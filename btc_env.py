import os
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env


class BTCTradingEnv(gym.Env):
    """
    BTC交易环境 - 目标持仓法

    观测空间: [lookback × features] 历史价格和技术指标矩阵
    动作空间: [-1, 1] 目标持仓比例 (-1=空仓, 0=中性, 1=满仓)
    奖励函数: 持仓收益 - 交易手续费
    """

    def __init__(
        self,
        data_path: str,
        lookback: int = 60,
        initial_balance: float = 10000.0,
        fee_rate: float = 0.001,  # 0.1% 手续费
        max_episode_steps: int = 10000,
    ):
        super().__init__()

        # 参数配置
        self.lookback = lookback
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.max_episode_steps = max_episode_steps

        # 加载数据
        self.data = self._load_data(data_path)
        self.data_length = len(self.data)

        # 动作空间: 目标持仓比例 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 观测空间: [lookback × features]
        # 实际特征: OHLCV + returns (ma_ratio已注释掉)
        self.feature_count = 6  # OHLCV + returns + volume_log
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback, self.feature_count),
            dtype=np.float32,
        )

        # 状态变量
        self.current_step = 0
        self.position = 0.0  # 当前持仓比例 [-1, 1]
        self.balance = initial_balance
        self.equity = initial_balance
        self.total_trades = 0
        self.total_fees = 0.0
        self.peak_equity = initial_balance
        self.max_drawdown_allowed = 0.10  # 10%

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """加载并预处理K线数据"""
        print(f"📂 加载训练数据: {data_path}")
        try:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"数据文件不存在: {data_path}")
            df = pd.read_csv(data_path)

            required_cols = ["open", "high", "low", "close", "volume"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"数据文件缺少必要列: {missing}")

            # 若存在时间列，进行排序
            time_cols = [c for c in ["timestamp", "datetime", "time"] if c in df.columns]
            if time_cols:
                tc = time_cols[0]
                df[tc] = pd.to_datetime(df[tc], unit="s", errors="ignore")
                df = df.sort_values(tc)

            df = self._calculate_features(df).dropna().reset_index(drop=True)
            if len(df) < self.max_episode_steps + self.lookback + 1:
                raise ValueError(
                    f"数据量不足：至少需要 {self.max_episode_steps + self.lookback + 1} 行，当前 {len(df)}"
                )

            print(f"✅ 数据加载成功: {len(df)} 条记录, 特征数: {df.shape[1]}")
            return df
        except Exception as e:
            print(f"❌ 训练数据加载失败: {e}")
            print(f"❌ 数据路径: {data_path}")
            raise


    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标特征"""
        # 收益率
        df["returns"] = df["close"].pct_change()

        # 对数变换处理volume的偏态分布
        df["volume_log"] = np.log1p(df["volume"])

        # 移动平均线比率
        # df['ma_20'] = df['close'].rolling(20, min_periods=1).mean()
        # df['ma_ratio'] = df['close'] / df['ma_20'] - 1

        # 特征归一化（使用滚动窗口）
        window = min(100, len(df) // 4)
        if window >= 20:
            # 价格特征相对归一化
            rolling_mean = df["close"].rolling(window, min_periods=10).mean()
            rolling_std = df["close"].rolling(window, min_periods=10).std()

            for col in ["open", "high", "low", "close"]:
                df[f"{col}_norm"] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

            # 成交量归一化
            vol_mean = df["volume_log"].rolling(window, min_periods=10).mean()
            vol_std = df["volume_log"].rolling(window, min_periods=10).std()
            df["volume_norm"] = (df["volume_log"] - vol_mean) / (vol_std + 1e-8)

            # 收益率已经相对稳定，简单clip即可
            df["returns"] = np.clip(df["returns"], -0.1, 0.1)  # 限制在±10%
        else:
            # 数据量太少时简单处理
            for col in ["open", "high", "low", "close"]:
                df[f"{col}_norm"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
            df["volume_norm"] = (df["volume_log"] - df["volume_log"].mean()) / (
                df["volume_log"].std() + 1e-8
            )
            df["returns"] = np.clip(df["returns"], -0.1, 0.1)

        return df

    def _get_observation(self) -> np.ndarray:
        """获取当前观测"""
        if self.current_step < self.lookback:
            # 数据不足时用重复填充
            available = self.current_step + 1
            repeat_count = self.lookback - available

            start_idx = max(0, self.current_step - available + 1)
            end_idx = self.current_step + 1
            real_data = self.data.iloc[start_idx:end_idx]

            # 用第一行重复填充
            if repeat_count > 0:
                first_row = real_data.iloc[[0]]
                padding = pd.concat([first_row] * repeat_count, ignore_index=True)
                obs_data = pd.concat([padding, real_data], ignore_index=True)
            else:
                obs_data = real_data
        else:
            start_idx = self.current_step - self.lookback + 1
            end_idx = self.current_step + 1
            obs_data = self.data.iloc[start_idx:end_idx]

        # 选择特征列
        # 使用归一化特征: OHLCV_norm + returns
        feature_cols = [
            "open_norm",
            "high_norm",
            "low_norm",
            "close_norm",
            "volume_norm",
            "returns",
        ]

        obs = obs_data[feature_cols].values.astype(np.float32)
        if obs.shape[0] != self.lookback:
            print(
                f"⚠️  观测维度不匹配: 期望 {self.lookback}, 实际 {obs.shape[0]}, step={self.current_step}"
            )
            # 强制调整维度
            if obs.shape[0] < self.lookback:
                padding = np.tile(obs[0:1], (self.lookback - obs.shape[0], 1))
                obs = np.vstack([padding, obs])
            else:
                obs = obs[-self.lookback :]

        return obs

    def _get_current_price(self) -> float:
        """获取当前价格"""
        return float(self.data.iloc[self.current_step]["close"])

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境"""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # 随机选择起始位置（确保有足够的数据）
        max_start = self.data_length - self.max_episode_steps - 1
        self.current_step = np.random.randint(self.lookback, max(self.lookback + 1, max_start))

        # 重置状态
        self.position = 0.0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.total_trades = 0
        self.total_fees = 0.0
        self.peak_equity = self.initial_balance

        observation = self._get_observation()
        info = {
            "step": self.current_step,
            "price": self._get_current_price(),
            "position": self.position,
            "equity": self.equity,
        }

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        if self.current_step >= self.data_length - 1:
            return self._get_observation(), 0.0, True, False, {"reason": "数据结束"}

        # 获取动作
        target_position = np.clip(float(action[0]), -1.0, 1.0)
        current_price = self._get_current_price()

        # 计算持仓变化和手续费
        position_change = abs(target_position - self.position)
        trade_fee = position_change * current_price * self.fee_rate
        self.total_fees += trade_fee

        if position_change > 0.01:  # 0.01 = 1% 最小变化阈值
            self.total_trades += 1

        # 移动到下一步
        self.current_step += 1
        next_price = self._get_current_price()

        # 计算收益
        if self.position != 0:
            price_return = (next_price - current_price) / current_price
            position_return = self.position * price_return
            self.equity *= 1 + position_return

        # 扣除手续费
        self.equity -= trade_fee
        self.position = target_position

        # 更新峰值
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # 计算奖励
        if self.position != 0:
            holding_return = self.position * (next_price - current_price) / current_price
            reward = holding_return - (trade_fee / self.initial_balance)
        else:
            reward = -(trade_fee / self.initial_balance)  # 只有手续费成本

        # 检查终止条件
        terminated = False
        truncated = False

        # 回撤过大终止
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.equity) / self.peak_equity
            if drawdown > self.max_drawdown_allowed:
                terminated = True

        # 资金耗尽终止
        if self.equity <= self.initial_balance * 0.1:
            terminated = True

        # 达到最大步数截断
        if self.current_step >= self.data_length - 1:
            truncated = True

        # 构造信息
        info = {
            "step": self.current_step,
            "price": next_price,
            "position": self.position,
            "equity": self.equity,
            "total_trades": self.total_trades,
            "total_fees": self.total_fees,
            "drawdown": (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0,
        }

        return self._get_observation(), reward, terminated, truncated, info


def check_btc_env(data_path: str = "train/btc_data.csv") -> bool:
    """检查BTC交易环境"""
    print("🔍 检查BTC交易环境...")
    try:
        # 创建环境
        env = BTCTradingEnv(
            data_path=data_path, lookback=60, initial_balance=10000, fee_rate=0.001
        )

        print(f"观测空间: {env.observation_space}")
        print(f"动作空间: {env.action_space}")
        print(f"数据长度: {env.data_length}")

        # 使用stable_baselines3的环境检查器
        print("🧪 运行环境一致性检查...")
        check_env(env, warn=True)

        # 测试重置和步进
        print("🔄 测试环境重置...")
        obs, info = env.reset()
        print(f"初始观测形状: {obs.shape}")
        print(f"初始信息: {info}")

        print("👟 测试环境步进...")
        for i in range(5):
            action = env.action_space.sample()  # 随机动作
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"步骤 {i+1}: 奖励={reward:.6f}, 持仓={info['position']:.3f}, "
                f"净值={info['equity']:.2f}, 价格=${info['price']:.2f}"
            )
            if terminated or truncated:
                print("环境提前终止")
                break

        print("✅ BTC交易环境检查通过！")
        return True

    except Exception as e:
        print(f"❌ 环境检查失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 检查环境
    success = check_btc_env("btc_data.csv")
    if success:
        print("🎉 环境测试成功！可以开始训练了。")
    else:
        print("💥 环境测试失败，请检查数据和代码。")
        exit(1)