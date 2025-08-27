import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
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
        max_episode_steps: int = 10000
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
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # 观测空间: [lookback × features] 
        self.feature_count = 6  # OHLCV + returns + ma_ratio
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.lookback, self.feature_count), 
            dtype=np.float32
        )
        
        # 状态变量
        self.current_step = 0
        self.position = 0.0  # 当前持仓比例 [-1, 1]
        self.balance = initial_balance
        self.equity = initial_balance
        self.total_trades = 0
        self.total_fees = 0.0
        
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """加载并预处理K线数据"""
        try:
            # 读取CSV文件 (假设列名: timestamp, open, high, low, close, volume)
            df = pd.read_csv(data_path)
            
            # 确保必要列存在
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"缺少必要的列: {missing_cols}")
            
            # 计算技术指标
            df = self._calculate_features(df)
            
            # 移除包含NaN的行
            df = df.dropna().reset_index(drop=True)
            
            print(f"数据加载成功: {len(df)} 条记录, 特征数: {df.shape[1]}")
            return df
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            # 如果加载失败，生成模拟数据用于测试
            return self._generate_mock_data()
    
    def _generate_mock_data(self) -> pd.DataFrame:
        """生成模拟BTC数据用于测试"""
        print("生成模拟BTC数据...")
        np.random.seed(42)
        
        # 生成5000个数据点
        n_points = 5000
        
        # 模拟价格走势 (基于随机游走 + 趋势)
        price_returns = np.random.normal(0, 0.02, n_points)  # 2%标准差
        price_returns[0] = 0
        prices = 50000 * np.exp(np.cumsum(price_returns))  # 起始价格50000
        
        # 生成OHLCV数据
        data = []
        for i in range(n_points):
            close_price = prices[i]
            # 简化模拟: open ≈ 上一个close, high/low 基于波动
            open_price = prices[i-1] if i > 0 else close_price
            volatility = abs(price_returns[i])
            high_price = close_price * (1 + volatility * np.random.uniform(0.5, 1.5))
            low_price = close_price * (1 - volatility * np.random.uniform(0.5, 1.5))
            volume = np.random.uniform(100, 1000)
            
            data.append({
                'open': open_price,
                'high': max(open_price, high_price, close_price),
                'low': min(open_price, low_price, close_price),
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df = self._calculate_features(df)
        return df.dropna().reset_index(drop=True)
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标特征"""
        # 收益率
        df['returns'] = df['close'].pct_change()
        
        # 移动平均线比率
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_ratio'] = df['close'] / df['ma_20'] - 1  # 偏离程度
        
        return df
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 随机选择起始位置 (保证有足够的历史数据)
        max_start = self.data_length - self.max_episode_steps - self.lookback
        if max_start <= 0:
            raise ValueError(f"数据不足以支持训练。需要至少 {self.max_episode_steps + self.lookback} 条数据，当前只有 {self.data_length} 条")
        
        self.current_step = np.random.randint(self.lookback, max_start)
        
        # 重置状态
        self.position = 0.0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.total_trades = 0
        self.total_fees = 0.0
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步动作"""
        # 解析动作 (目标持仓比例)
        target_position = np.clip(action[0], -1.0, 1.0)
        
        # 计算当前价格
        current_price = self.data.iloc[self.current_step]['close']
        
        # 计算仓位变化和交易成本
        position_change = target_position - self.position
        trade_cost = abs(position_change) * self.fee_rate
        
        # 计算收益 (基于上一时刻的持仓和价格变化)
        if self.current_step > self.lookback:
            prev_price = self.data.iloc[self.current_step - 1]['close']
            price_return = (current_price - prev_price) / prev_price
            position_return = self.position * price_return
            
            # 更新权益
            self.equity *= (1 + position_return - trade_cost)
        
        # 更新持仓
        self.position = target_position
        
        # 记录交易统计
        if abs(position_change) > 0.01:  # 仓位变化超过1%才算作交易
            self.total_trades += 1
        self.total_fees += trade_cost * self.equity
        
        # 计算奖励 = 本期收益 - 交易成本
        reward = 0.0
        if self.current_step > self.lookback:
            reward = position_return - trade_cost
        
        # 更新步数
        self.current_step += 1
        
        # 检查结束条件
        terminated = self.current_step >= self.data_length - 1
        truncated = (self.current_step - self.lookback) >= self.max_episode_steps
        
        # 风控: 权益下跌过多则提前结束
        if self.equity < self.initial_balance * 0.5:  # 亏损超过50%
            terminated = True
            reward -= 1.0  # 额外惩罚
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观测值"""
        start_idx = self.current_step - self.lookback
        end_idx = self.current_step
        
        # 获取历史数据窗口
        window_data = self.data.iloc[start_idx:end_idx]
        
        # 构建特征矩阵 [lookback × features]
        features = []
        for _, row in window_data.iterrows():
            feature_vector = [
                row['open'],
                row['high'], 
                row['low'],
                row['close'],
                row['volume'],
                row['returns'] if not np.isnan(row['returns']) else 0.0,
                # row['ma_ratio'] if not np.isnan(row['ma_ratio']) else 0.0
            ]
            features.append(feature_vector)
        
        obs = np.array(features, dtype=np.float32)
        
        # 归一化处理 (简单的价格归一化)
        if len(obs) > 0:
            # 价格特征 (OHLC) 相对于最新收盘价归一化
            latest_close = obs[-1, 3]  # 最新收盘价
            obs[:, :4] = obs[:, :4] / latest_close - 1.0  # 转为收益率形式
            
            # 成交量对数归一化
            obs[:, 4] = np.log1p(obs[:, 4]) / 10.0  # 压缩到合理范围
            
            # 收益率已经是合理范围,不需要额外处理
            
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        return {
            'step': self.current_step,
            'position': self.position,
            'equity': self.equity,
            'total_trades': self.total_trades,
            'total_fees': self.total_fees,
            'current_price': self.data.iloc[self.current_step]['close'] if self.current_step < len(self.data) else 0
        }
    
    def render(self, mode: str = 'human') -> Optional[Any]:
        """渲染环境状态"""
        if mode == 'human':
            info = self._get_info()
            print(f"Step: {info['step']}, Price: {info['current_price']:.2f}, "
                  f"Position: {info['position']:.3f}, Equity: {info['equity']:.2f}, "
                  f"Trades: {info['total_trades']}")
        return None
    
    def close(self):
        """关闭环境"""
        pass


def check_btc_env(data_path: str = None) -> bool:
    """检查BTC交易环境是否正常工作"""
    print("🔍 检查BTC交易环境...")
    
    try:
        # 创建环境 (使用模拟数据)
        env = BTCTradingEnv(
            data_path=data_path or "dummy_path.csv",  # 触发模拟数据生成
            lookback=30,
            max_episode_steps=100
        )
        
        print(f"✅ 环境创建成功")
        print(f"   - 观测空间: {env.observation_space}")
        print(f"   - 动作空间: {env.action_space}")
        print(f"   - 数据长度: {env.data_length}")
        
        # 使用stable-baselines3内置检查
        check_env(env)
        print("✅ Stable-Baselines3 环境检查通过")
        
        # 运行几步测试
        obs, info = env.reset()
        print(f"✅ 重置成功，观测形状: {obs.shape}")
        
        for i in range(5):
            action = env.action_space.sample()  # 随机动作
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {i+1}: action={action[0]:.3f}, reward={reward:.6f}, "
                  f"position={info['position']:.3f}, equity={info['equity']:.2f}")
            
            if terminated or truncated:
                print(f"   Episode ended: terminated={terminated}, truncated={truncated}")
                break
        
        env.close()
        print("🎉 BTC交易环境检查完成，所有测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ BTC交易环境检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行环境检查
    check_btc_env()