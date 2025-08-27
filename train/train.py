import os
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch

from btc_env import BTCTradingEnv, check_btc_env
from google.cloud import storage


class TrainingCallback:
    """训练过程中的自定义回调"""
    
    def __init__(self, save_freq: int = 10000):
        self.save_freq = save_freq
        self.step_count = 0
        
    def __call__(self, locals_, globals_):
        self.step_count += 1
        
        if self.step_count % self.save_freq == 0:
            print(f"训练步数: {self.step_count}")
        
        return True


def create_training_env(data_path: str, lookback: int = 60) -> BTCTradingEnv:
    """创建训练环境"""
    print(f"📊 创建训练环境...")
    
    env = BTCTradingEnv(
        data_path=data_path,
        lookback=lookback,
        initial_balance=10000.0,
        fee_rate=0.001,
        max_episode_steps=5000
    )
    
    # 包装为Monitor环境以记录训练统计
    env = Monitor(env)
    
    return env


def train_ppo_model(
    env: BTCTradingEnv,
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    verbose: int = 1
) -> PPO:
    """训练PPO模型"""
    
    print(f"🤖 开始训练PPO模型...")
    print(f"   - 总训练步数: {total_timesteps:,}")
    print(f"   - 学习率: {learning_rate}")
    print(f"   - 批次大小: {batch_size}")
    
    # 设置设备 (优先GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   - 使用设备: {device}")
    
    # 创建向量化环境
    vec_env = DummyVecEnv([lambda: env])
    
    # 配置PPO模型
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=verbose,
        device=device,
        tensorboard_log="./tensorboard_logs/"
    )
    
    print(f"✅ PPO模型配置完成")
    print(f"   - 策略网络: MLP")
    print(f"   - 参数数量: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # 创建回调函数
    callbacks = []
    
    # 每10000步保存检查点
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="ppo_btc"
    )
    callbacks.append(checkpoint_callback)
    
    # 开始训练
    print(f"🚀 开始训练...")
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        training_time = datetime.now() - start_time
        print(f"🎉 训练完成！耗时: {training_time}")
        
        return model
        
    except KeyboardInterrupt:
        print("⚠️ 训练被用户中断")
        return model
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        raise


def evaluate_model(model: PPO, env: BTCTradingEnv, n_episodes: int = 10) -> dict:
    """评估模型性能"""
    print(f"📈 评估模型性能 (运行 {n_episodes} 个episode)...")
    
    episode_returns = []
    episode_lengths = []
    total_trades = []
    final_equities = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_return = 0
        episode_length = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_return += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        total_trades.append(info.get('total_trades', 0))
        final_equities.append(info.get('equity', 0))
        
        print(f"   Episode {episode + 1}: Return={episode_return:.4f}, "
              f"Length={episode_length}, Trades={info.get('total_trades', 0)}, "
              f"Final Equity={info.get('equity', 0):.2f}")
    
    # 计算评估统计
    eval_stats = {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'mean_trades': np.mean(total_trades),
        'mean_final_equity': np.mean(final_equities),
        'min_return': np.min(episode_returns),
        'max_return': np.max(episode_returns)
    }
    
    print(f"📊 评估结果:")
    for key, value in eval_stats.items():
        print(f"   - {key}: {value:.4f}")
    
    return eval_stats


def save_model_to_gcs(model: PPO, gcs_path: str, local_model_path: str = "ppo_btc.zip"):
    """保存模型到Google Cloud Storage"""
    print(f"☁️ 保存模型到GCS: {gcs_path}")
    
    try:
        # 先保存到本地
        model.save(local_model_path)
        print(f"✅ 模型已保存到本地: {local_model_path}")
        
        # 解析GCS路径
        if gcs_path.startswith("gs://"):
            gcs_path = gcs_path[5:]  # 移除gs://前缀
        
        bucket_name = gcs_path.split("/")[0]
        blob_name = "/".join(gcs_path.split("/")[1:])
        
        # 上传到GCS
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(local_model_path)
        
        print(f"✅ 模型已上传到GCS: gs://{bucket_name}/{blob_name}")
        
        # 清理本地文件
        if os.path.exists(local_model_path):
            os.remove(local_model_path)
        
    except Exception as e:
        print(f"❌ 上传模型到GCS失败: {e}")
        print(f"💡 模型已保存到本地: {local_model_path}")


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description="训练BTC交易PPO模型")
    
    parser.add_argument("--data_csv", type=str, default="btc_data.csv",
                       help="BTC历史数据CSV文件路径")
    parser.add_argument("--out_dir", type=str, default="gs://your-bucket/models/ppo/",
                       help="模型输出GCS路径")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="总训练步数")
    parser.add_argument("--lookback", type=int, default=60,
                       help="历史数据回望长度")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="学习率")
    parser.add_argument("--eval", action="store_true",
                       help="训练后进行模型评估")
    parser.add_argument("--check_only", action="store_true",
                       help="仅检查环境，不进行训练")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 BTC DRL 自动交易训练系统")
    print("=" * 60)
    print(f"📊 数据文件: {args.data_csv}")
    print(f"📁 输出路径: {args.out_dir}")
    print(f"⏱️  训练步数: {args.timesteps:,}")
    print(f"👀 回望长度: {args.lookback}")
    print("=" * 60)
    
    # 检查环境
    if not check_btc_env(args.data_csv):
        print("❌ 环境检查失败，训练终止")
        return
    
    if args.check_only:
        print("✅ 环境检查完成，仅检查模式结束")
        return
    
    # 创建输出目录
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./tensorboard_logs", exist_ok=True)
    
    try:
        # 创建训练环境
        env = create_training_env(args.data_csv, args.lookback)
        
        # 训练模型
        model = train_ppo_model(
            env=env,
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            verbose=1
        )
        
        # 评估模型 (可选)
        if args.eval:
            eval_stats = evaluate_model(model, env, n_episodes=5)
        
        # 保存模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"ppo_btc_{timestamp}.zip"
        gcs_full_path = args.out_dir.rstrip("/") + "/" + model_filename
        
        save_model_to_gcs(model, gcs_full_path, model_filename)
        
        print("🎉 训练流程完成！")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()