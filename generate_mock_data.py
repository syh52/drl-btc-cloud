#!/usr/bin/env python3
"""
生成模拟BTCUSDT数据用于快速测试
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from google.cloud import storage

def generate_mock_btc_data(days=90):
    """生成模拟的BTCUSDT数据"""
    print(f"📊 生成{days}天模拟BTCUSDT数据...")
    
    # 时间范围
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # 创建分钟级时间序列
    time_index = pd.date_range(start=start_time, end=end_time, freq='1min')
    n_points = len(time_index)
    
    print(f"📅 时间范围: {start_time} 到 {end_time}")
    print(f"🔢 数据点数: {n_points}")
    
    # 模拟比特币价格走势
    np.random.seed(42)  # 保证可重现
    
    # 基础价格趋势
    base_price = 65000
    trend = np.linspace(-5000, 5000, n_points)  # 价格趋势
    
    # 添加随机波动
    volatility = 0.02  # 2% 波动率
    random_walk = np.cumsum(np.random.normal(0, volatility, n_points))
    
    # 计算收盘价
    close_prices = base_price + trend + (base_price * random_walk)
    
    # 生成OHLCV数据
    data = []
    for i, (timestamp, close) in enumerate(zip(time_index, close_prices)):
        # 模拟开盘、最高、最低价
        volatility_range = close * 0.005  # 0.5%的波动范围
        high = close + np.random.uniform(0, volatility_range)
        low = close - np.random.uniform(0, volatility_range)
        
        if i == 0:
            open_price = close
        else:
            open_price = close_prices[i-1]
        
        # 模拟成交量
        volume = np.random.uniform(100, 1000)
        
        data.append({
            'timestamp': int(timestamp.timestamp() * 1000),
            'open': open_price,
            'high': max(open_price, high, close),
            'low': min(open_price, low, close),
            'close': close,
            'volume': volume,
            'datetime': timestamp
        })
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    df = df.set_index('datetime')
    
    # 计算技术指标
    print("📊 计算技术指标...")
    
    # 收益率
    df['returns'] = df['close'].pct_change()
    
    # 移动平均
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_60'] = df['close'].rolling(60).mean()
    
    # MA比率
    df['ma_ratio'] = df['close'] / df['ma_20']
    
    # 波动率
    df['volatility'] = df['returns'].rolling(60).std()
    
    # 成交量移动平均
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # 价格标准化
    df['price_norm'] = (df['close'] - df['close'].rolling(60).mean()) / df['close'].rolling(60).std()
    
    # 删除前60行（因为指标需要热身）
    df_clean = df.iloc[60:].copy()
    
    # 填充剩余的NaN
    df_clean = df_clean.fillna(0)
    
    print(f"📈 最终数据形状: {df_clean.shape}")
    
    return df_clean

def save_data(df, save_local=True, save_gcs=True):
    """保存数据到本地和GCS"""
    
    # 保存本地文件
    if save_local:
        local_path = "train/btc_data.csv"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        df.to_csv(local_path)
        print(f"💾 数据已保存到: {local_path}")
    
    # 保存到GCS
    if save_gcs:
        try:
            bucket_name = os.getenv('BUCKET_NAME', 'ai4fnew-drl-btc-20250827')
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            # 保存CSV到GCS
            gcs_path = "data/btc_data.csv"
            blob = bucket.blob(gcs_path)
            blob.upload_from_string(df.to_csv(), content_type='text/csv')
            
            print(f"☁️ 数据已备份到 GCS: gs://{bucket_name}/{gcs_path}")
            
        except Exception as e:
            print(f"⚠️ 保存到GCS时出错: {e}")

def main():
    try:
        # 生成模拟数据
        df = generate_mock_btc_data(days=90)
        
        # 保存数据
        save_data(df, save_local=True, save_gcs=True)
        
        print("\n✅ 模拟数据生成完成!")
        print(f"📊 总共 {len(df)} 条记录")
        print(f"🔢 特征列: {list(df.columns)}")
        print(f"📅 时间范围: {df.index[0]} 到 {df.index[-1]}")
        
        # 显示基础统计
        print("\n📋 数据统计:")
        stats = df[['open', 'high', 'low', 'close', 'volume']].describe()
        print(stats.round(2))
        
        # 显示最近5行数据
        print("\n📋 最近5行数据:")
        print(df[['open', 'high', 'low', 'close', 'volume', 'returns']].tail().round(4))
        
    except Exception as e:
        print(f"❌ 生成数据失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())