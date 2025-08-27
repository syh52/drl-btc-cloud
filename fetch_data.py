#!/usr/bin/env python3
"""
使用CCXT获取BTCUSDT 1分钟K线数据（最近90天）
补齐分钟与基础特征，保存到本地和GCS
"""

import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.cloud import storage
import argparse
import time

def fetch_btc_data(days=90, save_local=True, save_gcs=True):
    """获取BTCUSDT数据并处理"""
    print(f"🔄 开始获取BTCUSDT最近{days}天的1分钟数据...")
    
    # 初始化Binance交易所
    exchange = ccxt.binance({
        'rateLimit': 1200,  # 限制请求频率
        'enableRateLimit': True,
    })
    
    # 计算时间范围
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    print(f"📅 时间范围: {start_time.strftime('%Y-%m-%d %H:%M')} 到 {end_time.strftime('%Y-%m-%d %H:%M')}")
    
    # 获取数据
    symbol = 'BTC/USDT'
    timeframe = '1m'
    
    # CCXT使用毫秒时间戳
    since = int(start_time.timestamp() * 1000)
    end_timestamp = int(end_time.timestamp() * 1000)
    
    all_ohlcv = []
    current_since = since
    
    print("📡 正在获取数据...")
    batch_count = 0
    
    while current_since < end_timestamp:
        try:
            # 获取最多1000条数据（Binance限制）
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            
            if not ohlcv:
                print("⚠️ 没有更多数据")
                break
                
            all_ohlcv.extend(ohlcv)
            batch_count += 1
            
            # 更新时间戳到下一批
            current_since = ohlcv[-1][0] + 60000  # 加1分钟（毫秒）
            
            print(f"📊 已获取批次 {batch_count}, 最新时间: {datetime.fromtimestamp(ohlcv[-1][0]/1000)}")
            
            # 避免API限制
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ 获取数据时出错: {e}")
            time.sleep(2)  # 出错时等待更久
            continue
    
    if not all_ohlcv:
        raise ValueError("❌ 未能获取到任何数据")
    
    print(f"✅ 共获取 {len(all_ohlcv)} 条1分钟K线数据")
    
    # 转换为DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 转换时间戳
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime')
    
    # 补齐分钟（确保连续性）
    print("🔧 补齐分钟数据...")
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1min')
    df_reindexed = df.reindex(full_range)
    
    # 向前填充缺失值
    df_reindexed = df_reindexed.fillna(method='ffill')
    
    # 计算基础特征
    print("📊 计算技术指标...")
    
    # 收益率
    df_reindexed['returns'] = df_reindexed['close'].pct_change()
    
    # 移动平均
    df_reindexed['ma_20'] = df_reindexed['close'].rolling(20).mean()
    df_reindexed['ma_60'] = df_reindexed['close'].rolling(60).mean()
    
    # MA比率
    df_reindexed['ma_ratio'] = df_reindexed['close'] / df_reindexed['ma_20']
    
    # 波动率
    df_reindexed['volatility'] = df_reindexed['returns'].rolling(60).std()
    
    # 成交量移动平均
    df_reindexed['volume_ma'] = df_reindexed['volume'].rolling(20).mean()
    df_reindexed['volume_ratio'] = df_reindexed['volume'] / df_reindexed['volume_ma']
    
    # 价格标准化（为了ML模型）
    df_reindexed['price_norm'] = (df_reindexed['close'] - df_reindexed['close'].rolling(60).mean()) / df_reindexed['close'].rolling(60).std()
    
    # 删除前60行（因为指标需要热身）
    df_clean = df_reindexed.iloc[60:].copy()
    
    # 填充剩余的NaN
    df_clean = df_clean.fillna(0)
    
    print(f"📈 最终数据形状: {df_clean.shape}")
    print(f"📅 数据时间范围: {df_clean.index.min()} 到 {df_clean.index.max()}")
    
    # 保存本地文件
    if save_local:
        local_path = "train/btc_data.csv"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        df_clean.to_csv(local_path)
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
            blob.upload_from_string(df_clean.to_csv(), content_type='text/csv')
            
            print(f"☁️ 数据已备份到 GCS: gs://{bucket_name}/{gcs_path}")
            
        except Exception as e:
            print(f"⚠️ 保存到GCS时出错: {e}")
    
    return df_clean

def main():
    parser = argparse.ArgumentParser(description='获取BTCUSDT历史数据')
    parser.add_argument('--days', type=int, default=90, help='获取最近N天数据')
    parser.add_argument('--no-local', action='store_true', help='不保存到本地')
    parser.add_argument('--no-gcs', action='store_true', help='不保存到GCS')
    
    args = parser.parse_args()
    
    try:
        df = fetch_btc_data(
            days=args.days,
            save_local=not args.no_local,
            save_gcs=not args.no_gcs
        )
        
        print("\n✅ 数据获取完成!")
        print(f"📊 总共 {len(df)} 条记录")
        print(f"🔢 特征列: {list(df.columns)}")
        print(f"📅 时间范围: {df.index[0]} 到 {df.index[-1]}")
        
        # 显示最近5行数据
        print("\n📋 最近5行数据:")
        print(df.tail().round(4))
        
    except Exception as e:
        print(f"❌ 获取数据失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())