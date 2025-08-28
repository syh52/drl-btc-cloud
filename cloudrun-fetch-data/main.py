#!/usr/bin/env python3
"""
Cloud Run: BTC历史数据获取服务
- 支持多时间周期（1m、5m、15m、1h、4h、1d）
- 自动保存到GCS
- 处理大数据量（18个月数据）
- 云环境直连，无需代理
- 适合长时间运行任务
"""

import os
import json
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import logging

# 延迟导入重量级库，避免启动时加载失败
def get_ccxt():
    import ccxt
    return ccxt

def get_pandas():
    import pandas as pd
    return pd

def get_storage():
    from google.cloud import storage
    return storage

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_btc_data_cloud(timeframe='5m', days=540, save_gcs=True):
    """
    在云环境中获取BTC数据
    Args:
        timeframe: 时间周期 (1m, 5m, 15m, 1h, 4h, 1d)
        days: 获取天数 (540天 ≈ 18个月)
        save_gcs: 是否保存到GCS
    """
    # 时间周期配置
    timeframe_config = {
        '1m': {'freq': '1min', 'interval_ms': 60000, 'display': '1分钟'},
        '5m': {'freq': '5min', 'interval_ms': 300000, 'display': '5分钟'},
        '15m': {'freq': '15min', 'interval_ms': 900000, 'display': '15分钟'},
        '1h': {'freq': '1H', 'interval_ms': 3600000, 'display': '1小时'},
        '4h': {'freq': '4H', 'interval_ms': 14400000, 'display': '4小时'},
        '1d': {'freq': '1D', 'interval_ms': 86400000, 'display': '1天'}
    }
    
    if timeframe not in timeframe_config:
        raise ValueError(f"不支持的时间周期: {timeframe}，支持: {list(timeframe_config.keys())}")
    
    config = timeframe_config[timeframe]
    logger.info(f"🔄 Cloud Run获取BTCUSDT最近{days}天的{config['display']}数据...")
    
    # 云环境交易所配置（直连，不需要代理）
    exchange_config = {
        'rateLimit': 800,   # Cloud Run可以更快一些
        'enableRateLimit': True,
        'timeout': 30000,   # 30秒超时
        'sandbox': False,
    }
    
    logger.info("🌐 Cloud Run云环境直连模式")
    
    try:
        ccxt = get_ccxt()
        exchange = ccxt.binance(exchange_config)
        logger.info("🔍 测试云环境交易所连接...")
        server_time = exchange.fetch_time()
        if server_time:
            logger.info(f"✅ Cloud Run连接成功，服务器时间: {datetime.fromtimestamp(server_time/1000)}")
    except Exception as e:
        logger.error(f"❌ Cloud Run连接失败: {e}")
        raise
    
    # 计算数据量
    interval_minutes = config['interval_ms'] // 60000
    total_expected_records = (days * 24 * 60) // interval_minutes
    logger.info(f"📊 预期获取约 {total_expected_records:,} 条数据")
    
    # 计算时间范围
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    logger.info(f"📅 时间范围: {start_time.strftime('%Y-%m-%d %H:%M')} 到 {end_time.strftime('%Y-%m-%d %H:%M')}")
    
    # 获取数据
    symbol = 'BTC/USDT'
    since = int(start_time.timestamp() * 1000)
    end_timestamp = int(end_time.timestamp() * 1000)
    
    all_ohlcv = []
    current_since = since
    batch_count = 0
    start_time_fetch = time.time()
    
    logger.info("📡 Cloud Run开始批量获取数据...")
    
    # Cloud Run优化的批处理参数
    max_retries = 5
    base_delay = 0.3  # Cloud Run可以更快
    
    while current_since < end_timestamp:
        retry_count = 0
        batch_success = False
        
        while retry_count < max_retries and not batch_success:
            try:
                logger.info(f"🔄 批次 {batch_count + 1} (总进度: {len(all_ohlcv):,})")
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
                
                if not ohlcv:
                    logger.info("⚠️ 没有更多数据")
                    batch_success = True
                    break
                    
                all_ohlcv.extend(ohlcv)
                batch_count += 1
                batch_success = True
                
                # 详细的进度显示
                current_records = len(all_ohlcv)
                progress_percent = min((current_records / total_expected_records) * 100, 100)
                elapsed_time = time.time() - start_time_fetch
                
                if elapsed_time > 0:
                    records_per_second = current_records / elapsed_time
                    if records_per_second > 0:
                        eta_seconds = max(0, (total_expected_records - current_records) / records_per_second)
                        eta_str = f", ETA: {int(eta_seconds//60)}分{int(eta_seconds%60)}秒"
                    else:
                        eta_str = ""
                else:
                    eta_str = ""
                
                current_since = ohlcv[-1][0] + config['interval_ms']
                
                logger.info(f"📊 批次 {batch_count} | {progress_percent:.1f}% ({current_records:,}/{total_expected_records:,}){eta_str}")
                
                # Cloud Run延迟优化
                time.sleep(base_delay)
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"⚠️ 批次 {batch_count + 1} 重试 {retry_count}/{max_retries}: {e}")
                    time.sleep(base_delay * retry_count * 2)  # 递增延迟
                else:
                    logger.error(f"❌ 批次 {batch_count + 1} 失败: {e}")
                    # 跳过这批数据，继续下一批
                    current_since += config['interval_ms'] * 1000
                    batch_success = True
    
    if not all_ohlcv:
        raise ValueError("❌ 未能获取到任何数据")
    
    total_fetch_time = time.time() - start_time_fetch
    logger.info(f"✅ 获取完成! {len(all_ohlcv)} 条数据，耗时 {total_fetch_time:.1f}秒")
    
    # 数据处理（快速版本，减少内存使用）
    logger.info("🔧 处理数据...")
    pd = get_pandas()
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime')
    
    # 简化的技术指标计算（适合云环境）
    logger.info("📊 计算基础技术指标...")
    
    if timeframe == '5m':
        short_ma, long_ma = 4, 12
    elif timeframe == '1h':
        short_ma, long_ma = 20, 60
    else:
        short_ma, long_ma = 10, 30
    
    df['returns'] = df['close'].pct_change()
    df['ma_short'] = df['close'].rolling(short_ma, min_periods=1).mean()
    df['ma_long'] = df['close'].rolling(long_ma, min_periods=1).mean()
    df['ma_ratio'] = df['close'] / df['ma_short']
    df['volatility'] = df['returns'].rolling(short_ma, min_periods=1).std()
    
    # 清理数据
    df_clean = df.fillna(0)
    
    logger.info(f"📈 处理完成: {df_clean.shape} | 时间范围: {df_clean.index.min()} 到 {df_clean.index.max()}")
    
    # 保存到GCS
    gcs_path = None
    if save_gcs:
        try:
            bucket_name = os.getenv('BUCKET_NAME', 'ai4fnew-drl-btc-20250827')
            storage = get_storage()
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            gcs_filename = f"btc_data_{timeframe}_{days}d.csv"
            gcs_path = f"data/{gcs_filename}"
            blob = bucket.blob(gcs_path)
            
            # 直接上传CSV数据
            csv_data = df_clean.to_csv()
            blob.upload_from_string(csv_data, content_type='text/csv')
            
            logger.info(f"☁️ 数据已保存到 GCS: gs://{bucket_name}/{gcs_path}")
            
        except Exception as e:
            logger.error(f"⚠️ 保存到GCS时出错: {e}")
    
    return {
        'success': True,
        'records_count': len(df_clean),
        'data_shape': list(df_clean.shape),
        'time_range': {
            'start': str(df_clean.index.min()),
            'end': str(df_clean.index.max())
        },
        'columns': list(df_clean.columns),
        'gcs_path': f"gs://{bucket_name}/{gcs_path}" if gcs_path else None,
        'processing_time_seconds': total_fetch_time
    }

@app.route('/', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'service': 'btc-data-fetcher',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/fetch', methods=['POST', 'GET'])
def fetch_data():
    """获取BTC数据的主接口"""
    try:
        # 解析参数
        if request.method == 'POST':
            data = request.get_json() or {}
        else:
            data = request.args.to_dict()
        
        timeframe = data.get('timeframe', '5m')
        days = int(data.get('days', 540))  # 默认18个月
        save_gcs_value = data.get('save_gcs', 'true')
        if isinstance(save_gcs_value, bool):
            save_gcs = save_gcs_value
        else:
            save_gcs = str(save_gcs_value).lower() == 'true'
        
        logger.info(f"🚀 开始获取数据: timeframe={timeframe}, days={days}, save_gcs={save_gcs}")
        
        # 执行数据获取
        result = fetch_btc_data_cloud(timeframe=timeframe, days=days, save_gcs=save_gcs)
        
        logger.info(f"✅ 数据获取完成: {result['records_count']} 条记录")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ 数据获取失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '数据获取失败'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)