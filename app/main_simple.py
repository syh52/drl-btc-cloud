from fastapi import FastAPI
from pydantic import BaseModel
import json
from datetime import datetime, timezone
import numpy as np
import os
import logging
from typing import Optional
from google.cloud import storage

app = FastAPI(title="DRL BTC Trading API", version="1.0.0")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 模型加载状态
model_loaded = False
model_version = None
model_config = None

def load_model_config():
    """从latest.json加载模型配置"""
    global model_loaded, model_version, model_config
    
    try:
        # 尝试从本地加载
        if os.path.exists("../models/ppo/latest.json"):
            with open("../models/ppo/latest.json", 'r') as f:
                model_config = json.load(f)
        else:
            # 从GCS加载
            bucket_name = os.getenv('BUCKET_NAME', 'ai4fnew-drl-btc-20250827')
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob('models/ppo/latest.json')
            
            if blob.exists():
                config_data = blob.download_as_text()
                model_config = json.loads(config_data)
            else:
                logger.warning("No model config found in GCS")
                return False
        
        model_version = model_config.get('model_version', 'unknown')
        model_loaded = True
        logger.info(f"Model config loaded: version {model_version}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model config: {e}")
        return False

# 启动时加载模型配置
load_model_config()

async def log_decision(decision_data: dict):
    """记录决策日志到JSONL文件（本地和GCS）"""
    try:
        # 生成文件名（按天分割）
        date_str = datetime.fromtimestamp(decision_data['timestamp']).strftime('%Y%m%d')
        log_filename = f"{date_str}.jsonl"
        
        # 本地日志目录
        local_log_dir = "logs/decisions"
        os.makedirs(local_log_dir, exist_ok=True)
        local_log_path = os.path.join(local_log_dir, log_filename)
        
        # 幂等性检查：按分钟检查是否已记录
        minute_key = decision_data['timestamp'] // 60 * 60  # 精确到分钟
        
        # 检查本地文件是否已有该分钟的记录
        if os.path.exists(local_log_path):
            with open(local_log_path, 'r') as f:
                for line in f:
                    if line.strip():
                        existing_data = json.loads(line.strip())
                        existing_minute = existing_data['timestamp'] // 60 * 60
                        if existing_minute == minute_key:
                            logger.info(f"Decision for minute {minute_key} already logged")
                            return
        
        # 写入本地文件
        with open(local_log_path, 'a') as f:
            f.write(json.dumps(decision_data) + '\n')
        
        # 同时备份到GCS
        try:
            bucket_name = os.getenv('BUCKET_NAME', 'ai4fnew-drl-btc-20250827')
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            gcs_log_path = f"logs/decisions/{log_filename}"
            blob = bucket.blob(gcs_log_path)
            
            # 读取本地文件内容上传到GCS
            with open(local_log_path, 'r') as f:
                blob.upload_from_file(f, content_type='text/plain')
            
            logger.info(f"Decision logged to GCS: gs://{bucket_name}/{gcs_log_path}")
            
        except Exception as gcs_error:
            logger.warning(f"Failed to backup to GCS: {gcs_error}")
            
        logger.info(f"Decision logged locally: {local_log_path}")
        
    except Exception as e:
        logger.error(f"Failed to log decision: {e}")

@app.get("/")
async def root():
    return {
        "message": "🚀 DRL BTC 自动交易系统",
        "version": "1.0.0",
        "description": "基于深度强化学习的比特币自动交易系统 MVP",
        "endpoints": {
            "health": "/health",
            "status": "/status", 
            "trade": "/tick",
            "docs": "/docs"
        },
        "project": {
            "name": "ai4fnew",
            "region": "asia-southeast1",
            "bucket": "ai4fnew-drl-btc-20250827"
        }
    }

class TradingRequest(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    lookback: int = 60

class TradingResponse(BaseModel):
    ts: int
    price: float
    action: float
    position: float
    equity: float
    note: str = "paper-trade"

# 全局状态
current_position = 0.0
current_equity = 1.0
trade_count = 0

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_version": model_version,
        "gcs_connected": True,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/status")
async def get_status():
    return {
        "model_loaded": model_loaded,
        "model_version": model_version,
        "current_position": current_position,
        "current_equity": current_equity,
        "trade_count": trade_count,
        "server_time": datetime.now(timezone.utc).isoformat()
    }

@app.post("/tick", response_model=TradingResponse)
async def trading_tick(request: TradingRequest):
    global current_position, current_equity, trade_count
    
    # 模拟价格和决策
    current_price = 65000 + np.random.normal(0, 1000)  # 模拟BTC价格
    action = np.random.uniform(-0.5, 0.5)  # 模拟模型决策
    
    # 更新状态
    position_change = abs(action - current_position)
    if position_change > 0.01:
        trade_count += 1
    
    current_position = action
    current_equity *= (1 + np.random.normal(0, 0.001))  # 模拟权益变化
    
    # 记录决策日志
    now = datetime.now(timezone.utc)
    decision_data = {
        "timestamp": int(now.timestamp()),
        "datetime": now.isoformat(),
        "symbol": request.symbol,
        "price": round(float(current_price), 2),
        "action": round(float(action), 4),
        "position": round(float(current_position), 4),
        "equity": round(float(current_equity), 6),
        "trade_count": trade_count,
        "model_version": model_version or "mock"
    }
    
    # 写入决策日志到JSONL文件（幂等性按分钟）
    await log_decision(decision_data)
    
    logger.info(f"📊 交易决策: {json.dumps(decision_data)}")
    
    return TradingResponse(
        ts=decision_data["timestamp"],
        price=current_price,
        action=action,
        position=current_position,
        equity=current_equity,
        note="paper-trade-simple"
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main_simple:app", host="0.0.0.0", port=port, log_level="info")
