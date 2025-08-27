import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path

import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from google.cloud import storage, secretmanager
from google.cloud import logging as cloud_logging
import ccxt

# 导入训练组件 (需要将train目录添加到路径)
import sys
sys.path.append('/app/train')

try:
    from stable_baselines3 import PPO
    from train.btc_env import BTCTradingEnv
except ImportError as e:
    print(f"警告: 无法导入训练模块: {e}")
    # 在生产环境中可能需要另外处理


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 如果在GCP环境中，使用Cloud Logging
if os.getenv('GOOGLE_CLOUD_PROJECT'):
    client = cloud_logging.Client()
    client.setup_logging()


# API请求/响应模型
class TradingRequest(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    lookback: int = 60
    source: str = "mock"  # mock, binance_testnet


class TradingResponse(BaseModel):
    ts: int  # 时间戳
    price: float  # 当前价格
    action: float  # 模型决策 [-1, 1]
    position: float  # 目标持仓比例
    equity: float  # 当前净值
    note: str = "paper-trade"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gcs_connected: bool
    timestamp: str


class ModelManager:
    """模型管理器 - 负责加载和管理PPO模型"""
    
    def __init__(self, bucket_name: str, model_prefix: str = "models/ppo/"):
        self.bucket_name = bucket_name
        self.model_prefix = model_prefix
        self.model: Optional[PPO] = None
        self.model_path: Optional[str] = None
        self.last_update: Optional[datetime] = None
        
    async def load_latest_model(self) -> bool:
        """从GCS加载最新模型"""
        try:
            logger.info("🔍 正在查找最新模型...")
            
            # 连接GCS
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            
            # 列出所有模型文件
            blobs = bucket.list_blobs(prefix=self.model_prefix)
            model_files = [blob for blob in blobs if blob.name.endswith('.zip')]
            
            if not model_files:
                logger.warning(f"⚠️ 在 gs://{self.bucket_name}/{self.model_prefix} 中未找到模型文件")
                return False
            
            # 按时间排序，获取最新模型
            latest_blob = max(model_files, key=lambda b: b.time_created)
            model_gcs_path = f"gs://{self.bucket_name}/{latest_blob.name}"
            
            logger.info(f"📦 找到最新模型: {model_gcs_path}")
            
            # 下载模型到本地临时目录
            local_model_path = f"/tmp/{latest_blob.name.split('/')[-1]}"
            latest_blob.download_to_filename(local_model_path)
            
            logger.info(f"⬇️ 模型已下载到: {local_model_path}")
            
            # 加载PPO模型
            self.model = PPO.load(local_model_path)
            self.model_path = model_gcs_path
            self.last_update = datetime.now(timezone.utc)
            
            logger.info(f"✅ 模型加载成功: {model_gcs_path}")
            
            # 清理本地文件
            if os.path.exists(local_model_path):
                os.remove(local_model_path)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None


class DataProvider:
    """数据提供器 - 获取BTC价格数据"""
    
    def __init__(self):
        self.exchange = None
        self.mock_data = self._generate_mock_data()
        
    def _generate_mock_data(self) -> pd.DataFrame:
        """生成模拟价格数据"""
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        
        # 生成最近24小时的1分钟数据
        n_points = 24 * 60  # 24小时 * 60分钟
        
        # 基础价格和随机波动
        base_price = 65000 + np.random.normal(0, 5000)
        returns = np.random.normal(0, 0.001, n_points)  # 0.1%标准差
        
        prices = [base_price]
        for ret in returns[1:]:
            next_price = prices[-1] * (1 + ret)
            prices.append(max(next_price, 1000))  # 最低价格1000
        
        # 构建OHLCV数据
        data = []
        current_time = datetime.now(timezone.utc)
        
        for i, close_price in enumerate(prices):
            timestamp = current_time.timestamp() - (n_points - i) * 60
            
            # 模拟OHLC
            open_price = prices[i-1] if i > 0 else close_price
            volatility = abs(returns[i])
            high_price = max(open_price, close_price) * (1 + volatility * np.random.uniform(0.5, 1.5))
            low_price = min(open_price, close_price) * (1 - volatility * np.random.uniform(0.5, 1.5))
            volume = np.random.uniform(100, 1000)
            
            data.append({
                'timestamp': int(timestamp * 1000),  # 毫秒时间戳
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        
        # 计算技术指标 (与训练环境保持一致)
        df['returns'] = df['close'].pct_change()
        df['ma_20'] = df['close'].rolling(20, min_periods=1).mean()
        df['ma_ratio'] = df['close'] / df['ma_20'] - 1
        
        return df.fillna(0)
    
    async def get_recent_candles(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """获取最近的K线数据"""
        try:
            if symbol == "BTCUSDT" and interval == "1m":
                # 使用模拟数据 (取最新limit个数据点)
                recent_data = self.mock_data.tail(limit).copy()
                
                # 更新最新价格 (模拟实时变化)
                latest_price = recent_data.iloc[-1]['close']
                price_change = np.random.normal(0, 0.002)  # 0.2%标准差
                new_price = latest_price * (1 + price_change)
                
                # 添加最新数据点
                current_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
                new_row = {
                    'timestamp': current_timestamp,
                    'open': latest_price,
                    'high': max(latest_price, new_price),
                    'low': min(latest_price, new_price),
                    'close': new_price,
                    'volume': np.random.uniform(100, 1000),
                    'returns': price_change,
                    'ma_20': recent_data['ma_20'].iloc[-1],  # 简化处理
                    'ma_ratio': new_price / recent_data['ma_20'].iloc[-1] - 1
                }
                
                # 使用pd.concat代替append
                new_df = pd.concat([recent_data, pd.DataFrame([new_row])], ignore_index=True)
                return new_df.tail(limit)
            
            else:
                raise ValueError(f"暂不支持 {symbol} {interval}")
                
        except Exception as e:
            logger.error(f"❌ 获取价格数据失败: {e}")
            raise HTTPException(status_code=500, detail=f"数据获取失败: {e}")


class TradingDecisionLogger:
    """交易决策日志记录器"""
    
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        
    async def log_decision(self, decision_data: Dict[str, Any]):
        """记录交易决策到GCS和标准输出"""
        try:
            # 记录到标准输出 (Cloud Logging会自动收集)
            logger.info(f"📊 交易决策: {json.dumps(decision_data, indent=2)}")
            
            # 同时记录到GCS文件 (以日期为单位)
            current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            blob_path = f"logs/paper/{current_date}/decisions.jsonl"
            
            # 转换为JSONL格式
            log_line = json.dumps(decision_data) + "\n"
            
            # 上传到GCS (追加模式)
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            
            # 如果文件已存在，先下载现有内容
            existing_content = ""
            if blob.exists():
                existing_content = blob.download_as_text()
            
            # 追加新内容
            new_content = existing_content + log_line
            blob.upload_from_string(new_content, content_type='text/plain')
            
            logger.info(f"✅ 决策已记录到: gs://{self.bucket_name}/{blob_path}")
            
        except Exception as e:
            logger.error(f"❌ 决策日志记录失败: {e}")


# 全局变量
model_manager: Optional[ModelManager] = None
data_provider: Optional[DataProvider] = None
decision_logger: Optional[TradingDecisionLogger] = None

# 模拟交易状态
current_position = 0.0  # 当前持仓比例
current_equity = 1.0   # 当前净值 (相对于初始资金)
trade_count = 0


# FastAPI应用
app = FastAPI(title="DRL BTC Auto Trading API", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global model_manager, data_provider, decision_logger
    
    logger.info("🚀 启动DRL BTC自动交易服务...")
    
    # 获取环境变量
    bucket_name = os.getenv("GCS_BUCKET_NAME", "your-bucket-name")
    
    # 初始化组件
    model_manager = ModelManager(bucket_name)
    data_provider = DataProvider()
    decision_logger = TradingDecisionLogger(bucket_name)
    
    # 尝试加载最新模型
    try:
        model_loaded = await model_manager.load_latest_model()
        if model_loaded:
            logger.info("✅ 模型已在启动时加载")
        else:
            logger.warning("⚠️ 启动时未能加载模型，将在运行时重试")
    except Exception as e:
        logger.error(f"❌ 启动时模型加载失败: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    gcs_connected = False
    try:
        # 测试GCS连接
        client = storage.Client()
        # 简单的连接测试
        list(client.list_buckets())
        gcs_connected = True
    except:
        pass
    
    return HealthResponse(
        status="healthy" if model_manager and model_manager.is_loaded() else "degraded",
        model_loaded=model_manager.is_loaded() if model_manager else False,
        gcs_connected=gcs_connected,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.post("/tick", response_model=TradingResponse)
async def trading_tick(request: TradingRequest, background_tasks: BackgroundTasks):
    """处理交易决策请求"""
    global current_position, current_equity, trade_count
    
    logger.info(f"📈 收到交易决策请求: {request.symbol} {request.interval}")
    
    try:
        # 检查模型是否已加载
        if not model_manager or not model_manager.is_loaded():
            logger.warning("⚠️ 模型未加载，尝试重新加载...")
            if model_manager:
                model_loaded = await model_manager.load_latest_model()
                if not model_loaded:
                    raise HTTPException(status_code=503, detail="模型未加载，请稍后重试")
            else:
                raise HTTPException(status_code=503, detail="模型管理器未初始化")
        
        # 获取最新价格数据
        candles = await data_provider.get_recent_candles(
            request.symbol, 
            request.interval, 
            request.lookback + 20  # 多获取一些数据以计算技术指标
        )
        
        if len(candles) < request.lookback:
            raise HTTPException(status_code=400, detail=f"可用数据不足，需要至少{request.lookback}根K线")
        
        # 准备观测数据 (与训练环境一致)
        recent_candles = candles.tail(request.lookback)
        
        # 构建特征矩阵
        features = []
        for _, row in recent_candles.iterrows():
            feature_vector = [
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume'],
                row['returns'] if not np.isnan(row['returns']) else 0.0
            ]
            features.append(feature_vector)
        
        obs = np.array(features, dtype=np.float32)
        
        # 归一化处理 (与训练环境保持一致)
        if len(obs) > 0:
            latest_close = obs[-1, 3]
            obs[:, :4] = obs[:, :4] / latest_close - 1.0  # 价格归一化
            obs[:, 4] = np.log1p(obs[:, 4]) / 10.0      # 成交量归一化
        
        # 模型预测
        action, _ = model_manager.model.predict(obs[None, :], deterministic=True)
        target_position = np.clip(action[0], -1.0, 1.0)
        
        # 计算当前价格和时间戳
        current_price = candles.iloc[-1]['close']
        current_timestamp = int(candles.iloc[-1]['timestamp'])
        
        # 更新纸面交易状态
        position_change = abs(target_position - current_position)
        if position_change > 0.01:  # 仓位变化超过1%
            trade_count += 1
        
        # 简化的权益计算 (实际应该基于价格变化)
        price_return = candles.iloc[-1]['returns'] if not np.isnan(candles.iloc[-1]['returns']) else 0.0
        current_equity *= (1 + current_position * price_return - position_change * 0.001)  # 扣除手续费
        
        # 更新持仓
        current_position = target_position
        
        # 构建响应
        response = TradingResponse(
            ts=current_timestamp,
            price=current_price,
            action=float(target_position),
            position=float(current_position),
            equity=float(current_equity),
            note="paper-trade"
        )
        
        # 异步记录决策日志
        decision_data = {
            "timestamp": current_timestamp,
            "symbol": request.symbol,
            "interval": request.interval,
            "price": current_price,
            "action": float(target_position),
            "position": float(current_position),
            "equity": float(current_equity),
            "trade_count": trade_count,
            "model_path": model_manager.model_path,
            "note": "paper-trade"
        }
        
        background_tasks.add_task(decision_logger.log_decision, decision_data)
        
        logger.info(f"✅ 决策完成: 价格={current_price:.2f}, 动作={target_position:.3f}, 持仓={current_position:.3f}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 交易决策失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"交易决策失败: {str(e)}")


@app.get("/status")
async def get_status():
    """获取服务状态信息"""
    return {
        "model_loaded": model_manager.is_loaded() if model_manager else False,
        "model_path": model_manager.model_path if model_manager and model_manager.is_loaded() else None,
        "last_model_update": model_manager.last_update.isoformat() if model_manager and model_manager.last_update else None,
        "current_position": current_position,
        "current_equity": current_equity,
        "trade_count": trade_count,
        "server_time": datetime.now(timezone.utc).isoformat()
    }


@app.post("/reload_model")
async def reload_model():
    """手动重新加载模型"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="模型管理器未初始化")
    
    logger.info("🔄 手动重新加载模型...")
    model_loaded = await model_manager.load_latest_model()
    
    if model_loaded:
        return {"status": "success", "message": "模型重新加载成功", "model_path": model_manager.model_path}
    else:
        raise HTTPException(status_code=500, detail="模型重新加载失败")


if __name__ == "__main__":
    # 本地开发时的启动配置
    port = int(os.getenv("PORT", 8080))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=os.getenv("ENV") == "development"
    )