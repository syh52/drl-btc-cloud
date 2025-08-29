import asyncio
import glob
import json
import logging
import os

# 导入训练组件 (动态路径处理)
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import ccxt
import numpy as np
import pandas as pd
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from google.cloud import logging as cloud_logging
from google.cloud import secretmanager, storage
from pydantic import BaseModel

# 安全的路径处理
app_root = Path(__file__).parent.parent
train_path = app_root / "train"
if train_path.exists():
    sys.path.append(str(train_path))
else:
    # 备用路径 (容器环境)
    sys.path.append("/app/train")

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
if os.getenv("GOOGLE_CLOUD_PROJECT"):
    client = cloud_logging.Client()
    client.setup_logging()


# API请求/响应模型
class TradingRequest(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    lookback: int = 60
    source: str = "binance"  # binance, binance_testnet


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
            model_files = [blob for blob in blobs if blob.name.endswith(".zip")]

            if not model_files:
                logger.warning(
                    f"⚠️ 在 gs://{self.bucket_name}/{self.model_prefix} 中未找到模型文件"
                )
                return False

            # 按时间排序，获取最新模型
            latest_blob = max(model_files, key=lambda b: b.time_created)
            model_gcs_path = f"gs://{self.bucket_name}/{latest_blob.name}"

            logger.info(f"📦 找到最新模型: {model_gcs_path}")

            # 下载模型到本地临时目录 (安全路径处理)
            import tempfile

            temp_dir = Path(tempfile.gettempdir())
            model_filename = Path(latest_blob.name).name  # 安全获取文件名
            local_model_path = temp_dir / model_filename

            latest_blob.download_to_filename(str(local_model_path))

            logger.info(f"⬇️ 模型已下载到: {local_model_path}")

            # 加载PPO模型
            self.model = PPO.load(str(local_model_path))
            self.model_path = model_gcs_path
            self.last_update = datetime.now(timezone.utc)

            logger.info(f"✅ 模型加载成功: {model_gcs_path}")

            # 注意: 清理操作移到finally块中处理

            return True

        except storage.exceptions.NotFound as e:
            logger.error(f"❌ GCS存储桶或路径不存在: {e}")
            return False
        except storage.exceptions.Forbidden as e:
            logger.error(f"❌ GCS访问权限不足: {e}")
            return False
        except FileNotFoundError as e:
            logger.error(f"❌ 模型文件下载失败: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ 模型加载未知错误: {type(e).__name__}: {e}")
            import traceback

            logger.debug(f"详细错误信息: {traceback.format_exc()}")
            return False
        finally:
            # 确保临时文件被清理
            if "local_model_path" in locals() and local_model_path.exists():
                try:
                    local_model_path.unlink()
                    logger.debug(f"🧹 已清理临时文件: {local_model_path}")
                except OSError as e:
                    logger.warning(f"⚠️ 临时文件清理失败: {e}")

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None


class DataProvider:
    """数据提供器 - 从Cloud Run获取历史数据，CCXT获取实时价格"""

    def __init__(self):
        self.exchange = None
        self.cloud_run_url = os.getenv("CLOUD_RUN_DATA_URL")
        self.last_price = None
        self.last_price_update = None

    async def initialize_exchange(self):
        """初始化CCXT交易所（仅用于实时价格）"""
        if self.exchange:
            return

        try:
            logger.info("🔗 初始化CCXT交易所（实时价格）")

            config = {
                "enableRateLimit": True,
                "timeout": 30000,
                "sandbox": False,
            }

            proxy_url = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
            if proxy_url:
                logger.info(f"🌐 使用代理: {proxy_url}")

            self.exchange = ccxt.binance(config)

            if not self.exchange.has.get("fetchTicker"):
                raise Exception("交易所不支持ticker数据")

            logger.info("✅ 交易所初始化成功")

        except Exception as e:
            logger.error(f"❌ 交易所初始化失败: {e}")
            raise

    async def get_current_price(self) -> float:
        """获取当前BTC价格"""
        # 缓存机制：每分钟最多更新一次
        now = datetime.now()
        if (
            self.last_price_update
            and (now - self.last_price_update).total_seconds() < 60
            and self.last_price
        ):
            return self.last_price

        if not self.exchange:
            await self.initialize_exchange()

        ticker = await asyncio.to_thread(self.exchange.fetch_ticker, "BTC/USDT")

        if not ticker or not ticker.get("last"):
            raise Exception("无法获取实时价格数据")

        self.last_price = float(ticker["last"])
        self.last_price_update = now

        logger.info(f"✅ 获取实时价格: ${self.last_price:.2f}")
        return self.last_price

    async def get_recent_candles_from_cloud_run(
        self, symbol: str, interval: str, limit: int = 100
    ) -> pd.DataFrame:
        """从GCS获取历史K线数据（直接使用已存在的数据文件）"""
        try:
            bucket_name = os.getenv("BUCKET_NAME", "ai4fnew-drl-btc-20250827")

            # 映射时间间隔
            interval_mapping = {
                "1m": "1m",
                "5m": "5m",
                "15m": "15m",
                "1h": "1h",
                "4h": "4h",
                "1d": "1d",
            }

            mapped_interval = interval_mapping.get(interval, "5m")
            gcs_path = f"data/btc_data_{mapped_interval}_540d.csv"

            logger.info(f"📥 从GCS读取历史数据: gs://{bucket_name}/{gcs_path}")

            # 从GCS读取数据
            from google.cloud import storage

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(gcs_path)

            if not blob.exists():
                raise Exception(f"GCS数据文件不存在: gs://{bucket_name}/{gcs_path}")

            # 下载CSV数据
            csv_content = blob.download_as_text()

            # 使用StringIO读取CSV
            from io import StringIO

            df = pd.read_csv(StringIO(csv_content))

            # 确保有datetime列
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            else:
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

            # 排序并取最新的数据
            df = df.sort_values("datetime")
            df = df.tail(limit)

            # 确保所需的列存在
            required_columns = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "returns",
            ]
            for col in required_columns:
                if col not in df.columns:
                    if col == "timestamp":
                        df["timestamp"] = (
                            df["datetime"].astype(int) // 10**6
                        )  # 转换为毫秒时间戳
                    else:
                        df[col] = 0.0

            # 重新计算技术指标（确保与训练环境一致）
            df["returns"] = df["close"].pct_change()
            df["ma_20"] = df["close"].rolling(20, min_periods=1).mean()
            df["ma_ratio"] = df["close"] / df["ma_20"] - 1

            df = df.fillna(0)

            logger.info(
                f"✅ 从GCS获取 {len(df)} 条历史K线数据，时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}"
            )
            return df

        except Exception as e:
            logger.error(f"❌ 从GCS获取历史数据失败: {e}")
            raise


class TradingDecisionLogger:
    """交易决策日志记录器 - 本地和GCS双重记录"""

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name

    async def log_decision(self, decision_data: Dict[str, Any]):
        """记录交易决策到本地和GCS"""
        try:
            # 生成文件名（按天分割）
            date_str = datetime.fromtimestamp(decision_data["timestamp"]).strftime(
                "%Y%m%d"
            )
            log_filename = f"{date_str}.jsonl"

            # 本地日志目录
            local_log_dir = "logs/decisions"
            os.makedirs(local_log_dir, exist_ok=True)
            local_log_path = os.path.join(local_log_dir, log_filename)

            # 幂等性检查：按分钟检查是否已记录
            minute_key = decision_data["timestamp"] // 60 * 60

            if os.path.exists(local_log_path):
                with open(local_log_path, "r") as f:
                    for line in f:
                        if line.strip():
                            existing_data = json.loads(line.strip())
                            existing_minute = existing_data["timestamp"] // 60 * 60
                            if existing_minute == minute_key:
                                logger.info(
                                    f"Decision for minute {minute_key} already logged"
                                )
                                return

            # 写入本地文件
            with open(local_log_path, "a") as f:
                f.write(json.dumps(decision_data) + "\n")

            # 同时备份到GCS
            try:
                client = storage.Client()
                bucket = client.bucket(self.bucket_name)

                gcs_log_path = f"logs/decisions/{log_filename}"
                blob = bucket.blob(gcs_log_path)

                # 上传本地文件到GCS
                with open(local_log_path, "r") as f:
                    blob.upload_from_file(f, content_type="text/plain")

                logger.info(f"✅ 决策记录到: {local_log_path} + GCS")

            except Exception as gcs_error:
                logger.warning(f"GCS备份失败: {gcs_error}")

        except Exception as e:
            logger.error(f"❌ 决策日志记录失败: {e}")
            raise


# 全局变量
model_manager: Optional[ModelManager] = None
data_provider: Optional[DataProvider] = None
decision_logger: Optional[TradingDecisionLogger] = None

# 模拟交易状态
current_position = 0.0  # 当前持仓比例
current_equity = 1.0  # 当前净值 (相对于初始资金)
trade_count = 0


# FastAPI应用
app = FastAPI(title="DRL BTC Auto Trading API", version="1.0.0")


@app.get("/")
async def root():
    return {
        "message": "🚀 DRL BTC 自动交易系统",
        "version": "1.0.0",
        "description": "基于深度强化学习的比特币自动交易系统",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "trade": "/tick",
            "dashboard": "/dashboard",
            "recent_data": "/recent",
            "reload_model": "/reload_model",
            "docs": "/docs",
        },
        "features": [
            "PPO深度强化学习模型",
            "Cloud Run历史数据",
            "实时Dashboard",
            "本地+GCS日志",
        ],
        "model_status": (
            "已加载" if model_manager and model_manager.is_loaded() else "未加载"
        ),
    }


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global model_manager, data_provider, decision_logger

    logger.info("🚀 启动DRL BTC自动交易服务...")

    # 获取环境变量
    bucket_name = os.getenv(
        "GCS_BUCKET_NAME", os.getenv("BUCKET_NAME", "ai4fnew-drl-btc-20250827")
    )

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
        timestamp=datetime.now(timezone.utc).isoformat(),
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
                    raise HTTPException(
                        status_code=503, detail="模型未加载，请稍后重试"
                    )
            else:
                raise HTTPException(status_code=503, detail="模型管理器未初始化")

        # 获取最新价格数据
        candles = await data_provider.get_recent_candles_from_cloud_run(
            request.symbol,
            request.interval,
            request.lookback + 20,  # 多获取一些数据以计算技术指标
        )

        if len(candles) < request.lookback:
            raise HTTPException(
                status_code=400, detail=f"可用数据不足，需要至少{request.lookback}根K线"
            )

        # 准备观测数据 (与训练环境一致)
        recent_candles = candles.tail(request.lookback)

        # 构建特征矩阵
        features = []
        for _, row in recent_candles.iterrows():
            feature_vector = [
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
                row["returns"] if not np.isnan(row["returns"]) else 0.0,
            ]
            features.append(feature_vector)

        obs = np.array(features, dtype=np.float32)

        # 归一化处理 (与训练环境保持一致)
        if len(obs) > 0:
            latest_close = obs[-1, 3]
            obs[:, :4] = obs[:, :4] / latest_close - 1.0  # 价格归一化
            obs[:, 4] = np.log1p(obs[:, 4]) / 10.0  # 成交量归一化

        # 模型预测
        action, _ = model_manager.model.predict(obs[None, :], deterministic=True)
        target_position = np.clip(action[0], -1.0, 1.0)

        # 计算当前价格和时间戳
        current_price = candles.iloc[-1]["close"]
        current_timestamp = int(candles.iloc[-1]["timestamp"])

        # 更新纸面交易状态
        position_change = abs(target_position - current_position)
        if position_change > 0.01:  # 仓位变化超过1%
            trade_count += 1

        # 简化的权益计算 (实际应该基于价格变化)
        price_return = (
            candles.iloc[-1]["returns"]
            if not np.isnan(candles.iloc[-1]["returns"])
            else 0.0
        )
        current_equity *= (
            1 + current_position * price_return - position_change * 0.001
        )  # 扣除手续费

        # 更新持仓
        current_position = target_position

        # 构建响应
        response = TradingResponse(
            ts=current_timestamp,
            price=current_price,
            action=float(target_position),
            position=float(current_position),
            equity=float(current_equity),
            note="paper-trade",
        )

        # 异步记录决策日志
        decision_data = {
            "timestamp": current_timestamp,
            "datetime": datetime.fromtimestamp(
                current_timestamp, timezone.utc
            ).isoformat(),
            "symbol": request.symbol,
            "interval": request.interval,
            "price": current_price,
            "action": float(target_position),
            "position": float(current_position),
            "equity": float(current_equity),
            "trade_count": trade_count,
            "model_path": model_manager.model_path,
            "note": "paper-trade",
        }

        background_tasks.add_task(decision_logger.log_decision, decision_data)

        logger.info(
            f"✅ 决策完成: 价格={current_price:.2f}, 动作={target_position:.3f}, 持仓={current_position:.3f}"
        )

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
        "model_path": (
            model_manager.model_path
            if model_manager and model_manager.is_loaded()
            else None
        ),
        "last_model_update": (
            model_manager.last_update.isoformat()
            if model_manager and model_manager.last_update
            else None
        ),
        "current_position": current_position,
        "current_equity": current_equity,
        "trade_count": trade_count,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/reload_model")
async def reload_model():
    """手动重新加载模型"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="模型管理器未初始化")

    logger.info("🔄 手动重新加载模型...")
    model_loaded = await model_manager.load_latest_model()

    if model_loaded:
        return {
            "status": "success",
            "message": "模型重新加载成功",
            "model_path": model_manager.model_path,
        }
    else:
        raise HTTPException(status_code=500, detail="模型重新加载失败")


@app.get("/recent")
async def get_recent_data(limit: int = 500):
    """获取最近的决策数据用于Dashboard展示"""
    try:
        data_points = []

        # 读取本地决策日志文件
        log_files = glob.glob("logs/decisions/*.jsonl")
        log_files.sort(reverse=True)  # 最新的文件在前

        for log_file in log_files[:3]:  # 最多读取最近3天的数据
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line.strip())
                                data_points.append(
                                    {
                                        "timestamp": data["timestamp"],
                                        "datetime": data["datetime"],
                                        "price": data["price"],
                                        "action": data["action"],
                                        "position": data["position"],
                                        "equity": data["equity"],
                                        "model_version": data.get(
                                            "model_version", "unknown"
                                        ),
                                    }
                                )
                            except json.JSONDecodeError:
                                continue

        # 按时间排序并限制数量
        data_points.sort(key=lambda x: x["timestamp"])
        data_points = data_points[-limit:]

        return {
            "data": data_points,
            "total_points": len(data_points),
            "model_version": (
                model_manager.model_path
                if model_manager and model_manager.is_loaded()
                else "unknown"
            ),
            "data_source": "local_logs" if data_points else "no_data",
        }

    except Exception as e:
        logger.error(f"Failed to get recent data: {e}")
        raise HTTPException(status_code=500, detail=f"获取历史数据失败: {str(e)}")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """极简Dashboard页面"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DRL BTC 自动交易系统 Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f0f23;
                color: #cccccc;
                overflow-x: hidden;
            }
            .header {
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 20px;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.5);
            }
            .header h1 { color: #00d4aa; font-size: 2.2em; margin-bottom: 10px; }
            .status-bar {
                display: flex;
                justify-content: center;
                gap: 30px;
                margin-top: 15px;
                flex-wrap: wrap;
            }
            .status-item {
                background: rgba(0, 212, 170, 0.1);
                padding: 8px 16px;
                border-radius: 20px;
                border: 1px solid #00d4aa;
                font-size: 0.9em;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            .chart-container {
                background: linear-gradient(145deg, #1e1e1e, #2d2d2d);
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                border: 1px solid #333;
            }
            .chart-title {
                color: #00d4aa;
                font-size: 1.3em;
                margin-bottom: 15px;
                text-align: center;
                font-weight: 600;
            }
            .chart-wrapper {
                position: relative;
                height: 400px;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 10px;
                margin-top: 15px;
            }
            .stat-item {
                background: rgba(0, 212, 170, 0.05);
                padding: 10px;
                border-radius: 8px;
                text-align: center;
                border: 1px solid rgba(0, 212, 170, 0.2);
            }
            .stat-value {
                font-size: 1.2em;
                font-weight: bold;
                color: #00d4aa;
            }
            .stat-label {
                font-size: 0.8em;
                color: #888;
                margin-top: 5px;
            }
            @media (max-width: 768px) {
                .container { grid-template-columns: 1fr; }
                .status-bar { gap: 15px; }
                .header h1 { font-size: 1.8em; }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 DRL BTC 自动交易系统</h1>
            <div class="status-bar">
                <div class="status-item" id="model-status">📡 模型加载中...</div>
                <div class="status-item" id="data-status">📊 数据加载中...</div>
                <div class="status-item" id="version-status">🔖 版本检查中...</div>
            </div>
        </div>

        <div class="container">
            <div class="chart-container">
                <div class="chart-title">📈 BTC价格 & AI决策信号</div>
                <div class="chart-wrapper">
                    <canvas id="priceChart"></canvas>
                </div>
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-value" id="current-price">--</div>
                        <div class="stat-label">当前价格</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="price-change">--</div>
                        <div class="stat-label">24h变化</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="total-trades">--</div>
                        <div class="stat-label">总交易数</div>
                    </div>
                </div>
            </div>

            <div class="chart-container">
                <div class="chart-title">💹 账户权益曲线</div>
                <div class="chart-wrapper">
                    <canvas id="equityChart"></canvas>
                </div>
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-value" id="current-equity">--</div>
                        <div class="stat-label">当前权益</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="total-return">--</div>
                        <div class="stat-label">总收益率</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="current-position">--</div>
                        <div class="stat-label">当前仓位</div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let priceChart, equityChart;

            function initCharts() {
                const priceCtx = document.getElementById('priceChart').getContext('2d');
                const equityCtx = document.getElementById('equityChart').getContext('2d');

                priceChart = new Chart(priceCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'BTC价格',
                            data: [],
                            borderColor: '#00d4aa',
                            backgroundColor: 'rgba(0, 212, 170, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.4
                        }, {
                            label: '买入信号',
                            data: [],
                            backgroundColor: '#4ade80',
                            borderColor: '#4ade80',
                            pointRadius: 6,
                            showLine: false,
                            pointStyle: 'triangle'
                        }, {
                            label: '卖出信号',
                            data: [],
                            backgroundColor: '#f87171',
                            borderColor: '#f87171',
                            pointRadius: 6,
                            showLine: false,
                            pointStyle: 'triangleDown'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { labels: { color: '#cccccc' } } },
                        scales: {
                            x: { ticks: { color: '#888' }, grid: { color: '#333' } },
                            y: { ticks: { color: '#888' }, grid: { color: '#333' } }
                        }
                    }
                });

                equityChart = new Chart(equityCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: '账户权益',
                            data: [],
                            borderColor: '#fbbf24',
                            backgroundColor: 'rgba(251, 191, 36, 0.1)',
                            borderWidth: 3,
                            fill: true,
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { labels: { color: '#cccccc' } } },
                        scales: {
                            x: { ticks: { color: '#888' }, grid: { color: '#333' } },
                            y: { ticks: { color: '#888' }, grid: { color: '#333' } }
                        }
                    }
                });
            }

            function updateCharts(data) {
                if (!data || data.length === 0) return;

                const labels = data.map(d => {
                    const date = new Date(d.datetime || d.timestamp * 1000);
                    return date.toLocaleTimeString('zh-CN', {
                        hour: '2-digit',
                        minute: '2-digit'
                    });
                });

                const prices = data.map(d => parseFloat(d.price));
                const equities = data.map(d => parseFloat(d.equity));
                const buySignals = [];
                const sellSignals = [];

                data.forEach((d, i) => {
                    const action = parseFloat(d.action);
                    if (action > 0.3) {
                        buySignals.push({x: i, y: parseFloat(d.price)});
                    } else if (action < -0.3) {
                        sellSignals.push({x: i, y: parseFloat(d.price)});
                    }
                });

                priceChart.data.labels = labels;
                priceChart.data.datasets[0].data = prices;
                priceChart.data.datasets[1].data = buySignals;
                priceChart.data.datasets[2].data = sellSignals;
                priceChart.update();

                equityChart.data.labels = labels;
                equityChart.data.datasets[0].data = equities;
                equityChart.update();

                updateStats(data);
            }

            function updateStats(data) {
                if (data.length === 0) return;

                const latest = data[data.length - 1];
                const first = data[0];
                
                document.getElementById('current-price').textContent = '$' + latest.price.toLocaleString();
                document.getElementById('current-equity').textContent = latest.equity.toFixed(6);
                document.getElementById('current-position').textContent = latest.position.toFixed(4);

                const priceChange = ((latest.price - first.price) / first.price * 100).toFixed(2);
                document.getElementById('price-change').textContent = priceChange + '%';
                
                const totalReturn = ((latest.equity - first.equity) / first.equity * 100).toFixed(2);
                document.getElementById('total-return').textContent = totalReturn + '%';

                const totalTrades = data.filter(d => Math.abs(d.action) > 0.1).length;
                document.getElementById('total-trades').textContent = totalTrades;
            }

            async function loadData() {
                try {
                    const response = await fetch('/recent?limit=200');
                    const result = await response.json();
                    
                    if (result.error) {
                        throw new Error(result.error);
                    }

                    document.getElementById('model-status').textContent = '🤖 PPO模型已加载';
                    document.getElementById('data-status').textContent = `📊 数据: ${result.total_points} 条`;
                    document.getElementById('version-status').textContent = `🔖 ${result.model_version || 'unknown'}`;

                    updateCharts(result.data || []);

                } catch (error) {
                    console.error('加载数据失败:', error);
                    document.getElementById('model-status').textContent = '❌ 数据加载失败';
                    document.getElementById('data-status').textContent = '❌ 无数据';
                    document.getElementById('version-status').textContent = '❌ 未知版本';
                }
            }

            async function initApp() {
                initCharts();
                await loadData();
                setInterval(loadData, 30000);
            }

            document.addEventListener('DOMContentLoaded', initApp);
        </script>
    </body>
    </html>
    """
    return html_content


if __name__ == "__main__":
    # 本地开发时的启动配置
    port = int(os.getenv("PORT", 8080))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=os.getenv("ENV") == "development",
    )