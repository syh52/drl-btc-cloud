from fastapi import FastAPI
from pydantic import BaseModel
import json
from datetime import datetime, timezone
import numpy as np
import os
import logging
from typing import Optional, List
from google.cloud import storage
from fastapi.responses import HTMLResponse
import glob
import pandas as pd

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
            bucket_name = os.getenv('GCS_BUCKET_NAME', os.getenv('BUCKET_NAME', 'ai4fnew-drl-btc-20250827'))
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
            bucket_name = os.getenv('GCS_BUCKET_NAME', os.getenv('BUCKET_NAME', 'ai4fnew-drl-btc-20250827'))
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
                with open(log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line.strip())
                                data_points.append({
                                    "timestamp": data["timestamp"],
                                    "datetime": data["datetime"],
                                    "price": data["price"],
                                    "action": data["action"],
                                    "position": data["position"],
                                    "equity": data["equity"],
                                    "model_version": data.get("model_version", "unknown")
                                })
                            except json.JSONDecodeError:
                                continue
        
        # 如果没有本地数据，生成一些模拟数据用于演示
        if not data_points:
            logger.info("No local data found, generating mock data for dashboard")
            base_time = int(datetime.now(timezone.utc).timestamp()) - 3600  # 从1小时前开始
            for i in range(min(limit, 60)):  # 生成最多60分钟的数据
                timestamp = base_time + (i * 60)
                price = 65000 + np.random.normal(0, 500)
                action = np.random.uniform(-0.8, 0.8)
                equity = 1.0 + (np.random.random() - 0.5) * 0.02  # ±1%波动
                
                data_points.append({
                    "timestamp": timestamp,
                    "datetime": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
                    "price": round(price, 2),
                    "action": round(action, 4),
                    "position": round(action, 4),
                    "equity": round(equity, 6),
                    "model_version": model_version or "demo"
                })
        
        # 按时间排序并限制数量
        data_points.sort(key=lambda x: x["timestamp"])
        data_points = data_points[-limit:]
        
        return {
            "data": data_points,
            "total_points": len(data_points),
            "model_version": model_version,
            "data_source": "local_logs" if glob.glob("logs/decisions/*.jsonl") else "mock_data"
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent data: {e}")
        return {"error": str(e), "data": []}

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
            .loading {
                text-align: center;
                color: #888;
                padding: 50px;
                font-size: 1.1em;
            }
            .error {
                text-align: center;
                color: #ff6b6b;
                padding: 30px;
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
            let dashboardData = [];

            // 初始化图表
            function initCharts() {
                const priceCtx = document.getElementById('priceChart').getContext('2d');
                const equityCtx = document.getElementById('equityChart').getContext('2d');

                // 价格图表配置
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
                            pointHoverRadius: 8,
                            showLine: false,
                            pointStyle: 'triangle'
                        }, {
                            label: '卖出信号',
                            data: [],
                            backgroundColor: '#f87171',
                            borderColor: '#f87171',
                            pointRadius: 6,
                            pointHoverRadius: 8,
                            showLine: false,
                            pointStyle: 'triangleDown'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { 
                                labels: { color: '#cccccc' }
                            }
                        },
                        scales: {
                            x: { 
                                ticks: { color: '#888' },
                                grid: { color: '#333' }
                            },
                            y: { 
                                ticks: { color: '#888' },
                                grid: { color: '#333' }
                            }
                        }
                    }
                });

                // 权益图表配置
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
                        plugins: {
                            legend: { 
                                labels: { color: '#cccccc' }
                            }
                        },
                        scales: {
                            x: { 
                                ticks: { color: '#888' },
                                grid: { color: '#333' }
                            },
                            y: { 
                                ticks: { color: '#888' },
                                grid: { color: '#333' }
                            }
                        }
                    }
                });
            }

            // 更新图表数据
            function updateCharts(data) {
                if (!data || data.length === 0) return;

                const labels = data.map(d => new Date(d.datetime).toLocaleTimeString('zh-CN', {
                    hour: '2-digit',
                    minute: '2-digit'
                }));

                const prices = data.map(d => d.price);
                const equities = data.map(d => d.equity);
                const buySignals = [];
                const sellSignals = [];

                // 分离买卖信号
                data.forEach((d, i) => {
                    if (d.action > 0.1) {
                        buySignals.push({x: i, y: d.price});
                    } else if (d.action < -0.1) {
                        sellSignals.push({x: i, y: d.price});
                    }
                });

                // 更新价格图表
                priceChart.data.labels = labels;
                priceChart.data.datasets[0].data = prices;
                priceChart.data.datasets[1].data = buySignals;
                priceChart.data.datasets[2].data = sellSignals;
                priceChart.update();

                // 更新权益图表
                equityChart.data.labels = labels;
                equityChart.data.datasets[0].data = equities;
                equityChart.update();

                // 更新统计数据
                updateStats(data);
            }

            // 更新统计数据
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

            // 加载数据
            async function loadData() {
                try {
                    const response = await fetch('/recent?limit=200');
                    const result = await response.json();
                    
                    if (result.error) {
                        throw new Error(result.error);
                    }

                    dashboardData = result.data || [];
                    
                    // 更新状态栏
                    document.getElementById('model-status').textContent = '🤖 模型已加载';
                    document.getElementById('data-status').textContent = `📊 数据: ${result.total_points} 条`;
                    document.getElementById('version-status').textContent = `🔖 ${result.model_version || 'unknown'}`;

                    // 更新图表
                    updateCharts(dashboardData);

                } catch (error) {
                    console.error('加载数据失败:', error);
                    document.getElementById('model-status').textContent = '❌ 数据加载失败';
                    document.getElementById('data-status').textContent = '❌ 无数据';
                    document.getElementById('version-status').textContent = '❌ 未知版本';
                }
            }

            // 初始化应用
            async function initApp() {
                initCharts();
                await loadData();
                
                // 定时刷新数据 (每30秒)
                setInterval(loadData, 30000);
            }

            // 页面加载完成后初始化
            document.addEventListener('DOMContentLoaded', initApp);
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main_simple:app", host="0.0.0.0", port=port, log_level="info")
