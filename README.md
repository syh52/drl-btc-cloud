# 🚀 DRL BTC 自动交易系统 MVP

基于深度强化学习(PPO)的比特币自动交易系统，运行在Google Cloud Platform上，实现"训练→部署→纸面单"的完整闭环。

## 📋 项目概述

本项目是一个最小可行产品(MVP)，专注于验证DRL交易策略的可行性。系统使用PPO算法训练交易模型，通过Cloud Run提供推理服务，由Cloud Scheduler每分钟触发进行交易决策记录。

### 🎯 核心目标

- ✅ **训练闭环**: Vertex AI训练PPO模型并保存到GCS
- ✅ **推理闭环**: Cloud Run加载模型进行实时决策
- ✅ **触发闭环**: 每分钟自动触发交易决策
- ✅ **记录闭环**: 完整的纸面交易日志追踪

### 🚫 MVP范围限制

- **仅BTC**: 只支持BTCUSDT交易对
- **仅纸面**: 不连接实盘，仅记录决策
- **仅MLP**: 使用简单MLP网络，不使用LSTM
- **仅CPU**: 训练和推理都使用CPU资源

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Google Cloud Platform                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌───────────────┐    ┌──────────────┐   │
│  │ Vertex AI   │───▶│  GCS Storage  │◀───│ Cloud Run    │   │
│  │ (训练PPO)   │    │  (模型/数据)   │    │ (推理服务)    │   │
│  └─────────────┘    └───────────────┘    └──────────────┘   │
│                             ▲                     ▲         │
│  ┌─────────────┐            │                     │         │
│  │Cloud        │    ┌───────────────┐    ┌──────────────┐   │
│  │Scheduler    │───▶│   Pub/Sub     │───▶│  Eventarc    │   │
│  │(每分钟)     │    │   (消息队列)   │    │  (触发器)    │   │
│  └─────────────┘    └───────────────┘    └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
drl-btc-cloud/
├── train/                    # 训练模块
│   ├── btc_env.py            # BTC交易环境(Gymnasium)
│   ├── train.py              # PPO训练脚本
│   └── submit_job.py         # Vertex AI任务提交
├── app/                      # 推理服务
│   ├── main.py               # FastAPI应用
│   └── requirements.txt      # 推理依赖
├── infra/                    # 基础设施
│   └── deploy.sh             # 一键部署脚本
├── requirements.txt          # 完整项目依赖
├── config.yaml               # 配置文件
├── Makefile                  # 常用命令
└── README.md                 # 项目文档

注: app/ 目录下还有独立的 requirements.txt，用于 Cloud Run 部署
```

## 🛠️ 快速开始

### 先决条件

- Google Cloud SDK (`gcloud`)
- Python 3.9+ (推荐3.12+)
- Docker (用于Cloud Run部署)
- GCP项目 (已启用计费)

### 1. 克隆并安装

```bash
git clone <repository-url>
cd drl-btc-cloud

# 安装依赖 (推荐使用 pip3)
make install
# 或
pip3 install -r requirements.txt

# 注意: 建议在虚拟环境中安装
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
pip3 install -r requirements.txt
```

### 2. 环境检查

```bash
# 运行冒烟测试
make smoke-test

# 或单独测试交易环境  
cd train && python3 btc_env.py
```

### 3. 一键部署到GCP

```bash
# 运行部署脚本 (交互式配置)
make deploy
# 或
cd infra && ./deploy.sh
```

部署脚本会自动：
- 启用必需的GCP API
- 创建GCS存储桶
- 部署Cloud Run服务
- 配置Pub/Sub消息队列
- 设置Cloud Scheduler定时任务
- 创建Eventarc触发器

### 4. 训练模型

```bash
# 提交Vertex AI训练任务
make train-submit PROJECT_ID=your-project-id BUCKET=your-bucket-name

# 或本地训练 (小规模测试)
make train

# 注意: 需要先安装依赖
make install
```

### 5. 启动自动交易

```bash
# 手动触发一次 (测试)
curl -X POST https://your-service-url/tick \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT","interval":"1m"}'

# 启动定时任务 (每分钟自动)
gcloud scheduler jobs run every-minute --location=asia-southeast1
```

## 🔍 监控和管理

### 系统状态检查

```bash
# 检查部署状态
make status PROJECT_ID=your-project-id

# 查看服务日志
make logs PROJECT_ID=your-project-id

# 实时监控
make monitor PROJECT_ID=your-project-id
```

### API接口测试

```bash
# 健康检查
curl https://your-service-url/health

# 获取服务状态
curl https://your-service-url/status

# 手动触发交易决策
curl -X POST https://your-service-url/tick \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT","interval":"1m","lookback":60}'
```

### Web界面监控

- **Cloud Run**: https://console.cloud.google.com/run
- **Cloud Logging**: https://console.cloud.google.com/logs
- **Cloud Scheduler**: https://console.cloud.google.com/cloudscheduler
- **Vertex AI**: https://console.cloud.google.com/vertex-ai

## 📊 核心组件详解

### 1. BTC交易环境 (`btc_env.py`)

基于Gymnasium标准的强化学习环境：

- **观测空间**: `[lookback × 5features]` 历史价格矩阵  
- **动作空间**: `[-1, 1]` 目标持仓比例
- **奖励函数**: `持仓收益 - 交易手续费`
- **特征工程**: OHLCV + 收益率 (当前版本)

```python
# 环境使用示例
from btc_env import BTCTradingEnv

env = BTCTradingEnv(
    data_path="btc_data.csv",
    lookback=60,
    fee_rate=0.001
)

obs, info = env.reset()
action = [0.5]  # 50%持仓
obs, reward, done, truncated, info = env.step(action)
```

### 2. PPO训练 (`train.py`)

使用Stable-Baselines3实现的PPO算法：

- **网络结构**: MLP (多层感知机)
- **训练步数**: 100K (可配置)
- **批次大小**: 64
- **学习率**: 3e-4

```bash
# 本地训练
python train.py --data_csv btc_data.csv --timesteps 100000

# Vertex AI训练
python3 submit_job.py --project_id your-project --bucket your-bucket
```

### 3. 推理服务 (`main.py`)

FastAPI Web服务，提供交易决策API：

- **模型管理**: 自动从GCS加载最新模型
- **数据获取**: 模拟BTC价格数据 (可扩展到真实数据)
- **决策记录**: 同步到Cloud Logging和GCS
- **健康监控**: 实时状态检查

```python
# API响应格式
{
  "ts": 1699999999,          # 时间戳
  "price": 65000.0,          # 当前价格
  "action": 0.35,            # 模型决策
  "position": 0.35,          # 目标持仓
  "equity": 1.0021,          # 当前净值
  "note": "paper-trade"      # 标记
}
```

### 4. 自动化部署 (`deploy.sh`)

一键部署脚本包含：

- GCP项目配置和API启用
- GCS存储桶创建和目录结构
- Cloud Run服务部署 (容器化)
- Pub/Sub消息队列配置
- Eventarc触发器设置
- Cloud Scheduler定时任务

## ⚙️ 配置说明

### 环境变量

```bash
# GCP配置
export PROJECT_ID="your-project-id"
export REGION="asia-southeast1"
export BUCKET_NAME="your-bucket-name"

# 服务配置
export GCS_BUCKET_NAME="your-bucket-name"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### 配置文件 (`config.yaml`)

主要配置项：

```yaml
# 训练配置
training:
  total_timesteps: 100000
  lookback: 60
  fee_rate: 0.001

# 推理配置
inference:
  api:
    port: 8080
  data_source:
    default: "mock"

# GCP配置
gcp:
  region: "asia-southeast1"
  cloud_run:
    memory: "2Gi"
    cpu: "2"
```

## 📈 性能和成本

### 预期性能指标

- **训练时间**: 10-30分钟 (100K步，CPU)
- **推理延迟**: < 200ms (不含冷启动)
- **触发频率**: 每分钟1次
- **数据存储**: ~1MB/天 (决策日志)

### 成本估算 (每月)

- **Vertex AI训练**: ~$5-10 (偶尔训练)
- **Cloud Run**: ~$5-15 (按请求计费)
- **GCS存储**: ~$1-3 (模型和日志)
- **Pub/Sub**: ~$1 (消息传递)
- **总计**: **~$12-30/月**

## 🔧 开发指南

### 本地开发

```bash
# 启动开发环境  
make dev

# 运行单元测试
make test

# 代码检查
python3 train/btc_env.py  # 环境检查
```

### 自定义扩展

1. **新增交易对**: 修改 `config.yaml` 中的 `symbols`
2. **新增特征**: 在 `btc_env.py` 的 `_calculate_features` 中添加
3. **新增数据源**: 在 `main.py` 的 `DataProvider` 中扩展
4. **新增算法**: 替换 `train.py` 中的PPO为其他算法

### 调试技巧

```bash
# 查看详细日志
gcloud logging read 'resource.type=cloud_run_revision' --limit=100

# 本地测试API
cd app && python3 main.py

# 检查模型文件
gsutil ls gs://your-bucket/models/ppo/

# 手动触发训练
python3 train/submit_job.py --project_id your-project --bucket your-bucket
```

## 🚨 故障排查

### 常见问题

1. **模型加载失败**
   - 检查GCS权限
   - 确认模型文件存在
   - 查看Cloud Run日志

2. **训练任务失败**
   - 检查Vertex AI配额
   - 验证数据格式
   - 查看训练日志

3. **API响应超时**
   - 检查模型文件大小
   - 增加Cloud Run内存
   - 优化数据处理逻辑

4. **触发器不工作**
   - 确认Cloud Scheduler启用
   - 检查Pub/Sub权限
   - 验证Eventarc配置

### 日志分析

```bash
# Cloud Run错误日志
gcloud logging read 'resource.type=cloud_run_revision AND severity=ERROR'

# Vertex AI训练日志
gcloud logging read 'resource.type=aiplatform.googleapis.com/CustomJob'

# Scheduler执行日志
gcloud logging read 'resource.type=cloud_scheduler_job'
```

## 🔮 未来路线图

### 短期计划 (1-2个月)

- [ ] 接入Binance Testnet真实数据
- [ ] 添加更多技术指标特征
- [ ] 实现LSTM/GRU网络结构
- [ ] 增强风险管理机制

### 中期计划 (3-6个月)

- [ ] 多币种支持 (ETH, BNB等)
- [ ] 高级算法 (SAC, A2C, DQN)
- [ ] 实盘连接和资金管理
- [ ] Web仪表板和可视化

### 长期计划 (6个月+)

- [ ] 分布式训练和推理
- [ ] 多策略组合优化
- [ ] 机器学习运维(MLOps)
- [ ] 量化风控和回测系统

## 📝 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献指南

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 提交Pull Request

## 📞 联系方式

- 项目维护者: [Your Name]
- 邮箱: [your.email@example.com]
- 项目主页: [GitHub Repository URL]

## 🙏 致谢

- **Stable-Baselines3**: 强化学习算法库
- **Google Cloud Platform**: 云计算基础设施
- **FastAPI**: 高性能Web框架
- **Gymnasium**: 强化学习环境标准

---

⚡ **重要提醒**: 本项目仅用于教育和研究目的，不构成投资建议。加密货币交易具有高风险，请谨慎使用。