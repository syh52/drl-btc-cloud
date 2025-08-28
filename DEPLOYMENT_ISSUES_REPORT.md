# Cloud Run 部署问题分析报告

## 📋 项目概况

- **项目名称**: DRL BTC 自动交易系统
- **服务名称**: drl-trader
- **项目ID**: ai4fnew
- **部署区域**: asia-southeast1
- **当前状态**: 部署持续失败，运行旧版本

## 🚨 核心问题

### 1. 部署超时问题
**症状**: 所有 `gcloud run deploy` 命令都在2-10分钟后超时
```
Error: Command timed out after 2m 0.0s Building using Dockerfile and deploying container to Cloud Run service [drl-trader]
Building and deploying...
Uploading sources................done
```

**影响**: 新代码无法部署到生产环境

### 2. 版本未更新问题
**当前生产版本**:
- Revision: `drl-trader-00001-njk`
- 部署时间: 2025-08-25 18:35:43 UTC (3天前的旧版本)
- 健康检查: `{"status":"healthy","model_loaded":false,"gcs_connected":true}`

**期望版本**:
- 包含模型加载功能
- 健康检查应返回: `{"model_loaded":true,"model_version":"20250827_121449"}`

## 🔍 详细问题分析

### 部署脚本语法问题 (已修复)
**问题**: Bash续行符使用错误
```bash
# 错误写法 (已修复)
gcloud run deploy $CLOUD_RUN_SERVICE \\  # 双反斜杠
    --source . \\
    
# 正确写法
gcloud run deploy "$CLOUD_RUN_SERVICE" \  # 单反斜杠
    --source . \
```

### Dockerfile配置问题 (已修复)
**发现问题**:
1. 缺少 `curl` 命令用于健康检查
2. 错误的模块路径: `main:app` → `main_simple:app`
3. 训练模块路径问题: `../train` → `train`

**修复内容**:
```dockerfile
# 添加curl依赖
RUN apt-get install -y gcc g++ curl

# 修复启动命令
CMD exec uvicorn main_simple:app --host 0.0.0.0 --port $PORT

# 修复训练模块路径
COPY train /app/train
```

### 环境变量不一致问题 (已修复)
**问题**: 代码与部署配置的环境变量名不匹配
- 部署时设置: `GCS_BUCKET_NAME`
- 代码中读取: `BUCKET_NAME`

**修复方案**:
```python
# 兼容两种环境变量名
bucket_name = os.getenv('GCS_BUCKET_NAME', os.getenv('BUCKET_NAME', 'ai4fnew-drl-btc-20250827'))
```

## 🛠️ 尝试过的解决方案

### 1. 直接部署命令
```bash
gcloud run deploy drl-trader \
  --source . \
  --region asia-southeast1 \
  --project ai4fnew \
  --no-traffic
```
**结果**: 超时失败

### 2. 使用部署脚本
```bash
bash infra/deploy.sh
```
**结果**: 交互式确认后开始执行，但被用户中断

### 3. 分步部署策略
使用 `--no-traffic` 参数先部署新版本但不切流量
**结果**: 构建阶段超时

## 📊 当前系统状态

### 生产环境
- **服务URL**: https://drl-trader-veojdmk2ca-as.a.run.app
- **健康状态**: ✅ 服务运行正常
- **功能状态**: ❌ 模型未加载 (`model_loaded: false`)
- **数据流**: ❌ 无法获取实时交易决策

### 本地开发环境
- **运行状态**: ✅ 正常运行在端口 8081
- **功能状态**: ✅ 所有功能正常
  - `model_loaded: true`
  - `model_version: "20250827_121449"`
  - Dashboard 可视化正常工作
- **数据流**: ✅ 可以正常读取决策日志

## 🎯 未完成的关键目标

根据用户的3步计划，我们完成了Step 3 (Dashboard开发)，但Step 1和Step 2受阻：

### Step 1: 重新部署 Cloud Run ❌
- **目标**: 让线上 `/health` 返回 `model_loaded:true` + 正确 `model_version`
- **状态**: 未完成 - 部署持续失败
- **阻塞原因**: Cloud Build 超时问题

### Step 2: 验证生产环境 ⏸️
- **目标**: 验证模型在生产环境正常运行
- **状态**: 无法进行 - 依赖Step 1完成
- **预期**: 
  - `GET /health` 返回 `{"model_loaded":true,"model_version":"..."}`
  - `POST /tick` 能正常记录决策到日志
  - Cloud Scheduler 每分钟自动触发

### Step 3: 极简 Dashboard ✅
- **状态**: 已完成
- **功能**: K线图 + 信号点 + 权益曲线 + 模型版本显示
- **访问**: http://localhost:8081/dashboard (本地测试正常)

## 💡 推荐解决方案

### 方案A: 继续排查Cloud Build问题
1. **增加超时时间**: 使用 `--timeout=1800` (30分钟)
2. **启用构建日志**: `gcloud builds log --stream [BUILD_ID]`
3. **检查配额限制**: 验证项目的Cloud Build配额

### 方案B: 使用容器镜像部署
1. **本地构建镜像**:
   ```bash
   docker build -t gcr.io/ai4fnew/drl-trader .
   docker push gcr.io/ai4fnew/drl-trader
   ```
2. **从镜像部署**:
   ```bash
   gcloud run deploy drl-trader --image gcr.io/ai4fnew/drl-trader
   ```

### 方案C: 使用Cloud Build配置文件
创建 `cloudbuild.yaml` 文件，明确控制构建过程

### 方案D: 临时解决方案
在当前旧版本服务上手动更新环境变量，部分缓解问题

## ⚠️ 风险评估

### 高风险
- **生产环境功能缺失**: 模型未加载，无法进行实际交易决策
- **数据丢失风险**: 新的交易信号无法记录到日志

### 中风险
- **用户体验**: Dashboard只能在本地访问，无法远程监控
- **运维复杂性**: 需要手动维护本地服务进行监控

### 低风险
- **服务可用性**: 当前服务仍然稳定运行，基础健康检查正常

## 🔄 下一步建议

1. **优先级1**: 解决Cloud Build超时问题，完成生产部署
2. **优先级2**: 验证新版本的模型加载和决策记录功能
3. **优先级3**: 将Dashboard部署到生产环境进行远程监控
4. **优先级4**: 恢复Cloud Scheduler自动触发机制

## 📝 技术备注

- **本地功能验证**: 所有代码修改已在本地环境验证通过
- **配置一致性**: 环境变量、启动命令等已统一修复
- **代码完整性**: 模型文件、配置文件、Dashboard代码均准备就绪

---

**报告生成时间**: 2025-08-28 08:00 UTC  
**报告状态**: 等待部署问题解决方案  
**联系人**: 开发团队  