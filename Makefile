# DRL BTC 自动交易系统 Makefile

.PHONY: help install test train deploy status clean smoke-test

# 默认目标
help:
	@echo "DRL BTC 自动交易系统 - 可用命令:"
	@echo ""
	@echo "🔧 开发环境:"
	@echo "  make install      - 安装依赖"
	@echo "  make test         - 运行测试"
	@echo "  make smoke-test   - 快速冒烟测试"
	@echo ""
	@echo "🤖 模型训练:"
	@echo "  make train        - 训练PPO模型"
	@echo "  make train-submit - 提交Vertex AI训练任务"
	@echo ""
	@echo "☁️ 部署操作:"
	@echo "  make deploy       - 一键部署到GCP"
	@echo "  make status       - 检查部署状态"
	@echo "  make logs         - 查看服务日志"
	@echo ""
	@echo "🧹 清理操作:"
	@echo "  make clean        - 清理临时文件"
	@echo "  make clean-gcp    - 清理GCP资源"
	@echo ""
	@echo "📊 监控工具:"
	@echo "  make monitor      - 实时监控交易决策"
	@echo "  make test-api     - 测试API接口"

# 安装依赖
install:
	@echo "📦 安装项目依赖..."
	pip install -r requirements.txt
	@echo "✅ 依赖安装完成"

# 运行测试
test:
	@echo "🧪 运行环境测试..."
	cd train && python btc_env.py
	@echo "✅ 测试完成"

# 冒烟测试 - 快速验证整个流程
smoke-test:
	@echo "🚀 运行冒烟测试..."
	@echo "1. 测试交易环境..."
	cd train && python btc_env.py
	@echo "2. 快速训练测试 (1000步)..."
	cd train && python train.py --timesteps 1000 --check_only
	@echo "3. 测试API服务..."
	@echo "启动本地API服务进行测试..."
	cd app && python main.py &
	@sleep 5
	@curl -f http://localhost:8080/health || echo "API服务测试失败"
	@pkill -f "python main.py" || true
	@echo "✅ 冒烟测试完成"

# 本地训练
train:
	@echo "🤖 开始本地训练..."
	cd train && python train.py --timesteps 50000 --eval
	@echo "✅ 训练完成"

# 提交Vertex AI训练任务
train-submit:
	@echo "☁️ 提交Vertex AI训练任务..."
	@if [ -z "$(PROJECT_ID)" ]; then \
		echo "❌ 请设置PROJECT_ID环境变量"; \
		echo "例如: make train-submit PROJECT_ID=your-project-id BUCKET=your-bucket"; \
		exit 1; \
	fi
	@if [ -z "$(BUCKET)" ]; then \
		echo "❌ 请设置BUCKET环境变量"; \
		exit 1; \
	fi
	cd train && python submit_job.py --project_id $(PROJECT_ID) --bucket $(BUCKET)
	@echo "✅ 训练任务已提交"

# 一键部署
deploy:
	@echo "🚀 开始部署到GCP..."
	cd infra && ./deploy.sh
	@echo "✅ 部署完成"

# 检查部署状态
status:
	@echo "📊 检查系统状态..."
	@if [ -z "$(PROJECT_ID)" ]; then \
		echo "❌ 请设置PROJECT_ID环境变量"; \
		exit 1; \
	fi
	@echo "Cloud Run服务状态:"
	@gcloud run services list --platform=managed --regions=asia-southeast1 --filter="metadata.name:drl-trader" --format="table(metadata.name,status.url,status.conditions[0].status)" || echo "无法获取Cloud Run状态"
	@echo ""
	@echo "Pub/Sub主题状态:"
	@gcloud pubsub topics list --filter="name:drl-tick" --format="table(name)" || echo "无法获取Pub/Sub状态"
	@echo ""
	@echo "Cloud Scheduler任务状态:"
	@gcloud scheduler jobs list --location=asia-southeast1 --filter="name:every-minute" --format="table(name,schedule,state)" || echo "无法获取Scheduler状态"

# 查看服务日志
logs:
	@echo "📋 查看服务日志 (最近50条)..."
	@if [ -z "$(PROJECT_ID)" ]; then \
		echo "❌ 请设置PROJECT_ID环境变量"; \
		exit 1; \
	fi
	gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=drl-trader' \
		--limit=50 \
		--format="table(timestamp,severity,textPayload)" \
		--project=$(PROJECT_ID)

# 测试API接口
test-api:
	@echo "🧪 测试API接口..."
	@if [ -z "$(SERVICE_URL)" ]; then \
		echo "❌ 请设置SERVICE_URL环境变量"; \
		echo "例如: make test-api SERVICE_URL=https://your-service-url"; \
		exit 1; \
	fi
	@echo "1. 健康检查..."
	curl -f "$(SERVICE_URL)/health" | python -m json.tool
	@echo "\n2. 获取状态..."
	curl -f "$(SERVICE_URL)/status" | python -m json.tool
	@echo "\n3. 交易决策测试..."
	curl -X POST "$(SERVICE_URL)/tick" \
		-H "Content-Type: application/json" \
		-d '{"symbol":"BTCUSDT","interval":"1m"}' | python -m json.tool
	@echo "✅ API测试完成"

# 实时监控
monitor:
	@echo "📊 开始实时监控..."
	@if [ -z "$(PROJECT_ID)" ]; then \
		echo "❌ 请设置PROJECT_ID环境变量"; \
		exit 1; \
	fi
	@echo "监控Cloud Run日志 (Ctrl+C停止)..."
	gcloud logging tail 'resource.type=cloud_run_revision AND resource.labels.service_name=drl-trader' \
		--format="value(timestamp,severity,textPayload)" \
		--project=$(PROJECT_ID)

# 清理临时文件
clean:
	@echo "🧹 清理临时文件..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.log" -delete
	rm -rf train/checkpoints/ || true
	rm -rf train/tensorboard_logs/ || true
	rm -rf app/Dockerfile || true
	rm -rf app/.gcloudignore || true
	@echo "✅ 清理完成"

# 清理GCP资源 (危险操作)
clean-gcp:
	@echo "⚠️ 清理GCP资源 (危险操作)..."
	@read -p "确认删除所有GCP资源? (yes/NO): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		echo "删除Cloud Run服务..."; \
		gcloud run services delete drl-trader --region=asia-southeast1 --quiet || true; \
		echo "删除Cloud Scheduler任务..."; \
		gcloud scheduler jobs delete every-minute --location=asia-southeast1 --quiet || true; \
		echo "删除Eventarc触发器..."; \
		gcloud eventarc triggers delete drl-trigger --location=asia-southeast1 --quiet || true; \
		echo "删除Pub/Sub主题..."; \
		gcloud pubsub topics delete drl-tick --quiet || true; \
		echo "⚠️ 注意: GCS存储桶和Secret需要手动删除"; \
		echo "✅ 主要资源已清理"; \
	else \
		echo "取消清理操作"; \
	fi

# 开发环境快速启动
dev:
	@echo "🔧 启动开发环境..."
	@echo "启动本地API服务 (http://localhost:8080)..."
	cd app && GCS_BUCKET_NAME=dev-bucket python main.py

# 获取项目信息
info:
	@echo "📋 项目信息:"
	@echo "  名称: DRL BTC 自动交易系统"
	@echo "  版本: 1.0.0 MVP"
	@echo "  架构: Vertex AI + Cloud Run + Pub/Sub"
	@echo "  区域: asia-southeast1"
	@echo ""
	@echo "📁 项目结构:"
	@find . -name "*.py" -o -name "*.sh" -o -name "*.yaml" -o -name "Makefile" | \
		grep -E "\.(py|sh|yaml)$$|Makefile$$" | \
		head -20
	@echo ""
	@echo "📊 代码统计:"
	@find . -name "*.py" -exec cat {} \; | wc -l | awk '{print "  Python代码行数: " $$1}'
	@find . -name "*.sh" -exec cat {} \; | wc -l | awk '{print "  Shell脚本行数: " $$1}'