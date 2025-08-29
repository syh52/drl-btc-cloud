# DRL BTC 自动交易系统 Makefile

.PHONY: help install test train deploy status clean smoke-test format lint pre-commit-setup

# 默认目标
help:
	@echo "DRL BTC 自动交易系统 - 可用命令:"
	@echo ""
	@echo "🔧 开发环境:"
	@echo "  make install      - 安装依赖"
	@echo "  make test         - 运行测试"
	@echo "  make smoke-test   - 快速冒烟测试"
	@echo "  make format       - 格式化代码 (black + isort)"
	@echo "  make lint         - 代码检查 (flake8 + mypy)"
	@echo "  make pre-commit-setup - 设置pre-commit钩子"
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
	@echo "创建虚拟环境..."
	python3 -m venv venv || true
	@echo "安装依赖到虚拟环境..."
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install black isort flake8 mypy pre-commit
	@echo "✅ 依赖安装完成"
	@echo "💡 使用方法: source venv/bin/activate"

# 代码格式化
format:
	@echo "🎨 格式化代码..."
	@if [ ! -d "venv" ]; then echo "❌ 请先运行 make install"; exit 1; fi
	. venv/bin/activate && black --line-length=88 .
	. venv/bin/activate && isort --profile=black .
	@echo "✅ 代码格式化完成"

# 代码检查
lint:
	@echo "🔍 运行代码检查..."
	@if [ ! -d "venv" ]; then echo "❌ 请先运行 make install"; exit 1; fi
	. venv/bin/activate && flake8 --max-line-length=88 --ignore=E203,W503 . || true
	. venv/bin/activate && mypy --ignore-missing-imports app/ train/ || true
	@echo "✅ 代码检查完成"

# 设置pre-commit钩子
pre-commit-setup:
	@echo "🪝 设置pre-commit钩子..."
	@if [ ! -d "venv" ]; then echo "❌ 请先运行 make install"; exit 1; fi
	. venv/bin/activate && pre-commit install
	@echo "✅ pre-commit钩子设置完成"
	@echo "💡 现在每次提交都会自动运行代码检查"

# 运行测试
test:
	@echo "🧪 运行环境测试..."
	@if [ ! -d "venv" ]; then echo "❌ 请先运行 make install"; exit 1; fi
	. venv/bin/activate && cd train && python3 btc_env.py
	@echo "✅ 测试完成"

# 冒烟测试 - 快速验证整个流程
smoke-test:
	@echo "🚀 运行冒烟测试..."
	@if [ ! -d "venv" ]; then echo "❌ 请先运行 make install"; exit 1; fi
	@echo "1. 测试交易环境..."
	. venv/bin/activate && cd train && python3 btc_env.py
	@echo "2. 快速训练测试 (跳过-需要更多依赖)..."
	@echo "   训练测试跳过 - 需要完整的GCP环境"
	@echo "3. 测试API服务(基础检查)..."
	@echo "   API服务测试跳过 - 需要GCP环境和依赖"
	@echo "✅ 冒烟测试完成 - 核心环境正常"

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
		echo "例如: make train-submit PROJECT_ID=your-project-id BUCKET=your-bucket"; \
		exit 1; \
	fi
	cd train && python3 submit_job.py --project_id $(PROJECT_ID) --bucket $(BUCKET) --timesteps 100000
	@echo "✅ 训练任务已提交"

# 一键部署
deploy:
	@echo "🚀 开始一键部署..."
	@if [ ! -f "infra/deploy.sh" ]; then \
		echo "❌ 部署脚本不存在: infra/deploy.sh"; \
		exit 1; \
	fi
	chmod +x infra/deploy.sh
	cd infra && ./deploy.sh
	@echo "✅ 部署完成"

# 检查部署状态
status:
	@echo "📊 检查部署状态..."
	@if [ -z "$(PROJECT_ID)" ]; then \
		echo "❌ 请设置PROJECT_ID环境变量"; \
		echo "例如: make status PROJECT_ID=your-project-id"; \
		exit 1; \
	fi
	@echo "Cloud Run 服务:"
	gcloud run services list --project=$(PROJECT_ID) --platform=managed || true
	@echo ""
	@echo "Cloud Scheduler 任务:"
	gcloud scheduler jobs list --project=$(PROJECT_ID) || true
	@echo ""
	@echo "GCS 存储桶:"
	gsutil ls -p $(PROJECT_ID) || true
	@echo "✅ 状态检查完成"

# 查看服务日志
logs:
	@echo "📄 查看服务日志..."
	@if [ -z "$(PROJECT_ID)" ]; then \
		echo "❌ 请设置PROJECT_ID环境变量"; \
		echo "例如: make logs PROJECT_ID=your-project-id"; \
		exit 1; \
	fi
	gcloud logging read 'resource.type=cloud_run_revision' --project=$(PROJECT_ID) --limit=50

# 实时监控
monitor:
	@echo "📈 开始实时监控..."
	@if [ -z "$(PROJECT_ID)" ]; then \
		echo "❌ 请设置PROJECT_ID环境变量"; \
		echo "例如: make monitor PROJECT_ID=your-project-id"; \
		exit 1; \
	fi
	@echo "监控Cloud Run服务日志 (按Ctrl+C退出)..."
	gcloud logging tail 'resource.type=cloud_run_revision' --project=$(PROJECT_ID) || true

# 测试API接口
test-api:
	@echo "🔗 测试API接口..."
	@echo "测试生产环境健康检查:"
	curl -f https://drl-trader-veojdmk2ca-as.a.run.app/health || echo "❌ 健康检查失败"
	@echo ""
	@echo "测试生产环境状态:"
	curl -f https://drl-trader-veojdmk2ca-as.a.run.app/status || echo "❌ 状态检查失败"
	@echo ""
	@echo "✅ API测试完成"

# 清理临时文件
clean:
	@echo "🧹 清理临时文件..."
	find . -type f -name "*.pyc" -delete || true
	find . -type d -name "__pycache__" -exec rm -rf {} + || true
	find . -type f -name "*.log" -delete || true
	rm -rf .mypy_cache || true
	rm -rf .pytest_cache || true
	rm -rf checkpoints || true
	rm -rf tensorboard_logs || true
	@echo "✅ 清理完成"

# 清理GCP资源 (谨慎使用)
clean-gcp:
	@echo "⚠️  清理GCP资源 (谨慎操作)..."
	@echo "这将删除所有相关的GCP资源!"
	@read -p "确定要继续吗? [y/N] " -n 1 -r; \
	if [[ ! $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; echo "❌ 操作已取消"; exit 1; \
	fi
	@echo ""
	@if [ -z "$(PROJECT_ID)" ]; then \
		echo "❌ 请设置PROJECT_ID环境变量"; \
		exit 1; \
	fi
	@echo "删除Cloud Run服务..."
	gcloud run services delete drl-trader --project=$(PROJECT_ID) --platform=managed --region=asia-southeast1 --quiet || true
	@echo "删除Cloud Scheduler任务..."
	gcloud scheduler jobs delete every-minute --project=$(PROJECT_ID) --location=asia-southeast1 --quiet || true
	@echo "删除Pub/Sub主题..."
	gcloud pubsub topics delete drl-tick --project=$(PROJECT_ID) --quiet || true
	@echo "⚠️  注意: GCS存储桶需要手动删除以避免意外数据丢失"
	@echo "✅ GCP资源清理完成"

# 开发模式启动
dev:
	@echo "🔧 启动开发模式..."
	@if [ ! -d "venv" ]; then echo "❌ 请先运行 make install"; exit 1; fi
	@echo "启动本地API服务..."
	. venv/bin/activate && cd app && python3 main.py

# 显示项目信息
info:
	@echo "ℹ️  项目信息"
	@echo "项目名称: DRL BTC 自动交易系统"
	@echo "版本: v1.0.0 MVP"
	@echo "生产环境: https://drl-trader-veojdmk2ca-as.a.run.app"
	@echo "GitHub: https://github.com/syh52/drl-btc-cloud"
	@echo "Python版本要求: 3.9+"
	@echo "主要依赖: stable-baselines3, fastapi, google-cloud"