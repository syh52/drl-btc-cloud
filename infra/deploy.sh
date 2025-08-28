#!/bin/bash

# DRL BTC 自动交易系统 - 一键部署脚本
# 适用于 Google Cloud Platform

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查必需的命令
check_prerequisites() {
    log_info "检查必需的命令..."
    
    commands=("gcloud" "gsutil" "docker")
    for cmd in "${commands[@]}"; do
        if ! command -v $cmd &> /dev/null; then
            log_error "$cmd 未安装。请先安装 Google Cloud SDK。"
            exit 1
        fi
    done
    
    log_success "必需命令检查完成"
}

# 设置默认配置
setup_config() {
    # 默认配置值
    DEFAULT_PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
    DEFAULT_REGION="asia-southeast1"
    DEFAULT_BUCKET_SUFFIX="-drl-btc-$(date +%Y%m%d)"
    
    # 读取用户输入
    echo "================================"
    log_info "配置部署参数"
    echo "================================"
    
    read -p "项目ID [${DEFAULT_PROJECT_ID}]: " PROJECT_ID
    PROJECT_ID=${PROJECT_ID:-$DEFAULT_PROJECT_ID}
    
    if [[ -z "$PROJECT_ID" ]]; then
        log_error "项目ID不能为空"
        exit 1
    fi
    
    read -p "部署区域 [${DEFAULT_REGION}]: " REGION
    REGION=${REGION:-$DEFAULT_REGION}
    
    BUCKET_NAME="${PROJECT_ID}${DEFAULT_BUCKET_SUFFIX}"
    read -p "GCS存储桶名称 [${BUCKET_NAME}]: " INPUT_BUCKET
    BUCKET_NAME=${INPUT_BUCKET:-$BUCKET_NAME}
    
    # 其他配置
    CLOUD_RUN_SERVICE="drl-trader"
    PUBSUB_TOPIC="drl-tick"
    SCHEDULER_JOB="every-minute"
    EVENTARC_TRIGGER="drl-trigger"
    
    # 显示配置
    echo ""
    log_info "部署配置:"
    echo "  项目ID: $PROJECT_ID"
    echo "  区域: $REGION" 
    echo "  存储桶: $BUCKET_NAME"
    echo "  Cloud Run服务: $CLOUD_RUN_SERVICE"
    echo "  Pub/Sub主题: $PUBSUB_TOPIC"
    echo ""
    
    read -p "确认部署? (y/N): " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        log_warning "部署已取消"
        exit 0
    fi
}

# 设置GCloud项目
setup_gcloud() {
    log_info "配置gcloud项目: $PROJECT_ID"
    
    gcloud config set project $PROJECT_ID
    gcloud config set compute/region $REGION
    
    log_success "gcloud配置完成"
}

# 启用必需的API
enable_apis() {
    log_info "启用必需的Google Cloud API..."
    
    apis=(
        "aiplatform.googleapis.com"
        "run.googleapis.com" 
        "pubsub.googleapis.com"
        "eventarc.googleapis.com"
        "cloudscheduler.googleapis.com"
        "secretmanager.googleapis.com"
        "cloudbuild.googleapis.com"
        "artifactregistry.googleapis.com"
        "storage.googleapis.com"
        "logging.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        log_info "启用 $api..."
        gcloud services enable $api --quiet
    done
    
    log_success "所有API已启用"
    
    # 等待API生效
    log_info "等待API生效..."
    sleep 30
}

# 创建GCS存储桶
create_gcs_bucket() {
    log_info "创建GCS存储桶: $BUCKET_NAME"
    
    # 检查存储桶是否已存在
    if gsutil ls gs://$BUCKET_NAME &> /dev/null; then
        log_warning "存储桶 gs://$BUCKET_NAME 已存在，跳过创建"
    else
        # 创建存储桶
        gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME
        log_success "存储桶创建完成"
    fi
    
    # 创建必要的目录结构
    log_info "创建目录结构..."
    echo "" | gsutil cp - gs://$BUCKET_NAME/data/.keep
    echo "" | gsutil cp - gs://$BUCKET_NAME/models/ppo/.keep
    echo "" | gsutil cp - gs://$BUCKET_NAME/logs/paper/.keep
    echo "" | gsutil cp - gs://$BUCKET_NAME/vertex_output/.keep
    
    log_success "目录结构创建完成"
}

# 创建Pub/Sub主题
create_pubsub_topic() {
    log_info "创建Pub/Sub主题: $PUBSUB_TOPIC"
    
    # 检查主题是否已存在
    if gcloud pubsub topics describe $PUBSUB_TOPIC &> /dev/null; then
        log_warning "主题 $PUBSUB_TOPIC 已存在，跳过创建"
    else
        gcloud pubsub topics create $PUBSUB_TOPIC
        log_success "Pub/Sub主题创建完成"
    fi
}

# 部署Cloud Run服务
deploy_cloud_run() {
    log_info "部署Cloud Run服务: $CLOUD_RUN_SERVICE"
    
    # 切换到app目录
    cd ../app
    
    # 创建.gcloudignore文件
    cat > .gcloudignore << EOF
.git
.gitignore
*.pyc
__pycache__/
.pytest_cache/
.coverage
.venv/
venv/
env/
*.log
.DS_Store
EOF

    # 创建Dockerfile
    cat > Dockerfile << EOF
FROM python:3.9-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制训练模块 (需要用于推理)
COPY ../train /app/train

# 复制应用代码
COPY . .

# 设置环境变量
ENV PORT=8080
ENV GCS_BUCKET_NAME=$BUCKET_NAME

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 启动应用
CMD exec uvicorn main:app --host 0.0.0.0 --port \$PORT
EOF
    
    log_info "部署到Cloud Run..."
    
    gcloud run deploy "$CLOUD_RUN_SERVICE" \
        --source . \
        --region="$REGION" \
        --allow-unauthenticated \
        --memory=2Gi \
        --cpu=2 \
        --timeout=3600 \
        --max-instances=10 \
        --set-env-vars="GCS_BUCKET_NAME=$BUCKET_NAME,GOOGLE_CLOUD_PROJECT=$PROJECT_ID" \
        --quiet
    
    # 获取服务URL
    SERVICE_URL=$(gcloud run services describe $CLOUD_RUN_SERVICE --region=$REGION --format='value(status.url)')
    
    log_success "Cloud Run服务部署完成: $SERVICE_URL"
    
    # 回到原目录
    cd ../infra
}

# 创建Eventarc触发器
create_eventarc_trigger() {
    log_info "创建Eventarc触发器: $EVENTARC_TRIGGER"
    
    # 检查触发器是否已存在
    if gcloud eventarc triggers describe $EVENTARC_TRIGGER --location=$REGION &> /dev/null; then
        log_warning "触发器 $EVENTARC_TRIGGER 已存在，跳过创建"
    else
        # 获取Cloud Run服务URL
        SERVICE_URL=$(gcloud run services describe $CLOUD_RUN_SERVICE --region=$REGION --format='value(status.url)')
        
        gcloud eventarc triggers create $EVENTARC_TRIGGER \\
            --location=$REGION \\
            --destination-run-service=$CLOUD_RUN_SERVICE \\
            --destination-run-region=$REGION \\
            --destination-run-path="/tick" \\
            --event-filters="type=google.cloud.pubsub.topic.v1.messagePublished" \\
            --event-filters="resource=projects/$PROJECT_ID/topics/$PUBSUB_TOPIC"
        
        log_success "Eventarc触发器创建完成"
    fi
}

# 创建Cloud Scheduler任务
create_scheduler_job() {
    log_info "创建Cloud Scheduler任务: $SCHEDULER_JOB"
    
    # 检查任务是否已存在
    if gcloud scheduler jobs describe $SCHEDULER_JOB --location=$REGION &> /dev/null; then
        log_warning "调度任务 $SCHEDULER_JOB 已存在，更新配置"
        
        # 更新现有任务
        gcloud scheduler jobs update pubsub $SCHEDULER_JOB \\
            --location=$REGION \\
            --schedule="* * * * *" \\
            --topic=$PUBSUB_TOPIC \\
            --message-body='{"symbol":"BTCUSDT","interval":"1m","source":"mock"}' \\
            --quiet
    else
        # 创建新任务
        gcloud scheduler jobs create pubsub $SCHEDULER_JOB \\
            --location=$REGION \\
            --schedule="* * * * *" \\
            --topic=$PUBSUB_TOPIC \\
            --message-body='{"symbol":"BTCUSDT","interval":"1m","source":"mock"}' \\
            --quiet
    fi
    
    log_success "Cloud Scheduler任务配置完成 (每分钟触发)"
}

# 创建Secret Manager密钥占位
create_secrets() {
    log_info "创建Secret Manager密钥占位..."
    
    secrets=("BINANCE_KEY" "BINANCE_SECRET")
    
    for secret in "${secrets[@]}"; do
        if gcloud secrets describe $secret &> /dev/null; then
            log_warning "密钥 $secret 已存在，跳过创建"
        else
            # 创建密钥并设置占位值
            echo "placeholder-key-for-future-use" | gcloud secrets create $secret --data-file=-
            log_success "密钥 $secret 创建完成"
        fi
    done
}

# 验证部署
verify_deployment() {
    log_info "验证部署状态..."
    
    # 检查Cloud Run服务
    SERVICE_URL=$(gcloud run services describe $CLOUD_RUN_SERVICE --region=$REGION --format='value(status.url)')
    if [[ -n "$SERVICE_URL" ]]; then
        log_success "✅ Cloud Run服务运行正常: $SERVICE_URL"
        
        # 测试健康检查
        log_info "测试服务健康检查..."
        if curl -f "$SERVICE_URL/health" &> /dev/null; then
            log_success "✅ 健康检查通过"
        else
            log_warning "⚠️ 健康检查失败，服务可能需要时间启动"
        fi
    else
        log_error "❌ Cloud Run服务状态异常"
    fi
    
    # 检查Pub/Sub主题
    if gcloud pubsub topics describe $PUBSUB_TOPIC &> /dev/null; then
        log_success "✅ Pub/Sub主题正常"
    else
        log_error "❌ Pub/Sub主题异常"
    fi
    
    # 检查存储桶
    if gsutil ls gs://$BUCKET_NAME &> /dev/null; then
        log_success "✅ GCS存储桶正常"
    else
        log_error "❌ GCS存储桶异常"
    fi
    
    # 检查调度任务
    if gcloud scheduler jobs describe $SCHEDULER_JOB --location=$REGION &> /dev/null; then
        log_success "✅ Cloud Scheduler任务正常"
    else
        log_error "❌ Cloud Scheduler任务异常"
    fi
}

# 显示部署总结
show_summary() {
    echo ""
    echo "================================"
    log_success "🎉 部署完成！"
    echo "================================"
    echo ""
    echo "🔗 资源信息:"
    echo "  项目ID: $PROJECT_ID"
    echo "  区域: $REGION"
    echo "  Cloud Run服务: $SERVICE_URL"
    echo "  GCS存储桶: gs://$BUCKET_NAME"
    echo "  Pub/Sub主题: $PUBSUB_TOPIC"
    echo "  调度任务: $SCHEDULER_JOB (每分钟)"
    echo ""
    echo "📋 下一步操作:"
    echo "  1. 训练模型: cd ../train && python submit_job.py --project_id $PROJECT_ID --bucket $BUCKET_NAME"
    echo "  2. 测试API: curl -X POST $SERVICE_URL/tick -H 'Content-Type: application/json' -d '{\"symbol\":\"BTCUSDT\"}'"
    echo "  3. 启动调度: gcloud scheduler jobs run $SCHEDULER_JOB --location=$REGION"
    echo "  4. 查看日志: gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=$CLOUD_RUN_SERVICE' --limit=50"
    echo ""
    echo "🔍 监控地址:"
    echo "  - Cloud Run: https://console.cloud.google.com/run/detail/$REGION/$CLOUD_RUN_SERVICE?project=$PROJECT_ID"
    echo "  - Cloud Logging: https://console.cloud.google.com/logs/query?project=$PROJECT_ID"
    echo "  - Cloud Scheduler: https://console.cloud.google.com/cloudscheduler?project=$PROJECT_ID"
    echo ""
}

# 主函数
main() {
    echo "================================"
    echo "🚀 DRL BTC 自动交易系统部署器"
    echo "================================"
    echo ""
    
    # 检查先决条件
    check_prerequisites
    
    # 配置参数
    setup_config
    
    # 部署流程
    setup_gcloud
    enable_apis
    create_gcs_bucket
    create_pubsub_topic
    deploy_cloud_run
    create_eventarc_trigger
    create_scheduler_job
    create_secrets
    
    # 验证部署
    verify_deployment
    
    # 显示总结
    show_summary
}

# 错误处理
trap 'log_error "部署过程中发生错误，脚本已退出"; exit 1' ERR

# 运行主函数
main "$@"