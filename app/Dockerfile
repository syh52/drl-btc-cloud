FROM python:3.9-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
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
ENV GCS_BUCKET_NAME=ai4fnew-drl-btc-20250827

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 启动应用
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT
