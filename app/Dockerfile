# 使用官方Python 3.11镜像作为基础
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 复制训练模块（如果需要）
COPY ../btc_env.py /app/

# 暴露端口
EXPOSE 8080

# 设置环境变量
ENV PYTHONPATH=/app
ENV PORT=8080

# 启动命令
CMD ["python", "main.py"]