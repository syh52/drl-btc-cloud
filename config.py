# 统一的项目配置
PROJECT_ID = "ai4fnew"
REGION = "asia-southeast1"
BUCKET_NAME = "ai4fnew-drl-btc-20250827"

# GCS 路径配置
DATA_PATH = f"gs://{BUCKET_NAME}/data/btc_data_5m_540d.csv"
MODEL_OUTPUT_PATH = f"gs://{BUCKET_NAME}/models/ppo/"
VERTEX_OUTPUT_PATH = f"gs://{BUCKET_NAME}/vertex_output"

# 训练参数默认值
DEFAULT_TIMESTEPS = 100000
DEFAULT_LOOKBACK = 60
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_MACHINE_TYPE = "n1-standard-4"

# 显示配置摘要
def print_config():
    print("=" * 50)
    print("🔧 项目配置")
    print("=" * 50)
    print(f"项目ID: {PROJECT_ID}")
    print(f"区域: {REGION}")
    print(f"存储桶: {BUCKET_NAME}")
    print(f"数据路径: {DATA_PATH}")
    print(f"模型输出: {MODEL_OUTPUT_PATH}")
    print(f"Vertex输出: {VERTEX_OUTPUT_PATH}")
    print("=" * 50)