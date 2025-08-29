import argparse
import os
from datetime import datetime
from typing import Optional

from google.cloud import aiplatform, storage


def create_gcs_training_data(
    project_id: str, bucket_name: str, local_csv_path: Optional[str] = None
):
    """创建或上传训练数据到GCS"""
    print(f"📊 准备训练数据...")

    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)

        # 如果传入的是GCS路径，直接验证并使用
        if local_csv_path and local_csv_path.startswith("gs://"):
            # 验证GCS文件是否存在
            try:
                # 解析GCS路径 gs://bucket/path
                path_parts = local_csv_path.replace("gs://", "").split("/", 1)
                bucket_name_from_path = path_parts[0]
                blob_path = path_parts[1] if len(path_parts) > 1 else ""
                
                check_bucket = client.bucket(bucket_name_from_path)
                check_blob = check_bucket.blob(blob_path)
                
                if check_blob.exists():
                    print(f"✅ 使用现有GCS数据: {local_csv_path}")
                    return local_csv_path
                else:
                    raise FileNotFoundError(f"GCS数据文件不存在: {local_csv_path}")
            except FileNotFoundError:
                raise
            except Exception as e:
                print(f"❌ 验证GCS数据失败: {e}")
                raise
        
        # 如果有本地CSV文件，上传到GCS
        elif local_csv_path and os.path.exists(local_csv_path):
            blob_name = "data/btc_1m.csv"
            blob = bucket.blob(blob_name)

            print(
                f"📤 上传本地数据: {local_csv_path} -> gs://{bucket_name}/{blob_name}"
            )
            blob.upload_from_filename(local_csv_path)
            print(f"✅ 数据上传完成")

            return f"gs://{bucket_name}/{blob_name}"
        else:
            # 这里不要再"遍历 GCS 默认路径尝试"，直接硬失败
            raise FileNotFoundError("未提供有效的数据文件路径（本地或 GCS），无法提交训练。")

    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        raise


def submit_vertex_training_job(
    project_id: str,
    region: str,
    bucket_name: str,
    job_display_name: str,
    data_csv_path: Optional[str] = None,
    timesteps: int = 100000,
    machine_type: str = "n1-standard-4",
) -> str:
    """提交Vertex AI自定义训练任务"""

    # 提交前再次断言（双保险）
    if not data_csv_path:
        raise ValueError("必须提供 --data_csv 且该路径有效（本地将被上传或 GCS 必须存在）。")

    print(f"☁️ 提交Vertex AI训练任务...")
    print(f"   - 项目ID: {project_id}")
    print(f"   - 区域: {region}")
    print(f"   - 任务名称: {job_display_name}")
    print(f"   - 机器类型: {machine_type}")

    # 初始化Vertex AI
    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=f"gs://{bucket_name}/vertex_output",
    )

    try:
        # 定义训练容器参数
        output_dir = f"gs://{bucket_name}/models/ppo"
        args = [
            "--out_dir",
            output_dir,
            "--timesteps",
            str(timesteps),
            "--lookback",
            "60",
            "--lr",
            "3e-4",
        ]

        # 如果有训练数据路径，添加到参数中
        if data_csv_path:
            args.extend(["--data_csv", data_csv_path])

        print(f"📋 训练参数: {' '.join(args)}")

        # 创建自定义训练任务
        job = aiplatform.CustomTrainingJob(
            display_name=job_display_name,
            script_path="train.py",  # 训练脚本
            container_uri=f"gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13.py39:latest",
            requirements=[
                "stable-baselines3>=2.0.0",
                "gymnasium>=0.28.0",
                "numpy>=1.21.0",
                "pandas>=1.3.0",
                "google-cloud-storage>=2.10.0",
                "torch>=1.13.0",
                "tensorboard>=2.11.0",
            ],
            model_serving_container_image_uri=None,  # 不用于推理
            project=project_id,
            location=region,
        )

        print(f"✅ 训练任务配置完成")

        # 提交训练任务
        print(f"🚀 提交训练任务...")

        model = job.run(
            args=args,
            replica_count=1,
            machine_type=machine_type,
            accelerator_type=None,  # CPU训练
            accelerator_count=0,
            base_output_dir=f"gs://{bucket_name}/vertex_output",
            sync=False,  # 异步运行
        )

        job_resource_name = job.resource_name
        print(f"🎉 训练任务已提交成功！")
        print(f"   - 任务资源名称: {job_resource_name}")
        print(
            f"   - 在Google Cloud Console查看: https://console.cloud.google.com/vertex-ai/training/custom-jobs/{job_resource_name}?project={project_id}"
        )

        return job_resource_name

    except Exception as e:
        print(f"❌ 提交训练任务失败: {e}")
        import traceback

        traceback.print_exc()
        raise


def check_job_status(project_id: str, region: str, job_resource_name: str):
    """检查训练任务状态"""
    print(f"🔍 检查训练任务状态...")

    try:
        aiplatform.init(project=project_id, location=region)

        # 从resource name获取job
        job = aiplatform.CustomTrainingJob.get(job_resource_name)

        state = job.state
        print(f"   - 当前状态: {state}")

        if hasattr(job, "error") and job.error:
            print(f"   - 错误信息: {job.error}")

        if hasattr(job, "create_time"):
            print(f"   - 创建时间: {job.create_time}")

        if hasattr(job, "update_time"):
            print(f"   - 更新时间: {job.update_time}")

        return state

    except Exception as e:
        print(f"❌ 检查任务状态失败: {e}")
        return None


def list_recent_jobs(project_id: str, region: str, limit: int = 5):
    """列出最近的训练任务"""
    print(f"📋 最近的训练任务 (最多 {limit} 个):")

    try:
        aiplatform.init(project=project_id, location=region)

        # 获取最近的训练任务
        jobs = aiplatform.CustomTrainingJob.list(
            filter=None,
            order_by="create_time desc",
            project=project_id,
            location=region,
        )

        count = 0
        for job in jobs:
            if count >= limit:
                break

            print(f"   {count + 1}. {job.display_name}")
            print(f"      - 状态: {job.state}")
            print(f"      - 资源名称: {job.resource_name}")
            if hasattr(job, "create_time"):
                print(f"      - 创建时间: {job.create_time}")
            print()

            count += 1

        if count == 0:
            print("   暂无训练任务")

    except Exception as e:
        print(f"❌ 获取任务列表失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="提交Vertex AI BTC训练任务")

    parser.add_argument("--project_id", type=str, required=True, help="GCP项目ID")
    parser.add_argument(
        "--region", type=str, default="asia-southeast1", help="Vertex AI区域"
    )
    parser.add_argument("--bucket", type=str, required=True, help="GCS存储桶名称")
    parser.add_argument("--data_csv", type=str, help="本地训练数据CSV路径 (可选)或GCS路径")
    parser.add_argument("--timesteps", type=int, default=100000, help="训练步数")
    parser.add_argument(
        "--machine_type", type=str, default="n1-standard-4", help="训练机器类型"
    )
    parser.add_argument("--job_name", type=str, help="自定义任务名称")
    parser.add_argument("--list_jobs", action="store_true", help="列出最近的训练任务")
    parser.add_argument(
        "--check_job", type=str, help="检查指定任务状态 (提供job resource name)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("☁️ Vertex AI BTC 训练任务提交器")
    print("=" * 60)

    # 列出最近任务
    if args.list_jobs:
        list_recent_jobs(args.project_id, args.region, 10)
        return

    # 检查特定任务状态
    if args.check_job:
        check_job_status(args.project_id, args.region, args.check_job)
        return

    # 生成任务名称
    if not args.job_name:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"drl-btc-training-{timestamp}"
    else:
        job_name = args.job_name

    try:
        # 准备训练数据
        gcs_data_path = create_gcs_training_data(
            args.project_id, args.bucket, args.data_csv
        )

        # 提交训练任务
        job_resource_name = submit_vertex_training_job(
            project_id=args.project_id,
            region=args.region,
            bucket_name=args.bucket,
            job_display_name=job_name,
            data_csv_path=gcs_data_path,
            timesteps=args.timesteps,
            machine_type=args.machine_type,
        )

        print("=" * 60)
        print("✅ 训练任务已成功提交！")
        print(f"📝 任务名称: {job_name}")
        print(f"🔗 资源名称: {job_resource_name}")
        print("💡 使用以下命令检查状态:")
        print(
            f"   python {__file__} --project_id {args.project_id} --region {args.region} --bucket {args.bucket} --check_job {job_resource_name}"
        )
        print("=" * 60)

    except Exception as e:
        print(f"❌ 任务提交失败: {e}")
        exit(1)


if __name__ == "__main__":
    main()