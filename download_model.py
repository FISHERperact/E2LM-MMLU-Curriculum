#!/usr/bin/env python3
"""
预下载 SmolLM2 模型脚本
在评估前先下载模型，避免评估时等待
"""

import os
from huggingface_hub import snapshot_download
from tqdm import tqdm

# 设置镜像源（加速下载）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 配置
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints"
CHECKPOINTS = [
    "step-125000",
    # "step-100000",  # 取消注释以下载更多 checkpoint
    # "step-150000",
]

def download_checkpoint(model_name, revision):
    """下载特定 checkpoint"""
    print(f"\n{'='*60}")
    print(f"下载 {model_name} - {revision}")
    print(f"{'='*60}\n")
    
    try:
        snapshot_download(
            repo_id=model_name,
            revision=revision,
            cache_dir=None,  # 使用默认缓存目录
            resume_download=True,  # 支持断点续传
        )
        print(f"\n✅ {revision} 下载完成")
        return True
    except Exception as e:
        print(f"\n❌ {revision} 下载失败: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("SmolLM2 模型预下载工具")
    print("="*60)
    print(f"\n使用镜像源: {os.environ.get('HF_ENDPOINT', '默认源')}")
    print(f"模型: {MODEL_NAME}")
    print(f"Checkpoints: {', '.join(CHECKPOINTS)}")
    print("\n" + "="*60)
    
    input("\n按 Enter 开始下载...")
    
    success_count = 0
    fail_count = 0
    
    for checkpoint in CHECKPOINTS:
        if download_checkpoint(MODEL_NAME, checkpoint):
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "="*60)
    print("下载完成")
    print("="*60)
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {fail_count}")
    
    if success_count > 0:
        print("\n模型已缓存，现在可以快速运行评估:")
        print("  bash run_evaluation.sh")
    
    print("\n缓存位置: ~/.cache/huggingface/hub/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

