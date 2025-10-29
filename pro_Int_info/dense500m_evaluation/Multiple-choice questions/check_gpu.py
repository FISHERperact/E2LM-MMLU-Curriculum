#!/usr/bin/env python3
"""快速检查GPU状态"""
import torch

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA版本: {torch.version.cuda}")
else:
    print("⚠️  警告: 未检测到GPU")

