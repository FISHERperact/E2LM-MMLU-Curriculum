#!/usr/bin/env python3
"""
快速检查GPU状态和PyTorch配置
"""

import torch
import sys

print("=" * 70)
print("GPU 和 PyTorch 状态检查")
print("=" * 70)

# 检查CUDA
print(f"\n1. CUDA 可用性: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA 版本: {torch.version.cuda}")
    print(f"   GPU 数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n   GPU {i}:")
        print(f"     名称: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"     总内存: {props.total_memory / 1024**3:.1f} GB")
        print(f"     已用内存: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"     缓存内存: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        print(f"     计算能力: {props.major}.{props.minor}")
else:
    print("   ⚠️  CUDA 不可用！")
    print("   原因可能是:")
    print("     - 没有安装支持CUDA的PyTorch")
    print("     - 没有NVIDIA GPU")
    print("     - CUDA驱动未正确安装")
    print("\n   当前将使用 CPU（会非常慢）")

# 检查PyTorch版本
print(f"\n2. PyTorch 版本: {torch.__version__}")

# 测试简单的GPU操作
if torch.cuda.is_available():
    print(f"\n3. GPU 测试:")
    try:
        # 创建一个小张量
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print(f"   ✓ GPU 矩阵乘法测试成功")
        print(f"   设备: {z.device}")
        del x, y, z
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ✗ GPU 测试失败: {e}")

# 检查推荐设置
print(f"\n4. 推荐设置:")
if torch.cuda.is_available():
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_memory_gb >= 16:
        print(f"   ✓ GPU内存充足 ({total_memory_gb:.1f} GB)")
        print(f"   推荐使用: dtype=bfloat16 或 float16")
    elif total_memory_gb >= 8:
        print(f"   ⚠️  GPU内存有限 ({total_memory_gb:.1f} GB)")
        print(f"   推荐使用: dtype=float16")
    else:
        print(f"   ⚠️  GPU内存较小 ({total_memory_gb:.1f} GB)")
        print(f"   建议使用: dtype=float16 且减少batch_size")
else:
    print(f"   ⚠️  无GPU，将使用CPU（非常慢）")

print("\n" + "=" * 70)

# 退出码
if torch.cuda.is_available():
    print("✅ GPU 可用，可以开始评估")
    sys.exit(0)
else:
    print("❌ GPU 不可用，评估会非常慢")
    sys.exit(1)

