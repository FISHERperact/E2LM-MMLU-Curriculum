#!/usr/bin/env python3
"""
绘制 Dense-500M-Arch1 模型在 MMLU 上的评估结果图表
"""

import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(checkpoint_dir):
    """加载指定 checkpoint 的评估结果"""
    result_files = glob.glob(f"{checkpoint_dir}/**/results_*.json", recursive=True)
    
    if not result_files:
        return None
    
    with open(result_files[0], 'r') as f:
        data = json.load(f)
    
    return data.get('results', {})

def extract_iter_number(checkpoint_name):
    """从 checkpoint 名称提取迭代次数"""
    # iter_0002000 -> 2000
    try:
        return int(checkpoint_name.replace('iter_', ''))
    except:
        return 0

def main():
    results_dir = "results_dense500m"
    
    if not os.path.exists(results_dir):
        print(f"❌ 错误: 找不到结果目录 {results_dir}")
        print("   请先运行评估: bash run_evaluation_dense500m.sh")
        return
    
    # 收集所有 checkpoint 的结果
    checkpoints = []
    iter_numbers = []
    mmlu_acc = []
    mmlu_acc_norm = []
    
    checkpoint_dirs = sorted([d for d in Path(results_dir).iterdir() if d.is_dir()])
    
    if not checkpoint_dirs:
        print(f"❌ 错误: {results_dir} 目录为空")
        return
    
    print("正在收集数据...")
    
    for checkpoint_dir in sorted(checkpoint_dirs, key=lambda x: extract_iter_number(x.name)):
        checkpoint_name = checkpoint_dir.name
        results = load_results(checkpoint_dir)
        
        if results and 'mmlu' in results:
            checkpoints.append(checkpoint_name)
            iter_numbers.append(extract_iter_number(checkpoint_name))
            
            acc = results['mmlu'].get('acc,none', 0)
            acc_n = results['mmlu'].get('acc_norm,none', acc)
            
            mmlu_acc.append(acc * 100)  # 转换为百分比
            mmlu_acc_norm.append(acc_n * 100)
            
            print(f"  ✓ {checkpoint_name}: {acc_n:.2%}")
    
    if not checkpoints:
        print("没有找到任何评估结果")
        return
    
    print(f"\n找到 {len(checkpoints)} 个 checkpoint 的结果")
    print("正在生成图表...")
    
    # 创建图表
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 图表 1: MMLU 准确率随训练步数变化
    ax1 = axes[0]
    ax1.plot(iter_numbers, mmlu_acc, 'o-', linewidth=2, markersize=8, 
             label='原始准确率 (acc)', color='#2E86AB')
    ax1.plot(iter_numbers, mmlu_acc_norm, 's-', linewidth=2, markersize=8,
             label='归一化准确率 (acc_norm)', color='#A23B72')
    
    ax1.set_xlabel('训练迭代次数 (Iterations)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MMLU 准确率 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Dense-500M-Arch1 模型在 MMLU 上的表现\n(随训练步数变化)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签（每 4 个点标注一次，避免拥挤）
    for i in range(0, len(iter_numbers), 4):
        ax1.annotate(f'{mmlu_acc_norm[i]:.1f}%', 
                    xy=(iter_numbers[i], mmlu_acc_norm[i]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9, alpha=0.7)
    
    # 图表 2: 对比柱状图（首尾对比）
    ax2 = axes[1]
    
    if len(checkpoints) >= 2:
        x = np.arange(2)
        width = 0.35
        
        first_acc = [mmlu_acc[0], mmlu_acc_norm[0]]
        last_acc = [mmlu_acc[-1], mmlu_acc_norm[-1]]
        
        bars1 = ax2.bar(x - width/2, first_acc, width, label=checkpoints[0], 
                       color='#E76F51', alpha=0.8)
        bars2 = ax2.bar(x + width/2, last_acc, width, label=checkpoints[-1],
                       color='#2A9D8F', alpha=0.8)
        
        ax2.set_ylabel('MMLU 准确率 (%)', fontsize=12, fontweight='bold')
        ax2.set_title('首尾 Checkpoint 对比', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['原始准确率 (acc)', '归一化准确率 (acc_norm)'])
        ax2.legend(fontsize=11)
        ax2.grid(True, axis='y', alpha=0.3)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.2f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=10, fontweight='bold')
        
        # 计算提升
        improvement_acc = last_acc[0] - first_acc[0]
        improvement_norm = last_acc[1] - first_acc[1]
        
        ax2.text(0.5, 0.95, 
                f'提升: 原始 +{improvement_acc:.2f}% | 归一化 +{improvement_norm:.2f}%',
                transform=ax2.transAxes,
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    output_file = "dense500m_mmlu_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图表已保存: {output_file}")
    
    # 生成详细的统计信息
    print("\n" + "=" * 80)
    print("统计信息:")
    print("-" * 80)
    
    if len(mmlu_acc_norm) > 0:
        print(f"最低准确率: {min(mmlu_acc_norm):.2f}% ({checkpoints[mmlu_acc_norm.index(min(mmlu_acc_norm))]})")
        print(f"最高准确率: {max(mmlu_acc_norm):.2f}% ({checkpoints[mmlu_acc_norm.index(max(mmlu_acc_norm))]})")
        print(f"平均准确率: {np.mean(mmlu_acc_norm):.2f}%")
        print(f"标准差: {np.std(mmlu_acc_norm):.2f}%")
        
        if len(mmlu_acc_norm) >= 2:
            improvement = mmlu_acc_norm[-1] - mmlu_acc_norm[0]
            improvement_pct = (improvement / mmlu_acc_norm[0]) * 100
            print(f"总提升: {improvement:.2f}% (相对提升 {improvement_pct:.1f}%)")
    
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()

