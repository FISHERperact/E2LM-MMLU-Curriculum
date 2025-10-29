#!/usr/bin/env python3
"""
Plot Dense-1B-Arch1 model evaluation results on MMLU benchmark
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
    results_dir = "results_dense1b"
    
    if not os.path.exists(results_dir):
        print(f"❌ Error: Results directory not found: {results_dir}")
        print("   Please run evaluation first: bash run_evaluation_dense1b.sh")
        return
    
    # 收集所有 checkpoint 的结果
    checkpoints = []
    iter_numbers = []
    mmlu_acc = []
    mmlu_acc_norm = []
    
    checkpoint_dirs = sorted([d for d in Path(results_dir).iterdir() if d.is_dir()])
    
    if not checkpoint_dirs:
        print(f"❌ Error: {results_dir} directory is empty")
        return
    
    print("Collecting data...")
    
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
        print("No evaluation results found")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoint results")
    print("Generating plots...")
    
    # 创建图表
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: MMLU accuracy vs training iterations
    ax1 = axes[0]
    ax1.plot(iter_numbers, mmlu_acc, 'o-', linewidth=2, markersize=8, 
             label='Raw Accuracy (acc)', color='#2E86AB')
    ax1.plot(iter_numbers, mmlu_acc_norm, 's-', linewidth=2, markersize=8,
             label='Normalized Accuracy (acc_norm)', color='#A23B72')
    
    ax1.set_xlabel('Training Iterations', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MMLU Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Dense-1B-Arch1 Model Performance on MMLU\n(Training Progress)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels (every 4th point to avoid crowding)
    for i in range(0, len(iter_numbers), 4):
        ax1.annotate(f'{mmlu_acc_norm[i]:.1f}%', 
                    xy=(iter_numbers[i], mmlu_acc_norm[i]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9, alpha=0.7)
    
    # Plot 2: First vs Last checkpoint comparison
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
        
        ax2.set_ylabel('MMLU Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('First vs Last Checkpoint Comparison', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Raw Accuracy (acc)', 'Normalized Accuracy (acc_norm)'])
        ax2.legend(fontsize=11)
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.2f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=10, fontweight='bold')
        
        # Calculate improvement
        improvement_acc = last_acc[0] - first_acc[0]
        improvement_norm = last_acc[1] - first_acc[1]
        
        ax2.text(0.5, 0.95, 
                f'Improvement: Raw +{improvement_acc:.2f}% | Normalized +{improvement_norm:.2f}%',
                transform=ax2.transAxes,
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_file = "dense1b_mmlu_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved: {output_file}")
    
    # Generate detailed statistics
    print("\n" + "=" * 80)
    print("Statistics Summary:")
    print("-" * 80)
    
    if len(mmlu_acc_norm) > 0:
        print(f"Lowest Accuracy:  {min(mmlu_acc_norm):.2f}% ({checkpoints[mmlu_acc_norm.index(min(mmlu_acc_norm))]})")
        print(f"Highest Accuracy: {max(mmlu_acc_norm):.2f}% ({checkpoints[mmlu_acc_norm.index(max(mmlu_acc_norm))]})")
        print(f"Average Accuracy: {np.mean(mmlu_acc_norm):.2f}%")
        print(f"Std Deviation:    {np.std(mmlu_acc_norm):.2f}%")
        
        if len(mmlu_acc_norm) >= 2:
            improvement = mmlu_acc_norm[-1] - mmlu_acc_norm[0]
            improvement_pct = (improvement / mmlu_acc_norm[0]) * 100
            print(f"Total Improvement: {improvement:.2f}% (Relative: {improvement_pct:.1f}%)")
    
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()

