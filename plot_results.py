#!/usr/bin/env python3
"""
结果可视化脚本
根据评估结果生成性能对比图表
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_results(results_dir="results"):
    """加载所有评估结果"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ 错误: 结果目录不存在: {results_dir}")
        return []
    
    results = []
    
    # 遍历所有 checkpoint 目录
    for checkpoint_dir in sorted(results_path.iterdir()):
        if not checkpoint_dir.is_dir():
            continue
        
        # 查找 JSON 结果文件
        json_files = list(checkpoint_dir.glob("**/*.json"))
        
        if not json_files:
            print(f"⚠️  警告: {checkpoint_dir.name} 目录中没有找到结果文件")
            continue
        
        # 读取第一个 JSON 文件
        json_file = json_files[0]
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append({
                    'checkpoint': checkpoint_dir.name,
                    'data': data
                })
                print(f"✓ 加载: {checkpoint_dir.name}")
        except Exception as e:
            print(f"✗ 错误: 无法读取 {json_file}: {e}")
    
    return results


def plot_comparison(results, output_file="results_comparison.png"):
    """绘制性能对比图"""
    
    if not results:
        print("❌ 没有可用的结果数据")
        return
    
    # 提取数据
    checkpoints = []
    task_scores = defaultdict(lambda: {'acc': [], 'acc_norm': []})
    
    for result in results:
        checkpoint = result['checkpoint']
        checkpoints.append(checkpoint)
        
        # 提取每个任务的分数
        if 'results' in result['data']:
            for task_name, metrics in result['data']['results'].items():
                # 提取 acc 和 acc_norm
                acc = metrics.get('acc,none', metrics.get('acc', None))
                acc_norm = metrics.get('acc_norm,none', metrics.get('acc_norm', None))
                
                if acc is not None:
                    task_scores[task_name]['acc'].append(acc)
                if acc_norm is not None:
                    task_scores[task_name]['acc_norm'].append(acc_norm)
    
    # 创建图表
    n_tasks = len(task_scores)
    
    if n_tasks == 0:
        print("❌ 没有找到任务结果")
        return
    
    # 设置图表大小
    fig, axes = plt.subplots(
        nrows=(n_tasks + 1) // 2,
        ncols=2,
        figsize=(15, 5 * ((n_tasks + 1) // 2))
    )
    
    if n_tasks == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # 为每个任务绘制子图
    for idx, (task_name, scores) in enumerate(sorted(task_scores.items())):
        ax = axes[idx]
        
        x = np.arange(len(checkpoints))
        width = 0.35
        
        # 绘制 acc 和 acc_norm
        if scores['acc']:
            ax.bar(x - width/2, scores['acc'], width, label='ACC', alpha=0.8)
        if scores['acc_norm']:
            ax.bar(x + width/2, scores['acc_norm'], width, label='ACC (Normalized)', alpha=0.8)
        
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{task_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(checkpoints, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
    
    # 隐藏多余的子图
    for idx in range(n_tasks, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_file}")
    plt.close()


def plot_average_performance(results, output_file="average_performance.png"):
    """绘制平均性能图"""
    
    if not results:
        return
    
    checkpoints = []
    avg_acc = []
    avg_acc_norm = []
    
    for result in results:
        checkpoint = result['checkpoint']
        checkpoints.append(checkpoint)
        
        # 计算平均分数
        acc_scores = []
        acc_norm_scores = []
        
        if 'results' in result['data']:
            for task_name, metrics in result['data']['results'].items():
                acc = metrics.get('acc,none', metrics.get('acc', None))
                acc_norm = metrics.get('acc_norm,none', metrics.get('acc_norm', None))
                
                if acc is not None:
                    acc_scores.append(acc)
                if acc_norm is not None:
                    acc_norm_scores.append(acc_norm)
        
        avg_acc.append(np.mean(acc_scores) if acc_scores else 0)
        avg_acc_norm.append(np.mean(acc_norm_scores) if acc_norm_scores else 0)
    
    # 绘制图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(checkpoints))
    
    ax.plot(x, avg_acc, 'o-', label='Average ACC', linewidth=2, markersize=8)
    ax.plot(x, avg_acc_norm, 's-', label='Average ACC (Normalized)', linewidth=2, markersize=8)
    
    ax.set_xlabel('Checkpoint', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title('Average Performance Across All Tasks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 平均性能图已保存: {output_file}")
    plt.close()


def print_summary(results):
    """打印结果摘要"""
    print("\n" + "="*80)
    print("📊 评估结果摘要")
    print("="*80)
    
    for result in results:
        checkpoint = result['checkpoint']
        print(f"\n{checkpoint}:")
        print("-" * 60)
        
        if 'results' in result['data']:
            for task_name, metrics in sorted(result['data']['results'].items()):
                acc = metrics.get('acc,none', metrics.get('acc', None))
                acc_norm = metrics.get('acc_norm,none', metrics.get('acc_norm', None))
                
                print(f"  {task_name:20s}", end="")
                if acc is not None:
                    print(f"  acc: {acc:.4f}", end="")
                if acc_norm is not None:
                    print(f"  acc_norm: {acc_norm:.4f}", end="")
                print()
    
    print("\n" + "="*80)


def main():
    print("\n" + "="*80)
    print("📈 SmolLM2 评估结果可视化")
    print("="*80 + "\n")
    
    # 加载结果
    print("📂 加载评估结果...")
    results = load_results()
    
    if not results:
        print("\n❌ 没有找到评估结果")
        print("请先运行评估: bash run_evaluation.sh")
        return
    
    print(f"\n✓ 成功加载 {len(results)} 个 checkpoint 的结果\n")
    
    # 打印摘要
    print_summary(results)
    
    # 生成图表
    print("\n📊 生成可视化图表...")
    plot_comparison(results)
    plot_average_performance(results)
    
    print("\n" + "="*80)
    print("✅ 完成!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

