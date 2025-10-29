#!/usr/bin/env python3
"""
查看 Dense-1B-Arch1 模型在 MMLU 上的评估结果
"""

import json
import glob
import os
from pathlib import Path

def load_results(checkpoint_dir):
    """加载指定 checkpoint 的评估结果"""
    result_files = glob.glob(f"{checkpoint_dir}/**/results_*.json", recursive=True)
    
    if not result_files:
        return None
    
    with open(result_files[0], 'r') as f:
        data = json.load(f)
    
    return data.get('results', {})

def main():
    results_dir = "results_dense1b"
    
    if not os.path.exists(results_dir):
        print(f"❌ 错误: 找不到结果目录 {results_dir}")
        print("   请先运行评估: bash run_evaluation_dense1b.sh")
        return
    
    # 收集所有 checkpoint 的结果
    all_results = {}
    
    checkpoint_dirs = sorted([d for d in Path(results_dir).iterdir() if d.is_dir()])
    
    if not checkpoint_dirs:
        print(f"❌ 错误: {results_dir} 目录为空")
        return
    
    print("=" * 80)
    print(f"Dense-1B-Arch1 模型 MMLU 评估结果")
    print("=" * 80)
    print()
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_name = checkpoint_dir.name
        results = load_results(checkpoint_dir)
        
        if results:
            all_results[checkpoint_name] = results
            
            # 提取 MMLU 总分
            mmlu_acc = results.get('mmlu', {}).get('acc,none', 0)
            mmlu_acc_norm = results.get('mmlu', {}).get('acc_norm,none', mmlu_acc)
            
            print(f"✅ {checkpoint_name:20} | MMLU 准确率: {mmlu_acc:.4f} ({mmlu_acc:.2%}) | 归一化: {mmlu_acc_norm:.4f} ({mmlu_acc_norm:.2%})")
        else:
            print(f"❌ {checkpoint_name:20} | 结果未找到")
    
    print()
    print("=" * 80)
    
    if not all_results:
        print("没有找到任何评估结果")
        return
    
    # 显示详细的 MMLU 子任务结果（可选：显示第一个和最后一个 checkpoint）
    print()
    print("详细子任务结果:")
    print("-" * 80)
    
    first_checkpoint = list(all_results.keys())[0]
    last_checkpoint = list(all_results.keys())[-1]
    
    print(f"\n📊 {first_checkpoint} (第一个 checkpoint):")
    print_mmlu_breakdown(all_results[first_checkpoint])
    
    if len(all_results) > 1:
        print(f"\n📊 {last_checkpoint} (最后一个 checkpoint):")
        print_mmlu_breakdown(all_results[last_checkpoint])
    
    # 找出最佳 checkpoint
    print()
    print("=" * 80)
    print("最佳表现:")
    print("-" * 80)
    
    best_checkpoint = None
    best_score = 0
    
    for checkpoint, results in all_results.items():
        mmlu_acc_norm = results.get('mmlu', {}).get('acc_norm,none', 0)
        if mmlu_acc_norm > best_score:
            best_score = mmlu_acc_norm
            best_checkpoint = checkpoint
    
    if best_checkpoint:
        print(f"🏆 最佳 checkpoint: {best_checkpoint}")
        print(f"   MMLU 归一化准确率: {best_score:.4f} ({best_score:.2%})")
    
    print()
    print("=" * 80)
    print()
    print("💡 提示:")
    print("  - 运行 plot_results_dense1b.py 生成可视化图表")
    print("  - 查看详细结果: results_dense1b/")
    print()

def print_mmlu_breakdown(results):
    """打印 MMLU 子任务的详细结果"""
    
    # 提取所有 MMLU 子任务
    mmlu_tasks = {k: v for k, v in results.items() 
                  if k.startswith('mmlu_') and k != 'mmlu'}
    
    if not mmlu_tasks:
        print("  没有子任务数据")
        return
    
    # 按类别分组
    humanities = {}
    social_sciences = {}
    stem = {}
    other = {}
    
    # 学科分类（简化版）
    humanities_keywords = ['history', 'law', 'philosophy', 'religion', 'logic']
    social_keywords = ['economics', 'geography', 'politics', 'psychology', 'sociology']
    stem_keywords = ['math', 'physics', 'chemistry', 'biology', 'computer', 'statistics', 'engineering']
    
    for task, metrics in mmlu_tasks.items():
        task_lower = task.lower()
        acc = metrics.get('acc,none', 0)
        
        if any(kw in task_lower for kw in humanities_keywords):
            humanities[task] = acc
        elif any(kw in task_lower for kw in social_keywords):
            social_sciences[task] = acc
        elif any(kw in task_lower for kw in stem_keywords):
            stem[task] = acc
        else:
            other[task] = acc
    
    # 显示分类统计
    print(f"  人文学科 (Humanities): {len(humanities)} 个任务")
    if humanities:
        avg_humanities = sum(humanities.values()) / len(humanities)
        print(f"    平均准确率: {avg_humanities:.2%}")
    
    print(f"  社会科学 (Social Sciences): {len(social_sciences)} 个任务")
    if social_sciences:
        avg_social = sum(social_sciences.values()) / len(social_sciences)
        print(f"    平均准确率: {avg_social:.2%}")
    
    print(f"  STEM: {len(stem)} 个任务")
    if stem:
        avg_stem = sum(stem.values()) / len(stem)
        print(f"    平均准确率: {avg_stem:.2%}")
    
    print(f"  其他: {len(other)} 个任务")
    if other:
        avg_other = sum(other.values()) / len(other)
        print(f"    平均准确率: {avg_other:.2%}")
    
    # 显示最佳和最差的 5 个任务
    sorted_tasks = sorted(mmlu_tasks.items(), key=lambda x: x[1].get('acc,none', 0), reverse=True)
    
    print(f"\n  表现最好的 5 个任务:")
    for task, metrics in sorted_tasks[:5]:
        acc = metrics.get('acc,none', 0)
        task_name = task.replace('mmlu_', '').replace('_', ' ').title()
        print(f"    {task_name:40} {acc:.2%}")
    
    print(f"\n  表现最差的 5 个任务:")
    for task, metrics in sorted_tasks[-5:]:
        acc = metrics.get('acc,none', 0)
        task_name = task.replace('mmlu_', '').replace('_', ' ').title()
        print(f"    {task_name:40} {acc:.2%}")

if __name__ == "__main__":
    main()

