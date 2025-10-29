#!/usr/bin/env python3
"""
查看填空题MMLU评估结果
"""

import json
import os
import glob
from pathlib import Path

def load_results(results_dir="results_fillblank3b_batch"):
    """加载所有评估结果"""
    results = {}
    
    # 查找所有结果文件
    pattern = os.path.join(results_dir, "iter_*", "results_*.json")
    result_files = glob.glob(pattern)
    
    for file_path in result_files:
        # 提取 checkpoint 名称
        checkpoint = Path(file_path).parent.name
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 存储关键指标
                results[checkpoint] = {
                    "accuracy": data.get("accuracy", 0),
                    "total": data.get("total", 0),
                    "correct": data.get("correct", 0),
                    "exact_match": data.get("exact_match", 0),
                    "contains_match": data.get("contains_match", 0),
                    "numeric_match": data.get("numeric_match", 0),
                    "by_subject": data.get("by_subject", {})
                }
        except Exception as e:
            print(f"警告: 无法加载 {file_path}: {e}")
    
    return results

def print_overall_results(results):
    """打印总体结果"""
    if not results:
        print("没有找到评估结果")
        return
    
    print("\n" + "="*80)
    print("总体评估结果")
    print("="*80)
    
    # 按 checkpoint 排序
    sorted_checkpoints = sorted(results.keys())
    
    print(f"\n{'Checkpoint':<20} {'准确率':<12} {'正确/总数':<15} {'精确匹配':<12} {'包含匹配':<12} {'数字匹配':<12}")
    print("-" * 80)
    
    for checkpoint in sorted_checkpoints:
        data = results[checkpoint]
        print(f"{checkpoint:<20} "
              f"{data['accuracy']:>10.2%}  "
              f"{data['correct']:>6}/{data['total']:<6}  "
              f"{data['exact_match']:>10}  "
              f"{data['contains_match']:>10}  "
              f"{data['numeric_match']:>10}")
    
    # 找出最佳 checkpoint
    best_checkpoint = max(results.items(), key=lambda x: x[1]['accuracy'])
    print("\n" + "-" * 80)
    print(f"🏆 最佳 Checkpoint: {best_checkpoint[0]} ({best_checkpoint[1]['accuracy']:.2%})")
    print("="*80)

def print_subject_analysis(results, top_n=10):
    """打印学科分析"""
    if not results:
        return
    
    print("\n" + "="*80)
    print("学科表现分析（使用最佳checkpoint）")
    print("="*80)
    
    # 找出最佳 checkpoint
    best_checkpoint = max(results.items(), key=lambda x: x[1]['accuracy'])
    checkpoint_name, data = best_checkpoint
    
    print(f"\nCheckpoint: {checkpoint_name}")
    print(f"总体准确率: {data['accuracy']:.2%}")
    
    if not data['by_subject']:
        print("没有学科级别的数据")
        return
    
    # 按准确率排序
    sorted_subjects = sorted(
        data['by_subject'].items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )
    
    # 最佳学科
    print(f"\n📈 表现最好的 {top_n} 个学科:")
    print(f"{'学科':<45} {'准确率':<12} {'正确/总数':<12}")
    print("-" * 80)
    for subject, stats in sorted_subjects[:top_n]:
        print(f"{subject:<45} {stats['accuracy']:>10.2%}  "
              f"{stats['correct']:>4}/{stats['total']:<4}")
    
    # 最差学科
    print(f"\n📉 表现最差的 {top_n} 个学科:")
    print(f"{'学科':<45} {'准确率':<12} {'正确/总数':<12}")
    print("-" * 80)
    for subject, stats in sorted_subjects[-top_n:][::-1]:
        print(f"{subject:<45} {stats['accuracy']:>10.2%}  "
              f"{stats['correct']:>4}/{stats['total']:<4}")
    
    print("="*80)

def print_progress_comparison(results):
    """打印学习进度对比"""
    if len(results) < 2:
        return
    
    print("\n" + "="*80)
    print("学习进度分析")
    print("="*80)
    
    sorted_checkpoints = sorted(results.keys())
    
    if len(sorted_checkpoints) >= 2:
        first = sorted_checkpoints[0]
        last = sorted_checkpoints[-1]
        
        first_acc = results[first]['accuracy']
        last_acc = results[last]['accuracy']
        improvement = last_acc - first_acc
        
        print(f"\n起始 Checkpoint ({first}):")
        print(f"  准确率: {first_acc:.2%}")
        print(f"  正确数: {results[first]['correct']}/{results[first]['total']}")
        
        print(f"\n最终 Checkpoint ({last}):")
        print(f"  准确率: {last_acc:.2%}")
        print(f"  正确数: {results[last]['correct']}/{results[last]['total']}")
        
        print(f"\n提升幅度:")
        print(f"  绝对提升: {improvement:+.2%}")
        print(f"  相对提升: {(improvement/first_acc*100 if first_acc > 0 else 0):+.1f}%")
    
    print("="*80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="查看填空题MMLU评估结果")
    parser.add_argument("--results_dir", type=str, default="results_fillblank3b_batch",
                        help="结果目录")
    parser.add_argument("--top_n", type=int, default=10, 
                        help="显示Top N学科")
    
    args = parser.parse_args()
    
    # 加载结果
    results = load_results(args.results_dir)
    
    if not results:
        print(f"\n❌ 在 {args.results_dir} 中没有找到评估结果")
        print(f"\n请先运行评估:")
        print(f"  bash quick_test_fillblank.sh         # 快速测试")
        print(f"  bash run_evaluation_fillblank.sh     # 完整评估")
        return
    
    # 打印结果
    print_overall_results(results)
    print_subject_analysis(results, args.top_n)
    print_progress_comparison(results)
    
    print(f"\n💡 提示:")
    print(f"  - 详细结果位于: {args.results_dir}/")
    print(f"  - 查看日志: {args.results_dir}/iter_*/evaluation.log")
    print(f"  - 绘制图表: python3 plot_results_fillblank.py")

if __name__ == "__main__":
    main()

