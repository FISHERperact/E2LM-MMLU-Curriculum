#!/usr/bin/env python3
"""
查看评估题目的实用工具

用法:
    python view_questions.py hellaswag 5         # 查看 5 个 HellaSwag 题目
    python view_questions.py arc_challenge 3     # 查看 3 个 ARC-Challenge 题目
    python view_questions.py mmlu_anatomy 10     # 查看 10 个 MMLU anatomy 题目
"""

import json
import sys
import glob
import random
from pathlib import Path

def format_question(data, index=None):
    """格式化输出一个题目"""
    doc = data['doc']
    target = int(data['target'])
    
    # 模型答案
    scores = [s[0] for s in data['filtered_resps']]
    model_answer = scores.index(max(scores))
    is_correct = model_answer == target
    
    # 分隔线
    print("\n" + "="*70)
    if index is not None:
        print(f"题目 #{index}")
    print("="*70)
    
    # 根据不同数据集格式显示题目
    if 'query' in doc:  # HellaSwag
        print(f"📝 情境: {doc.get('activity_label', '')}")
        print(f"   {doc['query']}")
        print(f"\n📋 选项:")
        for i, ending in enumerate(doc['endings']):
            marker = "✓" if i == target else " "
            model_marker = "🤖" if i == model_answer else "  "
            print(f"   {model_marker} [{i}] {ending} {marker}")
    
    elif 'question' in doc:  # ARC, OpenBookQA, MMLU
        print(f"📝 问题: {doc['question']}")
        
        if 'choices' in doc and isinstance(doc['choices'], dict):
            print(f"\n📋 选项:")
            for i, choice in enumerate(doc['choices']['text']):
                label = doc['choices']['label'][i]
                marker = "✓" if i == target else " "
                model_marker = "🤖" if i == model_answer else "  "
                print(f"   {model_marker} [{label}] {choice} {marker}")
        elif 'choices' in doc and isinstance(doc['choices'], list):
            print(f"\n📋 选项:")
            for i, choice in enumerate(doc['choices']):
                marker = "✓" if i == target else " "
                model_marker = "🤖" if i == model_answer else "  "
                print(f"   {model_marker} [{i}] {choice} {marker}")
    
    elif 'sentence' in doc:  # WinoGrande
        print(f"📝 句子: {doc['sentence']}")
        print(f"\n📋 选项:")
        if 'option1' in doc and 'option2' in doc:
            options = [doc['option1'], doc['option2']]
            for i, option in enumerate(options):
                marker = "✓" if i == target else " "
                model_marker = "🤖" if i == model_answer else "  "
                print(f"   {model_marker} [{i}] {option} {marker}")
    
    elif 'passage' in doc:  # BoolQ
        print(f"📝 段落: {doc['passage'][:200]}..." if len(doc.get('passage', '')) > 200 else f"📝 段落: {doc.get('passage', '')}")
        print(f"   问题: {doc.get('question', '')}")
        print(f"\n📋 答案:")
        options = ['False', 'True']
        for i, option in enumerate(options):
            marker = "✓" if i == target else " "
            model_marker = "🤖" if i == model_answer else "  "
            print(f"   {model_marker} [{i}] {option} {marker}")
    
    else:
        print(f"📝 题目数据: {doc}")
    
    # 显示结果
    formatted_scores = [f'{float(s):.2f}' if isinstance(s, (int, float)) else str(s) for s in scores]
    print(f"\n📊 模型分数: {formatted_scores}")
    print(f"✅ 正确答案: {target}")
    print(f"🤖 模型答案: {model_answer}")
    
    if is_correct:
        print(f"🎉 结果: ✅ 答对了！")
    else:
        print(f"❌ 结果: 答错了")
    
    print("="*70)

def find_sample_file(task_name, results_dir):
    """查找样本文件"""
    pattern = f"{results_dir}/*/samples_{task_name}_*.jsonl"
    files = glob.glob(pattern)
    
    if not files:
        # 尝试不同的目录结构
        pattern = f"{results_dir}/samples_{task_name}_*.jsonl"
        files = glob.glob(pattern)
    
    if files:
        return files[0]
    return None

def view_questions(task_name, num_questions=5, random_sample=True, results_dir=None):
    """查看指定任务的题目"""
    
    # 默认结果目录
    if results_dir is None:
        results_dir = "results/step-125000"
    
    # 查找样本文件
    sample_file = find_sample_file(task_name, results_dir)
    
    if not sample_file:
        print(f"❌ 错误: 找不到任务 '{task_name}' 的样本文件")
        print(f"\n可用的任务:")
        
        # 列出所有可用任务
        pattern = f"{results_dir}/*/samples_*.jsonl"
        files = glob.glob(pattern)
        if not files:
            pattern = f"{results_dir}/samples_*.jsonl"
            files = glob.glob(pattern)
        
        tasks = set()
        for f in files:
            name = Path(f).name
            # 提取任务名: samples_TASKNAME_timestamp.jsonl
            if name.startswith('samples_'):
                task = name.replace('samples_', '').rsplit('_', 1)[0]
                tasks.add(task)
        
        for task in sorted(tasks):
            print(f"  - {task}")
        
        return
    
    print(f"\n📁 文件: {sample_file}")
    
    # 读取所有题目
    questions = []
    with open(sample_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                questions.append(json.loads(line))
            except:
                continue
    
    total = len(questions)
    print(f"📊 总共 {total} 个题目")
    
    # 选择要显示的题目
    if random_sample and num_questions < total:
        selected = random.sample(questions, num_questions)
        print(f"🎲 随机选择 {num_questions} 个题目:")
    else:
        selected = questions[:num_questions]
        print(f"📖 显示前 {num_questions} 个题目:")
    
    # 显示题目
    for i, q in enumerate(selected, 1):
        format_question(q, index=i)

def show_statistics(task_name, results_dir=None):
    """显示任务统计信息"""
    
    if results_dir is None:
        results_dir = "results/step-125000"
    
    sample_file = find_sample_file(task_name, results_dir)
    
    if not sample_file:
        print(f"❌ 找不到任务 '{task_name}' 的样本文件")
        return
    
    correct = 0
    total = 0
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                total += 1
                
                scores = [s[0] for s in data['filtered_resps']]
                model_answer = scores.index(max(scores))
                correct_answer = int(data['target'])
                
                if model_answer == correct_answer:
                    correct += 1
            except:
                continue
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"📊 任务统计: {task_name}")
    print(f"{'='*70}")
    print(f"总题目数: {total}")
    print(f"答对数量: {correct}")
    print(f"答错数量: {total - correct}")
    print(f"准确率: {accuracy:.2%}")
    print(f"{'='*70}\n")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法:")
        print("  python view_questions.py <任务名> [数量] [--stats]")
        print("")
        print("示例:")
        print("  python view_questions.py hellaswag 5")
        print("  python view_questions.py arc_challenge 3")
        print("  python view_questions.py mmlu_anatomy 10")
        print("  python view_questions.py hellaswag --stats")
        print("")
        print("可用任务:")
        
        # 列出所有可用任务
        results_dir = "results/step-125000"
        pattern = f"{results_dir}/*/samples_*.jsonl"
        files = glob.glob(pattern)
        if not files:
            pattern = f"{results_dir}/samples_*.jsonl"
            files = glob.glob(pattern)
        
        tasks = set()
        for f in files:
            name = Path(f).name
            if name.startswith('samples_'):
                task = name.replace('samples_', '').rsplit('_', 1)[0]
                tasks.add(task)
        
        for task in sorted(tasks):
            print(f"  - {task}")
        
        return
    
    task_name = sys.argv[1]
    
    # 检查是否是统计模式
    if '--stats' in sys.argv:
        show_statistics(task_name)
        return
    
    num_questions = 5
    if len(sys.argv) >= 3 and sys.argv[2].isdigit():
        num_questions = int(sys.argv[2])
    
    view_questions(task_name, num_questions)

if __name__ == "__main__":
    main()

