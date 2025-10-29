#!/usr/bin/env python3
"""
评估 Dense-500M 模型在自定义 MMLU 填空题数据集上的表现
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import json
import os
import argparse
from datetime import datetime
import re

def load_model_and_tokenizer(model_name, subfolder=None, dtype="bfloat16"):
    """加载模型和tokenizer"""
    print(f"正在加载模型: {model_name}")
    if subfolder:
        print(f"  Subfolder: {subfolder}")
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("  ⚠️  警告: 未检测到GPU，将使用CPU（会非常慢）")
    
    # 设置dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    print(f"  数据类型: {dtype}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        subfolder=subfolder,
        trust_remote_code=True
    )
    
    # 加载模型
    print("  正在加载模型权重...")
    
    # 强制使用单张GPU以提高效率（500M模型很小，不需要多卡）
    device_map = {"": 0}  # 使用GPU 0
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        subfolder=subfolder,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    model.eval()
    
    # 显示模型设备信息
    if hasattr(model, 'hf_device_map'):
        print(f"  模型设备映射: {model.hf_device_map}")
    
    print(f"  ✓ 模型加载完成")
    
    return model, tokenizer

def generate_answer(model, tokenizer, prompt, max_new_tokens=50):
    """生成答案"""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 只保留模型需要的参数，移除token_type_ids等
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs.get('attention_mask', None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 使用贪心解码（确定性生成）
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 只取新生成的部分
    generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

def normalize_answer(text):
    """标准化答案文本"""
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    # 转换为小写
    text = text.lower().strip()
    # 移除标点符号
    text = re.sub(r'[.,;:!?]', '', text)
    return text

def check_answer(generated, reference):
    """检查答案是否正确"""
    # 精确匹配
    if normalize_answer(generated) == normalize_answer(reference):
        return True, "exact_match"
    
    # 包含匹配
    if normalize_answer(reference) in normalize_answer(generated):
        return True, "contains"
    
    # 数字匹配（如果参考答案是数字）
    try:
        ref_num = float(reference.replace(',', ''))
        # 从生成的文本中提取数字
        numbers = re.findall(r'-?\d+\.?\d*', generated.replace(',', ''))
        if numbers and abs(float(numbers[0]) - ref_num) < 0.01:
            return True, "numeric_match"
    except:
        pass
    
    return False, "no_match"

def evaluate_dataset(model, tokenizer, dataset, batch_size=1, max_samples=None, output_path=None):
    """评估数据集"""
    results = {
        "total": 0,
        "correct": 0,
        "exact_match": 0,
        "contains_match": 0,
        "numeric_match": 0,
        "by_subject": {},
        "samples": []
    }
    
    # 限制样本数（用于测试）
    data = dataset['test']
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
    
    print(f"\n开始评估 {len(data)} 个样本...")
    
    # 显示GPU使用情况
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"初始GPU内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    import time
    start_time = time.time()
    
    for idx, item in enumerate(tqdm(data, desc="评估进度", unit="样本")):
        # 准备prompt
        fill_blank_question = item['fill_blank_question']
        reference_answer = item['fill_blank_answer']
        subject = item['subject']
        
        # 创建prompt（可以根据需要调整）
        prompt = f"Question: {fill_blank_question}\nAnswer:"
        
        # 生成答案
        generated_answer = generate_answer(model, tokenizer, prompt)
        
        # 检查答案
        is_correct, match_type = check_answer(generated_answer, reference_answer)
        
        # 更新统计
        results["total"] += 1
        if is_correct:
            results["correct"] += 1
            if match_type == "exact_match":
                results["exact_match"] += 1
            elif match_type == "contains":
                results["contains_match"] += 1
            elif match_type == "numeric_match":
                results["numeric_match"] += 1
        
        # 按学科统计
        if subject not in results["by_subject"]:
            results["by_subject"][subject] = {"total": 0, "correct": 0}
        results["by_subject"][subject]["total"] += 1
        if is_correct:
            results["by_subject"][subject]["correct"] += 1
        
        # 保存样本（可选）
        if output_path:
            sample_result = {
                "index": idx,
                "subject": subject,
                "question": fill_blank_question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct,
                "match_type": match_type
            }
            results["samples"].append(sample_result)
    
    # 计算准确率
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    
    # 计算每个学科的准确率
    for subject, stats in results["by_subject"].items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    
    # 计算速度统计
    elapsed_time = time.time() - start_time
    results["elapsed_time"] = elapsed_time
    results["samples_per_second"] = results["total"] / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n评估完成!")
    print(f"总耗时: {elapsed_time / 60:.1f} 分钟")
    print(f"平均速度: {results['samples_per_second']:.2f} 样本/秒")
    
    if torch.cuda.is_available():
        print(f"最终GPU内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    return results

def save_results(results, output_path):
    """保存结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")

def print_results(results):
    """打印结果"""
    print("\n" + "="*70)
    print("评估结果")
    print("="*70)
    print(f"总样本数: {results['total']}")
    print(f"正确数量: {results['correct']}")
    print(f"准确率: {results['accuracy']:.2%}")
    print(f"\n匹配类型分布:")
    print(f"  精确匹配: {results['exact_match']}")
    print(f"  包含匹配: {results['contains_match']}")
    print(f"  数字匹配: {results['numeric_match']}")
    
    # 按学科统计（前10）
    print(f"\n按学科准确率 (Top 10):")
    sorted_subjects = sorted(
        results["by_subject"].items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True
    )[:10]
    
    for subject, stats in sorted_subjects:
        print(f"  {subject:40s}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description="评估填空题MMLU数据集")
    parser.add_argument("--model_name", type=str, default="tiiuae/dense-500m-arch1",
                        help="模型名称")
    parser.add_argument("--subfolder", type=str, default=None,
                        help="模型子文件夹（checkpoint）")
    parser.add_argument("--dataset_path", type=str,
                        default="/home2/yth/.cache/huggingface/datasets/mmlu_fill_blank_dataset_8500/test",
                        help="数据集路径")
    parser.add_argument("--output_dir", type=str, default="results_fillblank",
                        help="输出目录")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="数据类型")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大样本数（用于测试）")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="生成的最大token数")
    
    args = parser.parse_args()
    
    # 加载数据集
    print("正在加载数据集...")
    dataset = load_from_disk(args.dataset_path)
    print(f"数据集已加载: {len(dataset['test'])} 个样本")
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        args.subfolder,
        args.dtype
    )
    
    # 评估
    results = evaluate_dataset(
        model,
        tokenizer,
        dataset,
        max_samples=args.max_samples,
        output_path=True
    )
    
    # 打印结果
    print_results(results)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = args.subfolder if args.subfolder else "base"
    output_path = os.path.join(
        args.output_dir,
        checkpoint_name,
        f"results_{timestamp}.json"
    )
    save_results(results, output_path)
    
    # 打印一些示例
    if results["samples"]:
        print("\n" + "="*70)
        print("示例结果 (前5个):")
        print("="*70)
        for sample in results["samples"][:5]:
            print(f"\n问题: {sample['question']}")
            print(f"参考答案: {sample['reference_answer']}")
            print(f"生成答案: {sample['generated_answer']}")
            print(f"是否正确: {'✓' if sample['is_correct'] else '✗'} ({sample['match_type']})")
            print("-" * 70)

if __name__ == "__main__":
    main()

