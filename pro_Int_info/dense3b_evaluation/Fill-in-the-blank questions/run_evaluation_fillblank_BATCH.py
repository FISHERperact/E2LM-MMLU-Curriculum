#!/usr/bin/env python3
"""
Evaluate Dense-3B Model on Custom Fill-in-the-blank MMLU Dataset - Batch Version
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
    
    # 设置pad_token和padding方向
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # decoder模型必须左padding
    
    print(f"  Tokenizer padding side: {tokenizer.padding_side}")
    
    # 加载模型
    print("  正在加载模型权重...")
    device_map = {"": 0}  # 使用GPU 0
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        subfolder=subfolder,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    model.eval()
    print(f"  ✓ 模型加载完成")
    
    return model, tokenizer

def generate_answers_batch(model, tokenizer, prompts, max_new_tokens=10):
    """批处理生成答案"""
    # Tokenize所有prompts
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码生成的文本
    generated_texts = []
    for i, output in enumerate(outputs):
        # 只取新生成的部分
        generated = tokenizer.decode(
            output[input_ids.shape[1]:],
            skip_special_tokens=True
        )
        generated_texts.append(generated.strip())
    
    return generated_texts

def normalize_answer(text):
    """标准化答案文本"""
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
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
    
    # 数字匹配
    try:
        ref_num = float(reference.replace(',', ''))
        numbers = re.findall(r'-?\d+\.?\d*', generated.replace(',', ''))
        if numbers and abs(float(numbers[0]) - ref_num) < 0.01:
            return True, "numeric_match"
    except:
        pass
    
    return False, "no_match"

def evaluate_dataset(model, tokenizer, dataset, batch_size=16, max_samples=None, max_new_tokens=10):
    """评估数据集 - 批处理版本"""
    results = {
        "total": 0,
        "correct": 0,
        "exact_match": 0,
        "contains_match": 0,
        "numeric_match": 0,
        "by_subject": {},
        "samples": []
    }
    
    data = dataset['test']
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
    
    print(f"\n开始评估 {len(data)} 个样本...")
    print(f"批处理大小: {batch_size}")
    print(f"最大生成长度: {max_new_tokens} tokens")
    
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"初始GPU内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    import time
    start_time = time.time()
    
    # 批处理评估
    num_batches = (len(data) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="批处理进度"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(data))
        
        # 准备批次prompts
        prompts = []
        batch_info = []  # 存储每个样本的信息
        
        for i in range(start_idx, end_idx):
            item = data[i]
            prompt = f"Question: {item['fill_blank_question']}\nAnswer:"
            prompts.append(prompt)
            batch_info.append({
                'question': item['fill_blank_question'],
                'answer': item['fill_blank_answer'],
                'subject': item['subject'],
                'index': i
            })
        
        # 批量生成
        generated_answers = generate_answers_batch(model, tokenizer, prompts, max_new_tokens)
        
        # 检查答案
        for i, info in enumerate(batch_info):
            reference_answer = info['answer']
            generated_answer = generated_answers[i]
            subject = info['subject']
            
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
            
            # 保存样本
            sample_result = {
                "index": info['index'],
                "subject": subject,
                "question": info['question'],
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct,
                "match_type": match_type
            }
            results["samples"].append(sample_result)
    
    # 计算统计
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    
    for subject, stats in results["by_subject"].items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    
    elapsed_time = time.time() - start_time
    results["elapsed_time"] = elapsed_time
    results["samples_per_second"] = results["total"] / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n评估完成!")
    print(f"总耗时: {elapsed_time / 60:.1f} 分钟 ({elapsed_time:.1f} 秒)")
    print(f"平均速度: {results['samples_per_second']:.2f} 样本/秒")
    print(f"🚀 提速效果: 批处理 vs 单个 = {results['samples_per_second'] / 0.5:.1f}x 倍")
    
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
    parser = argparse.ArgumentParser(description="Evaluate Fill-in-the-blank MMLU - Batch Version")
    parser.add_argument("--model_name", type=str, default="tiiuae/dense-3b-arch1")
    parser.add_argument("--subfolder", type=str, default=None)
    parser.add_argument("--dataset_path", type=str,
                        default="/home2/yth/.cache/huggingface/datasets/mmlu_fill_blank_dataset_8500/test")
    parser.add_argument("--output_dir", type=str, default="results_fillblank_batch")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批处理大小（建议8-32）")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="生成的最大token数（建议10-30）")
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 Fill-in-the-blank Evaluation (Batch Version)")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print("="*70)
    
    # 加载数据集
    print("\n正在加载数据集...")
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
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens
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
    
    # 打印示例
    if results["samples"]:
        print("\n" + "="*70)
        print("示例结果 (前5个):")
        print("="*70)
        for sample in results["samples"][:5]:
            print(f"\n问题: {sample['question'][:80]}...")
            print(f"参考答案: {sample['reference_answer']}")
            print(f"生成答案: {sample['generated_answer']}")
            print(f"是否正确: {'✓' if sample['is_correct'] else '✗'} ({sample['match_type']})")
            print("-" * 70)

if __name__ == "__main__":
    main()

