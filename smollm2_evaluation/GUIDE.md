# SmolLM2 评估详细指南

> **适用对象**: 需要深入了解技术细节、遇到问题需要排查、或想扩展项目的用户

**提示**: 大部分操作请参考 [README.md](README.md)，本文档仅供深入阅读。

---

## 📋 目录

1. [技术架构](#技术架构)
2. [数据来源详解](#数据来源详解)
3. [题目文件格式](#题目文件格式)
4. [评估工作原理](#评估工作原理)
5. [故障排查](#故障排查)
6. [高级用法](#高级用法)
7. [问题解决历史](#问题解决历史)

---

## 🏗️ 技术架构

### 系统组成

```
┌──────────────────────────────────────────────────────────────┐
│                        完整技术栈                              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. 评估框架 (lm-evaluation-harness)                          │
│     • 位置: /home2/yth/lm-evaluation-harness-competition/    │
│     • 版本: 0.4.9                                             │
│     • 功能: 提供标准化的评估流程和任务定义                      │
│                                                               │
│  2. 模型 (SmolLM2-1.7B)                                       │
│     • 来源: HuggingFace Hub                                   │
│     • 大小: 3.42GB                                            │
│     • 参数量: 17亿                                            │
│     • 检查点: step-125000                                     │
│                                                               │
│  3. 运行环境                                                  │
│     • Python: 3.12                                           │
│     • PyTorch: 2.9.0 + CUDA 12.8                            │
│     • GPU: 8张 (24GB 显存/张)                                │
│     • 内存: 314GB                                             │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 数据流

```
[HuggingFace Hub]
     ↓ (下载模型和数据集)
[本地缓存: ~/.cache/huggingface/]
     ↓ (加载到内存)
[GPU 内存]
     ↓ (运行推理)
[评估结果]
     ↓ (保存)
[results/step-125000/*.json 和 *.jsonl]
     ↓ (可选: 打包提交)
[submission.zip]
```

---

## 📡 数据来源详解

### 所有数据集的 HuggingFace 配置

数据集地址都在 `lm-evaluation-harness` 的 YAML 配置文件中预先定义：

| 数据集 | 配置文件路径 | dataset_path | HuggingFace 地址 |
|--------|-------------|--------------|-----------------|
| HellaSwag | `lm_eval/tasks/hellaswag/hellaswag.yaml` | `hellaswag` | https://huggingface.co/datasets/hellaswag |
| ARC | `lm_eval/tasks/arc/arc_easy.yaml` | `allenai/ai2_arc` | https://huggingface.co/datasets/allenai/ai2_arc |
| MMLU | `lm_eval/tasks/mmlu/default/_default_template_yaml` | `cais/mmlu` | https://huggingface.co/datasets/cais/mmlu |
| PIQA | `lm_eval/tasks/piqa/piqa.yaml` | `baber/piqa` | https://huggingface.co/datasets/baber/piqa |
| WinoGrande | `lm_eval/tasks/winogrande/default.yaml` | `winogrande` | https://huggingface.co/datasets/winogrande |
| OpenBookQA | `lm_eval/tasks/openbookqa/openbookqa.yaml` | `openbookqa` | https://huggingface.co/datasets/openbookqa |
| BoolQ | 内置配置 | `boolq` | https://huggingface.co/datasets/boolq |

### 数据下载流程

```python
# lm_eval 内部工作流程（简化）

from datasets import load_dataset

# 1. 读取 YAML 配置
task_config = yaml.load("hellaswag.yaml")
dataset_path = task_config['dataset_path']  # "hellaswag"

# 2. 从 HuggingFace 下载/加载数据集
dataset = load_dataset(
    dataset_path,           # "hellaswag"
    split="validation",     # 使用验证集
    cache_dir="~/.cache/huggingface/datasets/"
)

# 3. 对每个样本进行评估
for sample in dataset:
    prompt = format_prompt(sample)
    answer = model.generate(prompt)
    score = evaluate(answer, sample['correct_answer'])
```

### 缓存位置

```bash
# 模型缓存
~/.cache/huggingface/hub/
└── models--HuggingFaceTB--SmolLM2-1.7B-intermediate-checkpoints/
    └── snapshots/
        └── [revision-hash]/
            ├── model.safetensors  (3.42GB)
            ├── config.json
            └── tokenizer.json

# 数据集缓存
~/.cache/huggingface/datasets/
├── hellaswag/              (180 MB)
├── allenai___ai2_arc/      (30 MB)
├── cais___mmlu/            (250 MB)
├── baber___piqa/           (10 MB)
├── winogrande/             (5 MB)
├── openbookqa/             (5 MB)
└── super_glue/             (10 MB, 包含 BoolQ)

总计: ~490 MB
```

---

## 📄 题目文件格式

### JSON Lines 格式

每个 `samples_*.jsonl` 文件包含多行，每行是一个完整的 JSON 对象（一个题目）：

```jsonl
{"doc_id": 0, "doc": {...}, "target": "2", "resps": [...], ...}
{"doc_id": 1, "doc": {...}, "target": "1", "resps": [...], ...}
{"doc_id": 2, "doc": {...}, "target": "3", "resps": [...], ...}
```

### 完整字段说明

```json
{
    "doc_id": 0,                              // 题目编号
    "doc": {                                  // 原始题目数据
        "id": "Mercury_7175875",              // 数据集中的 ID
        "question": "题目文本...",
        "choices": {
            "text": ["选项A", "选项B", "选项C", "选项D"],
            "label": ["A", "B", "C", "D"]
        },
        "answerKey": "C"                      // 正确答案标签
    },
    "target": "2",                            // 正确答案索引（0-based）
    "arguments": {                            // 模型看到的完整提示
        "gen_args_0": {
            "arg_0": "Question: ...\nAnswer:",
            "arg_1": " 选项A文本"
        },
        "gen_args_1": {...},
        "gen_args_2": {...},
        "gen_args_3": {...}
    },
    "resps": [                                // 模型原始响应
        [[-18.875, false]],                   // 选项 A 的分数
        [[-25.125, false]],                   // 选项 B 的分数
        [[-21.875, false]],                   // 选项 C 的分数（正确）
        [[-15.250, false]]                    // 选项 D 的分数（模型选择）
    ],
    "filtered_resps": [                       // 处理后的分数
        [-18.875],
        [-25.125],
        [-21.875],
        [-15.250]
    ],
    "doc_hash": "...",                        // 文档哈希（去重）
    "prompt_hash": "...",                     // 提示哈希
    "target_hash": "..."                      // 目标哈希
}
```

### 分数含义

- **负数越小（绝对值越大）= 越不可能**
- **负数越大（越接近 0）= 越可能**

示例:
```
-15.250  ← 最高分，模型选择这个
-18.875  ← 第二高分
-21.875  ← 正确答案（第三）
-25.125  ← 最低分
```

在这个例子中，模型选错了（选了 -15.250，而正确答案是 -21.875）。

---

## ⚙️ 评估工作原理

### lm-evaluation-harness 流程

```python
# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints",
    revision="step-125000",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(...)

# 2. 加载数据集
dataset = load_dataset("hellaswag", split="validation")

# 3. 对每个样本进行评估
results = []
for sample in dataset:
    # 构建 4 个完整提示（题目 + 每个选项）
    prompts = [
        f"{sample['query']} {choice}"
        for choice in sample['endings']
    ]
    
    # 计算每个提示的对数概率
    log_probs = []
    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)
            logits = outputs.logits
            # 计算对数概率
            log_prob = compute_log_prob(logits, tokens)
            log_probs.append(log_prob)
    
    # 选择概率最高的选项
    predicted = log_probs.index(max(log_probs))
    correct = int(sample['label'])
    
    results.append({
        'predicted': predicted,
        'correct': correct,
        'is_correct': predicted == correct
    })

# 4. 计算准确率
accuracy = sum(r['is_correct'] for r in results) / len(results)
```

### 归一化准确率计算

**问题**: 多 token 答案会因为概率相乘而得分更低

```python
# 例子
answer_1 = "Dubai"        # 1 token
answer_2 = "Abu Dhabi"    # 2 tokens ["Abu", "Dhabi"]

# 未归一化
P(Dubai) = 0.8
P(Abu Dhabi) = 0.7 * 0.6 = 0.42

# Dubai 得分更高，但这不公平！

# 归一化：平均每个 token 的对数概率
log_P_norm(Dubai) = log(0.8) / 1 = -0.097
log_P_norm(Abu Dhabi) = log(0.42) / 2 = -0.434

# 现在可以公平比较了
```

**实现**:
```python
def normalized_accuracy(log_probs, num_tokens):
    """归一化准确率计算"""
    return log_probs / num_tokens
```

---

## 🔧 故障排查

### 常见错误及解决

#### 1. SafetensorError

**错误信息**:
```
safetensors_rust.SafetensorError: 
Error while deserializing header: 
invalid JSON in header: control character found
```

**原因**: 模型文件下载不完整或损坏

**解决**:
```bash
# 1. 清除损坏缓存
rm -rf ~/.cache/huggingface/hub/models--HuggingFaceTB*

# 2. 设置镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 3. 重新运行评估
cd /home2/yth/smollm2_evaluation
bash run_evaluation.sh
```

#### 2. CUDA Out of Memory

**错误信息**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解决方案**:

```bash
# 方法 1: 使用更小的 batch size
lm_eval --batch_size 4 ...

# 方法 2: 使用 float16
lm_eval --model_args dtype=float16 ...

# 方法 3: 使用 CPU（慢）
lm_eval --device cpu ...

# 方法 4: 分任务评估
lm_eval --tasks hellaswag ...  # 一次一个任务
```

#### 3. 下载速度慢

**症状**: 下载只有几十 KB/s

**解决**:
```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或使用代理
export http_proxy=http://proxy.example.com:8080
export https_proxy=http://proxy.example.com:8080
```

#### 4. 数据集加载失败

**错误**: `DatasetNotFoundError`

**解决**:
```bash
# 手动预下载数据集
python3 << EOF
from datasets import load_dataset
load_dataset("hellaswag")
load_dataset("allenai/ai2_arc", "ARC-Easy")
load_dataset("cais/mmlu", "abstract_algebra")
EOF
```

#### 5. 重复警告信息

**现象**: `trust_remote_code is not supported anymore` 重复出现

**解释**: 这是正常的警告，不影响评估。MMLU 有 57 个子任务，每个都会显示一次。

**无需处理**，可以安全忽略。

---

## 🚀 高级用法

### 自定义任务配置

创建自定义任务 YAML 文件:

```yaml
# custom_task.yaml
task: my_custom_task
dataset_path: path/to/dataset
output_type: multiple_choice
doc_to_text: "Question: {{question}}\nAnswer:"
doc_to_target: "{{answer}}"
doc_to_choice: "{{choices}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
```

使用自定义任务:
```bash
lm_eval --model hf \
    --model_args pretrained=... \
    --tasks custom_task \
    --include_path /path/to/custom_task.yaml
```

### 批量评估多个模型

```bash
#!/bin/bash

MODELS=(
    "HuggingFaceTB/SmolLM2-135M"
    "HuggingFaceTB/SmolLM2-360M"
    "HuggingFaceTB/SmolLM2-1.7B"
)

for model in "${MODELS[@]}"; do
    echo "Evaluating $model..."
    lm_eval --model hf \
        --model_args pretrained=$model \
        --tasks hellaswag,arc_easy,mmlu \
        --output_path results/$(basename $model)/
done
```

### 使用 Few-shot 评估

```bash
# 5-shot 评估（提供 5 个示例）
lm_eval --model hf \
    --model_args pretrained=... \
    --tasks hellaswag \
    --num_fewshot 5
```

### 保存详细日志

```bash
lm_eval --model hf \
    --model_args pretrained=... \
    --tasks hellaswag \
    --log_samples \                    # 保存所有样本
    --output_path results/ \
    --verbosity DEBUG > eval.log 2>&1  # 保存详细日志
```

---

## 📊 数据集详细信息

### MMLU 的 57 个子学科

#### 人文学科 (13个)
- formal_logic, high_school_european_history, high_school_us_history
- high_school_world_history, international_law, jurisprudence
- logical_fallacies, moral_disputes, moral_scenarios
- philosophy, prehistory, professional_law, world_religions

#### 社会科学 (12个)
- econometrics, high_school_geography, high_school_government_and_politics
- high_school_macroeconomics, high_school_microeconomics, high_school_psychology
- human_sexuality, professional_psychology, public_relations
- security_studies, sociology, us_foreign_policy

#### STEM (24个)
- abstract_algebra, anatomy, astronomy, college_biology
- college_chemistry, college_computer_science, college_mathematics, college_physics
- computer_security, conceptual_physics, electrical_engineering, elementary_mathematics
- high_school_biology, high_school_chemistry, high_school_computer_science
- high_school_mathematics, high_school_physics, high_school_statistics
- machine_learning, ...

#### 其他 (8个)
- business_ethics, clinical_knowledge, college_medicine, global_facts
- human_aging, management, marketing, medical_genetics
- miscellaneous, nutrition, professional_accounting, professional_medicine, virology

### 查看 MMLU 各学科表现

```bash
cd /home2/yth/smollm2_evaluation

python3 << 'EOF'
import json
import glob

# 读取结果文件
result_files = glob.glob('results/step-125000/**/results_*.json', recursive=True)
with open(result_files[0], 'r') as f:
    data = json.load(f)

# 提取 MMLU 子任务
mmlu_tasks = {k: v for k, v in data['results'].items() 
              if k.startswith('mmlu_') and k != 'mmlu'}

# 按准确率排序
sorted_tasks = sorted(mmlu_tasks.items(), 
                     key=lambda x: x[1].get('acc,none', 0), 
                     reverse=True)

print("MMLU 子任务性能排名 (Top 10):")
print("=" * 70)
for i, (task, metrics) in enumerate(sorted_tasks[:10], 1):
    acc = metrics.get('acc,none', 0)
    task_name = task.replace('mmlu_', '').replace('_', ' ').title()
    print(f"{i:2}. {task_name:40} {acc:.2%}")
EOF
```

---

## 🛠️ 问题解决历史

### 遇到的问题及解决方案

#### 问题 1: 下载速度慢 (122 KB/s)

**时间**: 2025-10-19 11:30

**发现**:
```
model.safetensors: [进度条] 122kB/s
预计时间: 8+ 小时
```

**原因**: 直连 HuggingFace 服务器（国外），国际带宽限制

**解决**:
```bash
export HF_ENDPOINT=https://hf-mirror.com
# 速度提升到 5-20 MB/s (提升 40-160 倍)
```

**效果**: 下载时间从 8 小时降至 3-12 分钟

#### 问题 2: SafetensorError

**时间**: 2025-10-19 11:45

**错误**:
```python
safetensors_rust.SafetensorError: 
Error while deserializing header: 
invalid JSON in header: control character found
```

**原因**: 之前下载中断，缓存了 810MB (23.7%) 的不完整文件

**解决**:
```bash
# 清除损坏缓存
rm -rf ~/.cache/huggingface/hub/models--HuggingFaceTB*

# 使用镜像重新下载
export HF_ENDPOINT=https://hf-mirror.com
bash run_evaluation.sh
```

**经验教训**:
- 下载大文件前先设置镜像源
- 不要中断下载
- 如果中断了，先清除缓存再重试

#### 问题 3: 重复警告

**时间**: 2025-10-19 12:20

**现象**:
```
`trust_remote_code` is not supported anymore.
... (重复 57 次)
```

**解释**: MMLU 有 57 个子任务，每个任务加载时都会显示警告

**结论**: 这是正常现象，不影响评估结果，可以安全忽略

---

## 📚 参考资源

### 论文

- **HellaSwag**: [Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830)
- **ARC**: [Think you have Solved Question Answering?](https://arxiv.org/abs/1803.05457)
- **MMLU**: [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
- **PIQA**: [Physical Interaction QA](https://arxiv.org/abs/1911.11641)
- **WinoGrande**: [An Adversarial Winograd Schema Challenge](https://arxiv.org/abs/1907.10641)

### 在线资源

- [LM Evaluation Harness 文档](https://github.com/EleutherAI/lm-evaluation-harness)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [集成基准测试指南](https://huggingface.co/blog/integrating-benchmarks-lm-eval)

---

**最后更新**: 2025-10-19  
**维护者**: AI Assistant  
**反馈**: 如有问题或建议，请参考 README.md 或查看项目文档

