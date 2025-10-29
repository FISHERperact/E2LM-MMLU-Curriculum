# 自定义 MMLU 填空题数据集格式

## 📊 数据集概览

- **名称**: mmlu_fill_blank_dataset_8500
- **总样本数**: 8,500
- **学科数量**: 43 个
- **任务类型**: 填空题（与标准MMLU的多选题不同）

## 📂 存储位置

```bash
~/.cache/huggingface/datasets/mmlu_fill_blank_dataset_8500/
├── test/     # 测试集（8500个样本）
├── val/      # 验证集
└── dev/      # 开发集
```

## 📋 数据格式对比

### 标准 MMLU（多选题）

```json
{
  "question": "What is the time complexity of binary search?",
  "subject": "computer_science",
  "choices": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
  "answer": 1  // 索引 0-3
}
```

### 自定义 MMLU（填空题）

```json
{
  "original_question": "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
  "original_answer_text": "4",
  "subject": "abstract_algebra",
  "fill_blank_question": "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q, which is [BLANK].",
  "fill_blank_answer": "4"
}
```

## 🔑 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `original_question` | string | 原始问题文本 |
| `original_answer_text` | string | 原始答案（文本形式） |
| `subject` | string | 学科名称 |
| `fill_blank_question` | string | 填空题形式（含[BLANK]标记） |
| `fill_blank_answer` | string | 填空答案 |

## 📊 数据样本示例

### 示例 1: Abstract Algebra

```
原始问题: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
原始答案: 4
学科: abstract_algebra
填空题: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q, which is [BLANK].
填空答案: 4
```

### 示例 2: Abstract Algebra (多个空)

```
原始问题: Find all zeros in the indicated finite field of the given polynomial with coefficients in that field. x^5 + 3x^3 + x^2 + 2x in Z_5
原始答案: 0,4
学科: abstract_algebra
填空题: Find all zeros in the indicated finite field of the given polynomial with coefficients in that field. x^5 + 3x^3 + x^2 + 2x in Z_5. The zeros are [BLANK] and [BLANK].
填空答案: 0,4
```

## 📈 学科分布（Top 10）

| 学科 | 样本数 |
|------|--------|
| miscellaneous | 783 |
| high_school_psychology | 545 |
| high_school_macroeconomics | 390 |
| elementary_mathematics | 378 |
| moral_disputes | 333 |
| high_school_biology | 310 |
| high_school_mathematics | 270 |
| clinical_knowledge | 265 |
| high_school_microeconomics | 238 |
| high_school_world_history | 237 |

**完整学科列表** (43个):
```
abstract_algebra, anatomy, astronomy, business_ethics, clinical_knowledge,
college_biology, college_chemistry, college_computer_science, college_mathematics,
college_medicine, college_physics, computer_security, conceptual_physics,
econometrics, electrical_engineering, elementary_mathematics, formal_logic,
global_facts, high_school_biology, high_school_chemistry, high_school_computer_science,
high_school_european_history, high_school_geography, high_school_government_and_politics,
high_school_macroeconomics, high_school_mathematics, high_school_microeconomics,
high_school_physics, high_school_psychology, high_school_statistics, high_school_us_history,
high_school_world_history, human_aging, human_sexuality, international_law,
jurisprudence, logical_fallacies, machine_learning, management, marketing,
medical_genetics, miscellaneous, moral_disputes
```

## 🎯 任务类型差异

### 标准 MMLU (多选题)
- **任务**: 从4个选项中选择正确答案
- **评估**: 计算选项概率，选最高的
- **指标**: 准确率（ACC）

### 自定义 MMLU (填空题)
- **任务**: 生成文本填入空白处
- **评估**: 需要生成式评估或精确匹配
- **指标**: 精确匹配、ROUGE、或自定义评分

## ⚠️ 重要注意事项

1. **不兼容标准MMLU任务**: 这个数据集不能直接用于标准的 `mmlu` 任务
2. **需要自定义任务定义**: 需要为lm-evaluation-harness创建自定义任务
3. **评估方式不同**: 填空题需要生成式评估，而非多选题的分类评估

## 🔧 使用此数据集的选项

### 选项 1: 创建自定义 lm-eval 任务（推荐）

创建自定义任务配置文件，适配填空题格式。

### 选项 2: 转换为标准 MMLU 格式

将填空题转换为多选题格式（需要生成干扰选项）。

### 选项 3: 直接使用 HuggingFace Transformers

编写自定义评估脚本，不依赖 lm-evaluation-harness。

## 📚 数据集元信息

```json
{
  "dataset_name": "mmlu_fill_blank_simplified",
  "description": "MMLU填空题数据集（简化版）",
  "total_samples": 8500,
  "subject_count": 43,
  "splits": {
    "test": 8500
  },
  "created_at": "2024-01-01 12:00:00"
}
```

## 🎓 总结

这是一个**自定义的MMLU衍生数据集**，采用填空题格式而非标准的多选题格式。要在评估中使用它，需要：

1. ✅ 数据已正确存储在 HuggingFace datasets 缓存中
2. ⚠️ 需要创建自定义任务配置
3. ⚠️ 或者转换为标准MMLU格式
4. ⚠️ 或者编写独立的评估脚本

