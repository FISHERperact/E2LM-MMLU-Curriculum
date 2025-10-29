# Dense-500M 填空题MMLU评估指南

## 📋 概述

本目录包含了评估 `tiiuae/dense-500m-arch1` 模型在**自定义MMLU填空题数据集**上的完整工具链。

### 数据集信息

- **名称**: mmlu_fill_blank_dataset_8500
- **位置**: `~/.cache/huggingface/datasets/mmlu_fill_blank_dataset_8500/`
- **样本数**: 8,500
- **学科数**: 43 个
- **任务类型**: 填空题（生成式任务）

### 与标准MMLU的区别

| 特性 | 标准MMLU | 填空题MMLU |
|------|----------|------------|
| 任务类型 | 多选题（4选1） | 填空题（生成式） |
| 输入格式 | 问题 + 4个选项 | 问题 + [BLANK]标记 |
| 输出格式 | 选择A/B/C/D | 生成文本答案 |
| 评估方式 | 概率比较 | 文本匹配 |
| 样本数 | ~14,000 | 8,500 |
| 学科数 | 57 | 43 |

## 🚀 快速开始

### 方法 1: 快速测试（推荐第一次运行）

```bash
cd /home2/yth/dense500m_evaluation

# 测试 3 个 checkpoints，每个 50 样本（~15-30分钟）
bash quick_test_fillblank.sh

# 查看测试结果
python3 view_results_fillblank.py --results_dir results_fillblank_test
```

### 方法 2: 完整评估

```bash
cd /home2/yth/dense500m_evaluation

# 评估所有 27 个 checkpoints，每个 8500 样本（预计数小时）
bash run_evaluation_fillblank.sh

# 查看结果
python3 view_results_fillblank.py
```

## 📂 文件说明

### 评估脚本

| 文件 | 说明 |
|------|------|
| `run_evaluation_fillblank.py` | Python评估脚本（核心） |
| `run_evaluation_fillblank.sh` | Bash批量评估脚本 |
| `quick_test_fillblank.sh` | 快速测试脚本 |
| `view_results_fillblank.py` | 结果查看脚本 |

### 文档

| 文件 | 说明 |
|------|------|
| `README_FILLBLANK.md` | 本文件 |
| `CUSTOM_MMLU_FORMAT.md` | 数据集格式详细说明 |
| `check_mmlu_data.sh` | 数据集检查工具 |

## 🔧 评估参数配置

### 在 `run_evaluation_fillblank.sh` 中修改：

```bash
# 选择要评估的 checkpoints
CHECKPOINTS=(
    "iter_0002000"
    "iter_0028000"
    "iter_0054000"
    # ... 添加或删除
)

# 调整评估参数
DTYPE="bfloat16"        # 数据类型: bfloat16, float16, float32
MAX_NEW_TOKENS=50       # 生成的最大token数
MAX_SAMPLES=""          # 限制样本数（空=全部，如"100"=测试100个）
```

### 在 `run_evaluation_fillblank.py` 中修改：

```python
# 生成参数
temperature=0.1       # 降低以获得更确定的答案
do_sample=False       # True=采样，False=贪心
max_new_tokens=50     # 最大生成长度
```

## 📊 评估流程

### 1. 数据加载
```python
from datasets import load_from_disk
dataset = load_from_disk("~/.cache/huggingface/datasets/mmlu_fill_blank_dataset_8500/test")
```

### 2. Prompt 格式
```
Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q, which is [BLANK].
Answer:
```

### 3. 模型生成
模型生成答案文本（最多50个token）

### 4. 答案匹配
三种匹配方式：
- **精确匹配**: 标准化后完全相同
- **包含匹配**: 生成的答案包含参考答案
- **数字匹配**: 提取数字进行比较

### 5. 结果统计
- 总体准确率
- 各学科准确率
- 匹配类型分布
- 详细样本记录

## 📈 结果输出

### 目录结构

```
results_fillblank/
├── iter_0002000/
│   ├── results_20250101_120000.json    # 评估结果
│   └── evaluation.log                  # 运行日志
├── iter_0004000/
│   └── ...
...
└── iter_0054000/
    └── ...
```

### 结果文件格式

```json
{
  "total": 8500,
  "correct": 3200,
  "accuracy": 0.3765,
  "exact_match": 2500,
  "contains_match": 500,
  "numeric_match": 200,
  "by_subject": {
    "abstract_algebra": {
      "total": 100,
      "correct": 45,
      "accuracy": 0.45
    },
    ...
  },
  "samples": [
    {
      "index": 0,
      "subject": "abstract_algebra",
      "question": "...",
      "reference_answer": "4",
      "generated_answer": "4",
      "is_correct": true,
      "match_type": "exact_match"
    },
    ...
  ]
}
```

## 📊 查看结果

### 命令行查看

```bash
# 查看所有结果
python3 view_results_fillblank.py

# 指定结果目录
python3 view_results_fillblank.py --results_dir results_fillblank_test

# 显示更多学科
python3 view_results_fillblank.py --top_n 20
```

### 输出示例

```
==============================================================================
总体评估结果
==============================================================================

Checkpoint           准确率       正确/总数        精确匹配      包含匹配      数字匹配
------------------------------------------------------------------------------
iter_0002000         25.50%       2168/8500          1800          250          118
iter_0028000         35.20%       2992/8500          2400          400          192
iter_0054000         42.10%       3579/8500          2900          500          179

------------------------------------------------------------------------------
🏆 最佳 Checkpoint: iter_0054000 (42.10%)
==============================================================================

==============================================================================
学科表现分析（使用最佳checkpoint）
==============================================================================

Checkpoint: iter_0054000
总体准确率: 42.10%

📈 表现最好的 10 个学科:
学科                                          准确率       正确/总数
------------------------------------------------------------------------------
computer_security                             65.00%       65/100
formal_logic                                  58.73%       74/126
...
```

## 🎯 性能优化

### GPU 内存不足

```bash
# 方法 1: 降低数据类型精度
DTYPE="float16"

# 方法 2: 减少生成长度
MAX_NEW_TOKENS=30

# 方法 3: 限制样本数
MAX_SAMPLES=100  # 只评估100个样本
```

### 加速评估

```bash
# 方法 1: 减少 checkpoints
CHECKPOINTS=(
    "iter_0002000"
    "iter_0028000"
    "iter_0054000"
)

# 方法 2: 并行评估（如果有多GPU）
# 手动在不同终端运行不同的 checkpoint

# 方法 3: 使用更大的batch size
# （当前实现为逐个样本，可修改代码支持批处理）
```

### 后台运行

```bash
# 方法 1: nohup
nohup bash run_evaluation_fillblank.sh > fillblank_eval.log 2>&1 &
tail -f fillblank_eval.log

# 方法 2: screen
screen -S fillblank_eval
bash run_evaluation_fillblank.sh
# 按 Ctrl+A, D 分离
# screen -r fillblank_eval 恢复
```

## ⚠️ 注意事项

### 1. 数据格式不兼容

❌ **不能使用** lm-evaluation-harness 的标准 MMLU 任务
✅ **必须使用** 本目录提供的自定义评估脚本

### 2. 评估时间

- 快速测试（3个checkpoints × 50样本）: ~15-30分钟
- 完整评估（27个checkpoints × 8500样本）: 预计数小时

### 3. 答案匹配限制

填空题评估比多选题更具挑战性：
- 模型可能生成正确含义但格式不同的答案
- 当前匹配策略可能过于严格或宽松
- 建议检查 `samples` 字段中的具体样本

### 4. GPU 要求

- 推荐: 16GB+ VRAM
- 最低: 8GB VRAM (使用 float16)

## 🔍 故障排除

### Q1: ModuleNotFoundError: No module named 'datasets'

```bash
# 激活虚拟环境
cd /home2/yth/lm-evaluation-harness-competition
source .venv/bin/activate

# 安装缺失的包
pip install datasets transformers torch
```

### Q2: CUDA out of memory

```bash
# 编辑 run_evaluation_fillblank.sh
DTYPE="float16"          # 使用更低精度
MAX_NEW_TOKENS=30        # 减少生成长度
MAX_SAMPLES=1000         # 限制样本数
```

### Q3: 数据集未找到

```bash
# 检查数据集是否存在
ls -la ~/.cache/huggingface/datasets/mmlu_fill_blank_dataset_8500/

# 如果不存在，请确认数据集路径
python3 inspect_custom_mmlu.py
```

### Q4: 生成的答案总是错误

检查以下几点：
1. Prompt 格式是否合适
2. max_new_tokens 是否足够
3. 匹配策略是否需要调整
4. 查看具体样本判断模型输出质量

## 📚 相关命令

```bash
# 查看数据集信息
python3 inspect_custom_mmlu.py

# 检查标准MMLU数据集
bash check_mmlu_data.sh

# 查看数据格式说明
cat CUSTOM_MMLU_FORMAT.md

# 比较两个数据集
cat MMLU_DATA_FORMAT.md
cat CUSTOM_MMLU_FORMAT.md
```

## 🎓 总结

### 核心流程

1. ✅ 数据集已准备好（8500个填空题）
2. ✅ 评估脚本已创建（独立于lm-evaluation-harness）
3. ✅ 支持批量评估所有checkpoints
4. ✅ 自动生成详细结果和统计

### 开始评估

```bash
# 第一步：快速测试
cd /home2/yth/dense500m_evaluation
bash quick_test_fillblank.sh

# 第二步：查看测试结果
python3 view_results_fillblank.py --results_dir results_fillblank_test

# 第三步：如果正常，运行完整评估
bash run_evaluation_fillblank.sh

# 第四步：查看完整结果
python3 view_results_fillblank.py
```

---

**创建时间**: 2025-10-22  
**状态**: ✅ 就绪

🚀 **立即开始**: `bash quick_test_fillblank.sh`

