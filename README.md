# NeurIPS-2025-E2LM-Competition-Early-Training-Evaluation-of-Language-Models-our-solution-
We propose MMLU-Curriculum for the E2LM challenge, a curriculum-based benchmark that ranks MMLU tasks by difficulty using small model performance, assigns easier tasks early and harder ones later, and converts MCQs to cloze-style generation for smoother, more discriminative evaluation.

# SmolLM2-1.7B 模型评估项目（后续随意更改其他模型）

> **一句话总结**: 使用 lm-evaluation-harness 评估 SmolLM2-1.7B 模型在 8 个标准基准测试上的性能

**评估完成时间**: 2025-10-19 12:36  
**总题目数**: ~34,544 题  
**评估任务**: 8 个主任务 + 61 个 MMLU 子任务

---

## ⚠️ 重要依赖

**本项目依赖于 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)，必须先安装！**

### 快速安装

```bash
# 1. 克隆 lm-evaluation-harness
cd /home2/yth  # 或您的工作目录
git clone https://github.com/tiiuae/lm-evaluation-harness-competition
cd lm-evaluation-harness-competition

# 2. 创建虚拟环境并安装
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple  # 使用清华镜像加速

# 3. 验证安装
lm_eval --help
```

**注意**: 如果您将 `lm-evaluation-harness-competition` 安装到其他位置，需要修改 `run_evaluation.sh` 中的 `LM_EVAL_DIR` 变量。

---

## 📊 SmlLLM2评估结果概览

| 数据集 | 题目数 | 准确率 | 归一化准确率 | 评级 |
|--------|--------|--------|--------------|------|
| **PIQA** | 1,838 | 74.65% | 74.81% | ⭐⭐⭐⭐⭐ |
| **ARC-Easy** | 2,376 | 72.73% | 69.70% | ⭐⭐⭐⭐⭐ |
| **BoolQ** | 3,270 | 62.94% | 62.94% | ⭐⭐⭐⭐ |
| **HellaSwag** | 10,042 | 46.92% | 61.96% | ⭐⭐⭐⭐ |
| **WinoGrande** | 1,267 | 60.85% | 60.85% | ⭐⭐⭐⭐ |
| **ARC-Challenge** | 1,172 | 37.63% | 40.27% | ⭐⭐⭐ |
| **OpenBookQA** | 500 | 28.00% | 39.40% | ⭐⭐⭐ |
| **MMLU** | ~14,079 | 23.62% | 23.62% | ⭐⭐ |

**模型优势**: 物理常识 (74.65%)、基础科学 (72.73%)、阅读理解 (62.94%)

---

## 🚀 快速开始

### 1. 查看评估结果

```bash
cd /home2/yth/smollm2_evaluation
 
# 查看结果 JSON
cat results/step-125000/HuggingFaceTB__SmolLM2-1.7B-intermediate-checkpoints/results_*.json | python3 -m json.tool

# 或使用脚本查看
python3 view_questions.py hellaswag --stats
```

### 2. 查看具体题目

```bash
# 查看 5 个 HellaSwag 题目
python3 view_questions.py hellaswag 5

# 查看 3 个 ARC-Challenge 题目
python3 view_questions.py arc_challenge 3

# 查看任何任务的统计
python3 view_questions.py mmlu_anatomy --stats
```

### 3. 重新运行评估（如需要）

```bash
# 激活环境
cd /home2/yth/lm-evaluation-harness-competition
source .venv/bin/activate

# 运行评估（使用镜像加速）
cd /home2/yth/smollm2_evaluation
bash run_evaluation.sh
```

---

## 📂 项目结构

```
smollm2_evaluation/
│
├── 📊 评估结果
│   └── results/step-125000/
│       ├── results_*.json                    # 总结果（准确率等）
│       └── samples_*.jsonl (69个文件)        # 所有题目 + 模型答案
│
├── 🔧 脚本工具
│   ├── view_questions.py                     # 查看题目（推荐使用！）
│   ├── run_evaluation.sh                     # 运行评估
│   ├── create_submission.sh                  # 创建提交文件
│   └── plot_results.py                       # 结果可视化
│
└── 📖 文档
    ├── README.md                             # 本文件（主入口）
    └── GUIDE.md                              # 详细指南（可选阅读）
```

---

## 🎯 核心任务说明

### 您在做什么？

**类比**: 像是给 AI 学生（SmolLM2）进行全面考试

```
输入: SmolLM2-1.7B 模型 (checkpoint: step-125000)
  ↓
处理: 在 8 个标准测试上运行
  • HellaSwag     - 常识推理
  • ARC           - 科学问答
  • MMLU          - 57 个学科知识
  • PIQA          - 物理常识
  • WinoGrande    - 语言理解
  • OpenBookQA    - 开放问答
  • BoolQ         - 是非判断
  ↓
输出: 性能指标（准确率、标准误差等）
```

### 数据从哪里来？

✅ **所有数据来自 HuggingFace Hub**（自动下载）

| 数据集 | HuggingFace 地址 |
|--------|------------------|
| HellaSwag | `hellaswag` |
| ARC | `allenai/ai2_arc` |
| MMLU | `cais/mmlu` |
| PIQA | `baber/piqa` |
| WinoGrande | `winogrande` |
| OpenBookQA | `openbookqa` |
| BoolQ | `boolq` |

**缓存位置**:
- 模型: `~/.cache/huggingface/hub/`
- 数据集: `~/.cache/huggingface/datasets/`

---

## 📝 题目在哪里？

### 主要位置（推荐查看）⭐

```bash
/home2/yth/smollm2_evaluation/results/step-125000/
  HuggingFaceTB__SmolLM2-1.7B-intermediate-checkpoints/
    └── samples_*.jsonl  (69个文件)
```

**文件列表**:
```
samples_hellaswag_*.jsonl          → 10,042 题
samples_arc_easy_*.jsonl           → 2,376 题
samples_arc_challenge_*.jsonl      → 1,172 题
samples_piqa_*.jsonl               → 1,838 题
samples_winogrande_*.jsonl         → 1,267 题
samples_openbookqa_*.jsonl         → 500 题
samples_boolq_*.jsonl              → 3,270 题
samples_mmlu_*_*.jsonl (57个)      → ~14,079 题
```

### 查看方法

#### 方法 1: 使用脚本（最简单！）

```bash
cd /home2/yth/smollm2_evaluation

# 查看题目（带格式化输出）
python3 view_questions.py hellaswag 3

# 输出示例:
# ======================================================================
# 题目 #1
# ======================================================================
# 📝 问题: A glass of cold water is set on a desktop...
# 📋 选项:
#    🤖 [A] some water dripped over the side.  
#       [B] the water soaked through the glass.  
#       [C] water vapor condensed on the sides. ✓
#       [D] someone sprayed the glass with water.  
# ✅ 正确答案: 2 (C)
# 🤖 模型答案: 0 (A)
# ❌ 结果: 答错了
```

#### 方法 2: 直接查看 JSON

```bash
cd results/step-125000/HuggingFaceTB__SmolLM2-1.7B-intermediate-checkpoints/

# 查看第一个题目（格式化）
head -1 samples_hellaswag_*.jsonl | python3 -m json.tool

# 在编辑器中打开
nano samples_hellaswag_*.jsonl
```

#### 方法 3: 统计分析

```bash
# 统计题目总数
wc -l samples_*.jsonl

# 查看任务统计
python3 view_questions.py hellaswag --stats
```

### 题目文件格式

每个 `.jsonl` 文件包含多行，每行是一个 JSON 对象（一个题目）：

```json
{
  "doc_id": 0,
  "doc": {
    "question": "题目文本",
    "choices": {"text": ["A", "B", "C", "D"]},
    "answerKey": "C"
  },
  "target": "2",                           // 正确答案索引
  "resps": [[...], [...], [...], [...]],  // 模型对每个选项的评分
  "filtered_resps": [[-18.875], [-25.125], [-21.875], [-15.250]]
}
```

**关键字段**:
- `doc`: 原始题目（问题、选项、正确答案）
- `target`: 正确答案的索引（0, 1, 2, 3）
- `filtered_resps`: 模型分数（越高 = 越可能选择）

---

## 🎓 评估任务详解

### 数据集说明

#### 1. HellaSwag（常识推理）✅
- **题目数**: 10,042
- **类型**: 情境续写
- **示例**: "一个人坐在屋顶上。他..." → 选择最合理的后续
- **您的表现**: 61.96% (归一化)

#### 2. ARC（科学问答）✅
- **ARC-Easy**: 2,376 题，小学到初中水平，**72.73%** ⭐⭐⭐⭐⭐
- **ARC-Challenge**: 1,172 题，高中水平，40.27%
- **类型**: 多项选择科学题
- **示例**: "哪种能源利用重力？" → A. 潮汐能和水能

#### 3. MMLU（多学科理解）✅
- **题目数**: ~14,079（57 个学科）
- **学科**: 数学、物理、化学、生物、历史、法律、医学等
- **类型**: 4 选 1 多项选择
- **您的表现**: 23.62%（接近随机 25%，这对小模型是正常的）

#### 4. PIQA（物理常识）✅
- **题目数**: 1,838
- **类型**: 选择实现目标的合理方法
- **示例**: "如何在家做冰淇淋？" → A. 放冰箱过夜
- **您的表现**: **74.65%** ⭐⭐⭐⭐⭐（最佳！）

#### 5. WinoGrande（语言理解）✅
- **题目数**: 1,267
- **类型**: 代词消歧和填空
- **示例**: "The trophy doesn't fit in the suitcase because ___ is too large."
- **您的表现**: 60.85%

#### 6. OpenBookQA（开放问答）✅
- **题目数**: 500
- **类型**: 需要推理的科学问答
- **您的表现**: 39.40% (归一化)

#### 7. BoolQ（是非判断）✅
- **题目数**: 3,270
- **类型**: 基于段落回答 True/False
- **您的表现**: 62.94%

---

## 🔧 自定义评估

### 评估单个任务

```bash
cd /home2/yth/lm-evaluation-harness-competition
source .venv/bin/activate

lm_eval --model hf \
    --model_args pretrained=HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints,revision=step-125000,dtype=bfloat16 \
    --tasks hellaswag \
    --batch_size auto \
    --output_path results/single_task/
```

### 评估多个 checkpoint

编辑 `run_evaluation.sh` 中的 `CHECKPOINTS` 数组：

```bash
CHECKPOINTS=(
    "step-100000"
    "step-125000"  # 已完成
    "step-150000"
)

# 重新运行
bash run_evaluation.sh
```

### 快速测试（限制样本数）

```bash
lm_eval --model hf \
    --model_args pretrained=HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints,revision=step-125000 \
    --tasks hellaswag \
    --limit 10 \
    --output_path results/quick_test/
```

---

## 📈 结果可视化

### 生成图表

```bash
python plot_results.py
```

生成文件:
- `results_comparison.png` - 各任务性能对比
- `average_performance.png` - 平均性能趋势

---

## 📦 创建提交文件

```bash
# 创建 submission.zip
bash create_submission.sh --include-results

# 验证
unzip -l submission.zip
```

**提交内容**:
```
submission.zip
├── evaluation.patch      # Git diff（如有代码修改）
├── metadata.yaml         # 评估元数据
└── results/              # 评估结果（可选）
```

---

## 💡 评估指标说明

### 准确率类型

- **acc (Accuracy)**: 原始准确率 = 正确数 / 总题数
- **acc_norm (Normalized Accuracy)**: 归一化准确率（推荐使用！）

### 为什么使用归一化准确率？

**问题**: 多 token 答案会受惩罚

```
"Abu Dhabi" → 2 tokens: P(Abu) × P(Dhabi|Abu)
"Dubai"     → 1 token:  P(Dubai)

因为概率 ≤ 1，多 token 答案的概率会更低
```

**解决**: 归一化 = 平均每个 token 的对数概率

```
acc_norm = log(概率) / token数量
```

这样可以**公平比较不同长度的答案**。

---

## 🛠️ 常用命令

### 查看题目

```bash
# 查看题目（推荐！）
python3 view_questions.py hellaswag 5
python3 view_questions.py arc_challenge 3
python3 view_questions.py mmlu_anatomy 10

# 查看统计
python3 view_questions.py hellaswag --stats

# 列出所有可用任务
python3 view_questions.py
```

### 查看结果

```bash
# 进入结果目录
cd results/step-125000/HuggingFaceTB__SmolLM2-1.7B-intermediate-checkpoints/

# 查看所有样本文件
ls -lh samples_*.jsonl

# 统计题目总数
wc -l samples_*.jsonl

# 查看结果 JSON
cat results_*.json | python3 -m json.tool | less
```

### 系统检查

```bash
# 查看 GPU
nvidia-smi

# 查看缓存大小
du -sh ~/.cache/huggingface/*

# 列出可用任务
cd /home2/yth/lm-evaluation-harness-competition
source .venv/bin/activate
lm_eval --tasks list | grep -E "hellaswag|arc|mmlu"
```

---

## ⚠️ 常见问题

### 1. 下载速度慢

**问题**: 模型/数据集下载只有几十 KB/s

**解决**: 使用国内镜像

```bash
# 临时设置
export HF_ENDPOINT=https://hf-mirror.com

# 永久设置
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

### 2. GPU 内存不足

**解决方案**:
```bash
# 方法 1: 减小 batch_size
lm_eval ... --batch_size 4

# 方法 2: 使用更小的数据类型
lm_eval ... --model_args dtype=float16

# 方法 3: 单独评估任务
lm_eval ... --tasks hellaswag  # 一次只评估一个
```

### 3. 模型文件损坏

**错误**: `SafetensorError`

**解决**:
```bash
# 清除损坏缓存
rm -rf ~/.cache/huggingface/hub/models--HuggingFaceTB*

# 重新下载（使用镜像）
export HF_ENDPOINT=https://hf-mirror.com
bash run_evaluation.sh
```

### 4. 重复警告信息

**现象**: `trust_remote_code is not supported anymore` 重复多次

**解释**: 
- 这是**警告**，不是错误
- MMLU 有 57 个子任务，每个都显示一次
- **不影响评估结果**，可以安全忽略

---

## 📊 性能分析

### 模型优势领域

✅ **物理常识** (74.65%)  
✅ **基础科学** (72.73%)  
✅ **阅读理解** (62.94%)  
✅ **情境推理** (61.96%)  
✅ **语言理解** (60.85%)  

### 待改进领域

⚠️ **专业知识** (23.62%) - MMLU 涵盖大学级知识，对 1.7B 小模型有挑战  
⚠️ **深层推理** (39.40%) - OpenBookQA 需要多步推理  

### 与随机猜测对比

| 数据集 | 随机准确率 | 您的模型 | 提升 |
|--------|-----------|---------|------|
| HellaSwag | 25% | **61.96%** | +147% |
| ARC-Easy | 25% | **72.73%** | +191% |
| PIQA | 50% | **74.65%** | +49% |
| BoolQ | 50% | **62.94%** | +26% |

**结论**: 模型在所有任务上都**明显优于随机猜测**！

---

## 🔗 重要路径

```bash
# 项目目录
/home2/yth/smollm2_evaluation/

# 评估框架
/home2/yth/lm-evaluation-harness-competition/

# 模型缓存
~/.cache/huggingface/hub/

# 数据集缓存
~/.cache/huggingface/datasets/

# 评估结果
/home2/yth/smollm2_evaluation/results/step-125000/
```

---

## 📚 参考资源

### 文档

- **README.md** (本文件) - 快速入口和常用操作
- **GUIDE.md** - 详细技术指南（深入了解时阅读）

### 外部链接

- [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [SmolLM2 模型](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints)
- [集成指南](https://huggingface.co/blog/integrating-benchmarks-lm-eval)

### 数据集主页

- [HellaSwag](https://rowanzellers.com/hellaswag/)
- [ARC](https://allenai.org/data/arc)
- [MMLU](https://github.com/hendrycks/test)
- [PIQA](https://yonatanbisk.com/piqa/)

---

## 🎓 总结

### 您完成了什么

1. ✅ **环境搭建** - Python, PyTorch, lm-eval
2. ✅ **模型评估** - 在 8 个基准测试上运行
3. ✅ **结果生成** - 获得了详细的性能数据
4. ✅ **题目保存** - 所有 ~34,544 个题目都已保存

### 关键成果

- **性能数据**: 8 个任务的完整评估结果
- **题目库**: 34,544 个题目 + 模型答案
- **可视化工具**: 查看和分析脚本
- **可复用框架**: 可用于其他模型评估

### 下一步

1. **分析结果**: 使用 `view_questions.py` 查看具体题目
2. **生成图表**: 运行 `plot_results.py`
3. **创建提交**: 运行 `create_submission.sh`
4. **评估其他版本**: 修改 checkpoint 重新评估

---

**项目创建**: 2025-10-19  
**评估完成**: 2025-10-19 12:36  
**总耗时**: 约 33 分钟

**🎉 评估完成！祝您使用愉快！**

如有疑问，请查看 `GUIDE.md` 获取详细说明。
