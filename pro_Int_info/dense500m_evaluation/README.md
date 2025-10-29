# Dense-500M-Arch1 模型 MMLU 评估项目

> **项目总览**: 评估 tiiuae/dense-500m-arch1 模型在两种不同MMLU数据集上的性能

**模型**: [tiiuae/dense-500m-arch1](https://huggingface.co/tiiuae/dense-500m-arch1) (500M 参数)  
**Checkpoints**: 27 个 (iter_0002000 到 iter_0054000)

---

## 📂 项目结构

```
dense500m_evaluation/
├── Multiple-choice questions/      # 标准MMLU（多选题）评估
│   ├── 数据集: cais/mmlu (57学科, ~14,079题)
│   ├── 任务类型: 4选1多项选择题
│   └── 结果目录: results_dense500m/
│
├── Fill-in-the-blank questions/    # 自定义MMLU（填空题）评估
│   ├── 数据集: mmlu_fill_blank_dataset_8500 (43学科, 8,500题)
│   ├── 任务类型: 生成式填空题
│   └── 结果目录: results_fillblank/
│
└── README.md                        # 本文件
```

---

## 🚀 快速开始

### 方法 1: 标准MMLU（多选题）评估

```bash
cd "/home2/yth/dense500m_evaluation/Multiple-choice questions"

# 快速测试（推荐第一次运行）
bash quick_test_dense500m.sh

# 或使用一键启动菜单
bash START_DENSE500M.sh

# 查看完整使用说明
cat README.md
```

**特点**:
- ✅ 使用 lm-evaluation-harness 官方框架
- ✅ 标准的多选题格式（4选1）
- ✅ 57个学科，~14,079道题
- ✅ 与官方MMLU基准完全兼容

### 方法 2: 填空题MMLU评估

```bash
cd "/home2/yth/dense500m_evaluation/Fill-in-the-blank questions"

# 快速测试（3个checkpoints × 50样本）
bash quick_test_fillblank.sh

# 完整评估（27个checkpoints × 8,500样本）
bash run_evaluation_fillblank.sh

# 查看完整使用说明
cat README_FILLBLANK.md
```

**特点**:
- ✅ 自定义评估脚本（生成式任务）
- ✅ 填空题格式（含[BLANK]标记）
- ✅ 43个学科，8,500道题
- ✅ 支持精确匹配、包含匹配、数字匹配

---

## 📊 两种评估方式对比

| 特性 | 标准MMLU（多选题） | 填空题MMLU |
|------|-------------------|-----------|
| **数据集** | cais/mmlu | mmlu_fill_blank_dataset_8500 |
| **学科数** | 57 个 | 43 个 |
| **样本数** | ~14,079 | 8,500 |
| **任务类型** | 多选题（4选1） | 填空题（生成式） |
| **输入格式** | 问题 + ABCD选项 | 问题 + [BLANK]标记 |
| **输出方式** | 选择A/B/C/D | 生成文本答案 |
| **评估方法** | 对数概率比较 | 文本匹配 |
| **评估框架** | lm-evaluation-harness | 自定义Python脚本 |
| **兼容性** | ✅ 官方基准 | ❌ 自定义格式 |
| **评估时间** | ~30分钟/checkpoint | ~20分钟/checkpoint |
| **适用场景** | 标准benchmark对比 | 生成能力评估 |

---

## 🎯 评估流程示例

### 标准MMLU评估流程

```bash
# 1. 进入多选题目录
cd "/home2/yth/dense500m_evaluation/Multiple-choice questions"

# 2. 运行快速测试
bash quick_test_dense500m.sh

# 3. 查看测试结果
python3 view_results_dense500m.py

# 4. 如果测试正常，运行完整评估
bash run_evaluation_dense500m.sh

# 5. 生成可视化图表
python3 plot_results_dense500m.py

# 6. 查看具体题目
python3 view_questions.py mmlu_anatomy 5
```

### 填空题评估流程

```bash
# 1. 进入填空题目录
cd "/home2/yth/dense500m_evaluation/Fill-in-the-blank questions"

# 2. 运行快速测试
bash quick_test_fillblank.sh

# 3. 查看测试结果
python3 view_results_fillblank.py --results_dir results_fillblank_test

# 4. 如果测试正常，运行完整评估
bash run_evaluation_fillblank.sh

# 5. 查看完整结果
python3 view_results_fillblank.py
```

---

## 📁 结果存储说明

### 标准MMLU结果

```
Multiple-choice questions/
└── results_dense500m/              # 已有结果（请勿删除）
    ├── iter_0002000/
    │   └── tiiuae__dense-500m-arch1/
    │       ├── results_*.json
    │       └── samples_mmlu_*.jsonl (57个学科)
    ├── iter_0004000/
    └── ...
```

### 填空题结果

```
Fill-in-the-blank questions/
├── results_fillblank/              # 完整评估结果（新）
│   ├── iter_0002000/
│   │   ├── results_*.json
│   │   └── evaluation.log
│   └── ...
└── results_fillblank_test/         # 快速测试结果（新）
    └── ...
```

**⚠️ 重要**: 两种评估的结果分别存储在各自的目录中，互不影响。

---

## 📚 详细文档

- **标准MMLU评估**: `Multiple-choice questions/README.md`
- **填空题评估**: `Fill-in-the-blank questions/README_FILLBLANK.md`
- **数据格式说明**:
  - 标准MMLU: `Multiple-choice questions/MMLU_DATA_FORMAT.md`
  - 填空题: `Fill-in-the-blank questions/CUSTOM_MMLU_FORMAT.md`

---

## ⚙️ 依赖要求

### 标准MMLU评估

必须安装 [lm-evaluation-harness](https://github.com/tiiuae/lm-evaluation-harness-competition):

```bash
git clone https://github.com/tiiuae/lm-evaluation-harness-competition /home2/yth/lm-evaluation-harness-competition
cd /home2/yth/lm-evaluation-harness-competition
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 填空题评估

使用相同的虚拟环境，需要以下Python包:
- `transformers`
- `torch`
- `datasets`
- `tqdm`

---

## 🔧 常用命令

```bash
# 查看项目结构
tree -L 2 /home2/yth/dense500m_evaluation

# 查看已完成的评估
ls -lh "Multiple-choice questions/results_dense500m/"
ls -lh "Fill-in-the-blank questions/results_fillblank/"

# 查看GPU状态
watch -n 1 nvidia-smi

# 后台运行评估（推荐）
cd "Multiple-choice questions"
nohup bash run_evaluation_dense500m.sh > eval.log 2>&1 &
tail -f eval.log
```

---

## 🎓 使用建议

### 如果您想要...

**1. 与官方基准对比** → 使用 **Multiple-choice questions**
- 标准化的多选题格式
- 官方lm-evaluation-harness框架
- 可直接与其他模型对比

**2. 评估生成能力** → 使用 **Fill-in-the-blank questions**
- 生成式任务，更有挑战性
- 评估模型的实际生成质量
- 自定义评分标准

**3. 全面评估** → **两者都运行**
- 多角度评估模型性能
- 对比多选题vs生成式任务的表现
- 更完整的模型能力画像

---

## 📖 快速参考

| 操作 | 标准MMLU | 填空题MMLU |
|------|---------|-----------|
| 进入目录 | `cd "Multiple-choice questions"` | `cd "Fill-in-the-blank questions"` |
| 快速测试 | `bash quick_test_dense500m.sh` | `bash quick_test_fillblank.sh` |
| 完整评估 | `bash run_evaluation_dense500m.sh` | `bash run_evaluation_fillblank.sh` |
| 查看结果 | `python3 view_results_dense500m.py` | `python3 view_results_fillblank.py` |
| 生成图表 | `python3 plot_results_dense500m.py` | *(待实现)* |

---

## ⚠️ 注意事项

1. **不要混淆结果目录**: 两种评估的结果存储在各自独立的目录中
2. **虚拟环境**: 两种评估可以共用同一个虚拟环境
3. **GPU内存**: 两种评估都需要足够的GPU内存（推荐16GB+）
4. **评估时间**: 完整评估需要较长时间，建议后台运行

---

## 🆘 故障排除

### Q: 找不到lm-evaluation-harness

```bash
# 确认安装路径
ls /home2/yth/lm-evaluation-harness-competition

# 如果不存在，请安装（见依赖要求部分）
```

### Q: 找不到数据集

```bash
# 标准MMLU
ls ~/.cache/huggingface/datasets/cais___mmlu/

# 填空题MMLU
ls ~/.cache/huggingface/datasets/mmlu_fill_blank_dataset_8500/
```

### Q: GPU内存不足

编辑评估脚本，降低精度或batch size:
```bash
DTYPE="float16"    # 使用float16而非bfloat16
BATCH_SIZE="4"     # 减小batch size
```

---

**项目创建**: 2025-10-20  
**最后更新**: 2025-10-22  
**状态**: ✅ 就绪

🚀 **立即开始**: 选择一个评估方式，进入对应目录开始评估！

