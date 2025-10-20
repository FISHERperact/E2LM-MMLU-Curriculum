# Dense-500M-Arch1 模型 MMLU 评估项目

> **一句话总结**: 评估 tiiuae/dense-500m-arch1 模型的 27 个训练 checkpoints 在 MMLU 上的性能，观察学习曲线

**模型**: [tiiuae/dense-500m-arch1](https://huggingface.co/tiiuae/dense-500m-arch1) (500M 参数)  
**任务**: MMLU (57 个学科，~14,079 题)  
**Checkpoints**: 27 个 (iter_0002000 到 iter_0054000)  
**目标**: 观察训练过程中 2k-54k 步的性能变化趋势

---

## ⚠️ 重要说明

### 依赖

**必须先安装 [lm-evaluation-harness](https://github.com/tiiuae/lm-evaluation-harness-competition)！**

```bash
# 1. 克隆并安装
git clone https://github.com/tiiuae/lm-evaluation-harness-competition /home2/yth/lm-evaluation-harness-competition
cd /home2/yth/lm-evaluation-harness-competition
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 验证安装
lm_eval --help
```

### 模型结构（关键！）

**Dense-500M 与 SmolLM2 不同**：

```
SmolLM2:        Checkpoints 是 git branches  → 使用 revision=step-125000
Dense-500M:     Checkpoints 是子目录        → 使用 subfolder=iter_0002000
```

**本项目已正确配置！** 使用 `subfolder` 参数。

---

## 🚀 快速开始

### 方法 1: 一键启动（推荐）

```bash
cd /home2/yth/dense500m_evaluation
bash START_DENSE500M.sh
```

交互式菜单：
1. 快速测试 (~10 分钟) - 测试 3 个 checkpoints，每个 100 样本
2. 完整评估 (~13-27 小时) - 评估所有 27 个 checkpoints
3. 查看结果
4. 生成图表
5. 查看题目

### 方法 2: 手动运行

```bash
cd /home2/yth/dense500m_evaluation

# 步骤 1: 快速测试（强烈推荐先运行！）
bash quick_test_dense500m.sh                # ~10 分钟

# 步骤 2: 查看测试结果
python3 view_results_dense500m.py

# 步骤 3: 运行完整评估（如果测试正常）
bash run_evaluation_dense500m.sh            # 13-27 小时

# 步骤 4: 生成可视化图表
python3 plot_results_dense500m.py
```

---

## 📊 Checkpoints 配置

### 全部 27 个 Checkpoints

```
iter_0002000  (2k)    iter_0012000  (12k)   iter_0022000  (22k)   iter_0032000  (32k)   iter_0042000  (42k)   iter_0052000  (52k)
iter_0004000  (4k)    iter_0014000  (14k)   iter_0024000  (24k)   iter_0034000  (34k)   iter_0044000  (44k)   iter_0054000  (54k)
iter_0006000  (6k)    iter_0016000  (16k)   iter_0026000  (26k)   iter_0036000  (36k)   iter_0046000  (46k)
iter_0008000  (8k)    iter_0018000  (18k)   iter_0028000  (28k)   iter_0038000  (38k)   iter_0048000  (48k)
iter_0010000  (10k)   iter_0020000  (20k)   iter_0030000  (30k)   iter_0040000  (40k)   iter_0050000  (50k)
```

### 自定义 Checkpoints

编辑 `run_evaluation_dense500m.sh`，只评估关键点：

```bash
# 示例：只评估 7 个关键点（每 8k 步），耗时降至 ~7 小时
CHECKPOINTS=(
    "iter_0002000"
    "iter_0010000"
    "iter_0018000"
    "iter_0026000"
    "iter_0034000"
    "iter_0042000"
    "iter_0050000"
)
```

---

## 📈 结果分析

### 查看文字结果

```bash
python3 view_results_dense500m.py
```

输出：
- 每个 checkpoint 的 MMLU 分数
- 按类别统计（人文、社科、STEM、其他）
- 表现最好/最差的 5 个学科
- 最佳 checkpoint 识别

### 生成可视化图表

```bash
python3 plot_results_dense500m.py
```

生成 `dense500m_mmlu_results.png`：
1. 性能曲线（准确率 vs 训练步数）
2. 首尾对比柱状图
3. 统计信息（最低、最高、平均、提升幅度）

### 查看具体题目

```bash
# 查看 5 个 MMLU anatomy 题目
python3 view_questions.py mmlu_anatomy 5

# 查看任务统计
python3 view_questions.py mmlu_computer_science --stats
```

---

## 🎯 MMLU 数据集

### 简介

**MMLU** = Massive Multitask Language Understanding
- **题目数**: ~14,079
- **学科数**: 57 个
- **类型**: 4 选 1 多项选择题
- **论文**: [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)

### 学科分类

- **人文学科 (13个)**: 历史、法律、哲学、宗教等
- **社会科学 (12个)**: 经济学、心理学、政治学、社会学等
- **STEM (24个)**: 数学、物理、化学、生物、计算机等
- **其他 (8个)**: 商业、医学、营养学等

### 题目示例

```
[Computer Science]
问题: What is the time complexity of binary search?
A. O(1)
B. O(log n)  ✓
C. O(n)
D. O(n log n)

模型会计算每个选项的对数概率，选择最高的作为答案。
```

---

## 📂 输出结构

```
results_dense500m/
├── iter_0002000/
│   ├── tiiuae__dense-500m-arch1/
│   │   ├── results_*.json                    # 总结果
│   │   └── samples_mmlu_*.jsonl (57个)       # 每个学科的样本
│   └── evaluation.log
├── iter_0004000/
│   └── ...
... (共 27 个)
└── iter_0054000/
    └── ...
```

---

## 💡 预期结果

### 性能曲线

```
MMLU 准确率
    ↑
45% |                                      ╱────────
40% |                               ╱─────╯
35% |                        ╱─────╯
30% |                 ╱─────╯
25% |─────────╱──────╯
20% |────────╯
    └──────────────────────────────────────────────→
    2k    10k    20k    30k    40k    50k   训练步数

阶段:
  早期 (2k-10k):  20-25% (接近随机 25%)
  中期 (10k-30k): 25-35% (快速提升)
  后期 (30k-54k): 35-45% (趋于收敛)
```

### 与 SmolLM2-1.7B 对比

| 模型 | 参数量 | MMLU 准确率 |
|------|--------|------------|
| SmolLM2-1.7B | 1.7B | 23.62% |
| Dense-500M (预期) | 500M | ~35-45% |

---

## 🔧 自定义配置

### GPU 内存不足

```bash
# 编辑 run_evaluation_dense500m.sh

BATCH_SIZE="4"      # 减小 batch size
DTYPE="float16"     # 使用 float16 节省内存
```

### 加速评估

```bash
# 方法 1: 减少 checkpoints（推荐）
# 只评估 7 个关键点

# 方法 2: 增大 batch size
BATCH_SIZE="16"     # 如果 GPU 内存足够

# 方法 3: 快速测试模式
lm_eval ... --limit 500  # 只测试 500 个样本
```

### 后台运行

```bash
# 方法 1: nohup
nohup bash run_evaluation_dense500m.sh > evaluation.log 2>&1 &
tail -f evaluation.log

# 方法 2: screen (推荐)
screen -S dense500m
bash run_evaluation_dense500m.sh
# 按 Ctrl+A, D 分离
# screen -r dense500m 恢复
```

---

## ⚠️ 常见问题

### Q1: RevisionNotFoundError

**错误**: `Invalid rev id: iter_0002000`

**原因**: Dense-500M 的 checkpoints 是子目录，不是 git branches

**解决**: ✅ 脚本已使用 `subfolder` 参数，无需修改

### Q2: 下载速度慢

```bash
# 使用镜像（脚本中已设置）
export HF_ENDPOINT=https://hf-mirror.com
```

### Q3: 评估太慢

- 减少 checkpoints 数量（编辑脚本，只保留 7-10 个）
- 使用 `--limit 500` 限制样本数
- 增大 `batch_size`

### Q4: 中断和恢复

- 按 `Ctrl+C` 中断
- 已完成的 checkpoint 结果会保存
- 重新运行会从下一个 checkpoint 继续

---

## 🛠️ 常用命令

```bash
# 查看评估进度
ls -lh results_dense500m/
ls -d results_dense500m/*/ | wc -l

# 查看最新日志
tail -f results_dense500m/iter_*/evaluation.log

# 查看 GPU
watch -n 1 nvidia-smi

# 查看具体题目
python3 view_questions.py mmlu_anatomy 5

# 清理测试结果
rm -rf results_dense500m_test/
```

---

## 📚 参考资源

- **模型**: https://huggingface.co/tiiuae/dense-500m-arch1
- **MMLU 论文**: https://arxiv.org/abs/2009.03300
- **MMLU 数据集**: https://huggingface.co/datasets/cais/mmlu
- **lm-evaluation-harness**: https://github.com/EleutherAI/lm-evaluation-harness
- **集成指南**: https://huggingface.co/blog/integrating-benchmarks-lm-eval

---

## 🎓 总结

### 项目特点

✅ 自动化 - 一键评估所有 27 个 checkpoints  
✅ 可观测 - 实时进度和日志  
✅ 可中断 - 随时中断和恢复  
✅ 可视化 - 自动生成性能曲线  
✅ 快速测试 - 10 分钟验证环境  

### 预期收获

1. **性能曲线**: 清晰看到模型的学习过程
2. **最佳 checkpoint**: 识别性能最优的训练步数
3. **学科分析**: 了解模型在不同领域的表现
4. **研究价值**: 可用于论文、报告或进一步研究

---

## 📖 快速参考

| 命令 | 用途 | 耗时 |
|------|------|------|
| `bash START_DENSE500M.sh` | 一键启动菜单 | - |
| `bash quick_test_dense500m.sh` | 快速测试 | 10 分钟 |
| `bash run_evaluation_dense500m.sh` | 完整评估 | 13-27 小时 |
| `python3 view_results_dense500m.py` | 查看结果 | - |
| `python3 plot_results_dense500m.py` | 生成图表 | - |
| `python3 view_questions.py mmlu_anatomy 5` | 查看题目 | - |

---

**项目创建**: 2025-10-20  
**状态**: 已修复 subfolder 参数，可以运行

🚀 **立即开始**: `bash quick_test_dense500m.sh`
