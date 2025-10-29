# 标准MMLU（多选题）评估 - 快速指南

## 📍 当前位置
您在 `Multiple-choice questions/` 目录中

## 🎯 评估内容
- **数据集**: cais/mmlu（标准MMLU基准）
- **学科数**: 57 个
- **样本数**: ~14,079 题
- **任务**: 4选1多项选择题
- **结果**: 存储在 `results_dense500m/`（已包含27个checkpoints的评估结果）

## 🚀 快速开始

```bash
# 方法1: 使用一键启动菜单（推荐）
bash START_DENSE500M.sh

# 方法2: 直接运行快速测试
bash quick_test_dense500m.sh

# 方法3: 查看已有结果
python3 view_results_dense500m.py

# 方法4: 生成图表
python3 plot_results_dense500m.py
```

## 📊 查看结果

已完成的评估结果在 `results_dense500m/` 目录中：
- 27个checkpoints（iter_0002000 到 iter_0054000）
- 每个checkpoint包含57个学科的详细结果

## 📚 详细文档

完整使用说明请查看: `README.md`

## 🔙 返回主目录

```bash
cd ..
cat README.md  # 查看项目总览
```

