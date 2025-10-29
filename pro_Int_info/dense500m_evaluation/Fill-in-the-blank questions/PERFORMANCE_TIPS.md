# 性能优化建议

## 🚀 当前硬件配置

**检测到硬件:**
- **GPU**: 8 × NVIDIA GeForce RTX 3090
- **显存**: 每张 23.6 GB
- **总显存**: ~189 GB
- **状态**: ✅ GPU完全可用

## ⚡ 为什么评估慢？

当前脚本的问题：
1. **逐个样本处理** - 没有批处理
2. **单GPU使用** - device_map="auto" 只用了1张GPU
3. **生成长度** - max_new_tokens=50 可能太长

## 🎯 优化方案

### 方案 1: 减少max_new_tokens（立即生效）

大多数填空题答案很短（1-10个token），设置50太长了。

编辑 `quick_test_fillblank.sh` 或 `run_evaluation_fillblank.sh`:
```bash
MAX_NEW_TOKENS=20  # 从50改为20
```

**预期提升**: 2-3倍速度

### 方案 2: 减少样本数（快速测试）

编辑 `quick_test_fillblank.sh`:
```bash
MAX_SAMPLES=50     # 从默认改为50，快速测试
```

**预期耗时**: ~5-10分钟/checkpoint

### 方案 3: 只评估关键checkpoints

编辑 `run_evaluation_fillblank.sh`，只评估部分checkpoints:
```bash
CHECKPOINTS=(
    "iter_0002000"   # 早期
    "iter_0020000"   # 中期
    "iter_0040000"   # 后期
    "iter_0054000"   # 最终
)
```

**预期耗时**: 从27个减少到4个

### 方案 4: 使用批处理（需要修改代码）

当前是逐个样本生成，改为批处理可以提速5-10倍。

（需要较大修改，可以后续优化）

## 📊 速度参考

### 当前速度（逐个样本）
- **速度**: ~0.5-2 样本/秒
- **8500样本耗时**: ~70-280分钟/checkpoint
- **27个checkpoints**: ~31-126小时

### 优化后速度（减少max_new_tokens到20）
- **速度**: ~1-4 样本/秒
- **8500样本耗时**: ~35-140分钟/checkpoint
- **27个checkpoints**: ~16-63小时

### 如果使用批处理（未实现）
- **速度**: ~5-20 样本/秒
- **8500样本耗时**: ~7-28分钟/checkpoint
- **27个checkpoints**: ~3-13小时

## ✅ 立即可用的优化

### 1. 修改快速测试脚本

```bash
cd "/home2/yth/dense500m_evaluation/Fill-in-the-blank questions"
nano quick_test_fillblank.sh

# 修改以下行:
MAX_NEW_TOKENS=20    # 原来是50
MAX_SAMPLES=50       # 快速测试50个样本
```

### 2. 查看当前评估进度

```bash
# 查看正在运行的评估
watch -n 1 "tail -20 results_fillblank/iter_*/evaluation.log"

# 查看GPU使用
watch -n 1 nvidia-smi
```

### 3. 如果当前评估太慢，可以中断并重新配置

```bash
# Ctrl+C 中断
# 修改配置后重新运行
bash quick_test_fillblank.sh
```

## 💡 推荐配置（快速测试）

```bash
# quick_test_fillblank.sh
CHECKPOINTS=(
    "iter_0002000"   # 早期
    "iter_0054000"   # 最终
)
MAX_NEW_TOKENS=20
MAX_SAMPLES=50
```

**预期耗时**: ~10-20分钟总共

## 💡 推荐配置（完整评估，但更快）

```bash
# run_evaluation_fillblank.sh
MAX_NEW_TOKENS=20    # 从50减到20
```

选择关键checkpoints（如每10000步一个）:
```bash
CHECKPOINTS=(
    "iter_0002000"
    "iter_0010000"
    "iter_0020000"
    "iter_0030000"
    "iter_0040000"
    "iter_0050000"
    "iter_0054000"
)
```

**预期耗时**: ~4-10小时总共

## 🔍 监控评估进度

```bash
# 实时查看日志
tail -f results_fillblank/iter_*/evaluation.log

# 查看GPU使用率
nvidia-smi -l 1

# 查看已完成的数量
ls -d results_fillblank/iter_* | wc -l
```

## 📝 总结

**立即优化（无需改代码）**:
1. ✅ 减少 MAX_NEW_TOKENS 到 20
2. ✅ 减少评估的 checkpoints 数量
3. ✅ 快速测试时限制样本数到 50-100

**未来优化（需改代码）**:
1. 🔄 实现批处理生成
2. 🔄 多GPU并行（同时评估多个checkpoints）
3. 🔄 提前停止生成（检测到EOS token）

---

💡 **建议**: 先运行快速测试（2个checkpoints × 50样本），验证速度后再决定是否运行完整评估。

