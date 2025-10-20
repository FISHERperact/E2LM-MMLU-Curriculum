#!/bin/bash
# 创建提交文件脚本
# 按照竞赛要求生成 submission.zip

set -e

echo "=================================================="
echo "创建提交文件"
echo "=================================================="
echo ""

# 配置
LM_EVAL_DIR="/home2/yth/lm-evaluation-harness-competition"
SUBMISSION_DIR="$(pwd)/submission"
RESULTS_DIR="$(pwd)/results"

# 创建提交目录
mkdir -p "$SUBMISSION_DIR"

# 步骤 1: 生成 evaluation.patch
echo "步骤 1/3: 生成 evaluation.patch"
cd "$LM_EVAL_DIR"

# 检查是否有修改
if git diff --quiet && git diff --cached --quiet; then
    echo "  ℹ️  没有代码修改，创建空 patch 文件"
    echo "# No changes - using standard lm-evaluation-harness" > "${SUBMISSION_DIR}/evaluation.patch"
else
    echo "  ✓ 检测到代码修改，生成 patch"
    # 生成包含所有修改的 patch
    git diff > "${SUBMISSION_DIR}/evaluation.patch"
    git diff --cached >> "${SUBMISSION_DIR}/evaluation.patch"
fi

echo "  ✓ evaluation.patch 已生成"
echo ""

# 步骤 2: 创建 metadata.yaml
echo "步骤 2/3: 创建 metadata.yaml"

cat > "${SUBMISSION_DIR}/metadata.yaml" << 'EOF'
# 提交元数据
# 根据竞赛要求填写

# 基本信息
submission:
  name: "SmolLM2-1.7B Evaluation"
  author: "yth"
  email: "your.email@example.com"
  date: "2025-10-19"

# 模型信息
model:
  name: "SmolLM2-1.7B"
  checkpoint: "HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints"
  revision: "step-125000"

# 评估任务
evaluation:
  tasks:
    - hellaswag
    - arc_easy
    - arc_challenge
    - mmlu
    - piqa
    - winogrande
    - openbookqa
    - boolq
  
  # 主要指标
  primary_metric: "acc_norm"
  
  # 评估设置
  settings:
    num_fewshot: 0
    batch_size: "auto"
    dtype: "bfloat16"

# HuggingFace Token（如果数据集是私有的）
hf_token: ""

# 备注
notes: |
  Standard evaluation of SmolLM2-1.7B on multiple benchmarks.
  Using zero-shot setting with normalized accuracy as primary metric.
EOF

echo "  ✓ metadata.yaml 已创建"
echo ""

# 步骤 3: 创建 submission.zip
echo "步骤 3/3: 打包 submission.zip"

cd "$SUBMISSION_DIR"
zip -r ../submission.zip ./* > /dev/null 2>&1

if [ -f "../submission.zip" ]; then
    echo "  ✓ submission.zip 已创建"
else
    echo "  ✗ submission.zip 创建失败"
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ 提交文件创建完成!"
echo "=================================================="
echo ""
echo "📦 提交文件位置: $(pwd)/../submission.zip"
echo ""
echo "📄 包含文件:"
cd ..
unzip -l submission.zip
echo ""
echo "🔍 验证步骤:"
echo "1. 检查 patch: cat ${SUBMISSION_DIR}/evaluation.patch"
echo "2. 检查 metadata: cat ${SUBMISSION_DIR}/metadata.yaml"
echo "3. 验证 zip: unzip -t submission.zip"
echo ""
echo "📝 提交前请确认:"
echo "  [ ] metadata.yaml 中的信息正确"
echo "  [ ] evaluation.patch 包含所有必要的修改"
echo "  [ ] 评估结果符合预期"
echo ""

