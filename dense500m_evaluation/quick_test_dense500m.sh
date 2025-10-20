#!/bin/bash
# Dense-500M-Arch1 快速测试脚本
# 只评估 3 个关键 checkpoint + 限制 100 个样本

set -e

echo "=================================================="
echo "Dense-500M-Arch1 快速测试"
echo "只评估 3 个 checkpoints + 每个限制 100 个样本"
echo "=================================================="
echo ""

# 配置变量
MODEL_NAME="tiiuae/dense-500m-arch1"

if [ -z "$LM_EVAL_DIR" ]; then
    LM_EVAL_DIR="$(dirname "$(pwd)")/lm-evaluation-harness-competition"
fi

OUTPUT_BASE_DIR="$(pwd)/results_dense500m_test"

# 只测试 3 个关键 checkpoints
CHECKPOINTS=(
    "iter_0002000"   # 早期
    "iter_0028000"   # 中期
    "iter_0054000"   # 后期
)

TASKS="mmlu"
BATCH_SIZE="auto"
NUM_FEWSHOT=0
DTYPE="bfloat16"
LIMIT=100  # 只测试 100 个样本

# 使用镜像加速
export HF_ENDPOINT=https://hf-mirror.com

echo "测试配置:"
echo "  模型: ${MODEL_NAME}"
echo "  Checkpoints: ${#CHECKPOINTS[@]} 个 (早期、中期、后期)"
echo "  任务: ${TASKS}"
echo "  样本限制: ${LIMIT} 个 (快速测试)"
echo "  输出目录: ${OUTPUT_BASE_DIR}"
echo ""

# 检查环境
if [ ! -d "${LM_EVAL_DIR}" ]; then
    echo "❌ 错误: 找不到 lm-evaluation-harness"
    echo "   期望路径: ${LM_EVAL_DIR}"
    exit 1
fi

source "${LM_EVAL_DIR}/.venv/bin/activate"

if ! command -v lm_eval &> /dev/null; then
    echo "❌ 错误: lm_eval 命令不可用"
    exit 1
fi

echo "✅ 环境检查通过"
echo ""

mkdir -p "${OUTPUT_BASE_DIR}"

START_TIME=$(date +%s)
TOTAL=${#CHECKPOINTS[@]}
CURRENT=0

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    echo "=================================================="
    echo "测试 checkpoint: ${CHECKPOINT} (${CURRENT}/${TOTAL})"
    echo "=================================================="
    
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${CHECKPOINT}"
    mkdir -p "${OUTPUT_DIR}"
    
    echo ""
    echo "运行快速测试（限制 ${LIMIT} 个样本）..."
    echo ""
    
    if lm_eval --model hf \
        --model_args pretrained=${MODEL_NAME},subfolder=${CHECKPOINT},dtype=${DTYPE} \
        --tasks ${TASKS} \
        --batch_size ${BATCH_SIZE} \
        --num_fewshot ${NUM_FEWSHOT} \
        --limit ${LIMIT} \
        --output_path ${OUTPUT_DIR}/ \
        2>&1 | tee "${OUTPUT_DIR}/test.log"; then
        
        echo ""
        echo "✅ ${CHECKPOINT} 测试完成"
    else
        echo ""
        echo "❌ ${CHECKPOINT} 测试失败"
        echo "   查看日志: ${OUTPUT_DIR}/test.log"
    fi
    
    echo ""
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "=================================================="
echo "快速测试完成！"
echo "=================================================="
echo ""
echo "总耗时: $((DURATION / 60)) 分 $((DURATION % 60)) 秒"
echo "输出目录: ${OUTPUT_BASE_DIR}"
echo ""
echo "💡 提示:"
echo "  这只是快速测试（限制 ${LIMIT} 个样本）"
echo "  完整评估请运行: bash run_evaluation_dense500m.sh"
echo ""
echo "查看测试结果:"
echo "  python3 view_results_dense500m.py"
echo ""

