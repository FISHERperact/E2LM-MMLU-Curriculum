#!/bin/bash
# 快速测试填空题评估脚本
# 只评估少量 checkpoints 和样本，用于验证环境和配置

set -e

echo "=================================================="
echo "填空题MMLU快速测试"
echo "测试 3 个 checkpoints，每个 50 样本"
echo "预计耗时: ~15-30 分钟"
echo "=================================================="
echo ""

# 配置变量
MODEL_NAME="tiiuae/dense-500m-arch1"
DATASET_PATH="/home2/yth/.cache/huggingface/datasets/mmlu_fill_blank_dataset_8500/test"
OUTPUT_BASE_DIR="$(pwd)/results_fillblank_test"

# lm-evaluation-harness 路径
if [ -z "$LM_EVAL_DIR" ]; then
    LM_EVAL_DIR="$(dirname "$(dirname "$(pwd)")")/lm-evaluation-harness-competition"
fi

# 测试用的 checkpoints（早期、中期、晚期各一个）
CHECKPOINTS=(
    "iter_0002000"  # 早期
    "iter_0028000"  # 中期
    "iter_0054000"  # 晚期
)

# 评估参数
DTYPE="bfloat16"
MAX_NEW_TOKENS=20  # 减少生成长度以提速（大多数答案很短）
MAX_SAMPLES=50     # 只测试50个样本

export HF_ENDPOINT=https://hf-mirror.com

echo "配置信息:"
echo "  模型: ${MODEL_NAME}"
echo "  测试 Checkpoints: ${#CHECKPOINTS[@]} 个"
echo "  每个样本数: ${MAX_SAMPLES}"
echo "  数据集: ${DATASET_PATH}"
echo "  输出目录: ${OUTPUT_BASE_DIR}"
echo ""

# 检查数据集
if [ ! -d "${DATASET_PATH}" ]; then
    echo "❌ 错误: 找不到数据集目录"
    echo "   期望路径: ${DATASET_PATH}"
    exit 1
fi

echo "✅ 数据集检查通过"
echo ""

# 激活虚拟环境
if [ -f "${LM_EVAL_DIR}/.venv/bin/activate" ]; then
    source "${LM_EVAL_DIR}/.venv/bin/activate"
fi

# 创建输出目录
mkdir -p "${OUTPUT_BASE_DIR}"

# 记录开始时间
START_TIME=$(date +%s)
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 遍历每个 checkpoint
CURRENT=0
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    echo "=================================================="
    echo "测试 checkpoint: ${CHECKPOINT} (${CURRENT}/${#CHECKPOINTS[@]})"
    echo "=================================================="
    
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${CHECKPOINT}"
    mkdir -p "${OUTPUT_DIR}"
    
    CHECKPOINT_START=$(date +%s)
    
    # 运行评估
    echo ""
    echo "运行命令:"
    echo "python3 run_evaluation_fillblank.py \\"
    echo "    --model_name ${MODEL_NAME} \\"
    echo "    --subfolder ${CHECKPOINT} \\"
    echo "    --dataset_path ${DATASET_PATH} \\"
    echo "    --output_dir ${OUTPUT_BASE_DIR} \\"
    echo "    --dtype ${DTYPE} \\"
    echo "    --max_new_tokens ${MAX_NEW_TOKENS} \\"
    echo "    --max_samples ${MAX_SAMPLES}"
    echo ""
    
    if python3 run_evaluation_fillblank.py \
        --model_name "${MODEL_NAME}" \
        --subfolder "${CHECKPOINT}" \
        --dataset_path "${DATASET_PATH}" \
        --output_dir "${OUTPUT_BASE_DIR}" \
        --dtype "${DTYPE}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --max_samples "${MAX_SAMPLES}" \
        2>&1 | tee "${OUTPUT_DIR}/evaluation.log"; then
        CHECKPOINT_END=$(date +%s)
        CHECKPOINT_DURATION=$((CHECKPOINT_END - CHECKPOINT_START))
        
        echo ""
        echo "✅ ${CHECKPOINT} 测试完成"
        echo "   耗时: $((CHECKPOINT_DURATION / 60)) 分 $((CHECKPOINT_DURATION % 60)) 秒"
        echo ""
    else
        echo ""
        echo "❌ ${CHECKPOINT} 测试失败"
        exit 1
    fi
done

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "=================================================="
echo "快速测试完成！"
echo "=================================================="
echo ""
echo "总耗时: $((TOTAL_DURATION / 60)) 分 $((TOTAL_DURATION % 60)) 秒"
echo "结果目录: ${OUTPUT_BASE_DIR}"
echo ""
echo "如果测试正常，可以运行完整评估:"
echo "  bash run_evaluation_fillblank.sh"
echo ""

