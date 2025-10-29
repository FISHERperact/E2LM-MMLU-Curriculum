#!/bin/bash
# Dense-500M-Arch1 模型在填空题MMLU数据集上的评估脚本
# 评估所有 checkpoint 在自定义填空题数据集上的表现

set -e  # 遇到错误立即退出

echo "=================================================="
echo "Dense-500M-Arch1 模型填空题MMLU评估"
echo "模型: tiiuae/dense-500m-arch1"
echo "数据集: MMLU填空题 (8500 样本, 43 学科)"
echo "=================================================="
echo ""

# 配置变量
MODEL_NAME="tiiuae/dense-500m-arch1"
DATASET_PATH="/home2/yth/.cache/huggingface/datasets/mmlu_fill_blank_dataset_8500/test"
OUTPUT_BASE_DIR="$(pwd)/results_fillblank"

# lm-evaluation-harness 路径（用于激活虚拟环境）
if [ -z "$LM_EVAL_DIR" ]; then
    LM_EVAL_DIR="$(dirname "$(dirname "$(pwd)")")/lm-evaluation-harness-competition"
fi

# 所有要评估的 checkpoint
CHECKPOINTS=(
    "iter_0002000"
    "iter_0004000"
    "iter_0006000"
    "iter_0008000"
    "iter_0010000"
    "iter_0012000"
    "iter_0014000"
    "iter_0016000"
    "iter_0018000"
    "iter_0020000"
    "iter_0022000"
    "iter_0024000"
    "iter_0026000"
    "iter_0028000"
    "iter_0030000"
    "iter_0032000"
    "iter_0034000"
    "iter_0036000"
    "iter_0038000"
    "iter_0040000"
    "iter_0042000"
    "iter_0044000"
    "iter_0046000"
    "iter_0048000"
    "iter_0050000"
    "iter_0052000"
    "iter_0054000"
)

# 评估参数
DTYPE="bfloat16"  # 或 float16, float32
MAX_NEW_TOKENS=50
MAX_SAMPLES=""  # 留空表示评估所有样本，设置数字（如100）用于快速测试

# 使用 HuggingFace 镜像加速下载（可选）
export HF_ENDPOINT=https://hf-mirror.com

echo "配置信息:"
echo "  模型: ${MODEL_NAME}"
echo "  Checkpoints: ${#CHECKPOINTS[@]} 个"
echo "  数据集: ${DATASET_PATH}"
echo "  数据类型: ${DTYPE}"
echo "  输出目录: ${OUTPUT_BASE_DIR}"
if [ -n "$MAX_SAMPLES" ]; then
    echo "  最大样本数: ${MAX_SAMPLES} (测试模式)"
else
    echo "  最大样本数: 全部 (8500)"
fi
echo ""

# 检查数据集是否存在
if [ ! -d "${DATASET_PATH}" ]; then
    echo "❌ 错误: 找不到数据集目录"
    echo "   期望路径: ${DATASET_PATH}"
    exit 1
fi

echo "✅ 数据集检查通过"
echo ""

# 激活虚拟环境（如果需要transformers等库）
if [ -f "${LM_EVAL_DIR}/.venv/bin/activate" ]; then
    echo "激活虚拟环境..."
    source "${LM_EVAL_DIR}/.venv/bin/activate"
fi

# 创建输出目录
mkdir -p "${OUTPUT_BASE_DIR}"

# 记录开始时间
START_TIME=$(date +%s)
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 遍历每个 checkpoint
TOTAL_CHECKPOINTS=${#CHECKPOINTS[@]}
CURRENT=0

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    echo "=================================================="
    echo "正在评估 checkpoint: ${CHECKPOINT} (${CURRENT}/${TOTAL_CHECKPOINTS})"
    echo "=================================================="
    
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${CHECKPOINT}"
    mkdir -p "${OUTPUT_DIR}"
    
    # 记录 checkpoint 开始时间
    CHECKPOINT_START=$(date +%s)
    
    # 运行评估（直接运行，不使用eval以避免空格问题）
    echo ""
    echo "运行命令:"
    echo "python3 run_evaluation_fillblank.py \\"
    echo "    --model_name ${MODEL_NAME} \\"
    echo "    --subfolder ${CHECKPOINT} \\"
    echo "    --dataset_path ${DATASET_PATH} \\"
    echo "    --output_dir ${OUTPUT_BASE_DIR} \\"
    echo "    --dtype ${DTYPE} \\"
    echo "    --max_new_tokens ${MAX_NEW_TOKENS}"
    if [ -n "$MAX_SAMPLES" ]; then
        echo "    --max_samples ${MAX_SAMPLES}"
    fi
    echo ""
    
    # 构建参数数组
    ARGS=(
        "--model_name" "${MODEL_NAME}"
        "--subfolder" "${CHECKPOINT}"
        "--dataset_path" "${DATASET_PATH}"
        "--output_dir" "${OUTPUT_BASE_DIR}"
        "--dtype" "${DTYPE}"
        "--max_new_tokens" "${MAX_NEW_TOKENS}"
    )
    
    if [ -n "$MAX_SAMPLES" ]; then
        ARGS+=("--max_samples" "${MAX_SAMPLES}")
    fi
    
    if python3 run_evaluation_fillblank.py "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/evaluation.log"; then
        # 计算耗时
        CHECKPOINT_END=$(date +%s)
        CHECKPOINT_DURATION=$((CHECKPOINT_END - CHECKPOINT_START))
        
        echo ""
        echo "✅ ${CHECKPOINT} 评估完成"
        echo "   耗时: $((CHECKPOINT_DURATION / 60)) 分 $((CHECKPOINT_DURATION % 60)) 秒"
        echo ""
    else
        echo ""
        echo "❌ ${CHECKPOINT} 评估失败"
        echo "   查看日志: ${OUTPUT_DIR}/evaluation.log"
        echo ""
        
        # 询问是否继续
        read -p "是否继续评估下一个 checkpoint? (y/n): " continue_choice
        if [ "$continue_choice" != "y" ] && [ "$continue_choice" != "Y" ]; then
            echo "评估已中止"
            exit 1
        fi
    fi
    
    echo ""
done

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "=================================================="
echo "所有评估完成！"
echo "=================================================="
echo ""
echo "总结:"
echo "  模型: ${MODEL_NAME}"
echo "  Checkpoints: ${TOTAL_CHECKPOINTS} 个"
echo "  数据集: MMLU 填空题 (8500 样本)"
echo "  输出目录: ${OUTPUT_BASE_DIR}"
echo "  总耗时: $((TOTAL_DURATION / 3600)) 小时 $(((TOTAL_DURATION % 3600) / 60)) 分"
echo ""
echo "后续步骤:"
echo "  1. 查看结果: python3 view_results_fillblank.py"
echo "  2. 绘制图表: python3 plot_results_fillblank.py"
echo ""

