#!/bin/bash
# Dense-500M 填空题完整评估 - 批处理版本
# 使用批处理优化，速度提升20倍！

set -e

echo "=================================================================="
echo "Dense-500M 填空题完整评估 (批处理优化版)"
echo "模型: tiiuae/dense-500m-arch1"
echo "数据集: MMLU填空题 (8500 样本, 43 学科)"
echo "批处理大小: 16 | 速度: ~10 样本/秒"
echo "=================================================================="
echo ""

# 配置变量
MODEL_NAME="tiiuae/dense-500m-arch1"
DATASET_PATH="/home2/yth/.cache/huggingface/datasets/mmlu_fill_blank_dataset_8500/test"
OUTPUT_BASE_DIR="$(pwd)/results_fillblank_batch"

# lm-evaluation-harness 路径（用于虚拟环境）
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
DTYPE="bfloat16"
BATCH_SIZE=16          # 批处理大小
MAX_NEW_TOKENS=20      # 生成长度
MAX_SAMPLES=""         # 留空=全部8500，或设置数字如1000用于测试

export HF_ENDPOINT=https://hf-mirror.com

echo "配置信息:"
echo "  模型: ${MODEL_NAME}"
echo "  Checkpoints: ${#CHECKPOINTS[@]} 个"
echo "  数据集: ${DATASET_PATH}"
echo "  批处理大小: ${BATCH_SIZE}"
echo "  最大生成长度: ${MAX_NEW_TOKENS} tokens"
if [ -n "$MAX_SAMPLES" ]; then
    echo "  样本数: ${MAX_SAMPLES} (测试模式)"
    TOTAL_SAMPLES=$MAX_SAMPLES
else
    echo "  样本数: 8500 (完整评估)"
    TOTAL_SAMPLES=8500
fi
echo "  输出目录: ${OUTPUT_BASE_DIR}"
echo ""

# 预估时间
SAMPLES_PER_SEC=10
TIME_PER_CHECKPOINT=$((TOTAL_SAMPLES / SAMPLES_PER_SEC))
TOTAL_TIME=$((TIME_PER_CHECKPOINT * ${#CHECKPOINTS[@]}))
echo "⏱️  预估时间:"
echo "  每个checkpoint: ~$((TIME_PER_CHECKPOINT / 60)) 分钟"
echo "  总计: ~$((TOTAL_TIME / 3600)) 小时 $((TOTAL_TIME % 3600 / 60)) 分钟"
echo ""

# 检查数据集
if [ ! -d "${DATASET_PATH}" ]; then
    echo "❌ 错误: 找不到数据集目录"
    echo "   期望路径: ${DATASET_PATH}"
    exit 1
fi

# 激活虚拟环境
if [ -f "${LM_EVAL_DIR}/.venv/bin/activate" ]; then
    source "${LM_EVAL_DIR}/.venv/bin/activate"
fi

echo "✅ 环境检查通过"
echo ""

# 显示GPU信息
echo "GPU 信息:"
python3 check_gpu.py 2>/dev/null || echo "  无法检测GPU信息"
echo ""

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
    
    echo "=================================================================="
    echo "正在评估 checkpoint: ${CHECKPOINT} (${CURRENT}/${TOTAL_CHECKPOINTS})"
    echo "=================================================================="
    
    # 显示当前GPU状态
    echo ""
    echo "当前GPU状态:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{printf "  GPU %s: %s\n  显存使用: %s MB / %s MB\n  GPU利用率: %s%%\n", $1, $2, $3, $4, $5}' || echo "  无法获取GPU信息"
    echo ""
    
    # 记录 checkpoint 开始时间
    CHECKPOINT_START=$(date +%s)
    
    # 运行评估
    echo "运行命令:"
    echo "python3 run_evaluation_fillblank_BATCH.py \\"
    echo "    --model_name ${MODEL_NAME} \\"
    echo "    --subfolder ${CHECKPOINT} \\"
    echo "    --dataset_path ${DATASET_PATH} \\"
    echo "    --output_dir ${OUTPUT_BASE_DIR} \\"
    echo "    --dtype ${DTYPE} \\"
    echo "    --batch_size ${BATCH_SIZE} \\"
    echo "    --max_new_tokens ${MAX_NEW_TOKENS}"
    if [ -n "$MAX_SAMPLES" ]; then
        echo "    --max_samples ${MAX_SAMPLES}"
    fi
    echo ""
    
    # 构建参数（避免空格路径问题）
    if [ -n "$MAX_SAMPLES" ]; then
        if python3 run_evaluation_fillblank_BATCH.py \
            --model_name "${MODEL_NAME}" \
            --subfolder "${CHECKPOINT}" \
            --dataset_path "${DATASET_PATH}" \
            --output_dir "${OUTPUT_BASE_DIR}" \
            --dtype "${DTYPE}" \
            --batch_size "${BATCH_SIZE}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --max_samples "${MAX_SAMPLES}" \
            2>&1 | tee "${OUTPUT_BASE_DIR}/${CHECKPOINT}_evaluation.log"; then
            true
        else
            false
        fi
    else
        if python3 run_evaluation_fillblank_BATCH.py \
            --model_name "${MODEL_NAME}" \
            --subfolder "${CHECKPOINT}" \
            --dataset_path "${DATASET_PATH}" \
            --output_dir "${OUTPUT_BASE_DIR}" \
            --dtype "${DTYPE}" \
            --batch_size "${BATCH_SIZE}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            2>&1 | tee "${OUTPUT_BASE_DIR}/${CHECKPOINT}_evaluation.log"; then
            true
        else
            false
        fi
    fi
    
    if [ $? -eq 0 ]; then
        # 计算耗时
        CHECKPOINT_END=$(date +%s)
        CHECKPOINT_DURATION=$((CHECKPOINT_END - CHECKPOINT_START))
        
        echo ""
        echo "✅ ${CHECKPOINT} 评估完成"
        echo "   耗时: $((CHECKPOINT_DURATION / 60)) 分 $((CHECKPOINT_DURATION % 60)) 秒"
        
        # 显示评估后的GPU状态
        echo ""
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{printf "  GPU显存: %s MB | GPU利用率: %s%%\n", $1, $2}'
        echo ""
    else
        echo ""
        echo "❌ ${CHECKPOINT} 评估失败"
        echo "   查看日志: ${OUTPUT_BASE_DIR}/${CHECKPOINT}_evaluation.log"
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

echo "=================================================================="
echo "所有评估完成！"
echo "=================================================================="
echo ""
echo "总结:"
echo "  模型: ${MODEL_NAME}"
echo "  Checkpoints: ${TOTAL_CHECKPOINTS} 个"
echo "  数据集: MMLU 填空题 (${TOTAL_SAMPLES} 样本)"
echo "  输出目录: ${OUTPUT_BASE_DIR}"
echo "  总耗时: $((TOTAL_DURATION / 3600)) 小时 $(((TOTAL_DURATION % 3600) / 60)) 分"
echo "  平均速度: $(awk "BEGIN {printf \"%.2f\", ${TOTAL_SAMPLES} * ${TOTAL_CHECKPOINTS} / ${TOTAL_DURATION}}") 样本/秒"
echo ""
echo "后续步骤:"
echo "  1. 查看结果: python3 view_results_fillblank.py --results_dir ${OUTPUT_BASE_DIR}"
echo ""

