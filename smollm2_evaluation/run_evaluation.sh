#!/bin/bash
# SmolLM2 模型评估脚本
# 按照 LM Evaluation Harness 标准范式

set -e  # 遇到错误立即退出

echo "=================================================="
echo "SmolLM2-1.7B 模型评估"
echo "=================================================="
echo ""

# 配置变量
MODEL_NAME="HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints"

# lm-evaluation-harness 路径（自动检测或手动设置）
# 方法1: 自动检测（假设在父目录）
if [ -z "$LM_EVAL_DIR" ]; then
    LM_EVAL_DIR="$(dirname "$(pwd)")/lm-evaluation-harness-competition"
fi

# 方法2: 手动指定（如果自动检测失败，取消下面这行的注释）
# LM_EVAL_DIR="/home2/yth/lm-evaluation-harness-competition"

OUTPUT_BASE_DIR="$(pwd)/results"

# 要评估的 checkpoint 列表
CHECKPOINTS=(
    "step-125000"
    # "step-100000"  # 取消注释以评估更多 checkpoint
    # "step-150000"
)

# 要评估的任务列表
TASKS=(
    "hellaswag"
    "arc_easy"
    "arc_challenge"
    "mmlu"
    "piqa"
    "winogrande"
    "openbookqa"
    "boolq"
)

# 将任务列表转换为逗号分隔的字符串
TASKS_STR=$(IFS=,; echo "${TASKS[*]}")

# 其他配置
BATCH_SIZE="auto"
DTYPE="bfloat16"
NUM_FEWSHOT=0

# 激活虚拟环境
echo "▶ 激活虚拟环境..."
cd "$LM_EVAL_DIR"
source .venv/bin/activate

# 验证 lm_eval 可用
if ! command -v lm_eval &> /dev/null; then
    echo "❌ 错误: lm_eval 未安装"
    echo "请运行: pip install -e ."
    exit 1
fi

echo "✓ 环境就绪"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_BASE_DIR"

# 循环评估每个 checkpoint
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    echo "=================================================="
    echo "评估 checkpoint: $CHECKPOINT"
    echo "=================================================="
    
    # 设置输出目录
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${CHECKPOINT}"
    mkdir -p "$OUTPUT_DIR"
    
    # 构建命令
    CMD="lm_eval --model hf \
        --model_args pretrained=${MODEL_NAME},revision=${CHECKPOINT},dtype=${DTYPE} \
        --tasks ${TASKS_STR} \
        --batch_size ${BATCH_SIZE} \
        --num_fewshot ${NUM_FEWSHOT} \
        --output_path ${OUTPUT_DIR}/ \
        --log_samples"
    
    echo ""
    echo "📊 执行命令:"
    echo "$CMD"
    echo ""
    
    # 运行评估
    if eval "$CMD"; then
        echo "✓ Checkpoint ${CHECKPOINT} 评估完成"
        echo "  结果保存在: ${OUTPUT_DIR}/"
    else
        echo "✗ Checkpoint ${CHECKPOINT} 评估失败"
        exit 1
    fi
    
    echo ""
done

echo "=================================================="
echo "✅ 所有评估完成!"
echo "=================================================="
echo ""
echo "结果目录: $OUTPUT_BASE_DIR"
echo ""
echo "下一步:"
echo "1. 查看结果: ls -lh ${OUTPUT_BASE_DIR}/"
echo "2. 可视化结果: python plot_results.py"
echo "3. 创建提交: bash create_submission.sh"
echo ""

