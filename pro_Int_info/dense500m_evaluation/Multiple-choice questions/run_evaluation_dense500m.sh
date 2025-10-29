#!/bin/bash
# Dense-500M-Arch1 模型 MMLU 评估脚本
# 评估所有 checkpoint 在 MMLU 数据集上的表现

set -e  # 遇到错误立即退出

echo "=================================================="
echo "Dense-500M-Arch1 模型 MMLU 评估"
echo "模型: tiiuae/dense-500m-arch1"
echo "任务: MMLU (57 个学科)"
echo "=================================================="
echo ""

# 配置变量
MODEL_NAME="tiiuae/dense-500m-arch1"

# lm-evaluation-harness 路径（自动检测或手动设置）
if [ -z "$LM_EVAL_DIR" ]; then
    LM_EVAL_DIR="$(dirname "$(dirname "$(pwd)")")/lm-evaluation-harness-competition"
fi

# 如果自动检测失败，取消下面这行的注释并手动指定
# LM_EVAL_DIR="/home2/yth/lm-evaluation-harness-competition"

OUTPUT_BASE_DIR="$(pwd)/results_dense500m"

# 所有要评估的 checkpoint（从 iter_0002000 到 iter_0054000，步长 2000）
CHECKPOINTS=(
    #"iter_0002000"
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

# 只评估 MMLU 任务
TASKS="mmlu"

# 评估参数
BATCH_SIZE="auto"
NUM_FEWSHOT=0
DTYPE="bfloat16"  # 或 float16, float32

# 使用 HuggingFace 镜像加速下载（可选）
export HF_ENDPOINT=https://hf-mirror.com

echo "配置信息:"
echo "  模型: ${MODEL_NAME}"
echo "  Checkpoints: ${#CHECKPOINTS[@]} 个"
echo "  任务: ${TASKS}"
echo "  Few-shot: ${NUM_FEWSHOT}"
echo "  数据类型: ${DTYPE}"
echo "  输出目录: ${OUTPUT_BASE_DIR}"
echo ""

# 检查 lm-evaluation-harness 是否存在
if [ ! -d "${LM_EVAL_DIR}" ]; then
    echo "❌ 错误: 找不到 lm-evaluation-harness 目录"
    echo "   期望路径: ${LM_EVAL_DIR}"
    echo ""
    echo "请安装 lm-evaluation-harness:"
    echo "  git clone https://github.com/tiiuae/lm-evaluation-harness-competition ${LM_EVAL_DIR}"
    echo "  cd ${LM_EVAL_DIR}"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -e .[dev]"
    exit 1
fi

# 激活虚拟环境
if [ ! -f "${LM_EVAL_DIR}/.venv/bin/activate" ]; then
    echo "❌ 错误: 虚拟环境不存在"
    echo "   期望路径: ${LM_EVAL_DIR}/.venv"
    echo ""
    echo "请创建虚拟环境:"
    echo "  cd ${LM_EVAL_DIR}"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -e .[dev]"
    exit 1
fi

source "${LM_EVAL_DIR}/.venv/bin/activate"

# 验证 lm_eval 命令
if ! command -v lm_eval &> /dev/null; then
    echo "❌ 错误: lm_eval 命令不可用"
    echo "请确保已安装 lm-evaluation-harness"
    exit 1
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
    
    echo "=================================================="
    echo "正在评估 checkpoint: ${CHECKPOINT} (${CURRENT}/${TOTAL_CHECKPOINTS})"
    echo "=================================================="
    
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${CHECKPOINT}"
    mkdir -p "${OUTPUT_DIR}"
    
    # 显示当前GPU状态
    echo ""
    echo "当前GPU状态:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{printf "  GPU %s: %s\n  显存使用: %s MB / %s MB (%.1f%%)\n  GPU利用率: %s%%\n", $1, $2, $3, $4, ($3/$4)*100, $5}' || echo "  无法获取GPU信息"
    echo ""
    
    # 记录 checkpoint 开始时间
    CHECKPOINT_START=$(date +%s)
    
    # 运行评估
    echo ""
    echo "运行命令:"
    echo "lm_eval --model hf \\"
    echo "    --model_args pretrained=${MODEL_NAME},subfolder=${CHECKPOINT},dtype=${DTYPE} \\"
    echo "    --tasks ${TASKS} \\"
    echo "    --batch_size ${BATCH_SIZE} \\"
    echo "    --num_fewshot ${NUM_FEWSHOT} \\"
    echo "    --output_path ${OUTPUT_DIR}/ \\"
    echo "    --log_samples"
    echo ""
    
    # 提示用户可以监控GPU
    echo "💡 提示: 可在另一个终端运行 'watch -n 1 nvidia-smi' 实时监控GPU"
    echo ""
    
    if lm_eval --model hf \
        --model_args "pretrained=${MODEL_NAME},subfolder=${CHECKPOINT},dtype=${DTYPE}" \
        --tasks "${TASKS}" \
        --batch_size "${BATCH_SIZE}" \
        --num_fewshot "${NUM_FEWSHOT}" \
        --output_path "${OUTPUT_DIR}/" \
        --log_samples \
        2>&1 | tee "${OUTPUT_DIR}/evaluation.log"; then
        
        # 计算耗时
        CHECKPOINT_END=$(date +%s)
        CHECKPOINT_DURATION=$((CHECKPOINT_END - CHECKPOINT_START))
        
        # 计算速度（从日志中提取样本数）
        TOTAL_SAMPLES=$(grep -o "Total.*samples" "${OUTPUT_DIR}/evaluation.log" | grep -o "[0-9]*" | head -1 || echo "0")
        if [ "$TOTAL_SAMPLES" -gt 0 ]; then
            SAMPLES_PER_SEC=$(awk "BEGIN {printf \"%.2f\", $TOTAL_SAMPLES / $CHECKPOINT_DURATION}")
            echo ""
            echo "✅ ${CHECKPOINT} 评估完成"
            echo "   样本数: ${TOTAL_SAMPLES}"
            echo "   耗时: $((CHECKPOINT_DURATION / 60)) 分 $((CHECKPOINT_DURATION % 60)) 秒"
            echo "   速度: ${SAMPLES_PER_SEC} 样本/秒"
        else
            echo ""
            echo "✅ ${CHECKPOINT} 评估完成"
            echo "   耗时: $((CHECKPOINT_DURATION / 60)) 分 $((CHECKPOINT_DURATION % 60)) 秒"
        fi
        
        # 显示评估后的GPU状态
        echo ""
        echo "评估后GPU状态:"
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{printf "  显存使用: %s MB\n  GPU利用率: %s%%\n", $1, $2}' || echo "  无法获取GPU信息"
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
echo "  任务: ${TASKS}"
echo "  输出目录: ${OUTPUT_BASE_DIR}"
echo "  总耗时: $((TOTAL_DURATION / 3600)) 小时 $(((TOTAL_DURATION % 3600) / 60)) 分"
echo ""
echo "后续步骤:"
echo "  1. 查看结果: python3 view_results_dense500m.py"
echo "  2. 绘制图表: python3 plot_results_dense500m.py"
echo "  3. 查看具体题目: python3 view_questions.py mmlu_anatomy 5"
echo ""

