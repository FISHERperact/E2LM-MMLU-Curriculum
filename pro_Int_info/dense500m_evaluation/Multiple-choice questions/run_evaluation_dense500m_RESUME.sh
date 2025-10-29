#!/bin/bash
# Resume Dense-500M Multiple Choice Evaluation from iter_0026000

set -e

echo "=================================================================="
echo "Resuming Dense-500M Multiple Choice Evaluation"
echo "Starting from: iter_0026000"
echo "=================================================================="
echo ""

MODEL_NAME="tiiuae/dense-500m-arch1"

if [ -z "$LM_EVAL_DIR" ]; then
    LM_EVAL_DIR="/home2/yth/lm-evaluation-harness-competition"
fi

OUTPUT_BASE_DIR="$(pwd)/results_dense500m"

# Only missing checkpoints (iter_0026000 onwards)
CHECKPOINTS=(
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

TASKS="mmlu"
BATCH_SIZE="auto"
NUM_FEWSHOT=0
DTYPE="bfloat16"

export HF_ENDPOINT=https://hf-mirror.com

echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  Remaining checkpoints: ${#CHECKPOINTS[@]}"
echo "  Tasks: ${TASKS}"
echo "  Output dir: ${OUTPUT_BASE_DIR}"
echo ""

# Check environment
if [ ! -d "${LM_EVAL_DIR}" ]; then
    echo "❌ Error: lm-evaluation-harness not found"
    exit 1
fi

source "${LM_EVAL_DIR}/.venv/bin/activate"

if ! command -v lm_eval &> /dev/null; then
    echo "❌ Error: lm_eval command not available"
    exit 1
fi

echo "✅ Environment check passed"
echo ""

mkdir -p "${OUTPUT_BASE_DIR}"

START_TIME=$(date +%s)
TOTAL=${#CHECKPOINTS[@]}
CURRENT=0

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    echo "=================================================================="
    echo "Evaluating checkpoint: ${CHECKPOINT} (${CURRENT}/${TOTAL})"
    echo "=================================================================="
    
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${CHECKPOINT}"
    mkdir -p "${OUTPUT_DIR}"
    
    # Check if already has results
    if [ -f "${OUTPUT_DIR}"/*/results_*.json ]; then
        echo "⏭️  ${CHECKPOINT} already has results, skipping..."
        echo ""
        continue
    fi
    
    echo ""
    echo "Current GPU status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{printf "  GPU %s: %s\n  Memory: %s MB / %s MB\n  Utilization: %s%%\n", $1, $2, $3, $4, $5}'
    echo ""
    
    CHECKPOINT_START=$(date +%s)
    
    echo "Running evaluation..."
    echo ""
    
    # Add retry logic for network errors
    MAX_RETRIES=3
    RETRY=0
    SUCCESS=0
    
    while [ $RETRY -lt $MAX_RETRIES ] && [ $SUCCESS -eq 0 ]; do
        if [ $RETRY -gt 0 ]; then
            echo "⚠️  Retry attempt $RETRY/$MAX_RETRIES for ${CHECKPOINT}"
            sleep 10
        fi
        
        if lm_eval --model hf \
            --model_args "pretrained=${MODEL_NAME},subfolder=${CHECKPOINT},dtype=${DTYPE}" \
            --tasks "${TASKS}" \
            --batch_size "${BATCH_SIZE}" \
            --num_fewshot "${NUM_FEWSHOT}" \
            --output_path "${OUTPUT_DIR}/" \
            --log_samples \
            2>&1 | tee "${OUTPUT_DIR}/evaluation.log"; then
            SUCCESS=1
        else
            RETRY=$((RETRY + 1))
            if [ $RETRY -lt $MAX_RETRIES ]; then
                echo "❌ Failed, will retry..."
            fi
        fi
    done
    
    if [ $SUCCESS -eq 1 ]; then
        CHECKPOINT_END=$(date +%s)
        CHECKPOINT_DURATION=$((CHECKPOINT_END - CHECKPOINT_START))
        
        echo ""
        echo "✅ ${CHECKPOINT} evaluation completed"
        echo "   Duration: $((CHECKPOINT_DURATION / 60)) min $((CHECKPOINT_DURATION % 60)) sec"
        echo ""
    else
        echo ""
        echo "❌ ${CHECKPOINT} evaluation failed after $MAX_RETRIES attempts"
        echo "   Check log: ${OUTPUT_DIR}/evaluation.log"
        echo ""
        
        read -p "Continue with next checkpoint? (y/n): " continue_choice
        if [ "$continue_choice" != "y" ] && [ "$continue_choice" != "Y" ]; then
            echo "Evaluation aborted"
            exit 1
        fi
    fi
    
    echo ""
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "=================================================================="
echo "Resume evaluation completed!"
echo "=================================================================="
echo ""
echo "Total time: $((TOTAL_DURATION / 3600)) hours $(((TOTAL_DURATION % 3600) / 60)) min"
echo ""
echo "View results:"
echo "  python3 view_results_dense500m.py"
echo "  python3 plot_results_dense500m.py"
echo ""

