#!/bin/bash
# Resume Dense-500M Fill-in-the-blank Evaluation - Batch Version
# Continue from missing checkpoints

set -e

echo "=================================================================="
echo "Resuming Dense-500M Fill-in-the-blank Evaluation (Batch Version)"
echo "Completing missing checkpoints"
echo "=================================================================="
echo ""

MODEL_NAME="tiiuae/dense-500m-arch1"
DATASET_PATH="/home2/yth/.cache/huggingface/datasets/mmlu_fill_blank_dataset_8500/test"
OUTPUT_BASE_DIR="$(pwd)/results_fillblank"

LM_EVAL_DIR="/home2/yth/lm-evaluation-harness-competition"

# Missing checkpoints (iter_0034000 and iter_0038000 onwards)
CHECKPOINTS=(
    "iter_0034000"
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

DTYPE="bfloat16"
BATCH_SIZE=16
MAX_NEW_TOKENS=20

export HF_ENDPOINT=https://hf-mirror.com

echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  Remaining checkpoints: ${#CHECKPOINTS[@]}"
echo "  Dataset: ${DATASET_PATH}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Output dir: ${OUTPUT_BASE_DIR}"
echo ""

# Check dataset
if [ ! -d "${DATASET_PATH}" ]; then
    echo "❌ Error: Dataset not found"
    echo "   Expected path: ${DATASET_PATH}"
    exit 1
fi

# Activate virtual environment
if [ -f "${LM_EVAL_DIR}/.venv/bin/activate" ]; then
    source "${LM_EVAL_DIR}/.venv/bin/activate"
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
    
    # Check if already has results
    if [ -f "${OUTPUT_BASE_DIR}/${CHECKPOINT}/results_"*.json ]; then
        echo "⏭️  ${CHECKPOINT} already has results, skipping..."
        echo ""
        continue
    fi
    
    echo ""
    echo "Current GPU status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{printf "  GPU %s: %s\n  Memory: %s MB / %s MB\n  Utilization: %s%%\n", $1, $2, $3, $4, $5}'
    echo ""
    
    CHECKPOINT_START=$(date +%s)
    
    echo "Running command:"
    echo "python3 run_evaluation_fillblank_BATCH.py \\"
    echo "    --model_name ${MODEL_NAME} \\"
    echo "    --subfolder ${CHECKPOINT} \\"
    echo "    --dataset_path ${DATASET_PATH} \\"
    echo "    --output_dir ${OUTPUT_BASE_DIR} \\"
    echo "    --dtype ${DTYPE} \\"
    echo "    --batch_size ${BATCH_SIZE} \\"
    echo "    --max_new_tokens ${MAX_NEW_TOKENS}"
    echo ""
    
    # Add retry logic
    MAX_RETRIES=3
    RETRY=0
    SUCCESS=0
    
    while [ $RETRY -lt $MAX_RETRIES ] && [ $SUCCESS -eq 0 ]; do
        if [ $RETRY -gt 0 ]; then
            echo "⚠️  Retry attempt $RETRY/$MAX_RETRIES for ${CHECKPOINT}"
            sleep 10
        fi
        
        if python3 run_evaluation_fillblank_BATCH.py \
            --model_name "${MODEL_NAME}" \
            --subfolder "${CHECKPOINT}" \
            --dataset_path "${DATASET_PATH}" \
            --output_dir "${OUTPUT_BASE_DIR}" \
            --dtype "${DTYPE}" \
            --batch_size "${BATCH_SIZE}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            2>&1 | tee "${OUTPUT_BASE_DIR}/${CHECKPOINT}_evaluation.log"; then
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
        
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{printf "  GPU Memory: %s MB | Utilization: %s%%\n", $1, $2}'
        echo ""
    else
        echo ""
        echo "❌ ${CHECKPOINT} evaluation failed after $MAX_RETRIES attempts"
        echo "   Check log: ${OUTPUT_BASE_DIR}/${CHECKPOINT}_evaluation.log"
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
echo "  python3 view_results_fillblank.py"
echo "  python3 plot_results_fillblank.py"
echo ""

