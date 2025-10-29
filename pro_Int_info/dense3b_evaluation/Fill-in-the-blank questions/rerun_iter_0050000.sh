#!/bin/bash
# Re-run Dense-3B Fill-in-the-blank evaluation for iter_0050000

set -e

echo "=================================================================="
echo "Re-running Dense-3B Fill-in-the-blank Evaluation"
echo "Checkpoint: iter_0050000"
echo "=================================================================="
echo ""

MODEL_NAME="tiiuae/dense-3b-arch1"
CHECKPOINT="iter_0050000"
DATASET_PATH="/home2/yth/.cache/huggingface/datasets/mmlu_fill_blank_dataset_8500/test"
OUTPUT_BASE_DIR="$(pwd)/results_fillblank3b_batch"

# lm-evaluation-harness path (for virtual environment)
LM_EVAL_DIR="/home2/yth/lm-evaluation-harness-competition"

DTYPE="bfloat16"
BATCH_SIZE=16
MAX_NEW_TOKENS=20

export HF_ENDPOINT=https://hf-mirror.com

echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  Checkpoint: ${CHECKPOINT}"
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
    echo "✅ Virtual environment activated"
else
    echo "⚠️  Warning: Virtual environment not found, using system Python"
fi

echo ""
echo "Starting evaluation..."
echo ""

START_TIME=$(date +%s)

# Show GPU status
echo "Current GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{printf "  GPU %s: %s\n  Memory: %s MB / %s MB\n  Utilization: %s%%\n", $1, $2, $3, $4, $5}'
echo ""

# Run evaluation with retry logic
MAX_RETRIES=3
RETRY=0
SUCCESS=0

while [ $RETRY -lt $MAX_RETRIES ] && [ $SUCCESS -eq 0 ]; do
    if [ $RETRY -gt 0 ]; then
        echo "⚠️  Retry attempt $RETRY/$MAX_RETRIES"
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

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
if [ $SUCCESS -eq 1 ]; then
    echo "=================================================================="
    echo "✅ Evaluation completed successfully!"
    echo "=================================================================="
    echo ""
    echo "Duration: $((DURATION / 60)) minutes $((DURATION % 60)) seconds"
    echo ""
    echo "Results saved to: ${OUTPUT_BASE_DIR}/${CHECKPOINT}/"
    echo ""
    echo "View results:"
    echo "  python3 view_results_fillblank.py"
    echo "  python3 plot_results_fillblank.py"
    echo ""
else
    echo "=================================================================="
    echo "❌ Evaluation failed after $MAX_RETRIES attempts"
    echo "=================================================================="
    echo ""
    echo "Check log: ${OUTPUT_BASE_DIR}/${CHECKPOINT}_evaluation.log"
    echo ""
    exit 1
fi

