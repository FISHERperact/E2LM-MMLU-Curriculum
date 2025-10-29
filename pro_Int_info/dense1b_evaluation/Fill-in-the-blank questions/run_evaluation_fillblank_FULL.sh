#!/bin/bash
# Dense-1B Fill-in-the-blank MMLU Full Evaluation - Batch Version
# 20x faster with batch processing!

set -e

echo "=================================================================="
echo "Dense-1B Fill-in-the-blank MMLU Full Evaluation (Batch Optimized)"
echo "Model: tiiuae/dense-1b-arch1"
echo "Dataset: MMLU Fill-in-the-blank (8500 samples, 43 subjects)"
echo "Batch size: 16 | Speed: ~10 samples/sec"
echo "=================================================================="
echo ""

# Configuration
MODEL_NAME="tiiuae/dense-1b-arch1"
DATASET_PATH="/home2/yth/.cache/huggingface/datasets/mmlu_fill_blank_dataset_8500/test"
OUTPUT_BASE_DIR="$(pwd)/results_fillblank1b_batch"

# lm-evaluation-harness path (for virtual environment)
if [ -z "$LM_EVAL_DIR" ]; then
    LM_EVAL_DIR="$(dirname "$(dirname "$(pwd)")")/lm-evaluation-harness-competition"
fi

# All checkpoints to evaluate
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

# Evaluation parameters
DTYPE="bfloat16"
BATCH_SIZE=16
MAX_NEW_TOKENS=20
MAX_SAMPLES=""  # Empty=all 8500, or set number like 1000 for testing

export HF_ENDPOINT=https://hf-mirror.com

echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  Checkpoints: ${#CHECKPOINTS[@]}"
echo "  Dataset: ${DATASET_PATH}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max new tokens: ${MAX_NEW_TOKENS}"
if [ -n "$MAX_SAMPLES" ]; then
    echo "  Samples: ${MAX_SAMPLES} (test mode)"
    TOTAL_SAMPLES=$MAX_SAMPLES
else
    echo "  Samples: 8500 (full evaluation)"
    TOTAL_SAMPLES=8500
fi
echo "  Output dir: ${OUTPUT_BASE_DIR}"
echo ""

# Time estimate
SAMPLES_PER_SEC=10
TIME_PER_CHECKPOINT=$((TOTAL_SAMPLES / SAMPLES_PER_SEC))
TOTAL_TIME=$((TIME_PER_CHECKPOINT * ${#CHECKPOINTS[@]}))
echo "⏱️  Time estimate:"
echo "  Per checkpoint: ~$((TIME_PER_CHECKPOINT / 60)) minutes"
echo "  Total: ~$((TOTAL_TIME / 3600)) hours $((TOTAL_TIME % 3600 / 60)) minutes"
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

# Create output directory
mkdir -p "${OUTPUT_BASE_DIR}"

# Record start time
START_TIME=$(date +%s)
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Iterate through checkpoints
TOTAL_CHECKPOINTS=${#CHECKPOINTS[@]}
CURRENT=0

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    echo "=================================================================="
    echo "Evaluating checkpoint: ${CHECKPOINT} (${CURRENT}/${TOTAL_CHECKPOINTS})"
    echo "=================================================================="
    
    # Show GPU status
    echo ""
    echo "Current GPU status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{printf "  GPU %s: %s\n  Memory: %s MB / %s MB\n  Utilization: %s%%\n", $1, $2, $3, $4, $5}' || echo "  Unable to get GPU info"
    echo ""
    
    # Record checkpoint start time
    CHECKPOINT_START=$(date +%s)
    
    # Run evaluation
    echo "Running command:"
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
    
    # Run with proper argument handling
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
        # Calculate duration
        CHECKPOINT_END=$(date +%s)
        CHECKPOINT_DURATION=$((CHECKPOINT_END - CHECKPOINT_START))
        
        echo ""
        echo "✅ ${CHECKPOINT} evaluation completed"
        echo "   Duration: $((CHECKPOINT_DURATION / 60)) min $((CHECKPOINT_DURATION % 60)) sec"
        
        # Show post-evaluation GPU status
        echo ""
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{printf "  GPU Memory: %s MB | Utilization: %s%%\n", $1, $2}'
        echo ""
    else
        echo ""
        echo "❌ ${CHECKPOINT} evaluation failed"
        echo "   Check log: ${OUTPUT_BASE_DIR}/${CHECKPOINT}_evaluation.log"
        echo ""
        
        # Ask to continue
        read -p "Continue with next checkpoint? (y/n): " continue_choice
        if [ "$continue_choice" != "y" ] && [ "$continue_choice" != "Y" ]; then
            echo "Evaluation aborted"
            exit 1
        fi
    fi
    
    echo ""
done

# Calculate total duration
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "=================================================================="
echo "All evaluations completed!"
echo "=================================================================="
echo ""
echo "Summary:"
echo "  Model: ${MODEL_NAME}"
echo "  Checkpoints: ${TOTAL_CHECKPOINTS}"
echo "  Dataset: MMLU Fill-in-the-blank (${TOTAL_SAMPLES} samples)"
echo "  Output dir: ${OUTPUT_BASE_DIR}"
echo "  Total time: $((TOTAL_DURATION / 3600)) hours $(((TOTAL_DURATION % 3600) / 60)) min"
echo "  Average speed: $(awk "BEGIN {printf \"%.2f\", ${TOTAL_SAMPLES} * ${TOTAL_CHECKPOINTS} / ${TOTAL_DURATION}}") samples/sec"
echo ""
echo "Next steps:"
echo "  1. View results: python3 view_results_fillblank.py --results_dir ${OUTPUT_BASE_DIR}"
echo "  2. Plot charts: python3 plot_results_fillblank.py"
echo ""

