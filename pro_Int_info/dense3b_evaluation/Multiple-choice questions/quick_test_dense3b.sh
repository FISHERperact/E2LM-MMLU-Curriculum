#!/bin/bash
# Dense-3B-Arch1 Quick Test Script
# Test 3 key checkpoints + limit 100 samples

set -e

echo "=================================================="
echo "Dense-3B-Arch1 Quick Test"
echo "Testing 3 checkpoints + 100 samples each"
echo "=================================================="
echo ""

# Configuration
MODEL_NAME="tiiuae/dense-3b-arch1"

if [ -z "$LM_EVAL_DIR" ]; then
    LM_EVAL_DIR="$(dirname "$(dirname "$(pwd)")")/lm-evaluation-harness-competition"
fi

OUTPUT_BASE_DIR="$(pwd)/results_dense3b_test"

# Test 3 key checkpoints
CHECKPOINTS=(
    "iter_0002000"   # Early
    "iter_0028000"   # Middle
    "iter_0054000"   # Late
)

TASKS="mmlu"
BATCH_SIZE="auto"
NUM_FEWSHOT=0
DTYPE="bfloat16"
LIMIT=100  # Only test 100 samples

# Use mirror for faster download
export HF_ENDPOINT=https://hf-mirror.com

echo "Test configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  Checkpoints: ${#CHECKPOINTS[@]} (early, middle, late)"
echo "  Tasks: ${TASKS}"
echo "  Sample limit: ${LIMIT} (quick test)"
echo "  Output dir: ${OUTPUT_BASE_DIR}"
echo ""

# Check environment
if [ ! -d "${LM_EVAL_DIR}" ]; then
    echo "❌ Error: lm-evaluation-harness not found"
    echo "   Expected path: ${LM_EVAL_DIR}"
    exit 1
fi

source "${LM_EVAL_DIR}/.venv/bin/activate"

if ! command -v lm_eval &> /dev/null; then
    echo "❌ Error: lm_eval command not available"
    exit 1
fi

echo "✅ Environment check passed"
echo ""

# Show GPU info
echo "GPU Info:"
python3 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('  No GPU')" 2>/dev/null || echo "  Unable to detect GPU"
echo ""

mkdir -p "${OUTPUT_BASE_DIR}"

START_TIME=$(date +%s)
TOTAL=${#CHECKPOINTS[@]}
CURRENT=0

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    echo "=================================================="
    echo "Testing checkpoint: ${CHECKPOINT} (${CURRENT}/${TOTAL})"
    echo "=================================================="
    
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${CHECKPOINT}"
    mkdir -p "${OUTPUT_DIR}"
    
    # Show GPU status
    echo ""
    echo "Current GPU status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{printf "  GPU %s: %s\n  Memory: %s MB / %s MB\n  Utilization: %s%%\n", $1, $2, $3, $4, $5}' || echo "  Unable to get GPU info"
    
    echo ""
    echo "Running quick test (limit ${LIMIT} samples)..."
    echo ""
    
    CHECKPOINT_START=$(date +%s)
    
    if lm_eval --model hf \
        --model_args "pretrained=${MODEL_NAME},subfolder=${CHECKPOINT},dtype=${DTYPE}" \
        --tasks "${TASKS}" \
        --batch_size "${BATCH_SIZE}" \
        --num_fewshot "${NUM_FEWSHOT}" \
        --limit "${LIMIT}" \
        --output_path "${OUTPUT_DIR}/" \
        2>&1 | tee "${OUTPUT_DIR}/test.log"; then
        
        CHECKPOINT_END=$(date +%s)
        CHECKPOINT_DURATION=$((CHECKPOINT_END - CHECKPOINT_START))
        
        echo ""
        echo "✅ ${CHECKPOINT} test completed"
        echo "   Duration: $((CHECKPOINT_DURATION / 60)) min $((CHECKPOINT_DURATION % 60)) sec"
        echo ""
        
        # Show GPU status
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{printf "  GPU Memory: %s MB | Utilization: %s%%\n", $1, $2}'
    else
        echo ""
        echo "❌ ${CHECKPOINT} test failed"
        echo "   Check log: ${OUTPUT_DIR}/test.log"
    fi
    
    echo ""
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "=================================================="
echo "Quick test completed!"
echo "=================================================="
echo ""
echo "Total time: $((DURATION / 60)) min $((DURATION % 60)) sec"
echo "Output dir: ${OUTPUT_BASE_DIR}"
echo ""
echo "💡 Tip:"
echo "  This is just a quick test (limited to ${LIMIT} samples)"
echo "  For full evaluation: bash run_evaluation_dense3b.sh"
echo ""
echo "View test results:"
echo "  python3 view_results_dense3b.py"
echo ""

