#!/bin/bash
# Dense-3B-Arch1 Model MMLU Evaluation Script
# Evaluate all checkpoints on MMLU dataset

set -e  # Exit on error

echo "=================================================="
echo "Dense-3B-Arch1 Model MMLU Evaluation"
echo "Model: tiiuae/dense-3b-arch1"
echo "Tasks: MMLU (57 subjects)"
echo "=================================================="
echo ""

# Configuration
MODEL_NAME="tiiuae/dense-3b-arch1"

# lm-evaluation-harness path
if [ -z "$LM_EVAL_DIR" ]; then
    LM_EVAL_DIR="$(dirname "$(dirname "$(pwd)")")/lm-evaluation-harness-competition"
fi

# Manual override if needed
# LM_EVAL_DIR="/home2/yth/lm-evaluation-harness-competition"

OUTPUT_BASE_DIR="$(pwd)/results_dense3b"

# All checkpoints to evaluate (iter_0002000 to iter_0054000, step 2000)
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

# Evaluation tasks
TASKS="mmlu"

# Evaluation parameters
BATCH_SIZE="auto"
NUM_FEWSHOT=0
DTYPE="bfloat16"  # or float16, float32

# Use HuggingFace mirror for faster download
export HF_ENDPOINT=https://hf-mirror.com

echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  Checkpoints: ${#CHECKPOINTS[@]} total"
echo "  Tasks: ${TASKS}"
echo "  Few-shot: ${NUM_FEWSHOT}"
echo "  Data type: ${DTYPE}"
echo "  Output dir: ${OUTPUT_BASE_DIR}"
echo ""

# Check lm-evaluation-harness
if [ ! -d "${LM_EVAL_DIR}" ]; then
    echo "❌ Error: lm-evaluation-harness not found"
    echo "   Expected path: ${LM_EVAL_DIR}"
    echo ""
    echo "Please install lm-evaluation-harness:"
    echo "  git clone https://github.com/tiiuae/lm-evaluation-harness-competition ${LM_EVAL_DIR}"
    echo "  cd ${LM_EVAL_DIR}"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -e .[dev]"
    exit 1
fi

# Activate virtual environment
if [ ! -f "${LM_EVAL_DIR}/.venv/bin/activate" ]; then
    echo "❌ Error: Virtual environment not found"
    echo "   Expected path: ${LM_EVAL_DIR}/.venv"
    exit 1
fi

source "${LM_EVAL_DIR}/.venv/bin/activate"

# Verify lm_eval command
if ! command -v lm_eval &> /dev/null; then
    echo "❌ Error: lm_eval command not available"
    exit 1
fi

echo "✅ Environment check passed"
echo ""

# Show GPU info
echo "GPU Info:"
python3 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('  No GPU detected')" 2>/dev/null || echo "  Unable to detect GPU"
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
    
    echo "=================================================="
    echo "Evaluating checkpoint: ${CHECKPOINT} (${CURRENT}/${TOTAL_CHECKPOINTS})"
    echo "=================================================="
    
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${CHECKPOINT}"
    mkdir -p "${OUTPUT_DIR}"
    
    # Show current GPU status
    echo ""
    echo "Current GPU status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{printf "  GPU %s: %s\n  Memory: %s MB / %s MB (%.1f%%)\n  Utilization: %s%%\n", $1, $2, $3, $4, ($3/$4)*100, $5}' || echo "  Unable to get GPU info"
    echo ""
    
    # Record checkpoint start time
    CHECKPOINT_START=$(date +%s)
    
    # Run evaluation
    echo ""
    echo "Running command:"
    echo "lm_eval --model hf \\"
    echo "    --model_args pretrained=${MODEL_NAME},subfolder=${CHECKPOINT},dtype=${DTYPE} \\"
    echo "    --tasks ${TASKS} \\"
    echo "    --batch_size ${BATCH_SIZE} \\"
    echo "    --num_fewshot ${NUM_FEWSHOT} \\"
    echo "    --output_path ${OUTPUT_DIR}/ \\"
    echo "    --log_samples"
    echo ""
    
    # Tip for GPU monitoring
    echo "💡 Tip: Run 'watch -n 1 nvidia-smi' in another terminal to monitor GPU"
    echo ""
    
    if lm_eval --model hf \
        --model_args "pretrained=${MODEL_NAME},subfolder=${CHECKPOINT},dtype=${DTYPE}" \
        --tasks "${TASKS}" \
        --batch_size "${BATCH_SIZE}" \
        --num_fewshot "${NUM_FEWSHOT}" \
        --output_path "${OUTPUT_DIR}/" \
        --log_samples \
        2>&1 | tee "${OUTPUT_DIR}/evaluation.log"; then
        
        # Calculate duration
        CHECKPOINT_END=$(date +%s)
        CHECKPOINT_DURATION=$((CHECKPOINT_END - CHECKPOINT_START))
        
        # Extract sample count from log
        TOTAL_SAMPLES=$(grep -o "Total.*samples" "${OUTPUT_DIR}/evaluation.log" | grep -o "[0-9]*" | head -1 || echo "0")
        if [ "$TOTAL_SAMPLES" -gt 0 ]; then
            SAMPLES_PER_SEC=$(awk "BEGIN {printf \"%.2f\", $TOTAL_SAMPLES / $CHECKPOINT_DURATION}")
            echo ""
            echo "✅ ${CHECKPOINT} evaluation completed"
            echo "   Samples: ${TOTAL_SAMPLES}"
            echo "   Duration: $((CHECKPOINT_DURATION / 60)) min $((CHECKPOINT_DURATION % 60)) sec"
            echo "   Speed: ${SAMPLES_PER_SEC} samples/sec"
        else
            echo ""
            echo "✅ ${CHECKPOINT} evaluation completed"
            echo "   Duration: $((CHECKPOINT_DURATION / 60)) min $((CHECKPOINT_DURATION % 60)) sec"
        fi
        
        # Show post-evaluation GPU status
        echo ""
        echo "Post-evaluation GPU status:"
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{printf "  Memory: %s MB\n  Utilization: %s%%\n", $1, $2}' || echo "  Unable to get GPU info"
        echo ""
    else
        echo ""
        echo "❌ ${CHECKPOINT} evaluation failed"
        echo "   Check log: ${OUTPUT_DIR}/evaluation.log"
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

echo "=================================================="
echo "All evaluations completed!"
echo "=================================================="
echo ""
echo "Summary:"
echo "  Model: ${MODEL_NAME}"
echo "  Checkpoints: ${TOTAL_CHECKPOINTS}"
echo "  Tasks: ${TASKS}"
echo "  Output dir: ${OUTPUT_BASE_DIR}"
echo "  Total time: $((TOTAL_DURATION / 3600)) hours $(((TOTAL_DURATION % 3600) / 60)) min"
echo ""
echo "Next steps:"
echo "  1. View results: python3 view_results_dense3b.py"
echo "  2. Plot charts: python3 plot_results_dense3b.py"
echo ""

