#!/bin/bash
# Dense-3B Fill-in-the-blank Evaluation with GPU Selection

# ========================================
# 🎯 GPU SELECTION - Change this number!
# ========================================
GPU_ID=6  # <-- Change this to use different GPU (0, 1, 2, 3, 4, 5, 6, or 7)
# ========================================

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=================================================================="
echo "Dense-3B Fill-in-the-blank Evaluation (Batch Version)"
echo "Selected GPU: $GPU_ID"
echo "=================================================================="
echo ""

# Show selected GPU info
nvidia-smi -i $GPU_ID --query-gpu=index,name,memory.free,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU %s: %s\n  Free Memory: %s MB / %s MB\n\n", $1, $2, $3, $4}'

read -p "Continue with GPU $GPU_ID? (y/n): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Cancelled. Edit this script to change GPU_ID."
    exit 0
fi

echo ""
echo "Starting evaluation on GPU $GPU_ID..."
echo ""

# Run the original script
bash run_evaluation_fillblank_FULL.sh

