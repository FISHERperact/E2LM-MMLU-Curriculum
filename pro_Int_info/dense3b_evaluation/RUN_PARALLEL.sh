#!/bin/bash
# Dense-3B Parallel Evaluation - Run on Different GPUs Simultaneously

echo "=================================================================="
echo "Dense-3B Parallel Evaluation Setup"
echo "Running Multiple Choice and Fill-in-the-blank on separate GPUs"
echo "=================================================================="
echo ""

# GPU assignment
MC_GPU=0      # Multiple Choice uses GPU 0
FB_GPU=1      # Fill-in-the-blank uses GPU 1

echo "GPU Assignment:"
echo "  Multiple Choice (Standard MMLU)  → GPU ${MC_GPU}"
echo "  Fill-in-the-blank (Custom MMLU)  → GPU ${FB_GPU}"
echo ""

# Check GPU availability
echo "Checking GPU status..."
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits | nl -v 0
echo ""

read -p "Confirm GPU assignment and start parallel evaluation? (y/n): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "=================================================================="
echo "Starting Parallel Evaluation..."
echo "=================================================================="
echo ""

# Create log directory
LOG_DIR="/home2/yth/dense3b_evaluation/parallel_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Logs will be saved to: $LOG_DIR"
echo ""

# Start Fill-in-the-blank on GPU 1 (faster, finishes first)
echo "🚀 Starting Fill-in-the-blank evaluation on GPU ${FB_GPU}..."
cd "/home2/yth/dense3b_evaluation/Fill-in-the-blank questions"
CUDA_VISIBLE_DEVICES=${FB_GPU} nohup bash run_evaluation_fillblank_FULL.sh \
    > "${LOG_DIR}/fillblank_gpu${FB_GPU}.log" 2>&1 &
FB_PID=$!
echo "   PID: $FB_PID"
echo "   Log: ${LOG_DIR}/fillblank_gpu${FB_GPU}.log"
echo ""

# Wait a moment to avoid resource contention during model loading
sleep 10

# Start Multiple Choice on GPU 0
echo "🚀 Starting Multiple Choice evaluation on GPU ${MC_GPU}..."
cd "/home2/yth/dense3b_evaluation/Multiple-choice questions"
CUDA_VISIBLE_DEVICES=${MC_GPU} nohup bash run_evaluation_dense3b.sh \
    > "${LOG_DIR}/multiple_choice_gpu${MC_GPU}.log" 2>&1 &
MC_PID=$!
echo "   PID: $MC_PID"
echo "   Log: ${LOG_DIR}/multiple_choice_gpu${MC_GPU}.log"
echo ""

echo "=================================================================="
echo "Both evaluations started!"
echo "=================================================================="
echo ""
echo "Process IDs:"
echo "  Fill-in-the-blank:  PID ${FB_PID} (GPU ${FB_GPU})"
echo "  Multiple Choice:    PID ${MC_PID} (GPU ${MC_GPU})"
echo ""
echo "📊 Monitor Progress:"
echo ""
echo "  # Watch both GPUs"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "  # View Fill-in-the-blank log"
echo "  tail -f ${LOG_DIR}/fillblank_gpu${FB_GPU}.log"
echo ""
echo "  # View Multiple Choice log"
echo "  tail -f ${LOG_DIR}/multiple_choice_gpu${MC_GPU}.log"
echo ""
echo "  # Check if processes are running"
echo "  ps aux | grep -E '${FB_PID}|${MC_PID}'"
echo ""
echo "  # Kill if needed"
echo "  kill ${FB_PID}  # Stop Fill-in-the-blank"
echo "  kill ${MC_PID}  # Stop Multiple Choice"
echo ""
echo "⏱️ Estimated Time:"
echo "  Fill-in-the-blank: ~6-7 hours (will finish first)"
echo "  Multiple Choice:   ~13-27 hours"
echo ""
echo "🎯 When Complete:"
echo "  cd /home2/yth/dense3b_evaluation"
echo "  bash VIEW_RESULTS.sh"
echo ""
echo "=================================================================="

# Save PIDs for reference
echo "$FB_PID" > "${LOG_DIR}/fillblank_pid.txt"
echo "$MC_PID" > "${LOG_DIR}/multiple_choice_pid.txt"
echo "GPU ${FB_GPU}" > "${LOG_DIR}/fillblank_gpu.txt"
echo "GPU ${MC_GPU}" > "${LOG_DIR}/multiple_choice_gpu.txt"

echo "✅ Setup complete! Both evaluations running in background."
echo ""

