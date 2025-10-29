#!/bin/bash
# Monitor Parallel Evaluation Progress

clear

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║         Dense-1B Parallel Evaluation - Real-time Monitor             ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Find latest log directory
LOG_DIR=$(ls -td /home2/yth/dense1b_evaluation/parallel_logs_* 2>/dev/null | head -1)

if [ -z "$LOG_DIR" ]; then
    echo "❌ No parallel evaluation found"
    echo ""
    echo "Start parallel evaluation first:"
    echo "  bash RUN_PARALLEL.sh"
    exit 1
fi

echo "Log Directory: $LOG_DIR"
echo ""

# Read PIDs
if [ -f "$LOG_DIR/fillblank_pid.txt" ]; then
    FB_PID=$(cat "$LOG_DIR/fillblank_pid.txt")
    FB_GPU=$(cat "$LOG_DIR/fillblank_gpu.txt" | grep -o '[0-9]')
else
    FB_PID="N/A"
    FB_GPU="?"
fi

if [ -f "$LOG_DIR/multiple_choice_pid.txt" ]; then
    MC_PID=$(cat "$LOG_DIR/multiple_choice_pid.txt")
    MC_GPU=$(cat "$LOG_DIR/multiple_choice_gpu.txt" | grep -o '[0-9]')
else
    MC_PID="N/A"
    MC_GPU="?"
fi

# Check process status
echo "Process Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Fill-in-the-blank
if ps -p $FB_PID > /dev/null 2>&1; then
    echo "  Fill-in-the-blank:  ✅ Running (PID: $FB_PID, GPU: $FB_GPU)"
else
    echo "  Fill-in-the-blank:  ⏹️  Stopped (PID: $FB_PID)"
fi

# Multiple Choice
if ps -p $MC_PID > /dev/null 2>&1; then
    echo "  Multiple Choice:    ✅ Running (PID: $MC_PID, GPU: $MC_GPU)"
else
    echo "  Multiple Choice:    ⏹️  Stopped (PID: $MC_PID)"
fi

echo ""

# GPU Status
echo "GPU Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
while IFS=, read -r idx name mem_used mem_total util; do
    mem_pct=$(awk "BEGIN {printf \"%.1f\", ($mem_used/$mem_total)*100}")
    
    # Highlight GPUs in use
    if [ "$idx" == "$FB_GPU" ] || [ "$idx" == "$MC_GPU" ]; then
        echo "  GPU $idx: ${name}"
        echo "    Memory: ${mem_used} MB / ${mem_total} MB (${mem_pct}%)"
        echo "    Utilization: ${util}%"
        if [ "$idx" == "$FB_GPU" ]; then
            echo "    Task: Fill-in-the-blank"
        else
            echo "    Task: Multiple Choice"
        fi
        echo ""
    fi
done

echo ""

# Progress
echo "Progress:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Fill-in-the-blank progress
FB_RESULTS="/home2/yth/dense1b_evaluation/Fill-in-the-blank questions/results_fillblank1b_batch"
if [ -d "$FB_RESULTS" ]; then
    FB_COUNT=$(ls -d "$FB_RESULTS"/iter_* 2>/dev/null | wc -l)
    FB_PROGRESS=$(awk "BEGIN {printf \"%.1f\", ($FB_COUNT/27)*100}")
    echo "  Fill-in-the-blank:  $FB_COUNT/27 checkpoints (${FB_PROGRESS}%)"
else
    echo "  Fill-in-the-blank:  0/27 checkpoints (0%)"
fi

# Multiple Choice progress
MC_RESULTS="/home2/yth/dense1b_evaluation/Multiple-choice questions/results_dense1b"
if [ -d "$MC_RESULTS" ]; then
    MC_COUNT=$(ls -d "$MC_RESULTS"/iter_* 2>/dev/null | wc -l)
    MC_PROGRESS=$(awk "BEGIN {printf \"%.1f\", ($MC_COUNT/27)*100}")
    echo "  Multiple Choice:    $MC_COUNT/27 checkpoints (${MC_PROGRESS}%)"
else
    echo "  Multiple Choice:    0/27 checkpoints (0%)"
fi

echo ""

# Latest log entries
echo "Latest Log Entries:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "$LOG_DIR/fillblank_gpu${FB_GPU}.log" ]; then
    echo "  Fill-in-the-blank (last 3 lines):"
    tail -3 "$LOG_DIR/fillblank_gpu${FB_GPU}.log" | sed 's/^/    /'
    echo ""
fi

if [ -f "$LOG_DIR/multiple_choice_gpu${MC_GPU}.log" ]; then
    echo "  Multiple Choice (last 3 lines):"
    tail -3 "$LOG_DIR/multiple_choice_gpu${MC_GPU}.log" | sed 's/^/    /'
    echo ""
fi

echo ""

# Quick commands
echo "Quick Commands:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Monitor GPUs:          watch -n 1 nvidia-smi"
echo "  View FB log:           tail -f $LOG_DIR/fillblank_gpu${FB_GPU}.log"
echo "  View MC log:           tail -f $LOG_DIR/multiple_choice_gpu${MC_GPU}.log"
echo "  Stop FB:               kill $FB_PID"
echo "  Stop MC:               kill $MC_PID"
echo "  Re-run this monitor:   bash MONITOR_PARALLEL.sh"
echo ""
echo "╚══════════════════════════════════════════════════════════════════════╝"

