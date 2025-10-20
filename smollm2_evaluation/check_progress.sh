#!/bin/bash
# 检查评估进度

echo "评估进度检查"
echo "========================================"
echo ""

# 检查进程
if ps aux | grep -v grep | grep lm_eval > /dev/null; then
    echo "✅ 评估进程正在运行"
    
    # 显示运行时间
    RUNTIME=$(ps aux | grep -v grep | grep lm_eval | awk '{print $10}')
    echo "   运行时间: $RUNTIME"
    
    # 显示 CPU 使用率
    CPU=$(ps aux | grep -v grep | grep lm_eval | awk '{print $3}')
    echo "   CPU 使用率: ${CPU}%"
    
    # 显示内存使用
    MEM=$(ps aux | grep -v grep | grep lm_eval | awk '{print $4}')
    echo "   内存使用率: ${MEM}%"
else
    echo "❌ 评估进程未运行"
fi

echo ""

# 检查 GPU
if command -v nvidia-smi > /dev/null 2>&1; then
    echo "GPU 状态:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | \
    while read line; do
        echo "   $line"
    done
fi

echo ""

# 检查结果目录
RESULT_DIR="/home2/yth/smollm2_evaluation/results/step-125000"
if [ -d "$RESULT_DIR" ]; then
    FILE_COUNT=$(find "$RESULT_DIR" -type f 2>/dev/null | wc -l)
    if [ $FILE_COUNT -gt 0 ]; then
        echo "✅ 已生成 $FILE_COUNT 个结果文件"
        echo ""
        echo "最新文件:"
        ls -lht "$RESULT_DIR" | head -5
    else
        echo "⏳ 结果文件尚未生成（评估进行中）"
    fi
else
    echo "⏳ 结果目录尚未创建"
fi

echo ""
echo "========================================"
echo ""
echo "说明:"
echo "• 如果进程正在运行且 GPU 有使用率 → 正常"
echo "• MMLU 任务需要 30-60 分钟"
echo "• 完整评估需要 1-2 小时"
echo "• 重复的警告信息是正常的"
echo ""

