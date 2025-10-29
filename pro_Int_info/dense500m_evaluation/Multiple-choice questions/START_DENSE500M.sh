#!/bin/bash
# Dense-500M-Arch1 一键启动脚本

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║         Dense-500M-Arch1 评估 - 一键启动                            ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "请选择要执行的操作:"
echo ""
echo "  1) 快速测试 (推荐先运行，~10 分钟)"
echo "  2) 完整评估 (所有 27 个 checkpoints，~13-27 小时)"
echo "  3) 查看结果"
echo "  4) 生成图表"
echo "  5) 查看具体题目"
echo "  6) 查看指南"
echo "  0) 退出"
echo ""
read -p "请输入选项 (0-6): " choice

case $choice in
    1)
        echo ""
        echo "🚀 运行快速测试..."
        bash quick_test_dense500m.sh
        ;;
    2)
        echo ""
        echo "⚠️  完整评估需要 13-27 小时，确定继续吗？"
        read -p "输入 'yes' 确认: " confirm
        if [ "$confirm" = "yes" ]; then
            echo ""
            echo "🚀 运行完整评估..."
            bash run_evaluation_dense500m.sh
        else
            echo "已取消"
        fi
        ;;
    3)
        echo ""
        echo "📊 查看结果..."
        python3 view_results_dense500m.py
        ;;
    4)
        echo ""
        echo "📈 生成图表..."
        python3 plot_results_dense500m.py
        ;;
    5)
        echo ""
        read -p "请输入任务名 (如: mmlu_anatomy): " task_name
        read -p "显示多少题 (默认 5): " num
        num=${num:-5}
        echo ""
        python3 view_questions.py "$task_name" "$num"
        ;;
    6)
        echo ""
        echo "📖 显示快速指南..."
        cat DENSE500M_QUICK_START.txt
        ;;
    0)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "完成！"
echo ""

