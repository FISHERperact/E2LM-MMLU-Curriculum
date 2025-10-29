#!/bin/bash
# One-click Launch Menu for Dense-3B Evaluation

clear
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║         Dense-3B MMLU Evaluation - Launch Menu                       ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Choose evaluation to run:"
echo ""
echo "  1) Fill-in-the-blank Full Evaluation (Batch - Recommended!)"
echo "     - 27 checkpoints × 8500 samples"
echo "     - Estimated time: ~6-7 hours"
echo "     - Speed: ~10 samples/sec (20x faster with batch processing)"
echo ""
echo "  2) Multiple Choice Full Evaluation"
echo "     - 27 checkpoints × ~14,079 samples"
echo "     - Estimated time: ~13-27 hours"
echo "     - Speed: ~800 samples/sec"
echo ""
echo "  3) Quick Test (Multiple Choice)"
echo "     - 3 checkpoints × 100 samples"
echo "     - Estimated time: ~10 minutes"
echo ""
echo "  4) Both Evaluations"
echo "     - Fill-in-the-blank first, then multiple choice"
echo "     - Estimated total: ~20-34 hours"
echo ""
echo "  0) Exit"
echo ""
read -p "Select (0-4): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting Fill-in-the-blank Full Evaluation (Batch Version)..."
        cd "/home2/yth/dense3b_evaluation/Fill-in-the-blank questions"
        bash run_evaluation_fillblank_FULL.sh
        ;;
    2)
        echo ""
        echo "🚀 Starting Multiple Choice Full Evaluation..."
        cd "/home2/yth/dense3b_evaluation/Multiple-choice questions"
        bash run_evaluation_dense3b.sh
        ;;
    3)
        echo ""
        echo "🚀 Starting Quick Test..."
        cd "/home2/yth/dense3b_evaluation/Multiple-choice questions"
        bash quick_test_dense3b.sh
        ;;
    4)
        echo ""
        echo "🚀 Starting Both Evaluations..."
        echo ""
        echo "==================== 1/2: Fill-in-the-blank ===================="
        cd "/home2/yth/dense3b_evaluation/Fill-in-the-blank questions"
        bash run_evaluation_fillblank_FULL.sh
        
        echo ""
        echo "==================== 2/2: Multiple Choice ===================="
        cd "/home2/yth/dense3b_evaluation/Multiple-choice questions"
        bash run_evaluation_dense3b.sh
        
        echo ""
        echo "✅ All evaluations completed!"
        ;;
    0)
        echo "Exit"
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                         Evaluation Complete!                         ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

