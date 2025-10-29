#!/bin/bash
# 检查MMLU数据集的完整性和格式

echo "=============================================="
echo "MMLU 数据集检查工具"
echo "=============================================="
echo ""

MMLU_DIR="$HOME/.cache/huggingface/datasets/cais___mmlu"

# 检查目录是否存在
if [ ! -d "$MMLU_DIR" ]; then
    echo "❌ 错误: MMLU 数据集未找到"
    echo "   位置: $MMLU_DIR"
    echo ""
    echo "请先下载数据集:"
    echo "  python3 -c \"from datasets import load_dataset; load_dataset('cais/mmlu', 'all')\""
    exit 1
fi

echo "✅ MMLU 数据集目录存在"
echo "   位置: $MMLU_DIR"
echo ""

# 统计学科数量
SUBJECT_COUNT=$(ls -d "$MMLU_DIR"/*/ 2>/dev/null | wc -l)
echo "📊 学科数量: $SUBJECT_COUNT 个"
echo ""

# 列出所有学科
echo "📚 所有学科列表:"
echo "----------------------------------------------"
ls "$MMLU_DIR" | column -c 80
echo "----------------------------------------------"
echo ""

# 检查一个示例学科的结构
EXAMPLE_SUBJECT="anatomy"
EXAMPLE_DIR=$(find "$MMLU_DIR/$EXAMPLE_SUBJECT" -name "*.arrow" -type f | head -1 | xargs dirname)

if [ -n "$EXAMPLE_DIR" ]; then
    echo "📂 示例学科结构 ($EXAMPLE_SUBJECT):"
    echo "----------------------------------------------"
    ls -lh "$EXAMPLE_DIR" 2>/dev/null
    echo "----------------------------------------------"
    echo ""
    
    # 显示 dataset_info.json 的关键信息
    INFO_FILE="$EXAMPLE_DIR/dataset_info.json"
    if [ -f "$INFO_FILE" ]; then
        echo "📋 数据集元信息 ($EXAMPLE_SUBJECT):"
        echo "----------------------------------------------"
        python3 -c "
import json
import sys

try:
    with open('$INFO_FILE', 'r') as f:
        info = json.load(f)
    
    print(f\"学科: {info.get('config_name', 'N/A')}\")
    print(f\"版本: {info['version']['version_str']}\")
    print(f\"\\n数据分割:\")
    
    for split_name, split_info in info.get('splits', {}).items():
        print(f\"  - {split_name:12s}: {split_info['num_examples']:4d} 样本\")
    
    print(f\"\\n字段定义:\")
    for field_name, field_info in info.get('features', {}).items():
        field_type = field_info.get('_type', field_info.get('dtype', 'unknown'))
        print(f\"  - {field_name:12s}: {field_type}\")
    
    print(f\"\\n总大小: {info.get('dataset_size', 0):,} bytes\")
except Exception as e:
    print(f\"错误: {e}\", file=sys.stderr)
" 2>/dev/null || echo "无法读取 dataset_info.json"
        echo "----------------------------------------------"
        echo ""
    fi
fi

# 统计总样本数（从一些学科中采样）
echo "📊 数据规模统计 (采样分析):"
echo "----------------------------------------------"

SAMPLE_SUBJECTS=("anatomy" "computer_security" "clinical_knowledge" "high_school_chemistry" "college_mathematics")
TOTAL_SAMPLES=0
CHECKED=0

for subject in "${SAMPLE_SUBJECTS[@]}"; do
    INFO_FILE=$(find "$MMLU_DIR/$subject" -name "dataset_info.json" -type f | head -1)
    if [ -f "$INFO_FILE" ]; then
        SAMPLES=$(python3 -c "
import json
try:
    with open('$INFO_FILE', 'r') as f:
        info = json.load(f)
    test_samples = info.get('splits', {}).get('test', {}).get('num_examples', 0)
    print(test_samples)
except:
    print(0)
" 2>/dev/null)
        if [ "$SAMPLES" -gt 0 ]; then
            printf "  %-30s: %4d 样本\n" "$subject" "$SAMPLES"
            TOTAL_SAMPLES=$((TOTAL_SAMPLES + SAMPLES))
            CHECKED=$((CHECKED + 1))
        fi
    fi
done

if [ $CHECKED -gt 0 ]; then
    AVG_SAMPLES=$((TOTAL_SAMPLES / CHECKED))
    ESTIMATED_TOTAL=$((AVG_SAMPLES * SUBJECT_COUNT))
    echo ""
    echo "  采样平均: ~$AVG_SAMPLES 样本/学科"
    echo "  估计总量: ~$ESTIMATED_TOTAL 样本 (全部 $SUBJECT_COUNT 个学科)"
fi
echo "----------------------------------------------"
echo ""

# 显示缓存大小
CACHE_SIZE=$(du -sh "$MMLU_DIR" 2>/dev/null | cut -f1)
echo "💾 缓存大小: $CACHE_SIZE"
echo ""

echo "✅ 检查完成"
echo ""
echo "后续操作:"
echo "  - 查看详细格式: cat MMLU_DATA_FORMAT.md"
echo "  - 查看题目示例: python3 view_questions.py mmlu_anatomy 5"
echo "  - 运行评估: bash quick_test_dense500m.sh"

