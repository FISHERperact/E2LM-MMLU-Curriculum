#!/bin/bash
# 快速上传到 GitHub 的脚本

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║           SmolLM2 评估项目 - GitHub 上传脚本                         ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# 检查是否已经初始化 git
if [ ! -d ".git" ]; then
    echo "❌ 错误: 未检测到 .git 目录"
    echo "请先运行: git init"
    exit 1
fi

# 检查是否配置了用户名和邮箱
if [ -z "$(git config user.name)" ] || [ -z "$(git config user.email)" ]; then
    echo "⚠️  请先配置 Git 用户信息："
    echo ""
    read -p "请输入您的名字: " git_name
    read -p "请输入您的邮箱: " git_email
    
    git config user.name "$git_name"
    git config user.email "$git_email"
    
    echo "✅ Git 用户信息配置完成"
    echo ""
fi

# 显示当前状态
echo "📊 当前 Git 状态："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
git status --short
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 询问是否继续
read -p "是否继续提交这些文件？(y/n): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "❌ 已取消"
    exit 0
fi

# 添加所有文件
echo ""
echo "📦 添加文件到 Git..."
git add .

# 创建提交
echo ""
echo "📝 创建提交..."
git commit -m "Initial commit: SmolLM2-1.7B evaluation project

Features:
- Evaluate SmolLM2-1.7B on 8 standard benchmarks
- Support for multiple checkpoints
- Comprehensive documentation (README.md, GUIDE.md)
- Visualization tools (plot_results.py)
- Question viewer (view_questions.py)
- Results: 74.65% PIQA, 72.73% ARC-Easy, 62.94% BoolQ

Tasks evaluated:
- HellaSwag (10,042 questions)
- ARC-Easy & Challenge (3,548 questions)
- MMLU (57 subjects, ~14,079 questions)
- PIQA, WinoGrande, OpenBookQA, BoolQ

Total: ~34,544 questions evaluated"

echo "✅ 提交创建完成"
echo ""

# 检查是否已添加远程仓库
if ! git remote | grep -q "origin"; then
    echo "⚠️  未检测到远程仓库"
    echo ""
    echo "请按照以下步骤操作："
    echo ""
    echo "1. 访问 https://github.com/new"
    echo "2. 创建新仓库："
    echo "   - Repository name: smollm2-evaluation"
    echo "   - Description: Evaluation of SmolLM2-1.7B model on 8 standard benchmarks"
    echo "   - Public/Private: 根据需要选择"
    echo "   - 不要勾选 'Add README file'"
    echo "3. 创建后，复制仓库 URL"
    echo ""
    read -p "请输入您的 GitHub 仓库 URL (如: https://github.com/username/smollm2-evaluation.git): " repo_url
    
    if [ -n "$repo_url" ]; then
        git remote add origin "$repo_url"
        echo "✅ 远程仓库已添加: $repo_url"
    else
        echo "❌ 未输入仓库 URL，请手动添加："
        echo "   git remote add origin https://github.com/YOUR_USERNAME/smollm2-evaluation.git"
        exit 1
    fi
fi

# 获取远程仓库 URL
remote_url=$(git remote get-url origin)
echo ""
echo "🔗 远程仓库: $remote_url"
echo ""

# 推送到 GitHub
read -p "是否现在推送到 GitHub？(y/n): " push_confirm
if [ "$push_confirm" = "y" ] || [ "$push_confirm" = "Y" ]; then
    echo ""
    echo "🚀 推送到 GitHub..."
    
    # 设置默认分支为 main
    git branch -M main
    
    # 推送
    if git push -u origin main; then
        echo ""
        echo "╔══════════════════════════════════════════════════════════════════════╗"
        echo "║                    ✅ 上传成功！                                     ║"
        echo "╚══════════════════════════════════════════════════════════════════════╝"
        echo ""
        echo "🎉 您的项目已成功上传到 GitHub！"
        echo ""
        echo "📍 访问您的仓库: $remote_url"
        echo ""
        echo "接下来可以："
        echo "  1. 在 GitHub 上查看您的项目"
        echo "  2. 编辑仓库描述和 Topics"
        echo "  3. 添加 LICENSE 文件"
        echo "  4. 邀请其他人协作"
        echo ""
    else
        echo ""
        echo "❌ 推送失败"
        echo ""
        echo "可能的原因："
        echo "  1. 认证失败 - 需要使用 Personal Access Token"
        echo "  2. 远程仓库已有内容 - 使用 git pull 先拉取"
        echo "  3. 网络问题 - 检查网络连接"
        echo ""
        echo "手动推送："
        echo "  git push -u origin main"
        echo ""
        echo "如果需要强制推送（会覆盖远程）："
        echo "  git push -f origin main"
        echo ""
    fi
else
    echo ""
    echo "ℹ️  跳过推送，您可以稍后手动推送："
    echo "   git push -u origin main"
fi

echo ""
echo "📖 更多信息请参考: GITHUB_UPLOAD_GUIDE.md"

