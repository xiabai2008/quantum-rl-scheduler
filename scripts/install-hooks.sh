#!/bin/bash
# ============================================================
#  一键安装 Git Hooks - 量子RL调度系统项目
#  用法: bash scripts/install-hooks.sh
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo ""
echo "=========================================="
echo "  安装 Git Hooks - quantum-rl-scheduler"
echo "=========================================="
echo ""

# 配置 core.hooksPath 指向项目 githooks 目录
cd "$PROJECT_ROOT"
git config core.hooksPath githooks

echo "  commit-msg hook  -> 检查 commit message 格式"
echo "  pre-push hook    -> 阻止直接推送 main 分支"
echo ""
echo "  验证: git config core.hooksPath"
echo "  输出应为: githooks"
echo ""
echo "  如需卸载: git config --unset core.hooksPath"
echo "=========================================="
echo ""
