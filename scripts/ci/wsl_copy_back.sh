#!/bin/bash
# =============================================================================
# WSL 复制回传脚本：将 WSL 侧实验结果复制回 Windows 侧仓库
#
# ⚠️ 隐私说明（8.7-v4 红队审查 P0 修复）：本脚本不含任何硬编码的
#    开发者用户路径/主机名。请通过环境变量 QRL_WIN_REPO 指定 Windows 侧
#    仓库在 WSL 中的挂载路径，例如：
#       export QRL_WIN_REPO="/mnt/c/Users/<你的用户名>/pathto/quantum-rl-scheduler"
#    若未设置，脚本将报错退出，避免泄漏私人路径。
# =============================================================================
set -e

if [ -z "${QRL_WIN_REPO:-}" ]; then
  echo "ERROR: 未设置 QRL_WIN_REPO（Windows 侧仓库在 WSL 中的路径）。" >&2
  echo '  示例: export QRL_WIN_REPO="/mnt/c/Users/<user>/x/quantum-rl-scheduler"' >&2
  exit 1
fi

if [ ! -d "$QRL_WIN_REPO" ]; then
  echo "ERROR: QRL_WIN_REPO 目录不存在: $QRL_WIN_REPO" >&2
  exit 1
fi

cp "$HOME/qrl/results/noise_feedback_v2/noise_feedback_v2_results.json" \
  "$QRL_WIN_REPO/results/noise_feedback_v2/noise_feedback_v2_results.json"
cp "$HOME/qrl/results/noise_feedback_v2/noise_feedback_v2_report.md" \
  "$QRL_WIN_REPO/results/noise_feedback_v2/noise_feedback_v2_report.md" 2>/dev/null || true
echo "结果已复制回 Windows"