#!/bin/bash
# =============================================================================
# WSL 同步脚本：将 Windows 侧仓库源码同步到 WSL 侧 qrl 目录用于运行实验
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

cd "$HOME"
# 覆盖 qrl 源码（保留 .venv；本地是最新含 device 可配置）
echo "=== 复制源码 ==="
for d in src scripts tests config docs deliverable_models results; do
  if [ -d "$QRL_WIN_REPO/$d" ]; then
    cp -rf "$QRL_WIN_REPO/$d" "$HOME/qrl/" 2>/dev/null || true
  fi
done
cp -f "$QRL_WIN_REPO/src/scheduler/ppo_agent.py" "$HOME/qrl/src/scheduler/ppo_agent.py" 2>/dev/null || true
# 复制单文件
for f in pyproject.toml README.md; do
  cp -f "$QRL_WIN_REPO/$f" "$HOME/qrl/" 2>/dev/null || true
done
echo "=== 验证 device 配置 ==="
grep -n "QRL_DEVICE" "$HOME/qrl/src/scheduler/ppo_agent.py" | head -2
echo "=== 快速 import 验证 ==="
cd "$HOME/qrl"
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from src.scheduler.env import QuantumSchedulingEnv, DEFAULT_MACHINE_CONFIGS
from src.scheduler.ppo_agent import PPOAgent
import torch
print('env/PPOAgent import OK | CUDA:', torch.cuda.is_available())
" 2>&1 | tail -3
echo "=== DONE ==="