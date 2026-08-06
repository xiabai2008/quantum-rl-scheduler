#!/bin/bash
set -e
cd "$HOME"
# 覆盖 qrl 源码（保留 .venv；本地是最新含 device 可配置）
echo "=== 复制源码 ==="
for d in src scripts tests config docs deliverable_models results; do
  if [ -d "/mnt/c/Users/HZR/Desktop/揭榜挂帅擂台赛/qrl-fix-732/$d" ]; then
    cp -rf "/mnt/c/Users/HZR/Desktop/揭榜挂帅擂台赛/qrl-fix-732/$d" "$HOME/qrl/" 2>/dev/null || true
  fi
done
cp -f "/mnt/c/Users/HZR/Desktop/揭榜挂帅擂台赛/qrl-fix-732/src/scheduler/ppo_agent.py" "$HOME/qrl/src/scheduler/ppo_agent.py" 2>/dev/null || true
# 复制单文件
for f in pyproject.toml README.md; do
  cp -f "/mnt/c/Users/HZR/Desktop/揭榜挂帅擂台赛/qrl-fix-732/$f" "$HOME/qrl/" 2>/dev/null || true
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
