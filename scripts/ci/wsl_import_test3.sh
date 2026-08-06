#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/qrl"
# 批量装项目所有可选依赖（避免逐个 import 试错）
uv pip install --python .venv/bin/python python-dotenv aiohttp httpx orjson 2>&1 | tail -1
# 若 pyproject 有项目自身依赖则装
uv pip install --python .venv/bin/python -e . 2>&1 | tail -2 || echo "editable 安装失败（跳过，手动加依赖）"
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from src.scheduler.env import QuantumSchedulingEnv, DEFAULT_MACHINE_CONFIGS
from src.scheduler.ppo_agent import PPOAgent
import torch
print('FINAL import OK | CUDA:', torch.cuda.is_available())
" 2>&1 | tail -3
echo "=== DONE ==="
