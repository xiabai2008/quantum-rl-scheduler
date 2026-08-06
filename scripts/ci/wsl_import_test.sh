#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/qrl"
uv pip install --python .venv/bin/python requests tqdm colorama 2>&1 | tail -1
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from src.scheduler.env import QuantumSchedulingEnv, DEFAULT_MACHINE_CONFIGS
from src.scheduler.ppo_agent import PPOAgent
from src.scheduler.marl import MultiAgentPPO
import torch
print('全部 import OK | CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))
" 2>&1 | tail -3
echo "=== DONE ==="
