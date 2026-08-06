#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME"
# 克隆项目（最新 main 229db1f）
if [ ! -d qrl ]; then
  git clone --depth 1 https://github.com/xiabai2008/quantum-rl-scheduler.git qrl 2>&1 | tail -1
else
  cd qrl && git fetch origin main && git checkout main && git pull origin main 2>&1 | tail -1
fi
cd "$HOME/qrl"
# 建 venv（Python 3.12）
uv venv --python 3.12 .venv 2>&1 | tail -1
# 安装依赖（torch CUDA 版 + SB3）
uv pip install --python .venv/bin/python \
  torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
uv pip install --python .venv/bin/python \
  stable-baselines3 sb3-contrib gymnasium numpy pandas pyyaml loguru psutil scipy 2>&1 | tail -2
echo "=== 验证 ==="
.venv/bin/python -c "
import torch, stable_baselines3
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('SB3:', stable_baselines3.__version__)
"
