#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/qrl"
echo "=== 安装 torch cu128 ==="
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3
echo "=== 安装其余依赖 ==="
uv pip install --python .venv/bin/python \
  stable-baselines3 sb3-contrib gymnasium numpy pandas pyyaml loguru psutil scipy python-pptx 2>&1 | tail -2
echo "=== 验证 ==="
.venv/bin/python -c "
import torch, stable_baselines3, gymnasium, numpy
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
print('SB3:', stable_baselines3.__version__, '| gymnasium:', gymnasium.__version__, '| numpy:', numpy.__version__)
" 2>&1 | tail -3
echo "=== DONE ==="
