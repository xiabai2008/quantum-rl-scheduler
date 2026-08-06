#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/qrl"
echo "=== 安装 CPU 版 torch + 依赖 ==="
uv pip install --python .venv/bin/python \
  torch torchvision \
  stable-baselines3 sb3-contrib gymnasium numpy pandas pyyaml loguru psutil scipy python-pptx 2>&1 | tail -3
echo "=== 验证 ==="
.venv/bin/python -c "
import torch, stable_baselines3, gymnasium, numpy, sys
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('SB3:', stable_baselines3.__version__, '| gym:', gymnasium.__version__, '| numpy:', numpy.__version__)
print('PY:', sys.version.split()[0])
" 2>&1 | tail -4
echo "=== DONE ==="
