#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
echo "=== qrl 目录 ==="
ls "$HOME/qrl" 2>/dev/null | head -5
echo "=== venv ==="
if [ -x "$HOME/qrl/.venv/bin/python" ]; then
  echo "venv 存在"
  "$HOME/qrl/.venv/bin/python" -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" 2>&1 | tail -2
else
  echo "venv 不存在（需重建）"
fi
echo "=== 进程 ==="
ps aux 2>/dev/null | grep -E "pip|uv" | grep -v grep | head -3
