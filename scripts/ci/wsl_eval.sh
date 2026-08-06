#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/qrl"
PY="$HOME/qrl/.venv/bin/python"
echo "=== 开始评估 50 seeds ==="
QRL_DEVICE=cpu OMP_NUM_THREADS=4 $PY scripts/training/train_noise_feedback_v2.py \
  --eval-only --seed-start 42 --seed-end 91 2>&1 | grep -vE "^\s*$" | tail -30
echo "=== 评估完成 ==="
