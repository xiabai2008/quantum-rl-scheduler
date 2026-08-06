#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/qrl"
PY="$HOME/qrl/.venv/bin/python"
echo "=== 装 tensorboard ==="
uv pip install --python .venv/bin/python tensorboard 2>&1 | tail -1
# 清理旧半成品
mv models/noise_feedback_v2/*.zip models/noise_feedback_v2/_50k_backup/ 2>/dev/null || true
rm -rf logs/noise_feedback_v2/standard logs/noise_feedback_v2/noise 2>/dev/null || true
# 重启 4 进程
for spec in "42 54 A" "55 67 B" "68 80 C" "81 91 D"; do
  set -- $spec
  nohup $PY scripts/training/train_noise_feedback_v2.py \
    --timesteps 150000 --train-only --seed-start $1 --seed-end $2 \
    > logs/noise_150k_proc_$3.log 2>&1 &
  echo "启动 $3: seeds $1-$2"
done
sleep 10
echo "=== 进程数 ==="
ps aux | grep train_noise | grep -v grep | wc -l
echo "=== A 日志 ==="
tail -4 logs/noise_150k_proc_A.log
