#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/qrl"
PY="$HOME/qrl/.venv/bin/python"
pkill -f train_noise_feedback_v2 2>/dev/null || true
sleep 2
# 清理半成品
mv models/noise_feedback_v2/*.zip models/noise_feedback_v2/_50k_backup/ 2>/dev/null || true
rm -rf logs/noise_feedback_v2/standard logs/noise_feedback_v2/noise 2>/dev/null || true
# setsid + nohup 启动 4 进程（QRL_DEVICE=cpu；setsid 确保不随会话退出）
for spec in "42 54 A" "55 67 B" "68 80 C" "81 91 D"; do
  set -- $spec
  QRL_DEVICE=cpu OMP_NUM_THREADS=1 setsid nohup $PY scripts/training/train_noise_feedback_v2.py \
    --timesteps 150000 --train-only --seed-start $1 --seed-end $2 \
    > logs/noise_150k_proc_$3.log 2>&1 &
  echo "启动 $3: seeds $1-$2"
done
sleep 10
echo "=== 进程数 ==="
ps -ef | grep train_noise | grep -v grep | wc -l
