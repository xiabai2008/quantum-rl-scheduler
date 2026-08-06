#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/qrl"
PY="$HOME/qrl/.venv/bin/python"
pkill -f train_noise_feedback_v2 2>/dev/null || true
sleep 3
echo "已完成: $(ls models/noise_feedback_v2/*.zip 2>/dev/null | wc -l) 个（SKIP 保留）"
# 6 进程 × 2 线程（12 线程 < 16 核；内存 6×0.8GB=4.8GB < 7.8GB）
declare -A SEGS=( [1]="42 49 A" [2]="50 57 B" [3]="58 65 C" [4]="66 73 D" [5]="74 81 E" [6]="82 91 F" )
for key in 1 2 3 4 5 6; do
  set -- ${SEGS[$key]}
  QRL_DEVICE=cpu OMP_NUM_THREADS=2 setsid nohup $PY scripts/training/train_noise_feedback_v2.py \
    --timesteps 150000 --train-only --seed-start $1 --seed-end $2 \
    > logs/noise_150k_proc_$3.log 2>&1 &
  echo "启动 $3: seeds $1-$2"
done
sleep 10
echo "进程: $(ps -ef | grep train_noise | grep -v grep | wc -l)"
