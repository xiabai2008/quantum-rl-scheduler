#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/qrl"
PY="$HOME/qrl/.venv/bin/python"
pkill -f train_noise_feedback_v2 2>/dev/null || true
sleep 2
# 保留已完成模型（seed42/55/68/81 standard），其余重训
echo "已完成: $(ls models/noise_feedback_v2/*.zip 2>/dev/null | wc -l) 个（保留，SKIP 机制自动跳过）"
# 8 进程 × 2 线程（16 核满载）
declare -A SEGS=( [1]="42 47 A" [2]="48 53 B" [3]="54 59 C" [4]="60 65 D" [5]="66 71 E" [6]="72 77 F" [7]="78 83 G" [8]="84 91 H" )
for key in 1 2 3 4 5 6 7 8; do
  set -- ${SEGS[$key]}
  QRL_DEVICE=cpu OMP_NUM_THREADS=2 setsid nohup $PY scripts/training/train_noise_feedback_v2.py \
    --timesteps 150000 --train-only --seed-start $1 --seed-end $2 \
    > logs/noise_150k_proc_$3.log 2>&1 &
  echo "启动 $3: seeds $1-$2"
done
sleep 10
echo "=== 进程数 ==="
ps -ef | grep train_noise | grep -v grep | wc -l
