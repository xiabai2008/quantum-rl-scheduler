#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/qrl"
PY="$HOME/qrl/.venv/bin/python"
pkill -f train_noise_feedback_v2 2>/dev/null || true
sleep 3
mkdir -p models/noise_feedback_v2/_150k_backup
mv models/noise_feedback_v2/*.zip models/noise_feedback_v2/_150k_backup/ 2>/dev/null || true
rm -rf logs/noise_feedback_v2/standard logs/noise_feedback_v2/noise 2>/dev/null || true
echo "清理后待训练: $(ls models/noise_feedback_v2/*.zip 2>/dev/null | wc -l) 个"
for spec in "42 54 A" "55 67 B" "68 80 C" "81 91 D"; do
  set -- $spec
  QRL_DEVICE=cpu OMP_NUM_THREADS=2 setsid nohup $PY scripts/training/train_noise_feedback_v2.py \
    --timesteps 100000 --train-only --seed-start $1 --seed-end $2 \
    > logs/noise_100k_proc_$3.log 2>&1 &
  echo "启动 $3"
done
sleep 10
echo "进程: $(ps -ef | grep train_noise | grep -v grep | wc -l)"
echo "模型目录: $(ls models/noise_feedback_v2/ | grep -c zip)"
