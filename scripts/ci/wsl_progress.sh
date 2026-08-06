#!/bin/bash
cd "$HOME/qrl"
echo "=== 模型数: $(ls models/noise_feedback_v2/*.zip 2>/dev/null | wc -l) ==="
for f in A B C D; do
  last=$(grep -oE "Training (standard|noise) seed=[0-9]+|\[done\].*seed=[0-9]+" logs/noise_150k_proc_$f.log 2>/dev/null | tail -1)
  echo "  $f: ${last:-启动中}"
done
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1
echo "=== 进程 ==="
ps aux | grep train_noise | grep -v grep | wc -l
