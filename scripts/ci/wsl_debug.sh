#!/bin/bash
cd "$HOME/qrl"
echo "=== 进程 CPU ==="
ps aux | grep train_noise | grep -v grep | awk '{print $3"%", $10}'
echo "=== tensorboard 事件最新 ==="
find logs/noise_feedback_v2 -name "events*" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -4 | awk '{print $1, $2}'
echo "=== A 日志尾部 ==="
tail -5 logs/noise_150k_proc_A.log
