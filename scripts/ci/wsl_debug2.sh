#!/bin/bash
cd "$HOME/qrl"
echo "=== 训练进程 ==="
ps aux | grep python | grep -v grep | grep -v "ps aux" | awk '{print $2, $3"%", $4"%", $11}' | head -8
echo "=== A 日志尾部 ==="
tail -6 logs/noise_150k_proc_A.log
echo "=== tensorboard 事件 ==="
find logs/noise_feedback_v2 -name "events*" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -3
