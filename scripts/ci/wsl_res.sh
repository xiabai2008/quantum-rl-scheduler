#!/bin/bash
echo "=== 内存 ==="
free -m | head -2
echo "=== 进程 CPU/内存 ==="
ps -eo pid,pcpu,pmem,cmd --sort=-pcpu 2>/dev/null | grep train_noise | head -5
echo "=== 各进程日志最新 ==="
for f in A B C D; do
  echo "--- $f: $(grep -cE '\[done\]' logs/noise_150k_proc_$f.log 2>/dev/null) 个完成 ---"
done
