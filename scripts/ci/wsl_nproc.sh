#!/bin/bash
echo "nproc: $(nproc)"
echo "内存: $(free -g | awk '/Mem:/{print $2}')GB"
echo "=== CPU 占用 ==="
top -bn1 | head -5 | tail -2
echo "=== 训练进程 CPU ==="
ps -ef | grep train_noise | grep -v grep | awk '{print $2, $8}' | head -8
