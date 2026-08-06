#!/bin/bash
echo "=== 所有 python 进程 ==="
ps -ef | grep -i python | grep -v grep
echo "=== GPU 进程 ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null
