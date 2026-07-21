#!/bin/bash
set -e

echo "========================================="
echo "  量子RL调度系统 - Docker 一键复现"
echo "  Quantum RL Scheduler - One-Click Demo"
echo "========================================="
echo ""

# 准备目录
mkdir -p /app/results /app/logs /app/models

# 阶段 1：后台启动策略对比仿真
echo "[1/2] 后台启动策略对比仿真（50 episodes x 100 tasks）..."
echo "       对比策略: FCFS / Random / Greedy / SJF / PPO / DQN 等"
echo ""

python scripts/evaluation/run_simulation.py \
    --episodes 50 \
    --tasks-per-episode 100 \
    --output-dir ./results/ \
    --verbose &

SIM_PID=$!
echo "  仿真进程 PID: $SIM_PID"
echo ""

# 错误处理：容器退出时自动 kill 仿真进程
trap "kill $SIM_PID 2>/dev/null; exit 0" EXIT INT TERM

# 后台监控仿真状态
(
    wait $SIM_PID
    SIM_EXIT=$?
    if [ $SIM_EXIT -eq 0 ]; then
        echo ""
        echo "[✅] 仿真完成，结果已写入 /app/results/"
        echo ""
    else
        echo ""
        echo "[❌] 仿真进程异常退出（exit=$SIM_EXIT），结果可能不完整"
        echo "     Web 面板仍可正常访问，请查看日志排查原因"
        echo ""
    fi
) &

# 阶段 2：前台启动 Web 监控面板
echo "[2/2] 启动 Web 监控面板..."
echo "  🌐 访问: http://localhost:8000"
echo "  📊 API:  http://localhost:8000/api/status"
echo "  📈 对比: http://localhost:8000/api/ppo/comparison"
echo ""
echo "  💡 仿真在后台运行，完成后自动写入 results/"
echo "  💡 按 Ctrl+C 停止所有服务"
echo "========================================="
echo ""

exec python -m uvicorn src.visualization.app:app --host 0.0.0.0 --port 8000
