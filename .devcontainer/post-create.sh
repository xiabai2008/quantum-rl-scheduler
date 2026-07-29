#!/bin/bash
# =============================================================================
# Dev Container 初始化脚本
# 容器首次创建后自动执行，完成项目初始化
# =============================================================================
set -e

echo "=========================================="
echo " Quantum RL Scheduler — Dev Environment"
echo "=========================================="

# 1. 创建项目目录
echo ""
echo "[1/6] Creating project directories..."
mkdir -p /workspace/logs /workspace/models /workspace/data
echo "  [PASS] Directories ready."

# 2. 复制环境变量模板
echo ""
echo "[2/6] Setting up environment variables..."
if [ ! -f /workspace/.env ]; then
    if [ -f /workspace/.env.example ]; then
        cp /workspace/.env.example /workspace/.env
        echo "  [PASS] Created .env from .env.example (Mock mode by default)."
    else
        echo "  [WARN] .env.example not found, skipping."
    fi
else
    echo "  [PASS] .env already exists, keeping existing config."
fi

# 3. 安装 pre-commit hooks（如果配置了）
echo ""
echo "[3/6] Setting up Git hooks..."
if [ -f /workspace/.pre-commit-config.yaml ]; then
    pre-commit install --install-hooks
    echo "  [PASS] Pre-commit hooks installed."
else
    echo "  [INFO] No .pre-commit-config.yaml found, skipping."
fi

# 4. GPU 检测
echo ""
echo "[4/6] Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null && \
    echo "  [PASS] GPU detected and accessible." || \
    echo "  [WARN] nvidia-smi found but GPU not accessible."
else
    echo "  [INFO] No GPU detected (CPU-only mode)."
    echo "  [INFO] To enable GPU: use .devcontainer/docker-compose.gpu.yml"
fi

# PyTorch CUDA 检测
python -c "
import torch
if torch.cuda.is_available():
    print(f'  [PASS] PyTorch CUDA: {torch.cuda.get_device_name(0)}')
    print(f'  [PASS] CUDA version: {torch.version.cuda}')
else:
    print('  [INFO] PyTorch running in CPU mode')
" 2>/dev/null || echo "  [WARN] PyTorch not installed yet"

# 5. 验证关键依赖
echo ""
echo "[5/6] Verifying critical dependencies..."
python -c "import gymnasium; print(f'  [PASS] gymnasium {gymnasium.__version__}')" || echo "  [WARN] gymnasium"
python -c "import stable_baselines3; print(f'  [PASS] stable-baselines3')" || echo "  [WARN] stable-baselines3"
python -c "import torch; print(f'  [PASS] torch {torch.__version__}')" || echo "  [WARN] torch"
python -c "import qiskit; print(f'  [PASS] qiskit {qiskit.__version__}')" || echo "  [WARN] qiskit"
python -c "import fastapi; print(f'  [PASS] fastapi')" || echo "  [WARN] fastapi"

# 6. 打印快速开始信息
echo ""
echo "[6/6] Environment setup complete!"
echo ""
echo "=========================================="
echo " Quick Start"
echo "=========================================="
echo "  # 运行测试"
echo "  python -m pytest tests/ -v"
echo ""
echo "  # 快速训练 (5000步验证)"
echo "  python scripts/quick_train.py"
echo ""
echo "  # 启动 Web 监控 (端口 8000)"
echo "  python -m uvicorn src.visualization.app:app --host 0.0.0.0 --port 8000"
echo ""
echo "  # 8种策略对比仿真"
echo "  python scripts/run_simulation.py"
echo ""
echo "  # Mock API 测试"
echo "  python scripts/test_mock_api.py"
echo ""
echo "=========================================="
echo " Happy coding! "
echo "=========================================="
