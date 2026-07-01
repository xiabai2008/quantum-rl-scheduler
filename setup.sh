#!/bin/bash
# =============================================================================
# 量子RL调度系统 — 一键环境初始化脚本
#
# 功能：
#   1. 检测 Python 版本并创建虚拟环境
#   2. 安装项目依赖
#   3. 配置环境变量（Mock 模式默认开启）
#   4. 创建必要的目录结构
#   5. 验证关键依赖可用
#
# 用法：
#   bash setup.sh              # 默认安装（Linux/macOS）
#   bash setup.sh --dev        # 额外安装开发工具（pre-commit, debugpy等）
#   bash setup.sh --no-venv    # 不创建虚拟环境（全局安装）
# =============================================================================
set -e

# ---------------------------------------------------------------------------
# 颜色定义
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
USE_VENV=true
DEV_MODE=false

for arg in "$@"; do
    case $arg in
        --dev)
            DEV_MODE=true
            shift
            ;;
        --no-venv)
            USE_VENV=false
            shift
            ;;
        --help|-h)
            echo "Usage: bash setup.sh [--dev] [--no-venv]"
            echo ""
            echo "Options:"
            echo "  --dev       Install additional dev tools (pre-commit, debugpy, etc.)"
            echo "  --no-venv   Skip virtual environment creation"
            echo "  --help      Show this help message"
            exit 0
            ;;
    esac
done

# ---------------------------------------------------------------------------
# 打印标题
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}==========================================${NC}"
echo -e "${BOLD} ${BLUE}Quantum RL Scheduler${NC} — Environment Setup"
echo -e "${BOLD}==========================================${NC}"
echo ""

# ---------------------------------------------------------------------------
# 1. 检测 Python
# ---------------------------------------------------------------------------
echo -e "${BOLD}[1/6]${NC} Checking Python..."

PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" &> /dev/null; then
        version=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$("$candidate" -c "import sys; print(sys.version_info.major)")
        minor=$("$candidate" -c "import sys; print(sys.version_info.minor)")
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$candidate"
            echo -e "  ${GREEN}[PASS]${NC} Found $candidate (Python $version)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "  ${RED}[FAIL]${NC} Python 3.10+ is required but not found."
    echo "  Please install Python from https://www.python.org/downloads/"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. 创建虚拟环境
# ---------------------------------------------------------------------------
VENV_PATH=".venv"

if [ "$USE_VENV" = true ]; then
    echo ""
    echo -e "${BOLD}[2/6]${NC} Creating virtual environment..."

    if [ -d "$VENV_PATH" ]; then
        echo -e "  ${YELLOW}[INFO]${NC} Virtual environment already exists at $VENV_PATH"
        read -p "  Recreate? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_PATH"
            echo "  Removing old virtual environment..."
        else
            echo "  Using existing virtual environment."
        fi
    fi

    if [ ! -d "$VENV_PATH" ]; then
        $PYTHON -m venv "$VENV_PATH"
        echo -e "  ${GREEN}[PASS]${NC} Virtual environment created at $VENV_PATH"
    fi

    # 激活虚拟环境路径
    if [ -f "$VENV_PATH/Scripts/activate" ]; then
        # Windows Git Bash / MSYS2
        PYTHON="$VENV_PATH/Scripts/python.exe"
        PIP="$VENV_PATH/Scripts/pip.exe"
    elif [ -f "$VENV_PATH/bin/activate" ]; then
        # Linux / macOS
        PYTHON="$VENV_PATH/bin/python"
        PIP="$VENV_PATH/bin/pip"
    fi
else
    echo ""
    echo -e "${BOLD}[2/6]${NC} Skipped (--no-venv, using system Python)"
    PIP="$PYTHON -m pip"
fi

# ---------------------------------------------------------------------------
# 3. 升级 pip 并安装依赖
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}[3/6]${NC} Installing dependencies..."

$PYTHON -m pip install --upgrade pip --quiet

if [ "$DEV_MODE" = true ]; then
    echo "  (dev mode: installing extra tools)"
    $PIP install -r requirements.txt
    $PIP install pre-commit pytest-watch debugpy bandit
else
    $PIP install -r requirements.txt
fi

echo -e "  ${GREEN}[PASS]${NC} Dependencies installed."

# ---------------------------------------------------------------------------
# 4. 配置环境变量
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}[4/6]${NC} Setting up environment variables..."

if [ ! -f .env ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "  ${GREEN}[PASS]${NC} Created .env from .env.example (Mock mode enabled by default)."
    else
        # 创建默认 .env
        cat > .env << 'ENVEOF'
# Quantum RL Scheduler — Environment Configuration
TIANYAN_API_KEY=your_api_key_here
TIANYAN_API_SECRET=your_api_secret_here
TIANYAN_MOCK_MODE=true
LOG_LEVEL=INFO
ENVEOF
        echo -e "  ${GREEN}[PASS]${NC} Created default .env (Mock mode)."
    fi
else
    echo -e "  ${YELLOW}[INFO]${NC} .env already exists, keeping existing configuration."
fi

# ---------------------------------------------------------------------------
# 5. 创建项目目录
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}[5/6]${NC} Creating project directories..."

mkdir -p logs models data results
echo -e "  ${GREEN}[PASS]${NC} Created: logs/ models/ data/ results/"

# ---------------------------------------------------------------------------
# 6. 验证安装
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}[6/6]${NC} Verifying installation..."

PASS_COUNT=0
FAIL_COUNT=0

verify_module() {
    local import_name="$1"
    local display_name="$2"
    if $PYTHON -c "import $import_name" 2>/dev/null; then
        echo -e "  ${GREEN}[PASS]${NC} $display_name"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo -e "  ${RED}[FAIL]${NC} $display_name (may need special installation)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

verify_module "numpy"          "numpy"
verify_module "gymnasium"      "gymnasium"
verify_module "stable_baselines3" "stable-baselines3"
verify_module "torch"          "PyTorch"
verify_module "qiskit"         "Qiskit"
verify_module "fastapi"        "FastAPI"
verify_module "sqlalchemy"     "SQLAlchemy"
verify_module "loguru"         "Loguru"
verify_module "pytest"         "pytest"
verify_module "black"          "Black"
verify_module "mypy"           "mypy"

# ---------------------------------------------------------------------------
# 完成
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}==========================================${NC}"
echo -e "${BOLD} ${GREEN}Setup Complete!${NC} ($PASS_COUNT/$((PASS_COUNT + FAIL_COUNT)) modules verified)"
echo -e "${BOLD}==========================================${NC}"
echo ""

if [ "$USE_VENV" = true ]; then
    echo -e "  ${BOLD}Activate virtual environment:${NC}"
    if [ -f "$VENV_PATH/Scripts/activate" ]; then
        echo "    source $VENV_PATH/Scripts/activate"
    else
        echo "    source $VENV_PATH/bin/activate"
    fi
    echo ""
fi

echo -e "  ${BOLD}Quick Start:${NC}"
echo "    # Run tests"
echo "    python -m pytest tests/ -v"
echo ""
echo "    # Quick training (5000 steps)"
echo "    python scripts/quick_train.py"
echo ""
echo "    # Start Web monitor (port 8000)"
echo "    python -m uvicorn src.visualization.app:app --host 0.0.0.0 --port 8000"
echo ""
echo "    # Strategy comparison (8 strategies, 200 tasks)"
echo "    python scripts/run_simulation.py"
echo ""
echo "    # Format code"
echo "    black src/ scripts/ tests/"
echo "    isort src/ scripts/ tests/"
echo ""

if [ "$DEV_MODE" = true ]; then
    echo -e "  ${BOLD}Dev tools installed:${NC}"
    echo "    pre-commit install    # Setup Git pre-commit hooks"
    echo "    ptw                   # Auto-run tests on file changes"
    echo "    bandit -r src/        # Security scan"
    echo ""
fi

echo -e "${BOLD}Happy coding!${NC}"
echo ""
