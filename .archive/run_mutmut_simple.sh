#!/bin/bash
cd "$(dirname "$0")"

# 检查虚拟环境
if [ ! -d ".venv-mutmut" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv-mutmut
fi

source .venv-mutmut/bin/activate

# 安装依赖（如果缺少）
python -c "import numpy" 2>/dev/null || pip install numpy pytest pytest-cov stable-baselines3 gymnasium torch fastapi uvicorn websockets pyyaml python-dotenv loguru -q

# 安装 mutmut
pip install mutmut -q

echo "=== Running mutmut on src/scheduler/env.py ==="
timeout 300 mutmut run src/scheduler/env.py 2>&1 | tee mutmut_env_results.txt

echo ""
echo "=== Results ==="
mutmut results 2>&1 | tee mutmut_summary.txt
