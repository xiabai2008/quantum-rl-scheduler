#!/bin/bash
cd "$(dirname "$0")"
source .venv-mutmut/bin/activate

# 检查依赖
python -c "import numpy, pytest; print('Dependencies OK')" 2>&1
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install numpy pytest pytest-cov stable-baselines3 gymnasium torch fastapi uvicorn websockets pyyaml python-dotenv loguru -q
fi

echo "=== Running mutmut on src/scheduler/env.py ==="
mutmut run src/scheduler/env.py 2>&1 | tee mutmut_env_results.txt

echo "=== Running mutmut on src/scheduler/marl.py ==="
mutmut run src/scheduler/marl.py 2>&1 | tee mutmut_marl_results.txt

echo "=== Running mutmut on src/api/tianyan_client.py ==="
mutmut run src/api/tianyan_client.py 2>&1 | tee mutmut_client_results.txt

echo "=== Results ==="
mutmut results 2>&1 | tee mutmut_all_results.txt
