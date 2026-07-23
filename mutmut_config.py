"""mutation testing 配置 — Issue #34

用法:
    mutmut run --paths-to-mutate src/scheduler/ppo_agent.py
    mutmut results
    mutmut show <mutant-id>
"""

# 目标模块（按优先级）
TARGETS = [
    "src/scheduler/ppo_agent.py",      # PPO核心，覆盖率72%
    "src/scheduler/env.py",            # 环境，覆盖率72%
    "src/scheduler/marl.py",           # MAPPO，覆盖率64%
    "src/api/circuit_breaker.py",      # 熔断器，覆盖率~70%
]

# mutmut CLI参考:
#   pip install mutmut
#   mutmut run --paths-to-mutate "src/scheduler/ppo_agent.py" --runner "python -m pytest tests/test_scheduler.py -x -q"
#   mutmut results
#   mutmut html
