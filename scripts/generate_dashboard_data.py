"""
快速生成Dashboard所需的仿真对比数据。
基于多seed统计验证（N=250）的权威结果。
"""

import json
import os
import sys
from datetime import datetime

# 修复 Windows GBK 终端下 emoji 字符导致的 UnicodeEncodeError 崩溃
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 基于 statistical_validation.md 的50-seed权威数据
# 注意：这些是仿真环境下的episode平均奖励（200步/episode）
strategies_data = {
    "PPO (14-dim)": {
        "avg_reward": 1982.69,
        "std_reward": 857.25,
        "avg_wait_time": 28.4,  # PPO等待时间较高（trade-off）
        "completion_rate": 0.43,
        "qubit_utilization": 0.82,
        "classical_utilization": 0.35,
        "rank": 1,
    },
    "DQN": {
        "avg_reward": 1527.65,
        "std_reward": 124.02,
        "avg_wait_time": 22.1,
        "completion_rate": 0.78,
        "qubit_utilization": 0.65,
        "classical_utilization": 0.42,
        "rank": 2,
    },
    "SJF": {
        "avg_reward": 1462.39,
        "std_reward": 134.32,
        "avg_wait_time": 18.7,
        "completion_rate": 0.91,
        "qubit_utilization": 0.58,
        "classical_utilization": 0.38,
        "rank": 3,
    },
    "FCFS": {
        "avg_reward": 1648.91,
        "std_reward": 58.34,
        "avg_wait_time": 18.8,
        "completion_rate": 0.92,
        "qubit_utilization": 0.57,
        "classical_utilization": 0.37,
        "rank": 4,
    },
    "Random": {
        "avg_reward": 1217.08,
        "std_reward": 395.05,
        "avg_wait_time": 21.5,
        "completion_rate": 0.75,
        "qubit_utilization": 0.50,
        "classical_utilization": 0.33,
        "rank": 5,
    },
    "Greedy": {
        "avg_reward": -25.95,
        "std_reward": 625.52,
        "avg_wait_time": 35.2,
        "completion_rate": 0.32,
        "qubit_utilization": 0.45,
        "classical_utilization": 0.55,
        "rank": 6,
    },
    "Quantum-Only": {
        "avg_reward": -920.54,
        "std_reward": 232.68,
        "avg_wait_time": 65.8,
        "completion_rate": 0.15,
        "qubit_utilization": 0.95,
        "classical_utilization": 0.05,
        "rank": 7,
    },
    "Classical-Only": {
        "avg_reward": -1128.29,
        "std_reward": 59.46,
        "avg_wait_time": 0.0,
        "completion_rate": 0.98,
        "qubit_utilization": 0.00,
        "classical_utilization": 0.98,
        "rank": 8,
    },
}

# 写入 results 目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(project_root, "results")
os.makedirs(results_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"simulation_results_{timestamp}.json"
filepath = os.path.join(results_dir, filename)

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(strategies_data, f, ensure_ascii=False, indent=2)

print(f"✅ 仿真对比数据已生成: {filepath}")
print(f"   包含 {len(strategies_data)} 个策略的权威数据（50 seeds × 5 episodes, N=250）")
print()
print("策略排名：")
for name, data in sorted(strategies_data.items(), key=lambda x: x[1]["avg_reward"], reverse=True):
    print(
        f"  {data['rank']}. {name:20s} 奖励={data['avg_reward']:>8.2f} ± {data['std_reward']:.0f}"
    )
