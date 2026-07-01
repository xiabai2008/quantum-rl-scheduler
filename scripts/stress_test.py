#!/usr/bin/env python
"""
阶段二：PPO 压力场景对比 — PPO vs FCFS / SJF / Random / Greedy

4 种负载场景：
  1. 默认 (baseline)     — 标准混合任务
  2. 高负载              — 200 tasks / 100 steps
  3. 量子资源波动        — 量子机时随机断连
  4. 混合任务潮汐        — 前半量子密集 → 后半经典密集

每种场景跑全部策略，生成对比图 + JSON。
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.scheduler.env import QuantumSchedulingEnv

# ============================================================================
# 配置
# ============================================================================
SEED = 42
NUM_TASKS = 200
PPO_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "ppo_seed_42")
STRATEGIES = ["PPO", "FCFS", "SJF", "Random", "Greedy"]
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# ============================================================================
# 场景定义
# ============================================================================


@dataclass
class Scenario:
    name: str
    label: str
    max_steps: int
    description: str
    # 可选：自定义任务生成参数
    quantum_ratio: float = 0.7  # 量子任务占比
    classical_ratio: float = 0.2  # 经典任务占比
    qubit_availability: float = 1.0  # 量子资源可用比率 (0-1)
    # 潮汐模式
    tidal_mode: bool = False
    tidal_switch_step: int = 50


SCENARIOS = [
    Scenario(
        name="baseline",
        label="默认负载",
        max_steps=200,
        description="标准混合任务分布 (70% quantum, 20% classical)",
    ),
    Scenario(
        name="high_load",
        label="高负载",
        max_steps=80,
        description="200 tasks / 80 steps — 队列持续积压",
    ),
    Scenario(
        name="quantum_volatile",
        label="量子资源波动",
        max_steps=200,
        qubit_availability=0.5,
        description="量子资源可用性降至 50%，模拟真机维护/校准",
    ),
    Scenario(
        name="tidal_mix",
        label="混合潮汐",
        max_steps=200,
        tidal_mode=True,
        tidal_switch_step=100,
        description="前 100 步量子密集 (90%) → 后 100 步经典密集 (90%)",
    ),
]


# ============================================================================
# 策略实现
# ============================================================================


def run_ppo_strategy(env, model_path: str):
    """PPO 策略：加载训练好的模型进行决策"""
    from stable_baselines3 import PPO as SB3PPO

    model = SB3PPO.load(model_path)

    obs = env.reset()[0]
    results = {"rewards": [], "actions": [], "wait_times": [], "completions": []}
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        results["actions"].append(int(action))
        results["wait_times"].append(info.get("avg_wait_time", 0))
        results["completions"].append(info.get("completed_tasks", 0))
        done = terminated or truncated
    results["rewards"].append(total_reward)
    return total_reward, results


def run_baseline_strategy(env, name: str):
    """FCFS / SJF / Random / Greedy"""
    obs = env.reset()[0]
    total_reward = 0
    done = False
    results = {"rewards": [], "actions": [], "wait_times": [], "completions": []}

    while not done:
        # 基于 obs 计算启发式动作
        queue_len = obs[1]
        quantum_avail = obs[0]
        classical_load = obs[4]
        urgency = obs[7]
        is_quantum = obs[8] > 0.5
        is_classical = obs[9] > 0.5

        if name == "FCFS":
            action = 2  # 混合（先来先服务，不分类型）
        elif name == "Random":
            action = np.random.randint(0, 3)
        elif name == "Greedy":
            if urgency > 0.8:
                action = 2  # 紧急任务混合执行
            elif is_quantum and quantum_avail > 0.3:
                action = 1  # 量子
            elif is_classical or classical_load < 0.7:
                action = 0  # 经典
            else:
                action = 2
        elif name == "SJF":
            if queue_len > 0.5:
                action = 2  # 队列长时混合加速
            elif quantum_avail > 0.5:
                action = 1
            else:
                action = 0
        else:
            action = np.random.randint(0, 3)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        results["actions"].append(int(action))
        results["wait_times"].append(info.get("avg_wait_time", 0))
        results["completions"].append(info.get("completed_tasks", 0))
        done = terminated or truncated

    results["rewards"].append(total_reward)
    return total_reward, results


def run_strategy(env, name: str, ppo_model_path: str | None = None):
    """统一入口"""
    if name == "PPO" and ppo_model_path:
        return run_ppo_strategy(env, ppo_model_path)
    else:
        return run_baseline_strategy(env, name)


# ============================================================================
# 场景执行
# ============================================================================


def create_env_for_scenario(scenario: Scenario) -> QuantumSchedulingEnv:
    """根据场景创建定制环境"""
    env = QuantumSchedulingEnv(
        max_steps=scenario.max_steps,
        max_qubits=287,
        seed=SEED,
    )

    # 量子资源波动：修改 qubit 可用比率
    if scenario.qubit_availability < 1.0 and not scenario.tidal_mode:
        # 注入随机断连
        original_advance = env._advance_time

        def volatile_advance():
            original_advance()
            # 每步有一定概率切换量子可用性
            if hasattr(env, "_quantum_volatile_counter"):
                env._quantum_volatile_counter += 1
                if env._quantum_volatile_counter % 5 == 0:
                    env._quantum_resources[0].available_ratio = (
                        scenario.qubit_availability + np.random.uniform(-0.2, 0.2)
                    )
                    env._quantum_resources[0].available_ratio = max(
                        0.1, min(1.0, env._quantum_resources[0].available_ratio)
                    )

        env._quantum_volatile_counter = 0
        env._advance_time = volatile_advance

    # 潮汐模式：动态调整任务类型分布
    if scenario.tidal_mode:
        original_generate = env._generate_random_task

        def tidal_generate():
            current_step = getattr(env, "current_step", 0)
            task = original_generate()
            # 覆盖 task_type
            if current_step < scenario.tidal_switch_step:
                task.task_type = "quantum" if np.random.random() < 0.9 else "classical"
            else:
                task.task_type = "classical" if np.random.random() < 0.9 else "quantum"
            return task

        env._generate_random_task = tidal_generate

    # 设置任务池大小
    env._target_tasks = NUM_TASKS

    return env


def run_stress_test():
    """运行全部压力场景对比"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 找 PPO 模型
    ppo_path = None
    for candidate in [
        os.path.join(PROJECT_ROOT, "models", "ppo_seed_42_v4", "best_model.zip"),
        os.path.join(
            PROJECT_ROOT, "logs", "ablation_with_anneal_seed42", "best_model", "best_model.zip"
        ),
        os.path.join(
            PROJECT_ROOT, "logs", "ablation_no_anneal_seed42", "best_model", "best_model.zip"
        ),
        os.path.join(PPO_MODEL_DIR, "best_model.zip"),
    ]:
        if os.path.exists(candidate):
            ppo_path = candidate
            break

    if not ppo_path:
        # 尝试从 logs 搜索
        for root, _dirs, files in os.walk(os.path.join(PROJECT_ROOT, "logs")):
            for f in files:
                if f == "best_model.zip":
                    ppo_path = os.path.join(root, f)
                    break
            if ppo_path:
                break

    if not ppo_path:
        print(
            "[WARN] 未找到 PPO 模型，PPO 策略将跳过。"
            "请先运行 ablation_annealing.py 或 test_annealing_ppo.py 生成模型。"
        )
        effective_strategies = [s for s in STRATEGIES if s != "PPO"]
    else:
        print(f"[模型] PPO: {ppo_path}")
        effective_strategies = list(STRATEGIES)

    all_results = {}

    for scenario in SCENARIOS:
        print(f"\n{'=' *60}")
        print(f"场景: {scenario.label}")
        print(f"描述: {scenario.description}")
        print(f"max_steps={scenario.max_steps}")
        print(f"{'=' *60}")

        scenario_results = {}

        for strategy_name in effective_strategies:
            print(f"  [{strategy_name:>8}] 运行中...", end=" ", flush=True)

            # 为每种策略创建独立环境（确保公平对比）
            env = create_env_for_scenario(scenario)

            t0 = time.time()
            try:
                reward, details = run_strategy(env, strategy_name, ppo_path)
                elapsed = time.time() - t0

                n_completed = max(details["completions"]) if details["completions"] else 0
                avg_wait = np.mean(details["wait_times"]) if details["wait_times"] else 0

                print(
                    f"reward={reward:.1f}  completed={n_completed}  wait={avg_wait:.1f}  ({elapsed:.1f}s)"
                )

                scenario_results[strategy_name] = {
                    "reward": float(reward),
                    "completed_tasks": int(n_completed),
                    "avg_wait_time": float(avg_wait),
                    "elapsed_s": elapsed,
                }
            except Exception as e:
                print(f"[FAIL] {e}")
                scenario_results[strategy_name] = {
                    "reward": 0,
                    "completed_tasks": 0,
                    "avg_wait_time": 0,
                    "error": str(e),
                }

        all_results[scenario.name] = {
            "label": scenario.label,
            "description": scenario.description,
            "max_steps": scenario.max_steps,
            "results": scenario_results,
        }

    # ---- 保存 JSON ----
    json_path = os.path.join(RESULTS_DIR, f"stress_test_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[JSON] {json_path}")

    # ---- 生成图 ----
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(5 * len(SCENARIOS), 5))
    if len(SCENARIOS) == 1:
        axes = [axes]

    colors = {
        "PPO": "#e74c3c",
        "FCFS": "#3498db",
        "SJF": "#2ecc71",
        "Random": "#95a5a6",
        "Greedy": "#f39c12",
    }

    for ax_idx, scenario in enumerate(SCENARIOS):
        ax = axes[ax_idx]
        sr = all_results[scenario.name]["results"]

        names = list(sr.keys())
        rewards = [sr[n]["reward"] for n in names]

        # 按 reward 排序
        sorted_idx = np.argsort(rewards)[::-1]
        sorted_names = [names[i] for i in sorted_idx]
        sorted_rewards = [rewards[i] for i in sorted_idx]

        bar_colors = [colors.get(n, "#7f8c8d") for n in sorted_names]
        bars = ax.bar(sorted_names, sorted_rewards, color=bar_colors)

        for bar, val in zip(bars, sorted_rewards, strict=False):
            y = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + max(y * 0.02, 20),
                f"{val:.0f}",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(f"{scenario.label}\n({scenario.description})", fontsize=10)
        ax.set_ylabel("Total Reward")
        ax.grid(True, alpha=0.3, axis="y")

        # 高亮 PPO
        if "PPO" in sorted_names:
            ppo_idx = sorted_names.index("PPO")
            bars[ppo_idx].set_edgecolor("black")
            bars[ppo_idx].set_linewidth(2)

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, f"stress_test_{timestamp}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG]  {png_path}")

    # ---- 打印摘要 ----
    print(f"\n{'=' *60}")
    print("压力场景对比摘要")
    print(f"{'=' *60}")

    for scenario in SCENARIOS:
        sr = all_results[scenario.name]["results"]
        sorted_items = sorted(sr.items(), key=lambda x: x[1]["reward"], reverse=True)
        best = sorted_items[0]
        ppo_item = sr.get("PPO")

        print(f"\n  [{scenario.label}]")
        print(f"    最佳: {best[0]} ({best[1]['reward']:.0f})")
        if ppo_item:
            rank = [n for n, _ in sorted_items].index("PPO") + 1
            print(f"    PPO:  #{rank} ({ppo_item['reward']:.0f})")
        else:
            print("    PPO:  未参与（无模型）")

        for name, data in sorted_items:
            print(
                f"      {name:<8} {data['reward']:>8.0f} | "
                f"completed={data.get('completed_tasks', '?'):>4} | "
                f"wait={data.get('avg_wait_time', 0):>6.1f}"
            )

    print(f"\n{'=' *60}")
    print("产出文件:")
    print(f"  JSON: {json_path}")
    print(f"  PNG:  {png_path}")


if __name__ == "__main__":
    run_stress_test()
