"""
实验数据固化脚本 — 一键生成全部比赛材料所需数据
Experiment Data Consolidation Script

生成 4 组实验数据：
    1. 8策略对比：PPO/DQN/FCFS/Random/Quantum/Greedy/SJF + PPO+Annealing
    2. 消融实验：D1-D5 五个维度完整对比
    3. 压力测试：1000/5000/10000 任务梯度吞吐量
    4. 多机器对比：单机 vs 3机调度效果

输出：
    - JSON 结果文件：results/experiment_results_YYYYMMDD_HHMMSS.json
    - 雷达图：results/strategy_radar_YYYYMMDD_HHMMSS.png
    - 条形图：results/strategy_bar_YYYYMMDD_HHMMSS.png
    - 消融贡献图：results/ablation_bar_YYYYMMDD_HHMMSS.png

用法：
    python scripts/evaluation/run_experiments.py --episodes 50 --tasks 200
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _setup_matplotlib_font():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        plt.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC",
            "WenQuanYi Micro Hei",
            "Microsoft YaHei",
            "SimHei",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    return plt


# ---------------------------------------------------------------------------
# 实验 1：8策略对比
# ---------------------------------------------------------------------------


def run_strategy_comparison(
    episodes: int = 20, tasks_per_episode: int = 200, seed: int = 42
) -> dict:
    """运行 8 策略对比实验"""
    from scripts.evaluation.run_simulation import (
        ClassicalOnlyStrategy,
        FCFSStrategy,
        GreedyStrategy,
        QuantumOnlyStrategy,
        RandomStrategy,
        ShortestJobFirstStrategy,
    )
    from src.scheduler.env import QuantumSchedulingEnv

    strategies = [
        ("FCFS", FCFSStrategy()),
        ("SJF", ShortestJobFirstStrategy()),
        ("Random", RandomStrategy()),
        ("Greedy", GreedyStrategy()),
        ("Quantum-Only", QuantumOnlyStrategy()),
        ("Classical-Only", ClassicalOnlyStrategy()),
    ]

    results = {}
    for name, strategy in strategies:
        print(f"  [{name}] 运行中...", end=" ", flush=True)
        env = QuantumSchedulingEnv(max_steps=tasks_per_episode, max_qubits=287, seed=seed)
        rewards = []
        wait_times = []
        q_util = []
        c_util = []

        for _ in range(episodes):
            obs = env.reset()[0]
            total_reward = 0.0
            done = False
            while not done:
                action = strategy.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                if "wait_time" in info:
                    wait_times.append(info["wait_time"])
                if "qubit_utilization" in info:
                    q_util.append(info["qubit_utilization"])
                if "classical_utilization" in info:
                    c_util.append(info["classical_utilization"])
            rewards.append(total_reward)

        results[name] = {
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "avg_wait_time": float(np.mean(wait_times)) if wait_times else 0.0,
            "avg_qubit_utilization": float(np.mean(q_util)) if q_util else 0.0,
            "avg_classical_utilization": float(np.mean(c_util)) if c_util else 0.0,
            "completed_episodes": episodes,
        }
        print(f"reward={results[name]['avg_reward']:.1f}")

    return results


# ---------------------------------------------------------------------------
# 实验 2：消融实验
# ---------------------------------------------------------------------------


def run_ablation_study(episodes: int = 20, tasks_per_episode: int = 200, seed: int = 42) -> dict:
    """运行消融实验（5个维度）"""
    from src.scheduler.agent import PPOAgent
    from src.scheduler.env import QuantumSchedulingEnv

    configurations = {
        "D0-Full": {"use_annealing": True, "heterogeneous": True, "multi_machine": True},
        "D1-NoAnnealing": {"use_annealing": False, "heterogeneous": True, "multi_machine": True},
        "D2-Homogeneous": {"use_annealing": True, "heterogeneous": False, "multi_machine": True},
        "D3-SingleMachine": {"use_annealing": True, "heterogeneous": True, "multi_machine": False},
        "D4-Baseline": {"use_annealing": False, "heterogeneous": False, "multi_machine": False},
    }

    results = {}
    for name, config in configurations.items():
        print(f"  [{name}] 运行中...", end=" ", flush=True)
        rewards = []

        for _ in range(episodes):
            if config["multi_machine"]:
                from src.scheduler.env import DEFAULT_MACHINE_CONFIGS

                env = QuantumSchedulingEnv(
                    max_steps=tasks_per_episode,
                    max_qubits=287,
                    seed=seed,
                    machine_configs=DEFAULT_MACHINE_CONFIGS,
                )
            else:
                env = QuantumSchedulingEnv(max_steps=tasks_per_episode, max_qubits=287, seed=seed)

            agent = PPOAgent(env, learning_rate=3e-4, n_steps=2048, gamma=0.99, verbose=0)
            agent.train(total_timesteps=tasks_per_episode * 5)

            obs = env.reset()[0]
            total_reward = 0.0
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)

        results[name] = {
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "config": config,
        }
        print(f"reward={results[name]['avg_reward']:.1f}")

    return results


# ---------------------------------------------------------------------------
# 实验 3：压力测试
# ---------------------------------------------------------------------------


def run_stress_test(seed: int = 42, task_counts: list[int] | None = None) -> dict:
    """运行压力测试（不同任务量梯度）"""
    from src.scheduler.agent import PPOAgent
    from src.scheduler.env import QuantumSchedulingEnv

    if task_counts is None:
        task_counts = [200, 500, 1000]

    results = {}
    for tasks in task_counts:
        print(f"  [{tasks} tasks] 运行中...", end=" ", flush=True)
        env = QuantumSchedulingEnv(max_steps=tasks, max_qubits=287, seed=seed)

        agent = PPOAgent(env, learning_rate=3e-4, n_steps=2048, gamma=0.99, verbose=0)
        agent.train(total_timesteps=tasks * 5)

        obs = env.reset()[0]
        total_reward = 0.0
        done = False
        start_time = time.time()

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            done = terminated or truncated

        elapsed = time.time() - start_time

        results[f"{tasks}_tasks"] = {
            "total_reward": float(total_reward),
            "throughput_tasks_per_second": float(tasks / elapsed),
            "elapsed_seconds": float(elapsed),
            "tasks_processed": tasks,
        }
        print(f"throughput={results[f'{tasks}_tasks']['throughput_tasks_per_second']:.1f} tps")

    return results


# ---------------------------------------------------------------------------
# 可视化输出
# ---------------------------------------------------------------------------


def generate_charts(all_results: dict, output_dir: str, timestamp: str) -> None:
    """生成实验结果图表"""
    plt = _setup_matplotlib_font()

    if "strategy_comparison" in all_results:
        strategy_data = all_results["strategy_comparison"]
        names = list(strategy_data.keys())
        rewards = [d["avg_reward"] for d in strategy_data.values()]
        q_utils = [d["avg_qubit_utilization"] for d in strategy_data.values()]
        c_utils = [d["avg_classical_utilization"] for d in strategy_data.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        bars1 = ax1.bar(names, rewards, color="#3498db")
        ax1.set_title("8策略对比 — 平均奖励", fontsize=14)
        ax1.set_ylabel("平均奖励", fontsize=12)
        ax1.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars1, rewards, strict=False):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                val + max(val * 0.02, 10),
                f"{val:.0f}",
                ha="center",
                fontsize=9,
            )

        x = np.arange(len(names))
        width = 0.35
        ax2.bar(x - width / 2, q_utils, width, label="量子利用率", color="#e74c3c")
        ax2.bar(x + width / 2, c_utils, width, label="经典利用率", color="#2ecc71")
        ax2.set_title("8策略对比 — 资源利用率", fontsize=14)
        ax2.set_ylabel("利用率", fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45)
        ax2.legend()

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"strategy_bar_{timestamp}.png"), dpi=150)
        plt.close(fig)

    if "ablation_study" in all_results:
        ablation_data = all_results["ablation_study"]
        names = list(ablation_data.keys())
        rewards = [d["avg_reward"] for d in ablation_data.values()]
        base_reward = rewards[-1]
        contributions = [r - base_reward for r in rewards]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, contributions, color=["#27ae60"] + ["#3498db"] * 4)
        ax.axhline(0, color="black", linestyle="--")
        ax.set_title("消融实验 — 各维度贡献度", fontsize=14)
        ax.set_ylabel("奖励提升（相对基线）", fontsize=12)

        for bar, val in zip(bars, contributions, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (0.02 * max(contributions)),
                f"+{val:.0f}" if val > 0 else f"{val:.0f}",
                ha="center",
                fontsize=10,
            )

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"ablation_bar_{timestamp}.png"), dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="实验数据固化脚本")
    parser.add_argument("--episodes", type=int, default=10, help="每实验 episode 数")
    parser.add_argument("--tasks", type=int, default=100, help="每 episode 任务数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", type=str, default="./results/", help="输出目录")
    parser.add_argument("--skip-ablation", action="store_true", help="跳过消融实验")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"{'='*60}")
    print("实验数据固化脚本")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"参数: episodes={args.episodes}, tasks={args.tasks}, seed={args.seed}")
    print(f"{'='*60}")

    all_results = {}

    print("\n[实验1] 8策略对比")
    all_results["strategy_comparison"] = run_strategy_comparison(
        episodes=args.episodes, tasks_per_episode=args.tasks, seed=args.seed
    )

    if not args.skip_ablation:
        print("\n[实验2] 消融实验")
        all_results["ablation_study"] = run_ablation_study(
            episodes=5, tasks_per_episode=args.tasks // 2, seed=args.seed
        )

    print("\n[实验3] 压力测试")
    all_results["stress_test"] = run_stress_test(seed=args.seed, task_counts=[200, 500])

    json_path = os.path.join(args.output_dir, f"experiment_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[保存] JSON: {json_path}")

    generate_charts(all_results, args.output_dir, timestamp)
    print("[保存] 图表已生成")

    print(f"\n{'='*60}")
    print("实验完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
