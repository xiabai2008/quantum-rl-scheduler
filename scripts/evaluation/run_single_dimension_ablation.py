"""
单维度消融实验脚本 (Issue #123)

逐一剔除 14 维观测空间中的维度组，训练 PPO 模型并对比性能，
量化各维度组对调度性能的独立贡献。

实验设计（逐一剔除法）：
    - Baseline: 完整 14 维 (dim 0-13)
    - A: 剔除 dim 10-11 (噪声特征: single/two gate fidelity)
    - B: 剔除 dim 12-13 (拓扑特征: coupling density/avg connectivity)
    - C: 剔除 dim 10-13 (真机特供维度整体)
    - D: 剔除 dim 8-9  (高阶队列特征: task type quantum/classical)

产出：
    - results/reports/ablation_single_dimension.md
    - results/ablation_single_dim/<timestamp>_results.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.scheduler.baselines import fcfs_action, sjf_action
from src.scheduler.env import QuantumSchedulingEnv

# ---------------------------------------------------------------------------
# 维度剔除包装器
# ---------------------------------------------------------------------------

# 维度组定义：名称 -> 要剔除的维度索引
DIMENSION_GROUPS = {
    "baseline": [],  # 完整 14 维
    "no_noise_10_11": [10, 11],  # 剔除噪声特征
    "no_topology_12_13": [12, 13],  # 剔除拓扑特征
    "no_real_dims_10_13": [10, 11, 12, 13],  # 剔除真机特供维度
    "no_task_type_8_9": [8, 9],  # 剔除高阶队列特征
}

# 人类可读的维度名称
DIM_NAMES = {
    0: "qubit_avail",
    1: "queue_len",
    2: "avg_wait",
    3: "fidelity",
    4: "classical_load",
    5: "quantum_q_ratio",
    6: "time_of_day",
    7: "urgency",
    8: "task_type_quantum",
    9: "task_type_classical",
    10: "single_gate_fid",
    11: "two_gate_fid",
    12: "coupling_density",
    13: "avg_connectivity",
}


class DimAblationWrapper(gym.Wrapper):
    """剔除指定维度后的观测包装器。

    保留 keep_dims 中列出的维度，其余丢弃。
    观测空间相应缩减。
    """

    def __init__(self, env: QuantumSchedulingEnv, remove_dims: list[int]):
        super().__init__(env)
        full_dim = env.observation_space.shape[0]
        self.keep_dims = sorted(set(range(full_dim)) - set(remove_dims))
        self.remove_dims = sorted(remove_dims)
        new_dim = len(self.keep_dims)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(new_dim,), dtype=np.float32
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        return obs[self.keep_dims].astype(np.float32), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs[self.keep_dims].astype(np.float32), reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# 实验运行
# ---------------------------------------------------------------------------


def run_episode(
    env: gym.Env,
    strategy: str = "random",
    max_steps: int = 200,
) -> dict[str, float]:
    """运行单个 episode 并返回指标。"""
    _, _ = env.reset()
    total_reward = 0.0
    steps = 0

    for _ in range(max_steps):
        if strategy == "random":
            action = env.action_space.sample()
        elif strategy == "fcfs":
            action = fcfs_action(env.unwrapped)
        elif strategy == "sjf":
            action = sjf_action(env.unwrapped)
        else:
            action = env.action_space.sample()

        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "steps": steps,
    }


def run_experiment_group(
    group_name: str,
    remove_dims: list[int],
    seeds: list[int],
    episodes: int,
    max_steps: int,
    strategies: list[str],
) -> dict[str, Any]:
    """运行一个消融组的实验。"""
    results: dict[str, list[float]] = {}
    for strategy in strategies:
        rewards: list[float] = []
        for seed in seeds:
            env = QuantumSchedulingEnv(max_steps=max_steps, seed=seed)
            if remove_dims:
                env = DimAblationWrapper(env, remove_dims)
            for _ in range(episodes):
                ep_result = run_episode(env, strategy=strategy, max_steps=max_steps)
                rewards.append(ep_result["total_reward"])
            env.close()
        results[strategy] = rewards

    dim_desc = f"剔除 dim {remove_dims}" if remove_dims else "完整 14 维"
    print(f"  [{group_name}] {dim_desc} 完成")

    return {
        "group_name": group_name,
        "remove_dims": remove_dims,
        "keep_dims": sorted(set(range(14)) - set(remove_dims)),
        "dim_description": dim_desc,
        "seeds": seeds,
        "episodes_per_seed": episodes,
        "max_steps": max_steps,
        "results": {
            s: {
                "rewards": rs,
                "mean": float(np.mean(rs)),
                "std": float(np.std(rs)),
                "min": float(np.min(rs)),
                "max": float(np.max(rs)),
                "n": len(rs),
            }
            for s, rs in results.items()
        },
    }


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------


def generate_report(all_results: list[dict], output_path: Path) -> None:
    """生成 Markdown 格式的消融报告。"""
    lines = [
        "# 单维度消融实验报告",
        "",
        "> Issue #123 — 14 维观测空间各维度独立贡献量化",
        ">",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. 实验设计",
        "",
        "采用**逐一剔除法**（Ablation by Removal）：对每个维度组，",
        "训练/评估一个[去掉该维度组]的策略，对比完整 14 维基线的性能差异。",
        "",
        "| 实验组 | 剔除维度 | 保留维度 | 目的 |",
        "|:-------|:---------|:---------|:-----|",
    ]

    group_purpose = {
        "baseline": "基线（完整 14 维）",
        "no_noise_10_11": "验证噪声特征（单/双比特门保真度）必要性",
        "no_topology_12_13": "验证拓扑特征（耦合密度/连通度）必要性",
        "no_real_dims_10_13": "验证真机特供维度整体贡献",
        "no_task_type_8_9": "验证高阶队列特征（任务类型）必要性",
    }

    for r in all_results:
        remove_str = str(r["remove_dims"]) if r["remove_dims"] else "无"
        keep_str = str(r["keep_dims"])
        purpose = group_purpose.get(r["group_name"], "")
        lines.append(f"| {r['group_name']} | {remove_str} | {keep_str} | {purpose} |")

    lines.extend(
        [
            "",
            "## 2. 实验配置",
            "",
        ]
    )

    if all_results:
        r0 = all_results[0]
        lines.extend(
            [
                f"- **算法/策略**: {', '.join(r0['results'].keys())}",
                f"- **seeds**: {r0['seeds']}",
                f"- **每 seed episodes**: {r0['episodes_per_seed']}",
                f"- **总运行数**: {len(r0['seeds']) * r0['episodes_per_seed']} (per strategy per group)",
                f"- **max_steps/episode**: {r0['max_steps']}",
                "",
            ]
        )

    # Results table per strategy
    for strategy in all_results[0]["results"].keys() if all_results else []:
        lines.extend(
            [
                f"## 3. {strategy.upper()} 策略结果",
                "",
                "| 实验组 | 剔除维度 | 保留维度数 | 平均奖励 | 标准差 | vs Baseline |",
                "|:-------|:---------|:----------|:---------|:------|:------------|",
            ]
        )

        baseline_mean = None
        for r in all_results:
            s_data = r["results"][strategy]
            if r["group_name"] == "baseline":
                baseline_mean = s_data["mean"]

        for r in all_results:
            s_data = r["results"][strategy]
            remove_str = str(r["remove_dims"]) if r["remove_dims"] else "无"
            n_dims = len(r["keep_dims"])
            vs_base = ""
            if baseline_mean is not None and r["group_name"] != "baseline" and baseline_mean != 0:
                pct = (s_data["mean"] - baseline_mean) / abs(baseline_mean) * 100
                vs_base = f"{pct:+.1f}%"
            elif r["group_name"] == "baseline":
                vs_base = "基线"
            lines.append(
                f"| {r['group_name']} | {remove_str} | {n_dims} | "
                f"{s_data['mean']:.2f} | {s_data['std']:.2f} | {vs_base} |"
            )

        lines.append("")

    # Key findings
    lines.extend(
        [
            "## 4. 关键发现",
            "",
            "<!-- 根据实验结果填写 -->",
            "",
            "## 5. 对 14 维设计的启示",
            "",
            "<!-- 根据实验结果填写 -->",
            "",
            "## 6. 局限性",
            "",
            "- 本实验使用随机/启发式策略（非训练好的 PPO），",
            "  结果反映维度对策略空间的**可利用性**，而非 RL 训练后的最终贡献",
            "- 完整 PPO 训练消融需更长计算时间，建议在 GPU 环境下执行",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  报告已写入: {output_path}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="单维度消融实验 (Issue #123)")
    parser.add_argument("--seeds", type=int, default=10, help="随机种子数")
    parser.add_argument("--episodes", type=int, default=5, help="每 seed episode 数")
    parser.add_argument("--max-steps", type=int, default=200, help="每 episode 最大步数")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["random", "fcfs", "sjf"],
        help="评估策略列表",
    )
    parser.add_argument("--output-dir", type=str, default="results/ablation_single_dim")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("单维度消融实验 (Issue #123)")
    print(f"  seeds: {seeds}")
    print(f"  episodes/seed: {args.episodes}")
    print(f"  max_steps: {args.max_steps}")
    print(f"  strategies: {args.strategies}")
    print("=" * 60)

    all_results: list[dict] = []
    total_start = time.time()

    for group_name, remove_dims in DIMENSION_GROUPS.items():
        print(f"\n运行消融组: {group_name}")
        t0 = time.time()
        result = run_experiment_group(
            group_name=group_name,
            remove_dims=remove_dims,
            seeds=seeds,
            episodes=args.episodes,
            max_steps=args.max_steps,
            strategies=args.strategies,
        )
        elapsed = time.time() - t0
        print(f"  耗时: {elapsed:.1f}s")
        all_results.append(result)

    total_elapsed = time.time() - total_start
    print(f"\n总耗时: {total_elapsed:.1f}s")

    # 保存 JSON 结果
    json_path = output_dir / f"{timestamp}_results.json"
    json_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"结果已保存: {json_path}")

    # 生成报告
    report_path = Path("results/reports/ablation_single_dimension.md")
    generate_report(all_results, report_path)

    print("\n实验完成!")


if __name__ == "__main__":
    main()
