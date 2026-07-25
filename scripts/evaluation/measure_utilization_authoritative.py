#!/usr/bin/env python
"""
Issue #116: 量子利用率权威数据测量脚本

测量 PPO/FCFS/SJF 三策略在 14 维原生环境下 10 seeds × 3 episodes 的
量子比特利用率、经典资源利用率、平均等待时间，作为统一口径的权威数据源。

输出：
    - results/utilization_authoritative/utilization_authoritative.json
    - results/utilization_authoritative/utilization_authoritative.md
"""

import contextlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "evaluation"))
from run_issue_38_67_experiments import (
    SimulationEnv,
    SimulationTaskGenerator,
    build_strategies,
    make_env,
)


def measure_utilization(
    seeds: list[int],
    episodes_per_seed: int = 3,
    tasks_per_episode: int = 200,
    obs_dim: int = 14,
    ppo_model: str = "deliverable_models/ppo_best_model_14dim.zip",
    dqn_model: str | None = None,
    target_strategies: list[str] | None = None,
) -> dict:
    """测量多策略×多seed的量子利用率权威数据。

    Args:
        seeds: 随机种子列表
        episodes_per_seed: 每个seed的episode数
        tasks_per_episode: 每episode的最大步数
        obs_dim: 观测维度（14=原生权威配置）
        ppo_model: PPO模型路径
        dqn_model: DQN模型路径（None表示不加载DQN）
        target_strategies: 只保留指定策略（None=全部）

    Returns:
        完整测量结果字典
    """
    print("\n[Issue #116] 量子利用率权威数据测量")
    print(f"  配置: {len(seeds)} seeds × {episodes_per_seed} episodes × {tasks_per_episode} 步")
    print(f"  观测维度: {obs_dim}（原生权威配置）")
    print(f"  Seeds: {seeds}")

    # 构建策略（加载 PPO/DQN 模型）
    strategies = build_strategies(dqn_path=dqn_model, ppo_path=ppo_model, obs_dim=obs_dim)
    if target_strategies:
        strategies = [s for s in strategies if s.name in target_strategies]
    print(f"  策略: {[s.name for s in strategies]}")

    # 收集数据: {strategy_name: {metric: [all_samples]}}
    all_metrics: dict[str, dict[str, list[float]]] = {
        s.name: {"qubit_util": [], "classical_util": [], "wait_time": [], "reward": []}
        for s in strategies
    }
    seed_details: dict[str, dict[str, dict]] = {s.name: {} for s in strategies}

    start_time = time.time()

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed_idx + 1}/{len(seeds)} (seed={seed}) ---")
        for strategy in strategies:
            env = make_env(tasks_per_episode, seed=seed, obs_dim=obs_dim)
            sim_env = SimulationEnv(
                env=env,
                task_generator=SimulationTaskGenerator(seed=seed),
            )

            ep_metrics = {"qubit_util": [], "classical_util": [], "wait_time": [], "reward": []}
            for ep in range(episodes_per_seed):
                obs, info = sim_env.reset(seed=seed + ep)
                ep_reward = 0.0
                step = 0
                while step < tasks_per_episode:
                    action = strategy.select_action(obs)
                    obs, reward, terminated, truncated, info = sim_env.step(action)
                    ep_reward += reward
                    step += 1
                    if terminated or truncated:
                        break
                sim_env.record_episode_stats(info)
                summary = sim_env.get_summary()
                ep_metrics["qubit_util"].append(float(summary["qubit_utilization"]))
                ep_metrics["classical_util"].append(float(summary["classical_utilization"]))
                ep_metrics["wait_time"].append(float(summary["avg_wait_time"]))
                ep_metrics["reward"].append(float(ep_reward))

            # 汇总该 seed 下该策略的统计
            seed_details[strategy.name][str(seed)] = {
                "mean_qubit_util": float(np.mean(ep_metrics["qubit_util"])),
                "mean_classical_util": float(np.mean(ep_metrics["classical_util"])),
                "mean_wait_time": float(np.mean(ep_metrics["wait_time"])),
                "mean_reward": float(np.mean(ep_metrics["reward"])),
                "episodes": ep_metrics,
            }
            all_metrics[strategy.name]["qubit_util"].extend(ep_metrics["qubit_util"])
            all_metrics[strategy.name]["classical_util"].extend(ep_metrics["classical_util"])
            all_metrics[strategy.name]["wait_time"].extend(ep_metrics["wait_time"])
            all_metrics[strategy.name]["reward"].extend(ep_metrics["reward"])

            with contextlib.suppress(Exception):
                env.close()

        # 打印当前 seed 摘要
        for sname in [s.name for s in strategies]:
            d = seed_details[sname][str(seed)]
            print(
                f"  {sname}: qutil={d['mean_qubit_util']:.4f}, "
                f"reward={d['mean_reward']:.1f}, wait={d['mean_wait_time']:.2f}"
            )

    total_elapsed = time.time() - start_time
    print(f"\n所有 {len(seeds)} seeds 完成，总耗时 {total_elapsed:.1f}s")

    # 汇总统计（mean ± std）
    summary_stats: dict[str, dict[str, float]] = {}
    for sname, metrics in all_metrics.items():
        summary_stats[sname] = {
            "qubit_util_mean": float(np.mean(metrics["qubit_util"])),
            "qubit_util_std": float(np.std(metrics["qubit_util"])),
            "classical_util_mean": float(np.mean(metrics["classical_util"])),
            "classical_util_std": float(np.std(metrics["classical_util"])),
            "wait_time_mean": float(np.mean(metrics["wait_time"])),
            "wait_time_std": float(np.std(metrics["wait_time"])),
            "reward_mean": float(np.mean(metrics["reward"])),
            "reward_std": float(np.std(metrics["reward"])),
            "n_samples": len(metrics["qubit_util"]),
        }

    return {
        "type": "utilization_authoritative",
        "timestamp": datetime.now().astimezone().isoformat(),
        "config": {
            "seeds": seeds,
            "episodes_per_seed": episodes_per_seed,
            "tasks_per_episode": tasks_per_episode,
            "observation_dim": obs_dim,
            "ppo_model": ppo_model,
            "dqn_model": dqn_model,
            "total_episodes": len(seeds) * episodes_per_seed,
        },
        "summary": summary_stats,
        "seed_details": seed_details,
        "total_elapsed_sec": round(total_elapsed, 2),
    }


def generate_report(result: dict, output_path: Path) -> None:
    """生成 Markdown 报告。"""
    cfg = result["config"]
    summary = result["summary"]
    lines = [
        "# 量子利用率权威数据测量报告（Issue #116）",
        "",
        f"> **测量时间**: {result['timestamp']}",
        "> **数据用途**: 统一所有文档中量子利用率口径，消除 +48.9% vs -7.4% 矛盾",
        "",
        "## 实验配置",
        "",
        f"- 观测维度: **{cfg['observation_dim']} 维（原生权威配置）**",
        f"- Seeds: {cfg['seeds']}",
        f"- 每 seed episode 数: {cfg['episodes_per_seed']}",
        f"- 每 episode 步数: {cfg['tasks_per_episode']}",
        f"- 总 episode 数: {cfg['total_episodes']}",
        f"- PPO 模型: `{cfg['ppo_model']}`",
        f"- DQN 模型: `{cfg['dqn_model']}`",
        "",
        "## 权威数据汇总",
        "",
        "| 策略 | 量子利用率 | 经典利用率 | 平均等待时间 | 平均奖励 |",
        "|:--|:--:|:--:|:--:|:--:|",
    ]
    for sname in ["PPO", "SJF", "FCFS"]:
        if sname not in summary:
            continue
        s = summary[sname]
        lines.append(
            f"| **{sname}** | {s['qubit_util_mean']:.4f} ± {s['qubit_util_std']:.4f} | "
            f"{s['classical_util_mean']:.4f} ± {s['classical_util_std']:.4f} | "
            f"{s['wait_time_mean']:.2f} ± {s['wait_time_std']:.2f} | "
            f"{s['reward_mean']:.2f} ± {s['reward_std']:.2f} |"
        )
    # 计算提升
    if "PPO" in summary and "FCFS" in summary:
        ppo_q = summary["PPO"]["qubit_util_mean"]
        fcfs_q = summary["FCFS"]["qubit_util_mean"]
        ppo_r = summary["PPO"]["reward_mean"]
        fcfs_r = summary["FCFS"]["reward_mean"]
        q_imp = (ppo_q - fcfs_q) / fcfs_q * 100 if fcfs_q != 0 else 0
        r_imp = (ppo_r - fcfs_r) / abs(fcfs_r) * 100 if fcfs_r != 0 else 0
        q_pp = (ppo_q - fcfs_q) * 100  # 百分点
        lines.extend(
            [
                "",
                "## PPO vs FCFS 提升",
                "",
                f"- **量子利用率**: PPO {ppo_q:.4f} vs FCFS {fcfs_q:.4f} → **{q_imp:+.2f}%** ({q_pp:+.2f} pp)",
                f"- **综合奖励**: PPO {ppo_r:.2f} vs FCFS {fcfs_r:.2f} → **{r_imp:+.2f}%**",
                "",
                "## 统一口径声明",
                "",
                "**以下数据为本项目量子利用率的唯一权威口径**：",
                "",
                f"- PPO 量子利用率: **{ppo_q:.4f} ± {summary['PPO']['qubit_util_std']:.4f}**",
                f"- FCFS 量子利用率: **{fcfs_q:.4f} ± {summary['FCFS']['qubit_util_std']:.4f}**",
                f"- 提升: **{q_imp:+.2f}%** ({q_pp:+.2f} 个百分点)",
                f"- 实验配置: {cfg['observation_dim']}维原生环境 × {cfg['total_episodes']} episodes × {cfg['tasks_per_episode']}步",
                "",
                "## 数据来源与历史矛盾说明",
                "",
                "### 历史矛盾",
                "",
                "| 文档 | FCFS | PPO | 差异 | 数据来源 |",
                "|:--|:--:|:--:|:--:|:--|",
                "| tradeoff_analysis.md（旧） | 33.61% | 50.03% | +48.9% | 2026-07-02 单次运行（非权威） |",
                "| 答辩PPT大纲.md（旧） | 45.36% | 41.98% | -7.4% | 另一批次单次运行（非权威） |",
                "| value_quantification.md（旧） | ~55% | ~72% | +30%（估算） | 估算值（非实测） |",
                "| **本报告（权威）** | **见上表** | **见上表** | **见上表** | **10 seeds × 3 episodes 实测** |",
                "",
                "### 矛盾根因",
                "",
                "1. **tradeoff_analysis.md**: 2026-07-02 单次运行，seed 和 episode 数不足，随机性大",
                "2. **答辩PPT大纲.md**: 另一批次单次运行，实验配置不同（可能不同 seed/步数/环境版本）",
                "3. **value_quantification.md**: 基于消融实验的估算值，非直接测量",
                "",
                "### 统一原则",
                "",
                "- 所有文档统一引用本报告的权威数据",
                "- 利用率数据必须标注实验配置（seeds/episodes/obs_dim）",
                "- 单次运行数据不再作为对外口径",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已生成: {output_path}")


def main() -> None:
    """主入口：运行权威测量并生成报告。"""
    # 10 seeds × 3 episodes = 30 次独立 episode，足以消除单次随机性
    seeds = [42, 137, 274, 411, 548, 685, 822, 959, 1096, 1233]
    result = measure_utilization(
        seeds=seeds,
        episodes_per_seed=3,
        tasks_per_episode=200,
        obs_dim=14,
        target_strategies=["PPO", "SJF", "FCFS"],
    )

    output_dir = _PROJECT_ROOT / "results" / "utilization_authoritative"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "utilization_authoritative.json"
    md_path = output_dir / "utilization_authoritative.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"JSON 已保存: {json_path}")

    generate_report(result, md_path)


if __name__ == "__main__":
    main()
