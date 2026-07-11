#!/usr/bin/env python
"""
多 Seed 策略对比与统计显著性检验
Multi-Seed Strategy Comparison with Statistical Significance Testing

复用 run_issue_38_67_experiments.py 的环境/策略/仿真基础设施，
在 N 个随机种子下运行 8 策略评估，收集每个 episode 的奖励数据，
输出统计显著性检验报告。

用法：
    python scripts/evaluation/run_multiseed_evaluation.py --seeds 10 --episodes 5
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 复用 run_issue_38_67_experiments.py 的基础设施
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "evaluation"))
from run_issue_38_67_experiments import (  # noqa: E402
    Obs10Wrapper,
    SimulationEnv,
    SimulationTaskGenerator,
    FCFSStrategy,
    RandomStrategy,
    QuantumOnlyStrategy,
    ClassicalOnlyStrategy,
    GreedyStrategy,
    ShortestJobFirstStrategy,
    PPOStrategy,
    DQNModelStrategy,
    build_strategies,
    make_env,
)


def run_multiseed(
    seeds: int = 10,
    episodes_per_seed: int = 5,
    tasks_per_episode: int = 200,
    ppo_model: str = "models/ppo_seed_42_v4/best_model.zip",
    dqn_model: str = "models/dqn_fair_v2/seed_42/best_model.zip",
    alpha: float = 0.05,
) -> dict:
    """运行多seed评估并生成统计显著性报告。"""
    print("=" * 70)
    print("  多 Seed 策略对比与统计显著性检验")
    print("=" * 70)
    print(f"  Seeds:           {seeds}")
    print(f"  Episodes/Seed:   {episodes_per_seed}")
    print(f"  Max Steps/Ep:    {tasks_per_episode}")
    print(f"  PPO Model:       {ppo_model}")
    print(f"  DQN Model:       {dqn_model}")
    print(f"  Alpha:           {alpha}")
    print("=" * 70)

    # 构建策略列表（加载模型）
    strategies = build_strategies(dqn_path=dqn_model, ppo_path=ppo_model)
    strategy_names = [s.name for s in strategies]
    print(f"\n已加载 {len(strategies)} 个策略: {strategy_names}")

    # 种子列表
    seed_list = [42 + i * 137 for i in range(seeds)]  # 使用质数步长增加多样性

    # 收集数据: {strategy_name: [所有episode奖励]}
    all_episode_rewards: dict[str, list[float]] = {s.name: [] for s in strategies}
    seed_details: dict[str, dict] = {}

    start_time = time.time()

    for seed_idx, seed in enumerate(seed_list):
        print(f"\n--- Seed {seed_idx+1}/{seeds} (seed={seed}) ---")
        seed_start = time.time()
        seed_data: dict[str, dict] = {}

        for strategy in strategies:
            # 为每个策略创建独立环境（用相同seed保证任务序列一致，公平对比）
            env = make_env(tasks_per_episode, seed=seed)
            sim_env = SimulationEnv(
                env=env,
                task_generator=SimulationTaskGenerator(seed=seed),
            )

            # 逐episode收集奖励
            ep_rewards = []
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
                ep_rewards.append(float(ep_reward))
                sim_env.record_episode_stats(info)

            all_episode_rewards[strategy.name].extend(ep_rewards)
            seed_data[strategy.name] = {
                "mean_reward": float(np.mean(ep_rewards)),
                "std_reward": float(np.std(ep_rewards)),
                "rewards": ep_rewards,
            }

            # 关闭环境
            try:
                env.close()
            except Exception:
                pass

        seed_elapsed = time.time() - seed_start
        seed_details[str(seed)] = seed_data

        # 打印当前seed摘要
        ppo_mean = seed_data.get("PPO", {}).get("mean_reward", 0)
        fcfs_mean = seed_data.get("FCFS", {}).get("mean_reward", 0)
        if fcfs_mean != 0:
            imp = (ppo_mean - fcfs_mean) / abs(fcfs_mean) * 100
        else:
            imp = 0
        print(f"  完成 ({seed_elapsed:.1f}s) | PPO={ppo_mean:.1f}, FCFS={fcfs_mean:.1f}, "
              f"Δ={imp:+.1f}%")

    total_elapsed = time.time() - start_time
    n_total = seeds * episodes_per_seed
    print(f"\n所有 {seeds} seeds 完成，总耗时 {total_elapsed:.1f}s（共 {n_total} 次独立episode）")

    # -----------------------------------------------------------------------
    # 保存原始奖励数据
    # -----------------------------------------------------------------------
    output_dir = _PROJECT_ROOT / "results" / "multiseed_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rewards_json = {
        "config": {
            "seeds": seed_list,
            "episodes_per_seed": episodes_per_seed,
            "tasks_per_episode": tasks_per_episode,
            "total_episodes": n_total,
            "ppo_model": ppo_model,
            "dqn_model": dqn_model,
            "observation_dim": 10,
            "wrapper": "Obs10Wrapper (14→10，公平对比)",
            "arrival_lambda": 0.5,
            "quantum_ratio": 0.7,
            "timestamp": timestamp,
        },
        "rewards": {k: [float(r) for r in v] for k, v in all_episode_rewards.items()},
        "seed_details": seed_details,
    }

    rewards_path = output_dir / f"rewards_multiseed_{timestamp}.json"
    with open(rewards_path, "w", encoding="utf-8") as f:
        json.dump(rewards_json, f, ensure_ascii=False, indent=2)
    canonical_path = output_dir / "rewards_multiseed.json"
    with open(canonical_path, "w", encoding="utf-8") as f:
        json.dump(rewards_json, f, ensure_ascii=False, indent=2)
    print(f"[保存] 奖励数据: {rewards_path}")

    # -----------------------------------------------------------------------
    # 打印汇总统计
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  多 Seed 汇总统计（按平均奖励降序）")
    print("=" * 70)
    print(f"  {'策略':<16} {'平均奖励':>10} {'标准差':>10} {'StdErr':>8} {'最小':>10} {'最大':>10} {'N':>6}")
    print("  " + "-" * 70)

    sorted_strategies = sorted(
        all_episode_rewards.keys(),
        key=lambda s: np.mean(all_episode_rewards[s]),
        reverse=True,
    )
    for sname in sorted_strategies:
        rewards = all_episode_rewards[sname]
        n = len(rewards)
        m = np.mean(rewards)
        s = np.std(rewards, ddof=1) if n > 1 else 0
        se = s / np.sqrt(n) if n > 0 else 0
        print(f"  {sname:<16} {m:>10.2f} {s:>10.2f} {se:>8.2f} "
              f"{np.min(rewards):>10.2f} {np.max(rewards):>10.2f} {n:>6}")

    ppo_rewards = all_episode_rewards.get("PPO", [])
    fcfs_rewards = all_episode_rewards.get("FCFS", [])
    ppo_mean = np.mean(ppo_rewards) if ppo_rewards else 0
    fcfs_mean = np.mean(fcfs_rewards) if fcfs_rewards else 0
    fcfs_std = np.std(fcfs_rewards, ddof=1) if len(fcfs_rewards) > 1 else 0
    improvement = (ppo_mean - fcfs_mean) / abs(fcfs_mean) * 100 if fcfs_mean != 0 else 0

    print(f"\n  核心结论：PPO={ppo_mean:.2f}±{np.std(ppo_rewards,ddof=1)/np.sqrt(len(ppo_rewards)):.2f} "
          f"vs FCFS={fcfs_mean:.2f}±{fcfs_std/np.sqrt(len(fcfs_rewards)):.2f}，"
          f"提升 {improvement:+.1f}%（N={n_total}）")

    # -----------------------------------------------------------------------
    # 统计显著性检验
    # -----------------------------------------------------------------------
    print("\n[统计] 运行显著性检验...")
    from src.utils.stats_significance import compare_strategies
    sig_results = compare_strategies(all_episode_rewards, alpha=alpha)

    # 生成 Markdown 报告
    from scripts.evaluation.statistical_significance import _generate_markdown_report
    base_report = _generate_markdown_report(all_episode_rewards, sig_results, alpha, canonical_path)

    # 构建权威数字摘要头部
    header_lines = [
        "",
        "## 零、权威实验数字（多 Seed 验证）",
        "",
        f"> **实验配置**: {seeds} seeds × {episodes_per_seed} episodes = {n_total} 次独立运行",
        f"> **环境**: 10 维公平对比环境（Obs10Wrapper 截断 14 维原生环境，兼容所有已训练模型）",
        f"> **任务规模**: 每 episode {tasks_per_episode} 步，泊松到达 λ=0.5，量子任务占比 70%",
        f"> **PPO 模型**: `{ppo_model}`（10维，Actor-Critic）",
        f"> **DQN 模型**: `{dqn_model}`（10维，Dueling DQN）",
        f"> **显著性水平**: α = {alpha}（Bonferroni 校正）",
        "",
        "| 排名 | 策略 | 平均奖励 | 标准差 | 标准误 | 提升 vs FCFS |",
        "|:--:|:--|:--:|:--:|:--:|:--:|",
    ]
    for rank, sname in enumerate(sorted_strategies, 1):
        rewards = all_episode_rewards[sname]
        n = len(rewards)
        m = np.mean(rewards)
        s = np.std(rewards, ddof=1) if n > 1 else 0
        se = s / np.sqrt(n) if n > 0 else 0
        if sname != "FCFS" and fcfs_mean != 0:
            imp = ((m - fcfs_mean) / abs(fcfs_mean) * 100)
            imp_str = f"{imp:+.1f}%"
        else:
            imp_str = "基线"
        header_lines.append(f"| {rank} | {sname} | {m:.2f} | {s:.2f} | {se:.2f} | {imp_str} |")

    header_lines.extend([
        "",
        f"**核心结论：PPO 平均奖励 {ppo_mean:.2f} vs FCFS {fcfs_mean:.2f}，提升 {improvement:+.1f}%**",
        f"（N={n_total} 次独立episode，α={alpha}，Bonferroni多重比较校正）",
        "",
        "---",
        "",
    ])

    # 插入到报告中
    report_lines = base_report.split("\n")
    insert_idx = 0
    for i, line in enumerate(report_lines):
        if line.startswith("## ") and i > 0:
            insert_idx = i
            break
    final_report = "\n".join(report_lines[:insert_idx] + header_lines + report_lines[insert_idx:])

    # 更新报告标题和数据来源说明
    final_report = final_report.replace(
        "# 策略对比统计显著性检验报告",
        "# 统计显著性检验报告（多Seed验证）\n\n"
        f"> 本报告为提交清单 `EXP_STAT` 必需文件，使用 {n_total} 次独立episode验证PPO相对于基线策略的统计显著性。"
    )

    # 写报告
    reports_dir = _PROJECT_ROOT / "results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "statistical_validation.md"
    report_path.write_text(final_report, encoding="utf-8")
    print(f"[保存] 统计显著性报告: {report_path}")

    # -----------------------------------------------------------------------
    # 显著性摘要
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  统计显著性检验摘要")
    print("=" * 70)
    sig_count = sum(1 for r in sig_results.values() if r["significant"])
    print(f"  共 {len(sig_results)} 次两两比较，{sig_count} 次显著（Bonferroni校正，α={alpha}）")
    print()
    for pair, info in sig_results.items():
        sig_mark = "✅" if info["significant"] else "❌"
        print(f"  {sig_mark} {pair}: {info['test']}, p={info['p_value']:.4g}, "
              f"{info['effect_size_type']}={info['effect_size']:.4f}")

    # PPO vs FCFS 详情
    for pair, info in sig_results.items():
        if "PPO" in pair and "FCFS" in pair:
            print(f"\n  >>> PPO vs FCFS: p={info['p_value']:.4g}, "
                  f"显著={'是' if info['significant'] else '否'}, "
                  f"{info['interpretation'][:80]}...")

    print("=" * 70)
    print(f"\n完成！权威数字：PPO={ppo_mean:.2f} vs FCFS={fcfs_mean:.2f}，提升 {improvement:+.1f}%")

    return {
        "rewards": all_episode_rewards,
        "ppo_mean": ppo_mean,
        "fcfs_mean": fcfs_mean,
        "improvement_pct": improvement,
        "n_total": n_total,
        "sorted_strategies": sorted_strategies,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="多Seed策略对比统计显著性检验")
    parser.add_argument("--seeds", type=int, default=10, help="随机种子数量（默认10）")
    parser.add_argument("--episodes", type=int, default=5, help="每个seed的episode数（默认5）")
    parser.add_argument("--tasks-per-episode", type=int, default=200, help="每episode最大步数（默认200）")
    parser.add_argument("--ppo-model", type=str, default="models/ppo_seed_42_v4/best_model.zip")
    parser.add_argument("--dqn-model", type=str, default="models/dqn_fair_v2/seed_42/best_model.zip")
    parser.add_argument("--alpha", type=float, default=0.05, help="显著性水平")
    args = parser.parse_args()

    run_multiseed(
        seeds=args.seeds,
        episodes_per_seed=args.episodes,
        tasks_per_episode=args.tasks_per_episode,
        ppo_model=args.ppo_model,
        dqn_model=args.dqn_model,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
