#!/usr/bin/env python
"""Issue #457: 105 数据比特规模稳健性复跑验证。

在天衍-287 真实数据比特规模（105 数据比特+182 耦合比特）下，
用已训练的 PPO 模型（ppo_best_model_16dim.zip，287 规模训练）跑 10 seeds × 5 episodes = N=50，
对比 PPO vs FCFS 提升幅度，验证 +123.4% 权威数字的稳健性。

验收标准（Issue #457）：
- 提升幅度与 123.4% 同量级（缩水 ≤10pp，即 ≥113.4%）→ 通过
- 缩水 >10pp（<113.4%）→ 触发预案 B（48h 全仓统一新数字，冻结顺延 3 天）

用法：
    python scripts/evaluation/run_issue_457_105_qubits_validation.py
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
    BaseStrategy,
    SimulationEnv,
    SimulationTaskGenerator,
    build_strategies,
)

from src.scheduler.env import QuantumSchedulingEnv

# ── 实验配置（与权威 50 seed 评估对齐，缩减为 10 seed 抽查）──
SEEDS = 10
EPISODES_PER_SEED = 5
TASKS_PER_EPISODE = 200
OBS_DIM = 14
MAX_QUBITS_105 = 105  # Issue #457: 天衍-287 真实数据比特数
ALPHA = 0.05

# 权威基准（50 seed N=250, max_qubits=287）
AUTHORITATIVE_PPO_MEAN = 2348.91
AUTHORITATIVE_FCFS_MEAN = 1051.59
AUTHORITATIVE_IMPROVEMENT_PCT = 123.4  # 2348.91/1051.59-1，见 config/statistics.yaml（旧 88.3 为 14 维口径，已废弃）
SHRINK_THRESHOLD_PP = 10.0  # 缩水阈值（百分点）


def make_env_105(tasks_per_episode: int, seed: int | None = None) -> QuantumSchedulingEnv:
    """创建 max_qubits=105 的仿真环境（Issue #457 真实数据比特规模）。"""
    return QuantumSchedulingEnv(
        max_steps=tasks_per_episode,
        max_qubits=MAX_QUBITS_105,
        seed=seed,
    )


def run_single_seed(
    seed: int,
    seed_idx: int,
    strategies: list[BaseStrategy],
) -> tuple[int, dict[str, dict], float]:
    """运行单个 seed 下所有策略×episodes 的评估。

    Args:
        seed: 随机种子
        seed_idx: seed 索引（用于进度打印）
        strategies: 已加载的策略列表

    Returns:
        (seed, seed_data, elapsed) 元组
    """
    seed_start = time.time()
    seed_data: dict[str, dict] = {}

    for strategy in strategies:
        env = make_env_105(TASKS_PER_EPISODE, seed=seed)
        sim_env = SimulationEnv(
            env=env,
            task_generator=SimulationTaskGenerator(seed=seed),
        )

        ep_rewards: list[float] = []
        for ep in range(EPISODES_PER_SEED):
            obs, info = sim_env.reset(seed=seed + ep)
            ep_reward = 0.0
            step = 0
            while step < TASKS_PER_EPISODE:
                action = strategy.select_action(obs)
                obs, reward, terminated, truncated, info = sim_env.step(action)
                ep_reward += reward
                step += 1
                if terminated or truncated:
                    break
            ep_rewards.append(float(ep_reward))
            sim_env.record_episode_stats(info)

        seed_data[strategy.name] = {
            "mean_reward": float(np.mean(ep_rewards)),
            "std_reward": float(np.std(ep_rewards)),
            "rewards": ep_rewards,
        }

        with contextlib.suppress(Exception):
            env.close()

    elapsed = time.time() - seed_start
    print(f"  Seed {seed_idx + 1}/{SEEDS} (seed={seed}) 完成 ({elapsed:.1f}s)")
    return seed, seed_data, elapsed


def run_validation() -> dict:
    """运行 105 规模 10 seed 稳健性复跑。"""
    print("=" * 70)
    print("  Issue #457: 105 数据比特规模稳健性复跑验证")
    print("=" * 70)
    print(f"  Seeds:           {SEEDS}")
    print(f"  Episodes/Seed:   {EPISODES_PER_SEED}")
    print(f"  Max Steps/Ep:    {TASKS_PER_EPISODE}")
    print(f"  Max Qubits:      {MAX_QUBITS_105}（天衍-287 真实数据比特）")
    print(f"  Obs Dim:         {OBS_DIM}（原生环境）")
    print("  PPO Model:       deliverable_models/ppo_best_model_16dim.zip")
    print(f"  Alpha:           {ALPHA}")
    print(
        f"  权威基准:        PPO vs FCFS +{AUTHORITATIVE_IMPROVEMENT_PCT}%（N=250, max_qubits=287）"
    )
    print(f"  缩水阈值:        >{SHRINK_THRESHOLD_PP}pp 触发预案 B")
    print("=" * 70)

    # 加载策略模型（复用权威评估的模型）
    ppo_path = "deliverable_models/ppo_best_model_16dim.zip"
    dqn_path = None  # DELETED: DQN 模型已在 v9 删除，原 dqn_best_model_14dim.zip 不再提供
    strategies = build_strategies(dqn_path=dqn_path, ppo_path=ppo_path, obs_dim=OBS_DIM)
    strategy_names = [s.name for s in strategies]
    print(f"\n已加载 {len(strategies)} 个策略: {strategy_names}")

    # 种子列表（与权威 50 seed 评估前 10 个种子一致，确保可比性）
    seed_list = [42 + i * 137 for i in range(SEEDS)]

    # 收集数据
    all_episode_rewards: dict[str, list[float]] = {s.name: [] for s in strategies}
    seed_details: dict[str, dict] = {}

    start_time = time.time()
    for seed_idx, seed in enumerate(seed_list):
        print(f"\n--- Seed {seed_idx + 1}/{SEEDS} (seed={seed}) ---")
        seed, seed_data, seed_elapsed = run_single_seed(seed, seed_idx, strategies)
        seed_details[str(seed)] = seed_data
        for sname, sdata in seed_data.items():
            all_episode_rewards[sname].extend(sdata["rewards"])

        # 打印当前 seed 摘要
        ppo_mean = seed_data.get("PPO", {}).get("mean_reward", 0)
        fcfs_mean = seed_data.get("FCFS", {}).get("mean_reward", 0)
        imp = (ppo_mean - fcfs_mean) / abs(fcfs_mean) * 100 if fcfs_mean != 0 else 0
        print(
            f"  完成 ({seed_elapsed:.1f}s) | PPO={ppo_mean:.1f}, FCFS={fcfs_mean:.1f}, "
            f"Δ={imp:+.1f}%"
        )

    total_elapsed = time.time() - start_time
    n_total = SEEDS * EPISODES_PER_SEED
    print(f"\n所有 {SEEDS} seeds 完成，总耗时 {total_elapsed:.1f}s（共 {n_total} 次独立episode）")

    # -----------------------------------------------------------------------
    # 保存原始数据
    # -----------------------------------------------------------------------
    output_dir = _PROJECT_ROOT / "results" / "issue_457_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rewards_json = {
        "config": {
            "issue": "#457",
            "description": "105 数据比特规模稳健性复跑验证",
            "seeds": seed_list,
            "episodes_per_seed": EPISODES_PER_SEED,
            "tasks_per_episode": TASKS_PER_EPISODE,
            "total_episodes": n_total,
            "max_qubits": MAX_QUBITS_105,
            "max_qubits_note": "天衍-287 真实数据比特数（105 数据比特+182 耦合比特）",
            "ppo_model": ppo_path,
            "dqn_model": dqn_path,
            "observation_dim": OBS_DIM,
            "wrapper": "原生 14 维环境",
            "arrival_lambda": 0.5,
            "quantum_ratio": 0.7,
            "timestamp": timestamp,
            "authoritative_baseline": {
                "ppo_mean": AUTHORITATIVE_PPO_MEAN,
                "fcfs_mean": AUTHORITATIVE_FCFS_MEAN,
                "improvement_pct": AUTHORITATIVE_IMPROVEMENT_PCT,
                "n_total": 250,
                "max_qubits": 287,
            },
        },
        "rewards": {k: [float(r) for r in v] for k, v in all_episode_rewards.items()},
        "seed_details": seed_details,
    }

    data_path = output_dir / f"rewards_105q_{timestamp}.json"
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(rewards_json, f, ensure_ascii=False, indent=2)
    canonical_path = output_dir / "rewards_105q_canonical.json"
    with open(canonical_path, "w", encoding="utf-8") as f:
        json.dump(rewards_json, f, ensure_ascii=False, indent=2)
    print(f"[保存] 奖励数据: {data_path}")

    # -----------------------------------------------------------------------
    # 汇总统计
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  105 规模汇总统计（按平均奖励降序）")
    print("=" * 70)
    print(
        f"  {'策略':<16} {'平均奖励':>10} {'标准差':>10} {'StdErr':>8} {'最小':>10} {'最大':>10} {'N':>6}"
    )
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
        print(
            f"  {sname:<16} {m:>10.2f} {s:>10.2f} {se:>8.2f} "
            f"{np.min(rewards):>10.2f} {np.max(rewards):>10.2f} {n:>6}"
        )

    # -----------------------------------------------------------------------
    # PPO vs FCFS 核心对比
    # -----------------------------------------------------------------------
    ppo_rewards = all_episode_rewards.get("PPO", [])
    fcfs_rewards = all_episode_rewards.get("FCFS", [])
    ppo_mean = float(np.mean(ppo_rewards)) if ppo_rewards else 0.0
    fcfs_mean = float(np.mean(fcfs_rewards)) if fcfs_rewards else 0.0
    ppo_std = float(np.std(ppo_rewards, ddof=1)) if len(ppo_rewards) > 1 else 0.0
    fcfs_std = float(np.std(fcfs_rewards, ddof=1)) if len(fcfs_rewards) > 1 else 0.0
    improvement = (ppo_mean - fcfs_mean) / abs(fcfs_mean) * 100 if fcfs_mean != 0 else 0.0

    # 统计检验
    from src.utils.stats_significance import compare_strategies

    sig_results = compare_strategies(all_episode_rewards, alpha=ALPHA)

    print("\n" + "=" * 70)
    print("  PPO vs FCFS 核心对比（105 规模 vs 权威 287 规模）")
    print("=" * 70)
    print(f"  {'指标':<24} {'105 规模（本次）':>20} {'287 规模（权威）':>20}")
    print("  " + "-" * 70)
    print(f"  {'PPO 平均奖励':<24} {ppo_mean:>20.2f} {AUTHORITATIVE_PPO_MEAN:>20.2f}")
    print(f"  {'FCFS 平均奖励':<24} {fcfs_mean:>20.2f} {AUTHORITATIVE_FCFS_MEAN:>20.2f}")
    print(f"  {'提升幅度 %':<24} {improvement:>+19.1f}% {AUTHORITATIVE_IMPROVEMENT_PCT:>+19.1f}%")
    print(f"  {'N':<24} {n_total:>20} {250:>20}")

    # 缩水判断
    shrink_pp = AUTHORITATIVE_IMPROVEMENT_PCT - improvement
    print(f"\n  缩水幅度: {shrink_pp:+.1f}pp（阈值: >{SHRINK_THRESHOLD_PP}pp 触发预案 B）")

    if shrink_pp > SHRINK_THRESHOLD_PP:
        verdict = "❌ 缩水超过阈值，需触发预案 B"
        plan_b_triggered = True
    else:
        verdict = f"✅ 缩水 ≤{SHRINK_THRESHOLD_PP}pp，+{AUTHORITATIVE_IMPROVEMENT_PCT}% 在 105 规模下稳健"
        plan_b_triggered = False

    print(f"  验收结论: {verdict}")

    # PPO vs FCFS 显著性
    for pair, info in sig_results.items():
        if "PPO" in pair and "FCFS" in pair:
            sig_mark = "✅" if info["significant"] else "❌"
            print(
                f"\n  统计检验: {info['test']}, p={info['p_value']:.4g}, "
                f"{info['effect_size_type']}={info['effect_size']:.4f} {sig_mark}"
            )
            print(f"  解读: {info['interpretation'][:100]}...")

    print("=" * 70)
    print(f"\n完成！105 规模 PPO={ppo_mean:.2f} vs FCFS={fcfs_mean:.2f}，提升 {improvement:+.1f}%")

    # -----------------------------------------------------------------------
    # 生成 Markdown 报告
    # -----------------------------------------------------------------------
    _generate_report(
        all_episode_rewards=all_episode_rewards,
        ppo_mean=ppo_mean,
        fcfs_mean=fcfs_mean,
        ppo_std=ppo_std,
        fcfs_std=fcfs_std,
        improvement=improvement,
        shrink_pp=shrink_pp,
        verdict=verdict,
        plan_b_triggered=plan_b_triggered,
        sig_results=sig_results,
        seed_details=seed_details,
        data_path=data_path,
        timestamp=timestamp,
    )

    return {
        "rewards": all_episode_rewards,
        "ppo_mean": ppo_mean,
        "fcfs_mean": fcfs_mean,
        "improvement_pct": improvement,
        "shrink_pp": shrink_pp,
        "plan_b_triggered": plan_b_triggered,
        "verdict": verdict,
        "n_total": n_total,
    }


def _generate_report(
    all_episode_rewards: dict[str, list[float]],
    ppo_mean: float,
    fcfs_mean: float,
    ppo_std: float,
    fcfs_std: float,
    improvement: float,
    shrink_pp: float,
    verdict: str,
    plan_b_triggered: bool,
    sig_results: dict,
    seed_details: dict,
    data_path: Path,
    timestamp: str,
) -> None:
    """生成 105 规模稳健性复跑 Markdown 报告。"""
    from scipy import stats

    reports_dir = _PROJECT_ROOT / "results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "issue_457_105_qubits_validation_report.md"

    n_total = SEEDS * EPISODES_PER_SEED

    lines = [
        "# Issue #457: 105 数据比特规模稳健性复跑报告",
        "",
        f"> **实验配置**: {SEEDS} seeds × {EPISODES_PER_SEED} episodes = {n_total} 次独立运行",
        f"> **仿真规模**: max_qubits={MAX_QUBITS_105}（天衍-287 真实数据比特数）",
        "> **PPO 模型**: deliverable_models/ppo_best_model_16dim.zip（287 规模训练，验证跨规模泛化）",
        f"> **观测维度**: {OBS_DIM} 维（原生环境）",
        f"> **生成时间**: {datetime.now().astimezone().isoformat()}",
        f"> **数据文件**: `{data_path}`",
        "",
        "## 一、背景与任务",
        "",
        "天衍-287 实际为 **105 数据比特 + 182 耦合比特** 超导量子计算机（祖冲之三号同款芯片）。",
        "原仿真使用 `max_qubits=287` 在 2.7 倍于真实数据比特的规模下验证调度，结论外推性存疑。",
        "",
        f"本实验在 `max_qubits=105` 规模下复跑 PPO vs FCFS，验证 +{AUTHORITATIVE_IMPROVEMENT_PCT}% 权威数字的稳健性。",
        "",
        "## 二、验收标准",
        "",
        f"- 提升幅度与 {AUTHORITATIVE_IMPROVEMENT_PCT}% 同量级（缩水 ≤{SHRINK_THRESHOLD_PP}pp，即 ≥{AUTHORITATIVE_IMPROVEMENT_PCT - SHRINK_THRESHOLD_PP:.1f}%）→ 通过",
        f"- 缩水 >{SHRINK_THRESHOLD_PP}pp（<{AUTHORITATIVE_IMPROVEMENT_PCT - SHRINK_THRESHOLD_PP:.1f}%）→ 触发预案 B",
        "",
        "## 三、105 规模汇总统计（全策略）",
        "",
        "| 策略 | 平均奖励 | 标准差 | N |",
        "|:--|:--:|:--:|:--:|",
    ]

    for sname in sorted(
        all_episode_rewards.keys(),
        key=lambda s: np.mean(all_episode_rewards[s]),
        reverse=True,
    ):
        rewards = all_episode_rewards[sname]
        m = float(np.mean(rewards))
        s = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
        lines.append(f"| {sname} | {m:.2f} ± {s:.2f} | {len(rewards)} |")

    # PPO vs FCFS 对比
    ppo_rewards = all_episode_rewards.get("PPO", [])
    fcfs_rewards = all_episode_rewards.get("FCFS", [])

    lines.extend(
        [
            "",
            "## 四、PPO vs FCFS 核心对比（105 规模 vs 权威 287 规模）",
            "",
            "| 指标 | 105 规模（本次） | 287 规模（权威） |",
            "|:--|:--:|:--:|",
            f"| PPO 平均奖励 | {ppo_mean:.2f} ± {ppo_std:.2f} | {AUTHORITATIVE_PPO_MEAN:.2f} |",
            f"| FCFS 平均奖励 | {fcfs_mean:.2f} ± {fcfs_std:.2f} | {AUTHORITATIVE_FCFS_MEAN:.2f} |",
            f"| 提升幅度 % | {improvement:+.1f}% | {AUTHORITATIVE_IMPROVEMENT_PCT:+.1f}% |",
            f"| N | {n_total} | 250 |",
            f"| 缩水幅度 | {shrink_pp:+.1f}pp | — |",
        ]
    )

    # 统计检验
    if len(ppo_rewards) > 1 and len(fcfs_rewards) > 1:
        t_stat, p_val = stats.ttest_ind(ppo_rewards, fcfs_rewards, equal_var=False)
        # Cohen's d
        pooled_std = np.sqrt(
            (
                (len(ppo_rewards) - 1) * np.var(ppo_rewards, ddof=1)
                + (len(fcfs_rewards) - 1) * np.var(fcfs_rewards, ddof=1)
            )
            / (len(ppo_rewards) + len(fcfs_rewards) - 2)
        )
        cohen_d = (ppo_mean - fcfs_mean) / pooled_std if pooled_std > 0 else 0.0

        # 95% CI
        se_diff = np.sqrt(ppo_std**2 / len(ppo_rewards) + fcfs_std**2 / len(fcfs_rewards))
        ci_lower = (ppo_mean - fcfs_mean) - 1.96 * se_diff
        ci_upper = (ppo_mean - fcfs_mean) + 1.96 * se_diff

        lines.extend(
            [
                "",
                "### 4.1 统计检验",
                "",
                "| 统计检验 | 值 |",
                "|:--|:--:|",
                f"| Welch t 统计量 | {t_stat:.4f} |",
                f"| p 值 | {p_val:.4g} |",
                f"| Cohen's d | {cohen_d:.4f} |",
                f"| 95% CI（PPO-FCFS 差值） | [{ci_lower:.2f}, {ci_upper:.2f}] |",
                f"| 显著性（α={ALPHA}） | {'✅ 显著' if p_val < ALPHA else '❌ 不显著'} |",
            ]
        )

    # 验收结论
    lines.extend(
        [
            "",
            "## 五、验收结论",
            "",
            f"**缩水幅度**: {shrink_pp:+.1f}pp（阈值: >{SHRINK_THRESHOLD_PP}pp 触发预案 B）",
            "",
            f"**验收结论**: {verdict}",
            "",
        ]
    )

    if plan_b_triggered:
        lines.extend(
            [
                "## 六、预案 B（已触发）",
                "",
                "缩水超过阈值，需执行预案 B：",
                "- 48h 内全仓统一新数字（105 规模下的权威提升幅度）",
                "- 代码冻结顺延 3 天定位原因",
                "- 重新训练 105 规模 PPO 模型",
                f"- 更新 PPT/白皮书/答辩材料中的所有 +{AUTHORITATIVE_IMPROVEMENT_PCT}% 数字",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 六、结论与说明",
                "",
                f"+{AUTHORITATIVE_IMPROVEMENT_PCT}% 在 105 数据比特规模下稳健（缩水 ≤{SHRINK_THRESHOLD_PP}pp），",
                "已训练的 PPO 模型（287 规模训练）在 105 规模下仍保持显著优势。",
                "",
                "> **口径说明**: 本次复跑使用 287 规模训练的 PPO 模型，在 105 规模环境下评估，",
                "> 验证模型的跨规模泛化能力。观测空间归一化常数（parser.py:618 的 287.0）",
                "> 保持不变以兼容已训练模型，详见 Issue #457 方案 A。",
                "",
            ]
        )

    # 修改说明
    lines.extend(
        [
            "## 七、代码改动说明（方案 A）",
            "",
            "采用最小改动方案，只修改 Issue 点名的 2 处脚本：",
            "- `scripts/evaluation/compilation_full.py:145`: `max_qubits=287` → `max_qubits=105`",
            "- `scripts/evaluation/quantum_noise_calibration.py:62`: `max_qubits=287` → `max_qubits=105`",
            "",
            "核心默认值保持 287（含耦合比特命名规模），加注释说明：",
            "- `src/scheduler/env.py:143`: 注释说明 287=含耦合比特命名规模，数据比特 105",
            "- `src/scheduler/parser.py:618`: 注释说明 287.0 为历史归一化常数，保持以兼容已训练模型",
            "",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[保存] 报告: {report_path}")


if __name__ == "__main__":
    result = run_validation()
    sys.exit(0 if not result["plan_b_triggered"] else 1)
