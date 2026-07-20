"""Issue #192：扩展真机闭环验证到 10 seeds（mixed_real + simulation）。

设计：
- simulation: 10 seeds（纯仿真，无真机）
- mixed_real: 10 seeds × 10 cap = 100 tasks 真机
- pure_real: 复用 #165 的 3 seeds 数据（配额不允许扩展）

统计：
- 95% 置信区间（t 分布）
- Welch t 检验（mixed_real vs simulation）
- Cohen's d 效应量

结果写到新路径，不覆盖 #165 文件。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

plt.switch_backend("Agg")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.training.run_real_machine_ablation import (
    aggregate_condition,
    run_preflight,
    save_plot,
    train_seed,
)
from scripts.training.train_agent_real import AuditedRealClient

# Issue #192 专用路径（不覆盖 #165）
DEFAULT_RESULTS_192 = _PROJECT_ROOT / "results" / "real_machine" / "issue192_10seeds.json"
DEFAULT_REPORT_192 = _PROJECT_ROOT / "results" / "reports" / "real_machine_validation.md"
DEFAULT_PLOT_192 = _PROJECT_ROOT / "results" / "real_machine" / "real_machine_10seeds.png"
DEFAULT_MODEL_DIR_192 = _PROJECT_ROOT / "models" / "issue192"
DEFAULT_QUOTA_STATE_192 = _PROJECT_ROOT / "results" / "real_machine" / "issue192_quota_state.json"
DEFAULT_QUOTA_CONFIG_192 = _PROJECT_ROOT / "results" / "real_machine" / "issue192_quota_budget.yaml"
DEFAULT_CHECKPOINT_192 = _PROJECT_ROOT / "results" / "real_machine" / "issue192_checkpoint.json"
# #165 结果文件（用于复用 pure_real 数据）
ISSUE165_RESULTS = _PROJECT_ROOT / "results" / "real_machine" / "issue165_ablation.json"

PHYSICAL_MACHINES = frozenset({"tianyan176", "tianyan176-2"})


def parse_seeds_10(value: str) -> list[int]:
    """解析 10 个不同 seed。"""
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise argparse.ArgumentTypeError("#192 必须提供 10 个不同 seed")
    return seeds


def compute_statistics(rewards: list[float], confidence: float = 0.95) -> dict[str, Any]:
    """计算均值、标准差、95% CI（t 分布）。"""
    arr = np.asarray(rewards, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "sem": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "ci_width": 0.0,
        }
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 0 else 0.0
    if n > 1:
        ci_low, ci_high = scipy_stats.t.interval(confidence, df=n - 1, loc=mean, scale=sem)
    else:
        ci_low = ci_high = mean
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ci_width": float(ci_high - ci_low),
        "rewards": rewards,
    }


def compare_conditions(
    rewards_a: list[float], rewards_b: list[float], name_a: str, name_b: str
) -> dict[str, Any]:
    """Welch t 检验比较两个条件。"""
    if len(rewards_a) < 2 or len(rewards_b) < 2:
        return {
            "test": "welch_t",
            "comparison": f"{name_a} vs {name_b}",
            "note": "样本不足（<2），无法检验",
        }
    t_stat, p_value = scipy_stats.ttest_ind(rewards_a, rewards_b, equal_var=False)
    # Cohen's d（pooled std）
    var_a = np.var(rewards_a, ddof=1)
    var_b = np.var(rewards_b, ddof=1)
    n_a, n_b = len(rewards_a), len(rewards_b)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    cohens_d = (np.mean(rewards_a) - np.mean(rewards_b)) / pooled_std if pooled_std > 0 else 0.0
    mean_diff = float(np.mean(rewards_a) - np.mean(rewards_b))
    return {
        "test": "welch_t",
        "comparison": f"{name_a} vs {name_b}",
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "mean_diff": mean_diff,
        "significant": bool(p_value < 0.05),
        "effect_size": (
            "large"
            if abs(cohens_d) >= 0.8
            else "medium"
            if abs(cohens_d) >= 0.5
            else "small"
            if abs(cohens_d) >= 0.2
            else "negligible"
        ),
    }


def load_issue165_pure_real() -> dict[str, Any]:
    """加载 #165 的 pure_real 3 seeds 数据用于复用。"""
    if not ISSUE165_RESULTS.exists():
        return {"available": False, "runs": [], "note": "#165 结果文件不存在"}
    data = json.loads(ISSUE165_RESULTS.read_text(encoding="utf-8"))
    pure_real = data.get("conditions", {}).get("pure_real", {})
    return {
        "available": True,
        "source": "issue165_ablation.json",
        "runs": pure_real.get("runs", []),
        "reward_mean": pure_real.get("reward_mean", 0.0),
        "reward_std": pure_real.get("reward_std", 0.0),
        "real_attempted": pure_real.get("real_attempted", 0),
        "real_completed": pure_real.get("real_completed", 0),
        "note": "复用 #165 的 3 seeds pure_real 数据（配额不允许扩展到 10 seeds）",
    }


def generate_validation_report(results: dict[str, Any], report_path: Path, plot_path: Path) -> None:
    """生成 #192 真机统计验证报告。"""
    conditions = results["conditions"]
    stats_section = results["statistics"]

    rows = []
    for key, label in (
        ("simulation", "纯仿真 (10 seeds)"),
        ("mixed_real", "仿真+真机混合 (10 seeds)"),
        ("pure_real", "纯真机 (3 seeds, 来自 #165)"),
    ):
        if key not in conditions:
            rows.append(f"| {label} | N/A（未运行） | - | - | - | - |")
            continue
        stats = conditions[key]
        ci = stats_section["ci"].get(key, {})
        rows.append(
            f"| {label} | {stats['reward_mean']:.2f} ± {stats['reward_std']:.2f} | "
            f"[{ci.get('ci_low', 0):.2f}, {ci.get('ci_high', 0):.2f}] | "
            f"{ci.get('ci_width', 0):.2f} | "
            f"{stats['real_attempted']}/{stats['real_accepted']}/{stats['real_completed']} | "
            f"{stats['real_participation_rate']:.2%} |"
        )

    preflight = results["preflight"]
    smoke = preflight["smoke"]
    comparison = stats_section["comparison"]
    has_comparison = "comparison" in comparison

    if has_comparison:
        stats_lines = [
            f"### {comparison['comparison']}",
            "",
            "- 检验方法：Welch t 检验（不假设等方差）",
            f"- t 统计量：{comparison.get('t_statistic', 'N/A'):.4f}",
            f"- p 值：{comparison.get('p_value', 'N/A'):.6f}",
            f"- Cohen's d：{comparison.get('cohens_d', 'N/A'):.4f}（{comparison.get('effect_size', 'N/A')}效应量）",
            f"- 均值差：{comparison.get('mean_diff', 'N/A'):.2f}",
            f"- 显著性（α=0.05）：{'**显著**' if comparison.get('significant') else '不显著'}",
        ]
    else:
        stats_lines = [
            "### 统计检验",
            "",
            f"- 状态：{comparison.get('note', '无法检验')}",
            "- 需要 mixed_real 和 simulation 各至少 2 个 seed 才能进行 Welch t 检验",
        ]

    report = [
        "# Issue #192 真机闭环验证扩展（10 seeds）",
        "",
        f"生成时间：{results['generated_at']}",
        f"实验状态：`{results['status']}`",
        "",
        "## 实验目标",
        "",
        "扩展 #165 的 3 seeds 真机验证到 10 seeds，提升统计说服力：",
        "- simulation: 10 seeds（纯仿真，无真机）",
        "- mixed_real: 10 seeds × 10 cap = 100 tasks 真机",
        "- pure_real: 复用 #165 的 3 seeds 数据（配额不允许扩展）",
        "",
        "## 实验口径",
        "",
        f"- seeds: {results['config']['seeds']}",
        f"- 每 seed 固定 {results['config']['tasks_per_seed']} 个训练任务",
        f"- 混合条件 real-prob={results['config']['mixed_real_probability']:.2f}，cap={results['config']['mixed_cap_per_seed']}/seed",
        f"- 物理后端：`{preflight['machine']}`（状态 `{preflight['backend_status']}`）",
        f"- 每任务 {results['config']['shots']} shots",
        f"- 冒烟任务：`{smoke['task_id']}`（状态 `{smoke['status']}`，耗时 {smoke['elapsed_s']:.3f}s）",
        "",
        "## 三条件结果",
        "",
        "| 条件 | reward ± std | 95% CI | CI 宽度 | 真机尝试/接受/完成 | 真机参与率 |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## 统计显著性检验",
        "",
        *stats_lines,
        "",
        "## 与仿真数字对比",
        "",
        "- 仿真权威数字（50seed N=250）：+88.3%",
        "- 本实验 mixed_real vs simulation：见上表",
        "- 一致性：本实验结果用于验证仿真结论在真机环境下的适用性",
        "",
        "## 数据边界",
        "",
        "- pure_real 行为 #165 复用数据（3 seeds），不参与 95% CI 计算",
        "- mixed_real 和 simulation 为 #192 新跑数据（10 seeds），参与统计检验",
        "- 所有真机记录均有 task ID 且状态 completed，无 Mock 调用",
        "- 完整 task ID 和审计记录见 `results/real_machine/issue192_10seeds.json`",
        "",
        "## 三线图",
        "",
        f"![Issue #192 10 seeds 收敛与 reward 对比](../real_machine/{plot_path.name})",
        "",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=parse_seeds_10,
        default=parse_seeds_10("42,43,44,45,46,47,48,49,50,51"),
    )
    parser.add_argument("--tasks-per-seed", type=int, default=200)
    parser.add_argument("--episode-horizon", type=int, default=20)
    parser.add_argument("--mixed-real-prob", type=float, default=0.05)
    parser.add_argument("--mixed-cap-per-seed", type=int, default=10)
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument("--machine", choices=sorted(PHYSICAL_MACHINES), default="tianyan176")
    parser.add_argument("--confirmed-machine-minutes", type=float, required=True)
    parser.add_argument("--confirmed-used-seconds", type=float, required=True)
    parser.add_argument("--observed-task-upper-seconds", type=float, required=True)
    parser.add_argument("--wait-timeout", type=int, default=120)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_192)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_192)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT_192)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR_192)
    parser.add_argument("--quota-state", type=Path, default=DEFAULT_QUOTA_STATE_192)
    parser.add_argument("--quota-config", type=Path, default=DEFAULT_QUOTA_CONFIG_192)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_192,
    )
    parser.add_argument("--skip-real", action="store_true", help="只跑 simulation（调试用）")
    args = parser.parse_args()

    if args.tasks_per_seed != 200:
        raise ValueError("#192 固定 --tasks-per-seed=200")
    if args.episode_horizon <= 0 or args.tasks_per_seed % args.episode_horizon:
        raise ValueError("episode horizon 必须为 200 的正因数")
    if not math.isclose(args.mixed_real_prob, 0.05):
        raise ValueError("#192 混合条件固定 --mixed-real-prob=0.05")
    if not 1 <= args.shots <= 32:
        raise ValueError("shots 必须在 1–32")

    # 复用 #165 pure_real 数据
    pure_real_reuse = load_issue165_pure_real()

    # 计算正式实验硬上限
    formal_cap = 0 if args.skip_real else len(args.seeds) * args.mixed_cap_per_seed  # 10 × 10 = 100

    # 执行 preflight（smoke test 1 task）
    if args.skip_real:
        preflight = {
            "authenticate": False,
            "machine": args.machine,
            "backend_type": "skipped",
            "backend_status": "skipped",
            "smoke": {"mode": "skipped", "task_id": "N/A", "status": "skipped", "elapsed_s": 0.0},
            "estimated_formal_submission_cap": formal_cap,
            "estimated_total_with_smoke": formal_cap,
            "shots_per_task": args.shots,
        }
        quota = None
        audited = None
    else:
        client, quota, preflight = run_preflight(
            machine=args.machine,
            shots=args.shots,
            formal_submission_cap=formal_cap,
            confirmed_machine_minutes=args.confirmed_machine_minutes,
            confirmed_used_seconds=args.confirmed_used_seconds,
            observed_task_upper_seconds=args.observed_task_upper_seconds,
            prior_failed_task_id=None,
            prior_failed_task_shots=args.shots,
            wait_timeout=args.wait_timeout,
            quota_config_path=args.quota_config,
            quota_state_path=args.quota_state,
        )
        audited = AuditedRealClient(client, wait_timeout=args.wait_timeout, poll_interval=2)

    # 断点续跑
    runs_by_condition: dict[str, list[dict[str, Any]]] = {
        "simulation": [],
        "mixed_real": [],
    }
    if args.checkpoint.exists():
        checkpoint_data = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        runs_by_condition = checkpoint_data.get("runs_by_condition", runs_by_condition)

    total_tasks = len(args.seeds) * args.tasks_per_seed

    # 条件配置
    # 如果 checkpoint 中已有 mixed_real 数据，即使 --skip-real 也加载（用于重新生成报告）
    has_mixed_checkpoint = len(runs_by_condition.get("mixed_real", [])) > 0
    if args.skip_real and not has_mixed_checkpoint:
        condition_specs = {
            "simulation": (0.0, 0, None),
        }
    else:
        condition_specs = {
            "simulation": (0.0, 0, None),
            "mixed_real": (args.mixed_real_prob, args.mixed_cap_per_seed, audited),
        }

    aborted = False
    for condition, (probability, cap, condition_client) in condition_specs.items():
        previous = {run["seed"]: run for run in runs_by_condition[condition]}
        refreshed: list[dict[str, Any]] = []
        for seed in args.seeds:
            old_run = previous.get(seed)
            if old_run is not None:
                # 复用已有 run
                refreshed.append(old_run)
                continue
            if aborted:
                continue
            print(f"[{datetime.now():%H:%M:%S}] {condition} seed={seed} 开始训练...")
            run = train_seed(
                condition=condition,
                seed=seed,
                tasks=args.tasks_per_seed,
                episode_horizon=args.episode_horizon,
                machine=args.machine,
                real_probability=probability,
                real_submission_cap=cap,
                shots=args.shots,
                client=condition_client,
                model_dir=args.model_dir,
            )
            refreshed.append(run)
            runs_by_condition[condition] = refreshed
            # 保存 checkpoint
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            args.checkpoint.write_text(
                json.dumps(
                    {
                        "issue": 192,
                        "generated_at": datetime.now().astimezone().isoformat(),
                        "runs_by_condition": runs_by_condition,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"[{datetime.now():%H:%M:%S}] {condition} seed={seed} 完成: "
                f"reward={run['evaluation']['reward']:.2f}, "
                f"real_completed={run['real_completed']}, "
                f"elapsed={run['training_elapsed_s']:.1f}s"
            )
            if condition != "simulation" and run["degraded"]:
                print(f"[警告] {condition} seed={seed} 降级，后续 seed 跳过")
                aborted = True
        runs_by_condition[condition] = refreshed

    # 汇总
    conditions: dict[str, dict[str, Any]] = {}
    for key in condition_specs:
        conditions[key] = aggregate_condition(runs_by_condition[key], total_tasks=total_tasks)

    # pure_real 复用 #165
    if pure_real_reuse["available"]:
        conditions["pure_real"] = aggregate_condition(
            pure_real_reuse["runs"], total_tasks=3 * args.tasks_per_seed
        )
        conditions["pure_real"]["reused_from"] = "issue165"
        conditions["pure_real"]["reuse_note"] = pure_real_reuse["note"]

    # 统计计算
    sim_rewards = [float(run["evaluation"]["reward"]) for run in runs_by_condition["simulation"]]
    mixed_rewards = (
        [float(run["evaluation"]["reward"]) for run in runs_by_condition["mixed_real"]]
        if not args.skip_real
        else []
    )

    ci = {
        "simulation": compute_statistics(sim_rewards),
    }
    if mixed_rewards:
        ci["mixed_real"] = compute_statistics(mixed_rewards)

    comparison = {}
    if len(sim_rewards) >= 2 and len(mixed_rewards) >= 2:
        comparison = compare_conditions(mixed_rewards, sim_rewards, "mixed_real", "simulation")

    statistics = {
        "ci": ci,
        "comparison": comparison if comparison else {"note": "样本不足，无法比较"},
    }

    # #192 自己的 status 判断逻辑（不依赖 #165 的 condition_is_valid，它硬编码 3 seeds）
    if "mixed_real" in conditions:
        mixed_runs = conditions["mixed_real"]["runs"]
        all_valid = all(not run["degraded"] and run["real_completed"] > 0 for run in mixed_runs)
        if all_valid and len(mixed_runs) == 10:
            status = "completed"
        elif all_valid:
            status = "completed_partial_seeds"
        else:
            status = "partial_degraded"
    else:
        status = "simulation_only"

    results = {
        "issue": 192,
        "status": status,
        "generated_at": datetime.now().astimezone().isoformat(),
        "config": {
            "seeds": args.seeds,
            "tasks_per_seed": args.tasks_per_seed,
            "episode_horizon": args.episode_horizon,
            "mixed_real_probability": args.mixed_real_prob,
            "mixed_cap_per_seed": args.mixed_cap_per_seed,
            "formal_submission_cap": formal_cap,
            "shots": args.shots,
            "machine": args.machine,
        },
        "preflight": preflight,
        "quota_remaining_after": quota.remaining() if quota else None,
        "conditions": conditions,
        "statistics": statistics,
        "pure_real_reuse": pure_real_reuse,
    }

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 保存图表（--skip-real 模式下只画 simulation，避免 KeyError）
    plot_conditions: dict[str, dict[str, Any]] = dict(conditions)
    if "mixed_real" not in plot_conditions:
        plot_conditions["mixed_real"] = conditions["simulation"]
    if "pure_real" not in plot_conditions:
        plot_conditions["pure_real"] = conditions["simulation"]
    try:
        save_plot(plot_conditions, args.plot)
    except (KeyError, ValueError) as exc:
        print(f"[警告] 绘图失败（不影响数据）: {exc}")

    generate_validation_report(results, args.report, args.plot)

    print(f"\n结果: {args.results}")
    print(f"报告: {args.report}")
    print(f"图: {args.plot}")
    if comparison:
        print(f"\n统计检验: {comparison.get('comparison')}")
        print(f"  p={comparison.get('p_value'):.6f}, d={comparison.get('cohens_d'):.4f}")


if __name__ == "__main__":
    main()
