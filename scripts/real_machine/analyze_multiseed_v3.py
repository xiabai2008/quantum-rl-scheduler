#!/usr/bin/env python
"""
真机多seed扩样合并分析脚本（实验③：v2 N=10 → N=20）

用途：
    合并 v2 权威数据（results/real_machine/tianyan287_multiseed/multiseed_data_20260727_005558.json，
    10 seeds）与新扩样数据（tianyan287_multiseed.py --formal 新增 10 seeds 的 multiseed_data_<ts>.json），
    按 v2 报告口径（multiseed_real_machine_report_10seeds_v2.md §6.1-6.3）重新计算
    N=20 的均值/标准差/Cohen's d/95% CI/Welch t p/Bonferroni/配对敏感性，
    并执行预注册检查项（效应量异常预警、数据质量审计），输出 v3 报告。

用法：
    python scripts/real_machine/analyze_multiseed_v3.py \
        --primary results/real_machine/tianyan287_multiseed/multiseed_data_20260727_005558.json \
        --extension results/real_machine/tianyan287_multiseed/multiseed_data_<新增时间戳>.json \
        --output results/reports/multiseed_real_machine_report_20seeds_v3.md

退出码：0 = 输出报告；1 = 数据质量审计失败（mock/degraded/完成率/task_id 留档率不达标）

统计口径（与 v2 完全一致）：
    - 主分析：Welch t-test（PPO vs FCFS / PPO vs SJF / SJF vs FCFS）
    - 多重比较：Bonferroni（3 比较，α=0.0167）
    - 效应量：Cohen's d + 95% CI（ddof=1）
    - 配对敏感性：同 seed PPO vs FCFS 配对 t（消除 seed 间方差）
    - 指标：total_reward（与 v2 报告一致，96 步/episode，1 真机任务/run）
    - 预注册检查项：任一策略 d>3 时输出效应量异常分析提示（v2 曾 d=5.33）

数据质量门槛（任一不过则 exit 1）：
    - 每记录 mock=False、degraded=False
    - unified_protocol=True（config 字段）
    - real_tasks_completed 合计 = seeds×策略数（30 或 60）
    - task_id 留档率 100%（real_records 每项 task_id 非空）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ALPHA = 0.05
N_COMPARISONS = 3
BONFERRONI_ALPHA = ALPHA / N_COMPARISONS
STRATEGIES = ["PPO", "FCFS", "SJF"]


def load_data(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def audit_data(data: dict[str, Any], label: str) -> list[str]:
    """数据质量审计（预注册门槛）。"""
    issues: list[str] = []
    cfg = data.get("config", {})
    if not cfg.get("unified_protocol"):
        issues.append(f"[{label}] unified_protocol != true")
    if cfg.get("machine") != "tianyan-287":
        issues.append(f"[{label}] machine={cfg.get('machine')} ≠ tianyan-287")
    if cfg.get("shots") != 32:
        issues.append(f"[{label}] shots={cfg.get('shots')} ≠ 32")
    results = [r for r in data.get("results", []) if not r.get("smoke_test")]
    completed = 0
    total_real = 0
    id_kept = 0
    for r in results:
        if r.get("mock"):
            issues.append(f"[{label}] seed={r.get('seed')} {r.get('strategy')} mock=True")
        if r.get("degraded"):
            issues.append(f"[{label}] seed={r.get('seed')} {r.get('strategy')} degraded=True")
        m = r.get("metrics", {})
        completed += int(m.get("real_tasks_completed", 0) or 0)
        for rec in r.get("real_records", []):
            total_real += 1
            if rec.get("task_id"):
                id_kept += 1
    if completed != len(results):
        issues.append(f"[{label}] real_tasks_completed={completed} ≠ 记录数 {len(results)}")
    if total_real and id_kept != total_real:
        issues.append(f"[{label}] task_id 留档率 {id_kept}/{total_real} < 100%")
    return issues


def compute_stats(
    rewards: dict[str, list[float]], seed_map: dict[str, list[int]]
) -> dict[str, Any]:
    """按 v2 报告口径计算统计量。"""
    out: dict[str, Any] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    ns: dict[str, int] = {}
    for s in STRATEGIES:
        arr = np.asarray(rewards[s], dtype=float)
        ns[s] = len(arr)
        means[s] = float(arr.mean())
        stds[s] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        out[s] = {
            "n": ns[s],
            "mean": round(means[s], 2),
            "std": round(stds[s], 2),
            "min": round(float(arr.min()), 2),
            "max": round(float(arr.max()), 2),
        }

    pairs = [("PPO", "FCFS"), ("PPO", "SJF"), ("SJF", "FCFS")]
    pair_stats: dict[str, Any] = {}
    for a, b in pairs:
        x = np.asarray(rewards[a], dtype=float)
        y = np.asarray(rewards[b], dtype=float)
        t = stats.ttest_ind(x, y, equal_var=False)
        # Cohen's d（pooled, ddof=1，与 v2 报告一致）
        n1, n2 = len(x), len(y)
        v1, v2 = x.var(ddof=1), y.var(ddof=1)
        pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        d = float((x.mean() - y.mean()) / pooled) if pooled > 0 else float("nan")
        # Welch CI（均值差）
        se = np.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        tc = stats.t.ppf(1 - ALPHA / 2, df)
        md = float(x.mean() - y.mean())
        pair_stats[f"{a} vs {b}"] = {
            "mean_diff": round(float(md), 2),
            "ci_95": [round(float(md - tc * se), 2), round(float(md + tc * se), 2)],
            "welch_t": round(float(t.statistic), 4),
            "p_value": float(t.pvalue),
            "cohens_d": round(float(d), 2),
            "bonferroni_significant": bool(t.pvalue < BONFERRONI_ALPHA),
            "effect_level": "大效应" if abs(d) >= 0.8 else "中效应" if abs(d) >= 0.5 else "小效应",
        }
    out["pairwise"] = pair_stats

    # 配对敏感性分析（同 seed PPO vs FCFS，v2 口径）：按 seed 字典对齐，不依赖列表顺序
    seeds_common = sorted(
        set(seed_map.get("PPO", [])) & set(seed_map.get("FCFS", [])) & set(seed_map.get("SJF", []))
    )
    if seeds_common:
        seed_reward: dict[str, dict[int, float]] = {
            s: {seed: rewards[s][i] for i, seed in enumerate(seed_map[s])} for s in STRATEGIES
        }
        ppo_seed = np.asarray([seed_reward["PPO"][s] for s in seeds_common], dtype=float)
        fcfs_seed = np.asarray([seed_reward["FCFS"][s] for s in seeds_common], dtype=float)
        pt = stats.ttest_rel(ppo_seed, fcfs_seed)
        out["paired_sensitivity"] = {
            "n_pairs": len(seeds_common),
            "paired_t": round(float(pt.statistic), 4),
            "p_value": float(pt.pvalue),
            "significant": bool(pt.pvalue < ALPHA),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="真机多seed扩样合并分析（实验③ v3）")
    parser.add_argument("--primary", required=True, type=Path, help="v2 权威 JSON（10 seeds）")
    parser.add_argument(
        "--extension", required=True, type=Path, help="新增扩样 JSON（新 10 seeds）"
    )
    parser.add_argument("--output", required=True, type=Path, help="v3 报告 md 输出路径")
    args = parser.parse_args()

    issues: list[str] = []
    primary = load_data(args.primary)
    extension = load_data(args.extension)
    issues += audit_data(primary, "v2-primary")
    issues += audit_data(extension, "extension")
    if issues:
        for i in issues:
            print(f"  [数据质量] {i}")
        print("❌ 数据质量审计未通过（见上），终止分析。")
        return 1

    # 合并（按 strategy 聚合 total_reward；记录 seed 配对信息）
    rewards: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    seed_map: dict[str, list[int]] = {s: [] for s in STRATEGIES}
    for data in (primary, extension):
        for r in data.get("results", []):
            if r.get("smoke_test"):
                continue
            name = str(r.get("strategy", "")).upper()
            m = r.get("metrics", {})
            if name in STRATEGIES and "total_reward" in m:
                rewards[name].append(float(m["total_reward"]))
                seed_map[name].append(int(r.get("seed", -1)))

    sizes = {s: len(rewards[s]) for s in STRATEGIES}
    if not (sizes["PPO"] == sizes["FCFS"] == sizes["SJF"] == 20):
        print(f"  [错误] 合并后样本数 {sizes}（期望每策略 20），请检查 JSON。")
        return 1

    stats_out = compute_stats(rewards, seed_map)

    # 预注册检查项：效应量异常预警（v2 曾 d=5.33 过大）
    warnings: list[str] = []
    for pair, ps in stats_out["pairwise"].items():
        if abs(ps["cohens_d"]) > 3.0:
            warnings.append(
                f"  ⚠ {pair}: d={ps['cohens_d']} 异常大（预注册检查项）——"
                "需在报告中补充效应量异常分析（如 reward 分布双峰/真机任务占比 1/96 机制说明）"
            )

    # 生成 v3 报告
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("# 多seed真机实验 v3（N=20 扩样权威版）")
    lines.append("")
    lines.append("> **生成时间**: " + now)
    lines.append(
        f"> **数据源**: `{args.primary.name}`（v2 权威 10 seeds）+ `{args.extension.name}`（扩样 10 seeds）"
    )
    lines.append(
        "> **协议**: unified_protocol=true, tianyan-287, shots=32, `H Q1/M Q1`, 96 步/episode, 1 真机任务/run"
    )
    lines.append(
        "> **统计方法**: Welch t 主分析 + Bonferroni（3 比较 α=0.0167）+ Cohen's d + 95% CI + 同 seed 配对敏感性"
    )
    lines.append(
        "> **口径说明**: 真机任务成功完成（N=20×3），但真机 reward 占总 reward 比例仍低（1 真机步/96 步），策略间差异仍主要由仿真动力学驱动（与 v2 相同边界，诚实披露）"
    )
    lines.append("")
    lines.append("## 一、策略汇总（N=20）")
    lines.append("")
    lines.append("| 策略 | N | 均值 | 标准差 | min | max |")
    lines.append("|:--|:--:|:--:|:--:|:--:|:--:|")
    for s in STRATEGIES:
        v = stats_out[s]
        lines.append(f"| {s} | {v['n']} | {v['mean']} | {v['std']} | {v['min']} | {v['max']} |")
    lines.append("")
    lines.append("## 二、两两对比（Welch t + Bonferroni）")
    lines.append("")
    lines.append("| 对比 | 均值差 | 95% CI | t | p 值 | Cohen's d | 效应 | Bonferroni(α=0.0167) |")
    lines.append("|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|")
    for pair, ps in stats_out["pairwise"].items():
        lines.append(
            f"| {pair} | {ps['mean_diff']} | {ps['ci_95']} | {ps['welch_t']} | "
            f"{ps['p_value']:.3g} | {ps['cohens_d']} | {ps['effect_level']} | "
            f"{'显著' if ps['bonferroni_significant'] else '不显著'} |"
        )
    lines.append("")
    if "paired_sensitivity" in stats_out:
        ps2 = stats_out["paired_sensitivity"]
        lines.append("## 三、配对敏感性分析（同 seed PPO vs FCFS）")
        lines.append("")
        lines.append(
            f"- N={ps2['n_pairs']} 对，配对 t={ps2['paired_t']}，p={ps2['p_value']:.4g}，"
            f"{'显著' if ps2['significant'] else '不显著'}"
        )
        lines.append("")
    lines.append("## 四、预注册检查项")
    lines.append("")
    if warnings:
        lines.append("- " + "\n- ".join(warnings))
    else:
        lines.append("- 无效应量异常预警（|d|≤3）")
    lines.append("- 数据质量审计通过：mock=false / degraded=false / task_id 100% 留档")
    lines.append("")
    lines.append("## 五、结论（按预注册条款）")
    lines.append("")
    lines.append("- 无论显著性如何，本报告如实呈现；负向/不显著结果不删除（预注册诚实披露条款）")
    lines.append(
        "- 若 N=20 下效应量较 v2（d=5.33）明显收缩，属小样本回归正常化，需在答辩中主动说明"
    )
    lines.append("")
    lines.append(f"*v3 报告自动生成 | {now}*")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ v3 报告已生成: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
