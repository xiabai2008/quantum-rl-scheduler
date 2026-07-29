#!/usr/bin/env python
"""
天衍-287 多seed真机实验统计分析脚本（Issue #58 重构）

按 Issue #58 要求重构：
1. 统一统计方法：Welch t-test 作为主分析，不得根据正态性临时选择不同检验
2. 同 seed 设计另提供配对敏感性分析（paired t-test + Cohen's d_z）
3. 统一效应量和 CI
4. 统一 JSON 字段
5. 删除重复的 pairwise/full_stats 区块，保留一个权威统计结构
6. bonferroni_significant=false 时 judgment 必须为"不支持"
7. 旧混合机器/混合shots数据标记为 invalid_for_formal_comparison，不参与新的权威统计

输入 JSON 格式（由 tianyan287_multiseed.py 生成）：
    {
        "config": {"machine": "tianyan-287", "shots": 32, ...},
        "results": [
            {"strategy": "PPO", "seed": 42, "metrics": {"total_reward": 1560.86, ...}},
            ...
        ]
    }

用法：
    python scripts/evaluation/multiseed_real_machine_analysis.py \\
        --input results/real_machine/tianyan287_multiseed/multiseed_data_XXX.json \\
        --output results/reports/multiseed_real_machine_report.md
"""

import json
import sys
from pathlib import Path

import click
import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.stats_significance import (
    cohen_d,
    rank_biserial,
)

#: 参与比较的策略（顺序固定）
STRATEGIES = ["PPO", "SJF", "FCFS"]

#: 两两比较对（A vs B，H1: A > B）
COMPARISONS = [
    ("PPO", "FCFS"),
    ("PPO", "SJF"),
    ("SJF", "FCFS"),
]

#: Bonferroni 校正比较次数
N_COMPARISONS = len(COMPARISONS)
ALPHA_BONFERRONI = 0.05 / N_COMPARISONS  # 0.0167

#: 已核实：统一机器和 shots 配置
# 正确后端代码为 tianyan-287（有连字符），tianyan287 不存在
EXPECTED_MACHINE = "tianyan-287"
EXPECTED_SHOTS = 32


def _effect_level(abs_d: float) -> str:
    """Cohen's d 效应量等级。"""
    if abs_d < 0.2:
        return "可忽略"
    if abs_d < 0.5:
        return "小效应"
    if abs_d < 0.8:
        return "中效应"
    return "大效应"


def _welch_t_test(a: list[float], b: list[float]) -> dict:
    """Welch t 检验（主分析）+ 均值差 95% CI。

    Issue #58：统一使用 Welch t-test 作为主分析，不根据正态性临时切换。
    """
    # 防御：空输入应返回中性结果（不崩溃）
    if len(a) == 0 or len(b) == 0:
        return {
            "mean_diff": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "ci_95": [0.0, 0.0],
            "ci_crosses_zero": True,
            "df": 0.0,
            "se": 0.0,
        }

    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    res = sp_stats.ttest_ind(arr_a, arr_b, equal_var=False)
    t_stat = float(res.statistic)
    p_value = float(res.pvalue)
    mean_diff = float(arr_a.mean() - arr_b.mean())

    n1, n2 = len(arr_a), len(arr_b)
    var_a = float(arr_a.var(ddof=1)) if n1 > 1 else 0.0
    var_b = float(arr_b.var(ddof=1)) if n2 > 1 else 0.0
    se = float(np.sqrt(var_a / n1 + var_b / n2)) if (var_a + var_b) > 0 else 0.0

    # Welch-Satterthwaite 自由度
    num = (var_a / n1 + var_b / n2) ** 2
    den = (var_a / n1) ** 2 / max(n1 - 1, 1) + (var_b / n2) ** 2 / max(n2 - 1, 1)
    df = num / den if den > 0 else n1 + n2 - 2
    t_crit = float(sp_stats.t.ppf(0.975, df)) if df > 0 else 0.0
    margin = t_crit * se

    return {
        "mean_diff": mean_diff,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_95": [mean_diff - margin, mean_diff + margin],
        "ci_crosses_zero": (mean_diff - margin) <= 0 <= (mean_diff + margin),
        "df": df,
        "se": se,
    }


def _paired_t_test(a: list[float], b: list[float]) -> dict:
    """配对 t 检验（敏感性分析）+ Cohen's d_z。

    Issue #58：同 seed 设计提供配对敏感性分析。
    要求 a 和 b 长度相同且按 seed 对齐。
    """
    if len(a) != len(b) or len(a) < 2:
        return {
            "paired_available": False,
            "reason": f"配对样本不足: len(a)={len(a)}, len(b)={len(b)}",
        }

    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    diffs = arr_a - arr_b

    res = sp_stats.ttest_rel(arr_a, arr_b)
    t_stat = float(res.statistic)
    p_value = float(res.pvalue)
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    d_z = mean_diff / sd_diff if sd_diff > 0 else 0.0

    n = len(diffs)
    se = sd_diff / np.sqrt(n) if n > 0 else 0.0
    t_crit = float(sp_stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    margin = t_crit * se

    return {
        "paired_available": True,
        "mean_diff": mean_diff,
        "t_stat": t_stat,
        "p_value": p_value,
        "cohens_d_z": d_z,
        "effect_level": _effect_level(abs(d_z)),
        "ci_95": [mean_diff - margin, mean_diff + margin],
        "ci_crosses_zero": (mean_diff - margin) <= 0 <= (mean_diff + margin),
        "n_pairs": n,
    }


def _validate_data_provenance(data: dict) -> dict:
    """Issue #58：数据来源验证。

    检查机器/shots 是否混合，混合数据标记为 invalid_for_formal_comparison。
    """
    config = data.get("config", {})
    results = data.get("results", [])

    machines = set()
    shots_set = set()
    for r in results:
        if r.get("smoke_test"):
            continue  # 跳过冒烟记录
        m = r.get("machine", config.get("machine", ""))
        s = r.get("shots", config.get("shots"))
        if m:
            machines.add(m)
        if s is not None:
            shots_set.add(str(s))

    valid = (
        len(machines) == 1
        and EXPECTED_MACHINE in machines
        and len(shots_set) == 1
        and str(EXPECTED_SHOTS) in shots_set
    )

    return {
        "valid": valid,
        "machines_found": sorted(machines),
        "shots_found": sorted(shots_set),
        "expected_machine": EXPECTED_MACHINE,
        "expected_shots": EXPECTED_SHOTS,
        "invalid_for_formal_comparison": not valid,
        "reason": ("mixed_machines_or_shots" if not valid else "unified_protocol"),
    }


def analyze(data: dict) -> dict:
    """执行完整统计分析（Issue #58 统一口径）。

    Args:
        data: 从 JSON 加载的实验数据

    Returns:
        分析结果字典（单一权威统计结构）
    """
    # Issue #58：数据来源验证
    provenance = _validate_data_provenance(data)

    # 按策略分组（仅包含非冒烟记录）
    groups: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    seed_rewards: dict[str, dict[int, float]] = {s: {} for s in STRATEGIES}
    for r in data["results"]:
        if r.get("smoke_test"):
            continue
        strategy = r["strategy"]
        if strategy in groups and "total_reward" in r.get("metrics", {}):
            reward = float(r["metrics"]["total_reward"])
            groups[strategy].append(reward)
            seed = r.get("seed")
            if seed is not None:
                seed_rewards[strategy][seed] = reward

    # 描述性统计
    descriptive = {}
    for name in STRATEGIES:
        arr = np.array(groups[name], dtype=float)
        descriptive[name] = {
            "n": len(arr),
            "mean": float(arr.mean()) if len(arr) > 0 else 0.0,
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(arr)) if len(arr) > 0 else 0.0,
            "min": float(arr.min()) if len(arr) > 0 else 0.0,
            "max": float(arr.max()) if len(arr) > 0 else 0.0,
        }

    # Issue #58：统一两两比较（Welch t-test 主分析 + 配对敏感性）
    pairwise = {}
    for name_a, name_b in COMPARISONS:
        rewards_a = groups[name_a]
        rewards_b = groups[name_b]

        # 主分析：Welch t-test + Cohen's d
        welch = _welch_t_test(rewards_a, rewards_b)
        d = float(cohen_d(rewards_a, rewards_b))
        rb = float(rank_biserial(rewards_a, rewards_b))

        # 敏感性分析：配对 t-test（同 seed 设计）
        # 需要相同 seed 对齐
        paired_a = []
        paired_b = []
        common_seeds = set(seed_rewards[name_a].keys()) & set(seed_rewards[name_b].keys())
        for s in sorted(common_seeds):
            paired_a.append(seed_rewards[name_a][s])
            paired_b.append(seed_rewards[name_b][s])
        paired = _paired_t_test(paired_a, paired_b)

        bonf_significant = welch["p_value"] < ALPHA_BONFERRONI

        # Issue #58：bonferroni_significant=false 时 judgment 必须为"不支持"
        if bonf_significant and not welch["ci_crosses_zero"] and d >= 0.5:
            judgment = "支持"
        else:
            judgment = "不支持"

        pairwise[f"{name_a}_vs_{name_b}"] = {
            "mean_a": float(np.mean(rewards_a)) if rewards_a else 0.0,
            "mean_b": float(np.mean(rewards_b)) if rewards_b else 0.0,
            "mean_diff": welch["mean_diff"],
            # 主分析：Welch t-test
            "welch_t": welch["t_stat"],
            "welch_p": welch["p_value"],
            "welch_df": welch["df"],
            "welch_se": welch["se"],
            "mean_diff_ci_95": welch["ci_95"],
            "ci_crosses_zero": welch["ci_crosses_zero"],
            # 效应量
            "cohens_d": d,
            "effect_level": _effect_level(abs(d)),
            "rank_biserial": rb,
            # 敏感性分析：配对 t-test
            "paired_analysis": paired,
            # Bonferroni 校正
            "bonferroni_significant": bonf_significant,
            "alpha_bonferroni": ALPHA_BONFERRONI,
            # Issue #58：judgment 严格按 bonferroni_significant
            "judgment": judgment,
        }

    return {
        "config": data.get("config", {}),
        "total_elapsed_seconds": data.get("total_elapsed_seconds", 0),
        "data_provenance": provenance,
        "descriptive": descriptive,
        "pairwise": pairwise,
        # Issue #58：删除重复的 full_stats 区块，pairwise 即权威结构
        "alpha_bonferroni": ALPHA_BONFERRONI,
        "analysis_protocol": {
            "main_test": "Welch t-test",
            "sensitivity_test": "paired t-test + Cohen's d_z",
            "correction": "Bonferroni",
            "n_comparisons": N_COMPARISONS,
            "judgment_rule": "bonferroni_significant=false -> judgment='不支持'",
        },
    }


def generate_report(analysis: dict, data_file: Path) -> str:
    """生成 Markdown 报告（Issue #58 统一口径）。"""
    lines = []
    lines.append("# 天衍-287 多seed真机实验统计分析报告（Issue #58 统一口径）\n")
    lines.append(f"**数据文件**: `{data_file}`\n")
    lines.append(f"**实验时间**: {analysis['config'].get('timestamp', 'N/A')}\n")
    lines.append(
        f"**实验配置**: {len(analysis['config'].get('seeds', []))} seeds × "
        f"{len(analysis['config'].get('strategies', []))} 策略\n"
    )
    lines.append(f"**总耗时**: {analysis['total_elapsed_seconds']:.1f}s\n")
    lines.append(f"**Bonferroni 校正 α**: {analysis['alpha_bonferroni']:.4f} (3 次比较)\n")
    lines.append(
        f"**统计协议**: {analysis['analysis_protocol']['main_test']} (主) + "
        f"{analysis['analysis_protocol']['sensitivity_test']} (敏感性)\n"
    )
    lines.append(f"**判定规则**: {analysis['analysis_protocol']['judgment_rule']}\n")
    lines.append("\n---\n")

    # Issue #58：数据来源验证
    prov = analysis["data_provenance"]
    lines.append("## 0. 数据来源验证\n")
    lines.append(f"- **期望机器**: {prov['expected_machine']}\n")
    lines.append(f"- **期望 shots**: {prov['expected_shots']}\n")
    lines.append(f"- **实际机器**: {prov['machines_found']}\n")
    lines.append(f"- **实际 shots**: {prov['shots_found']}\n")
    if prov["valid"]:
        lines.append(f"- **判定**: ✅ 统一协议（{prov['reason']}）\n")
    else:
        lines.append(
            f"- **判定**: ❌ {prov['reason']}，"
            f"标记为 `invalid_for_formal_comparison`，不参与权威统计\n"
        )
    lines.append("")

    # 描述性统计
    lines.append("## 1. 描述性统计\n")
    lines.append("| 策略 | N | 均值 | 标准差 | 中位数 | min | max |")
    lines.append("|:--:|:--:|:--:|:--:|:--:|:--:|:--:|")
    for name in STRATEGIES:
        d = analysis["descriptive"][name]
        lines.append(
            f"| {name} | {d['n']} | {d['mean']:.2f} | {d['std']:.2f} | "
            f"{d['median']:.2f} | {d['min']:.2f} | {d['max']:.2f} |"
        )
    lines.append("")

    # 两两比较
    lines.append("## 2. 两两比较（Welch t-test 主分析）\n")
    lines.append(
        "> **Issue #58 统一规则**: Welch t-test 作为主分析，"
        "不根据正态性临时切换。同 seed 设计另提供配对敏感性分析。\n"
        "> **判定**: `bonferroni_significant=false` 时 judgment 必须为「不支持」。\n"
    )

    for name_a, name_b in COMPARISONS:
        key = f"{name_a}_vs_{name_b}"
        p = analysis["pairwise"][key]
        lines.append(f"### 2.{COMPARISONS.index((name_a, name_b)) + 1} {name_a} vs {name_b}\n")
        lines.append(
            f"- **均值差**: {p['mean_diff']:.2f} ({name_a}={p['mean_a']:.2f} vs {name_b}={p['mean_b']:.2f})"
        )
        lines.append(f"- **Cohen's d**: {p['cohens_d']:.4f}（{p['effect_level']}）")
        lines.append(f"- **rank-biserial**: {p['rank_biserial']:.4f}")
        ci_lo, ci_hi = p["mean_diff_ci_95"]
        lines.append(
            f"- **均值差 95% CI**: [{ci_lo:.2f}, {ci_hi:.2f}] "
            f"{'(跨0)' if p['ci_crosses_zero'] else '(不跨0)'}"
        )
        lines.append(
            f"- **Welch t 检验**: t={p['welch_t']:.4f}, p={p['welch_p']:.6f}, "
            f"df={p['welch_df']:.2f} "
            f"{'(Bonferroni显著)' if p['bonferroni_significant'] else '(Bonferroni不显著)'}"
        )
        # 配对敏感性
        paired = p.get("paired_analysis", {})
        if paired.get("paired_available"):
            lines.append(
                f"- **配对敏感性** (N={paired['n_pairs']}): "
                f"t={paired['t_stat']:.4f}, p={paired['p_value']:.6f}, "
                f"d_z={paired['cohens_d_z']:.4f}（{paired['effect_level']}）"
            )
            pci_lo, pci_hi = paired["ci_95"]
            lines.append(
                f"  - 配对均值差 95% CI: [{pci_lo:.2f}, {pci_hi:.2f}] "
                f"{'(跨0)' if paired['ci_crosses_zero'] else '(不跨0)'}"
            )
        else:
            lines.append(f"- **配对敏感性**: 不可用（{paired.get('reason', 'N/A')}）")
        lines.append(f"- **判定**: **{p['judgment']}**\n")

    # 汇总表
    lines.append("## 3. 汇总表\n")
    lines.append(
        "| 比较 | 均值差 | Cohen's d | 效应等级 | 95% CI | 跨0? | Welch p | Bonferroni | 判定 |"
    )
    lines.append("|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|")
    for name_a, name_b in COMPARISONS:
        key = f"{name_a}_vs_{name_b}"
        p = analysis["pairwise"][key]
        ci_lo, ci_hi = p["mean_diff_ci_95"]
        lines.append(
            f"| {name_a} vs {name_b} | {p['mean_diff']:.2f} | {p['cohens_d']:.4f} | "
            f"{p['effect_level']} | [{ci_lo:.2f}, {ci_hi:.2f}] | "
            f"{'是' if p['ci_crosses_zero'] else '否'} | "
            f"{p['welch_p']:.6f} | "
            f"{'显著' if p['bonferroni_significant'] else '不显著'} | "
            f"{p['judgment']} |"
        )
    lines.append("")

    # 结论
    lines.append("## 4. 结论\n")
    ppo_fcfs = analysis["pairwise"]["PPO_vs_FCFS"]
    ppo_sjf = analysis["pairwise"]["PPO_vs_SJF"]
    lines.append(
        f"1. **PPO vs FCFS**: Cohen's d={ppo_fcfs['cohens_d']:.2f}（{ppo_fcfs['effect_level']}），"
        f"Welch p={ppo_fcfs['welch_p']:.2e}, "
        f"Bonferroni校正后{'显著' if ppo_fcfs['bonferroni_significant'] else '不显著'}。"
        f"判定：**{ppo_fcfs['judgment']}**。\n"
    )
    lines.append(
        f"2. **PPO vs SJF**: Cohen's d={ppo_sjf['cohens_d']:.2f}（{ppo_sjf['effect_level']}），"
        f"Welch p={ppo_sjf['welch_p']:.2e}, "
        f"Bonferroni校正后{'显著' if ppo_sjf['bonferroni_significant'] else '不显著'}。"
        f"判定：**{ppo_sjf['judgment']}**。\n"
    )

    if not prov["valid"]:
        lines.append(
            "3. **数据来源警告**: 检测到混合机器/shots，"
            "本数据标记为 `invalid_for_formal_comparison`，不参与权威统计。\n"
        )

    return "\n".join(lines)


@click.command()
@click.option("--input", "input_file", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "output_file", required=True, type=click.Path(path_type=Path))
@click.option("--json-output", "json_output", type=click.Path(path_type=Path), default=None)
def main(input_file: Path, output_file: Path, json_output: Path | None) -> None:
    """执行多seed真机实验统计分析并生成报告（Issue #58 统一口径）。"""
    with input_file.open(encoding="utf-8") as f:
        data = json.load(f)

    analysis = analyze(data)
    report = generate_report(analysis, input_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report, encoding="utf-8")
    click.echo(f"报告已生成: {output_file}")

    if json_output:
        # 序列化分析结果（统一 JSON 字段）
        safe_analysis = json.loads(json.dumps(analysis, default=str))
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(safe_analysis, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        click.echo(f"JSON 结果已生成: {json_output}")

    # 打印关键结论
    click.echo("\n" + "=" * 60)
    click.echo("关键结论（Issue #58 统一口径）:")
    click.echo(
        f"  数据来源: {'✅ 统一协议' if analysis['data_provenance']['valid'] else '❌ 混合协议'}"
    )
    for name_a, name_b in COMPARISONS:
        key = f"{name_a}_vs_{name_b}"
        p = analysis["pairwise"][key]
        click.echo(
            f"  {name_a} vs {name_b}: d={p['cohens_d']:.2f}, "
            f"Welch p={p['welch_p']:.2e}, "
            f"Bonferroni={'显著' if p['bonferroni_significant'] else '不显著'}, "
            f"判定={p['judgment']}"
        )


if __name__ == "__main__":
    main()
