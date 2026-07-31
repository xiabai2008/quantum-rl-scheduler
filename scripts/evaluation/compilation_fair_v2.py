"""
Issue #451: 编译层公平对比重做（同池配对+固定种子+逐电路明细+显著性）

修复原 `compilation_full.py` 的不公平对比设计：
- 原 SABRE 评测 60 电路含深电路，PPO 仅评测 20 个浅电路，非配对、分布不同
- 全程 `np.random` 未播种，不可复现
- 产出仅 4 字段，无逐电路明细、无 std、无显著性

本脚本实现公平对比：
1. 固定种子（seed=42）生成 60 电路统一池（3 难度 × 20，含深电路）
2. SABRE 与 PPO 在 **同一电路池** 配对评测
3. 逐电路明细入库（qubits/depth/双方 SWAP 数）
4. Wilcoxon 符号秩检验 + 效应量 + 95% CI
5. 生成 `compilation_fair_v2_report.md` 与逐电路 JSON

产出:
    - results/compilation/fair_v2_per_circuit.json
    - results/compilation/fair_v2_summary.json
    - results/reports/compilation_fair_v2_report.md
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(str(_PROJECT_ROOT))

# 修复 Windows GBK 终端下 emoji 字符导致的 UnicodeEncodeError 崩溃
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from qiskit.circuit.random import random_circuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreLayout, SabreSwap
from stable_baselines3 import PPO

from src.quantum.compilation_env import QuantumCompilationEnv

# ---------------------------------------------------------------------------
# 配置：固定种子 + 统一电路池
# ---------------------------------------------------------------------------

SEED = 42
N_PER_CATEGORY = 20
CATEGORIES: dict[str, dict[str, Any]] = {
    "shallow": {"qubits": (5, 8), "gates": (5, 10), "label": "浅电路"},
    "medium": {"qubits": (9, 14), "gates": (10, 20), "label": "中电路"},
    "deep": {"qubits": (14, 16), "gates": (20, 30), "label": "深电路"},
}

# 4×4 2D网格拓扑（与 compilation_env.py COUPLING_GRAPH 一致，匹配天衍真机 nearest-neighbor 结构）
_GRID_ROWS, _GRID_COLS = 4, 4
_grid_edges: list[tuple[int, int]] = []
for _r in range(_GRID_ROWS):
    for _c in range(_GRID_COLS):
        _q = _r * _GRID_COLS + _c
        if _c + 1 < _GRID_COLS:
            _grid_edges.append((_q, _q + 1))
            _grid_edges.append((_q + 1, _q))
        if _r + 1 < _GRID_ROWS:
            _grid_edges.append((_q, _q + _GRID_COLS))
            _grid_edges.append((_q + _GRID_COLS, _q))
COUPLING = CouplingMap(_grid_edges)

MODEL_PATH = "deliverable_models/ppo_compilation_agent.zip"


# ---------------------------------------------------------------------------
# 1. 固定种子生成统一电路池
# ---------------------------------------------------------------------------


def generate_circuit_pool(seed: int = SEED) -> list[dict[str, Any]]:
    """生成 60 个统一电路池（3 难度 × 20），固定种子可复现。

    Returns:
        list of {"circuit": QuantumCircuit, "category": str, "qubits": int, "depth": int,
                 "gates": int, "index": int}
    """
    rng = np.random.default_rng(seed)
    pool: list[dict[str, Any]] = []
    for cat_name, cfg in CATEGORIES.items():
        for _i in range(N_PER_CATEGORY):
            n_qubits = int(rng.integers(cfg["qubits"][0], cfg["qubits"][1] + 1))
            n_gates = int(rng.integers(cfg["gates"][0], cfg["gates"][1] + 1))
            qc = random_circuit(
                n_qubits, n_gates, measure=False, seed=int(rng.integers(0, 2**31 - 1))
            )
            pool.append(
                {
                    "circuit": qc,
                    "category": cat_name,
                    "qubits": n_qubits,
                    "depth": qc.depth(),
                    "gates": qc.size(),
                    "index": len(pool),
                }
            )
    return pool


# ---------------------------------------------------------------------------
# 2. SABRE Baseline 评测
# ---------------------------------------------------------------------------


def evaluate_sabre(pool: list[dict[str, Any]], seed: int = SEED) -> list[dict[str, Any]]:
    """对统一电路池执行 SABRE 编译，返回逐电路 SWAP 数。

    SabreLayout/SabreSwap 均传入固定 seed，消除批间 p 值漂移（未播种时
    启发式搜索随机源不固定，同 seed 多次运行 p 值会在 0.26~0.87 间漂移）。
    """
    results: list[dict[str, Any]] = []
    for item in pool:
        pm = PassManager(
            [
                SabreLayout(COUPLING, swap_trials=8, layout_trials=8, seed=seed),
                SabreSwap(COUPLING, trials=8, seed=seed),
            ]
        )
        compiled = pm.run(item["circuit"])
        swap_count = compiled.count_ops().get("swap", 0)
        results.append(
            {
                "index": item["index"],
                "category": item["category"],
                "qubits": item["qubits"],
                "depth": item["depth"],
                "gates": item["gates"],
                "sabre_swap": int(swap_count),
            }
        )
    return results


# ---------------------------------------------------------------------------
# 3. PPO 评测（同池配对）
# ---------------------------------------------------------------------------


def evaluate_ppo(
    pool: list[dict[str, Any]], model_path: str = MODEL_PATH, max_steps: int = 200
) -> list[dict[str, Any]]:
    """对统一电路池执行 PPO 编译，返回逐电路 SWAP 数。

    使用 deterministic=True 保证可复现。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PPO 模型不存在: {model_path}")
    model = PPO.load(model_path)
    results: list[dict[str, Any]] = []
    for item in pool:
        env = QuantumCompilationEnv(item["circuit"], max_steps=max_steps)
        obs, _ = env.reset(seed=SEED)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break
        stats_dict = env.get_stats()
        results.append(
            {
                "index": item["index"],
                "category": item["category"],
                "qubits": item["qubits"],
                "depth": item["depth"],
                "gates": item["gates"],
                "ppo_swap": int(stats_dict["swap_count"]),
            }
        )
    return results


# ---------------------------------------------------------------------------
# 4. 统计分析：Wilcoxon 符号秩检验 + 效应量 + 95% CI
# ---------------------------------------------------------------------------


def compute_statistics(sabre_swaps: list[int], ppo_swaps: list[int]) -> dict[str, Any]:
    """计算配对 Wilcoxon 符号秩检验 + 效应量 + 95% CI。

    Returns:
        dict with all statistics
    """
    sabre_arr = np.array(sabre_swaps, dtype=np.float64)
    ppo_arr = np.array(ppo_swaps, dtype=np.float64)
    n_pairs = len(sabre_arr)

    # 均值与标准差
    sabre_mean = float(np.mean(sabre_arr))
    ppo_mean = float(np.mean(ppo_arr))
    sabre_std = float(np.std(sabre_arr, ddof=1))
    ppo_std = float(np.std(ppo_arr, ddof=1))

    # 提升百分比（基于均值）
    improvement_pct = (1.0 - ppo_mean / max(1.0, sabre_mean)) * 100.0

    # 配对 Wilcoxon signed-rank 检验（单侧：PPO < SABRE）
    diffs = sabre_arr - ppo_arr
    nonzero_diffs = diffs[diffs != 0]
    if len(nonzero_diffs) >= 1:
        wilcox_stat, p_value = stats.wilcoxon(
            ppo_arr, sabre_arr, alternative="less", zero_method="wilcox"
        )
        wilcox_stat = float(wilcox_stat)
        p_value = float(p_value)
    else:
        wilcox_stat = 0.0
        p_value = 1.0

    # 效应量：配对 rank-biserial correlation (r = Z / sqrt(N))
    # Wilcoxon 单侧：r = 1 - 2W / (N*(N+1)/2)，正值表示 PPO 更优
    n_nonzero = len(nonzero_diffs)
    if n_nonzero > 0:
        rank_biserial = 1.0 - 2.0 * wilcox_stat / (n_nonzero * (n_nonzero + 1) / 2.0)
    else:
        rank_biserial = 0.0

    # 逐电路改进率（基于差值）
    per_circuit_improvement = [
        (s - p) / max(1.0, s) * 100.0 for s, p in zip(sabre_arr, ppo_arr, strict=False)
    ]
    ci_low = float(np.percentile(per_circuit_improvement, 2.5))
    ci_high = float(np.percentile(per_circuit_improvement, 97.5))

    # Bootstrap 95% CI for mean improvement percentage
    rng = np.random.default_rng(SEED)
    bootstrap_means: list[float] = []
    for _ in range(2000):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        sample_sabre = sabre_arr[idx]
        sample_ppo = ppo_arr[idx]
        sample_imp = (1.0 - np.mean(sample_ppo) / max(1.0, np.mean(sample_sabre))) * 100.0
        bootstrap_means.append(float(sample_imp))
    boot_ci_low = float(np.percentile(bootstrap_means, 2.5))
    boot_ci_high = float(np.percentile(bootstrap_means, 97.5))

    # Cohen's d for paired samples
    cohen_d = float(np.mean(diffs) / np.std(diffs, ddof=1)) if len(nonzero_diffs) > 1 else 0.0

    return {
        "n_pairs": int(n_pairs),
        "sabre_mean": sabre_mean,
        "ppo_mean": ppo_mean,
        "sabre_std": sabre_std,
        "ppo_std": ppo_std,
        "improvement_pct": float(improvement_pct),
        "wilcoxon_w": wilcox_stat,
        "p_value": p_value,
        "rank_biserial": float(rank_biserial),
        "cohen_d": cohen_d,
        "per_circuit_improvement_mean": float(np.mean(per_circuit_improvement)),
        "per_circuit_improvement_median": float(np.median(per_circuit_improvement)),
        "per_circuit_improvement_ci_low": ci_low,
        "per_circuit_improvement_ci_high": ci_high,
        "bootstrap_ci_low": boot_ci_low,
        "bootstrap_ci_high": boot_ci_high,
        "significant": bool(p_value < 0.05),
    }


def compute_category_breakdown(
    per_circuit: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """分类别统计 SABRE/PPO SWAP 数与改进率。"""
    breakdown: dict[str, dict[str, float]] = {}
    for cat_name in CATEGORIES:
        cat_items = [item for item in per_circuit if item["category"] == cat_name]
        if not cat_items:
            continue
        sabre_vals = [item["sabre_swap"] for item in cat_items]
        ppo_vals = [item["ppo_swap"] for item in cat_items]
        sabre_mean = float(np.mean(sabre_vals))
        ppo_mean = float(np.mean(ppo_vals))
        improvement = (1.0 - ppo_mean / max(1.0, sabre_mean)) * 100.0
        breakdown[cat_name] = {
            "n": len(cat_items),
            "sabre_mean": sabre_mean,
            "sabre_std": float(np.std(sabre_vals, ddof=1)) if len(sabre_vals) > 1 else 0.0,
            "ppo_mean": ppo_mean,
            "ppo_std": float(np.std(ppo_vals, ddof=1)) if len(ppo_vals) > 1 else 0.0,
            "improvement_pct": float(improvement),
        }
    return breakdown


def compute_subset_analysis(per_circuit: list[dict[str, Any]]) -> dict[str, Any]:
    """计算中电路+深电路子集（n=40）的统计分析。

    科学依据：浅电路（5-8 比特）SABRE 几乎不需 SWAP，PPO 反而引入额外 SWAP，
    属于 PPO 不适用的场景。中电路+深电路子集反映 PPO 在有意义场景下的真实性能。
    """
    subset = [item for item in per_circuit if item["category"] in ("medium", "deep")]
    sabre_swaps = [item["sabre_swap"] for item in subset]
    ppo_swaps = [item["ppo_swap"] for item in subset]
    stats_dict = compute_statistics(sabre_swaps, ppo_swaps)
    return {
        "subset": "medium+deep",
        "n_pairs": len(subset),
        "stats": stats_dict,
        "rationale": (
            "浅电路（5-8 比特）SABRE 几乎不需 SWAP，PPO 反而引入额外 SWAP；"
            "中电路+深电路子集反映 PPO 在有意义场景下的真实性能。"
        ),
    }


# ---------------------------------------------------------------------------
# 5. 生成报告
# ---------------------------------------------------------------------------


def generate_report(
    stats_summary: dict[str, Any],
    breakdown: dict[str, dict[str, float]],
    subset_analysis: dict[str, Any],
    per_circuit: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    """生成 Markdown 报告。"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    subset_stats = subset_analysis["stats"]

    ppo_better_overall = stats_summary["improvement_pct"] > 0
    ppo_better_subset = subset_stats["improvement_pct"] > 0

    # PPO 劣势幅度（正值表示 PPO 比 SABRE 高出多少 %）
    ppo_worse_pct_overall = max(0.0, -stats_summary["improvement_pct"])
    ppo_worse_pct_subset = max(0.0, -subset_stats["improvement_pct"])

    if ppo_better_overall and stats_summary["significant"]:
        sig_text = f"p = {stats_summary['p_value']:.2e} < 0.05，PPO 的 SWAP 数显著低于 SABRE，差异具有统计显著性。"
    elif ppo_better_overall:
        sig_text = f"p = {stats_summary['p_value']:.2e}，未达到 0.05 显著性水平，PPO 平均 SWAP 数较低但差异不显著。"
    elif stats_summary["p_value"] >= 0.999:
        sig_text = (
            f"p = {stats_summary['p_value']:.2e} ≈ 1.0，PPO 不优于 SABRE（单侧检验完全反向），"
            f"PPO 平均 SWAP 数高于 SABRE {ppo_worse_pct_overall:.1f}%。"
        )
    else:
        sig_text = (
            f"p = {stats_summary['p_value']:.2e}，PPO 平均 SWAP 数高于 SABRE {ppo_worse_pct_overall:.1f}%，"
            f"PPO 不优于 SABRE。"
        )

    if ppo_better_subset and subset_stats["significant"]:
        subset_sig_text = (
            f"p = {subset_stats['p_value']:.2e} < 0.05，PPO 在中电路+深电路子集上显著优于 SABRE。"
        )
    elif ppo_better_subset:
        subset_sig_text = f"p = {subset_stats['p_value']:.2e}，子集检验未达到 0.05 显著性水平，PPO 平均较低但差异不显著。"
    elif subset_stats["p_value"] >= 0.999:
        subset_sig_text = (
            f"p = {subset_stats['p_value']:.2e} ≈ 1.0，PPO 在中电路+深电路子集上不优于 SABRE（完全反向），"
            f"PPO SWAP 数高于 SABRE {ppo_worse_pct_subset:.1f}%。"
        )
    else:
        subset_sig_text = (
            f"p = {subset_stats['p_value']:.2e}，PPO 在中电路+深电路子集上 SWAP 数高于 SABRE "
            f"{ppo_worse_pct_subset:.1f}%，PPO 不优于 SABRE。"
        )

    # 数据驱动的统计结论文字（避免数据与结论矛盾）
    deep_improvement = breakdown.get("deep", {}).get("improvement_pct", 0.0)
    if deep_improvement > 0:
        deep_conclusion_text = f"深电路类别改进{deep_improvement:.1f}%，提示PPO在复杂电路上有优势"
    else:
        deep_conclusion_text = (
            f"深电路类别PPO SWAP数高{abs(deep_improvement):.1f}%，PPO在复杂电路上不优于SABRE"
        )

    if ppo_better_overall and stats_summary["significant"]:
        stats_conclusion = (
            f"PPO平均SWAP数显著低于SABRE(p<0.05)，{deep_conclusion_text}。"
        )
    elif ppo_better_overall:
        stats_conclusion = (
            f"PPO平均SWAP数低于SABRE但差异未达统计显著性(p>0.05)，{deep_conclusion_text}，"
            f"提示需更大规模训练验证。"
        )
    else:
        stats_conclusion = (
            f"PPO平均SWAP数高于SABRE {ppo_worse_pct_overall:.1f}%（p={stats_summary['p_value']:.2e}），"
            f"PPO不优于SABRE；{deep_conclusion_text}，"
            f"提示模型与观测空间可能不匹配，需重新训练或对齐观测维度。"
        )

    # 显著性汇总（避免硬编码"均未显著"）
    both_insignificant = not stats_summary["significant"] and not subset_stats["significant"]
    sig_summary = (
        "差异均未达到统计显著性阈值(α=0.05)，提示需进一步训练或优化奖励函数。"
        if both_insignificant
        else "详见上方显著性检验结果。"
    )

    # 类别表格行
    cat_rows = []
    for cat_name, cfg in CATEGORIES.items():
        if cat_name in breakdown:
            b = breakdown[cat_name]
            cat_rows.append(
                f"| {cfg['label']} ({b['n']} 电路) | {b['sabre_mean']:.2f} ± {b['sabre_std']:.2f} | "
                f"{b['ppo_mean']:.2f} ± {b['ppo_std']:.2f} | {b['improvement_pct']:+.1f}% |"
            )

    # 前 10 个电路明细
    detail_rows = []
    for item in per_circuit[:10]:
        diff = item["sabre_swap"] - item["ppo_swap"]
        detail_rows.append(
            f"| {item['index']} | {CATEGORIES[item['category']]['label']} | {item['qubits']} | "
            f"{item['depth']} | {item['gates']} | {item['sabre_swap']} | {item['ppo_swap']} | "
            f"{diff:+d} |"
        )

    return f"""# 编译层 PPO SWAP 公平对比报告 v2

**生成时间**: {ts}
**Issue**: #451 — 修复 `compilation_full.py` 不公平对比设计
**模型**: `{config["model_path"]}` (PPO, 200k timesteps, 4×4 2D网格+全电路分布训练)
**种子**: {config["seed"]}（电路池可复现）

---

## 一、公平对比设计

### 原对比问题（Issue #451 实锤）

`scripts/evaluation/compilation_full.py` 的 76.4% 对比设计不公平：
- SABRE 评测 60 电路含深电路类（dp: 14-16 比特/20-30 门）
- PPO 仅评测 20 个浅电路（5-13 比特/5-21 门），非配对、分布不同
- 全程 `np.random` 未播种，不可复现
- 产出 JSON 仅 4 字段，无逐电路明细、无 std、无显著性

### 本实验修复

| 修复项 | 实现 |
|:--|:--|
| 固定种子 | `np.random.default_rng({config["seed"]})` 生成电路池 |
| 同池配对 | SABRE 与 PPO 在 **同一 60 电路池** 评测 |
| 逐电路明细 | 入库 `results/compilation/fair_v2_per_circuit.json`（60 条） |
| 显著性检验 | Wilcoxon signed-rank 单侧检验 + rank-biserial + 95% CI |
| Bootstrap CI | 2000 次重采样估计提升百分比 95% CI |

### 实验配置

| 参数 | 值 |
|:--|:--|
| 物理拓扑 | 4×4 2D网格拓扑（匹配天衍真机 nearest-neighbor 结构，与 compilation_env.py COUPLING_GRAPH 一致） |
| 评估电路数 | 60 (3 类别 × 20 电路) |
| 电路类别 | 浅 (5-8 qubits, 5-10 gates)、中 (9-14 qubits, 10-20 gates)、深 (14-16 qubits, 20-30 gates) |
| SABRE 配置 | swap_trials=8, layout_trials=8 |
| PPO 配置 | MlpPolicy, lr=3e-4, n_steps=2048, batch_size=64, 200k timesteps（4×4 2D网格，全电路分布训练） |
| 评估模式 | deterministic=True |
| 种子 | {config["seed"]} |

---

## 二、总体对比

| 指标 | SABRE | PPO | 变化 |
|:--|:--|:--|:--|
| 平均 SWAP 数 | {stats_summary["sabre_mean"]:.2f} | {stats_summary["ppo_mean"]:.2f} | **{stats_summary["improvement_pct"]:+.1f}%** |
| 标准差 | {stats_summary["sabre_std"]:.2f} | {stats_summary["ppo_std"]:.2f} | — |
| 样本数 | {stats_summary["n_pairs"]} | {stats_summary["n_pairs"]} | — |

---

## 三、分类别对比

| 类别 | SABRE avg ± std | PPO avg ± std | 改进率 |
|:--|:--:|:--:|:--:|
{chr(10).join(cat_rows)}

---

## 四、统计显著性检验

### Wilcoxon signed-rank 检验（配对设计，同一电路池）

| 统计量 | 值 |
|:--|:--|
| W 统计量 | {stats_summary["wilcoxon_w"]:.0f} |
| p 值（单侧 PPO < SABRE） | {stats_summary["p_value"]:.2e} |
| 配对数 | {stats_summary["n_pairs"]} |
| rank-biserial 效应量 | {stats_summary["rank_biserial"]:.3f} |
| Cohen's d（配对） | {stats_summary["cohen_d"]:.3f} |
| 显著性 (α=0.05) | {"[PASS] 显著" if stats_summary["significant"] else "[FAIL] 不显著"} |

**结论**: {sig_text}

### 提升百分比 95% CI

| 方法 | CI |
|:--|:--|
| Bootstrap (2000 次) | [{stats_summary["bootstrap_ci_low"]:.1f}%, {stats_summary["bootstrap_ci_high"]:.1f}%] |
| 逐电路改进率百分位 | [{stats_summary["per_circuit_improvement_ci_low"]:.1f}%, {stats_summary["per_circuit_improvement_ci_high"]:.1f}%] |
| 逐电路改进率均值 | {stats_summary["per_circuit_improvement_mean"]:.1f}% |
| 逐电路改进率中位数 | {stats_summary["per_circuit_improvement_median"]:.1f}% |

---

## 五、子集分析（中电路+深电路，n={subset_analysis["n_pairs"]}）

### 科学依据

{subset_analysis["rationale"]}

### 子集统计

| 指标 | SABRE | PPO | 变化 |
|:--|:--|:--|:--|
| 平均 SWAP 数 | {subset_stats["sabre_mean"]:.2f} | {subset_stats["ppo_mean"]:.2f} | **{subset_stats["improvement_pct"]:+.1f}%** |
| 标准差 | {subset_stats["sabre_std"]:.2f} | {subset_stats["ppo_std"]:.2f} | — |
| 样本数 | {subset_stats["n_pairs"]} | {subset_stats["n_pairs"]} | — |

### 子集显著性检验

| 统计量 | 值 |
|:--|:--|
| Wilcoxon W | {subset_stats["wilcoxon_w"]:.0f} |
| p 值（单侧 PPO < SABRE） | {subset_stats["p_value"]:.2e} |
| rank-biserial 效应量 | {subset_stats["rank_biserial"]:.3f} |
| Cohen's d（配对） | {subset_stats["cohen_d"]:.3f} |
| Bootstrap 95% CI | [{subset_stats["bootstrap_ci_low"]:.1f}%, {subset_stats["bootstrap_ci_high"]:.1f}%] |
| 显著性 (α=0.05) | {"[PASS] 显著" if subset_stats["significant"] else "[FAIL] 不显著"} |

**子集结论**: {subset_sig_text}

---

## 六、逐电路明细（前 10 条，完整数据见 JSON）

| # | 类别 | 量子比特 | 深度 | 门数 | SABRE SWAP | PPO SWAP | 差值 (S-P) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
{chr(10).join(detail_rows)}

完整逐电路数据：`results/compilation/fair_v2_per_circuit.json`

---

## 七、与原 76.4% 数字对比

| 来源 | SABRE avg | PPO avg | 改进率 | 配对设计 | 固定种子 | 显著性 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| 原 `compilation_full.py` (Issue #451) | 27.6 | 6.5 | -76.4% | [FAIL] 非配对 (60 vs 20) | [FAIL] | 未检验 |
| 本公平对比 v2（全 60 电路） | {stats_summary["sabre_mean"]:.2f} | {stats_summary["ppo_mean"]:.2f} | {stats_summary["improvement_pct"]:+.1f}% | [PASS] 同池配对 | [PASS] seed={config["seed"]} | p={stats_summary["p_value"]:.2e} |
| 本公平对比 v2（中+深 40 电路） | {subset_stats["sabre_mean"]:.2f} | {subset_stats["ppo_mean"]:.2f} | {subset_stats["improvement_pct"]:+.1f}% | [PASS] 同池配对 | [PASS] seed={config["seed"]} | p={subset_stats["p_value"]:.2e} |

### 数字差异说明

原 76.4% 来自不公平对比（SABRE 评测含深电路，PPO 仅评测浅电路，非配对设计），
本公平对比在相同电路池上配对评测，拓扑统一为 4×4 2D网格（匹配天衍真机）。

改进率 = (1 - PPO均值/SABRE均值) × 100%：正值表示 PPO 更优，负值表示 SABRE 更优。

当前模型在4×4 2D网格+全电路分布上训练200k steps。
全60电路平均变化 {stats_summary["improvement_pct"]:+.1f}%（p={stats_summary["p_value"]:.2e}），
中+深电路子集变化 {subset_stats["improvement_pct"]:+.1f}%（p={subset_stats["p_value"]:.2e}），
{sig_summary}

---

## 八、结论

1. **公平对比方法论修复**：修复了原 `compilation_full.py` 非配对、无种子、无统计检验的不公平对比设计，
   实现同池配对（60电路）、固定种子、Wilcoxon符号秩检验、Bootstrap CI的科学对比框架。
2. **拓扑对齐**：SABRE与PPO统一在4×4 2D网格拓扑（匹配天衍真机nearest-neighbor结构）上评测，
   与 `compilation_env.py` 的 COUPLING_GRAPH 一致。
3. **当前模型评测结果**：全60电路 SABRE avg={stats_summary["sabre_mean"]:.2f} SWAP，
   PPO avg={stats_summary["ppo_mean"]:.2f} SWAP，变化 {stats_summary["improvement_pct"]:+.1f}%
   （p={stats_summary["p_value"]:.2e}）；中+深子集变化 {subset_stats["improvement_pct"]:+.1f}%（p={subset_stats["p_value"]:.2e}）。
4. **统计结论**：{stats_conclusion}
5. 完整逐电路数据入库 `results/compilation/fair_v2_per_circuit.json` 可复算。

---

## 九、复现方法

```bash
# 1. 使用已训练 PPO 模型运行公平对比
python scripts/evaluation/compilation_fair_v2.py

# 2. 输出
# - results/compilation/fair_v2_per_circuit.json  (逐电路明细)
# - results/compilation/fair_v2_summary.json       (统计汇总)
# - results/reports/compilation_fair_v2_report.md  (本报告)
```

---

*公平对比 v2 自动生成 | Issue #451 | 数据源: {config["model_path"]}*
"""


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  Issue #451: 编译层公平对比 v2")
    print("=" * 60)

    config = {"seed": SEED, "model_path": MODEL_PATH, "n_circuits": 60}

    # 1. 生成统一电路池
    print(f"\n[1/5] 生成 {config['n_circuits']} 电路统一池 (seed={SEED})...")
    pool = generate_circuit_pool(SEED)
    print(f"  电路池: {len(pool)} 个 (浅/中/深各 {N_PER_CATEGORY})")

    # 2. SABRE 评测
    print("\n[2/5] SABRE baseline 评测 (同池配对)...")
    sabre_results = evaluate_sabre(pool)
    sabre_swaps = [r["sabre_swap"] for r in sabre_results]
    print(
        f"  SABRE: avg={np.mean(sabre_swaps):.2f}, std={np.std(sabre_swaps, ddof=1):.2f}, "
        f"n={len(sabre_swaps)}"
    )

    # 3. PPO 评测
    print(f"\n[3/5] PPO 评测 (同池配对, model={MODEL_PATH})...")
    ppo_results = evaluate_ppo(pool, MODEL_PATH)
    ppo_swaps = [r["ppo_swap"] for r in ppo_results]
    print(
        f"  PPO: avg={np.mean(ppo_swaps):.2f}, std={np.std(ppo_swaps, ddof=1):.2f}, "
        f"n={len(ppo_swaps)}"
    )

    # 4. 合并逐电路明细 + 统计分析
    print("\n[4/5] 统计分析...")
    per_circuit: list[dict[str, Any]] = []
    for s, p in zip(sabre_results, ppo_results, strict=False):
        merged = {
            "index": s["index"],
            "category": s["category"],
            "qubits": s["qubits"],
            "depth": s["depth"],
            "gates": s["gates"],
            "sabre_swap": s["sabre_swap"],
            "ppo_swap": p["ppo_swap"],
            "diff_sabre_minus_ppo": s["sabre_swap"] - p["ppo_swap"],
        }
        per_circuit.append(merged)

    stats_summary = compute_statistics(sabre_swaps, ppo_swaps)
    breakdown = compute_category_breakdown(per_circuit)
    subset_analysis = compute_subset_analysis(per_circuit)
    subset_stats = subset_analysis["stats"]
    print(
        f"  变化率: {stats_summary['improvement_pct']:+.1f}% "
        f"(SABRE {stats_summary['sabre_mean']:.2f} → PPO {stats_summary['ppo_mean']:.2f})"
    )
    print(
        f"  Wilcoxon: W={stats_summary['wilcoxon_w']:.0f}, "
        f"p={stats_summary['p_value']:.2e}, rank-biserial={stats_summary['rank_biserial']:.3f}"
    )
    print(
        f"  Bootstrap 95% CI: [{stats_summary['bootstrap_ci_low']:.1f}%, "
        f"{stats_summary['bootstrap_ci_high']:.1f}%]"
    )
    print(
        f"  显著性 (α=0.05): {'[PASS] 显著' if stats_summary['significant'] else '[FAIL] 不显著'}"
    )
    print(
        f"  子集（中+深 n={subset_analysis['n_pairs']}）: "
        f"{subset_stats['improvement_pct']:+.1f}%, p={subset_stats['p_value']:.2e}, "
        f"{'[PASS] 显著' if subset_stats['significant'] else '[FAIL] 不显著'}"
    )

    # 5. 生成报告与 JSON
    print("\n[5/5] 生成报告与 JSON...")
    os.makedirs("results/compilation", exist_ok=True)
    os.makedirs("results/reports", exist_ok=True)

    # 逐电路 JSON
    per_circuit_path = "results/compilation/fair_v2_per_circuit.json"
    with open(per_circuit_path, "w", encoding="utf-8") as f:
        json.dump(
            {"config": config, "per_circuit": per_circuit, "categories": CATEGORIES},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  [OK] {per_circuit_path}")

    # 汇总 JSON
    summary_path = "results/compilation/fair_v2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config,
                "stats": stats_summary,
                "category_breakdown": breakdown,
                "subset_analysis": subset_analysis,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  [OK] {summary_path}")

    # Markdown 报告
    report = generate_report(stats_summary, breakdown, subset_analysis, per_circuit, config)
    report_path = "results/reports/compilation_fair_v2_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  [OK] {report_path}")

    print("\n" + "=" * 60)
    print("  公平对比 v2 完成")
    print(
        f"  全 60 电路: {stats_summary['improvement_pct']:+.1f}% (p={stats_summary['p_value']:.2e})"
    )
    print(
        f"  中+深 40 子集: {subset_stats['improvement_pct']:+.1f}% "
        f"(p={subset_stats['p_value']:.2e})"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
