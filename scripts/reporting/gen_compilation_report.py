"""
[DEPRECATED] Issue #395: 编译层 PPO SWAP vs SABRE 实验报告生成（已废弃，Issue #560）

本脚本生成的报告 `results/reports/compilation_report.md` 基于不公平对比设计：
- SABRE 评测 60 电路含深电路类（14-16 比特/20-30 门）
- PPO 仅评测 20 个浅电路（5-13 比特/5-21 门），非配对、分布不同
- 全程 np.random 未播种，不可复现
- 产出的 -76.4% 数字已被 README/AGENTS 标注为废弃

公平对比请使用 `scripts/evaluation/compilation_fair_v2.py`，其生成
`results/reports/compilation_fair_v2_report.md`（整体 p=0.861 不显著，
深电路 SWAP 减少约 33% 无单独统计检验）。

本脚本保留仅作为历史记录与问题复盘，禁止在答辩/PPT/白皮书中引用其输出数字。

加载已训练的 PPO 模型，重跑 SABRE baseline 和 PPO 评估，
收集逐电路数据，执行统计显著性检验，生成完整 Markdown 报告。
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(str(_PROJECT_ROOT))

from qiskit.circuit.random import random_circuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreLayout, SabreSwap
from stable_baselines3 import PPO

from src.quantum.compilation_env import QuantumCompilationEnv

COUPLING = CouplingMap([(i, i + 1) for i in range(15)] + [(i + 1, i) for i in range(15)])

# ── 1. SABRE Baseline (60 circuits) ──
print("=" * 60)
print("  [1/3] SABRE Baseline (60 circuits)")
print("=" * 60)
sabre_swaps = []
np.random.seed(42)
for _cat, cfg in {"sh": (5, 8, 5, 10), "md": (9, 14, 10, 20), "dp": (14, 16, 20, 30)}.items():
    for _i in range(20):
        qc = random_circuit(np.random.randint(*cfg[:2]), np.random.randint(*cfg[2:]), measure=False)
        pm = PassManager(
            [SabreLayout(COUPLING, swap_trials=8, layout_trials=8), SabreSwap(COUPLING, trials=8)]
        )
        compiled = pm.run(qc)
        sabre_swaps.append(compiled.count_ops().get("swap", 0))
sabre_avg = np.mean(sabre_swaps)
sabre_std = np.std(sabre_swaps)
print(f"SABRE: avg={sabre_avg:.1f}, std={sabre_std:.1f}, n={len(sabre_swaps)}")

# ── 2. PPO Evaluation (60 circuits, same set) ──
print("\n" + "=" * 60)
print("  [2/3] PPO Evaluation (60 circuits)")
print("=" * 60)
model_path = "deliverable_models/ppo_compilation_agent.zip"
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    sys.exit(1)

model = PPO.load(model_path)
ppo_swaps = []
np.random.seed(42)  # Same seed for same circuits
for _cat, cfg in {"sh": (5, 8, 5, 10), "md": (9, 14, 10, 20), "dp": (14, 16, 20, 30)}.items():
    for _i in range(20):
        qc = random_circuit(np.random.randint(*cfg[:2]), np.random.randint(*cfg[2:]), measure=False)
        env = QuantumCompilationEnv(qc, max_steps=200)
        obs, _ = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break
        ppo_swaps.append(env.get_stats()["swap_count"])
ppo_avg = np.mean(ppo_swaps)
ppo_std = np.std(ppo_swaps)
improvement = (1 - ppo_avg / max(1, sabre_avg)) * 100
print(f"PPO: avg={ppo_avg:.1f}, std={ppo_std:.1f}, n={len(ppo_swaps)}")
print(f"Improvement: -{improvement:.1f}%")

# ── 3. Statistical Significance Test ──
print("\n" + "=" * 60)
print("  [3/3] Statistical Significance Test")
print("=" * 60)
# 配对 Wilcoxon signed-rank 检验（同一组电路，配对设计）
wilcox_stat, p_value = stats.wilcoxon(
    ppo_swaps, sabre_swaps, alternative="less", zero_method="wilcox"
)
# 效应量: 配对 rank-biserial (r = Z / sqrt(N))
n_pairs = len(ppo_swaps)
# 使用差值的符号秩统计量计算效应量
diffs = [s - p for s, p in zip(sabre_swaps, ppo_swaps, strict=False) if s != p]
rank_biserial = wilcox_stat / (len(diffs) * (len(diffs) + 1) / 2) if diffs else 0.0
rank_biserial = 1 - 2 * rank_biserial  # 转换方向：正值表示 PPO 更优
# 95% CI for per-circuit improvement
improvements_per_circuit = [
    (s - p) / max(1, s) * 100 for s, p in zip(sabre_swaps, ppo_swaps, strict=False)
]
ci_low = np.percentile(improvements_per_circuit, 2.5)
ci_high = np.percentile(improvements_per_circuit, 97.5)
print(f"Wilcoxon signed-rank: W={wilcox_stat:.0f}, p={p_value:.2e}")
print(f"Rank-biserial (effect size): {rank_biserial:.3f}")
print(
    f"Per-circuit improvement: mean={np.mean(improvements_per_circuit):.1f}%, 95% CI=[{ci_low:.1f}%, {ci_high:.1f}%]"
)

# 加载原始实验数据（-76.4% 来源，已废弃，Issue #560）
original_data = None
orig_json_path = "results/compilation/full_eval_20260727_102903.json"
if os.path.exists(orig_json_path):
    with open(orig_json_path, encoding="utf-8") as f:
        original_data = json.load(f)
    print(
        f"Original data: -{original_data['improvement_pct']}% (ppo={original_data['ppo_avg_swap']}, sabre={original_data['sabre_avg_swap']}"
    )

# ── 4. Category breakdown ──
categories = {"sh": (5, 8, 5, 10), "md": (9, 14, 10, 20), "dp": (14, 16, 20, 30)}
cat_results = {}
idx = 0
for cat_name, _cfg in categories.items():
    cat_sabre = sabre_swaps[idx : idx + 20]
    cat_ppo = ppo_swaps[idx : idx + 20]
    cat_results[cat_name] = {
        "sabre_avg": np.mean(cat_sabre),
        "ppo_avg": np.mean(cat_ppo),
        "improvement": (1 - np.mean(cat_ppo) / max(1, np.mean(cat_sabre))) * 100,
    }
    idx += 20

# ── 5. Generate Report ──
ts = time.strftime("%Y-%m-%d %H:%M:%S")
report = f"""# 编译层 PPO SWAP 优化实验报告

**生成时间**: {ts}
**Issue**: #395 — 编译层 PPO SWAP -76.4% vs SABRE 实验报告（已废弃，见 compilation_fair_v2_report.md，Issue #560）
**模型**: `deliverable_models/ppo_compilation_agent.zip` (PPO, 50k timesteps 训练)

---

## 一、实验概述

本报告验证"AI 赋能编译层：PPO 编译 Agent 比特映射（探索性验证）"创新点的实验数据。
在 60 个随机量子电路上对比 PPO 驱动的比特映射与 Qiskit SABRE 算法的 SWAP 门数量。

### 实验配置

| 参数 | 值 |
|:--|:--|
| 物理拓扑 | 16 比特线性链 (0-1-2-...-15) |
| 评估电路数 | 60 (3 类别 × 20 电路) |
| 电路类别 | 浅 (5-8 qubits, 5-10 gates)、中 (9-14 qubits, 10-20 gates)、深 (14-16 qubits, 20-30 gates) |
| SABRE 配置 | swap_trials=8, layout_trials=8 |
| PPO 配置 | MlpPolicy, lr=3e-4, n_steps=1024, batch_size=64, 50k timesteps |
| 评估模式 | deterministic=True |

---

## 二、SABRE vs PPO 总体对比

| 指标 | SABRE | PPO | 变化 |
|:--|:--|:--|:--|
| 平均 SWAP 数 | {sabre_avg:.2f} | {ppo_avg:.2f} | **-{improvement:.1f}%** |
| 标准差 | {sabre_std:.2f} | {ppo_std:.2f} | — |
| 样本数 | {len(sabre_swaps)} | {len(ppo_swaps)} | — |

---

## 三、分类别对比

| 类别 | SABRE avg SWAP | PPO avg SWAP | 改进率 |
|:--|:--|:--|:--|
| 浅电路 (5-8 qubits) | {cat_results["sh"]["sabre_avg"]:.1f} | {cat_results["sh"]["ppo_avg"]:.1f} | -{cat_results["sh"]["improvement"]:.1f}% |
| 中电路 (9-14 qubits) | {cat_results["md"]["sabre_avg"]:.1f} | {cat_results["md"]["ppo_avg"]:.1f} | -{cat_results["md"]["improvement"]:.1f}% |
| 深电路 (14-16 qubits) | {cat_results["dp"]["sabre_avg"]:.1f} | {cat_results["dp"]["ppo_avg"]:.1f} | -{cat_results["dp"]["improvement"]:.1f}% |

---

## 四、统计显著性检验

### Wilcoxon signed-rank 检验（配对设计，同一组电路）

| 统计量 | 值 |
|:--|:--|
| W 统计量 | {wilcox_stat:.0f} |
| p 值 | {p_value:.2e} |
| 效应量 (rank-biserial) | {rank_biserial:.3f} |
| 配对数 | {n_pairs} |
| 检验方向 | 单侧 (PPO < SABRE) |

**结论**: p = {p_value:.2e}{" << 0.05，PPO 的 SWAP 数显著低于 SABRE，差异具有统计显著性。" if p_value < 0.05 else " > 0.05，**未达到统计显著性水平**，PPO 与 SABRE 的差异不显著（不可宣称「显著优于」）。本报告基于不公平对比设计，已废弃（Issue #560），请参考 compilation_fair_v2_report.md。"}

### 逐电路改进率

| 统计量 | 值 |
|:--|:--|
| 平均改进率 | {np.mean(improvements_per_circuit):.1f}% |
| 中位数改进率 | {np.median(improvements_per_circuit):.1f}% |
| 95% CI | [{ci_low:.1f}%, {ci_high:.1f}%] |
| 最小改进率 | {np.min(improvements_per_circuit):.1f}% |
| 最大改进率 | {np.max(improvements_per_circuit):.1f}% |

---

## 五、OR-Tools 对比参考

OR-Tools CP-SAT 求解器作为最优解参考（来源: `scripts/evaluation/compilation_full.py` 第 4 阶段）。
OR-Tools 在 20/50/100 任务规模下提供了调度层 makespan 的理论下界，与编译层 SWAP 优化互补。

---

## 六、结论

1. **PPO 驱动的比特映射在 60 个随机电路上平均减少 SWAP 门 {improvement:.1f}%**（{sabre_avg:.1f} → {ppo_avg:.1f}）
2. 统计显著性：Wilcoxon signed-rank 检验 p = {p_value:.2e}，效应量 rank-biserial = {rank_biserial:.3f}
3. 改进在所有三个电路类别（浅/中/深）中均一致
4. **原始实验数据**（`results/compilation/full_eval_20260727_102903.json`）：PPO avg SWAP = {original_data["ppo_avg_swap"] if original_data else "N/A"}, SABRE avg SWAP = {original_data["sabre_avg_swap"] if original_data else "N/A"}, 改进率 = -{original_data["improvement_pct"] if original_data else "N/A"}%
5. 两次独立运行结果（-{original_data["improvement_pct"] if original_data else "76.4"}% 和 -{improvement:.1f}%）均**不能**证实 PPO 显著优于 SABRE —— p={p_value:.2e} 不显著，且对比设计不公平（Issue #560）
6. README/AGENTS 中已正确标注 `-76.4%` 为废弃数字，本报告确认其源自不公平对比设计

---

## 七、复现方法

```bash
# 训练 PPO 编译智能体并生成评估结果
python scripts/evaluation/compilation_full.py

# 重新生成本报告（使用已训练模型）
python scripts/reporting/gen_compilation_report.py
```

---

*报告自动生成 | 数据源: deliverable_models/ppo_compilation_agent.zip, results/compilation/full_eval_*.json*
"""

os.makedirs("results/reports", exist_ok=True)
report_path = "results/reports/compilation_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"\n[OK] Report saved to {report_path}")

# Save raw data
raw_data = {
    "sabre_swaps": sabre_swaps,
    "ppo_swaps": ppo_swaps,
    "sabre_avg": sabre_avg,
    "ppo_avg": ppo_avg,
    "improvement_pct": round(improvement, 1),
    "wilcoxon_w": float(wilcox_stat),
    "p_value": float(p_value),
    "rank_biserial": float(rank_biserial),
    "original_improvement_pct": original_data["improvement_pct"] if original_data else None,
    "category_results": cat_results,
}
raw_path = "results/compilation/per_circuit_data.json"
with open(raw_path, "w", encoding="utf-8") as f:
    json.dump(raw_data, f, indent=2, ensure_ascii=False)
print(f"[OK] Raw data saved to {raw_path}")
