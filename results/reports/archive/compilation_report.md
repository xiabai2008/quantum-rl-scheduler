# 编译层 PPO SWAP 优化实验报告

> ⚠️ **[DEPRECATED] 本报告已废弃（Issue #560）**
>
> 本报告基于**不公平对比设计**（SABRE 评测60电路含深电路类，PPO 仅评测20个浅电路，非配对、分布不同，全程未播种），
> 产出的 `-76.4%` 数字已被 README/AGENTS 标注为废弃，不可作为成果引用。
> 公平对比请参考 **`results/reports/compilation_fair_v2_report.md`**：
> 整体 p=0.861 不显著，深电路(14-16q) SWAP 减少约 33%（无单独统计检验）。
>
> 本报告保留仅作为历史记录与问题复盘，**禁止在答辩/PPT/白皮书中引用 -76.4% 数字**。
> 生成脚本 `scripts/reporting/gen_compilation_report.py` 同步标注为 DEPRECATED。

**生成时间**: 2026-07-27 22:08:09
**Issue**: #395 — 编译层 PPO SWAP -76.4% vs SABRE 实验报告（已废弃，见公平对比v2）
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
| 平均 SWAP 数 | 34.02 | 10.75 | **-68.4%** |
| 标准差 | 69.72 | 6.23 | — |
| 样本数 | 60 | 60 | — |

---

## 三、分类别对比

| 类别 | SABRE avg SWAP | PPO avg SWAP | 改进率 |
|:--|:--|:--|:--|
| 浅电路 (5-8 qubits) | 1.0 | 3.3 | --230.0% |
| 中电路 (9-14 qubits) | 26.0 | 11.3 | -56.3% |
| 深电路 (14-16 qubits) | 75.0 | 17.6 | -76.5% |

---

## 四、统计显著性检验

### Wilcoxon signed-rank 检验（配对设计，同一组电路）

| 统计量 | 值 |
|:--|:--|
| W 统计量 | 697 |
| p 值 | 2.05e-01 |
| 效应量 (rank-biserial) | 0.127 |
| 配对数 | 60 |
| 检验方向 | 单侧 (PPO < SABRE) |

**结论**: p = 2.05e-01 > 0.05，**未达到统计显著性水平**，PPO 与 SABRE 的差异不显著（不可宣称"显著优于"）。本报告基于不公平对比设计，已废弃，请参考 `compilation_fair_v2_report.md`。

### 逐电路改进率

| 统计量 | 值 |
|:--|:--|
| 平均改进率 | -285.8% |
| 中位数改进率 | -100.0% |
| 95% CI | [-1752.5%, 92.1%] |
| 最小改进率 | -1900.0% |
| 最大改进率 | 94.6% |

---

## 五、OR-Tools 对比参考

OR-Tools CP-SAT 求解器作为最优解参考（来源: `scripts/evaluation/compilation_full.py` 第 4 阶段）。
OR-Tools 在 20/50/100 任务规模下提供了调度层 makespan 的理论下界，与编译层 SWAP 优化互补。

---

## 六、结论

> ⚠️ 本节结论基于不公平对比设计，已废弃（Issue #560）。以下文字保留作为问题复盘记录，**不可作为成果引用**。

1. **PPO 驱动的比特映射在 60 个随机电路上平均减少 SWAP 门 68.4%**（34.0 → 10.8）—— 但该数字来自不公平对比（SABRE 评测含深电路，PPO 仅评测浅电路）
2. 统计显著性：Wilcoxon signed-rank 检验 p = 2.05e-01 > 0.05，**不显著**，效应量 rank-biserial = 0.127（小效应）—— **不可宣称"显著优于 SABRE"**
3. 改进在所有三个电路类别（浅/中/深）中均一致 —— 但浅电路 PPO 反而更差（-230%），分类别数字亦不可信
4. **原始实验数据**（`results/compilation/full_eval_20260727_102903.json`）：PPO avg SWAP = 6.5, SABRE avg SWAP = 27.566666666666666, 改进率 = -76.4% —— **已废弃**（不公平对比，PPO 仅评测浅电路）
5. 两次独立运行结果（-76.4% 和 -68.4%）均**不能**证实 PPO 显著优于 SABRE —— p=0.205 不显著，且对比设计不公平
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
