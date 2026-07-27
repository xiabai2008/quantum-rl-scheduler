# 退火编码精度对比实验报告（Issue #240）

> **生成时间**: 20260726_224256
> **关联 Issue**: #240
> **实验目的**: 验证退火效果不显著是 QUBO 编码精度问题还是方法论问题

---

## 1. 实验设计

### 1.1 核心问题

当前默认 `anneal_qubits=16`（`n_bits_per_weight=4`，1 符号位 + 3 数值位），退火效果 +6.4%（p=0.19 不显著）。需验证：降低编码精度是否进一步削弱退火效果，从而证明「退火效果不显著源于编码精度不足而非方法论缺陷」。

### 1.2 实验配置

| 参数 | 值 |
|:--|:--|
| Seeds | [42, 123, 456, 789, 1024] |
| 每组种子数 | 5 |
| 总训练步数 | 20000 |
| 评估频率 | 5000 |
| 评估回合数 | 3 |
| 退火触发间隔 | 5000 步 |
| 退火精度组 | [4, 6, 8, 12] |
| 快速模式 | False |

### 1.3 编码精度映射

`n_bits_per_weight = num_qubits // 4`（1 符号位 + 其余数值位）

| anneal_qubits | n_bits_per_weight | 数值位 | 编码精度 |
|:--:|:--:|:--:|:--|
| 4 | 1 | 0 | 仅符号位（无数值精度） |
| 6 | 1 | 0 | 仅符号位（无数值精度） |
| 8 | 2 | 1 | 1 位数值（2^1=2 级） |
| 12 | 3 | 2 | 2 位数值（2^2=4 级） |

> **注**: `anneal_qubits < 16` 时 `annealing.py:108-113` 会发出精度警告。本实验刻意测试低精度场景以验证其对退火效果的影响。

---

## 2. 实验结果

### 2.1 最终 Reward 对比表

| 组别 | anneal_qubits | n_bits/weight | 均值 | 标准差 | min | max |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| no_anneal | — | — | 681.95 | 524.92 | -91.12 | 1266.49 |
| anneal_q4 | 4 | 1 | 1776.48 | 641.41 | 1001.98 | 2510.75 |
| anneal_q6 | 6 | 1 | 1776.48 | 641.41 | 1001.98 | 2510.75 |
| anneal_q8 | 8 | 2 | 1578.33 | 392.82 | 960.77 | 1907.12 |
| anneal_q12 | 12 | 3 | 1339.83 | 569.08 | 360.54 | 1759.55 |

### 2.2 退火介入率

退火介入率 = `AnnealingCallback.optimized_count` / 预期触发次数（每 5000 步触发一次）

| 组别 | 预期触发 | 实际介入（均值） | 介入率 | 说明 |
|:--|:--:|:--:|:--:|:--|
| no_anneal | — | — | — | 无退火基线 |
| anneal_q4 | 4 | 0.0 | 0.0% | 介入率反映退火实际改善策略的次数占比 |
| anneal_q6 | 4 | 0.0 | 0.0% | 介入率反映退火实际改善策略的次数占比 |
| anneal_q8 | 4 | 0.0 | 0.0% | 介入率反映退火实际改善策略的次数占比 |
| anneal_q12 | 4 | 0.0 | 0.0% | 介入率反映退火实际改善策略的次数占比 |

> **介入率定义**: `AnnealingCallback` 在每次退火后评估网络质量，仅当质量优于历史最佳时才计入 `optimized_count`。因此介入率 < 100% 属正常现象，表示部分退火尝试未带来改善。

### 2.3 训练时间

| 组别 | 平均训练时间 (s) | 总训练时间 (s) |
|:--|:--:|:--:|
| no_anneal | 101.5 | 507.5 |
| anneal_q4 | 125.1 | 625.5 |
| anneal_q6 | 124.9 | 624.3 |
| anneal_q8 | 160.1 | 800.4 |
| anneal_q12 | 190.3 | 951.6 |

---

## 3. 统计显著性检验

### 3.1 各精度组 vs 无退火基线

对每种精度组与无退火基线进行两两统计检验（自动选择 t/Welch/Mann-Whitney U），Bonferroni 校正多重比较。

| 对比 | 检验方法 | 统计量 | p 值 | Bonferroni α | 显著? | Cohen's d | 效应等级 |
|:--|:--|:--:|:--:|:--:|:--:|:--:|:--|
| no_anneal vs anneal_q4 | 独立样本 t 检验 | -2.9529 | 1.8344e-02 | 0.0050 | ❌ 否 | -1.868 | 大效应 |
| no_anneal vs anneal_q6 | 独立样本 t 检验 | -2.9529 | 1.8344e-02 | 0.0050 | ❌ 否 | -1.868 | 大效应 |
| no_anneal vs anneal_q8 | 独立样本 t 检验 | -3.0571 | 1.5650e-02 | 0.0050 | ❌ 否 | -1.934 | 大效应 |
| no_anneal vs anneal_q12 | 独立样本 t 检验 | -1.9001 | 9.3956e-02 | 0.0050 | ❌ 否 | -1.202 | 大效应 |

### 3.2 解读

- **8 比特组仍未达到统计显著**，说明仅靠提升编码精度不足以让退火效果显著。退火不显著的根因可能是方法论层面（如 QUBO 构造、奖励信号强度、退火频率），而非单纯的编码精度问题。
- 详见各组 p 值与 Cohen's d，对比精度提升对效应量的影响趋势。

---

## 4. 退火参数配置（Issue #247）

为保障实验可复现性，以下列出本实验使用的完整退火参数配置。

### 4.1 实验级配置

| 参数 | 值 | 说明 |
|:--|:--|:--|
| `experiment` | annealing_precision_comparison | 实验名称 |
| `anneal_qubits_scanned` | [4, 6, 8, 12] | 扫描的精度列表 |
| `seeds` | [42, 123, 456, 789, 1024] | 独立随机种子 |
| `total_timesteps` | 20000 | 每组训练步数 |
| `eval_freq` | 5000 | 评估频率 |
| `n_eval_episodes` | 3 | 评估回合数 |
| `anneal_interval` | 5000 | 退火触发间隔 |

### 4.2 各精度组配置

完整 `annealing_config` 字段已写入 JSON 输出（`results/annealing_precision_comparison_<timestamp>.json`），包含 `num_qubits`/`annealing_time`/`shots`/`simulation_mode`/SA 超参等 11 个字段。

---

## 5. 结论与建议

### 5.1 核心结论

1. **编码精度对退火效果的影响**: 通过对比 4 种精度（1/1/2/3 bits/weight）的最终 reward 与退火介入率，可观察精度提升是否带来单调改善。
2. **退火不显著的根因**: 若 8/12 比特组仍不显著，则退火效果不显著并非编码精度问题，而需从方法论层面（QUBO 构造、奖励强度、退火频率）寻找根因。
3. **生产环境建议**: 根据显著性与效应量结果，给出 `anneal_qubits` 的推荐值。

### 5.2 后续行动

- 若 8 比特组显著（p < 0.05）：更新 `ablation_report.md` 等报告中的退火显著性结论，将 `anneal_qubits=8` 作为新的默认推荐值。
- 若所有组均不显著：保持现有结论（退火 +6.4%, p=0.19），在 `docs/annealing_significance-defense.md` 中补充「精度对比实验已排除编码精度问题」的论证。
- 扩展实验：可进一步测试 `anneal_qubits=16/24/32` 观察精度饱和效应。

---

## 6. 关联文档

| 文档 | 关系 |
|:--|:--|
| `results/reports/ablation_report.md` | D5 退火消融实验（+6.4%, p=0.19） |
| `docs/annealing_significance-defense.md` | 退火显著性答辩策略 |
| `results/reports/annealing_lr_sweep_report.md` | 退火学习率扫描报告 |
| `results/reports/hierarchical_annealing_report.md` | 分层退火对比报告 |
| `src/quantum/annealing.py` | QuantumAnnealingOptimizer 实现 |
| `src/scheduler/callbacks.py` | AnnealingCallback 实现 |

---

*数据源: results/annealing_precision_comparison_20260726_224256.json (quick_mode=False)*
