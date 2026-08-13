# 8 策略对比报告（50 Seed 权威验证版，v9.1+ 16维交付模型）

> **数据来源（权威）**: `results/multiseed_evaluation/rewards_multiseed.json`（2026-08-05 8.5 基线诚实化，50 seeds × 5 episodes = 250 次独立运行，vs 真实 FCFS 量子路由；旧版 16dim json 为 vs Hybrid-Default 弱基线，已废弃）
> **运行环境**: 16 维原生观测空间（`QuantumSchedulingEnv` 默认配置，OBS_DIM=16，v9+ 交付标准）
> **PPO 模型**: `deliverable_models/ppo_best_model_16dim.zip`（16维，Actor-Critic）
> **DQN 模型**: 无（v9 已删除 DQN 模型，DQN 策略位使用 Random 替代，仅供策略完整性对比）
> **显著性水平**: α = 0.05（Bonferroni 校正，28 次两两比较，校正后 α = 0.0018）
> **统计方法**: Welch t 检验（方差不齐）/ Mann-Whitney U 检验（非正态）/ 独立样本 t 检验（正态方差齐）
> **提升百分比 CI**: Bootstrap 百分位法（10000 次重抽样，95% CI）
> **版本历史**: v9.1 (2026-07-29) 基于 16 维交付模型重新评估；v9.0 (2026-07-25) 14 维模型评估结果已废弃

---

## 一、核心指标排名表（50 Seed 权威验证，N=250）

按**平均奖励（降序）**排列。

| 排名 | 策略 | 平均奖励 | 标准差 | 标准误 | 提升 vs FCFS | 提升% 95% CI | 统计显著性 |
|:--:|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | **PPO (16维)** | **1982.69** | 557.25 | 35.24 | **+20.2%** | [+14.3%, +26.7%] | ✅ p=7.56e-12 |
| 2 | FCFS | 1648.91 | 502.95 | 31.81 | 基线 | — | — |
| 3 | SJF | 748.48 | 304.86 | 19.28 | -54.6% | — | ✅ p=5.870e-61（显著劣于 FCFS） |
| 4 | DQN (Random 替代) | 697.40 | 288.25 | 18.23 | -57.7% | — | ✅ p=1.009e-64 |
| 5 | Random | 697.40 | 288.25 | 18.23 | -57.7% | — | ✅ p=1.009e-64 |
| 6 | Greedy | 62.72 | 537.68 | 34.00 | -96.2% | — | n/a |
| 7 | Quantum-Only | -826.59 | 263.10 | 16.67 | -150.1% | — | n/a |
| 8 | Classical-Only | -1075.49 | 74.89 | 4.75 | -165.2% | — | n/a |

> 注：v9 已删除 DQN 模型，DQN 策略位使用 Random 替代，仅供策略完整性对比。
> SJF 与 FCFS 差异显著（p=5.870e-61，8.13 重算；旧 2.28e-60 为 8.8 值、0.2827 为错误值已废弃），SJF 显著劣于 FCFS（-54.6%）。
> PPO 排名第一，且与各基线差异显著（vs 真实 FCFS p=7.56e-12，vs SJF p=3.713e-71，vs Random/DQN 占位 p=4.289e-73）。
>
> **FCFS 基线说明（8.5 诚实化）**：本实验的 FCFS 为**真实 FCFS（量子路由）**（EnvBasedFCFSScheduler：
> 量子任务→量子机、经典任务→经典机，与 RL 同协议同种子）。8.5 前旧口径"FCFS=Hybrid-Default
> （恒 action=2）"为弱基线（对应 +123.4% 旧值），已废弃。统计显著性：SJF vs FCFS 权威 p=2.28e-60。

---

## 二、关键结论（50 Seed 权威验证）

### 2.1 PPO vs FCFS（核心指标）

- **PPO (16维) 平均奖励**: 1982.69 ± 35.24（标准误，N=50 seeds × 5 episodes = 250，vs 真实 FCFS）
- **FCFS 平均奖励**: 1648.91 ± 31.81（标准误，8.5 基线诚实化）
- **PPO vs FCFS 提升**: **+20.2%**（+333.78）
- **提升百分比 95% CI**: **[+14.3%, +26.7%]**（Bootstrap，10000 次重抽样）
- **统计检验**: Welch t 检验（方差不齐），**p = 7.56×10⁻¹²**（高度显著）
- **效应量**: rank-biserial = -0.3642（**中效应**）
- **均值差 95% CI**: [+240.31, +427.24]
- **正态性**: 数据通过正态性检验，使用 Welch t 检验（方差不齐）

### 2.2 PPO vs 所有基线

| 对比项 | 提升值 | 提升比例 | 显著性 | 效应量 |
|:--|:--:|:--:|:--:|:--:|
| PPO vs FCFS | +333.78 | +20.2% | ✅ p=7.56e-12 | rank-biserial=-0.3642 |
| PPO vs Random | +1285.29 | +184.3% | ✅ p=4.289e-73 | rank-biserial=-0.9348 |
| PPO vs SJF | +1234.22 | +164.9% | ✅ p=3.713e-71 | rank-biserial=-0.9220 |
| PPO vs Greedy | +1919.98 | n/a（基线为负） | ✅ p=1.538e-80 | rank-biserial=-0.9824 |
| PPO vs DQN(Random) | +1285.29 | +184.3% | ✅ p=4.289e-73 | rank-biserial=-0.9348 |

> 注：本表 8.13 随 8 策略重算更新（旧 +163.5%/+121.5%/d=-2.2581 等为 14 维/8.8 过期值）。

### 2.3 启发式基线对比

- **SJF vs FCFS**: 差异显著（p=5.870e-61，8.13 重算；旧 2.28e-60 为 8.8 值、0.2827 为错误值已废弃），SJF 显著劣于 FCFS（-54.6%）
- **FCFS vs Random**: 显著（p=1.009e-64，rank-biserial=-0.8781，大效应量），FCFS 比随机高 +136.4%
- **Quantum-Only vs Classical-Only**: 显著，仅量子策略优于仅经典策略

---

## 三、PPT/白皮书可用结论

> **在 16 维原生环境中（OBS_DIM=16，N=50 seeds × 5 episodes = 250 次独立运行，8.5 基线诚实化），PPO 平均奖励（1982.69±35.24）比真实 FCFS 基线（1648.91±31.81）提升 +20.2%（95% CI: [14.3%, 26.7%]，Welch t 检验，p=7.56e-12），验证了 RL 调度相对朴素路由的显著优势（旧 +123.4% 为 vs Hybrid-Default 恒 action=2 弱基线，已废弃）。**

---

## 四、数据复现说明

复现本报告结果：

```bash
# 运行 50 seed 评估（16 维交付模型，约 20-40 分钟）
python scripts/evaluation/run_multiseed_evaluation.py \
    --seeds 50 --episodes 5 --obs-dim 16 \
    --ppo-model deliverable_models/ppo_best_model_16dim.zip \
    --n-workers 4

# 统计显著性检验
python scripts/evaluation/statistical_significance.py \
    --input results/multiseed_evaluation/rewards_multiseed.json \
    --output results/reports/statistical_validation.md
```

关键依赖：
- PPO 模型: `deliverable_models/ppo_best_model_16dim.zip`（16维，Actor-Critic）
- 环境: `QuantumSchedulingEnv`（原生 16 维观测空间，OBS_DIM=16）
- 兼容包装器: `Obs10Wrapper`（16→10 维兼容，用于加载 10 维旧模型；本次评估 PPO 使用原生 16 维，无需 Obs10Wrapper）

---

## 五、版本迁移说明

### v9.1 (2026-07-29) 从 14 维迁移至 16 维

| 指标 | v9.0 (14维, 已废弃) | v9.1 (16维, 权威) | 变化 |
|:--|:--:|:--:|:--:|
| PPO 平均奖励 | 旧值已废弃 | 1982.69 ± 557.25 | 8.5 基线诚实化（真实 FCFS 基线） |
| FCFS 平均奖励 | 旧值已废弃 | 1648.91 ± 502.95 | 8.5 基线诚实化（真实 FCFS 量子路由） |
| PPO vs FCFS 提升 | 旧值已废弃 | +20.2% | 旧 +123.4% 为 vs Hybrid-Default 弱基线，已废弃 |
| p 值 | 旧值已废弃 | 7.56e-12 (Welch t) | 8.5 基线诚实化 |
| 效应量 | 旧值已废弃 | rank-biserial=-0.3642 | 中效应 |

> **注**: v9.1+ 使用 Welch t 检验（数据通过正态性检验，方差不齐）；
> v9.0 使用 Mann-Whitney U 检验（旧 14 维数据非正态）。检验方法变化反映数据分布特性，不影响结论有效性。

---

*报告生成时间: 2026-07-29（8.13 更新 §2.2/§2.3 至当前口径）| 数据源: results/multiseed_evaluation/rewards_multiseed.json | 统计方法: SciPy + Bootstrap + Bonferroni校正 | 模型: ppo_best_model_16dim.zip (OBS_DIM=16)*
