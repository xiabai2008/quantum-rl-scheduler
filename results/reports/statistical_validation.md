# 统计显著性检验报告（多Seed验证）

> 本报告为提交清单 `EXP_STAT` 必需文件，使用 250 次独立episode验证PPO相对于基线策略的统计显著性。

> **数据来源**: `C:\Users\HZR\Desktop\揭榜挂帅擂台赛\quantum-rl-scheduler\results\multiseed_evaluation\rewards_multiseed.json`
> **显著性水平 α**: 0.05
> **比较次数**: 28（Bonferroni 校正后 α = 0.0018）

---


## 零、权威实验数字（多 Seed 验证）

> **实验配置**: 50 seeds × 5 episodes = 250 次独立运行
> **环境**: 14 维观测空间（原生 14 维环境）
> **任务规模**: 每 episode 200 步，泊松到达 λ=0.5，量子任务占比 70%
> **PPO 模型**: `deliverable_models/ppo_best_model_14dim.zip`（14维，Actor-Critic）
> **DQN 模型**: `deliverable_models/dqn_best_model_14dim.zip`（14维，Double DQN + reward clip）
> **显著性水平**: α = 0.05（Bonferroni 校正）

| 排名 | 策略 | 平均奖励 | 标准差 | 标准误 | 提升 vs FCFS | 提升% 95% CI |
|:--:|:--|:--:|:--:|:--:|:--:|:--:|
| 1 | PPO | 2746.94 | 1160.72 | 73.41 | +88.3% | [+78.5%, +98.2%] |
| 2 | DQN | 1527.65 | 124.02 | 7.84 | +4.7% | [+3.6%, +6.0%] |
| 3 | SJF | 1462.39 | 134.32 | 8.50 | +0.2% | [-1.0%, +1.5%] |
| 4 | FCFS | 1458.77 | 60.47 | 3.82 | 基线 | — |
| 5 | Random | 1247.17 | 385.76 | 24.40 | -14.5% | [-17.8%, -11.2%] |
| 6 | Greedy | -25.95 | 625.52 | 39.56 | -101.8% | [-107.0%, -96.5%] |
| 7 | Quantum-Only | -920.54 | 232.68 | 14.72 | -163.1% | [-165.0%, -161.1%] |
| 8 | Classical-Only | -1128.29 | 59.46 | 3.76 | -177.3% | [-178.0%, -176.7%] |

**核心结论：PPO 平均奖励 2746.94 vs FCFS 1458.77，提升 +88.3%，95% CI: [+78.5%, +98.2%]**
（N=250 次独立episode，α=0.05，Bonferroni多重比较校正）

---

## 一、各策略奖励统计

| 策略 | 样本数 | 平均奖励 | 标准差 | 最小值 | 最大值 |
|:--|:--:|:--:|:--:|:--:|:--:|
| DQN | 250 | 1527.65 | 124.02 | 1321.54 | 2231.43 |
| FCFS | 250 | 1458.77 | 60.47 | 1312.01 | 1610.51 |
| Random | 250 | 1247.17 | 385.76 | 385.98 | 2310.58 |
| Quantum-Only | 250 | -920.54 | 232.68 | -1238.56 | 322.45 |
| Classical-Only | 250 | -1128.29 | 59.46 | -1267.91 | -946.13 |
| Greedy | 250 | -25.95 | 625.52 | -1231.43 | 1739.73 |
| SJF | 250 | 1462.39 | 134.32 | 1205.06 | 2083.44 |
| PPO | 250 | 2746.94 | 1160.72 | -619.82 | 6109.21 |

## 二、两两比较结果

| 对比 | 检验方法 | 统计量 | p 值 | 显著? | 效应量 | 均值差 | 95% CI | 提升% 95% CI |
|:--|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DQN vs FCFS | Mann-Whitney U 检验 | 42317.0000 | 7.343e-12 | ✅ 是 | rank-biserial correlation=0.3541 | 68.88 | [51.73, 86.03] | [+3.6%, +6.0%] |
| DQN vs Random | Mann-Whitney U 检验 | 48058.0000 | 2.357e-25 | ✅ 是 | rank-biserial correlation=0.5379 | 280.48 | [230.13, 330.83] | [+17.9%, +27.6%] |
| DQN vs Quantum-Only | Mann-Whitney U 检验 | 62500.0000 | 2.234e-83 | ✅ 是 | rank-biserial correlation=1.0000 | 2448.19 | [2415.43, 2480.95] | [+260.7%, +271.7%] |
| DQN vs Classical-Only | Mann-Whitney U 检验 | 62500.0000 | 2.234e-83 | ✅ 是 | rank-biserial correlation=1.0000 | 2655.94 | [2638.85, 2673.03] | [+233.8%, +237.1%] |
| DQN vs Greedy | Mann-Whitney U 检验 | 62030.0000 | 6.05e-81 | ✅ 是 | rank-biserial correlation=0.9850 | 1553.60 | [1474.36, 1632.84] | [+1580.1%, +93233.2%] |
| DQN vs SJF | Mann-Whitney U 检验 | 42120.0000 | 1.71e-11 | ✅ 是 | rank-biserial correlation=0.3478 | 65.25 | [42.53, 87.97] | [+2.9%, +6.0%] |
| DQN vs PPO | Mann-Whitney U 检验 | 9839.0000 | 4.258e-40 | ✅ 是 | rank-biserial correlation=-0.6852 | -1219.30 | [-1364.35, -1074.24] | [-47.2%, -41.3%] |
| FCFS vs Random | Mann-Whitney U 检验 | 45479.0000 | 1.271e-18 | ✅ 是 | rank-biserial correlation=0.4553 | 211.60 | [163.08, 260.12] | [+12.7%, +21.7%] |
| FCFS vs Quantum-Only | Mann-Whitney U 检验 | 62500.0000 | 2.234e-83 | ✅ 是 | rank-biserial correlation=1.0000 | 2379.31 | [2349.44, 2409.18] | [+253.7%, +263.7%] |
| FCFS vs Classical-Only | Mann-Whitney U 检验 | 62500.0000 | 2.234e-83 | ✅ 是 | rank-biserial correlation=1.0000 | 2587.06 | [2576.52, 2597.60] | [+228.2%, +230.4%] |
| FCFS vs Greedy | Mann-Whitney U 检验 | 61865.0000 | 4.238e-80 | ✅ 是 | rank-biserial correlation=0.9797 | 1484.72 | [1406.63, 1562.81] | [+1514.3%, +89134.3%] |
| FCFS vs SJF | Mann-Whitney U 检验 | 34387.0000 | 0.05218 | ❌ 否 | rank-biserial correlation=0.1004 | -3.63 | [-21.93, 14.68] | [-1.5%, +1.0%] |
| FCFS vs PPO | Mann-Whitney U 检验 | 9121.0000 | 1.032e-42 | ✅ 是 | rank-biserial correlation=-0.7081 | -1288.18 | [-1432.60, -1143.75] | [-49.6%, -44.0%] |
| Random vs Quantum-Only | Mann-Whitney U 检验 | 62500.0000 | 2.234e-83 | ✅ 是 | rank-biserial correlation=1.0000 | 2167.71 | [2111.73, 2223.69] | [+229.0%, +242.2%] |
| Random vs Classical-Only | Welch t 检验 | 96.2289 | 8.986e-206 | ✅ 是 | Cohen's d=8.6070 | 2375.46 | [2326.86, 2424.07] | [+206.3%, +214.8%] |
| Random vs Greedy | Welch t 检验 | 27.3909 | 4.956e-95 | ✅ 是 | Cohen's d=2.4499 | 1273.12 | [1181.76, 1364.49] | [+1303.4%, +77754.2%] |
| Random vs SJF | Mann-Whitney U 检验 | 17965.0000 | 1.97e-16 | ✅ 是 | rank-biserial correlation=-0.4251 | -215.22 | [-265.98, -164.47] | [-18.0%, -11.3%] |
| Random vs PPO | Welch t 检验 | -19.3874 | 4.92e-55 | ✅ 是 | Cohen's d=-1.7341 | -1499.77 | [-1652.00, -1347.55] | [-57.4%, -51.6%] |
| Quantum-Only vs Classical-Only | Mann-Whitney U 检验 | 51619.0000 | 1.877e-36 | ✅ 是 | rank-biserial correlation=0.6518 | 207.75 | [177.91, 237.59] | [+15.9%, +21.1%] |
| Quantum-Only vs Greedy | Mann-Whitney U 检验 | 5782.5000 | 5.369e-56 | ✅ 是 | rank-biserial correlation=-0.8150 | -894.59 | [-977.52, -811.66] | [-56324.7%, -800.3%] |
| Quantum-Only vs SJF | Mann-Whitney U 检验 | 0.0000 | 2.234e-83 | ✅ 是 | rank-biserial correlation=-1.0000 | -2382.94 | [-2416.32, -2349.55] | [-165.0%, -160.8%] |
| Quantum-Only vs PPO | Mann-Whitney U 检验 | 27.0000 | 3.089e-83 | ✅ 是 | rank-biserial correlation=-0.9991 | -3667.49 | [-3814.59, -3520.38] | [-135.6%, -131.6%] |
| Classical-Only vs Greedy | Welch t 检验 | -27.7390 | 9.249e-79 | ✅ 是 | Cohen's d=-2.4810 | -1102.34 | [-1180.61, -1024.08] | [-68894.1%, -1005.6%] |
| Classical-Only vs SJF | Mann-Whitney U 检验 | 0.0000 | 2.234e-83 | ✅ 是 | rank-biserial correlation=-1.0000 | -2590.69 | [-2608.94, -2572.44] | [-178.2%, -176.1%] |
| Classical-Only vs PPO | Welch t 检验 | -52.7197 | 1.546e-137 | ✅ 是 | Cohen's d=-4.7154 | -3875.24 | [-4020.01, -3730.47] | [-143.3%, -139.0%] |
| Greedy vs SJF | Mann-Whitney U 检验 | 737.0000 | 1.404e-79 | ✅ 是 | rank-biserial correlation=-0.9764 | -1488.35 | [-1567.85, -1408.85] | [-107.0%, -96.5%] |
| Greedy vs PPO | Welch t 检验 | -33.2514 | 7.07e-115 | ✅ 是 | Cohen's d=-2.9741 | -2772.89 | [-2936.86, -2608.93] | [-103.7%, -98.2%] |
| SJF vs PPO | Mann-Whitney U 检验 | 9070.0000 | 6.675e-43 | ✅ 是 | rank-biserial correlation=-0.7098 | -1284.55 | [-1429.74, -1139.35] | [-49.5%, -43.8%] |

## 三、详细解释

### DQN vs FCFS

> 使用 Mann-Whitney U 检验 比较 DQN 与 FCFS：DQN 平均奖励高于FCFS 68.88（95% CI: [51.73, 86.03]）；统计量=42317.0000，p=7.343e-12。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.3541（中效应）。

### DQN vs Random

> 使用 Mann-Whitney U 检验 比较 DQN 与 Random：DQN 平均奖励高于Random 280.48（95% CI: [230.13, 330.83]）；统计量=48058.0000，p=2.357e-25。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.5379（大效应）。

### DQN vs Quantum-Only

> 使用 Mann-Whitney U 检验 比较 DQN 与 Quantum-Only：DQN 平均奖励高于Quantum-Only 2448.19（95% CI: [2415.43, 2480.95]）；统计量=62500.0000，p=2.234e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=1.0000（大效应）。

### DQN vs Classical-Only

> 使用 Mann-Whitney U 检验 比较 DQN 与 Classical-Only：DQN 平均奖励高于Classical-Only 2655.94（95% CI: [2638.85, 2673.03]）；统计量=62500.0000，p=2.234e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=1.0000（大效应）。

### DQN vs Greedy

> 使用 Mann-Whitney U 检验 比较 DQN 与 Greedy：DQN 平均奖励高于Greedy 1553.60（95% CI: [1474.36, 1632.84]）；统计量=62030.0000，p=6.05e-81。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.9850（大效应）。

### DQN vs SJF

> 使用 Mann-Whitney U 检验 比较 DQN 与 SJF：DQN 平均奖励高于SJF 65.25（95% CI: [42.53, 87.97]）；统计量=42120.0000，p=1.71e-11。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.3478（中效应）。

### DQN vs PPO

> 使用 Mann-Whitney U 检验 比较 DQN 与 PPO：DQN 平均奖励低于PPO 1219.30（95% CI: [-1364.35, -1074.24]）；统计量=9839.0000，p=4.258e-40。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.6852（大效应）。

### FCFS vs Random

> 使用 Mann-Whitney U 检验 比较 FCFS 与 Random：FCFS 平均奖励高于Random 211.60（95% CI: [163.08, 260.12]）；统计量=45479.0000，p=1.271e-18。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.4553（中效应）。

### FCFS vs Quantum-Only

> 使用 Mann-Whitney U 检验 比较 FCFS 与 Quantum-Only：FCFS 平均奖励高于Quantum-Only 2379.31（95% CI: [2349.44, 2409.18]）；统计量=62500.0000，p=2.234e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=1.0000（大效应）。

### FCFS vs Classical-Only

> 使用 Mann-Whitney U 检验 比较 FCFS 与 Classical-Only：FCFS 平均奖励高于Classical-Only 2587.06（95% CI: [2576.52, 2597.60]）；统计量=62500.0000，p=2.234e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=1.0000（大效应）。

### FCFS vs Greedy

> 使用 Mann-Whitney U 检验 比较 FCFS 与 Greedy：FCFS 平均奖励高于Greedy 1484.72（95% CI: [1406.63, 1562.81]）；统计量=61865.0000，p=4.238e-80。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.9797（大效应）。

### FCFS vs SJF

> 使用 Mann-Whitney U 检验 比较 FCFS 与 SJF：FCFS 平均奖励低于SJF 3.63（95% CI: [-21.93, 14.68]）；统计量=34387.0000，p=0.05218。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 rank-biserial correlation=0.1004（小效应）。

### FCFS vs PPO

> 使用 Mann-Whitney U 检验 比较 FCFS 与 PPO：FCFS 平均奖励低于PPO 1288.18（95% CI: [-1432.60, -1143.75]）；统计量=9121.0000，p=1.032e-42。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.7081（大效应）。

### Random vs Quantum-Only

> 使用 Mann-Whitney U 检验 比较 Random 与 Quantum-Only：Random 平均奖励高于Quantum-Only 2167.71（95% CI: [2111.73, 2223.69]）；统计量=62500.0000，p=2.234e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=1.0000（大效应）。

### Random vs Classical-Only

> 使用 Welch t 检验 比较 Random 与 Classical-Only：Random 平均奖励高于Classical-Only 2375.46（95% CI: [2326.86, 2424.07]）；统计量=96.2289，p=8.986e-206。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=8.6070（大效应）。

### Random vs Greedy

> 使用 Welch t 检验 比较 Random 与 Greedy：Random 平均奖励高于Greedy 1273.12（95% CI: [1181.76, 1364.49]）；统计量=27.3909，p=4.956e-95。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=2.4499（大效应）。

### Random vs SJF

> 使用 Mann-Whitney U 检验 比较 Random 与 SJF：Random 平均奖励低于SJF 215.22（95% CI: [-265.98, -164.47]）；统计量=17965.0000，p=1.97e-16。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.4251（中效应）。

### Random vs PPO

> 使用 Welch t 检验 比较 Random 与 PPO：Random 平均奖励低于PPO 1499.77（95% CI: [-1652.00, -1347.55]）；统计量=-19.3874，p=4.92e-55。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=-1.7341（大效应）。

### Quantum-Only vs Classical-Only

> 使用 Mann-Whitney U 检验 比较 Quantum-Only 与 Classical-Only：Quantum-Only 平均奖励高于Classical-Only 207.75（95% CI: [177.91, 237.59]）；统计量=51619.0000，p=1.877e-36。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.6518（大效应）。

### Quantum-Only vs Greedy

> 使用 Mann-Whitney U 检验 比较 Quantum-Only 与 Greedy：Quantum-Only 平均奖励低于Greedy 894.59（95% CI: [-977.52, -811.66]）；统计量=5782.5000，p=5.369e-56。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.8150（大效应）。

### Quantum-Only vs SJF

> 使用 Mann-Whitney U 检验 比较 Quantum-Only 与 SJF：Quantum-Only 平均奖励低于SJF 2382.94（95% CI: [-2416.32, -2349.55]）；统计量=0.0000，p=2.234e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-1.0000（大效应）。

### Quantum-Only vs PPO

> 使用 Mann-Whitney U 检验 比较 Quantum-Only 与 PPO：Quantum-Only 平均奖励低于PPO 3667.49（95% CI: [-3814.59, -3520.38]）；统计量=27.0000，p=3.089e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.9991（大效应）。

### Classical-Only vs Greedy

> 使用 Welch t 检验 比较 Classical-Only 与 Greedy：Classical-Only 平均奖励低于Greedy 1102.34（95% CI: [-1180.61, -1024.08]）；统计量=-27.7390，p=9.249e-79。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=-2.4810（大效应）。

### Classical-Only vs SJF

> 使用 Mann-Whitney U 检验 比较 Classical-Only 与 SJF：Classical-Only 平均奖励低于SJF 2590.69（95% CI: [-2608.94, -2572.44]）；统计量=0.0000，p=2.234e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-1.0000（大效应）。

### Classical-Only vs PPO

> 使用 Welch t 检验 比较 Classical-Only 与 PPO：Classical-Only 平均奖励低于PPO 3875.24（95% CI: [-4020.01, -3730.47]）；统计量=-52.7197，p=1.546e-137。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=-4.7154（大效应）。

### Greedy vs SJF

> 使用 Mann-Whitney U 检验 比较 Greedy 与 SJF：Greedy 平均奖励低于SJF 1488.35（95% CI: [-1567.85, -1408.85]）；统计量=737.0000，p=1.404e-79。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.9764（大效应）。

### Greedy vs PPO

> 使用 Welch t 检验 比较 Greedy 与 PPO：Greedy 平均奖励低于PPO 2772.89（95% CI: [-2936.86, -2608.93]）；统计量=-33.2514，p=7.07e-115。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=-2.9741（大效应）。

### SJF vs PPO

> 使用 Mann-Whitney U 检验 比较 SJF 与 PPO：SJF 平均奖励低于PPO 1284.55（95% CI: [-1429.74, -1139.35]）；统计量=9070.0000，p=6.675e-43。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.7098（大效应）。

## 四、检验方法说明

- **正态性检验**：n < 50 使用 Shapiro-Wilk，n ≥ 50 使用 D'Agostino K²
- **方差齐性检验**：Levene 检验
- **检验选择**：
  - 两组均正态且方差齐 → 独立样本 t 检验
  - 两组均正态但方差不齐 → Welch t 检验
  - 任一组非正态 → Mann-Whitney U 检验
- **效应量**：正态用 Cohen's d，非参数用 rank-biserial correlation
- **多重比较校正**：Bonferroni（校正 α = α / 比较次数）
- **置信区间**：均值差的 95% CI
- **Cohen's d 等级**：< 0.2 可忽略，0.2-0.5 小，0.5-0.8 中，≥ 0.8 大
- **rank-biserial 等级**：< 0.1 可忽略，0.1-0.3 小，0.3-0.5 中，≥ 0.5 大

---
*报告自动生成 | 数据源: C:\Users\HZR\Desktop\揭榜挂帅擂台赛\quantum-rl-scheduler\results\multiseed_evaluation\rewards_multiseed.json*