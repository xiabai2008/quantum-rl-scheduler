# 统计显著性检验报告（多Seed验证）

> 本报告为提交清单 `EXP_STAT` 必需文件，使用 50 次独立episode验证PPO相对于基线策略的统计显著性。

> **数据来源**: `results/multiseed_evaluation/rewards_multiseed.json`
> **显著性水平 α**: 0.05
> **比较次数**: 28（Bonferroni 校正后 α = 0.0018）

---


## 零、权威实验数字（多 Seed 验证）

> **实验配置**: 10 seeds × 5 episodes = 50 次独立运行
> **环境**: 10 维公平对比环境（Obs10Wrapper 截断 14 维原生环境，兼容所有已训练模型）
> **任务规模**: 每 episode 200 步，泊松到达 λ=0.5，量子任务占比 70%
> **PPO 模型**: `models/ppo_seed_42_v4/best_model.zip`（10维，Actor-Critic）
> **DQN 模型**: `models/dqn_fair_v2/seed_42/best_model.zip`（10维，Dueling DQN）
> **显著性水平**: α = 0.05（Bonferroni 校正）

| 排名 | 策略 | 平均奖励 | 标准差 | 标准误 | 提升 vs FCFS |
|:--:|:--|:--:|:--:|:--:|:--:|
| 1 | PPO | 2814.19 | 1121.19 | 158.56 | +92.4% |
| 2 | SJF | 1468.17 | 119.08 | 16.84 | +0.4% |
| 3 | FCFS | 1462.48 | 55.85 | 7.90 | 基线 |
| 4 | Random | 1275.91 | 411.84 | 58.24 | -12.8% |
| 5 | Greedy | -71.87 | 619.50 | 87.61 | -104.9% |
| 6 | DQN | -897.08 | 289.90 | 41.00 | -161.3% |
| 7 | Quantum-Only | -897.08 | 289.90 | 41.00 | -161.3% |
| 8 | Classical-Only | -1134.35 | 64.04 | 9.06 | -177.6% |

**核心结论：PPO 平均奖励 2814.19 vs FCFS 1462.48，提升 +92.4%**
（N=50 次独立episode，α=0.05，Bonferroni多重比较校正）

---

## 一、各策略奖励统计

| 策略 | 样本数 | 平均奖励 | 标准差 | 最小值 | 最大值 |
|:--|:--:|:--:|:--:|:--:|:--:|
| DQN | 50 | -897.08 | 289.90 | -1230.44 | 322.45 |
| FCFS | 50 | 1462.48 | 55.85 | 1350.98 | 1606.32 |
| Random | 50 | 1275.91 | 411.84 | 385.98 | 2169.79 |
| Quantum-Only | 50 | -897.08 | 289.90 | -1230.44 | 322.45 |
| Classical-Only | 50 | -1134.35 | 64.04 | -1252.49 | -946.13 |
| Greedy | 50 | -71.87 | 619.50 | -1214.34 | 1124.31 |
| SJF | 50 | 1468.17 | 119.08 | 1241.50 | 1829.36 |
| PPO | 50 | 2814.19 | 1121.19 | -21.39 | 5316.67 |

## 二、两两比较结果

| 对比 | 检验方法 | 统计量 | p 值 | 显著? | 效应量 | 均值差 | 95% CI |
|:--|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| DQN vs FCFS | Mann-Whitney U 检验 | 0.0000 | 7.066e-18 | ✅ 是 | rank-biserial correlation=-1.0000 | -2359.55 | [-2442.41, -2276.70] |
| DQN vs Random | Mann-Whitney U 检验 | 0.0000 | 7.066e-18 | ✅ 是 | rank-biserial correlation=-1.0000 | -2172.98 | [-2314.33, -2031.64] |
| DQN vs Quantum-Only | Mann-Whitney U 检验 | 1250.0000 | 1 | ❌ 否 | rank-biserial correlation=0.0000 | 0.00 | [-115.06, 115.06] |
| DQN vs Classical-Only | Mann-Whitney U 检验 | 2087.0000 | 8.085e-09 | ✅ 是 | rank-biserial correlation=0.6696 | 237.27 | [153.95, 320.60] |
| DQN vs Greedy | Mann-Whitney U 检验 | 321.5000 | 1.579e-10 | ✅ 是 | rank-biserial correlation=-0.7428 | -825.20 | [-1017.16, -633.25] |
| DQN vs SJF | Mann-Whitney U 检验 | 0.0000 | 7.066e-18 | ✅ 是 | rank-biserial correlation=-1.0000 | -2365.25 | [-2453.21, -2277.29] |
| DQN vs PPO | Mann-Whitney U 检验 | 1.0000 | 7.504e-18 | ✅ 是 | rank-biserial correlation=-0.9992 | -3711.26 | [-4036.27, -3386.25] |
| FCFS vs Random | Welch t 检验 | 3.1743 | 0.002551 | ❌ 否 | Cohen's d=0.6349 | 186.57 | [68.56, 304.58] |
| FCFS vs Quantum-Only | Mann-Whitney U 检验 | 2500.0000 | 7.066e-18 | ✅ 是 | rank-biserial correlation=1.0000 | 2359.55 | [2276.70, 2442.41] |
| FCFS vs Classical-Only | 独立样本 t 检验 | 216.0950 | 4.325e-133 | ✅ 是 | Cohen's d=43.2190 | 2596.83 | [2572.98, 2620.68] |
| FCFS vs Greedy | Welch t 检验 | 17.4427 | 7.464e-23 | ✅ 是 | Cohen's d=3.4885 | 1534.35 | [1357.65, 1711.06] |
| FCFS vs SJF | Welch t 检验 | -0.3062 | 0.7603 | ❌ 否 | Cohen's d=-0.0612 | -5.70 | [-42.80, 31.41] |
| FCFS vs PPO | Welch t 检验 | -8.5143 | 3.037e-11 | ✅ 是 | Cohen's d=-1.7029 | -1351.71 | [-1670.70, -1032.71] |
| Random vs Quantum-Only | Mann-Whitney U 检验 | 2500.0000 | 7.066e-18 | ✅ 是 | rank-biserial correlation=1.0000 | 2172.98 | [2031.64, 2314.33] |
| Random vs Classical-Only | Welch t 检验 | 40.8916 | 7.315e-41 | ✅ 是 | Cohen's d=8.1783 | 2410.26 | [2291.94, 2528.57] |
| Random vs Greedy | Welch t 检验 | 12.8112 | 1.429e-21 | ✅ 是 | Cohen's d=2.5622 | 1347.78 | [1138.62, 1556.94] |
| Random vs SJF | Welch t 检验 | -3.1713 | 0.002441 | ❌ 否 | Cohen's d=-0.6343 | -192.27 | [-313.67, -70.87] |
| Random vs PPO | Welch t 检验 | -9.1066 | 4.89e-13 | ✅ 是 | Cohen's d=-1.8213 | -1538.28 | [-1875.95, -1200.61] |
| Quantum-Only vs Classical-Only | Mann-Whitney U 检验 | 2087.0000 | 8.085e-09 | ✅ 是 | rank-biserial correlation=0.6696 | 237.27 | [153.95, 320.60] |
| Quantum-Only vs Greedy | Mann-Whitney U 检验 | 321.5000 | 1.579e-10 | ✅ 是 | rank-biserial correlation=-0.7428 | -825.20 | [-1017.16, -633.25] |
| Quantum-Only vs SJF | Mann-Whitney U 检验 | 0.0000 | 7.066e-18 | ✅ 是 | rank-biserial correlation=-1.0000 | -2365.25 | [-2453.21, -2277.29] |
| Quantum-Only vs PPO | Mann-Whitney U 检验 | 1.0000 | 7.504e-18 | ✅ 是 | rank-biserial correlation=-0.9992 | -3711.26 | [-4036.27, -3386.25] |
| Classical-Only vs Greedy | Welch t 检验 | -12.0630 | 1.994e-16 | ✅ 是 | Cohen's d=-2.4126 | -1062.48 | [-1239.38, -885.57] |
| Classical-Only vs SJF | Welch t 检验 | -136.1036 | 1.051e-91 | ✅ 是 | Cohen's d=-27.2207 | -2602.52 | [-2640.62, -2564.43] |
| Classical-Only vs PPO | Welch t 检验 | -24.8619 | 1.477e-29 | ✅ 是 | Cohen's d=-4.9724 | -3948.54 | [-4267.64, -3629.43] |
| Greedy vs SJF | Welch t 检验 | -17.2624 | 2.531e-23 | ✅ 是 | Cohen's d=-3.4525 | -1540.05 | [-1719.02, -1361.08] |
| Greedy vs PPO | Welch t 检验 | -15.9315 | 5.44e-26 | ✅ 是 | Cohen's d=-3.1863 | -2886.06 | [-3246.83, -2525.29] |
| SJF vs PPO | Welch t 检验 | -8.4415 | 3.423e-11 | ✅ 是 | Cohen's d=-1.6883 | -1346.01 | [-1666.26, -1025.76] |

## 三、详细解释

### DQN vs FCFS

> 使用 Mann-Whitney U 检验 比较 DQN 与 FCFS：DQN 平均奖励低于FCFS 2359.55（95% CI: [-2442.41, -2276.70]）；统计量=0.0000，p=7.066e-18。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-1.0000（大效应）。

### DQN vs Random

> 使用 Mann-Whitney U 检验 比较 DQN 与 Random：DQN 平均奖励低于Random 2172.98（95% CI: [-2314.33, -2031.64]）；统计量=0.0000，p=7.066e-18。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-1.0000（大效应）。

### DQN vs Quantum-Only

> 使用 Mann-Whitney U 检验 比较 DQN 与 Quantum-Only：DQN 平均奖励等于Quantum-Only 0.00（95% CI: [-115.06, 115.06]）；统计量=1250.0000，p=1。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 rank-biserial correlation=0.0000（可忽略）。

### DQN vs Classical-Only

> 使用 Mann-Whitney U 检验 比较 DQN 与 Classical-Only：DQN 平均奖励高于Classical-Only 237.27（95% CI: [153.95, 320.60]）；统计量=2087.0000，p=8.085e-09。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.6696（大效应）。

### DQN vs Greedy

> 使用 Mann-Whitney U 检验 比较 DQN 与 Greedy：DQN 平均奖励低于Greedy 825.20（95% CI: [-1017.16, -633.25]）；统计量=321.5000，p=1.579e-10。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.7428（大效应）。

### DQN vs SJF

> 使用 Mann-Whitney U 检验 比较 DQN 与 SJF：DQN 平均奖励低于SJF 2365.25（95% CI: [-2453.21, -2277.29]）；统计量=0.0000，p=7.066e-18。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-1.0000（大效应）。

### DQN vs PPO

> 使用 Mann-Whitney U 检验 比较 DQN 与 PPO：DQN 平均奖励低于PPO 3711.26（95% CI: [-4036.27, -3386.25]）；统计量=1.0000，p=7.504e-18。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.9992（大效应）。

### FCFS vs Random

> 使用 Welch t 检验 比较 FCFS 与 Random：FCFS 平均奖励高于Random 186.57（95% CI: [68.56, 304.58]）；统计量=3.1743，p=0.002551。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 Cohen's d=0.6349（中效应）。

### FCFS vs Quantum-Only

> 使用 Mann-Whitney U 检验 比较 FCFS 与 Quantum-Only：FCFS 平均奖励高于Quantum-Only 2359.55（95% CI: [2276.70, 2442.41]）；统计量=2500.0000，p=7.066e-18。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=1.0000（大效应）。

### FCFS vs Classical-Only

> 使用 独立样本 t 检验 比较 FCFS 与 Classical-Only：FCFS 平均奖励高于Classical-Only 2596.83（95% CI: [2572.98, 2620.68]）；统计量=216.0950，p=4.325e-133。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=43.2190（大效应）。

### FCFS vs Greedy

> 使用 Welch t 检验 比较 FCFS 与 Greedy：FCFS 平均奖励高于Greedy 1534.35（95% CI: [1357.65, 1711.06]）；统计量=17.4427，p=7.464e-23。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=3.4885（大效应）。

### FCFS vs SJF

> 使用 Welch t 检验 比较 FCFS 与 SJF：FCFS 平均奖励低于SJF 5.70（95% CI: [-42.80, 31.41]）；统计量=-0.3062，p=0.7603。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 Cohen's d=-0.0612（可忽略）。

### FCFS vs PPO

> 使用 Welch t 检验 比较 FCFS 与 PPO：FCFS 平均奖励低于PPO 1351.71（95% CI: [-1670.70, -1032.71]）；统计量=-8.5143，p=3.037e-11。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=-1.7029（大效应）。

### Random vs Quantum-Only

> 使用 Mann-Whitney U 检验 比较 Random 与 Quantum-Only：Random 平均奖励高于Quantum-Only 2172.98（95% CI: [2031.64, 2314.33]）；统计量=2500.0000，p=7.066e-18。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=1.0000（大效应）。

### Random vs Classical-Only

> 使用 Welch t 检验 比较 Random 与 Classical-Only：Random 平均奖励高于Classical-Only 2410.26（95% CI: [2291.94, 2528.57]）；统计量=40.8916，p=7.315e-41。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=8.1783（大效应）。

### Random vs Greedy

> 使用 Welch t 检验 比较 Random 与 Greedy：Random 平均奖励高于Greedy 1347.78（95% CI: [1138.62, 1556.94]）；统计量=12.8112，p=1.429e-21。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=2.5622（大效应）。

### Random vs SJF

> 使用 Welch t 检验 比较 Random 与 SJF：Random 平均奖励低于SJF 192.27（95% CI: [-313.67, -70.87]）；统计量=-3.1713，p=0.002441。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 Cohen's d=-0.6343（中效应）。

### Random vs PPO

> 使用 Welch t 检验 比较 Random 与 PPO：Random 平均奖励低于PPO 1538.28（95% CI: [-1875.95, -1200.61]）；统计量=-9.1066，p=4.89e-13。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=-1.8213（大效应）。

### Quantum-Only vs Classical-Only

> 使用 Mann-Whitney U 检验 比较 Quantum-Only 与 Classical-Only：Quantum-Only 平均奖励高于Classical-Only 237.27（95% CI: [153.95, 320.60]）；统计量=2087.0000，p=8.085e-09。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.6696（大效应）。

### Quantum-Only vs Greedy

> 使用 Mann-Whitney U 检验 比较 Quantum-Only 与 Greedy：Quantum-Only 平均奖励低于Greedy 825.20（95% CI: [-1017.16, -633.25]）；统计量=321.5000，p=1.579e-10。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.7428（大效应）。

### Quantum-Only vs SJF

> 使用 Mann-Whitney U 检验 比较 Quantum-Only 与 SJF：Quantum-Only 平均奖励低于SJF 2365.25（95% CI: [-2453.21, -2277.29]）；统计量=0.0000，p=7.066e-18。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-1.0000（大效应）。

### Quantum-Only vs PPO

> 使用 Mann-Whitney U 检验 比较 Quantum-Only 与 PPO：Quantum-Only 平均奖励低于PPO 3711.26（95% CI: [-4036.27, -3386.25]）；统计量=1.0000，p=7.504e-18。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.9992（大效应）。

### Classical-Only vs Greedy

> 使用 Welch t 检验 比较 Classical-Only 与 Greedy：Classical-Only 平均奖励低于Greedy 1062.48（95% CI: [-1239.38, -885.57]）；统计量=-12.0630，p=1.994e-16。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=-2.4126（大效应）。

### Classical-Only vs SJF

> 使用 Welch t 检验 比较 Classical-Only 与 SJF：Classical-Only 平均奖励低于SJF 2602.52（95% CI: [-2640.62, -2564.43]）；统计量=-136.1036，p=1.051e-91。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=-27.2207（大效应）。

### Classical-Only vs PPO

> 使用 Welch t 检验 比较 Classical-Only 与 PPO：Classical-Only 平均奖励低于PPO 3948.54（95% CI: [-4267.64, -3629.43]）；统计量=-24.8619，p=1.477e-29。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=-4.9724（大效应）。

### Greedy vs SJF

> 使用 Welch t 检验 比较 Greedy 与 SJF：Greedy 平均奖励低于SJF 1540.05（95% CI: [-1719.02, -1361.08]）；统计量=-17.2624，p=2.531e-23。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=-3.4525（大效应）。

### Greedy vs PPO

> 使用 Welch t 检验 比较 Greedy 与 PPO：Greedy 平均奖励低于PPO 2886.06（95% CI: [-3246.83, -2525.29]）；统计量=-15.9315，p=5.44e-26。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=-3.1863（大效应）。

### SJF vs PPO

> 使用 Welch t 检验 比较 SJF 与 PPO：SJF 平均奖励低于PPO 1346.01（95% CI: [-1666.26, -1025.76]）；统计量=-8.4415，p=3.423e-11。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=-1.6883（大效应）。

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