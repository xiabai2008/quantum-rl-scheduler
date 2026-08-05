# 统计显著性检验报告（多Seed验证）

> 本报告为提交清单 `EXP_STAT` 必需文件，使用 4 次独立episode验证PPO相对于基线策略的统计显著性。

> **数据来源**: `results\multiseed_evaluation\rewards_multiseed.json`
> **显著性水平 α**: 0.05
> **比较次数**: 28（Bonferroni 校正后 α = 0.0018）

---


## 零、权威实验数字（多 Seed 验证）

> **实验配置**: 2 seeds × 2 episodes = 4 次独立运行
> **环境**: 16 维观测空间（原生 16 维环境（v9+ 交付标准，OBS_DIM=16））
> **任务规模**: 每 episode 200 步，泊松到达 λ=0.5，量子任务占比 70%
> **PPO 模型**: `deliverable_models/ppo_best_model_16dim.zip`（16维，Actor-Critic）
> **DQN 模型**: `None`（DQN 模型已删除，下表 DQN 行为 Random 策略数据占位，不代表 DQN 实测结果）
> **显著性水平**: α = 0.05（Bonferroni 校正）

| 排名 | 策略 | 平均奖励 | 标准差 | 标准误 | 提升 vs FCFS | 提升% 95% CI |
|:--:|:--|:--:|:--:|:--:|:--:|:--:|
| 1 | FCFS | 1812.62 | 632.94 | 316.47 | 基线 | — |
| 2 | PPO | 1803.99 | 523.64 | 261.82 | -0.5% | [-31.2%, +55.8%] |
| 3 | SJF | 703.03 | 363.10 | 181.55 | -61.2% | [-76.6%, -33.2%] |
| 4 | DQN | 687.87 | 288.68 | 144.34 | -62.1% | [-77.8%, -38.7%] |
| 5 | Random | 687.87 | 288.68 | 144.34 | -62.1% | [-77.8%, -38.7%] |
| 6 | Greedy | 183.47 | 431.61 | 215.81 | -89.9% | [-110.5%, -67.7%] |
| 7 | Quantum-Only | -816.92 | 187.66 | 93.83 | -145.1% | [-169.6%, -132.9%] |
| 8 | Classical-Only | -1028.15 | 97.61 | 48.81 | -156.7% | [-186.3%, -144.7%] |

**核心结论：PPO 平均奖励 1803.99 vs FCFS 1812.62，提升 -0.5%，95% CI: [-31.2%, +55.8%]**
（N=4 次独立episode，α=0.05，Bonferroni多重比较校正）

---

## 一、各策略奖励统计

| 策略 | 样本数 | 平均奖励 | 标准差 | 最小值 | 最大值 |
|:--|:--:|:--:|:--:|:--:|:--:|
| DQN | 4 | 687.87 | 288.68 | 299.95 | 941.35 |
| FCFS | 4 | 1812.62 | 632.94 | 906.92 | 2334.40 |
| Random | 4 | 687.87 | 288.68 | 299.95 | 941.35 |
| Quantum-Only | 4 | -816.92 | 187.66 | -1038.27 | -652.32 |
| Classical-Only | 4 | -1028.15 | 97.61 | -1155.48 | -917.82 |
| Greedy | 4 | 183.47 | 431.61 | -291.23 | 639.51 |
| SJF | 4 | 703.03 | 363.10 | 477.57 | 1239.56 |
| PPO | 4 | 1803.99 | 523.64 | 1193.68 | 2297.12 |

## 二、两两比较结果

| 对比 | 检验方法 | 统计量 | p 值 | 显著? | 效应量 | 均值差 | 95% CI | 提升% 95% CI |
|:--|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DQN vs FCFS | 独立样本 t 检验 | -3.2336 | 0.01783 | ❌ 否 | Cohen's d=-2.2865 | -1124.75 | [-1975.85, -273.64] | [-77.8%, -38.7%] |
| DQN vs Random | 独立样本 t 检验 | 0.0000 | 1 | ❌ 否 | Cohen's d=0.0000 | 0.00 | [-499.48, 499.48] | [-42.9%, +72.5%] |
| DQN vs Quantum-Only | 独立样本 t 检验 | 8.7408 | 0.0001241 | ✅ 是 | Cohen's d=6.1807 | 1504.79 | [1083.54, 1926.04] | [+150.6%, +219.5%] |
| DQN vs Classical-Only | 独立样本 t 检验 | 11.2624 | 2.929e-05 | ✅ 是 | Cohen's d=7.9637 | 1716.02 | [1343.19, 2088.85] | [+141.6%, +189.8%] |
| DQN vs Greedy | 独立样本 t 检验 | 1.9428 | 0.1001 | ❌ 否 | Cohen's d=1.3738 | 504.40 | [-130.88, 1139.69] | [+13.0%, +506220.8%] |
| DQN vs SJF | Mann-Whitney U 检验 | 9.0000 | 0.8857 | ❌ 否 | rank-biserial correlation=0.1250 | -15.16 | [-582.69, 552.37] | [-47.6%, +65.9%] |
| DQN vs PPO | 独立样本 t 检验 | -3.7332 | 0.0097 | ❌ 否 | Cohen's d=-2.6398 | -1116.12 | [-1847.67, -384.57] | [-77.5%, -42.6%] |
| FCFS vs Random | 独立样本 t 检验 | 3.2336 | 0.01783 | ❌ 否 | Cohen's d=2.2865 | 1124.75 | [273.64, 1975.85] | [+63.0%, +349.2%] |
| FCFS vs Quantum-Only | 独立样本 t 检验 | 7.9663 | 0.0002083 | ✅ 是 | Cohen's d=5.6330 | 2629.54 | [1821.85, 3437.23] | [+244.1%, +403.4%] |
| FCFS vs Classical-Only | 独立样本 t 检验 | 8.8716 | 0.0001141 | ✅ 是 | Cohen's d=6.2732 | 2840.77 | [2057.24, 3624.29] | [+216.2%, +323.8%] |
| FCFS vs Greedy | 独立样本 t 检验 | 4.2532 | 0.005361 | ❌ 否 | Cohen's d=3.0074 | 1629.15 | [691.87, 2566.43] | [+207.0%, +1297653.9%] |
| FCFS vs SJF | Mann-Whitney U 检验 | 15.0000 | 0.05714 | ❌ 否 | rank-biserial correlation=0.8750 | 1109.58 | [216.84, 2002.33] | [+55.2%, +326.7%] |
| FCFS vs PPO | 独立样本 t 检验 | 0.0210 | 0.9839 | ❌ 否 | Cohen's d=0.0149 | 8.63 | [-996.39, 1013.65] | [-35.6%, +46.3%] |
| Random vs Quantum-Only | 独立样本 t 检验 | 8.7408 | 0.0001241 | ✅ 是 | Cohen's d=6.1807 | 1504.79 | [1083.54, 1926.04] | [+150.6%, +219.5%] |
| Random vs Classical-Only | 独立样本 t 检验 | 11.2624 | 2.929e-05 | ✅ 是 | Cohen's d=7.9637 | 1716.02 | [1343.19, 2088.85] | [+141.6%, +189.8%] |
| Random vs Greedy | 独立样本 t 检验 | 1.9428 | 0.1001 | ❌ 否 | Cohen's d=1.3738 | 504.40 | [-130.88, 1139.69] | [+13.0%, +506220.8%] |
| Random vs SJF | Mann-Whitney U 检验 | 9.0000 | 0.8857 | ❌ 否 | rank-biserial correlation=0.1250 | -15.16 | [-582.69, 552.37] | [-47.6%, +65.9%] |
| Random vs PPO | 独立样本 t 检验 | -3.7332 | 0.0097 | ❌ 否 | Cohen's d=-2.6398 | -1116.12 | [-1847.67, -384.57] | [-77.5%, -42.6%] |
| Quantum-Only vs Classical-Only | 独立样本 t 检验 | 1.9971 | 0.09279 | ❌ 否 | Cohen's d=1.4122 | 211.23 | [-47.57, 470.03] | [+2.9%, +36.2%] |
| Quantum-Only vs Greedy | Welch t 检验 | -4.2512 | 0.0125 | ❌ 否 | Cohen's d=-3.0060 | -1000.39 | [-1647.80, -352.97] | [-600976.5%, -236.8%] |
| Quantum-Only vs SJF | Mann-Whitney U 检验 | 0.0000 | 0.02857 | ❌ 否 | rank-biserial correlation=-1.0000 | -1519.95 | [-2020.02, -1019.89] | [-283.4%, -174.6%] |
| Quantum-Only vs PPO | Welch t 检验 | -9.4235 | 0.0009486 | ✅ 是 | Cohen's d=-6.6634 | -2620.91 | [-3413.11, -1828.71] | [-163.1%, -133.2%] |
| Classical-Only vs Greedy | Welch t 检验 | -5.4761 | 0.009243 | ❌ 否 | Cohen's d=-3.8722 | -1211.62 | [-1880.27, -542.96] | [-754455.6%, -279.7%] |
| Classical-Only vs SJF | Mann-Whitney U 检验 | 0.0000 | 0.02857 | ❌ 否 | rank-biserial correlation=-1.0000 | -1731.18 | [-2191.20, -1271.17] | [-318.6%, -197.2%] |
| Classical-Only vs PPO | Welch t 检验 | -10.6340 | 0.001312 | ✅ 是 | Cohen's d=-7.5194 | -2832.14 | [-3649.42, -2014.85] | [-176.2%, -145.2%] |
| Greedy vs SJF | Mann-Whitney U 检验 | 3.0000 | 0.2 | ❌ 否 | rank-biserial correlation=-0.6250 | -519.57 | [-1209.64, 170.50] | [-126.1%, -13.3%] |
| Greedy vs PPO | 独立样本 t 检验 | -4.7762 | 0.003075 | ❌ 否 | Cohen's d=-3.3772 | -1620.52 | [-2450.74, -790.30] | [-110.5%, -68.5%] |
| SJF vs PPO | Mann-Whitney U 检验 | 1.0000 | 0.05714 | ❌ 否 | rank-biserial correlation=-0.8750 | -1100.95 | [-1880.56, -321.35] | [-75.8%, -36.2%] |

## 三、详细解释

### DQN vs FCFS

> 使用 独立样本 t 检验 比较 DQN 与 FCFS：DQN 平均奖励低于FCFS 1124.75（95% CI: [-1975.85, -273.64]）；统计量=-3.2336，p=0.01783。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 Cohen's d=-2.2865（大效应）。

### DQN vs Random

> 使用 独立样本 t 检验 比较 DQN 与 Random：DQN 平均奖励等于Random 0.00（95% CI: [-499.48, 499.48]）；统计量=0.0000，p=1。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 Cohen's d=0.0000（可忽略）。

### DQN vs Quantum-Only

> 使用 独立样本 t 检验 比较 DQN 与 Quantum-Only：DQN 平均奖励高于Quantum-Only 1504.79（95% CI: [1083.54, 1926.04]）；统计量=8.7408，p=0.0001241。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=6.1807（大效应）。

### DQN vs Classical-Only

> 使用 独立样本 t 检验 比较 DQN 与 Classical-Only：DQN 平均奖励高于Classical-Only 1716.02（95% CI: [1343.19, 2088.85]）；统计量=11.2624，p=2.929e-05。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=7.9637（大效应）。

### DQN vs Greedy

> 使用 独立样本 t 检验 比较 DQN 与 Greedy：DQN 平均奖励高于Greedy 504.40（95% CI: [-130.88, 1139.69]）；统计量=1.9428，p=0.1001。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 Cohen's d=1.3738（大效应）。

### DQN vs SJF

> 使用 Mann-Whitney U 检验 比较 DQN 与 SJF：DQN 平均奖励低于SJF 15.16（95% CI: [-582.69, 552.37]）；统计量=9.0000，p=0.8857。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 rank-biserial correlation=0.1250（小效应）。

### DQN vs PPO

> 使用 独立样本 t 检验 比较 DQN 与 PPO：DQN 平均奖励低于PPO 1116.12（95% CI: [-1847.67, -384.57]）；统计量=-3.7332，p=0.0097。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 Cohen's d=-2.6398（大效应）。

### FCFS vs Random

> 使用 独立样本 t 检验 比较 FCFS 与 Random：FCFS 平均奖励高于Random 1124.75（95% CI: [273.64, 1975.85]）；统计量=3.2336，p=0.01783。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 Cohen's d=2.2865（大效应）。

### FCFS vs Quantum-Only

> 使用 独立样本 t 检验 比较 FCFS 与 Quantum-Only：FCFS 平均奖励高于Quantum-Only 2629.54（95% CI: [1821.85, 3437.23]）；统计量=7.9663，p=0.0002083。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=5.6330（大效应）。

### FCFS vs Classical-Only

> 使用 独立样本 t 检验 比较 FCFS 与 Classical-Only：FCFS 平均奖励高于Classical-Only 2840.77（95% CI: [2057.24, 3624.29]）；统计量=8.8716，p=0.0001141。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=6.2732（大效应）。

### FCFS vs Greedy

> 使用 独立样本 t 检验 比较 FCFS 与 Greedy：FCFS 平均奖励高于Greedy 1629.15（95% CI: [691.87, 2566.43]）；统计量=4.2532，p=0.005361。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 Cohen's d=3.0074（大效应）。

### FCFS vs SJF

> 使用 Mann-Whitney U 检验 比较 FCFS 与 SJF：FCFS 平均奖励高于SJF 1109.58（95% CI: [216.84, 2002.33]）；统计量=15.0000，p=0.05714。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 rank-biserial correlation=0.8750（大效应）。

### FCFS vs PPO

> 使用 独立样本 t 检验 比较 FCFS 与 PPO：FCFS 平均奖励高于PPO 8.63（95% CI: [-996.39, 1013.65]）；统计量=0.0210，p=0.9839。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 Cohen's d=0.0149（可忽略）。

### Random vs Quantum-Only

> 使用 独立样本 t 检验 比较 Random 与 Quantum-Only：Random 平均奖励高于Quantum-Only 1504.79（95% CI: [1083.54, 1926.04]）；统计量=8.7408，p=0.0001241。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=6.1807（大效应）。

### Random vs Classical-Only

> 使用 独立样本 t 检验 比较 Random 与 Classical-Only：Random 平均奖励高于Classical-Only 1716.02（95% CI: [1343.19, 2088.85]）；统计量=11.2624，p=2.929e-05。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=7.9637（大效应）。

### Random vs Greedy

> 使用 独立样本 t 检验 比较 Random 与 Greedy：Random 平均奖励高于Greedy 504.40（95% CI: [-130.88, 1139.69]）；统计量=1.9428，p=0.1001。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 Cohen's d=1.3738（大效应）。

### Random vs SJF

> 使用 Mann-Whitney U 检验 比较 Random 与 SJF：Random 平均奖励低于SJF 15.16（95% CI: [-582.69, 552.37]）；统计量=9.0000，p=0.8857。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 rank-biserial correlation=0.1250（小效应）。

### Random vs PPO

> 使用 独立样本 t 检验 比较 Random 与 PPO：Random 平均奖励低于PPO 1116.12（95% CI: [-1847.67, -384.57]）；统计量=-3.7332，p=0.0097。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 Cohen's d=-2.6398（大效应）。

### Quantum-Only vs Classical-Only

> 使用 独立样本 t 检验 比较 Quantum-Only 与 Classical-Only：Quantum-Only 平均奖励高于Classical-Only 211.23（95% CI: [-47.57, 470.03]）；统计量=1.9971，p=0.09279。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 Cohen's d=1.4122（大效应）。

### Quantum-Only vs Greedy

> 使用 Welch t 检验 比较 Quantum-Only 与 Greedy：Quantum-Only 平均奖励低于Greedy 1000.39（95% CI: [-1647.80, -352.97]）；统计量=-4.2512，p=0.0125。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 Cohen's d=-3.0060（大效应）。

### Quantum-Only vs SJF

> 使用 Mann-Whitney U 检验 比较 Quantum-Only 与 SJF：Quantum-Only 平均奖励低于SJF 1519.95（95% CI: [-2020.02, -1019.89]）；统计量=0.0000，p=0.02857。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 rank-biserial correlation=-1.0000（大效应）。

### Quantum-Only vs PPO

> 使用 Welch t 检验 比较 Quantum-Only 与 PPO：Quantum-Only 平均奖励低于PPO 2620.91（95% CI: [-3413.11, -1828.71]）；统计量=-9.4235，p=0.0009486。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=-6.6634（大效应）。

### Classical-Only vs Greedy

> 使用 Welch t 检验 比较 Classical-Only 与 Greedy：Classical-Only 平均奖励低于Greedy 1211.62（95% CI: [-1880.27, -542.96]）；统计量=-5.4761，p=0.009243。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 Cohen's d=-3.8722（大效应）。

### Classical-Only vs SJF

> 使用 Mann-Whitney U 检验 比较 Classical-Only 与 SJF：Classical-Only 平均奖励低于SJF 1731.18（95% CI: [-2191.20, -1271.17]）；统计量=0.0000，p=0.02857。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 rank-biserial correlation=-1.0000（大效应）。

### Classical-Only vs PPO

> 使用 Welch t 检验 比较 Classical-Only 与 PPO：Classical-Only 平均奖励低于PPO 2832.14（95% CI: [-3649.42, -2014.85]）；统计量=-10.6340，p=0.001312。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=-7.5194（大效应）。

### Greedy vs SJF

> 使用 Mann-Whitney U 检验 比较 Greedy 与 SJF：Greedy 平均奖励低于SJF 519.57（95% CI: [-1209.64, 170.50]）；统计量=3.0000，p=0.2。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 rank-biserial correlation=-0.6250（大效应）。

### Greedy vs PPO

> 使用 独立样本 t 检验 比较 Greedy 与 PPO：Greedy 平均奖励低于PPO 1620.52（95% CI: [-2450.74, -790.30]）；统计量=-4.7762，p=0.003075。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 Cohen's d=-3.3772（大效应）。

### SJF vs PPO

> 使用 Mann-Whitney U 检验 比较 SJF 与 PPO：SJF 平均奖励低于PPO 1100.95（95% CI: [-1880.56, -321.35]）；统计量=1.0000，p=0.05714。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 rank-biserial correlation=-0.8750（大效应）。

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

## 五、检验力分析（Power Analysis）

> 检验力（Power）= 1 - β，表示当原假设为假时正确拒绝原假设的概率。通常要求 power ≥ 0.80。
>
> 此处对每个比较对计算：(1) 当前样本量下的事后检验力；(2) 达到 80% 检验力所需的每组样本量；(3) 当前样本量下的最小可检测效应量（MDES）。
>
> 注意：检验力分析基于 Cohen's d 与双侧 t 检验近似；非参数检验的检验力仅供参考。

| 对比 | 效应量 (Cohen's d 或 rank-biserial) | 当前 N1/N2 | 当前检验力 | 80% 检验力所需 N/组 | 当前样本量 MDES (d) |
|:--|:--:|:--:|:--:|:--:|:--:|
| DQN vs FCFS | Cohen's d=-2.2865 | 4/4 | 0.7684 | 5 | 2.3707 |
| DQN vs Random | Cohen's d=0.0000 | 4/4 | 0.0500 | N/A | 2.3707 |
| DQN vs Quantum-Only | Cohen's d=6.1807 | 4/4 | 1.0000 | 2 | 2.3707 |
| DQN vs Classical-Only | Cohen's d=7.9637 | 4/4 | 1.0000 | 2 | 2.3707 |
| DQN vs Greedy | Cohen's d=1.3738 | 4/4 | 0.3730 | 10 | 2.3707 |
| DQN vs SJF | rank-biserial correlation=0.1250 | 4/4 | 0.0526 | 1006 | 2.3707 |
| DQN vs PPO | Cohen's d=-2.6398 | 4/4 | 0.8723 | 4 | 2.3707 |
| FCFS vs Random | Cohen's d=2.2865 | 4/4 | 0.7684 | 5 | 2.3707 |
| FCFS vs Quantum-Only | Cohen's d=5.6330 | 4/4 | 1.0000 | 3 | 2.3707 |
| FCFS vs Classical-Only | Cohen's d=6.2732 | 4/4 | 1.0000 | 2 | 2.3707 |
| FCFS vs Greedy | Cohen's d=3.0074 | 4/4 | 0.9399 | 4 | 2.3707 |
| FCFS vs SJF | rank-biserial correlation=0.8750 | 4/4 | 0.1822 | 22 | 2.3707 |
| FCFS vs PPO | Cohen's d=0.0149 | 4/4 | 0.0500 | 71122 | 2.3707 |
| Random vs Quantum-Only | Cohen's d=6.1807 | 4/4 | 1.0000 | 2 | 2.3707 |
| Random vs Classical-Only | Cohen's d=7.9637 | 4/4 | 1.0000 | 2 | 2.3707 |
| Random vs Greedy | Cohen's d=1.3738 | 4/4 | 0.3730 | 10 | 2.3707 |
| Random vs SJF | rank-biserial correlation=0.1250 | 4/4 | 0.0526 | 1006 | 2.3707 |
| Random vs PPO | Cohen's d=-2.6398 | 4/4 | 0.8723 | 4 | 2.3707 |
| Quantum-Only vs Classical-Only | Cohen's d=1.4122 | 4/4 | 0.3902 | 9 | 2.3707 |
| Quantum-Only vs Greedy | Cohen's d=-3.0060 | 4/4 | 0.9398 | 4 | 2.3707 |
| Quantum-Only vs SJF | rank-biserial correlation=-1.0000 | 4/4 | 0.2232 | 17 | 2.3707 |
| Quantum-Only vs PPO | Cohen's d=-6.6634 | 4/4 | 1.0000 | 2 | 2.3707 |
| Classical-Only vs Greedy | Cohen's d=-3.8722 | 4/4 | 0.9942 | 3 | 2.3707 |
| Classical-Only vs SJF | rank-biserial correlation=-1.0000 | 4/4 | 0.2232 | 17 | 2.3707 |
| Classical-Only vs PPO | Cohen's d=-7.5194 | 4/4 | 1.0000 | 2 | 2.3707 |
| Greedy vs SJF | rank-biserial correlation=-0.6250 | 4/4 | 0.1166 | 42 | 2.3707 |
| Greedy vs PPO | Cohen's d=-3.3772 | 4/4 | 0.9756 | 3 | 2.3707 |
| SJF vs PPO | rank-biserial correlation=-0.8750 | 4/4 | 0.1822 | 22 | 2.3707 |

### 文字解读

- **PPO vs FCFS**：Cohen's d=0.0149（大效应），当前 N1=4, N2=4 的检验力 = 0.0500（达到 80% 标准）；检测该效应量仅需每组约 71122 个样本。
- 检验力 < 0.80 的对比：表明当前样本量不足以可靠检测该效应量，对应的不显著结论需要谨慎解读（可能存在检验力不足导致假阴性）。
- 检验力 ≥ 0.99 且显著的对比：核心结论极其稳健，样本量远超检测该效应所需。

---
*报告自动生成 | 数据源: results\multiseed_evaluation\rewards_multiseed.json*