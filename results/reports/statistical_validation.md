# 统计显著性检验报告（多Seed验证）

> 本报告为提交清单 `EXP_STAT` 必需文件，使用 250 次独立episode验证PPO相对于基线策略的统计显著性。

> **数据来源**: `results\multiseed_evaluation\rewards_multiseed.json`
> **显著性水平 α**: 0.05
> **比较次数**: 28（Bonferroni 校正后 α = 0.0018）
> **标准差口径**: 本报告标准差为样本标准差（ddof=1）；statistics.yaml/AGENTS.md 权威口径为总体标准差（ddof=0，PPO 557.25 / FCFS 502.95），两口径均合法，数值差异为自由度修正。

---


## 零、权威实验数字（多 Seed 验证）

> **实验配置**: 50 seeds × 5 episodes = 250 次独立运行
> **环境**: 16 维观测空间（原生 16 维环境（v9+ 交付标准，OBS_DIM=16））
> **任务规模**: 每 episode 200 步，泊松到达 λ=0.5，量子任务占比 70%
> **PPO 模型**: `deliverable_models/ppo_best_model_16dim.zip`（16维，Actor-Critic）
> **DQN 模型**: `None`（DQN 模型已删除，下表 DQN 行为 Random 策略数据占位，不代表 DQN 实测结果）
> **显著性水平**: α = 0.05（Bonferroni 校正）

| 排名 | 策略 | 平均奖励 | 标准差 | 标准误 | 提升 vs FCFS | 提升% 95% CI |
|:--:|:--|:--:|:--:|:--:|:--:|:--:|
| 1 | PPO | 1982.69 | 558.37 | 35.31 | +20.2% | [+14.3%, +26.7%] |
| 2 | FCFS | 1648.91 | 503.96 | 31.87 | 基线 | — |
| 3 | SJF | 774.86 | 276.30 | 17.47 | -53.0% | [-55.7%, -50.2%] |
| 4 | DQN | 602.37 | 262.61 | 16.61 | -63.5% | [-65.8%, -61.0%] |
| 5 | Random | 602.37 | 262.61 | 16.61 | -63.5% | [-65.8%, -61.0%] |
| 6 | Greedy | 80.71 | 550.22 | 34.80 | -95.1% | [-99.2%, -91.0%] |
| 7 | Quantum-Only | -826.59 | 263.63 | 16.67 | -150.1% | [-152.9%, -147.5%] |
| 8 | Classical-Only | -1075.49 | 75.04 | 4.75 | -165.2% | [-167.9%, -162.8%] |

**核心结论：PPO 平均奖励 1982.69 vs FCFS 1648.91，提升 +20.2%，95% CI: [+14.3%, +26.7%]**
（N=250 次独立episode，α=0.05，Bonferroni多重比较校正）

---

## 一、各策略奖励统计

| 策略 | 样本数 | 平均奖励 | 标准差 | 最小值 | 最大值 |
|:--|:--:|:--:|:--:|:--:|:--:|
| DQN | 250 | 602.37 | 262.61 | -121.52 | 1365.36 |
| FCFS | 250 | 1648.91 | 503.96 | 288.56 | 2813.56 |
| Random | 250 | 602.37 | 262.61 | -121.52 | 1365.36 |
| Quantum-Only | 250 | -826.59 | 263.63 | -1250.65 | 87.27 |
| Classical-Only | 250 | -1075.49 | 75.04 | -1264.92 | -876.90 |
| Greedy | 250 | 80.71 | 550.22 | -1200.11 | 1352.60 |
| SJF | 250 | 774.86 | 276.30 | 206.66 | 1691.67 |
| PPO | 250 | 1982.69 | 558.37 | 291.78 | 3115.95 |

## 二、两两比较结果

| 对比 | 检验方法 | 统计量 | p 值 | 显著? | 效应量 | 均值差 | 95% CI | 提升% 95% CI |
|:--|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DQN vs FCFS | Mann-Whitney U 检验 | 2659.0000 | 4.253e-70 | ✅ 是 | rank-biserial correlation=-0.9149 | -1046.55 | [-1117.16, -975.93] | [-65.8%, -61.0%] |
| DQN vs Random | 独立样本 t 检验 | 0.0000 | 1 | ❌ 否 | Cohen's d=0.0000 | 0.00 | [-46.15, 46.15] | [-7.3%, +7.7%] |
| DQN vs Quantum-Only | Mann-Whitney U 检验 | 62480.0000 | 2.841e-83 | ✅ 是 | rank-biserial correlation=0.9994 | 1428.96 | [1382.72, 1475.20] | [+168.1%, +177.8%] |
| DQN vs Classical-Only | Welch t 检验 | 97.1330 | 6.564e-223 | ✅ 是 | Cohen's d=8.6878 | 1677.86 | [1643.86, 1711.86] | [+152.9%, +159.0%] |
| DQN vs Greedy | Mann-Whitney U 检验 | 49773.0000 | 1.943e-30 | ✅ 是 | rank-biserial correlation=0.5927 | 521.66 | [445.90, 597.42] | [+304.3%, +4024.6%] |
| DQN vs SJF | Mann-Whitney U 检验 | 21041.0000 | 2.621e-10 | ✅ 是 | rank-biserial correlation=-0.3267 | -172.49 | [-219.86, -125.12] | [-27.4%, -16.7%] |
| DQN vs PPO | Mann-Whitney U 检验 | 1506.0000 | 1.034e-75 | ✅ 是 | rank-biserial correlation=-0.9518 | -1380.33 | [-1457.00, -1303.65] | [-71.5%, -67.6%] |
| FCFS vs Random | Mann-Whitney U 检验 | 59841.0000 | 4.253e-70 | ✅ 是 | rank-biserial correlation=0.9149 | 1046.55 | [975.93, 1117.16] | [+156.7%, +192.6%] |
| FCFS vs Quantum-Only | Mann-Whitney U 检验 | 62500.0000 | 2.234e-83 | ✅ 是 | rank-biserial correlation=1.0000 | 2475.51 | [2404.84, 2546.18] | [+288.8%, +310.8%] |
| FCFS vs Classical-Only | Mann-Whitney U 检验 | 62500.0000 | 2.234e-83 | ✅ 是 | rank-biserial correlation=1.0000 | 2724.41 | [2661.09, 2787.72] | [+247.5%, +259.3%] |
| FCFS vs Greedy | Mann-Whitney U 检验 | 61426.0000 | 7.148e-78 | ✅ 是 | rank-biserial correlation=0.9656 | 1568.21 | [1475.49, 1660.92] | [+1010.8%, +11352.0%] |
| FCFS vs SJF | Mann-Whitney U 检验 | 57725.0000 | 2.282e-60 | ✅ 是 | rank-biserial correlation=0.8472 | 874.06 | [802.64, 945.47] | [+100.7%, +125.9%] |
| FCFS vs PPO | Mann-Whitney U 检验 | 19870.0000 | 1.86e-12 | ✅ 是 | rank-biserial correlation=-0.3642 | -333.78 | [-427.24, -240.31] | [-21.0%, -12.4%] |
| Random vs Quantum-Only | Mann-Whitney U 检验 | 62480.0000 | 2.841e-83 | ✅ 是 | rank-biserial correlation=0.9994 | 1428.96 | [1382.72, 1475.20] | [+168.1%, +177.8%] |
| Random vs Classical-Only | Welch t 检验 | 97.1330 | 6.564e-223 | ✅ 是 | Cohen's d=8.6878 | 1677.86 | [1643.86, 1711.86] | [+152.9%, +159.0%] |
| Random vs Greedy | Mann-Whitney U 检验 | 49773.0000 | 1.943e-30 | ✅ 是 | rank-biserial correlation=0.5927 | 521.66 | [445.90, 597.42] | [+304.3%, +4024.6%] |
| Random vs SJF | Mann-Whitney U 检验 | 21041.0000 | 2.621e-10 | ✅ 是 | rank-biserial correlation=-0.3267 | -172.49 | [-219.86, -125.12] | [-27.4%, -16.7%] |
| Random vs PPO | Mann-Whitney U 检验 | 1506.0000 | 1.034e-75 | ✅ 是 | rank-biserial correlation=-0.9518 | -1380.33 | [-1457.00, -1303.65] | [-71.5%, -67.6%] |
| Quantum-Only vs Classical-Only | Mann-Whitney U 检验 | 51673.0000 | 1.227e-36 | ✅ 是 | rank-biserial correlation=0.6535 | 248.90 | [214.84, 282.96] | [+20.1%, +26.2%] |
| Quantum-Only vs Greedy | Mann-Whitney U 检验 | 5422.5000 | 1.539e-57 | ✅ 是 | rank-biserial correlation=-0.8265 | -907.30 | [-983.12, -831.49] | [-5700.4%, -656.8%] |
| Quantum-Only vs SJF | Mann-Whitney U 检验 | 0.0000 | 2.234e-83 | ✅ 是 | rank-biserial correlation=-1.0000 | -1601.45 | [-1648.90, -1554.00] | [-213.2%, -200.6%] |
| Quantum-Only vs PPO | Mann-Whitney U 检验 | 0.0000 | 2.234e-83 | ✅ 是 | rank-biserial correlation=-1.0000 | -2809.29 | [-2886.02, -2732.56] | [-143.9%, -139.5%] |
| Classical-Only vs Greedy | Mann-Whitney U 检验 | 1836.0000 | 4.404e-74 | ✅ 是 | rank-biserial correlation=-0.9412 | -1156.20 | [-1225.20, -1087.20] | [-7385.9%, -827.3%] |
| Classical-Only vs SJF | Mann-Whitney U 检验 | 0.0000 | 2.234e-83 | ✅ 是 | rank-biserial correlation=-1.0000 | -1850.35 | [-1885.92, -1814.77] | [-245.3%, -232.7%] |
| Classical-Only vs PPO | Mann-Whitney U 检验 | 0.0000 | 2.234e-83 | ✅ 是 | rank-biserial correlation=-1.0000 | -3058.19 | [-3128.19, -2988.18] | [-156.3%, -152.3%] |
| Greedy vs SJF | Mann-Whitney U 检验 | 7907.0000 | 2.493e-47 | ✅ 是 | rank-biserial correlation=-0.7470 | -694.15 | [-770.65, -617.64] | [-98.4%, -80.8%] |
| Greedy vs PPO | Mann-Whitney U 检验 | 617.0000 | 3.429e-80 | ✅ 是 | rank-biserial correlation=-0.9803 | -1901.99 | [-1999.40, -1804.58] | [-99.4%, -92.5%] |
| SJF vs PPO | Mann-Whitney U 检验 | 2537.0000 | 1.11e-70 | ✅ 是 | rank-biserial correlation=-0.9188 | -1207.84 | [-1285.25, -1130.42] | [-63.1%, -58.7%] |

## 三、详细解释

### DQN vs FCFS

> 使用 Mann-Whitney U 检验 比较 DQN 与 FCFS：DQN 平均奖励低于FCFS 1046.55（95% CI: [-1117.16, -975.93]）；统计量=2659.0000，p=4.253e-70。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.9149（大效应）。

### DQN vs Random

> 使用 独立样本 t 检验 比较 DQN 与 Random：DQN 平均奖励等于Random 0.00（95% CI: [-46.15, 46.15]）；统计量=0.0000，p=1。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 Cohen's d=0.0000（可忽略）。

### DQN vs Quantum-Only

> 使用 Mann-Whitney U 检验 比较 DQN 与 Quantum-Only：DQN 平均奖励高于Quantum-Only 1428.96（95% CI: [1382.72, 1475.20]）；统计量=62480.0000，p=2.841e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.9994（大效应）。

### DQN vs Classical-Only

> 使用 Welch t 检验 比较 DQN 与 Classical-Only：DQN 平均奖励高于Classical-Only 1677.86（95% CI: [1643.86, 1711.86]）；统计量=97.1330，p=6.564e-223。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=8.6878（大效应）。

### DQN vs Greedy

> 使用 Mann-Whitney U 检验 比较 DQN 与 Greedy：DQN 平均奖励高于Greedy 521.66（95% CI: [445.90, 597.42]）；统计量=49773.0000，p=1.943e-30。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.5927（大效应）。

### DQN vs SJF

> 使用 Mann-Whitney U 检验 比较 DQN 与 SJF：DQN 平均奖励低于SJF 172.49（95% CI: [-219.86, -125.12]）；统计量=21041.0000，p=2.621e-10。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.3267（中效应）。

### DQN vs PPO

> 使用 Mann-Whitney U 检验 比较 DQN 与 PPO：DQN 平均奖励低于PPO 1380.33（95% CI: [-1457.00, -1303.65]）；统计量=1506.0000，p=1.034e-75。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.9518（大效应）。

### FCFS vs Random

> 使用 Mann-Whitney U 检验 比较 FCFS 与 Random：FCFS 平均奖励高于Random 1046.55（95% CI: [975.93, 1117.16]）；统计量=59841.0000，p=4.253e-70。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.9149（大效应）。

### FCFS vs Quantum-Only

> 使用 Mann-Whitney U 检验 比较 FCFS 与 Quantum-Only：FCFS 平均奖励高于Quantum-Only 2475.51（95% CI: [2404.84, 2546.18]）；统计量=62500.0000，p=2.234e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=1.0000（大效应）。

### FCFS vs Classical-Only

> 使用 Mann-Whitney U 检验 比较 FCFS 与 Classical-Only：FCFS 平均奖励高于Classical-Only 2724.41（95% CI: [2661.09, 2787.72]）；统计量=62500.0000，p=2.234e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=1.0000（大效应）。

### FCFS vs Greedy

> 使用 Mann-Whitney U 检验 比较 FCFS 与 Greedy：FCFS 平均奖励高于Greedy 1568.21（95% CI: [1475.49, 1660.92]）；统计量=61426.0000，p=7.148e-78。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.9656（大效应）。

### FCFS vs SJF

> 使用 Mann-Whitney U 检验 比较 FCFS 与 SJF：FCFS 平均奖励高于SJF 874.06（95% CI: [802.64, 945.47]）；统计量=57725.0000，p=2.282e-60。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.8472（大效应）。

### FCFS vs PPO

> 使用 Mann-Whitney U 检验 比较 FCFS 与 PPO：FCFS 平均奖励低于PPO 333.78（95% CI: [-427.24, -240.31]）；统计量=19870.0000，p=1.86e-12。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.3642（中效应）。

### Random vs Quantum-Only

> 使用 Mann-Whitney U 检验 比较 Random 与 Quantum-Only：Random 平均奖励高于Quantum-Only 1428.96（95% CI: [1382.72, 1475.20]）；统计量=62480.0000，p=2.841e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.9994（大效应）。

### Random vs Classical-Only

> 使用 Welch t 检验 比较 Random 与 Classical-Only：Random 平均奖励高于Classical-Only 1677.86（95% CI: [1643.86, 1711.86]）；统计量=97.1330，p=6.564e-223。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=8.6878（大效应）。

### Random vs Greedy

> 使用 Mann-Whitney U 检验 比较 Random 与 Greedy：Random 平均奖励高于Greedy 521.66（95% CI: [445.90, 597.42]）；统计量=49773.0000，p=1.943e-30。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.5927（大效应）。

### Random vs SJF

> 使用 Mann-Whitney U 检验 比较 Random 与 SJF：Random 平均奖励低于SJF 172.49（95% CI: [-219.86, -125.12]）；统计量=21041.0000，p=2.621e-10。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.3267（中效应）。

### Random vs PPO

> 使用 Mann-Whitney U 检验 比较 Random 与 PPO：Random 平均奖励低于PPO 1380.33（95% CI: [-1457.00, -1303.65]）；统计量=1506.0000，p=1.034e-75。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.9518（大效应）。

### Quantum-Only vs Classical-Only

> 使用 Mann-Whitney U 检验 比较 Quantum-Only 与 Classical-Only：Quantum-Only 平均奖励高于Classical-Only 248.90（95% CI: [214.84, 282.96]）；统计量=51673.0000，p=1.227e-36。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.6535（大效应）。

### Quantum-Only vs Greedy

> 使用 Mann-Whitney U 检验 比较 Quantum-Only 与 Greedy：Quantum-Only 平均奖励低于Greedy 907.30（95% CI: [-983.12, -831.49]）；统计量=5422.5000，p=1.539e-57。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.8265（大效应）。

### Quantum-Only vs SJF

> 使用 Mann-Whitney U 检验 比较 Quantum-Only 与 SJF：Quantum-Only 平均奖励低于SJF 1601.45（95% CI: [-1648.90, -1554.00]）；统计量=0.0000，p=2.234e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-1.0000（大效应）。

### Quantum-Only vs PPO

> 使用 Mann-Whitney U 检验 比较 Quantum-Only 与 PPO：Quantum-Only 平均奖励低于PPO 2809.29（95% CI: [-2886.02, -2732.56]）；统计量=0.0000，p=2.234e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-1.0000（大效应）。

### Classical-Only vs Greedy

> 使用 Mann-Whitney U 检验 比较 Classical-Only 与 Greedy：Classical-Only 平均奖励低于Greedy 1156.20（95% CI: [-1225.20, -1087.20]）；统计量=1836.0000，p=4.404e-74。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.9412（大效应）。

### Classical-Only vs SJF

> 使用 Mann-Whitney U 检验 比较 Classical-Only 与 SJF：Classical-Only 平均奖励低于SJF 1850.35（95% CI: [-1885.92, -1814.77]）；统计量=0.0000，p=2.234e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-1.0000（大效应）。

### Classical-Only vs PPO

> 使用 Mann-Whitney U 检验 比较 Classical-Only 与 PPO：Classical-Only 平均奖励低于PPO 3058.19（95% CI: [-3128.19, -2988.18]）；统计量=0.0000，p=2.234e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-1.0000（大效应）。

### Greedy vs SJF

> 使用 Mann-Whitney U 检验 比较 Greedy 与 SJF：Greedy 平均奖励低于SJF 694.15（95% CI: [-770.65, -617.64]）；统计量=7907.0000，p=2.493e-47。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.7470（大效应）。

### Greedy vs PPO

> 使用 Mann-Whitney U 检验 比较 Greedy 与 PPO：Greedy 平均奖励低于PPO 1901.99（95% CI: [-1999.40, -1804.58]）；统计量=617.0000，p=3.429e-80。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.9803（大效应）。

### SJF vs PPO

> 使用 Mann-Whitney U 检验 比较 SJF 与 PPO：SJF 平均奖励低于PPO 1207.84（95% CI: [-1285.25, -1130.42]）；统计量=2537.0000，p=1.11e-70。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.9188（大效应）。

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
| DQN vs FCFS | rank-biserial correlation=-0.9149 | 250/250 | 1.0000 | 20 | 0.2511 |
| DQN vs Random | Cohen's d=0.0000 | 250/250 | 0.0500 | N/A | 0.2511 |
| DQN vs Quantum-Only | rank-biserial correlation=0.9994 | 250/250 | 1.0000 | 17 | 0.2511 |
| DQN vs Classical-Only | Cohen's d=8.6878 | 250/250 | 1.0000 | 2 | 0.2511 |
| DQN vs Greedy | rank-biserial correlation=0.5927 | 250/250 | 1.0000 | 46 | 0.2511 |
| DQN vs SJF | rank-biserial correlation=-0.3267 | 250/250 | 0.9541 | 149 | 0.2511 |
| DQN vs PPO | rank-biserial correlation=-0.9518 | 250/250 | 1.0000 | 19 | 0.2511 |
| FCFS vs Random | rank-biserial correlation=0.9149 | 250/250 | 1.0000 | 20 | 0.2511 |
| FCFS vs Quantum-Only | rank-biserial correlation=1.0000 | 250/250 | 1.0000 | 17 | 0.2511 |
| FCFS vs Classical-Only | rank-biserial correlation=1.0000 | 250/250 | 1.0000 | 17 | 0.2511 |
| FCFS vs Greedy | rank-biserial correlation=0.9656 | 250/250 | 1.0000 | 18 | 0.2511 |
| FCFS vs SJF | rank-biserial correlation=0.8472 | 250/250 | 1.0000 | 23 | 0.2511 |
| FCFS vs PPO | rank-biserial correlation=-0.3642 | 250/250 | 0.9823 | 120 | 0.2511 |
| Random vs Quantum-Only | rank-biserial correlation=0.9994 | 250/250 | 1.0000 | 17 | 0.2511 |
| Random vs Classical-Only | Cohen's d=8.6878 | 250/250 | 1.0000 | 2 | 0.2511 |
| Random vs Greedy | rank-biserial correlation=0.5927 | 250/250 | 1.0000 | 46 | 0.2511 |
| Random vs SJF | rank-biserial correlation=-0.3267 | 250/250 | 0.9541 | 149 | 0.2511 |
| Random vs PPO | rank-biserial correlation=-0.9518 | 250/250 | 1.0000 | 19 | 0.2511 |
| Quantum-Only vs Classical-Only | rank-biserial correlation=0.6535 | 250/250 | 1.0000 | 38 | 0.2511 |
| Quantum-Only vs Greedy | rank-biserial correlation=-0.8265 | 250/250 | 1.0000 | 24 | 0.2511 |
| Quantum-Only vs SJF | rank-biserial correlation=-1.0000 | 250/250 | 1.0000 | 17 | 0.2511 |
| Quantum-Only vs PPO | rank-biserial correlation=-1.0000 | 250/250 | 1.0000 | 17 | 0.2511 |
| Classical-Only vs Greedy | rank-biserial correlation=-0.9412 | 250/250 | 1.0000 | 19 | 0.2511 |
| Classical-Only vs SJF | rank-biserial correlation=-1.0000 | 250/250 | 1.0000 | 17 | 0.2511 |
| Classical-Only vs PPO | rank-biserial correlation=-1.0000 | 250/250 | 1.0000 | 17 | 0.2511 |
| Greedy vs SJF | rank-biserial correlation=-0.7470 | 250/250 | 1.0000 | 30 | 0.2511 |
| Greedy vs PPO | rank-biserial correlation=-0.9803 | 250/250 | 1.0000 | 18 | 0.2511 |
| SJF vs PPO | rank-biserial correlation=-0.9188 | 250/250 | 1.0000 | 20 | 0.2511 |

### 文字解读

- **PPO vs FCFS**：rank-biserial correlation=-0.3642（中效应），当前 N1=250, N2=250 的检验力 = 0.9823（达到 80% 标准）；检测该效应量仅需每组约 120 个样本。
- 检验力 < 0.80 的对比：表明当前样本量不足以可靠检测该效应量，对应的不显著结论需要谨慎解读（可能存在检验力不足导致假阴性）。
- 检验力 ≥ 0.99 且显著的对比：核心结论极其稳健，样本量远超检测该效应所需。

---
*报告自动生成 | 数据源: results\multiseed_evaluation\rewards_multiseed.json*