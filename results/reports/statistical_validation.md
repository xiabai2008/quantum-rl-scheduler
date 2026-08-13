# 统计显著性检验报告（多Seed验证）

> 本报告为提交清单 `EXP_STAT` 必需文件，使用 250 次独立episode验证PPO相对于基线策略的统计显著性。

> **数据来源**: `results\multiseed_evaluation\rewards_multiseed.json`
> **显著性水平 α**: 0.05
> **比较次数**: 28（Bonferroni 校正后 α = 0.0018）
> **标准差口径**: 本报告 std 列使用**样本标准差（ddof=1）**（由统计脚本自动生成）；statistics.yaml / AGENTS / 白皮书使用总体标准差（ddof=0，如 PPO 557.25 vs 本报告 558.37）。两者均为合法口径，N=250 时差异 <0.2%，对外引用以 statistics.yaml（ddof=0）为权威。

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
| 3 | SJF | 748.48 | 305.47 | 19.32 | -54.6% | [-57.4%, -51.6%] |
| 4 | DQN | 697.40 | 288.83 | 18.27 | -57.7% | [-60.3%, -54.9%] |
| 5 | Random | 697.40 | 288.83 | 18.27 | -57.7% | [-60.3%, -54.9%] |
| 6 | Greedy | 62.72 | 538.76 | 34.07 | -96.2% | [-100.2%, -92.2%] |
| 7 | Quantum-Only | -826.59 | 263.63 | 16.67 | -150.1% | [-152.9%, -147.5%] |
| 8 | Classical-Only | -1075.49 | 75.04 | 4.75 | -165.2% | [-167.9%, -162.8%] |

**核心结论：PPO 平均奖励 1982.69 vs FCFS 1648.91，提升 +20.2%，95% CI: [+14.3%, +26.7%]**
（N=250 次独立episode，α=0.05，Bonferroni多重比较校正）

---

## 一、各策略奖励统计

| 策略 | 样本数 | 平均奖励 | 标准差 | 最小值 | 最大值 |
|:--|:--:|:--:|:--:|:--:|:--:|
| DQN | 250 | 697.40 | 288.83 | -5.42 | 1619.49 |
| FCFS | 250 | 1648.91 | 503.96 | 288.56 | 2813.56 |
| Random | 250 | 697.40 | 288.83 | -5.42 | 1619.49 |
| Quantum-Only | 250 | -826.59 | 263.63 | -1250.65 | 87.27 |
| Classical-Only | 250 | -1075.49 | 75.04 | -1264.92 | -876.90 |
| Greedy | 250 | 62.72 | 538.76 | -1200.11 | 1298.08 |
| SJF | 250 | 748.48 | 305.47 | 65.09 | 1691.67 |
| PPO | 250 | 1982.69 | 558.37 | 291.78 | 3115.95 |

## 二、两两比较结果

| 对比 | 检验方法 | 统计量 | p 值 | 显著? | 效应量 | 均值差 | 95% CI | 提升% 95% CI |
|:--|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DQN vs FCFS | Mann-Whitney U 检验 | 3808.0000 | 1.009e-64 | [SIG] 是 | rank-biserial correlation=-0.8781 | -951.51 | [-1023.69, -879.33] | [-60.3%, -54.9%] |
| DQN vs Random | 独立样本 t 检验 | 0.0000 | 1 | [NS] 否 | Cohen's d=0.0000 | 0.00 | [-50.76, 50.76] | [-7.0%, +7.7%] |
| DQN vs Quantum-Only | Mann-Whitney U 检验 | 62496.0000 | 2.344e-83 | [SIG] 是 | rank-biserial correlation=0.9999 | 1524.00 | [1475.41, 1572.59] | [+179.0%, +190.1%] |
| DQN vs Classical-Only | Welch t 检验 | 93.9343 | 4.502e-215 | [SIG] 是 | Cohen's d=8.4017 | 1772.90 | [1735.74, 1810.05] | [+161.5%, +168.2%] |
| DQN vs Greedy | Mann-Whitney U 检验 | 53269.0000 | 2.629e-42 | [SIG] 是 | rank-biserial correlation=0.7046 | 634.69 | [558.73, 710.65] | [+435.8%, +12533.1%] |
| DQN vs SJF | Mann-Whitney U 检验 | 28727.0000 | 0.1184 | [NS] 否 | rank-biserial correlation=-0.0807 | -51.07 | [-103.31, 1.16] | [-13.3%, +0.2%] |
| DQN vs PPO | Mann-Whitney U 检验 | 2038.0000 | 4.289e-73 | [SIG] 是 | rank-biserial correlation=-0.9348 | -1285.29 | [-1363.41, -1207.17] | [-67.0%, -62.6%] |
| FCFS vs Random | Mann-Whitney U 检验 | 58692.0000 | 1.009e-64 | [SIG] 是 | rank-biserial correlation=0.8781 | 951.51 | [879.33, 1023.69] | [+122.2%, +152.3%] |
| FCFS vs Quantum-Only | Mann-Whitney U 检验 | 62500.0000 | 2.234e-83 | [SIG] 是 | rank-biserial correlation=1.0000 | 2475.51 | [2404.84, 2546.18] | [+288.8%, +310.8%] |
| FCFS vs Classical-Only | Mann-Whitney U 检验 | 62500.0000 | 2.234e-83 | [SIG] 是 | rank-biserial correlation=1.0000 | 2724.41 | [2661.09, 2787.72] | [+247.5%, +259.3%] |
| FCFS vs Greedy | Mann-Whitney U 检验 | 61573.0000 | 1.294e-78 | [SIG] 是 | rank-biserial correlation=0.9703 | 1586.20 | [1494.53, 1677.87] | [+1176.4%, +28933.4%] |
| FCFS vs SJF | Mann-Whitney U 检验 | 57858.0000 | 5.87e-61 | [SIG] 是 | rank-biserial correlation=0.8515 | 900.44 | [827.21, 973.66] | [+106.7%, +135.0%] |
| FCFS vs PPO | Mann-Whitney U 检验 | 19870.0000 | 1.86e-12 | [SIG] 是 | rank-biserial correlation=-0.3642 | -333.78 | [-427.24, -240.31] | [-21.0%, -12.4%] |
| Random vs Quantum-Only | Mann-Whitney U 检验 | 62496.0000 | 2.344e-83 | [SIG] 是 | rank-biserial correlation=0.9999 | 1524.00 | [1475.41, 1572.59] | [+179.0%, +190.1%] |
| Random vs Classical-Only | Welch t 检验 | 93.9343 | 4.502e-215 | [SIG] 是 | Cohen's d=8.4017 | 1772.90 | [1735.74, 1810.05] | [+161.5%, +168.2%] |
| Random vs Greedy | Mann-Whitney U 检验 | 53269.0000 | 2.629e-42 | [SIG] 是 | rank-biserial correlation=0.7046 | 634.69 | [558.73, 710.65] | [+435.8%, +12533.1%] |
| Random vs SJF | Mann-Whitney U 检验 | 28727.0000 | 0.1184 | [NS] 否 | rank-biserial correlation=-0.0807 | -51.07 | [-103.31, 1.16] | [-13.3%, +0.2%] |
| Random vs PPO | Mann-Whitney U 检验 | 2038.0000 | 4.289e-73 | [SIG] 是 | rank-biserial correlation=-0.9348 | -1285.29 | [-1363.41, -1207.17] | [-67.0%, -62.6%] |
| Quantum-Only vs Classical-Only | Mann-Whitney U 检验 | 51673.0000 | 1.227e-36 | [SIG] 是 | rank-biserial correlation=0.6535 | 248.90 | [214.84, 282.96] | [+20.1%, +26.2%] |
| Quantum-Only vs Greedy | Mann-Whitney U 检验 | 5537.5000 | 4.812e-57 | [SIG] 是 | rank-biserial correlation=-0.8228 | -889.31 | [-963.84, -814.78] | [-14635.6%, -737.8%] |
| Quantum-Only vs SJF | Mann-Whitney U 检验 | 1.0000 | 2.261e-83 | [SIG] 是 | rank-biserial correlation=-1.0000 | -1575.07 | [-1625.21, -1524.93] | [-217.7%, -203.6%] |
| Quantum-Only vs PPO | Mann-Whitney U 检验 | 0.0000 | 2.234e-83 | [SIG] 是 | rank-biserial correlation=-1.0000 | -2809.29 | [-2886.02, -2732.56] | [-143.9%, -139.5%] |
| Classical-Only vs Greedy | Mann-Whitney U 检验 | 1836.0000 | 4.404e-74 | [SIG] 是 | rank-biserial correlation=-0.9412 | -1138.21 | [-1205.80, -1070.62] | [-19038.5%, -933.7%] |
| Classical-Only vs SJF | Mann-Whitney U 检验 | 0.0000 | 2.234e-83 | [SIG] 是 | rank-biserial correlation=-1.0000 | -1823.97 | [-1863.06, -1784.88] | [-251.6%, -236.7%] |
| Classical-Only vs PPO | Mann-Whitney U 检验 | 0.0000 | 2.234e-83 | [SIG] 是 | rank-biserial correlation=-1.0000 | -3058.19 | [-3128.19, -2988.18] | [-156.3%, -152.3%] |
| Greedy vs SJF | Mann-Whitney U 检验 | 8184.0000 | 2.963e-46 | [SIG] 是 | rank-biserial correlation=-0.7381 | -685.76 | [-762.72, -608.80] | [-100.4%, -82.8%] |
| Greedy vs PPO | Mann-Whitney U 检验 | 549.0000 | 1.538e-80 | [SIG] 是 | rank-biserial correlation=-0.9824 | -1919.98 | [-2016.39, -1823.56] | [-100.2%, -93.5%] |
| SJF vs PPO | Mann-Whitney U 检验 | 2438.0000 | 3.713e-71 | [SIG] 是 | rank-biserial correlation=-0.9220 | -1234.22 | [-1313.30, -1155.13] | [-64.5%, -59.9%] |

## 三、详细解释

### DQN vs FCFS

> 使用 Mann-Whitney U 检验 比较 DQN 与 FCFS：DQN 平均奖励低于FCFS 951.51（95% CI: [-1023.69, -879.33]）；统计量=3808.0000，p=1.009e-64。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.8781（大效应）。

### DQN vs Random

> 使用 独立样本 t 检验 比较 DQN 与 Random：DQN 平均奖励等于Random 0.00（95% CI: [-50.76, 50.76]）；统计量=0.0000，p=1。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 Cohen's d=0.0000（可忽略）。

### DQN vs Quantum-Only

> 使用 Mann-Whitney U 检验 比较 DQN 与 Quantum-Only：DQN 平均奖励高于Quantum-Only 1524.00（95% CI: [1475.41, 1572.59]）；统计量=62496.0000，p=2.344e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.9999（大效应）。

### DQN vs Classical-Only

> 使用 Welch t 检验 比较 DQN 与 Classical-Only：DQN 平均奖励高于Classical-Only 1772.90（95% CI: [1735.74, 1810.05]）；统计量=93.9343，p=4.502e-215。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=8.4017（大效应）。

### DQN vs Greedy

> 使用 Mann-Whitney U 检验 比较 DQN 与 Greedy：DQN 平均奖励高于Greedy 634.69（95% CI: [558.73, 710.65]）；统计量=53269.0000，p=2.629e-42。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.7046（大效应）。

### DQN vs SJF

> 使用 Mann-Whitney U 检验 比较 DQN 与 SJF：DQN 平均奖励低于SJF 51.07（95% CI: [-103.31, 1.16]）；统计量=28727.0000，p=0.1184。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 rank-biserial correlation=-0.0807（可忽略）。

### DQN vs PPO

> 使用 Mann-Whitney U 检验 比较 DQN 与 PPO：DQN 平均奖励低于PPO 1285.29（95% CI: [-1363.41, -1207.17]）；统计量=2038.0000，p=4.289e-73。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.9348（大效应）。

### FCFS vs Random

> 使用 Mann-Whitney U 检验 比较 FCFS 与 Random：FCFS 平均奖励高于Random 951.51（95% CI: [879.33, 1023.69]）；统计量=58692.0000，p=1.009e-64。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.8781（大效应）。

### FCFS vs Quantum-Only

> 使用 Mann-Whitney U 检验 比较 FCFS 与 Quantum-Only：FCFS 平均奖励高于Quantum-Only 2475.51（95% CI: [2404.84, 2546.18]）；统计量=62500.0000，p=2.234e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=1.0000（大效应）。

### FCFS vs Classical-Only

> 使用 Mann-Whitney U 检验 比较 FCFS 与 Classical-Only：FCFS 平均奖励高于Classical-Only 2724.41（95% CI: [2661.09, 2787.72]）；统计量=62500.0000，p=2.234e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=1.0000（大效应）。

### FCFS vs Greedy

> 使用 Mann-Whitney U 检验 比较 FCFS 与 Greedy：FCFS 平均奖励高于Greedy 1586.20（95% CI: [1494.53, 1677.87]）；统计量=61573.0000，p=1.294e-78。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.9703（大效应）。

### FCFS vs SJF

> 使用 Mann-Whitney U 检验 比较 FCFS 与 SJF：FCFS 平均奖励高于SJF 900.44（95% CI: [827.21, 973.66]）；统计量=57858.0000，p=5.87e-61。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.8515（大效应）。

### FCFS vs PPO

> 使用 Mann-Whitney U 检验 比较 FCFS 与 PPO：FCFS 平均奖励低于PPO 333.78（95% CI: [-427.24, -240.31]）；统计量=19870.0000，p=1.86e-12。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.3642（中效应）。

### Random vs Quantum-Only

> 使用 Mann-Whitney U 检验 比较 Random 与 Quantum-Only：Random 平均奖励高于Quantum-Only 1524.00（95% CI: [1475.41, 1572.59]）；统计量=62496.0000，p=2.344e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.9999（大效应）。

### Random vs Classical-Only

> 使用 Welch t 检验 比较 Random 与 Classical-Only：Random 平均奖励高于Classical-Only 1772.90（95% CI: [1735.74, 1810.05]）；统计量=93.9343，p=4.502e-215。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 Cohen's d=8.4017（大效应）。

### Random vs Greedy

> 使用 Mann-Whitney U 检验 比较 Random 与 Greedy：Random 平均奖励高于Greedy 634.69（95% CI: [558.73, 710.65]）；统计量=53269.0000，p=2.629e-42。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.7046（大效应）。

### Random vs SJF

> 使用 Mann-Whitney U 检验 比较 Random 与 SJF：Random 平均奖励低于SJF 51.07（95% CI: [-103.31, 1.16]）；统计量=28727.0000，p=0.1184。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异不显著。效应量 rank-biserial correlation=-0.0807（可忽略）。

### Random vs PPO

> 使用 Mann-Whitney U 检验 比较 Random 与 PPO：Random 平均奖励低于PPO 1285.29（95% CI: [-1363.41, -1207.17]）；统计量=2038.0000，p=4.289e-73。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.9348（大效应）。

### Quantum-Only vs Classical-Only

> 使用 Mann-Whitney U 检验 比较 Quantum-Only 与 Classical-Only：Quantum-Only 平均奖励高于Classical-Only 248.90（95% CI: [214.84, 282.96]）；统计量=51673.0000，p=1.227e-36。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=0.6535（大效应）。

### Quantum-Only vs Greedy

> 使用 Mann-Whitney U 检验 比较 Quantum-Only 与 Greedy：Quantum-Only 平均奖励低于Greedy 889.31（95% CI: [-963.84, -814.78]）；统计量=5537.5000，p=4.812e-57。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.8228（大效应）。

### Quantum-Only vs SJF

> 使用 Mann-Whitney U 检验 比较 Quantum-Only 与 SJF：Quantum-Only 平均奖励低于SJF 1575.07（95% CI: [-1625.21, -1524.93]）；统计量=1.0000，p=2.261e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-1.0000（大效应）。

### Quantum-Only vs PPO

> 使用 Mann-Whitney U 检验 比较 Quantum-Only 与 PPO：Quantum-Only 平均奖励低于PPO 2809.29（95% CI: [-2886.02, -2732.56]）；统计量=0.0000，p=2.234e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-1.0000（大效应）。

### Classical-Only vs Greedy

> 使用 Mann-Whitney U 检验 比较 Classical-Only 与 Greedy：Classical-Only 平均奖励低于Greedy 1138.21（95% CI: [-1205.80, -1070.62]）；统计量=1836.0000，p=4.404e-74。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.9412（大效应）。

### Classical-Only vs SJF

> 使用 Mann-Whitney U 检验 比较 Classical-Only 与 SJF：Classical-Only 平均奖励低于SJF 1823.97（95% CI: [-1863.06, -1784.88]）；统计量=0.0000，p=2.234e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-1.0000（大效应）。

### Classical-Only vs PPO

> 使用 Mann-Whitney U 检验 比较 Classical-Only 与 PPO：Classical-Only 平均奖励低于PPO 3058.19（95% CI: [-3128.19, -2988.18]）；统计量=0.0000，p=2.234e-83。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-1.0000（大效应）。

### Greedy vs SJF

> 使用 Mann-Whitney U 检验 比较 Greedy 与 SJF：Greedy 平均奖励低于SJF 685.76（95% CI: [-762.72, -608.80]）；统计量=8184.0000，p=2.963e-46。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.7381（大效应）。

### Greedy vs PPO

> 使用 Mann-Whitney U 检验 比较 Greedy 与 PPO：Greedy 平均奖励低于PPO 1919.98（95% CI: [-2016.39, -1823.56]）；统计量=549.0000，p=1.538e-80。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.9824（大效应）。

### SJF vs PPO

> 使用 Mann-Whitney U 检验 比较 SJF 与 PPO：SJF 平均奖励低于PPO 1234.22（95% CI: [-1313.30, -1155.13]）；统计量=2438.0000，p=3.713e-71。经 Bonferroni 校正（28 次比较，校正 α=0.0018），差异显著。效应量 rank-biserial correlation=-0.9220（大效应）。

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
| DQN vs FCFS | rank-biserial correlation=-0.8781 | 250/250 | 1.0000 | 22 | 0.2511 |
| DQN vs Random | Cohen's d=0.0000 | 250/250 | 0.0500 | N/A | 0.2511 |
| DQN vs Quantum-Only | rank-biserial correlation=0.9999 | 250/250 | 1.0000 | 17 | 0.2511 |
| DQN vs Classical-Only | Cohen's d=8.4017 | 250/250 | 1.0000 | 2 | 0.2511 |
| DQN vs Greedy | rank-biserial correlation=0.7046 | 250/250 | 1.0000 | 33 | 0.2511 |
| DQN vs SJF | rank-biserial correlation=-0.0807 | 250/250 | 0.1469 | 2410 | 0.2511 |
| DQN vs PPO | rank-biserial correlation=-0.9348 | 250/250 | 1.0000 | 19 | 0.2511 |
| FCFS vs Random | rank-biserial correlation=0.8781 | 250/250 | 1.0000 | 22 | 0.2511 |
| FCFS vs Quantum-Only | rank-biserial correlation=1.0000 | 250/250 | 1.0000 | 17 | 0.2511 |
| FCFS vs Classical-Only | rank-biserial correlation=1.0000 | 250/250 | 1.0000 | 17 | 0.2511 |
| FCFS vs Greedy | rank-biserial correlation=0.9703 | 250/250 | 1.0000 | 18 | 0.2511 |
| FCFS vs SJF | rank-biserial correlation=0.8515 | 250/250 | 1.0000 | 23 | 0.2511 |
| FCFS vs PPO | rank-biserial correlation=-0.3642 | 250/250 | 0.9823 | 120 | 0.2511 |
| Random vs Quantum-Only | rank-biserial correlation=0.9999 | 250/250 | 1.0000 | 17 | 0.2511 |
| Random vs Classical-Only | Cohen's d=8.4017 | 250/250 | 1.0000 | 2 | 0.2511 |
| Random vs Greedy | rank-biserial correlation=0.7046 | 250/250 | 1.0000 | 33 | 0.2511 |
| Random vs SJF | rank-biserial correlation=-0.0807 | 250/250 | 0.1469 | 2410 | 0.2511 |
| Random vs PPO | rank-biserial correlation=-0.9348 | 250/250 | 1.0000 | 19 | 0.2511 |
| Quantum-Only vs Classical-Only | rank-biserial correlation=0.6535 | 250/250 | 1.0000 | 38 | 0.2511 |
| Quantum-Only vs Greedy | rank-biserial correlation=-0.8228 | 250/250 | 1.0000 | 25 | 0.2511 |
| Quantum-Only vs SJF | rank-biserial correlation=-1.0000 | 250/250 | 1.0000 | 17 | 0.2511 |
| Quantum-Only vs PPO | rank-biserial correlation=-1.0000 | 250/250 | 1.0000 | 17 | 0.2511 |
| Classical-Only vs Greedy | rank-biserial correlation=-0.9412 | 250/250 | 1.0000 | 19 | 0.2511 |
| Classical-Only vs SJF | rank-biserial correlation=-1.0000 | 250/250 | 1.0000 | 17 | 0.2511 |
| Classical-Only vs PPO | rank-biserial correlation=-1.0000 | 250/250 | 1.0000 | 17 | 0.2511 |
| Greedy vs SJF | rank-biserial correlation=-0.7381 | 250/250 | 1.0000 | 30 | 0.2511 |
| Greedy vs PPO | rank-biserial correlation=-0.9824 | 250/250 | 1.0000 | 18 | 0.2511 |
| SJF vs PPO | rank-biserial correlation=-0.9220 | 250/250 | 1.0000 | 20 | 0.2511 |

### 文字解读

- **PPO vs FCFS**：rank-biserial correlation=-0.3642（大效应），当前 N1=250, N2=250 的检验力 = 0.9823（达到 80% 标准）；检测该效应量仅需每组约 120 个样本。
- 检验力 < 0.80 的对比：表明当前样本量不足以可靠检测该效应量，对应的不显著结论需要谨慎解读（可能存在检验力不足导致假阴性）。
- 检验力 ≥ 0.99 且显著的对比：核心结论极其稳健，样本量远超检测该效应所需。

---
*报告自动生成 | 数据源: results\multiseed_evaluation\rewards_multiseed.json*