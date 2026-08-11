# 平均等待时间显著性检验报告（8.11）

> 日期：2026-08-11 | 数据源：`results/multiseed_evaluation/rewards_multiseed.json`（utilization.avg_wait_time）
> 权威登记：`config/statistics.yaml` wait_time_significance 段
> 目的：为"吞吐量+等待时间双优"提供与 +20.2% 同源（N=250）的统计显著证据

## 1. 结果

| 指标 | PPO | FCFS | 变化 |
|:--|--:|--:|--:|
| 平均等待时间（步） | 25.46 ± 10.13 | 29.61 ± 10.64 | **-14.0%** |

## 2. 统计检验（N=250）

| 检验 | 统计量 | p 值 | 结论 |
|:--|--:|--:|:--|
| 配对 t（逐 episode） | t=-4.536 | **8.917e-06** | 极显著 |
| Wilcoxon 符号秩 | — | **8.928e-06** | 极显著 |
| seed 聚合（50 组配对） | t=-2.461 | **0.0174** | 显著（稳健性确认） |

- 95% CI（均值差）：**[-5.95, -2.36]**（不含 0）
- PPO 等待更短的 episode：153/250（61.2%）

## 3. 结论

1. **PPO 平均等待时间显著低于 FCFS（-14.0%，p<0.001 配对检验）**——与 +20.2% 综合奖励提升
   同为 N=250 权威实验的同源指标，等待时间维度从描述性数字升级为统计显著证据。
2. seed 聚合 p=0.0174 确认结论不依赖 episode 级重复（非伪重复）。
3. 结合 +20.2%（Welch t p=7.56e-12），**吞吐量（综合奖励）与等待时间双优均统计显著**，
   不存在"以等待时间换吞吐量"的负权衡（旧 +51% 口径已废弃）。

## 4. 复现

```python
# rewards_multiseed.json utilization.avg_wait_time
# PPO vs FCFS 配对 t：scipy.stats.ttest_rel（t=-4.536, p=8.917e-06）
# Wilcoxon：scipy.stats.wilcoxon（p=8.928e-06）
```
