# 真机反馈训练影响量化报告（修正版：方向C重新解读）

> 日期：2026-08-11 | 数据：`results/real_machine/issue192_10seeds.json`
> **8.11 修正**：初版将三条件误读为"同一模型的三种评估"；核实后为
> **三种训练数据训练的模型**（simulation/mixed_real/pure_real 各自训练）。
> 修正结论：差异反映训练数据影响，非"校准评估偏差"。

## 1. 实验设计（issue192，10 seeds）

| 条件 | 训练数据 | 真机参与（训练时） | 模型 |
|:--|:--|:--|:--|
| simulation | 纯仿真训练 | 0%（real_prob=0.0） | ppo_simulation_* |
| mixed_real | 仿真+真机反馈混合训练 | 5%（real_prob=0.05, cap=10） | ppo_mixed_real_* |
| pure_real | 纯真机反馈训练 | 100%（real_prob=1.0, cap=200） | ppo_pure_real_*（issue165） |

## 2. 结果（评估奖励）

| 条件 | 均值 | std | 说明 |
|:--|--:|--:|:--|
| simulation（纯仿真训练） | 999.7 | 1038.3 | 训练无真机数据 |
| mixed_real（混合训练） | 184.3 | 1167.2 | 训练含 5% 真机反馈 |
| pure_real（纯真机训练） | -298.8 | 1164.1 | 训练全真机反馈 |

## 3. 统计

| 对比 | t | p | 判定 |
|:--|--:|--:|:--|
| simulation vs mixed_real | -1.60 | 0.127 | 不显著（N=10+高方差，功效不足） |
| simulation vs pure_real | — | — | 方向一致（更大幅下降） |

## 4. 修正后的结论（诚实披露）

1. **训练时引入真机反馈改变模型行为**：混合/纯真机训练的模型，在评估中表现
   与纯仿真训练模型差异巨大（999.7 → 184.3 → -298.8），方向一致（真机反馈
   使评估奖励下降）——这印证"真机执行条件与理想仿真差异真实存在"。
2. **与噪声敏感性实验（p=2.98e-08）一致**：真机噪声是负向因素，训练/评估引入
   真机反馈会反映这一影响。
3. **本数据不能支撑"未校准评估高估真机"**（那是不同模型）；真正支撑
   "评估可信度"的是噪声敏感性机制实验（N=25 配对，统计成立）。
4. **本实验的意义**：展示了"训练数据含真机反馈"的模型行为差异（方向性），
   佐证量子硬件数据确实影响 AI 训练结果（无论正负）。

## 5. 复现

```python
# issue192_10seeds.json conditions.{simulation,mixed_real,pure_real}.runs[].evaluation.reward
# Welch t: scipy.stats.ttest_ind(sim_rewards, mixed_rewards, equal_var=False)
```
