# 天衍-287 多seed真机实验统计分析报告

> **⚠️ INVALID FOR FORMAL COMPARISON（invalid_for_formal_comparison=true）**
>
> 本报告基于旧混合数据，**不得作为权威结论引用**。失效原因：
> - 混合机器：`tianyan287`（无连字符）+ `tianyan176`（无连字符），均非正确的 `tianyan-287`
> - 混合 shots：`1024`（old5）+ `32`（new5），不满足 Issue #58 统一口径
> - `real_tasks_completed=0`：无任何真机任务真正完成
> - SJF vs FCFS 的 `judgment` 自相矛盾（`bonferroni_significant=false` 却标"支持"，已修正为"不支持"）
> - 统计检验方法不统一（混合使用 Welch t / 独立样本 t / 配对 t）
>
> 报告中的 PPO=1736.32、FCFS=382.99、Cohen's d=5.33 **不是权威结论**。
> 正式 10-seed 真机验证需基于 `tianyan-287` + `shots=32` + `H Q1/M Q1` + 统一 Welch t-test 的全新数据。

**数据文件**: `results\real_machine\tianyan287_multiseed\multiseed_data_10seeds_merged.json`

**实验时间**: N/A

**实验配置**: 10 seeds × 3 策略

**总耗时**: 795.8s

**Bonferroni 校正 α**: 0.0167 (3 次比较)


---

## 1. 描述性统计

| 策略 | N | 均值 | 标准差 | 中位数 | min | max |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| PPO | 10 | 1736.32 | 355.78 | 1722.03 | 1224.13 | 2293.18 |
| SJF | 10 | 575.33 | 237.69 | 547.31 | 260.91 | 944.15 |
| FCFS | 10 | 383.00 | 49.13 | 406.64 | 288.77 | 432.96 |

## 2. 正态性检验

| 策略 | 检验方法 | p 值 | 结论 |
|:--:|:--:|:--:|:--:|
| PPO | Shapiro-Wilk | 0.7760 | 正态 |
| SJF | Shapiro-Wilk | 0.5578 | 正态 |
| FCFS | Shapiro-Wilk | 0.0687 | 正态 |

> 三组数据均通过正态性检验（p > 0.05），适用参数检验。

## 3. 两两比较（效应量决策范式）

> **决策规则**: 以 Cohen's d ≥ 0.5（中效应）且均值差 95% CI 不跨 0 为「支持」；
> d < 0.2 或 CI 跨 0 为「不支持」；其余为「不确定」。

### 3.1 PPO vs FCFS

- **均值差**: 1353.32 (PPO=1736.32 vs FCFS=383.00)
- **Cohen's d**: 5.3288（大效应）
- **rank-biserial**: 1.0000
- **均值差 95% CI**: [1097.83, 1608.82] (不跨0)
- **提升百分比**: 353.4% (95% CI: [290.8%, 421.1%])
- **Welch t 检验**: t=11.9155, p=0.000001 (Bonferroni显著)
- **判定**: **支持**

### 3.2 PPO vs SJF

- **均值差**: 1160.99 (PPO=1736.32 vs SJF=575.33)
- **Cohen's d**: 3.8373（大效应）
- **rank-biserial**: 1.0000
- **均值差 95% CI**: [873.71, 1448.28] (不跨0)
- **提升百分比**: 201.8% (95% CI: [133.4%, 305.0%])
- **Welch t 检验**: t=8.5804, p=0.000000 (Bonferroni显著)
- **判定**: **支持**

### 3.3 SJF vs FCFS

- **均值差**: 192.33 (SJF=575.33 vs FCFS=383.00)
- **Cohen's d**: 1.1206（大效应）
- **rank-biserial**: 0.4000
- **均值差 95% CI**: [20.76, 363.90] (不跨0)
- **提升百分比**: 50.2% (95% CI: [13.2%, 90.2%])
- **Welch t 检验**: t=2.5058, p=0.031647 (Bonferroni不显著)
- **判定**: **不支持**（bonferroni_significant=false，已修正自相矛盾）

## 4. 汇总表

| 比较 | 均值差 | Cohen's d | 效应等级 | 95% CI | 跨0? | p 值 | Bonferroni | 判定 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| PPO vs FCFS | 1353.32 | 5.3288 | 大效应 | [1097.83, 1608.82] | 否 | 0.000001 | 显著 | 支持 |
| PPO vs SJF | 1160.99 | 3.8373 | 大效应 | [873.71, 1448.28] | 否 | 0.000000 | 显著 | 支持 |
| SJF vs FCFS | 192.33 | 1.1206 | 大效应 | [20.76, 363.90] | 否 | 0.031647 | 不显著 | 不支持 |

## 5. compare_strategies 完整输出

**PPO vs SJF**: 使用 独立样本 t 检验 比较 PPO 与 SJF：PPO 平均奖励高于SJF 1160.99（95% CI: [876.72, 1445.26]）；统计量=8.5804，p=8.912e-08。经 Bonferroni 校正（3 次比较，校正 α=0.0167），差异显著。效应量 Cohen's d=3.8373（大效应）。

**PPO vs FCFS**: 使用 Welch t 检验 比较 PPO 与 FCFS：PPO 平均奖励高于FCFS 1353.32（95% CI: [1097.83, 1608.82]）；统计量=11.9155，p=5.842e-07。经 Bonferroni 校正（3 次比较，校正 α=0.0167），差异显著。效应量 Cohen's d=5.3288（大效应）。

**SJF vs FCFS**: 使用 Welch t 检验 比较 SJF 与 FCFS：SJF 平均奖励高于FCFS 192.33（95% CI: [20.76, 363.90]）；统计量=2.5058，p=0.03165。经 Bonferroni 校正（3 次比较，校正 α=0.0167），差异不显著。效应量 Cohen's d=1.1206（大效应）。

## 6. 结论

1. **PPO vs FCFS**: Cohen's d=5.33（大效应），均值差 95% CI [1097.83, 1608.82]不跨0，Bonferroni校正后显著（p=5.84e-07）。判定：**支持**。

2. **PPO vs SJF**: Cohen's d=3.84（大效应），均值差 95% CI [873.71, 1448.28]不跨0，Bonferroni校正后显著（p=2.55e-07）。判定：**支持**。

3. **多seed环境验证**: PPO 在 10 个独立 seed（tianyan176 真机环境，96步/episode，泊松到达λ=0.5）下均显著优于 FCFS 和 SJF，提升幅度 +353.4%（vs FCFS）/ +201.8%（vs SJF），与仿真实验（+88.3%, N=250）结论一致。

## 7. 真机任务执行情况披露

> **如实披露**：本实验中"真机环境"指调度环境在真机任务提交通路上的集成验证，统计指标（total_reward）来自调度环境仿真，而非真机保真度本身。

- **旧 5 seeds（20260721, shots=1024）**: 15 个真机任务（H Q0/M Q0）提交至 tianyan-287，全部提交成功获得 task_id，但执行阶段均失败（平台返回"任务运行失败"），real_tasks_completed=0，avg_fidelity=None
- **新 5 seeds（20260722, shots=32）**: 15 个真机任务提交至 tianyan176，全部提交成功获得 task_id，但 wait_for_task 轮询未返回 probability 字段（cqlib SDK query_experiment 接口兼容性问题），real_tasks_completed=0，avg_fidelity=None
- **统计指标有效性**: total_reward 为调度环境仿真奖励，与真机 fidelity 无关，统计结论（Cohen's d + Welch t + Bonferroni）基于仿真奖励，统计有效性不受真机 fidelity 缺失影响
- **tianyan-287 执行失败原因**: 经排查，tianyan-287 上 H Q0/M Q0 线路执行返回"运行失败"（code:1），可能与 Q0 比特校准状态或 paid 机器授权有关，需联系平台确认
- **tianyan176 稳定性**: 在 issue164 闭环实验中 8/8 真机任务成功（shots=32），本次 10 seeds 扩展中任务提交通路正常但结果查询接口异常
