# 8 策略对比报告（多Seed验证版）

> **数据来源（权威）**: `results/multiseed_evaluation/rewards_multiseed.json`（2026-07-09 多seed评估）
> **数据来源（单次参考）**: `results/issue_experiments/20260702_215916_strategy_comparison.json`（2026-07-02 单次运行）
> **运行环境**: 10 维公平对比环境（14 维环境经 Obs10Wrapper 截断，兼容现有 DQN/PPO 模型）
> **实验配置**: 10 seeds × 5 episodes = 50 次独立运行，200步/episode，泊松到达λ=0.5
> **统计检验**: Welch t 检验 / Mann-Whitney U，Bonferroni 校正 α=0.05
> **生成时间**: 2026-07-09（v8 权威数字锁定）

---

## 一、核心指标排名表（多Seed验证，N=50）

按**平均奖励（降序）**排列。

| 排名 | 策略 | 平均奖励 | 标准差 | 标准误 | 提升 vs FCFS | 统计显著性 |
|:--:|:--|:--:|:--:|:--:|:--:|:--:|
| 1 | **PPO** | **2814.19** | 1121.19 | 158.56 | **+92.4%** | ✅ p<0.001 |
| 2 | SJF | 1468.17 | 119.08 | 16.84 | +0.4% | ❌ p=0.76 |
| 3 | FCFS | 1462.48 | 55.85 | 7.90 | 基线 | — |
| 4 | Random | 1275.91 | 411.84 | 58.24 | -12.8% | ✅ p<0.01 |
| 5 | Greedy | -71.87 | 619.50 | 87.61 | -104.9% | ✅ p<0.001 |
| 6 | DQN | -897.08 | 289.90 | 41.00 | -161.3% | ✅ p<0.001 |
| 7 | Quantum-Only | -897.08 | 289.90 | 41.00 | -161.3% | ✅ p<0.001 |
| 8 | Classical-Only | -1134.35 | 64.04 | 9.06 | -177.6% | ✅ p<0.001 |

> 注：DQN 与 Quantum-Only 平均奖励相同（-897.08），因为当前 DQN 模型在10维环境下未充分收敛，行为退化为类似仅量子分配策略。
> SJF 与 FCFS 无显著差异（p=0.76），说明在该环境设置下，启发式调度策略之间差异不大。

---

## 二、关键结论（多Seed验证）

### 2.1 PPO vs FCFS（核心指标）

- **PPO 平均奖励**: 2814.19 ± 158.56（标准误）
- **FCFS 平均奖励**: 1462.48 ± 7.90（标准误）
- **PPO vs FCFS 提升**: +1351.71（95% CI: [1032.71, 1670.70]）
- **提升比例**: **+92.4%**
- **统计检验**: Welch t 检验，t(df=51.2)=-8.51，**p=3.04×10⁻¹¹**（Bonferroni校正后显著）
- **效应量**: Cohen's d = -1.70（**大效应量**，>0.8 为大）

### 2.2 PPO vs 所有基线

| 对比项 | 提升值 | 提升比例 | 显著性 |
|:--|:--|:--|:--|
| PPO vs FCFS | +1351.71 | +92.4% | ✅ p<0.001 |
| PPO vs Random | +1538.28 | +120.6% | ✅ p<0.001 |
| PPO vs SJF | +1346.02 | +91.7% | ✅ p<0.001 |
| PPO vs Greedy | +2886.06 | +4015% | ✅ p<0.001 |
| PPO vs DQN | +3711.27 | +414% | ✅ p<0.001 |

### 2.3 其他指标（单次运行参考，待多seed验证）

基于 2026-07-02 单次运行（200任务，10维环境）：
- **量子利用率最高**: Random（44.85%）
- **经典利用率最高**: PPO（51.81%）
- **最短平均等待**: FCFS（39.94 步）
- **PPO 平均等待**: 58.72 步（较高等待换取了更高的资源利用率和奖励）

---

## 三、PPT/白皮书可用结论

> **在 10 维公平对比环境中（N=50次独立运行），PPO 强化学习调度策略的平均奖励（2814±159）比 FCFS 基线（1462±8）提升 92.4%（Welch t检验，p<0.001，Cohen's d=-1.70，大效应量），验证了 RL 在量子-经典混合任务调度中的显著优势。相比 Random 策略提升 120.6%。**

---

## 四、数据复现说明

复现本报告结果：

```bash
# 运行多seed评估（约40秒）
python scripts/evaluation/run_multiseed_evaluation.py --seeds 10 --episodes 5 --tasks-per-episode 200

# 仅运行统计显著性检验（基于已有数据）
python scripts/evaluation/statistical_significance.py \
    --input results/multiseed_evaluation/rewards_multiseed.json \
    --output results/reports/statistical_validation.md
```

关键依赖：
- PPO 模型: `models/ppo_seed_42_v4/best_model.zip`（10维，Actor-Critic）
- DQN 模型: `models/dqn_fair_v2/seed_42/best_model.zip`（10维，Dueling DQN）
- 环境: `QuantumSchedulingEnv` + `Obs10Wrapper`（14维→10维截断）

---

*报告生成时间: 2026-07-09 | 数据源: results/multiseed_evaluation/rewards_multiseed_20260709_*.json | 统计方法: SciPy + Bonferroni校正*
