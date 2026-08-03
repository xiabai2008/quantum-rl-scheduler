# HybridScheduler 三路消融报告（规则 vs RL vs 混合）

- **场景**：5 episodes × 60 步，同种子同任务流
- **模型**：deliverable_models/ppo_best_model_16dim.zip

| 策略 | 平均奖励/步 |
|:--|:--|
| 纯规则（RuleEngine） | 6.923 |
| 纯 RL（PPO deterministic） | 14.447 |
| **混合（HybridScheduler）** | **5.779** |

**混合策略决策来源分布**：{"rule": 89.60000000000001, "rl": 2.1999999999999997, "default": 8.200000000000001}

**解读**：
- 混合策略应不差于最优单策略（规则兜底 + RL 补盲），来源分布说明规则/RL 各自承担的比例
- 完整数据：`results/hybrid/hybrid_ablation_result.json`
