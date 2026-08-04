# HybridScheduler 三路消融报告（规则 vs RL vs 混合）

- **场景**：5 episodes × 60 步，同种子同任务流
- **模型**：deliverable_models/ppo_best_model_16dim.zip

| 策略 | 平均奖励/步 |
|:--|:--|
| 纯规则（RuleEngine） | 6.923 |
| 纯 RL（PPO deterministic） | 14.447 |
| **混合（HybridScheduler）** | **5.779** |
| **混合（RL 优先，Issue #928）** | **14.447** |
| 混合（自适应，8.3 审查补充） | 12.835 |

**规则优先混合来源分布**：{"rule": 89.60000000000001, "rl": 2.1999999999999997, "default": 8.200000000000001}
**RL 优先混合来源分布**：{"rl": 69.8, "default": 30.2}

**解读**：
- RL 优先模式（Issue #928）：RL 高置信度优先 + 规则兜底，修复"规则优先锁死 RL"
- 自适应模式：基于 RL/规则历史奖励滑动窗口自动切换接管优先级
- 完整数据：`results/hybrid/hybrid_ablation_result.json`
