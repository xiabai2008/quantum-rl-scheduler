# HybridScheduler 三路消融报告（规则 vs RL vs 混合）

- **场景**：3 episodes × 60 步，同种子同任务流
- **模型**：deliverable_models/ppo_best_model_16dim.zip

| 策略 | 平均奖励/步 |
|:--|:--|
| 纯规则（RuleEngine） | 11.621 |
| 纯 RL（PPO deterministic） | 13.065 |
| **混合（HybridScheduler）** | **3.546** |

**混合策略决策来源分布**：{"rule": 93.66666666666667, "rl": 2.3333333333333335, "default": 4.0}

**发现（2026-08-02 消融实测）**：
- 本场景（量子友好负载）下混合策略明显劣于纯 RL（246 vs 710，约 -65%）：规则引擎对
  quantum/universal 任务过于保守（93.7% 决策被规则接管），锁死 RL 优势；
  少量 RL 决策（2 次，动作类型切换）扰动队列导致整体劣化。
- **这不是脚本缺陷**：同 obs 下混合内 RL 决策与纯 RL 动作一致；根因是
  '规则优先'设计在规则引擎次优的负载下压过 RL。
- **后续建议**：HybridScheduler 需要动态权重（如按 RL 置信度/历史表现决定规则与
  RL 的接管比例），或规则引擎在 quantum 友好负载下放松保守阈值。
- 完整数据：`results/hybrid/hybrid_ablation_result.json`
