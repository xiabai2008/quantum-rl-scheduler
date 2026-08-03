# 公平感知模型消融实验报告（17 维 vs 16 维基线）

- **场景**：租户不平衡负载（A 80% / B 10% / C 10%），15 episodes × 100 步
- **基线**：`deliverable_models/ppo_best_model_16dim.zip`（16 维，公平观测默认关闭）
- **公平模型**：`deliverable_models/ppo_fairness17dim.zip`（17 维，include_fairness_obs=True 训练，100K steps）

| 指标 | 16 维基线 | 17 维公平 | 变化 |
|:--|:--|:--|:--|
| Jain 完成率公平指数 | 0.8988 | 0.9338 | +0.0350 |
| max/min 完成率比率 | 0.5821 | 0.6129 | +0.0308 |
| 平均奖励/步 | 14.216 | 14.515 | +2.1% |

**结论解读**：
- 公平感知模型应显著提升 Jain 完成率公平指数（公平性维度）
- 调度效率（平均奖励/步）应持平或小幅变化——公平不免费但代价可控
- 完整数据：`results/fairness/fairness_ablation_result.json`
