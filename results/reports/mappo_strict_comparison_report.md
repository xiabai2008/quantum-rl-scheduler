# MAPPO 严格双基线对比报告（同 wrapper 同种子）

- **协议**：MultiAgentEnvWrapper 同实例、同种子序列（seed=42..+20）
- **模型**：deliverable_models/mappo/mappo.pt（50K 收敛）
- **FCFS**：每 agent 固定 hybrid（action=2），任务排序由 env 内部 FCFS 完成

| 策略 | mean_reward | std |
|:--|:--|:--|
| **MAPPO（协同）** | 5507.7 | 656.6 |
| FCFS（同环境） | 917.7 | 445.5 |
| 3独立PPO（无协同） | 4033.6 | 2066.1 |
| **增益 vs FCFS** | **+500.1%** | — |
| **增益 vs 独立PPO** | **+36.5%** | — |

**结论**：同 wrapper 同种子严格对比下，MAPPO（50K 收敛）相对 FCFS 增益
+500.1%、相对 3 独立 PPO（同训练量、无协同投票）增益
+36.5%——该数字消除了评估环境混杂，可直接归因于
多智能体协同调度算法（投票仲裁 + 共享 Critic 信用分配）。

**统计检验（8.10 补，N=20 配对）**：

| 对比 | 均值差 | 95% CI | Wilcoxon p | 配对t p | 显著 |
|:--|--:|:--|--:|--:|:--:|
| MAPPO vs FCFS | 4590.0 | [4277.3, 4902.7] | 0.0000 | 0.0000 | ✅ |
| MAPPO vs 独立PPO | 1474.1 | [464.4, 2483.9] | 0.0240 | 0.0100 | ✅ |

> 说明：独立 PPO 在部分 seed 出现性能崩溃（如 seed 56-59 仅 452-2232），MAPPO 保持稳定
> （std 656.6 vs 2066.1），
> 协同优势来自稳定性 + 协作投票。逐 seed 数据见 JSON per_seed 字段。

完整数据：`results/mappo_strict_comparison_result.json`
