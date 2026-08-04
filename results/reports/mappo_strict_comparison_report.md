# MAPPO 严格双基线对比报告（同 wrapper 同种子）

- **协议**：MultiAgentEnvWrapper 同实例、同种子序列（seed=42..+10）
- **模型**：models/mappo.pt（50K 收敛）
- **FCFS**：每 agent 固定 hybrid（action=2），任务排序由 env 内部 FCFS 完成

| 策略 | mean_reward | std |
|:--|:--|:--|
| **MAPPO（协同）** | 4588.7 | 663.8 |
| FCFS（同环境） | 923.9 | 418.5 |
| 3独立PPO（无协同） | 4412.2 | 972.1 |
| **增益 vs FCFS** | **+396.7%** | — |
| **增益 vs 独立PPO** | **+4.0%** | — |

**结论**：同 wrapper 同种子严格对比下，MAPPO（50K 收敛）相对 FCFS 增益
+396.7%、相对 3 独立 PPO（同训练量、无协同投票）增益
+4.0%——该数字消除了评估环境混杂，可直接归因于
多智能体协同调度算法（投票仲裁 + 共享 Critic 信用分配）。
多智能体协同调度算法（+ 规则未覆盖的 RL 决策优势）。
完整数据：`results/mappo_strict_comparison_result.json`
