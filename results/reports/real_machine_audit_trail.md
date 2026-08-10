# 真机实验审计轨迹

> **数据来源**: `results/real_machine/tianyan287_multiseed/`（N=10 v2 权威）+ `results/real_machine/mbs_expansion_20260810.json`（176 扩充）
> **生成时间**: 2026-08-10（8.10 外部审查修复：原文件为空壳模板）

## 真机任务记录（task_id 可审计）

| 机器 | 实验 | task_id | 状态 | MBS/保真度 |
|:--|:--|:--|:--|:--|
| tianyan-287 | smoke | 2081422018843197442 | completed | 0.7997 |
| tianyan-287 | seed42-PPO | 2081422149348966402 | completed | 0.9935 |
| tianyan-287 | seed42-FCFS | 2081422187768791042 | completed | 0.9419 |
| tianyan-287 | seed42-SJF | 2081422224678199297 | completed | 0.8773 |
| tianyan-287 | seed123-PPO | 2081422262443712514 | completed | 0.6706 |
| tianyan-287 | seed123-FCFS | 2081422300586713089 | completed | 0.8773 |
| tianyan-287 | seed123-SJF | 2081422338259951618 | completed | 0.6706 |
| tianyan-287 | seed456-PPO | 2081422375970938881 | completed | 0.8643 |
| tianyan-287 | seed456-FCFS | 2081422413942439938 | completed | 0.606 |
| tianyan-287 | seed456-SJF | 2081422451996893185 | completed | 0.8773 |
| tianyan-287 | seed789-PPO | 2081422489620267009 | completed | 0.9419 |
| tianyan-287 | seed789-FCFS | 2081422528576495618 | completed | 0.9935 |
| tianyan-287 | seed789-SJF | 2081422565797216257 | completed | 0.7997 |
| tianyan-287 | seed1024-PPO | 2081422603524513793 | completed | 0.8773 |
| tianyan-287 | seed1024-FCFS | 2081422641898668033 | completed | 0.9419 |
| tianyan-287 | seed1024-SJF | 2081422679978754050 | completed | 0.8773 |
| tianyan-287 | seed2025-PPO | 2081422718318419970 | completed | 0.8643 |
| tianyan-287 | seed2025-FCFS | 2081422756276871169 | completed | 0.8773 |
| tianyan-287 | seed2025-SJF | 2081422793849913346 | completed | 0.7482 |
| tianyan-287 | seed3141-PPO | 2081422832147636225 | completed | 0.9935 |
| tianyan-287 | seed3141-FCFS | 2081422870240772098 | completed | 0.9419 |
| tianyan-287 | seed3141-SJF | 2081422908181979137 | completed | 0.7997 |
| tianyan-287 | seed5678-PPO | 2081422945939103746 | completed | 0.9289 |
| tianyan-287 | seed5678-FCFS | 2081422983633313793 | completed | 0.606 |
| tianyan-287 | seed5678-SJF | 2081423023290458113 | completed | 0.5544 |
| tianyan-287 | seed8765-PPO | 2081423064319606785 | completed | 0.8643 |
| tianyan-287 | seed8765-FCFS | 2081423104874332161 | completed | 0.9935 |
| tianyan-287 | seed8765-SJF | 2081423143277379586 | completed | 0.9935 |
| tianyan-287 | seed9999-PPO | 2081423182573346817 | completed | 0.8643 |
| tianyan-287 | seed9999-FCFS | 2081423223283728386 | completed | 0.9419 |
| tianyan-287 | seed9999-SJF | 2081423260285878273 | completed | 0.9289 |
| tianyan176 | mbs#0 | 2086691216871866369 | - | 0.9648 |
| tianyan176 | mbs#1 | 2086691324787113985 | - | 0.949708 |
| tianyan176 | mbs#2 | 2086691556329005057 | - | 0.996392 |
| tianyan176 | mbs#3 | 2086691899289321473 | - | 0.949708 |
| tianyan176 | mbs#4 | 2086692006894190594 | - | 0.893652 |
| tianyan176 | mbs#5 | 2086692113978499074 | - | 0.919524 |
| tianyan176 | mbs#6 | 2086692194911789057 | - | 0.947552 |
| tianyan176 | mbs#7 | 2086692426366066690 | - | 0.989924 |
| tianyan176 | mbs#8 | 2086692775018053634 | - | 0.960488 |
| tianyan176 | mbs#9 | 2086694541243662338 | - | 0.977945 |
| tianyan176 | mbs#10 | 2086694887860477953 | - | 0.995042 |
| tianyan176 | mbs#11 | 2086695003115757569 | - | 0.973671 |
| tianyan176 | mbs#12 | 2086695435016290306 | - | 0.960848 |
| tianyan176 | mbs#13 | 2086696263143399425 | - | 0.911694 |
| tianyan176 | mbs#14 | 2086696852951089153 | - | 0.962985 |

**共 46 条记录，含 task_id 的 46 条**。

> 补充：287 可用性验证 315 次调用 100% 成功（284 主验证 + 31 审计）；N=10 v2 权威实验 30/30 真机任务完成（详见 `results/reports/multiseed_real_machine_report_10seeds_v2.md`）。