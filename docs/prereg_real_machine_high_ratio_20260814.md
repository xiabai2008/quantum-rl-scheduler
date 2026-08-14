# 真机提分实验预注册方案（2026-08-14 起草，冻结延期权）

> **📌 决策记录（2026-08-14，冻结终检）：实验①选 B 放弃实跑；实验③已完成（v3 权威 N=20）；实验②保持未启动**
>
> **实验①（真机高占比闭环）放弃理由**：v3 权威（N=20/组，8.14 扩样）已达 N≥18 功效线，
> 真机策略对比证据缺口已收窄（d=2.11, p=1.22e-07）；当日真机平台 ~30% 真实失败率、
> 仅支持单比特 H 电路，高占比档位下实验可比性风险高；新 reward 口径与 v3（1/96 步参与）
> 不一致的全仓同步成本高。**本方案保留为可恢复预案**，恢复条件：平台多比特支持稳定且
> 失败率 <10%，或机时包升级后评审阶段需补充高占比证据（详见 `docs/runbook_high_ratio_exp1_20260814.md` 决策横幅）。
>
> **实验③（v2 扩样 N=20）已完成**：结果见 `results/reports/multiseed_real_machine_report_20seeds_v3.md`、
> `config/statistics.yaml real_machine_20seed_v3`（PPO=1632.26±326.49 vs FCFS=782.94±467.61,
> d=2.11, p=1.22e-07；SJF vs FCFS p=0.16 不显著——v2 的 0.0316 边缘支持判为小样本噪声）。
>
> **背景**：8.13 冻结终检确认全项目最大短板为"真机性能证据不充分"——现有真机实验
> `real_submit_probability=0.15`（env_types.py:102 默认）下真机奖励仅占约 1/96 步，
> 策略间差异主要由仿真动力学驱动（`statistics.yaml real_machine_10seed_v2` 已诚实披露）。
> 机时可用 + 冻结日期可灵活延期后，以下三个预注册实验是"方案可行性 7.5→8.5、
> 主题契合度 8.0→8.5"的核心杠杆。
>
> **总纪律（必须逐条遵守）**：
> 1. 所有实验只用**冻结代码**（HEAD 5c15e44 / tag `v9.1-submission`，8.14 冻结终检收口）的**显式参数**运行；
>    **禁止修改任何 env/agent/baselines 行为或默认值**（`REAL_SUBMIT_PROBABILITY_DEFAULT`
>    等常量一动，全部权威仿真数字作废）。
> 2. 实验启动前先入库**预注册 JSON**（含本方案、seeds 清单、参数、假设、成功标准、
>    时间戳），运行后结果 JSON 与报告入库，**无论结果正负**。
> 3. 所有新数字进入 `config/statistics.yaml` 新实验段 + `check_stats_consistency.py`
>    登记 p 值白名单/黑名单，重跑门禁，最后重打 tag、重打包（`validate_submission.py --pack`）。
> 4. 任何真机结果不显著或方向为负，按预注册条款如实写入报告——严谨性 9 分是
>    本项目最大的竞争资产，不可因一次真机实验的波动而受损。
>
> 配套自检：`python scripts/ci/reproduce_authoritative.py`（8.13 冻结终检新增），
> 每次实验入库后运行确认权威数字未被污染。

---

## 实验①：真机高占比调度闭环（最大杠杆，建议 8/20-9/1 执行）

### 1.1 动机（解决评委最可能的一击）

> "你们的 PPO 提升 99% 来自仿真 reward，真机任务每 episode 只有 1 步，**真机对结果
> 没有任何贡献**。" —— 现有 `real_submit_probability=0.15` + 96 步/episode 的设计
> 无法反驳这一点。本实验把真机任务占比提到主导档位，直接验证"AI 调度决策在真机
> 奖励维度上优于基线"。

### 1.2 预注册假设

- **H1**（主假设，单侧）：在真机高占比档位下，PPO 的 episode 总奖励均值 > 真实 FCFS
  （EnvBasedFCFSScheduler，量子路由）——配对设计（同 seed 同档位两策略）。
- **H2**（稳健性，探索性）：真机 reward 占总 reward 比例 ≥ 30% 时，PPO vs FCFS 差异
  仍显著（证明差异非仿真主导）。
- **H3**（负向预案）：若 H1 不显著或方向为负 → 如实报告，结论定为"真机高占比下
  PPO 无优势/优势不显著"，并给出置信区间与功效分析——这本身就是有价值的诚实证据
  （预注册的意义：防止选择性报告）。

### 1.3 实验设计

| 项 | 取值 | 依据 |
|:--|:--|:--|
| 档位（real_submit_probability） | **0.15（对照，已有数据）** / **0.5** / **0.8** | 0.15=现状对照；0.5/0.8 为真机主导档 |
| 策略 | PPO（ppo_best_model_16dim.zip）vs FCFS | 权威对比对象 |
| 每档每策略 seeds | ≥ 8（预算允许则 10） | 配对设计 N=8 时 Wilcoxon 最小可达显著 |
| episode 长度 | 100 步（缩减以控制机时） | 原权威 200 步；真机任务数 = 步数×概率 |
| 机器 | tianyan-287（带连字符） | 与 v2 权威一致 |
| shots / 电路 | 32 / H Q1、M Q1 | 与 v2 权威一致（tianyan287_multiseed.py:80 SHOTS=32） |
| 配对方式 | 同 seed 先 PPO 后 FCFS（或随机交替） | 消除 seed 间方差 |
| 种子序列 | seed = 42 + i×137（i=0..N-1） | 与噪声配对实验同法 |
| 执行脚本 | `scripts/real_machine/ppo_closed_loop_async.py --real-submit-prob X`（现有脚本，**显式传参**）+ `EnvBasedFCFSScheduler` 对照 | 已有脚本支持该参数（:31 文档示例 --real-submit-prob 0.3） |

### 1.4 机时预算公式（启动前必算）

```
单 seed 真机任务数 ≈ episode_steps × real_submit_probability
预算(任务数) = N_seeds × 2 策略 × episode_steps × prob
0.5 档 N=8：8×2×100×0.5 = 800 真机任务
0.8 档 N=8：8×2×100×0.8 = 1280 真机任务
```
按剩余机时额度反推 N（额度未知时先跑 0.5 档 N=4 的成本评估，再决定完整规模）。

### 1.5 统计口径（预注册锁定，与 v2 一致）

- 主检验：配对 Wilcoxon signed-rank（单侧 H1: PPO > FCFS，同 seed 配对）
- 辅检验：Welch t、均值差 95% CI、Cliff's delta / 配对 rank-biserial
- 显著性：α=0.05（单侧）；多重比较：3 档位各独立预注册（不做跨档位校正，逐档报告）
- 功效：N=8、配对 d_z 需 ≥1.1 才达 80% 功效——**预先声明**小样本下不显著≠无差异
- 输出 JSON schema（对齐 noise_paired JSON 风格）：
  ```json
  {"config": {"experiment": "prereg_real_high_ratio", "real_submit_probability": 0.5,
    "seeds": [...], "episode_steps": 100, "machine": "tianyan-287", "shots": 32,
    "circuit": "H Q1/M Q1", "strategy_order": "paired", "timestamp": "..."},
   "raw_data": {"per_seed": {"PPO": [...], "FCFS": [...], "real_tasks_completed": n,
    "real_reward_share": [...]}},
   "statistics": {"wilcoxon": {...}, "welch": {...}, "ci_95": [...], "effect": {...}}}
  ```
- 入库路径：`results/real_machine/prereg_high_ratio_<timestamp>.json` +
  `results/reports/real_machine_high_ratio_report.md`（模板参考 multiseed_real_machine_report_10seeds_v2.md）

### 1.6 门禁衔接

- 新实验段：`config/statistics.yaml` 新增 `real_machine_high_ratio` 段（data_source、
  n、p 值、结论）；`check_stats_consistency.py` 的 `_KNOWN_LEGIT_P_VALUES` 登记新 p 值；
  `audit_authoritative_metrics.py` 若扫描真机实验需同步豁免/登记。

---

## 实验②：编译层真机验证（AI→量子第二支柱真机化，8/20-9/5，先 smoke test）

### 2.1 动机

> 编译层目前只有 SWAP 数仿真对比（深电路子集 33/33 全胜 p<0.001，主检验 p=0.177
> 方向性）。“AI 赋能量子”如果能拿到**真机保真度证据**，双向赋能叙事不再一边倒。

### 2.2 技术前提（必须先验证，风险最高点）

- 编译模型 `ppo_compilation_agent.zip` 训练于 **4×4 2D 网格**拓扑；天衍-287 为
  10×11 网格（105 数据+182 耦合）——**拓扑不同，需嵌入映射**：
  - 检查天衍-287 耦合图是否含 4×4 子网格（10×11 网格含多个 4×4 子网格，需用
    真机 topology 数据确认，`src/api/` 或文档中查 coupling map）；
  - 把 PPO/SABRE 编译输出（4×4 布局）映射到真机子网格坐标后提交执行。
- **smoke test（预注册门槛）**：先选 3 个深电路（SABRE 高成本区间）做 PPO vs SABRE
  各 1 次真机执行，确认：任务可提交、MBS/保真度可读、差异方向与仿真一致。
  smoke test 通过才启动正式实验；不通过则本实验终止，如实记录"拓扑适配不可行"。

### 2.3 正式实验（smoke test 通过后）

- 电路集：从 `results/compilation_tail_per_circuit.json` 取 SABRE>25 SWAP 的深电路
  20 个 + SABRE≤10 浅电路 10 个（对照）
- 对比：同线路 PPO 编译布局 vs SABRE 编译布局，真机执行，指标 = MBS 保真度 /
  成功概率（对齐噪声实验的 MBS 口径，mbs 均值 0.8863±0.0874 可作基线参照）
- 预注册 H：PPO 编译电路真机 MBS ≥ SABRE 编译电路（配对 Wilcoxon，同电路配对）
- 风险披露：真机噪声（std 0.069）可能淹没编译布局差异 → 预注册条款：不显著则
  报告"编译差异在真机噪声下不可分辨"，与跨机器噪声异质性实验（p=0.0197）互相印证
- 入库：`results/real_machine/compilation_real_<timestamp>.json` +
  `results/reports/compilation_real_machine_report.md`

---

## 实验③：真机 v2 扩样 N=10 → N=20（成功率最高，可最先跑）

### 3.1 动机

> 现 v2 权威（N=10/组）效应量 d=5.33、p=5.84e-07——**效应量大到可疑**（小样本
> 幸存者偏差是评委标准质疑）。扩到 N=20 让 CI 收紧，把"可疑的大效应"变"可信的显著"。

### 3.2 设计

- 复用 `scripts/real_machine/run_15seeds_multistrategy.py`（已按 N≥15 设计、Task ID
  100% 留档）；扩展至 N=20 seeds × 3 策略（PPO/SJF/FCFS）× 1 真机任务/run = 60 次
  真机调用（新增 30 次）
- 协议与 v2 完全一致：tianyan-287、96 步/episode、32 shots、H Q1/M Q1、
  unified_protocol=true
- 统计：统一 Welch t + Bonferroni（3 比较，α=0.0167），与 v2 报告同构
- 产出：`results/real_machine/tianyan287_multiseed/multiseed_data_<ts>.json` +
  报告 v3（或在 v2 报告追加 §扩样）；更新 `statistics.yaml real_machine_10seed_v2`
  的 n 与 p 值（或新增 v3 段并 supersedes v2）
- 若扩样后 d 仍 >3（异常大），需在报告增加效应量异常分析（如 reward 分布双峰、
  真机任务占比 1/96 的机制说明）——预注册中声明该检查项

---

## 时间线与决策门槛（建议）

| 时间 | 动作 | 门槛 |
|:--|:--|:--|
| 8/14-8/16 | 实验③启动（成本最低）+ ①的 0.5 档成本评估（N=4） | ③脚本 dry-run 通过 |
| 8/16-8/20 | 实验② smoke test（拓扑适配验证） | 通过才启动正式；不通过即终止并记录 |
| 8/20-9/1 | 实验①完整规模 + ②正式（若 smoke 通过）；③完成并入库 | 每个结果 JSON 入库 |
| 9/1-9/5 | 全链统计重算、statistics.yaml 新段、门禁登记、文档同步 | `reproduce_authoritative.py` exit 0 |
| 9/5-9/10 | 冻结：重打 tag（新 HEAD）+ `validate_submission.py --pack` + dist 旧包清理 | validate 0 错误 |
| 9/15 | 官方提交截止 | — |

**每条红线**：① 不改默认参数与行为代码；② 预注册先行、负结果照常入库；
③ 每次实验后跑 `reproduce_authoritative.py` 确认权威数字零污染；
④ 与另一会话（opencode）并行作业时，先 `git status` 确认无冲突文件再入库。
