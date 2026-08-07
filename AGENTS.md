# AGENTS.md — 量子RL调度系统项目通用记忆

> 此文件供所有 AI Agent（CodeBuddy / TRAE / Claude / Cursor 等）读取，以快速理解项目全貌。
> 每次重要变更后请更新本文档的"最后更新"日期和对应章节。
> Version: 9.1.0

**最后更新**：2026-08-07（8.7-v2 审查修复：演示视频脚本 demo_video_final_script.md 完整配音稿段落4仍残留废弃数字（奖励2349/比FCFS+123.4%/比随机+163.3%/多机+36.8%），且"噪声感知奖励整形是最大杠杆"叙事与噪声负向结论冲突，已全部对齐权威口径（1983/与FCFS+20.2%/多机协同与独立PPO+84.6%/压力测试+91.4%），等待时间"+51%帕累托权衡"改为权威"-14.0%双优"；门禁加固：check_stats_consistency 新增中文数字归一化（_normalize_chinese_numerals，堵住"百分之X"中文表述绕过ASCII黑名单的漏洞）+ 新增123.4%/163.3%/36.8%/等待时间51%黑名单 + _ORPHAN_DEPRECATED增补2349；新增回归测试16用例；前序：8.7 审查 P2 收尾：PPT 覆盖率数字改为 CI 门禁口径 + tianyan_cqlib 提交空结果返回 None 修复（避免 "None"/"[]" 无效 task_id 追踪）+ compilation_env SWAP 逻辑复核确认正确；前序：8.7 审查 P0/P1 完成 commit 83a299a：编译层+38.5%降级为事后子集方向性证据 + 噪声反馈负向措辞统一 + 利用率-3.3%全仓废弃+7.9% + check_stats_consistency 新增A9孤立废弃值检测 + run_issue_38_67 DQN维度修复；前序：8.6 评审修复：新增交付模型冒烟加载测试 tests/test_model_smoke_load.py + 移除 CI --ignore 改 -m "not benchmark"（模型加载路径纳入主套件覆盖）+ 测试用例数同步3697；前序：8.2 拉取最新 main 并 rebase：清理+88.3%权威残留统一16维口径(#766) + 噪声N=10标废弃统一N=25(#831) + submission打包修复 + 测试用例数同步3697；前序：8.1 #766/#831文档清理：全量扫描+88.3%残留，为剩余报告统一加"14维旧口径已废弃"声明并指向16维权威（+20.2%, N=250, Welch t p=1.449e-66）；修正strategy_comparison_report_v4/authoritative_metric_audit/real_machine_statistical_significance等历史快照引导口径；测试用例数统一为3696；check_doc_sync全绿；validate_submission.py打包修复（git_tag跳过+输出zip排除+CODE_ARCHIVE构建，15GB→5.5MB）；#873收尾（mypy/ruff/format全绿，46测试通过）（前序：7.31第五轮benchmark 5.x兼容修复：修复test_scheduler_cache_benchmark.py(8测试)+test_annealing_benchmark.py(7测试)的pytest-benchmark 5.x stats兼容性（TypeError: 'NoneType' object is not subscriptable），新增_get_stat helper安全访问stats属性/字典/嵌套对象，PR#866 Windows CI 3个Python版本失败→待CI验证；前序：7.31第四轮自审issue修复完成：处理自提10个issue(#840-#849)，修复12个测试失败→0 failed（3519 passed/25 skipped）；#840修复72测试失败(17/16维+大小写+ortools+DAG)；#841回退PR#759观测维度替换(保预训练模型兼容)；#842版本号0.5.0→9.1.0；#843 8份报告+88.3%加废弃声明；#844 Cohen's d 5.33/5.64统一；#845测试用例数统一为实测值；#847 marl.py nosec B614；#848 .gitignore补coverage/retry临时文件；#849 MODELS.md版本头注；MARL flaky test改为多机协同断言；修复test_api.py 4个预存失败(PR#717 _call_with_retry+QCIS验证)；修复test_performance_benchmarks.py 6个预存失败(pytest-benchmark 5.x API兼容)）（前序：7.31第三轮终审查完成：审查20个开放PR，合并6个(#611/#610/#612/#609/#601/#616)，关闭7个(main已有等效实现)，7个需修改(#614/#615/#613/#604/#603/#602/#600)；新增公平性特性：#587(公平性惩罚)、#588(公平性观测开关)、#585(观测维度消融配置)；修复pr_patrol文档Cohen's d=5.64→5.33统计口径一致性；编译环境可配置规模(#594/#616)：支持自定义物理比特数和耦合图，天衍-287预设10x11网格）（前序：退火优雅降级：默认关闭+依赖可选化+deprecated标注；量子赋能AI主方向为真机噪声反馈；新增编译AI/VQE/OR-Tools）（完成P0/P1批次issues清理：关闭16个(#94/#97/#98/#102/#114/#115/#117/#118/#119/#120/#122/#148/#150/#153/#162/#194)；新增分层QUBO退火模式(#148)、退火权重放大机制+介入率诊断(#194)、私有方法重构(#153)、状态持久化设计文档(#114)、扩展性梯度测试(#117)、覆盖率提升env\_real\_machine 29%→97%/marl 64%→99%(#97/#98)、变异测试增强86用例(#122)、权威市场数据9源+10篇2024-2026论文(#115/#119)；7.31第三轮终审查：处理30个审查issue(#776-#805)，0个open PR待修改；3519测试用例全通过）

***

## 开始工作前必读

### Git 推送规则

| 你是谁        | 怎么推送                                                     |
| ---------- | -------------------------------------------------------- |
| **普通队友**   | 创建功能分支 → `git push origin feature/xxx` → 创建 PR → 1人审批后合并 |
| **管理员/瑞哥** | `git push origin main`（GitHub 原生分支保护已启用）                 |

**Commit 格式**：

```
<type>: <简短描述>
feat / fix / docs / test / refactor / chore
```

***

## 1. 项目概述

**作品名称**：量子RL驱动的天衍云平台智能调度系统
**所属比赛**：2026年"揭榜挂帅"擂台赛 — 榜题"量子AI双向赋能的研究与应用探索"
**主办方**：共青团中央主办 / 中国电信发榜 / 中电信量子执行
**团队人数**：8人（含负责人）
**负责人**：瑞哥（GitHub: xiabai2008）

**核心创新—双向赋能**：

- AI 赋能 量子计算：调度层(PPO +20.2%, p<0.001) + 编译层(PPO替代SABRE，公平对比v2，Issue #451；4×4 2D网格拓扑下同池配对60电路，深电路N=80扩充（Issue #559）：SWAP减少+38.5%（Wilcoxon p=2.75e-02显著），全60电路p=8.40e-01不显著，浅/中电路无优势（诚实披露）；原-76.4%为不公平对比已废弃)
- 量子 赋能 AI：真机噪声特征建模与PPO噪声敏感性评估（tianyan-287 H门1024 shots→保真度0.976→噪声分布建模→PPO噪声敏感性基准，负向证据：噪声使奖励下降12.43%，N=25配对检验 p=2.98e-08。注：单seed等待时间-5.7%为探索性结果，10seeds分布实验揭示噪声对RL的挑战）
- 退火模块：探索性功能，默认关闭，不再投入开发（20seeds统计不显著 p=0.9430，实为经典模拟退火）
- 量化目标：综合调度收益+20.2%（核心目标，已达成）；资源利用率-3.3%（vs 真实 FCFS，N=250，多目标权衡维度，赛题≥30%目标未达成）

**2026-07-29 新增特性**：

- **公平性特性**（#587/#588/#585）：公平性惩罚机制（多租户Jain's指数优化）、公平性观测开关（可配置是否在观测空间中包含公平性维度）、观测维度消融配置（支持灵活消融实验，区分各观测维度贡献）
- **可配置编译环境**（#594/#616）：编译环境支持自定义物理比特数与耦合图拓扑，天衍-287预设10×11网格拓扑，便于跨硬件适配与扩展性实验

**目标平台**：天衍云平台真机"天衍-287"（105数据比特+182耦合比特超导量子计算机，搭载祖冲之三号同款芯片）

**仓库地址**：<https://github.com/xiabai2008/quantum-rl-scheduler>

## 2. 关键时间节点

| 日期         | 事项                          | 状态  |
| ---------- | --------------------------- | --- |
| 2026-06-30 | 报名截止                        | 已通过 |
| 2026-07-01 | Track A 工程收尾 / Track B 比赛材料 | 已完成 |
| 2026-07-09 | P0可信度修复（依赖/统计/数字）           | 已完成 |
| 2026-07-29 | PR审查+公平性特性+文档更新             | 已完成 |
| 2026-08-15 | 代码冻结                        | 📅  |
| 2026-09-15 | 作品提交截止                      | 📅  |
| 2026-09-30 | 初审结果公布                      | 📅  |
| 2026-11    | 终审擂台赛                       | 📅  |

## 3. 项目代码结构（v9.1）

```
quantum-rl-scheduler/
├── AGENTS.md                     # 本文档
├── README.md                     # 项目介绍 + 快速开始
├── requirements.txt              # Python 依赖清单（含dimod/dwave-neal）
├── requirements-quantum.txt      # 真机可选依赖（cqlib）
├── pyproject.toml                # 统一配置（ruff/bandit/mypy/pytest/coverage）
├── mypy.ini                      # 类型检查（8项严格配置，仅2模块豁免：annealing/scripts）
├── .editorconfig                 # 跨编辑器编码风格统一
├── .pre-commit-config.yaml       # Git pre-commit 自动检查
├── .env.example                  # 环境变量模板
├── CONTRIBUTING.md               # 贡献指南
├── Dockerfile + docker-compose.yml  # 一键部署

├── src/                          # 源代码（64 个 .py 文件）
│   ├── exceptions.py             # 统一异常体系（8类）
│   ├── config/                   # 配置管理（settings.py, schema.py）
│   ├── scheduler/                # 调度引擎（核心模块，~23文件）
│   │   ├── parser.py             # 量子任务解析
│   │   ├── env.py                # Gymnasium调度环境入口（16维/异质化/多机器）
│   │   ├── env_observation.py    # 观测空间（16维）
│   │   ├── env_dynamics.py       # 环境动力学（泊松任务生成）
│   │   ├── env_machines.py       # 多机器管理
│   │   ├── env_reward.py         # 奖励函数
│   │   ├── env_render.py         # 渲染
│   │   ├── env_types.py          # 类型定义（OBS_DIM=16）
│   │   ├── env_real_machine.py   # 真机集成
│   │   ├── agent.py              # DQN 智能体
│   │   ├── ppo_agent.py          # PPO 智能体
│   │   ├── networks.py           # 神经网络
│   │   ├── training.py           # 训练循环
│   │   ├── callbacks.py          # 训练回调
│   │   ├── marl.py               # MAPPO 多智能体调度
│   │   ├── multi_objective_env.py # 多目标奖励包装器
│   │   ├── async_annealing_callback.py # 异步退火回调
│   │   ├── baselines.py          # 基线启发式策略
│   │   ├── ablation.py           # 消融实验
│   │   ├── dag_scheduler.py      # DAG调度
│   │   ├── hybrid_scheduler.py   # 混合调度器
│   │   ├── tenant.py             # 多租户
│   │   ├── checkpoint_manager.py # 检查点管理
│   │   ├── training_logger.py    # 训练日志
│   │   ├── explainability.py     # 可解释性
│   │   ├── export.py             # 模型导出
│   │   └── cache.py              # 缓存
│   ├── api/                      # API层（~7文件）
│   │   ├── tianyan_client.py     # 天衍云 API 客户端
│   │   ├── tianyan_cqlib.py      # cqlib 真机客户端 + 多机器协调器
│   │   ├── cqlib_recorder.py     # 真机响应录制/回放客户端（Issue #175）
│   │   ├── mock_client.py        # Mock API 客户端
│   │   ├── circuit_breaker.py    # 熔断器（CLOSED/OPEN/HALF_OPEN）
│   │   └── quota_tracker.py      # 配额追踪
│   ├── quantum/                  # 量子计算（~3文件）
│   │   ├── annealing.py          # 量子退火优化器(探索性)

│   │   └── annealing_loop.py     # 异步退火闭环控制器
│   │   └── compilation_env.py  # 电路编译AI（PPO替代SABRE）
│   ├── evaluation/               # 评估模块（~4文件，Issue #170 防泄漏/OOD）
│   │   ├── data_split.py         # 数据分割（防泄漏）
│   │   ├── blind_test.py         # 留出盲测评估
│   │   └── ood_generalization.py # 分布外泛化验证
│   ├── visualization/            # Web监控（~10文件 + Vue3前端）
│   │   ├── app.py               # FastAPI 入口（含对战状态管理+前端静态服务）
│   │   ├── routes.py             # 路由（含/metrics端点+/api/explainability+/api/battle）
│   │   ├── state.py              # 共享全局状态唯一定义（Issue #179 打破循环依赖）
│   │   ├── simulator.py          # 仿真器（PPO真实env.step调度）
│   │   ├── websocket_handler.py  # WebSocket
│   │   ├── connection.py         # 连接管理
│   │   ├── fallback_template.py  # 降级模板
│   │   ├── models.py             # 数据模型
│   │   └── frontend/             # Vue3 前端（DecisionMagnifier/BattlePanel等组件）
│   └── utils/                    # 工具（~8文件）
│       ├── helpers.py            # 工具函数
│       ├── metrics.py            # Prometheus 7个指标
│       ├── stats_significance.py # 统计显著性检验
│       ├── platform_compat.py    # 平台兼容
│       ├── alerts.py             # 告警
│       └── seeds.py              # 随机种子管理

├── tests/                        # 测试（94+ 文件，3697 用例 + 21 benchmark = 3718）
│   ├── test_scheduler.py         # 调度环境测试
│   ├── test_marl.py              # MAPPO 测试
│   ├── test_annealing.py         # 量子启发式退火测试
│   ├── test_annealing_loop.py    # 异步退火闭环测试
│   ├── test_multi_objective.py   # 多目标奖励测试
│   ├── test_state_space.py       # 状态空间测试
│   ├── test_api.py               # API 层测试
│   ├── test_parser.py            # 解析器测试
│   ├── visualization/            # 可视化测试（Issue #730 拆分自 test_visualization.py）
│   │   ├── test_app.py           # app.py 辅助函数测试
│   │   ├── test_routes.py        # HTTP API 路由测试
│   │   ├── test_security.py      # 安全/输入校验测试
│   │   ├── test_state.py         # state.py 访问器测试
│   │   └── test_websocket.py     # WebSocket 端点测试
│   ├── test_helpers.py           # 工具函数测试
│   ├── test_property.py          # property-based testing
│   ├── test_callbacks.py         # 回调测试
│   ├── test_env_real_machine.py  # 真机环境测试
│   ├── test_baselines.py         # 基线策略测试
│   ├── test_stats_significance.py # 统计检验测试
│   ├── test_circuit_breaker.py   # 熔断器测试
│   ├── test_fairness_reward.py   # 公平性奖励测试（#587 公平性惩罚机制）
│   └── benchmarks/               # 性能基准

├── scripts/                      # 按功能分区
│   ├── cli.py                    # Click 统一入口（train/simulate/serve/demo）
│   ├── training/                 # train_agent.py, quick_train.py
│   ├── evaluation/               # run_simulation.py, run_multiseed_evaluation.py,
│   │                             # run_issue_38_67_experiments.py, statistical_significance.py,
│   │                             # preregistered_real_machine_analysis.py, multiseed_real_machine_analysis.py
│   ├── demo/                     # demo.py, demo_cqlib.py, demo_multi_machine.py
│   ├── testing/                  # e2e_test.py, calibrate_mock.py
│   ├── benchmarking/             # mock_vs_real.py, stress_test.py, high_load_fairness.py
│   ├── real_machine/             # tianyan287_experiment.py, tianyan287_multiseed.py
│   └── reporting/                # generate_report.py

├── models/                       # 训练模型（PPO/DQN 检查点）
├── results/
│   ├── reports/                  # 实验报告（18份，含statistical_validation.md, multiseed_real_machine_report.md, fair_scheduling_report.md, d3_reward_ablation_report.md, high_load_fairness_report.md）
│   ├── models/                   # 归档的权威模型（ppo_best_model_16dim.zip等）
│   ├── multiseed_evaluation/     # 多seed评估数据
│   ├── fair_comparison/          # 公平对比数据
│   ├── issue_experiments/        # Issue实验数据
│   └── real_machine/             # 真机实验数据（tianyan287/ + tianyan287_multiseed/）

├── docs/
│   ├── 新人上手指南.md            # 团队 onboarding
│   ├── 队友协同开发指南.md         # 精简版快速上手
│   ├── Git工作流.md              # 分支管理规范
│   ├── 团队分工.md               # 角色职责
│   ├── 开发计划.md               # 详细时间线
│   ├── requirements_traceability.md # 需求追溯矩阵
│   ├── defense_qa_handbook.md    # 答辩QA手册
│   ├── dependency_management.md  # 依赖管理
│   ├── api_reference.md          # API参考
│   ├── Code_Wiki.md              # 代码Wiki
│   ├── technical_bottlenecks.md  # 技术瓶颈分析（7项瓶颈+缓解策略）
│   ├── annealing_significance-defense.md # 退火显著性答辩策略（p=0.19应对话术）
│   ├── deployment.md # 部署架构（三阶段：原型→试点→生产）
│   ├── cross_hardware.md # 跨硬件兼容性（路线图+可扩展性论述）
│   └── value_quantification.md   # 价值量化报告

├── config/
│   ├── .env.example
│   ├── config.yaml
│   └── submission_manifest.yaml  # 提交清单（v9.1）

└── .github/
    └── workflows/
        ├── ci.yml                  # CI 4 Job：lint→test→typecheck→benchmarks
        └── pr-automation.yml       # PR 自动标签 + Commit 格式校验
```

## 4. 技术栈

| 层级   | 技术                                          | 用途                                                       |
| ---- | ------------------------------------------- | -------------------------------------------------------- |
| 语言   | Python 3.10+                                | 全部                                                       |
| RL   | Stable-Baselines3 (PPO/DQN/MAPPO)           | 双算法 + 多智能体                                               |
| RL   | Gymnasium                                   | 环境封装                                                     |
| DL   | PyTorch ≥2.0                                | 神经网络                                                     |
| 量子   | 天衍云 cqlib SDK                               | 105数据比特+182耦合比特超导处理器（可选，requirements-quantum.txt）        |
| 量子   | D-Wave dimod / dwave-neal                   | 量子启发式退火（QUBO+模拟退火，**探索性功能，默认关闭**，requirements.txt已注释为可选） |
| Web  | FastAPI + Uvicorn                           | 监控界面（routes.py含/metrics）                                 |
| 前端   | Vue3 + Echarts                              | 监控面板                                                     |
| CLI  | Click                                       | 统一命令行入口                                                  |
| 可观测  | Prometheus + prometheus\_client             | 7个指标（Gauge/Counter/Histogram），/metrics端点已暴露              |
| 统计   | SciPy                                       | 统计显著性检验（t/Welch/Mann-Whitney + Bonferroni校正）             |
| 代码质量 | ruff(10类) + mypy(8项) + bandit               | v1技术提升方案                                                 |
| CI   | GitHub Actions 4 Job + Codecov + Dependabot | 自动化质量门禁                                                  |

## 5. v1 技术提升方案落地成果

### 代码质量强化

- mypy：8项严格配置（disallow\_untyped\_defs + disallow\_incomplete\_defs + warn\_return\_any + strict\_equality 等），当前2模块豁免（annealing/scripts.\*）。2026-07-20 修复全部 26 个类型错误，CI mypy 从 baseline 升级为 strict mode
- ruff：完全替代 flake8 + black + isort，10类规则集（E/W/F/I/N/B/SIM/C4/UP/RUF）。2026-07-20 清理全部 142 个历史遗留错误，CI ruff check 从 --exit-zero 升级为严格阻断
- CI 工具栈对齐：2026-07-19 将 CI lint job 从 black+isort+flake8 迁移到 ruff format + ruff check + bandit，与 .pre-commit-config.yaml 完全一致
  - ruff format --check：严格阻断（格式基线）
  - ruff check：严格阻断（142→0，2026-07-20 完成）
  - mypy：严格阻断（26→0，2026-07-20 完成）
  - bandit：严格阻断（安全扫描）

### 工程韧性

- 统一异常体系：8类异常（QuantumSchedulerError → 5子类），code + retryable 语义
- API 熔断器：CLOSED/OPEN/HALF\_OPEN 三态转换
- Prometheus 指标：7个指标覆盖调度/API/退火三个维度，/metrics端点在routes.py暴露
- Click CLI：train/simulate/serve/demo 四子命令统一入口
- 依赖可复现：requirements.txt 核心依赖；cqlib 通过 requirements-quantum.txt 安装；dimod/dwave-neal 已注释为可选（退火默认关闭）

### 测试升级

- 测试文件：5 → 76（+71个专用测试模块）
- 测试用例：100+ → 3697（+21 benchmark = 3718 总收集）（Issue #398 P1-4 统一口径，pytest --collect-only 实测，2026-08-01）
- 测试用例：100+ → 3697（+21 benchmark = 3718 总收集）（Issue #398 P1-4 统一口径，pytest --collect-only 实测，2026-08-02）
- CI 强制覆盖率：40% → 80%（CI 实测通过，pyproject.toml `fail_under=80`；历史值 93.58% 随测试集扩充已变化，以 CI 报告为准）
- 新增：property-based testing + 性能基准测试 + mutation testing + 统计显著性检验

### 实验可信度（v8新增）

- 多seed评估：50 seeds × 5 episodes = 250 次独立运行（N=250）
- 统计显著性：Bonferroni校正，PPO vs FCFS p=7.56e-12（Welch t检验，vs 真实 FCFS；旧 p=1.449e-66 对应 Hybrid-Default 基线，8.5 已诚实化）
- 权威数字锁定：PPO=1982.69±557.25 vs FCFS=1648.91±502.95，提升 +20.2%

## 6. 权威实验成果（50seed N=250 验证，v9.1 基于16维交付模型重评，2026-07-29）

> **权威实验配置**：16维交付模型（ppo_best_model_16dim.zip）、50 seeds × 5 episodes = 250次独立运行（N=250）、200步/episode、泊松到达λ=0.5
> **统计显著性**：PPO vs FCFS（真实 FCFS）Welch t 检验 p=7.56e-12（8.5 基线诚实化），Bonferroni校正后显著

|  排名 | 策略             |     平均奖励    |   标准差   | 提升 vs FCFS |
| :-: | :------------- | :---------: | :-----: | :--------: |
|  1  | **PPO**        | **1982.69** | 557.25 | **+20.2%** |
|  2  | FCFS           |   1648.91   |  502.95  |     基线     |
|  3  | SJF            |   774.86   |  275.74 |    -53.0%   |
|  4  | DQN（Random占位） |    602.37   |  262.09 |   -63.5%   |
|  5  | Random         |    602.37   |  262.09 |   -63.5%   |
|  6  | Greedy         |    80.71   |  549.12 |   -95.1%  |
|  7  | Quantum-Only   |   -826.59   |  263.63 |   -150.1%  |
|  8  | Classical-Only |   -1075.49  |  75.04  |   -165.2%  |

> 注：v9 已删除 DQN 模型，DQN 策略位使用 Random 策略占位（见 `config/statistics.yaml` strategy_summary.DQN.note）。

### 消融实验（参考）

| 实验          | 核心结论                                                                                        |
| ----------- | ------------------------------------------------------------------------------------------- |
| 五维消融        | D1算法+20.2% > D4多机+86.3% > D5退火+6.4% > D2状态+2.1%                                             |
| 压力测试        | 4场景PPO综合稳定性最强；量子波动场景PPO +91.4%                                                              |
| 真机验证        | **可用性验证**：315次SDK调用100%成功，全链路验证通过                                                           |
| **多seed真机** | **小样本策略对比**（N=10/组，v2权威）：PPO d=5.33 vs FCFS, p=6.83e-04, Bonferroni显著（小样本探索性结果，效应量异常大，待更多seeds验证） |

> **⚠️ 真机验证结论边界（Issue #128）**
>
> 真机实验结论严格区分为**可用性验证**和**性能验证**两级：
>
> - ✅ **可用性验证（已达成）**：SDK认证、任务提交、状态轮询、结果获取全链路验证通过，315次真机调用100%成功（284次主验证+31次审计补充）；Issue #128 新增 tianyan176 H门任务成功（P(0)=50.9%, P(1)=49.1%）
> - ⚠️ **性能验证（不充分）**：mixed\_real vs simulation p=0.344不显著（N=5, 需N≥18）；多seed策略对比p=6.83e-04显著，但真机 reward 占比极低（1/96步），策略间差异主要由仿真 reward 驱动
> - **性能提升结论由仿真实验支撑**：PPO vs FCFS +20.2%（N=250, p=7.56e-12，vs 真实 FCFS）
>
> 详见 `docs/real_machine_verification_boundary.md`

### 多seed真机实验 v2（2026-07-27 权威版，N=10 per group，supersedes N=5）

> ⚠️ **真机版本选择规则（团队通用）**：对外答辩/报告/PPT 一律使用本节 N=10 v2 权威数值；旧版 N=5 表格仅用于解释数据迭代历史，严禁作为性能基准引用。

> **实验配置**：10 seeds × 3策略 \[PPO,FCFS,SJF] × 1真机任务/run = 30次运行
> **真机平台**：天衍-287（实际回退至 tianyan176），96步/episode，泊松到达λ=0.5
> **统计方法**：Cohen's d + 95% CI（效应量决策范式），Bonferroni校正
> **⚠️ 边界说明**：真机任务成功完成（30/30, mock=false），但真机 reward 占总 reward 比例极低（1/96步），策略间差异主要由仿真 reward 驱动

|    策略   |  N  |        均值       |     标准差    |     min     |     max     |
| :-----: | :-: | :-------------: | :--------: | :---------: | :---------: |
| **PPO** |  10 | **1736.32** | 355.78 | 1224.13 | 2293.18 |
|   SJF   |  10 |    575.33   | 237.69 |  383.93 |  854.43 |
|   FCFS  |  10 |    383.00   |  49.13  |  288.77 |  410.23 |

|      比较     | Cohen's d | 效应等级 |         95% CI         |    p值    | Bonferroni |   判定   |
| :---------: | :-------: | :--: | :--------------------: | :------: | :--------: | :----: |
| PPO vs FCFS |    5.33   |  大效应 | [1097.83, 1608.82] | p<0.001 |     显著     | **支持** |
|  PPO vs SJF |    4.04   |  大效应 | [911.78, 1410.20]  | p<0.001 |     显著     | **支持** |
| SJF vs FCFS |    1.13   |  大效应 |  [-38.82, 423.48]  |   0.080  |     不显著    |   不支持  |

| 版本 | 样本量 | PPO均值 / FCFS均值 | Cohen's d | 对外使用 |
| :--- | :----: | :-----------------: | :-------: | :------: |
| N=5 旧版 | 5/组 | 1665.22 / 353.22 | 5.33 | ❌ 仅用于历史追溯 |
| **N=10 v2 权威** | **10/组** | **1736.32 / 383.00** | **5.33** | ✅ 答辩/报告唯一标准 |

详见 `results/reports/multiseed_real_machine_report.md`。

详见 `results/reports/` 目录（共18份报告，含统计显著性检验报告、多seed真机实验报告、公平调度报告、D3奖励消融报告、高负载公平调度报告）。

## 7. 比赛材料

| 材料                | 路径                                                  | 状态                                                |
| ----------------- | --------------------------------------------------- | ------------------------------------------------- |
| 答辩PPT（制作中）.pptx\` | ✅ 制作中（+20.2%，p=7.56e-12，N=250，315次真机调用（284+31审计口径），新增2页应用价值）    | <br />                                            |
| 技术白皮书（7章）         | `docs/technical_whitepaper.pdf`                     | ✅ 已完成（+20.2%，315真机调用，100%成功率，2026-07-27 v9.1 口径） |
| 价值量化报告            | `docs/value_quantification.md`                      | ✅ 已完成（6节，10项指标，ROI分析，VQE场景案例）                     |
| 技术瓶颈分析            | `docs/technical_bottlenecks.md`                     | ✅ 已完成（7项瓶颈+缓解策略，2026-07-24）                       |
| 公平调度实验报告          | `results/reports/fair_scheduling_report.md`         | ✅ 已完成（5租户Jain's指数=0.9875，PPO总奖励+57.6%，2026-07-24） |
| 退火显著性答辩策略         | `docs/annealing_significance-defense.md`            | ✅ 已完成（5类评委问题应对话术，p=0.9430（20seeds）→delta=0.40，2026-07-24）    |
| 部署架构文档            | `docs/deployment.md`                                | ✅ 已完成（三阶段部署路径+ONNX优化+K8s配置，2026-07-24）            |
| D3奖励消融报告          | `results/reports/d3_reward_ablation_report.md`      | ✅ 已完成（7预设×2策略×10seeds，策略-奖励耦合分析，2026-07-24）       |
| 高负载公平调度报告         | `results/reports/high_load_fairness_report.md`      | ✅ 已完成（λ=1.2高负载5租户公平调度，PPO/FCFS/SJF对比，2026-07-25）  |
| MAPPO热力图          | `results/reports/marl_heatmap.html`                 | ✅ 已完成（多智能体调度策略热力图可视化，2026-07-25）                  |
| PPT数据分离可视化        | `results/reports/ppt_separation_visualization.html` | ✅ 已完成（策略奖励分布分离度可视化，2026-07-25）                    |
| 演示视频分镜脚本          | `演示视频分镜脚本.md`                                       | 已完成                                               |
| 演示视频（5分钟）         | —                                                   | 待录制                                               |
| 统计显著性报告           | `results/reports/statistical_validation.md`         | ✅ 已完成                                             |
| 分层退火对比报告          | `results/reports/hierarchical_annealing_report.md`  | ✅ 已完成（参数覆盖11.9%→100%，8.4x提升，2026-07-26）           |
| 扩展性梯度测试报告         | `results/reports/scalability_test.md`               | ✅ 已完成（5规模×3策略，PPO决策延迟O(1)，2026-07-26）             |
| 退火lr扫描报告          | `results/reports/annealing_lr_sweep_report.md`      | ✅ 已完成（根因诊断：lr=0.01导致退火无效化，2026-07-26）             |
| 状态持久化设计           | `docs/state_persistence_design.md`                  | ✅ 已完成（SQLite/Redis双方案+MVP路线图，2026-07-26）          |
| SOTA对比表           | `docs/sota_comparison.md`                           | ✅ 已完成（10篇2024-2026论文+差异化定位，2026-07-26）            |

## 8. 当前进度

> **仓库状态快照（2026-08-07）**：1 open PR（#934）/ 2 open issue（#933、#846）（doc-sync 检查用；PR/issue 数实时变化，仅作快照）

```
v1 技术提升   ████████████████████ 100%（ruff 142→0 + mypy 26→0 + CI全严格阻断 + 覆盖率80%）
Track A       ████████████████████ 100%
Track B       ████████████████████ 100%（PPT/白皮书/视频脚本/实验数据）
P0 可信度修复  ████████████████████ 100%（依赖/统计/数字锁定 2026-07-09）
Track C       ████████████████████ 100%（mypy 26→0 + 覆盖率 60%→80% + ruff 142→0）
真机闭环       ████████████████████ 100%（天衍-287套餐已开通，30个真机任务全部成功；单点实验PPO经典保真度0.9924，多seed实验使用测量平衡分数MBS，PPO均值0.8965）
深度分析文档   ████████████████████ 100%（技术瓶颈/公平调度/退火答辩/部署架构/D3消融，2026-07-24）
Demo可视化增强  ████████████████████ 100%（依赖清理/Docker优化/PPO调度修复/决策放大镜/对战面板/高负载公平调度/MARL热力图，2026-07-25）
Issues全面清理  ████████████████████ 100%（关闭16个P0/P1 open issues(#94-#194)，新增分层退火/退火诊断/扩展性测试/持久化设计/SOTA对比/市场数据/覆盖率提升，2026-07-26；当前仍有39个open issues(#430+批次为P2/P3清理与扩展实验)）
PR审查        ████████████████░░░░  80%（2026-07-29审查20个开放PR：合并6个(#611/#610/#612/#609/#601/#616)、关闭7个(main已有等效实现)、7个待修改(#614/#615/#613/#604/#603/#602/#600)）
公平性特性      ████████████████████ 100%（2026-07-29：#587公平性惩罚+#588公平性观测开关+#585观测维度消融配置+test_fairness_reward.py）
提交校验       ███████████████████░  85%（7.31第三轮审查修复10个issue(#840-#849)，仍缺git tag/dist zip/演示视频/PPT，待8/15冻结前补齐）
```

## 9. 下一步

- ~~P1：mypy 豁免 6→2~~ ✅ 已完成（2026-07-20，26个错误全部修复，CI mypy 严格阻断）
- ~~P1：清理 142 个 ruff check 历史遗留 errors~~ ✅ 已完成（2026-07-20，142→0，CI 移除 --exit-zero）
- ~~P1：更新PPT/白皮书中的实验数字为+20.2%~~ ✅ 已完成（2026-07-20，15个md文件105处替换，.pptx/.docx 待瑞哥手动更新）
- ~~P2：测试覆盖率提升~~ ✅ 已完成（2026-07-20，66个新测试，覆盖率门槛60%→80%）
- ~~P3：Docker 一键复现~~ ✅ 已完成（2026-07-20，#163 关闭）
- ~~P0：深度分析文档（5项行动计划）~~ ✅ 已完成（2026-07-24，技术瓶颈/公平调度/退火答辩/部署架构/D3消融）
- ~~P2：Day2-3 依赖清理与Docker优化~~ ✅ 已完成（2026-07-25，移除9个未使用依赖，Dockerfile多阶段前端构建，simulator.py PPO真实调度修复）
- ~~P2：Day4-7 可视化增强与实验补充~~ ✅ 已完成（2026-07-25，决策放大镜+对战面板+5个API端点+高负载公平调度+MARL热力图+PPT分离可视化）
- ~~P2：PR审查与公平性特性~~ ✅ 已完成（2026-07-29，审查20个PR：合并6个/关闭7个/7个待修改；新增#587/#588/#585公平性特性；可配置编译环境#594/#616；修复pr_patrol文档Cohen's d统计口径一致性）
- **P1**：处理7个待修改PR（#614/#615/#613/#604/#603/#602/#600）— 需作者按审查意见修订后重新提交
- **P2**：演示视频录制（4-5分钟，1080p）— 需瑞哥人工录制
- **P2**：PPT/白皮书 .pptx/.docx 源文件数字更新 — 需瑞哥手动更新
- **P3**：8/15代码冻结，9/15前打v9.1-submission标签
  - 冻结前检查清单:
    1. 所有 CI 检查全绿（lint/test/typecheck/security）
    2. `python scripts/ci/validate_submission.py --check` 通过
    3. PPT/白皮书数字与代码权威数字一致（+20.2%，16维交付模型权威实验N=250，p=7.56e-12，旧14维+88.3%已废弃）
    4. 演示视频已就位
    5. 打标签: `git tag -a v9.1-submission -m "v9.1 提交版本" && git push origin v9.1-submission`
    6. 打包: `python scripts/ci/validate_submission.py --pack`

详见 workspace 根目录 `项目状态审查与下一步工作建议_2026-07-09.md`。

### 观测维度口径管理规范（Issue #129）

项目中存在两种观测维度，严格按以下规范使用：

| 维度                | 适用场景                     | 包装器                        |
| :---------------- | :----------------------- | :------------------------- |
| 16维（原生·交付标准）     | PPO训练/评估、真机实验、答辩提交       | 无（QuantumSchedulingEnv 原生） |
| 14维（编译层专用）       | 量子比特映射（编译PPO vs SABRE）   | 编译环境compilation_env.py独立实现 |
| 10维（Obs10Wrapper） | DQN基线对比（与旧版10维 DQN 公平对比） | Obs10Wrapper（截断前10维）       |

**口径切换声明要求**：

- 10维和16维结果不可直接比较；调度层16维与编译层14维是独立任务，指标不可跨任务比较
- 报告/表格必须标注观测维度
- PPO +20.2% 为16维交付模型权威对比结果（v9.1+，OBS_DIM=16，50seed×5episodes=250次独立运行），PPO 模型文件为 ppo_best_model_16dim.zip（v9 已由 14 维迁移至 16 维，旧版 dqn_best_model_10dim.zip / dqn_best_model_14dim.zip 已删除）。10维 Obs10Wrapper 仅用于与旧版 10维 DQN 模型的公平对比

详见 `docs/observation_dim_standard.md`

### 生产落地路径（Issue #130）

| 阶段       | 时间       | 目标                         |
| :------- | :------- | :------------------------- |
| 阶段1 竞赛交付 | 截止 08/15 | 代码冻结 + 交付物完善 + 答辩准备        |
| 阶段2 试点部署 | 08-10月   | 状态持久化 + Redis接入 + 监控告警完善   |
| 阶段3 生产部署 | 10-01月   | 多租户集成 + 高可用 + 性能调优         |
| 阶段4 规模化  | 01-06月   | 多硬件适配 + K8s云原生 + SLA 99.9% |

**生产就绪度评级**：研究原型向工程原型过渡，具备试点部署能力（综合 8.0/10）

详见 `docs/production_roadmap.md`

## 10. 团队信息

| GitHub 用户名      | 权限    |
| --------------- | ----- |
| xiabai2008      | Admin |
| heka-ky         | Write |
| zyhsga          | Write |
| NN2914          | Write |
| qpqpalalzmzm112 | Write |
| Jackhock-1      | Write |
| DUMNOX          | Write |
| K1660729        | Write |

## 11. 快速命令参考

```bash
# ── CLI 统一入口 ──
python scripts/cli.py train --timesteps 50000 --algorithm ppo
python scripts/cli.py simulate --num-tasks 200 --strategies all
python scripts/cli.py serve --port 8000
python scripts/cli.py demo --multi-machine

# ── 多Seed评估与统计检验 ──
python scripts/evaluation/run_multiseed_evaluation.py --seeds 10 --episodes 5
python scripts/evaluation/statistical_significance.py --input results/multiseed_evaluation/rewards_multiseed.json

# ── 代码质量 ──
ruff check src/ scripts/ tests/           # 代码检查
ruff format src/ scripts/ tests/          # 代码格式化
mypy src/                                  # 类型检查
bandit -r src/ -c pyproject.toml -ll      # 安全扫描
pre-commit run --all-files                 # pre-commit 全量检查

# ── 测试 ──
pytest tests/ --cov=src --cov-fail-under=80  # 测试 + 覆盖率
pytest tests/benchmarks/ --benchmark-only    # 性能基准

# ── Web ──
uvicorn src.visualization.app:app --reload --port 8000
curl localhost:8000/metrics                  # Prometheus指标

# ── 依赖安装 ──
pip install -r requirements.txt              # 基础依赖（含退火）
pip install -r requirements-quantum.txt      # 真机依赖（cqlib）

# ── Docker ──
docker-compose up -d
```

## 12. 重要文件路径速查

| 用途           | 路径                                                    |
| ------------ | ----------------------------------------------------- |
| 权威PPO模型（16维） | `deliverable_models/ppo_best_model_16dim.zip`         |
| 编译层PPO模型       | `deliverable_models/ppo_compilation_agent.zip`         |
| 归档模型目录       | `deliverable_models/`（已入库，详见 MODELS.md）               |
| 多seed评估数据    | `results/multiseed_evaluation/rewards_multiseed.json` |
| 统计显著性报告      | `results/reports/statistical_validation.md`           |
| 策略对比报告       | `results/reports/strategy_comparison.md`              |
| 提交清单         | `config/submission_manifest.yaml`                     |
| Obs10Wrapper | `scripts/evaluation/run_issue_38_67_experiments.py`   |
