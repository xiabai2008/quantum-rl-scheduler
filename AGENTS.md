# AGENTS.md — 量子RL调度系统项目通用记忆

> 此文件供所有 AI Agent（CodeBuddy / TRAE / Claude / Cursor 等）读取，以快速理解项目全貌。
> 每次重要变更后请更新本文档的"最后更新"日期和对应章节。

**最后更新**：2026-07-22（多seed真机实验完成：5 seeds × 3策略，PPO d=5.64 vs FCFS, p<0.001，Bonferroni校正后显著）

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
- AI 赋能 量子计算：用强化学习（RL）智能调度量子/经典任务
- 量子 赋能 AI：用量子退火（QUBO映射）加速 RL 决策
- 量化目标：资源利用率提升 ≥30%（等待时间为多目标优化中的权衡维度，非硬性指标）

**目标平台**：天衍云平台真机"天衍-287"（287量子比特超导量子计算机）

**仓库地址**：<https://github.com/xiabai2008/quantum-rl-scheduler>

## 2. 关键时间节点

| 日期         | 事项     | 状态      |
| ---------- | ------ | ------- |
| 2026-06-30 | 报名截止   | 已通过 |
| 2026-07-01 | Track A 工程收尾 / Track B 比赛材料 | 已完成 |
| 2026-07-09 | P0可信度修复（依赖/统计/数字） | 已完成 |
| 2026-08-15 | 代码冻结   | 📅 |
| 2026-09-15 | 作品提交截止 | 📅 |
| 2026-09-30 | 初审结果公布 | 📅 |
| 2026-11    | 终审擂台赛  | 📅 |

## 3. 项目代码结构（v8）

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

├── src/                          # 源代码（~57 文件）
│   ├── exceptions.py             # 统一异常体系（8类）
│   ├── config/                   # 配置管理（settings.py, schema.py）
│   ├── scheduler/                # 调度引擎（核心模块，~23文件）
│   │   ├── parser.py             # 量子任务解析
│   │   ├── env.py                # Gymnasium调度环境入口（14维/异质化/多机器）
│   │   ├── env_observation.py    # 观测空间（14维）
│   │   ├── env_dynamics.py       # 环境动力学（泊松任务生成）
│   │   ├── env_machines.py       # 多机器管理
│   │   ├── env_reward.py         # 奖励函数
│   │   ├── env_render.py         # 渲染
│   │   ├── env_types.py          # 类型定义（OBS_DIM=14）
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
│   ├── api/                      # API层（~6文件）
│   │   ├── tianyan_client.py     # 天衍云 API 客户端
│   │   ├── tianyan_cqlib.py      # cqlib 真机客户端 + 多机器协调器
│   │   ├── mock_client.py        # Mock API 客户端
│   │   ├── circuit_breaker.py    # 熔断器（CLOSED/OPEN/HALF_OPEN）
│   │   └── quota_tracker.py      # 配额追踪
│   ├── quantum/                  # 量子计算（~3文件）
│   │   ├── annealing.py          # 量子退火优化器
│   │   └── annealing_loop.py     # 异步退火闭环控制器
│   ├── visualization/            # Web监控（~8文件）
│   │   ├── app.py               # FastAPI 入口
│   │   ├── routes.py             # 路由（含/metrics端点）
│   │   ├── simulator.py          # 仿真器
│   │   └── websocket_handler.py  # WebSocket
│   └── utils/                    # 工具（~8文件）
│       ├── helpers.py            # 工具函数
│       ├── metrics.py            # Prometheus 7个指标
│       ├── stats_significance.py # 统计显著性检验
│       ├── platform_compat.py    # 平台兼容
│       ├── alerts.py             # 告警
│       └── seeds.py              # 随机种子管理

├── tests/                        # 测试（~42 文件，500+ 用例）
│   ├── test_scheduler.py         # 调度环境测试
│   ├── test_marl.py              # MAPPO 测试
│   ├── test_annealing.py         # 量子退火测试
│   ├── test_annealing_loop.py    # 异步退火闭环测试
│   ├── test_multi_objective.py   # 多目标奖励测试
│   ├── test_state_space.py       # 状态空间测试
│   ├── test_api.py               # API 层测试
│   ├── test_parser.py            # 解析器测试
│   ├── test_visualization.py     # 可视化测试
│   ├── test_helpers.py           # 工具函数测试
│   ├── test_property.py          # property-based testing
│   ├── test_callbacks.py         # 回调测试
│   ├── test_env_real_machine.py  # 真机环境测试
│   ├── test_baselines.py         # 基线策略测试
│   ├── test_stats_significance.py # 统计检验测试
│   ├── test_circuit_breaker.py   # 熔断器测试
│   └── benchmarks/               # 性能基准

├── scripts/                      # 按功能分区
│   ├── cli.py                    # Click 统一入口（train/simulate/serve/demo）
│   ├── training/                 # train_agent.py, quick_train.py
│   ├── evaluation/               # run_simulation.py, run_multiseed_evaluation.py,
│   │                             # run_issue_38_67_experiments.py, statistical_significance.py,
│   │                             # preregistered_real_machine_analysis.py, multiseed_real_machine_analysis.py
│   ├── demo/                     # demo.py, demo_cqlib.py, demo_multi_machine.py
│   ├── testing/                  # e2e_test.py, calibrate_mock.py
│   ├── benchmarking/             # mock_vs_real.py, stress_test.py
│   ├── real_machine/             # tianyan287_experiment.py, tianyan287_multiseed.py
│   └── reporting/                # generate_report.py

├── models/                       # 训练模型（PPO/DQN 检查点）
├── results/
│   ├── reports/                  # 实验报告（10份，含statistical_validation.md, multiseed_real_machine_report.md）
│   ├── models/                   # 归档的权威模型（ppo_best_10dim.zip等）
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
│   └── Code_Wiki.md              # 代码Wiki

├── config/
│   ├── .env.example
│   ├── config.yaml
│   └── submission_manifest.yaml  # 提交清单（v8.0）

└── .github/
    └── workflows/
        ├── ci.yml                  # CI 4 Job：lint→test→typecheck→benchmarks
        └── pr-automation.yml       # PR 自动标签 + Commit 格式校验
```

## 4. 技术栈

| 层级  | 技术                     | 用途               |
| --- | ---------------------- | ---------------- |
| 语言  | Python 3.10+                | 全部               |
| RL  | Stable-Baselines3 (PPO/DQN/MAPPO)    | 双算法 + 多智能体    |
| RL  | Gymnasium              | 环境封装             |
| DL  | PyTorch ≥2.0                | 神经网络             |
| 量子  | 天衍云 cqlib SDK              | 287量子比特超导处理器（可选，requirements-quantum.txt） |
| 量子  | D-Wave dimod / dwave-neal     | 量子退火（requirements.txt） |
| Web | FastAPI + Uvicorn      | 监控界面（routes.py含/metrics） |
| 前端  | Vue3 + Echarts         | 监控面板             |
| CLI | Click | 统一命令行入口 |
| 可观测 | Prometheus + prometheus_client | 7个指标（Gauge/Counter/Histogram），/metrics端点已暴露 |
| 统计 | SciPy | 统计显著性检验（t/Welch/Mann-Whitney + Bonferroni校正） |
| 代码质量 | ruff(10类) + mypy(8项) + bandit | v1技术提升方案 |
| CI | GitHub Actions 4 Job + Codecov + Dependabot | 自动化质量门禁 |

## 5. v1 技术提升方案落地成果

### 代码质量强化
- mypy：8项严格配置（disallow_untyped_defs + disallow_incomplete_defs + warn_return_any + strict_equality 等），当前2模块豁免（annealing/scripts.*）。2026-07-20 修复全部 26 个类型错误，CI mypy 从 baseline 升级为 strict mode
- ruff：完全替代 flake8 + black + isort，10类规则集（E/W/F/I/N/B/SIM/C4/UP/RUF）。2026-07-20 清理全部 142 个历史遗留错误，CI ruff check 从 --exit-zero 升级为严格阻断
- CI 工具栈对齐：2026-07-19 将 CI lint job 从 black+isort+flake8 迁移到 ruff format + ruff check + bandit，与 .pre-commit-config.yaml 完全一致
  - ruff format --check：严格阻断（格式基线）
  - ruff check：严格阻断（142→0，2026-07-20 完成）
  - mypy：严格阻断（26→0，2026-07-20 完成）
  - bandit：严格阻断（安全扫描）

### 工程韧性
- 统一异常体系：8类异常（QuantumSchedulerError → 5子类），code + retryable 语义
- API 熔断器：CLOSED/OPEN/HALF_OPEN 三态转换
- Prometheus 指标：7个指标覆盖调度/API/退火三个维度，/metrics端点在routes.py暴露
- Click CLI：train/simulate/serve/demo 四子命令统一入口
- 依赖可复现：requirements.txt 含 dimod/dwave-neal；cqlib 通过 requirements-quantum.txt 安装

### 测试升级
- 测试文件：5 → 49（+44个专用测试模块）
- 测试用例：100+ → 1663+
- CI 强制覆盖率：40% → 70%（实际 91%）
- 新增：property-based testing + 性能基准测试 + mutation testing + 统计显著性检验

### 实验可信度（v8新增）
- 多seed评估：50 seeds × 5 episodes = 250 次独立运行（N=250）
- 统计显著性：Bonferroni校正，PPO vs FCFS p<0.001（Welch t检验）
- 权威数字锁定：PPO=2747±437 vs FCFS=1459±8，提升 +88.3%

## 6. v8 实验成果（50seed N=250 验证，2026-07-19）

> **权威实验配置**：10维公平对比环境（Obs10Wrapper）、50 seeds × 5 episodes = 250次独立运行（N=250）、200步/episode、泊松到达λ=0.5
> **统计显著性**：PPO vs FCFS 使用 Welch t 检验，p=3.04e-11，Cohen's d=-1.70（大效应量），Bonferroni校正后显著

| 排名 | 策略 | 平均奖励 | 标准差 | 提升 vs FCFS |
|:--:|:--|:--:|:--:|:--:|
| 1 | **PPO** | **2746.94** | 1121.19 | **+88.3%** |
| 2 | SJF | 1468.17 | 119.08 | +0.4% |
| 3 | FCFS | 1458.77 | 55.85 | 基线 |
| 4 | Random | 1275.91 | 411.84 | -12.8% |
| 5 | Greedy | -71.87 | 619.50 | - |
| 6 | DQN | -897.08 | 289.90 | - |
| 7 | Quantum-Only | -897.08 | 289.90 | - |
| 8 | Classical-Only | -1134.35 | 64.04 | - |

### 消融实验（参考）
| 实验 | 核心结论 |
|------|---------|
| 五维消融 | D4多机+86.3% > D1算法+88.3% > D5退火+6.4% > D2状态+2.1% |
| 压力测试 | 4场景PPO综合稳定性最强；量子波动场景PPO +91.4% |
| 真机验证 | 32任务100%成功率；Mock校准后偏差<5% |
| **多seed真机** | **5 seeds × 3策略，PPO d=5.64(大效应) vs FCFS, p=6.83e-04, Bonferroni校正后显著** |

### 多seed真机实验（2026-07-22新增，N=5 per group）

> **实验配置**：5 seeds [42,123,456,789,1024] × 3策略 [PPO,FCFS,SJF] × 1真机任务/run = 15次运行
> **真机平台**：天衍-287（实际回退至 tianyan176），96步/episode，泊松到达λ=0.5
> **统计方法**：Cohen's d + 95% CI（效应量决策范式），Bonferroni校正α=0.0167

| 策略 | N | 均值 | 标准差 | min | max |
|:--:|:--:|:--:|:--:|:--:|:--:|
| **PPO** | 5 | **1665.22** | 324.51 | 1224.13 | 2097.05 |
| SJF | 5 | 567.20 | 206.33 | 383.93 | 854.43 |
| FCFS | 5 | 353.22 | 53.33 | 288.77 | 410.23 |

| 比较 | Cohen's d | 效应等级 | 95% CI | p值 | Bonferroni | 判定 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| PPO vs FCFS | 5.64 | 大效应 | [911.78, 1712.22] | 6.83e-04 | 显著 | **支持** |
| PPO vs SJF | 4.04 | 大效应 | [688.67, 1507.37] | 4.25e-04 | 显著 | **支持** |
| SJF vs FCFS | 1.42 | 大效应 | [-38.82, 466.78] | 0.080 | 不显著 | 不支持 |

详见 `results/reports/multiseed_real_machine_report.md`。

详见 `results/reports/` 目录（共10份报告，含统计显著性检验报告和多seed真机实验报告）。

## 7. 比赛材料

| 材料 | 路径 | 状态 |
|------|------|------|
| 答辩PPT（17页） | `../答辩PPT_量子RL调度系统_v5.pptx` | ✅ v5已完成（+88.3%，p=4.92e-55，N=250，284真机调用，新增2页应用价值） |
| 技术白皮书（11章） | `../技术白皮书_量子RL调度系统_v5.docx` | ✅ v5已完成（+88.3%，284真机调用，100%成功率，新增第10章落地与价值量化） |
| 价值量化报告 | `docs/value_quantification.md` | ✅ 已完成（6节，10项指标，ROI分析，VQE场景案例） |
| 演示视频分镜脚本 | `演示视频分镜脚本.md` | 已完成 |
| 演示视频（5分钟） | — | 待录制 |
| 统计显著性报告 | `results/reports/statistical_validation.md` | ✅ 已完成 |

## 8. 当前进度

```
v1 技术提升   ████████████████████ 100%（ruff 142→0 + mypy 26→0 + CI全严格阻断 + 覆盖率70%）
Track A       ████████████████████ 100%
Track B       ████████████████████ 100%（PPT/白皮书/视频脚本/实验数据）
P0 可信度修复  ████████████████████ 100%（依赖/统计/数字锁定 2026-07-09）
Track C       ████████████████████ 100%（mypy 26→0 + 覆盖率 60%→70% + ruff 142→0）
真机闭环       ████████████████████ 100%（天衍-287套餐已开通，30个真机任务全部成功，PPO保真度0.9924）
提交校验       ███████████████████░  90%（13项中8通过,4缺失待8/15冻结,1警告待PDF转换）
```

## 9. 下一步

- ~~P1：mypy 豁免 6→2~~ ✅ 已完成（2026-07-20，26个错误全部修复，CI mypy 严格阻断）
- ~~P1：清理 142 个 ruff check 历史遗留 errors~~ ✅ 已完成（2026-07-20，142→0，CI 移除 --exit-zero）
- ~~P1：更新PPT/白皮书中的实验数字为+88.3%~~ ✅ 已完成（2026-07-20，15个md文件105处替换，.pptx/.docx 待瑞哥手动更新）
- ~~P2：测试覆盖率提升~~ ✅ 已完成（2026-07-20，66个新测试，覆盖率门槛60%→70%）
- ~~P3：Docker 一键复现~~ ✅ 已完成（2026-07-20，#163 关闭）
- **P2**：演示视频录制（4-5分钟，1080p）— 需瑞哥人工录制
- **P2**：PPT/白皮书 .pptx/.docx 源文件数字更新 — 需瑞哥手动更新
- **P3**：8/15代码冻结，9/15前打v8.0-submission标签
  - 冻结前检查清单:
    1. 所有 CI 检查全绿（lint/test/typecheck/security）
    2. `python scripts/ci/validate_submission.py --check` 通过
    3. PPT/白皮书数字与代码权威数字一致（+88.3%）
    4. 演示视频已就位
    5. 打标签: `git tag -a v8.0-submission -m "v8.0 提交版本" && git push origin v8.0-submission`
    6. 打包: `python scripts/ci/validate_submission.py --pack`

详见 workspace 根目录 `项目状态审查与下一步工作建议_2026-07-09.md`。

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
pytest tests/ --cov=src --cov-fail-under=60  # 测试 + 覆盖率
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

| 用途 | 路径 |
|------|------|
| 权威PPO模型（14维） | `deliverable_models/ppo_best_model_14dim.zip` |
| 权威DQN模型（10维） | `deliverable_models/dqn_best_model_10dim.zip` |
| 归档模型目录 | `deliverable_models/`（已入库，详见 MODELS.md） |
| 多seed评估数据 | `results/multiseed_evaluation/rewards_multiseed.json` |
| 统计显著性报告 | `results/reports/statistical_validation.md` |
| 策略对比报告 | `results/reports/strategy_comparison.md` |
| 提交清单 | `config/submission_manifest.yaml` |
| Obs10Wrapper | `scripts/evaluation/run_issue_38_67_experiments.py` |
