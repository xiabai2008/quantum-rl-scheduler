# 量子RL驱动的天衍云平台智能调度系统

> 2026年度"揭榜挂帅"擂台赛参赛项目
> 选题编号：XA-202609 | 发榜单位：中国电信集团有限公司

[![CI](https://github.com/xiabai2008/quantum-rl-scheduler/actions/workflows/ci.yml/badge.svg)](https://github.com/xiabai2008/quantum-rl-scheduler/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目简介

本项目面向"量子+AI双向赋能"核心命题，构建基于强化学习（RL）的天衍云平台智能调度系统。

**双向赋能机制：**
- AI赋能量子：RL Agent 实时决策任务在量子/经典资源间的最优分流
- 量子赋能AI：量子启发式退火[^annealing-sim]优化作为探索性方向（当前为经典模拟退火，训练有开销，奖励提升统计不显著）

**量化目标：** 资源利用率提升 ≥30%

## 项目状态（v8.0）

| 指标 | 数值 |
|------|------|
| 核心代码量 | 约 1.1 万行 Python（src/ 64 文件） |
| 测试文件 | 69 个文件，2500+ 测试用例 |
| CI 强制覆盖率 | 80%（实际 93.58%，pyproject.toml `fail_under=80`） |
| 真机可用性验证 | 天衍-287 30/30任务成功（N=10 seeds×3策略，100%成功率） |
| PPO vs FCFS（仿真） | 综合奖励提升 88.3%（14维模型，N=250，Mann-Whitney U 检验 p=1.032e-42，rank-biserial=-0.71） |
| PPO vs FCFS（真机） | +353%（N=10, PPO=1736 vs FCFS=383, p<0.001） |
| 多机器 MAPPO | 奖励 4,294（vs 单机 2,305，提升 +86.3%） |
| 电路编译 AI | PPO SWAP=6.5 vs SABRE=27.6，减少 76.4% |
| VQE 行业场景 | 10分子×100任务，PPO +97.5% vs FCFS |
| OR-Tools 对比 | 20/50/100任务，OR-Tools静态最优，PPO动态优势 |
| 消融实验 | 五维度全量完成（D1-D5） |
| 压力测试 | 4 种极限场景 PPO 综合稳定性最佳 |
| 工程韧性 | 熔断器 + 8类异常体系 + Prometheus 可观测性 |
| 代码质量 | ruff(10类规则) + mypy(8项收紧) + bandit 安全扫描 |

## 项目架构

```mermaid
graph TB
    User[用户/评委] --> CLI[Click CLI 统一入口]
    CLI --> Train[训练模块 training.py]
    CLI --> Demo[Demo演示 demo.py]
    CLI --> Serve[Web监控面板 app.py]
    Train --> PPO[PPO / DQN / MAPPO]
    PPO --> Env[Gymnasium 14维调度环境]
    Env --> Annealing[量子启发式退火优化器 QUBO]
    Env --> Tianyan[天衍云API / 真机cqlib]
    Serve --> FastAPI[FastAPI 后端]
    FastAPI --> Vue[Vue3 + ECharts 前端]
    Tianyan --> CircuitBreaker[熔断器 + 配额追踪]
```

```
quantum-rl-scheduler/
├── src/                      # 源代码（~64 文件）
│   ├── exceptions.py         # 统一异常体系（8 类）
│   ├── scheduler/            # RL调度引擎（env + agent + parser + marl + multi_objective_env）
│   ├── api/                  # 天衍云API封装（Mock/真实/cqlib 三模式 + 熔断器）
│   ├── quantum/              # 量子启发式退火加速模块（QUBO + 异步闭环）
│   ├── visualization/        # FastAPI + Vue3 + Echarts 监控面板
│   └── utils/                # 工具函数 + Prometheus 指标
├── tests/                    # 69 个测试文件，2500+ 用例
│   └── benchmarks/           # 性能基准测试
├── scripts/                  # 按功能分区（training/evaluation/demo/testing/benchmarking/reporting）
│   └── cli.py                # Click 统一命令行入口
├── docs/                     # 团队文档（上手指南、Git规范、分工、协同开发）
├── config/                   # 系统配置（config.yaml + .env.example）
├── results/reports/          # 实验数据固化报告（23份）
├── .github/workflows/        # CI/CD 4 Job 流水线 + PR 自动化
├── .devcontainer/            # VS Code 开发容器
├── pyproject.toml            # 统一配置（ruff/mypy/bandit/pytest/coverage）
├── mypy.ini                  # 类型检查（8项严格配置）
├── .pre-commit-config.yaml   # Git commit 自动检查
└── Dockerfile + compose      # 一键 Docker 部署
```

## 快速开始

### 方式一：Demo 一键启动（评委/演示推荐）

安装依赖后，一条命令启动完整 Demo（自动检测依赖、启动服务、打开浏览器）：

```bash
python scripts/demo_one_click.py
```

脚本会自动：
- 检查 Python 版本和必需依赖（缺失时自动安装）
- 检查 PPO 模型文件
- 寻找可用端口（默认 8000）
- 启动 uvicorn 服务器
- 等待服务就绪后自动打开浏览器
- 打印访问地址和操作指引

### 方式二：一键初始化（开发）

```bash
git clone https://github.com/xiabai2008/quantum-rl-scheduler.git
cd quantum-rl-scheduler

# Linux / macOS / Git Bash
bash setup.sh

# Windows PowerShell
powershell .\setup.ps1
```

### 方式三：手动安装

```bash
git clone https://github.com/xiabai2008/quantum-rl-scheduler.git
cd quantum-rl-scheduler

python -m venv .venv
source .venv/Scripts/activate   # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
cp .env.example .env            # Mock 模式默认开启
```

### 方式四：VS Code Dev Container

安装 [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) 扩展后，打开项目文件夹，点击右下角 "Reopen in Container"。

### 方式五：Docker 一键复现（Docker）

```bash
# 一条命令，5 分钟看到对比结果
docker compose up
```

### 验证环境

```bash
# CLI 统一入口
python scripts/cli.py --help

# 快速训练（5000步）
python scripts/cli.py train --timesteps 5000

# 8种策略对比仿真（200任务）
python scripts/cli.py simulate --num-tasks 200

# 启动 Web 监控界面
python scripts/cli.py serve --port 8000

# 运行全部测试
pytest tests/ --cov=src
```

## Mock 模式（开发阶段默认）

无需真实天衍云平台权限即可完整开发。Mock 客户端模拟：
- 量子任务提交（返回虚拟 task_id）
- 任务状态自动轮转（PENDING → RUNNING → COMPLETED）
- 量子测量结果（随机计数）
- 可配置网络延迟和失败率

切换到真实 API：
```bash
# 修改 .env
TIANYAN_MOCK_MODE=false
TIANYAN_API_KEY=你的真实API密钥
```

## 队友须知

> 认领 Issue 前先读这段，避免走弯路、避免浪费共享真机机时。

### 1. 默认纯本地开发，无需真机
项目默认 `TIANYAN_MOCK_MODE=true`（见 `.env.example`），**绝大多数开发、测试、仿真都在本地 Mock 模式下完成**，不需要天衍云权限。只有极少数任务在"真机验证环节"才需连接真实硬件。

### 2. 当前开放 Issue 的真机适用范围

> **Issue 状态请直接查看 [GitHub Issues 页面](https://github.com/xiabai2008/quantum-rl-scheduler/issues)**，避免文档与仓库实际状态不同步。
>
> 历史上曾列出 #142-#152 等开放 Issue，已于 2026-07-26 全部关闭。新增 Issue 的真机适用范围请参考 Issue 标签与描述。

**真机使用原则**（适用于所有 Issue）：
- 默认纯本地仿真开发（`TIANYAN_MOCK_MODE=true`）
- 真机仅用于"全量/分层 QUBO 上硬件验证"或"真机集成测试"的可选环节
- 真机客户端封装类（`tianyan_cqlib.py`、`env_real_machine.py`）的测试可用 Mock 模拟响应，无需真机
- 退火模块（`annealing.py`）默认 `simulation_mode=True`，使用本地 D-Wave neal 求解器

### 3. 真机机时珍贵，省着用
- 免费机时包仅 1-qubit 电路稳定（默认只用单比特门）。
- 真机只留给 **退火验证 / 真机集成测试 / 客户端封装验证** 等可选环节；开发期一律用 Mock。
- 连续失败 3 次会自动降级 Mock（正常现象，非 bug）。

### 4. 需要连真机时
读 [`docs/真机训练接入指南.md`](docs/真机训练接入指南.md)：装 `requirements-quantum.txt` → 配 `.env`（API Key 向瑞哥领取）→ `CqlibTianyanClient.authenticate()` 验证连接 → `python scripts/training/train_agent_real.py --timesteps 5000 --real-prob 0.05`。
**API Key 不外泄、`.env` 不入库。**

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 语言 | Python 3.10+ | 全部开发 |
| RL框架 | Stable-Baselines3 | PPO + DQN + MAPPO |
| RL环境 | Gymnasium | 标准化调度环境 |
| 深度学习 | PyTorch 2.0+ | 神经网络 |
| 量子仿真 | Qiskit / PennyLane | 量子电路仿真 |
| 量子真机 | 天衍云 cqlib SDK | 105数据比特+182耦合比特超导处理器 |
| 量子启发式退火 | D-Wave dimod / neal | QUBO求解 |
| Web后端 | FastAPI + Uvicorn | 监控API |
| Web前端 | Vue3 + Echarts | 监控面板 |
| CLI | Click | 统一命令行入口 |

## 团队基础设施

| 工具 | 用途 |
|------|------|
| `pyproject.toml` | Black + ruff + bandit + mypy + pytest + coverage + mutmut 统一配置 |
| `mypy.ini` | 8项严格类型检查（仅2模块暂时豁免：annealing/scripts） |
| `.pre-commit-config.yaml` | Git commit 前自动格式检查 + Commit 格式校验 |
| GitHub Actions CI | lint(ruff+bandit) + test(3.10/3.11/3.12矩阵) + typecheck(mypy) + benchmarks + QUBO形式化验证 + 覆盖率artifact |
| Dependabot | pip + GitHub Actions 自动依赖更新 |
| VS Code Dev Container | 一键开发环境（Docker + 12+ 扩展） |
| `setup.sh` / `setup.ps1` | 跨平台一键环境初始化 |

> **CI 覆盖率 artifact（Issue #260）**：每次 CI 运行都会生成 `coverage.xml`，
> 并作为名为 `coverage-xml` 的 artifact 上传，可在 Actions 页面下载查看本次构建的覆盖率快照；
> 在 PR 中还会由 `ci(#253,#260)` 自动发布一条覆盖率评论（当前展示本次构建的总行覆盖率）。
> **QUBO 形式化验证（Issue #253）**：CI 会运行 `tests/test_qubo_optimization.py` 中
> `-k "formal or property"` 选中的 QUBO 形式化/属性测试（基于 numpy 随机种子的属性测试），
> 验证 QUBO 矩阵对称性、能量公式正确性等数学性质。

## 工程韧性

| 组件 | 功能 |
|------|------|
| `src/exceptions.py` | 8类统一异常（code + retryable 语义） |
| `src/api/circuit_breaker.py` | 熔断器（CLOSED/OPEN/HALF_OPEN 三态） |
| `src/utils/metrics.py` | 7个 Prometheus 指标（Gauge/Counter/Histogram） |
| `scripts/cli.py` | Click 统一入口（train/simulate/serve/demo） |

## 核心功能

| 模块 | 功能 | 状态 |
|------|------|------|
| 任务解析器 | 解析QASM量子任务，资源预估 | 已验证 |
| RL智能体 | PPO（主力）+ DQN（备选）+ MAPPO（多智能体） | 已验证 |
| 调度环境 | 14维状态空间 / 3类动作 / 异质化任务 | 已验证 |
| 天衍API | Mock / REST / cqlib 三模式 + 多机器协调器 | 已验证 |
| 量子启发式退火 | QUBO映射 + 退火求解 + 异步闭环 | 已验证 |
| 多机器调度 | 3台机器MAPPO协同，奖励+86.3%（仿真验证） | 已验证 |
| 真机可用性验证 | 284次真机调用100%成功，SDK全链路验证 | 已完成 |
| Web可视化 | FastAPI + Vue3 + Echarts + WebSocket | 已验证 |
| 可观测性 | Prometheus /metrics 端点 | 已验证 |
| CI/CD | 4 Job流水线 + Codecov + Dependabot | 已配置 |
| Docker部署 | 一键容器化部署 | 已配置 |

## 实验成果

| 实验 | 核心结论 |
|------|---------|
| 8策略对比 | PPO奖励2746.94 vs FCFS 1458.77，+88.3%（14维，N=250，Mann-Whitney U 检验 p=1.032e-42，r=-0.71） |
| 真机验证 | N=10 seeds, PPO=1736 vs FCFS=383 (+353%), 30/30成功 |
| 五维消融 | D1算法+88.3% > D4多机+86.3% > D5退火+6.4% > D2状态+2.1% |
| 电路编译 | PPO SWAP=6.5 vs SABRE=27.6, -76.4% (60电路) |
| VQE行业 | 10分子×100任务, PPO +97.5% vs FCFS |
| OR-Tools | CP-SAT静态最优, PPO动态实时优势 |
| 压力测试 | 4场景PPO综合稳定性最强；量子波动场景PPO +91.4% |
| 多租户公平调度 | 5租户Jain's公平指数=0.9875，PPO总奖励+57.6% vs FCFS |
| D3奖励消融 | 7权重预设×2策略×10seeds，揭示策略-奖励耦合关系 |

详见 `results/reports/` 目录。

## 比赛材料

> 以下材料位于仓库上级目录（比赛提交材料，非代码仓库的一部分）：

| 材料 | 文件 | 版本 |
|------|------|------|
| 答辩PPT（17页） | `../答辩PPT_量子RL调度系统_v5.pptx` | v5 |
| 技术白皮书（11章） | `../技术白皮书_量子RL调度系统_v5.docx` | v5 |
| 演示视频分镜脚本 | `演示视频分镜脚本.md` | — |
| 答辩PPT大纲 | `答辩PPT大纲.md` | — |
| 白皮书更新计划 | `技术白皮书_更新计划.md` | — |
| B1 实验数据报告 | `results/reports/` 下 23 份报告 | — |

## 最终提交包说明

比赛最终提交物清单定义在 `config/submission_manifest.yaml`，使用校验工具管理：

```bash
# 准备提交物（创建 dist/ 目录 + 生成缺失项报告 + 输出检查清单）
python scripts/ci/validate_submission.py --prepare

# 校验所有提交物是否符合要求
python scripts/ci/validate_submission.py --check

# 生成缺失项清单报告
python scripts/ci/validate_submission.py --check --report results/reports/submission_validation_report.md

# 打包（校验通过后生成 dist/submission_v8.0_YYYYMMDD.zip）
python scripts/ci/validate_submission.py --pack
```

### 提交物清单（13 项）

| 编号 | 名称 | 类型 | 状态 |
|:--:|:--|:--:|:--:|
| CODE_REPO | 代码仓库（Git 标签 v8.0-submission） | git_tag | 8/15 冻结后创建 |
| CODE_ARCHIVE | 代码压缩包 | zip | 冻结后 --pack 生成 |
| WHITEPAPER | 技术白皮书（20-50页 PDF） | pdf | docx→PDF 转换 |
| PRESENTATION | 答辩 PPT（15-20页） | pptx | 人工制作 |
| DEMO_VIDEO | 演示视频（4-5分钟 1080p） | mp4 | 人工录制 |
| EXP_STRATEGY | 策略对比报告 | md | ✅ 已完成 |
| EXP_ABLATION | 消融实验报告 | md | ✅ 已完成 |
| EXP_STRESS | 压力测试报告 | md | ✅ 已完成 |
| EXP_REAL | 真机验证报告 | md | ✅ 已完成 |
| EXP_STAT | 统计显著性报告 | md | ✅ 已完成 |
| MODEL_PPO | PPO 权威模型 | zip | ✅ 已完成 |
| MODEL_DQN | DQN 权威模型 | zip | ✅ 已完成 |
| REQUIREMENTS_MATRIX | 需求追溯矩阵 | md | ✅ 已完成 |

### 代码冻结流程（8/15）

1. 确认所有 CI 检查全绿
2. 运行 `python scripts/ci/pre_freeze_check.sh` 执行冻结前检查
3. 运行 `python scripts/ci/validate_submission.py --check` 确认通过
4. 创建标签：`git tag -a v8.0-submission -m "v8.0 提交版本" && git push origin v8.0-submission`
5. 打包：`python scripts/ci/validate_submission.py --pack`
6. 提交压缩包至比赛平台

## 开发计划

| 里程碑 | 截止日期 | 内容 |
|--------|----------|------|
| Track A 工程收尾 | 7/1 已完成 | pre-commit + scripts/ 重组 |
| Track B 比赛材料 | 7/1 已完成 | PPT + 白皮书 + 视频脚本 + 实验数据 |
| Track C 质量深化 | 7-8月 | mypy豁免清理 + 覆盖率80% + mutation testing |
| PPO 真机闭环 | 7-8月 | cqlib 注入调度循环 |
| M5 参赛提交 | 9/15 | 最终材料提交 |

## 文档索引

| 文档 | 说明 |
|------|------|
| [新人上手指南](docs/新人上手指南.md) | 详细 onboarding（11步 + FAQ） |
| [队友协同开发指南](docs/队友协同开发指南.md) | 精简版快速上手（15分钟） |
| [真机训练接入指南](docs/真机训练接入指南.md) | 连接天衍云真机并进入训练（装 cqlib → 配 .env → 验证连接 → 跑训练） |
| [Git工作流](docs/Git工作流.md) | 分支策略 + Commit规范 + PR流程 |
| [团队分工](docs/团队分工.md) | 10人角色职责分配 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献指南 + 代码规范 |
| [AGENTS.md](AGENTS.md) | AI Agent 通用项目记忆 |
| [技术瓶颈分析](docs/technical_bottlenecks.md) | 7项技术瓶颈 + 缓解策略 |
| [退火显著性答辩策略](docs/annealing_significance-defense.md) | p=0.19应对话术 + 5类评委问题 |
| [部署架构](docs/deployment.md) | 三阶段部署路径（原型→试点→生产） |
| [跨硬件兼容性](docs/cross_hardware.md) | 三层解耦架构 + 扩展路径 |
| [价值量化报告](docs/value_quantification.md) | 10项指标 + ROI分析 + VQE场景案例 |
| [公平调度报告](results/reports/fair_scheduling_report.md) | 5租户Jain's指数=0.9875 |
| [D3奖励消融报告](results/reports/d3_reward_ablation_report.md) | 7预设×2策略×10seeds消融 |

## 许可证

MIT License © 2026 胡展瑞

---

[^annealing-sim]: 当前版本使用经典模拟退火（neal库）求解QUBO问题，真机量子退火为后续工作。详见 [技术瓶颈分析](docs/technical_bottlenecks.md) 瓶颈1。
