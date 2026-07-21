# 量子RL驱动的天衍云平台智能调度系统

> 2026年度"揭榜挂帅"擂台赛参赛项目
> 选题编号：XA-202609 | 发榜单位：中国电信集团有限公司

[![CI](https://github.com/xiabai2004/quantum-rl-scheduler/actions/workflows/ci.yml/badge.svg)](https://github.com/xiabai2004/quantum-rl-scheduler/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目简介

本项目面向"量子+AI双向赋能"核心命题，构建基于强化学习（RL）的天衍云平台智能调度系统。

**双向赋能机制：**
- AI赋能量子：RL Agent 实时决策任务在量子/经典资源间的最优分流
- 量子赋能AI：利用量子退火算法加速 RL 策略搜索过程

**量化目标：** 资源利用率提升 ≥30%

## 项目状态（v8.0）

| 指标 | 数值 |
|------|------|
| 核心代码量 | 约 1.1 万行 Python（src/ 57 文件） |
| 测试文件 | 42 个文件，500+ 测试用例 |
| CI 强制覆盖率 | 60%（目标 80%） |
| 真机验证 | 32 个量子任务成功提交天衍云 3 台超导量子计算机 |
| PPO vs FCFS | 综合奖励提升 88.3%（14维模型，N=50，p=3.5e-8，Cohen's d=4.09） |
| 多机器 MAPPO | 奖励 4,294（vs 单机 2,305，提升 +86.3%） |
| 消融实验 | 五维度全量完成（D1-D5） |
| 压力测试 | 4 种极限场景 PPO 综合稳定性最强 |
| 工程韧性 | 熔断器 + 8类异常体系 + Prometheus 可观测性 |
| 代码质量 | ruff(10类规则) + mypy(8项收紧) + bandit 安全扫描 |
| 比赛材料 | PPT 15页 + 白皮书 10章 + 视频分镜脚本 6段 |

## 项目架构

```
quantum-rl-scheduler/
├── src/                      # 源代码（22 文件）
│   ├── exceptions.py         # 统一异常体系（8 类）
│   ├── scheduler/            # RL调度引擎（env + agent + parser + marl + multi_objective_env）
│   ├── api/                  # 天衍云API封装（Mock/真实/cqlib 三模式 + 熔断器）
│   ├── quantum/              # 量子退火加速模块（QUBO + 异步闭环）
│   ├── visualization/        # FastAPI + Vue3 + Echarts 监控面板
│   └── utils/                # 工具函数 + Prometheus 指标
├── tests/                    # 14 个测试文件，100+ 用例
│   └── benchmarks/           # 性能基准测试
├── scripts/                  # 按功能分区（training/evaluation/demo/testing/benchmarking/reporting）
│   └── cli.py                # Click 统一命令行入口
├── docs/                     # 团队文档（上手指南、Git规范、分工、协同开发）
├── config/                   # 系统配置（config.yaml + .env.example）
├── results/reports/          # 实验数据固化报告（9份）
├── .github/workflows/        # CI/CD 4 Job 流水线 + PR 自动化
├── .devcontainer/            # VS Code 开发容器
├── pyproject.toml            # 统一配置（Black/ruff/bandit/mypy/pytest/coverage/mutmut）
├── mypy.ini                  # 类型检查（8项严格配置）
├── .pre-commit-config.yaml   # Git commit 自动检查
└── Dockerfile + compose      # 一键 Docker 部署
```

## 快速开始

### 方式一：一键初始化（推荐）

```bash
git clone https://github.com/xiabai2004/quantum-rl-scheduler.git
cd quantum-rl-scheduler

# Linux / macOS / Git Bash
bash setup.sh

# Windows PowerShell
powershell .\setup.ps1
```

### 方式二：手动安装

```bash
git clone https://github.com/xiabai2004/quantum-rl-scheduler.git
cd quantum-rl-scheduler

python -m venv .venv
source .venv/Scripts/activate   # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
cp .env.example .env            # Mock 模式默认开启
```

### 方式三：VS Code Dev Container

安装 [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) 扩展后，打开项目文件夹，点击右下角 "Reopen in Container"。

### 方式四：Docker 一键复现（Docker）

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
| Issue | 标题 | 是否需要真机 | 说明 |
|---|---|---|---|
| #142 | 制作系统演示视频 | 否 | 视频制作 |
| #143 | 技术白皮书 v3 终稿 | 否 | 文档 |
| #144 | 答辩 PPT 终稿 | 否 | 文档 |
| #145 | 清理 v5 克隆 | 否 | 仓库卫生 |
| #147 | 数字一致性复核 | 否 | 文档核对 |
| #148 | 突破 head_only 退火限制 | 验证环节需要 | 默认 `simulation_mode=True` 用本地 D-Wave neal 求解器，开发全程可纯仿真；真机仅作"全量/分层 QUBO 上硬件验证"的可选项 |
| #149 | 补齐 env_real_machine 测试 | 测试用 Mock 即可 | 模块本身即真机集成；按 `docs/真机训练接入指南.md` 用 Mock 降级，不强制连真机 |
| #150 | 提升 marl（MAPPO）覆盖率 | 否 | `marl.py` 为纯仿真模块，不依赖真机 |
| #151 | 提升 tianyan_cqlib 覆盖率 | 测试用 Mock 即可 | 真机客户端封装，测试用 Mock 模拟响应即可省机时 |
| #152 | 提升 env/ppo_agent 覆盖率 | 否 | 默认 `use_real_machine=False` |

> 真机"上手前置"指引仅对 **#148 / #149 / #151** 三个 Issue 适用；其余任务请纯本地仿真完成。

### 3. 真机机时珍贵，省着用
- 免费机时包仅 1-qubit 电路稳定（默认只用单比特门）。
- 真机只留给 **#148 / #149 / #151 的验证环节**；开发期一律用 Mock。
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
| 量子真机 | 天衍云 cqlib SDK | 287量子比特超导处理器 |
| 量子退火 | D-Wave dimod / neal | QUBO求解 |
| Web后端 | FastAPI + Uvicorn | 监控API |
| Web前端 | Vue3 + Echarts | 监控面板 |
| CLI | Click | 统一命令行入口 |

## 团队基础设施

| 工具 | 用途 |
|------|------|
| `pyproject.toml` | Black + ruff + bandit + mypy + pytest + coverage + mutmut 统一配置 |
| `mypy.ini` | 8项严格类型检查（仅2模块暂时豁免：annealing/scripts） |
| `.pre-commit-config.yaml` | Git commit 前自动格式检查 + Commit 格式校验 |
| GitHub Actions CI | lint(ruff+bandit) + test(3.10/3.11/3.12矩阵) + typecheck(mypy) + benchmarks |
| Dependabot | pip + GitHub Actions 自动依赖更新 |
| VS Code Dev Container | 一键开发环境（Docker + 12+ 扩展） |
| `setup.sh` / `setup.ps1` | 跨平台一键环境初始化 |

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
| 量子退火 | QUBO映射 + 退火求解 + 异步闭环 | 已验证 |
| 多机器调度 | 3台真机MAPPO协同，奖励+86.3% | 已验证 |
| 真机验证 | 32个量子任务成功提交，Mock偏差<5% | 已完成 |
| Web可视化 | FastAPI + Vue3 + Echarts + WebSocket | 已验证 |
| 可观测性 | Prometheus /metrics 端点 | 已验证 |
| CI/CD | 4 Job流水线 + Codecov + Dependabot | 已配置 |
| Docker部署 | 一键容器化部署 | 已配置 |

## 实验成果

| 实验 | 核心结论 |
|------|---------|
| 8策略对比 | PPO奖励2747 vs FCFS 1457，+88.3%（14维，p=3.5e-8，d=4.09） |
| 五维消融 | D1算法+88.3% > D4多机+86.3% > D5退火+6.4% > D2状态+2.1% |
| 压力测试 | 4场景PPO综合稳定性最强；量子波动场景PPO +91.4% |
| 真机验证 | 32任务100%成功率；Mock校准后偏差<5% |

详见 `results/reports/` 目录。

## 比赛材料

| 材料 | 文件 |
|------|------|
| 答辩PPT（15页） | `../答辩PPT_量子RL调度系统.pptx` |
| 技术白皮书（10章） | `../技术白皮书_量子RL调度系统_v2.docx` |
| 演示视频分镜脚本 | `演示视频分镜脚本.md` |
| 答辩PPT大纲 | `答辩PPT大纲.md` |
| 白皮书更新计划 | `技术白皮书_更新计划.md` |
| B1 实验数据报告 | `results/reports/` 下 4 份报告 |

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

## 许可证

MIT License © 2026 胡展瑞
