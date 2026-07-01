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

**量化目标：** 资源利用率提升 ≥30%，平均等待时间降低 ≥40%

## 项目状态（v5）

| 指标 | 数值 |
|------|------|
| 核心代码量 | ~9,500 行 Python |
| 单元测试 | 67 用例全部通过 |
| 真机验证 | 17 个量子任务成功提交天衍云 |
| PPO vs FCFS | 提升 92.5% |
| 多机器 PPO | 奖励 4,294（vs 单机 2,305，提升 +86.3%） |

## 项目架构

```
quantum-rl-scheduler/
├── src/                      # 源代码
│   ├── scheduler/            # RL调度引擎（env.py + agent.py + parser.py）
│   ├── api/                  # 天衍云API封装（Mock/真实/cqlib 三模式）
│   ├── quantum/              # 量子退火加速模块
│   ├── visualization/        # FastAPI + Vue3 + Echarts 监控面板
│   └── utils/                # 工具函数
├── tests/                    # 67 个单元测试用例
├── scripts/                  # 训练/仿真/对比脚本（21个）
├── docs/                     # 团队文档（上手指南、Git规范、分工）
├── config/                   # 系统配置
├── .github/workflows/        # CI/CD 流水线
├── .devcontainer/            # VS Code 开发容器
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

### 验证环境

```bash
# Mock API 测试
python scripts/test_mock_api.py

# 快速训练（5000步）
python scripts/quick_train.py

# 8种策略对比仿真（200任务）
python scripts/run_simulation.py

# 启动 Web 监控界面
python -m uvicorn src.visualization.app:app --reload --port 8000
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

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 语言 | Python 3.10+ | 全部开发 |
| RL框架 | Stable-Baselines3 | PPO + Dueling DQN |
| RL环境 | Gymnasium | 标准化调度环境 |
| 深度学习 | PyTorch 2.0+ | 神经网络 |
| 量子仿真 | Qiskit / PennyLane | 量子电路仿真 |
| 量子退火 | D-Wave dimod / neal | QUBO求解 |
| Web后端 | FastAPI + Uvicorn | 监控API |
| Web前端 | Vue3 + Echarts | 监控面板 |
| 数据库 | SQLite / Redis | 任务持久化 |

## 团队基础设施

| 工具 | 用途 |
|------|------|
| `pyproject.toml` | Black + isort + flake8 + mypy + pytest 统一配置 |
| `.pre-commit-config.yaml` | Git commit 前自动格式检查 |
| GitHub Actions CI | 自动 lint + test（3.10/3.11/3.12）+ mypy |
| VS Code Dev Container | 一键开发环境（Docker + 12+ 扩展） |
| `setup.sh` / `setup.ps1` | 跨平台一键环境初始化 |

## 核心功能

| 模块 | 功能 | 状态 |
|------|------|------|
| 任务解析器 | 解析QASM量子任务，资源预估 | ✅ 已验证 |
| RL智能体 | PPO + DQN 双算法调度决策 | ✅ 已验证 |
| 调度环境 | Gymnasium 10维状态/3类动作 | ✅ 已验证 |
| 天衍API | Mock/真实/cqlib 三模式自动切换 | ✅ 已验证 |
| 量子退火 | QUBO映射 + 退火求解 | ✅ 已验证 |
| 多机器调度 | 3台真机智能路由（v5） | ✅ 已验证 |
| 真机验证 | 17个量子任务成功提交天衍云 | ✅ 已完成 |
| Web可视化 | FastAPI + Vue3 + Echarts | ✅ 已验证 |
| CI/CD | GitHub Actions 自动化测试 | ✅ 已配置 |
| Docker部署 | 一键容器化部署 | ✅ 已配置 |

## 开发计划

| 里程碑 | 截止日期 | 内容 |
|--------|----------|------|
| M1 | 7/10 | 环境搭建与基础模块 |
| M2 | 7/25 | 核心算法开发 |
| M3 | 8/5 | 系统集成与可视化 |
| M4 | 8/25 | 测试与真机验证 |
| M5 | 9/15 | 文档完善与参赛提交 |

详见 [docs/开发计划.md](docs/开发计划.md)

## 文档索引

| 文档 | 说明 |
|------|------|
| [新人上手指南](docs/新人上手指南.md) | 详细 onboarding（11步 + FAQ） |
| [队友协同开发指南](docs/队友协同开发指南.md) | 精简版快速上手（15分钟） |
| [Git工作流](docs/Git工作流.md) | 分支策略 + Commit规范 + PR流程 |
| [团队分工](docs/团队分工.md) | 10人角色职责分配 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献指南 + 代码规范 |

## 许可证

MIT License © 2026 胡展瑞
