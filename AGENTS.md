# AGENTS.md — 量子RL调度系统项目通用记忆

> 此文件供所有 AI Agent（CodeBuddy / TRAE / Claude / Cursor 等）读取，以快速理解项目全貌。
> 每次重要变更后请更新本文档的"最后更新"日期和对应章节。

**最后更新**：2026-06-27  


## 1. 项目概述

**作品名称**：量子RL驱动的天衍云平台智能调度系统  
**所属比赛**：2026年"揭榜挂帅"擂台赛 — 榜题"量子AI双向赋能的研究与应用探索"  
**主办方**：共青团中央 / 中国电信发榜 / 中电信量子执行  
**团队人数**：10人（含负责人）  
**负责人**：瑞哥（GitHub: xiabai2004）  

**核心创新—双向赋能**：
- AI 赋能 量子计算：用强化学习（RL）智能调度量子/经典任务
- 量子 赋能 AI：用量子退火（QUBO映射）加速 RL 决策
- 量化目标：资源利用率提升 ≥30%，平均等待时间降低 ≥40%

**目标平台**：天衍云平台真机"天衍-287"（祖冲之三号同款超导量子计算机）


## 2. 关键时间节点

| 日期 | 事项 | 状态 |
|------|------|------|
| 2026-06-30 | 报名截止 | ⚠️ 即将到期 |
| 2026-09-15 | 作品提交截止 | 📅 |
| 2026-09-30 | 初审结果公布 | 📅 |
| 2026-11 | 终审擂台赛 | 📅 |


## 3. 系统架构

```
用户界面 (FastAPI Web 监控, port 8000)
    │
    ├── 调度引擎 (src/scheduler/)
    │   ├── parser.py   — 任务解析器（QASM → 标准化Task）
    │   ├── env.py      — Gymnasium 调度环境（状态/动作/奖励）
    │   └── agent.py    — DQN 智能体（Dueling 架构）
    │
    ├── 量子加速模块 (src/quantum/)
    │   └── annealing.py — 量子退火求解器（QUBO映射 + 求解）
    │
    └── API 客户端 (src/api/)
        ├── tianyan_client.py — 天衍云真实 API（Mock 可切换）
        └── mock_client.py    — Mock 客户端（开发阶段使用）
```


## 4. 项目代码结构

```
quantum-rl-scheduler/
├── AGENTS.md                     # 本文档—通用项目记忆
├── README.md                     # 项目介绍 + 快速开始
├── requirements.txt              # Python 依赖清单
├── .env.example                  # 环境变量模板
├── .gitignore
├── CONTRIBUTING.md               # 贡献指南
│
├── config/
│   └── config.yaml               # 系统配置（mock_mode: true 表示 Mock 模式）
│
├── src/
│   ├── scheduler/                # 调度引擎（核心模块）
│   │   ├── __init__.py           # SchedulerAgent, QuantumSchedulingEnv, Task 等
│   │   ├── parser.py             # 量化任务解析（867行）
│   │   ├── env.py                # Gymnasium 调度环境（845行）
│   │   └── agent.py              # Dueling DQN 智能体（669行）
│   │
│   ├── api/
│   │   ├── __init__.py           # 工厂函数 create_tianyan_client()
│   │   ├── tianyan_client.py     # 天衍云 API 客户端（Mock/真实 自动切换）
│   │   └── mock_client.py        # Mock API 客户端（831行）
│   │
│   ├── quantum/
│   │   ├── __init__.py
│   │   └── annealing.py          # 量子退火优化器（766行）
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── app.py               # FastAPI Web 监控界面（1145行）
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── scripts/
│   ├── quick_train.py            # 快速训练验证（5000步）★ 新增
│   ├── train_agent.py            # 完整训练脚本（401行）
│   ├── run_simulation.py         # 仿真对比脚本（641行）
│   ├── test_mock_api.py          # Mock API 测试
│   └── install-hooks.sh          # Git Hooks 安装脚本
│
├── tests/                        # 单元测试（待补充）
│   └── test_scheduler.py
│
├── docs/
│   ├── 新人上手指南.md            # 团队 onboarding
│   ├── Git工作流.md               # 分支管理规范
│   ├── 团队分工.md                # 角色职责
│   └── 开发计划.md               # 详细时间线
│
└── .github/
    ├── PULL_REQUEST_TEMPLATE.md
    └── ISSUE_TEMPLATE/
```

**总核心代码量**：约 5,500 行 Python（不含测试和文档）


## 5. 技术栈

| 层级 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 语言 | Python | 3.12+ (TRAE 用 3.12.9) | 全部 |
| RL | Stable-Baselines3 | 2.9.0 | DQN 算法 |
| RL | Gymnasium | 1.3.0 | 环境封装 |
| DL | PyTorch | 2.12.0 | 神经网络 |
| 量子 | Qiskit / PennyLane | ≥1.0 | 量子电路仿真 |
| Web | FastAPI + Uvicorn | ≥0.104 | 监控界面 |
| 前端 | Vue3 + Echarts | — | 监控面板 |
| 配置 | PyYAML + python-dotenv | ≥6.0 | 配置管理 |


## 6. 开发模式

### 6.1 Mock 模式（当前默认）
- **配置**：`config/config.yaml` 中 `tianyan.mock_mode: true`
- **行为**：所有天衍云 API 调用走 `MockTianyanClient`，无需真实平台
- **特性**：模拟任务提交/轮转/结果、可配置延迟和失败率
- **切换真实**：修改 `.env` 中 `TIANYAN_MOCK_MODE=false`，填写 `TIANYAN_API_KEY`

### 6.2 关键接口速查

```python
# 任务解析
from src.scheduler.parser import TaskParser, Task, TaskBuilder
parser = TaskParser()
task = parser.parse_qasm(qasm_string, priority=5)

# 调度环境
from src.scheduler.env import QuantumSchedulingEnv
env = QuantumSchedulingEnv(max_qubits=20)
obs, _ = env.reset()
obs, reward, terminated, truncated, info = env.step(action)

# RL 智能体
from src.scheduler.agent import SchedulerAgent
agent = SchedulerAgent(env)
agent.train(total_timesteps=5000, save_path="./models")

# Mock API
from src.api.tianyan_client import TianyanClient
client = TianyanClient()  # 自动选择 Mock/真实模式
task_id = client.submit_quantum_task(circuit_qasm=qasm, shots=1024)

# 量子退火
from src.quantum.annealing import QuantumAnnealingOptimizer
opt = QuantumAnnealingOptimizer()
result = opt.solve_qubo(Q_matrix)
```


## 7. Git 工作流

- **主分支**：`main`（受保护，禁止直接推送）
- **功能分支**：`feature/<模块名>` 或 `fix/<问题>`
- **Commit 格式**：`<type>: <简短描述>`
  - type: `feat` / `fix` / `docs` / `test` / `refactor` / `chore`
- **Hooks**：`commit-msg` 检查格式 + `pre-push` 拦截直接推 main
- **安装 Hooks**：`bash scripts/install-hooks.sh`
- **推送时用**：`git push --no-verify origin main`（绕过 hook，仅限紧急情况）


## 8. 当前开发进度

### 已就绪
| 模块 | 文件 | 行数 | 验证状态 |
|------|------|------|---------|
| Mock API | mock_client.py | 831 | ✅ 已测试 |
| API 客户端 | tianyan_client.py | 600+ | ✅ Mock 委托已实现 |
| 任务解析器 | parser.py | 867 | ⚠️ 待验证 |
| 调度环境 | env.py | 845 | ⚠️ 待验证 |
| RL 智能体 | agent.py | 669 | ⚠️ 待验证 |
| 量子退火 | annealing.py | 766 | ⚠️ 待验证 |
| Web 界面 | app.py | 1145 | ⚠️ 待验证 |
| 训练脚本 | train_agent.py | 401 | ⚠️ 待验证 |
| 仿真脚本 | run_simulation.py | 641 | ⚠️ 待验证 |
| 快速训练 | quick_train.py | 63 | ✅ 已创建 |

### 待紧急处理
- [ ] **发送平台申请邮件**（截止 6/30）→ 收件人 `saiyuan@chinatelecom.cn`
- [ ] 端到端训练验证（5000步 DQN）
- [ ] 仿真对比测试（RL vs 贪心 vs FIFO）


## 9. 团队信息

| GitHub 用户名 | 权限 | 分工 |
|---------------|------|------|
| xiabai2004 | Admin | 项目负责人 + 架构 |
| heka-ky | Write | 待分配 |
| zyhsga | Write | 待分配 |
| NN2914 | Write | 待分配 |
| qpqpalalzmzm112 | Write | 待分配 |
| Jackhock-1 | Write | 待分配 |
| DUMNOX | Write | 待分配 |
| K1660729 | Write | 待分配 |

仓库地址：https://github.com/xiabai2004/quantum-rl-scheduler（Private）


## 10. 重要注意事项

1. **不要改 `config/config.yaml` 的 `mock_mode: true`**，除非获得平台权限
2. **每次修改 Python 文件后记得更新 `requirements.txt`**（如有新依赖）
3. **所有路径使用相对于项目根目录的相对路径**
4. **TRAE 的 Python 环境**：`D:\tools\Python 3.12.9\python.exe`
5. **运行命令始终在项目根目录**：`C:\Users\HZR\Desktop\揭榜挂帅擂台赛\quantum-rl-scheduler`
6. **GitHub 仓库为 Private**，比赛结束前不公开
7. **不要删除 docs/ 下的任何指南文件**


## 11. 快速命令参考

```bash
# 验证 Mock API
python scripts/test_mock_api.py

# 快速训练验证
python scripts/quick_train.py

# 完整训练
python scripts/train_agent.py --config config/config.yaml

# 仿真对比
python scripts/run_simulation.py --mock-mode --num-tasks 50

# 启动 Web 界面
uvicorn src.visualization.app:app --reload --port 8000

# 推送到 GitHub（注意 hook 拦截）
git push --no-verify origin main
```
