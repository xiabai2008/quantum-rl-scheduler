# AGENTS.md — 量子RL调度系统项目通用记忆

> 此文件供所有 AI Agent（CodeBuddy / TRAE / Claude / Cursor 等）读取，以快速理解项目全貌。
> 每次重要变更后请更新本文档的"最后更新"日期和对应章节。

**最后更新**：2026-06-27  

---

# 🚨🚨🚨 **HOOK 警告 — 必读！** 🚨🚨🚨

## ⛔ Git Hook 会拦截直接推送到 main 分支！

此项目安装了 `pre-push` hook，**直接执行 `git push origin main` 会被拦截并报错！**

### ✅ 正确的推送方式

| 场景 | 命令 |
|------|------|
| **紧急/直接推送** | `git push --no-verify origin main` |
| **正常流程** | 创建功能分支 → 推送功能分支 → 创建 PR → Review 后合并 |

### 📝 Commit 格式要求

`commit-msg` hook 会检查 commit 信息格式：

```
<type>: <简短描述>
```

**合法 type**：`feat` / `fix` / `docs` / `test` / `refactor` / `chore` / `perf` / `style` / `ci` / `build`

| 错误示例 | 正确示例 |
|---------|---------|
| `remove: xxx` ❌ | `chore: 移除某文件` ✅ |
| `update: xxx` ❌ | `feat: 更新某功能` ✅ |
| `添加新功能` ❌ | `feat: 添加新功能` ✅ |

---

**如果你看到 `⛔ 禁止直接推送到 main 分支` 错误，不代表你不能推送！**
**请使用 `git push --no-verify origin main` 或创建功能分支走 PR 流程。**

---

## ⚠️ 开始工作前必读

### Git 推送会被 Hook 拦截！

此项目在 `main` 分支安装了 `pre-push` hook，**直接 `git push origin main` 会被拦截**。

**推送代码到 main 分支的正确命令**：
```bash
git push --no-verify origin main
```

**Commit 信息格式要求**（`commit-msg` hook 检查）：
- 格式：`<type>: <简短描述>`
- type 必须是：`feat` / `fix` / `docs` / `test` / `refactor` / `chore` / `perf` / `style` / `ci` / `build`
- 错误示例：`remove: xxx` → 被拦截（`remove` 不是合法 type）
- 正确示例：`chore: 移除某文件`

如果 `git push` 报错 `⛔ 禁止直接推送到 main 分支`，**不代表你不能推送**，用 `--no-verify` 即可。

---

## 1. 项目概述

**作品名称**：量子RL驱动的天衍云平台智能调度系统  
**所属比赛**：2026年"揭榜挂帅"擂台赛 — 榜题"量子AI双向赋能的研究与应用探索"  
**主办方**：共青团中央主办 / 中国电信发榜 / 中电信量子执行  
**团队人数**：8人（含负责人）  
**负责人**：瑞哥（GitHub: xiabai2004）  

**核心创新—双向赋能**：
- AI 赋能 量子计算：用强化学习（RL）智能调度量子/经典任务
- 量子 赋能 AI：用量子退火（QUBO映射）加速 RL 决策
- 量化目标：资源利用率提升 ≥30%，平均等待时间降低 ≥40%

**目标平台**：天衍云平台真机"天衍-287"（祖冲之三号同款超导量子计算机）

**赛事奖励**：¥26.5万现金 + 价值超¥200万的真机机时  
**官方邮箱**：saiyuan@chinatelecom.cn（平台申请邮件）  
**仓库地址**：https://github.com/xiabai2004/quantum-rl-scheduler（Private）


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
├── LICENSE                       # 许可证
├── .trae/
│   └── instructions.md           # TRAE Agent 指令文件
├── githooks/
│   ├── commit-msg                # Commit 格式检查 hook
│   └── pre-push                  # 主分支保护 hook
│
├── config/
│   ├── .env.example              # 环境变量模板
│   └── config.yaml               # 系统配置（mock_mode: true 表示 Mock 模式）
│
├── src/
│   ├── __init__.py
│   ├── scheduler/                # 调度引擎（核心模块）
│   │   ├── __init__.py           # SchedulerAgent, QuantumSchedulingEnv, Task 等导出
│   │   ├── parser.py             # 量化任务解析（816行）
│   │   ├── env.py                # Gymnasium 调度环境（720行）
│   │   └── agent.py              # Dueling DQN 智能体（592行）
│   │
│   ├── api/
│   │   ├── __init__.py           # 工厂函数 get_client() / create_tianyan_client()
│   │   ├── tianyan_client.py     # 天衍云 API 客户端（574行）
│   │   └── mock_client.py        # Mock API 客户端（510行）
│   │
│   ├── quantum/
│   │   ├── __init__.py
│   │   └── annealing.py          # 量子退火优化器（682行）
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── app.py               # FastAPI Web 监控界面（1109行）
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py            # 工具函数（285行）
│
├── scripts/
│   ├── quick_train.py            # 快速训练验证（63行）★ 新增
│   ├── train_agent.py            # 完整训练脚本
│   ├── run_simulation.py         # 仿真对比脚本
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
│   ├── 开发计划.md               # 详细时间线
│   └── 项目记忆_给TRAE.md         # TRAE 专用记忆文件
│
└── .github/
    ├── PULL_REQUEST_TEMPLATE.md
    └── ISSUE_TEMPLATE/
        ├── bug.md
        └── task.md
```

**总核心代码量**：约 4,788 行 Python（不含测试和文档）


## 5. 技术栈

| 层级 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 语言 | Python | ≥3.10（TRAE 使用 3.12.9） | 全部 |
| RL | Stable-Baselines3 | ≥2.0.0 | DQN 算法 |
| RL | Gymnasium | ≥0.28.0 | 环境封装 |
| DL | PyTorch | ≥2.0.0 | 神经网络 |
| 量子 | Qiskit / PennyLane | ≥1.0 | 量子电路仿真 |
| 量子 | D-Wave Ocean SDK | 可选 | 量子退火（dimod/neal） |
| Web | FastAPI + Uvicorn | ≥0.104 | 监控界面 |
| 前端 | Vue3 + Echarts | — | 监控面板 |
| 配置 | PyYAML + python-dotenv | ≥6.0 | 配置管理 |
| 日志 | Loguru | ≥0.7.0 | 日志框架 |


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

# 调度环境（8维状态空间，Discrete(3)动作空间）
from src.scheduler.env import QuantumSchedulingEnv
env = QuantumSchedulingEnv(max_qubits=20)
obs, _ = env.reset()
obs, reward, terminated, truncated, info = env.step(action)

# RL 智能体（Dueling DQN）
from src.scheduler.agent import SchedulerAgent
agent = SchedulerAgent(env, learning_rate=1e-4, batch_size=32)
model = agent.train(total_timesteps=5000, eval_freq=500, log_dir="./logs")
agent.save("./models/model")
eval_result = agent.evaluate(num_episodes=5)

# API 客户端（自动选择 Mock/真实模式）
from src.api import get_client
client = get_client(mock_mode=True)  # 或 get_client() 自动检测
task_id = client.submit_quantum_task(circuit_qasm=qasm, shots=1024)

# 量子退火（QUBO 映射 + 求解）
from src.quantum.annealing import QuantumAnnealingOptimizer
opt = QuantumAnnealingOptimizer(simulation_mode=True)
result = opt.solve_qubo(Q_matrix)

# 工具函数
from src.utils import setup_logging, load_config, MetricsCalculator
setup_logging()
config = load_config()
metrics = MetricsCalculator()
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
| 模块 | 文件 | 实际行数 | 验证状态 | 备注 |
|------|------|---------|---------|------|
| Mock API | mock_client.py | 510 | ✅ 已测试 | |
| API 客户端 | tianyan_client.py | 574 | ✅ Mock 委托已实现 | |
| 任务解析器 | parser.py | 816 | ✅ 已验证 | TaskParser + Builder + Legacy |
| 调度环境 | env.py | 720 | ✅ 已验证 | Gymnasium 接口完整 |
| RL 智能体 | agent.py | 592 | ✅ 已验证 | Dueling DQN 架构 |
| 量子退火 | annealing.py | 682 | ✅ 已验证 | QUBO映射 + numpy仿真 |
| Web 界面 | app.py | 1109 | ✅ 已验证 | FastAPI + WebSocket |
| 快速训练 | quick_train.py | 63 | ✅ 已验证 | 端到端训练通过 |
| 工具函数 | helpers.py | 285 | ✅ 已验证 | |
| 训练脚本 | train_agent.py | — | ⚠️ 待验证 | |
| 仿真脚本 | run_simulation.py | — | ⚠️ 待验证 | |

### 待紧急处理
- [ ] **发送平台申请邮件**（截止 6/30）→ 收件人 `saiyuan@chinatelecom.cn`
- [x] 端到端训练验证（5000步 DQN）✅ 已完成
- [ ] 仿真对比测试（RL vs 贪心 vs FIFO）
- [ ] 单元测试补充


## 9. 团队信息

| GitHub 用户名 | 权限 | 分工 | 状态 |
|---------------|------|------|------|
| xiabai2004 | Admin | 项目负责人 + 架构 | ✅ 已加入 |
| heka-ky | Write | 待分配 | ✅ 已加入 |
| zyhsga | Write | 待分配 | ✅ 已加入 |
| NN2914 | Write | 待分配 | ✅ 已加入 |
| qpqpalalzmzm112 | Write | 待分配 | ✅ 已加入 |
| Jackhock-1 | Write | 待分配 | ✅ 已加入 |
| DUMNOX | Write | 待分配 | ✅ 已加入 |
| K1660729 | Write | 待分配 | ✅ 已加入 |

**团队分工建议**：
- 算法开发（2-3人）：env.py、agent.py、annealing.py、真机测试
- 后端开发（2人）：parser.py、tianyan_client.py、run_simulation.py
- 前端开发（1-2人）：app.py（Vue3 + Echarts）
- 测试与DevOps（1人）：单元测试、CI/CD、Docker
- 文档与项目管理（1人）：训练脚本、文档完善、PPT制作

仓库地址：https://github.com/xiabai2004/quantum-rl-scheduler（Private）


## 10. 重要注意事项

1. **不要改 `config/config.yaml` 的 `mock_mode: true`**，除非获得平台权限
2. **每次修改 Python 文件后记得更新 `requirements.txt`**（如有新依赖）
3. **所有路径使用相对于项目根目录的相对路径**
4. **TRAE 的 Python 环境**：`D:\\tools\\Python 3.12.9\\python.exe`
5. **运行命令始终在项目根目录**：`C:\\Users\\HZR\\Desktop\\揭榜挂帅擂台赛\\quantum-rl-scheduler`
6. **GitHub 仓库为 Private**，比赛结束前不公开
7. **不要删除 docs/ 下的任何指南文件**


## 11. 代码规范

| 规范项 | 要求 |
|--------|------|
| Python 版本 | ≥3.10（TRAE 使用 3.12.9） |
| 代码格式化 | Black（line-length=88） |
| 注释语言 | 中文 |
| 函数/方法 | 必须有文档字符串（docstrings） |
| 命名规范 | 类名 PascalCase，函数/变量 snake_case |

**开发优先级顺序**（由高到低）：
1. `src/scheduler/env.py` — RL 调度环境（Gymnasium 接口）
2. `src/scheduler/agent.py` — RL 智能体（DQN 训练）
3. `src/api/tianyan_client.py` — 天衍云 API 封装
4. `src/quantum/annealing.py` — 量子退火加速模块
5. `src/visualization/app.py` — Web 监控界面

## 12. 快速命令参考

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