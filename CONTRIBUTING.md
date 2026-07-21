# 贡献指南

> 欢迎加入 **量子RL驱动的天衍云平台智能调度系统** 开发团队！本指南帮助你快速了解协作规范。
>
> 完整规范请参见：
> - [docs/Git工作流.md](docs/Git工作流.md) — 分支策略、Commit、PR、冲突解决完整流程
> - [docs/队友协同开发指南.md](docs/队友协同开发指南.md) — 15 分钟快速上手
> - [AGENTS.md](AGENTS.md) — 项目全貌（架构、技术栈、命令速查）

---

## 1. 项目简介

量子RL调度系统是 2026 年"揭榜挂帅"擂台赛参赛作品，核心创新为 **AI 赋能量子计算 + 量子赋能 AI** 的双向赋能：

- 用强化学习（PPO/DQN/MAPPO）智能调度量子/经典任务
- 用量子退火（QUBO 映射）加速 RL 决策

目标平台为天衍云平台真机"天衍-287"（287 量子比特超导量子计算机）。

---

## 2. 环境准备

### 2.1 系统要求

- Python **3.10+**（支持 3.10 / 3.11 / 3.12）
- Git 2.30+
- 推荐使用 TRAE / VS Code 编辑器

### 2.2 克隆与安装

```bash
# 克隆仓库
git clone https://github.com/xiabai2008/quantum-rl-scheduler.git
cd quantum-rl-scheduler

# 一键初始化（推荐）
bash setup.sh                # Git Bash / Linux / macOS
# 或
powershell .\setup.ps1       # Windows PowerShell
```

`setup.sh` / `setup.ps1` 会自动完成：创建虚拟环境 → 安装依赖 → 配置 `.env` → 创建项目目录 → 验证关键模块。

**手动安装（备选）：**

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
cp .env.example .env
```

### 2.3 安装 Pre-commit Hooks（强烈推荐）

```bash
pip install pre-commit
pre-commit install
```

`pre-commit` 会在每次 `git commit` 前自动运行代码检查（ruff + mypy + bandit 等），不合格的提交会被拦截。

### 2.4 验证环境

```bash
python -c "from src.scheduler.env import QuantumSchedulingEnv; print('OK')"
```

### 2.5 Mock 模式 vs 真机模式

- 🟢 **Mock 模式**（默认，无需改 `.env`）：不依赖天衍云平台，仿真开发即可
- 🟡 **真机模式**：去 https://qc.zdxlz.com 注册 → API 管理 → 生成 Key → 编辑 `.env`
- ⚠️ `.env` 已在 `.gitignore` 中，不会被推送

---

## 3. Git 工作流

我们采用 **简化版 GitHub Flow**（详见 [docs/Git工作流.md](docs/Git工作流.md)）。

### 3.1 分支策略

```
main（保护分支，只接受 PR 合并）
  │
  ├── feature/xxx      ← 新功能开发
  ├── fix/xxx          ← 修 Bug
  ├── docs/xxx         ← 文档更新
  ├── refactor/xxx     ← 重构
  └── chore/xxx        ← 杂项（依赖升级、CI 配置等）
```

| 你是谁        | 怎么推送                                                     |
| ---------- | -------------------------------------------------------- |
| **普通队友**   | 创建功能分支 → `git push origin feature/xxx` → 创建 PR → 1 人审批后合并 |
| **管理员/瑞哥** | `git push origin main`（GitHub 原生分支保护已启用）                 |

> ⚠️ **禁止直接 push 到 `main` 分支！** main 分支受保护，必须走 PR 流程。

### 3.2 日常工作流

```bash
# 早上：同步最新代码
git checkout main
git pull origin main
git checkout feature/你的分支
git merge main                # 把 main 合入自己的分支（避免累积冲突）

# 白天：写代码
# ... 编辑文件 ...

# 提交代码
git add src/scheduler/agent.py    # 只 add 自己改的文件！
git commit -m "feat: 实现 DQN 网络架构，完成训练循环"

# 推送并提 PR
git push origin feature/你的分支
# 然后去 GitHub 网页创建 Pull Request
```

---

## 4. Commit 格式

所有提交必须遵循 **Conventional Commits** 规范：

```
<type>: <简短描述>

<详细说明（可选）>
```

### type 类型

| type | 含义 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat: 实现 DQN 智能体训练循环` |
| `fix` | 修 Bug | `fix: 修复量子退火 QUBO 矩阵维度不匹配` |
| `docs` | 文档 | `docs: 更新 API 接口文档` |
| `test` | 添加测试 | `test: 添加任务解析器边界条件测试` |
| `refactor` | 重构（不改功能） | `refactor: 提取奖励计算为独立函数` |
| `chore` | 杂项（构建、依赖、CI） | `chore: 升级 torch 到 2.1` |
| `perf` | 性能优化 | `perf: 优化任务队列数据结构` |
| `style` | 格式（空格、逗号等） | `style: 用 ruff format 格式化所有文件` |

### 好的示例

```bash
✅ feat: 实现 Duelling DQN 网络架构
✅ fix: 修复训练时 epsilon 衰减溢出 bug
✅ refactor: 将状态编码从 8 维扩展到 14 维
❌ update code
❌ fix bug
❌ 修改了一些东西
```

> 💡 CI 会自动校验 Commit 格式，不符合规范的 PR 会被阻塞。

---

## 5. 代码规范

所有代码质量工具的配置统一在 **`pyproject.toml`** 中管理。

### 5.1 工具链

| 工具 | 用途 | 命令 |
|------|------|------|
| **ruff** | 代码检查 + 格式化（替代 flake8 + isort） | `ruff check src/ scripts/ tests/`<br>`ruff format src/ scripts/ tests/` |
| **mypy** | 类型检查（8 项严格配置） | `mypy src/` |
| **bandit** | 安全扫描（检测密钥泄露、不安全函数） | `bandit -r src/ -c pyproject.toml -ll` |
| **black** | 代码格式化（line-length=100，向后兼容保留） | `black src/ scripts/ tests/` |
| **pre-commit** | Git commit 前自动检查 | `pre-commit run --all-files` |

### 5.2 代码风格

- **行宽**：100 字符（`pyproject.toml` 中 `line-length = 100`）
- **缩进**：4 空格
- **类名**：PascalCase（`SchedulerAgent`）
- **函数/变量**：snake_case（`train_model()`）
- **常量**：UPPER_SNAKE_CASE（`MAX_RETRIES`）
- **私有方法/变量**：以下划线开头（`_build_network()`）

### 5.3 注释与 Docstring 规范

```python
def sample_function(param1: int, param2: str) -> bool:
    """
    函数功能的简短描述。

    Args:
        param1: 参数1的说明
        param2: 参数2的说明

    Returns:
        返回值的说明

    Raises:
        ValueError: 什么情况下抛出此异常

    Example:
        >>> sample_function(1, "test")
        True
    """
    pass
```

### 5.4 手动触发检查

```bash
# 一次性运行所有 pre-commit 检查
pre-commit run --all-files

# 或分工具运行
ruff check src/ scripts/ tests/           # 代码检查
ruff format src/ scripts/ tests/          # 代码格式化
mypy src/                                  # 类型检查
bandit -r src/ -c pyproject.toml -ll      # 安全扫描
```

---

## 6. 测试要求

### 6.1 测试命令

```bash
# 运行全部测试 + 覆盖率（CI 强制 ≥ 60%）
pytest tests/ --cov=src --cov-fail-under=60

# 运行指定测试文件
pytest tests/test_scheduler.py -v

# 跳过慢测试
pytest tests/ -m "not slow"

# 性能基准测试
pytest tests/benchmarks/ --benchmark-only
```

### 6.2 测试规范

- 每个新功能必须有对应的测试用例
- 测试文件与 `src/` 目录结构对应（如 `src/scheduler/agent.py` ↔ `tests/test_agent.py`）
- 函数/类必须写 docstring（中文）
- 测试覆盖率目标：**≥ 60%**（CI 强制门禁），长期目标 80%+

### 6.3 测试标记

```python
import pytest

@pytest.mark.slow              # 慢测试，可用 -m "not slow" 跳过
@pytest.mark.real_machine      # 需要真机访问
@pytest.mark.integration       # 集成测试
@pytest.mark.unit              # 单元测试
```

---

## 7. PR 流程

### 7.1 完整流程

1. **创建分支**：`git checkout -b feature/xxx`（从最新 `main` 切出）
2. **写代码**：遵循第 5 节代码规范
3. **跑测试**：`pytest tests/ --cov=src --cov-fail-under=60`
4. **跑检查**：`pre-commit run --all-files`
5. **提交并推送**：`git push origin feature/xxx`
6. **创建 PR**：去 GitHub 网页点击 "Compare & pull request"
7. **填写 PR 模板**：概述、关联 Issue、改动类型、改动内容、验证、检查清单
8. **等待 CI**：4 个 Job 必须全部通过（Lint / Test / Type Check / Benchmarks）
9. **Review**：至少 1 人 Approve 才能合并
10. **合并**：由项目经理或作者在 Review 通过后合并

### 7.2 CI 自动检查

创建 PR 后，GitHub Actions 会自动运行 4 个 Job：

| Job | 内容 |
|-----|------|
| **Lint** | ruff check + ruff format + bandit 安全扫描 |
| **Test** | pytest 多版本测试（Python 3.10 / 3.11 / 3.12）+ 覆盖率（≥ 60%） |
| **Type Check** | mypy 类型检查 |
| **Benchmarks** | 性能基准测试 |

另外还有 **PR Automation**：基于修改文件自动打标签 + Commit 格式校验。

**所有检查必须通过 PR 才能合并。** 如果失败：

1. 点击 PR 页面底部的 "Details" 查看错误日志
2. 本地修复后重新 push 即可自动重新触发

### 7.3 Review 规则

- 每个 PR 至少 **1 人 Review + 1 人 Approve** 才能合并
- 项目经理合并所有 PR（或在 Review 通过后由作者自行合并）
- 禁止直接 push 到 `main` 分支！

---

## 8. Issue 报告规范

### 8.1 报告 Bug

使用 [Bug 报告模板](https://github.com/xiabai2008/quantum-rl-scheduler/issues/new?labels=bug&template=bug.md) 创建 Issue，需包含：

- **描述**：清晰描述遇到的问题
- **复现步骤**：1. 2. 3. 编号步骤
- **期望行为**：你期望发生什么
- **实际行为**：实际发生了什么
- **环境信息**：OS / Python 版本 / 分支 / 相关依赖版本
- **日志/截图**：粘贴完整错误日志或截图

### 8.2 提交功能建议

使用 [功能建议模板](https://github.com/xiabai2008/quantum-rl-scheduler/issues/new?labels=enhancement&template=feature_request.md) 创建 Issue，需包含：

- **描述**：你希望实现什么功能
- **动机/背景**：为什么需要这个功能，解决什么问题
- **建议方案**：你设想的实现方式
- **备选方案**：其他可能的方案
- **优先级**：高 / 中 / 低

### 8.3 Issue 标签

| 标签 | 含义 |
|------|------|
| `bug` | Bug 报告 |
| `enhancement` | 功能增强 |
| `documentation` | 文档相关 |
| `algorithm` | 算法/调度核心 |
| `backend` | 后端/API |
| `frontend` | 前端/可视化 |
| `testing` | 测试相关 |
| `ci` | CI/基础设施 |
| `dependencies` | 依赖升级 |
| `help wanted` | 需要帮助 |
| `good first issue` | 适合新人的简单任务 |

---

## 9. 团队成员

| GitHub 用户名 | 权限 | 角色 |
| ------------ | ----- | ---- |
| xiabai2008 | Admin | 项目负责人（瑞哥） |
| heka-ky | Write | 队员 |
| zyhsga | Write | 队员 |
| NN2914 | Write | 队员 |
| qpqpalalzmzm112 | Write | 队员 |
| Jackhock-1 | Write | 队员 |
| DUMNOX | Write | 队员 |
| K1660729 | Write | 队员 |

---

## 10. 沟通规范

### 异步沟通（推荐）

- **GitHub Issues**：任务讨论、Bug 报告、功能建议
- **GitHub PR Comments**：代码 Review
- **飞书/微信群**：日常交流、紧急通知

### 同步沟通

- **每日站会**：21:00（15 分钟）
- **每周周会**：周日 20:00（30 分钟）

### 遇到问题

1. 先自己查文档（`docs/` 目录）、搜 GitHub Issues
2. 在群聊里问
3. 创建 GitHub Issue（附上错误日志）
4. @ 项目经理或相关负责人

---

## 11. 常用命令速查

```bash
# ── 环境初始化 ──
bash setup.sh                                   # 一键初始化
pre-commit install                              # 安装 Git hooks

# ── CLI 统一入口 ──
python scripts/cli.py train --timesteps 50000 --algorithm ppo
python scripts/cli.py simulate --num-tasks 200 --strategies all
python scripts/cli.py serve --port 8000
python scripts/cli.py demo --multi-machine

# ── 代码质量 ──
ruff check src/ scripts/ tests/                 # 代码检查
ruff format src/ scripts/ tests/                # 代码格式化
mypy src/                                       # 类型检查
bandit -r src/ -c pyproject.toml -ll            # 安全扫描
pre-commit run --all-files                      # 全量检查

# ── 测试 ──
pytest tests/ --cov=src --cov-fail-under=60     # 测试 + 覆盖率
pytest tests/benchmarks/ --benchmark-only       # 性能基准

# ── Web ──
uvicorn src.visualization.app:app --reload --port 8000

# ── Docker ──
docker-compose up -d
```

---

## 12. 注意事项

- ⚠️ 不要修改 `config/config.yaml` 的 `mock_mode: true`（除非真机闭环测试）
- ⚠️ 不要提交 `.env` 文件（已在 `.gitignore` 中）
- ⚠️ 不要提交 `models/`、`logs/`、`results/` 下的训练产物
- ✅ Python 文件用 4 空格缩进
- ✅ 函数/类必须写 docstring（中文）
- ✅ 新功能必须配套测试用例

---

感谢你的贡献！🎉 如有任何问题，请在 [Issues](https://github.com/xiabai2008/quantum-rl-scheduler/issues) 或群聊中提出。
