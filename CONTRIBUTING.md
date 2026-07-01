# 团队贡献指南

## 欢迎！

感谢你加入量子RL调度系统开发团队。请花5分钟阅读本文档，了解我们的协作方式。

---

## 快速开始

### 1. 环境准备
```bash
# 克隆仓库
git clone https://github.com/xiabai2004/quantum-rl-scheduler.git
cd quantum-rl-scheduler

# 一键初始化（推荐）
bash setup.sh                # Git Bash / Linux / macOS
# 或
powershell .\setup.ps1       # Windows PowerShell

# 手动安装（备选）
python -m venv .venv
source .venv/Scripts/activate   # Windows
pip install -r requirements.txt

# 安装 Git pre-commit hooks（强烈推荐）
pip install pre-commit
pre-commit install
```

### 2. 选择你的任务
1. 查看 GitHub Issues 面板，找 `help wanted` 标签的任务
2. 在 Issue 下留言 "我来做这个"，然后 Assign 给自己
3. 创建功能分支：`git checkout -b feature/任务名`

### 3. 开始编码
- 使用 TRAE（https://trae.com.cn）辅助编码
- 遵循 PEP8 规范
- 函数必须有 docstring（中文）
- 写完代码后运行 `pytest` 确保不破坏已有功能

### 4. 提交代码
```bash
git add -A
git commit -m "feat: 简要描述你做了什么"
git push origin feature/你的分支
```
然后去 GitHub 创建 Pull Request。

---

## 代码规范

所有代码质量工具的配置统一在 **`pyproject.toml`** 中管理。

### 自动检查

| 层次 | 工具 | 触发时机 |
|------|------|----------|
| 本地 | `pre-commit` | Git commit 前 |
| 云端 | GitHub Actions CI | Push/PR 时 |

```bash
# 手动触发
black src/ scripts/ tests/      # 代码格式化
isort src/ scripts/ tests/      # import 排序
flake8 src/                      # 代码检查
mypy src/                        # 类型检查

# 一次性运行所有 pre-commit 检查
pre-commit run --all-files
```

### Python 代码风格
- 使用 `black` 自动格式化（line-length=100，通过 pyproject.toml 配置）
- 类名：PascalCase（`SchedulerAgent`）
- 函数/变量：snake_case（`train_model()`）
- 常量：UPPER_SNAKE_CASE（`MAX_RETRIES`）
- 私有方法/变量：以下划线开头（`_build_network()`）

### 注释规范
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

### 文件头注释
```python
"""
模块名称：RL智能体模块
功能描述：实现基于DQN的调度策略智能体
负责人：张三
创建日期：2026-07-01
"""
```

---

## 目录结构约定

```
quantum-rl-scheduler/
├── src/                    # 源代码
│   ├── scheduler/          # ⚠️ 核心：调度引擎（不要随意重构！）
│   ├── api/                # 天衍云API封装
│   ├── quantum/            # 量子计算模块
│   ├── visualization/      # Web界面
│   └── utils/              # 工具函数
├── tests/                  # ⚠️ 测试与src/对应
├── config/                 # 配置文件
├── scripts/                # 运行脚本
├── docs/                   # 文档
├── models/                 # 训练好的模型（不提交到Git）
├── logs/                   # 训练日志（不提交到Git）
└── results/                # 实验结果（不提交到Git）
```

---

## CI/CD 自动检查

每次 push 或创建 PR 时，GitHub Actions 会自动运行以下检查：

| 检查项 | 内容 |
|--------|------|
| Lint | Black 格式化 + isort 排序 + flake8 检查 |
| Test | pytest 多版本测试（Python 3.10/3.11/3.12）+ 覆盖率 |
| Type Check | mypy 类型检查 |
| PR Labels | 基于修改文件自动打标签 |
| Commit Format | Conventional Commits 格式校验 |

**所有检查必须通过 PR 才能合并。** 如果失败：
1. 点击 PR 页面底部的检查详情查看错误日志
2. 本地修复后重新 push 即可自动重新触发

## 测试要求

- 每个新功能必须有对应的测试用例
- 运行 `python -m pytest tests/ -v` 确保全部通过
- 测试覆盖率目标：>80%

---

## 沟通规范

### 异步沟通（推荐）
- GitHub Issues：任务讨论、Bug报告
- GitHub PR Comments：代码review
- 飞书/微信群：日常交流、紧急通知

### 同步沟通
- 每日站会：21:00（15分钟）
- 每周周会：周日20:00（30分钟）

### 遇到问题
1. 先自己查文档、搜GitHub Issues
2. 在群聊里问
3. 创建GitHub Issue（附上错误日志）
4. @项目经理或相关负责人
