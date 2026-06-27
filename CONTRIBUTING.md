# 团队贡献指南

## 欢迎！

感谢你加入量子RL调度系统开发团队。请花5分钟阅读本文档，了解我们的协作方式。

---

## 快速开始

### 1. 环境准备
```bash
# 克隆仓库
git clone https://github.com/你的用户名/quantum-rl-scheduler.git
cd quantum-rl-scheduler

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 安装依赖
pip install -r requirements.txt

# 运行测试确认环境正常
python -m pytest tests/
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

### Python 代码风格
- 使用 `black` 自动格式化（line-length=88）
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
