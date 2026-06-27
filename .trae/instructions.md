# 量子RL调度系统 - AI 开发指引

## 项目概述
本项目是"揭榜挂帅"参赛作品，题目为"量子RL驱动的天衍云平台智能调度系统"。
核心目标：用强化学习（RL）实现量子-经典混合计算任务的智能调度，
双向赋能：AI赋能量子（RL调度决策）+ 量子赋能AI（量子退火加速RL）。

## 技术栈
- Python 3.10+
- 强化学习：Stable-Baselines3（DQN）
- 量子计算：Qiskit、PennyLane（仿真）
- Web界面：FastAPI + Vue3
- 配置管理：PyYAML

## 代码规范
- 使用 Black 格式化（line-length=88）
- 中文注释，函数必须有 docstring
- 类名 PascalCase，函数/变量 snake_case

## 当前开发优先级
1. src/scheduler/env.py — RL环境（Gymnasium接口）
2. src/scheduler/agent.py — RL智能体（DQN训练）
3. src/api/tianyan_client.py — 天衍云API封装
4. src/quantum/annealing.py — 量子退火加速模块
5. src/visualization/app.py — Web监控界面
