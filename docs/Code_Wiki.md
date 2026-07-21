# 量子RL调度系统 — Code Wiki

> **作品名称**：量子RL驱动的天衍云平台智能调度系统
> **核心创新**：AI 赋能量子计算（RL 智能调度） + 量子赋能 AI（量子退火加速 RL 决策）
> **目标平台**：天衍云真机"天衍-287"（祖冲之三号同款超导量子计算机）
> **文档版本**：v6（MAPPO 多智能体 + 退火异步闭环 + 14维状态空间 + 多目标奖励）
> **最后更新**：2026-07-01

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 系统架构](#2-系统架构)
- [3. 模块职责详解](#3-模块职责详解)
  - [3.1 调度引擎 src/scheduler/](#31-调度引擎-srcscheduler)
  - [3.2 API 客户端 src/api/](#32-api-客户端-srcapi)
  - [3.3 量子加速 src/quantum/](#33-量子加速-srcquantum)
  - [3.4 可视化 src/visualization/](#34-可视化-srcvisualization)
  - [3.5 工具函数 src/utils/](#35-工具函数-srcutils)
- [4. 关键类与函数参考](#4-关键类与函数参考)
- [5. 依赖关系](#5-依赖关系)
- [6. 项目运行方式](#6-项目运行方式)
- [7. 测试与验证](#7-测试与验证)
- [8. 部署与容器化](#8-部署与容器化)

---

## 1. 项目概述

本项目是 2026 年"揭榜挂帅"擂台赛榜题"量子AI双向赋能的研究与应用探索"的参赛作品，由共青团中央主办、中国电信发榜、中电信量子执行。

**双向赋能核心**：

- **AI 赋能量子计算**：用强化学习（RL）智能调度量子/经典混合任务，量化目标为资源利用率提升 ≥30%
- **量子赋能 AI**：用量子退火（QUBO 映射）加速 RL 策略搜索

**技术栈**：Python ≥3.10 + Stable-Baselines3（DQN/PPO）+ Gymnasium + PyTorch + Qiskit + D-Wave Ocean SDK + FastAPI + Vue3 + Echarts

**核心代码量**：约 13,100 行 Python（不含测试和文档），561 个单元测试用例。

### 1.1 版本演进

| 版本 | 关键变化 | 核心成果 |
|------|---------|---------|
| v1-v3 | 初始代码 + 训练脚本 + 10维状态 + reward 归一化 | DQN reward 从 -843 提升至 -145 |
| v4 | 环境异质化 + PPO 主力算法 | PPO 单机平均奖励 +2,804，超越所有基线 92.5% |
| v5 | 多机器调度 + 真机验证 | 多机器 PPO 奖励 +4,294（+86.3%），17 个任务成功提交天衍云真机 |
| v6 | MAPPO 多智能体 + 异步退火闭环 + 14维状态 + 多目标奖励 | 新增 ~3,600 行代码 + 71 测试用例 |

---

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    用户界面 (FastAPI Web 监控, port 8000)         │
│              Vue3 + Echarts 实时面板 + WebSocket 推送              │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST + WebSocket
┌────────────────────────────┴────────────────────────────────────┐
│                       调度引擎 (src/scheduler/)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  parser.py   │  │   env.py     │  │      agent.py          │ │
│  │ 任务解析器    │→ │ Gymnasium 环境│← │ DQN + PPO + LSTM 智能体 │ │
│  │ QASM→Task    │  │ 14维状态/3动作 │  │ + 退火回调 + 真机回调   │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
│  ┌──────────────┐  ┌──────────────────────┐  ┌───────────────┐  │
│  │   marl.py    │  │ multi_objective_env  │  │ async_anneal  │  │
│  │ MAPPO多智能体 │  │  多目标奖励包装器     │  │ _callback.py  │  │
│  └──────────────┘  └──────────────────────┘  └───────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐   ┌─────────────────┐   ┌──────────────┐
│ API 客户端     │   │  量子加速模块     │   │  工具函数     │
│ src/api/      │   │  src/quantum/   │   │  src/utils/  │
│               │   │                  │   │              │
│ ┌───────────┐ │   │ ┌──────────────┐│   │ ┌──────────┐ │
│ │tianyan_   │ │   │ │ annealing.py ││   │ │helpers.py│ │
│ │client.py  │ │   │ │ QUBO退火优化  ││   │ └──────────┘ │
│ └───────────┘ │   │ └──────────────┘│   │ ┌──────────┐ │
│ ┌───────────┐ │   │ ┌──────────────┐│   │ │metrics.py│ │
│ │tianyan_   │ │   │ │annealing_    ││   │ │Prometheus│ │
│ │cqlib.py   │ │   │ │loop.py 异步闭环││   │ └──────────┘ │
│ └───────────┘ │   │ └──────────────┘│   └──────────────┘
│ ┌───────────┐ │   └─────────────────┘
│ │mock_      │ │
│ │client.py  │ │
│ └───────────┘ │
└───────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│        天衍云真机 (天衍-287 / tianyan_s/sw/tn)    │
│        cqlib SDK + QCIS 量子指令集                │
└──────────────────────────────────────────────────┘
```

### 2.2 数据流

```
用户提交任务 (QASM/字典/YAML)
      │
      ▼
TaskParser.parse() → 规范化 Task 对象
      │
      ▼
QuantumSchedulingEnv.reset() → 任务入队，生成 14维观测
      │
      ▼
RL Agent (PPO/MAPPO/DQN) → 输出动作 (0=经典/1=量子/2=混合)
      │
      ├── 多机器调度：_select_best_machine() 启发式评分择优
      │
      ▼
真机提交 (可选, real_submit_probability 控制)
      │
      ▼
量子退火优化 (异步闭环)
      │   ├── RL 训练 → 周期性触发退火
      │   ├── QUBO 映射 → 退火求解 → 权重更新
      │   └── 验证环境评估 → 自适应频率调整
      ▼
Web 界面实时展示 (WebSocket 推送)
```

### 2.3 核心设计模式

| 模式 | 应用位置 | 说明 |
|------|---------|------|
| 工厂模式 | `src/api/__init__.py` | `get_client()` 自动选择 Mock/真实客户端 |
| 包装器模式 | `multi_objective_env.py` | `MultiObjectiveRewardWrapper` 包装原环境 |
| 生产者-消费者 | `annealing_loop.py` | RL 训练线程生产任务，退火工作线程消费 |
| CTDE 架构 | `marl.py` | 多智能体集中训练、分布执行 |
| 熔断器模式 | `tianyan_client.py` | 连续失败自动熔断，保护天衍云 API |
| 策略模式 | `run_simulation.py` | 8 种调度策略可插拔对比 |
| Builder 模式 | `parser.py` | `TaskBuilder` 链式构造 Task |

---

## 3. 模块职责详解

### 3.1 调度引擎 src/scheduler/

调度引擎是项目核心模块，基于 Gymnasium 框架实现量子-经典混合任务调度环境，并提供多种 RL 智能体。

#### 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| [env.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/scheduler/env.py) | 1398 | Gymnasium 调度环境（14维状态/3动作/异质化任务/多机器调度） |
| [parser.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/scheduler/parser.py) | 864 | 任务解析器（字典/QASM/YAML/文本 → Task） |
| [agent.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/scheduler/agent.py) | 1261 | DQN + PPO/LSTM 智能体 + 退火/真机回调 |
| [marl.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/scheduler/marl.py) | 1134 | MAPPO 多智能体调度（CTDE 架构） |
| [multi_objective_env.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/scheduler/multi_objective_env.py) | 372 | 多目标奖励包装器（吞吐量/平衡/服务质量） |
| [async_annealing_callback.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/scheduler/async_annealing_callback.py) | 132 | 异步量子退火训练回调 |
| [__init__.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/scheduler/__init__.py) | 92 | 模块统一导出（延迟导入） |

#### 3.1.1 状态空间与动作空间

**14维状态空间**（`OBS_DIM = 14`）：

| 索引 | 常量 | 含义 |
|------|------|------|
| 0 | `OBS_QUBIT_AVAILABILITY` | 可用量子比特比率 |
| 1 | `OBS_QUEUE_LENGTH` | 任务队列长度（归一化） |
| 2 | `OBS_AVG_WAIT_TIME` | 平均等待时间（归一化） |
| 3 | `OBS_FIDELITY` | 量子比特平均保真度 |
| 4 | `OBS_CLASSICAL_LOAD` | 经典计算资源负载 |
| 5 | `OBS_QUANTUM_QUEUE_RATIO` | 量子专用队列占比 |
| 6 | `OBS_TIME_OF_DAY` | 时间段（昼夜模拟） |
| 7 | `OBS_URGENCY_LEVEL` | 当前任务紧急程度 |
| 8 | `OBS_TASK_TYPE_QUANTUM` | quantum 类型标识 |
| 9 | `OBS_TASK_TYPE_CLASSICAL` | classical 类型标识 |
| 10 | `OBS_SINGLE_GATE_FIDELITY` | 单比特门保真度（v6 新增） |
| 11 | `OBS_TWO_GATE_FIDELITY` | 两比特门保真度（v6 新增） |
| 12 | `OBS_COUPLING_DENSITY` | 耦合图密度（v6 新增） |
| 13 | `OBS_AVG_CONNECTIVITY` | 平均连通度（v6 新增） |

**3类动作空间**（`spaces.Discrete(3)`）：

| 动作 | 常量 | 含义 |
|------|------|------|
| 0 | `ACTION_CLASSICAL` | 分配到经典资源 |
| 1 | `ACTION_QUANTUM` | 分配到量子资源 |
| 2 | `ACTION_HYBRID` | 混合执行 |

#### 3.1.2 奖励机制

| 场景 | 奖励值 | 说明 |
|------|--------|------|
| 量子执行 | `10.0 × speedup × fidelity_factor + 3.0` | speedup ∈ [2,5]，保真度<0.9 再乘 0.6 |
| 经典执行 | `5.0 + 3.0 = 8.0` | 基础奖励 + 成功奖励 |
| 混合执行 | `7.0 × (0.5 + 0.5×available_ratio) + 3.0` | 受资源可用率影响 |
| 错误分配 | `-2.0` | 任务类型与资源不匹配，重新入队 |
| 等待超时 | `-0.1` | 每步惩罚 |
| 量子利用率低 | `-1.0` | available_ratio > 0.7 时触发 |

#### 3.1.3 多机器调度机制

`DEFAULT_MACHINE_CONFIGS` 基于天衍云真实超导机器：

| 机器名 | 量子比特数 | 支持门集 |
|--------|-----------|---------|
| tianyan_s | 287 | H, CZ, M |
| tianyan_sw | 72 | H, CZ, M, X, Y |
| tianyan_tn | 176 | H, CZ, M, RX, RY, RZ |

**机器选择启发式**：`评分 = fidelity × available_ratio / (1 + quantum_queue)`，兼顾质量与负载。

#### 3.1.4 算法层次

| 算法 | 类 | 说明 |
|------|-----|------|
| DQN（备选） | `SchedulerAgent` | Dueling DQN 架构，异质化环境下表现不佳 |
| PPO（主力） | `PPOAgent` | 已验证超越所有基线（v4 单机 +2804，v5 多机 +4294） |
| PPO + LSTM | `PPOAgent(use_lstm=True)` | v6 新增，时序依赖建模 |
| MAPPO | `MultiAgentPPO` | v6 新增，CTDE 多机器协调 |
| 多目标 RL | `MultiObjectiveRewardWrapper` | v6 新增，3 目标加权（可运行时切换） |

---

### 3.2 API 客户端 src/api/

封装天衍云平台 API，支持 Mock/真实模式自动切换，提供熔断器保护。

#### 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| [tianyan_client.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/api/tianyan_client.py) | 789 | 主客户端（REST/cqlib/Mock 三路委托 + 熔断器） |
| [tianyan_cqlib.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/api/tianyan_cqlib.py) | 456 | cqlib 真机客户端 + 多机器协调器 |
| [mock_client.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/api/mock_client.py) | 602 | Mock 客户端 + 工厂函数 |
| [circuit_breaker.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/api/circuit_breaker.py) | 121 | 独立熔断器（包裹式 API） |
| [__init__.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/api/__init__.py) | 64 | 工厂函数 `get_client()` / `get_cqlib_client()` |

#### 3.2.1 三路委托架构

`TianyanClient` 内部根据模式委托到不同后端：

```
TianyanClient
   ├── Mock 模式 → MockTianyanClient（内存模拟 + 状态轮转）
   ├── 真实模式 → CqlibTianyanClient（cqlib SDK 真机提交，优先）
   │      └── REAL_MACHINES 故障切换（8 台备用机）
   └── REST 路径（deprecated，被 WAF 拦截）
          └── _request() + 指数退避重试 + 熔断器
```

#### 3.2.2 Mock 模式判定优先级

1. 显式传参 `mock_mode`
2. 环境变量 `TIANYAN_MOCK_MODE`（true/1/yes）
3. 配置文件 `config/config.yaml` 中 `tianyan.mock_mode`
4. 默认：True（Mock 模式）

#### 3.2.3 熔断器机制

| 状态 | 行为 |
|------|------|
| CLOSED | 正常放行，累计失败计数 |
| OPEN | 熔断拒绝，抛 `CircuitOpenError` |
| HALF_OPEN | 恢复超时后放行一次试探 |

默认配置：`failure_threshold=5`（连续失败 5 次熔断），`recovery_timeout=60.0`（60 秒后试探恢复）。

#### 3.2.4 真机列表

`CqlibTianyanClient.REAL_MACHINES` 包含 8 台天衍云超导真机：

```
tianyan_sw, tianyan_s, tianyan_tn, tianyan_tnn,
tianyan_swn, tianyan_sa, tianyan176, tianyan176-2
```

---

### 3.3 量子加速 src/quantum/

实现量子退火加速 RL 策略搜索，核心思想是将神经网络权重优化问题映射为 QUBO 问题。

#### 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| [annealing.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/quantum/annealing.py) | 1286 | 量子退火策略优化器（QUBO 映射 + 退火求解） |
| [annealing_loop.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/quantum/annealing_loop.py) | 343 | 异步退火闭环控制器（生产者-消费者） |
| [__init__.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/quantum/__init__.py) | 34 | 模块导出与旧版兼容别名 |

#### 3.3.1 QUBO 映射机制

将 DQN 策略网络权重优化问题映射为 QUBO（Quadratic Unconstrained Binary Optimization）：

**目标函数**：`min_x  g^T Δw(x) + λ × ||Δw(x)||²`

其中 `Δw(x)` 是从二进制变量 `x` 解码的权重更新量，`g` 是梯度向量，`λ` 是正则化系数。

**编码格式**（每权重 `n_bits_per_weight` bit）：
- `bit 0`：符号位（1=负更新，0=正更新）
- `bit 1..n-1`：数值位，权重为 `1/2, 1/4, 1/8, ...`

**v2 改进**：权重差编码 + 梯度引导 + L2 正则化 + 符号-数值表示，确保 QUBO 最小化对应梯度下降方向。

#### 3.3.2 退火求解路径

```
anneal(qubo_matrix)
   ├── 路径1: 真机退火 (cqlib_client.submit_annealing_task)
   │      └── 天衍云为门控量子 SDK，不提供 QUBO 退火 → 降级仿真
   ├── 路径2: D-Wave neal (neal.SimulatedAnnealingSampler)
   └── 路径3: numpy 内置模拟退火 (Metropolis-Hastings)
```

#### 3.3.3 异步退火闭环

`AsyncAnnealingLoop` 实现生产者-消费者模式：

```
RL 训练线程（生产者）              退火工作线程（消费者）
   │ step 达到间隔                    │
   │ submit(policy快照) ──────► queue.Queue(maxsize=1)
   │ 继续训练（不阻塞）                │ _worker_loop:
   │                                  │   1. 评估旧奖励
   │                                  │   2. 退火优化（带重试+降级）
   │                                  │   3. 评估新奖励
   │                                  │   4. 自适应调整间隔
   │ rollout 开始前                    │   5. 写入 _pending_result
   │ ◄── get_pending_result() ────    │
   │ load_state_dict() 回写权重        │
```

**自适应频率**：连续 3 次有效（delta > threshold）→ 间隔减半；连续 3 次无效 → 间隔加倍。

---

### 3.4 可视化 src/visualization/

基于 FastAPI + Vue3 + Echarts 的实时监控界面。

#### 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| [app.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/visualization/app.py) | 1534 | FastAPI 监控后端 + WebSocket + Prometheus |
| [frontend/index.html](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/visualization/frontend/index.html) | 920 | Vue3 + Echarts 前端监控面板 |
| [__init__.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/visualization/__init__.py) | 12 | 模块导出 |

#### 3.4.1 REST API 端点

| 方法 | 路径 | 用途 |
|------|------|------|
| GET | `/` | 返回 Vue3 监控面板 HTML |
| GET | `/api/status` | 获取当前系统状态 |
| GET | `/api/real-machines` | 查询天衍云真机状态（实时 cqlib 轮询） |
| GET | `/api/real-submissions` | 查询最近真机提交记录 |
| GET | `/api/tasks` | 按状态过滤任务列表 |
| POST | `/api/tasks` | 提交新任务 |
| GET | `/api/metrics` | Prometheus 格式指标端点 |
| POST | `/api/strategy` | 切换调度策略 |
| POST | `/api/update` | 调度引擎回写状态 |
| GET | `/api/ppo/comparison` | 策略对比数据 |
| GET | `/api/ppo/predict` | PPO 单步推理 |
| GET | `/api/ppo/stats` | PPO 性能指标 |

#### 3.4.2 WebSocket 实时推送

| 消息类型 | 触发场景 |
|---------|---------|
| `init` | 客户端首次连接 |
| `status_update` | 每 3 秒模拟调度推送 |
| `task_added` | 提交新任务 |
| `strategy_changed` | 切换策略 |
| `pong` | 心跳响应 |

#### 3.4.3 Prometheus 指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `quantum_scheduler_qubit_utilization` | gauge | 量子比特利用率 |
| `quantum_scheduler_queue_length` | gauge | 任务队列长度 |
| `quantum_scheduler_completed_tasks` | counter | 已完成任务总数 |
| `quantum_scheduler_avg_wait_time` | gauge | 平均等待时间 |
| `quantum_scheduler_current_step` | counter | 当前调度步数 |

---

### 3.5 工具函数 src/utils/

#### 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| [helpers.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/utils/helpers.py) | 305 | 日志/配置/数据预处理/性能评估工具 |
| [metrics.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/utils/metrics.py) | 111 | Prometheus 指标定义（Counter/Gauge/Histogram） |
| [__init__.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/utils/__init__.py) | 36 | 模块导出 |

`metrics.py` 定义了更完整的 9 个 Prometheus 指标，覆盖调度引擎、API 调用、量子退火与运行时状态。

---

## 4. 关键类与函数参考

### 4.1 调度引擎核心类

#### `QuantumSchedulingEnv(gym.Env)`

量子-经典混合任务调度环境，项目核心。

```python
class QuantumSchedulingEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        max_steps: int = 500,                    # 最大步数
        max_qubits: int = 287,                    # 天衍-287
        render_mode: str | None = None,
        seed: int | None = None,
        machine_configs: list[dict] | None = None,  # None→单机模式
        real_submit_probability: float = 0.0,       # 真机提交概率
    ) -> None
```

**关键方法**：

| 方法 | 用途 |
|------|------|
| `attach_real_clients(clients: dict)` | 绑定真机客户端，启用真机验证 |
| `reset(*, seed, options) -> (obs, info)` | 重置环境，随机初始化 5-20 任务 |
| `step(action: int) -> (obs, reward, terminated, truncated, info)` | 执行一步调度决策 |
| `get_random_pending_task() -> Task \| None` | 随机取待处理任务（真机抽样用） |
| `machine_names -> list[str]` | 所有机器名称（property） |
| `num_machines -> int` | 机器数量（property） |

#### `PPOAgent`

PPO 调度智能体（主力算法）。

```python
class PPOAgent:
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_lstm: bool = False,              # v6: LSTM 支持
        n_lstm_layers: int = 1,
        lstm_hidden_size: int = 64,
        use_annealing: bool = False,         # 量子退火
        anneal_interval: int = 1000,
        anneal_qubits: int = 10,
        annealing_time: float = 20.0,
        anneal_shots: int = 1000,
        anneal_simulation_mode: bool = True,
        anneal_cqlib_client=None,
    )
```

**关键方法**：`train(total_timesteps, eval_freq, ...)`、`predict(state, deterministic)`、`evaluate(num_episodes)`、`save(path)` / `load(path)`、`get_config()`

#### `SchedulerAgent`

基于 Dueling DQN 的调度智能体（备选算法）。

```python
class SchedulerAgent:
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_BUFFER_SIZE = 10000
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_GAMMA = 0.99
    NET_ARCH = [128, 64]

    def __init__(self, env, learning_rate=..., buffer_size=..., batch_size=...,
                 gamma=..., target_update_interval=..., train_freq=...,
                 epsilon_start=..., epsilon_end=..., epsilon_decay=...,
                 learning_starts=..., tau=..., log_dir=..., verbose=..., seed=None)
```

#### `MultiAgentPPO`

MAPPO 多智能体调度（CTDE 架构）。

```python
class MultiAgentPPO:
    def __init__(
        self,
        env: QuantumSchedulingEnv,
        learning_rate: float = 3e-4,
        n_steps: int = 1024,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        actor_hidden: tuple = (128, 64),
        critic_hidden: tuple = (256, 128),
        seed: int | None = None,
        device: str = "auto",
    )
```

**关键属性**：`actors: list[ActorNet]`（每台机器一个 Actor）、`critic: CentralizedCritic`（共享 Critic）、`wrapper: MultiAgentEnvWrapper`

#### `MultiObjectiveRewardWrapper(gym.Wrapper)`

多目标奖励包装器，将标量奖励分解为 3 个目标。

```python
class MultiObjectiveRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: QuantumSchedulingEnv,
        weights: list[float] | None = None,       # [w_throughput, w_balance, w_quality]
        weight_preset: str | None = None,         # 预设名称
    )
```

**3 个目标**：
- `throughput`：本步完成任务数 [0,1]
- `balance`：`-|quantum_available - classical_load|` [-1,0]
- `quality`：`-avg_wait / MAX_WAIT_STEPS` [-1,0]

**权重预设**：`throughput_heavy` / `balance_heavy` / `quality_heavy` / `balanced` / `throughput_only` / `balance_only` / `quality_only`

#### 数据类

| 类 | 文件 | 用途 |
|-----|------|------|
| `Task` (env) | env.py | 队列中的待调度任务（含 wait_steps/urgency） |
| `Task` (parser) | parser.py | 规范化任务（含 deadline/status/algorithm） |
| `QuantumResource` | env.py | 量子资源聚合状态 |
| `ClassicalResource` | env.py | 经典计算资源状态 |
| `QuantumMachine` | env.py | 单台量子计算机状态（含噪声/拓扑特征） |

> 注意：`env.py` 的 `Task` 与 `parser.py` 的 `Task` 是两个不同的数据类，前者面向调度队列，后者面向解析。

#### 回调类

| 类 | 继承 | 用途 |
|-----|------|------|
| `AnnealingCallback` | `BaseCallback` | 同步量子退火优化（每 N 步阻塞训练） |
| `AsyncAnnealingCallback` | `BaseCallback` | 异步量子退火（工作线程，不阻塞训练） |
| `RealMachineCallback` | `BaseCallback` | 真机抽样回调（按概率提交天衍云） |
| `EpsilonExplorationCallback` | `BaseCallback` | Epsilon-Greedy 探索率衰减 |

---

### 4.2 API 客户端核心类

#### `TianyanClient`

天衍云主客户端，三路委托 + 熔断器。

```python
class TianyanClient:
    MAX_RETRIES = 3                    # 指数退避最大重试次数
    RETRY_BACKOFF_FACTOR = 2
    RETRY_INITIAL_WAIT = 1.0

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        mock_mode: bool | None = None,
        enable_circuit_breaker: bool = True,
    )
```

**关键方法**：`authenticate()`、`submit_quantum_task(circuit_qasm, shots, backend, qcis, task_name)`、`get_task_status(task_id)`、`get_task_result(task_id)`、`list_backends()`、`wait_for_task(task_id, poll_interval, timeout)`

#### `CqlibTianyanClient`

cqlib 真机客户端。

```python
class CqlibTianyanClient:
    REAL_MACHINES = ["tianyan_sw", "tianyan_s", "tianyan_tn", "tianyan_tnn",
                     "tianyan_swn", "tianyan_sa", "tianyan176", "tianyan176-2"]

    def __init__(
        self,
        login_key: str,
        machine_name: str = "tianyan_s",
        auto_retry_machine: bool = True,
    )
```

**关键方法**：`submit_quantum_task(qcis, circuit, shots, task_name)`（三步策略：预检→提交→故障切换）、`list_backends()`、`wait_for_task(task_id, timeout, poll_interval)`

#### `MultiMachineCqlibCoordinator`

多机器协调器。

```python
class MultiMachineCqlibCoordinator:
    def __init__(
        self,
        login_key: str,
        machine_names: list[str],
        auto_retry_machine: bool = False,   # 多机器场景默认关闭
    )
```

**关键方法**：`submit_to_machine(machine_name, qcis, shots, task_name)`、`get_all_status()`、`get_submit_stats()`、`as_client_map()`（返回 `{name: client}` 映射）

#### `MockTianyanClient`

Mock 客户端，内存模拟任务生命周期。

```python
class MockTianyanClient:
    def __init__(
        self,
        mock_delay: float = 1.0,
        mock_failure_rate: float = 0.0,
        api_key: str | None = None,
        base_url: str | None = None,
    )
```

**模拟机制**：PENDING → RUNNING（30% 概率）→ COMPLETED（40% 概率）；Bell 态电路返回 `{"00": shots/2, "11": shots/2}`。

#### 工厂函数

```python
# src/api/__init__.py
def get_client(mock_mode: bool | None = None) -> Any
def get_cqlib_client(machine_name: str = "tianyan_s") -> CqlibTianyanClient

# src/api/tianyan_cqlib.py
def create_multi_machine_clients(login_key: str, machine_names: list[str]) -> dict[str, CqlibTianyanClient]
```

---

### 4.3 量子加速核心类

#### `QuantumAnnealingOptimizer`

量子退火策略优化器。

```python
class QuantumAnnealingOptimizer:
    def __init__(
        self,
        num_qubits: int = 16,
        annealing_time: float = 20.0,
        shots: int = 1000,
        simulation_mode: bool = True,
        cqlib_client: Any = None,
    )
```

**关键方法**：

| 方法 | 用途 |
|------|------|
| `network_to_qubo(weights, gradients, td_errors) -> np.ndarray` | 神经网络权重映射为 QUBO 矩阵 |
| `anneal(qubo_matrix) -> str` | 退火求解，返回最优比特串 |
| `bitstring_to_weights(bitstring, original_shape, current_weights) -> list[np.ndarray]` | 比特串解码为权重 |
| `optimize_policy(agent, num_iterations, learning_rate, callback, replay_buffer, head_only, max_head_tensors) -> Any` | 主优化循环 |

**`optimize_policy` 接受准则**：只有当 loss 下降或上升幅度不超过 1% 阈值时才接受更新，否则回滚。

**`head_only` 模式**：默认仅优化 PPO 网络最后 4 个参数张量（action_net + value_net，约 260 参数），避免 QUBO 矩阵 OOM。

#### `AsyncAnnealingLoop`

异步退火闭环控制器。

```python
class AsyncAnnealingLoop:
    def __init__(
        self,
        optimizer: Any,
        validation_env: Any,
        eval_episodes: int = 3,
        eval_deterministic: bool = True,
        initial_interval: int = 5000,        # 初始触发间隔（步数）
        min_interval: int = 1000,
        max_interval: int = 20000,
        improvement_threshold: float = 0.0,
        retry_delays: list[float] | None = None,  # 默认 [5.0, 15.0]
        log_path: str = "results/annealing_loop_log.json",
        queue_maxsize: int = 1,
    )
```

**关键方法**：`start()`、`shutdown(wait, timeout)`、`submit(policy, step) -> bool`（非阻塞）、`get_pending_result() -> dict | None`、`get_current_interval() -> int`、`get_history() -> list[dict]`

**真机降级**：重试 `[5s, 15s]` 后仍失败 → `optimizer.simulation_mode = True` → 仿真退火兜底。

---

### 4.4 可视化核心类

#### `ConnectionManager`

WebSocket 连接管理器。

```python
class ConnectionManager:
    def __init__(self)
    # 属性：active_connections: list[WebSocket]

    async def connect(self, websocket: WebSocket) -> None    # 接受新连接
    def disconnect(self, websocket: WebSocket) -> None       # 移除连接
    async def broadcast(self, message: dict) -> None         # 广播消息
```

#### `start_web_server`

```python
def start_web_server(host: str = "0.0.0.0", port: int = 8000) -> None
```

通过 `uvicorn.run(app, host, port)` 启动 Web 服务器。

---

### 4.5 工具函数参考

#### 日志与配置

```python
def setup_logging(log_dir: str = "logs", log_level: str = "INFO", log_file: str = "scheduler.log") -> Any
def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]
def save_config(config: dict, config_path: str = "config/config.yaml") -> None
```

#### 数据预处理

```python
def normalize_vector(vector: list[float], min_val: float = 0.0, max_val: float = 1.0) -> list[float]
def one_hot_encode(category: str, categories: list[str]) -> list[int]
```

#### 性能评估

```python
def calculate_completion_rate(completed: int, total: int) -> float
def calculate_average_wait_time(wait_times: list[float]) -> float
def calculate_resource_utilization(used: float, total: float) -> float
```

#### `MetricsCalculator`

```python
class MetricsCalculator:
    @staticmethod
    def calculate_reward(completion_rate, avg_wait_time, resource_utilization, max_wait_time=3600.0) -> float
    # reward = 0.4 * completion_rate + 0.3 * normalized_wait + 0.3 * resource_utilization

    @staticmethod
    def calculate_improvement(new_value: float, baseline_value: float) -> float
```

---

## 5. 依赖关系

### 5.1 模块间依赖

```
src/scheduler/__init__.py
  ├── env.py（核心，无外部 RL 依赖）
  ├── parser.py（独立，依赖 yaml）
  ├── agent.py（依赖 stable_baselines3 + sb3_contrib + src.quantum.annealing）
  ├── marl.py（依赖 torch + env.py 的 OBS_DIM/MAX_QUEUE_SIZE/QuantumSchedulingEnv）
  ├── multi_objective_env.py（依赖 gymnasium + env.py）
  └── async_annealing_callback.py（依赖 stable_baselines3 + src.quantum.annealing_loop）

src/api/__init__.py
  ├── tianyan_client.py（依赖 requests/yaml/dotenv/loguru + src.exceptions）
  │      ├── 运行时导入 MockTianyanClient（Mock 模式）
  │      └── 运行时导入 CqlibTianyanClient（真实模式）
  ├── tianyan_cqlib.py（依赖 cqlib SDK + loguru）
  ├── mock_client.py（依赖 loguru，动态导入 TianyanAPIError）
  └── circuit_breaker.py（依赖 src.exceptions.CircuitOpenError）

src/quantum/__init__.py
  ├── annealing.py（依赖 numpy + 可选 neal/dimod + 可选 cqlib）
  └── annealing_loop.py（依赖 threading/queue + annealing.py）

src/visualization/app.py
  ├── 依赖 fastapi/uvicorn + src.scheduler.env + src.scheduler.agent
  ├── 依赖 src.api.tianyan_cqlib（真机状态查询）
  └── 依赖 src.utils.helpers（配置加载）

src/utils/
  ├── helpers.py（依赖 numpy/yaml/loguru）
  └── metrics.py（依赖 prometheus_client）
```

### 5.2 外部依赖（requirements.txt）

| 分组 | 依赖 | 版本 | 用途 |
|------|------|------|------|
| 核心 | numpy / pandas / scipy | ≥1.24 / ≥2.0 / ≥1.10 | 数值计算 |
| 强化学习 | gymnasium / stable-baselines3 / sb3-contrib | ≥0.28 / ≥2.0 / ≥2.0 | RL 框架 |
| 深度学习 | torch / tensorboard | ≥2.0 / ≥2.14 | 神经网络 |
| 量子计算 | qiskit / qiskit-aer / pennylane | ≥1.0 / ≥0.14 / ≥0.35 | 量子电路仿真 |
| 量子退火 | D-Wave Ocean SDK（可选） | — | dimod/neal 退火求解 |
| Web | fastapi / uvicorn / pydantic | ≥0.104 / ≥0.24 / ≥2.0 | 监控界面 |
| 前端 | Vue3 + Echarts（CDN） | — | 监控面板 |
| 配置 | python-dotenv / pyyaml | ≥1.0 / ≥6.0 | 配置管理 |
| 日志 | loguru | ≥0.7 | 日志框架 |
| 测试 | pytest / pytest-cov / pytest-timeout / hypothesis | ≥7.4 / ≥4.1 / ≥2.1 / ≥6.100 | 单元测试 |
| 代码质量 | black / isort / mypy / ruff / bandit | ≥23 / ≥5.12 / ≥1.5 / ≥0.4 / ≥1.7 | 代码规范 |
| 可观测性 | prometheus_client | ≥0.19 | 指标收集 |
| CLI | click | ≥8.1 | 命令行接口 |
| 真机 | cqlib（可选） | — | 天衍云真机 SDK |

### 5.3 环境变量

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `TIANYAN_API_KEY` | 天衍云 API 密钥 | — |
| `TIANYAN_API_SECRET` | 天衍云 API 密钥 | — |
| `TIANYAN_MOCK_MODE` | Mock 模式开关 | true |
| `TIANYAN_MACHINE` | 真机名称 | tianyan_s |
| `QUANTUM_ACCELERATION_ENABLED` | 量子加速全局开关 | 0 |
| `SIMULATION_MODE` | 仿真模式 | true |
| `ANNEALING_ENABLED` | 退火启用 | true |
| `ANNEALING_SIMULATION` | 退火仿真模式 | true |

---

## 6. 项目运行方式

### 6.1 环境初始化

```bash
# Linux/macOS/Git Bash 一键初始化
bash setup.sh

# Windows PowerShell 一键初始化
powershell .\setup.ps1
```

### 6.2 快速验证

```bash
# Mock API 功能测试
python scripts/testing/test_mock_api.py

# 端到端集成测试（parser→env→agent→annealing 全链路）
python scripts/testing/e2e_test.py

# 快速训练验证（5000步 DQN）
python scripts/training/quick_train.py
```

### 6.3 训练

```bash
# DQN 完整训练（10万步）
python scripts/training/train_agent.py --config config/config.yaml

# PPO 训练（主力算法）
python -c "from src.scheduler.env import QuantumSchedulingEnv; from src.scheduler.agent import PPOAgent; env=QuantumSchedulingEnv(max_qubits=20); agent=PPOAgent(env, learning_rate=3e-4, n_steps=2048, gamma=0.99); agent.train(total_timesteps=50000); agent.save('./models/ppo_model')"

# v6 算法深化训练
python scripts/train_marl.py --machines 3 --timesteps 50000              # MAPPO 多智能体
python scripts/train_lstm_agent.py --timesteps 50000                     # PPO+LSTM
python scripts/train_multi_objective.py --weights 1.0 0.5 0.5            # 多目标 RL
python scripts/train_with_annealing_loop.py --timesteps 50000            # 异步退火闭环
```

### 6.4 仿真对比

```bash
# 8 策略对比（Mock 模式）
python scripts/evaluation/run_simulation.py --mock-mode --num-tasks 200

# 超参数网格搜索
python scripts/evaluation/hyperparameter_search.py --timesteps 20000
```

### 6.5 消融实验

```bash
# 快速验证所有消融维度（dry-run）
python scripts/ablation_study.py --all --dry-run

# 指定维度 3 seed
python scripts/ablation_study.py --dim D1 D4 --seeds 3

# 退火消融（PPO vs PPO+退火, 5 seed）
python scripts/ablation_annealing.py

# 生成学术报告
python scripts/generate_ablation_report.py results/ablation_study_XXX.json
```

### 6.6 多机器调度与真机验证

```bash
# 纯仿真对比（单机 vs 多机）
python scripts/demo/demo_multi_machine.py --episodes 20

# 真机验证（5% 抽样提交天衍云）
python scripts/demo/demo_multi_machine.py --real --real-prob 0.05 --episodes 5
```

### 6.7 Web 界面

```bash
# 启动监控界面（默认 port 8000）
uvicorn src.visualization.app:app --reload --port 8000

# 或通过统一 CLI
python scripts/cli.py serve --port 8000
```

访问 `http://localhost:8000` 查看实时监控面板。

### 6.8 一键演示

```bash
# 完整流程：PPO 快速训练 → 8 策略仿真 → 生成报告 → 启动 Web
python scripts/demo/demo.py

# 跳过部分步骤
python scripts/demo/demo.py --skip-train --skip-simulation
```

### 6.9 代码质量检查

```bash
black src/ scripts/ tests/                 # 代码格式化
isort src/ scripts/ tests/                 # import 排序
ruff check src/ scripts/ tests/            # 代码检查（替代 flake8）
mypy src/                                  # 类型检查
bandit -r src/ -ll                         # 安全扫描
pre-commit run --all-files                 # 手动触发 pre-commit
```

### 6.10 关键接口速查

```python
# 任务解析
from src.scheduler.parser import TaskParser, Task, TaskBuilder
parser = TaskParser()
task = parser.parse({"task_id": "T1", "task_type": "quantum", "qubits_required": 5, ...})

# 调度环境（14维状态，3类动作）
from src.scheduler.env import QuantumSchedulingEnv
env = QuantumSchedulingEnv(max_qubits=20)
obs, _ = env.reset()
obs, reward, terminated, truncated, info = env.step(action)

# 多机器调度
from src.scheduler.env import QuantumSchedulingEnv, DEFAULT_MACHINE_CONFIGS
env = QuantumSchedulingEnv(machine_configs=DEFAULT_MACHINE_CONFIGS)  # 3 台真机
env.attach_real_clients({"tianyan_s": client})  # 可选：绑定真机

# PPO 训练（主力算法）
from src.scheduler.agent import PPOAgent
agent = PPOAgent(env, learning_rate=3e-4, n_steps=2048, gamma=0.99)
agent.train(total_timesteps=50000)
agent.model.save("./models/ppo_model")

# API 客户端（自动选择 Mock/真实）
from src.api import get_client
client = get_client(mock_mode=True)
task_id = client.submit_quantum_task(circuit_qasm=qasm, shots=1024)

# 量子退火
from src.quantum.annealing import QuantumAnnealingOptimizer
opt = QuantumAnnealingOptimizer(simulation_mode=True)
optimized_agent = opt.optimize_policy(agent, num_iterations=10)

# 工具函数
from src.utils import setup_logging, load_config, MetricsCalculator
setup_logging()
config = load_config()
```

---

## 7. 测试与验证

### 7.1 测试概览

测试目录共 **12 个测试文件**，**561 个测试用例**，覆盖率 **85.42%**。

| 测试文件 | 用例数 | 覆盖模块 |
|---------|-------|---------|
| [test_api.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/tests/test_api.py) | 136 | API 客户端（最大用例集，95.96% 覆盖） |
| [test_parser.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/tests/test_parser.py) | 107 | 任务解析器（100% 覆盖） |
| [test_scheduler.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/tests/test_scheduler.py) | 67 | 调度核心（含 11 个多机器用例） |
| [test_annealing.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/tests/test_annealing.py) | 62 | 量子退火 |
| [test_helpers.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/tests/test_helpers.py) | 63 | 工具函数 |
| [test_visualization.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/tests/test_visualization.py) | 42 | Web 可视化（90% 覆盖） |
| [test_multi_objective.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/tests/test_multi_objective.py) | 33 | 多目标奖励（v6） |
| [test_marl.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/tests/test_marl.py) | 18 | MAPPO 多智能体（v6） |
| [test_state_space.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/tests/test_state_space.py) | 14 | 状态空间（v6） |
| [test_annealing_loop.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/tests/test_annealing_loop.py) | 6 | 异步退火闭环（v6） |
| [test_property.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/tests/test_property.py) | 6 | 属性测试（hypothesis） |
| [benchmarks/test_annealing_benchmark.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/tests/benchmarks/test_annealing_benchmark.py) | 7 | 退火性能基准 |

### 7.2 运行测试

```bash
# 全量测试 + 覆盖率
pytest tests/ --cov=src

# 带标记运行
pytest tests/ -m "not slow"              # 跳过慢测试
pytest tests/ -m "real_machine"          # 仅真机测试
pytest tests/ -m "integration"           # 仅集成测试

# 类型检查（Windows 需设置 PYTHONUTF8=1）
PYTHONUTF8=1 python -m mypy src
```

### 7.3 pytest 标记

| 标记 | 用途 |
|------|------|
| `slow` | 慢测试（完整训练） |
| `real_machine` | 真机测试（需 API Key） |
| `integration` | 集成测试 |
| `unit` | 单元测试 |

### 7.4 性能基准

```bash
# 退火性能基准
pytest tests/benchmarks/test_annealing_benchmark.py --benchmark-only

# 8 策略对比（多 Seed 权威结果）
# 排名：PPO(2746.94) > SJF(1468.17) > FCFS(1458.77) > Random(1275.91)
#       > Greedy(-71.87) > DQN/Quantum-Only(-897.08) > Classical-Only(-1134.35)
```

---

## 8. 部署与容器化

### 8.1 Docker 部署

项目提供多阶段 Docker 构建（[Dockerfile](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/Dockerfile)）：

- **builder 阶段**：`python:3.11-slim`，安装依赖到 `/root/.local`
- **运行时阶段**：`python:3.11-slim`，仅复制依赖与代码，暴露 8000（FastAPI）+ 6006（TensorBoard）

```bash
# 一键部署
docker-compose up -d

# 仅启动 Web 服务（默认 profile）
docker-compose up web

# 启动 Web + TensorBoard（monitoring profile）
docker-compose --profile monitoring up

# 启动 Web + Redis（production profile）
docker-compose --profile production up
```

### 8.2 docker-compose 服务

| 服务 | 镜像 | 端口 | 资源限制 | profile |
|------|------|------|---------|---------|
| `web` | 本地 Dockerfile | 8000:8000 | 2 CPU / 4G 内存 | 默认 |
| `tensorboard` | tensorboard/tensorboard | 6006:6006 | — | monitoring |
| `redis` | redis:7-alpine | 6379:6379 | — | production |

### 8.3 健康检查

Docker 容器健康检查：`curl -f http://localhost:8000/api/status`，30 秒间隔。

### 8.4 配置文件

- [config/config.yaml](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/config/config.yaml)：系统主配置（mock_mode、调度参数、退火参数、天衍云配置）
- [config/.env.example](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/config/.env.example)：环境变量模板
- [pyproject.toml](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/pyproject.toml)：代码质量工具统一配置（Black/isort/ruff/mypy/pytest/coverage/bandit）

> **重要约束**：不要修改 `config/config.yaml` 的 `mock_mode: true`，除非获得天衍云平台权限。

---

## 附录：关键文件索引

| 模块 | 核心文件 | 行数 |
|------|---------|------|
| 调度环境 | [src/scheduler/env.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/scheduler/env.py) | 1398 |
| 任务解析 | [src/scheduler/parser.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/scheduler/parser.py) | 864 |
| RL 智能体 | [src/scheduler/agent.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/scheduler/agent.py) | 1261 |
| MAPPO | [src/scheduler/marl.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/scheduler/marl.py) | 1134 |
| 多目标 | [src/scheduler/multi_objective_env.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/scheduler/multi_objective_env.py) | 372 |
| 异步退火回调 | [src/scheduler/async_annealing_callback.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/scheduler/async_annealing_callback.py) | 132 |
| 天衍云客户端 | [src/api/tianyan_client.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/api/tianyan_client.py) | 789 |
| cqlib 真机 | [src/api/tianyan_cqlib.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/api/tianyan_cqlib.py) | 456 |
| Mock 客户端 | [src/api/mock_client.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/api/mock_client.py) | 602 |
| 量子退火 | [src/quantum/annealing.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/quantum/annealing.py) | 1286 |
| 退火闭环 | [src/quantum/annealing_loop.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/quantum/annealing_loop.py) | 343 |
| Web 后端 | [src/visualization/app.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/visualization/app.py) | 1534 |
| Web 前端 | [src/visualization/frontend/index.html](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/visualization/frontend/index.html) | 920 |
| 工具函数 | [src/utils/helpers.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/utils/helpers.py) | 305 |
| Prometheus | [src/utils/metrics.py](file:///c:/Users/HZR/Desktop/揭榜挂帅擂台赛/quantum-rl-scheduler/src/utils/metrics.py) | 111 |
