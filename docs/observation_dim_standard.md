# 观测维度口径管理标准

> **Issue #129 / #404** | 最后更新: 2026-07-28
> **适用范围**: 全项目脚本、文档、报告、答辩材料

---

## 一、观测维度定义

### 1.1 当前标准维度

| 维度 | 定义 | 实现方式 | 观测空间形状 | 状态 |
|:--|:--|:--|:--|:--|
| **16维（原生·当前标准）** | 完整状态空间，含物理噪声、拓扑特征、串扰风险、到达率MA | `QuantumSchedulingEnv`（`src/scheduler/env.py`） | `Box(0, 1, (16,))` | ✅ **交付标准** |
| **16维（截断·历史）** | 前14个维度，不含串扰风险和到达率MA | 旧版本兼容 | `Box(0, 1, (14,))` | ⚠️ 编译Agent使用；调度层已废弃 |
| **10维（截断·历史）** | 前10个维度，不含物理噪声和拓扑特征 | `Obs10Wrapper` | `Box(0, 1, (10,))` | ⚠️ 历史基线对比 |

### 1.2 16维观测空间详细定义（调度层当前标准）

**定义文件**：`src/scheduler/env_types.py`（`OBS_DIM = 16`）

| 索引 | 常量名 | 含义 | 类别 |
|:--|:--|:--|:--|
| 0 | OBS_QUBIT_AVAILABILITY | 量子比特可用率 | 基础 |
| 1 | OBS_QUEUE_LENGTH | 队列长度 | 基础 |
| 2 | OBS_AVG_WAIT_TIME | 平均等待时间 | 基础 |
| 3 | OBS_FIDELITY | 量子保真度 | 基础 |
| 4 | OBS_CLASSICAL_LOAD | 经典负载 | 基础 |
| 5 | OBS_QUANTUM_QUEUE_RATIO | 量子队列占比 | 基础 |
| 6 | OBS_TIME_OF_DAY | 时段 | 基础 |
| 7 | OBS_URGENCY_LEVEL | 紧急程度 | 基础 |
| 8 | OBS_TASK_TYPE_QUANTUM | 量子任务类型 | 基础 |
| 9 | OBS_TASK_TYPE_CLASSICAL | 经典任务类型 | 基础 |
| 10 | OBS_SINGLE_GATE_FIDELITY | 单比特门保真度 | 物理噪声 |
| 11 | OBS_TWO_GATE_FIDELITY | 双比特门保真度 | 物理噪声 |
| 12 | OBS_COUPLING_DENSITY | 耦合密度 | 拓扑特征 |
| 13 | OBS_AVG_CONNECTIVITY | 平均连接度 | 拓扑特征 |
| 14 | OBS_CROSSTALK_RISK | 串扰风险（基于空间并发） | 并发特征（v9新增） |
| 15 | OBS_ARRIVAL_RATE_MA | 任务到达率滑动平均 | 时序特征（v9新增） |

### 1.3 14维观测空间（编译层专用）

编译层PPO Agent（`ppo_compilation_agent.zip`）使用独立的14维观测空间，定义在`src/quantum/compilation_env.py`：
- 包含：电路特征（深度、两比特门比例）、当前映射状态、耦合图拓扑特征、SWAP候选动作评估
- 这与调度层的16维观测空间完全不同，是量子比特映射任务专用

### 1.4 耦合图拓扑（v9更新）

**定义文件**：`src/quantum/compilation_env.py`

- v8及以前：线性链耦合图（qubit 0-1-2-...-15），SWAP距离=线性abs差值
- v9起：**4×4 2D网格耦合图**（匹配天衍-287真机nearest-neighbor拓扑）
  - SWAP距离=BFS图最短路径
  - 2D网格直径=6（vs 线性链直径=15）
  - 拓扑消融：SABRE在2D网格上SWAP比线性链少~62%

---

## 二、适用场景规范

### 2.1 口径使用规则

| 场景 | 使用维度 | 模型类型 | 理由 |
|:--|:--|:--|:--|
| **官方仿真评估（交付）** | 16维（原生） | PPO-MLP | `run_simulation.py` 默认加载 ppo_best_model_16dim.zip |
| **PPO 训练与评估** | 16维（原生） | PPO-MLP/LSTM | 使用完整状态信息，最大化策略性能 |
| **真机实验** | 16维（原生） | PPO-MLP | 真实环境使用完整观测 |
| **消融实验 D2** | 10维→16维对比 | PPO-MLP | 测试维度扩展效果 |
| **答辩/PPT/白皮书** | 16维为主口径 | PPO-MLP | 16维PPO-MLP为最终提交版本 |
| **电路编译评估** | 编译层14维（独立任务专用） | PPO编译Agent | `compilation_fair_v2.py`独立环境，`compilation_env.py` shape=(14,) |
| **历史基线对比** | 10维/16维（调度层） | 旧模型（已归档） | 仅用于消融实验参考 |

### 2.2 模型与维度对应关系

| 模型 | 文件 | 观测维度 | 架构 | 用途 |
|:--|:--|:--|:--|:--|
| **PPO 权威交付模型** | `deliverable_models/ppo_best_model_16dim.zip` | 16维（调度层） | MLP (use_lstm=False) | ✅ 答辩/提交/官方评估 |
| PPO编译优化Agent | `deliverable_models/ppo_compilation_agent.zip` | 14维（编译层专用） | MLP | 量子比特映射（公平对比v2） |
| PPO-LSTM（消融用） | `models/` 目录（不入库） | 16维 | LSTM | 消融实验参考，非交付 |
| DQN 10维（旧·已删除） | — | 10维 | MLP | 历史归档，已清理 |

### 2.3 口径切换不可比性声明

**核心原则**：不同维度/不同环境的实验结果**不可直接比较**。

| 声明项 | 要求 |
|:--|:--|
| 报告标题 | 必须标注观测维度（如"16维 PPO-MLP vs FCFS"） |
| 数据表格 | 必须在表头或脚注标注维度 |
| 跨维度引用 | 必须注明"不同维度结果不可直接比较" |
| 调度vs编译 | 调度层和编译层是独立任务，指标不可跨任务比较 |
| 答辩口径 | +123.4% 基于 16 维交付模型 N=250 实验（config/statistics.yaml，v9.1+已验证）；14维旧模型（已删除）历史参考 |

---

## 三、模型架构说明

### 3.1 为什么交付模型使用MLP而非LSTM

消融实验验证：在本调度任务上，**MLP与LSTM收敛到相同策略**，MLP训练效率更高且兼容性更好。

选择MLP的理由：
1. **兼容性**：标准`PPO.load()`可直接加载，无需RecurrentPPO依赖
2. **效率**：训练更快，推理无需维护LSTM隐状态
3. **性能等价**：MLP与LSTM收敛到同一最优策略

---

## 四、数据完整性声明

### 权威数字一致性（v9）

| 指标 | 值 | 维度/环境 | 来源 |
|:--|:--|:--|:--|
| 仿真 PPO 均值 | 2348.91 ± 857.25 | **16维交付模型（新奖励参数）** | 50 seeds × 5 episodes = N=250 |
| 仿真 FCFS 均值 | 1051.59 ± 58.34 | **16维交付模型（新奖励参数）** | 同上 |
| 仿真 PPO 提升 | +123.4% | **16维交付模型（新奖励参数）** | 同上（Welch t, p=1.449e-66） |
| 编译PPO vs SABRE | 深电路SWAP减少~33%(n=20, 整体p=0.86不显著) | 编译层14维, 4×4 2D网格 | 公平对比v2, 60电路同池配对, p=0.86不显著 |
| 2D网格拓扑优势 | SWAP减少~62% vs 线性链 | 编译环境 | 拓扑消融（BFS验证） |
| 交付模型 | `ppo_best_model_16dim.zip` | 16维调度 | PPO-MLP, 100K steps |

> **重要声明（Issue #559/#530，v9.1+更新）**：+123.4%权威数字基于**16维交付模型 + 新奖励参数**的50seed实验（N=250, config/statistics.yaml），
> 对应模型为 `ppo_best_model_16dim.zip`。
> 14维旧模型（`ppo_best_model_14dim.zip`，已删除）使用旧奖励参数，已被v9.1+ 16维交付模型替代。
> v9.1+已完成N=250多seed评估验证，+123.4%为16维交付模型权威对比结果。
> 编译层PPO使用独立的14维观测空间（compilation_env.py, shape=(14,)），与调度层16维观测维度不可混淆。

---

## 五、关联文档

| 文档 | 路径 | 说明 |
|:--|:--|:--|
| 环境类型定义 | `src/scheduler/env_types.py` | OBS_DIM=16（调度层） |
| 观测构建 | `src/scheduler/env_observation.py` | 16维观测实现 |
| 2D网格耦合图 | `src/quantum/compilation_env.py` | 4×4网格+BFS距离 |
| 编译环境 | `src/quantum/compilation_env.py` | 编译层14维观测 |
| 权威评估脚本 | `scripts/evaluation/run_simulation.py` | 官方仿真入口 |
| 公平对比v2 | `scripts/evaluation/compilation_fair_v2.py` | PPO vs SABRE公平评估 |
| 模型登记 | `MODELS.md` | 模型维度与架构记录 |
| 训练脚本 | `scripts/training/train_16dim_ppo.py` | 16维MLP训练 |
| 消融实验 | `scripts/evaluation/ablation_ppo_variants.py` | MLP vs LSTM对比 |
| 新颖性声明 | `docs/novelty_statement.md` | 创新点口径 |

---

*Issue #129/#404 验收文件 | 2026-07-28 v9*
