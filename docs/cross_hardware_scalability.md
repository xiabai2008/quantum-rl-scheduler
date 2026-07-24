# 跨硬件可扩展性论述

> 本文档作为答辩材料中的理论扩展性论证，说明当前调度框架的解耦设计及其向其他量子硬件后端扩展的理论路径。
> 关联 Issue: #28
> 最后更新：2026-07-24

## 1. 定位与范围

### 1.1 当前实现聚焦

本项目的所有真机实验均在**天衍云平台超导量子计算机**（天衍-287 / tianyan176）上完成。天衍云平台本身是超导架构，赛题目标平台明确，不存在离子阱/光量子后端的实际需求。

### 1.2 本文档目标

- 阐明调度框架的**后端解耦设计**，证明架构层面具备跨硬件扩展能力
- 给出向其他量子硬件（离子阱、光量子、中性原子）扩展的**理论接口路径**
- 明确区分**已实现功能**与**理论分析**，避免过度声称

## 2. 架构解耦设计

### 2.1 三层架构概览

```
┌─────────────────────────────────────────────────────┐
│              RL 调度引擎层 (scheduler/)              │
│   env.py / env_real_machine.py / agent.py           │
│   ── 仅依赖 QuantumMachine 数据类与协议方法 ──       │
├─────────────────────────────────────────────────────┤
│              API 抽象层 (api/)                       │
│   ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│   │TianyanClient │  │CqlibTianyan  │  │MockClient │ │
│   │(REST API)    │  │Client(cqlib) │  │(开发测试) │ │
│   └──────────────┘  └──────────────┘  └───────────┘ │
│   ── 三个实现共享同一方法签名（鸭子类型协议）──       │
├─────────────────────────────────────────────────────┤
│              量子硬件层 (外部)                       │
│   天衍-287 (287q 超导) / tianyan176 (176q 超导)     │
└─────────────────────────────────────────────────────┘
```

### 2.2 鸭子类型协议

三个客户端类（`TianyanClient`、`CqlibTianyanClient`、`MockTianyanClient`）未继承自统一抽象基类，但通过**鸭子类型协议**实现了接口一致性。核心协议方法如下：

| 方法 | 签名 | 职责 |
|------|------|------|
| `submit_quantum_task()` | `(qcis/circuit_qasm, shots, ...) -> str` | 提交量子任务，返回 task_id |
| `get_task_status()` | `(task_id) -> dict` | 非阻塞查询任务状态 |
| `get_task_result()` | `(task_id) -> dict` | 获取已完成任务的测量结果 |
| `wait_for_task()` | `(task_id, timeout, poll_interval) -> dict` | 阻塞等待任务完成 |

调度引擎（`env_real_machine.py`）通过 `env._real_clients[machine.name]` 字典持有客户端实例，调用时**不关心具体实现类**，仅依赖协议方法签名。这一设计是跨硬件扩展的核心基础。

### 2.3 QuantumMachine 数据类

`QuantumMachine`（`env_types.py`）是硬件资源状态的抽象表示，与具体硬件后端解耦：

```python
@dataclass
class QuantumMachine:
    name: str                    # 机器名称
    total_qubits: int            # 物理比特数
    available_ratio: float       # 可用比特比率
    fidelity: float              # 平均保真度
    supported_gates: tuple       # 支持的门集合
    is_real: bool                # 是否对接真机
    single_gate_fidelity: float  # 单比特门保真度
    two_gate_fidelity: float     # 两比特门保真度
    coupling_density: float      # 耦合图密度
    avg_connectivity: float      # 平均连通度
```

该数据类通过 `supported_gates` 和噪声特征字段（`single_gate_fidelity`、`two_gate_fidelity`、`coupling_density`）描述不同硬件的物理特性，使得调度策略能根据硬件特征做出差异化决策。

### 2.4 降级与熔断机制

`CircuitBreaker`（`circuit_breaker.py`）实现 CLOSED/OPEN/HALF_OPEN 三态转换，当某个硬件后端连续失败时自动隔离，不影响其他后端的调度。这一机制天然支持多硬件后端并存场景：某个硬件后端不可用时，调度器自动降级到可用后端。

## 3. 跨硬件扩展路径

### 3.1 扩展步骤

向新硬件后端（如离子阱、光量子）扩展的理论路径：

| 步骤 | 内容 | 工作量估计 |
|:--:|------|:--:|
| 1 | 实现新客户端类，遵循鸭子类型协议（submit/get_status/get_result） | 中 |
| 2 | 实现电路格式转换器（如 QCIS → 目标硬件原生格式） | 中-高 |
| 3 | 配置 `QuantumMachine` 实例的硬件特征参数（门集、保真度、拓扑） | 低 |
| 4 | 在 `env._real_clients` 中注册新客户端 | 低 |
| 5 | 验证熔断器与降级机制在多后端场景下的正确性 | 低 |

### 3.2 各硬件后端的适配分析

| 硬件类型 | 电路格式 | 门集差异 | 拓扑特征 | 适配难点 |
|:--:|:--:|:--:|:--:|------|
| **超导（当前）** | QCIS | H, X, Y, Z, RX, RY, RZ, CNOT, CZ | 二维网格耦合 | 已完成适配 |
| **离子阱** | QASM/OpenQASM | 全连通拓扑，Mølmer-Sørensen 门 | 全连通 | 优势：全连通简化调度；难点：MS 门的时间参数化 |
| **光量子** | Xanadu Borealis 格式 | 连续变量门，高斯操作 | 线性光学网络 | 难点：离散/连续变量映射，后处理概率采样 |
| **中性原子** | QuEra格式 | Rydberg blockade 门 | 可重构二维阵列 | 优势：可重构拓扑；难点：原子重排时间建模 |

### 3.3 电路格式转换层

当前 `generate_qcis_circuit()` 生成天衍云原生 QCIS 格式电路。跨硬件扩展需引入电路格式转换层：

```
任务参数 → generate_qcis_circuit() → QCIS电路
                                      ↓
                           ┌─────────┴─────────┐
                           ↓                   ↓
                    QCIS→QASM 转换器     QCIS→目标格式 转换器
                           ↓                   ↓
                    离子阱后端           其他后端
```

借助 Qiskit 的 Intermediate Representation (IR) 或 OpenQASM 3.0 作为中间格式，可实现跨硬件电路转换。当前项目已在 `requirements.txt` 中包含 Qiskit 依赖。

## 4. 调度策略的硬件无关性

### 4.1 状态空间设计

RL 环境的 14 维观测空间设计为硬件无关：

| 维度 | 含义 | 硬件无关性 |
|:--:|------|:--:|
| 队列长度 | 待调度任务数 | ✓ |
| 量子保真度 | 机器平均保真度 | ✓（通过 QuantumMachine.fidelity） |
| 等待时间 | 任务累积等待 | ✓ |
| 拓扑连接度 | 耦合图密度 | ✓（通过 QuantumMachine.coupling_density） |
| 单/双比特门保真度 | 物理噪声特征 | ✓（通过 QuantumMachine 字段） |

观测空间不包含任何硬件特定信息（如硬件型号、门名称），所有硬件特征通过 `QuantumMachine` 数据类抽象传入。

### 4.2 奖励函数

奖励函数基于任务完成度、资源利用率和量子保真度计算，不依赖具体硬件类型。真机测量结果通过 `compute_result_fidelity()` 计算保真度后映射为 reward，该函数接受概率分布字典作为输入，与硬件无关。

### 4.3 策略迁移

PPO 策略网络输入 14 维硬件无关观测，输出调度动作。在不同硬件后端上，只要 `QuantumMachine` 实例正确配置硬件特征参数，同一策略网络可直接迁移，无需重新训练。若硬件噪声特征差异显著（如保真度分布不同），可通过少量微调（fine-tuning）快速适配。

## 5. 现状与边界

### 5.1 已实现

- 超导量子后端（天衍-287/tianyan176）的完整适配
- 三客户端鸭子类型协议（Mock/REST/cqlib）
- 多机器调度（3台机器 MAPPO 协同，+86.3%）
- 熔断器与降级机制
- 硬件无关的 14 维观测空间与奖励函数
- 真机测量结果→保真度→reward 闭环

### 5.2 理论分析（未实现）

- 离子阱/光量子/中性原子后端的客户端实现
- QCIS→其他格式的电路转换器
- 多硬件异构后端并存的调度策略优化

### 5.3 诚实声明

本项目聚焦天衍云超导平台，跨硬件扩展为架构层面的理论论证，**未实际实现非超导硬件的适配代码**。框架的解耦设计确保了扩展的理论可行性，但实际适配工作量（尤其是电路格式转换）不可忽视。

## 6. 答辩问答指引

**Q: 你们的系统支持哪些量子硬件？**

A: 当前完整适配天衍云平台超导量子计算机（天衍-287，287量子比特）。天衍云平台本身是超导架构，赛题目标平台明确。

**Q: 如果要扩展到其他硬件呢？**

A: 框架采用三层解耦设计，调度引擎仅依赖鸭子类型协议（submit/get_status/get_result）和 QuantumMachine 数据类。扩展新硬件只需实现协议客户端 + 电路格式转换器，调度策略无需修改。14 维观测空间硬件无关，策略网络可直接迁移。

**Q: 跨硬件调度有什么挑战？**

A: 主要挑战在电路格式转换（不同硬件门集不同）和硬件噪声建模（保真度、拓扑、门时间差异）。调度框架本身已通过 QuantumMachine 数据类抽象了这些差异，难点在硬件特定的适配层实现。

## 7. 关联文档

- 架构设计：`src/api/`（三客户端实现）、`src/scheduler/env_types.py`（QuantumMachine）
- 真机闭环：`src/scheduler/env_real_machine.py`、`results/reports/real_machine_closed_loop.md`
- 真机性能：`results/reports/real_machine_performance.md`
- 技术白皮书：`../技术白皮书_量子RL调度系统_v5.docx`
