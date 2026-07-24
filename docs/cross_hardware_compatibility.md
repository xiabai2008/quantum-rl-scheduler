# 跨硬件兼容路线图：超导 / 离子阱 / 光量子

> Issue #100 — 比赛方案要求"兼容主流及新兴量子硬件技术路线（如超导、离子阱、光量子等）"
>
> 关联 Issue：#28（同一主题）
> 比赛方案引用：P11 L221（跨硬件要求）、P12 L290（光量子资源）
>
> 生成时间：2026-07-24

---

## 1. 硬件技术路线概览

| 维度 | 超导量子 | 离子阱 | 光量子 |
|------|---------|--------|--------|
| **物理载体** | 约瑟夫森结 | 囚禁离子 | 光子 |
| **典型比特数** | 50–1000+ | 10–50 | 50–200（模式数） |
| **门保真度** | 99.5%–99.9%（单比特） | 99.9%+（单比特） | 受光学损耗限制 |
| **相干时间** | 10–100 μs | 1–10 s | 由光路长度决定 |
| **工作温度** | ~15 mK（稀释制冷机） | 室温–液氦 | 室温 |
| **连线方式** | 2D/3D 耦合图 | 全连接 | 线性光路 / 簇态 |
| **编程模型** | 电路模型 | 电路模型 | 电路模型 / 高斯玻色采样 |
| **商业化程度** | 最高（IBM/Google/中电信） | 中（IonQ/Quantinuum/启科） | 中低（九章/Xanadu） |

---

## 2. 已支持硬件：超导量子（天衍云平台）

### 2.1 纳管真机清单

`CqlibTianyanClient.REAL_MACHINES` 当前纳管 **9 台** 超导量子计算机：

| 机器名 | 比特数 | 计费 | 用途 |
|--------|--------|------|------|
| `tianyan-287` | 105 (付费套餐) | 付费 | 深度实验 |
| `tianyan_sw` | — | 免费 | 轻调度备用 |
| `tianyan_s` | — | 免费 | 默认机器 |
| `tianyan_tn` | — | 免费 | 转调度备用 |
| `tianyan_tnn` | — | 免费 | 转调度备用 |
| `tianyan_swn` | — | 免费 | 转调度备用 |
| `tianyan_sa` | — | 免费 | 转调度备用 |
| `tianyan176` | 176 | 免费 | 转调度 / 深度实验回退 |
| `tianyan176-2` | 176 | 免费 | 转调度备用 |

> 比赛方提供"天衍-287"套餐，实际回退至 `tianyan176`。

### 2.2 已验证数据

| 指标 | 数值 | 来源 |
|------|------|------|
| 真机调用次数 | 284 | `results/reports/` |
| 成功率 | 100% | 真机验证报告 |
| 多 seed 真机 | 5 seeds × 3 策略 | `results/reports/multiseed_real_machine_report.md` |
| PPO vs FCFS (真机) | Cohen's d = 5.64 (大效应) | 同上 |
| 电路格式 | QCIS | `src/scheduler/env_real_machine.py` |

### 2.3 架构抽象层

```
┌─────────────────────────────────────────┐
│          调度器 (scheduler/env.py)        │
│   策略决策与资源分配，不直接操作硬件       │
└──────────────────┬──────────────────────┘
                   │ QuantumMachine 数据类
┌──────────────────▼──────────────────────┐
│      硬件抽象层 (env_types.py)            │
│  QuantumMachine: name/qubits/fidelity/  │
│  supported_gates/is_real/...            │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│      API 客户端层                         │
│  CqlibTianyanClient  (超导/cqlib)       │
│  MockClient           (开发测试)          │
│  TianyanClient        (REST/legacy)      │
└─────────────────────────────────────────┘
```

**关键设计**：调度器仅通过 `QuantumMachine` 数据类与硬件交互，不直接调用 API 客户端。客户端层负责将硬件特定协议（QCIS/cqlib）转换为统一接口。这使得扩展新硬件类型时**无需修改调度器核心逻辑**。

---

## 3. 可扩展硬件：光量子

### 3.1 赛源资源

比赛方案 P12 L290 明确提供 **"九章四号同款光量子原型机"** 资源。这是本项目跨硬件兼容的首要扩展目标。

### 3.2 光量子与超导的关键差异

| 差异点 | 超导 | 光量子 | 对调度器的影响 |
|--------|------|--------|--------------|
| 计算模型 | 通用电路 | 高斯玻色采样 (GBS) 为主 | 需支持非电路型任务 |
| 电路格式 | QCIS | GBS 参数（协方差矩阵/挤压参数） | API 客户端需新增提交协议 |
| 结果解读 | 比特串 | 光子数分布 | 任务状态轮询逻辑需适配 |
| 全连接性 | 受耦合图限制 | 天然全连接 | 调度器可简化拓扑约束 |
| 噪声模型 | 退极化/退相干 | 光子损耗 | 保真度指标含义不同 |

### 3.3 扩展路径

```
Phase 0 (当前):  超导-only，CqlibTianyanClient
    │
Phase 1 (设计):  抽象 HardwareBackend 基类
    │  - submit_task(circuit, shots, task_type) → task_id
    │  - poll_result(task_id) → result
    │  - query_status() → MachineStatus
    │  - supported_task_types() → ["circuit", "gbs", ...]
    │
Phase 2 (光量子):  实现 OpticalBackend(HardwareBackend)
    │  - 对接九章光量子 API
    │  - GBS 任务提交与结果轮询
    │  - 保真度映射：光子损耗率 → 0-1 指标
    │
Phase 3 (融合):  调度器感知硬件类型
    │  - QuantumMachine 增加 hardware_type 字段
    │  - 任务路由：电路任务→超导，GBS任务→光量子
    │  - 跨硬件协同：混合电路-GBS 工作流
    │
Phase 4 (验证):  光量子真机实验
       - 对接九章原型机
       - GBS 任务调度对比实验
```

### 3.4 预计工作量

| Phase | 内容 | 依赖 | 备注 |
|-------|------|------|------|
| Phase 1 | `HardwareBackend` 抽象基类 | 无 | 纯设计，不改动现有代码 |
| Phase 2 | `OpticalBackend` 实现 | 九章 API 文档 | 需比赛方提供 SDK/文档 |
| Phase 3 | 调度器硬件感知 | Phase 1 + 2 | QuantumMachine 扩展 |
| Phase 4 | 光量子真机验证 | Phase 2 + 九章接入 | 实验数据 |

---

## 4. 可扩展硬件：离子阱

### 4.1 技术特点

离子阱平台（如启科量子、IonQ、Quantinuum）具有以下差异化优势：

- **全连接拓扑**：任意两比特可直接交互，无需 SWAP 门插入
- **高保真度**：单比特门 > 99.9%，两比特门 > 99.5%
- **长相干时间**：秒级（比超导高 4–5 个数量级）
- **QASM 兼容**：多数平台支持 OpenQASM 2.0/3.0 输入

### 4.2 与超导的关键差异

| 差异点 | 超导 | 离子阱 | 对调度器的影响 |
|--------|------|--------|--------------|
| 拓扑 | 受耦合图限制 | 全连接 | 可省去 SWAP 路由优化 |
| 比特数 | 50–1000+ | 10–50 | 需处理容量约束 |
| 电路格式 | QCIS (cqlib) | QASM | API 客户端需新增协议 |
| 门集 | {H, CZ, M} | {H, CX, M, Rz} | 门兼容性检查需扩展 |
| 串扰 | 中等 | 低（串行门操作） | 噪声模型简化 |

### 4.3 扩展路径

```
Phase 1 (设计):  同光量子 Phase 1，共用 HardwareBackend 基类
    │
Phase 2 (实现):  IonTrapBackend(HardwareBackend)
    │  - 对接启科量子 / IonQ API
    │  - QCIS → QASM 电路转译层
    │  - 保真度映射：门误差率 → 0-1 指标
    │
Phase 3 (验证):  离子阱仿真/真机实验
       - 启科量子云 API 对接
       - 調度对比实验
```

> 注：离子阱不在本次比赛提供的硬件资源中，扩展优先级低于光量子。

---

## 5. 硬件抽象层架构设计

### 5.1 HardwareBackend 抽象基类

```python
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

class HardwareType(Enum):
    SUPERCONDUCTING = "superconducting"  # 超导
    ION_TRAP = "ion_trap"                # 离子阱
    PHOTONIC = "photonic"                # 光量子

class TaskType(Enum):
    CIRCUIT = "circuit"    # 通用电路
    GBS = "gbs"            # 高斯玻色采样
    ANNEALING = "annealing" # 退火

class MachineStatus(Enum):
    ONLINE = "online"
    CALIBRATING = "calibrating"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"

class HardwareBackend(ABC):
    """量子硬件后端抽象基类"""

    @property
    @abstractmethod
    def hardware_type(self) -> HardwareType:
        """硬件类型"""

    @abstractmethod
    def list_machines(self) -> list[dict[str, Any]]:
        """列出可用机器"""

    @abstractmethod
    def submit_task(
        self,
        circuit: str,
        shots: int,
        task_type: TaskType = TaskType.CIRCUIT,
        task_name: str = "",
    ) -> str | None:
        """提交任务，返回 task_id"""

    @abstractmethod
    def poll_result(self, task_id: str) -> dict[str, Any] | None:
        """轮询结果，未完成返回 None"""

    @abstractmethod
    def query_status(self, machine_name: str) -> MachineStatus:
        """查询机器状态"""

    @abstractmethod
    def supported_task_types(self) -> list[TaskType]:
        """支持的任务类型"""

    @abstractmethod
    def supported_gates(self) -> tuple[str, ...]:
        """支持的门集合"""
```

### 5.2 现有客户端映射

| 现有类 | 对应 HardwareBackend | hardware_type |
|--------|---------------------|---------------|
| `CqlibTianyanClient` | `SuperconductingBackend` | `SUPERCONDUCTING` |
| `MockClient` | `MockBackend` | `SUPERCONDUCTING`（模拟） |

### 5.3 QuantumMachine 扩展

```python
class QuantumMachine:
    # ... 现有字段 ...
    hardware_type: HardwareType = HardwareType.SUPERCONDUCTING  # 新增
    supported_task_types: tuple[TaskType, ...] = (TaskType.CIRCUIT,)  # 新增
```

---

## 6. 調度器跨硬件适配策略

### 6.1 任务路由

```
任务到达
    │
    ├── 电路任务 (task_type=CIRCUIT)
    │   ├── 超导可用且比特数足够 → 超导后端
    │   ├── 离子阱可用且比特数足够 → 离子阱后端
    │   └── 无可用后端 → 排队等待
    │
    ├── GBS 任务 (task_type=GBS)
    │   ├── 光量子可用 → 光量子后端
    │   └── 无光量子 → 仿真回退
    │
    └── 退火任务 (task_type=ANNEALING)
        └── dwave-neal 仿真退火
```

### 6.2 硬件感知观测空间

当前 14 维观测空间中，维度 10-13 为真机特供特征（噪声/拓扑），其语义随硬件类型变化：

| 维度 | 超导含义 | 光量子含义 | 离子阱含义 |
|------|---------|-----------|-----------|
| 10 | 单比特门保真度 | 挤压参数稳定性 | 单比特门保真度 |
| 11 | 两比特门保真度 | 干涉仪可见度 | 两比特门保真度 |
| 12 | 耦合图密度 | 光路连通度 | 1.0（全连接） |
| 13 | 平均连通度 | 模式匹配率 | 1.0（全连接） |

> 此设计使得观测空间维度不变，RL 策略网络无需修改即可适配不同硬件。

### 6.3 保真度统一度量

不同硬件的保真度指标含义不同，需统一为 0-1 标量：

| 硬件 | 保真度来源 | 映射函数 |
|------|-----------|---------|
| 超导 | 随机基准测试 (RB) | 直接使用 RB 保真度 |
| 光量子 | 光子损耗率 η | fidelity = η^N (N 为模式数) |
| 离子阱 | 门误差率 ε | fidelity = 1 - ε |

---

## 7. 验收标准对照

| Issue #100 验收项 | 状态 | 说明 |
|-------------------|------|------|
| `docs/cross_hardware_compatibility.md` 产出 | ✅ | 本文档 |
| 含超导实测数据 + 光量子扩展路径 | ✅ | §2.2 实测数据 + §3.3 扩展路径 |
| 架构图说明硬件抽象层 | ✅ | §2.3 + §5 架构设计 |

---

## 8. 实施优先级与时间线

| 优先级 | 事项 | 前置依赖 | 时间窗口 |
|--------|------|---------|---------|
| **P1** | 本文档完成 | 无 | ✅ 已完成 |
| **P1** | `HardwareBackend` 抽象基类设计与实现 | 本文档 | 代码冻结前（8/15） |
| **P2** | 光量子 API 调研与 `OpticalBackend` 设计 | 九章 API 文档 | 9/15 提交前 |
| **P2** | QuantumMachine 增加 `hardware_type` 字段 | HardwareBackend | 8/15 前 |
| **P3** | 光量子真机验证实验 | OpticalBackend + 九章接入 | 赛赛后 |

> 注：光量子扩展的实际推进需比赛方提供九章原型机的 API 接入文档与 SDK，当前路径为设计级就绪。

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解策略 |
|------|------|---------|
| 九章 API 文档未及时提供 | Phase 2 无法启动 | 先完成 Phase 1 抽象层，答辩中展示设计就绪 |
| 光量子 GBS 与电路模型计算范式不同 | 任务路由逻辑复杂化 | TaskType 枚举区分，调度器按 task_type 路由 |
| 离子阱平台 API 不统一 | 每家需单独适配 | 优先适配启科量子（国内），其余按需扩展 |
| 保真度指标跨硬件不可比 | 調度决策失准 | 统一度量映射（§6.3），并在答辩中说明假设 |

---

## 10. 与比赛方案的映射

| 比赛方案要求 | 本文档对应 | 状态 |
|-------------|-----------|------|
| P11 L221: "兼容主流及新兴量子硬件技术路线" | §1 概览 + §5 抽象层设计 | 架构就绪 |
| P12 L290: "九章四号同款光量子原型机" | §3 光量子扩展路径 | 设计级就绪，待 API |
| 天衍-287 超导真机 | §2 已支持 + 实测数据 | ✅ 已验证 |
