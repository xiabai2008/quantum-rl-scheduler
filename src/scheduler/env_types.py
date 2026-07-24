"""
量子任务调度环境的类型与常量定义模块
Types and Constants for Quantum-Classical Hybrid Task Scheduling Environment

本模块集中定义调度环境所用的：
    - 状态向量索引常量（OBS_*）
    - 动作常量（ACTION_*）
    - 奖励参数（REWARD_*）
    - 环境参数（MAX_* / QUBIT_* / INITIAL_*）
    - 真机闭环参数（REAL_SUBMIT_* / REAL_MACHINE_*）
    - 数据类：Task / QuantumResource / ClassicalResource / QuantumMachine
    - 默认多机器配置 DEFAULT_MACHINE_CONFIGS

该模块不依赖 env.py，避免循环导入；env_reward.py 与 env_real_machine.py
仅依赖本模块。
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 状态向量索引（扩展版：14维，包含物理噪声和拓扑特征）
# ---------------------------------------------------------------------------
OBS_QUBIT_AVAILABILITY = 0  # 当前可用量子比特比率
OBS_QUEUE_LENGTH = 1  # 任务队列长度（归一化）
OBS_AVG_WAIT_TIME = 2  # 平均等待时间（归一化）
OBS_FIDELITY = 3  # 量子比特平均保真度
OBS_CLASSICAL_LOAD = 4  # 经典计算资源负载
OBS_QUANTUM_QUEUE_RATIO = 5  # 量子专用队列占比
OBS_TIME_OF_DAY = 6  # 一天中的时间段（昼夜模拟）
OBS_URGENCY_LEVEL = 7  # 当前任务紧急程度
OBS_TASK_TYPE_QUANTUM = 8  # 当前任务是quantum类型
OBS_TASK_TYPE_CLASSICAL = 9  # 当前任务是classical类型
OBS_SINGLE_GATE_FIDELITY = 10  # 单比特门平均保真度（SPAM error 补数）
OBS_TWO_GATE_FIDELITY = 11  # 两比特门平均保真度（CZ门误差率补数）
OBS_COUPLING_DENSITY = 12  # 耦合图密度 = 实际连接数 / 全连接数
OBS_AVG_CONNECTIVITY = 13  # 量子比特平均连通度 = 平均连接数 / max_connections

OBS_DIM = 14  # 状态空间维度（从10扩展到14）

# ---------------------------------------------------------------------------
# 动作常量
# ---------------------------------------------------------------------------
ACTION_CLASSICAL = 0  # 分配到经典计算资源
ACTION_QUANTUM = 1  # 分配到量子计算资源
ACTION_HYBRID = 2  # 混合执行

# ---------------------------------------------------------------------------
# 奖励参数（修改后：增强正确执行的奖励）
# ---------------------------------------------------------------------------
REWARD_QUANTUM_BASE = 10.0  # 量子执行基础奖励（不变）
REWARD_CLASSICAL = 5.0  # 经典执行奖励（从3.0提升到5.0）
REWARD_HYBRID = 7.0  # 混合执行奖励（新增，介于经典和量子之间）
REWARD_WAIT_OVER_THRESHOLD = -0.1  # 等待超时惩罚（从-0.5降低到-0.1，减少惩罚强度）
REWARD_LOW_QUBIT_UTIL = -1.0  # 量子比特利用率惩罚（从-2.0降低到-1.0）
REWARD_MISMATCH = -2.0  # 错误分配惩罚（从-5.0降低到-2.0）
REWARD_SUCCESS_BONUS = 3.0  # 任务成功完成奖励（新增）
QUANTUM_SPEEDUP_RANGE = (2.0, 5.0)  # 量子加速比范围

# ---------------------------------------------------------------------------
# 环境参数
# ---------------------------------------------------------------------------
MAX_QUEUE_SIZE = 30  # 队列最大长度（用于归一化）
MAX_WAIT_STEPS = 50  # 最大等待步数（超过此阈值开始惩罚）
MAX_STEPS_DEFAULT = 500  # 默认最大步数（一个 episode）
QUBIT_UTIL_THRESHOLD = 0.3  # 量子比特利用率低阈值
INITIAL_QUEUE_RANGE = (5, 20)  # reset 时初始任务队列大小范围

# ---------------------------------------------------------------------------
# 真机闭环参数（Issue #64）
# ---------------------------------------------------------------------------
# 真机提交抽样概率（控制真机机时消耗：每个量子任务以此概率真正上真机）
REAL_SUBMIT_PROBABILITY_DEFAULT = 0.0
# 真机任务成功完成时的奖励加成（叠加到 step reward，status_only 模式使用）
REAL_MACHINE_SUCCESS_BONUS = 2.0
# 真机任务失败时的惩罚（叠加到 step reward）
REAL_MACHINE_FAIL_PENALTY = -1.0
# 连续失败次数达到阈值后自动降级到 Mock（避免持续消耗机时）
REAL_MACHINE_DEGRADE_FAIL_THRESHOLD = 3
# 单个真机任务结果轮询的最大次数（超过则视为超时失败）
REAL_MACHINE_MAX_POLL_STEPS = 20

# ---------------------------------------------------------------------------
# 真机结果反馈模式（Issue #235）
# ---------------------------------------------------------------------------
# status_only   : 仅使用 completed 状态给固定 bonus（旧行为，向后兼容）
# result_aware  : 解析真机测量分布，按解质量计算 reward（语义闭环）
# shuffled      : 打乱真机测量结果（消融对照组，检验是否只是噪声注入）
REAL_FEEDBACK_STATUS_ONLY = "status_only"
REAL_FEEDBACK_RESULT_AWARE = "result_aware"
REAL_FEEDBACK_SHUFFLED = "shuffled"
REAL_FEEDBACK_MODES = (
    REAL_FEEDBACK_STATUS_ONLY,
    REAL_FEEDBACK_RESULT_AWARE,
    REAL_FEEDBACK_SHUFFLED,
)

# result_aware 模式下的最大奖励上限（防止高保真度任务奖励爆炸）
REAL_RESULT_REWARD_MAX = 5.0
# result_aware 模式下的最小奖励下限（即使质量为 0 也给少量完成奖励）
REAL_RESULT_REWARD_MIN = 0.5


# ---------------------------------------------------------------------------
# 辅助数据结构
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """
    表示队列中的单个待调度任务。

    Attributes:
        task_id        : 唯一任务标识符
        task_type      : 任务类型，"quantum"（仅量子可执行）、"classical"（仅经典可执行）、"universal"（两者皆可）
        qubit_count    : 该任务所需的量子比特数
        wait_steps     : 该任务已在队列中等待的步数
        urgency        : 紧急程度 0-1，越高越紧急
        priority       : 优先级 1-5
        execution_time : 预估执行时间（步数），与任务规模正相关
        qcis           : QCIS 格式量子电路（仅量子任务，用于真机提交）
        tenant_id      : 租户 ID（多租户配额隔离，Issue #97）
        required_gates : 任务所需的量子门集合（如 ("H","CZ","M")），None 表示不限制
    """

    task_id: str
    task_type: str  # "quantum", "classical", "universal"
    qubit_count: int = 0
    wait_steps: int = 0
    urgency: float = 0.5
    priority: int = 3
    execution_time: int = 3
    qcis: str | None = None  # QCIS 格式电路，None 表示未生成
    tenant_id: str | None = None  # 租户 ID（多租户配额隔离，Issue #97）
    required_gates: tuple[str, ...] | None = None  # 任务所需的量子门集合，None 表示不限制


@dataclass
class QuantumResource:
    """
    量子计算资源状态。

    Attributes:
        total_qubits   : 物理机总量子比特数
        available_ratio: 当前可用的量子比特比率（0-1）
        fidelity        : 当前量子比特平均保真度（0-1）
        quantum_queue   : 量子专用队列中的任务数
    """

    total_qubits: int = 287
    available_ratio: float = 1.0
    fidelity: float = 0.98
    quantum_queue: int = 0


@dataclass
class ClassicalResource:
    """
    经典计算资源状态。

    Attributes:
        load  : 经典计算资源负载（0-1），1 表示满载
        queue : 经典资源队列中的任务数
    """

    load: float = 0.0
    queue: int = 0


@dataclass
class QuantumMachine:
    """
    单台量子计算机的资源状态（多机器调度扩展）。

    每台机器独立维护可用比特、保真度、队列与在线状态，
    并声明其支持的门集合，用于任务-机器兼容性匹配。

    Attributes:
        name            : 机器名称（如 "tianyan_s"）
        total_qubits    : 物理机总量子比特数
        available_ratio : 当前可用量子比特比率（0-1）
        fidelity        : 当前量子比特平均保真度（0-1）
        quantum_queue   : 该机器专属队列中的任务数
        available       : 是否在线可用（False 表示维护/校准中）
        supported_gates : 支持的门集合（如 ("H","CZ","M")）
        is_real         : 是否对接真机（True 时可走 cqlib 提交）
        single_gate_fidelity : 单比特门平均保真度（SPAM error 补数）
        two_gate_fidelity : 两比特门平均保真度（CZ门误差率补数）
        coupling_density : 耦合图密度 = 实际连接数 / 全连接数
        avg_connectivity : 量子比特平均连通度 = 平均连接数 / max_connections
    """

    name: str = "tianyan_s"
    total_qubits: int = 287
    available_ratio: float = 1.0
    fidelity: float = 0.98
    quantum_queue: int = 0
    available: bool = True
    supported_gates: tuple = ("H", "CZ", "M")
    is_real: bool = False
    # 物理噪声特征（阶段1）
    single_gate_fidelity: float = 0.99
    two_gate_fidelity: float = 0.95
    # 拓扑特征（阶段2）
    coupling_density: float = 0.5
    avg_connectivity: float = 0.5

    def update_noise_features(self, rng: np.random.Generator) -> None:
        """
        更新物理噪声特征（基于保真度噪声衰减模型）。

        单比特门保真度：fidelity * 0.99 + random(-0.02, 0)
        两比特门保真度：fidelity * 0.95 + random(-0.03, 0.01)

        Args:
            rng: NumPy 随机数生成器
        """
        # 单比特门保真度：基于 fidelity 的噪声衰减
        noise_single = rng.uniform(-0.02, 0.0)
        self.single_gate_fidelity = float(np.clip(self.fidelity * 0.99 + noise_single, 0.0, 1.0))

        # 两比特门保真度：基于 fidelity 的噪声衰减（两比特门误差更大）
        noise_two = rng.uniform(-0.03, 0.01)
        self.two_gate_fidelity = float(np.clip(self.fidelity * 0.95 + noise_two, 0.0, 1.0))

    def update_topology_features(self) -> None:
        """
        更新拓扑特征（基于 total_qubits 估算网格密度）。

        耦合图密度：小芯片密度高，大芯片密度低
        平均连通度：基于网格拓扑估算
        """
        # 耦合图密度：基于 total_qubits 的网格拓扑估算
        # 小芯片（<100 qubits）密度高（~0.7），大芯片（>200 qubits）密度低（~0.3）
        if self.total_qubits <= 100:
            base_density = 0.7
        elif self.total_qubits <= 200:
            base_density = 0.5
        else:
            base_density = 0.3
        self.coupling_density = float(np.clip(base_density, 0.0, 1.0))

        # 平均连通度：网格拓扑中每个比特平均连接数 / max_connections
        # 网格拓扑：内部节点连接数=4，边缘节点连接数=2或3
        # 估算平均连接数约为 2.5-3.5，max_connections=4
        avg_conn = 3.0 if self.total_qubits > 50 else 2.5
        max_conn = 4.0
        self.avg_connectivity = float(np.clip(avg_conn / max_conn, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 默认多机器配置（基于天衍云真实超导机器列表）
# ---------------------------------------------------------------------------
# is_real=False 表示仅仿真，不消耗真机机时；True 时需 attach 真实客户端
DEFAULT_MACHINE_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "tianyan_s",
        "total_qubits": 287,
        "supported_gates": ("H", "CZ", "M"),
        "is_real": False,
    },
    {
        "name": "tianyan_sw",
        "total_qubits": 72,
        "supported_gates": ("H", "CZ", "M", "X", "Y"),
        "is_real": False,
    },
    {
        "name": "tianyan_tn",
        "total_qubits": 176,
        "supported_gates": ("H", "CZ", "M", "RX", "RY", "RZ"),
        "is_real": False,
    },
]
