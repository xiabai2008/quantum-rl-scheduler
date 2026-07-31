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

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 状态向量索引（扩展版：16维，包含物理噪声和拓扑特征）
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
OBS_CROSSTALK_RISK = 14  # 串扰风险（基于空间并发的任务密度）
OBS_ARRIVAL_RATE_MA = 15  # 任务到达率滑动平均（流量突发感知）

OBS_DIM = 16  # 状态空间维度（从15扩展到16）

# Issue #588: 公平性观测（可选第17维，不影响默认 OBS_DIM=16）
OBS_FAIRNESS_INDEX = 16  # 索引16，仅在 include_fairness_obs=True 时使用
OBS_DIM_WITH_FAIRNESS = 17

# ---------------------------------------------------------------------------
# 动作常量
# ---------------------------------------------------------------------------
ACTION_CLASSICAL = 0  # 分配到经典计算资源
ACTION_QUANTUM = 1  # 分配到量子计算资源
ACTION_HYBRID = 2  # 混合执行
ACTION_QUANTUM_QEM = 3  # 使用误差缓释（QEM）的量子执行

# 动作空间维度（动作枚举 0..3 共 4 个，统一引用避免硬编码不一致，Issue #787）
N_ACTIONS = 4

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

# Issue #587: 公平性惩罚参数
FAIRNESS_PENALTY_THRESHOLD = 0.3
FAIRNESS_PENALTY_FACTOR = 2.0

# Issue #801: 奖励函数魔法数字提取为命名常量（数值未变，仅集中管理）
CROSSTALK_PENALTY_FACTOR = 2.0
REQUEUE_PENALTY_FACTOR = 0.5
FIDELITY_NORM_REF = 0.99
LOW_FIDELITY_THRESHOLD = 0.9
LOW_FIDELITY_DISCOUNT = 0.6
QEM_TIME_PENALTY_FACTOR = 3.0

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
# Issue #576: 从 0.0 提升至 0.15，确保真机在训练中有足够参与率（>=15%）
REAL_SUBMIT_PROBABILITY_DEFAULT = 0.15
# 真机提交间隔步数（Issue #243：间隔触发保底，每N步强制提交一次）
# 确保 probability-only 触发不会因路由机会少而完全错过真机参与
REAL_MACHINE_SUBMIT_INTERVAL = 20
# 真机任务成功完成时的奖励加成（叠加到 step reward，status_only 模式使用）
REAL_MACHINE_SUCCESS_BONUS = 2.0
# 真机任务失败时的惩罚（叠加到 step reward）
REAL_MACHINE_FAIL_PENALTY = -1.0
# 连续失败次数达到阈值后自动降级到 Mock（避免持续消耗机时）
REAL_MACHINE_DEGRADE_FAIL_THRESHOLD = 3
# 单个真机任务结果轮询的最大次数（超过则视为超时失败）
REAL_MACHINE_MAX_POLL_STEPS = 20
# 每步最多轮询的 pending 任务数（Issue #524：避免 step() 中同步阻塞）
# 默认值 1 确保每个 step() 最多发起 1 次网络请求，将阻塞时间控制在单次请求延迟内
REAL_MACHINE_MAX_POLL_PER_STEP_DEFAULT = 1

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
REAL_RESULT_REWARD_MIN = 0.0  # 测量解析失败时给 0 奖励（不鼓励失败）

# Issue #576: 真机提交训练级硬上限默认值（配合 200 步训练约 4-5 次真机提交）
REAL_MACHINE_MAX_SUBMISSIONS_DEFAULT = 30

# Issue #577: 噪声感知奖励整形默认参数
NOISE_AWARE_REWARD_ENABLED_DEFAULT = True
NOISE_AWARE_PENALTY_THRESHOLD = 0.9
NOISE_AWARE_BONUS_THRESHOLD = 0.95
NOISE_AWARE_PENALTY_STEPS = 5
NOISE_AWARE_DECAY_FACTOR = 0.7
NOISE_AWARE_PENALTY_STRENGTH = 2.0
NOISE_AWARE_BONUS_STRENGTH = 0.5
# 向后兼容别名（拼写修正：NOISE_AWARD → NOISE_AWARE，Issue #657）
NOISE_AWARD_PENALTY_STRENGTH = NOISE_AWARE_PENALTY_STRENGTH
NOISE_AWARD_BONUS_STRENGTH = NOISE_AWARE_BONUS_STRENGTH


# ---------------------------------------------------------------------------
# 辅助数据结构
# ---------------------------------------------------------------------------


@dataclass
class RealMachineConfig:
    """真机闭环参数统一配置（Issue #576）。

    集中管理真机提交、轮询、降级与奖励相关的全部参数，
    替代分散在模块级常量和环境构造函数中的零散配置。

    使用方式::

        config = RealMachineConfig(submit_probability=0.2)
        env = QuantumSchedulingEnv(
            real_submit_probability=config.submit_probability,
            max_real_submissions=config.max_submissions,
        )

    Attributes:
        submit_probability     : 真机提交抽样概率（0-1），控制机时消耗
        submit_interval        : 间隔触发步数（保底机制，每N步强制提交一次）
        max_submissions        : 训练级真机提交硬上限（跨 episode 累积）
        degrade_fail_threshold : 连续失败次数阈值，达到后自动降级到 Mock
        success_bonus          : 真机任务成功完成时的奖励加成
        fail_penalty           : 真机任务失败时的惩罚
        max_poll_steps         : 单任务结果轮询最大次数（超时视为失败）
        max_poll_per_step      : 每步最多轮询的 pending 任务数（Issue #524，避免阻塞）
        result_reward_max      : result_aware 模式最大奖励上限
        result_reward_min      : result_aware 模式最小奖励下限
        noise_aware_reward     : 是否启用噪声感知奖励整形（Issue #577）
        noise_penalty_threshold: 低保真度惩罚阈值（低于此值施加惩罚）
        noise_bonus_threshold  : 高保真度奖励阈值（高于此值给与加成）
        noise_penalty_steps    : 惩罚/加成持续步数
        noise_decay_factor     : 指数衰减因子（每步乘以该因子）
        noise_penalty_strength : 惩罚强度系数
        noise_bonus_strength   : 奖励加成强度系数
    """

    submit_probability: float = REAL_SUBMIT_PROBABILITY_DEFAULT
    submit_interval: int = REAL_MACHINE_SUBMIT_INTERVAL
    max_submissions: int = REAL_MACHINE_MAX_SUBMISSIONS_DEFAULT
    degrade_fail_threshold: int = REAL_MACHINE_DEGRADE_FAIL_THRESHOLD
    success_bonus: float = 5.0
    fail_penalty: float = REAL_MACHINE_FAIL_PENALTY
    max_poll_steps: int = REAL_MACHINE_MAX_POLL_STEPS
    max_poll_per_step: int = REAL_MACHINE_MAX_POLL_PER_STEP_DEFAULT
    result_reward_max: float = REAL_RESULT_REWARD_MAX
    result_reward_min: float = REAL_RESULT_REWARD_MIN
    noise_aware_reward: bool = NOISE_AWARE_REWARD_ENABLED_DEFAULT
    noise_penalty_threshold: float = NOISE_AWARE_PENALTY_THRESHOLD
    noise_bonus_threshold: float = NOISE_AWARE_BONUS_THRESHOLD
    noise_penalty_steps: int = NOISE_AWARE_PENALTY_STEPS
    noise_decay_factor: float = NOISE_AWARE_DECAY_FACTOR
    noise_penalty_strength: float = NOISE_AWARE_PENALTY_STRENGTH
    noise_bonus_strength: float = NOISE_AWARE_BONUS_STRENGTH

    def __post_init__(self) -> None:
        """参数校验。"""
        if not 0.0 <= self.submit_probability <= 1.0:
            raise ValueError(f"submit_probability must be in [0, 1], got {self.submit_probability}")
        if self.submit_interval < 1:
            raise ValueError(f"submit_interval must be >= 1, got {self.submit_interval}")
        if self.max_submissions < 0:
            raise ValueError(f"max_submissions must be non-negative, got {self.max_submissions}")
        if self.degrade_fail_threshold < 1:
            raise ValueError(
                f"degrade_fail_threshold must be >= 1, got {self.degrade_fail_threshold}"
            )
        if self.max_poll_steps < 1:
            raise ValueError(f"max_poll_steps must be >= 1, got {self.max_poll_steps}")
        if self.max_poll_per_step < 1:
            raise ValueError(f"max_poll_per_step must be >= 1, got {self.max_poll_per_step}")
        if not 0.0 <= self.noise_penalty_threshold <= 1.0:
            raise ValueError(
                f"noise_penalty_threshold must be in [0, 1], got {self.noise_penalty_threshold}"
            )
        if not 0.0 <= self.noise_bonus_threshold <= 1.0:
            raise ValueError(
                f"noise_bonus_threshold must be in [0, 1], got {self.noise_bonus_threshold}"
            )
        if self.noise_penalty_threshold > self.noise_bonus_threshold:
            raise ValueError(
                f"noise_penalty_threshold ({self.noise_penalty_threshold}) must be <= "
                f"noise_bonus_threshold ({self.noise_bonus_threshold})"
            )
        if self.noise_penalty_steps < 1:
            raise ValueError(f"noise_penalty_steps must be >= 1, got {self.noise_penalty_steps}")
        if not 0.0 < self.noise_decay_factor <= 1.0:
            raise ValueError(f"noise_decay_factor must be in (0, 1], got {self.noise_decay_factor}")
        if self.noise_penalty_strength < 0:
            raise ValueError(
                f"noise_penalty_strength must be non-negative, got {self.noise_penalty_strength}"
            )
        if self.noise_bonus_strength < 0:
            raise ValueError(
                f"noise_bonus_strength must be non-negative, got {self.noise_bonus_strength}"
            )


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
        readout_error   : 读出错误率（0-1）
        coupling_density : 耦合图密度 = 实际连接数 / 全连接数
        avg_connectivity : 量子比特平均连通度 = 平均连接数 / max_connections
        active_tasks    : 当前正在该机器上并行执行的任务列表
        used_qubits     : 当前已占用的量子比特数
        _noise_profile  : 注入的真机噪声画像（Issue #591），None 表示使用随机生成
    """

    name: str = "tianyan_s"
    total_qubits: int = 287
    available_ratio: float = 1.0
    fidelity: float = 0.98
    quantum_queue: int = 0
    available: bool = True
    supported_gates: tuple[str, ...] = ("H", "CZ", "M")
    is_real: bool = False
    # 物理噪声特征（阶段1）
    single_gate_fidelity: float = 0.99
    two_gate_fidelity: float = 0.95
    readout_error: float = 0.015
    # 拓扑特征（阶段2）
    coupling_density: float = 0.5
    avg_connectivity: float = 0.5
    active_tasks: list = field(default_factory=list)
    used_qubits: int = 0
    _noise_profile: dict[str, Any] | None = field(default=None, repr=False)

    def inject_noise_profile(self, profile: dict[str, Any]) -> None:
        """注入来自 NoiseModelExtractor 的真机噪声画像（Issue #591）。

        接收 ``NoiseModelExtractor.extract_noise_profile()`` 返回的噪声字典，
        从中提取平均保真度和读出错误率，用于驱动仿真环境的噪声特征。

        Args:
            profile: NoiseModelExtractor 返回的噪声画像字典，结构为::

                {
                    "readout_error": {"Q0": 0.012, ...},
                    "gate_error": {"Q0_H": 0.0008, "Q0_Q1_CZ": 0.012, ...},
                    "t1_time": {"Q0": 45.2, ...},
                    "metadata": {"source": "real"|"mock", ...}
                }
        """
        self._noise_profile = profile

        gate_errors = profile.get("gate_error", {})
        readout_errors = profile.get("readout_error", {})

        single_gate_errors: list[float] = []
        two_gate_errors: list[float] = []
        for key, err in gate_errors.items():
            if key.count("_") == 1:
                single_gate_errors.append(float(err))
            elif key.count("_") == 2:
                two_gate_errors.append(float(err))

        if single_gate_errors:
            avg_single_err = float(np.mean(single_gate_errors))
            self.single_gate_fidelity = float(np.clip(1.0 - avg_single_err, 0.0, 1.0))

        if two_gate_errors:
            avg_two_err = float(np.mean(two_gate_errors))
            self.two_gate_fidelity = float(np.clip(1.0 - avg_two_err, 0.0, 1.0))

        if readout_errors:
            self.readout_error = float(np.mean(list(readout_errors.values())))

        if single_gate_errors and two_gate_errors:
            avg_total_err = float(np.mean(single_gate_errors + two_gate_errors))
            self.fidelity = float(np.clip(1.0 - avg_total_err, 0.0, 1.0))

    def update_noise_features(self, rng: np.random.Generator) -> None:
        """
        更新物理噪声特征。

        当已通过 ``inject_noise_profile()`` 注入真机噪声画像时，
        基于注入值添加小幅随机扰动（模拟真实噪声漂移）；
        否则使用原有随机生成逻辑（Mock 模式，保持向后兼容）。

        Args:
            rng: NumPy 随机数生成器
        """
        if self._noise_profile is not None:
            noise_single = rng.uniform(-0.005, 0.005)
            self.single_gate_fidelity = float(
                np.clip(self.single_gate_fidelity + noise_single, 0.0, 1.0)
            )
            noise_two = rng.uniform(-0.008, 0.008)
            self.two_gate_fidelity = float(np.clip(self.two_gate_fidelity + noise_two, 0.0, 1.0))
            noise_readout = rng.uniform(-0.003, 0.003)
            self.readout_error = float(np.clip(self.readout_error + noise_readout, 0.0, 1.0))
            noise_fid = rng.uniform(-0.003, 0.003)
            self.fidelity = float(np.clip(self.fidelity + noise_fid, 0.0, 1.0))
        else:
            noise_single = rng.uniform(-0.02, 0.0)
            self.single_gate_fidelity = float(
                np.clip(self.fidelity * 0.99 + noise_single, 0.0, 1.0)
            )

            noise_two = rng.uniform(-0.03, 0.01)
            self.two_gate_fidelity = float(np.clip(self.fidelity * 0.95 + noise_two, 0.0, 1.0))

            self.readout_error = float(
                np.clip((1.0 - self.fidelity) * 0.5 + rng.uniform(-0.005, 0.01), 0.0, 1.0)
            )

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
