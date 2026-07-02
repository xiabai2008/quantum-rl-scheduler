"""
量子任务调度环境模块
Quantum-Classical Hybrid Task Scheduling Environment for RL Training

基于 Gymnasium 框架构建的量子-经典混合计算调度环境。
RL Agent 需要决定每个任务是在经典计算资源上执行，还是在量子计算资源上执行，
或者采用量子-经典混合协同执行方式。

状态空间（10维，float32）：
    - qubit_availability : 当前可用量子比特比率（0-1）
    - queue_length       : 当前任务队列长度（归一化到 0-1）
    - avg_wait_time      : 队列中任务平均等待时间（归一化）
    - fidelity           : 当前量子比特平均保真度（0-1）
    - classical_load     : 经典计算资源负载（0-1）
    - quantum_queue_ratio: 量子专用队列占比（0-1）
    - time_of_day        : 一天中的时间段（0-1，模拟昼夜负载差异）
    - urgency_level      : 当前任务的紧急程度（0-1）
    - task_type_quantum  : 当前任务是否为 quantum 类型（0-1）
    - task_type_classical: 当前任务是否为 classical 类型（0-1）

动作空间（Discrete(3)）：
    - 0 : 分配到经典计算资源
    - 1 : 分配到量子计算资源
    - 2 : 混合执行（量子-经典协同）

奖励函数设计：
    - 任务在量子资源上成功执行     : +10 * 量子加速比
    - 任务在经典资源上执行          : +3（基准奖励）
    - 任务等待超过阈值             : -0.5 * 超时比例
    - 量子比特利用率低（<30%）     : -2（惩罚资源浪费）
    - 任务被错误分配到不兼容资源    : -5（大惩罚）
"""

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 状态向量索引（扩展版：14维，包含物理噪声和拓扑特征）
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

# 动作常量
ACTION_CLASSICAL = 0  # 分配到经典计算资源
ACTION_QUANTUM = 1  # 分配到量子计算资源
ACTION_HYBRID = 2  # 混合执行

# 奖励参数（修改后：增强正确执行的奖励）
REWARD_QUANTUM_BASE = 10.0  # 量子执行基础奖励（不变）
REWARD_CLASSICAL = 5.0  # 经典执行奖励（从3.0提升到5.0）
REWARD_HYBRID = 7.0  # 混合执行奖励（新增，介于经典和量子之间）
REWARD_WAIT_OVER_THRESHOLD = -0.1  # 等待超时惩罚（从-0.5降低到-0.1，减少惩罚强度）
REWARD_LOW_QUBIT_UTIL = -1.0  # 量子比特利用率惩罚（从-2.0降低到-1.0）
REWARD_MISMATCH = -2.0  # 错误分配惩罚（从-5.0降低到-2.0）
REWARD_SUCCESS_BONUS = 3.0  # 任务成功完成奖励（新增）
QUANTUM_SPEEDUP_RANGE = (2.0, 5.0)  # 量子加速比范围

# 环境参数
MAX_QUEUE_SIZE = 30  # 队列最大长度（用于归一化）
MAX_WAIT_STEPS = 50  # 最大等待步数（超过此阈值开始惩罚）
MAX_STEPS_DEFAULT = 500  # 默认最大步数（一个 episode）
QUBIT_UTIL_THRESHOLD = 0.3  # 量子比特利用率低阈值
INITIAL_QUEUE_RANGE = (5, 20)  # reset 时初始任务队列大小范围


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
    """

    task_id: str
    task_type: str  # "quantum", "classical", "universal"
    qubit_count: int = 0
    wait_steps: int = 0
    urgency: float = 0.5
    priority: int = 3
    execution_time: int = 3


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


# 默认多机器配置（基于天衍云真实超导机器列表）
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

# 真机提交抽样概率（控制真机机时消耗：每个量子任务以此概率真正上真机）
REAL_SUBMIT_PROBABILITY_DEFAULT = 0.0


# ---------------------------------------------------------------------------
# 核心环境类
# ---------------------------------------------------------------------------


class QuantumSchedulingEnv(gym.Env):
    """
    量子-经典混合计算调度环境

    该环境模拟了一个量子-经典混合计算平台的任务调度场景。RL Agent 在每个时间步
    需要做出调度决策：将当前任务分配到经典资源、量子资源，还是采用混合执行模式。

    状态空间为 10 维连续向量（Box），每个维度取值范围 [0, 1]，数据类型 float32。
    动作空间为 3 个离散动作（Discrete(3)）。

    典型用法::

        import gymnasium as gym
        env = gym.make("QuantumScheduling-v0")  # 需要注册
        # 或者直接实例化：
        env = QuantumSchedulingEnv()
        obs, info = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}  # noqa: RUF012

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(
        self,
        max_steps: int = MAX_STEPS_DEFAULT,
        max_qubits: int = 287,
        render_mode: str | None = None,
        seed: int | None = None,
        machine_configs: list[dict[str, Any]] | None = None,
        real_submit_probability: float = REAL_SUBMIT_PROBABILITY_DEFAULT,
    ):
        """
        初始化量子任务调度环境。

        Args:
            max_steps               : 单个 episode 的最大步数，超过后 episode 终止
            max_qubits              : 物理机总量子比特数，默认 287（对应天衍-287 真机）
            render_mode             : 渲染模式，支持 "human" 或 "ansi"，None 表示不渲染
            seed                    : 随机种子，用于可复现的实验
            machine_configs         : 多机器配置列表，None 时退化为单机模式（向后兼容）。
                                      每项字段见 DEFAULT_MACHINE_CONFIGS。
            real_submit_probability : 当机器 is_real=True 且已 attach 真实客户端时，
                                      每个量子任务以此概率真正提交真机（控制机时消耗）。
                                      0.0 表示纯仿真，1.0 表示每次都上真机。
        """
        super().__init__()

        self.max_steps = max_steps
        self.max_qubits = max_qubits
        self.render_mode = render_mode
        self.real_submit_probability = float(real_submit_probability)

        # Gymnasium 标准空间定义（保持 10 维 obs + Discrete(3) 不变，确保 PPO 模型可复用）
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # ---- 多机器调度扩展 ----
        # machine_configs=None → 单机模式（与旧版完全等价）
        if machine_configs is None:
            machine_configs = [
                {
                    "name": "tianyan_s",
                    "total_qubits": max_qubits,
                    "supported_gates": ("H", "CZ", "M"),
                    "is_real": False,
                }
            ]
        self._machines: list[QuantumMachine] = [
            QuantumMachine(
                name=cfg.get("name", "tianyan_s"),
                total_qubits=cfg.get("total_qubits", max_qubits),
                supported_gates=tuple(cfg.get("supported_gates", ("H", "CZ", "M"))),
                is_real=bool(cfg.get("is_real", False)),
            )
            for cfg in machine_configs
        ]

        # 真机客户端映射：machine_name -> client（由 attach_real_clients 注入）
        self._real_clients: dict[str, Any] = {}

        # 内部状态
        self._current_step: int = 0
        self._task_queue: list[Task] = []
        self._current_task: Task | None = None
        # self._quantum 保留为所有机器的聚合视图，确保旧版 obs/reward 逻辑不变
        self._quantum: QuantumResource = QuantumResource(total_qubits=max_qubits)
        self._classical: ClassicalResource = ClassicalResource()
        self._time_of_day: float = 0.0
        self._quantum_available: bool = True

        # 多机器调度记录
        self._last_selected_machine: str | None = None
        self._machine_schedule_count: dict[str, int] = {m.name: 0 for m in self._machines}
        self._machine_real_submits: dict[str, int] = {m.name: 0 for m in self._machines}

        # 统计信息（用于 info 字典和渲染）
        self._total_scheduled: int = 0
        self._quantum_success: int = 0
        self._classical_success: int = 0
        self._hybrid_success: int = 0
        self._mismatch_count: int = 0
        self._episode_reward: float = 0.0

        # 用于 ANSI 渲染的日志缓冲区
        self._render_log: list[str] = []

    # ------------------------------------------------------------------
    # 多机器调度：真机客户端接入
    # ------------------------------------------------------------------

    def attach_real_clients(self, clients: dict[str, Any]) -> None:
        """绑定真机客户端，启用选择性真机验证。

        Args:
            clients: 机器名 -> 客户端实例的映射（如 CqlibTianyanClient）。
                     绑定后，对应机器的 is_real 会被置为 True。
        """
        self._real_clients.update(clients)
        for m in self._machines:
            if m.name in clients:
                m.is_real = True

    @property
    def machine_names(self) -> list[str]:
        """返回当前所有机器名称列表。"""
        return [m.name for m in self._machines]

    @property
    def num_machines(self) -> int:
        """返回量子机器数量。"""
        return len(self._machines)

    # ------------------------------------------------------------------
    # 真机抽样辅助接口（供 RealMachineCallback 等外部模块使用）
    # ------------------------------------------------------------------

    def get_random_pending_task(self) -> Task | None:
        """从当前任务队列中随机取一个待处理任务（用于真机抽样提交）。

        优先从 ``_task_queue`` 中随机抽取；队列空时退化为当前正在调度的
        ``_current_task``；两者皆空时返回 ``None``。

        Returns:
            一个待处理的 Task，或 None（无任务可提交）。
        """
        if self._task_queue:
            idx = int(self.np_random.integers(0, len(self._task_queue)))
            return self._task_queue[idx]
        return self._current_task

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        重置环境到初始状态。

        随机初始化以下内容：
            - 任务队列：随机生成 5-20 个任务，任务类型从
              {"quantum", "classical", "universal"} 中均匀采样
            - 量子比特状态：保真度随机在 [0.85, 0.99] 之间，可用比率随机在 [0.3, 1.0]
            - 经典计算负载：随机在 [0.1, 0.7] 之间
            - 时间段：随机初始化一天中的时刻

        Args:
            seed   : 随机种子，覆盖构造时传入的种子
            options : 可选的额外参数，当前未使用

        Returns:
            observation (np.ndarray): 10 维初始状态向量，dtype=float32
            info (dict): 包含环境元数据的字典
        """
        super().reset(seed=seed)
        rng = self.np_random

        # 重置步数和统计
        self._current_step = 0
        self._total_scheduled = 0
        self._quantum_success = 0
        self._classical_success = 0
        self._hybrid_success = 0
        self._mismatch_count = 0
        self._episode_reward = 0.0
        self._render_log = []

        # 重置多机器调度记录
        self._last_selected_machine = None
        self._machine_schedule_count = {m.name: 0 for m in self._machines}
        self._machine_real_submits = {m.name: 0 for m in self._machines}

        # 随机初始化任务队列（5-20 个任务）
        self._task_queue = []
        initial_count = rng.integers(INITIAL_QUEUE_RANGE[0], INITIAL_QUEUE_RANGE[1] + 1)
        for i in range(initial_count):
            self._task_queue.append(self._generate_random_task(rng, task_id=i))

        # 随机初始化每台量子机器状态（多机器调度扩展）
        for m in self._machines:
            m.available_ratio = rng.uniform(0.3, 1.0)
            m.fidelity = rng.uniform(0.85, 0.99)
            m.quantum_queue = 0
            m.available = True
            # 初始化物理噪声特征和拓扑特征
            m.update_noise_features(rng)
            m.update_topology_features()
        # 聚合到 self._quantum，保证旧版 obs/reward 逻辑不变
        self._recompute_aggregate()

        # 随机初始化经典计算负载
        self._classical.load = rng.uniform(0.1, 0.7)
        self._classical.queue = 0

        # 随机初始化时间段
        self._time_of_day = rng.uniform(0.0, 1.0)

        # 取出队首任务作为当前任务
        self._pick_next_task()

        return self._get_observation(), self._get_info()

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        执行一步调度决策。

        根据智能体选择的动作将当前任务分配到对应资源，模拟执行过程，
        更新系统状态，计算奖励。

        执行流程：
            1. 根据 action 将当前任务分配到经典/量子/混合资源
            2. 判断是否为兼容分配（错误分配给予 -5 惩罚）
            3. 计算即时奖励（量子加速、经典基准、等待超时等）
            4. 检查量子比特利用率（<30% 时惩罚 -2）
            5. 推进仿真时间：更新资源状态、任务等待步数、生成新任务
            6. 推进时间段（模拟昼夜变化）

        Args:
            action : 动作索引，0=经典资源，1=量子资源，2=混合执行

        Returns:
            observation (np.ndarray): 下一步的 10 维状态向量
            reward (float): 本步获得的奖励值
            terminated (bool): 是否自然终止（达到最大步数）
            truncated (bool): 是否被截断（当前实现中始终为 False）
            info (dict): 包含本步详细信息的字典
        """
        self._current_step += 1
        rng = self.np_random

        # 本步总奖励从 0 开始累加。后续会把它拆成三类：
        # 1) 执行动作本身的收益或惩罚，例如正确执行、错误分配、量子不可用；
        # 2) 队列层面的全局惩罚，例如任务等待过久；
        # 3) 资源利用率惩罚，例如大量量子比特长期空闲。
        reward = 0.0

        if self._current_task is not None:
            task = self._current_task
            is_compatible = self._check_compatibility(task, action)

            # ---- 判断兼容性 ----
            # 兼容性是奖励计算的第一道门：
            # - quantum 任务不能走纯经典；
            # - classical 任务不能走纯量子；
            # - universal 任务可以走经典、量子或混合。
            # 不兼容时不执行任务，而是扣分并重新入队，提醒智能体避免错误路由。
            if not is_compatible:
                reward += REWARD_MISMATCH
                self._mismatch_count += 1
                # 不兼容任务被拒绝，重新入队尾部（如果有空间）
                task.wait_steps += 1
                if len(self._task_queue) < MAX_QUEUE_SIZE:
                    self._task_queue.append(task)
                self._last_selected_machine = None
                log_msg = (
                    f"[步骤{self._current_step}] 任务{task.task_id} 分配到不兼容资源"
                    f"(action={action})，惩罚{REWARD_MISMATCH}"
                )
            else:
                # ---- 多机器调度：为量子任务选择最佳机器 ----
                quantum_action = action in (ACTION_QUANTUM, ACTION_HYBRID)
                selected_machine = None
                if quantum_action:
                    selected_machine = self._select_best_machine(task)
                # 量子不可用 = 需要量子但没有任何机器能接下该任务。
                # 这里把“决策正确但资源暂时不可用”和“动作类型完全错误”区分开：
                # - 纯量子动作：任务重新排队，只给半个 mismatch 惩罚；
                # - 混合动作：允许降级为经典执行，避免系统完全空转。
                quantum_unavailable = quantum_action and selected_machine is None

                if quantum_unavailable:
                    if action == ACTION_QUANTUM:
                        reward += REWARD_MISMATCH * 0.5
                        task.wait_steps += 1
                        if len(self._task_queue) < MAX_QUEUE_SIZE:
                            self._task_queue.append(task)
                        self._last_selected_machine = None
                        log_msg = (
                            f"[步骤{self._current_step}] 量子资源不可用，"
                            f"任务{task.task_id} 重新入队，惩罚{REWARD_MISMATCH * 0.5:.1f}"
                        )
                    else:
                        reward += self._compute_execution_reward(task, ACTION_CLASSICAL, rng)
                        self._total_scheduled += 1
                        self._classical_success += 1
                        self._last_selected_machine = None
                        log_msg = (
                            f"[步骤{self._current_step}] 量子资源不可用，"
                            f"混合任务{task.task_id}降级为经典执行，reward={reward:.2f}"
                        )
                else:
                    # ---- 兼容分配：计算执行奖励 ----
                    # 执行奖励由 _compute_execution_reward() 负责：
                    # 经典执行给稳定基准分，量子执行按加速比和保真度放大，
                    # 混合执行介于二者之间并受当前量子可用率影响。
                    reward += self._compute_execution_reward(task, action, rng)
                    self._total_scheduled += 1

                    if action == ACTION_QUANTUM:
                        self._quantum_success += 1
                        self._route_to_machine(selected_machine, task, rng)
                    elif action == ACTION_CLASSICAL:
                        self._classical_success += 1
                        self._last_selected_machine = None
                    else:
                        self._hybrid_success += 1
                        # 混合执行同样占用量子机器（若有）
                        self._route_to_machine(selected_machine, task, rng)

                    machine_tag = (
                        f"@{selected_machine.name}" if selected_machine is not None else ""
                    )
                    log_msg = (
                        f"[步骤{self._current_step}] 任务{task.task_id}({task.task_type})"
                        f" → action={action}{machine_tag}, reward={reward:.2f}"
                    )

            self._render_log.append(log_msg)

        else:
            # 无任务可调度，轻微惩罚
            reward -= 1.0

        # ---- 等待超时惩罚（遍历队列中所有任务） ----
        # 这是一个全局队列惩罚：即使当前任务执行成功，如果队列里有任务
        # 已经等待太久，本步总奖励也会被扣分。这样智能体会学到“别只挑
        # 容易得分的任务，也要照顾快超时的任务”。
        reward += self._compute_wait_penalty()

        # ---- 量子比特利用率惩罚 ----
        # available_ratio 越高表示空闲量子比特越多。若空闲比例高于
        # 1 - QUBIT_UTIL_THRESHOLD，即实际利用率低于 30%，扣一小分，
        # 鼓励策略在合适的时候把任务送到量子资源上。
        if self._quantum.available_ratio > (1.0 - QUBIT_UTIL_THRESHOLD):
            reward += REWARD_LOW_QUBIT_UTIL

        # ---- 推进仿真时间 ----
        self._advance_time(rng)

        # ---- 推进时间段（昼夜循环） ----
        self._time_of_day = (self._time_of_day + 1.0 / self.max_steps) % 1.0

        # ---- 取出下一个任务 ----
        self._pick_next_task()

        # ---- 累计奖励 ----
        self._episode_reward += reward

        # ---- 判断终止 ----
        terminated = self._current_step >= self.max_steps
        truncated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # render()
    # ------------------------------------------------------------------

    def render(self) -> Any | None:
        """
        渲染当前环境状态。

        - "human" 模式：在标准输出打印格式化的状态面板
        - "ansi" 模式：返回 ANSI 字符串（适合日志或测试）
        - None（未设置渲染模式）：不执行任何操作

        Returns:
            render_mode == "ansi" 时返回格式化字符串，否则返回 None
        """
        if self.render_mode is None:
            return None

        # 构建状态面板
        sep = "=" * 64
        lines = [
            sep,
            f"  量子任务调度环境  |  步骤: {self._current_step}/{self.max_steps}"
            f"  |  累计奖励: {self._episode_reward:.2f}"
            f"  |  机器数: {len(self._machines)}",
            "-" * 64,
            f"  [量子资源(聚合)] 可用比率: {self._quantum.available_ratio:.2%}"
            f"  |  保真度: {self._quantum.fidelity:.4f}"
            f"  |  量子队列: {self._quantum.quantum_queue}",
            f"  [经典资源] 负载: {self._classical.load:.2%}"
            f"  |  经典队列: {self._classical.queue}",
            f"  [任务队列]  长度: {len(self._task_queue)}" f"  |  已调度: {self._total_scheduled}",
            f"  [统计] 量子成功: {self._quantum_success}"
            f"  |  经典成功: {self._classical_success}"
            f"  |  混合成功: {self._hybrid_success}"
            f"  |  不兼容: {self._mismatch_count}",
        ]

        # 多机器明细表
        if len(self._machines) > 1:
            lines.append("-" * 64)
            lines.append("  [量子机器明细]")
            for m in self._machines:
                status = "在线" if m.available else "维护"
                real_tag = "真机" if m.is_real else "仿真"
                sched_cnt = self._machine_schedule_count.get(m.name, 0)
                lines.append(
                    f"    {m.name:14s} | {m.total_qubits:3d}q | "
                    f"可用{m.available_ratio:5.1%} | 保真{m.fidelity:.3f} | "
                    f"队列{m.quantum_queue:2d} | {status} | {real_tag} | "
                    f"调度{sched_cnt}"
                )
            if self._last_selected_machine:
                lines.append(f"  [本步路由] → {self._last_selected_machine}")

        lines.append("-" * 64)

        # 附加当前任务信息
        if self._current_task is not None:
            t = self._current_task
            lines.append(
                f"  [当前任务] ID={t.task_id}  类型={t.task_type}"
                f"  紧急={t.urgency:.2f}  等待={t.wait_steps}步"
            )
        else:
            lines.append("  [当前任务] 无")

        # 附加最近日志
        if self._render_log:
            recent = self._render_log[-5:]  # 最近5条
            lines.append("-" * 64)
            lines.append("  最近日志:")
            for log in recent:
                lines.append(f"    {log}")

        lines.append(sep)
        output = "\n".join(lines) + "\n"

        if self.render_mode == "human":
            logger.info(output)
        elif self.render_mode == "ansi":
            return output

        return None

    # ------------------------------------------------------------------
    # close()
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        关闭环境，释放资源。

        当前实现中无额外资源需要释放，仅清空内部状态。
        """
        self._task_queue.clear()
        self._current_task = None
        self._render_log.clear()

    # ------------------------------------------------------------------
    # 私有方法：任务生成
    # ------------------------------------------------------------------

    def _generate_random_task(self, rng: np.random.Generator, task_id: int) -> Task:
        """
        生成具有真实差异性的随机任务（异质化版本）。

        任务参数采用不均匀分布，模拟真实场景中的任务多样性：
            - qubits: 偏向中小规模，但偶发大规模（长尾分布）
            - urgency: 大部分正常，少数紧急
            - execution_time: 与 qubits 正相关
            - task_type: 混合分布，量子任务居多

        Args:
            rng     : NumPy 随机数生成器
            task_id : 任务编号

        Returns:
            Task: 生成的随机任务对象
        """
        qubit_options = [2, 3, 5, 5, 8, 10, 10, 15, 20, 30, 50, 100]
        qubit_probs = [0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02]
        qubits = int(rng.choice(qubit_options, p=qubit_probs))

        urgency_options = [0.1, 0.3, 0.5, 0.5, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
        urgency_probs = [0.05, 0.10, 0.20, 0.20, 0.15, 0.10, 0.08, 0.05, 0.05, 0.02]
        urgency = float(rng.choice(urgency_options, p=urgency_probs))

        base_time = int(qubits**0.6)
        execution_time = max(1, base_time + int(rng.choice([-1, 0, 0, 0, 1, 2])))

        task_type_options = ["quantum", "quantum", "classical", "classical", "universal"]
        task_type_probs = [0.35, 0.35, 0.10, 0.10, 0.10]
        task_type = str(rng.choice(task_type_options, p=task_type_probs))

        qubit_count = 0 if task_type == "classical" else qubits

        return Task(
            task_id=f"T{task_id:04d}",
            task_type=task_type,
            qubit_count=qubit_count,
            wait_steps=0,
            urgency=urgency,
            priority=int(rng.integers(1, 6)),
            execution_time=execution_time,
        )

    # ------------------------------------------------------------------
    # 私有方法：兼容性检查
    # ------------------------------------------------------------------

    def _check_compatibility(self, task: Task, action: int) -> bool:
        """
        检查任务类型与所选资源的兼容性。

        兼容性规则（修改后）：
            - classical 任务 → 经典资源 (action=0) 或混合执行 (action=2)
            - quantum 任务   → 量子资源 (action=1) 或混合执行 (action=2)
            - universal 任务 → 三种动作均兼容

        注意：混合执行 (action=2) 现在对所有任务类型都兼容，
        因为它会根据实际情况灵活选择资源。

        Args:
            task   : 待检查的任务
            action : 智能体选择的动作

        Returns:
            bool: True 表示兼容，False 表示不兼容
        """
        if task.task_type == "classical":
            # classical任务：经典资源或混合执行都兼容
            return action in (ACTION_CLASSICAL, ACTION_HYBRID)
        elif task.task_type == "quantum":
            # quantum任务：量子资源或混合执行都兼容
            return action in (ACTION_QUANTUM, ACTION_HYBRID)
        else:  # universal
            return True

    # ------------------------------------------------------------------
    # 私有方法：多机器调度
    # ------------------------------------------------------------------

    def _select_best_machine(self, task: Task) -> QuantumMachine | None:
        """
        为给定任务选择最合适的量子机器（多机器调度核心启发式）。

        选择策略：
            1. 过滤：机器在线、可用比特数 >= 任务需求、支持任务所需门集合
            2. 评分：score = fidelity * available_ratio / (1 + quantum_queue)
            3. 返回得分最高的机器；若无机器通过过滤，返回 None

        评分兼顾保真度、可用资源与队列负载，倾向于把任务分发给
        当前质量最好且最空闲的机器，实现负载均衡与质量择优。

        Args:
            task: 待调度的任务

        Returns:
            QuantumMachine 或 None（无机器可承接）
        """
        candidates = []
        for m in self._machines:
            if not m.available:
                continue
            # 比特数检查（available_ratio * total_qubits 为估算可用比特）
            usable_qubits = int(m.total_qubits * m.available_ratio)
            if usable_qubits < task.qubit_count:
                continue
            # 门集合检查（任务所需门需被机器支持；这里按常见量子门粗粒度匹配）
            if not self._machine_supports_task(m, task):
                continue
            candidates.append(m)

        if not candidates:
            return None

        # 评分：保真度 * 可用比率 / (1 + 队列长度)
        def _score(m: QuantumMachine) -> float:
            return m.fidelity * m.available_ratio / (1.0 + m.quantum_queue)

        return max(candidates, key=_score)

    def _machine_supports_task(self, machine: QuantumMachine, task: Task) -> bool:
        """检查机器门集合是否兼容任务需求。

        当前实现：tianyan_s 仅支持 H/CZ/M（真机实测约束），
        含参数化门（RX/RY/RZ）的任务需路由到 tianyan_tn 等支持该门集的机器。
        任务的 required_gates 字段若未声明，默认按机器声明集合宽松放行。

        Args:
            machine: 候选机器
            task:    待执行任务

        Returns:
            bool: 机器能否执行该任务
        """
        required = getattr(task, "required_gates", None)
        if not required:
            # 任务未声明门需求，默认放行（保持与旧版行为一致）
            return True
        return set(required).issubset(set(machine.supported_gates))

    def _route_to_machine(
        self,
        machine: QuantumMachine | None,
        task: Task,
        rng: np.random.Generator,
    ) -> None:
        """将任务路由到选定的量子机器，更新队列与调度记录。

        若机器为真机模式（is_real=True 且已 attach 客户端），则以
        ``real_submit_probability`` 概率真正提交到天衍云真机，控制机时消耗。

        Args:
            machine: 选定的量子机器（None 时不做任何操作）
            task:    被执行的任务
            rng:     随机数生成器（用于抽样是否上真机）
        """
        if machine is None:
            self._last_selected_machine = None
            return

        machine.quantum_queue += 1
        self._last_selected_machine = machine.name
        self._machine_schedule_count[machine.name] = (
            self._machine_schedule_count.get(machine.name, 0) + 1
        )

        # 选择性真机提交（控制机时成本）
        if (
            machine.is_real
            and machine.name in self._real_clients
            and self.real_submit_probability > 0.0
            and float(rng.random()) < self.real_submit_probability
        ):
            self._submit_to_real_machine(machine, task)

    def _submit_to_real_machine(self, machine: QuantumMachine, task: Task) -> None:
        """向真机提交一个量子任务（异常安全，失败仅记录日志）。

        真机提交在仿真循环中是非阻塞的：提交后不等待结果，仅计入
        真机提交计数，避免阻塞 RL 训练。结果可在 demo 脚本中单独轮询。

        Args:
            machine: 目标真机
            task:    待提交任务
        """
        client = self._real_clients.get(machine.name)
        if client is None:
            return
        # 构造一个最小可执行的 QCIS 电路（H 门 + 测量），用于真机验证
        # 真实场景下应由 parser 从 task 生成 QCIS，这里用占位电路做连通性验证
        qcis = getattr(task, "qcis", None) or "H Q0\nM Q0"
        try:
            client.submit_quantum_task(
                qcis=qcis,
                shots=512,
                task_name=f"RL_{task.task_id}",
            )
            self._machine_real_submits[machine.name] = (
                self._machine_real_submits.get(machine.name, 0) + 1
            )
        except Exception as e:
            # 真机 API 提交可能因网络/认证/服务端等多种原因失败，无法精确收窄
            logger.error(f"[真机] {machine.name} 提交失败: {e}")
            self._render_log.append(f"[真机] {machine.name} 提交失败: {str(e)[:60]}")

    def _recompute_aggregate(self) -> None:
        """根据所有机器状态重算 self._quantum 聚合视图。

        聚合规则：
            - available_ratio : 各机器可用比率的加权均值（按 total_qubits 加权）
            - fidelity        : 各机器保真度的加权均值
            - quantum_queue   : 所有机器队列长度之和
            - _quantum_available : 任一机器在线即为 True

        该聚合保证旧版 10 维观测与奖励计算在多机器模式下仍然有效。
        """
        if not self._machines:
            return
        total_q = sum(m.total_qubits for m in self._machines)
        if total_q <= 0:
            total_q = 1
        self._quantum.available_ratio = float(
            sum(m.available_ratio * m.total_qubits for m in self._machines) / total_q
        )
        self._quantum.fidelity = float(
            sum(m.fidelity * m.total_qubits for m in self._machines) / total_q
        )
        self._quantum.quantum_queue = sum(m.quantum_queue for m in self._machines)
        self._quantum_available = any(m.available for m in self._machines)

    # ------------------------------------------------------------------
    # 私有方法：执行奖励计算
    # ------------------------------------------------------------------

    def _compute_execution_reward(
        self,
        task: Task,
        action: int,
        rng: np.random.Generator,
    ) -> float:
        """
        计算任务执行成功后的即时奖励。

        这个函数只处理“任务已经被安排执行”的正向收益，不处理错误分配、
        等待超时和低利用率惩罚；那些全局项在 step() 中统一累加。

        奖励规则：
            - 经典执行 (action=0):
              REWARD_CLASSICAL + REWARD_SUCCESS_BONUS，作为稳定基准。
            - 量子执行 (action=1):
              REWARD_QUANTUM_BASE * speedup + REWARD_SUCCESS_BONUS。
              speedup 从 QUANTUM_SPEEDUP_RANGE 随机采样，并乘以保真度因子；
              当保真度低于 0.9 时再乘 0.6，表示低质量量子结果的折扣。
            - 混合执行 (action=2):
              REWARD_HYBRID * hybrid_factor + REWARD_SUCCESS_BONUS。
              hybrid_factor 随量子可用率从 0.5 到 1.0 变化，表示量子资源越充足，
              混合执行越接近完整收益。

        Args:
            task   : 被执行的任务
            action : 执行方式
            rng    : 随机数生成器

        Returns:
            float: 计算得到的即时奖励
        """
        if action == ACTION_CLASSICAL:
            # 经典执行不依赖量子机器状态，奖励最稳定，用作所有策略的基准线。
            return REWARD_CLASSICAL + REWARD_SUCCESS_BONUS

        elif action == ACTION_QUANTUM:
            # 基础加速比在 [2, 5] 之间随机
            speedup = rng.uniform(*QUANTUM_SPEEDUP_RANGE)
            # 保真度加成：保真度越高，加速比越大
            fidelity_factor = self._quantum.fidelity / 0.99  # 归一化到 ~1.0
            speedup *= fidelity_factor
            reward = REWARD_QUANTUM_BASE * speedup
            # 保真度过低时打折，避免智能体盲目偏向低质量量子资源。
            if self._quantum.fidelity < 0.9:
                reward *= 0.6
            return reward + REWARD_SUCCESS_BONUS

        else:  # ACTION_HYBRID
            # 混合执行奖励介于经典和量子之间，并随量子可用率动态调整。
            base = REWARD_HYBRID
            # available_ratio=0 时 factor=0.5，available_ratio=1 时 factor=1.0。
            hybrid_factor = 0.5 + 0.5 * self._quantum.available_ratio
            return base * hybrid_factor + REWARD_SUCCESS_BONUS

    # ------------------------------------------------------------------
    # 私有方法：等待超时惩罚
    # ------------------------------------------------------------------

    def _compute_wait_penalty(self) -> float:
        """
        计算队列中所有任务的等待超时惩罚。

        当任务的等待步数超过 MAX_WAIT_STEPS 时，每超一步惩罚 -0.5。
        惩罚与超时比例成正比。

        Returns:
            float: 总等待惩罚（通常为负值或零）
        """
        penalty = 0.0
        for task in self._task_queue:
            if task.wait_steps > MAX_WAIT_STEPS:
                overtime_ratio = (task.wait_steps - MAX_WAIT_STEPS) / MAX_WAIT_STEPS
                penalty += REWARD_WAIT_OVER_THRESHOLD * overtime_ratio
        return penalty

    # ------------------------------------------------------------------
    # 私有方法：时间推进与状态更新
    # ------------------------------------------------------------------

    def _advance_time(self, rng: np.random.Generator) -> None:
        """
        推进一个仿真时间步，更新所有资源状态。

        更新内容：
            1. 量子资源：每台机器独立波动（可用比特、保真度、在线状态、队列完成）
            2. 经典资源：负载随机波动，任务完成后释放
            3. 队列中任务：每个任务等待步数 +1
            4. 随机生成 0-3 个新任务（泊松分布，均值 1.2）
            5. 聚合所有机器状态到 self._quantum（保证旧版 obs 一致）
        """
        # ---- 每台量子机器独立波动 ----
        for m in self._machines:
            # 可用比特比率波动
            m.available_ratio = float(
                np.clip(m.available_ratio + rng.uniform(-0.1, 0.1), 0.05, 1.0)
            )
            # 保真度衰减 + 随机恢复
            m.fidelity = float(np.clip(m.fidelity - 0.002 + rng.uniform(0.0, 0.01), 0.7, 0.999))
            # 更新物理噪声特征（基于新的 fidelity）
            m.update_noise_features(rng)
            # 在线状态随机波动（模拟真机维护/校准，5% 概率翻转）
            if rng.random() < 0.05:
                m.available = not m.available
            # 该机器队列完成任务
            completed_m = 0
            for _ in range(m.quantum_queue):
                if rng.random() < 0.15:  # 15% 概率完成一个量子任务
                    completed_m += 1
            m.quantum_queue = max(0, m.quantum_queue - completed_m)

        # 聚合到 self._quantum（保持旧版 obs/reward 逻辑不变）
        self._recompute_aggregate()

        # 经典资源波动
        self._classical.load = np.clip(self._classical.load + rng.uniform(-0.15, 0.15), 0.0, 1.0)
        # 经典队列完成一些任务
        completed_c = 0
        for _ in range(self._classical.queue):
            if rng.random() < 0.2:  # 20% 概率完成一个经典任务
                completed_c += 1
        self._classical.queue = max(0, self._classical.queue - completed_c)
        # 完成任务后释放负载
        self._classical.load = np.clip(self._classical.load - 0.05 * completed_c, 0.0, 1.0)

        # 队列中任务等待步数 +1
        for task in self._task_queue:
            task.wait_steps += 1

        # 随机生成新任务（泊松分布，均值 1.2）
        new_task_count = int(rng.poisson(1.2))
        for _ in range(new_task_count):
            if len(self._task_queue) < MAX_QUEUE_SIZE:
                new_id = self._total_scheduled + len(self._task_queue)
                self._task_queue.append(self._generate_random_task(rng, task_id=new_id))

    # ------------------------------------------------------------------
    # 私有方法：取出下一个任务
    # ------------------------------------------------------------------

    def _pick_next_task(self) -> None:
        """
        从队列中取出下一个待调度任务作为当前任务。

        优先调度紧急程度最高的任务（priority 降序、wait_steps 降序）。
        """
        if not self._task_queue:
            self._current_task = None
            return

        # 按紧急程度和等待时间排序，选出最紧急的任务
        self._task_queue.sort(key=lambda t: (-t.priority, -t.wait_steps, -t.urgency))
        self._current_task = self._task_queue.pop(0)

    # ------------------------------------------------------------------
    # 私有方法：构建观测向量
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """
        构建并返回当前 14 维状态向量（扩展版：包含物理噪声和拓扑特征）。

        各维度含义及计算方式：
            [0] qubit_availability  : 量子比特可用比率（直接取值）
            [1] queue_length        : 队列长度 / MAX_QUEUE_SIZE
            [2] avg_wait_time       : 队列中任务平均等待步数 / MAX_WAIT_STEPS
            [3] fidelity            : 量子比特平均保真度
            [4] classical_load     : 经典计算负载（直接取值）
            [5] quantum_queue_ratio : 量子队列 / (量子队列 + 经典队列 + 1)
            [6] time_of_day        : 当前模拟时刻
            [7] urgency_level      : 当前任务紧急程度（无任务时为 0）
            [8] task_type_quantum  : 当前任务是quantum类型（1=是，0=不是）
            [9] task_type_classical: 当前任务是classical类型（1=是，0=不是）
            [10] single_gate_fidelity : 单比特门平均保真度（所有机器加权平均）
            [11] two_gate_fidelity : 两比特门平均保真度（所有机器加权平均）
            [12] coupling_density  : 耦合图密度（所有机器加权平均）
            [13] avg_connectivity  : 量子比特平均连通度（所有机器加权平均）

        Returns:
            np.ndarray: 形状 (14,)，dtype=float32，值域 [0, 1]
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        obs[OBS_QUBIT_AVAILABILITY] = float(np.clip(self._quantum.available_ratio, 0.0, 1.0))

        obs[OBS_QUEUE_LENGTH] = float(np.clip(len(self._task_queue) / MAX_QUEUE_SIZE, 0.0, 1.0))

        if self._task_queue:
            avg_wait = sum(t.wait_steps for t in self._task_queue) / len(self._task_queue)
            obs[OBS_AVG_WAIT_TIME] = float(np.clip(avg_wait / MAX_WAIT_STEPS, 0.0, 1.0))
        else:
            obs[OBS_AVG_WAIT_TIME] = 0.0

        obs[OBS_FIDELITY] = float(np.clip(self._quantum.fidelity, 0.0, 1.0))

        obs[OBS_CLASSICAL_LOAD] = float(np.clip(self._classical.load, 0.0, 1.0))

        total_running = self._quantum.quantum_queue + self._classical.queue + 1
        obs[OBS_QUANTUM_QUEUE_RATIO] = float(
            np.clip(self._quantum.quantum_queue / total_running, 0.0, 1.0)
        )

        obs[OBS_TIME_OF_DAY] = float(np.clip(self._time_of_day, 0.0, 1.0))

        if self._current_task is not None:
            obs[OBS_URGENCY_LEVEL] = float(np.clip(self._current_task.urgency, 0.0, 1.0))
        else:
            obs[OBS_URGENCY_LEVEL] = 0.0

        # 添加任务类型编码
        if self._current_task is not None:
            obs[OBS_TASK_TYPE_QUANTUM] = 1.0 if self._current_task.task_type == "quantum" else 0.0
            obs[OBS_TASK_TYPE_CLASSICAL] = (
                1.0 if self._current_task.task_type == "classical" else 0.0
            )
        else:
            obs[OBS_TASK_TYPE_QUANTUM] = 0.0
            obs[OBS_TASK_TYPE_CLASSICAL] = 0.0

        # 阶段1：物理噪声特征（所有机器加权平均）
        if self._machines:
            total_q = sum(m.total_qubits for m in self._machines)
            if total_q > 0:
                obs[OBS_SINGLE_GATE_FIDELITY] = float(
                    np.clip(
                        sum(m.single_gate_fidelity * m.total_qubits for m in self._machines)
                        / total_q,
                        0.0,
                        1.0,
                    )
                )
                obs[OBS_TWO_GATE_FIDELITY] = float(
                    np.clip(
                        sum(m.two_gate_fidelity * m.total_qubits for m in self._machines) / total_q,
                        0.0,
                        1.0,
                    )
                )
                # 阶段2：拓扑特征（所有机器加权平均）
                obs[OBS_COUPLING_DENSITY] = float(
                    np.clip(
                        sum(m.coupling_density * m.total_qubits for m in self._machines) / total_q,
                        0.0,
                        1.0,
                    )
                )
                obs[OBS_AVG_CONNECTIVITY] = float(
                    np.clip(
                        sum(m.avg_connectivity * m.total_qubits for m in self._machines) / total_q,
                        0.0,
                        1.0,
                    )
                )

        return obs

    # ------------------------------------------------------------------
    # 私有方法：构建信息字典
    # ------------------------------------------------------------------

    def _get_info(self) -> dict[str, Any]:
        """
        构建环境信息字典，供调试和监控使用。

        Returns:
            dict: 包含当前步数、统计摘要、资源状态、多机器调度详情等信息
        """
        info: dict[str, Any] = {
            "current_step": self._current_step,
            "max_steps": self._max_steps,
            "task_queue_length": len(self._task_queue),
            "total_scheduled": self._total_scheduled,
            "quantum_success": self._quantum_success,
            "classical_success": self._classical_success,
            "hybrid_success": self._hybrid_success,
            "mismatch_count": self._mismatch_count,
            "episode_reward": self._episode_reward,
            "qubit_availability": self._quantum.available_ratio,
            "fidelity": self._quantum.fidelity,
            "classical_load": self._classical.load,
            "time_of_day": self._time_of_day,
            # 多机器调度信息
            "num_machines": len(self._machines),
            "last_selected_machine": self._last_selected_machine,
            "machine_schedule_count": dict(self._machine_schedule_count),
            "machine_real_submits": dict(self._machine_real_submits),
            "machines": [
                {
                    "name": m.name,
                    "total_qubits": m.total_qubits,
                    "available_ratio": m.available_ratio,
                    "fidelity": m.fidelity,
                    "quantum_queue": m.quantum_queue,
                    "available": m.available,
                    "is_real": m.is_real,
                    "supported_gates": list(m.supported_gates),
                }
                for m in self._machines
            ],
        }
        if self._current_task is not None:
            info["current_task"] = {
                "task_id": self._current_task.task_id,
                "task_type": self._current_task.task_type,
                "urgency": self._current_task.urgency,
                "wait_steps": self._current_task.wait_steps,
            }
        return info

    @property
    def _max_steps(self) -> int:
        """返回最大步数，便于 info 字典引用。"""
        return self.max_steps


# ---------------------------------------------------------------------------
# Gymnasium 注册辅助函数
# ---------------------------------------------------------------------------


def register_env() -> None:
    """
    将 QuantumSchedulingEnv 注册到 Gymnasium 注册表。

    注册后可通过 ``gym.make("QuantumScheduling-v0")`` 创建环境实例。

    Usage::

        from scheduler.env import register_env
        register_env()
        env = gym.make("QuantumScheduling-v0", max_steps=1000)
    """
    from gymnasium.envs.registration import register

    try:  # noqa: SIM105
        register(
            id="QuantumScheduling-v0",
            entry_point="src.scheduler.env:QuantumSchedulingEnv",
        )
    except gym.error.Error:
        # 已注册，忽略
        pass


# ---------------------------------------------------------------------------
# 命令行测试入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  量子任务调度环境 - 功能测试")
    print("=" * 60)

    # 创建环境（带 human 渲染模式）
    env = QuantumSchedulingEnv(max_steps=20, render_mode="human")

    # 测试 reset
    print("\n--- reset() 测试 ---")
    obs, info = env.reset(seed=42)
    print(f"状态维度: {env.observation_space.shape}")
    print(f"状态向量: {obs}")
    print(f"动作空间: {env.action_space}")
    print(f"初始队列长度: {info['task_queue_length']}")

    # 动作名称映射
    action_names = {
        ACTION_CLASSICAL: "经典资源",
        ACTION_QUANTUM: "量子资源",
        ACTION_HYBRID: "混合执行",
    }

    # 随机执行若干步
    print("\n--- step() 测试（随机策略） ---")
    for _i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            print(f"\nEpisode 结束（terminated={terminated}, truncated={truncated}）")
            break

    # 打印最终统计
    print("\n--- Episode 统计 ---")
    final_info = env._get_info()
    print(f"总步数: {final_info['current_step']}")
    print(f"累计奖励: {final_info['episode_reward']:.2f}")
    print(f"已调度任务: {final_info['total_scheduled']}")
    print(f"  量子成功: {final_info['quantum_success']}")
    print(f"  经典成功: {final_info['classical_success']}")
    print(f"  混合成功: {final_info['hybrid_success']}")
    print(f"  不兼容分配: {final_info['mismatch_count']}")

    # 测试无渲染模式
    print("\n--- 无渲染模式测试 ---")
    env_silent = QuantumSchedulingEnv(max_steps=10)
    obs, info = env_silent.reset(seed=123)
    total_reward = 0.0
    for _ in range(10):
        obs, reward, terminated, truncated, info = env_silent.step(int(np.argmax(obs[:3])))
        total_reward += reward
        if terminated:
            break
    print(f"简单策略累计奖励: {total_reward:.2f}")

    env.close()
    env_silent.close()

    print("\n所有测试完成。")
