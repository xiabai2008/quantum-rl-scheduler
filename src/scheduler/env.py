"""
量子任务调度环境模块
Quantum-Classical Hybrid Task Scheduling Environment for RL Training

基于 Gymnasium 框架构建的量子-经典混合计算调度环境。
RL Agent 需要决定每个任务是在经典计算资源上执行，还是在量子计算资源上执行，
或者采用量子-经典混合协同执行方式。

状态空间（8维，float32）：
    - qubit_availability : 当前可用量子比特比率（0-1）
    - queue_length       : 当前任务队列长度（归一化到 0-1）
    - avg_wait_time      : 队列中任务平均等待时间（归一化）
    - fidelity           : 当前量子比特平均保真度（0-1）
    - classical_load     : 经典计算资源负载（0-1）
    - quantum_queue_ratio: 量子专用队列占比（0-1）
    - time_of_day        : 一天中的时间段（0-1，模拟昼夜负载差异）
    - urgency_level      : 当前任务的紧急程度（0-1）

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

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 状态向量索引
OBS_QUBIT_AVAILABILITY = 0   # 当前可用量子比特比率
OBS_QUEUE_LENGTH = 1          # 任务队列长度（归一化）
OBS_AVG_WAIT_TIME = 2         # 平均等待时间（归一化）
OBS_FIDELITY = 3              # 量子比特平均保真度
OBS_CLASSICAL_LOAD = 4        # 经典计算资源负载
OBS_QUANTUM_QUEUE_RATIO = 5   # 量子专用队列占比
OBS_TIME_OF_DAY = 6           # 一天中的时间段（昼夜模拟）
OBS_URGENCY_LEVEL = 7         # 当前任务紧急程度

OBS_DIM = 8  # 状态空间维度

# 动作常量
ACTION_CLASSICAL = 0   # 分配到经典计算资源
ACTION_QUANTUM = 1     # 分配到量子计算资源
ACTION_HYBRID = 2      # 混合执行

# 奖励参数
REWARD_QUANTUM_BASE = 10.0       # 量子执行基础奖励
REWARD_CLASSICAL = 3.0            # 经典执行基准奖励
REWARD_WAIT_OVER_THRESHOLD = -0.5  # 等待超时惩罚系数
REWARD_LOW_QUBIT_UTIL = -2.0     # 量子比特利用率过低的惩罚
REWARD_MISMATCH = -5.0           # 错误分配（不兼容资源）惩罚
QUANTUM_SPEEDUP_RANGE = (2.0, 5.0)  # 量子加速比范围

# 环境参数
MAX_QUEUE_SIZE = 30              # 队列最大长度（用于归一化）
MAX_WAIT_STEPS = 50              # 最大等待步数（超过此阈值开始惩罚）
MAX_STEPS_DEFAULT = 500          # 默认最大步数（一个 episode）
QUBIT_UTIL_THRESHOLD = 0.3       # 量子比特利用率低阈值
INITIAL_QUEUE_RANGE = (5, 20)    # reset 时初始任务队列大小范围


# ---------------------------------------------------------------------------
# 辅助数据结构
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """
    表示队列中的单个待调度任务。

    Attributes:
        task_id      : 唯一任务标识符
        task_type    : 任务类型，"quantum"（仅量子可执行）、"classical"（仅经典可执行）、"universal"（两者皆可）
        qubit_count  : 该任务所需的量子比特数
        wait_steps   : 该任务已在队列中等待的步数
        urgency      : 紧急程度 0-1，越高越紧急
        priority     : 优先级 1-5
    """
    task_id: str
    task_type: str  # "quantum", "classical", "universal"
    qubit_count: int = 0
    wait_steps: int = 0
    urgency: float = 0.5
    priority: int = 3


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


# ---------------------------------------------------------------------------
# 核心环境类
# ---------------------------------------------------------------------------

class QuantumSchedulingEnv(gym.Env):
    """
    量子-经典混合计算调度环境

    该环境模拟了一个量子-经典混合计算平台的任务调度场景。RL Agent 在每个时间步
    需要做出调度决策：将当前任务分配到经典资源、量子资源，还是采用混合执行模式。

    状态空间为 8 维连续向量（Box），每个维度取值范围 [0, 1]，数据类型 float32。
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

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(
        self,
        max_steps: int = MAX_STEPS_DEFAULT,
        max_qubits: int = 287,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        初始化量子任务调度环境。

        Args:
            max_steps   : 单个 episode 的最大步数，超过后 episode 终止
            max_qubits  : 物理机总量子比特数，默认 287（对应天衍-287 真机）
            render_mode : 渲染模式，支持 "human" 或 "ansi"，None 表示不渲染
            seed        : 随机种子，用于可复现的实验
        """
        super().__init__()

        self.max_steps = max_steps
        self.max_qubits = max_qubits
        self.render_mode = render_mode

        # Gymnasium 标准空间定义
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # 内部状态
        self._current_step: int = 0
        self._task_queue: List[Task] = []
        self._current_task: Optional[Task] = None
        self._quantum: QuantumResource = QuantumResource(total_qubits=max_qubits)
        self._classical: ClassicalResource = ClassicalResource()
        self._time_of_day: float = 0.0

        # 统计信息（用于 info 字典和渲染）
        self._total_scheduled: int = 0
        self._quantum_success: int = 0
        self._classical_success: int = 0
        self._hybrid_success: int = 0
        self._mismatch_count: int = 0
        self._episode_reward: float = 0.0

        # 用于 ANSI 渲染的日志缓冲区
        self._render_log: List[str] = []

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
            observation (np.ndarray): 8 维初始状态向量，dtype=float32
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

        # 随机初始化任务队列（5-20 个任务）
        self._task_queue = []
        initial_count = rng.integers(INITIAL_QUEUE_RANGE[0], INITIAL_QUEUE_RANGE[1] + 1)
        for i in range(initial_count):
            self._task_queue.append(self._generate_random_task(rng, task_id=i))

        # 随机初始化量子比特状态
        self._quantum.available_ratio = rng.uniform(0.3, 1.0)
        self._quantum.fidelity = rng.uniform(0.85, 0.99)
        self._quantum.quantum_queue = 0

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

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
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
            observation (np.ndarray): 下一步的 8 维状态向量
            reward (float): 本步获得的奖励值
            terminated (bool): 是否自然终止（达到最大步数）
            truncated (bool): 是否被截断（当前实现中始终为 False）
            info (dict): 包含本步详细信息的字典
        """
        self._current_step += 1
        rng = self.np_random

        reward = 0.0
        scheduled = False

        if self._current_task is not None:
            task = self._current_task
            is_compatible = self._check_compatibility(task, action)

            # ---- 判断兼容性 ----
            if not is_compatible:
                reward += REWARD_MISMATCH
                self._mismatch_count += 1
                # 不兼容任务被拒绝，重新入队尾部（如果有空间）
                task.wait_steps += 1
                if len(self._task_queue) < MAX_QUEUE_SIZE:
                    self._task_queue.append(task)
                log_msg = f"[步骤{self._current_step}] 任务{task.task_id} 分配到不兼容资源(action={action})，惩罚{REWARD_MISMATCH}"
            else:
                # ---- 兼容分配：计算执行奖励 ----
                reward += self._compute_execution_reward(task, action, rng)
                scheduled = True
                self._total_scheduled += 1

                if action == ACTION_QUANTUM:
                    self._quantum_success += 1
                elif action == ACTION_CLASSICAL:
                    self._classical_success += 1
                else:
                    self._hybrid_success += 1

                log_msg = f"[步骤{self._current_step}] 任务{task.task_id}({task.task_type}) → action={action}, reward={reward:.2f}"

            self._render_log.append(log_msg)

        else:
            # 无任务可调度，轻微惩罚
            reward -= 1.0

        # ---- 等待超时惩罚（遍历队列中所有任务） ----
        reward += self._compute_wait_penalty()

        # ---- 量子比特利用率惩罚 ----
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

    def render(self) -> Optional[Any]:
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
            f"  |  累计奖励: {self._episode_reward:.2f}",
            "-" * 64,
            f"  [量子资源] 可用比率: {self._quantum.available_ratio:.2%}"
            f"  |  保真度: {self._quantum.fidelity:.4f}"
            f"  |  量子队列: {self._quantum.quantum_queue}",
            f"  [经典资源] 负载: {self._classical.load:.2%}"
            f"  |  经典队列: {self._classical.queue}",
            f"  [任务队列]  长度: {len(self._task_queue)}"
            f"  |  已调度: {self._total_scheduled}",
            f"  [统计] 量子成功: {self._quantum_success}"
            f"  |  经典成功: {self._classical_success}"
            f"  |  混合成功: {self._hybrid_success}"
            f"  |  不兼容: {self._mismatch_count}",
            "-" * 64,
        ]

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
            print(output, end="")
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
        生成一个随机任务。

        任务类型从 {"quantum", "classical", "universal"} 中均匀采样。
        量子比特数根据类型设定，紧急程度在 [0, 1] 间随机。

        Args:
            rng     : NumPy 随机数生成器
            task_id : 任务编号

        Returns:
            Task: 生成的随机任务对象
        """
        task_type = rng.choice(["quantum", "classical", "universal"])

        if task_type == "quantum":
            qubit_count = int(rng.integers(5, 50))
        elif task_type == "classical":
            qubit_count = 0
        else:
            qubit_count = int(rng.integers(2, 20))

        return Task(
            task_id=f"T{task_id:04d}",
            task_type=task_type,
            qubit_count=qubit_count,
            wait_steps=0,
            urgency=float(rng.uniform(0.0, 1.0)),
            priority=int(rng.integers(1, 6)),
        )

    # ------------------------------------------------------------------
    # 私有方法：兼容性检查
    # ------------------------------------------------------------------

    def _check_compatibility(self, task: Task, action: int) -> bool:
        """
        检查任务类型与所选资源的兼容性。

        兼容性规则：
            - classical 任务 → 只能分配到经典资源 (action=0)
            - quantum 任务   → 只能分配到量子资源 (action=1) 或混合执行 (action=2)
            - universal 任务 → 三种动作均兼容

        Args:
            task   : 待检查的任务
            action : 智能体选择的动作

        Returns:
            bool: True 表示兼容，False 表示不兼容
        """
        if task.task_type == "classical":
            return action == ACTION_CLASSICAL
        elif task.task_type == "quantum":
            return action in (ACTION_QUANTUM, ACTION_HYBRID)
        else:  # universal
            return True

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

        奖励规则：
            - 经典执行 (action=0) : +3（基准奖励）
            - 量子执行 (action=1) : +10 * 量子加速比（加速比在 [2, 5] 间随机）
            - 混合执行 (action=2) : +7（介于经典和量子之间）

        量子加速比会受到当前保真度的影响：保真度越高，加速比越大。
        当保真度低于 0.9 时，量子执行奖励会打折。

        Args:
            task   : 被执行的任务
            action : 执行方式
            rng    : 随机数生成器

        Returns:
            float: 计算得到的即时奖励
        """
        if action == ACTION_CLASSICAL:
            return REWARD_CLASSICAL

        elif action == ACTION_QUANTUM:
            # 基础加速比在 [2, 5] 之间随机
            speedup = rng.uniform(*QUANTUM_SPEEDUP_RANGE)
            # 保真度加成：保真度越高，加速比越大
            fidelity_factor = self._quantum.fidelity / 0.99  # 归一化到 ~1.0
            speedup *= fidelity_factor
            reward = REWARD_QUANTUM_BASE * speedup
            # 保真度过低时打折
            if self._quantum.fidelity < 0.9:
                reward *= 0.6
            return reward

        else:  # ACTION_HYBRID
            # 混合执行奖励介于经典和量子之间
            base = (REWARD_CLASSICAL + REWARD_QUANTUM_BASE) / 2.0
            # 根据量子资源可用性调整
            hybrid_factor = 0.5 + 0.5 * self._quantum.available_ratio
            return base * hybrid_factor

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
            1. 量子资源：可用比特比率随机波动，保真度缓慢衰减后部分恢复
            2. 经典资源：负载随机波动，任务完成后释放
            3. 队列中任务：每个任务等待步数 +1
            4. 随机生成 0-3 个新任务（泊松分布，均值 1.2）
            5. 随机完成一些正在执行的任务，释放资源
        """
        # 量子资源波动
        self._quantum.available_ratio = np.clip(
            self._quantum.available_ratio + rng.uniform(-0.1, 0.1), 0.05, 1.0
        )
        # 保真度衰减 + 随机恢复
        self._quantum.fidelity = np.clip(
            self._quantum.fidelity - 0.002 + rng.uniform(0.0, 0.01), 0.7, 0.999
        )
        # 量子队列完成一些任务
        completed_q = 0
        for _ in range(self._quantum.quantum_queue):
            if rng.random() < 0.15:  # 15% 概率完成一个量子任务
                completed_q += 1
        self._quantum.quantum_queue = max(0, self._quantum.quantum_queue - completed_q)

        # 经典资源波动
        self._classical.load = np.clip(
            self._classical.load + rng.uniform(-0.15, 0.15), 0.0, 1.0
        )
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
        构建并返回当前 8 维状态向量。

        各维度含义及计算方式：
            [0] qubit_availability  : 量子比特可用比率（直接取值）
            [1] queue_length        : 队列长度 / MAX_QUEUE_SIZE
            [2] avg_wait_time       : 队列中任务平均等待步数 / MAX_WAIT_STEPS
            [3] fidelity            : 量子比特平均保真度
            [4] classical_load     : 经典计算负载（直接取值）
            [5] quantum_queue_ratio : 量子队列 / (量子队列 + 经典队列 + 1)
            [6] time_of_day        : 当前模拟时刻
            [7] urgency_level      : 当前任务紧急程度（无任务时为 0）

        Returns:
            np.ndarray: 形状 (8,)，dtype=float32，值域 [0, 1]
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

        return obs

    # ------------------------------------------------------------------
    # 私有方法：构建信息字典
    # ------------------------------------------------------------------

    def _get_info(self) -> Dict[str, Any]:
        """
        构建环境信息字典，供调试和监控使用。

        Returns:
            dict: 包含当前步数、统计摘要、资源状态等详细信息
        """
        info = {
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

def register_env():
    """
    将 QuantumSchedulingEnv 注册到 Gymnasium 注册表。

    注册后可通过 ``gym.make("QuantumScheduling-v0")`` 创建环境实例。

    Usage::

        from scheduler.env import register_env
        register_env()
        env = gym.make("QuantumScheduling-v0", max_steps=1000)
    """
    from gymnasium.envs.registration import register

    try:
        register(
            id="QuantumScheduling-v0",
            entry_point="src.scheduler.env:QuantumSchedulingEnv",
            max_entry_points=1,
        )
    except gymnasium.error.Error:
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
    for i in range(20):
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
