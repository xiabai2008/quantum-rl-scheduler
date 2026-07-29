"""量子任务调度环境模块（Gymnasium 接口）。

模块拆分：常量/数据类→env_types.py，奖励→env_reward.py，真机闭环→env_real_machine.py，
渲染→env_render.py，多机器调度→env_machines.py，动态演化→env_dynamics.py，
观测构建→env_observation.py。本文件保留核心类与薄包装，重新导出全部符号以保持向后兼容。
"""

import copy
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

# 从子模块重新导出（向后兼容：from src.scheduler.env import Task, OBS_DIM, ... ）
from src.scheduler.env_dynamics import (
    advance_time,
    check_compatibility,
    generate_random_task,
    pick_next_task,
)
from src.scheduler.env_machines import (
    machine_supports_task,
    recompute_aggregate,
    route_to_machine,
    select_best_machine,
)
from src.scheduler.env_observation import get_info, get_observation
from src.scheduler.env_real_machine import (
    FREE_TIER_MAX_QUBITS,
    poll_pending_real_tasks,
    record_real_failure,
    submit_to_real_machine,
)
from src.scheduler.env_render import close_env, render_env
from src.scheduler.env_reward import (
    compute_execution_reward,
    compute_fairness_penalty,
    compute_wait_penalty,
)
from src.scheduler.env_types import (
    ACTION_CLASSICAL,
    ACTION_HYBRID,
    ACTION_QUANTUM,
    DEFAULT_MACHINE_CONFIGS,
    INITIAL_QUEUE_RANGE,
    MAX_QUEUE_SIZE,
    MAX_STEPS_DEFAULT,
    MAX_WAIT_STEPS,
    OBS_ARRIVAL_RATE_MA,
    OBS_AVG_CONNECTIVITY,
    OBS_AVG_WAIT_TIME,
    OBS_CLASSICAL_LOAD,
    OBS_COUPLING_DENSITY,
    OBS_CROSSTALK_RISK,
    OBS_DIM,
    OBS_DIM_WITH_FAIRNESS,
    OBS_FIDELITY,
    OBS_QUANTUM_QUEUE_RATIO,
    OBS_QUBIT_AVAILABILITY,
    OBS_QUEUE_LENGTH,
    OBS_SINGLE_GATE_FIDELITY,
    OBS_TASK_TYPE_CLASSICAL,
    OBS_TASK_TYPE_QUANTUM,
    OBS_TIME_OF_DAY,
    OBS_TWO_GATE_FIDELITY,
    OBS_URGENCY_LEVEL,
    QUANTUM_SPEEDUP_RANGE,
    QUBIT_UTIL_THRESHOLD,
    REAL_MACHINE_DEGRADE_FAIL_THRESHOLD,
    REAL_MACHINE_FAIL_PENALTY,
    REAL_MACHINE_MAX_POLL_STEPS,
    REAL_MACHINE_MAX_SUBMISSIONS_DEFAULT,
    REAL_MACHINE_SUBMIT_INTERVAL,
    REAL_MACHINE_SUCCESS_BONUS,
    REAL_SUBMIT_PROBABILITY_DEFAULT,
    REWARD_CLASSICAL,
    REWARD_HYBRID,
    REWARD_LOW_QUBIT_UTIL,
    REWARD_MISMATCH,
    REWARD_QUANTUM_BASE,
    REWARD_SUCCESS_BONUS,
    REWARD_WAIT_OVER_THRESHOLD,
    ClassicalResource,
    QuantumMachine,
    QuantumResource,
    RealMachineConfig,
    Task,
)

__all__ = [
    "ACTION_CLASSICAL",
    "ACTION_HYBRID",
    "ACTION_QUANTUM",
    "DEFAULT_MACHINE_CONFIGS",
    "INITIAL_QUEUE_RANGE",
    "MAX_QUEUE_SIZE",
    "MAX_STEPS_DEFAULT",
    "MAX_WAIT_STEPS",
    "OBS_ARRIVAL_RATE_MA",
    "OBS_AVG_CONNECTIVITY",
    "OBS_AVG_WAIT_TIME",
    "OBS_CLASSICAL_LOAD",
    "OBS_COUPLING_DENSITY",
    "OBS_CROSSTALK_RISK",
    "OBS_DIM",
    "OBS_DIM_WITH_FAIRNESS",
    "OBS_FIDELITY",
    "OBS_QUANTUM_QUEUE_RATIO",
    "OBS_QUBIT_AVAILABILITY",
    "OBS_QUEUE_LENGTH",
    "OBS_SINGLE_GATE_FIDELITY",
    "OBS_TASK_TYPE_CLASSICAL",
    "OBS_TASK_TYPE_QUANTUM",
    "OBS_TIME_OF_DAY",
    "OBS_TWO_GATE_FIDELITY",
    "OBS_URGENCY_LEVEL",
    "QUANTUM_SPEEDUP_RANGE",
    "QUBIT_UTIL_THRESHOLD",
    "REAL_MACHINE_DEGRADE_FAIL_THRESHOLD",
    "REAL_MACHINE_FAIL_PENALTY",
    "REAL_MACHINE_MAX_POLL_STEPS",
    "REAL_MACHINE_SUBMIT_INTERVAL",
    "REAL_MACHINE_SUCCESS_BONUS",
    "REAL_SUBMIT_PROBABILITY_DEFAULT",
    "REWARD_CLASSICAL",
    "REWARD_HYBRID",
    "REWARD_LOW_QUBIT_UTIL",
    "REWARD_MISMATCH",
    "REWARD_QUANTUM_BASE",
    "REWARD_SUCCESS_BONUS",
    "REWARD_WAIT_OVER_THRESHOLD",
    "ClassicalResource",
    "QuantumMachine",
    "QuantumResource",
    "QuantumSchedulingEnv",
    "RealMachineConfig",
    "Task",
    "register_env",
]


class QuantumSchedulingEnv(gym.Env[Any, Any]):
    """量子-经典混合计算调度环境（Gymnasium 接口）。

    状态空间 16 维 Box(float32)，动作空间 Discrete(4)。
    详见模块文档与各子模块实现。
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}  # noqa: RUF012

    def __init__(
        self,
        max_steps: int = MAX_STEPS_DEFAULT,
        max_qubits: int = 287,
        render_mode: str | None = None,
        seed: int | None = None,
        machine_configs: list[dict[str, Any]] | None = None,
        real_submit_probability: float = REAL_SUBMIT_PROBABILITY_DEFAULT,
        real_submit_interval: int = REAL_MACHINE_SUBMIT_INTERVAL,
        use_real_machine: bool = False,
        real_machine_feedback_weight: float = 1.0,
        max_real_submissions: int | None = REAL_MACHINE_MAX_SUBMISSIONS_DEFAULT,
        real_machine_shots: int = 512,
        real_feedback_mode: str = "status_only",
        tenant_manager: Any | None = None,
        arrival_lambda: float | Callable[[int, int], float] | None = None,
        quantum_task_ratio: float | None = None,
        real_machine_max_qubits: int = FREE_TIER_MAX_QUBITS,
        noise_profile: str | dict[str, Any] | None = None,
        include_fairness_obs: bool = False,
        observation_dim: int | None = None,
    ):
        """初始化量子任务调度环境（参数详见子模块文档）。"""
        super().__init__()

        self.max_steps = max_steps
        self.max_qubits = max_qubits
        self.render_mode = render_mode
        self.real_submit_probability = float(real_submit_probability)
        # Issue #243: 间隔触发保底（每N步强制提交一次），与概率触发共存
        if real_submit_interval < 1:
            raise ValueError("real_submit_interval must be >= 1")
        self.real_submit_interval = int(real_submit_interval)
        self.use_real_machine = bool(use_real_machine)
        self.real_machine_feedback_weight = float(real_machine_feedback_weight)
        if max_real_submissions is not None and max_real_submissions < 0:
            raise ValueError("max_real_submissions must be non-negative or None")
        if real_machine_shots <= 0:
            raise ValueError("real_machine_shots must be positive")
        if real_machine_max_qubits <= 0:
            raise ValueError("real_machine_max_qubits must be positive")
        self.max_real_submissions = max_real_submissions
        self.real_machine_shots = int(real_machine_shots)
        self.real_machine_max_qubits = int(real_machine_max_qubits)
        # 真机结果反馈模式（Issue #235）：status_only / result_aware / shuffled
        from src.scheduler.env_types import REAL_FEEDBACK_MODES

        if real_feedback_mode not in REAL_FEEDBACK_MODES:
            raise ValueError(
                f"real_feedback_mode must be one of {REAL_FEEDBACK_MODES}, "
                f"got {real_feedback_mode!r}"
            )
        self.real_feedback_mode = real_feedback_mode
        if arrival_lambda is not None and not callable(arrival_lambda):
            if float(arrival_lambda) < 0.0:
                raise ValueError("arrival_lambda must be non-negative")
            arrival_lambda = float(arrival_lambda)
        if quantum_task_ratio is not None and not 0.0 <= float(quantum_task_ratio) <= 1.0:
            raise ValueError("quantum_task_ratio must be between 0 and 1")
        self.arrival_lambda = arrival_lambda
        self.quantum_task_ratio = (
            float(quantum_task_ratio) if quantum_task_ratio is not None else None
        )
        self.noise_profile = self._resolve_noise_profile(noise_profile)

        # Issue #588: 公平性观测开关（不影响默认 OBS_DIM=16，保持向后兼容）
        self._include_fairness_obs = bool(include_fairness_obs)
        # Issue #585: 消融实验支持截断观测空间
        self._observation_dim = observation_dim

        # Gymnasium 标准空间定义（16 维 obs + Discrete(4)，确保 PPO 模型可复用）
        # Issue #585: observation_dim 截断优先级最高
        # Issue #588: include_fairness_obs 扩展到 17 维
        if observation_dim is not None and observation_dim < OBS_DIM:
            eff_dim = observation_dim
        elif self._include_fairness_obs:
            eff_dim = OBS_DIM_WITH_FAIRNESS
        else:
            eff_dim = OBS_DIM
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(eff_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 0: classical, 1: quantum, 2: hybrid, 3: qem

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
        # 保存原始机器配置，供 _create_eval_env 创建独立副本（Issue #399）
        self._machine_configs: list[dict[str, Any]] = copy.deepcopy(machine_configs)
        self._machines: list[QuantumMachine] = [
            QuantumMachine(
                name=cfg.get("name", "tianyan_s"),
                total_qubits=cfg.get("total_qubits", max_qubits),
                supported_gates=tuple(cfg.get("supported_gates", ("H", "CZ", "M"))),
                is_real=bool(cfg.get("is_real", False)),
            )
            for cfg in machine_configs
        ]

        # 缓存机器总量子比特数（Issue #219）
        # _machines 列表在初始化后基本不变，total_qubits 是机器静态属性，
        # 每步重新计算 sum() 是不必要的性能开销。在 __init__ 和 attach_real_clients
        # 时更新此缓存，env_observation.py 直接读取缓存值。
        self._total_qubits_cache: int = sum(m.total_qubits for m in self._machines)

        # 真机客户端映射：machine_name -> client（由 attach_real_clients 注入）
        self._real_clients: dict[str, Any] = {}

        # 真机闭环状态（Issue #64）
        # _pending_real_tasks: 已提交但未拿到结果的真机任务列表
        self._pending_real_tasks: list[dict[str, Any]] = []
        # _real_result_records: 真机结果详细记录（Issue #235 可追溯性）
        self._real_result_records: list[dict[str, Any]] = []
        # _real_feedback_log: 真机因果记录（Issue #235，"RL动作→真机任务→结果→reward"因果链）
        self._real_feedback_log: list[dict[str, Any]] = []
        self._real_machine_degraded: bool = False  # 降级标志：True 时跳过真机提交
        self._real_consecutive_failures: int = 0  # 连续失败计数（触发降级）
        self._real_success_count: int = 0
        self._real_fail_count: int = 0
        # 跨 episode 累积，确保训练级硬上限不会被 reset 绕过。
        self._real_submission_attempts_total: int = 0

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

        # 连续无任务步数（Issue #400）：用于提前终止，减少无效空转步
        self._consecutive_idle_steps: int = 0

        # 用于 ANSI 渲染的日志缓冲区
        self._render_log: list[str] = []

        # 多租户配额管理器（Issue #97）
        self._tenant_manager: Any | None = tenant_manager

        # Issue #587: 公平性跟踪器（可选，用于计算公平性惩罚；默认 None）
        # 通过 set_fairness_tracker() 显式设置；include_fairness_obs 时也可设置
        self._fairness_tracker: Any | None = None

        # LSTM 时序流量感知 (Superpower)
        self.max_arrival_history_length = 10
        self.arrival_history: list[int] = []
        self.current_time_window_arrivals = 0

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
        # 更新 total_qubits 缓存（Issue #219）
        # attach_real_clients 不修改 _machines 列表本身，但保守起见同步缓存
        self._total_qubits_cache = sum(m.total_qubits for m in self._machines)

    def set_fairness_tracker(self, tracker: Any | None) -> None:
        """设置公平性跟踪器，启用奖励函数中的公平性惩罚（Issue #587）。

        Args:
            tracker: MultiTenantFairnessTracker 实例，或 None 清除。
        """
        self._fairness_tracker = tracker

    def inject_noise_profile(self, noise_params: dict[str, Any]) -> None:
        """注入真机噪声参数到仿真环境（Issue #591）。

        将 NoiseModelExtractor.extract_all() 提取的噪声参数注入环境的
        ``noise_profile``，使奖励函数感知真机噪声水平。

        调用示例::

            from src.scheduler.env_real_machine import NoiseModelExtractor

            extractor = NoiseModelExtractor()
            results = extractor.extract_all(
                measurement_results={"0": 0.45, "1": 0.55},
                rb_results=[{"m": 1, "fidelity": 0.99}, ...],
                delay_results=[{"t": 10, "p1": 0.9}, ...],
            )
            env.inject_noise_profile(results)

        注入后，量子执行奖励将根据 readout_error 和 gate_error 折扣，
        混合执行奖励以半权重折扣。T1 参数记录但不参与当前奖励计算。

        Args:
            noise_params: 噪声参数字典，可包含：
                - "readout_error": 读出误差率 (0-1)
                - "gate_error": 平均门误差率 (0-1)
                - "decoherence": T1 拟合结果子字典（含 "t1" 键）
        """
        profile: dict[str, float] = {}

        # readout_error 直接取值
        if "readout_error" in noise_params:
            profile["readout_error"] = float(noise_params["readout_error"])

        # gate_error 直接取值
        if "gate_error" in noise_params:
            profile["gate_error"] = float(noise_params["gate_error"])

        # decoherence 是子字典，提取 t1 值
        decoherence = noise_params.get("decoherence")
        if isinstance(decoherence, dict) and "t1" in decoherence:
            profile["t1"] = float(decoherence["t1"])

        self.noise_profile = profile if profile else None

    @property
    def machine_names(self) -> list[str]:
        """返回当前所有机器名称列表。"""
        return [m.name for m in self._machines]

    @property
    def num_machines(self) -> int:
        """返回量子机器数量。"""
        return len(self._machines)

    def get_random_pending_task(self) -> Task | None:
        """从当前任务队列中随机取一个待处理任务（用于真机抽样提交）。

        优先从 ``_task_queue`` 中随机抽取；队列空时退化为 ``_current_task``；
        两者皆空时返回 ``None``。
        """
        if self._task_queue:
            idx = int(self.np_random.integers(0, len(self._task_queue)))
            return self._task_queue[idx]
        return self._current_task

    def is_real_machine_degraded(self) -> bool:
        """返回真机是否已降级到 Mock。"""
        return self._real_machine_degraded

    def get_real_machine_stats(self) -> dict[str, Any]:
        """返回真机闭环统计信息（供 info 字典和报告使用）。"""
        return {
            "pending_count": len(self._pending_real_tasks),
            "submission_attempts_total": self._real_submission_attempts_total,
            "max_real_submissions": self.max_real_submissions,
            "success_count": self._real_success_count,
            "fail_count": self._real_fail_count,
            "degraded": self._real_machine_degraded,
            "consecutive_failures": self._real_consecutive_failures,
            "noise_profile": self.noise_profile,
        }

    def export_real_feedback_log(self, path: str) -> int:
        """导出真机反馈因果记录为 JSON 文件，返回记录条数（Issue #236）。

        将 ``_real_feedback_log``（"RL动作→真机任务→结果→reward" 完整因果链）
        序列化为 JSON，包含元数据（实验时间、环境参数、seed）和记录列表。

        Args:
            path: 输出 JSON 文件路径。父目录会自动创建。

        Returns:
            导出的因果记录条数（0 表示无真机反馈记录）。
        """
        import json
        from datetime import datetime
        from pathlib import Path

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        records = list(self._real_feedback_log)
        payload = {
            "type": "real_feedback_log",
            "exported_at": datetime.now().astimezone().isoformat(),
            "metadata": {
                "max_steps": self.max_steps,
                "max_qubits": self.max_qubits,
                "real_submit_probability": self.real_submit_probability,
                "use_real_machine": self.use_real_machine,
                "real_machine_feedback_weight": self.real_machine_feedback_weight,
                "real_machine_shots": self.real_machine_shots,
                "real_feedback_mode": self.real_feedback_mode,
                "arrival_lambda": self.arrival_lambda,
                "quantum_task_ratio": self.quantum_task_ratio,
                "num_machines": len(self._machines),
                "machine_names": [m.name for m in self._machines],
            },
            "stats": self.get_real_machine_stats(),
            "record_count": len(records),
            "records": records,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        return len(records)

    def get_tenant_stats(self) -> list[dict[str, Any]]:
        """返回所有租户的配额使用状态；未启用租户管理时返回空列表。"""
        if self._tenant_manager is None:
            return []
        result: list[dict[str, Any]] = self._tenant_manager.get_all_tenants_info()
        return result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[Any], dict[str, Any]]:
        """重置环境：随机初始化任务队列、量子比特状态、经典负载和时间段。"""
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

        # 重置 LSTM 时序流量感知状态
        self.arrival_history = []
        self.current_time_window_arrivals = 0

        # 重置连续无任务步数（Issue #400）
        self._consecutive_idle_steps = 0

        # 重置多机器调度记录
        self._last_selected_machine = None
        self._machine_schedule_count = {m.name: 0 for m in self._machines}
        self._machine_real_submits = {m.name: 0 for m in self._machines}

        # 重置真机闭环状态
        # 注意：success/fail/consecutive_failures 计数器跨 episode 累积，
        # 仅在 __init__ 中初始化，reset 不清零，确保训练汇总统计准确
        self._pending_real_tasks = []
        self._real_result_records = []
        self._real_feedback_log = []

        # 随机初始化任务队列（5-20 个任务）
        self._task_queue = []
        initial_count = rng.integers(INITIAL_QUEUE_RANGE[0], INITIAL_QUEUE_RANGE[1] + 1)
        for i in range(initial_count):
            self._task_queue.append(self._generate_random_task(rng, task_id=i))

        # 随机初始化每台量子机器状态（多机器调度扩展）
        for m in self._machines:
            m.available_ratio = rng.uniform(0.3, 1.0)
            m.fidelity = self._sample_initial_fidelity(rng)
            m.quantum_queue = 0
            m.available = True
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

    def step(self, action: int) -> tuple[NDArray[Any], float, bool, bool, dict[str, Any]]:
        """执行一步调度决策：根据 action 分配到经典/量子/混合资源并计算奖励。"""
        self._current_step += 1
        rng = self.np_random

        # 本步总奖励 = 执行收益 + 队列等待惩罚 + 资源利用率惩罚
        reward = 0.0

        # Issue #522 性能优化：本步只构建一次观测并缓存复用。
        # 原实现中 step() 兼容分支、_compute_execution_reward()、返回值各调用一次
        # _get_observation()，单步共 3 次观测构建（含 numpy 向量化加权平均），开销大。
        # 缓存后保证单步内观测一致，并将构建次数降为 1。
        obs = self._get_observation()

        if self._current_task is not None:
            task = self._current_task
            is_compatible = self._check_compatibility(task, action)

            if not is_compatible:
                # 不兼容：扣分并重新入队
                reward += REWARD_MISMATCH
                self._mismatch_count += 1
                task.wait_steps += 1
                if len(self._task_queue) < MAX_QUEUE_SIZE:
                    self._task_queue.append(task)
                self._last_selected_machine = None
                log_msg = (
                    f"[步骤{self._current_step}] 任务{task.task_id} 分配到不兼容资源"
                    f"(action={action})，惩罚{REWARD_MISMATCH}"
                )
            else:
                # 兼容分配：为量子任务选择最佳机器
                quantum_action = action in (ACTION_QUANTUM, ACTION_HYBRID)
                selected_machine = None
                if quantum_action:
                    selected_machine = self._select_best_machine(task)
                # 量子不可用 = 需要量子但无机器能接下任务
                quantum_unavailable = quantum_action and selected_machine is None

                if quantum_unavailable:
                    if action == ACTION_QUANTUM:
                        # 纯量子动作：任务重新排队，半个 mismatch 惩罚
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
                        # 混合动作：降级为经典执行，避免系统空转
                        reward += self._compute_execution_reward(task, ACTION_CLASSICAL, rng, obs)
                        self._total_scheduled += 1
                        self._classical_success += 1
                        self._last_selected_machine = None
                        log_msg = (
                            f"[步骤{self._current_step}] 量子资源不可用，"
                            f"混合任务{task.task_id}降级为经典执行，reward={reward:.2f}"
                        )
                else:
                    # 兼容分配：计算执行奖励（复用步首缓存的 obs，避免重复构建观测）
                    crosstalk_risk = obs[OBS_CROSSTALK_RISK]
                    crosstalk_penalty = crosstalk_risk * 2.0

                    reward += (
                        self._compute_execution_reward(task, action, rng, obs) - crosstalk_penalty
                    )
                    self._total_scheduled += 1

                    # 构建观测快照（Issue #234）：记录关键状态字段用于因果追溯
                    _obs_snapshot: dict[str, Any] = {
                        "queue_length": len(self._task_queue),
                        "qubit_avail": self._quantum.available_ratio,
                        "fidelity": self._quantum.fidelity,
                        "classical_load": self._classical.load,
                        "quantum_queue": self._quantum.quantum_queue,
                    }

                    if action == ACTION_QUANTUM:
                        self._quantum_success += 1
                        self._route_to_machine(
                            selected_machine,
                            task,
                            rng,
                            rl_action=action,
                            observation_snapshot=_obs_snapshot,
                        )
                    elif action == ACTION_CLASSICAL:
                        self._classical_success += 1
                        self._last_selected_machine = None
                    else:
                        self._hybrid_success += 1
                        self._route_to_machine(
                            selected_machine,
                            task,
                            rng,
                            rl_action=action,
                            observation_snapshot=_obs_snapshot,
                        )

                    machine_tag = (
                        f"@{selected_machine.name}" if selected_machine is not None else ""
                    )
                    log_msg = (
                        f"[步骤{self._current_step}] 任务{task.task_id}({task.task_type})"
                        f" → action={action}{machine_tag}, reward={reward:.2f}"
                    )

            self._render_log.append(log_msg)

            # 有任务可调度，重置连续空转计数器（Issue #400）
            self._consecutive_idle_steps = 0

        else:
            # 无任务可调度，轻微惩罚
            reward -= 1.0
            # 追踪连续无任务步数（Issue #400）
            self._consecutive_idle_steps += 1

        # 等待超时惩罚（全局队列惩罚）
        reward += self._compute_wait_penalty()

        # 量子比特利用率惩罚（利用率低于 30% 时扣分）
        if self._quantum.available_ratio > (1.0 - QUBIT_UTIL_THRESHOLD):
            reward += REWARD_LOW_QUBIT_UTIL

        # 真机闭环反馈（Issue #64）：非阻塞轮询已提交真机任务结果
        if self.use_real_machine and self._pending_real_tasks:
            reward += self._poll_pending_real_tasks()

        # 推进仿真时间
        self._advance_time(rng)

        # 推进时间段（昼夜循环）
        self._time_of_day = (self._time_of_day + 1.0 / self.max_steps) % 1.0

        # 取出下一个任务
        self._pick_next_task()

        # 累计奖励
        self._episode_reward += reward

        # 判断终止（Issue #400: 修复 Gymnasium 语义）
        # terminated=True → 自然终止（连续无任务，环境无意义继续），不 Bootstrap
        # truncated=True → 因 max_steps 外部限制截断（任务可能未完成），需要 Bootstrap
        idle_termination_threshold = 10
        truncated = self._current_step >= self.max_steps
        terminated = (not truncated) and (
            self._consecutive_idle_steps >= idle_termination_threshold
        )

        # Issue #522: 步首缓存的 obs 用于奖励计算，步尾重新构建观测以反映
        # advance_time 后的状态（如 arrival_rate_ma 已更新），保证返回给 Agent
        # 的观测与原始实现语义一致（原实现返回 advance_time 后的观测）。
        return_obs = self._get_observation()
        return return_obs, reward, terminated, truncated, self._get_info()

    # -- 薄包装方法：委托给子模块，保留实例方法签名以兼容现有测试 --

    def render(self) -> Any | None:
        return render_env(self)

    def close(self) -> None:
        close_env(self)

    @staticmethod
    def _resolve_noise_profile(
        profile: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        """解析保真度噪声模型配置（Issue #456 噪声反馈 v2）。

        Args:
            profile: 噪声模型名称或自定义参数字典。
                - None/"uniform": 默认均匀分布 U(0.85, 0.99)
                - "real_machine": 真机噪声分布 Beta(μ=0.886, σ=0.087) 截断到 [0.671, 0.994]
                - dict: 自定义参数 {distribution, mean, std, low, high}

        Returns:
            标准化的噪声配置字典。
        """
        if profile is None or profile == "uniform":
            return {"distribution": "uniform", "low": 0.85, "high": 0.99}
        if profile == "real_machine":
            return _beta_params_from_mean_std(0.8863, 0.0874, 0.671, 0.994)
        if isinstance(profile, dict):
            return dict(profile)
        raise ValueError(f"Unknown noise_profile: {profile!r}")

    def _sample_initial_fidelity(self, rng: np.random.Generator) -> float:
        """根据 noise_profile 采样初始保真度。"""
        p = self.noise_profile
        if p is None:
            return float(rng.uniform(0.85, 0.99))
        dist = p.get("distribution")
        if dist == "uniform":
            return float(rng.uniform(p["low"], p["high"]))
        if dist == "beta":
            for _ in range(100):
                val = rng.beta(p["a"], p["b"])
                val = val * (p["high"] - p["low"]) + p["low"]
                if p["low"] <= val <= p["high"]:
                    return float(val)
            return float(p["mean"])
        return float(rng.uniform(0.85, 0.99))

    def _generate_random_task(self, rng: np.random.Generator, task_id: int) -> Task:
        return generate_random_task(rng, task_id, self.quantum_task_ratio)

    def _get_arrival_lambda(self) -> float:
        """返回当前步的泊松到达率，供多负载评估使用。"""
        if self.arrival_lambda is None:
            return 1.2
        if callable(self.arrival_lambda):
            value = float(self.arrival_lambda(self._current_step, self.max_steps))
            if value < 0.0:
                raise ValueError("arrival_lambda schedule returned a negative value")
            return value
        return float(self.arrival_lambda)

    def _check_compatibility(self, task: Task, action: int) -> bool:
        return check_compatibility(task, action)

    def _select_best_machine(self, task: Task) -> QuantumMachine | None:
        return select_best_machine(self, task)

    def _machine_supports_task(self, machine: QuantumMachine, task: Task) -> bool:
        return machine_supports_task(machine, task)

    def _route_to_machine(
        self,
        machine: QuantumMachine | None,
        task: Task,
        rng: np.random.Generator,
        rl_action: int = -1,
        rl_action_prob: float = 0.0,
        observation_snapshot: dict[str, Any] | None = None,
    ) -> None:
        route_to_machine(self, machine, task, rng, rl_action, rl_action_prob, observation_snapshot)

    def _submit_to_real_machine(
        self,
        machine: QuantumMachine,
        task: Task,
        rl_action: int = -1,
        rl_action_prob: float = 0.0,
        observation_snapshot: dict[str, Any] | None = None,
    ) -> None:
        submit_to_real_machine(self, machine, task, rl_action, rl_action_prob, observation_snapshot)

    def _record_real_failure(self, machine_name: str, reason: str) -> None:
        record_real_failure(self, machine_name, reason)

    def _poll_pending_real_tasks(self) -> float:
        return poll_pending_real_tasks(self)

    def _recompute_aggregate(self) -> None:
        recompute_aggregate(self)

    def _compute_execution_reward(
        self,
        task: Task,
        action: int,
        rng: np.random.Generator,
        obs: NDArray[Any] | None = None,
        fairness_penalty: float | None = None,
    ) -> float:
        """计算执行奖励（委托给 env_reward.compute_execution_reward）。

        Args:
            task: 待执行任务。
            action: 调度动作（ACTION_CLASSICAL / ACTION_QUANTUM / ACTION_HYBRID）。
            rng: 随机数生成器。
            obs: 步首缓存的全局观测向量。若提供则直接读取串扰风险，避免重复构建观测
                （Issue #522 性能优化）；若为 None 则回退到即时构建。
            fairness_penalty: 公平性惩罚值（Issue #587）。为 None 时自动从
                ``_fairness_tracker`` 计算；非 None 时直接使用传入值。

        Returns:
            执行奖励标量。
        """
        # Issue #522: 优先复用步首缓存的观测，避免在奖励计算中重复调用 _get_observation()
        if obs is None:
            obs = self._get_observation()
        crosstalk_risk = obs[OBS_CROSSTALK_RISK]
        crosstalk_penalty = crosstalk_risk * 2.0  # 惩罚因子可调

        # Issue #587: 公平性惩罚嵌入奖励函数
        # 若未显式传入 fairness_penalty，则根据 tenant_manager 是否存在决定是否计算
        if fairness_penalty is None:
            fairness_penalty = self._compute_fairness_penalty_for_task(task)

        return compute_execution_reward(
            task=task,
            action=action,
            rng=rng,
            quantum_fidelity=self._quantum.fidelity,
            quantum_available_ratio=self._quantum.available_ratio,
            crosstalk_penalty=crosstalk_penalty,
            fairness_penalty=fairness_penalty,
            noise_profile=self.noise_profile,
        )

    def _compute_fairness_penalty_for_task(self, task: Task) -> float:
        """计算单个任务的公平性惩罚（Issue #587）。

        当 ``_fairness_tracker`` 为 None 时（未设置跟踪器），返回 0.0。
        否则从 ``_fairness_tracker`` 提取各租户的平均等待时间字典，
        调用 ``compute_fairness_penalty`` 计算惩罚。

        Args:
            task: 待执行任务（使用 tenant_id 字段）

        Returns:
            公平性惩罚值（非正数）
        """
        # Issue #587: 未设置公平性跟踪器时无惩罚
        if self._fairness_tracker is None:
            return 0.0
        # 从 fairness_tracker 提取各租户的平均等待时间字典
        wait_times = self._fairness_tracker.get_wait_times_dict()
        tenant_id = getattr(task, "tenant_id", None)
        return compute_fairness_penalty(
            tenant_id=tenant_id,
            fairness_wait_times=wait_times,
        )

    def _compute_wait_penalty(self) -> float:
        return compute_wait_penalty(self._task_queue)

    def _advance_time(self, rng: np.random.Generator) -> None:
        advance_time(self, rng)

    def _pick_next_task(self) -> None:
        pick_next_task(self)

    def _get_observation(self) -> NDArray[Any]:
        return get_observation(self)

    def _get_info(self) -> dict[str, Any]:
        return get_info(self)

    @property
    def _max_steps(self) -> int:
        return self.max_steps


def _beta_params_from_mean_std(mean: float, std: float, low: float, high: float) -> dict[str, Any]:
    """将均值/标准差映射到 Beta 分布参数 a, b（已线性缩放到 [low, high]）。

    方法矩估计：μ = a/(a+b), σ² = ab/((a+b)²(a+b+1))
    """
    range_ = high - low
    if range_ <= 0:
        return {"distribution": "beta", "a": 1.0, "b": 1.0, "low": low, "high": high, "mean": mean}
    m = (mean - low) / range_
    v = (std / range_) ** 2
    if v >= m * (1 - m):
        v = m * (1 - m) * 0.9
    a = m * (m * (1 - m) / v - 1)
    b = (1 - m) * (m * (1 - m) / v - 1)
    a = max(a, 0.5)
    b = max(b, 0.5)
    return {
        "distribution": "beta",
        "a": float(a),
        "b": float(b),
        "low": float(low),
        "high": float(high),
        "mean": float(mean),
        "std": float(std),
    }


def register_env() -> None:
    """将 QuantumSchedulingEnv 注册到 Gymnasium 注册表。"""
    from gymnasium.envs.registration import register

    try:  # noqa: SIM105
        register(
            id="QuantumScheduling-v0",
            entry_point="src.scheduler.env:QuantumSchedulingEnv",
        )
    except gym.error.Error:
        pass  # 已注册，忽略
