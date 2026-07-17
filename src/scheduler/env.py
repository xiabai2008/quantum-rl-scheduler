"""量子任务调度环境模块（Gymnasium 接口）。

模块拆分：常量/数据类→env_types.py，奖励→env_reward.py，真机闭环→env_real_machine.py，
渲染→env_render.py，多机器调度→env_machines.py，动态演化→env_dynamics.py，
观测构建→env_observation.py。本文件保留核心类与薄包装，重新导出全部符号以保持向后兼容。
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

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
    poll_pending_real_tasks,
    record_real_failure,
    submit_to_real_machine,
)
from src.scheduler.env_render import close_env, render_env
from src.scheduler.env_reward import compute_execution_reward, compute_wait_penalty
from src.scheduler.env_types import (
    ACTION_CLASSICAL,
    ACTION_HYBRID,
    ACTION_QUANTUM,
    DEFAULT_MACHINE_CONFIGS,
    INITIAL_QUEUE_RANGE,
    MAX_QUEUE_SIZE,
    MAX_STEPS_DEFAULT,
    MAX_WAIT_STEPS,
    OBS_AVG_CONNECTIVITY,
    OBS_AVG_WAIT_TIME,
    OBS_CLASSICAL_LOAD,
    OBS_COUPLING_DENSITY,
    OBS_DIM,
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
    Task,
)

__all__ = [
    "ACTION_CLASSICAL", "ACTION_HYBRID", "ACTION_QUANTUM", "DEFAULT_MACHINE_CONFIGS",
    "INITIAL_QUEUE_RANGE", "MAX_QUEUE_SIZE", "MAX_STEPS_DEFAULT", "MAX_WAIT_STEPS",
    "OBS_AVG_CONNECTIVITY", "OBS_AVG_WAIT_TIME", "OBS_CLASSICAL_LOAD",
    "OBS_COUPLING_DENSITY", "OBS_DIM", "OBS_FIDELITY", "OBS_QUANTUM_QUEUE_RATIO",
    "OBS_QUBIT_AVAILABILITY", "OBS_QUEUE_LENGTH", "OBS_SINGLE_GATE_FIDELITY",
    "OBS_TASK_TYPE_CLASSICAL", "OBS_TASK_TYPE_QUANTUM", "OBS_TIME_OF_DAY",
    "OBS_TWO_GATE_FIDELITY", "OBS_URGENCY_LEVEL", "QUANTUM_SPEEDUP_RANGE",
    "QUBIT_UTIL_THRESHOLD", "REAL_MACHINE_DEGRADE_FAIL_THRESHOLD", "REAL_MACHINE_FAIL_PENALTY",
    "REAL_MACHINE_MAX_POLL_STEPS", "REAL_MACHINE_SUCCESS_BONUS", "REAL_SUBMIT_PROBABILITY_DEFAULT",
    "REWARD_CLASSICAL", "REWARD_HYBRID", "REWARD_LOW_QUBIT_UTIL", "REWARD_MISMATCH",
    "REWARD_QUANTUM_BASE", "REWARD_SUCCESS_BONUS", "REWARD_WAIT_OVER_THRESHOLD",
    "ClassicalResource", "QuantumMachine", "QuantumResource", "QuantumSchedulingEnv",
    "Task", "register_env",
]


class QuantumSchedulingEnv(gym.Env):
    """量子-经典混合计算调度环境（Gymnasium 接口）。

    状态空间 14 维 Box(float32)，动作空间 Discrete(3)。
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
        use_real_machine: bool = False,
        real_machine_feedback_weight: float = 1.0,
        tenant_manager: Any | None = None,
    ):
        """初始化量子任务调度环境（参数详见子模块文档）。"""
        super().__init__()

        self.max_steps = max_steps
        self.max_qubits = max_qubits
        self.render_mode = render_mode
        self.real_submit_probability = float(real_submit_probability)
        self.use_real_machine = bool(use_real_machine)
        self.real_machine_feedback_weight = float(real_machine_feedback_weight)

        # Gymnasium 标准空间定义（保持 14 维 obs + Discrete(3) 不变，确保 PPO 模型可复用）
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
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

        # 真机闭环状态（Issue #64）
        # _pending_real_tasks: 已提交但未拿到结果的真机任务列表
        self._pending_real_tasks: list[dict[str, Any]] = []
        self._real_machine_degraded: bool = False  # 降级标志：True 时跳过真机提交
        self._real_consecutive_failures: int = 0  # 连续失败计数（触发降级）
        self._real_success_count: int = 0
        self._real_fail_count: int = 0

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

        # 多租户配额管理器（Issue #97）
        self._tenant_manager: Any | None = tenant_manager

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
            "success_count": self._real_success_count,
            "fail_count": self._real_fail_count,
            "degraded": self._real_machine_degraded,
            "consecutive_failures": self._real_consecutive_failures,
        }

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
    ) -> tuple[np.ndarray, dict[str, Any]]:
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

        # 重置多机器调度记录
        self._last_selected_machine = None
        self._machine_schedule_count = {m.name: 0 for m in self._machines}
        self._machine_real_submits = {m.name: 0 for m in self._machines}

        # 重置真机闭环状态
        # 注意：success/fail/consecutive_failures 计数器跨 episode 累积，
        # 仅在 __init__ 中初始化，reset 不清零，确保训练汇总统计准确
        self._pending_real_tasks = []

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

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """执行一步调度决策：根据 action 分配到经典/量子/混合资源并计算奖励。"""
        self._current_step += 1
        rng = self.np_random

        # 本步总奖励 = 执行收益 + 队列等待惩罚 + 资源利用率惩罚
        reward = 0.0

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
                        reward += self._compute_execution_reward(task, ACTION_CLASSICAL, rng)
                        self._total_scheduled += 1
                        self._classical_success += 1
                        self._last_selected_machine = None
                        log_msg = (
                            f"[步骤{self._current_step}] 量子资源不可用，"
                            f"混合任务{task.task_id}降级为经典执行，reward={reward:.2f}"
                        )
                else:
                    # 兼容分配：计算执行奖励
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

        # 判断终止
        terminated = self._current_step >= self.max_steps
        truncated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # -- 薄包装方法：委托给子模块，保留实例方法签名以兼容现有测试 --

    def render(self) -> Any | None:
        return render_env(self)

    def close(self) -> None:
        close_env(self)

    def _generate_random_task(self, rng: np.random.Generator, task_id: int) -> Task:
        return generate_random_task(rng, task_id)

    def _check_compatibility(self, task: Task, action: int) -> bool:
        return check_compatibility(task, action)

    def _select_best_machine(self, task: Task) -> QuantumMachine | None:
        return select_best_machine(self, task)

    def _machine_supports_task(self, machine: QuantumMachine, task: Task) -> bool:
        return machine_supports_task(machine, task)

    def _route_to_machine(
        self, machine: QuantumMachine | None, task: Task, rng: np.random.Generator
    ) -> None:
        route_to_machine(self, machine, task, rng)

    def _submit_to_real_machine(self, machine: QuantumMachine, task: Task) -> None:
        submit_to_real_machine(self, machine, task)

    def _record_real_failure(self, machine_name: str, reason: str) -> None:
        record_real_failure(self, machine_name, reason)

    def _poll_pending_real_tasks(self) -> float:
        return poll_pending_real_tasks(self)

    def _recompute_aggregate(self) -> None:
        recompute_aggregate(self)

    def _compute_execution_reward(
        self, task: Task, action: int, rng: np.random.Generator
    ) -> float:
        return compute_execution_reward(
            task=task, action=action, rng=rng,
            quantum_fidelity=self._quantum.fidelity,
            quantum_available_ratio=self._quantum.available_ratio,
        )

    def _compute_wait_penalty(self) -> float:
        return compute_wait_penalty(self._task_queue)

    def _advance_time(self, rng: np.random.Generator) -> None:
        advance_time(self, rng)

    def _pick_next_task(self) -> None:
        pick_next_task(self)

    def _get_observation(self) -> np.ndarray:
        return get_observation(self)

    def _get_info(self) -> dict[str, Any]:
        return get_info(self)

    @property
    def _max_steps(self) -> int:
        return self.max_steps


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
