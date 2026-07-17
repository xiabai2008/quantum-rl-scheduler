"""QuantumSchedulingEnv 核心边界和恢复行为测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import src.scheduler.env as env_module
from src.scheduler.env import (
    ACTION_CLASSICAL,
    ACTION_HYBRID,
    ACTION_QUANTUM,
    MAX_QUEUE_SIZE,
    QuantumSchedulingEnv,
    Task,
    register_env,
)


def _task(task_id: str, task_type: str = "quantum", qubits: int = 2) -> Task:
    """构造字段最少且可识别的调度任务。"""
    return Task(task_id=task_id, task_type=task_type, qubit_count=qubits)


def test_random_pending_task_uses_queue_then_current_then_none() -> None:
    """随机待处理任务应按队列、当前任务、空值顺序降级。"""
    env = QuantumSchedulingEnv(max_steps=5, seed=3)
    env.reset(seed=3)
    queued = _task("queued")
    current = _task("current")
    env._task_queue = [queued]
    env._current_task = current
    assert env.get_random_pending_task() is queued

    env._task_queue = []
    assert env.get_random_pending_task() is current
    env._current_task = None
    assert env.get_random_pending_task() is None


def test_mismatch_does_not_overflow_full_queue() -> None:
    """资源不兼容时，满队列不得再次塞入当前任务。"""
    env = QuantumSchedulingEnv(max_steps=5, seed=4)
    env.reset(seed=4)
    rejected = _task("rejected", task_type="classical", qubits=0)
    env._current_task = rejected
    env._task_queue = [_task(f"queued-{index}") for index in range(MAX_QUEUE_SIZE)]

    _, reward, _, _, _ = env.step(ACTION_QUANTUM)

    assert reward < 0
    assert rejected.wait_steps == 1
    assert all(task.task_id != rejected.task_id for task in env._task_queue)
    assert env._mismatch_count == 1


def test_quantum_unavailable_requeues_task_without_crashing() -> None:
    """没有机器能承接时，纯量子动作应保留任务并给出惩罚。"""
    env = QuantumSchedulingEnv(
        max_steps=5,
        max_qubits=4,
        machine_configs=[
            {
                "name": "small",
                "total_qubits": 4,
                "supported_gates": ("H", "CZ", "M"),
            }
        ],
        seed=5,
    )
    env.reset(seed=5)
    oversized = _task("oversized", qubits=20)
    env._current_task = oversized
    env._task_queue = []

    _, reward, _, _, info = env.step(ACTION_QUANTUM)

    assert reward < 0
    # 重新入队一次，随后仿真时间推进又为队列任务增加一次等待。
    assert oversized.wait_steps == 2
    pending = [env._current_task, *env._task_queue]
    assert oversized in pending
    assert info["total_scheduled"] == 0
    assert info["last_selected_machine"] is None


def test_hybrid_unavailable_falls_back_to_classical() -> None:
    """混合动作缺少量子资源时应回退到经典执行，而非卡住队列。"""
    env = QuantumSchedulingEnv(max_steps=5, max_qubits=2, seed=6)
    env.reset(seed=6)
    hybrid = _task("hybrid", task_type="hybrid", qubits=50)
    env._current_task = hybrid
    env._task_queue = []

    _, _, _, _, info = env.step(ACTION_HYBRID)

    assert info["total_scheduled"] == 1
    assert info["classical_success"] == 1
    assert info["hybrid_success"] == 0
    assert info["last_selected_machine"] is None


def test_empty_queue_step_is_safe_and_terminates() -> None:
    """空队列仍应返回合法状态，并按 max_steps 正常终止。"""
    env = QuantumSchedulingEnv(max_steps=1, seed=7)
    env.reset(seed=7)
    env._task_queue = []
    env._current_task = None

    obs, reward, terminated, truncated, info = env.step(ACTION_CLASSICAL)

    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert terminated is True
    assert truncated is False
    assert info["total_scheduled"] == 0


def test_pending_real_feedback_is_added_to_step_reward(monkeypatch) -> None:
    """存在待轮询真机任务时，非阻塞反馈应合入本步奖励。"""
    env = QuantumSchedulingEnv(max_steps=5, use_real_machine=True, seed=8)
    env.reset(seed=8)
    env._task_queue = []
    env._current_task = None
    env._pending_real_tasks = [{"task_id": "real-1"}]
    poll = MagicMock(return_value=4.5)
    monkeypatch.setattr(env, "_poll_pending_real_tasks", poll)
    env._quantum.available_ratio = 0.0

    _, reward, _, _, _ = env.step(ACTION_CLASSICAL)

    poll.assert_called_once_with()
    assert reward == pytest.approx(3.5)


def test_thin_wrappers_delegate_to_split_modules(monkeypatch) -> None:
    """拆分后的薄包装应保持原实例方法签名和返回值。"""
    env = QuantumSchedulingEnv(max_steps=5, seed=9)
    env.reset(seed=9)
    task = _task("wrapper")
    machine = env._machines[0]
    rng = np.random.default_rng(9)

    monkeypatch.setattr(env_module, "check_compatibility", MagicMock(return_value=True))
    monkeypatch.setattr(env_module, "machine_supports_task", MagicMock(return_value=True))
    monkeypatch.setattr(env_module, "compute_wait_penalty", MagicMock(return_value=-2.0))
    monkeypatch.setattr(env_module, "poll_pending_real_tasks", MagicMock(return_value=1.0))

    assert env._check_compatibility(task, ACTION_QUANTUM) is True
    assert env._machine_supports_task(machine, task) is True
    assert env._compute_wait_penalty() == -2.0
    assert env._poll_pending_real_tasks() == 1.0
    assert env._max_steps == 5
    assert isinstance(rng, np.random.Generator)


def test_register_env_is_idempotent() -> None:
    """重复注册 Gym 环境不应向调用方抛出异常。"""
    register_env()
    register_env()
