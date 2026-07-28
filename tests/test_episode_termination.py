"""
量子RL调度系统 - PPO episode 终止与自动重置路径测试

Issue #518: 补齐 episode termination 与 auto-reset 路径的测试覆盖。

测试覆盖：
- max_steps 截断 (truncated=True)：到达最大步数时截断，terminated=False
- 全部任务完成后的自然终止 (terminated=True)：队列清空后连续空闲触发终止
- 自动重置后产生有效观测：episode 结束后 reset() 返回合法 observation
- info 字典中包含 completion_rate：终止时 info 含完成率字段
- 连续空闲步数触发终止 (_consecutive_idle_steps 逻辑)：阈值 10 步

测试风格：pytest 函数式，中文注释，seed=42 保证可复现
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.env_types import OBS_DIM

# 与 env.step() 中硬编码的 idle_termination_threshold 保持一致
IDLE_TERMINATION_THRESHOLD = 10


def test_episode_truncated_at_max_steps():
    """测试 episode 在 max_steps 时截断 (truncated=True)。

    通过设置较高的 arrival_lambda 确保队列持续有任务到达，
    使 episode 不会因空闲提前终止，而是到达 max_steps 截断。
    """
    # arrival_lambda=5.0 保证每步平均到达 5 个新任务，队列不会清空
    env = QuantumSchedulingEnv(max_steps=50, seed=42, arrival_lambda=5.0)
    env.reset(seed=42)

    # 额外填充任务队列，确保 50 步内不会出现连续空闲
    for i in range(60):
        env._task_queue.append(env._generate_random_task(env.np_random, task_id=10000 + i))

    terminated = False
    truncated = False
    for _ in range(env.max_steps + 5):
        _, _, terminated, truncated, _ = env.step(2)  # action=2 hybrid，全兼容
        if terminated or truncated:
            break

    assert truncated is True, "应在 max_steps 时截断"
    assert terminated is False, "截断时 terminated 必须为 False"
    assert env._current_step == env.max_steps, "截断时步数应等于 max_steps"
    # 确保没有触发空闲终止逻辑
    assert env._consecutive_idle_steps < IDLE_TERMINATION_THRESHOLD


def test_episode_terminated_when_all_tasks_complete():
    """测试全部任务完成后 episode 自然终止 (terminated=True)。

    通过设置 arrival_lambda=0.0 禁止新任务到达，使用全兼容动作 action=2
    快速调度完所有任务。队列清空后连续空闲 10 步触发 terminated=True。
    """
    # arrival_lambda=0.0 禁止新任务到达，队列只减不增
    env = QuantumSchedulingEnv(max_steps=50, seed=42, arrival_lambda=0.0)
    env.reset(seed=42)

    terminated = False
    truncated = False
    for _ in range(env.max_steps + IDLE_TERMINATION_THRESHOLD + 5):
        _, _, terminated, truncated, _ = env.step(2)  # action=2 hybrid，全兼容
        if terminated or truncated:
            break

    assert terminated is True, "全部任务完成后应触发 terminated=True"
    assert truncated is False, "自然终止时 truncated 必须为 False"
    # 终止时应发生在 max_steps 之前
    assert env._current_step < env.max_steps
    # 队列和当前任务都应为空
    assert len(env._task_queue) == 0
    assert env._current_task is None


def test_auto_reset_produces_valid_observation():
    """测试 episode 结束后自动重置产生有效的新观测。

    验证 reset() 返回的观测：
    - 形状为 (OBS_DIM,)
    - dtype 为 float32
    - 所有值在 observation_space 范围 [0, 1] 内
    - 与上一个 episode 的最终观测不同（确认是全新状态）
    """
    env = QuantumSchedulingEnv(max_steps=50, seed=42, arrival_lambda=0.0)
    env.reset(seed=42)

    # 运行至 episode 结束
    for _ in range(env.max_steps + IDLE_TERMINATION_THRESHOLD + 5):
        obs_final, _, terminated, truncated, _ = env.step(2)
        if terminated or truncated:
            break

    assert terminated or truncated, "episode 应已结束"

    # 模拟 PPO 训练循环中的 auto-reset：episode 结束后调用 reset()
    obs_reset, info_reset = env.reset(seed=123)

    # 观测形状与 dtype 校验
    assert obs_reset.shape == (OBS_DIM,), f"观测形状应为 ({OBS_DIM},)"
    assert obs_reset.dtype == np.float32, "观测 dtype 应为 float32"
    # 观测值域校验
    assert np.all(obs_reset >= 0.0), "观测所有维度应 >= 0.0"
    assert np.all(obs_reset <= 1.0), "观测所有维度应 <= 1.0"
    # 确认 reset 产生了新状态（与最终观测不同）
    assert not np.array_equal(obs_reset, obs_final), "reset 应产生新的观测状态"
    # info 是有效字典
    assert isinstance(info_reset, dict)
    assert info_reset["current_step"] == 0, "reset 后步数应为 0"


def test_info_contains_completion_rate_after_termination():
    """测试终止时 info 字典包含 completion_rate 字段。

    completion_rate = total_scheduled / (total_scheduled + queue_length)，
    应为 [0.0, 1.0] 范围内的浮点数。
    """
    env = QuantumSchedulingEnv(max_steps=50, seed=42, arrival_lambda=0.0)
    env.reset(seed=42)

    final_info: dict = {}
    for _ in range(env.max_steps + IDLE_TERMINATION_THRESHOLD + 5):
        _, _, terminated, truncated, final_info = env.step(2)
        if terminated or truncated:
            break

    assert "completion_rate" in final_info, "info 应包含 completion_rate 字段"
    cr = final_info["completion_rate"]
    assert isinstance(cr, float), "completion_rate 应为 float 类型"
    assert 0.0 <= cr <= 1.0, f"completion_rate 应在 [0, 1] 范围内，实际: {cr}"
    # 全部任务完成后终止，完成率应为 1.0（队列已空）
    assert cr == pytest.approx(1.0), "队列清空后 completion_rate 应为 1.0"


def test_consecutive_idle_steps_trigger_termination():
    """测试连续空闲步数 (_consecutive_idle_steps) 达到阈值时触发终止。

    直接清空队列并设置 _current_task=None，模拟无任务可调度的场景。
    每步 _consecutive_idle_steps 递增，达到阈值 10 时 terminated=True。
    """
    env = QuantumSchedulingEnv(max_steps=50, seed=42, arrival_lambda=0.0)
    env.reset(seed=42)

    # 手动清空队列与当前任务，模拟所有任务已完成的状态
    env._task_queue.clear()
    env._current_task = None
    env._consecutive_idle_steps = 0

    # 前 9 步：_consecutive_idle_steps 递增但未达阈值，不应终止
    for i in range(IDLE_TERMINATION_THRESHOLD - 1):
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated, f"第 {i + 1} 步不应触发 terminated（未达阈值）"
        assert not truncated, "未达 max_steps，不应触发 truncated"
        assert env._consecutive_idle_steps == i + 1, (
            f"_consecutive_idle_steps 应为 {i + 1}，"
            f"实际: {env._consecutive_idle_steps}"
        )

    # 第 10 步：达到阈值，应触发 terminated=True
    _, _, terminated, truncated, info = env.step(0)
    assert terminated is True, "连续空闲 10 步应触发 terminated=True"
    assert truncated is False, "未达 max_steps，truncated 应为 False"
    assert env._consecutive_idle_steps == IDLE_TERMINATION_THRESHOLD
    # info 中仍应包含 completion_rate
    assert "completion_rate" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
