"""MAPPO 多机协调的故障与边界场景测试。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import src.scheduler.marl as marl_module
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumMachine, QuantumSchedulingEnv, Task
from src.scheduler.marl import MultiAgentEnvWrapper, MultiAgentPPO, RolloutBuffer


def _env(machine_count: int = 2) -> QuantumSchedulingEnv:
    """创建固定数量机器的短回合环境。"""
    return QuantumSchedulingEnv(
        max_steps=8,
        machine_configs=DEFAULT_MACHINE_CONFIGS[:machine_count],
        seed=42,
    )


def test_wrapper_refreshes_dynamically_joined_machine() -> None:
    """运行期间加入机器后，包装器应能刷新观测与名称映射。"""
    env = _env(1)
    wrapper = MultiAgentEnvWrapper(env)
    env.reset(seed=42)
    env._machines.append(
        QuantumMachine(
            name="dynamic-machine",
            total_qubits=32,
            available_ratio=0.8,
            fidelity=0.96,
            supported_gates=("H", "CZ", "M"),
        )
    )

    assert wrapper.refresh_machines() is True
    assert wrapper.refresh_machines() is False
    assert wrapper.num_agents == 2
    assert wrapper.machine_names[-1] == "dynamic-machine"
    observations = wrapper.get_local_observations()
    assert set(observations) == set(wrapper.machine_names)


def test_offline_and_missing_votes_fall_back_to_classical() -> None:
    """机器全部离线或通信缺失时应稳定回退到经典动作。"""
    env = _env(2)
    wrapper = MultiAgentEnvWrapper(env)
    env.reset(seed=1)
    for machine in env._machines:
        machine.available = False

    assert wrapper.aggregate_actions({}) == (0, None)
    assert wrapper.aggregate_actions(dict.fromkeys(wrapper.machine_names, 2)) == (0, None)


def test_quantum_vote_has_priority_over_hybrid_vote() -> None:
    """同时存在量子和混合投票时应优先选择量子动作。"""
    env = _env(2)
    wrapper = MultiAgentEnvWrapper(env)
    env.reset(seed=2)
    for machine in env._machines:
        machine.available = True

    action, chosen = wrapper.aggregate_actions(
        {wrapper.machine_names[0]: 2, wrapper.machine_names[1]: 1}
    )

    assert action == 1
    assert chosen == 1


@pytest.mark.parametrize(
    ("available", "ratio", "qubits", "gate_result", "expected"),
    [
        (False, 1.0, 1, True, False),
        (True, 0.1, 100, True, False),
        (True, 1.0, 1, False, False),
        (True, 1.0, 1, True, True),
    ],
)
def test_machine_can_handle_all_filter_branches(
    available: bool,
    ratio: float,
    qubits: int,
    gate_result: bool,
    expected: bool,
    monkeypatch,
) -> None:
    """离线、容量和门集合过滤应逐层生效。"""
    env = _env(1)
    wrapper = MultiAgentEnvWrapper(env)
    machine = env._machines[0]
    machine.available = available
    machine.available_ratio = ratio
    monkeypatch.setattr(env, "_machine_supports_task", MagicMock(return_value=gate_result))

    assert wrapper._machine_can_handle(0, Task("edge", "quantum", qubits)) is expected


def test_step_restores_machine_state_when_environment_raises(monkeypatch) -> None:
    """底层通信异常时，临时路由状态必须在 finally 中恢复。"""
    env = _env(2)
    wrapper = MultiAgentEnvWrapper(env)
    env.reset(seed=3)
    env._current_task = Task("route", "quantum", 1)
    original = [True, False]
    for machine, state in zip(env._machines, original, strict=True):
        machine.available = state
    monkeypatch.setattr(wrapper, "aggregate_actions", MagicMock(return_value=(1, 0)))
    monkeypatch.setattr(wrapper, "_machine_can_handle", MagicMock(return_value=True))
    monkeypatch.setattr(env, "step", MagicMock(side_effect=ConnectionError("offline")))

    with pytest.raises(ConnectionError, match="offline"):
        wrapper.step(dict.fromkeys(wrapper.machine_names, 1))

    assert [machine.available for machine in env._machines] == original


def test_step_without_compatible_choice_uses_environment_fallback(monkeypatch) -> None:
    """Agent 误选不兼容机器时不应强制修改机器在线状态。"""
    env = _env(2)
    wrapper = MultiAgentEnvWrapper(env)
    env.reset(seed=4)
    original = [machine.available for machine in env._machines]
    monkeypatch.setattr(wrapper, "aggregate_actions", MagicMock(return_value=(2, 0)))
    monkeypatch.setattr(wrapper, "_machine_can_handle", MagicMock(return_value=False))
    monkeypatch.setattr(
        env,
        "step",
        MagicMock(
            return_value=(
                np.zeros(14, dtype=np.float32),
                1.25,
                False,
                False,
                {"completion_rate": 0.0},
            )
        ),
    )

    _, reward, terminated, truncated, info = wrapper.step({})

    assert reward == 1.25
    assert terminated is False
    assert truncated is False
    assert info["env_action"] == 2
    assert info["chosen_machine"] == wrapper.machine_names[0]
    assert [machine.available for machine in env._machines] == original


def _buffer(capacity: int = 1) -> RolloutBuffer:
    """创建双 Agent 最小缓冲区。"""
    return RolloutBuffer(capacity, num_agents=2, local_obs_dim=3, global_state_dim=6)


def _add_valid(buffer: RolloutBuffer, reward: float = 1.0, value: float = 0.5) -> None:
    """写入一个合法时间步。"""
    buffer.add(
        local_obs=[np.zeros(3, dtype=np.float32) for _ in range(2)],
        actions=[0, 1],
        log_probs=[-0.2, -0.3],
        reward=reward,
        global_state=np.zeros(6, dtype=np.float32),
        done=False,
        value=value,
    )


def test_rollout_buffer_rejects_overflow_and_bad_agent_payload() -> None:
    """满载写入和 Agent 数据数量不一致应给出明确异常。"""
    buffer = _buffer()
    _add_valid(buffer)
    assert buffer.full is True
    with pytest.raises(OverflowError, match="已满"):
        _add_valid(buffer)
    buffer.reset()
    with pytest.raises(ValueError, match="每个 Agent"):
        buffer.add(
            local_obs=[np.zeros(3, dtype=np.float32)],
            actions=[0],
            log_probs=[-0.2],
            reward=1.0,
            global_state=np.zeros(6, dtype=np.float32),
            done=False,
            value=0.5,
        )


@pytest.mark.parametrize(("reward", "value"), [(np.nan, 0.0), (1.0, np.inf)])
def test_rollout_buffer_rejects_non_finite_reward_or_value(reward: float, value: float) -> None:
    """异常奖励或价值不得污染 GAE 和梯度。"""
    with pytest.raises(ValueError, match="有限数值"):
        _add_valid(_buffer(), reward=reward, value=value)


def test_empty_buffer_gae_and_update_are_well_defined() -> None:
    """空缓冲区应返回空 GAE 和全零更新统计。"""
    buffer = _buffer(capacity=2)
    advantages, returns = buffer.compute_gae(0.0, 0.99, 0.95)
    assert len(advantages) == 2
    assert all(item.size == 0 for item in advantages)
    assert returns.size == 0

    agent = MultiAgentPPO(
        _env(1),
        n_steps=2,
        batch_size=1,
        actor_hidden=(4,),
        critic_hidden=(4,),
        verbose=0,
    )
    result = agent._update([np.array([], dtype=np.float32)], np.array([], dtype=np.float32))
    assert result == {
        "mean_reward": 0.0,
        "mean_actor_loss": 0.0,
        "critic_loss": 0.0,
        "mean_entropy": 0.0,
    }


def test_collect_rollout_requires_training_initialization() -> None:
    """跳过 train 直接收集轨迹时应给出清楚的断言信息。"""
    agent = MultiAgentPPO(
        _env(1),
        n_steps=2,
        actor_hidden=(4,),
        critic_hidden=(4,),
        verbose=0,
    )
    with pytest.raises(AssertionError, match="必须先调用 train"):
        agent._collect_rollout()


def test_load_legacy_format_and_verbose_helpers(monkeypatch) -> None:
    """旧版嵌入配置模型仍应可加载，配置和 repr 应可读。"""
    agent = MultiAgentPPO(
        _env(1),
        n_steps=2,
        actor_hidden=(4,),
        critic_hidden=(4,),
        verbose=1,
    )
    state = {
        "actors": [actor.state_dict() for actor in agent.actors],
        "critic": agent.critic.state_dict(),
        "config": {"num_agents": 1},
    }
    monkeypatch.setattr(marl_module.os.path, "exists", lambda _path: False)
    load = MagicMock(return_value=state)
    monkeypatch.setattr(torch, "load", load)

    agent.load("legacy-model")

    load.assert_called_once_with(
        "legacy-model.pt",
        map_location=agent.device,
        weights_only=False,
    )
    assert agent.get_config()["architecture"] == "MAPPO"
    assert "Agent数=1" in repr(agent)


def test_set_seed_none_and_cuda_branch(monkeypatch) -> None:
    """无 seed 应无副作用，有 CUDA 时应同步设置设备随机种子。"""
    agent = MultiAgentPPO(
        _env(1),
        n_steps=2,
        actor_hidden=(4,),
        critic_hidden=(4,),
        verbose=0,
    )
    agent._set_seed(None)
    cuda_seed = MagicMock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", cuda_seed)
    agent._set_seed(123)
    # torch.manual_seed 会同步调用一次 CUDA，随后实现显式再同步一次。
    assert cuda_seed.call_count == 2
    cuda_seed.assert_called_with(123)
