"""MAPPO 多机协调的故障与边界场景测试。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import src.scheduler.marl as marl_module
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumMachine, QuantumSchedulingEnv, Task
from src.scheduler.marl import (
    ActorNet,
    CentralizedCritic,
    MultiAgentEnvWrapper,
    MultiAgentPPO,
    RolloutBuffer,
)


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
        "scorer_loss": 0.0,
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

    # 8.7-v4 修复：先 weights_only=True（安全加载）失败后回退 False（旧格式兼容）。
    # mock 第一次调用（True）抛异常模拟旧格式模型，验证回退路径。
    def _fake_load(*args, **kwargs):
        if kwargs.get("weights_only", True):
            raise TypeError("legacy format not loadable with weights_only=True")
        return state

    load = MagicMock(side_effect=_fake_load)
    monkeypatch.setattr(torch, "load", load)

    agent.load("legacy-model")

    load.assert_any_call(
        "legacy-model.pt",
        map_location=agent.device,
        weights_only=True,
    )
    load.assert_any_call(
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


# ---------------------------------------------------------------------------
# Issue #261: MAPPO 边界情况测试（平票/非法动作/梯度/CTDE）
# ---------------------------------------------------------------------------


def test_vote_tie_breaking() -> None:
    """投票平票时应使用确定性打破逻辑（取第一个最大评分的机器）。

    多个 Agent 投票相同且机器评分完全相同时，aggregate_actions 使用
    Python max() 的"首个最大值"语义确定性地选择索引最小的机器。
    """
    env = _env(3)
    wrapper = MultiAgentEnvWrapper(env)
    env.reset(seed=42)
    # 让所有机器评分完全相同（同 fidelity / available_ratio / queue）
    for machine in env._machines:
        machine.available = True
        machine.fidelity = 0.9
        machine.available_ratio = 0.8
        machine.quantum_queue = 2

    actions = dict.fromkeys(wrapper.machine_names, 1)  # 全部投票量子
    # 多次聚合应返回相同结果（确定性）
    results = [wrapper.aggregate_actions(actions) for _ in range(5)]
    assert all(result == results[0] for result in results)
    # 平票时应选中索引 0（max 返回第一个最大值）
    assert results[0] == (1, 0)


def test_invalid_action_handling() -> None:
    """Agent 输出非法动作（负数、超范围）时应被安全处理，不影响其他 Agent。

    当前实现：aggregate_actions 中 int(actions.get(name, 0)) 将非 {0,1,2}
    的值视为弃权（既不投量子也不投混合），等同于经典动作。
    限制说明：源码未做 clamp 或抛异常，而是静默回退到经典，测试验证此既定行为。
    """
    env = _env(2)
    wrapper = MultiAgentEnvWrapper(env)
    env.reset(seed=42)
    for machine in env._machines:
        machine.available = True

    # 全部非法动作 → 回退到经典
    invalid_actions = {
        wrapper.machine_names[0]: -1,  # 负数
        wrapper.machine_names[1]: 99,  # 超范围
    }
    env_action, chosen = wrapper.aggregate_actions(invalid_actions)
    assert env_action == 0
    assert chosen is None

    # 混合：一个有效量子投票 + 一个非法动作 → 应执行量子
    mixed_actions = {
        wrapper.machine_names[0]: 1,  # 有效：量子
        wrapper.machine_names[1]: -5,  # 非法：负数
    }
    env_action, chosen = wrapper.aggregate_actions(mixed_actions)
    assert env_action == 1
    assert chosen == 0


def test_gradient_flow() -> None:
    """使用 torch.autograd.gradcheck 验证 Actor/Critic 梯度反向传播正确。

    gradcheck 通过有限差分法验证解析梯度与数值梯度一致，确保反向传播实现无误。
    所有张量必须使用 double 精度（torch.float64）。
    """
    obs_dim = 4
    action_dim = 4
    batch_size = 2

    # === Actor 梯度流：obs -> feature -> logits -> log_prob ===
    actor = ActorNet(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=(6,)).double()
    obs = torch.randn(batch_size, obs_dim, dtype=torch.float64, requires_grad=True)
    actions = torch.zeros(batch_size, dtype=torch.long)

    def actor_fn(obs_input: torch.Tensor) -> torch.Tensor:
        log_prob, _entropy = actor.evaluate_actions(obs_input, actions)
        return log_prob.sum()

    assert torch.autograd.gradcheck(actor_fn, (obs,), eps=1e-6, atol=1e-4)

    # === Critic 梯度流：global_state -> value ===
    num_agents = 2
    global_state_dim = obs_dim * num_agents
    critic = CentralizedCritic(global_state_dim=global_state_dim, hidden_sizes=(6,)).double()
    gs = torch.randn(batch_size, global_state_dim, dtype=torch.float64, requires_grad=True)

    def critic_fn(gs_input: torch.Tensor) -> torch.Tensor:
        return critic(gs_input).sum()  # type: ignore[no-any-return]

    assert torch.autograd.gradcheck(critic_fn, (gs,), eps=1e-6, atol=1e-4)

    # === 多 Agent 梯度独立性：一个 Actor 的反向传播不影响另一个 ===
    actor_a = ActorNet(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=(6,)).double()
    actor_b = ActorNet(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=(6,)).double()
    obs_a = torch.randn(1, obs_dim, dtype=torch.float64)

    actor_a.zero_grad()
    actor_b.zero_grad()
    log_prob_a, _ = actor_a.evaluate_actions(obs_a, torch.zeros(1, dtype=torch.long))
    log_prob_a.sum().backward()

    # actor_a 应有梯度
    has_grad_a = any(p.grad is not None and torch.any(p.grad != 0) for p in actor_a.parameters())
    assert has_grad_a, "Actor A 参数未收到梯度"
    # actor_b 不应有梯度（多 Agent 梯度独立性）
    for name, param in actor_b.named_parameters():
        assert param.grad is None or torch.all(param.grad == 0), (
            f"Actor B 参数 {name} 不应收到梯度（多 Agent 梯度独立性被破坏）"
        )


def test_ctde_consistency() -> None:
    """验证 CTDE：训练时 Critic 使用全局状态，执行时 Actor 仅用局部观测。

    CTDE (Centralized Training, Decentralized Execution):
    - 执行时：每个 Actor 仅看本机局部观测（去中心化）
    - 训练时：Critic 看所有 Agent 的全局状态拼接（中心化）
    """
    num_agents = 2
    env = _env(num_agents)
    agent = MultiAgentPPO(
        env,
        n_steps=4,
        batch_size=2,
        actor_hidden=(4,),
        critic_hidden=(4,),
        verbose=0,
    )

    # 1. 架构一致性：Critic 输入维度 = local_obs_dim * num_agents（全局状态）
    assert agent.global_state_dim == agent.local_obs_dim * num_agents

    # 2. 执行时去中心化：Actor 第一层输入维度 = local_obs_dim（仅局部观测）
    for i, actor in enumerate(agent.actors):
        first_layer = actor.feature[0]
        assert isinstance(first_layer, torch.nn.Linear)
        assert first_layer.in_features == agent.local_obs_dim, (
            f"Actor {i} 输入维度应为 local_obs_dim={agent.local_obs_dim}，"
            f"实际为 {first_layer.in_features}"
        )
    # Actor 输入维度 < 全局状态维度（证明仅用局部，非全局）
    assert agent.local_obs_dim < agent.global_state_dim

    # 3. 训练时中心化：Critic 第一层输入维度 = global_state_dim
    critic_first_layer = agent.critic.net[0]
    assert isinstance(critic_first_layer, torch.nn.Linear)
    assert critic_first_layer.in_features == agent.global_state_dim

    # 4. 行为一致性：_sample_actions 调用 Critic 时传入全局状态
    env.reset(seed=42)
    local_obs = agent.wrapper.get_local_observations()
    captured_inputs: list[torch.Tensor] = []
    original_forward = agent.critic.forward

    def spy_forward(x: torch.Tensor) -> torch.Tensor:
        captured_inputs.append(x.detach().clone())
        return original_forward(x)

    agent.critic.forward = spy_forward  # type: ignore[assignment]
    try:
        agent._sample_actions(local_obs, deterministic=True)
    finally:
        agent.critic.forward = original_forward  # type: ignore[method-assign]

    assert len(captured_inputs) == 1, "Critic 应被调用一次"
    assert captured_inputs[0].shape == (1, agent.global_state_dim), (
        f"Critic 输入形状应为 (1, {agent.global_state_dim})，"
        f"实际为 {tuple(captured_inputs[0].shape)}"
    )

    # 5. 训练缓冲区存储的是全局状态（而非单 Agent 局部观测）
    assert agent.buffer.global_states.shape[1] == agent.global_state_dim
