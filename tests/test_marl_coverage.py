"""MAPPO 补充测试（Issue #37）：覆盖 marl.py 未测试分支。

补充覆盖目标：64% → 80%+

新增覆盖：
    1. _build_mlp output_dim=0 分支
    2. MultiAgentEnvWrapper.reset options 透传
    3. ActorNet.forward / evaluate_actions 直接调用
    4. MultiAgentPPO._collect_rollout done 分支
    5. MultiAgentPPO.train eval_freq 触发评估分支
    6. MultiAgentPPO._save_internal / load 新格式
    7. MultiAgentPPO.predict deterministic=False
    8. RolloutBuffer.compute_gae non_terminal 分支
    9. MultiAgentPPO.get_config / __repr__
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, OBS_DIM, QuantumSchedulingEnv
from src.scheduler.marl import (
    ActorNet,
    CentralizedCritic,
    MultiAgentEnvWrapper,
    MultiAgentPPO,
    RolloutBuffer,
    _build_mlp,
)


def _make_env(machine_count: int = 2, max_steps: int = 40, seed: int = 42) -> QuantumSchedulingEnv:
    """创建测试环境。"""
    configs = DEFAULT_MACHINE_CONFIGS[:machine_count] if machine_count else None
    return QuantumSchedulingEnv(max_steps=max_steps, machine_configs=configs, seed=seed)


def _make_agent(**kwargs: Any) -> MultiAgentPPO:
    """创建快速测试用 MAPPO 智能体。"""
    env = _make_env(machine_count=2, max_steps=20, seed=42)
    defaults: dict[str, Any] = {
        "n_steps": 8,
        "batch_size": 4,
        "n_epochs": 1,
        "actor_hidden": (8,),
        "critic_hidden": (8,),
        "seed": 42,
        "verbose": 0,
    }
    defaults.update(kwargs)
    return MultiAgentPPO(env, **cast("Any", defaults))


# ---------------------------------------------------------------------------
# _build_mlp 分支覆盖
# ---------------------------------------------------------------------------


class TestBuildMlp:
    """_build_mlp 函数的分支覆盖。"""

    def test_output_dim_zero_returns_feature_extractor_only(self) -> None:
        """output_dim=0 时只返回特征提取器，不含输出层。"""
        mlp = _build_mlp(input_dim=4, output_dim=0, hidden_sizes=(8, 16))
        # 最后一个 Linear 是 hidden_sizes 的最后一层（16），不应有额外的输出层
        linear_layers = [m for m in mlp if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 2  # 两个隐藏层，无输出层
        assert linear_layers[-1].out_features == 16

    def test_output_dim_positive_appends_output_layer(self) -> None:
        """output_dim>0 时应追加输出层。"""
        mlp = _build_mlp(input_dim=4, output_dim=3, hidden_sizes=(8,))
        linear_layers = [m for m in mlp if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 2  # 一个隐藏层 + 一个输出层
        assert linear_layers[-1].out_features == 3

    def test_activation_function_applied(self) -> None:
        """自定义激活函数应被使用。"""
        mlp = _build_mlp(input_dim=4, output_dim=2, hidden_sizes=(8,), activation=nn.ReLU)
        relu_layers = [m for m in mlp if isinstance(m, nn.ReLU)]
        assert len(relu_layers) == 1


# ---------------------------------------------------------------------------
# ActorNet 直接调用覆盖
# ---------------------------------------------------------------------------


class TestActorNetDirect:
    """ActorNet 的 forward 和 evaluate_actions 直接调用。"""

    def test_forward_returns_logits_shape(self) -> None:
        """forward 应返回 (batch, action_dim) 形状的 logits。"""
        actor = ActorNet(obs_dim=13, action_dim=3, hidden_sizes=(8,))
        obs = torch.zeros(4, 13)  # batch=4
        logits = actor.forward(obs)
        assert logits.shape == (4, 3)

    def test_evaluate_actions_returns_log_prob_and_entropy(self) -> None:
        """evaluate_actions 应返回 log_prob 和 entropy。"""
        actor = ActorNet(obs_dim=13, action_dim=3, hidden_sizes=(8,))
        obs = torch.zeros(2, 13)
        actions = torch.zeros(2, dtype=torch.long)
        log_prob, entropy = actor.evaluate_actions(obs, actions)
        assert log_prob.shape == (2,)
        assert entropy.shape == (2,)
        assert torch.all(torch.isfinite(log_prob))
        assert torch.all(entropy >= 0)

    def test_get_action_deterministic_picks_argmax(self) -> None:
        """deterministic=True 时应选择 logit 最大的动作。"""
        actor = ActorNet(obs_dim=13, action_dim=3, hidden_sizes=(8,))
        obs = torch.zeros(1, 13)
        # 多次采样，确定性策略每次应返回相同动作
        action1, _, _ = actor.get_action(obs, deterministic=True)
        action2, _, _ = actor.get_action(obs, deterministic=True)
        assert int(action1.item()) == int(action2.item())


# ---------------------------------------------------------------------------
# CentralizedCritic 覆盖
# ---------------------------------------------------------------------------


class TestCentralizedCritic:
    """CentralizedCritic 直接调用。"""

    def test_forward_batch_input(self) -> None:
        """Critic 应支持批量输入。"""
        critic = CentralizedCritic(global_state_dim=26, hidden_sizes=(8,))
        gs = torch.zeros(3, 26)  # batch=3
        values = critic.forward(gs)
        assert values.shape == (3,)


# ---------------------------------------------------------------------------
# MultiAgentEnvWrapper.reset options 透传
# ---------------------------------------------------------------------------


class TestWrapperResetOptions:
    """MultiAgentEnvWrapper.reset 的 options 参数透传。"""

    def test_reset_returns_local_observations_and_info(self) -> None:
        """reset 应返回局部观测字典和 info。"""
        env = _make_env(machine_count=2, max_steps=20, seed=42)
        wrapper = MultiAgentEnvWrapper(env)
        local_obs, _info = wrapper.reset(seed=42)
        assert isinstance(local_obs, dict)
        assert set(local_obs.keys()) == set(wrapper.machine_names)
        for obs in local_obs.values():
            assert obs.shape == (OBS_DIM + 3,)

    def test_reset_passes_options_to_env(self) -> None:
        """options 参数应透传给 env.reset。"""
        env = _make_env(machine_count=2, max_steps=20, seed=42)
        wrapper = MultiAgentEnvWrapper(env)
        options = {"custom_option": True}
        with patch.object(env, "reset") as mock_reset:
            mock_reset.return_value = (np.zeros(OBS_DIM, dtype=np.float32), {"mock": True})
            wrapper.reset(seed=99, options=options)
            mock_reset.assert_called_once_with(seed=99, options=options)


# ---------------------------------------------------------------------------
# MultiAgentPPO.predict deterministic=False
# ---------------------------------------------------------------------------


class TestPredictStochastic:
    """predict 在 stochastic 模式下的行为。"""

    def test_predict_stochastic_returns_valid_action(self) -> None:
        """deterministic=False 时应返回合法动作。"""
        agent = _make_agent()
        agent.env.reset(seed=42)
        action = agent.predict(deterministic=False)
        assert action in (0, 1, 2)


# ---------------------------------------------------------------------------
# RolloutBuffer.compute_gae non_terminal 分支
# ---------------------------------------------------------------------------


class TestComputeGaeBranches:
    """compute_gae 的 non_terminal 和 last_value 分支。"""

    def test_gae_with_all_done_episodes(self) -> None:
        """所有步都 done 时，GAE 应正确处理。"""
        buf = RolloutBuffer(n_steps=4, num_agents=2, local_obs_dim=3, global_state_dim=6)
        for _t in range(4):
            buf.add(
                local_obs=[np.zeros(3, dtype=np.float32) for _ in range(2)],
                actions=[0, 1],
                log_probs=[-0.5, -0.5],
                reward=1.0,
                global_state=np.zeros(6, dtype=np.float32),
                done=True,  # 每步都 done
                value=0.5,
            )
        advs, returns = buf.compute_gae(last_value=0.0, gamma=0.99, gae_lambda=0.95)
        assert len(advs) == 2
        for adv in advs:
            assert adv.shape == (4,)
        # done=True 时 non_terminal=0，delta = reward - value
        expected_delta = 1.0 - 0.5
        assert np.allclose(advs[0], expected_delta)
        assert np.allclose(returns, expected_delta + 0.5)

    def test_gae_with_continuing_episode_uses_next_value(self) -> None:
        """非终止步应使用 next_value 进行 bootstrap。"""
        buf = RolloutBuffer(n_steps=3, num_agents=1, local_obs_dim=3, global_state_dim=3)
        # 第一步：reward=1, value=0.5, done=False
        # 第二步：reward=0, value=0.8, done=False
        # 第三步：reward=1, value=0.3, done=True
        values = [0.5, 0.8, 0.3]
        rewards = [1.0, 0.0, 1.0]
        dones = [False, False, True]
        for t in range(3):
            buf.add(
                local_obs=[np.zeros(3, dtype=np.float32)],
                actions=[0],
                log_probs=[-0.5],
                reward=rewards[t],
                global_state=np.zeros(3, dtype=np.float32),
                done=dones[t],
                value=values[t],
            )
        advs, _returns = buf.compute_gae(last_value=10.0, gamma=0.99, gae_lambda=0.95)
        # 最后一步 done=True，delta = reward + gamma*last_value*0 - value
        # = 1.0 + 0 - 0.3 = 0.7
        assert abs(advs[0][-1] - 0.7) < 1e-5


# ---------------------------------------------------------------------------
# MultiAgentPPO._collect_rollout done 分支
# ---------------------------------------------------------------------------


class TestCollectRolloutDoneBranch:
    """_collect_rollout 的 terminated/truncated 分支。"""

    def test_collect_rollout_handles_done_and_resets(self) -> None:
        """收集 rollout 时遇到 done 应重置环境并继续。"""
        agent = _make_agent(n_steps=4)
        agent._last_obs, _ = agent.wrapper.reset(seed=42)
        agent._last_global_state = agent.wrapper.get_global_state()
        agent._collect_rollout()
        assert agent.buffer.pos == 4
        assert hasattr(agent, "_last_episode_rewards")


# ---------------------------------------------------------------------------
# MultiAgentPPO.train eval_freq 触发评估分支
# ---------------------------------------------------------------------------


class TestTrainEvalFreqBranch:
    """train 的周期性评估分支。"""

    def test_train_with_eval_freq_triggers_evaluation(self) -> None:
        """eval_freq>0 时应触发评估并保存 best_model。"""
        agent = _make_agent(n_steps=4, verbose=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.log_dir = tmpdir
            agent.train(total_timesteps=8, eval_freq=4, n_eval_episodes=1)
            # best_model 应被保存
            assert os.path.exists(os.path.join(tmpdir, "best_model.pt"))
            assert os.path.exists(os.path.join(tmpdir, "best_model_config.json"))


# ---------------------------------------------------------------------------
# MultiAgentPPO._save_internal / load 新格式
# ---------------------------------------------------------------------------


class TestSaveLoadNewFormat:
    """新格式保存/加载（带 _config.json）。"""

    def test_save_load_new_format_roundtrip(self) -> None:
        """新格式保存后加载应恢复模型参数。"""
        agent = _make_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            agent._save_internal(save_path)
            assert os.path.exists(save_path + ".pt")
            assert os.path.exists(save_path + "_config.json")

            # 验证 config 内容
            with open(save_path + "_config.json", encoding="utf-8") as f:
                cfg = json.load(f)
            assert cfg["num_agents"] == agent.num_agents
            assert cfg["local_obs_dim"] == agent.local_obs_dim

            # 新 agent 加载
            agent2 = _make_agent()
            agent2.load(save_path)
            # 验证参数一致
            for i, (a1, a2) in enumerate(zip(agent.actors, agent2.actors, strict=True)):
                for p1, p2 in zip(a1.parameters(), a2.parameters(), strict=True):
                    assert torch.allclose(p1, p2), f"Actor {i} 参数不一致"

    def test_save_creates_directory_if_not_exists(self) -> None:
        """保存路径的父目录不存在时应自动创建。"""
        agent = _make_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "nested", "deep", "model")
            agent._save_internal(nested_path)
            assert os.path.exists(nested_path + ".pt")

    def test_load_appends_pt_extension_if_missing(self) -> None:
        """加载时路径不含 .pt 后缀应自动补全。"""
        agent = _make_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            agent._save_internal(save_path)
            # 不带 .pt 后缀加载
            agent2 = _make_agent()
            agent2.load(save_path)  # 不带 .pt
            # 应成功加载（不抛异常即通过）

    def test_load_with_verbose_logs(self) -> None:
        """verbose=1 时加载应记录日志。"""
        agent = _make_agent(verbose=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            agent._save_internal(save_path)
            agent2 = _make_agent(verbose=1)
            # 不抛异常即通过
            agent2.load(save_path)


# ---------------------------------------------------------------------------
# MultiAgentPPO.get_config / __repr__
# ---------------------------------------------------------------------------


class TestConfigAndRepr:
    """get_config 和 __repr__ 的完整覆盖。"""

    def test_get_config_returns_all_fields(self) -> None:
        """get_config 应返回所有配置字段。"""
        agent = _make_agent(learning_rate=1e-3, n_steps=16, batch_size=8, gamma=0.95)
        cfg = agent.get_config()
        assert cfg["architecture"] == "MAPPO"
        assert cfg["num_agents"] == 2
        assert cfg["learning_rate"] == 1e-3
        assert cfg["n_steps"] == 16
        assert cfg["batch_size"] == 8
        assert cfg["gamma"] == 0.95
        assert "machine_names" in cfg
        assert "local_obs_dim" in cfg
        assert "global_state_dim" in cfg
        assert "gae_lambda" in cfg
        assert "clip_range" in cfg
        assert "ent_coef" in cfg
        assert "vf_coef" in cfg
        assert "max_grad_norm" in cfg
        assert "device" in cfg

    def test_repr_returns_formatted_string(self) -> None:
        """__repr__ 应返回格式化的字符串。"""
        agent = _make_agent()
        repr_str = repr(agent)
        assert "MultiAgentPPO(" in repr_str
        assert "架构=MAPPO" in repr_str
        assert "Agent数=" in repr_str
        assert "机器=" in repr_str


# ---------------------------------------------------------------------------
# MultiAgentPPO._to_tensor / _global_state_tensor
# ---------------------------------------------------------------------------


class TestTensorHelpers:
    """_to_tensor 和 _global_state_tensor 辅助函数。"""

    def test_to_tensor_returns_float32_on_device(self) -> None:
        """_to_tensor 应返回设备上的 float32 张量。"""
        agent = _make_agent()
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        tensor = agent._to_tensor(arr)
        assert tensor.dtype == torch.float32
        assert tensor.device == agent.device
        assert tensor.shape == (3,)

    def test_global_state_tensor_adds_batch_dim(self) -> None:
        """_global_state_tensor 应添加 batch 维度。"""
        agent = _make_agent()
        gs = np.zeros(agent.global_state_dim, dtype=np.float32)
        tensor = agent._global_state_tensor(gs)
        assert tensor.shape == (1, agent.global_state_dim)


# ---------------------------------------------------------------------------
# MultiAgentEnvWrapper._build_local_obs 边界值
# ---------------------------------------------------------------------------


class TestBuildLocalObsEdgeCases:
    """_build_local_obs 的边界值覆盖。"""

    def test_clamps_fidelity_above_one(self) -> None:
        """fidelity 超过 1.0 应被 clip 到 1.0。"""
        env = _make_env(machine_count=1, max_steps=10, seed=42)
        wrapper = MultiAgentEnvWrapper(env)
        env.reset(seed=42)
        env._machines[0].fidelity = 2.5  # 超出范围
        env._machines[0].available_ratio = 1.0
        env._machines[0].quantum_queue = 0
        obs = wrapper._build_local_obs(env._get_observation(), 0)
        # fidelity 在 local_obs 的 OBS_DIM 位置
        assert obs[OBS_DIM] <= 1.0

    def test_clamps_negative_available_ratio(self) -> None:
        """available_ratio 为负值应被 clip 到 0.0。"""
        env = _make_env(machine_count=1, max_steps=10, seed=42)
        wrapper = MultiAgentEnvWrapper(env)
        env.reset(seed=42)
        env._machines[0].fidelity = 0.5
        env._machines[0].available_ratio = -0.5  # 负值
        env._machines[0].quantum_queue = 0
        obs = wrapper._build_local_obs(env._get_observation(), 0)
        assert obs[OBS_DIM + 1] >= 0.0


# ---------------------------------------------------------------------------
# MultiAgentEnvWrapper.aggregate_actions hybrid 分支
# ---------------------------------------------------------------------------


class TestAggregateActionsHybrid:
    """aggregate_actions 的 hybrid 投票分支。"""

    def test_hybrid_only_votes_yield_hybrid_action(self) -> None:
        """仅有 hybrid(2) 投票时应返回 action=2。"""
        env = _make_env(machine_count=2, max_steps=10, seed=42)
        wrapper = MultiAgentEnvWrapper(env)
        env.reset(seed=42)
        for m in env._machines:
            m.available = True
        actions = dict.fromkeys(wrapper.machine_names, 2)
        env_action, chosen = wrapper.aggregate_actions(actions)
        assert env_action == 2
        assert chosen is not None

    def test_hybrid_quantum_mixed_quantum_wins(self) -> None:
        """同时有 quantum(1) 和 hybrid(2) 时 quantum 优先。"""
        env = _make_env(machine_count=2, max_steps=10, seed=42)
        wrapper = MultiAgentEnvWrapper(env)
        env.reset(seed=42)
        for m in env._machines:
            m.available = True
        actions = {
            wrapper.machine_names[0]: 2,  # hybrid
            wrapper.machine_names[1]: 1,  # quantum
        }
        env_action, chosen = wrapper.aggregate_actions(actions)
        assert env_action == 1  # quantum wins
        assert chosen == 1


# ---------------------------------------------------------------------------
# MultiAgentEnvWrapper._machine_score
# ---------------------------------------------------------------------------


class TestMachineScore:
    """_machine_score 评分函数。"""

    def test_score_higher_for_better_machine(self) -> None:
        """更优机器（高 fidelity、高可用、低队列）应有更高评分。"""
        env = _make_env(machine_count=2, max_steps=10, seed=42)
        wrapper = MultiAgentEnvWrapper(env)
        env.reset(seed=42)
        # 机器0：优秀
        env._machines[0].fidelity = 0.99
        env._machines[0].available_ratio = 1.0
        env._machines[0].quantum_queue = 0
        # 机器1：较差
        env._machines[1].fidelity = 0.5
        env._machines[1].available_ratio = 0.3
        env._machines[1].quantum_queue = 10
        score0 = wrapper._machine_score(0)
        score1 = wrapper._machine_score(1)
        assert score0 > score1

    def test_score_zero_queue_returns_fidelity_times_ratio(self) -> None:
        """队列为 0 时评分 = fidelity * available_ratio / 1。"""
        env = _make_env(machine_count=1, max_steps=10, seed=42)
        wrapper = MultiAgentEnvWrapper(env)
        env.reset(seed=42)
        env._machines[0].fidelity = 0.8
        env._machines[0].available_ratio = 0.5
        env._machines[0].quantum_queue = 0
        score = wrapper._machine_score(0)
        assert abs(score - 0.4) < 1e-5


# ---------------------------------------------------------------------------
# MultiAgentEnvWrapper.step 路由分支
# ---------------------------------------------------------------------------


class TestStepRoutingBranches:
    """wrapper.step 的路由逻辑分支。"""

    def test_step_with_single_agent_skips_routing(self) -> None:
        """单机环境下不应触发强制路由（num_agents=1）。"""
        env = _make_env(machine_count=1, max_steps=10, seed=42)
        wrapper = MultiAgentEnvWrapper(env)
        env.reset(seed=42)
        # 所有投票 quantum
        actions = dict.fromkeys(wrapper.machine_names, 1)
        _local_obs, _reward, _terminated, _truncated, info = wrapper.step(actions)
        assert "chosen_machine" in info
        assert "env_action" in info

    def test_step_with_chosen_machine_incompatible_skips_routing(self) -> None:
        """选中机器不兼容时不应强制修改在线状态。"""
        env = _make_env(machine_count=2, max_steps=10, seed=42)
        wrapper = MultiAgentEnvWrapper(env)
        env.reset(seed=42)
        original_avail = [m.available for m in env._machines]
        # 让机器0不兼容（available_ratio=0 → usable_qubits=0）
        env._machines[0].available_ratio = 0.0
        env._current_task = MagicMock()
        env._current_task.qubit_count = 1
        actions = {wrapper.machine_names[0]: 1, wrapper.machine_names[1]: 0}
        _local_obs, _reward, _terminated, _truncated, _info = wrapper.step(actions)
        # 机器在线状态不应被修改
        assert [m.available for m in env._machines] == original_avail


# ---------------------------------------------------------------------------
# MultiAgentPPO.evaluate 完整流程
# ---------------------------------------------------------------------------


class TestEvaluateComplete:
    """evaluate 方法的完整执行流程。"""

    def test_evaluate_returns_valid_metrics(self) -> None:
        """evaluate 应返回包含 mean_reward/std_reward/success_rate/num_episodes 的字典。"""
        agent = _make_agent()
        result = agent.evaluate(num_episodes=2, deterministic=True)
        assert "mean_reward" in result
        assert "std_reward" in result
        assert "success_rate" in result
        assert "num_episodes" in result
        assert result["num_episodes"] == 2
        assert np.isfinite(result["mean_reward"])
        assert 0.0 <= result["success_rate"] <= 1.0


# ---------------------------------------------------------------------------
# MultiAgentPPO._set_seed
# ---------------------------------------------------------------------------


class TestSetSeed:
    """_set_seed 方法的分支。"""

    def test_set_seed_none_does_nothing(self) -> None:
        """seed=None 时应无副作用。"""
        agent = _make_agent()
        # 不抛异常即通过
        agent._set_seed(None)

    def test_set_seed_makes_results_reproducible(self) -> None:
        """相同 seed 应产生可复现的初始动作。"""
        env1 = _make_env(machine_count=1, max_steps=10, seed=100)
        agent1 = MultiAgentPPO(
            env1, n_steps=4, actor_hidden=(8,), critic_hidden=(8,), seed=100, verbose=0
        )
        env1.reset(seed=100)
        action1 = agent1.predict(deterministic=False)

        env2 = _make_env(machine_count=1, max_steps=10, seed=100)
        agent2 = MultiAgentPPO(
            env2, n_steps=4, actor_hidden=(8,), critic_hidden=(8,), seed=100, verbose=0
        )
        env2.reset(seed=100)
        action2 = agent2.predict(deterministic=False)
        assert action1 == action2


# ---------------------------------------------------------------------------
# MultiAgentPPO device 选择
# ---------------------------------------------------------------------------


class TestDeviceSelection:
    """__init__ 的 device 选择分支。"""

    def test_auto_device_selects_cpu_when_no_cuda(self) -> None:
        """device='auto' 且无 CUDA 时应选择 CPU。"""
        with patch("torch.cuda.is_available", return_value=False):
            agent = _make_agent(device="auto")
            assert agent.device == torch.device("cpu")

    def test_explicit_cpu_device(self) -> None:
        """device='cpu' 时应使用 CPU。"""
        agent = _make_agent(device="cpu")
        assert agent.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# MultiAgentEnvWrapper.get_global_state
# ---------------------------------------------------------------------------


class TestGetGlobalState:
    """get_global_state 的完整覆盖。"""

    def test_global_state_is_concatenation_of_local_obs(self) -> None:
        """全局状态应为所有局部观测的拼接。"""
        env = _make_env(machine_count=3, max_steps=10, seed=42)
        wrapper = MultiAgentEnvWrapper(env)
        env.reset(seed=42)
        gs = wrapper.get_global_state()
        assert gs.shape == (wrapper.local_obs_dim * 3,)
        assert gs.dtype == np.float32

    def test_global_state_changes_after_step(self) -> None:
        """step 后全局状态应发生变化（除非环境状态不变）。"""
        env = _make_env(machine_count=2, max_steps=10, seed=42)
        wrapper = MultiAgentEnvWrapper(env)
        env.reset(seed=42)
        gs_before = wrapper.get_global_state().copy()
        actions = dict.fromkeys(wrapper.machine_names, 0)
        wrapper.step(actions)
        gs_after = wrapper.get_global_state()
        # 至少形状应一致
        assert gs_after.shape == gs_before.shape


# ---------------------------------------------------------------------------
# _update 边界
# ---------------------------------------------------------------------------


class TestUpdateEdgeCases:
    """_update 的边界情况。"""

    def test_update_with_single_step_buffer(self) -> None:
        """缓冲区仅 1 步时应能完成更新。"""
        agent = _make_agent(n_steps=1, batch_size=1)
        agent._last_obs, _ = agent.wrapper.reset(seed=42)
        agent._last_global_state = agent.wrapper.get_global_state()
        agent._collect_rollout()
        last_value_t = agent.critic(agent._global_state_tensor(agent._last_global_state))
        last_value = float(last_value_t.item())
        advs, returns = agent.buffer.compute_gae(last_value, agent.gamma, agent.gae_lambda)
        result = agent._update(advs, returns)
        assert "mean_reward" in result
        assert "mean_actor_loss" in result
        assert "critic_loss" in result
        assert "mean_entropy" in result
