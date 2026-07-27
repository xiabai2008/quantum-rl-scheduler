"""
退火梯度计算单元测试 —— 覆盖 Issue #357 与 Issue #358。

测试目标（对应 src/quantum/annealing.py 中的 QuantumAnnealingOptimizer）：

Issue #358（算法守卫）:
    ``_compute_gradients`` 必须先调用 ``_is_dqn_agent`` 判断 agent 是否为 DQN
    类型；对非 DQN agent（如 PPO/SAC）应显式抛出 ``ValueError``，而非静默产出
    无效梯度。

Issue #357（target_net 用于 next-Q）:
    ``_compute_gradients`` 计算 TD 目标时，``next_q_values`` 必须取自
    ``target_net``（经由 ``_get_target_net`` 获取），而非在线 ``policy_net``，
    以避免"移动目标"问题。
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import nn

# 将项目根目录加入 sys.path，使 ``from src.quantum.annealing import ...`` 可用
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.quantum.annealing import QuantumAnnealingOptimizer


# ============================================================================
# 辅助：构造 mock 经验回放缓冲区
# ============================================================================
def _make_replay_buffer(batch_size, obs_dim=4, n_actions=2, seed=0):
    """构造一个 mock replay buffer，其 ``sample`` 返回 5 个 numpy 数组组成的元组。

    返回元组顺序与 ``_compute_gradients`` 的解析约定一致：
        (observations, actions, rewards, next_observations, dones)

    其中 actions 形状为 (batch_size, 1) 的 int64，以适配
    ``q_values.gather(1, actions)``。
    """
    rng = np.random.RandomState(seed)
    observations = rng.rand(batch_size, obs_dim).astype(np.float32)
    actions = rng.randint(0, n_actions, size=(batch_size, 1)).astype(np.int64)
    rewards = rng.rand(batch_size).astype(np.float32)
    next_observations = rng.rand(batch_size, obs_dim).astype(np.float32)
    dones = np.zeros(batch_size, dtype=np.float32)

    rb = MagicMock()
    rb.sample.return_value = (
        observations,
        actions,
        rewards,
        next_observations,
        dones,
    )
    return rb, (observations, actions, rewards, next_observations, dones)


# ============================================================================
# 1. _is_dqn_agent —— DQN 类型判定（Issue #358 守卫的前置判定）
# ============================================================================
class TestIsDqnAgent:
    """``_is_dqn_agent`` 静态方法的判定逻辑。"""

    def test_classic_dqn_agent_returns_true(self):
        """具备 policy_net + target_net（均为 nn.Module）的 agent 应判为 DQN。"""
        policy_net = nn.Linear(4, 2)
        target_net = nn.Linear(4, 2)
        agent = SimpleNamespace(policy_net=policy_net, target_net=target_net)

        assert QuantumAnnealingOptimizer._is_dqn_agent(agent) is True

    def test_sb3_dqn_agent_returns_true(self):
        """SB3 DQN agent（具有 policy.q_net）应判为 DQN。"""
        policy_net = nn.Linear(4, 2)
        # SB3 风格：agent.policy.q_net 存在
        agent = SimpleNamespace(policy=SimpleNamespace(q_net=policy_net))

        assert QuantumAnnealingOptimizer._is_dqn_agent(agent) is True

    def test_ppo_agent_returns_false(self):
        """PPO 风格 agent（有 policy 但无 q_net / target_net）不应判为 DQN。"""
        # PPO 的 policy 通常是 actor-critic，无 q_net 属性
        agent = SimpleNamespace(policy=SimpleNamespace(actor=nn.Linear(4, 2)))

        assert QuantumAnnealingOptimizer._is_dqn_agent(agent) is False

    def test_plain_object_returns_false(self):
        """没有任何 DQN 特征的普通对象不应判为 DQN。"""

        class Plain:
            pass

        assert QuantumAnnealingOptimizer._is_dqn_agent(Plain()) is False
        assert QuantumAnnealingOptimizer._is_dqn_agent(object()) is False


# ============================================================================
# 2. _get_target_net —— target 网络获取（Issue #357 的支撑方法）
# ============================================================================
class TestGetTargetNet:
    """``_get_target_net`` 静态方法的选取 / 回退逻辑。"""

    def test_returns_agent_target_net_when_present(self):
        """agent 拥有 nn.Module 类型的 target_net 时，应返回该 target_net。"""
        policy_net = nn.Linear(4, 2)
        target_net = nn.Linear(4, 2)
        agent = SimpleNamespace(policy_net=policy_net, target_net=target_net)

        result = QuantumAnnealingOptimizer._get_target_net(agent, policy_net)

        assert result is target_net

    def test_falls_back_to_policy_net_when_no_target_net(self):
        """agent 无 target_net（如 SB3 DQN 仅含 policy.q_net）时应回退到 policy_net。"""
        policy_net = nn.Linear(4, 2)
        agent = SimpleNamespace(policy=SimpleNamespace(q_net=policy_net))

        result = QuantumAnnealingOptimizer._get_target_net(agent, policy_net)

        assert result is policy_net


# ============================================================================
# 3. _compute_gradients —— Issue #357（target_net 用于 next-Q）与 Issue #358（守卫）
# ============================================================================
class TestComputeGradients:
    """``_compute_gradients`` 的核心行为测试。"""

    def test_uses_target_net_for_next_q_values(self):
        """Issue #357：TD 目标的 next_q_values 必须取自 target_net 而非 policy_net。

        构造权重不同的 policy_net 与 target_net，使得两者给出的 next-Q 不同；
        若实现错误地使用 policy_net，则 TD 误差将与"正确（target_net）"版本不一致。
        """
        torch.manual_seed(0)
        obs_dim, n_actions = 4, 2
        batch_size = 8

        policy_net = nn.Linear(obs_dim, n_actions)
        target_net = nn.Linear(obs_dim, n_actions)
        # 使 target_net 与 policy_net 权重不同，从而 next-Q 不同
        with torch.no_grad():
            for p in target_net.parameters():
                p.add_(1.0)

        gamma = 0.9
        agent = SimpleNamespace(
            policy_net=policy_net, target_net=target_net, gamma=gamma
        )

        rb, (observations, actions, rewards, next_observations, dones) = (
            _make_replay_buffer(batch_size, obs_dim, n_actions, seed=0)
        )

        optimizer = QuantumAnnealingOptimizer()
        gradients, td_errors, loss = optimizer._compute_gradients(
            policy_net, rb, agent, batch_size=batch_size
        )

        # --- 手动复算"正确"TD 误差（next-Q 取自 target_net）---
        with torch.no_grad():
            obs_t = torch.from_numpy(observations).float()
            act_t = torch.from_numpy(actions).long()
            rew_t = torch.from_numpy(rewards).float()
            next_t = torch.from_numpy(next_observations).float()
            done_t = torch.from_numpy(dones).float()

            q_value = policy_net(obs_t).gather(1, act_t).squeeze(1)
            next_q_target = target_net(next_t).max(1)[0]
            target_q = rew_t + gamma * next_q_target * (1.0 - done_t)
            expected_td = (q_value - target_q).numpy()

            # --- 手动复算"错误"TD 误差（next-Q 取自 policy_net）---
            next_q_policy = policy_net(next_t).max(1)[0]
            wrong_target_q = rew_t + gamma * next_q_policy * (1.0 - done_t)
            wrong_td = (q_value - wrong_target_q).numpy()

        # 关键断言：实际 TD 误差 == target_net 版本
        np.testing.assert_allclose(
            td_errors, expected_td, rtol=1e-5, atol=1e-6
        )
        # 前置有效性：target_net 与 policy_net 给出的 next-Q 确实不同，
        # 否则上述等价断言无法区分两种实现。
        assert not np.allclose(expected_td, wrong_td), (
            "target_net 与 policy_net 给出的 TD 误差相同，测试无法区分 Issue #357 行为"
        )

        # 梯度应被成功计算：数量与参数一致，且至少有一个非零梯度
        param_list = list(policy_net.parameters())
        assert len(gradients) == len(param_list)
        assert any(np.any(g != 0) for g in gradients)
        # 每个梯度形状与对应参数一致
        for grad, param in zip(gradients, param_list, strict=False):
            assert grad.shape == param.shape

        # 损失为有限标量
        assert np.isfinite(loss)

    def test_raises_value_error_for_non_dqn_agent(self):
        """Issue #358：对非 DQN agent（PPO 风格）应抛出 ValueError。"""
        policy_net = nn.Linear(4, 2)
        # PPO 风格 agent：有 policy 但无 q_net / target_net
        agent = SimpleNamespace(policy=SimpleNamespace(actor=nn.Linear(4, 2)))
        # 守卫在采样前触发，replay buffer 不会被实际使用
        rb = MagicMock()

        optimizer = QuantumAnnealingOptimizer()
        with pytest.raises(ValueError):
            optimizer._compute_gradients(policy_net, rb, agent, batch_size=8)
