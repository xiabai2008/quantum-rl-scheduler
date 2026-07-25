"""
Quantum RL Scheduler - DuelingQNetwork Unit Tests
Unit Tests for src/scheduler/networks.py

Test coverage:
  - DuelingQNetwork default net_arch=[128, 64] construction
  - DuelingQNetwork custom net_arch construction
  - forward output shape (batch_size, action_dim)
  - Dueling Q-value decomposition: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
  - Different action_space.n output dimensions
  - Backward gradient flow verification
"""

from __future__ import annotations

import os
import sys
import unittest

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import FlattenExtractor
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.networks import DuelingQNetwork


def _make_network(
    obs_dim: int = 8,
    action_dim: int = 3,
    net_arch: list[int] | None = None,
) -> DuelingQNetwork:
    """Helper to build a DuelingQNetwork instance.

    Args:
        obs_dim: Observation space dimension.
        action_dim: Action space dimension.
        net_arch: Hidden layer architecture, None uses default [128, 64].

    Returns:
        An initialized DuelingQNetwork instance.
    """
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,))
    act_space = spaces.Discrete(action_dim)
    features_extractor = FlattenExtractor(obs_space)
    features_dim = obs_dim
    return DuelingQNetwork(
        observation_space=obs_space,
        action_space=act_space,
        features_extractor=features_extractor,
        features_dim=features_dim,
        net_arch=net_arch,
    )


class TestDuelingQNetworkInit(unittest.TestCase):
    """Tests for DuelingQNetwork initialization."""

    def test_default_net_arch(self):
        """Default net_arch should be [128, 64]."""
        net = _make_network()
        linear_layers = [m for m in net.q_net.modules() if isinstance(m, nn.Linear)]
        self.assertEqual(len(linear_layers), 2)
        self.assertEqual(linear_layers[0].out_features, 128)
        self.assertEqual(linear_layers[1].out_features, 64)

    def test_custom_net_arch(self):
        """Custom net_arch [64, 32] should be respected."""
        net = _make_network(net_arch=[64, 32])
        linear_layers = [m for m in net.q_net.modules() if isinstance(m, nn.Linear)]
        self.assertEqual(len(linear_layers), 2)
        self.assertEqual(linear_layers[0].out_features, 64)
        self.assertEqual(linear_layers[1].out_features, 32)

    def test_value_stream_architecture(self):
        """value_stream output dim=1, advantage_stream output dim=action_dim."""
        net = _make_network(obs_dim=8, net_arch=[128, 64])
        value_layers = [m for m in net.value_stream.modules() if isinstance(m, nn.Linear)]
        self.assertEqual(value_layers[-1].out_features, 1)
        adv_layers = [m for m in net.advantage_stream.modules() if isinstance(m, nn.Linear)]
        self.assertEqual(adv_layers[-1].out_features, 3)


class TestDuelingQNetworkForward(unittest.TestCase):
    """Tests for DuelingQNetwork forward pass."""

    def test_forward_output_shape(self):
        """Forward output shape should be (batch_size, action_dim)."""
        net = _make_network(obs_dim=8, action_dim=3)
        obs = th.randn(4, 8)
        with th.no_grad():
            output = net(obs)
        self.assertEqual(output.shape, (4, 3))

    def test_q_value_decomposition(self):
        """Verify Q = V + A - mean(A) identity."""
        net = _make_network(obs_dim=8, action_dim=5)
        obs = th.randn(2, 8)
        with th.no_grad():
            features = net.extract_features(obs, net.features_extractor)
            shared = net.q_net(features)
            value = net.value_stream(shared)
            advantage = net.advantage_stream(shared)
            expected_q = value + advantage - advantage.mean(dim=-1, keepdim=True)
            actual_q = net(obs)
        th.testing.assert_close(actual_q, expected_q)

    def test_different_action_dim(self):
        """Different action_space.n should produce correct output dims."""
        for action_dim in [2, 4, 7]:
            net = _make_network(obs_dim=8, action_dim=action_dim)
            obs = th.randn(1, 8)
            with th.no_grad():
                output = net(obs)
            self.assertEqual(output.shape[-1], action_dim)

    def test_gradient_flow(self):
        """loss.backward() should produce non-None gradients."""
        net = _make_network(obs_dim=8, action_dim=3)
        obs = th.randn(4, 8)
        target = th.randn(4, 3)
        output = net(obs)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        for name, param in net.value_stream.named_parameters():
            self.assertIsNotNone(param.grad, f"value_stream.{name} grad is None")
        for name, param in net.advantage_stream.named_parameters():
            self.assertIsNotNone(param.grad, f"advantage_stream.{name} grad is None")
        for name, param in net.q_net.named_parameters():
            self.assertIsNotNone(param.grad, f"q_net.{name} grad is None")


class TestDuelingQNetworkBatchConsistency(unittest.TestCase):
    """Tests for batch consistency."""

    def test_single_vs_batch_output(self):
        """Single-sample forward should match first result of batch forward."""
        net = _make_network(obs_dim=8, action_dim=3)
        net.eval()
        obs_single = th.randn(1, 8)
        obs_batch = obs_single.expand(4, -1).clone()
        with th.no_grad():
            out_single = net(obs_single)
            out_batch = net(obs_batch)
        th.testing.assert_close(out_single, out_batch[0:1])


if __name__ == "__main__":
    unittest.main()
