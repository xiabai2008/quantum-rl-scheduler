"""多seed评估并行化与决策缓存的单元测试。

覆盖：
- CachedPPOStrategy：缓存命中/未命中、cache_stats
- PPOAgent 决策缓存集成：predict 缓存路径、cache_stats、get_config
- _run_single_seed：返回值结构与策略统计
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from gymnasium import spaces

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "evaluation"))

from run_issue_38_67_experiments import (
    FCFSStrategy,
    RandomStrategy,
)
from run_multiseed_evaluation import (
    CachedPPOStrategy,
    _run_single_seed,
)

from src.scheduler.cache import SchedulerCache
from src.scheduler.ppo_agent import PPOAgent

LOG_DIR = "logs/test_multiseed_parallel"


def _make_mock_ppo_model(action: int = 1) -> MagicMock:
    """创建返回指定动作的 mock PPO 模型。"""
    model = MagicMock()
    model.predict.return_value = (np.array([action]), None)
    return model


class TinyEnv:
    """不触发真实训练的最小 Gym 风格环境。"""

    observation_space = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)
    action_space = spaces.Discrete(3)

    def __init__(self) -> None:
        self.steps = 0

    def reset(self):
        """开始一个两步回合。"""
        self.steps = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        """返回固定奖励和可预测的结束状态。"""
        del action
        self.steps += 1
        done = self.steps >= 2
        return (
            np.full(4, self.steps / 2, dtype=np.float32),
            1.5,
            done,
            False,
            {"completion_rate": 0.75 if done else 0.0},
        )


# ---------------------------------------------------------------------------
# CachedPPOStrategy 测试
# ---------------------------------------------------------------------------


class TestCachedPPOStrategy:
    """CachedPPOStrategy 缓存策略测试。"""

    def test_cache_miss_on_first_call(self) -> None:
        """首次调用应未命中缓存，触发模型推理。"""
        model = _make_mock_ppo_model(action=1)
        cache = SchedulerCache(max_size=10)
        strategy = CachedPPOStrategy(model, cache)

        obs = np.array([0.1, 0.2, 0.3, 0.4])
        action = strategy.select_action(obs)

        assert action == 1
        stats = strategy.cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1
        model.predict.assert_called_once()

    def test_cache_hit_on_identical_state(self) -> None:
        """相同状态应命中缓存，跳过模型推理。"""
        model = _make_mock_ppo_model(action=2)
        cache = SchedulerCache(max_size=10)
        strategy = CachedPPOStrategy(model, cache)

        obs = np.array([0.5, 0.5, 0.5, 0.5])
        first_action = strategy.select_action(obs)
        second_action = strategy.select_action(obs)

        assert first_action == 2
        assert second_action == 2
        stats = strategy.cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        model.predict.assert_called_once()

    def test_cache_hit_on_similar_state(self) -> None:
        """高相似度状态应命中缓存（余弦相似度≥0.95）。"""
        model = _make_mock_ppo_model(action=0)
        cache = SchedulerCache(max_size=10, similarity_threshold=0.95)
        strategy = CachedPPOStrategy(model, cache)

        obs1 = np.array([1.0, 2.0, 3.0, 4.0])
        obs2 = np.array([1.01, 2.01, 3.01, 4.01])

        strategy.select_action(obs1)
        action2 = strategy.select_action(obs2)

        stats = strategy.cache_stats()
        assert stats["hits"] >= 1
        assert stats["misses"] == 1
        model.predict.assert_called_once()
        assert action2 == 0

    def test_cache_miss_on_dissimilar_state(self) -> None:
        """低相似度状态应未命中缓存，触发模型推理。"""
        model = _make_mock_ppo_model(action=1)
        cache = SchedulerCache(max_size=10, similarity_threshold=0.95)
        strategy = CachedPPOStrategy(model, cache)

        obs1 = np.array([1.0, 0.0, 0.0, 0.0])
        obs2 = np.array([0.0, 0.0, 0.0, 1.0])

        strategy.select_action(obs1)
        strategy.select_action(obs2)

        stats = strategy.cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 2
        assert model.predict.call_count == 2

    def test_cache_stats_returns_dict(self) -> None:
        """cache_stats 应返回包含所有键的字典。"""
        model = _make_mock_ppo_model()
        cache = SchedulerCache(max_size=5)
        strategy = CachedPPOStrategy(model, cache)

        stats = strategy.cache_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "size" in stats
        assert "evictions" in stats

    def test_name_attribute(self) -> None:
        """策略名称应为 'PPO'。"""
        model = _make_mock_ppo_model()
        cache = SchedulerCache(max_size=5)
        strategy = CachedPPOStrategy(model, cache)
        assert strategy.name == "PPO"


# ---------------------------------------------------------------------------
# PPOAgent 决策缓存集成测试
# ---------------------------------------------------------------------------


class TestPPOAgentCacheIntegration:
    """PPOAgent 决策缓存集成测试。"""

    @pytest.fixture
    def tiny_env(self) -> TinyEnv:
        """提供独立的最小环境。"""
        return TinyEnv()

    def test_predict_without_cache(self, tiny_env: TinyEnv) -> None:
        """未启用缓存时，predict 应正常调用模型推理。"""
        agent = PPOAgent(tiny_env, log_dir=LOG_DIR, verbose=0)
        model = MagicMock()
        model.predict.return_value = (np.array([1]), None)
        agent.model = model

        state = np.zeros(4, dtype=np.float32)
        action = agent.predict(state)

        assert action == 1
        model.predict.assert_called_once()
        assert agent.cache_stats() == {}

    def test_predict_with_cache_hit(self, tiny_env: TinyEnv) -> None:
        """启用缓存时，相同状态第二次调用应命中缓存。"""
        cache = SchedulerCache(max_size=10)
        agent = PPOAgent(tiny_env, log_dir=LOG_DIR, verbose=0, cache=cache)
        model = MagicMock()
        model.predict.return_value = (np.array([2]), None)
        agent.model = model

        state = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        action1 = agent.predict(state)
        action2 = agent.predict(state)

        assert action1 == 2
        assert action2 == 2
        model.predict.assert_called_once()

        stats = agent.cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_predict_with_cache_miss(self, tiny_env: TinyEnv) -> None:
        """启用缓存时，不同状态应未命中缓存。"""
        cache = SchedulerCache(max_size=10, similarity_threshold=0.99)
        agent = PPOAgent(tiny_env, log_dir=LOG_DIR, verbose=0, cache=cache)
        model = MagicMock()
        model.predict.return_value = (np.array([0]), None)
        agent.model = model

        state1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state2 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        agent.predict(state1)
        agent.predict(state2)

        stats = agent.cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 2
        assert model.predict.call_count == 2

    def test_cache_stats_empty_without_cache(self, tiny_env: TinyEnv) -> None:
        """未启用缓存时，cache_stats 应返回空字典。"""
        agent = PPOAgent(tiny_env, log_dir=LOG_DIR, verbose=0)
        assert agent.cache_stats() == {}

    def test_get_config_includes_use_cache(self, tiny_env: TinyEnv) -> None:
        """get_config 应包含 use_cache 字段。"""
        agent_no_cache = PPOAgent(tiny_env, log_dir=LOG_DIR, verbose=0)
        config_no_cache = agent_no_cache.get_config()
        assert "use_cache" in config_no_cache
        assert config_no_cache["use_cache"] is False

        cache = SchedulerCache(max_size=5)
        agent_with_cache = PPOAgent(tiny_env, log_dir=LOG_DIR, verbose=0, cache=cache)
        config_with_cache = agent_with_cache.get_config()
        assert config_with_cache["use_cache"] is True

    def test_predict_2d_state_with_cache(self, tiny_env: TinyEnv) -> None:
        """启用缓存时，2D 状态不应被 reshape，直接传给模型。"""
        cache = SchedulerCache(max_size=10)
        agent = PPOAgent(tiny_env, log_dir=LOG_DIR, verbose=0, cache=cache)
        model = MagicMock()
        model.predict.return_value = (np.array([1]), None)
        agent.model = model

        state_2d = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        action = agent.predict(state_2d)

        assert action == 1
        called_state = model.predict.call_args.args[0]
        assert called_state.shape == (1, 4)


# ---------------------------------------------------------------------------
# _run_single_seed 测试
# ---------------------------------------------------------------------------


class TestRunSingleSeed:
    """_run_single_seed 函数测试。"""

    def test_returns_correct_structure(self, monkeypatch) -> None:
        """测试返回值为4元组且结构正确。"""
        simple_strategies = [FCFSStrategy(), RandomStrategy(action_dim=3, seed=42)]

        def mock_build_strategies(
            dqn_path: str | None = None,
            ppo_path: str | None = None,
            obs_dim: int = 10,
        ) -> list:
            del dqn_path, ppo_path, obs_dim
            return simple_strategies

        monkeypatch.setattr("run_multiseed_evaluation.build_strategies", mock_build_strategies)

        result = _run_single_seed(
            seed=42,
            seed_idx=0,
            total_seeds=1,
            ppo_model="fake_ppo.zip",
            dqn_model=None,
            obs_dim=14,
            episodes_per_seed=1,
            tasks_per_episode=5,
            use_cache=False,
        )

        assert len(result) == 4
        seed, seed_data, rewards_map, elapsed = result

        assert seed == 42
        assert isinstance(seed_data, dict)
        assert isinstance(rewards_map, dict)
        assert isinstance(elapsed, float)
        assert elapsed >= 0

        for sname in ["FCFS", "Random"]:
            assert sname in seed_data
            assert "mean_reward" in seed_data[sname]
            assert "std_reward" in seed_data[sname]
            assert "rewards" in seed_data[sname]
            assert len(seed_data[sname]["rewards"]) == 1

            assert sname in rewards_map
            assert len(rewards_map[sname]) == 1

    def test_use_cache_wraps_ppo_strategy(self, monkeypatch) -> None:
        """use_cache=True 时应将 PPO 策略包装为 CachedPPOStrategy。"""
        from run_issue_38_67_experiments import PPOStrategy

        mock_ppo_model = _make_mock_ppo_model(action=1)
        ppo_strategy = PPOStrategy(mock_ppo_model)

        def mock_build_strategies(
            dqn_path: str | None = None,
            ppo_path: str | None = None,
            obs_dim: int = 10,
        ) -> list:
            del dqn_path, ppo_path, obs_dim
            return [FCFSStrategy(), ppo_strategy]

        monkeypatch.setattr("run_multiseed_evaluation.build_strategies", mock_build_strategies)

        result = _run_single_seed(
            seed=42,
            seed_idx=0,
            total_seeds=1,
            ppo_model="fake_ppo.zip",
            dqn_model=None,
            obs_dim=14,
            episodes_per_seed=1,
            tasks_per_episode=5,
            use_cache=True,
        )

        seed, seed_data, rewards_map, _ = result
        assert seed == 42
        assert "PPO" in seed_data
        assert "FCFS" in seed_data
        assert len(rewards_map["PPO"]) == 1

    def test_multiple_episodes(self, monkeypatch) -> None:
        """测试多 episode 场景下奖励列表长度正确。"""
        simple_strategies = [FCFSStrategy()]

        def mock_build_strategies(
            dqn_path: str | None = None,
            ppo_path: str | None = None,
            obs_dim: int = 10,
        ) -> list:
            del dqn_path, ppo_path, obs_dim
            return simple_strategies

        monkeypatch.setattr("run_multiseed_evaluation.build_strategies", mock_build_strategies)

        result = _run_single_seed(
            seed=99,
            seed_idx=2,
            total_seeds=5,
            ppo_model="fake.zip",
            dqn_model=None,
            obs_dim=14,
            episodes_per_seed=3,
            tasks_per_episode=5,
            use_cache=False,
        )

        seed, seed_data, rewards_map, _ = result
        assert seed == 99
        assert len(rewards_map["FCFS"]) == 3
        assert len(seed_data["FCFS"]["rewards"]) == 3

    def test_obs_dim_passed_to_build_strategies(self, monkeypatch) -> None:
        """Issue #435: _run_single_seed 应将 obs_dim 传递给 build_strategies。

        原实现漏传 obs_dim，导致 build_strategies 使用默认值 10，
        与主流程 run_multiseed 传入的 obs_dim=14 不一致，
        加载 14 维 DQN 检查点时维度不匹配崩溃。
        """
        captured_obs_dim: list[int] = []
        simple_strategies = [FCFSStrategy()]

        def mock_build_strategies(
            dqn_path: str | None = None,
            ppo_path: str | None = None,
            obs_dim: int = 10,
        ) -> list:
            del dqn_path, ppo_path
            captured_obs_dim.append(obs_dim)
            return simple_strategies

        monkeypatch.setattr("run_multiseed_evaluation.build_strategies", mock_build_strategies)

        _run_single_seed(
            seed=42,
            seed_idx=0,
            total_seeds=1,
            ppo_model="fake_ppo.zip",
            dqn_model=None,
            obs_dim=14,
            episodes_per_seed=1,
            tasks_per_episode=5,
            use_cache=False,
        )

        # Issue #435: obs_dim 应被正确传递，而非使用默认值 10
        assert len(captured_obs_dim) == 1
        assert captured_obs_dim[0] == 14, (
            f"obs_dim 应为 14，实际传递 {captured_obs_dim[0]}（#435 回归）"
        )

    def test_obs_dim_10_passed_correctly(self, monkeypatch) -> None:
        """Issue #435: obs_dim=10 时也应正确传递。"""
        captured_obs_dim: list[int] = []
        simple_strategies = [FCFSStrategy()]

        def mock_build_strategies(
            dqn_path: str | None = None,
            ppo_path: str | None = None,
            obs_dim: int = 10,
        ) -> list:
            del dqn_path, ppo_path
            captured_obs_dim.append(obs_dim)
            return simple_strategies

        monkeypatch.setattr("run_multiseed_evaluation.build_strategies", mock_build_strategies)

        _run_single_seed(
            seed=42,
            seed_idx=0,
            total_seeds=1,
            ppo_model="fake_ppo.zip",
            dqn_model=None,
            obs_dim=10,
            episodes_per_seed=1,
            tasks_per_episode=5,
            use_cache=False,
        )

        assert len(captured_obs_dim) == 1
        assert captured_obs_dim[0] == 10
