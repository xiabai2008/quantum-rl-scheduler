"""
学习率调度器单元测试（Issue #403）

测试覆盖：
    1. create_lr_schedule 工具函数：
       - linear: 线性衰减
       - cosine: 余弦退火
       - constant: 恒定不变
       - 边界与异常：base_lr 非正、未知 schedule_type
    2. PPOAgent lr_schedule 集成：
       - 默认 lr_schedule="linear"
       - lr_schedule="constant" 向后兼容
       - lr_schedule="cosine" 余弦退火
       - get_config 包含 lr_schedule 字段
    3. SchedulerAgent (DQN) lr_schedule 集成：
       - 默认 lr_schedule="linear"
       - get_config 包含 lr_schedule 字段
    4. MultiAgentPPO (MAPPO) lr_schedule 集成：
       - 默认 lr_schedule="linear"
       - _update_learning_rate 正确更新优化器 lr
       - get_config 包含 lr_schedule 字段
       - constant 模式下训练中 lr 不变
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.lr_schedule import (
    SUPPORTED_SCHEDULE_TYPES,
    compute_lr_at_progress,
    create_lr_schedule,
)


# ============================================================================
# 1. create_lr_schedule 工具函数测试
# ============================================================================
class TestCreateLrSchedule:
    """create_lr_schedule 函数测试。"""

    def test_linear_schedule_start(self) -> None:
        """线性调度：progress_remaining=1.0 时 lr=base_lr。"""
        lr_fn = create_lr_schedule(3e-4, "linear")
        assert lr_fn(1.0) == pytest.approx(3e-4)

    def test_linear_schedule_mid(self) -> None:
        """线性调度：progress_remaining=0.5 时 lr=base_lr*0.5。"""
        lr_fn = create_lr_schedule(3e-4, "linear")
        assert lr_fn(0.5) == pytest.approx(1.5e-4)

    def test_linear_schedule_end(self) -> None:
        """线性调度：progress_remaining=0.0 时 lr=0。"""
        lr_fn = create_lr_schedule(3e-4, "linear")
        assert lr_fn(0.0) == pytest.approx(0.0)

    def test_linear_schedule_negative_progress_clamped(self) -> None:
        """线性调度：progress_remaining<0 时 lr 被截断为 0。"""
        lr_fn = create_lr_schedule(3e-4, "linear")
        assert lr_fn(-0.5) == pytest.approx(0.0)

    def test_cosine_schedule_start(self) -> None:
        """余弦退火：progress_remaining=1.0 时 lr=base_lr。"""
        lr_fn = create_lr_schedule(1e-3, "cosine")
        assert lr_fn(1.0) == pytest.approx(1e-3, rel=1e-6)

    def test_cosine_schedule_end(self) -> None:
        """余弦退火：progress_remaining=0.0 时 lr=0。"""
        lr_fn = create_lr_schedule(1e-3, "cosine")
        assert lr_fn(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_cosine_schedule_mid(self) -> None:
        """余弦退火：progress_remaining=0.5 时 lr=base_lr*0.5。"""
        lr_fn = create_lr_schedule(1e-3, "cosine")
        # progress=0.5, cos(pi*0.5)=0, lr = base * 0.5 * (1+0) = base*0.5
        assert lr_fn(0.5) == pytest.approx(5e-4, rel=1e-6)

    def test_cosine_schedule_monotonic_decreasing(self) -> None:
        """余弦退火：lr 应随 progress_remaining 递减而单调递减。"""
        lr_fn = create_lr_schedule(1e-3, "cosine")
        progress_values = np.linspace(1.0, 0.0, 20)
        lr_values = [lr_fn(p) for p in progress_values]
        for i in range(len(lr_values) - 1):
            assert lr_values[i] >= lr_values[i + 1]

    def test_constant_schedule_always_base_lr(self) -> None:
        """恒定调度：任何 progress_remaining 时 lr=base_lr。"""
        lr_fn = create_lr_schedule(5e-4, "constant")
        for p in [1.0, 0.75, 0.5, 0.25, 0.0, -0.1]:
            assert lr_fn(p) == pytest.approx(5e-4)

    def test_linear_schedule_monotonic_decreasing(self) -> None:
        """线性调度：lr 应随 progress_remaining 递减而单调递减。"""
        lr_fn = create_lr_schedule(3e-4, "linear")
        progress_values = np.linspace(1.0, 0.0, 20)
        lr_values = [lr_fn(p) for p in progress_values]
        for i in range(len(lr_values) - 1):
            assert lr_values[i] >= lr_values[i + 1]


class TestCreateLrScheduleErrors:
    """create_lr_schedule 异常输入测试。"""

    def test_negative_base_lr_raises(self) -> None:
        """base_lr 为负数应抛出 ValueError。"""
        with pytest.raises(ValueError, match="base_lr"):
            create_lr_schedule(-1e-4, "linear")

    def test_zero_base_lr_raises(self) -> None:
        """base_lr 为 0 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="base_lr"):
            create_lr_schedule(0.0, "linear")

    def test_unknown_schedule_type_raises(self) -> None:
        """未知的 schedule_type 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="schedule_type"):
            create_lr_schedule(3e-4, "exponential")  # type: ignore[arg-type]

    def test_supported_types_constant(self) -> None:
        """SUPPORTED_SCHEDULE_TYPES 应包含三种类型。"""
        assert set(SUPPORTED_SCHEDULE_TYPES) == {"linear", "cosine", "constant"}


class TestComputeLrAtProgress:
    """compute_lr_at_progress 便捷函数测试。"""

    def test_matches_schedule_fn(self) -> None:
        """compute_lr_at_progress 应与 create_lr_schedule 返回的函数一致。"""
        base_lr = 7e-4
        for schedule_type in ("linear", "cosine", "constant"):
            lr_fn = create_lr_schedule(base_lr, schedule_type)  # type: ignore[arg-type]
            for p in [1.0, 0.7, 0.3, 0.0]:
                direct = compute_lr_at_progress(base_lr, schedule_type, p)  # type: ignore[arg-type]
                assert direct == pytest.approx(lr_fn(p))


# ============================================================================
# 2. PPOAgent lr_schedule 集成测试
# ============================================================================
class TestPPOAgentLrSchedule:
    """PPOAgent 学习率调度集成测试。"""

    def test_default_lr_schedule_is_linear(self) -> None:
        """PPOAgent 默认 lr_schedule 应为 'linear'。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=10, seed=42)
        agent_cls = __import__("src.scheduler.ppo_agent", fromlist=["PPOAgent"]).PPOAgent
        agent = agent_cls(env, verbose=0)
        assert agent.lr_schedule == "linear"

    def test_constant_lr_schedule_backward_compat(self) -> None:
        """lr_schedule='constant' 时 _lr_fn 应始终返回固定值。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=10, seed=42)
        agent_cls = __import__("src.scheduler.ppo_agent", fromlist=["PPOAgent"]).PPOAgent
        agent = agent_cls(env, learning_rate=1e-3, lr_schedule="constant", verbose=0)
        for p in [1.0, 0.5, 0.0]:
            assert agent._lr_fn(p) == pytest.approx(1e-3)

    def test_cosine_lr_schedule(self) -> None:
        """lr_schedule='cosine' 时 _lr_fn 应符合余弦退火公式。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=10, seed=42)
        agent_cls = __import__("src.scheduler.ppo_agent", fromlist=["PPOAgent"]).PPOAgent
        agent = agent_cls(env, learning_rate=2e-4, lr_schedule="cosine", verbose=0)
        # start
        assert agent._lr_fn(1.0) == pytest.approx(2e-4, rel=1e-6)
        # end
        assert agent._lr_fn(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_get_config_includes_lr_schedule(self) -> None:
        """get_config 应包含 lr_schedule 字段。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=10, seed=42)
        agent_cls = __import__("src.scheduler.ppo_agent", fromlist=["PPOAgent"]).PPOAgent
        agent = agent_cls(env, lr_schedule="cosine", verbose=0)
        config = agent.get_config()
        assert "lr_schedule" in config
        assert config["lr_schedule"] == "cosine"


# ============================================================================
# 3. SchedulerAgent (DQN) lr_schedule 集成测试
# ============================================================================
class TestSchedulerAgentLrSchedule:
    """SchedulerAgent (DQN) 学习率调度集成测试。"""

    def test_default_lr_schedule_is_linear(self) -> None:
        """SchedulerAgent 默认 lr_schedule 应为 'linear'。"""
        from src.scheduler.agent import SchedulerAgent
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=10, seed=42)
        agent = SchedulerAgent(env, verbose=0)
        assert agent.lr_schedule == "linear"

    def test_constant_lr_schedule(self) -> None:
        """lr_schedule='constant' 时 _lr_fn 应始终返回固定值。"""
        from src.scheduler.agent import SchedulerAgent
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=10, seed=42)
        agent = SchedulerAgent(env, learning_rate=5e-4, lr_schedule="constant", verbose=0)
        for p in [1.0, 0.5, 0.0]:
            assert agent._lr_fn(p) == pytest.approx(5e-4)

    def test_get_config_includes_lr_schedule(self) -> None:
        """get_config 应包含 lr_schedule 字段。"""
        from src.scheduler.agent import SchedulerAgent
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=10, seed=42)
        agent = SchedulerAgent(env, lr_schedule="cosine", verbose=0)
        config = agent.get_config()
        assert "lr_schedule" in config
        assert config["lr_schedule"] == "cosine"


# ============================================================================
# 4. MultiAgentPPO (MAPPO) lr_schedule 集成测试
# ============================================================================
class TestMAPPOLrSchedule:
    """MultiAgentPPO 学习率调度集成测试。"""

    @staticmethod
    def _make_agent(lr_schedule: str = "linear"):
        """创建快速测试用 MAPPO 智能体。"""
        from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv
        from src.scheduler.marl import MultiAgentPPO

        env = QuantumSchedulingEnv(
            max_steps=20,
            machine_configs=DEFAULT_MACHINE_CONFIGS[:2],
            seed=42,
        )
        return MultiAgentPPO(
            env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            actor_hidden=(8,),
            critic_hidden=(8,),
            seed=42,
            verbose=0,
            lr_schedule=lr_schedule,  # type: ignore[arg-type]
        )

    def test_default_lr_schedule_is_linear(self) -> None:
        """MAPPO 默认 lr_schedule 应为 'linear'。"""
        agent = self._make_agent()
        assert agent.lr_schedule == "linear"

    def test_constant_lr_schedule(self) -> None:
        """lr_schedule='constant' 时 _lr_fn 应始终返回固定值。"""
        agent = self._make_agent("constant")
        for p in [1.0, 0.5, 0.0]:
            assert agent._lr_fn(p) == pytest.approx(3e-4)

    def test_get_config_includes_lr_schedule(self) -> None:
        """get_config 应包含 lr_schedule 字段。"""
        agent = self._make_agent("cosine")
        config = agent.get_config()
        assert "lr_schedule" in config
        assert config["lr_schedule"] == "cosine"

    def test_update_learning_rate_changes_optimizer_lr(self) -> None:
        """_update_learning_rate 应正确更新优化器 param_groups 的 lr。"""
        agent = self._make_agent("linear")
        agent.total_timesteps = 50
        total_timesteps = 100
        agent._update_learning_rate(total_timesteps)
        # progress_remaining = 1 - 50/100 = 0.5
        expected_lr = 3e-4 * 0.5
        for opt in agent.actor_optimizers:
            assert opt.param_groups[0]["lr"] == pytest.approx(expected_lr, rel=1e-6)
        assert agent.critic_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr, rel=1e-6)

    def test_update_learning_rate_zero_timesteps_noop(self) -> None:
        """total_timesteps=0 时 _update_learning_rate 应为空操作。"""
        agent = self._make_agent("linear")
        original_lr = agent.actor_optimizers[0].param_groups[0]["lr"]
        agent._update_learning_rate(0)
        assert agent.actor_optimizers[0].param_groups[0]["lr"] == pytest.approx(original_lr)

    def test_constant_schedule_lr_unchanged_after_update(self) -> None:
        """constant 模式下 _update_learning_rate 不应改变 lr。"""
        agent = self._make_agent("constant")
        agent.total_timesteps = 50
        agent._update_learning_rate(100)
        # constant 模式下 lr 始终为初始值
        assert agent.actor_optimizers[0].param_groups[0]["lr"] == pytest.approx(3e-4)
        assert agent.critic_optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)

    def test_cosine_schedule_lr_decreases_in_train(self) -> None:
        """cosine 模式训练后 lr 应低于初始值。"""
        agent = self._make_agent("cosine")
        initial_lr = agent.actor_optimizers[0].param_groups[0]["lr"]
        # 训练少量步数
        agent.train(total_timesteps=16, eval_freq=0)
        final_lr = agent.actor_optimizers[0].param_groups[0]["lr"]
        # 训练后 lr 应低于初始值
        assert final_lr < initial_lr
