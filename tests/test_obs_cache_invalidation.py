"""Issue #775: 观测缓存失效保护单元测试。

验证所有修改 env 聚合/队列状态的 mutator（含委托给子模块的
route_to_machine / recompute_aggregate / advance_time / pick_next_task）
在状态变更后失效观测缓存，避免外部绕开 step() 直接调用时返回过期观测。

不依赖 torch；CI 需 gymnasium + numpy。
"""

import pytest

from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.env_dynamics import advance_time, pick_next_task
from src.scheduler.env_machines import recompute_aggregate, route_to_machine


@pytest.fixture
def env():
    e = QuantumSchedulingEnv(max_steps=20, max_qubits=20, seed=42)
    e.reset()
    return e


def _fill_cache(e: QuantumSchedulingEnv) -> None:
    e._get_observation()
    assert e._cached_obs is not None, "precondition: cache should be filled"


def test_step_keeps_cache_consistent(env: QuantumSchedulingEnv) -> None:
    """回归：step() 内部仍能正常填充/失效缓存，行为不变。"""
    _fill_cache(env)
    obs, _, _, _, _ = env.step(0)
    assert obs is not None
    assert env._cached_obs is not None


def test_route_to_machine_invalidates_cache(env: QuantumSchedulingEnv) -> None:
    _fill_cache(env)
    task = env._current_task
    assert task is not None
    route_to_machine(env, env._machines[0], task, env.np_random)
    assert env._cached_obs is None


def test_recompute_aggregate_invalidates_cache(env: QuantumSchedulingEnv) -> None:
    _fill_cache(env)
    recompute_aggregate(env)
    assert env._cached_obs is None


def test_advance_time_invalidates_cache(env: QuantumSchedulingEnv) -> None:
    _fill_cache(env)
    advance_time(env, env.np_random)
    assert env._cached_obs is None


def test_pick_next_task_invalidates_cache(env: QuantumSchedulingEnv) -> None:
    _fill_cache(env)
    pick_next_task(env)
    assert env._cached_obs is None


def test_get_observation_recomputes_after_invalidation(env: QuantumSchedulingEnv) -> None:
    """失效后 _get_observation() 应重新构建（不抛错、维度正确）。"""
    _fill_cache(env)
    route_to_machine(env, env._machines[0], env._current_task, env.np_random)
    assert env._cached_obs is None
    obs = env._get_observation()
    assert obs is not None
    assert obs.shape[0] in (16, 17)


def test_invalidation_is_idempotent(env: QuantumSchedulingEnv) -> None:
    """多次失效无副作用。"""
    _fill_cache(env)
    env._invalidate_obs_cache()
    env._invalidate_obs_cache()
    assert env._cached_obs is None
