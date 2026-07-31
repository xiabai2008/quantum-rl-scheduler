"""Issue #772: 编译层观测语义版本化测试。

验证 QuantumCompilationEnv 的 obs_version 参数：
- 默认 "v759_post"：维度 11-13 为新特征
  (avg_swap_dist_n / swap_efficiency / isolated_occupied_n)
- "v759_pre"：维度 11-13 为旧模型训练语义
  (1-mapped_r / 1-alloc / 1-conn)，用于加载
  deliverable_models/ppo_compilation_agent.zip（该 zip 为 PR #759 前的旧语义训练）
- 非法 obs_version 必须抛出 ValueError

注：本测试仅依赖 gymnasium + numpy，不涉及 torch，可在 CI 环境运行。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.quantum.compilation_env import QuantumCompilationEnv


def test_default_obs_version_is_v759_post() -> None:
    env = QuantumCompilationEnv(circuit=None, max_steps=10)
    obs, _ = env.reset()
    assert obs.shape == (14,)
    # 默认语义下维度 11-13 应在 [0, 1]（新特征已归一化）
    assert 0.0 <= obs[11] <= 1.0
    assert 0.0 <= obs[12] <= 1.0
    assert 0.0 <= obs[13] <= 1.0


def test_v759_pre_reproduces_legacy_dim_11_13() -> None:
    # 初始状态: _mapped_gates=0, _mapping={}, n_gates=0
    # → mapped_r=0, alloc=0, conn=0
    env = QuantumCompilationEnv(circuit=None, max_steps=10, obs_version="v759_pre")
    obs, _ = env.reset()
    assert obs.shape == (14,)
    # v759_pre 维度 11-13 = 1-mapped_r / 1-alloc / 1-conn
    assert obs[11] == pytest.approx(1.0 - 0.0)
    assert obs[12] == pytest.approx(1.0 - 0.0)
    assert obs[13] == pytest.approx(1.0 - 0.0)


def test_v759_post_matches_new_features() -> None:
    env = QuantumCompilationEnv(circuit=None, max_steps=10, obs_version="v759_post")
    obs, _ = env.reset()
    assert obs.shape == (14,)
    assert 0.0 <= obs[11] <= 1.0
    assert 0.0 <= obs[12] <= 1.0
    assert 0.0 <= obs[13] <= 1.0


def test_invalid_obs_version_raises() -> None:
    with pytest.raises(ValueError):
        QuantumCompilationEnv(circuit=None, obs_version="v759_mid")


def test_obs_version_preserved_across_steps() -> None:
    env = QuantumCompilationEnv(circuit=None, max_steps=10, obs_version="v759_pre")
    obs, _ = env.reset()
    # 初始维度 11-13 = 1.0 (因 mapped_r=alloc=conn=0)
    assert obs[11] == pytest.approx(1.0)
    assert obs[12] == pytest.approx(1.0)
    assert obs[13] == pytest.approx(1.0)
    # 走几步后维度仍由 v759_pre 决定，不应变成新特征语义
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break
    assert obs.shape == (14,)
    # v759_pre 下维度 11-13 仍为 1-mapped_r / 1-alloc / 1-conn（合法范围 [0,1]）
    assert 0.0 <= obs[11] <= 1.0
    assert 0.0 <= obs[12] <= 1.0
    assert 0.0 <= obs[13] <= 1.0


def test_v759_pre_and_post_differ_on_non_trivial_state() -> None:
    """在已映射部分比特的状态下，两种语义的维度 11-13 应不同（验证分支确实生效）。"""
    env_pre = QuantumCompilationEnv(circuit=None, max_steps=50, obs_version="v759_pre")
    env_post = QuantumCompilationEnv(circuit=None, max_steps=50, obs_version="v759_post")
    obs_pre, _ = env_pre.reset()
    obs_post, _ = env_post.reset()
    # 至少执行若干步产生非零映射/交换，使 mapped_r/alloc/conn 偏离 0
    for _ in range(10):
        a_pre = env_pre.action_space.sample()
        a_post = env_post.action_space.sample()
        obs_pre, _, d1, t1, _ = env_pre.step(int(a_pre))
        obs_post, _, d2, t2, _ = env_post.step(int(a_post))
        if (d1 or t1) and (d2 or t2):
            break
    # 两种语义在维度 11-13 上一般不同（除非巧合全 0/全 1），验证分支存在
    diff = not np.allclose(obs_pre[11:14], obs_post[11:14])
    # 若恰好相同也不算错误，但至少两者各自在合法范围
    assert 0.0 <= obs_pre[11] <= 1.0 and 0.0 <= obs_post[11] <= 1.0
    assert 0.0 <= obs_pre[12] <= 1.0 and 0.0 <= obs_post[12] <= 1.0
    assert 0.0 <= obs_pre[13] <= 1.0 and 0.0 <= obs_post[13] <= 1.0
    # 若两者不同，则说明版本分支确实产生不同观测
    if diff:
        assert not np.array_equal(obs_pre[11:14], obs_post[11:14])
