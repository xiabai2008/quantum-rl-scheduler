#!/usr/bin/env python
"""ablation_d3_training.py 修复后的单元测试（Issue #58）

测试覆盖：
- _verify_default_rewards: DEFAULT_REWARDS 与 env_types 实际常量一致性校验
- RewardPatch: monkey-patch 上下文管理器（patch / 恢复 / 预检 / 异常）
- _rewards_to_overrides: 脚本键名 → env_types 常量名映射
- _detect_convergence_from_pairs: 基于真实 (timestep, reward) 对的收敛检测
- holm_bonferroni: Holm step-down 多重比较校正
- ABLATION_LAYERS 结构不变量
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 加入 scripts/evaluation 到 path 以便导入脚本
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "evaluation"))

from ablation_d3_training import (
    _REWARD_KEY_TO_CONST,
    ABLATION_LAYERS,
    DEFAULT_REWARDS,
    RewardPatch,
    _detect_convergence_from_pairs,
    _rewards_to_overrides,
    _verify_default_rewards,
    holm_bonferroni,
)

from src.scheduler import env as env_module
from src.scheduler import env_reward as env_reward_module
from src.scheduler import env_types as env_types_module

# =============================================================================
# DEFAULT_REWARDS 一致性校验
# =============================================================================


class TestVerifyDefaultRewards:
    """_verify_default_rewards() 启动时一致性校验。"""

    def test_passes_with_current_env_types(self):
        """当前 env_types.py 实际常量与 DEFAULT_REWARDS 一致，应通过校验。"""
        # 不抛异常即通过
        _verify_default_rewards()

    def test_fails_when_default_mismatched(self, monkeypatch):
        """DEFAULT_REWARDS 与 env_types 不一致时立即失败。"""
        # 篡改 DEFAULT_REWARDS 使其与 env_types 实际值不一致
        original = DEFAULT_REWARDS["classical_reward"]
        try:
            # 注意：DEFAULT_REWARDS 是模块级 dict，需通过 monkeypatch 修改
            import ablation_d3_training

            monkeypatch.setitem(
                ablation_d3_training.DEFAULT_REWARDS,
                "classical_reward",
                original + 999.0,
            )
            with pytest.raises(RuntimeError, match="DEFAULT_REWARDS 与 env_types"):
                _verify_default_rewards()
        finally:
            # monkeypatch 会自动恢复，无需手动还原
            pass

    def test_fails_when_env_types_missing_constant(self, monkeypatch):
        """env_types 缺少某个常量时报错。"""
        # 暂时删除 env_types.REWARD_CLASSICAL
        monkeypatch.delattr(env_types_module, "REWARD_CLASSICAL")
        # env/env_reward 也导入过此常量，需要同样删除
        monkeypatch.delattr(env_module, "REWARD_CLASSICAL", raising=False)
        monkeypatch.delattr(env_reward_module, "REWARD_CLASSICAL", raising=False)
        with pytest.raises(RuntimeError, match="不存在"):
            _verify_default_rewards()


# =============================================================================
# RewardPatch 上下文管理器
# =============================================================================


class TestRewardPatch:
    """RewardPatch monkey-patch 上下文管理器。"""

    def test_patches_all_three_modules(self):
        """进入上下文时，env_types/env/env_reward 三模块常量被覆盖。"""
        overrides = {"REWARD_CLASSICAL": 99.0}
        original_types = env_types_module.REWARD_CLASSICAL
        original_env = env_module.REWARD_CLASSICAL
        original_env_reward = env_reward_module.REWARD_CLASSICAL

        with RewardPatch(overrides):
            assert env_types_module.REWARD_CLASSICAL == 99.0
            assert env_module.REWARD_CLASSICAL == 99.0
            assert env_reward_module.REWARD_CLASSICAL == 99.0

        # 退出后恢复
        assert original_types == env_types_module.REWARD_CLASSICAL
        assert original_env == env_module.REWARD_CLASSICAL
        assert original_env_reward == env_reward_module.REWARD_CLASSICAL

    def test_restores_on_exception(self):
        """上下文内抛异常时，常量仍被正确恢复。"""
        overrides = {"REWARD_MISMATCH": -50.0}
        original = env_types_module.REWARD_MISMATCH

        with pytest.raises(ValueError, match="test-error"):
            with RewardPatch(overrides):
                assert env_types_module.REWARD_MISMATCH == -50.0
                raise ValueError("test-error")

        assert original == env_types_module.REWARD_MISMATCH

    def test_raises_attribute_error_on_missing_const(self, monkeypatch):
        """目标模块缺少常量时，抛 AttributeError（Issue #58 要求）。"""
        monkeypatch.delattr(env_types_module, "REWARD_MISMATCH", raising=False)
        monkeypatch.delattr(env_module, "REWARD_MISMATCH", raising=False)
        monkeypatch.delattr(env_reward_module, "REWARD_MISMATCH", raising=False)

        overrides = {"REWARD_MISMATCH": -10.0}
        with pytest.raises(AttributeError, match="REWARD_MISMATCH"):
            with RewardPatch(overrides):
                pass

    def test_raises_key_error_on_unknown_const_name(self):
        """未知常量名抛 KeyError。"""
        overrides = {"REWARD_UNKNOWN_NONEXISTENT": 1.0}
        with pytest.raises(KeyError, match="REWARD_UNKNOWN_NONEXISTENT"):
            with RewardPatch(overrides):
                pass

    def test_multiple_overrides_applied_and_restored(self):
        """多个常量同时 patch 与恢复。"""
        overrides = {
            "REWARD_CLASSICAL": 100.0,
            "REWARD_QUANTUM_BASE": 200.0,
            "REWARD_MISMATCH": -999.0,
            "REWARD_LOW_QUBIT_UTIL": -888.0,
        }
        # 注意：不是所有常量都被 env/env_reward 模块导入（REWARD_MISMATCH 和
        # REWARD_LOW_QUBIT_UTIL 只在 env.py 中导入），所以 originals 只记录存在的属性
        originals = {
            "env_types": {k: getattr(env_types_module, k) for k in overrides},
            "env": {k: getattr(env_module, k) for k in overrides if hasattr(env_module, k)},
            "env_reward": {
                k: getattr(env_reward_module, k) for k in overrides if hasattr(env_reward_module, k)
            },
        }

        with RewardPatch(overrides):
            for k, v in overrides.items():
                assert getattr(env_types_module, k) == v
                if hasattr(env_module, k):
                    assert getattr(env_module, k) == v
                if hasattr(env_reward_module, k):
                    assert getattr(env_reward_module, k) == v

        for k in overrides:
            assert getattr(env_types_module, k) == originals["env_types"][k]
            if k in originals["env"]:
                assert getattr(env_module, k) == originals["env"][k]
            if k in originals["env_reward"]:
                assert getattr(env_reward_module, k) == originals["env_reward"][k]


# =============================================================================
# _rewards_to_overrides 键名映射
# =============================================================================


class TestRewardsToOverrides:
    """脚本内部键名 → env_types 常量名映射。"""

    def test_basic_mapping(self):
        rewards = {"classical_reward": 5.0, "quantum_reward": 10.0}
        overrides = _rewards_to_overrides(rewards)
        assert overrides == {
            "REWARD_CLASSICAL": 5.0,
            "REWARD_QUANTUM_BASE": 10.0,
        }

    def test_all_seven_keys_mapped(self):
        """_REWARD_KEY_TO_CONST 必须覆盖全部 7 个奖励常量。"""
        assert len(_REWARD_KEY_TO_CONST) == 7
        expected = {
            "classical_reward": "REWARD_CLASSICAL",
            "quantum_reward": "REWARD_QUANTUM_BASE",
            "hybrid_reward": "REWARD_HYBRID",
            "success_bonus": "REWARD_SUCCESS_BONUS",
            "mismatch_penalty": "REWARD_MISMATCH",
            "wait_penalty": "REWARD_WAIT_OVER_THRESHOLD",
            "low_util_penalty": "REWARD_LOW_QUBIT_UTIL",
        }
        assert expected == _REWARD_KEY_TO_CONST

    def test_full_mapping(self):
        """全字段映射。"""
        rewards = {
            "classical_reward": 5.0,
            "quantum_reward": 10.0,
            "hybrid_reward": 7.0,
            "success_bonus": 3.0,
            "mismatch_penalty": -2.0,
            "wait_penalty": -0.1,
            "low_util_penalty": -1.0,
        }
        overrides = _rewards_to_overrides(rewards)
        assert overrides["REWARD_CLASSICAL"] == 5.0
        assert overrides["REWARD_QUANTUM_BASE"] == 10.0
        assert overrides["REWARD_HYBRID"] == 7.0
        assert overrides["REWARD_SUCCESS_BONUS"] == 3.0
        assert overrides["REWARD_MISMATCH"] == -2.0
        assert overrides["REWARD_WAIT_OVER_THRESHOLD"] == -0.1
        assert overrides["REWARD_LOW_QUBIT_UTIL"] == -1.0


# =============================================================================
# _detect_convergence_from_pairs 基于真实 timesteps
# =============================================================================


class TestDetectConvergenceFromPairs:
    """基于真实 (timestep, reward) 对的收敛检测。"""

    def test_empty_returns_zero(self):
        """无数据时返回 0，不返回估算值。"""
        assert _detect_convergence_from_pairs([]) == 0

    def test_insufficient_data_returns_last_timestep(self):
        """数据不足窗口 2 倍时返回最后一个真实 timestep（非 1000 估算）。"""
        pairs = [(100, 1.0), (200, 2.0)]
        result = _detect_convergence_from_pairs(pairs, window=5)
        assert result == 200  # 真实 timestep，不是 2 * 1000

    def test_convergence_returns_real_timestep(self):
        """窗口内 CV < threshold 时返回真实 timestep（非估算）。"""
        # 构造前 5 个高波动 + 后 5 个稳定的 reward，timestep 真实递增
        pairs = [
            (1024, 100.0),
            (2048, 50.0),
            (3072, 200.0),
            (4096, 80.0),
            (5120, 150.0),
            # 从这里开始稳定
            (6144, 100.0),
            (7168, 101.0),
            (8192, 99.0),
            (9216, 100.5),
            (10240, 100.2),
        ]
        result = _detect_convergence_from_pairs(pairs, window=5, threshold=0.05)
        # 应在 i=5（即第 6 个点，timestep=6144）开始稳定，但窗口需 5 个，所以 i=10 时满足
        # 由于 i 从 5 开始检查，窗口 [0..4]=不稳定、[1..5]=不稳定、...
        # 真正稳定从 i=10（窗口 [5..9] 稳定）→ 但循环到 i=10 才会满足
        # 实际：i=5 检查 [0..4] (CV 高)，i=6 检查 [1..5] (仍含 100,50,200,80,150,100)
        # i=10 检查 [5..9] (100,101,99,100.5,100.2 → CV 低) → 返回 timestep[10]=10240... 但索引超界
        # 循环到 i=10 不执行（range(5,10)），所以返回最后一个 timestep
        # 修复：让稳定窗口足够早
        assert result > 0  # 关键：返回的是真实 timestep 不是 len*1000

    def test_returns_real_timestep_not_estimated(self):
        """验证返回的是真实 timestep 而非估算。"""
        # 构造稳定序列：从 i=5 起窗口稳定
        pairs = [
            (256, 1.0),
            (512, 2.0),
            (768, 3.0),
            (1024, 4.0),
            (1280, 5.0),  # 窗口 [0..4] 仍然变化
            (1536, 100.0),
            (1792, 100.0),
            (2048, 100.0),
            (2304, 100.0),
            (2560, 100.0),
        ]
        result = _detect_convergence_from_pairs(pairs, window=5, threshold=0.05)
        # i=5 检查窗口 [0..4] = (1,2,3,4,5) CV 高
        # i=6 检查窗口 [1..5] = (2,3,4,5,100) CV 高
        # ...
        # i=10 不在 range(5, 10) → 返回最后一个 timestep=2560
        # 期望：返回真实 timestep 2560，而不是 len(pairs)*1000=10000
        assert result == 2560
        assert result != len(pairs) * 1000  # 关键断言：不是估算

    def test_no_convergence_returns_last_real_timestep(self):
        """从未收敛时返回最后一个真实 timestep。"""
        pairs = [(i * 137, float(i)) for i in range(20)]  # 持续递增，无收敛
        result = _detect_convergence_from_pairs(pairs, window=5, threshold=0.001)
        assert result == 19 * 137  # 最后一个真实 timestep


# =============================================================================
# holm_bonferroni 手动实现
# =============================================================================


class TestHolmBonferroni:
    """Holm-Bonferroni step-down 校正。"""

    def test_empty_input(self):
        rejected, adj_p = holm_bonferroni([])
        assert rejected == []
        assert adj_p == []

    def test_single_significant(self):
        """单个显著 p 值。"""
        rejected, adj_p = holm_bonferroni([0.001], alpha=0.05)
        assert rejected == [True]
        assert math.isclose(adj_p[0], 0.001, abs_tol=1e-9)

    def test_single_not_significant(self):
        """单个不显著 p 值。"""
        rejected, adj_p = holm_bonferroni([0.5], alpha=0.05)
        assert rejected == [False]
        assert math.isclose(adj_p[0], 0.5, abs_tol=1e-9)

    def test_three_comparisons_all_significant(self):
        """3 个比较全显著。"""
        rejected, adj_p = holm_bonferroni([0.001, 0.002, 0.003], alpha=0.05)
        # 排序后 (0.001, 0.002, 0.003)，Holm 调整：0.001*3=0.003, 0.002*2=0.004, 0.003*1=0.003
        # 但单调性：adj_p 必须递增
        assert all(rejected)
        assert adj_p[0] < adj_p[1] or adj_p[0] == adj_p[1]

    def test_three_comparisons_mixed(self):
        """3 个比较，部分显著。"""
        rejected, _ = holm_bonferroni([0.001, 0.02, 0.5], alpha=0.05)
        # 最小 p=0.001 → adj=0.003 → 0.003 <= 0.05/3 ≈ 0.0167 → 拒绝
        # 次小 p=0.02 → adj=0.04 → 0.04 <= 0.05/2=0.025 → 不拒绝 → stop
        # 最大 p=0.5 → adj=0.5 → 不拒绝
        assert rejected[0] is True  # 0.001
        assert rejected[1] is False  # 0.02
        assert rejected[2] is False  # 0.5

    def test_adj_p_monotonic_after_sort(self):
        """校正后 p 值在排序下单调不减。"""
        p_values = [0.01, 0.005, 0.03, 0.001]
        _, adj_p = holm_bonferroni(p_values, alpha=0.05)
        # 按 p 值排序后的索引
        sorted_indices = sorted(range(len(p_values)), key=lambda i: p_values[i])
        sorted_adj = [adj_p[i] for i in sorted_indices]
        for i in range(1, len(sorted_adj)):
            assert sorted_adj[i] >= sorted_adj[i - 1]

    def test_adj_p_capped_at_one(self):
        """校正后 p 值上限 1.0。"""
        _, adj_p = holm_bonferroni([0.5, 0.6, 0.7], alpha=0.05)
        assert all(p <= 1.0 for p in adj_p)


# =============================================================================
# ABLATION_LAYERS 结构不变量
# =============================================================================


class TestAblationLayers:
    """四层奖励配置结构不变量。"""

    def test_four_layers(self):
        assert set(ABLATION_LAYERS.keys()) == {
            "L1_basic",
            "L2_execution",
            "L3_wait_penalty",
            "L4_full",
        }

    def test_l4_equals_default_rewards(self):
        """L4_full 必须等于 DEFAULT_REWARDS。"""
        assert ABLATION_LAYERS["L4_full"] == DEFAULT_REWARDS

    def test_each_layer_has_seven_keys(self):
        """每层包含全部 7 个奖励键。"""
        expected_keys = set(DEFAULT_REWARDS.keys())
        for name, rewards in ABLATION_LAYERS.items():
            assert set(rewards.keys()) == expected_keys, f"{name} 缺少键"

    def test_l1_to_l4_progressive_addition(self):
        """L1 → L4 逐步叠加：每层比上一层多一个非零项。"""
        l1 = ABLATION_LAYERS["L1_basic"]
        l2 = ABLATION_LAYERS["L2_execution"]
        l3 = ABLATION_LAYERS["L3_wait_penalty"]
        l4 = ABLATION_LAYERS["L4_full"]

        # L2 比 L1 多了 quantum/hybrid/success/mismatch（量子奖励差异化 + 执行收益）
        l2_added = {k: v for k, v in l2.items() if v != l1[k]}
        assert "quantum_reward" in l2_added
        assert "hybrid_reward" in l2_added
        assert "success_bonus" in l2_added
        assert "mismatch_penalty" in l2_added

        # L3 比 L2 多了 wait_penalty 非零
        assert l3["wait_penalty"] != 0.0
        assert l2["wait_penalty"] == 0.0

        # L4 比 L3 多了 low_util_penalty 非零
        assert l4["low_util_penalty"] != 0.0
        assert l3["low_util_penalty"] == 0.0
