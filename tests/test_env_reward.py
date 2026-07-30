"""Issue #401 奖励函数改进的单元测试。

测试 ``src/scheduler/env_reward.py`` 中三个函数的行为：
    - ``_compute_task_speedup``     : 基于任务比特数的对数加速比
    - ``_compute_task_weighting``   : urgency/priority 加权因子
    - ``compute_execution_reward``  : 经典/量子/混合执行的即时奖励

重点验证：
    1. 小比特数任务不获得额外加速比加成（qubit_factor == 1.0）
    2. 大比特数任务获得更高加速比且随比特数单调递增
    3. urgency/priority 加权符合 [0.5,1.0]×[0.7,1.1] 的设计，越界值被钳制
    4. 三种动作（经典/量子/混合）的奖励均应用任务加权
    5. 低保真度（<0.9）触发 0.6 折扣
    6. 串扰惩罚按权重缩减量子奖励
    7. 相比“纯均匀采样”旧方案，新方案的奖励方差显著降低
"""

from __future__ import annotations

import numpy as np
import pytest

from src.scheduler.env_reward import (
    _compute_task_speedup,
    _compute_task_weighting,
    compute_execution_reward,
)
from src.scheduler.env_types import (
    ACTION_CLASSICAL,
    ACTION_HYBRID,
    ACTION_QUANTUM,
    OBS_CROSSTALK_RISK,
    QUANTUM_SPEEDUP_RANGE,
    REWARD_CLASSICAL,
    REWARD_HYBRID,
    REWARD_QUANTUM_BASE,
    REWARD_SUCCESS_BONUS,
    Task,
)


def _make_task(
    task_id: str = "t0",
    qubit_count: int = 10,
    urgency: float = 0.5,
    priority: int = 3,
) -> Task:
    """构造用于测试的 Task 实例（默认字段与 dataclass 默认值一致）。"""
    return Task(
        task_id=task_id,
        task_type="quantum",
        qubit_count=qubit_count,
        urgency=urgency,
        priority=priority,
    )


# ---------------------------------------------------------------------------
# _compute_task_speedup
# ---------------------------------------------------------------------------
class TestComputeTaskSpeedup:
    """``_compute_task_speedup`` 的测试。"""

    def test_small_qubit_count_no_boost(self) -> None:
        """比特数 <= 参考值(20) 时 qubit_factor == 1.0，加速比等于基础采样值。"""
        for qubit_count in (0, 1, 10, 20):
            task = _make_task(qubit_count=qubit_count)
            rng = np.random.default_rng(7)
            speedup = _compute_task_speedup(task, rng)

            # 用相同种子的独立 rng 还原基础采样值（函数内部仅消费一次 uniform）
            rng_base = np.random.default_rng(7)
            base = float(rng_base.uniform(*QUANTUM_SPEEDUP_RANGE))

            assert speedup == pytest.approx(base), f"qubit_count={qubit_count} 时加速比应等于基础值"

    def test_large_qubit_count_higher_speedup(self) -> None:
        """比特数=100 时 qubit_factor>1，加速比高于基础采样值。"""
        task = _make_task(qubit_count=100)
        rng = np.random.default_rng(3)
        speedup = _compute_task_speedup(task, rng)

        rng_base = np.random.default_rng(3)
        base = float(rng_base.uniform(*QUANTUM_SPEEDUP_RANGE))

        # qubit_factor = 1 + log(100/20) = 1 + log(5) ≈ 2.609
        assert speedup > base
        assert speedup == pytest.approx(base * (1.0 + np.log(5.0)))

    def test_speedup_monotonic_with_qubits(self) -> None:
        """相同随机种子下，加速比随比特数非递减，且大比特数严格更高。"""
        qubit_counts = [10, 20, 50, 100, 200]
        speedups: list[float] = []
        for qc in qubit_counts:
            # 每次使用相同种子，保证基础采样值相同，差异仅来自 qubit_factor
            rng = np.random.default_rng(2024)
            speedups.append(_compute_task_speedup(_make_task(qubit_count=qc), rng))

        # 非递减（<=20 时 qubit_factor 恒为 1.0，相等；之后随比特数单调上升）
        assert all(speedups[i] <= speedups[i + 1] for i in range(len(speedups) - 1))
        # 最大比特数的加速比严格大于最小比特数
        assert speedups[-1] > speedups[0]

    def test_speedup_within_reasonable_range(self) -> None:
        """加速比始终为正，且不低于基础加速比下界。"""
        for qubit_count in (0, 1, 10, 20, 100, 1000):
            rng = np.random.default_rng(99)
            speedup = _compute_task_speedup(_make_task(qubit_count=qubit_count), rng)
            assert speedup > 0
            # base >= QUANTUM_SPEEDUP_RANGE[0]=2.0 且 qubit_factor >= 1.0
            assert speedup >= QUANTUM_SPEEDUP_RANGE[0]


# ---------------------------------------------------------------------------
# _compute_task_weighting
# ---------------------------------------------------------------------------
class TestComputeTaskWeighting:
    """``_compute_task_weighting`` 的测试。"""

    def test_default_task_weight(self) -> None:
        """默认任务(urgency=0.5, priority=3)的加权因子。

        实际公式：
            urgency_factor  = 0.5 + 0.5 * 0.5 = 0.75
            priority_factor = 0.6 + 0.1 * 3   = 0.9
            weight          = 0.75 * 0.9      = 0.675
        """
        task = _make_task(urgency=0.5, priority=3)
        weight = _compute_task_weighting(task)

        expected = (0.5 + 0.5 * 0.5) * (0.6 + 0.1 * 3)
        assert weight == pytest.approx(expected)
        assert weight == pytest.approx(0.675)

    def test_high_urgency_higher_weight(self) -> None:
        """高紧急高优先级任务的权重大于默认任务。"""
        default_weight = _compute_task_weighting(_make_task(urgency=0.5, priority=3))
        high_weight = _compute_task_weighting(_make_task(urgency=1.0, priority=5))

        # urgency=1.0 → 1.0, priority=5 → 1.1, weight = 1.0 * 1.1 = 1.1
        assert high_weight == pytest.approx(1.0 * 1.1)
        assert high_weight > default_weight

    def test_low_urgency_lower_weight(self) -> None:
        """低紧急低优先级任务的权重小于默认任务。"""
        default_weight = _compute_task_weighting(_make_task(urgency=0.5, priority=3))
        low_weight = _compute_task_weighting(_make_task(urgency=0.0, priority=1))

        # urgency=0.0 → 0.5, priority=1 → 0.7, weight = 0.5 * 0.7 = 0.35
        assert low_weight == pytest.approx(0.5 * 0.7)
        assert low_weight < default_weight

    def test_weight_clamps_out_of_range(self) -> None:
        """urgency/priority 越界时被钳制到合法区间。"""
        # urgency=-1 钳制到 0，priority=10 钳制到 5
        clamped = _compute_task_weighting(_make_task(urgency=-1.0, priority=10))
        # 等价于 urgency=0.0, priority=5
        equivalent = _compute_task_weighting(_make_task(urgency=0.0, priority=5))

        assert clamped == pytest.approx(equivalent)
        assert clamped == pytest.approx(0.5 * 1.1)  # 0.55
        # 钳制后仍在设计区间 [0.35, 1.1] 内
        assert 0.35 <= clamped <= 1.1


# ---------------------------------------------------------------------------
# compute_execution_reward
# ---------------------------------------------------------------------------
class TestComputeExecutionReward:
    """``compute_execution_reward`` 的测试。"""

    def test_classical_reward_has_task_weight(self) -> None:
        """经典执行奖励 = (REWARD_CLASSICAL + REWARD_SUCCESS_BONUS) * weight。"""
        rng = np.random.default_rng(0)
        task_default = _make_task(urgency=0.5, priority=3)  # weight 0.675
        task_high = _make_task(urgency=1.0, priority=5)  # weight 1.1

        r_default = compute_execution_reward(task_default, ACTION_CLASSICAL, rng, 0.95, 1.0)
        r_high = compute_execution_reward(task_high, ACTION_CLASSICAL, rng, 0.95, 1.0)

        assert r_default == pytest.approx((REWARD_CLASSICAL + REWARD_SUCCESS_BONUS) * 0.675)
        assert r_high == pytest.approx((REWARD_CLASSICAL + REWARD_SUCCESS_BONUS) * 1.1)
        assert r_high > r_default

    def test_quantum_reward_uses_qubit_count(self) -> None:
        """相同种子下，大比特数任务获得更高的量子奖励。"""
        task_small = _make_task(qubit_count=10)  # qubit_factor=1.0
        task_large = _make_task(qubit_count=100)  # qubit_factor≈2.609

        rng_s = np.random.default_rng(11)
        r_small = compute_execution_reward(task_small, ACTION_QUANTUM, rng_s, 0.95, 1.0)
        rng_l = np.random.default_rng(11)
        r_large = compute_execution_reward(task_large, ACTION_QUANTUM, rng_l, 0.95, 1.0)

        assert r_large > r_small

    def test_quantum_reward_applies_urgency(self) -> None:
        """相同种子、相同比特数下，高紧急任务获得更高量子奖励。"""
        task_low = _make_task(urgency=0.2, priority=3)
        task_high = _make_task(urgency=0.9, priority=3)

        rng_lo = np.random.default_rng(21)
        r_low = compute_execution_reward(task_low, ACTION_QUANTUM, rng_lo, 0.95, 1.0)
        rng_hi = np.random.default_rng(21)
        r_high = compute_execution_reward(task_high, ACTION_QUANTUM, rng_hi, 0.95, 1.0)

        assert r_high > r_low

    def test_hybrid_reward_applies_weight(self) -> None:
        """混合执行奖励随任务加权变化。"""
        rng = np.random.default_rng(0)
        task_default = _make_task(urgency=0.5, priority=3)  # weight 0.675
        task_high = _make_task(urgency=1.0, priority=5)  # weight 1.1

        # available_ratio=1.0 → hybrid_factor = 0.5 + 0.5*1.0 = 1.0
        r_default = compute_execution_reward(task_default, ACTION_HYBRID, rng, 0.95, 1.0)
        r_high = compute_execution_reward(task_high, ACTION_HYBRID, rng, 0.95, 1.0)

        assert r_default == pytest.approx(REWARD_HYBRID * 1.0 * 0.675 + REWARD_SUCCESS_BONUS)
        assert r_high == pytest.approx(REWARD_HYBRID * 1.0 * 1.1 + REWARD_SUCCESS_BONUS)
        assert r_high > r_default

    def test_low_fidelity_discount(self) -> None:
        """保真度 < 0.9 时，量子奖励被额外乘 0.6 折扣。"""
        task = _make_task(qubit_count=10)  # qubit_factor=1.0
        weight = _compute_task_weighting(task)

        rng_actual = np.random.default_rng(31)
        actual = compute_execution_reward(task, ACTION_QUANTUM, rng_actual, 0.85, 1.0)

        # 用相同种子的独立 rng 还原基础加速比（量子分支仅消费一次 uniform）
        rng_base = np.random.default_rng(31)
        base = float(rng_base.uniform(*QUANTUM_SPEEDUP_RANGE))
        fidelity_factor = 0.85 / 0.99

        expected_discounted = (
            REWARD_QUANTUM_BASE * base * fidelity_factor * 0.6
        ) * weight + REWARD_SUCCESS_BONUS
        expected_no_discount = (
            REWARD_QUANTUM_BASE * base * fidelity_factor
        ) * weight + REWARD_SUCCESS_BONUS

        # 精确验证 0.6 折扣被应用
        assert actual == pytest.approx(expected_discounted)
        # 折扣使奖励降低
        assert actual < expected_no_discount

        # 对照：保真度 >= 0.9 不打折
        rng_hi = np.random.default_rng(31)
        r_hi = compute_execution_reward(task, ACTION_QUANTUM, rng_hi, 0.95, 1.0)
        expected_hi = (REWARD_QUANTUM_BASE * base * (0.95 / 0.99)) * weight + REWARD_SUCCESS_BONUS
        assert r_hi == pytest.approx(expected_hi)
        assert r_hi > actual

    def test_crosstalk_penalty_reduces_reward(self) -> None:
        """正的串扰惩罚按权重缩减量子奖励。"""
        task = _make_task(qubit_count=10)
        weight = _compute_task_weighting(task)

        rng_a = np.random.default_rng(41)
        r_no_penalty = compute_execution_reward(task, ACTION_QUANTUM, rng_a, 0.95, 1.0, 0.0)
        rng_b = np.random.default_rng(41)
        r_penalty = compute_execution_reward(task, ACTION_QUANTUM, rng_b, 0.95, 1.0, 5.0)

        assert r_penalty < r_no_penalty
        # 惩罚在加权前扣除，因此奖励差值 = crosstalk_penalty * weight
        assert (r_no_penalty - r_penalty) == pytest.approx(5.0 * weight)

    def test_reward_variance_reduced(self) -> None:
        """相比“纯均匀采样”旧方案，新方案奖励方差显著降低。

        旧方案（纯均匀）：reward = REWARD_QUANTUM_BASE * U(2,5) + bonus，
        不应用任务加权，奖励在 [23, 53] 上均匀分布。
        新方案（Issue #401）：reward = REWARD_QUANTUM_BASE * U(2,5) * weight + bonus，
        确定的任务加权把奖励范围压缩到 weight 倍，方差按 weight^2 缩减。
        对默认任务(weight≈0.675)，方差缩减约 1 - 0.675^2 ≈ 54%。
        """
        task = _make_task(qubit_count=10, urgency=0.5, priority=3)  # weight 0.675
        n_samples = 100

        # 新方案：调用真实函数（保真度 0.99 不触发折扣，比特数 10 → qubit_factor=1.0）
        rng_new = np.random.default_rng(42)
        new_rewards = np.array(
            [
                compute_execution_reward(task, ACTION_QUANTUM, rng_new, 0.99, 1.0)
                for _ in range(n_samples)
            ]
        )

        # 旧方案（纯均匀）：相同随机种子下的同一批均匀采样，但不应用任务加权
        rng_old = np.random.default_rng(42)
        old_rewards = (
            rng_old.uniform(*QUANTUM_SPEEDUP_RANGE, size=n_samples) * REWARD_QUANTUM_BASE
            + REWARD_SUCCESS_BONUS
        )

        var_new = float(np.var(new_rewards))
        var_old = float(np.var(old_rewards))

        assert var_new < var_old
        # 方差缩减应显著（理论约 54%，留出采样余量后至少缩减 20%）
        assert var_new < 0.8 * var_old


# ---------------------------------------------------------------------------
# _compute_execution_reward crosstalk_risk 参数化（Issue #746）
# ---------------------------------------------------------------------------
class TestComputeExecutionRewardCrosstalkParam:
    """``_compute_execution_reward`` 新增 ``crosstalk_risk`` 形参的行为测试。

    核心目标：传入步首已算出的 ``crosstalk_risk`` 时，奖励计算不再重复调用
    ``_get_observation()`` 重建观测，且奖励语义与回退路径逐位一致。
    """

    def test_crosstalk_risk_param_skips_get_observation(self) -> None:
        """传入 crosstalk_risk 时不应调用 _get_observation() 重建观测。"""
        from unittest.mock import patch

        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=5, seed=42)
        env.reset(seed=42)
        task = _make_task(qubit_count=10)
        rng = np.random.default_rng(42)

        call_count = 0
        original = env._get_observation

        def spy() -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return original()

        with patch.object(env, "_get_observation", spy):
            # 传入 crosstalk_risk，不应触发 _get_observation
            env._compute_execution_reward(task, ACTION_QUANTUM, rng, crosstalk_risk=0.3)

        assert call_count == 0, "传入 crosstalk_risk 时不应调用 _get_observation()"

    def test_crosstalk_risk_param_consistent_with_fallback(self) -> None:
        """传入 crosstalk_risk 的奖励值应与回退路径（从 _get_observation 读取）一致。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=5, seed=42)
        env.reset(seed=42)
        task = _make_task(qubit_count=10)

        # 从环境观测中读取真实的 crosstalk_risk
        obs = env._get_observation()
        crosstalk_risk = float(obs[OBS_CROSSTALK_RISK])

        # 路径1：传入 crosstalk_risk 参数
        rng_a = np.random.default_rng(99)
        reward_param = env._compute_execution_reward(
            task, ACTION_QUANTUM, rng_a, crosstalk_risk=crosstalk_risk
        )

        # 路径2：不传 crosstalk_risk，回退到 _get_observation()
        rng_b = np.random.default_rng(99)
        reward_fallback = env._compute_execution_reward(task, ACTION_QUANTUM, rng_b)

        assert reward_param == pytest.approx(reward_fallback)

    def test_step_reward_unchanged_with_param(self) -> None:
        """step() 在兼容分配分支传入 crosstalk_risk 后，奖励应保持稳定。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=5, seed=42)
        env.reset(seed=42)

        # 连续执行几步，验证不抛异常且奖励为有限值
        for _ in range(3):
            _obs, reward, terminated, truncated, _ = env.step(1)
            assert np.isfinite(reward), "奖励应为有限值"
            if terminated or truncated:
                env.reset()
