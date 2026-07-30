"""噪声感知奖励整形测试 — Issue #577。

验证 src/scheduler/env_real_machine.py 中的噪声感知奖励状态机：
    - trigger_noise_aware_feedback : 根据保真度触发惩罚/加成
    - advance_noise_aware_to_next_step : 推进衰减状态机
    - get_noise_aware_adjustment : 获取当前步调整值

以及 src/scheduler/env_reward.py 中的 compute_fairness_penalty 边界条件。

阈值与参数来自 src/scheduler/env_types.py：
    NOISE_AWARE_PENALTY_THRESHOLD = 0.9   (低于此值施加惩罚)
    NOISE_AWARE_BONUS_THRESHOLD   = 0.95  (高于此值施加加成)
    NOISE_AWARE_PENALTY_STEPS     = 5     (惩罚持续步数)
    NOISE_AWARE_DECAY_FACTOR      = 0.7   (每步衰减因子)
    NOISE_AWARD_PENALTY_STRENGTH  = 2.0
    NOISE_AWARD_BONUS_STRENGTH    = 0.5
"""

from types import SimpleNamespace

import pytest

from src.scheduler.env_real_machine import (
    advance_noise_aware_to_next_step,
    get_noise_aware_adjustment,
    init_noise_aware_state,
    trigger_noise_aware_feedback,
)
from src.scheduler.env_reward import compute_fairness_penalty
from src.scheduler.env_types import (
    ACTION_QUANTUM,
    NOISE_AWARD_BONUS_STRENGTH,
    NOISE_AWARD_PENALTY_STRENGTH,
    NOISE_AWARE_BONUS_THRESHOLD,
    NOISE_AWARE_DECAY_FACTOR,
    NOISE_AWARE_PENALTY_STEPS,
    NOISE_AWARE_PENALTY_THRESHOLD,
    RealMachineConfig,
)


def _make_env(config: RealMachineConfig | None = None) -> SimpleNamespace:
    """构造一个最小化的 env 对象用于噪声感知奖励测试。

    真实环境为 QuantumSchedulingEnv，但噪声感知函数仅访问以下属性：
        real_machine_config          : RealMachineConfig 配置
        _current_step                : 当前步数
        _noise_aware_*               : 噪声感知状态机字段
    """
    env = SimpleNamespace()
    env.real_machine_config = config if config is not None else RealMachineConfig()
    env._current_step = 0
    init_noise_aware_state(env)
    return env


def test_noise_aware_penalty_low_fidelity():
    """fidelity < NOISE_AWARE_PENALTY_THRESHOLD(0.9) 时应触发负向惩罚。"""
    env = _make_env()
    fidelity = 0.80  # 低于惩罚阈值 0.9
    trigger_noise_aware_feedback(env, fidelity)

    # pending 已设置且为负
    assert env._noise_aware_has_pending is True
    expected = -NOISE_AWARD_PENALTY_STRENGTH * (NOISE_AWARE_PENALTY_THRESHOLD - fidelity)
    assert env._noise_aware_pending_value == pytest.approx(expected)
    assert env._noise_aware_pending_value < 0.0

    # 推进一步激活 pending，对量子动作应返回负向调整
    advance_noise_aware_to_next_step(env)
    adj = get_noise_aware_adjustment(env, ACTION_QUANTUM)
    assert adj < 0.0
    assert adj == pytest.approx(expected)


def test_noise_aware_bonus_high_fidelity():
    """fidelity > NOISE_AWARE_BONUS_THRESHOLD(0.95) 时应触发正向加成。"""
    env = _make_env()
    fidelity = 0.99  # 高于加成阈值 0.95
    trigger_noise_aware_feedback(env, fidelity)

    assert env._noise_aware_has_pending is True
    expected = NOISE_AWARD_BONUS_STRENGTH * (fidelity - NOISE_AWARE_BONUS_THRESHOLD)
    assert env._noise_aware_pending_value == pytest.approx(expected)
    assert env._noise_aware_pending_value > 0.0

    advance_noise_aware_to_next_step(env)
    adj = get_noise_aware_adjustment(env, ACTION_QUANTUM)
    assert adj > 0.0
    assert adj == pytest.approx(expected)


def test_noise_aware_neutral():
    """fidelity 在两个阈值之间时不触发任何调整（应为 0）。"""
    env = _make_env()
    # 0.9 <= fidelity <= 0.95 为中性区间
    fidelity = 0.92
    trigger_noise_aware_feedback(env, fidelity)

    # 未触发 pending
    assert env._noise_aware_has_pending is False
    assert env._noise_aware_pending_value == 0.0

    advance_noise_aware_to_next_step(env)
    adj = get_noise_aware_adjustment(env, ACTION_QUANTUM)
    assert adj == 0.0


def test_penalty_decay():
    """测试 NOISE_AWARE_PENALTY_STEPS 和 NOISE_AWARE_DECAY_FACTOR 的衰减效果。

    时序（steps=5, decay=0.7，触发值 -X）：
        第1步: 完整强度 -X
        第2步: -X * 0.7
        第3步: -X * 0.7^2
        第4步: -X * 0.7^3
        第5步: -X * 0.7^4
        第6步: 0（衰减结束归零）
    """
    config = RealMachineConfig()  # 默认 steps=5, decay=0.7
    env = _make_env(config)
    fidelity = 0.85
    trigger_noise_aware_feedback(env, fidelity)
    expected_initial = -NOISE_AWARD_PENALTY_STRENGTH * (NOISE_AWARE_PENALTY_THRESHOLD - fidelity)

    values: list[float] = []
    for _ in range(NOISE_AWARE_PENALTY_STEPS):
        advance_noise_aware_to_next_step(env)
        values.append(get_noise_aware_adjustment(env, ACTION_QUANTUM))

    # 第1步：完整强度
    assert values[0] == pytest.approx(expected_initial)
    # 后续每步乘以 DECAY_FACTOR
    for i in range(1, NOISE_AWARE_PENALTY_STEPS):
        assert values[i] == pytest.approx(values[i - 1] * NOISE_AWARE_DECAY_FACTOR)

    # 持续步数耗尽后应归零
    advance_noise_aware_to_next_step(env)
    assert get_noise_aware_adjustment(env, ACTION_QUANTUM) == 0.0


def test_compute_fairness_penalty_none_tenant():
    """tenant_id=None 时公平性惩罚应返回 0.0。"""
    # 即使有足量租户数据，tenant_id=None 也应返回 0
    assert compute_fairness_penalty(None, {"t1": 1.0, "t2": 2.0}) == 0.0
    # 无数据时也应返回 0
    assert compute_fairness_penalty(None, None) == 0.0
    assert compute_fairness_penalty(None, {}) == 0.0
