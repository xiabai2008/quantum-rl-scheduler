"""
量子RL调度系统 - 熔断器模块单元测试
Unit Tests for src/api/circuit_breaker.py

测试覆盖：
- CircuitState 枚举（CLOSED / OPEN / HALF_OPEN 三态语义）
- CircuitBreaker 初始化（默认参数与自定义参数）
- CLOSED 状态：成功清零计数、失败累加计数、达阈值转 OPEN、alert_critical 触发
- OPEN 状态：拒绝调用抛出 CircuitOpenError、超过 recovery_timeout 转 HALF_OPEN
- HALF_OPEN 状态：试探成功回 CLOSED、试探失败回 OPEN
- is_available() 在三态下的返回值（含 OPEN 超时与未超时分支）
- reset() 从任意状态回到 CLOSED
- call() 参数透传
- 时间通过 unittest.mock 控制 time.monotonic，告警通过 mock 屏蔽
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api.circuit_breaker import CircuitBreaker, CircuitState
from src.exceptions import CircuitOpenError


class TestCircuitStateEnum(unittest.TestCase):
    """测试 CircuitState 枚举的三态语义。"""

    def test_closed_state_value(self):
        """CLOSED 状态的字符串值应为 'closed'。"""
        self.assertEqual(CircuitState.CLOSED.value, "closed")

    def test_open_state_value(self):
        """OPEN 状态的字符串值应为 'open'。"""
        self.assertEqual(CircuitState.OPEN.value, "open")

    def test_half_open_state_value(self):
        """HALF_OPEN 状态的字符串值应为 'half_open'。"""
        self.assertEqual(CircuitState.HALF_OPEN.value, "half_open")

    def test_three_distinct_states(self):
        """三态应互不相等。"""
        states = {CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN}
        self.assertEqual(len(states), 3)


class TestCircuitBreakerInit(unittest.TestCase):
    """测试 CircuitBreaker 初始化。"""

    def test_default_init_values(self):
        """默认初始化应使用 failure_threshold=5 与 recovery_timeout=60.0。"""
        cb = CircuitBreaker()
        self.assertEqual(cb.failure_threshold, 5)
        self.assertEqual(cb.recovery_timeout, 60.0)
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertEqual(cb.failure_count, 0)
        self.assertEqual(cb.last_failure_time, 0.0)

    def test_custom_init_values(self):
        """自定义参数应被正确存储。"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        self.assertEqual(cb.failure_threshold, 3)
        self.assertEqual(cb.recovery_timeout, 30.0)
        self.assertEqual(cb.state, CircuitState.CLOSED)


class TestCircuitBreakerClosedState(unittest.TestCase):
    """测试 CLOSED 状态下的行为。"""

    @patch("src.api.circuit_breaker.alert_critical")
    def test_successful_call_returns_result(self, _mock_alert):
        """CLOSED 状态下成功调用应返回函数结果。"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        result = cb.call(lambda x: x + 1, 41)
        self.assertEqual(result, 42)

    @patch("src.api.circuit_breaker.alert_critical")
    def test_successful_call_resets_failure_count(self, _mock_alert):
        """CLOSED 状态下成功调用应清零 failure_count。"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        # 先制造一次失败，使 failure_count > 0
        with self.assertRaises(ValueError):
            cb.call(_raise_value_error)
        self.assertEqual(cb.failure_count, 1)
        # 成功调用后应清零
        cb.call(lambda: "ok")
        self.assertEqual(cb.failure_count, 0)

    @patch("src.api.circuit_breaker.alert_critical")
    def test_failed_call_increments_failure_count(self, _mock_alert):
        """CLOSED 状态下失败调用应累加 failure_count。"""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
        with self.assertRaises(ValueError):
            cb.call(_raise_value_error)
        self.assertEqual(cb.failure_count, 1)
        with self.assertRaises(ValueError):
            cb.call(_raise_value_error)
        self.assertEqual(cb.failure_count, 2)

    @patch("src.api.circuit_breaker.alert_critical")
    def test_failed_call_raises_original_exception(self, _mock_alert):
        """CLOSED 状态下失败调用应抛出原始异常（而非包装异常）。"""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
        with self.assertRaises(RuntimeError) as ctx:
            cb.call(_raise_runtime_error)
        self.assertIn("boom", str(ctx.exception))

    @patch("src.api.circuit_breaker.alert_critical")
    def test_not_open_before_threshold(self, _mock_alert):
        """未达到 failure_threshold 时应保持 CLOSED。"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        for _ in range(2):
            with self.assertRaises(ValueError):
                cb.call(_raise_value_error)
        # 仅 2 次失败，未达阈值 3，仍应处于 CLOSED
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertEqual(cb.failure_count, 2)

    @patch("src.api.circuit_breaker.alert_critical")
    def test_closed_to_open_after_threshold(self, _mock_alert):
        """连续失败达到 failure_threshold 时应转为 OPEN。"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        for _ in range(3):
            with self.assertRaises(ValueError):
                cb.call(_raise_value_error)
        self.assertEqual(cb.state, CircuitState.OPEN)
        self.assertEqual(cb.failure_count, 3)

    @patch("src.api.circuit_breaker.alert_critical")
    def test_alert_critical_called_on_closed_to_open(self, mock_alert):
        """CLOSED→OPEN 转换时 alert_critical 应被调用一次。"""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
        # 第一次失败：未达阈值，不告警
        with self.assertRaises(ValueError):
            cb.call(_raise_value_error)
        mock_alert.assert_not_called()
        # 第二次失败：达到阈值，触发告警
        with self.assertRaises(ValueError):
            cb.call(_raise_value_error)
        mock_alert.assert_called_once()
        # 第一个参数应为类别 "circuit_breaker"
        args, _ = mock_alert.call_args
        self.assertEqual(args[0], "circuit_breaker")
        # 第二个参数为消息，应包含失败计数信息
        self.assertIn("CLOSED→OPEN", args[1])

    @patch("src.api.circuit_breaker.alert_critical")
    def test_partial_failures_then_success_resets_count(self, _mock_alert):
        """部分失败后一次成功应清零 failure_count，避免累积误触发。"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        # 两次失败
        with self.assertRaises(ValueError):
            cb.call(_raise_value_error)
        with self.assertRaises(ValueError):
            cb.call(_raise_value_error)
        self.assertEqual(cb.failure_count, 2)
        # 一次成功，清零计数
        cb.call(lambda: "ok")
        self.assertEqual(cb.failure_count, 0)
        # 再两次失败，仍未达阈值
        with self.assertRaises(ValueError):
            cb.call(_raise_value_error)
        with self.assertRaises(ValueError):
            cb.call(_raise_value_error)
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertEqual(cb.failure_count, 2)


class TestCircuitBreakerOpenState(unittest.TestCase):
    """测试 OPEN 状态下的行为，使用 mock 控制 time.monotonic。"""

    @patch("src.api.circuit_breaker.alert_critical")
    def test_open_state_rejects_call(self, _mock_alert):
        """OPEN 状态下应直接拒绝调用并抛出 CircuitOpenError。"""
        with patch("src.api.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 0.0
            cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
            # 触发熔断
            for _ in range(2):
                with self.assertRaises(ValueError):
                    cb.call(_raise_value_error)
            self.assertEqual(cb.state, CircuitState.OPEN)
            # 此时 time.monotonic() 仍为 0，未过恢复超时
            with self.assertRaises(CircuitOpenError):
                cb.call(lambda: "should be rejected")

    @patch("src.api.circuit_breaker.alert_critical")
    def test_circuit_open_error_has_correct_code(self, _mock_alert):
        """CircuitOpenError 应携带 code='CIRCUIT_OPEN' 与 retryable=True。"""
        with patch("src.api.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 0.0
            cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
            with self.assertRaises(ValueError):
                cb.call(_raise_value_error)
            self.assertEqual(cb.state, CircuitState.OPEN)
            with self.assertRaises(CircuitOpenError) as ctx:
                cb.call(lambda: "rejected")
            self.assertEqual(ctx.exception.code, "CIRCUIT_OPEN")
            self.assertTrue(ctx.exception.retryable)

    @patch("src.api.circuit_breaker.alert_critical")
    def test_open_state_no_recovery_within_timeout(self, _mock_alert):
        """OPEN 状态未超时时应保持 OPEN，调用被拒绝。"""
        with patch("src.api.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 0.0
            cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
            with self.assertRaises(ValueError):
                cb.call(_raise_value_error)
            self.assertEqual(cb.state, CircuitState.OPEN)
            # 时间仅前进了 30 秒，未达 60 秒
            mock_time.return_value = 30.0
            with self.assertRaises(CircuitOpenError):
                cb.call(lambda: "rejected")
            # 状态应仍为 OPEN（未转 HALF_OPEN）
            self.assertEqual(cb.state, CircuitState.OPEN)

    @patch("src.api.circuit_breaker.alert_critical")
    def test_open_to_half_open_after_recovery_timeout(self, _mock_alert):
        """OPEN 状态超过 recovery_timeout 后，调用应转入 HALF_OPEN 并放行试探。"""
        with patch("src.api.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 0.0
            cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
            with self.assertRaises(ValueError):
                cb.call(_raise_value_error)
            self.assertEqual(cb.state, CircuitState.OPEN)
            # 时间前进 100 秒，超过 60 秒恢复超时
            mock_time.return_value = 100.0
            # 调用应转入 HALF_OPEN 并放行（试探成功）
            result = cb.call(lambda: "recovered")
            self.assertEqual(result, "recovered")
            # 试探成功后应回到 CLOSED
            self.assertEqual(cb.state, CircuitState.CLOSED)


class TestCircuitBreakerHalfOpenState(unittest.TestCase):
    """测试 HALF_OPEN 状态下的试探行为。"""

    @patch("src.api.circuit_breaker.alert_critical")
    def test_half_open_success_transitions_to_closed(self, _mock_alert):
        """HALF_OPEN 状态下试探成功应重置为 CLOSED。"""
        with patch("src.api.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 0.0
            cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
            # 触发熔断
            with self.assertRaises(ValueError):
                cb.call(_raise_value_error)
            self.assertEqual(cb.state, CircuitState.OPEN)
            # 超过恢复超时，进入 HALF_OPEN
            mock_time.return_value = 100.0
            # 手动设置 HALF_OPEN 以隔离测试
            cb.state = CircuitState.HALF_OPEN
            # 试探成功 → 回到 CLOSED
            cb.call(lambda: "ok")
            self.assertEqual(cb.state, CircuitState.CLOSED)
            self.assertEqual(cb.failure_count, 0)

    @patch("src.api.circuit_breaker.alert_critical")
    def test_half_open_failure_transitions_to_open(self, _mock_alert):
        """HALF_OPEN 状态下试探失败应回到 OPEN。"""
        with patch("src.api.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 0.0
            cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
            # 触发熔断
            with self.assertRaises(ValueError):
                cb.call(_raise_value_error)
            self.assertEqual(cb.state, CircuitState.OPEN)
            # 超过恢复超时，进入 HALF_OPEN
            mock_time.return_value = 100.0
            # 手动设置 HALF_OPEN 以隔离测试
            cb.state = CircuitState.HALF_OPEN
            # 试探失败 → 回到 OPEN
            with self.assertRaises(ValueError):
                cb.call(_raise_value_error)
            self.assertEqual(cb.state, CircuitState.OPEN)

    @patch("src.api.circuit_breaker.alert_critical")
    def test_half_open_success_resets_last_failure_time(self, _mock_alert):
        """HALF_OPEN 试探成功调用 reset() 应将 last_failure_time 清零。"""
        with patch("src.api.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 0.0
            cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
            with self.assertRaises(ValueError):
                cb.call(_raise_value_error)
            # last_failure_time 应已被设置为 0.0（mock 时间）
            self.assertEqual(cb.last_failure_time, 0.0)
            mock_time.return_value = 100.0
            cb.state = CircuitState.HALF_OPEN
            cb.call(lambda: "ok")
            # reset() 应将 last_failure_time 清零
            self.assertEqual(cb.last_failure_time, 0.0)


class TestIsAvailable(unittest.TestCase):
    """测试 is_available() 在三态下的判定逻辑。"""

    @patch("src.api.circuit_breaker.alert_critical")
    def test_is_available_closed_returns_true(self, _mock_alert):
        """CLOSED 状态 is_available() 应返回 True。"""
        cb = CircuitBreaker()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertTrue(cb.is_available())

    @patch("src.api.circuit_breaker.alert_critical")
    def test_is_available_half_open_returns_true(self, _mock_alert):
        """HALF_OPEN 状态 is_available() 应返回 True（放行试探）。"""
        cb = CircuitBreaker()
        cb.state = CircuitState.HALF_OPEN
        self.assertTrue(cb.is_available())

    @patch("src.api.circuit_breaker.alert_critical")
    def test_is_available_open_not_expired_returns_false(self, _mock_alert):
        """OPEN 状态未超时 is_available() 应返回 False。"""
        with patch("src.api.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 100.0
            cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
            cb.state = CircuitState.OPEN
            cb.last_failure_time = 50.0  # 100 - 50 = 50 < 60，未超时
            self.assertFalse(cb.is_available())

    @patch("src.api.circuit_breaker.alert_critical")
    def test_is_available_open_expired_returns_true(self, _mock_alert):
        """OPEN 状态已超时 is_available() 应返回 True。"""
        with patch("src.api.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 200.0
            cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
            cb.state = CircuitState.OPEN
            cb.last_failure_time = 100.0  # 200 - 100 = 100 >= 60，已超时
            self.assertTrue(cb.is_available())

    @patch("src.api.circuit_breaker.alert_critical")
    def test_is_available_open_boundary_equal_timeout(self, _mock_alert):
        """OPEN 状态恰好到达恢复超时（差值等于 recovery_timeout）应返回 True。"""
        with patch("src.api.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 160.0
            cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
            cb.state = CircuitState.OPEN
            cb.last_failure_time = 100.0  # 160 - 100 = 60 == 60，边界值
            self.assertTrue(cb.is_available())


class TestReset(unittest.TestCase):
    """测试 reset() 方法从任意状态回到 CLOSED。"""

    @patch("src.api.circuit_breaker.alert_critical")
    def test_reset_from_closed(self, _mock_alert):
        """CLOSED 状态调用 reset 应保持 CLOSED 并清零计数。"""
        cb = CircuitBreaker()
        cb.failure_count = 2
        cb.reset()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertEqual(cb.failure_count, 0)
        self.assertEqual(cb.last_failure_time, 0.0)

    @patch("src.api.circuit_breaker.alert_critical")
    def test_reset_from_open(self, _mock_alert):
        """OPEN 状态调用 reset 应回到 CLOSED。"""
        with patch("src.api.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 0.0
            cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
            with self.assertRaises(ValueError):
                cb.call(_raise_value_error)
            self.assertEqual(cb.state, CircuitState.OPEN)
            cb.reset()
            self.assertEqual(cb.state, CircuitState.CLOSED)
            self.assertEqual(cb.failure_count, 0)
            self.assertEqual(cb.last_failure_time, 0.0)

    @patch("src.api.circuit_breaker.alert_critical")
    def test_reset_from_half_open(self, _mock_alert):
        """HALF_OPEN 状态调用 reset 应回到 CLOSED。"""
        cb = CircuitBreaker()
        cb.state = CircuitState.HALF_OPEN
        cb.failure_count = 1
        cb.last_failure_time = 99.0
        cb.reset()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertEqual(cb.failure_count, 0)
        self.assertEqual(cb.last_failure_time, 0.0)

    @patch("src.api.circuit_breaker.alert_critical")
    def test_reset_allows_call_again(self, _mock_alert):
        """reset 后应能立即放行调用（验证实际可用性）。"""
        with patch("src.api.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 0.0
            cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
            with self.assertRaises(ValueError):
                cb.call(_raise_value_error)
            self.assertEqual(cb.state, CircuitState.OPEN)
            # reset 后立即可调用，无需等待 recovery_timeout
            cb.reset()
            result = cb.call(lambda: "ok")
            self.assertEqual(result, "ok")
            self.assertEqual(cb.state, CircuitState.CLOSED)


class TestCallArgumentPassThrough(unittest.TestCase):
    """测试 call() 对位置参数与关键字参数的透传。"""

    @patch("src.api.circuit_breaker.alert_critical")
    def test_call_passes_positional_and_keyword_args(self, _mock_alert):
        """call 应将 *args 与 **kwargs 透传给被包裹函数。"""
        cb = CircuitBreaker()
        mock_func = MagicMock(return_value="done")
        result = cb.call(mock_func, 1, 2, key="value")
        self.assertEqual(result, "done")
        mock_func.assert_called_once_with(1, 2, key="value")

    @patch("src.api.circuit_breaker.alert_critical")
    def test_call_with_no_args(self, _mock_alert):
        """call 应支持无参数函数调用。"""
        cb = CircuitBreaker()
        mock_func = MagicMock(return_value=42)
        result = cb.call(mock_func)
        self.assertEqual(result, 42)
        mock_func.assert_called_once_with()


class TestFullStateTransitionCycle(unittest.TestCase):
    """测试完整的状态机循环：CLOSED → OPEN → HALF_OPEN → CLOSED/OPEN。"""

    @patch("src.api.circuit_breaker.alert_critical")
    def test_full_cycle_recover_via_half_open(self, _mock_alert):
        """完整恢复路径：CLOSED → OPEN → HALF_OPEN → CLOSED。"""
        with patch("src.api.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 0.0
            cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
            # CLOSED：连续失败触发熔断
            with self.assertRaises(ValueError):
                cb.call(_raise_value_error)
            with self.assertRaises(ValueError):
                cb.call(_raise_value_error)
            self.assertEqual(cb.state, CircuitState.OPEN)
            # OPEN：未超时拒绝调用
            with self.assertRaises(CircuitOpenError):
                cb.call(lambda: "rejected")
            # 时间前进 100 秒，超过 60 秒恢复超时
            mock_time.return_value = 100.0
            # 应转入 HALF_OPEN 并放行试探（成功 → CLOSED）
            result = cb.call(lambda: "recovered")
            self.assertEqual(result, "recovered")
            self.assertEqual(cb.state, CircuitState.CLOSED)
            self.assertEqual(cb.failure_count, 0)

    @patch("src.api.circuit_breaker.alert_critical")
    def test_full_cycle_reopen_via_half_open_failure(self, _mock_alert):
        """完整重熔路径：CLOSED → OPEN → HALF_OPEN → OPEN。"""
        with patch("src.api.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 0.0
            cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
            # CLOSED → OPEN
            with self.assertRaises(ValueError):
                cb.call(_raise_value_error)
            self.assertEqual(cb.state, CircuitState.OPEN)
            # 时间前进，超过恢复超时
            mock_time.return_value = 100.0
            # HALF_OPEN 试探失败 → 回到 OPEN
            with self.assertRaises(ValueError):
                cb.call(_raise_value_error)
            self.assertEqual(cb.state, CircuitState.OPEN)
            # 再次未超时拒绝调用（last_failure_time 已更新为 100.0）
            mock_time.return_value = 130.0
            with self.assertRaises(CircuitOpenError):
                cb.call(lambda: "rejected")


# ── 测试辅助函数：定义在模块级以便 MagicMock 等正常使用 ──


def _raise_value_error() -> None:
    """总是抛出 ValueError 的辅助函数，用于触发熔断器失败计数。"""
    raise ValueError("test failure")


def _raise_runtime_error() -> None:
    """总是抛出 RuntimeError 的辅助函数，用于验证原始异常透传。"""
    raise RuntimeError("boom")


if __name__ == "__main__":
    unittest.main()
