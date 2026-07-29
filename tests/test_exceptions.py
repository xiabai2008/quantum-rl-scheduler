"""
异常模块单元测试
Unit tests for src/exceptions.py
"""

from __future__ import annotations

import pytest

from src.exceptions import (
    CircuitOpenError,
    ConfigurationError,
    QuantumAnnealingError,
    QuantumSchedulerError,
    RateLimitError,
    ResourceExhaustedError,
    SchedulingError,
    TaskParseError,
    TianyanAPIError,
)


class TestQuantumSchedulerError:
    """测试基类 QuantumSchedulerError"""

    def test_defaults(self) -> None:
        """验证默认属性值"""
        err = QuantumSchedulerError("test error")
        assert str(err) == "test error"
        assert err.code == "UNKNOWN"
        assert err.retryable is False

    def test_custom_code(self) -> None:
        """验证自定义错误码"""
        err = QuantumSchedulerError("test error", code="CUSTOM_ERROR")
        assert err.code == "CUSTOM_ERROR"
        assert err.retryable is False

    def test_retryable_true(self) -> None:
        """验证可重试标志"""
        err = QuantumSchedulerError("test error", code="RETRYABLE", retryable=True)
        assert err.code == "RETRYABLE"
        assert err.retryable is True

    def test_is_exception_subclass(self) -> None:
        """验证是 Exception 的子类"""
        err = QuantumSchedulerError("test")
        assert isinstance(err, Exception)


class TestTianyanAPIError:
    """测试 TianyanAPIError"""

    def test_inheritance(self) -> None:
        """验证继承关系"""
        err = TianyanAPIError("api error")
        assert isinstance(err, QuantumSchedulerError)
        assert isinstance(err, Exception)

    def test_defaults(self) -> None:
        """验证默认值"""
        err = TianyanAPIError("api failed")
        assert str(err) == "api failed"
        assert err.code == "TIANYAN_API_ERROR"
        assert err.retryable is False


class TestCircuitOpenError:
    """测试 CircuitOpenError"""

    def test_inheritance(self) -> None:
        """验证继承关系"""
        err = CircuitOpenError("circuit open")
        assert isinstance(err, QuantumSchedulerError)

    def test_default_retryable(self) -> None:
        """验证默认不可重试"""
        err = CircuitOpenError("circuit open")
        assert err.retryable is False


class TestConfigurationError:
    """测试 ConfigurationError"""

    def test_inheritance(self) -> None:
        """验证继承关系"""
        err = ConfigurationError("config error")
        assert isinstance(err, QuantumSchedulerError)

    def test_not_retryable(self) -> None:
        """验证配置错误默认不可重试"""
        err = ConfigurationError("missing config")
        assert err.retryable is False


class TestTaskParseError:
    """测试 TaskParseError"""

    def test_inheritance(self) -> None:
        """验证继承关系"""
        err = TaskParseError("parse error")
        assert isinstance(err, QuantumSchedulerError)


class TestSchedulingError:
    """测试 SchedulingError"""

    def test_inheritance(self) -> None:
        """验证继承关系"""
        err = SchedulingError("scheduling failed")
        assert isinstance(err, QuantumSchedulerError)


class TestQuantumAnnealingError:
    """测试 QuantumAnnealingError"""

    def test_inheritance(self) -> None:
        """验证继承关系"""
        err = QuantumAnnealingError("annealing failed")
        assert isinstance(err, QuantumSchedulerError)


class TestResourceExhaustedError:
    """测试 ResourceExhaustedError"""

    def test_inheritance(self) -> None:
        """验证继承关系"""
        err = ResourceExhaustedError("no qubits left")
        assert isinstance(err, QuantumSchedulerError)


class TestRateLimitError:
    """测试 RateLimitError"""

    def test_inheritance(self) -> None:
        """验证继承关系"""
        err = RateLimitError("rate limited")
        assert isinstance(err, QuantumSchedulerError)

    def test_default_retryable(self) -> None:
        """验证限流错误默认可重试"""
        err = RateLimitError("rate limited")
        assert err.retryable is True

    def test_default_code(self) -> None:
        """验证默认错误码"""
        err = RateLimitError("rate limited")
        assert err.code == "RATE_LIMIT"

    def test_retry_after_default_none(self) -> None:
        """验证 retry_after 默认为 None"""
        err = RateLimitError("rate limited")
        assert err.retry_after is None

    def test_retry_after_custom(self) -> None:
        """验证自定义 retry_after"""
        err = RateLimitError("rate limited", retry_after=30.0)
        assert err.retry_after == 30.0

    def test_custom_code_and_retryable(self) -> None:
        """验证自定义 code 和 retryable"""
        err = RateLimitError(
            "rate limited",
            code="MY_RATE_LIMIT",
            retryable=False,
            retry_after=5.0,
        )
        assert err.code == "MY_RATE_LIMIT"
        assert err.retryable is False
        assert err.retry_after == 5.0


class TestAllExceptionsInstantiable:
    """参数化测试：所有异常类均可正常实例化"""

    @pytest.mark.parametrize(
        "exc_class",
        [
            QuantumSchedulerError,
            TianyanAPIError,
            CircuitOpenError,
            ConfigurationError,
            TaskParseError,
            SchedulingError,
            QuantumAnnealingError,
            ResourceExhaustedError,
            RateLimitError,
        ],
    )
    def test_can_instantiate(self, exc_class: type[QuantumSchedulerError]) -> None:
        """验证所有异常类都能用消息字符串实例化"""
        err = exc_class(f"test {exc_class.__name__}")
        assert isinstance(err, QuantumSchedulerError)
        assert isinstance(err, Exception)
        assert str(err) == f"test {exc_class.__name__}"
