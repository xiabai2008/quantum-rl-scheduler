"""scripts/real_machine 包初始化模块。"""

import time
from collections.abc import Callable
from typing import TypeVar

from loguru import logger

T = TypeVar("T")

# Issue #690: 可重试的网络/临时错误类型
_RETRYABLE_ERRORS = (ConnectionError, TimeoutError, OSError)


def with_retry(
    func: Callable[..., T],
    *args: object,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs: object,
) -> T:
    """带指数退避的重试包装器，用于真机任务提交和轮询。

    仅对网络/临时错误（ConnectionError/TimeoutError/OSError）重试，
    对编程错误（ValueError/TypeError）和业务错误（QCIS校验失败）直接抛出。

    Args:
        func: 待调用的函数
        *args: 透传给 func 的位置参数
        max_retries: 最大重试次数
        base_delay: 基础退避延迟（秒），实际延迟 = base_delay * 2^attempt
        **kwargs: 透传给 func 的关键字参数

    Returns:
        func 的返回值

    Raises:
        最后一次重试的异常
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except _RETRYABLE_ERRORS as e:
            last_exc = e
            if attempt < max_retries - 1:
                wait = base_delay * (2**attempt)
                logger.warning(
                    f"[Retry] 第{attempt + 1}/{max_retries}次失败，"
                    f"{wait:.1f}s后重试: {type(e).__name__}: {e}"
                )
                time.sleep(wait)
            else:
                logger.error(f"[Retry] 重试耗尽（{max_retries}次）: {e}")
        except (ValueError, TypeError, KeyError, AttributeError):
            raise
    assert last_exc is not None
    raise last_exc
