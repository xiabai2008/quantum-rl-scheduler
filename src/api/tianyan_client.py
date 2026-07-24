"""
天衍云平台 API 封装客户端
Tianyan Cloud Platform API Client

封装天衍量子云平台的 API 接口，支持量子/经典任务提交、
状态查询、结果获取、后端管理等功能。

认证方式：cqlib SDK（从环境变量 TIANYAN_API_KEY 读取）
配置来源：config/config.yaml + .env
"""

import os
import random
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import monotonic
from typing import Any, TypeVar, cast

import requests
import yaml
from dotenv import load_dotenv
from loguru import logger

from src.api.circuit_breaker import CircuitBreaker
from src.exceptions import QuantumSchedulerError, RateLimitError
from src.utils.metrics import (
    api_calls,
    api_errors,
    api_request_duration,
    tianyan_cb_state,
)

T = TypeVar("T")


def mask_token(token: str) -> str:
    """对 API 令牌进行脱敏处理，仅保留首尾各 4 个字符。

    用于日志输出时隐藏完整令牌，防止敏感信息泄露。
    短令牌（≤8 字符）全部以 * 替换；空字符串原样返回。

    Args:
        token: 原始令牌字符串

    Returns:
        脱敏后的字符串，例如 ``"abcd****wxyz"``；空字符串返回 ``""``
    """
    if not token:
        return ""
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}{'*' * (len(token) - 8)}{token[-4:]}"


class TianyanAPIError(QuantumSchedulerError):
    """天衍云平台 API 自定义异常

    当 API 返回非 200 状态码时抛出，携带状态码和响应详情。
    继承自 QuantumSchedulerError，纳入统一异常体系。

    Attributes:
        status_code: HTTP 响应状态码
        message: 错误描述信息
        response_body: 原始响应体（JSON）
    """

    def __init__(self, status_code: int, message: str, response_body: dict | None = None):
        self.status_code = status_code
        self.message = message
        self.response_body = response_body or {}
        super().__init__(
            f"[{status_code}] {message}", code=str(status_code), retryable=status_code >= 500
        )


class TokenBucketRateLimiter:
    """令牌桶限流器

    基于令牌桶算法控制请求频率：桶以恒定速率 ``rate`` 生成令牌，每次请求
    消费一定数量令牌；桶满时多余的令牌溢出，桶空时请求需等待令牌补充。

    适用于平滑突发流量、保护下游服务不被过载的场景。

    Args:
        capacity: 桶容量（最大突发请求数），达到上限后多余令牌溢出
        rate: 令牌生成速率（每秒生成的令牌数），即稳态最大 QPS

    Attributes:
        capacity: 桶容量
        rate: 令牌生成速率（令牌/秒）
    """

    def __init__(self, capacity: float, rate: float) -> None:
        """初始化令牌桶限流器

        Args:
            capacity: 桶容量（最大突发请求数）
            rate: 令牌生成速率（每秒令牌数）
        """
        self.capacity: float = capacity
        self.rate: float = rate
        self._tokens: float = capacity
        self._last_refill: float = monotonic()

    def _refill(self) -> None:
        """根据流逝时间补充令牌

        使用单调时钟计算距上次补充的时间差，按 ``rate`` 补充令牌，
        但不超过 ``capacity`` 上限。
        """
        now = monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last_refill = now

    def acquire(self, tokens: float = 1.0) -> float:
        """获取令牌，返回需要等待的时间

        若桶中令牌充足则立即消费并返回 0；若不足则计算补充所需时间并返回。
        调用方应自行 ``time.sleep`` 等待返回的秒数后再重试。

        Args:
            tokens: 需要消费的令牌数，默认 1.0

        Returns:
            需要等待的时间（秒）；0.0 表示无需等待
        """
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return 0.0
        deficit = tokens - self._tokens
        wait_time = deficit / self.rate if self.rate > 0 else float("inf")
        return wait_time

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """尝试获取令牌（非阻塞）

        若桶中令牌充足则消费并返回 True；否则返回 False，不阻塞。

        Args:
            tokens: 需要消费的令牌数，默认 1.0

        Returns:
            ``True`` 表示获取成功，``False`` 表示令牌不足
        """
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    @property
    def available_tokens(self) -> float:
        """当前可用令牌数（补充后）"""
        self._refill()
        return self._tokens


class QuotaTracker:
    """API 配额追踪器

    按小时与日两个窗口统计 API 调用次数，窗口过期后自动重置计数。
    使用 ``time.time()``（墙上时钟）而非单调时钟，以便与服务器侧配额窗口对齐。

    Attributes:
        hourly_count: 当前小时窗口内的调用次数
        daily_count: 当前日窗口内的调用次数
    """

    HOUR_SECONDS: float = 3600.0
    DAY_SECONDS: float = 86400.0

    def __init__(self) -> None:
        """初始化配额追踪器，窗口起点设为当前时间"""
        self._hourly_count: int = 0
        self._daily_count: int = 0
        self._hourly_window_start: float = time.time()
        self._daily_window_start: float = time.time()

    def record(self) -> None:
        """记录一次 API 调用，自动检查并重置过期窗口"""
        now = time.time()
        if now - self._hourly_window_start >= self.HOUR_SECONDS:
            self._hourly_count = 0
            self._hourly_window_start = now
        if now - self._daily_window_start >= self.DAY_SECONDS:
            self._daily_count = 0
            self._daily_window_start = now
        self._hourly_count += 1
        self._daily_count += 1

    @property
    def hourly_count(self) -> int:
        """当前小时窗口内的调用次数（过期窗口返回 0）"""
        if time.time() - self._hourly_window_start >= self.HOUR_SECONDS:
            return 0
        return self._hourly_count

    @property
    def daily_count(self) -> int:
        """当前日窗口内的调用次数（过期窗口返回 0）"""
        if time.time() - self._daily_window_start >= self.DAY_SECONDS:
            return 0
        return self._daily_count

    def reset(self) -> None:
        """手动重置所有计数与窗口起点（主要用于测试）"""
        now = time.time()
        self._hourly_count = 0
        self._daily_count = 0
        self._hourly_window_start = now
        self._daily_window_start = now


class TianyanClient:
    """
    天衍量子云平台客户端

    封装天衍量子云平台的所有 API 接口，提供：
    - 量子电路任务提交（QCIS/QASM 格式）
    - 经典计算任务提交
    - 任务状态查询与结果获取
    - 量子后端信息查询
    - 队列状态监控
    - 熔断器模式（可选）

    真实模式使用 cqlib SDK；Mock 模式使用模拟客户端（开发阶段使用）。

    使用示例::

        # 真实模式（需要有效 API Key）
        client = TianyanClient()

        # Mock 模式（开发/测试用）
        client = TianyanClient(mock_mode=True)

        # 自动检测配置（推荐）
        client = TianyanClient()  # 会根据 config.yaml 和环境变量自动选择

        # 禁用熔断器
        client = TianyanClient(enable_circuit_breaker=False)

        if client.authenticate():
            task_id = client.submit_quantum_task(qcis="H Q0\\nM Q0")
            status = client.get_task_status(task_id)
            result = client.get_task_result(task_id)

    Args:
        api_key: API 密钥（默认从环境变量 TIANYAN_API_KEY 读取）
        base_url: API 基础 URL（默认从 config/config.yaml 读取，仅用于日志）
        mock_mode: 是否使用 Mock 模式（None 表示自动检测）
        enable_circuit_breaker: 是否启用熔断器（默认 True）
        timeout: 单次 API 请求超时时间（秒），默认 30.0
        max_retries: API 请求失败最大重试次数，默认 3
        retry_delay: 重试间隔（秒），默认 1.0
        max_requests_per_second: 每秒最大请求数（令牌桶限流），默认 None 不限流
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        mock_mode: bool | None = None,
        enable_circuit_breaker: bool = True,
        timeout: float | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
        max_requests_per_second: float | None = None,
    ):
        """初始化天衍云客户端

        按优先级读取配置：显式传参 > 环境变量 > config/config.yaml 默认值。

        Mock 模式检测顺序：
        1. 显式传参 ``mock_mode``
        2. 环境变量 ``TIANYAN_MOCK_MODE``（true/1/yes）
        3. 配置文件 ``config/config.yaml`` 中的 ``tianyan.mock_mode``
        4. 默认：True（开发阶段默认使用 Mock 模式）

        请求超时与重试配置读取顺序：显式传参 > 环境变量 > 默认值：
        - ``timeout`` ← ``TIANYAN_API_TIMEOUT``（默认 30.0）
        - ``max_retries`` ← ``TIANYAN_API_MAX_RETRIES``（默认 3）
        - ``retry_delay`` ← ``TIANYAN_API_RETRY_DELAY``（默认 1.0）

        限流配置（Issue #84）：
        - ``max_requests_per_second`` ← ``TIANYAN_API_RATE_LIMIT``（默认 None 不限流）
        - 启用后使用令牌桶算法控制请求频率，429 响应触发自适应指数退避 + jitter
        - 限流触发的失败（RateLimitError）不计入熔断器失败计数

        Args:
            api_key: API 密钥，若为 None 则从环境变量 ``TIANYAN_API_KEY``
                （或别名 ``TIANYAN_API_TOKEN``）读取。真实模式下若未提供且
                环境变量也未配置，将抛出 ``ValueError``。
            base_url: API 基础 URL，若为 None 则从 ``config/config.yaml`` 读取。
            mock_mode: 是否使用 Mock 模式，None 表示自动检测。
            enable_circuit_breaker: 是否启用熔断器模式。
            timeout: 单次 API 请求超时时间（秒），默认 30.0，
                可通过环境变量 ``TIANYAN_API_TIMEOUT`` 覆盖。
            max_retries: API 请求失败最大重试次数，默认 3,
                可通过环境变量 ``TIANYAN_API_MAX_RETRIES`` 覆盖。
            retry_delay: 重试间隔（秒），默认 1.0，
                可通过环境变量 ``TIANYAN_API_RETRY_DELAY`` 覆盖。
            max_requests_per_second: 每秒最大请求数（令牌桶限流），
                None 或 0 表示不限流；也可通过环境变量
                ``TIANYAN_API_RATE_LIMIT`` 配置。启用后 429 响应将触发
                自适应指数退避 + 随机 jitter，且限流失败不计入熔断器。

        Raises:
            ValueError: 真实模式下未提供 ``api_key`` 且环境变量
                ``TIANYAN_API_KEY`` / ``TIANYAN_API_TOKEN`` 均未配置时。
        """
        # 按需加载 .env 文件中的环境变量
        load_dotenv()

        # 请求超时与重试配置（显式传参 > 环境变量 > 默认值）
        self.timeout: float = (
            timeout if timeout is not None else float(os.getenv("TIANYAN_API_TIMEOUT", "30.0"))
        )
        self.max_retries: int = (
            max_retries
            if max_retries is not None
            else int(os.getenv("TIANYAN_API_MAX_RETRIES", "3"))
        )
        self.retry_delay: float = (
            retry_delay
            if retry_delay is not None
            else float(os.getenv("TIANYAN_API_RETRY_DELAY", "1.0"))
        )

        # 限流配置（Issue #84）：显式传参 > 环境变量 > None（不限流）
        rate_limit = (
            max_requests_per_second
            if max_requests_per_second is not None
            else float(os.getenv("TIANYAN_API_RATE_LIMIT", "0"))
        )
        if rate_limit > 0:
            self._rate_limiter: TokenBucketRateLimiter | None = TokenBucketRateLimiter(
                capacity=rate_limit, rate=rate_limit
            )
            logger.info(f"API 限流已启用：{rate_limit} 请求/秒")
        else:
            self._rate_limiter = None

        # 配额追踪（始终启用，仅记录不影响行为）
        self._quota_tracker: QuotaTracker = QuotaTracker()

        # 确定是否使用 Mock 模式
        self.mock_mode = self._detect_mock_mode(mock_mode)
        logger.info(f"Mock 模式: {self.mock_mode}")

        # cqlib 委托客户端（真实模式才创建，先置 None 保证属性始终存在）
        self._cqlib = None
        self._mock_client: Any = None

        # 熔断器
        self._circuit_breaker: CircuitBreaker | None = (
            CircuitBreaker() if enable_circuit_breaker else None
        )

        if self.mock_mode:
            # Mock 模式：创建 Mock 客户端并委托所有 API 调用
            from src.api.mock_client import MockTianyanClient

            self._mock_client = MockTianyanClient(
                mock_delay=float(os.getenv("TIANYAN_MOCK_DELAY", "1.0")),
                mock_failure_rate=float(os.getenv("TIANYAN_MOCK_FAILURE_RATE", "0.0")),
            )
            logger.info("✅ 使用 Mock 模式（不依赖真实平台）")
            return

        # 真实模式：初始化真实 API 客户端
        self._mock_client = None

        # 加载 API 凭证（必需，缺失则抛出 ValueError）
        creds = self._load_credentials(api_key=api_key)
        self.api_key = creds["api_key"]
        # 仅记录脱敏后的密钥，避免敏感信息泄露到日志
        logger.info(f"已加载 API 密钥: {mask_token(self.api_key)}")

        # 读取 base_url（从配置文件回退）
        self.base_url = base_url or self._load_base_url_from_config()
        logger.info(f"天衍客户端初始化完成，base_url={self.base_url}")

        # 真实模式统一走 cqlib SDK（REST API 被 WAF 拦截，已弃用）
        machine_name = os.getenv("TIANYAN_MACHINE", "tianyan_s")
        try:
            from src.api.tianyan_cqlib import CqlibTianyanClient

            self._cqlib = CqlibTianyanClient(
                login_key=self.api_key,
                machine_name=machine_name,
                auto_retry_machine=True,
            )
            logger.info(f"✅ 真实模式委托 cqlib（机器={machine_name}）")
        except Exception as e:
            # 涉及 cqlib SDK 导入与初始化，异常类型无法穷举，保留宽捕获并记录日志
            logger.warning(f"cqlib 客户端初始化失败: {e}，回退 REST 路径")

        self.session = None

    @staticmethod
    def _detect_mock_mode(explicit_mock_mode: bool | None) -> bool:
        """检测是否使用 Mock 模式

        Args:
            explicit_mock_mode: 显式传参的 mock_mode 值

        Returns:
            是否使用 Mock 模式
        """
        # 1. 显式传参优先
        if explicit_mock_mode is not None:
            return explicit_mock_mode

        # 2. 环境变量
        mock_env = os.getenv("TIANYAN_MOCK_MODE", "").lower()
        if mock_env in ("true", "1", "yes"):
            return True
        if mock_env in ("false", "0", "no"):
            return False

        # 3. 配置文件
        try:
            with open("config/config.yaml", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            mock_config = config.get("tianyan", {}).get("mock_mode", True)
            return cast(bool, mock_config)
        except (yaml.YAMLError, OSError, AttributeError) as e:
            # YAML 解析失败、文件读取失败或配置结构异常时，默认使用 Mock 模式
            logger.debug(f"读取 mock_mode 配置失败: {e}，默认使用 Mock 模式")
            return True  # 默认使用 Mock 模式

    @staticmethod
    def _load_credentials(api_key: str | None = None) -> dict[str, str]:
        """从环境变量加载 API 凭证

        按优先级读取凭证：显式传参 > 环境变量。对于敏感字段（密钥/令牌）
        不提供任何默认值，缺失时抛出 ``ValueError`` 以便上层显式处理。

        读取的凭证变量：
        - ``TIANYAN_API_KEY``：天衍云 API 密钥（必需，别名 ``TIANYAN_API_TOKEN``）
        - ``TIANYAN_API_SECRET``：API 密钥对应的 Secret（可选）
        - ``TIANYAN_APP_ID``：应用 ID（可选）

        Args:
            api_key: 显式传入的 API 密钥，优先于环境变量

        Returns:
            凭证字典，至少包含 ``api_key`` 字段；可选字段仅在环境变量设置时出现

        Raises:
            ValueError: 当必需的 API 密钥既未显式传入、也未在环境变量中配置时
        """
        credentials: dict[str, str] = {}

        # API Key（必需）：显式传参 > TIANYAN_API_KEY > TIANYAN_API_TOKEN（别名）
        key = api_key or os.getenv("TIANYAN_API_KEY") or os.getenv("TIANYAN_API_TOKEN")
        if not key:
            raise ValueError(
                "未配置天衍云 API 密钥：请设置环境变量 TIANYAN_API_KEY"
                "（或别名 TIANYAN_API_TOKEN）后再启动真实模式，"
                "或在初始化 TianyanClient 时显式传入 api_key 参数。"
            )
        credentials["api_key"] = key

        # API Secret（可选）：仅当环境变量设置时才使用，不提供默认值
        secret = os.getenv("TIANYAN_API_SECRET")
        if secret:
            credentials["api_secret"] = secret

        # App ID（可选）：仅当环境变量设置时才使用，不提供默认值
        app_id = os.getenv("TIANYAN_APP_ID")
        if app_id:
            credentials["app_id"] = app_id

        return credentials

    @staticmethod
    def _load_base_url_from_config(config_path: str = "config/config.yaml") -> str:
        """从 config.yaml 读取 api.base_url

        Args:
            config_path: 配置文件路径

        Returns:
            API 基础 URL，读取失败时返回默认值
        """
        default_url = "https://api.tianyanyun.cn/v1"
        try:
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            url = config.get("tianyan", {}).get("base_url", default_url)
            return cast(str, url)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 不存在，使用默认 base_url")
            return default_url
        except (yaml.YAMLError, AttributeError, OSError) as e:
            logger.warning(f"读取配置文件失败: {e}，使用默认 base_url")
            return default_url

    # ------------------------------------------------------------------
    # 内部工具：限流、重试、指标埋点、熔断器状态同步
    # ------------------------------------------------------------------

    def _apply_rate_limit(self) -> None:
        """应用令牌桶限流

        若限流器已启用且令牌不足，阻塞等待令牌补充后再继续。
        限流器未启用时（``_rate_limiter is None``）直接返回。
        """
        if self._rate_limiter is None:
            return
        wait_time = self._rate_limiter.acquire()
        if wait_time > 0:
            logger.debug(f"令牌桶限流：等待 {wait_time:.3f}s 后放行")
            time.sleep(wait_time)

    def _track_quota(self) -> None:
        """记录一次 API 调用到配额追踪器"""
        self._quota_tracker.record()

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        """判断异常是否由限流（HTTP 429）引起

        检查异常是否为 ``RateLimitError`` 或携带 ``status_code == 429`` 属性。

        Args:
            exc: 待检查的异常对象

        Returns:
            ``True`` 表示该异常由限流引起
        """
        if isinstance(exc, RateLimitError):
            return True
        status_code = getattr(exc, "status_code", None)
        return status_code == 429

    def _compute_adaptive_backoff(self, attempt: int, retry_after: float | None = None) -> float:
        """计算自适应退避时间（指数退避 + 随机 jitter）

        基数为 ``retry_delay * 2^attempt``，叠加 0~10% 的随机 jitter，
        若服务端返回 ``Retry-After`` 则取其与指数退避的较大值。

        Args:
            attempt: 当前重试次数（从 0 开始）
            retry_after: 服务端建议的等待时间（秒），可选

        Returns:
            退避时间（秒）
        """
        base: float = self.retry_delay * (2**attempt)
        jitter: float = random.uniform(0, base * 0.1)
        backoff: float = base + jitter
        if retry_after is not None and retry_after > backoff:
            return retry_after
        return backoff

    def _call_with_retry(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """带重试与限流的 API 调用包装器

        集成令牌桶限流、配额追踪、429 自适应退避与常规重试：

        - 每次调用前应用令牌桶限流（若已启用）并记录配额
        - 429 限流异常使用指数退避 + jitter 重试，不计入熔断器失败计数
        - 其他异常按 ``retry_delay`` 固定间隔重试
        - 重试耗尽后抛出最后一次异常

        Args:
            func: 待调用的可调用对象。
            *args: 透传给 ``func`` 的位置参数。
            **kwargs: 透传给 ``func`` 的关键字参数。

        Returns:
            ``func`` 的返回值。

        Raises:
            Exception: 重试耗尽后抛出最后一次异常。
            RateLimitError: 429 限流重试耗尽后抛出。
        """
        last_exc: Exception | None = None
        total_attempts = max(1, self.max_retries + 1)
        for attempt in range(total_attempts):
            try:
                # 限流：获取令牌（必要时阻塞等待）
                self._apply_rate_limit()
                # 配额追踪
                self._track_quota()
                return func(*args, **kwargs)
            except Exception as e:
                # 429 限流：转换为 RateLimitError 并使用自适应退避
                if self._is_rate_limited(e):
                    retry_after = getattr(e, "retry_after", None)
                    if not isinstance(e, RateLimitError):
                        e = RateLimitError(
                            f"API 限流（429）: {e}",
                            retry_after=retry_after,
                        )
                    last_exc = e
                    if attempt < total_attempts - 1:
                        backoff = self._compute_adaptive_backoff(attempt, retry_after)
                        logger.debug(
                            f"API 限流（429），第 {attempt + 1}/{total_attempts} 次退避 "
                            f"{backoff:.3f}s 后重试"
                        )
                        time.sleep(backoff)
                    else:
                        logger.debug(f"API 限流重试耗尽（共 {total_attempts} 次）: {e}")
                    continue
                last_exc = e
                if attempt < total_attempts - 1:
                    logger.debug(
                        f"API 调用第 {attempt + 1}/{total_attempts} 次失败: "
                        f"{type(e).__name__}: {e}，{self.retry_delay}s 后重试"
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.debug(
                        f"API 调用重试耗尽（共 {total_attempts} 次）: {type(e).__name__}: {e}"
                    )
        assert last_exc is not None
        raise last_exc

    @contextmanager
    def _observe_api_call(self, method: str, endpoint: str) -> Iterator[None]:
        """记录 API 调用指标的上下文管理器

        进入时递增请求计数器；退出时将耗时记录到延迟直方图；
        若调用抛出异常则递增错误计数器；最后同步熔断器状态 Gauge。

        Args:
            method: 调用方法名称（如 ``"submit_quantum_task"``）。
            endpoint: API 端点名称（如 ``"quantum_task"``）。
        """
        api_calls.labels(method=method, endpoint=endpoint).inc()
        start = monotonic()
        try:
            yield
        except Exception as e:
            api_errors.labels(method=method, endpoint=endpoint, error_type=type(e).__name__).inc()
            raise
        finally:
            duration = monotonic() - start
            api_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
            self._update_circuit_breaker_gauge()

    def _update_circuit_breaker_gauge(self) -> None:
        """同步熔断器状态到 Prometheus Gauge

        将熔断器字符串状态映射为数值并写入 ``tianyan_circuit_breaker_state``：
        closed=0, open=1, half_open=2。
        """
        state_map = {"closed": 0, "open": 1, "half_open": 2}
        state = self.get_circuit_state()
        tianyan_cb_state.set(state_map.get(state, 0))

    # ------------------------------------------------------------------
    # 1. 认证验证
    # ------------------------------------------------------------------

    def authenticate(self) -> bool:
        """验证 API 密钥有效性

        真实模式委托 cqlib 认证；Mock 模式委托 Mock 客户端。

        Returns:
            ``True`` 表示认证通过，``False`` 表示认证失败。
        """
        # Mock 模式委托
        if self.mock_mode and hasattr(self, "_mock_client") and self._mock_client:
            with self._observe_api_call("authenticate", "auth"):
                return cast(bool, self._mock_client.authenticate())

        # 真实模式委托 cqlib
        if self._cqlib is not None:
            with self._observe_api_call("authenticate", "auth"):
                return self._call_with_retry(self._cqlib.authenticate)

        logger.error("认证失败：未配置有效 API 密钥或 cqlib 客户端")
        return False

    # ------------------------------------------------------------------
    # 2. 量子任务提交
    # ------------------------------------------------------------------

    def submit_quantum_task(
        self,
        circuit_qasm: str = "",
        shots: int = 1024,
        backend: str = "tianyan_s",
        qcis: str = "",
        task_name: str = "Scheduler_Task",
    ) -> str:
        """提交量子计算任务

        真实模式委托 cqlib（接受 QCIS 格式）；Mock 模式委托 Mock 客户端（QASM 格式）。

        Args:
            circuit_qasm: QASM 格式量子电路字符串（Mock 模式用；真实模式建议用 qcis）
            shots: 重复测量次数，默认 1024
            backend: 量子后端名称，真实模式默认 ``tianyan_s``
            qcis: QCIS 指令字符串（真实模式优先使用，如 ``"H Q0\\nM Q0"``）
            task_name: 任务名称（真实模式用）

        Returns:
            任务 ID（task_id）字符串
        """
        # Mock 模式委托
        if self.mock_mode and hasattr(self, "_mock_client") and self._mock_client:
            with self._observe_api_call("submit_quantum_task", "quantum_task"):
                return cast(
                    str,
                    self._mock_client.submit_quantum_task(
                        circuit_qasm=circuit_qasm, shots=shots, backend=backend
                    ),
                )

        # 真实模式委托 cqlib
        if self._cqlib is not None:
            qcis_str = qcis or circuit_qasm
            if not qcis_str:
                raise ValueError("真实模式需提供 qcis 或 circuit_qasm")
            with self._observe_api_call("submit_quantum_task", "quantum_task"):
                task_id = self._call_with_retry(
                    self._cqlib.submit_quantum_task,
                    qcis=qcis_str,
                    shots=shots,
                    task_name=task_name,
                )
                if task_id is None:
                    raise TianyanAPIError(500, "cqlib did not return a task_id")
                return task_id

        raise TianyanAPIError(
            status_code=500,
            message="未配置有效 API 密钥或 cqlib 客户端，无法提交量子任务",
        )

    # ------------------------------------------------------------------
    # 3. 查询任务状态
    # ------------------------------------------------------------------

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """查询任务执行状态

        Args:
            task_id: 任务 ID

        Returns:
            状态字典，至少包含 ``status`` 字段
        """
        if self._circuit_breaker:
            self._circuit_breaker.before_request()

        try:
            with self._observe_api_call("get_task_status", "task_status"):
                if self.mock_mode and hasattr(self, "_mock_client") and self._mock_client:
                    result = cast(dict[str, Any], self._mock_client.get_task_status(task_id))
                    if self._circuit_breaker:
                        self._circuit_breaker.on_success()
                    return result

                # 真实模式委托 cqlib
                if self._cqlib is not None:
                    result = self._cqlib.get_task_status(task_id)
                    if self._circuit_breaker:
                        self._circuit_breaker.on_success()
                    return result

                raise TianyanAPIError(
                    status_code=500,
                    message="未配置有效 API 密钥或 cqlib 客户端，无法查询任务状态",
                )
        except Exception as e:
            # 429 限流：转换为 RateLimitError，不计入熔断器失败计数（Issue #84）
            if self._is_rate_limited(e):
                if not isinstance(e, RateLimitError):
                    e = RateLimitError(
                        f"API 限流（429）: {e}",
                        retry_after=getattr(e, "retry_after", None),
                    )
                logger.debug(f"get_task_status 限流触发，不计入熔断器: {e}")
                raise e
            # 其他异常计入熔断器失败计数，原异常重新抛出由上层处理
            logger.debug(f"get_task_status 失败，已触发熔断器失败计数: {type(e).__name__}: {e}")
            if self._circuit_breaker:
                self._circuit_breaker.on_failure()
            raise

    # ------------------------------------------------------------------
    # 4. 获取任务结果
    # ------------------------------------------------------------------

    def get_task_result(self, task_id: str) -> dict[str, Any]:
        """获取任务执行结果

        仅当任务状态为 ``COMPLETED`` 时返回有效测量结果。

        Args:
            task_id: 任务 ID

        Returns:
            结果字典，包含 ``counts``（测量计数）、``metadata``（元数据）等字段

        Raises:
            TianyanAPIError: 查询失败或任务尚未完成时抛出
        """
        if self.mock_mode and hasattr(self, "_mock_client") and self._mock_client:
            with self._observe_api_call("get_task_result", "task_result"):
                return cast(dict[str, Any], self._mock_client.get_task_result(task_id))

        # 真实模式委托 cqlib
        if self._cqlib is not None:
            with self._observe_api_call("get_task_result", "task_result"):
                return self._call_with_retry(self._cqlib.get_task_result, task_id)

        raise TianyanAPIError(
            status_code=500,
            message="未配置有效 API 密钥或 cqlib 客户端，无法获取任务结果",
        )

    # ------------------------------------------------------------------
    # 5. 列出可用量子后端
    # ------------------------------------------------------------------

    def list_backends(self) -> list[dict[str, Any]]:
        """列出平台上所有可用的量子计算后端

        Returns:
            后端信息列表，每个元素为字典，包含 ``name``、``type``
            （superconducting / photonic）等字段

        Raises:
            TianyanAPIError: 查询失败时抛出
        """
        if self.mock_mode and hasattr(self, "_mock_client") and self._mock_client:
            with self._observe_api_call("list_backends", "backends"):
                return cast(list[dict[str, Any]], self._mock_client.list_backends())

        # 真实模式委托 cqlib
        if self._cqlib is not None:
            with self._observe_api_call("list_backends", "backends"):
                return self._call_with_retry(self._cqlib.list_backends)

        raise TianyanAPIError(
            status_code=500,
            message="未配置有效 API 密钥或 cqlib 客户端，无法获取后端列表",
        )

    # ------------------------------------------------------------------
    # 6. 获取后端详细信息
    # ------------------------------------------------------------------

    def get_backend_info(self, backend_name: str) -> dict[str, Any]:
        """获取指定量子后端的详细信息

        Args:
            backend_name: 后端名称，如 ``tianyan-287``

        Returns:
            后端详情字典，包含：
            - ``name``: 后端名称
            - ``num_qubits``: 可用量子比特数
            - ``fidelity``: 单/双量子比特门保真度
            - ``queue_depth``: 当前队列中的任务数
            - ``status``: 在线/离线状态

        Raises:
            TianyanAPIError: 查询失败或后端不存在时抛出
        """
        if self.mock_mode and hasattr(self, "_mock_client") and self._mock_client:
            with self._observe_api_call("get_backend_info", "backend_info"):
                return cast(dict[str, Any], self._mock_client.get_backend_info(backend_name))

        # 真实模式委托 cqlib
        if self._cqlib is not None:
            with self._observe_api_call("get_backend_info", "backend_info"):
                return self._call_with_retry(self._cqlib.get_backend_info, backend_name)

        raise TianyanAPIError(
            status_code=500,
            message="未配置有效 API 密钥或 cqlib 客户端，无法获取后端信息",
        )

    # ------------------------------------------------------------------
    # 7. 提交经典计算任务
    # ------------------------------------------------------------------

    def submit_classical_task(self, code: str, language: str = "python3") -> str:
        """提交经典计算任务

        将经典计算代码提交至平台的经典计算节点执行。

        Args:
            code: 要执行的代码字符串
            language: 编程语言，默认 ``python3``，支持 ``python3`` / ``c`` / ``cpp``

        Returns:
            任务 ID（task_id）字符串

        Raises:
            TianyanAPIError: 提交失败时抛出
        """
        if self.mock_mode and hasattr(self, "_mock_client") and self._mock_client:
            with self._observe_api_call("submit_classical_task", "classical_task"):
                return cast(
                    str,
                    self._mock_client.submit_classical_task(code=code, language=language),
                )

        raise TianyanAPIError(
            status_code=500,
            message="未配置有效 API 密钥或 cqlib 客户端，无法提交经典任务",
        )

    # ------------------------------------------------------------------
    # 8. 获取队列状态
    # ------------------------------------------------------------------

    def get_queue_status(self) -> dict[str, Any]:
        """获取当前平台任务队列状态

        Returns:
            队列状态字典，包含：
            - ``total_pending``: 排队中任务数
            - ``total_running``: 执行中任务数
            - ``queue_capacity``: 队列总容量
            - ``estimated_wait_time``: 预估等待时间（秒）
            - ``by_backend``: 按后端分组的队列详情

        Raises:
            TianyanAPIError: 查询失败时抛出
        """
        if self.mock_mode and hasattr(self, "_mock_client") and self._mock_client:
            with self._observe_api_call("get_queue_status", "queue_status"):
                return cast(dict[str, Any], self._mock_client.get_queue_status())

        # 真实模式委托 cqlib
        if self._cqlib is not None:
            with self._observe_api_call("get_queue_status", "queue_status"):
                return self._call_with_retry(self._cqlib.get_queue_status)

        raise TianyanAPIError(
            status_code=500,
            message="未配置有效 API 密钥或 cqlib 客户端，无法获取队列状态",
        )

    # ------------------------------------------------------------------
    # 便捷方法：等待任务完成
    # ------------------------------------------------------------------

    def get_circuit_state(self) -> str:
        """获取熔断器当前状态

        Returns:
            熔断器状态字符串："closed" / "open" / "half_open"
        """
        if self._circuit_breaker is None:
            return "closed"
        return self._circuit_breaker.get_state()

    def get_hourly_quota(self) -> int:
        """获取当前小时窗口内的 API 调用次数

        Returns:
            当前小时窗口内的调用计数
        """
        return self._quota_tracker.hourly_count

    def get_daily_quota(self) -> int:
        """获取当前日窗口内的 API 调用次数

        Returns:
            当前日窗口内的调用计数
        """
        return self._quota_tracker.daily_count

    def get_available_rate_limit_tokens(self) -> float | None:
        """获取令牌桶当前可用令牌数

        Returns:
            可用令牌数；限流未启用时返回 None
        """
        if self._rate_limiter is None:
            return None
        return self._rate_limiter.available_tokens

    def wait_for_task(
        self,
        task_id: str,
        poll_interval: float = 5.0,
        timeout: float = 3600.0,
    ) -> dict[str, Any]:
        """轮询等待任务完成并返回结果

        周期性查询任务状态，直到任务完成或失败，或超过超时时间。

        Args:
            task_id: 任务 ID
            poll_interval: 轮询间隔（秒），默认 5.0
            timeout: 最大等待时间（秒），默认 3600.0（1 小时）

        Returns:
            任务最终结果字典

        Raises:
            TianyanAPIError: 任务失败或超时时抛出
        """
        # 真实模式委托 cqlib（cqlib 内部轮询逻辑）
        if not self.mock_mode and self._cqlib is not None:
            return self._cqlib.wait_for_task(
                task_id, timeout=int(timeout), poll_interval=int(poll_interval)
            )

        elapsed = 0.0
        while elapsed < timeout:
            status_info = self.get_task_status(task_id)
            status = status_info.get("status", "UNKNOWN")

            if status == "COMPLETED":
                logger.info(f"任务 {task_id} 已完成")
                return self.get_task_result(task_id)
            elif status == "FAILED":
                error_msg = status_info.get("error", "未知错误")
                raise TianyanAPIError(
                    status_code=400,
                    message=f"任务 {task_id} 执行失败: {error_msg}",
                    response_body=status_info,
                )

            logger.debug(f"任务 {task_id} 状态={status}，{poll_interval}s 后再次查询")
            time.sleep(poll_interval)
            elapsed += poll_interval

        raise TianyanAPIError(
            status_code=408,
            message=f"任务 {task_id} 等待超时（{timeout}s）",
        )


# ======================================================================
# 模块入口示例
# ======================================================================
if __name__ == "__main__":
    import sys

    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    # 初始化客户端
    client = TianyanClient()

    # 验证 API 密钥
    if not client.authenticate():
        logger.error("认证失败，请检查 TIANYAN_API_KEY 环境变量")
        sys.exit(1)

    logger.info("认证通过")

    # 提交量子任务示例
    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """

    try:
        task_id = client.submit_quantum_task(circuit_qasm=qasm_str, shots=1024)
        logger.info(f"任务提交成功，task_id={task_id}")

        # 等待结果
        result = client.wait_for_task(task_id, poll_interval=3.0, timeout=120.0)
        logger.info(f"任务结果: {result}")

    except TianyanAPIError as e:
        logger.error(f"API 错误: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"网络错误: {e}")
