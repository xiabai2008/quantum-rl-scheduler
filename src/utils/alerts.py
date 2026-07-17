"""异常告警与通知集成模块。

监控熔断器状态、退火失败、API错误等异常事件，
支持多种通知渠道（日志、Prometheus 指标、Webhook）。

设计要点
--------
- :class:`AlertManager` 集中接收告警，统一分发到日志、Prometheus、Webhook 三个通道
- 内置滑动窗口速率限制（默认每分钟最多 100 条），防止告警风暴
- 模块级单例 :data:`alert_manager` 与四个便捷函数，便于在各模块一行调用
- Webhook 通道可选：未配置 ``ALERT_WEBHOOK_URL`` 时自动跳过，不影响主流程

典型用法
--------
直接使用便捷函数::

    from src.utils.alerts import alert_error, alert_critical

    alert_error("api", "API 连续失败", endpoint="/v1/tasks")
    alert_critical("circuit_breaker", "熔断器打开", failure_count=5)

或通过单例调用以获取 :class:`Alert` 对象::

    from src.utils.alerts import alert_manager, AlertLevel

    alert = alert_manager.alert(AlertLevel.WARNING, "annealing", "退火未收敛", step=100)
"""

import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

import requests
from loguru import logger
from prometheus_client import Counter

__all__ = [
    "Alert",
    "AlertLevel",
    "AlertManager",
    "alert_critical",
    "alert_error",
    "alert_info",
    "alert_manager",
    "alert_warning",
]


# Prometheus 告警计数器，按级别与类别统计
alerts_total = Counter(
    "scheduler_alerts_total",
    "Total alerts dispatched",
    ["level", "category"],
)


class AlertLevel(Enum):
    """告警级别枚举

    Attributes:
        INFO: 信息级告警，用于记录一般性事件。
        WARNING: 警告级告警，表示潜在问题，需关注。
        ERROR: 错误级告警，表示已发生的故障，需处理。
        CRITICAL: 严重告警，表示影响系统可用性的重大故障，需立即处理。
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """告警数据对象

    Attributes:
        level: 告警级别（:class:`AlertLevel`）。
        category: 告警类别（如 ``"circuit_breaker"``、``"annealing"``、``"api"``）。
        message: 告警描述信息。
        timestamp: 告警产生时间戳（``time.time()``，Unix 秒）。
        context: 告警附加上下文（键值对），便于后续诊断与统计。
    """

    level: AlertLevel
    category: str
    message: str
    timestamp: float = field(default_factory=time.time)
    context: dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """告警管理器

    集中接收、记录与分发异常告警，支持日志、Prometheus 指标与 Webhook 三个通道。
    内置滑动窗口速率限制（默认每分钟最多 100 条），防止告警风暴淹没下游通道。

    Args:
        webhook_url: Webhook 通知地址，若为 ``None`` 则尝试从环境变量
            ``ALERT_WEBHOOK_URL`` 读取；两者均未设置时禁用 Webhook 通道。
        max_alerts_per_minute: 每分钟最多分发的告警数量，默认 100。
            超过该阈值的告警将被丢弃并记录一条 warning 日志。

    Note:
        Webhook 发送失败（网络异常或非 2xx 响应）仅记录 warning 日志，
        不会抛出异常，避免影响调用方主流程。
    """

    # 各告警级别对应的 loguru 日志级别名称
    _LEVEL_TO_LOGURU: ClassVar[dict[AlertLevel, str]] = {
        AlertLevel.INFO: "INFO",
        AlertLevel.WARNING: "WARNING",
        AlertLevel.ERROR: "ERROR",
        AlertLevel.CRITICAL: "CRITICAL",
    }

    def __init__(
        self,
        webhook_url: str | None = None,
        max_alerts_per_minute: int = 100,
    ) -> None:
        """初始化告警管理器

        Args:
            webhook_url: Webhook 通知地址，``None`` 时从环境变量读取。
            max_alerts_per_minute: 每分钟最多分发的告警数量。
        """
        self.webhook_url: str | None = (
            webhook_url if webhook_url is not None else os.getenv("ALERT_WEBHOOK_URL")
        )
        self.max_alerts_per_minute: int = int(max_alerts_per_minute)
        # 滑动窗口速率限制：保留最近一分钟内的告警时间戳
        self._recent_timestamps: deque[float] = deque()
        # 已分发的告警列表（用于测试与诊断）
        self._alerts: list[Alert] = []

    def alert(
        self,
        level: AlertLevel,
        category: str,
        message: str,
        **context: Any,
    ) -> Alert | None:
        """记录并分发一条告警

        依次执行：速率限制检查 → 记录到内部列表 → 日志输出 →
        Prometheus 指标更新 → Webhook 发送（若已配置）。

        Args:
            level: 告警级别。
            category: 告警类别（如 ``"circuit_breaker"``）。
            message: 告警描述信息。
            **context: 告警附加上下文键值对。

        Returns:
            已分发的 :class:`Alert` 对象；若因速率限制被丢弃则返回 ``None``。
        """
        # 速率限制：清理 60 秒外的时间戳，判断窗口内是否已达上限
        now = time.time()
        while self._recent_timestamps and now - self._recent_timestamps[0] > 60.0:
            self._recent_timestamps.popleft()
        if len(self._recent_timestamps) >= self.max_alerts_per_minute:
            logger.warning(f"告警速率限制触发，丢弃告警: [{level.value}] {category}: {message}")
            return None
        self._recent_timestamps.append(now)

        alert = Alert(
            level=level,
            category=category,
            message=message,
            timestamp=now,
            context=dict(context),
        )

        self._alerts.append(alert)
        self._log_alert(alert)
        self._record_metric(alert)
        self._send_webhook(alert)
        return alert

    def _log_alert(self, alert: Alert) -> None:
        """按告警级别使用 loguru 输出日志

        Args:
            alert: 待记录的告警对象。
        """
        log_level = self._LEVEL_TO_LOGURU.get(alert.level, "INFO")
        ctx_str = ", ".join(f"{k}={v}" for k, v in alert.context.items())
        msg = f"[{alert.category}] {alert.message}"
        if ctx_str:
            msg += f" ({ctx_str})"
        logger.log(log_level, msg)

    def _record_metric(self, alert: Alert) -> None:
        """更新 Prometheus 告警计数器

        Args:
            alert: 待记录的告警对象。
        """
        alerts_total.labels(level=alert.level.value, category=alert.category).inc()

    def _send_webhook(self, alert: Alert) -> None:
        """发送告警到 Webhook（若已配置）

        使用 POST JSON 请求；网络异常或非 2xx 响应仅记录 warning 日志，不抛出。

        Args:
            alert: 待发送的告警对象。
        """
        if not self.webhook_url:
            return
        payload = {
            "level": alert.level.value,
            "category": alert.category,
            "message": alert.message,
            "timestamp": alert.timestamp,
            "context": alert.context,
        }
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=5.0)
            if response.status_code >= 400:
                logger.warning(f"Webhook 返回非 2xx 状态码: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Webhook 发送失败: {type(e).__name__}: {e}")

    def get_alerts(self) -> list[Alert]:
        """返回已记录的告警列表（拷贝，避免外部修改内部状态）"""
        return list(self._alerts)

    def clear(self) -> None:
        """清空已记录的告警与速率限制窗口

        主要用于测试场景下的状态隔离。
        """
        self._alerts.clear()
        self._recent_timestamps.clear()


# 模块级单例，供全局便捷访问
alert_manager = AlertManager()


def alert_info(category: str, message: str, **context: Any) -> Alert | None:
    """记录一条 INFO 级别告警

    Args:
        category: 告警类别。
        message: 告警描述信息。
        **context: 告警附加上下文键值对。

    Returns:
        已分发的 :class:`Alert` 对象；若因速率限制被丢弃则返回 ``None``。
    """
    return alert_manager.alert(AlertLevel.INFO, category, message, **context)


def alert_warning(category: str, message: str, **context: Any) -> Alert | None:
    """记录一条 WARNING 级别告警

    Args:
        category: 告警类别。
        message: 告警描述信息。
        **context: 告警附加上下文键值对。

    Returns:
        已分发的 :class:`Alert` 对象；若因速率限制被丢弃则返回 ``None``。
    """
    return alert_manager.alert(AlertLevel.WARNING, category, message, **context)


def alert_error(category: str, message: str, **context: Any) -> Alert | None:
    """记录一条 ERROR 级别告警

    Args:
        category: 告警类别。
        message: 告警描述信息。
        **context: 告警附加上下文键值对。

    Returns:
        已分发的 :class:`Alert` 对象；若因速率限制被丢弃则返回 ``None``。
    """
    return alert_manager.alert(AlertLevel.ERROR, category, message, **context)


def alert_critical(category: str, message: str, **context: Any) -> Alert | None:
    """记录一条 CRITICAL 级别告警

    Args:
        category: 告警类别。
        message: 告警描述信息。
        **context: 告警附加上下文键值对。

    Returns:
        已分发的 :class:`Alert` 对象；若因速率限制被丢弃则返回 ``None``。
    """
    return alert_manager.alert(AlertLevel.CRITICAL, category, message, **context)
