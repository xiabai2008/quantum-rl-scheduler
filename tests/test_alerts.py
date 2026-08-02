"""
量子RL调度系统 - 异常告警模块单元测试
Unit Tests for src/utils/alerts.py

测试覆盖：
- AlertManager 告警记录（alert 被正确保存到内部列表）
- 各告警级别对应 loguru 日志级别
- 速率限制防止告警风暴（max_alerts_per_minute 阈值生效）
- Webhook 未配置时不发送（requests.post 不被调用）
- 便捷函数 alert_info/warning/error/critical 使用正确级别
"""

import os
import sys
import threading
import unittest
from typing import Any
from unittest.mock import patch

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loguru import logger

from src.utils.alerts import (
    Alert,
    AlertLevel,
    AlertManager,
    alert_critical,
    alert_error,
    alert_info,
    alert_manager,
    alert_warning,
)


class TestAlertManager(unittest.TestCase):
    """测试 AlertManager 核心功能。"""

    def setUp(self):
        """每个测试前清空模块级单例，避免跨用例污染。"""
        alert_manager.clear()

    def test_alert_manager_records_alert(self):
        """alert() 应将告警对象记录到内部列表并返回该对象。"""
        manager = AlertManager()
        result = manager.alert(AlertLevel.WARNING, "circuit_breaker", "熔断器打开", failure_count=5)
        self.assertIsNotNone(result)
        alerts = manager.get_alerts()
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].level, AlertLevel.WARNING)
        self.assertEqual(alerts[0].category, "circuit_breaker")
        self.assertEqual(alerts[0].message, "熔断器打开")
        self.assertEqual(alerts[0].context["failure_count"], 5)
        # 返回的对象应与记录的对象一致
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "circuit_breaker")

    def test_alert_level_logging(self):
        """每个告警级别应使用对应的 loguru 日志级别输出。"""
        captured_levels: list[str] = []

        def sink(message: Any) -> None:
            captured_levels.append(message.record["level"].name)

        sink_id = logger.add(sink, level="DEBUG")
        try:
            manager = AlertManager()
            manager._log_alert(Alert(AlertLevel.INFO, "cat", "info msg"))
            manager._log_alert(Alert(AlertLevel.WARNING, "cat", "warn msg"))
            manager._log_alert(Alert(AlertLevel.ERROR, "cat", "err msg"))
            manager._log_alert(Alert(AlertLevel.CRITICAL, "cat", "crit msg"))
        finally:
            logger.remove(sink_id)

        self.assertIn("INFO", captured_levels)
        self.assertIn("WARNING", captured_levels)
        self.assertIn("ERROR", captured_levels)
        self.assertIn("CRITICAL", captured_levels)

    def test_alert_rate_limiting(self):
        """速率限制应丢弃超过阈值的告警。"""
        manager = AlertManager(max_alerts_per_minute=3)
        results: list[Alert | None] = []
        for i in range(5):
            results.append(manager.alert(AlertLevel.INFO, "test", f"alert {i}"))

        # 前 3 条被接受，后 2 条因速率限制被丢弃
        accepted = [r for r in results if r is not None]
        self.assertEqual(len(accepted), 3)
        self.assertEqual(len(manager.get_alerts()), 3)
        # 被丢弃的告警返回 None
        self.assertIsNone(results[3])
        self.assertIsNone(results[4])

    def test_alert_webhook_disabled(self):
        """未配置 webhook_url 时不应调用 requests.post。"""
        manager = AlertManager()
        # 强制禁用 webhook，避免环境变量干扰
        manager.webhook_url = None
        with patch("src.utils.alerts.requests.post") as mock_post:
            manager.alert(AlertLevel.ERROR, "api", "API 调用失败")
        mock_post.assert_not_called()

    def test_alert_convenience_functions(self):
        """便捷函数应使用正确级别记录告警到模块级单例。"""
        alert_manager.clear()
        alert_info("test", "info msg")
        alert_warning("test", "warning msg")
        alert_error("test", "error msg")
        alert_critical("test", "critical msg")

        alerts = alert_manager.get_alerts()
        self.assertEqual(len(alerts), 4)
        self.assertEqual(alerts[0].level, AlertLevel.INFO)
        self.assertEqual(alerts[1].level, AlertLevel.WARNING)
        self.assertEqual(alerts[2].level, AlertLevel.ERROR)
        self.assertEqual(alerts[3].level, AlertLevel.CRITICAL)


class TestAlertManagerWebhookRetry(unittest.TestCase):
    """Issue #878: Webhook 指数退避重试逻辑测试。

    覆盖 ERROR/CRITICAL 级别重试 3 次、其他级别仅 1 次，
    以及非 2xx / 网络异常时的最终失败行为。
    """

    def _make_manager(self, url: str = "http://example.com/hook") -> AlertManager:
        return AlertManager(webhook_url=url, max_alerts_per_minute=10000)

    def test_error_retries_three_times_on_http_error(self):
        """ERROR 级别 + 连续 5xx 响应应重试 3 次。"""
        manager = self._make_manager()
        with (
            patch("src.utils.alerts.requests.post") as mock_post,
            patch("src.utils.alerts.time.sleep") as mock_sleep,
        ):
            mock_post.return_value.status_code = 500
            manager._send_webhook(Alert(AlertLevel.ERROR, "api", "boom"))
        self.assertEqual(mock_post.call_count, 3)
        mock_sleep.assert_any_call(0.5)
        mock_sleep.assert_any_call(1.0)

    def test_critical_retries_three_times(self):
        """CRITICAL 级别 + 网络异常应重试 3 次。"""
        manager = self._make_manager()
        with (
            patch(
                "src.utils.alerts.requests.post",
                side_effect=requests.exceptions.ConnectionError("refused"),
            ) as mock_post,
            patch("src.utils.alerts.time.sleep"),
        ):
            manager._send_webhook(Alert(AlertLevel.CRITICAL, "cb", "open"))
        self.assertEqual(mock_post.call_count, 3)

    def test_warning_does_not_retry(self):
        """WARNING 级别 + 5xx 响应应仅尝试 1 次（不重试）。"""
        manager = self._make_manager()
        with (
            patch("src.utils.alerts.requests.post") as mock_post,
            patch("src.utils.alerts.time.sleep") as mock_sleep,
        ):
            mock_post.return_value.status_code = 500
            manager._send_webhook(Alert(AlertLevel.WARNING, "api", "warn"))
        self.assertEqual(mock_post.call_count, 1)
        mock_sleep.assert_not_called()

    def test_success_on_first_attempt_no_retry(self):
        """首次 2xx 成功应立即返回，不重试。"""
        manager = self._make_manager()
        with (
            patch("src.utils.alerts.requests.post") as mock_post,
            patch("src.utils.alerts.time.sleep") as mock_sleep,
        ):
            mock_post.return_value.status_code = 200
            manager._send_webhook(Alert(AlertLevel.CRITICAL, "cb", "ok"))
        self.assertEqual(mock_post.call_count, 1)
        mock_sleep.assert_not_called()


class TestAlertManagerMemoryLimit(unittest.TestCase):
    """Issue #878/#706: _alerts 内存上限测试。"""

    def test_alerts_list_capped_at_max(self):
        """超过 1000 条告警时只保留最近 1000 条。"""
        manager = AlertManager(max_alerts_per_minute=10000)
        for i in range(1005):
            manager.alert(AlertLevel.INFO, "test", f"alert {i}")
        alerts = manager.get_alerts()
        self.assertEqual(len(alerts), 1000)
        # 保留的是最近的 1000 条（编号 5..1004）
        self.assertEqual(alerts[0].message, "alert 5")
        self.assertEqual(alerts[-1].message, "alert 1004")


class TestAlertManagerThreadSafety(unittest.TestCase):
    """Issue #878: 多线程并发 alert() 的竞态测试。"""

    def test_concurrent_alerts_no_loss(self):
        """200 个线程并发告警，应全部记录无丢失、无异常。"""
        manager = AlertManager(max_alerts_per_minute=10000)
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                manager.alert(AlertLevel.INFO, "test", f"alert {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(200)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(manager.get_alerts()), 200)


class TestAlertManagerGetAlertsIsolation(unittest.TestCase):
    """Issue #878: get_alerts() 返回深拷贝，外部修改不影响内部状态。"""

    def test_external_context_modification_is_isolated(self):
        """修改返回 Alert 的 context 字典不应污染内部状态。"""
        manager = AlertManager(max_alerts_per_minute=10000)
        manager.alert(AlertLevel.ERROR, "api", "boom", job_id=1)

        returned = manager.get_alerts()
        returned[0].context["job_id"] = 999
        returned[0].message = "hacked"

        internal = manager.get_alerts()
        self.assertEqual(internal[0].context["job_id"], 1)
        self.assertEqual(internal[0].message, "boom")

    def test_external_list_modification_is_isolated(self):
        """修改返回的列表本身不应影响内部记录。"""
        manager = AlertManager(max_alerts_per_minute=10000)
        manager.alert(AlertLevel.WARNING, "test", "warn")

        returned = manager.get_alerts()
        returned.clear()

        self.assertEqual(len(manager.get_alerts()), 1)


if __name__ == "__main__":
    unittest.main()
