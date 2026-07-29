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
import unittest
from typing import Any
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
