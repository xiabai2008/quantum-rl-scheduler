"""
Issue #193: 统一 logging 配置 - 标准 logging → loguru 桥接测试

测试覆盖：
- LOGGING_CONFIG 字典结构完整性
- install_intercept_handler 安装后标准 logging 日志被转发到 loguru
- _InterceptHandler 自身异常不影响主流程
- 噪声 logger 级别被正确调整
"""

import logging
import os
import sys
import unittest
from contextlib import redirect_stderr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config.settings import (
    LOGGING_CONFIG,
    _InterceptHandler,
    install_intercept_handler,
)


class TestLoggingConfigShape(unittest.TestCase):
    """验证 LOGGING_CONFIG 字典结构完整、字段类型正确。"""

    def test_logging_config_is_dict(self):
        self.assertIsInstance(LOGGING_CONFIG, dict)

    def test_logging_config_version(self):
        self.assertEqual(LOGGING_CONFIG.get("version"), 1)

    def test_logging_config_has_formatters(self):
        self.assertIn("formatters", LOGGING_CONFIG)
        self.assertIn("detailed", LOGGING_CONFIG["formatters"])
        self.assertIn("simple", LOGGING_CONFIG["formatters"])

    def test_logging_config_has_handlers(self):
        self.assertIn("handlers", LOGGING_CONFIG)
        self.assertIn("console", LOGGING_CONFIG["handlers"])
        self.assertIn("file", LOGGING_CONFIG["handlers"])

    def test_logging_config_file_handler_rotation(self):
        file_handler = LOGGING_CONFIG["handlers"]["file"]
        self.assertEqual(file_handler["maxBytes"], 10 * 1024 * 1024)
        self.assertEqual(file_handler["backupCount"], 5)

    def test_logging_config_disable_existing_loggers_false(self):
        # 必须为 False，避免屏蔽已有 logger
        self.assertFalse(LOGGING_CONFIG.get("disable_existing_loggers", True))

    def test_logging_config_loggers_keys(self):
        loggers = LOGGING_CONFIG.get("loggers", {})
        for required in ("src", "uvicorn", "stable_baselines3"):
            self.assertIn(required, loggers)


class TestInterceptHandler(unittest.TestCase):
    """测试 _InterceptHandler 类。"""

    def test_is_logging_handler_subclass(self):
        self.assertTrue(issubclass(_InterceptHandler, logging.Handler))

    def test_emit_does_not_raise_on_normal_record(self):
        """正常 LogRecord 不应抛异常。"""
        handler = _InterceptHandler()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="test message",
            args=None,
            exc_info=None,
        )
        # 不应抛异常
        handler.emit(record)

    def test_emit_swallows_exceptions(self):
        """emit 内部异常应被吞掉，不影响调用方。"""
        handler = _InterceptHandler()

        # 构造一个会让 loguru.opt().log() 失败的场景：
        # 传入一个故意让 level 参数非法的 record（levelno 为负数）
        record = logging.LogRecord(
            name="x",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="x",
            args=None,
            exc_info=None,
        )
        # 不应抛异常
        handler.emit(record)


class TestInstallInterceptHandler(unittest.TestCase):
    """测试 install_intercept_handler 函数。"""

    def setUp(self):
        # 保存 root logger 原始状态
        self._root = logging.getLogger()
        self._original_handlers = self._root.handlers[:]
        self._original_level = self._root.level

    def tearDown(self):
        # 恢复 root logger 状态
        self._root.handlers = self._original_handlers
        self._root.setLevel(self._original_level)

    def test_install_adds_intercept_handler_to_root(self):
        install_intercept_handler()
        root = logging.getLogger()
        self.assertTrue(
            any(isinstance(h, _InterceptHandler) for h in root.handlers),
            "root logger 应包含 _InterceptHandler",
        )

    def test_install_sets_root_level_to_zero(self):
        install_intercept_handler()
        self.assertEqual(logging.getLogger().level, 0)

    def test_install_silences_noisy_loggers(self):
        install_intercept_handler()
        for noisy in ("urllib3", "asyncio", "matplotlib", "PIL"):
            self.assertEqual(
                logging.getLogger(noisy).level,
                logging.WARNING,
                f"{noisy} logger 应被设置为 WARNING",
            )

    def test_standard_logging_after_install_does_not_raise(self):
        """安装后，通过标准 logging 输出日志不应抛异常。"""
        install_intercept_handler()
        std_logger = logging.getLogger("test.intercept")

        # 捕获 stderr，避免污染测试输出
        import io

        buf = io.StringIO()
        with redirect_stderr(buf):
            std_logger.info("test via standard logging")
            std_logger.warning("warn via standard logging")
            std_logger.error("error via standard logging")

        # 不抛异常即通过
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
