"""
量子RL调度系统 - 结构化日志（JSON 格式）单元测试
Unit Tests for JSON logging support in src/utils/helpers.py

测试覆盖：
- setup_logging 文本格式（默认，LOG_FORMAT 未设置）
- setup_logging JSON 格式（LOG_FORMAT=json）
- _json_serializer 生成有效 JSON 且包含必需字段
- _json_serializer 保留中文字符（ensure_ascii=False）
- _json_serializer 合并 extra 上下文且不覆盖内置字段
- LOG_FORMAT 大小写不敏感 / 未知值回退到文本
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.helpers import _json_serializer, setup_logging


def _make_mock_record(
    message: str = "测试消息",
    level: str = "INFO",
    module: str = "test_module",
    function: str = "test_func",
    line: int = 42,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造模拟 loguru record 字典用于测试 _json_serializer。"""
    return {
        "time": datetime(2026, 7, 2, 10, 0, 0),
        "level": SimpleNamespace(name=level),
        "module": module,
        "function": function,
        "line": line,
        "message": message,
        "extra": extra or {},
    }


# ============================================================
# _json_serializer 测试
# ============================================================
class TestJsonSerializer(unittest.TestCase):
    """测试 _json_serializer 函数。"""

    def test_json_serializer(self):
        """_json_serializer 应生成包含所有必需字段的有效 JSON。"""
        record = _make_mock_record(message="hello", level="INFO")
        output = _json_serializer(record)

        # 应为有效 JSON
        data = json.loads(output)
        self.assertIsInstance(data, dict)

        # 必需字段全部存在
        required_fields = {"timestamp", "level", "module", "function", "line", "message"}
        self.assertTrue(required_fields.issubset(data.keys()))

        # 字段值正确
        self.assertEqual(data["message"], "hello")
        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["module"], "test_module")
        self.assertEqual(data["function"], "test_func")
        self.assertEqual(data["line"], 42)
        self.assertEqual(data["timestamp"], "2026-07-02T10:00:00")

    def test_json_serializer_chinese(self):
        """_json_serializer 应保留中文字符（ensure_ascii=False）。"""
        record = _make_mock_record(message="量子RL调度系统启动")
        output = _json_serializer(record)

        # 中文字符应直接出现在输出中（而非 \uXXXX 转义）
        self.assertIn("量子RL调度系统启动", output)

        # 解析后消息应正确
        data = json.loads(output)
        self.assertEqual(data["message"], "量子RL调度系统启动")

    def test_json_serializer_includes_extra_fields(self):
        """_json_serializer 应合并 record['extra'] 中的自定义字段。"""
        record = _make_mock_record(
            message="task scheduled",
            extra={"task_id": "T-001", "strategy": "ppo", "qubits": 287},
        )
        output = _json_serializer(record)
        data = json.loads(output)

        self.assertEqual(data["task_id"], "T-001")
        self.assertEqual(data["strategy"], "ppo")
        self.assertEqual(data["qubits"], 287)

    def test_json_serializer_extra_does_not_override_builtin(self):
        """extra 中的字段不应覆盖内置字段（如 message、level）。"""
        record = _make_mock_record(
            message="original",
            extra={"message": "hijacked", "level": "HACKED"},
        )
        output = _json_serializer(record)
        data = json.loads(output)

        self.assertEqual(data["message"], "original")
        self.assertEqual(data["level"], "INFO")

    def test_json_serializer_timestamp_is_iso_format(self):
        """timestamp 字段应为 ISO 8601 格式。"""
        record = _make_mock_record()
        output = _json_serializer(record)
        data = json.loads(output)

        # 应可被 fromisoformat 解析
        parsed = datetime.fromisoformat(data["timestamp"])
        self.assertIsInstance(parsed, datetime)

    def test_json_serializer_handles_non_serializable_extra(self):
        """extra 中的不可序列化对象应用 str() 兜底（default=str）。"""
        # datetime 是 JSON 不可序列化的，应用 str() 转换
        custom_dt = datetime(2026, 1, 1, 12, 0, 0)
        record = _make_mock_record(extra={"custom_time": custom_dt})
        output = _json_serializer(record)

        # 不应抛异常，且应包含该字段
        data = json.loads(output)
        self.assertIn("custom_time", data)

    def test_json_serializer_empty_extra(self):
        """extra 为空字典时应正常输出（仅含内置字段）。"""
        record = _make_mock_record(extra={})
        output = _json_serializer(record)
        data = json.loads(output)

        self.assertEqual(len(data), 6)  # 仅 6 个内置字段


# ============================================================
# setup_logging 文本/JSON 格式切换测试
# ============================================================
class TestSetupLoggingFormat(unittest.TestCase):
    """测试 setup_logging 的文本/JSON 格式切换。"""

    def _clear_log_format(self) -> None:
        """清理 LOG_FORMAT 环境变量。"""
        os.environ.pop("LOG_FORMAT", None)

    def setUp(self) -> None:
        self._clear_log_format()

    def tearDown(self) -> None:
        self._clear_log_format()

    def test_setup_logging_text_format(self):
        """默认（不设 LOG_FORMAT）应为文本格式。"""
        with tempfile.TemporaryDirectory() as tmp:
            lg = setup_logging(log_dir=tmp, log_level="INFO")
            try:
                self.assertIsNotNone(lg)
                # 捕获 stderr 验证输出为文本（非 JSON）
                buf = io.StringIO()
                with redirect_stderr(buf):
                    lg.info("text mode message")
                output = buf.getvalue().strip()

                # 文本格式输出应包含 | 分隔符和消息内容
                self.assertIn("|", output)
                self.assertIn("text mode message", output)
                # 不应能解析为 JSON
                with self.assertRaises(json.JSONDecodeError):
                    json.loads(output)
            finally:
                lg.remove()

    def test_setup_logging_json_format(self):
        """LOG_FORMAT=json 时日志应为 JSON 格式。"""
        os.environ["LOG_FORMAT"] = "json"
        with tempfile.TemporaryDirectory() as tmp:
            lg = setup_logging(log_dir=tmp, log_level="INFO")
            try:
                self.assertIsNotNone(lg)
                buf = io.StringIO()
                with redirect_stdout(buf):
                    lg.info("json mode message")
                output = buf.getvalue().strip()

                # 应为有效 JSON
                data = json.loads(output)
                self.assertEqual(data["message"], "json mode message")
                self.assertEqual(data["level"], "INFO")
            finally:
                lg.remove()

    def test_setup_logging_json_format_writes_file(self):
        """LOG_FORMAT=json 时应写入 JSON 格式的日志文件。"""
        os.environ["LOG_FORMAT"] = "json"
        with tempfile.TemporaryDirectory() as tmp:
            log_file = "test_json.log"
            lg = setup_logging(log_dir=tmp, log_file=log_file, log_level="INFO")
            try:
                # 重定向 stdout 避免控制台输出干扰测试
                with (
                    open(os.devnull, "w", encoding="utf-8") as devnull,
                    redirect_stdout(devnull),
                ):
                    lg.info("file json message")

                file_path = os.path.join(tmp, log_file)
                self.assertTrue(os.path.exists(file_path))

                with open(file_path, encoding="utf-8") as f:
                    content = f.read().strip()

                # 文件内容应为有效 JSON
                data = json.loads(content)
                self.assertEqual(data["message"], "file json message")
            finally:
                lg.remove()

    def test_setup_logging_json_format_case_insensitive(self):
        """LOG_FORMAT=JSON（大写）也应生效。"""
        os.environ["LOG_FORMAT"] = "JSON"
        with tempfile.TemporaryDirectory() as tmp:
            lg = setup_logging(log_dir=tmp, log_level="INFO")
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    lg.info("upper case test")
                output = buf.getvalue().strip()

                data = json.loads(output)
                self.assertEqual(data["message"], "upper case test")
            finally:
                lg.remove()

    def test_setup_logging_unknown_format_defaults_to_text(self):
        """LOG_FORMAT 为非 json 值时应回退到文本格式。"""
        os.environ["LOG_FORMAT"] = "xml"
        with tempfile.TemporaryDirectory() as tmp:
            lg = setup_logging(log_dir=tmp, log_level="INFO")
            try:
                buf = io.StringIO()
                with redirect_stderr(buf):
                    lg.info("fallback text")
                output = buf.getvalue().strip()

                # 文本格式包含 | 分隔符
                self.assertIn("|", output)
                self.assertIn("fallback text", output)
            finally:
                lg.remove()

    def test_setup_logging_returns_logger_in_both_modes(self):
        """两种模式下 setup_logging 都应返回有效 logger 且创建目录。"""
        for fmt in ("text", "json"):
            with self.subTest(format=fmt):
                os.environ["LOG_FORMAT"] = fmt
                with tempfile.TemporaryDirectory() as tmp:
                    log_dir = os.path.join(tmp, "subdir")
                    lg = setup_logging(log_dir=log_dir, log_file="x.log")
                    try:
                        self.assertIsNotNone(lg)
                        self.assertTrue(os.path.isdir(log_dir))
                    finally:
                        lg.remove()


if __name__ == "__main__":
    unittest.main()
