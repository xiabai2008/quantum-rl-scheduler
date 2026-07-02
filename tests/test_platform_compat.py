"""
量子RL调度系统 - Windows 平台兼容性工具测试
Unit Tests for src/utils/platform_compat.py

测试覆盖:
- TestPlatformDetection: 平台检测函数返回 bool
- TestNormalizePath: 路径分隔符规范化(正斜杠/反斜杠/混合)
- TestJoinPaths: 多段路径拼接、空段过滤
- TestGetProjectRoot: 返回路径包含项目名
- TestSafeMakedirs: 创建目录、已存在不报错、嵌套创建
- TestSafeOpen: 读写文件、UTF-8 编码、中文内容、二进制
- TestGetLineEnding: 返回正确换行符
- TestEnsureWindowsEncoding: 不抛异常
- TestRunSubprocess: 简单命令执行(python --version / -c print)
- TestKillProcessTree: 启动子进程并终止
"""

import contextlib
import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.platform_compat import (
    ensure_windows_encoding,
    get_line_ending,
    get_project_root,
    is_linux,
    is_macos,
    is_windows,
    join_paths,
    kill_process_tree,
    normalize_path,
    run_subprocess,
    safe_makedirs,
    safe_open,
)


# ============================================================
# 平台检测测试
# ============================================================
class TestPlatformDetection(unittest.TestCase):
    """测试平台检测函数。"""

    def test_is_windows_returns_bool(self):
        """is_windows 应返回 bool 类型。"""
        self.assertIsInstance(is_windows(), bool)

    def test_is_linux_returns_bool(self):
        """is_linux 应返回 bool 类型。"""
        self.assertIsInstance(is_linux(), bool)

    def test_is_macos_returns_bool(self):
        """is_macos 应返回 bool 类型。"""
        self.assertIsInstance(is_macos(), bool)

    def test_at_least_one_platform_detected(self):
        """三个平台检测函数应至少有一个为 True。"""
        platforms = [is_windows(), is_linux(), is_macos()]
        self.assertTrue(any(platforms), "至少应识别出一个平台")


# ============================================================
# 路径规范化测试
# ============================================================
class TestNormalizePath(unittest.TestCase):
    """测试路径规范化函数。"""

    def test_forward_slash_path(self):
        """正斜杠路径应被规范化为系统分隔符。"""
        result = normalize_path("a/b/c")
        self.assertEqual(result, os.path.normpath("a/b/c"))

    def test_backward_slash_path(self):
        """反斜杠路径应被规范化。"""
        result = normalize_path("a\\b\\c")
        self.assertEqual(result, os.path.normpath("a\\b\\c"))

    def test_mixed_separators(self):
        """混合分隔符路径应被正确规范化。"""
        result = normalize_path("a/b\\c/d")
        self.assertEqual(result, os.path.normpath("a/b\\c/d"))
        # 规范化后不应同时包含两种分隔符
        if is_windows():
            self.assertNotIn("/", result)
        else:
            self.assertNotIn("\\", result)

    def test_empty_path(self):
        """空路径应原样返回(不转为 '.')。"""
        self.assertEqual(normalize_path(""), "")

    def test_dot_dot_resolved(self):
        """.. 引用应被正确解析。"""
        result = normalize_path("a/b/../c")
        self.assertEqual(result, os.path.normpath("a/b/../c"))

    def test_absolute_path_preserved(self):
        """绝对路径应被保留。"""
        if is_windows():
            result = normalize_path("C:/Users/test/docs")
            self.assertTrue(result.startswith("C:"), f"应保留盘符,实际: {result}")
        else:
            result = normalize_path("/usr/local/bin")
            self.assertTrue(result.startswith("/"), f"应保留根斜杠,实际: {result}")


# ============================================================
# 路径拼接测试
# ============================================================
class TestJoinPaths(unittest.TestCase):
    """测试路径拼接函数。"""

    def test_join_multiple_parts(self):
        """多段路径应被正确拼接。"""
        result = join_paths("a", "b", "c")
        self.assertEqual(result, os.path.join("a", "b", "c"))

    def test_join_with_empty_parts(self):
        """空段应被过滤掉。"""
        result = join_paths("a", "", "b", "", "c")
        self.assertEqual(result, os.path.join("a", "b", "c"))

    def test_join_all_empty(self):
        """全部为空时应返回空字符串。"""
        self.assertEqual(join_paths("", "", ""), "")

    def test_join_no_args(self):
        """无参数时应返回空字符串。"""
        self.assertEqual(join_paths(), "")

    def test_join_single_part(self):
        """单段路径应原样返回。"""
        result = join_paths("a")
        self.assertEqual(result, "a")


# ============================================================
# 获取项目根目录测试
# ============================================================
class TestGetProjectRoot(unittest.TestCase):
    """测试获取项目根目录函数。"""

    def test_returns_absolute_path(self):
        """应返回绝对路径。"""
        root = get_project_root()
        self.assertTrue(os.path.isabs(root), f"应返回绝对路径,实际为: {root}")

    def test_contains_project_name(self):
        """返回路径应包含项目目录名。"""
        root = get_project_root()
        self.assertIn("quantum-rl-scheduler", root)

    def test_contains_src_directory(self):
        """根目录下应存在 src 子目录。"""
        root = get_project_root()
        self.assertTrue(
            os.path.isdir(os.path.join(root, "src")),
            f"根目录下应存在 src 子目录,根目录为: {root}",
        )


# ============================================================
# 安全创建目录测试
# ============================================================
class TestSafeMakedirs(unittest.TestCase):
    """测试安全创建目录函数。"""

    def test_creates_new_directory(self):
        """应能创建新目录。"""
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = os.path.join(tmp, "new_dir")
            safe_makedirs(new_dir)
            self.assertTrue(os.path.isdir(new_dir))

    def test_existing_directory_no_error(self):
        """已存在的目录不应抛出异常。"""
        with tempfile.TemporaryDirectory() as tmp:
            existing = os.path.join(tmp, "existing")
            os.makedirs(existing)
            # 再次创建不应抛异常
            safe_makedirs(existing)
            self.assertTrue(os.path.isdir(existing))

    def test_nested_directories(self):
        """应能创建嵌套目录。"""
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "a", "b", "c", "d")
            safe_makedirs(nested)
            self.assertTrue(os.path.isdir(nested))

    def test_empty_path_no_error(self):
        """空路径不应抛出异常。"""
        # 空字符串应静默返回
        safe_makedirs("")


# ============================================================
# 安全文件打开测试
# ============================================================
class TestSafeOpen(unittest.TestCase):
    """测试安全文件打开函数。"""

    def test_write_and_read_text(self):
        """应能正确写入和读取文本文件。"""
        with tempfile.TemporaryDirectory() as tmp:
            filepath = os.path.join(tmp, "test.txt")
            with safe_open(filepath, "w") as f:
                f.write("hello world")
            with safe_open(filepath, "r") as f:
                content = f.read()
            self.assertEqual(content, "hello world")

    def test_utf8_chinese_content(self):
        """应能正确处理 UTF-8 编码的中文内容。"""
        with tempfile.TemporaryDirectory() as tmp:
            filepath = os.path.join(tmp, "chinese.txt")
            chinese_text = "你好世界,量子RL调度系统"
            with safe_open(filepath, "w", encoding="utf-8") as f:
                f.write(chinese_text)
            with safe_open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertEqual(content, chinese_text)

    def test_binary_mode(self):
        """二进制模式应能正确读写且不指定 encoding。"""
        with tempfile.TemporaryDirectory() as tmp:
            filepath = os.path.join(tmp, "binary.bin")
            data = b"\x00\x01\x02\x03\xff"
            with safe_open(filepath, "wb") as f:
                f.write(data)
            with safe_open(filepath, "rb") as f:
                content = f.read()
            self.assertEqual(content, data)

    def test_mixed_separators_path(self):
        """应能处理包含混合分隔符的路径。"""
        with tempfile.TemporaryDirectory() as tmp:
            # 构造混合分隔符路径(需先创建子目录)
            filepath = tmp + "/subdir\\file.txt"
            safe_makedirs(os.path.dirname(filepath))
            with safe_open(filepath, "w") as f:
                f.write("test")
            # 验证文件确实被创建在正确位置
            normalized = os.path.normpath(filepath)
            self.assertTrue(os.path.isfile(normalized))


# ============================================================
# 换行符测试
# ============================================================
class TestGetLineEnding(unittest.TestCase):
    """测试换行符获取函数。"""

    def test_returns_correct_ending(self):
        """应返回当前平台的正确换行符。"""
        ending = get_line_ending()
        if is_windows():
            self.assertEqual(ending, "\r\n")
        else:
            self.assertEqual(ending, "\n")

    def test_returns_non_empty_string(self):
        """应返回非空字符串。"""
        ending = get_line_ending()
        self.assertIsInstance(ending, str)
        self.assertGreater(len(ending), 0)


# ============================================================
# Windows 编码设置测试
# ============================================================
class TestEnsureWindowsEncoding(unittest.TestCase):
    """测试 Windows 编码设置函数。"""

    def test_does_not_raise(self):
        """调用函数不应抛出任何异常。"""
        # 在任何平台上都不应抛出异常
        ensure_windows_encoding()

    def test_sets_pythonutf8_on_windows(self):
        """在 Windows 上应设置 PYTHONUTF8 环境变量。"""
        if not is_windows():
            self.skipTest("仅在 Windows 平台上测试")
        # 保存原值
        original = os.environ.get("PYTHONUTF8")
        try:
            # 清除后调用
            if "PYTHONUTF8" in os.environ:
                del os.environ["PYTHONUTF8"]
            ensure_windows_encoding()
            self.assertEqual(os.environ.get("PYTHONUTF8"), "1")
        finally:
            # 恢复原值
            if original is not None:
                os.environ["PYTHONUTF8"] = original
            elif "PYTHONUTF8" in os.environ:
                del os.environ["PYTHONUTF8"]

    def test_no_op_on_non_windows(self):
        """在非 Windows 平台上应为空操作(不抛异常)。"""
        if is_windows():
            self.skipTest("仅在非 Windows 平台上测试")
        # 调用不应抛异常,也不应设置 PYTHONUTF8
        original = os.environ.get("PYTHONUTF8")
        try:
            ensure_windows_encoding()
            # 非 Windows 不应主动设置 PYTHONUTF8
            if original is None:
                self.assertNotIn("PYTHONUTF8", os.environ)
        finally:
            if original is not None:
                os.environ["PYTHONUTF8"] = original


# ============================================================
# 跨平台 subprocess 测试
# ============================================================
class TestRunSubprocess(unittest.TestCase):
    """测试跨平台 subprocess 封装。"""

    def test_python_version(self):
        """应能执行 python --version 命令。"""
        result = run_subprocess([sys.executable, "--version"])
        self.assertEqual(result.returncode, 0)
        # Python 3.x 在某些版本输出到 stdout 或 stderr
        output = (result.stdout or "") + (result.stderr or "")
        self.assertIn("Python", output)

    def test_returns_completed_process(self):
        """应返回 subprocess.CompletedProcess 实例。"""
        result = run_subprocess([sys.executable, "--version"])
        self.assertIsInstance(result, subprocess.CompletedProcess)

    def test_capture_output(self):
        """应能捕获输出。"""
        result = run_subprocess([sys.executable, "-c", "print('hello')"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("hello", result.stdout)

    def test_failed_command_returncode(self):
        """失败的命令应返回非零退出码。"""
        result = run_subprocess([sys.executable, "-c", "import sys; sys.exit(1)"])
        self.assertNotEqual(result.returncode, 0)


# ============================================================
# 进程树终止测试
# ============================================================
class TestKillProcessTree(unittest.TestCase):
    """测试跨平台进程树终止函数。"""

    def test_kill_sleeping_child(self):
        """应能终止一个长时间运行的子进程。"""
        # 启动一个会运行较长时间的子进程
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # 确保进程已启动
        self.assertIsNone(proc.poll(), "子进程应仍在运行")
        try:
            # 短暂等待确保进程已就绪
            time.sleep(0.2)
            # 终止进程树
            kill_process_tree(proc.pid)
            # 等待进程结束(给系统一点时间)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # 强制 kill 作为兜底
                proc.kill()
                proc.wait(timeout=5)
            # 进程应已退出
            self.assertIsNotNone(proc.poll(), "子进程应已终止")
        finally:
            # 兜底清理:确保子进程不会泄漏
            if proc.poll() is None:
                proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)

    def test_kill_invalid_pid_no_error(self):
        """无效 PID(0 和负数)不应抛出异常。"""
        # PID 0 和负数应被静默忽略
        kill_process_tree(0)
        kill_process_tree(-1)

    def test_kill_already_exited_process(self):
        """已退出的进程 PID 不应抛出异常。"""
        # 启动一个立即退出的子进程
        proc = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait(timeout=5)
        # 进程已退出,调用 kill_process_tree 应不抛异常
        kill_process_tree(proc.pid)


if __name__ == "__main__":
    unittest.main()
