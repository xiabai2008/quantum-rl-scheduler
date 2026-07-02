"""Windows 平台兼容性工具模块

提供跨平台的路径处理、进程管理、编码处理等工具函数,
统一处理 Windows / Linux / macOS 之间的差异。

主要功能:
- 平台检测 (is_windows / is_linux / is_macos)
- 路径规范化与拼接 (normalize_path / join_paths / get_project_root)
- 安全文件/目录操作 (safe_makedirs / safe_open)
- 编码处理 (ensure_windows_encoding)
- 换行符 (get_line_ending)
- 跨平台进程管理 (kill_process_tree / run_subprocess)
"""

import contextlib
import os
import subprocess
import sys
from typing import Any

__all__ = [
    "ensure_windows_encoding",
    "get_line_ending",
    "get_project_root",
    "is_linux",
    "is_macos",
    "is_windows",
    "join_paths",
    "kill_process_tree",
    "normalize_path",
    "run_subprocess",
    "safe_makedirs",
    "safe_open",
]


# ---------------------------------------------------------------------------
# 平台检测
# ---------------------------------------------------------------------------
def is_windows() -> bool:
    """检测当前是否运行在 Windows 平台。

    Returns:
        True 表示当前为 Windows 平台,否则为 False
    """
    return sys.platform.startswith("win")


def is_linux() -> bool:
    """检测当前是否运行在 Linux 平台。

    Returns:
        True 表示当前为 Linux 平台,否则为 False
    """
    return sys.platform.startswith("linux")


def is_macos() -> bool:
    """检测当前是否运行在 macOS 平台。

    Returns:
        True 表示当前为 macOS 平台,否则为 False
    """
    return sys.platform == "darwin"


# ---------------------------------------------------------------------------
# 路径处理
# ---------------------------------------------------------------------------
def normalize_path(path: str) -> str:
    """将路径分隔符规范化为当前系统格式。

    支持正斜杠 (/)、反斜杠 (\\) 以及二者混合的路径,
    统一转换为当前操作系统使用的分隔符,并解析 . 与 .. 引用。

    Args:
        path: 待规范化的路径字符串

    Returns:
        规范化后的路径字符串;空字符串原样返回

    Example:
        >>> # 在 Windows 上
        >>> normalize_path("a/b\\\\c/d")
        'a\\\\b\\\\c\\\\d'
    """
    if not path:
        return path
    return os.path.normpath(path)


def join_paths(*parts: str) -> str:
    """跨平台路径拼接。

    包装 os.path.join,自动使用当前系统的分隔符。
    空字符串段会被自动跳过,避免产生多余的分隔符。

    Args:
        *parts: 路径片段(可变参数)

    Returns:
        拼接后的路径字符串;全部为空时返回空字符串

    Example:
        >>> join_paths("a", "b", "c")  # Windows
        'a\\\\b\\\\c'
    """
    filtered = [p for p in parts if p]
    if not filtered:
        return ""
    return os.path.join(*filtered)


def get_project_root() -> str:
    """获取项目根目录的绝对路径。

    基于当前文件位置推算:本文件位于 <root>/src/utils/platform_compat.py,
    因此根目录为上两级目录。

    Returns:
        项目根目录的规范化绝对路径

    Example:
        >>> root = get_project_root()
        >>> root.endswith("quantum-rl-scheduler")
        True
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    return normalize_path(project_root)


def safe_makedirs(path: str) -> None:
    """安全创建目录(支持已存在与嵌套创建)。

    使用 exist_ok=True 避免目录已存在时抛出异常,
    在 Windows 上对绝对路径添加 \\\\?\\ 前缀以支持长路径(超过 260 字符)。

    Args:
        path: 待创建的目录路径

    Raises:
        OSError: 当目录创建失败(如权限不足)时抛出
    """
    if not path:
        return
    normalized = os.path.normpath(path)
    # Windows 长路径支持:对绝对路径添加 \\?\ 前缀以绕过 260 字符限制
    if is_windows() and os.path.isabs(normalized) and not normalized.startswith("\\\\?\\"):
        normalized = "\\\\?\\" + normalized
    os.makedirs(normalized, exist_ok=True)


def safe_open(
    filepath: str,
    mode: str = "r",
    encoding: str = "utf-8",
) -> Any:
    """跨平台安全打开文件。

    自动处理路径规范化,确保在 Windows 上正确读写中文等内容。
    文件对象本身支持上下文管理器协议,推荐使用 with 语句调用。

    Args:
        filepath: 文件路径(支持正反斜杠混合)
        mode: 文件打开模式,默认 "r"
        encoding: 文件编码,默认 "utf-8"(二进制模式时忽略)

    Returns:
        已打开的文件对象(支持 with 语句使用)

    Example:
        >>> with safe_open("a/b/c.txt", "w") as f:
        ...     f.write("你好")
        >>> with safe_open("a/b/c.txt") as f:
        ...     print(f.read())
        你好
    """
    normalized = normalize_path(filepath)
    # 二进制模式不应指定 encoding,否则会抛出 ValueError
    if "b" in mode:
        return open(normalized, mode=mode)
    return open(normalized, mode=mode, encoding=encoding)


def get_line_ending() -> str:
    """获取当前平台的换行符。

    Windows 使用 \\r\\n,Linux/macOS 使用 \\n。

    Returns:
        当前平台的换行符字符串
    """
    if is_windows():
        return "\r\n"
    return "\n"


# ---------------------------------------------------------------------------
# 编码处理
# ---------------------------------------------------------------------------
def ensure_windows_encoding() -> None:
    """确保 Windows 平台下 stdout/stderr 使用 UTF-8 编码。

    Windows 默认控制台编码可能为 GBK/CP936,会导致输出中文/emoji 时抛出
    UnicodeEncodeError。此函数在 Windows 下重新配置标准流为 UTF-8,
    并设置 PYTHONUTF8=1 环境变量以提示子进程使用 UTF-8。

    在非 Windows 平台上为空操作(no-op)。
    """
    if not is_windows():
        return

    # 重新配置标准流为 UTF-8(避免中文/emoji 输出报错)
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            # 流已被关闭或不支持 reconfigure 时忽略
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")

    # 提示子进程使用 UTF-8(不覆盖用户已设置的值)
    if "PYTHONUTF8" not in os.environ:
        os.environ["PYTHONUTF8"] = "1"


# ---------------------------------------------------------------------------
# 进程管理
# ---------------------------------------------------------------------------
def kill_process_tree(pid: int) -> None:
    """跨平台终止进程树。

    Windows 上使用 taskkill /T /F 终止进程及其所有子进程,
    Unix 上使用 os.killpg 向进程组发送 SIGKILL。

    Args:
        pid: 待终止的根进程 PID

    Note:
        - Windows 上需要 taskkill 命令可用(系统自带)
        - Unix 上需要目标进程与调用者在同一进程组
        - 若进程已退出或 PID 无效,将忽略异常静默返回
    """
    if pid <= 0:
        return

    if is_windows():
        # Windows:taskkill /T(树) /F(强制) /PID
        # 进程可能已退出,或 taskkill 不可用时忽略
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True,
                check=False,
            )
    else:
        # Unix:向进程组发送 SIGKILL(signal.SIGKILL = 9)
        # 进程已退出 / 权限不足 / 其他错误时忽略
        # 使用 getattr 动态获取,避免 Windows 上 os 模块无此属性的类型错误
        killpg = getattr(os, "killpg", None)
        getpgid = getattr(os, "getpgid", None)
        if callable(killpg) and callable(getpgid):
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                killpg(getpgid(pid), 9)


def run_subprocess(
    cmd: list[str],
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """跨平台 subprocess 封装。

    Windows 下默认不使用 shell(避免 shell 注入风险与 cmd.exe 兼容问题),
    Unix 下允许通过 kwargs 中 shell=True 显式启用 shell。

    Args:
        cmd: 命令及其参数列表,如 ["python", "--version"]
        **kwargs: 透传给 subprocess.run 的额外参数(如 cwd / env / timeout)

    Returns:
        subprocess.CompletedProcess 实例,包含 returncode / stdout / stderr

    Raises:
        FileNotFoundError: 当命令不存在时抛出
        subprocess.SubprocessError: 其他子进程相关错误

    Example:
        >>> result = run_subprocess(["python", "--version"])
        >>> result.returncode
        0
    """
    defaults: dict[str, Any] = {
        "text": True,
        "capture_output": True,
    }
    for key, value in kwargs.items():
        defaults[key] = value
    # Windows 下强制不使用 shell,避免 cmd.exe 兼容性问题与注入风险
    if is_windows():
        defaults["shell"] = False
    return subprocess.run(cmd, **defaults)
