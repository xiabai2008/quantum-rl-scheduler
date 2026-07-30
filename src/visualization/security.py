"""
Web 可视化层安全组件

集中提供以下安全能力（Issue #514 / #515 / #516 / #517）：
    - ``RateLimiter``                 滑动窗口速率限制器（POST 端点保护）
    - ``validate_quantum_circuit``    QCIS/QASM 量子电路内容校验
    - ``sanitize_error_message``      错误消息净化（移除内部信息）
    - ``get_allowed_ws_origins``      WebSocket 允许的 Origin 列表
    - ``rate_limiter``                全局速率限制器单例

环境变量：
    - ``VIZ_WS_ALLOWED_ORIGINS``  逗号分隔的 WebSocket 允许 Origin 列表
    - ``VIZ_RATE_LIMIT``          每分钟最大请求数（默认 60）
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any

from fastapi import HTTPException

# ============================================================
# 常量
# ============================================================

#: WebSocket 单条消息最大字节数（1 MB）
WS_MAX_MESSAGE_BYTES: int = 1 * 1024 * 1024

#: WebSocket 最大并发连接数
WS_MAX_CONNECTIONS: int = 100

#: 量子电路最大门数（电路深度上限）
MAX_CIRCUIT_GATES: int = 10000

#: 量子电路最大量子比特数（天衍-287 可用比特上限）
MAX_CIRCUIT_QUBITS: int = 105

#: 速率限制默认值（每分钟请求数）
DEFAULT_RATE_LIMIT_PER_MINUTE: int = 60

#: 速率限制窗口（秒）
RATE_LIMIT_WINDOW_SECONDS: int = 60

#: 默认允许的 WebSocket Origin 列表（开发/本地环境）
_DEFAULT_ALLOWED_ORIGINS: list[str] = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
    # Starlette TestClient 默认 Origin
    "http://testserver",
    "https://testserver",
]

#: 允许的电路字符集（ASCII 可打印 + 换行/回车）
_CIRCUIT_ALLOWED_CHARS: set[int] = set(range(32, 127)) | {10, 13, 9}


# ============================================================
# 速率限制器（Issue #517）
# ============================================================


class RateLimiter:
    """基于滑动窗口的内存速率限制器。

    每个 key（通常是客户端 IP）在指定时间窗口内最多允许 ``max_requests`` 次请求。
    超出限制时 ``check()`` 返回 ``(False, limit, 0)``，调用方应返回 429。

    线程安全：使用 ``threading.Lock`` 保护内部状态。

    Attributes:
        max_requests: 时间窗口内最大请求数
        window_seconds: 时间窗口大小（秒）
    """

    def __init__(
        self,
        max_requests: int = DEFAULT_RATE_LIMIT_PER_MINUTE,
        window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
    ) -> None:
        """初始化速率限制器。

        Args:
            max_requests: 时间窗口内最大请求数
            window_seconds: 时间窗口大小（秒）
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int, int]:
        """检查 key 是否在速率限制范围内。

        如果在范围内，记录本次请求时间戳并返回 ``(True, limit, remaining)``。
        如果超出限制，不记录时间戳并返回 ``(False, limit, 0)``。

        Args:
            key: 限流键（通常是客户端 IP）

        Returns:
            ``(allowed, limit, remaining)`` 三元组：
            - allowed: 是否允许本次请求
            - limit: 时间窗口内最大请求数
            - remaining: 剩余可用请求数
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            # 清理过期时间戳
            timestamps = [t for t in self._requests.get(key, []) if t > cutoff]
            current_count = len(timestamps)
            if current_count >= self.max_requests:
                self._requests[key] = timestamps
                return (False, self.max_requests, 0)
            timestamps.append(now)
            self._requests[key] = timestamps
            remaining = self.max_requests - current_count - 1
            return (True, self.max_requests, remaining)

    def reset(self) -> None:
        """清空所有速率限制状态（用于测试隔离）。"""
        with self._lock:
            self._requests.clear()


#: 全局速率限制器单例（POST 端点保护，Issue #517）
rate_limiter = RateLimiter(
    max_requests=int(os.getenv("VIZ_RATE_LIMIT", str(DEFAULT_RATE_LIMIT_PER_MINUTE))),
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
)


def get_rate_limit_for_ip(client_ip: str) -> tuple[bool, int, int]:
    """检查指定 IP 是否在速率限制范围内。

    Args:
        client_ip: 客户端 IP 地址

    Returns:
        ``(allowed, limit, remaining)`` 三元组
    """
    return rate_limiter.check(client_ip)


# ============================================================
# 量子电路校验（Issue #515）
# ============================================================


def _estimate_qubit_count(circuit: str, fmt: str) -> int:
    """从电路内容中估算使用的量子比特数。

    QCIS 格式：查找 ``Q0``、``Q1`` 等模式，返回最大索引 + 1。
    QASM 格式：查找 ``qreg q[N];`` 声明，返回最大 N。

    Args:
        circuit: 电路内容字符串
        fmt: 电路格式（"qcis" 或 "openqasm"）

    Returns:
        估算的量子比特数
    """
    if fmt == "openqasm":
        # 查找 qreg q[N]; 声明
        matches = re.findall(r"qreg\s+\w+\s*\[(\d+)\s*\]", circuit, re.IGNORECASE)
        if matches:
            return max(int(m) for m in matches)
        # 回退：查找 q[N] 引用
        refs = re.findall(r"q\[(\d+)\s*\]", circuit, re.IGNORECASE)
        if refs:
            return max(int(r) for r in refs) + 1
        return 0
    else:
        # QCIS 格式：查找 Q0、Q1 等引用
        refs = re.findall(r"Q(\d+)", circuit, re.IGNORECASE)
        if refs:
            return max(int(r) for r in refs) + 1
        return 0


def _count_gates(circuit: str, fmt: str) -> int:
    """统计电路中的门数。

    QCIS 格式：非空、非注释行的数量。
    QASM 格式：以分号结尾的非注释指令数量。

    Args:
        circuit: 电路内容字符串
        fmt: 电路格式（"qcis" 或 "openqasm"）

    Returns:
        门数估算值
    """
    gate_count = 0
    for line in circuit.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # 跳过注释行
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        if fmt == "openqasm":
            # QASM 指令以分号结尾
            if stripped.endswith(";"):
                gate_count += 1
        else:
            # QCIS 每行一条指令
            gate_count += 1
    return gate_count


def validate_quantum_circuit(circuit: str, fmt: str = "qcis") -> dict[str, Any]:
    """验证量子电路内容（QCIS/QASM 格式）。

    校验规则（Issue #515）：
        1. 电路内容非空
        2. 仅包含合法可打印字符
        3. 门数不超过 ``MAX_CIRCUIT_GATES``（默认 10000）
        4. 量子比特数不超过 ``MAX_CIRCUIT_QUBITS``（默认 105）

    Args:
        circuit: 电路内容字符串
        fmt: 电路格式（"qcis" 或 "openqasm"）

    Returns:
        包含 gate_count 和 qubit_count 的字典

    Raises:
        HTTPException: 电路无效时抛出 400 错误
    """
    fmt_normalized = fmt.lower().strip()
    if fmt_normalized not in ("qcis", "openqasm", "qasm"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的电路格式: {fmt}（支持: qcis, openqasm）",
        )

    # 1. 非空检查
    if not circuit or not circuit.strip():
        raise HTTPException(status_code=400, detail="电路内容不能为空")

    # 2. 字符合法性检查：仅允许 ASCII 可打印字符 + 换行/回车/制表符
    invalid_chars = {ord(c) for c in circuit} - _CIRCUIT_ALLOWED_CHARS
    if invalid_chars:
        raise HTTPException(
            status_code=400,
            detail="电路内容包含非法字符（仅允许 ASCII 可打印字符）",
        )

    # 3. 门数检查
    gate_count = _count_gates(circuit, fmt_normalized)
    if gate_count > MAX_CIRCUIT_GATES:
        raise HTTPException(
            status_code=400,
            detail=f"电路深度（门数）{gate_count} 超过上限 {MAX_CIRCUIT_GATES}",
        )

    # 4. 量子比特数检查
    qubit_count = _estimate_qubit_count(circuit, fmt_normalized)
    if qubit_count > MAX_CIRCUIT_QUBITS:
        raise HTTPException(
            status_code=400,
            detail=f"量子比特数 {qubit_count} 超过上限 {MAX_CIRCUIT_QUBITS}",
        )

    return {"gate_count": gate_count, "qubit_count": qubit_count}


# ============================================================
# 错误消息净化（Issue #516）
# ============================================================

#: 匹配文件路径的正则（Unix 和 Windows 路径）
_FILE_PATH_PATTERN = re.compile(
    r"(?:[/\\][\w./-]+)|(?:[A-Za-z]:[/\\][\w./-]+)|(?:[\w./-]+\.(?:py|json|yaml|yml|txt|csv|log|db|sqlite))"
)

#: 匹配 Python 变量名/模块路径的正则
_INTERNAL_NAME_PATTERN = re.compile(r"\'[a-zA-Z_][\w.]*\'")


def sanitize_error_message(message: str) -> str:
    """净化错误消息，移除内部敏感信息。

    移除以下内容（Issue #516）：
        1. 内部文件路径（如 ``/home/user/project/src/...``）
        2. Python 变量名/模块路径（如 ``'some_variable'``）
        3. 异常类型名（如 ``RuntimeError``、``ValueError``）

    Args:
        message: 原始错误消息

    Returns:
        净化后的安全消息
    """
    sanitized = message
    # 移除文件路径
    sanitized = _FILE_PATH_PATTERN.sub("[path]", sanitized)
    # 移除内部变量名（单引号包裹的标识符）
    sanitized = _INTERNAL_NAME_PATTERN.sub("[internal]", sanitized)
    # 移除异常类型名模式（如 "RuntimeError: ..."）
    sanitized = re.sub(r"\b[A-Z]\w*(?:Error|Exception|Warning)\b", "[Error]", sanitized)
    return sanitized


# ============================================================
# WebSocket Origin 配置（Issue #514）
# ============================================================


def get_allowed_ws_origins() -> list[str]:
    """获取 WebSocket 允许的 Origin 列表。

    通过环境变量 ``VIZ_WS_ALLOWED_ORIGINS`` 配置（逗号分隔）。
    未配置时返回默认的 localhost 列表。

    Returns:
        允许的 Origin 字符串列表
    """
    env_value = os.getenv("VIZ_WS_ALLOWED_ORIGINS", "")
    if env_value.strip():
        return [o.strip() for o in env_value.split(",") if o.strip()]
    return list(_DEFAULT_ALLOWED_ORIGINS)


def is_origin_allowed(origin: str) -> bool:
    """检查指定 Origin 是否允许连接 WebSocket。

    Args:
        origin: Origin 头的值

    Returns:
        是否允许
    """
    if not origin:
        # Issue #736: 空 Origin 可被伪造绕过，默认拒绝
        # 如需允许非浏览器客户端（curl 等），设置环境变量 VIZ_WS_ALLOW_EMPTY_ORIGIN=1
        import os

        return os.environ.get("VIZ_WS_ALLOW_EMPTY_ORIGIN", "0") == "1"
    allowed = get_allowed_ws_origins()
    return origin in allowed
