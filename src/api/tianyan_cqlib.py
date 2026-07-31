"""
天衍云 cqlib SDK 封装
Cqlib Wrapper for Tianyan Cloud Platform

基于官方 cqlib 库封装的量子任务客户端，支持：
- 真机任务提交（QCIS 格式）
- 任务状态查询与结果获取
- 量子计算机列表查询
- 自动重试和异常处理

使用前需安装：pip install cqlib
"""

import threading
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.api.hardware_adapter import CircuitFormat, QuantumHardwareBackend
from src.api.types import TaskResult

if TYPE_CHECKING:
    from src.api.quota_tracker import QuotaTracker

# Issue #515: QCIS 电路内容验证常量
MAX_QCIS_LENGTH = 100_000
MAX_GATE_COUNT = 10_000
MAX_QUBITS_REFERENCED = 287
_QCIS_VALID_INSTRUCTIONS = frozenset(
    {"H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ", "CZ", "CNOT", "M", "B", "ISWAP", "I"}
)


def _validate_qcis(qcis_str: str) -> None:
    """验证 QCIS 电路内容，防止提交超深/非法电路（Issue #515）。

    Args:
        qcis_str: QCIS 指令字符串

    Raises:
        ValueError: 电路超过长度/门数/比特数上限，或包含非法指令
    """
    if len(qcis_str) > MAX_QCIS_LENGTH:
        raise ValueError(f"QCIS 电路超过最大长度 {MAX_QCIS_LENGTH} 字符")

    lines = [ln.strip() for ln in qcis_str.strip().split("\n") if ln.strip()]
    if len(lines) > MAX_GATE_COUNT:
        raise ValueError(f"QCIS 门数量超过上限 {MAX_GATE_COUNT}")

    referenced_qubits: set[str] = set()
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        op = parts[0].upper()
        if op not in _QCIS_VALID_INSTRUCTIONS:
            raise ValueError(f"QCIS 非法指令: {parts[0]}")
        for token in parts[1:]:
            token = token.strip(",")
            if token.upper().startswith("Q") and token[1:].isdigit():
                referenced_qubits.add(token)
    if len(referenced_qubits) > MAX_QUBITS_REFERENCED:
        raise ValueError(
            f"QCIS 引用比特数 {len(referenced_qubits)} 超过上限 {MAX_QUBITS_REFERENCED}"
        )


class CqlibTianyanClient(QuantumHardwareBackend):
    """基于 cqlib SDK 的天衍云真机客户端

    直接调用天衍云超导量子计算机执行量子电路。

    使用示例::

        client = CqlibTianyanClient(login_key="your_key")
        task_id = client.submit_quantum_task(qcis="H Q0\\nM Q0", shots=1024)
        result = client.wait_for_task(task_id)
    """

    # 已知可用的超导真机
    REAL_MACHINES = [  # noqa: RUF012
        "tianyan-287",  # 天衍-287 (105数据比特+182耦合比特 超导, paid)
        "tianyan_sw",  # 超导 free
        "tianyan_s",  # 超导 free
        "tianyan_tn",  # 超导 free
        "tianyan_tnn",  # 超导 free
        "tianyan_swn",  # 超导 free
        "tianyan_sa",  # 超导 free
        "tianyan176",  # 176比特 free
        "tianyan176-2",  # 176比特 free
    ]

    def __init__(
        self,
        login_key: str,
        machine_name: str = "tianyan_s",
        auto_retry_machine: bool = True,
        quota_tracker: "QuotaTracker | None" = None,
        api_secret: str | None = None,
        app_id: str | None = None,
    ):
        """初始化 cqlib 客户端

        Args:
            login_key: API Key（从个人中心获取）
            machine_name: 默认使用的量子计算机名称
            auto_retry_machine: 当前机器不可用时是否自动切换
            quota_tracker: 真机配额追踪器（可选，传入后提交前做配额预检，
                          提交成功后记录消耗；为 None 时不做配额控制）
            api_secret: API Secret（可选，当 SDK 支持时透传给 TianYanPlatform）
            app_id: App ID（可选，当 SDK 支持时透传给 TianYanPlatform）
        """
        import cqlib

        self.cqlib = cqlib
        self._login_key = login_key  # Issue #735: 私有属性，避免调试/序列化泄露
        self.machine_name = machine_name
        self.auto_retry_machine = auto_retry_machine
        self._platform = None
        self._quota_tracker = quota_tracker
        self._api_secret = api_secret
        self._app_id = app_id
        # Issue #701: 线程锁保护 platform 属性的懒加载
        self._platform_lock = threading.Lock()

        if api_secret or app_id:
            logger.info("[Cqlib] 额外凭证已加载（api_secret/app_id），将在平台初始化时透传")

        logger.info(f"[Cqlib] 客户端初始化，默认机器={machine_name}")

    @property
    def login_key(self) -> str:
        """返回 API Key（Issue #735: 私有属性的只读访问器）。

        内部存储为 ``_login_key`` 私有属性，避免在 ``repr()``、序列化或
        调试输出中泄露密钥；保留公开访问接口以维持向后兼容。
        """
        return self._login_key

    def __repr__(self) -> str:
        """返回脱敏的对象表示，避免泄露 API Key（Issue #735）。"""
        return f"CqlibTianyanClient(machine={self.machine_name!r}, login_key=***)"

    # ------------------------------------------------------------------
    # QuantumHardwareBackend ABC 接口实现（Issue #257）
    # ------------------------------------------------------------------

    @property
    def supported_gates(self) -> list[str]:
        """返回天衍云超导真机支持的量子门列表。"""
        return ["H", "X", "Y", "Z", "RX", "RY", "RZ", "CNOT", "CZ", "M"]

    @property
    def topology(self) -> dict[str, Any]:
        """返回天衍-287 的耦合图拓扑信息。

        天衍-287 超导量子计算机：105 数据比特 + 182 耦合比特，
        采用 2D 网格拓扑（最近邻耦合）。
        """
        return {
            "type": "2d_grid",
            "machine_name": self.machine_name,
            "total_qubits": 287,
            "data_qubits": 105,
            "coupler_qubits": 182,
            "connectivity": "nearest_neighbor",
            "description": "天衍-287 超导量子计算机，2D 网格拓扑",
        }

    @property
    def backend_type(self) -> str:
        """返回后端类型标识。"""
        return "superconducting"

    @property
    def circuit_format(self) -> CircuitFormat:
        """天衍云超导后端使用 QCIS 指令格式。"""
        return CircuitFormat.QCIS

    def submit_circuit(
        self,
        circuit: str,
        shots: int = 1024,
        task_name: str = "Scheduler_Task",
    ) -> str | None:
        """提交量子电路到天衍云真机（QuantumHardwareBackend 接口实现）。

        本方法是 ``submit_quantum_task`` 的 ABC 接口适配，
        将 ``circuit`` 参数映射为 QCIS 指令字符串。

        Args:
            circuit   : QCIS 指令字符串
            shots     : 测量次数
            task_name : 任务名称

        Returns:
            task_id 字符串；全部机器不可用时返回 None
        """
        return self.submit_quantum_task(
            qcis=circuit,
            shots=shots,
            task_name=task_name,
        )

    @property
    def platform(self) -> Any:
        """懒加载平台连接

        Issue #701: 加锁保护，避免并发首次访问时创建多个平台实例。
        """
        # 双重检查锁定模式：先无锁检查，再加锁创建
        if self._platform is not None:
            return self._platform
        with self._platform_lock:
            if self._platform is None:
                kwargs: dict[str, Any] = {
                    "login_key": self.login_key,
                    "machine_name": self.machine_name,
                }
                if self._api_secret:
                    kwargs["api_secret"] = self._api_secret
                if self._app_id:
                    kwargs["app_id"] = self._app_id
                self._platform = self.cqlib.TianYanPlatform(**kwargs)
            return self._platform

    def authenticate(self) -> bool:
        """验证 API Key 有效性。

        异常分类（Issue #218）：
        - 网络异常（ConnectionError/TimeoutError/OSError）：返回 False 但不记录为
          认证失败，调用方可通过 ``authenticate_strict()`` 区分网络问题和凭证错误。
        - 认证错误（凭证无效/权限不足）：记录 ERROR 级别日志，明确为永久性错误。

        Returns:
            bool: True 表示认证成功，False 表示失败（含网络问题和凭证错误）
        """
        try:
            _ = self.platform
            return True
        except OSError as e:
            # 网络问题：连接超时/拒绝/不可达，不应被视为认证失败
            logger.warning(f"[Cqlib] 认证时遭遇网络问题（不计入凭证失败）: {type(e).__name__}: {e}")
            return False
        except Exception as e:  # noqa: BLE001
            # cqlib 平台连接异常类型无法穷举，保留宽捕获并记录日志
            # 默认视为永久性错误（凭证无效/权限不足等）
            logger.error(f"[Cqlib] 认证失败（凭证或服务端问题）: {e}")
            return False

    def authenticate_strict(self) -> tuple[bool, str | None, bool]:
        """严格认证：返回认证结果、错误信息和是否为暂时性错误（Issue #218）。

        与 ``authenticate()`` 的区别：调用方可以基于返回的三元组明确区分
        "网络问题"（暂时性，可重试）和"凭证错误"（永久性，应降级）。

        Returns:
            tuple[bool, str | None, bool]:
            - 成功标志（True 表示认证通过）
            - 错误信息（成功时为 None）
            - 是否为暂时性错误（True 表示网络/服务端问题，可重试）
        """
        try:
            _ = self.platform
            return True, None, False
        except OSError as e:
            # 网络问题：暂时性错误，不应触发降级
            return False, f"网络异常: {type(e).__name__}: {e}", True
        except Exception as e:  # noqa: BLE001
            # cqlib 平台连接异常：默认视为永久性错误
            err_msg = str(e)
            # 部分关键字提示为暂时性服务端错误（5xx）
            transient_keywords = (
                "internal server error",
                "service unavailable",
                "gateway timeout",
                "bad gateway",
                "server error",
                "服务繁忙",
                "暂时不可用",
            )
            is_transient = any(kw in err_msg.lower() for kw in transient_keywords)
            return False, f"认证失败: {err_msg}", is_transient

    def list_backends(self) -> list[dict[str, Any]]:
        """列出所有可用的量子计算机"""
        try:
            machines = self.platform.query_quantum_computer_list()
            return [
                {
                    "id": m[0],
                    "type": m[1],
                    "status": m[2],
                    "name": m[3],
                }
                for m in machines
            ]
        except Exception as e:  # noqa: BLE001
            # cqlib 查询接口异常类型无法穷举，保留宽捕获并记录日志
            logger.error(f"[Cqlib] 获取机器列表失败: {e}")
            return []

    def get_backend_info(self, backend_name: str | None = None) -> dict[str, Any]:
        """获取指定后端信息"""
        name = backend_name or self.machine_name
        machines = self.list_backends()
        for m in machines:
            if m["name"] == name:
                return m
        return {}

    def submit_quantum_task(
        self,
        qcis: str = "",
        circuit: Any = None,
        shots: int = 1024,
        task_name: str = "Scheduler_Task",
    ) -> str | None:
        """提交量子任务到真机（含故障自动切换）

        提交策略：
            1. 预检当前机器状态：若非 running（校准中/维护中），立即跳过，不重试
            2. 尝试在当前机器提交；失败时按 auto_retry_machine 切换备用机
            3. 所有机器不可用时返回 None（不抛异常，保证调度循环不中断）

        Args:
            qcis: QCIS 指令字符串（"H Q0\\nM Q0"）
            circuit: cqlib.Circuit 对象（与 qcis 二选一）
            shots: 测量次数
            task_name: 任务名称

        Returns:
            task_id 字符串；全部机器不可用时返回 None
        """
        # 生成 QCIS
        if qcis:
            qcis_str = qcis
        elif circuit is not None:
            qcis_str = circuit.qcis if hasattr(circuit, "qcis") else str(circuit)
        else:
            raise ValueError("必须提供 qcis 或 circuit")

        # Issue #515: 验证 QCIS 电路内容，防止提交超深/非法电路
        _validate_qcis(qcis_str)

        # 配额预检查：配额不足时跳过提交，保持"全部不可用返回 None"语义
        if self._quota_tracker is not None and not self._quota_tracker.can_consume(
            shots=shots, tasks=1
        ):
            logger.warning(f"[Cqlib] 真机配额不足，跳过提交: {task_name}, shots={shots}")
            return None

        logger.info(f"[Cqlib] 提交量子任务: {task_name}, shots={shots}")
        logger.debug(f"[Cqlib] QCIS: {qcis_str[:100]}")

        # 预检当前机器状态（校准中/维护中立即跳过，不重试）
        if not self._is_machine_available(self.machine_name):
            logger.warning(f"[Cqlib] {self.machine_name} 不可用（校准/维护中），切换备用机")
            if self.auto_retry_machine:
                return self._retry_other_machine(qcis_str, shots, task_name)
            return None

        try:
            result = self.platform.submit_experiment(
                circuit=qcis_str,
                name=task_name,
                num_shots=shots,
                is_verify=False,
            )
            if isinstance(result, list) and len(result) > 0:
                task_id = str(result[0])
                logger.info(f"[Cqlib] 任务已提交: {task_id}")
                # 提交成功后记录配额消耗
                if self._quota_tracker is not None:
                    self._quota_tracker.consume(shots=shots, tasks=1)
                return task_id
            # 非列表结果同样视为提交成功，记录配额消耗
            if self._quota_tracker is not None:
                self._quota_tracker.consume(shots=shots, tasks=1)
            return str(result)
        except Exception as e:  # noqa: BLE001
            # cqlib 提交接口异常类型无法穷举，保留宽捕获并记录日志
            err_msg = str(e)
            logger.error(f"[Cqlib] {self.machine_name} 提交失败: {err_msg}")
            # 容量错误（电路超出免费额度）：不重试其他机器，直接返回 None
            if self._is_capacity_error(err_msg):
                logger.warning(f"[Cqlib] 容量不足，跳过提交（不重试其他机器）: {err_msg[:80]}")
                return None
            # 校准/不可用类错误立即切换，不重试当前机器
            if self._is_unavailable_error(err_msg) and self.auto_retry_machine:
                return self._retry_other_machine(qcis_str, shots, task_name)
            if self.auto_retry_machine:
                return self._retry_other_machine(qcis_str, shots, task_name)
            return None

    def _is_machine_available(self, machine_name: str) -> bool:
        """检查机器是否在线可用（status == running）。

        通过 query_quantum_computer_list 查询状态。查询本身失败时
        乐观返回 True（不阻塞提交，让 submit 自行暴露真实错误）。

        Args:
            machine_name: 机器名

        Returns:
            bool: running 返回 True，calibration/maintenance/unknown 返回 False
        """
        try:
            machines = self.list_backends()
            for m in machines:
                if m.get("name") == machine_name:
                    return m.get("status") == "running"
            # 未找到该机器，乐观放行
            return True
        except Exception as e:  # noqa: BLE001
            # cqlib 查询失败不阻塞，乐观放行；记录日志便于排查
            logger.debug(f"[Cqlib] 查询机器 {machine_name} 可用性失败: {e}，乐观放行")
            return True

    @staticmethod
    def _is_unavailable_error(err_msg: str) -> bool:
        """判断错误是否为机器不可用（校准/维护/忙）类错误。

        Args:
            err_msg: 异常消息字符串

        Returns:
            bool: 命中关键词返回 True
        """
        keywords = (
            "校准",
            "calibration",
            "维护",
            "maintenance",
            "不可用",
            "unavailable",
            "忙碌",
            "busy",
            "offline",
        )
        lower_msg = err_msg.lower()
        return any(kw.lower() in lower_msg for kw in keywords)

    @staticmethod
    def _is_capacity_error(err_msg: str) -> bool:
        """判断错误是否为机时包容量不足（电路超限，非机器问题）。

        此类错误不应触发机器切换，因为电路本身超出免费额度，
        换机器也无法解决。应直接返回 None 让上层跳过。

        Args:
            err_msg: 异常消息字符串

        Returns:
            bool: 命中关键词返回 True
        """
        keywords = (
            "最大比特数",
            "机时包",
            "qubit",
            "capacity",
            "超出",
        )
        lower_msg = err_msg.lower()
        return any(kw.lower() in lower_msg for kw in keywords)

    @staticmethod
    def _is_permission_error(err_msg: str) -> bool:
        """判断错误是否为权限不足（专属资源无权限）。

        此类错误应跳过当前机器，尝试其他机器（而非直接放弃）。

        Args:
            err_msg: 异常消息字符串

        Returns:
            bool: 命中关键词返回 True
        """
        keywords = (
            "专属资源",
            "权限",
            "permission",
            "forbidden",
        )
        lower_msg = err_msg.lower()
        return any(kw.lower() in lower_msg for kw in keywords)

    def _retry_other_machine(self, qcis: str, shots: int, task_name: str) -> str | None:
        """当前机器不可用时，按 REAL_MACHINES 列表尝试其他机器。

        每台候选机器先做可用性预检（跳过校准/维护中的），再尝试提交。
        全部不可用时返回 None（不抛异常）。

        Args:
            qcis: QCIS 指令字符串
            shots: 测量次数
            task_name: 任务名称

        Returns:
            task_id 字符串；全部失败返回 None
        """
        for machine in self.REAL_MACHINES:
            if machine == self.machine_name:
                continue
            # 预检：跳过不可用机器，避免无效重试
            if not self._is_machine_available(machine):
                logger.debug(f"[Cqlib] 跳过 {machine}（不可用）")
                continue
            try:
                logger.info(f"[Cqlib] 尝试备用机器: {machine}")
                alt_kwargs: dict[str, Any] = {
                    "login_key": self.login_key,
                    "machine_name": machine,
                }
                if self._api_secret:
                    alt_kwargs["api_secret"] = self._api_secret
                if self._app_id:
                    alt_kwargs["app_id"] = self._app_id
                alt = self.cqlib.TianYanPlatform(**alt_kwargs)
                try:
                    result = alt.submit_experiment(
                        circuit=qcis,
                        name=task_name,
                        num_shots=shots,
                        is_verify=False,
                    )
                finally:
                    # Issue #882: 备用机 Platform 连接用完即释放，避免多次
                    # 重试累积连接泄漏（SDK 提供 close 时调用，缺失则忽略）。
                    close = getattr(alt, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception as close_err:  # noqa: BLE001
                            logger.debug(f"[Cqlib] {machine} 连接释放失败: {close_err}")
                if isinstance(result, list) and len(result) > 0:
                    tid = str(result[0])
                    logger.info(f"[Cqlib] {machine} 提交成功: {tid}")
                else:
                    tid = str(result)
                # 备用机器提交成功后记录配额消耗（与主路径一致）
                if self._quota_tracker is not None:
                    self._quota_tracker.consume(shots=shots, tasks=1)
                return tid
            except Exception as e:  # noqa: BLE001
                # cqlib 备用机器提交异常类型无法穷举，保留宽捕获并记录日志
                err_msg = str(e)
                logger.debug(f"[Cqlib] {machine} 失败: {err_msg[:80]}")
                # 容量错误：电路本身超出免费额度，换机器也无效，直接放弃
                if self._is_capacity_error(err_msg):
                    logger.warning(f"[Cqlib] 容量不足，放弃所有重试: {err_msg[:80]}")
                    return None
                # 权限错误（专属资源）：跳过当前机器，继续尝试其他
                if self._is_permission_error(err_msg):
                    logger.debug(f"[Cqlib] {machine} 无权限（专属资源），跳过")
                continue
        logger.error("[Cqlib] 所有备用机器均不可用，放弃提交（返回 None）")
        return None

    def get_task_status(self, task_id: str) -> TaskResult:
        """查询任务状态（非阻塞，max_wait_time=2s 仅做一次 HTTP 尝试）。

        注意：cqlib 的 query_experiment 默认 max_wait_time=3600s，
        会导致长时间阻塞。本方法显式传入 max_wait_time=2 实现即时返回。
        任务未完成时返回 status="running"。
        """
        try:
            from cqlib.exceptions import CqlibRequestError

            result = self.platform.query_experiment(
                task_id,
                max_wait_time=2,
                sleep_time=1,
            )
            if isinstance(result, list) and len(result) > 0:
                data = result[0]
                if isinstance(data, dict):
                    has_result = "resultStatus" in data or "probability" in data
                    probability = data.get("probability")
                    counts = data.get("counts")
                    return TaskResult(
                        task_id=task_id,
                        status="completed" if has_result else "running",
                        probability=probability if isinstance(probability, dict) else {},
                        counts=counts if isinstance(counts, dict) else None,
                        shots=int(data.get("shots", 0) or 0),
                        backend=str(data.get("machine", self.machine_name)),
                        raw=data,
                    )
            return TaskResult(
                task_id=task_id,
                status="unknown",
                probability={},
                counts=None,
                shots=0,
                backend=self.machine_name,
                raw=result,
            )
        except CqlibRequestError as e:
            # SDK 同时使用 CqlibRequestError 表示"仍在运行"和服务端终态失败。
            # 终态失败必须立即返回 error，否则 wait_for_task 会无意义轮询到超时。
            message = str(e)
            terminal_failure_markers = ("运行失败", "run failure", "tasks have failed")
            if any(marker in message.lower() for marker in terminal_failure_markers):
                return TaskResult(
                    task_id=task_id,
                    status="error",
                    probability={},
                    counts=None,
                    shots=0,
                    backend=self.machine_name,
                    error=message,
                )
            # 已核实：不得把通用 CqlibRequestError 无条件标成 running
            # 可能是查询错误、网络问题或未知 SDK 状态
            # 标记为 query_error 而非 running，绝不能计为 completed
            logger.debug(f"[Cqlib] 查询任务 {task_id} CqlibRequestError: {message[:80]}")
            return TaskResult(
                task_id=task_id,
                status="query_error",
                probability={},
                counts=None,
                shots=0,
                backend=self.machine_name,
                error=message[:200],
                raw={},
            )
        except Exception as e:  # noqa: BLE001
            # cqlib 查询接口异常类型无法穷举，保留宽捕获并记录日志
            logger.debug(f"[Cqlib] 查询任务 {task_id} 状态失败: {e}")
            return TaskResult(
                task_id=task_id,
                status="error",
                probability={},
                counts=None,
                shots=0,
                backend=self.machine_name,
                error=str(e),
            )

    def get_task_result(self, task_id: str) -> TaskResult:
        """获取任务执行结果"""
        return self.get_task_status(task_id)

    def wait_for_task(self, task_id: str, timeout: int = 300, poll_interval: int = 5) -> TaskResult:
        """轮询等待任务完成并返回结果

        处理 ``query_error`` 状态：连续 3 次查询失败后快速终止，
        避免无意义轮询至超时（Issue #407）。
        Issue #719: 仅统计"连续"失败，任意非 query_error 状态重置计数器，
        避免 query_error 与 running/unknown 交替时累计误终止。

        Args:
            task_id: 任务 ID
            timeout: 超时秒数
            poll_interval: 轮询间隔秒数
        """
        start = time.time()
        query_fail_count = 0
        while time.time() - start < timeout:
            status = self.get_task_status(task_id)
            if status["status"] == "completed":
                return status
            if status["status"] == "error":
                return status
            if status["status"] == "query_error":
                query_fail_count += 1
                if query_fail_count >= 3:
                    return TaskResult(
                        task_id=task_id,
                        status="error",
                        probability={},
                        counts=None,
                        shots=0,
                        backend=self.machine_name,
                    )
            else:
                # Issue #719: 非 query_error 状态重置连续失败计数，
                # 避免 query_error 与 running/unknown 交替时累计误终止
                query_fail_count = 0
            time.sleep(poll_interval)
        return TaskResult(
            task_id=task_id,
            status="timeout",
            probability={},
            counts=None,
            shots=0,
            backend=self.machine_name,
        )

    def get_queue_status(self) -> dict[str, Any]:
        """获取队列状态（cqlib 无此接口，返回估算）"""
        machines = self.list_backends()
        running = sum(1 for m in machines if m.get("status") == "running")
        return {
            "total_machines": len(machines),
            "running": running,
            "available": [m["name"] for m in machines if m["status"] == "running"],
        }

    def is_available(self) -> bool:
        """检查真机是否可用（用于降级判断）。

        判定逻辑：
            1. 平台连接可建立（authenticate 通过）
            2. 当前机器在列表中且状态为 running

        任何异常均视为不可用，调用方可据此降级到 Mock。

        Returns:
            bool: True 表示真机可提交，False 表示应降级
        """
        try:
            if not self.authenticate():
                return False
            return self._is_machine_available(self.machine_name)
        except Exception as e:  # noqa: BLE001
            # cqlib 平台访问异常类型无法穷举，保留宽捕获并记录日志
            logger.debug(f"[Cqlib] is_available 检查失败: {e}")
            return False

    def submit_and_get_task_id(
        self,
        qcis: str,
        shots: int = 512,
        task_name: str = "Scheduler_Real_Task",
    ) -> str | None:
        """提交量子任务并立即返回 task_id（非阻塞）。

        本方法是 ``submit_quantum_task`` 的语义化别名，强调“提交后立即返回，
        不等待结果”，便于调度循环在 step() 中非阻塞调用，后续通过
        ``get_task_status`` 轮询结果。

        Args:
            qcis      : QCIS 指令字符串
            shots     : 测量次数
            task_name : 任务名称

        Returns:
            task_id 字符串；提交失败或真机不可用时返回 None
        """
        return self.submit_quantum_task(
            qcis=qcis,
            shots=shots,
            task_name=task_name,
        )


class MultiMachineCqlibCoordinator:
    """多机器 cqlib 协调器：统一管理多台天衍云真机的提交与状态聚合。

    每台机器对应一个独立的 CqlibTianyanClient 实例（独立 platform 连接），
    本协调器负责按机器名分发任务、聚合队列状态、汇总真机提交计数。

    使用示例::

        coord = MultiMachineCqlibCoordinator(
            login_key="xxx",
            machine_names=["tianyan_s", "tianyan_sw", "tianyan_tn"],
        )
        task_id = coord.submit_to_machine("tianyan_s", "H Q0\\nM Q0", shots=512)
        status = coord.get_all_status()
    """

    def __init__(
        self,
        login_key: str,
        machine_names: list[str],
        auto_retry_machine: bool = False,
        quota_tracker: "QuotaTracker | None" = None,
        api_secret: str | None = None,
        app_id: str | None = None,
    ):
        """初始化多机器协调器。

        Args:
            login_key        : 天衍云 API Key
            machine_names    : 要纳管的机器名列表
            auto_retry_machine: 单机提交失败时是否自动切换其他机器（默认 False，
                               多机器场景下由调度器决定路由，通常关闭单机重试）
            quota_tracker    : 真机配额追踪器（可选，传入后 submit_to_machine
                              成功时记录消耗；为 None 时不做配额记录）
            api_secret       : API Secret（可选，透传给各机器的 CqlibTianyanClient）
            app_id           : App ID（可选，透传给各机器的 CqlibTianyanClient）
        """
        self._login_key = login_key  # Issue #735: 私有属性，避免调试/序列化泄露
        self.machine_names = list(machine_names)
        self.auto_retry_machine = auto_retry_machine
        self._quota_tracker = quota_tracker
        self._api_secret = api_secret
        self._app_id = app_id
        self._clients: dict[str, CqlibTianyanClient] = {}
        self._submit_count: dict[str, int] = dict.fromkeys(self.machine_names, 0)
        self._fail_count: dict[str, int] = dict.fromkeys(self.machine_names, 0)
        # Issue #666: 线程锁，保护 _clients 懒加载与计数器并发读写
        self._lock = threading.Lock()

        logger.info(f"[MultiMachine] 纳管 {len(self.machine_names)} 台机器: {self.machine_names}")

    @property
    def login_key(self) -> str:
        """返回 API Key（Issue #735: 私有属性的只读访问器）。

        内部存储为 ``_login_key`` 私有属性，避免在 ``repr()``、序列化或
        调试输出中泄露密钥；保留公开访问接口以维持向后兼容。
        """
        return self._login_key

    def __repr__(self) -> str:
        """返回脱敏的对象表示，避免泄露 API Key（Issue #735）。"""
        return f"MultiMachineCqlibCoordinator(machines={self.machine_names!r}, login_key=***)"

    def _get_client(self, machine_name: str) -> CqlibTianyanClient:
        """懒加载指定机器的客户端（避免初始化时连接所有机器）。

        Issue #666: 加锁保护，避免并发下重复创建客户端导致连接泄漏。
        """
        with self._lock:
            if machine_name not in self._clients:
                if machine_name not in self.machine_names:
                    raise ValueError(f"机器 {machine_name} 未被纳管")
                self._clients[machine_name] = CqlibTianyanClient(
                    login_key=self.login_key,
                    machine_name=machine_name,
                    auto_retry_machine=self.auto_retry_machine,
                    api_secret=self._api_secret,
                    app_id=self._app_id,
                )
            return self._clients[machine_name]

    def submit_to_machine(
        self,
        machine_name: str,
        qcis: str,
        shots: int = 512,
        task_name: str = "MultiMachine_Task",
    ) -> str | None:
        """向指定机器提交量子任务。

        Args:
            machine_name: 目标机器名
            qcis        : QCIS 指令字符串
            shots       : 测量次数
            task_name   : 任务名称

        Returns:
            task_id 字符串；提交失败返回 None
        """
        try:
            client = self._get_client(machine_name)
            task_id = client.submit_quantum_task(qcis=qcis, shots=shots, task_name=task_name)
            # Issue #666: 加锁保护计数器更新，避免并发下计数丢失
            with self._lock:
                self._submit_count[machine_name] = self._submit_count.get(machine_name, 0) + 1
            # 提交成功后记录配额消耗（仅当 task_id 非 None 时）
            if self._quota_tracker is not None and task_id is not None:
                self._quota_tracker.consume(shots=shots, tasks=1)
            return task_id
        except Exception as e:  # noqa: BLE001
            # 涉及客户端获取（ValueError）与提交，异常类型无法穷举，保留宽捕获并记录日志
            with self._lock:
                self._fail_count[machine_name] = self._fail_count.get(machine_name, 0) + 1
            logger.error(f"[MultiMachine] {machine_name} 提交失败: {e}")
            return None

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """聚合所有纳管机器的队列状态。

        Returns:
            {machine_name: queue_status_dict} 映射
        """
        status = {}
        for name in self.machine_names:
            try:
                client = self._get_client(name)
                status[name] = client.get_queue_status()
            except Exception as e:  # noqa: BLE001
                # 涉及客户端获取与队列查询，异常类型无法穷举，保留宽捕获并记录日志
                logger.debug(f"[MultiMachine] 获取 {name} 队列状态失败: {e}")
                status[name] = {"error": str(e)[:80]}
        return status

    def get_submit_stats(self) -> dict[str, dict[str, int]]:
        """返回各机器的真机提交统计。

        Returns:
            {machine_name: {"submit": n, "fail": m}} 映射
        """
        return {
            name: {
                "submit": self._submit_count.get(name, 0),
                "fail": self._fail_count.get(name, 0),
            }
            for name in self.machine_names
        }

    def as_client_map(self) -> dict[str, CqlibTianyanClient]:
        """返回 {machine_name: client} 映射，便于注入 env.attach_real_clients。

        注意：此方法会触发所有纳管机器的客户端懒加载。
        """
        for name in self.machine_names:
            self._get_client(name)
        return dict(self._clients)


def create_multi_machine_clients(
    login_key: str,
    machine_names: list[str],
) -> dict[str, CqlibTianyanClient]:
    """工厂函数：为每台机器创建独立的 cqlib 客户端。

    Args:
        login_key    : 天衍云 API Key
        machine_names: 机器名列表

    Returns:
        {machine_name: CqlibTianyanClient} 映射，可直接传给 env.attach_real_clients
    """
    return {
        name: CqlibTianyanClient(
            login_key=login_key,
            machine_name=name,
            auto_retry_machine=False,
        )
        for name in machine_names
    }
