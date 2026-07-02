"""
天衍云平台 API 封装客户端
Tianyan Cloud Platform API Client

封装天衍量子云平台的 API 接口，支持量子/经典任务提交、
状态查询、结果获取、后端管理等功能。

认证方式：cqlib SDK（从环境变量 TIANYAN_API_KEY 读取）
配置来源：config/config.yaml + .env
"""

import os
import time
from enum import Enum
from time import monotonic
from typing import Any, cast

import requests
import yaml
from dotenv import load_dotenv
from loguru import logger

from src.exceptions import CircuitOpenError


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class TianyanAPIError(Exception):
    """天衍云平台 API 自定义异常

    当 API 返回非 200 状态码时抛出，携带状态码和响应详情。

    Attributes:
        status_code: HTTP 响应状态码
        message: 错误描述信息
        response_body: 原始响应体（JSON）
    """

    def __init__(self, status_code: int, message: str, response_body: dict | None = None):
        self.status_code = status_code
        self.message = message
        self.response_body = response_body or {}
        super().__init__(f"[{status_code}] {message}")


class CircuitBreaker:
    """熔断器模式实现

    状态：CLOSED（正常）→ OPEN（熔断）→ HALF_OPEN（试探）→ CLOSED

    Args:
        failure_threshold: 连续失败阈值，超过则熔断
        recovery_timeout: 熔断恢复超时时间（秒）
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0

    def before_request(self) -> None:
        """请求前检查熔断器状态"""
        if self.state == CircuitState.OPEN:
            if monotonic() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError("Circuit breaker is open")

    def on_success(self) -> None:
        """请求成功时重置状态"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def on_failure(self) -> None:
        """请求失败时增加失败计数"""
        self.failure_count += 1
        self.last_failure_time = monotonic()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def get_state(self) -> str:
        """返回当前状态字符串"""
        return self.state.value


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
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        mock_mode: bool | None = None,
        enable_circuit_breaker: bool = True,
    ):
        """初始化天衍云客户端

        按优先级读取配置：显式传参 > 环境变量 > config/config.yaml 默认值。

        Mock 模式检测顺序：
        1. 显式传参 ``mock_mode``
        2. 环境变量 ``TIANYAN_MOCK_MODE``（true/1/yes）
        3. 配置文件 ``config/config.yaml`` 中的 ``tianyan.mock_mode``
        4. 默认：True（开发阶段默认使用 Mock 模式）

        Args:
            api_key: API 密钥，若为 None 则从环境变量 ``TIANYAN_API_KEY`` 读取。
            base_url: API 基础 URL，若为 None 则从 ``config/config.yaml`` 读取。
            mock_mode: 是否使用 Mock 模式，None 表示自动检测。
            enable_circuit_breaker: 是否启用熔断器模式。
        """
        # 按需加载 .env 文件中的环境变量
        load_dotenv()

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

        # 读取 api_key
        self.api_key = api_key or os.getenv("TIANYAN_API_KEY", "")

        if not self.api_key:
            logger.warning("未配置 TIANYAN_API_KEY，API 调用将无法通过认证")

        # 读取 base_url（从配置文件回退）
        self.base_url = base_url or self._load_base_url_from_config()
        logger.info(f"天衍客户端初始化完成，base_url={self.base_url}")

        # 真实模式统一走 cqlib SDK（REST API 被 WAF 拦截，已弃用）
        machine_name = os.getenv("TIANYAN_MACHINE", "tianyan_s")
        if self.api_key:
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
            return cast(bool, self._mock_client.authenticate())

        # 真实模式委托 cqlib
        if self._cqlib is not None:
            return self._cqlib.authenticate()

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
            task_id = self._cqlib.submit_quantum_task(
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
            # 熔断器需捕获所有异常以记录失败计数，原异常重新抛出由上层处理
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
            return cast(dict[str, Any], self._mock_client.get_task_result(task_id))

        # 真实模式委托 cqlib
        if self._cqlib is not None:
            return self._cqlib.get_task_result(task_id)

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
            return cast(list[dict[str, Any]], self._mock_client.list_backends())

        # 真实模式委托 cqlib
        if self._cqlib is not None:
            return self._cqlib.list_backends()

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
            return cast(dict[str, Any], self._mock_client.get_backend_info(backend_name))

        # 真实模式委托 cqlib
        if self._cqlib is not None:
            return self._cqlib.get_backend_info(backend_name)

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
            return cast(str, self._mock_client.submit_classical_task(code=code, language=language))

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
            return cast(dict[str, Any], self._mock_client.get_queue_status())

        # 真实模式委托 cqlib
        if self._cqlib is not None:
            return self._cqlib.get_queue_status()

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
        print("认证失败，请检查 TIANYAN_API_KEY 环境变量")
        exit(1)

    print("认证通过")

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
        print(f"任务提交成功，task_id={task_id}")

        # 等待结果
        result = client.wait_for_task(task_id, poll_interval=3.0, timeout=120.0)
        print(f"任务结果: {result}")

    except TianyanAPIError as e:
        print(f"API 错误: {e}")
    except requests.exceptions.RequestException as e:
        print(f"网络错误: {e}")
