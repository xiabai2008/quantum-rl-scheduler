"""
天衍云平台 API 封装客户端
Tianyan Cloud Platform API Client

封装天衍量子云平台的 REST API 接口，支持量子/经典任务提交、
状态查询、结果获取、后端管理等功能。

认证方式：Bearer Token（从环境变量 TIANYAN_API_KEY 读取）
配置来源：config/config.yaml + .env
"""

import os
import time
from typing import Any, Dict, List, Optional

import yaml
import requests
from dotenv import load_dotenv
from loguru import logger

# 加载 .env 文件中的环境变量
load_dotenv()


class TianyanAPIError(Exception):
    """天衍云平台 API 自定义异常

    当 API 返回非 200 状态码时抛出，携带状态码和响应详情。

    Attributes:
        status_code: HTTP 响应状态码
        message: 错误描述信息
        response_body: 原始响应体（JSON）
    """

    def __init__(self, status_code: int, message: str, response_body: Optional[Dict] = None):
        self.status_code = status_code
        self.message = message
        self.response_body = response_body or {}
        super().__init__(f"[{status_code}] {message}")


class TianyanClient:
    """天衍量子云平台客户端

    封装天衍量子云平台的所有 API 接口，提供：
    - 量子电路任务提交（QASM 格式）
    - 经典计算任务提交
    - 任务状态查询与结果获取
    - 量子后端信息查询
    - 队列状态监控

    使用示例::

        client = TianyanClient()
        if client.authenticate():
            task_id = client.submit_quantum_task(circuit_qasm="OPENQASM 2.0; ...")
            status = client.get_task_status(task_id)
            result = client.get_task_result(task_id)

    Args:
        api_key: API 密钥（默认从环境变量 TIANYAN_API_KEY 读取）
        base_url: API 基础 URL（默认从 config/config.yaml 读取）
    """

    # 指数退避重试参数
    MAX_RETRIES = 3
    RETRY_BACKOFF_FACTOR = 2  # 每次重试的等待时间倍数
    RETRY_INITIAL_WAIT = 1.0  # 首次重试等待秒数

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """初始化天衍云客户端

        按优先级读取配置：显式传参 > 环境变量 > config/config.yaml 默认值。

        Args:
            api_key: API 密钥，若为 None 则从环境变量 ``TIANYAN_API_KEY`` 读取。
            base_url: API 基础 URL，若为 None 则从 ``config/config.yaml`` 读取。
        """
        # 读取 api_key
        self.api_key = api_key or os.getenv("TIANYAN_API_KEY", "")

        if not self.api_key:
            logger.warning("未配置 TIANYAN_API_KEY，API 调用将无法通过认证")

        # 读取 base_url（从配置文件回退）
        self.base_url = base_url or self._load_base_url_from_config()
        logger.info(f"天衍客户端初始化完成，base_url={self.base_url}")

        # 创建会话并设置 Bearer Token 认证头
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })

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
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            url = config.get("tianyan", {}).get("base_url", default_url)
            return url
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 不存在，使用默认 base_url")
            return default_url
        except Exception as e:
            logger.warning(f"读取配置文件失败: {e}，使用默认 base_url")
            return default_url

    # ------------------------------------------------------------------
    # 核心请求方法（含指数退避重试）
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """发送 API 请求，内置指数退避重试机制

        当请求因网络异常失败或服务端返回 5xx 错误时，自动进行最多
        ``MAX_RETRIES`` 次重试，每次等待时间按指数增长。

        Args:
            method: HTTP 方法（``GET`` / ``POST`` / ``PUT`` / ``DELETE``）
            endpoint: API 端点路径（如 ``/tasks``），不含 base_url
            data: JSON 请求体
            params: URL 查询参数

        Returns:
            API 返回的 JSON 数据字典

        Raises:
            TianyanAPIError: 服务端返回非 200 状态码且重试耗尽
            requests.exceptions.RequestException: 网络层异常且重试耗尽
        """
        url = f"{self.base_url}{endpoint}"
        last_exception: Optional[Exception] = None

        for attempt in range(self.MAX_RETRIES):
            try:
                logger.debug(f"API 请求 {method} {url}（第 {attempt + 1}/{self.MAX_RETRIES} 次）")

                response = self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    timeout=self._get_timeout(),
                )

                # 2xx 视为成功
                if response.status_code >= 200 and response.status_code < 300:
                    return response.json()

                # 5xx 服务端错误可重试
                if response.status_code >= 500:
                    try:
                        error_body = response.json()
                    except Exception:
                        error_body = {"raw": response.text}
                    raise TianyanAPIError(
                        status_code=response.status_code,
                        message=f"服务端错误: {response.reason}",
                        response_body=error_body,
                    )

                # 4xx 客户端错误不重试，直接抛出
                try:
                    error_body = response.json()
                except Exception:
                    error_body = {"raw": response.text}
                raise TianyanAPIError(
                    status_code=response.status_code,
                    message=f"客户端错误: {response.reason}",
                    response_body=error_body,
                )

            except TianyanAPIError as e:
                # 4xx 直接抛出，不重试
                if e.status_code < 500:
                    logger.error(f"API 客户端错误: {e}")
                    raise
                last_exception = e

            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.warning(f"API 请求异常: {e}")

            # 指数退避等待
            if attempt < self.MAX_RETRIES - 1:
                wait_time = self.RETRY_INITIAL_WAIT * (self.RETRY_BACKOFF_FACTOR ** attempt)
                logger.info(f"等待 {wait_time:.1f}s 后重试...")
                time.sleep(wait_time)

        # 重试耗尽
        logger.error(f"API 请求 {method} {url} 重试 {self.MAX_RETRIES} 次后仍失败")
        raise last_exception  # type: ignore[misc]

    @staticmethod
    def _get_timeout() -> int:
        """从配置文件读取请求超时时间（秒），默认 30

        Returns:
            超时秒数
        """
        try:
            with open("config/config.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return int(config.get("tianyan", {}).get("timeout", 30))
        except Exception:
            return 30

    # ------------------------------------------------------------------
    # 1. 认证验证
    # ------------------------------------------------------------------

    def authenticate(self) -> bool:
        """验证 API 密钥有效性

        向 ``/auth/verify`` 端点发送验证请求，确认当前 API Key 是否合法。

        Returns:
            ``True`` 表示认证通过，``False`` 表示认证失败。

        Raises:
            TianyanAPIError: 认证请求本身出现异常（非 401 错误）
        """
        try:
            result = self._request("GET", "/auth/verify")
            logger.info("API 密钥验证通过")
            return True
        except TianyanAPIError as e:
            if e.status_code == 401:
                logger.error("API 密钥无效或已过期")
                return False
            raise
        except Exception:
            logger.error("认证请求失败，请检查网络连接")
            return False

    # ------------------------------------------------------------------
    # 2. 量子任务提交
    # ------------------------------------------------------------------

    def submit_quantum_task(
        self,
        circuit_qasm: str,
        shots: int = 1024,
        backend: str = "tianyan-287",
    ) -> str:
        """提交量子计算任务

        将 QASM 格式的量子电路提交至指定的量子后端执行。

        Args:
            circuit_qasm: QASM 格式量子电路字符串，例如::

                    OPENQASM 2.0;
                    include "qelib1.inc";
                    qreg q[2];
                    creg c[2];
                    h q[0];
                    cx q[0], q[1];
                    measure q -> c;

            shots: 重复测量次数，默认 1024
            backend: 量子后端名称，默认 ``tianyan-287``

        Returns:
            任务 ID（task_id）字符串

        Raises:
            TianyanAPIError: 提交失败时抛出

        Examples:
            >>> client = TianyanClient()
            >>> task_id = client.submit_quantum_task(
            ...     circuit_qasm="OPENQASM 2.0;\\nqreg q[1];\\ncreg c[1];\\nh q[0];\\nmeasure q -> c;",
            ...     shots=2048,
            ... )
        """
        payload = {
            "type": "quantum",
            "format": "qasm",
            "circuit": circuit_qasm,
            "shots": shots,
            "backend": backend,
        }

        result = self._request("POST", "/tasks", data=payload)
        task_id = result.get("task_id", "")
        logger.info(f"量子任务提交成功，task_id={task_id}，后端={backend}，shots={shots}")
        return task_id

    # ------------------------------------------------------------------
    # 3. 查询任务状态
    # ------------------------------------------------------------------

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """查询任务执行状态

        Args:
            task_id: 任务 ID

        Returns:
            状态字典，至少包含 ``status`` 字段，取值为
            ``PENDING`` / ``RUNNING`` / ``COMPLETED`` / ``FAILED``

        Raises:
            TianyanAPIError: 查询失败时抛出
        """
        result = self._request("GET", f"/tasks/{task_id}/status")
        logger.debug(f"任务 {task_id} 状态: {result.get('status')}")
        return result

    # ------------------------------------------------------------------
    # 4. 获取任务结果
    # ------------------------------------------------------------------

    def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """获取任务执行结果

        仅当任务状态为 ``COMPLETED`` 时返回有效测量结果。

        Args:
            task_id: 任务 ID

        Returns:
            结果字典，包含 ``counts``（测量计数）、``metadata``（元数据）等字段

        Raises:
            TianyanAPIError: 查询失败或任务尚未完成时抛出
        """
        result = self._request("GET", f"/tasks/{task_id}/result")
        logger.info(f"获取任务 {task_id} 结果成功")
        return result

    # ------------------------------------------------------------------
    # 5. 列出可用量子后端
    # ------------------------------------------------------------------

    def list_backends(self) -> List[Dict[str, Any]]:
        """列出平台上所有可用的量子计算后端

        Returns:
            后端信息列表，每个元素为字典，包含 ``name``、``type``
            （superconducting / photonic）等字段

        Raises:
            TianyanAPIError: 查询失败时抛出
        """
        result = self._request("GET", "/backends")
        backends = result.get("backends", [])
        logger.info(f"可用后端数量: {len(backends)}")
        return backends

    # ------------------------------------------------------------------
    # 6. 获取后端详细信息
    # ------------------------------------------------------------------

    def get_backend_info(self, backend_name: str) -> Dict[str, Any]:
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
        result = self._request("GET", f"/backends/{backend_name}")
        logger.debug(f"后端 {backend_name} 信息: {result.get('num_qubits')} qubits")
        return result

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
        payload = {
            "type": "classical",
            "language": language,
            "code": code,
        }

        result = self._request("POST", "/tasks", data=payload)
        task_id = result.get("task_id", "")
        logger.info(f"经典任务提交成功，task_id={task_id}，语言={language}")
        return task_id

    # ------------------------------------------------------------------
    # 8. 获取队列状态
    # ------------------------------------------------------------------

    def get_queue_status(self) -> Dict[str, Any]:
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
        result = self._request("GET", "/queue/status")
        logger.info(f"队列状态: {result.get('total_pending', 0)} 待执行, "
                     f"{result.get('total_running', 0)} 执行中")
        return result

    # ------------------------------------------------------------------
    # 便捷方法：等待任务完成
    # ------------------------------------------------------------------

    def wait_for_task(
        self,
        task_id: str,
        poll_interval: float = 5.0,
        timeout: float = 3600.0,
    ) -> Dict[str, Any]:
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
