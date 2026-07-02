"""
天衍云平台 Mock API 客户端
Tianyan Cloud Platform Mock API Client

在开发阶段模拟天衍云平台的 API 响应，支持完整的任务生命周期模拟。
可用于离线开发、单元测试、CI/CD 流水线。

功能：
- 模拟量子任务提交（返回虚拟 task_id）
- 模拟任务状态轮转（PENDING → RUNNING → COMPLETED）
- 模拟量子计算结果（随机测量计数）
- 模拟后端信息查询
- 模拟队列状态查询

使用示例::

    from src.api.mock_client import MockTianyanClient

    client = MockTianyanClient()
    task_id = client.submit_quantum_task(circuit_qasm="OPENQASM 2.0;...")
    status = client.get_task_status(task_id)
    result = client.get_task_result(task_id)

作者：揭榜挂帅擂台赛团队
版本：1.0.0
日期：2026-06-27
"""

import os
import random
import time
import uuid
from datetime import datetime
from typing import Any, cast

from loguru import logger


class MockTianyanClient:
    """天衍云平台 Mock 客户端

    模拟天衍云平台的全部 API 接口，返回与真实 API 相同格式的数据。
    所有数据在内存中管理，任务状态会自动轮转。

    配置方式：
    - 设置环境变量 ``TIANYAN_MOCK_MODE=true`` 启用 Mock 模式
    - 设置环境变量 ``TIANYAN_MOCK_DELAY=2.0`` 调整模拟延迟（秒）
    - 设置环境变量 ``TIANYAN_MOCK_FAILURE_RATE=0.05`` 调整失败率（0-1）

    Attributes:
        mock_delay: 模拟网络延迟（秒），默认 1.0
        mock_failure_rate: 模拟失败率（0-1），默认 0.0（不失败）
        _tasks: 内存中的任务存储（task_id -> task_info）
        _backend_status: 模拟的后端状态
    """

    def __init__(
        self,
        mock_delay: float = 1.0,
        mock_failure_rate: float = 0.0,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """初始化 Mock 客户端

        Args:
            mock_delay: 模拟网络延迟（秒），默认 1.0
            mock_failure_rate: 模拟失败率（0-1），默认 0.0
            api_key: API 密钥（Mock 模式下可随意填写）
            base_url: API 基础 URL（Mock 模式下不使用）
        """
        self.mock_delay = mock_delay
        self.mock_failure_rate = mock_failure_rate
        self.api_key = api_key or os.getenv("TIANYAN_API_KEY", "mock-api-key")

        # 内存中的任务存储
        self._tasks: dict[str, dict[str, Any]] = {}

        # 模拟的后端信息
        self._backends = [
            {
                "name": "tianyan-287",
                "type": "superconducting",
                "num_qubits": 287,
                "fidelity": {
                    "single_qubit_gate": 0.9995,
                    "two_qubit_gate": 0.992,
                    "readout": 0.985,
                },
                "queue_depth": random.randint(5, 50),
                "status": "online",
                "max_shots": 10000,
                "calibration_date": "2026-06-20",
            },
            {
                "name": "tianyan-simulator",
                "type": "simulator",
                "num_qubits": 30,
                "fidelity": {
                    "single_qubit_gate": 1.0,
                    "two_qubit_gate": 1.0,
                    "readout": 1.0,
                },
                "queue_depth": 0,
                "status": "online",
                "max_shots": 1000000,
                "calibration_date": "N/A",
            },
        ]

        logger.info(f"Mock 天衍客户端初始化完成，延迟={mock_delay}s，失败率={mock_failure_rate}")

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    def _simulate_delay(self) -> None:
        """模拟网络延迟"""
        if self.mock_delay > 0:
            time.sleep(self.mock_delay)

    def _maybe_fail(self, operation: str) -> None:
        """按失败率随机抛出异常"""
        if random.random() < self.mock_failure_rate:
            from src.api.tianyan_client import TianyanAPIError

            raise TianyanAPIError(
                status_code=500,
                message=f"Mock 模拟失败: {operation}",
                response_body={"error": "mock_failure"},
            )

    def _generate_mock_result(self, circuit_qasm: str, shots: int) -> dict[str, Any]:
        """生成模拟的量子测量结果

        根据 QASM 电路生成合理的模拟结果。
        对于已知电路（如 Bell 态），返回确定的结果；
        对于未知电路，返回随机结果。

        Args:
            circuit_qasm: QASM 电路字符串
            shots: 测量次数

        Returns:
            包含 counts 的测量结果字典
        """
        # 简单启发式：根据电路内容生成合理结果
        num_qubits = circuit_qasm.count("qreg q[")
        if num_qubits == 0:
            num_qubits = 2  # 默认 2 qubit

        # 生成随机测量计数
        counts = {}

        # 限制最大 qubit 数，防止内存爆炸（2^20 ≈ 100万，可接受）
        MAX_QUBITS_FOR_FULL_ENUM = 20  # noqa: N806
        if num_qubits <= MAX_QUBITS_FOR_FULL_ENUM:
            #  qubit 数较少时使用全排列
            possible_states = [bin(i)[2:].zfill(num_qubits) for i in range(2**num_qubits)]

            # 随机分配 shots
            remaining_shots = shots
            for i, state in enumerate(possible_states):
                if i == len(possible_states) - 1:
                    counts[state] = remaining_shots
                else:
                    count = random.randint(0, remaining_shots)
                    counts[state] = count
                    remaining_shots -= count
        else:
            # qubit 数较多时使用随机采样，避免全排列内存爆炸
            num_samples = min(shots, 1000)  # 最多采样 1000 个不同状态
            for _ in range(num_samples):
                state = "".join(random.choice("01") for _ in range(num_qubits))
                if state not in counts:
                    counts[state] = 0
            # 分配 shots
            remaining_shots = shots
            states_list = list(counts.keys())
            for i, state in enumerate(states_list):
                if i == len(states_list) - 1:
                    counts[state] = remaining_shots
                else:
                    count = random.randint(0, remaining_shots)
                    counts[state] = count
                    remaining_shots -= count

        # 如果是 Bell 态电路（包含 h 和 cx），调整结果使其更接近 |00> + |11>
        if "h q[0]" in circuit_qasm and "cx q[0], q[1]" in circuit_qasm:
            counts = {
                "00": int(shots * 0.5),
                "11": int(shots * 0.5),
            }

        return {
            "task_id": "",  # 将在 submit 时填充
            "status": "COMPLETED",
            "backend": "tianyan-287",
            "shots": shots,
            "counts": counts,
            "metadata": {
                "execution_time": random.uniform(0.5, 5.0),
                "queue_time": random.uniform(10.0, 120.0),
                "timestamp": datetime.now().isoformat(),
            },
        }

    # ------------------------------------------------------------------
    # 1. 认证验证（Mock）
    # ------------------------------------------------------------------

    def authenticate(self) -> bool:
        """Mock 认证验证（始终返回 True）

        Returns:
            始终返回 True（Mock 模式不需要真实认证）
        """
        self._simulate_delay()
        self._maybe_fail("authenticate")

        logger.info("Mock 认证验证通过")
        return True

    # ------------------------------------------------------------------
    # 2. 量子任务提交（Mock）
    # ------------------------------------------------------------------

    def submit_quantum_task(
        self,
        circuit_qasm: str,
        shots: int = 1024,
        backend: str = "tianyan-287",
    ) -> str:
        """Mock 提交量子计算任务

        Args:
            circuit_qasm: QASM 格式量子电路字符串
            shots: 重复测量次数
            backend: 量子后端名称

        Returns:
            虚拟 task_id 字符串（格式：mock-{uuid}）
        """
        self._simulate_delay()
        self._maybe_fail("submit_quantum_task")

        # 生成虚拟 task_id
        task_id = f"mock-{uuid.uuid4().hex[:12]}"

        # 存储任务信息
        self._tasks[task_id] = {
            "task_id": task_id,
            "type": "quantum",
            "status": "PENDING",
            "backend": backend,
            "shots": shots,
            "circuit_qasm": circuit_qasm,
            "submitted_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "result": None,
        }

        # 异步模拟任务执行（在实际场景中，这里应该启动一个线程）
        # 为简化，我们在 get_task_status 中模拟状态轮转

        logger.info(f"Mock 量子任务提交成功，task_id={task_id}，后端={backend}")
        return task_id

    # ------------------------------------------------------------------
    # 3. 查询任务状态（Mock）
    # ------------------------------------------------------------------

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Mock 查询任务执行状态

        模拟任务状态轮转：
        - PENDING（0-2次查询）
        - RUNNING（3-5次查询）
        - COMPLETED（6+次查询）

        Args:
            task_id: 任务 ID

        Returns:
            状态字典

        Raises:
            ValueError: task_id 不存在时抛出
        """
        self._simulate_delay()
        self._maybe_fail("get_task_status")

        if task_id not in self._tasks:
            raise ValueError(f"Mock 任务 {task_id} 不存在")

        task = self._tasks[task_id]

        # 模拟状态轮转
        if task["status"] == "PENDING":
            # 随机决定是否进入 RUNNING
            if random.random() < 0.3:  # 30% 概率进入 RUNNING
                task["status"] = "RUNNING"
                task["started_at"] = datetime.now().isoformat()
                logger.debug(f"Mock 任务 {task_id} 状态变更: PENDING → RUNNING")

        elif task["status"] == "RUNNING":  # noqa: SIM102
            # 随机决定是否进入 COMPLETED
            if random.random() < 0.4:  # 40% 概率进入 COMPLETED
                task["status"] = "COMPLETED"
                task["completed_at"] = datetime.now().isoformat()

                # 生成模拟结果
                task["result"] = self._generate_mock_result(
                    circuit_qasm=task["circuit_qasm"],
                    shots=task["shots"],
                )
                task["result"]["task_id"] = task_id

                logger.debug(f"Mock 任务 {task_id} 状态变更: RUNNING → COMPLETED")

        logger.debug(f"Mock 任务 {task_id} 状态: {task['status']}")
        return {"task_id": task_id, "status": task["status"]}

    # ------------------------------------------------------------------
    # 4. 获取任务结果（Mock）
    # ------------------------------------------------------------------

    def get_task_result(self, task_id: str) -> dict[str, Any]:
        """Mock 获取任务执行结果

        Args:
            task_id: 任务 ID

        Returns:
            结果字典

        Raises:
            ValueError: task_id 不存在或任务未完成时抛出
        """
        self._simulate_delay()
        self._maybe_fail("get_task_result")

        if task_id not in self._tasks:
            raise ValueError(f"Mock 任务 {task_id} 不存在")

        task = self._tasks[task_id]

        if task["status"] != "COMPLETED":
            # 自动触发状态轮转，确保能获取到结果
            self.get_task_status(task_id)
            if task["status"] != "COMPLETED":
                raise ValueError(f"Mock 任务 {task_id} 尚未完成，当前状态: {task['status']}")

        logger.info(f"Mock 获取任务 {task_id} 结果成功")
        return cast(dict[str, Any], task["result"])

    # ------------------------------------------------------------------
    # 5. 列出可用量子后端（Mock）
    # ------------------------------------------------------------------

    def list_backends(self) -> list[dict[str, Any]]:
        """Mock 列出平台上所有可用的量子计算后端

        Returns:
            后端信息列表
        """
        self._simulate_delay()
        self._maybe_fail("list_backends")

        logger.info(f"Mock 可用后端数量: {len(self._backends)}")
        return self._backends

    # ------------------------------------------------------------------
    # 6. 获取后端详细信息（Mock）
    # ------------------------------------------------------------------

    def get_backend_info(self, backend_name: str) -> dict[str, Any]:
        """Mock 获取指定量子后端的详细信息

        Args:
            backend_name: 后端名称

        Returns:
            后端详情字典

        Raises:
            ValueError: 后端不存在时抛出
        """
        self._simulate_delay()
        self._maybe_fail("get_backend_info")

        for backend in self._backends:
            if backend["name"] == backend_name:
                logger.debug(f"Mock 后端 {backend_name} 信息")
                return backend

        raise ValueError(f"Mock 后端 {backend_name} 不存在")

    # ------------------------------------------------------------------
    # 7. 提交经典计算任务（Mock）
    # ------------------------------------------------------------------

    def submit_classical_task(self, code: str, language: str = "python3") -> str:
        """Mock 提交经典计算任务

        Args:
            code: 要执行的代码字符串
            language: 编程语言

        Returns:
            虚拟 task_id 字符串
        """
        self._simulate_delay()
        self._maybe_fail("submit_classical_task")

        task_id = f"mock-classical-{uuid.uuid4().hex[:12]}"

        self._tasks[task_id] = {
            "task_id": task_id,
            "type": "classical",
            "status": "COMPLETED",  # 经典任务立即完成
            "language": language,
            "code": code,
            "submitted_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "result": {"output": "Mock 执行结果", "exit_code": 0},
        }

        logger.info(f"Mock 经典任务提交成功，task_id={task_id}")
        return task_id

    # ------------------------------------------------------------------
    # 8. 获取队列状态（Mock）
    # ------------------------------------------------------------------

    def get_queue_status(self) -> dict[str, Any]:
        """Mock 获取当前平台任务队列状态

        Returns:
            队列状态字典
        """
        self._simulate_delay()
        self._maybe_fail("get_queue_status")

        # 统计内存中的任务
        pending_count = sum(1 for t in self._tasks.values() if t["status"] == "PENDING")
        running_count = sum(1 for t in self._tasks.values() if t["status"] == "RUNNING")

        result = {
            "total_pending": pending_count,
            "total_running": running_count,
            "queue_capacity": 1000,
            "estimated_wait_time": pending_count * 30,  # 每个任务约 30 秒
            "by_backend": {
                "tianyan-287": {
                    "pending": pending_count,
                    "running": running_count,
                    "capacity": 100,
                },
                "tianyan-simulator": {
                    "pending": 0,
                    "running": 0,
                    "capacity": 1000,
                },
            },
        }

        logger.info(f"Mock 队列状态: {pending_count} 待执行, {running_count} 执行中")
        return result

    # ------------------------------------------------------------------
    # 便捷方法：等待任务完成（Mock）
    # ------------------------------------------------------------------

    def wait_for_task(
        self,
        task_id: str,
        poll_interval: float = 1.0,  # Mock 模式下轮询间隔更短
        timeout: float = 60.0,  # Mock 模式下超时更短
    ) -> dict[str, Any]:
        """Mock 轮询等待任务完成并返回结果

        Args:
            task_id: 任务 ID
            poll_interval: 轮询间隔（秒），默认 1.0
            timeout: 最大等待时间（秒），默认 60.0

        Returns:
            任务最终结果字典
        """
        elapsed = 0.0
        while elapsed < timeout:
            status_info = self.get_task_status(task_id)
            status = status_info.get("status", "UNKNOWN")

            if status == "COMPLETED":
                logger.info(f"Mock 任务 {task_id} 已完成")
                return self.get_task_result(task_id)
            elif status == "FAILED":
                raise RuntimeError(f"Mock 任务 {task_id} 执行失败")

            logger.debug(f"Mock 任务 {task_id} 状态={status}，{poll_interval}s 后再次查询")
            time.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Mock 任务 {task_id} 等待超时（{timeout}s）")


# ======================================================================
# 工厂函数：根据配置创建客户端（真实或 Mock）
# ======================================================================


def create_tianyan_client(mock_mode: bool | None = None) -> Any:
    """工厂函数：根据配置创建天衍云客户端

    优先读取顺序：
    1. 显式传参 ``mock_mode``
    2. 环境变量 ``TIANYAN_MOCK_MODE``
    3. 配置文件 ``config/config.yaml`` 中的 ``tianyan.mock_mode``

    Args:
        mock_mode: 是否使用 Mock 模式（None 表示自动检测）

    Returns:
        TianyanClient 或 MockTianyanClient 实例

    Examples:
        >>> # 自动检测配置
        >>> client = create_tianyan_client()
        >>>
        >>> # 强制使用 Mock 模式
        >>> client = create_tianyan_client(mock_mode=True)
    """
    # 确定是否使用 Mock 模式
    if mock_mode is None:
        mock_mode_env = os.getenv("TIANYAN_MOCK_MODE", "").lower()
        if mock_mode_env in ("true", "1", "yes"):
            mock_mode = True
        else:
            # 从配置文件读取（使用基于 __file__ 的绝对路径）
            try:
                import yaml

                config_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "config",
                    "config.yaml",
                )
                with open(config_path, encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                mock_mode = config.get("tianyan", {}).get("mock_mode", True)
            except (ImportError, yaml.YAMLError, OSError, AttributeError) as e:
                # yaml 未安装、文件读取失败、YAML 解析错误或配置结构异常时默认 Mock 模式
                logger.debug(f"读取 mock_mode 配置失败: {e}，默认使用 Mock 模式")
                mock_mode = True  # 默认使用 Mock 模式

    # 创建客户端
    if mock_mode:
        mock_delay = float(os.getenv("TIANYAN_MOCK_DELAY", "1.0"))
        mock_failure_rate = float(os.getenv("TIANYAN_MOCK_FAILURE_RATE", "0.0"))

        logger.info(f"使用 Mock 模式（延迟={mock_delay}s，失败率={mock_failure_rate}）")
        return MockTianyanClient(
            mock_delay=mock_delay,
            mock_failure_rate=mock_failure_rate,
        )
    else:
        from src.api.tianyan_client import TianyanClient

        logger.info("使用真实 API 模式")
        return TianyanClient()


if __name__ == "__main__":
    # 测试 Mock 客户端
    client = MockTianyanClient(mock_delay=0.5)

    # 验证认证
    print("认证验证:", client.authenticate())

    # 提交量子任务
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """

    task_id = client.submit_quantum_task(circuit_qasm=qasm, shots=1024)
    print(f"任务提交成功，task_id={task_id}")

    # 等待任务完成
    result = client.wait_for_task(task_id, poll_interval=0.5, timeout=10.0)
    print(f"任务结果: {result}")

    # 查询后端信息
    backends = client.list_backends()
    print(f"可用后端: {[b['name'] for b in backends]}")

    print("\n✅ Mock 客户端测试通过！")
