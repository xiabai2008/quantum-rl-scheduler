"""cqlib 录制/回放客户端 — 为天衍真机测试提供确定性回放能力。

本模块实现了两种客户端，共同构成"录制-回放"测试框架（Issue #175）：

1. ``CqlibReplayClient`` —— 从 JSON fixtures 加载响应，无需安装 cqlib SDK
   即可运行真机交互逻辑的回归测试。作为 ``CqlibTianyanClient`` 的回放替代品。

2. ``CqlibRecordingClient`` —— 包装真实 ``CqlibTianyanClient``，在调用真机
   API 的同时将响应序列化为 JSON fixtures，供后续回放使用。

典型工作流::

    # 录制阶段（需真机凭证 + cqlib SDK）
    real = CqlibTianyanClient(login_key="xxx", machine_name="tianyan_s")
    recorder = CqlibRecordingClient(real, "tests/fixtures/cqlib_responses")
    recorder.list_backends()
    tid = recorder.submit_quantum_task(qcis="H Q0\\nM Q0", shots=1024)
    recorder.wait_for_task(tid)

    # 回放阶段（无需 cqlib SDK，CI 可直接运行）
    replay = CqlibReplayClient("tests/fixtures/cqlib_responses")
    assert replay.authenticate() is True
    backends = replay.list_backends()
    tid = replay.submit_quantum_task(qcis="H Q0\\nM Q0", shots=1024)
    result = replay.wait_for_task(tid)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.api.types import TaskResult

if TYPE_CHECKING:
    from src.api.tianyan_cqlib import CqlibTianyanClient


class CqlibReplayClient:
    """cqlib 回放客户端：从 JSON fixtures 加载响应，替代真实 SDK 调用。

    作为 ``CqlibTianyanClient`` 的回放替代品，无需安装 cqlib SDK 即可运行
    真机交互逻辑的回归测试。所有响应来自 ``fixtures_dir`` 下的 JSON 文件。

    任务状态采用轮询计数状态机模拟真实行为：

    - 首次查询某 ``task_id`` 返回 ``running`` 状态
    - 后续查询返回 ``completed`` 状态

    这使得 ``wait_for_task`` 能在回放中自然完成"等待→完成"流转。

    Attributes:
        fixtures_dir: JSON fixtures 目录路径
        machine_name: 默认机器名（影响可用性判断与重试逻辑）
        auto_retry_machine: 机器不可用时是否自动切换备用机
    """

    # 已知可用的超导真机（与 CqlibTianyanClient.REAL_MACHINES 保持一致）
    REAL_MACHINES: list[str] = [  # noqa: RUF012
        "tianyan-287",
        "tianyan_sw",
        "tianyan_s",
        "tianyan_tn",
        "tianyan_tnn",
        "tianyan_swn",
        "tianyan_sa",
        "tianyan176",
        "tianyan176-2",
    ]

    def __init__(
        self,
        fixtures_dir: str = "tests/fixtures/cqlib_responses",
        *,
        machine_name: str = "tianyan_s",
        auto_retry_machine: bool = True,
        error_mode: str | None = None,
    ) -> None:
        """初始化回放客户端并加载全部 fixtures。

        Args:
            fixtures_dir: JSON fixtures 目录路径
            machine_name: 默认机器名（用于可用性判断与重试逻辑）
            auto_retry_machine: 机器不可用时是否自动切换备用机
            error_mode: 错误注入模式（仅用于测试）。
                ``"capacity"`` 模拟机时包容量不足；
                ``"unavailable"`` 模拟所有机器不可用；
                ``None`` 为正常回放模式。
        """
        self.fixtures_dir = Path(fixtures_dir)
        self.machine_name = machine_name
        self.auto_retry_machine = auto_retry_machine
        self._error_mode = error_mode
        self._task_polls: dict[str, int] = {}

        self._machine_list: list[dict[str, Any]] = self._load("machine_list.json")["backends"]
        self._task_submit: dict[str, Any] = self._load("task_submit.json")
        self._status_running: dict[str, Any] = self._load("task_status_running.json")
        self._status_completed: dict[str, Any] = self._load("task_status_completed.json")
        self._task_result: dict[str, Any] = self._load("task_result.json")

        logger.info(
            f"[Replay] 回放客户端已加载 {len(self._machine_list)} 台机器，"
            f"fixtures_dir={self.fixtures_dir}"
        )

    def _load(self, name: str) -> dict[str, Any]:
        """从 fixtures 目录加载指定 JSON 文件。

        Args:
            name: JSON 文件名

        Returns:
            解析后的字典

        Raises:
            FileNotFoundError: fixture 文件不存在
            json.JSONDecodeError: JSON 格式错误
        """
        path = self.fixtures_dir / name
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        return data

    def authenticate(self) -> bool:
        """回放认证（总是返回 True）。

        Returns:
            始终返回 ``True``，模拟认证成功
        """
        logger.debug("[Replay] authenticate 回放 -> True")
        return True

    def list_backends(self) -> list[dict[str, Any]]:
        """从 machine_list.json 加载量子计算机列表。

        Returns:
            后端字典列表，每项含 ``id``/``type``/``status``/``name``
        """
        backends = [dict(b) for b in self._machine_list]
        logger.debug(f"[Replay] list_backends 回放 -> {len(backends)} 台机器")
        return backends

    def get_backend_info(self, backend_name: str | None = None) -> dict[str, Any]:
        """获取指定后端信息。

        Args:
            backend_name: 后端名称，为 None 时使用默认机器名

        Returns:
            匹配的后端字典；未找到返回空字典
        """
        name = backend_name or self.machine_name
        for m in self.list_backends():
            if m["name"] == name:
                return m
        return {}

    def _is_machine_available(self, machine_name: str) -> bool:
        """检查机器是否在线可用（status == running）。

        基于回放 fixtures 中的机器列表判断，逻辑与
        ``CqlibTianyanClient._is_machine_available`` 一致。

        Args:
            machine_name: 机器名

        Returns:
            running 返回 True，其他状态返回 False；未找到时乐观返回 True
        """
        for m in self._machine_list:
            if m.get("name") == machine_name:
                return m.get("status") == "running"
        return True

    def submit_quantum_task(
        self,
        qcis: str = "",
        circuit: Any = None,
        shots: int = 1024,
        task_name: str = "Scheduler_Task",
    ) -> str | None:
        """从 task_submit.json 加载任务提交响应。

        模拟真实提交逻辑：预检机器可用性、容量错误处理、备用机切换。
        当 ``error_mode="capacity"`` 时直接返回 None（模拟机时包容量不足）。

        Args:
            qcis: QCIS 指令字符串
            circuit: cqlib.Circuit 对象（与 qcis 二选一，回放中不使用）
            shots: 测量次数
            task_name: 任务名称

        Returns:
            task_id 字符串；机器不可用或容量不足时返回 None
        """
        # 容量错误注入：电路超出免费额度，换机器也无效
        if self._error_mode == "capacity":
            logger.warning("[Replay] 容量不足，跳过提交（error_mode=capacity）")
            return None

        # 预检当前机器状态
        if not self._is_machine_available(self.machine_name):
            logger.warning(f"[Replay] {self.machine_name} 不可用（校准/维护中）")
            if self.auto_retry_machine:
                return self._retry_other_machine(task_name)
            return None

        task_id = str(self._task_submit["task_id"])
        self._task_polls[task_id] = 0
        logger.info(f"[Replay] 提交任务回放: {task_name} -> {task_id}, shots={shots}")
        return task_id

    def _retry_other_machine(self, task_name: str) -> str | None:
        """当前机器不可用时，按 REAL_MACHINES 列表尝试其他机器。

        Args:
            task_name: 任务名称（仅用于日志）

        Returns:
            task_id 字符串；全部不可用返回 None
        """
        if self._error_mode == "unavailable":
            logger.error("[Replay] 所有机器不可用（error_mode=unavailable）")
            return None

        for machine in self.REAL_MACHINES:
            if machine == self.machine_name:
                continue
            if not self._is_machine_available(machine):
                continue
            task_id = str(self._task_submit["task_id"])
            self._task_polls[task_id] = 0
            logger.info(f"[Replay] 备用机器 {machine} 提交成功: {task_id}")
            return task_id

        logger.error("[Replay] 所有备用机器均不可用，放弃提交")
        return None

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """根据 task_id 轮询计数返回 running 或 completed 状态。

        状态机：首次查询返回 ``running``，后续查询返回 ``completed``。
        这模拟了真实场景中任务从运行到完成的自然流转。

        Args:
            task_id: 任务 ID

        Returns:
            状态字典，含 ``task_id``/``status``/``result``/``raw``
        """
        polls = self._task_polls.get(task_id, 0)
        self._task_polls[task_id] = polls + 1

        status = dict(self._status_running) if polls < 1 else dict(self._status_completed)

        status["task_id"] = task_id
        logger.debug(
            f"[Replay] get_task_status({task_id}) 第{polls + 1}次查询 -> {status['status']}"
        )
        return status

    def get_task_result(self, task_id: str) -> dict[str, Any]:
        """获取任务执行结果（从 task_result.json 加载）。

        Args:
            task_id: 任务 ID

        Returns:
            结果字典，含 ``task_id``/``status``/``result``/``raw``
        """
        result = dict(self._task_result)
        result["task_id"] = task_id
        logger.debug(f"[Replay] get_task_result({task_id}) -> {result['status']}")
        return result

    def wait_for_task(
        self, task_id: str, timeout: int = 300, poll_interval: int = 5
    ) -> dict[str, Any]:
        """轮询等待任务完成并返回结果。

        回放中通过 ``get_task_status`` 的轮询计数状态机实现：
        首次查询返回 running，第二次返回 completed 即结束等待。

        Args:
            task_id: 任务 ID
            timeout: 超时秒数
            poll_interval: 轮询间隔秒数

        Returns:
            完成状态字典；超时返回 ``{"task_id": ..., "status": "timeout"}``
        """
        start = time.time()
        while time.time() - start < timeout:
            status = self.get_task_status(task_id)
            if status["status"] == "completed":
                return status
            if status["status"] == "error":
                return status
            time.sleep(poll_interval)
        return {"task_id": task_id, "status": "timeout"}

    def get_queue_status(self) -> dict[str, Any]:
        """获取队列状态（基于 fixtures 机器列表统计）。

        Returns:
            含 ``total_machines``/``running``/``available`` 的字典
        """
        machines = self.list_backends()
        running = sum(1 for m in machines if m.get("status") == "running")
        return {
            "total_machines": len(machines),
            "running": running,
            "available": [m["name"] for m in machines if m["status"] == "running"],
        }

    def is_available(self) -> bool:
        """检查真机是否可用（回放中认证通过且默认机器 running）。

        Returns:
            True 表示可提交，False 表示应降级
        """
        return self.authenticate() and self._is_machine_available(self.machine_name)

    def submit_and_get_task_id(
        self,
        qcis: str,
        shots: int = 512,
        task_name: str = "Scheduler_Real_Task",
    ) -> str | None:
        """提交量子任务并立即返回 task_id（非阻塞，语义化别名）。

        Args:
            qcis: QCIS 指令字符串
            shots: 测量次数
            task_name: 任务名称

        Returns:
            task_id 字符串；提交失败返回 None
        """
        return self.submit_quantum_task(qcis=qcis, shots=shots, task_name=task_name)


class CqlibRecordingClient:
    """cqlib 录制客户端：包装真实 ``CqlibTianyanClient`` 并录制响应。

    所有方法调用转发给真实客户端，同时将响应序列化为 JSON fixtures
    保存到 ``fixtures_dir``，供 ``CqlibReplayClient`` 后续回放使用。

    录制时自动处理不可 JSON 序列化的对象（如 cqlib 自定义类型），
    将其递归转为字符串表示。

    使用示例::

        real = CqlibTianyanClient(login_key="xxx", machine_name="tianyan_s")
        recorder = CqlibRecordingClient(real, "tests/fixtures/cqlib_responses")
        backends = recorder.list_backends()
        tid = recorder.submit_quantum_task(qcis="H Q0\\nM Q0", shots=1024)
        recorder.wait_for_task(tid)

    Attributes:
        fixtures_dir: JSON fixtures 输出目录
    """

    def __init__(self, real_client: CqlibTianyanClient, fixtures_dir: str) -> None:
        """初始化录制客户端。

        Args:
            real_client: 真实 ``CqlibTianyanClient`` 实例（需已配置凭证）
            fixtures_dir: JSON fixtures 输出目录（不存在则自动创建）
        """
        self._client = real_client
        self.fixtures_dir = Path(fixtures_dir)
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[Record] 录制客户端已初始化，输出目录={self.fixtures_dir}")

    @staticmethod
    def _serialize(obj: Any) -> Any:
        """将不可 JSON 序列化的对象递归转为可序列化形式。

        cqlib SDK 返回的 ``raw`` 字段可能包含自定义类型，需在录制前
        转为 JSON 兼容格式。

        Args:
            obj: 任意 Python 对象

        Returns:
            JSON 可序列化的对象
        """
        if isinstance(obj, dict):
            return {str(k): CqlibRecordingClient._serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [CqlibRecordingClient._serialize(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    def _save(self, name: str, data: dict[str, Any]) -> None:
        """将响应数据序列化保存为 JSON fixture 文件。

        Args:
            name: 输出文件名
            data: 响应数据字典
        """
        path = self.fixtures_dir / name
        serialized = self._serialize(data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)
        logger.info(f"[Record] 已录制 fixture: {name}")

    def authenticate(self) -> bool:
        """录制认证结果。

        Returns:
            真实客户端的认证结果
        """
        result = self._client.authenticate()
        self._save("authenticate.json", {"description": "认证响应", "success": result})
        return result

    def list_backends(self) -> list[dict[str, Any]]:
        """录制量子计算机列表响应。

        Returns:
            真实客户端返回的后端字典列表
        """
        backends = self._client.list_backends()
        self._save(
            "machine_list.json",
            {
                "description": "天衍云量子计算机列表查询响应（录制）",
                "api_method": "platform.query_quantum_computer_list",
                "backends": backends,
            },
        )
        return backends

    def get_backend_info(self, backend_name: str | None = None) -> dict[str, Any]:
        """录制后端信息查询（不单独保存 fixture，随 list_backends 录制）。"""
        return self._client.get_backend_info(backend_name)

    def submit_quantum_task(
        self,
        qcis: str = "",
        circuit: Any = None,
        shots: int = 1024,
        task_name: str = "Scheduler_Task",
    ) -> str | None:
        """录制任务提交响应。

        Args:
            qcis: QCIS 指令字符串
            circuit: cqlib.Circuit 对象（与 qcis 二选一）
            shots: 测量次数
            task_name: 任务名称

        Returns:
            真实客户端返回的 task_id；失败返回 None
        """
        task_id = self._client.submit_quantum_task(
            qcis=qcis, circuit=circuit, shots=shots, task_name=task_name
        )
        self._save(
            "task_submit.json",
            {
                "description": "任务提交响应（录制）",
                "api_method": "platform.submit_experiment",
                "task_id": task_id,
                "task_name": task_name,
                "shots": shots,
                "machine_name": getattr(self._client, "machine_name", "unknown"),
                "qcis_preview": qcis[:100] if qcis else None,
            },
        )
        return task_id

    def get_task_status(self, task_id: str) -> TaskResult:
        """录制任务状态查询响应。

        根据 status 自动保存到 running 或 completed fixture 文件。

        Args:
            task_id: 任务 ID

        Returns:
            真实客户端返回的状态字典
        """
        status = self._client.get_task_status(task_id)
        label = status.get("status", "unknown")
        if label == "running":
            self._save(
                "task_status_running.json",
                {
                    "description": "任务运行中状态响应（录制）",
                    "api_method": "platform.query_experiment",
                    **status,
                },
            )
        elif label == "completed":
            self._save(
                "task_status_completed.json",
                {
                    "description": "任务完成状态响应（录制）",
                    "api_method": "platform.query_experiment",
                    **status,
                },
            )
        else:
            # Issue #720: error/query_error 等非完成状态不误存为 completed fixture
            logger.warning(f"[Recorder] 任务 {task_id} 状态为 {label}，不保存为 completed fixture")
        return status

    def get_task_result(self, task_id: str) -> TaskResult:
        """录制任务结果查询响应。

        Args:
            task_id: 任务 ID

        Returns:
            真实客户端返回的结果字典
        """
        result = self._client.get_task_result(task_id)
        self._save(
            "task_result.json",
            {
                "description": "任务结果查询响应（录制）",
                "api_method": "platform.query_experiment",
                **result,
            },
        )
        return result

    def wait_for_task(self, task_id: str, timeout: int = 300, poll_interval: int = 5) -> TaskResult:
        """录制等待任务完成的最终结果。

        轮询过程中的中间状态由 ``get_task_status`` 录制，
        此方法额外录制最终完成结果。

        Args:
            task_id: 任务 ID
            timeout: 超时秒数
            poll_interval: 轮询间隔秒数

        Returns:
            最终状态字典
        """
        result = self._client.wait_for_task(task_id, timeout=timeout, poll_interval=poll_interval)
        if result.get("status") == "completed":
            self._save(
                "task_status_completed.json",
                {
                    "description": "任务完成状态响应（wait_for_task 录制）",
                    "api_method": "platform.query_experiment",
                    **result,
                },
            )
        return result
