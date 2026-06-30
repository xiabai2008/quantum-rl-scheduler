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

import time
from typing import Any, Dict, List, Optional

from loguru import logger


class CqlibTianyanClient:
    """基于 cqlib SDK 的天衍云真机客户端

    直接调用天衍云超导量子计算机执行量子电路。

    使用示例::

        client = CqlibTianyanClient(login_key="your_key")
        task_id = client.submit_quantum_task(qcis="H Q0\\nM Q0", shots=1024)
        result = client.wait_for_task(task_id)
    """

    # 已知可用的超导真机
    REAL_MACHINES = [
        "tianyan_sw",    # 超导 free
        "tianyan_s",     # 超导 free
        "tianyan_tn",    # 超导 free
        "tianyan_tnn",   # 超导 free
        "tianyan_swn",   # 超导 free
        "tianyan_sa",    # 超导 free
        "tianyan176",    # 176比特 free
        "tianyan176-2",  # 176比特 free
    ]

    def __init__(
        self,
        login_key: str,
        machine_name: str = "tianyan_s",
        auto_retry_machine: bool = True,
    ):
        """初始化 cqlib 客户端

        Args:
            login_key: API Key（从个人中心获取）
            machine_name: 默认使用的量子计算机名称
            auto_retry_machine: 当前机器不可用时是否自动切换
        """
        import cqlib

        self.cqlib = cqlib
        self.login_key = login_key
        self.machine_name = machine_name
        self.auto_retry_machine = auto_retry_machine
        self._platform = None

        logger.info(f"[Cqlib] 客户端初始化，默认机器={machine_name}")

    @property
    def platform(self):
        """懒加载平台连接"""
        if self._platform is None:
            self._platform = self.cqlib.TianYanPlatform(
                login_key=self.login_key,
                machine_name=self.machine_name,
            )
        return self._platform

    def authenticate(self) -> bool:
        """验证 API Key 有效性"""
        try:
            _ = self.platform
            return True
        except Exception as e:
            logger.error(f"[Cqlib] 认证失败: {e}")
            return False

    def list_backends(self) -> List[Dict[str, Any]]:
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
        except Exception as e:
            logger.error(f"[Cqlib] 获取机器列表失败: {e}")
            return []

    def get_backend_info(self, backend_name: Optional[str] = None) -> Dict[str, Any]:
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
    ) -> str:
        """提交量子任务到真机

        Args:
            qcis: QCIS 指令字符串（"H Q0\\nM Q0"）
            circuit: cqlib.Circuit 对象（与 qcis 二选一）
            shots: 测量次数
            task_name: 任务名称

        Returns:
            task_id: 任务 ID 字符串
        """
        # 生成 QCIS
        if qcis:
            qcis_str = qcis
        elif circuit is not None:
            qcis_str = circuit.qcis if hasattr(circuit, "qcis") else str(circuit)
        else:
            raise ValueError("必须提供 qcis 或 circuit")

        logger.info(f"[Cqlib] 提交量子任务: {task_name}, shots={shots}")
        logger.debug(f"[Cqlib] QCIS: {qcis_str[:100]}")

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
                return task_id
            return str(result)
        except Exception as e:
            logger.error(f"[Cqlib] 任务提交失败: {e}")
            if self.auto_retry_machine:
                return self._retry_other_machine(qcis_str, shots, task_name)
            raise

    def _retry_other_machine(self, qcis: str, shots: int, task_name: str) -> str:
        """当前机器不可用时，尝试其他机器"""
        for machine in self.REAL_MACHINES:
            if machine == self.machine_name:
                continue
            try:
                logger.info(f"[Cqlib] 尝试备用机器: {machine}")
                alt = self.cqlib.TianYanPlatform(
                    login_key=self.login_key,
                    machine_name=machine,
                )
                result = alt.submit_experiment(
                    circuit=qcis,
                    name=task_name,
                    num_shots=shots,
                    is_verify=False,
                )
                if isinstance(result, list) and len(result) > 0:
                    tid = str(result[0])
                    logger.info(f"[Cqlib] {machine} 提交成功: {tid}")
                    return tid
            except Exception:
                continue
        raise RuntimeError("所有可用机器提交失败")

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """查询任务状态"""
        try:
            result = self.platform.query_experiment(task_id)
            if isinstance(result, list) and len(result) > 0:
                data = result[0]
                if isinstance(data, dict):
                    has_result = "resultStatus" in data or "probability" in data
                    return {
                        "task_id": task_id,
                        "status": "completed" if has_result else "running",
                        "result": data.get("probability"),
                        "raw": data,
                    }
            return {"task_id": task_id, "status": "unknown", "raw": result}
        except Exception as e:
            return {"task_id": task_id, "status": "error", "error": str(e)}

    def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """获取任务执行结果"""
        return self.get_task_status(task_id)

    def wait_for_task(self, task_id: str, timeout: int = 300, poll_interval: int = 5) -> Dict[str, Any]:
        """轮询等待任务完成并返回结果

        Args:
            task_id: 任务 ID
            timeout: 超时秒数
            poll_interval: 轮询间隔秒数
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

    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态（cqlib 无此接口，返回估算）"""
        machines = self.list_backends()
        running = sum(1 for m in machines if m.get("status") == "running")
        return {
            "total_machines": len(machines),
            "running": running,
            "available": [m["name"] for m in machines if m["status"] == "running"],
        }


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
        machine_names: List[str],
        auto_retry_machine: bool = False,
    ):
        """初始化多机器协调器。

        Args:
            login_key        : 天衍云 API Key
            machine_names    : 要纳管的机器名列表
            auto_retry_machine: 单机提交失败时是否自动切换其他机器（默认 False，
                               多机器场景下由调度器决定路由，通常关闭单机重试）
        """
        self.login_key = login_key
        self.machine_names = list(machine_names)
        self.auto_retry_machine = auto_retry_machine
        self._clients: Dict[str, CqlibTianyanClient] = {}
        self._submit_count: Dict[str, int] = {n: 0 for n in self.machine_names}
        self._fail_count: Dict[str, int] = {n: 0 for n in self.machine_names}

        logger.info(
            f"[MultiMachine] 纳管 {len(self.machine_names)} 台机器: {self.machine_names}"
        )

    def _get_client(self, machine_name: str) -> CqlibTianyanClient:
        """懒加载指定机器的客户端（避免初始化时连接所有机器）。"""
        if machine_name not in self._clients:
            if machine_name not in self.machine_names:
                raise ValueError(f"机器 {machine_name} 未被纳管")
            self._clients[machine_name] = CqlibTianyanClient(
                login_key=self.login_key,
                machine_name=machine_name,
                auto_retry_machine=self.auto_retry_machine,
            )
        return self._clients[machine_name]

    def submit_to_machine(
        self,
        machine_name: str,
        qcis: str,
        shots: int = 512,
        task_name: str = "MultiMachine_Task",
    ) -> Optional[str]:
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
            task_id = client.submit_quantum_task(
                qcis=qcis, shots=shots, task_name=task_name
            )
            self._submit_count[machine_name] = self._submit_count.get(machine_name, 0) + 1
            return task_id
        except Exception as e:  # noqa: BLE001
            self._fail_count[machine_name] = self._fail_count.get(machine_name, 0) + 1
            logger.error(f"[MultiMachine] {machine_name} 提交失败: {e}")
            return None

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
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
                status[name] = {"error": str(e)[:80]}
        return status

    def get_submit_stats(self) -> Dict[str, Dict[str, int]]:
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

    def as_client_map(self) -> Dict[str, CqlibTianyanClient]:
        """返回 {machine_name: client} 映射，便于注入 env.attach_real_clients。

        注意：此方法会触发所有纳管机器的客户端懒加载。
        """
        for name in self.machine_names:
            self._get_client(name)
        return dict(self._clients)


def create_multi_machine_clients(
    login_key: str,
    machine_names: List[str],
) -> Dict[str, CqlibTianyanClient]:
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
