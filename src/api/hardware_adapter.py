"""量子硬件后端抽象层（Issue #256/#258/#259）。

定义统一的 ``QuantumHardwareBackend`` 抽象基类（ABC），使超导、离子阱、
光量子等不同硬件路线可以通过同一接口接入调度系统。

当前实现状态：
    - **超导（superconducting）**：``CqlibTianyanClient`` 已完成真机验证（Issue #257）
    - **离子阱（ion_trap）**：``IonTrapBackend`` 桩实现，待接入真实平台
    - **光量子（photonic）**：``PhotonicBackend`` 桩实现，待接入真实平台

模块导出：
    - ``QuantumHardwareBackend``：抽象基类
    - ``CircuitFormat``：电路格式枚举
    - ``IonTrapBackend``：离子阱桩实现
    - ``PhotonicBackend``：光量子桩实现
    - ``create_hardware_backend``：工厂函数（Issue #259）
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, ClassVar

from loguru import logger


class CircuitFormat(Enum):
    """量子电路格式枚举（Issue #256）。

    不同硬件平台原生支持的电路描述格式不同，
    本枚举用于在提交电路时标注格式类型。
    """

    QCIS = "qcis"
    """天衍云 QCIS 指令格式（超导）"""

    OPENQASM = "openqasm"
    """OpenQASM 2.0/3.0 格式（跨平台通用）"""

    IONQ_JSON = "ionq_json"
    """IonQ JSON 格式（离子阱）"""

    PHOTONIC_HAMILTONIAN = "photonic_hamiltonian"
    """光量子哈密顿量描述格式"""

    QISKIT_CIRCUIT = "qiskit_circuit"
    """Qiskit Circuit 对象（内存对象，非文本格式）"""


class QuantumHardwareBackend(ABC):
    """量子硬件后端抽象基类（Issue #256）。

    所有具体硬件后端（超导/离子阱/光量子）继承本类并实现抽象接口，
    使调度系统可以通过统一接口操作不同硬件平台。

    抽象接口：
        - ``submit_circuit(circuit, shots, task_name) -> str | None``
        - ``get_task_status(task_id) -> dict``
        - ``supported_gates``（属性）
        - ``topology``（属性）
        - ``backend_type``（属性）

    使用示例::

        backend = create_hardware_backend({"hardware_type": "superconducting"})
        task_id = backend.submit_circuit("H Q0\\nM Q0", shots=1024)
        if task_id:
            status = backend.get_task_status(task_id)
    """

    @abstractmethod
    def submit_circuit(
        self,
        circuit: str,
        shots: int = 1024,
        task_name: str = "Scheduler_Task",
    ) -> str | None:
        """提交量子电路到硬件后端执行。

        Args:
            circuit   : 电路描述字符串（格式由子类的 ``circuit_format`` 决定）
            shots     : 测量次数
            task_name : 任务名称

        Returns:
            task_id 字符串；提交失败时返回 None
        """

    @abstractmethod
    def get_task_status(self, task_id: str) -> Mapping[str, Any]:
        """查询任务状态（非阻塞）。

        Args:
            task_id : 任务 ID

        Returns:
            状态字典，包含 ``status`` 字段（"running"/"completed"/"error"/"unknown"）
        """

    @property
    @abstractmethod
    def supported_gates(self) -> list[str]:
        """该后端支持的量子门列表。"""

    @property
    @abstractmethod
    def topology(self) -> dict[str, Any]:
        """硬件拓扑结构信息（耦合图、连接矩阵等）。"""

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """后端类型标识（如 "superconducting"/"ion_trap"/"photonic"）。"""

    @property
    def circuit_format(self) -> CircuitFormat:
        """该后端原生支持的电路格式（默认 QCIS，子类可覆盖）。"""
        return CircuitFormat.QCIS

    def is_available(self) -> bool:
        """检查后端是否可用（默认实现：尝试认证）。

        子类可覆盖此方法提供更精确的可用性检测。
        """
        return True


# ---------------------------------------------------------------------------
# 离子阱后端桩实现（Issue #258）
# ---------------------------------------------------------------------------


class IonTrapBackend(QuantumHardwareBackend):
    """离子阱量子计算后端桩实现（Issue #258）。

    离子阱量子计算机使用囚禁离子作为量子比特，特点是：
    - 全连通拓扑（任意两个离子均可直接纠缠）
    - 较长相干时间（秒级，远超超导的微秒级）
    - 典型门集：单比特旋转门 + Mølmer-Sørensen 两比特门

    .. note::
        当前为桩实现，``submit_circuit`` 返回模拟 task_id，
        不连接真实离子阱平台。

    TODO: 接入真实离子阱平台（如 IonQ Aria / Quantinuum H2）
    """

    _SUPPORTED_GATES: ClassVar[list[str]] = [
        "RZ",
        "RY",
        "RX",
        "RXX",  # Mølmer-Sørensen 门
        "RYY",
        "MS",  # Mølmer-Sørensen 简写
        "M",
    ]

    def __init__(
        self,
        num_ions: int = 20,
        api_key: str | None = None,
    ) -> None:
        """初始化离子阱后端桩实现。

        Args:
            num_ions : 离子数量（量子比特数），默认 20
            api_key  : 平台 API Key（桩实现不使用，预留）
        """
        self._num_ions = num_ions
        self._api_key = api_key
        logger.info(f"[IonTrap] 离子阱后端桩初始化，离子数={num_ions}")

    @property
    def supported_gates(self) -> list[str]:
        """返回离子阱典型门集。"""
        return list(self._SUPPORTED_GATES)

    @property
    def topology(self) -> dict[str, Any]:
        """返回全连通拓扑结构。

        离子阱的全连通性是其核心优势——任意两个离子之间
        都可以直接执行两比特门，无需 SWAP 插入。
        """
        return {
            "type": "all_to_all",
            "num_qubits": self._num_ions,
            "connectivity": "full",
            "description": "离子阱全连通拓扑，任意两离子可直接纠缠",
        }

    @property
    def backend_type(self) -> str:
        """返回后端类型标识。"""
        return "ion_trap"

    @property
    def circuit_format(self) -> CircuitFormat:
        """离子阱后端使用 IonQ JSON 格式。"""
        return CircuitFormat.IONQ_JSON

    def submit_circuit(
        self,
        circuit: str,
        shots: int = 1024,
        task_name: str = "IonTrap_Task",
    ) -> str | None:
        """提交电路到离子阱后端（桩实现：返回模拟 task_id）。

        TODO: 接入真实离子阱平台 API

        Args:
            circuit   : 电路描述（IonQ JSON 格式）
            shots     : 测量次数
            task_name : 任务名称

        Returns:
            模拟 task_id 字符串
        """
        task_id = f"iontrap_stub_{uuid.uuid4().hex[:12]}"
        logger.info(
            f"[IonTrap] 桩提交: {task_name}, shots={shots}, task_id={task_id} (未连接真实平台)"
        )
        return task_id

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """查询任务状态（桩实现：立即返回 completed）。

        TODO: 接入真实离子阱平台 API

        Args:
            task_id : 任务 ID

        Returns:
            模拟完成状态
        """
        logger.debug(f"[IonTrap] 桩查询: {task_id} → completed (模拟)")
        return {
            "task_id": task_id,
            "status": "completed",
            "result": {"0": 0.5, "1": 0.5},
            "raw": {"mock": True, "backend": "ion_trap"},
        }

    def is_available(self) -> bool:
        """桩实现始终返回 True（模拟可用）。"""
        return True


# ---------------------------------------------------------------------------
# 光量子后端桩实现（Issue #258）
# ---------------------------------------------------------------------------


class PhotonicBackend(QuantumHardwareBackend):
    """光量子计算后端桩实现（Issue #258）。

    光量子计算机使用光子作为量子比特，特点是：
    - 室温操作（无需极低温环境）
    - 高速并行处理（光速计算）
    - 典型门集：Hadamard + 分束器 + 相移器 + 光子探测
    - 主要范式：玻色采样 / 高斯玻色采样 / 离散变量光量子

    .. note::
        当前为桩实现，``submit_circuit`` 返回模拟 task_id，
        不连接真实光量子平台。

    TODO: 接入真实光量子平台（如 Xanadu Borealis / 国盾量子）
    """

    _SUPPORTED_GATES: ClassVar[list[str]] = [
        "H",  # Hadamard 门
        "BS",  # 分束器（Beam Splitter）
        "PS",  # 相移器（Phase Shifter）
        "S2",  # 二模压缩门
        "M",  # 光子数测量
        "PNR",  # 光子数分辨检测
    ]

    def __init__(
        self,
        num_modes: int = 16,
        api_key: str | None = None,
    ) -> None:
        """初始化光量子后端桩实现。

        Args:
            num_modes : 光学模式数（等效量子比特数），默认 16
            api_key   : 平台 API Key（桩实现不使用，预留）
        """
        self._num_modes = num_modes
        self._api_key = api_key
        logger.info(f"[Photonic] 光量子后端桩初始化，模式数={num_modes}")

    @property
    def supported_gates(self) -> list[str]:
        """返回光量子典型门集。"""
        return list(self._SUPPORTED_GATES)

    @property
    def topology(self) -> dict[str, Any]:
        """返回光量子拓扑结构。

        光量子芯片的拓扑取决于光路设计（波导布局），
        通常是线性链或网格结构。
        """
        return {
            "type": "linear_chain",
            "num_modes": self._num_modes,
            "connectivity": "nearest_neighbor",
            "description": "光量子线性波导阵列，最近邻耦合",
        }

    @property
    def backend_type(self) -> str:
        """返回后端类型标识。"""
        return "photonic"

    @property
    def circuit_format(self) -> CircuitFormat:
        """光量子后端使用哈密顿量描述格式。"""
        return CircuitFormat.PHOTONIC_HAMILTONIAN

    def submit_circuit(
        self,
        circuit: str,
        shots: int = 1024,
        task_name: str = "Photonic_Task",
    ) -> str | None:
        """提交电路到光量子后端（桩实现：返回模拟 task_id）。

        TODO: 接入真实光量子平台 API

        Args:
            circuit   : 电路描述（哈密顿量格式）
            shots     : 测量次数
            task_name : 任务名称

        Returns:
            模拟 task_id 字符串
        """
        task_id = f"photonic_stub_{uuid.uuid4().hex[:12]}"
        logger.info(
            f"[Photonic] 桩提交: {task_name}, shots={shots}, task_id={task_id} (未连接真实平台)"
        )
        return task_id

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """查询任务状态（桩实现：立即返回 completed）。

        TODO: 接入真实光量子平台 API

        Args:
            task_id : 任务 ID

        Returns:
            模拟完成状态
        """
        logger.debug(f"[Photonic] 桩查询: {task_id} → completed (模拟)")
        return {
            "task_id": task_id,
            "status": "completed",
            "result": {"0": 0.5, "1": 0.5},
            "raw": {"mock": True, "backend": "photonic"},
        }

    def is_available(self) -> bool:
        """桩实现始终返回 True（模拟可用）。"""
        return True


# ---------------------------------------------------------------------------
# 工厂函数（Issue #259）
# ---------------------------------------------------------------------------


# 后端类型 → 实现类的注册表
_BACKEND_REGISTRY: dict[str, type[QuantumHardwareBackend]] = {}


def _register_backend(
    backend_type: str,
) -> Callable[[type[QuantumHardwareBackend]], type[QuantumHardwareBackend]]:
    """装饰器：注册后端类型到全局注册表。

    Args:
        backend_type : 后端类型标识（如 "superconducting"）

    Returns:
        类装饰器
    """

    def decorator(cls: type[QuantumHardwareBackend]) -> type[QuantumHardwareBackend]:
        _BACKEND_REGISTRY[backend_type] = cls
        return cls

    return decorator


# 注册已知的后端类型
_BACKEND_REGISTRY["ion_trap"] = IonTrapBackend
_BACKEND_REGISTRY["photonic"] = PhotonicBackend


def create_hardware_backend(
    config: dict[str, Any] | None = None,
) -> QuantumHardwareBackend:
    """根据配置创建硬件后端实例（Issue #259）。

    工厂函数根据 ``config["hardware_type"]`` 选择对应的后端实现：
        - ``"superconducting"`` → ``CqlibTianyanClient``（超导真机）
        - ``"ion_trap"`` → ``IonTrapBackend``（离子阱桩）
        - ``"photonic"`` → ``PhotonicBackend``（光量子桩）

    Args:
        config : 配置字典，支持以下字段：
            - ``hardware_type`` (str)  : 后端类型，默认 "superconducting"
            - ``login_key`` (str)      : 超导后端的 API Key
            - ``machine_name`` (str)   : 超导后端的机器名
            - ``num_ions`` (int)       : 离子阱离子数
            - ``num_modes`` (int)      : 光量子模式数

    Returns:
        对应的 ``QuantumHardwareBackend`` 实例

    Raises:
        ValueError: 未知的 ``hardware_type`` 时

    Note:
        当前仅超导后端完成真机验证，离子阱和光量子为桩实现。
    """
    config = config or {}
    hardware_type = config.get("hardware_type", "superconducting")

    logger.info(f"[HardwareBackend] 创建后端: type={hardware_type}")

    if hardware_type == "superconducting":
        # 延迟导入避免循环依赖
        from src.api.tianyan_cqlib import CqlibTianyanClient

        login_key = config.get("login_key", "")
        machine_name = config.get("machine_name", "tianyan_s")
        return CqlibTianyanClient(
            login_key=login_key,
            machine_name=machine_name,
        )

    backend_cls = _BACKEND_REGISTRY.get(hardware_type)
    if backend_cls is None:
        raise ValueError(
            f"未知的 hardware_type: {hardware_type!r}，"
            f"支持的类型: {[*_BACKEND_REGISTRY, 'superconducting']}"
        )

    if hardware_type == "ion_trap":
        return backend_cls(  # type: ignore[call-arg]
            num_ions=config.get("num_ions", 20),
            api_key=config.get("api_key"),
        )

    if hardware_type == "photonic":
        return backend_cls(  # type: ignore[call-arg]
            num_modes=config.get("num_modes", 16),
            api_key=config.get("api_key"),
        )

    # 不应到达此处，但为了类型安全
    raise ValueError(f"无法创建后端: {hardware_type!r}")


__all__ = [
    "CircuitFormat",
    "IonTrapBackend",
    "PhotonicBackend",
    "QuantumHardwareBackend",
    "create_hardware_backend",
]
