"""
真机噪声模型提取器（Issue #579）
Noise Model Extractor for Real Quantum Hardware

从真机后端提取三类核心噪声参数：
- 读出错误率（readout error）：各比特测量时的错误概率
- 门错误率（gate error）：单比特门和双比特门的错误概率
- T1 弛豫时间（T1 relaxation time）：各比特能量弛豫时间常数

当无法连接真机（backend=None 或连接失败）时，自动降级为 Mock 模式，
返回基于真实超导量子比特统计分布的仿真噪声数据。

使用示例::

    from src.api.hardware_adapter import create_hardware_backend
    from src.real_machine import NoiseModelExtractor

    # 真机模式
    backend = create_hardware_backend({"hardware_type": "superconducting", "login_key": "xxx"})
    extractor = NoiseModelExtractor(backend=backend)
    profile = extractor.extract_noise_profile()

    # Mock 模式（无需真机连接）
    extractor = NoiseModelExtractor()
    readout_errors = extractor.extract_readout_error()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from src.api.hardware_adapter import QuantumHardwareBackend


def _first_present(*values: Any) -> Any:
    """返回第一个非 None 的值（Issue #675）。

    与 ``a or b or c`` 不同，本函数不会将 ``0``/``0.0``/``False``/``""`` 视为缺失，
    仅跳过 ``None``，从而保留 ``0.0`` 等合法值。
    """
    for v in values:
        if v is not None:
            return v
    return None


class NoiseModelExtractor:
    """真机噪声模型提取器

    从量子硬件后端提取噪声参数，构建完整的噪声画像（noise profile）。
    支持真机提取和 Mock 仿真两种模式，自动根据后端可用性降级。

    噪声参数物理含义：
        - **readout_error**：量子比特测量（读出）时发生错误的概率，
          超导量子比特典型值 0.5%~5%。
        - **gate_error**：量子门操作引入的错误概率。
          单比特门（H/X/Y/Z/RX/RY/RZ）典型值 0.05%~0.2%，
          双比特门（CNOT/CZ）典型值 0.5%~2%。
        - **t1_time**：能量弛豫时间（T1），描述量子比特从 |1⟩ 态
          退相干到 |0⟩ 态的时间常数，超导量子比特典型值 20~100 μs。

    Args:
        backend: 量子硬件后端实例（``QuantumHardwareBackend`` 子类）。
                 为 None 时使用 Mock 模式返回仿真数据。
        num_qubits: Mock 模式下模拟的量子比特数（默认 20，天衍云典型 free tier）。
        seed: 随机数种子（用于 Mock 模式可复现）。

    使用示例::

        extractor = NoiseModelExtractor(backend=None, num_qubits=10, seed=42)
        readout = extractor.extract_readout_error()
        gates = extractor.extract_gate_errors()
        t1 = extractor.extract_t1_times()
        profile = extractor.extract_noise_profile()
    """

    MOCK_SINGLE_GATES: ClassVar[list[str]] = ["H", "X", "Y", "Z", "RX", "RY", "RZ"]
    MOCK_TWO_QUBIT_GATES: ClassVar[list[str]] = ["CZ", "CNOT"]

    def __init__(
        self,
        backend: QuantumHardwareBackend | None = None,
        num_qubits: int = 20,
        seed: int | None = None,
    ) -> None:
        """初始化噪声模型提取器。

        Args:
            backend   : 量子硬件后端实例，None 时使用 Mock 模式。
            num_qubits: Mock 模式下模拟的量子比特数（默认 20）。
            seed      : NumPy 随机种子，用于 Mock 数据可复现。
        """
        self.backend = backend
        self._num_qubits = num_qubits
        self._rng = np.random.default_rng(seed)
        self._mock_cache: dict[str, Any] | None = None

        if backend is None:
            logger.info(f"[NoiseExtractor] 初始化 Mock 模式（{num_qubits} qubits, seed={seed}）")
        else:
            available = False
            try:
                available = backend.is_available()
            except (
                AttributeError,
                ConnectionError,
                TimeoutError,
                ValueError,
                TypeError,
                RuntimeError,
            ) as e:
                logger.warning(f"[NoiseExtractor] 后端可用性检查失败: {e}，将尝试真机提取")
            mode = "真机" if available else "Mock（后端不可用）"
            logger.info(f"[NoiseExtractor] 初始化{mode}模式，backend_type={backend.backend_type}")

    # ------------------------------------------------------------------
    # 公开提取方法
    # ------------------------------------------------------------------

    def extract_readout_error(self) -> dict[str, float]:
        """提取各比特的读出错误率。

        从真机校准数据中提取每个量子比特的读出错误概率；
        真机不可用时返回 Mock 数据。

        Returns:
            dict[str, float]: ``{"Q0": 0.012, "Q1": 0.008, ...}``，
            值范围 [0, 1]，键为比特标识（Q{n}）。
        """
        if self.backend is None:
            return cast(dict[str, float], self._mock_extract()["readout_error"])

        try:
            result = self._extract_readout_error_from_backend()
            if result:
                logger.info(f"[NoiseExtractor] 真机读出错误率提取成功，{len(result)} 个比特")
                return result
        except (
            AttributeError,
            ConnectionError,
            TimeoutError,
            ValueError,
            TypeError,
            RuntimeError,
        ) as e:
            logger.warning(f"[NoiseExtractor] 真机读出错误率提取失败，降级 Mock: {e}")

        return cast(dict[str, float], self._mock_extract()["readout_error"])

    def extract_gate_errors(self) -> dict[str, float]:
        """提取单比特门和双比特门的错误率。

        从真机校准数据中提取每个量子门上的操作错误概率；
        真机不可用时返回 Mock 数据。

        Returns:
            dict[str, float]: 键格式为：
            - 单比特门: ``"Q0_H"``, ``"Q3_RX"``（比特_门名）
            - 双比特门: ``"Q0_Q1_CZ"``, ``"Q5_Q6_CNOT"``（控制比特_目标比特_门名）
            值范围 [0, 1]。
        """
        if self.backend is None:
            return cast(dict[str, float], self._mock_extract()["gate_error"])

        try:
            result = self._extract_gate_errors_from_backend()
            if result:
                single = sum(1 for k in result if "_" in k and k.count("_") == 1)
                two = sum(1 for k in result if k.count("_") == 2)
                logger.info(
                    f"[NoiseExtractor] 真机门错误率提取成功，"
                    f"单比特门 {single} 个，双比特门 {two} 个"
                )
                return result
        except (
            AttributeError,
            ConnectionError,
            TimeoutError,
            ValueError,
            TypeError,
            RuntimeError,
        ) as e:
            logger.warning(f"[NoiseExtractor] 真机门错误率提取失败，降级 Mock: {e}")

        return cast(dict[str, float], self._mock_extract()["gate_error"])

    def extract_t1_times(self) -> dict[str, float]:
        """提取各比特的 T1 弛豫时间。

        从真机校准数据中提取每个量子比特的 T1 能量弛豫时间常数；
        真机不可用时返回 Mock 数据。

        Returns:
            dict[str, float]: ``{"Q0": 45.2, "Q1": 52.8, ...}``，
            值单位为微秒（μs），键为比特标识（Q{n}）。
        """
        if self.backend is None:
            return cast(dict[str, float], self._mock_extract()["t1_time"])

        try:
            result = self._extract_t1_times_from_backend()
            if result:
                logger.info(f"[NoiseExtractor] 真机 T1 时间提取成功，{len(result)} 个比特")
                return result
        except (
            AttributeError,
            ConnectionError,
            TimeoutError,
            ValueError,
            TypeError,
            RuntimeError,
        ) as e:
            logger.warning(f"[NoiseExtractor] 真机 T1 时间提取失败，降级 Mock: {e}")

        return cast(dict[str, float], self._mock_extract()["t1_time"])

    def extract_noise_profile(self) -> dict:
        """聚合三类噪声参数，返回完整噪声画像。

        将读出错误率、门错误率、T1 弛豫时间整合为一个完整的噪声画像字典，
        包含元数据（提取时间戳、后端类型、数据来源等）。

        Returns:
            dict: 完整噪声画像，结构::

                {
                    "readout_error": {"Q0": 0.012, ...},
                    "gate_error": {"Q0_H": 0.0008, "Q0_Q1_CZ": 0.012, ...},
                    "t1_time": {"Q0": 45.2, ...},
                    "metadata": {
                        "source": "mock" | "real",
                        "backend_type": str,
                        "num_qubits": int,
                        "timestamp": str,
                    }
                }
        """
        import datetime

        if self.backend is not None:
            try:
                readout = self._extract_readout_error_from_backend()
                gates = self._extract_gate_errors_from_backend()
                t1 = self._extract_t1_times_from_backend()
                if readout and gates and t1:
                    profile = {
                        "readout_error": readout,
                        "gate_error": gates,
                        "t1_time": t1,
                        "metadata": {
                            "source": "real",
                            "backend_type": self.backend.backend_type,
                            "num_qubits": len(readout),
                            "timestamp": datetime.datetime.now().isoformat(),
                        },
                    }
                    logger.info("[NoiseExtractor] 真机噪声画像提取完成")
                    return profile
            except (
                AttributeError,
                ConnectionError,
                TimeoutError,
                ValueError,
                TypeError,
                RuntimeError,
            ) as e:
                logger.warning(f"[NoiseExtractor] 真机噪声画像提取失败，降级 Mock: {e}")

        mock_data = self._mock_extract()
        mock_data["metadata"] = {
            "source": "mock",
            "backend_type": "mock",
            "num_qubits": self._num_qubits,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        logger.info("[NoiseExtractor] Mock 噪声画像生成完成")
        return mock_data

    # ------------------------------------------------------------------
    # Mock 数据生成
    # ------------------------------------------------------------------

    def _mock_extract(self) -> dict[str, Any]:
        """生成 Mock 噪声数据（无法连接真机时使用）。

        基于超导量子比特的真实噪声统计分布生成仿真数据：
        - 读出错误率：对数正态分布，中位数约 1.5%，范围 0.3%~5%
        - 单比特门错误率：对数正态分布，中位数约 0.1%，范围 0.03%~0.3%
        - 双比特门错误率：对数正态分布，中位数约 1.0%，范围 0.3%~2.5%
        - T1 时间：正态分布，均值约 50 μs，标准差约 15 μs，范围 15~90 μs

        使用缓存机制，同一实例多次调用返回相同数据（模拟校准数据不变）。

        Returns:
            dict: 包含 ``readout_error``、``gate_error``、``t1_time`` 的字典。
        """
        if self._mock_cache is not None:
            return self._mock_cache

        n = self._num_qubits
        qubit_ids = [f"Q{i}" for i in range(n)]

        # 读出错误率：对数正态分布（中位数 ~1.5%）
        readout_mu = np.log(0.015)
        readout_sigma = 0.5
        readout_errors = self._rng.lognormal(mean=readout_mu, sigma=readout_sigma, size=n)
        readout_errors = np.clip(readout_errors, 0.002, 0.06)
        readout_dict = {q: float(e) for q, e in zip(qubit_ids, readout_errors, strict=False)}

        # T1 弛豫时间：正态分布（均值 ~50 μs，标准差 ~15 μs）
        t1_mean = 50.0
        t1_std = 15.0
        t1_times = self._rng.normal(loc=t1_mean, scale=t1_std, size=n)
        t1_times = np.clip(t1_times, 10.0, 100.0)
        t1_dict = {q: float(t) for q, t in zip(qubit_ids, t1_times, strict=False)}

        # 单比特门错误率：对数正态分布（中位数 ~0.1%）
        gate_errors: dict[str, float] = {}
        single_mu = np.log(0.001)
        single_sigma = 0.4
        for q in qubit_ids:
            for gate in self.MOCK_SINGLE_GATES:
                err = self._rng.lognormal(mean=single_mu, sigma=single_sigma)
                err = float(np.clip(err, 0.0002, 0.005))
                gate_errors[f"{q}_{gate}"] = err

        # 双比特门错误率：对数正态分布（中位数 ~1.0%）
        # 使用最近邻耦合拓扑（线性链）
        two_mu = np.log(0.01)
        two_sigma = 0.4
        for i in range(n - 1):
            for gate in self.MOCK_TWO_QUBIT_GATES:
                err = self._rng.lognormal(mean=two_mu, sigma=two_sigma)
                err = float(np.clip(err, 0.002, 0.03))
                gate_errors[f"Q{i}_Q{i + 1}_{gate}"] = err

        self._mock_cache = {
            "readout_error": readout_dict,
            "gate_error": gate_errors,
            "t1_time": t1_dict,
        }

        logger.debug(
            f"[NoiseExtractor] Mock 数据生成完成: "
            f"{n} qubits, "
            f"readout avg={np.mean(readout_errors):.4f}, "
            f"single-gate avg≈{np.mean([v for k, v in gate_errors.items() if k.count('_') == 1]):.5f}, "
            f"two-gate avg≈{np.mean([v for k, v in gate_errors.items() if k.count('_') == 2]):.4f}, "
            f"T1 avg={np.mean(t1_times):.1f}μs"
        )
        return self._mock_cache

    # ------------------------------------------------------------------
    # 真机后端提取（尝试调用 cqlib SDK 校准接口）
    # ------------------------------------------------------------------

    def _extract_readout_error_from_backend(self) -> dict[str, float] | None:
        """尝试从真机后端提取读出错误率。

        优先尝试 cqlib SDK 的校准数据接口（``get_qubit_properties`` /
        ``query_calibration_data`` 等），若不可用则返回 None 触发降级。

        Returns:
            dict[str, float] | None: 成功返回比特→错误率映射，失败返回 None。
        """
        if self.backend is None:
            return None

        platform = getattr(self.backend, "platform", None)
        if platform is None:
            return None

        # 尝试 cqlib 常见的校准数据查询接口
        for method_name in (
            "get_qubit_properties",
            "query_calibration_data",
            "get_readout_fidelity",
        ):
            method = getattr(platform, method_name, None)
            if method is None:
                continue
            try:
                raw = method()
                return self._parse_readout_from_raw(raw)
            except (
                AttributeError,
                TypeError,
                ValueError,
                ConnectionError,
                TimeoutError,
                RuntimeError,
            ):
                logger.debug(f"[NoiseExtractor] {method_name}() 调用失败，尝试下一个接口")
                continue

        return None

    def _extract_gate_errors_from_backend(self) -> dict[str, float] | None:
        """尝试从真机后端提取门错误率。

        优先尝试 cqlib SDK 的门保真度/错误率查询接口，
        若不可用则返回 None 触发降级。

        Returns:
            dict[str, float] | None: 成功返回门键→错误率映射，失败返回 None。
        """
        if self.backend is None:
            return None

        platform = getattr(self.backend, "platform", None)
        if platform is None:
            return None

        for method_name in ("get_gate_errors", "query_gate_fidelity", "get_gate_properties"):
            method = getattr(platform, method_name, None)
            if method is None:
                continue
            try:
                raw = method()
                return self._parse_gate_errors_from_raw(raw)
            except (
                AttributeError,
                TypeError,
                ValueError,
                ConnectionError,
                TimeoutError,
                RuntimeError,
            ):
                logger.debug(f"[NoiseExtractor] {method_name}() 调用失败，尝试下一个接口")
                continue

        return None

    def _extract_t1_times_from_backend(self) -> dict[str, float] | None:
        """尝试从真机后端提取 T1 弛豫时间。

        优先尝试 cqlib SDK 的 T1/T2 相干时间查询接口，
        若不可用则返回 None 触发降级。

        Returns:
            dict[str, float] | None: 成功返回比特→T1(μs)映射，失败返回 None。
        """
        if self.backend is None:
            return None

        platform = getattr(self.backend, "platform", None)
        if platform is None:
            return None

        for method_name in ("get_t1_times", "query_coherence_times", "get_qubit_properties"):
            method = getattr(platform, method_name, None)
            if method is None:
                continue
            try:
                raw = method()
                return self._parse_t1_from_raw(raw)
            except (
                AttributeError,
                TypeError,
                ValueError,
                ConnectionError,
                TimeoutError,
                RuntimeError,
            ):
                logger.debug(f"[NoiseExtractor] {method_name}() 调用失败，尝试下一个接口")
                continue

        return None

    # ------------------------------------------------------------------
    # 原始数据解析（cqlib SDK 返回格式适配）
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_readout_from_raw(raw: Any) -> dict[str, float] | None:
        """解析 cqlib 返回的原始读出保真度/错误率数据。

        cqlib 不同版本返回格式可能不同，本方法做最大兼容解析：
        - dict 格式: ``{"Q0": {"readout_error": 0.012}, ...}`` 或 ``{"Q0": 0.012, ...}``
        - list 格式: ``[{"qubit": 0, "readout_error": 0.012}, ...]``

        Args:
            raw: cqlib SDK 返回的原始数据

        Returns:
            dict[str, float] | None: 解析失败返回 None
        """
        if raw is None:
            return None

        result: dict[str, float] = {}

        if isinstance(raw, dict):
            for key, value in raw.items():
                qid = f"Q{key}" if str(key).isdigit() else str(key)
                if isinstance(value, dict):
                    # Issue #703: 根据字段名判断是保真度还是错误率，不使用 val>0.5 启发式
                    readout_error = value.get("readout_error")
                    readout_fidelity = value.get("readout_fidelity")
                    if readout_error is not None:
                        result[qid] = float(readout_error)
                    elif readout_fidelity is not None:
                        result[qid] = 1.0 - float(readout_fidelity)
                elif isinstance(value, int | float):
                    # Issue #703: 标量值无字段名信息，假设为错误率（不转换）
                    result[qid] = float(value)

        elif isinstance(raw, list | tuple):
            for item in raw:
                if isinstance(item, dict):
                    q = item.get("qubit", item.get("q", item.get("id")))
                    # Issue #703: 根据字段名判断，不使用 val>0.5 启发式
                    readout_error = _first_present(item.get("readout_error"), item.get("error"))
                    readout_fidelity = item.get("readout_fidelity")
                    if q is not None and readout_error is not None:
                        qid = f"Q{q}" if str(q).isdigit() else str(q)
                        result[qid] = float(readout_error)
                    elif q is not None and readout_fidelity is not None:
                        qid = f"Q{q}" if str(q).isdigit() else str(q)
                        result[qid] = 1.0 - float(readout_fidelity)

        return result if result else None

    @staticmethod
    def _parse_gate_errors_from_raw(raw: Any) -> dict[str, float] | None:
        """解析 cqlib 返回的原始门错误率数据。

        兼容多种返回格式：
        - ``{"Q0_H": 0.0008, ...}`` 扁平字典
        - ``{"Q0": {"H": {"error": 0.0008}}, ...}`` 嵌套字典
        - ``[{"qubit": 0, "gate": "H", "error": 0.0008}, ...]`` 列表

        Args:
            raw: cqlib SDK 返回的原始数据

        Returns:
            dict[str, float] | None: 解析失败返回 None
        """
        if raw is None:
            return None

        result: dict[str, float] = {}

        if isinstance(raw, dict):
            for key, value in raw.items():
                if isinstance(value, dict):
                    # 嵌套格式: {"Q0": {"H": 0.0008, "CNOT_Q1": 0.012}}
                    qid = f"Q{key}" if str(key).isdigit() else str(key)
                    for gate_name, gate_val in value.items():
                        if isinstance(gate_val, dict):
                            err = _first_present(gate_val.get("error"), gate_val.get("gate_error"))
                            if err is not None:
                                result[f"{qid}_{gate_name}"] = float(err)
                        elif isinstance(gate_val, int | float):
                            # Issue #703: 标量值假设为错误率（不转换）
                            result[f"{qid}_{gate_name}"] = float(gate_val)
                elif isinstance(value, int | float):
                    # 扁平格式: {"Q0_H": 0.0008}
                    result[str(key)] = float(value)

        elif isinstance(raw, list | tuple):
            for item in raw:
                if isinstance(item, dict):
                    q = item.get("qubit", item.get("q"))
                    gate = item.get("gate", item.get("name"))
                    # Issue #703: 根据字段名判断，不使用 val>0.5 启发式
                    gate_error = _first_present(item.get("error"), item.get("gate_error"))
                    gate_fidelity = item.get("gate_fidelity")
                    if q is not None and gate is not None and gate_error is not None:
                        qid = f"Q{q}" if str(q).isdigit() else str(q)
                        val = float(gate_error)
                        # 双比特门可能有 target_qubit
                        tq = item.get("target_qubit", item.get("target_q"))
                        if tq is not None:
                            tqid = f"Q{tq}" if str(tq).isdigit() else str(tq)
                            result[f"{qid}_{tqid}_{gate}"] = val
                        else:
                            result[f"{qid}_{gate}"] = val
                    elif q is not None and gate is not None and gate_fidelity is not None:
                        # Issue #703: 保真度转错误率
                        qid = f"Q{q}" if str(q).isdigit() else str(q)
                        val = 1.0 - float(gate_fidelity)
                        tq = item.get("target_qubit", item.get("target_q"))
                        if tq is not None:
                            tqid = f"Q{tq}" if str(tq).isdigit() else str(tq)
                            result[f"{qid}_{tqid}_{gate}"] = val
                        else:
                            result[f"{qid}_{gate}"] = val

        return result if result else None

    @staticmethod
    def _parse_t1_from_raw(raw: Any) -> dict[str, float] | None:
        """解析 cqlib 返回的原始 T1 弛豫时间数据。

        兼容多种返回格式：
        - ``{"Q0": 45.2, ...}`` 扁平字典（值单位为 μs）
        - ``{"Q0": {"t1": 45.2}, ...}`` 嵌套字典
        - ``[{"qubit": 0, "t1": 45.2e-6}, ...]`` 列表（可能是秒为单位，自动转 μs）

        Args:
            raw: cqlib SDK 返回的原始数据

        Returns:
            dict[str, float] | None: 解析失败返回 None
        """
        if raw is None:
            return None

        result: dict[str, float] = {}

        if isinstance(raw, dict):
            for key, value in raw.items():
                qid = f"Q{key}" if str(key).isdigit() else str(key)
                if isinstance(value, dict):
                    t1 = _first_present(value.get("t1"), value.get("T1"), value.get("t1_time"))
                    if t1 is not None:
                        t1_val = float(t1)
                        # 若值 < 0.001，假定单位为秒，转换为微秒
                        if t1_val < 0.001:
                            t1_val *= 1e6
                        result[qid] = t1_val
                elif isinstance(value, int | float):
                    t1_val = float(value)
                    if 0 < t1_val < 0.001:
                        t1_val *= 1e6
                    result[qid] = t1_val

        elif isinstance(raw, list | tuple):
            for item in raw:
                if isinstance(item, dict):
                    q = item.get("qubit", item.get("q", item.get("id")))
                    t1 = _first_present(item.get("t1"), item.get("T1"), item.get("t1_time"))
                    if q is not None and t1 is not None:
                        qid = f"Q{q}" if str(q).isdigit() else str(q)
                        t1_val = float(t1)
                        if 0 < t1_val < 0.001:
                            t1_val *= 1e6
                        result[qid] = t1_val

        return result if result else None
