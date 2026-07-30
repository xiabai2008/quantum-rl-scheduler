"""NoiseModelExtractor 单元测试（Issue #579）。

测试覆盖：
    1. Mock 模式下三类噪声参数提取（读出误差/门误差/T1 时间）
    2. extract_noise_profile 聚合方法返回完整噪声画像
    3. 返回数据结构包含必要字段
    4. Mock 数据在合理范围内（错误率 [0,1]，T1 时间为正）
    5. 后端异常时正确降级到 Mock 模式
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.real_machine.noise_extractor import NoiseModelExtractor

# ---------------------------------------------------------------------------
# 辅助：构造会异常的 Mock 后端
# ---------------------------------------------------------------------------


class _BrokenBackend:
    """模拟所有校准接口均异常的真机后端，用于测试降级路径。"""

    backend_type = "broken"

    def __init__(self, available: bool = False) -> None:
        self._available = available
        # platform 上所有方法都抛异常
        self.platform = MagicMock()
        self.platform.get_qubit_properties.side_effect = RuntimeError("SDK 不支持")
        self.platform.query_calibration_data.side_effect = AttributeError("接口缺失")
        self.platform.get_readout_fidelity.side_effect = ValueError("数据格式错误")

    def is_available(self) -> bool:
        return self._available


# ---------------------------------------------------------------------------
# 测试 1-3：Mock 模式下三类噪声参数提取
# ---------------------------------------------------------------------------


class TestExtractMockData:
    """Mock 模式（backend=None）下各类噪声参数提取。"""

    def test_extract_readout_error_mock(self) -> None:
        """Mock 模式下应返回合理的读出误差字典。"""
        extractor = NoiseModelExtractor(backend=None, num_qubits=10, seed=42)
        result = extractor.extract_readout_error()

        assert isinstance(result, dict)
        assert len(result) == 10
        # 键格式应为 Q{n}
        assert all(k.startswith("Q") for k in result)
        # 值应为 float 类型
        assert all(isinstance(v, float) for v in result.values())

    def test_extract_gate_errors_mock(self) -> None:
        """Mock 模式下应返回合理的门误差字典。"""
        extractor = NoiseModelExtractor(backend=None, num_qubits=8, seed=42)
        result = extractor.extract_gate_errors()

        assert isinstance(result, dict)
        assert len(result) > 0
        # 应包含单比特门（Q0_H）和双比特门（Q0_Q1_CZ）键
        single_keys = [k for k in result if k.count("_") == 1]
        two_qubit_keys = [k for k in result if k.count("_") == 2]
        assert len(single_keys) > 0, "应包含单比特门误差"
        assert len(two_qubit_keys) > 0, "应包含双比特门误差"
        # 值应为 float 类型
        assert all(isinstance(v, float) for v in result.values())

    def test_extract_t1_times_mock(self) -> None:
        """Mock 模式下应返回合理的 T1 弛豫时间字典。"""
        extractor = NoiseModelExtractor(backend=None, num_qubits=12, seed=42)
        result = extractor.extract_t1_times()

        assert isinstance(result, dict)
        assert len(result) == 12
        # 键格式应为 Q{n}
        assert all(k.startswith("Q") for k in result)
        # 值应为 float 类型且为正数（单位 μs）
        assert all(isinstance(v, float) for v in result.values())
        assert all(v > 0 for v in result.values())


# ---------------------------------------------------------------------------
# 测试 4：extract_noise_profile 聚合方法
# ---------------------------------------------------------------------------


class TestExtractNoiseProfile:
    """extract_noise_profile 聚合方法测试。"""

    def test_extract_noise_profile_mock(self) -> None:
        """Mock 模式下 extract_noise_profile 应聚合三类噪声参数。"""
        extractor = NoiseModelExtractor(backend=None, num_qubits=10, seed=42)
        profile = extractor.extract_noise_profile()

        assert isinstance(profile, dict)
        # 应包含三类噪声参数
        assert "readout_error" in profile
        assert "gate_error" in profile
        assert "t1_time" in profile
        assert "metadata" in profile
        # metadata 应标记为 mock 来源
        assert profile["metadata"]["source"] == "mock"
        assert profile["metadata"]["num_qubits"] == 10
        assert "timestamp" in profile["metadata"]


# ---------------------------------------------------------------------------
# 测试 5：数据结构验证
# ---------------------------------------------------------------------------


class TestNoiseProfileStructure:
    """验证返回的数据结构包含必要字段。"""

    def test_noise_profile_structure(self) -> None:
        """噪声画像应包含所有必要字段且类型正确。"""
        extractor = NoiseModelExtractor(backend=None, num_qubits=5, seed=42)
        profile = extractor.extract_noise_profile()

        # 顶层字段
        required_keys = {"readout_error", "gate_error", "t1_time", "metadata"}
        assert set(profile.keys()) >= required_keys

        # readout_error: dict[str, float]
        assert isinstance(profile["readout_error"], dict)
        assert all(isinstance(v, float) for v in profile["readout_error"].values())

        # gate_error: dict[str, float]
        assert isinstance(profile["gate_error"], dict)
        assert all(isinstance(v, float) for v in profile["gate_error"].values())

        # t1_time: dict[str, float]
        assert isinstance(profile["t1_time"], dict)
        assert all(isinstance(v, float) for v in profile["t1_time"].values())

        # metadata: dict 含 source/backend_type/num_qubits/timestamp
        meta = profile["metadata"]
        assert isinstance(meta, dict)
        assert "source" in meta
        assert "backend_type" in meta
        assert "num_qubits" in meta
        assert "timestamp" in meta
        assert isinstance(meta["num_qubits"], int)


# ---------------------------------------------------------------------------
# 测试 6：Mock 数据合理性（范围内）
# ---------------------------------------------------------------------------


class TestMockDataReasonable:
    """Mock 数据应在物理合理的范围内。"""

    def test_mock_data_reasonable(self) -> None:
        """Mock 错误率应在 [0, 1] 范围内，T1 时间应为正数。"""
        extractor = NoiseModelExtractor(backend=None, num_qubits=20, seed=42)

        readout = extractor.extract_readout_error()
        for _q, err in readout.items():
            assert 0.0 <= err <= 1.0, f"读出误差 {err} 超出 [0,1]"

        gates = extractor.extract_gate_errors()
        for _k, err in gates.items():
            assert 0.0 <= err <= 1.0, f"门误差 {err} 超出 [0,1]"

        t1 = extractor.extract_t1_times()
        for _q, t in t1.items():
            assert t > 0, f"T1 时间 {t} 应为正数"
            assert t < 1000, f"T1 时间 {t} 异常大（应 <1000μs）"

    def test_mock_data_reproducible_with_seed(self) -> None:
        """相同种子应产生相同的 Mock 数据。"""
        e1 = NoiseModelExtractor(backend=None, num_qubits=8, seed=123)
        e2 = NoiseModelExtractor(backend=None, num_qubits=8, seed=123)
        assert e1.extract_readout_error() == e2.extract_readout_error()
        assert e1.extract_t1_times() == e2.extract_t1_times()


# ---------------------------------------------------------------------------
# 测试 7：异常降级到 Mock
# ---------------------------------------------------------------------------


class TestFallbackOnError:
    """后端异常时应正确降级到 Mock 模式。"""

    def test_fallback_on_error(self) -> None:
        """后端所有接口均异常时应降级返回 Mock 数据。"""
        backend = _BrokenBackend(available=True)
        extractor = NoiseModelExtractor(backend=backend, num_qubits=6, seed=42)

        # 读出误差：后端异常 → 降级 Mock
        readout = extractor.extract_readout_error()
        assert isinstance(readout, dict)
        assert len(readout) == 6  # Mock 模式按 num_qubits 生成

        # 门误差：后端异常 → 降级 Mock
        gates = extractor.extract_gate_errors()
        assert isinstance(gates, dict)
        assert len(gates) > 0

        # T1 时间：后端异常 → 降级 Mock
        t1 = extractor.extract_t1_times()
        assert isinstance(t1, dict)
        assert len(t1) == 6

    def test_fallback_profile_on_error(self) -> None:
        """extract_noise_profile 在后端异常时应降级返回 Mock 画像。"""
        backend = _BrokenBackend(available=True)
        extractor = NoiseModelExtractor(backend=backend, num_qubits=5, seed=42)
        profile = extractor.extract_noise_profile()

        assert profile["metadata"]["source"] == "mock"
        assert len(profile["readout_error"]) == 5

    def test_backend_not_available_falls_back_to_mock(self) -> None:
        """后端 is_available()=False 时应降级到 Mock。"""
        backend = _BrokenBackend(available=False)
        extractor = NoiseModelExtractor(backend=backend, num_qubits=4, seed=42)

        readout = extractor.extract_readout_error()
        assert len(readout) == 4  # Mock 数据按 num_qubits 生成


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
