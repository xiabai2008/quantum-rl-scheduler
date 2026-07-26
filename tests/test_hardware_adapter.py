"""量子硬件后端抽象层单元测试（Issue #256/#257/#258/#259）。

测试覆盖：
- TestQuantumHardwareBackendABC  : ABC 接口完整性验证
- TestCircuitFormat               : 电路格式枚举
- TestIonTrapBackend              : 离子阱桩实现
- TestPhotonicBackend             : 光量子桩实现
- TestSuperconductingBackendABC : 超导后端(CqlibTianyanClient) ABC 继承验证
- TestCreateHardwareBackend       : 工厂函数
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

from src.api.hardware_adapter import (
    CircuitFormat,
    IonTrapBackend,
    PhotonicBackend,
    QuantumHardwareBackend,
    create_hardware_backend,
)


# ============================================================
# TestQuantumHardwareBackendABC
# ============================================================
class TestQuantumHardwareBackendABC(unittest.TestCase):
    """测试 QuantumHardwareBackend ABC 接口。"""

    def test_cannot_instantiate_abc(self):
        """ABC 不能直接实例化。"""
        with self.assertRaises(TypeError):
            QuantumHardwareBackend()  # type: ignore[abstract]

    def test_abstract_methods_defined(self):
        """ABC 应定义所有抽象方法。"""
        abstract_methods = QuantumHardwareBackend.__abstractmethods__
        self.assertIn("submit_circuit", abstract_methods)
        self.assertIn("get_task_status", abstract_methods)
        self.assertIn("supported_gates", abstract_methods)
        self.assertIn("topology", abstract_methods)
        self.assertIn("backend_type", abstract_methods)

    def test_circuit_format_default(self):
        """circuit_format 属性应有默认值。"""
        # 通过子类验证默认值
        backend = IonTrapBackend()
        # IonTrapBackend 覆盖了 circuit_format
        self.assertEqual(backend.circuit_format, CircuitFormat.IONQ_JSON)

    def test_is_available_default(self):
        """is_available 默认返回 True。"""
        backend = IonTrapBackend()
        self.assertTrue(backend.is_available())


# ============================================================
# TestCircuitFormat
# ============================================================
class TestCircuitFormat(unittest.TestCase):
    """测试 CircuitFormat 枚举（Issue #256）。"""

    def test_all_formats_exist(self):
        """所有电路格式应存在。"""
        self.assertEqual(CircuitFormat.QCIS.value, "qcis")
        self.assertEqual(CircuitFormat.OPENQASM.value, "openqasm")
        self.assertEqual(CircuitFormat.IONQ_JSON.value, "ionq_json")
        self.assertEqual(CircuitFormat.PHOTONIC_HAMILTONIAN.value, "photonic_hamiltonian")
        self.assertEqual(CircuitFormat.QISKIT_CIRCUIT.value, "qiskit_circuit")

    def test_format_count(self):
        """应有 5 种格式。"""
        self.assertEqual(len(CircuitFormat), 5)


# ============================================================
# TestIonTrapBackend
# ============================================================
class TestIonTrapBackend(unittest.TestCase):
    """测试离子阱桩实现（Issue #258）。"""

    def setUp(self):
        """创建测试实例。"""
        self.backend = IonTrapBackend(num_ions=10)

    def test_backend_type(self):
        """后端类型应为 ion_trap。"""
        self.assertEqual(self.backend.backend_type, "ion_trap")

    def test_supported_gates(self):
        """应返回离子阱典型门集。"""
        gates = self.backend.supported_gates
        self.assertIn("RZ", gates)
        self.assertIn("RY", gates)
        self.assertIn("RX", gates)
        self.assertIn("RXX", gates)  # Mølmer-Sørensen
        self.assertIn("M", gates)
        self.assertGreater(len(gates), 0)

    def test_topology(self):
        """拓扑应为全连通。"""
        topo = self.backend.topology
        self.assertEqual(topo["type"], "all_to_all")
        self.assertEqual(topo["num_qubits"], 10)
        self.assertEqual(topo["connectivity"], "full")

    def test_circuit_format(self):
        """电路格式应为 IonQ JSON。"""
        self.assertEqual(self.backend.circuit_format, CircuitFormat.IONQ_JSON)

    def test_submit_circuit_returns_task_id(self):
        """submit_circuit 应返回模拟 task_id。"""
        task_id = self.backend.submit_circuit("test circuit", shots=512, task_name="test")
        self.assertIsNotNone(task_id)
        self.assertIn("iontrap_stub_", task_id)

    def test_get_task_status(self):
        """get_task_status 应返回 completed 状态。"""
        task_id = self.backend.submit_circuit("test", shots=100)
        status = self.backend.get_task_status(task_id)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["task_id"], task_id)

    def test_is_available(self):
        """桩实现应始终可用。"""
        self.assertTrue(self.backend.is_available())

    def test_isinstance_quantum_hardware_backend(self):
        """应为 QuantumHardwareBackend 实例。"""
        self.assertIsInstance(self.backend, QuantumHardwareBackend)

    def test_custom_num_ions(self):
        """应支持自定义离子数。"""
        backend = IonTrapBackend(num_ions=50)
        self.assertEqual(backend.topology["num_qubits"], 50)


# ============================================================
# TestPhotonicBackend
# ============================================================
class TestPhotonicBackend(unittest.TestCase):
    """测试光量子桩实现（Issue #258）。"""

    def setUp(self):
        """创建测试实例。"""
        self.backend = PhotonicBackend(num_modes=8)

    def test_backend_type(self):
        """后端类型应为 photonic。"""
        self.assertEqual(self.backend.backend_type, "photonic")

    def test_supported_gates(self):
        """应返回光量子典型门集。"""
        gates = self.backend.supported_gates
        self.assertIn("H", gates)
        self.assertIn("BS", gates)  # 分束器
        self.assertIn("PS", gates)  # 相移器
        self.assertIn("M", gates)
        self.assertGreater(len(gates), 0)

    def test_topology(self):
        """拓扑应为线性链。"""
        topo = self.backend.topology
        self.assertEqual(topo["type"], "linear_chain")
        self.assertEqual(topo["num_modes"], 8)
        self.assertEqual(topo["connectivity"], "nearest_neighbor")

    def test_circuit_format(self):
        """电路格式应为光量子哈密顿量。"""
        self.assertEqual(self.backend.circuit_format, CircuitFormat.PHOTONIC_HAMILTONIAN)

    def test_submit_circuit_returns_task_id(self):
        """submit_circuit 应返回模拟 task_id。"""
        task_id = self.backend.submit_circuit("test circuit", shots=256, task_name="test")
        self.assertIsNotNone(task_id)
        self.assertIn("photonic_stub_", task_id)

    def test_get_task_status(self):
        """get_task_status 应返回 completed 状态。"""
        task_id = self.backend.submit_circuit("test", shots=100)
        status = self.backend.get_task_status(task_id)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["task_id"], task_id)

    def test_is_available(self):
        """桩实现应始终可用。"""
        self.assertTrue(self.backend.is_available())

    def test_isinstance_quantum_hardware_backend(self):
        """应为 QuantumHardwareBackend 实例。"""
        self.assertIsInstance(self.backend, QuantumHardwareBackend)

    def test_custom_num_modes(self):
        """应支持自定义模式数。"""
        backend = PhotonicBackend(num_modes=32)
        self.assertEqual(backend.topology["num_modes"], 32)


# ============================================================
# TestSuperconductingBackendABC
# ============================================================
class TestSuperconductingBackendABC(unittest.TestCase):
    """测试 CqlibTianyanClient（超导后端）的 ABC 继承（Issue #257）。

    使用 ``sys.modules`` mock 替代真实 cqlib SDK，
    验证超导后端满足 QuantumHardwareBackend 契约。
    """

    def setUp(self):
        """在每个测试前将 cqlib 模块 mock 注入 sys.modules。"""
        self._mock_cqlib = MagicMock()
        self._modules_patch = patch.dict("sys.modules", {"cqlib": self._mock_cqlib})
        self._modules_patch.start()

    def tearDown(self):
        """测试后恢复 sys.modules。"""
        self._modules_patch.stop()
        # 清除可能被缓存的导入
        if "src.api.tianyan_cqlib" in sys.modules:
            del sys.modules["src.api.tianyan_cqlib"]

    def test_isinstance_quantum_hardware_backend(self):
        """CqlibTianyanClient 应为 QuantumHardwareBackend 实例。"""
        from src.api.tianyan_cqlib import CqlibTianyanClient

        client = CqlibTianyanClient(login_key="test_key")
        self.assertIsInstance(client, QuantumHardwareBackend)

    def test_backend_type_superconducting(self):
        """backend_type 应为 superconducting。"""
        from src.api.tianyan_cqlib import CqlibTianyanClient

        client = CqlibTianyanClient(login_key="test_key")
        self.assertEqual(client.backend_type, "superconducting")

    def test_supported_gates(self):
        """应返回超导门集。"""
        from src.api.tianyan_cqlib import CqlibTianyanClient

        client = CqlibTianyanClient(login_key="test_key")
        gates = client.supported_gates
        expected = ["H", "X", "Y", "Z", "RX", "RY", "RZ", "CNOT", "CZ", "M"]
        self.assertEqual(gates, expected)

    def test_topology(self):
        """应返回天衍-287 拓扑信息。"""
        from src.api.tianyan_cqlib import CqlibTianyanClient

        client = CqlibTianyanClient(login_key="test_key")
        topo = client.topology
        self.assertEqual(topo["type"], "2d_grid")
        self.assertEqual(topo["total_qubits"], 287)
        self.assertIn("data_qubits", topo)
        self.assertIn("coupler_qubits", topo)

    def test_circuit_format_qcis(self):
        """电路格式应为 QCIS。"""
        from src.api.tianyan_cqlib import CqlibTianyanClient

        client = CqlibTianyanClient(login_key="test_key")
        self.assertEqual(client.circuit_format, CircuitFormat.QCIS)

    def test_submit_circuit_delegates_to_submit_quantum_task(self):
        """submit_circuit 应委托给 submit_quantum_task。"""
        from src.api.tianyan_cqlib import CqlibTianyanClient

        client = CqlibTianyanClient(login_key="test_key")
        with patch.object(client, "submit_quantum_task", return_value="task_123") as mock:
            result = client.submit_circuit("H Q0\nM Q0", shots=1024, task_name="test")
            self.assertEqual(result, "task_123")
            mock.assert_called_once_with(qcis="H Q0\nM Q0", shots=1024, task_name="test")


# ============================================================
# TestCreateHardwareBackend
# ============================================================
class TestCreateHardwareBackend(unittest.TestCase):
    """测试工厂函数（Issue #259）。"""

    def test_create_ion_trap_backend(self):
        """应创建 IonTrapBackend。"""
        backend = create_hardware_backend({"hardware_type": "ion_trap"})
        self.assertIsInstance(backend, IonTrapBackend)
        self.assertEqual(backend.backend_type, "ion_trap")

    def test_create_photonic_backend(self):
        """应创建 PhotonicBackend。"""
        backend = create_hardware_backend({"hardware_type": "photonic"})
        self.assertIsInstance(backend, PhotonicBackend)
        self.assertEqual(backend.backend_type, "photonic")

    def test_create_superconducting_backend(self):
        """应创建 CqlibTianyanClient。"""
        mock_cqlib = MagicMock()
        with patch.dict("sys.modules", {"cqlib": mock_cqlib}):
            # 清除缓存的 tianyan_cqlib 模块，确保重新导入时使用 mock
            sys.modules.pop("src.api.tianyan_cqlib", None)

            backend = create_hardware_backend(
                {
                    "hardware_type": "superconducting",
                    "login_key": "test_key",
                }
            )
            from src.api.tianyan_cqlib import CqlibTianyanClient

            self.assertIsInstance(backend, CqlibTianyanClient)
            self.assertEqual(backend.backend_type, "superconducting")

    def test_unknown_hardware_type_raises(self):
        """未知 hardware_type 应抛出 ValueError。"""
        with self.assertRaises(ValueError) as ctx:
            create_hardware_backend({"hardware_type": "unknown_type"})
        self.assertIn("unknown_type", str(ctx.exception))

    def test_default_hardware_type(self):
        """不指定 hardware_type 时应默认为 superconducting。"""
        mock_cqlib = MagicMock()
        with patch.dict("sys.modules", {"cqlib": mock_cqlib}):
            sys.modules.pop("src.api.tianyan_cqlib", None)

            backend = create_hardware_backend({})
            self.assertEqual(backend.backend_type, "superconducting")

    def test_ion_trap_with_custom_config(self):
        """应支持自定义离子数配置。"""
        backend = create_hardware_backend(
            {
                "hardware_type": "ion_trap",
                "num_ions": 50,
            }
        )
        self.assertEqual(backend.topology["num_qubits"], 50)

    def test_photonic_with_custom_config(self):
        """应支持自定义模式数配置。"""
        backend = create_hardware_backend(
            {
                "hardware_type": "photonic",
                "num_modes": 32,
            }
        )
        self.assertEqual(backend.topology["num_modes"], 32)

    def test_all_backends_satisfy_abc_contract(self):
        """所有后端都应满足 ABC 契约。"""
        backends = [
            create_hardware_backend({"hardware_type": "ion_trap"}),
            create_hardware_backend({"hardware_type": "photonic"}),
        ]
        for backend in backends:
            self.assertIsInstance(backend, QuantumHardwareBackend)
            # 验证所有抽象方法已实现
            self.assertEqual(len(backend.__class__.__abstractmethods__), 0)
            # 验证接口可调用
            self.assertIsInstance(backend.supported_gates, list)
            self.assertIsInstance(backend.topology, dict)
            self.assertIsInstance(backend.backend_type, str)
            task_id = backend.submit_circuit("test", shots=100)
            self.assertIsNotNone(task_id)
            status = backend.get_task_status(task_id)
            self.assertIn("status", status)


if __name__ == "__main__":
    unittest.main()
