"""
量子RL调度系统 - API 层单元测试
Unit Tests for src/api/ layer

测试覆盖：
- MockTianyanClient（Mock 客户端全生命周期）
- create_tianyan_client / get_client / get_cqlib_client 工厂函数
- TianyanClient（Mock 委托路径 + 真实 REST 重试路径 + cqlib 委托路径）
- CqlibTianyanClient（cqlib SDK 封装，platform 用 mock 替代）
- MultiMachineCqlibCoordinator / create_multi_machine_clients（多机器协调器）

所有网络相关调用均通过 unittest.mock 替代，无需真实 TIANYAN_API_KEY。
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# 检测 cqlib 是否可用（CI 环境可能未安装真机 SDK）
try:
    import cqlib

    _HAS_CQLIB = True
except ImportError:
    _HAS_CQLIB = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api import get_client, get_cqlib_client
from src.api.circuit_breaker import CircuitState
from src.api.mock_client import MockTianyanClient, create_tianyan_client
from src.api.tianyan_client import (
    QuotaTracker,
    TianyanAPIError,
    TianyanClient,
    TokenBucketRateLimiter,
    mask_token,
)
from src.api.tianyan_cqlib import (
    CqlibTianyanClient,
    MultiMachineCqlibCoordinator,
    create_multi_machine_clients,
)
from src.exceptions import CircuitOpenError, RateLimitError

# 简单的 Bell 态 QASM 电路，用于提交任务测试
BELL_QASM = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""


def _env_without(*keys):
    """返回剔除指定键后的环境变量副本，配合 patch.dict(clear=True) 使用。"""
    return {k: v for k, v in os.environ.items() if k not in keys}


class TestMockTianyanClient(unittest.TestCase):
    """测试 Mock 天衍云客户端的全部接口与状态轮转逻辑。"""

    def setUp(self):
        """每个测试创建一个无延迟、不失败的 Mock 客户端。"""
        self.client = MockTianyanClient(mock_delay=0.0, mock_failure_rate=0.0)

    def test_init_defaults(self):
        """默认初始化应设置延迟、失败率与后端列表。"""
        client = MockTianyanClient()
        self.assertEqual(client.mock_delay, 1.0)
        self.assertEqual(client.mock_failure_rate, 0.0)
        self.assertIsInstance(client.api_key, str)
        self.assertEqual(len(client._backends), 2)

    def test_init_with_custom_config(self):
        """自定义配置应被正确存储。"""
        client = MockTianyanClient(
            mock_delay=0.5, mock_failure_rate=0.2, api_key="key-123", base_url="http://x"
        )
        self.assertEqual(client.mock_delay, 0.5)
        self.assertEqual(client.mock_failure_rate, 0.2)
        self.assertEqual(client.api_key, "key-123")

    def test_authenticate_returns_true(self):
        """Mock 认证始终返回 True。"""
        self.assertTrue(self.client.authenticate())

    def test_submit_quantum_task_returns_task_id(self):
        """提交量子任务应返回以 'mock-' 开头的 task_id。"""
        task_id = self.client.submit_quantum_task(circuit_qasm=BELL_QASM, shots=512)
        self.assertIsInstance(task_id, str)
        self.assertTrue(task_id.startswith("mock-"))
        self.assertIn(task_id, self.client._tasks)
        self.assertEqual(self.client._tasks[task_id]["status"], "PENDING")
        self.assertEqual(self.client._tasks[task_id]["shots"], 512)

    def test_submit_quantum_task_empty_circuit_accepted(self):
        """Mock 不校验空电路，仍返回 task_id（记录实际行为）。"""
        task_id = self.client.submit_quantum_task(circuit_qasm="", shots=128)
        self.assertTrue(task_id.startswith("mock-"))

    def test_submit_quantum_task_with_failure_rate_raises(self):
        """失败率为 1 时提交应抛出 TianyanAPIError。"""
        client = MockTianyanClient(mock_delay=0.0, mock_failure_rate=1.0)
        # 0.0 < 1.0 触发失败
        with patch("random.random", return_value=0.0), self.assertRaises(TianyanAPIError):
            client.submit_quantum_task(circuit_qasm=BELL_QASM, shots=64)

    def test_get_task_status_unknown_task_raises(self):
        """查询不存在的任务应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            self.client.get_task_status("nonexistent-id")

    def test_get_task_status_returns_pending_initially(self):
        """新提交任务首次查询应返回 PENDING（强制 random 不推进状态）。"""
        task_id = self.client.submit_quantum_task(circuit_qasm=BELL_QASM, shots=64)
        with patch("random.random", return_value=0.9):  # 0.9 >= 0.3 不推进
            status = self.client.get_task_status(task_id)
        self.assertEqual(status["status"], "PENDING")
        self.assertEqual(status["task_id"], task_id)

    def test_task_state_transitions_to_completed(self):
        """轮询多次应能经历 PENDING→RUNNING→COMPLETED 全流程。"""
        task_id = self.client.submit_quantum_task(circuit_qasm=BELL_QASM, shots=64)
        task = self.client._tasks[task_id]

        # PENDING -> RUNNING（random=0.0 < 0.3）
        with patch("random.random", return_value=0.0):
            self.client.get_task_status(task_id)
        self.assertEqual(task["status"], "RUNNING")
        self.assertIsNotNone(task["started_at"])

        # RUNNING -> COMPLETED（random=0.0 < 0.4）
        with patch("random.random", return_value=0.0):
            self.client.get_task_status(task_id)
        self.assertEqual(task["status"], "COMPLETED")
        self.assertIsNotNone(task["completed_at"])
        self.assertIsNotNone(task["result"])

    def test_get_task_result_unknown_task_raises(self):
        """获取不存在任务的结果应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            self.client.get_task_result("nonexistent-id")

    def test_get_task_result_not_completed_raises(self):
        """任务未完成时获取结果应抛出 ValueError。"""
        task_id = self.client.submit_quantum_task(circuit_qasm=BELL_QASM, shots=64)
        # 强制不推进状态
        with patch("random.random", return_value=0.9), self.assertRaises(ValueError):
            self.client.get_task_result(task_id)

    def test_get_task_result_returns_counts_when_completed(self):
        """任务完成后应返回包含 counts 的结果字典。"""
        task_id = self.client.submit_quantum_task(circuit_qasm=BELL_QASM, shots=64)
        # 直接置为完成并生成结果
        task = self.client._tasks[task_id]
        task["status"] = "COMPLETED"
        task["completed_at"] = "2026-07-01T00:00:00"
        task["result"] = self.client._generate_mock_result(BELL_QASM, 64)
        task["result"]["task_id"] = task_id

        result = self.client.get_task_result(task_id)
        self.assertIn("counts", result)
        self.assertEqual(result["shots"], 64)
        self.assertEqual(result["task_id"], task_id)

    def test_get_task_result_auto_triggers_completion(self):
        """get_task_result 内部触发 get_task_status，若完成则直接返回结果。"""
        task_id = self.client.submit_quantum_task(circuit_qasm=BELL_QASM, shots=64)
        task = self.client._tasks[task_id]
        task["status"] = "RUNNING"
        # get_task_status 中 RUNNING -> COMPLETED（random=0.0）
        with patch("random.random", return_value=0.0):
            result = self.client.get_task_result(task_id)
        self.assertEqual(task["status"], "COMPLETED")
        self.assertIn("counts", result)

    def test_list_backends(self):
        """list_backends 应返回包含 name 与 num_qubits 的后端字典列表。"""
        backends = self.client.list_backends()
        self.assertEqual(len(backends), 2)
        for b in backends:
            self.assertIn("name", b)
            self.assertIn("num_qubits", b)
        names = [b["name"] for b in backends]
        self.assertIn("tianyan-287", names)
        self.assertIn("tianyan-simulator", names)

    def test_get_backend_info_valid(self):
        """查询已知后端应返回其详情字典。"""
        info = self.client.get_backend_info("tianyan-287")
        self.assertEqual(info["name"], "tianyan-287")
        self.assertEqual(info["num_qubits"], 287)

    def test_get_backend_info_invalid_raises(self):
        """查询不存在后端应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            self.client.get_backend_info("no-such-backend")

    def test_submit_classical_task(self):
        """提交经典任务应立即完成并返回 task_id。"""
        task_id = self.client.submit_classical_task(code="print(1)", language="python3")
        self.assertTrue(task_id.startswith("mock-classical-"))
        task = self.client._tasks[task_id]
        self.assertEqual(task["status"], "COMPLETED")
        self.assertEqual(task["type"], "classical")
        self.assertEqual(task["result"]["exit_code"], 0)

    def test_get_queue_status(self):
        """队列状态应包含 pending/running 统计与 by_backend 分组。"""
        # 提交几个量子任务以产生 pending 计数
        self.client.submit_quantum_task(circuit_qasm=BELL_QASM, shots=64)
        self.client.submit_quantum_task(circuit_qasm=BELL_QASM, shots=64)
        q = self.client.get_queue_status()
        self.assertIn("total_pending", q)
        self.assertIn("total_running", q)
        self.assertIn("by_backend", q)
        self.assertEqual(q["total_pending"], 2)

    def test_wait_for_task_success(self):
        """wait_for_task 在任务完成时应返回结果。"""
        task_id = self.client.submit_quantum_task(circuit_qasm=BELL_QASM, shots=64)
        # 预置完成状态
        task = self.client._tasks[task_id]
        task["status"] = "COMPLETED"
        task["result"] = self.client._generate_mock_result(BELL_QASM, 64)
        task["result"]["task_id"] = task_id

        with patch("time.sleep"):
            result = self.client.wait_for_task(task_id, poll_interval=0.1, timeout=2.0)
        self.assertIn("counts", result)

    def test_wait_for_task_failed_raises(self):
        """任务 FAILED 时 wait_for_task 应抛出 RuntimeError。"""
        task_id = self.client.submit_quantum_task(circuit_qasm=BELL_QASM, shots=64)
        self.client._tasks[task_id]["status"] = "FAILED"
        with patch("time.sleep"), self.assertRaises(RuntimeError):
            self.client.wait_for_task(task_id, poll_interval=0.1, timeout=2.0)

    def test_wait_for_task_timeout(self):
        """任务始终未完成时 wait_for_task 应抛出 TimeoutError。"""
        task_id = self.client.submit_quantum_task(circuit_qasm=BELL_QASM, shots=64)
        # 让 get_task_status 始终返回非终态
        with (
            patch.object(self.client, "get_task_status", return_value={"status": "PENDING"}),
            patch("time.sleep"),
            self.assertRaises(TimeoutError),
        ):
            self.client.wait_for_task(task_id, poll_interval=0.05, timeout=0.1)

    def test_generate_mock_result_counts_sum_to_shots(self):
        """生成的测量计数总和应等于 shots（小 qubit 全排列路径）。"""
        result = self.client._generate_mock_result("qreg q[2];", shots=200)
        self.assertEqual(sum(result["counts"].values()), 200)
        self.assertEqual(result["shots"], 200)

    def test_generate_mock_result_bell_state(self):
        """Bell 态电路应返回 |00> 与 |11> 两个态。"""
        circuit = "qreg q[2]; h q[0]; cx q[0], q[1];"
        result = self.client._generate_mock_result(circuit, shots=1000)
        self.assertIn("00", result["counts"])
        self.assertIn("11", result["counts"])
        self.assertEqual(sum(result["counts"].values()), 1000)

    def test_generate_mock_result_large_qubit_path(self):
        """qubit 数 > 20 时应走随机采样分支且计数总和等于 shots。"""
        # count("qreg q[") 决定 num_qubits，重复 21 次使其 > 20
        circuit = "qreg q[" * 21
        result = self.client._generate_mock_result(circuit, shots=50)
        self.assertEqual(sum(result["counts"].values()), 50)

    def test_simulate_delay_zero_is_noop(self):
        """mock_delay=0 时 _simulate_delay 不应阻塞。"""
        client = MockTianyanClient(mock_delay=0.0)
        # 直接调用，若无异常即通过
        client._simulate_delay()
        self.assertEqual(client.mock_delay, 0.0)

    def test_simulate_delay_positive_sleeps(self):
        """mock_delay>0 时 _simulate_delay 应调用 time.sleep。"""
        client = MockTianyanClient(mock_delay=2.0)
        with patch("time.sleep") as mock_sleep:
            client._simulate_delay()
            mock_sleep.assert_called_once_with(2.0)

    def test_maybe_fail_triggers_error(self):
        """失败率触发时应抛出 TianyanAPIError。"""
        client = MockTianyanClient(mock_delay=0.0, mock_failure_rate=1.0)
        with patch("random.random", return_value=0.0), self.assertRaises(TianyanAPIError):
            client._maybe_fail("unit_test")
        # 错误对象应携带状态码与响应体
        try:
            with patch("random.random", return_value=0.0):
                client._maybe_fail("unit_test")
        except TianyanAPIError as e:
            self.assertEqual(e.status_code, 500)
            self.assertEqual(e.response_body, {"error": "mock_failure"})


class TestCreateTianyanClientFactory(unittest.TestCase):
    """测试 create_tianyan_client 工厂函数的多种模式选择。"""

    def test_explicit_mock_mode_true(self):
        """显式 mock_mode=True 应返回 MockTianyanClient。"""
        client = create_tianyan_client(mock_mode=True)
        self.assertIsInstance(client, MockTianyanClient)

    def test_env_mock_mode_true(self):
        """环境变量 TIANYAN_MOCK_MODE=true 时应使用 Mock 模式。"""
        with patch.dict(
            os.environ, {"TIANYAN_MOCK_MODE": "true", "TIANYAN_MOCK_DELAY": "0.0"}, clear=False
        ):
            client = create_tianyan_client(mock_mode=None)
        self.assertIsInstance(client, MockTianyanClient)
        self.assertEqual(client.mock_delay, 0.0)

    def test_env_mock_mode_yes_alias(self):
        """环境变量 TIANYAN_MOCK_MODE=yes 也应识别为 Mock 模式。"""
        with patch.dict(os.environ, {"TIANYAN_MOCK_MODE": "yes"}, clear=False):
            client = create_tianyan_client(mock_mode=None)
        self.assertIsInstance(client, MockTianyanClient)

    def test_config_file_fallback_to_mock(self):
        """无显式参数且无环境变量时，应回退读取 config.yaml（mock_mode=true）。"""
        env = _env_without("TIANYAN_MOCK_MODE")
        with patch.dict(os.environ, env, clear=True):
            client = create_tianyan_client(mock_mode=None)
        self.assertIsInstance(client, MockTianyanClient)

    def test_mock_failure_rate_from_env(self):
        """失败率应从 TIANYAN_MOCK_FAILURE_RATE 环境变量读取。"""
        with patch.dict(
            os.environ,
            {
                "TIANYAN_MOCK_MODE": "true",
                "TIANYAN_MOCK_DELAY": "0.0",
                "TIANYAN_MOCK_FAILURE_RATE": "0.3",
            },
            clear=False,
        ):
            client = create_tianyan_client(mock_mode=None)
        self.assertEqual(client.mock_failure_rate, 0.3)

    def test_explicit_mock_mode_false_returns_tianyan_client(self):
        """显式 mock_mode=False 应返回 TianyanClient（其内部会重新检测模式）。"""
        with patch("src.api.tianyan_client.load_dotenv"):
            client = create_tianyan_client(mock_mode=False)
        self.assertIsInstance(client, TianyanClient)


class TestGetClientFactory(unittest.TestCase):
    """测试 src.api.get_client 工厂函数。"""

    def test_get_client_mock_mode_true(self):
        """get_client(mock_mode=True) 应返回 MockTianyanClient。"""
        client = get_client(mock_mode=True)
        self.assertIsInstance(client, MockTianyanClient)

    def test_get_client_auto_detect(self):
        """get_client() 自动检测应返回客户端实例。"""
        client = get_client()
        # 默认配置为 mock 模式
        self.assertIsInstance(client, (MockTianyanClient, TianyanClient))


class TestGetCqlibClient(unittest.TestCase):
    """测试 src.api.get_cqlib_client 工厂函数。"""

    def test_missing_api_key_raises(self):
        """未设置 TIANYAN_API_KEY 时应抛出 ValueError。"""
        env = _env_without("TIANYAN_API_KEY")
        with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
            get_cqlib_client()

    def test_with_api_key_returns_client(self):
        """设置 TIANYAN_API_KEY 后应返回 CqlibTianyanClient 实例。"""
        with patch.dict(os.environ, {"TIANYAN_API_KEY": "fake-key-123"}, clear=False):
            client = get_cqlib_client(machine_name="tianyan_sw")
        self.assertIsInstance(client, CqlibTianyanClient)
        self.assertEqual(client.login_key, "fake-key-123")
        self.assertEqual(client.machine_name, "tianyan_sw")

    def test_default_machine_name(self):
        """默认机器名应为 tianyan_s。"""
        with patch.dict(os.environ, {"TIANYAN_API_KEY": "fake-key"}, clear=False):
            client = get_cqlib_client()
        self.assertEqual(client.machine_name, "tianyan_s")


class TestTianyanClientDetectAndConfig(unittest.TestCase):
    """测试 TianyanClient 的模式检测与配置加载静态方法。"""

    def test_detect_mock_mode_explicit(self):
        """显式传参应优先于环境变量与配置。"""
        self.assertTrue(TianyanClient._detect_mock_mode(True))
        self.assertFalse(TianyanClient._detect_mock_mode(False))

    def test_detect_mock_mode_env_true_values(self):
        """环境变量 true/1/yes 应识别为 Mock 模式。"""
        for val in ("true", "1", "yes"):
            with patch.dict(os.environ, {"TIANYAN_MOCK_MODE": val}, clear=False):
                self.assertTrue(TianyanClient._detect_mock_mode(None), f"failed for {val}")

    def test_detect_mock_mode_env_false_values(self):
        """环境变量 false/0/no 应识别为真实模式。"""
        for val in ("false", "0", "no"):
            with patch.dict(os.environ, {"TIANYAN_MOCK_MODE": val}, clear=False):
                self.assertFalse(TianyanClient._detect_mock_mode(None), f"failed for {val}")

    def test_detect_mock_mode_config_fallback(self):
        """无显式参数且无环境变量时，应回退读取 config.yaml（mock_mode=true）。"""
        env = _env_without("TIANYAN_MOCK_MODE")
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(TianyanClient._detect_mock_mode(None))

    def test_load_base_url_from_config_default_when_missing(self):
        """配置文件不存在时应返回默认 URL。"""
        url = TianyanClient._load_base_url_from_config("config/__not_exist__.yaml")
        self.assertEqual(url, "https://api.tianyanyun.cn/v1")

    def test_load_base_url_from_config_invalid_path(self):
        """配置路径无效（目录）时应返回默认 URL。"""
        url = TianyanClient._load_base_url_from_config("config")
        self.assertEqual(url, "https://api.tianyanyun.cn/v1")

    def test_load_base_url_from_config_valid(self):
        """合法配置文件应返回其中配置的 base_url。"""
        url = TianyanClient._load_base_url_from_config("config/config.yaml")
        self.assertEqual(url, "https://api.tianyanyun.cn/v1")


class TestTianyanClientMockDelegation(unittest.TestCase):
    """测试 TianyanClient 在 Mock 模式下对各方法的委托。"""

    def setUp(self):
        """构造 Mock 模式客户端，并替换内部 _mock_client 为 MagicMock 以便断言委托。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_MOCK_MODE", "TIANYAN_MOCK_DELAY")
        with patch.dict(os.environ, env, clear=True), patch("src.api.tianyan_client.load_dotenv"):
            self.client = TianyanClient(mock_mode=True)
        self.client._mock_client = MagicMock()

    def test_authenticate_delegates(self):
        """authenticate 应委托给 mock 客户端。"""
        self.client._mock_client.authenticate.return_value = True
        self.assertTrue(self.client.authenticate())
        self.client._mock_client.authenticate.assert_called_once()

    def test_submit_quantum_task_delegates(self):
        """submit_quantum_task 应委托给 mock 客户端。"""
        self.client._mock_client.submit_quantum_task.return_value = "mock-abc"
        tid = self.client.submit_quantum_task(
            circuit_qasm=BELL_QASM, shots=256, backend="tianyan-287"
        )
        self.assertEqual(tid, "mock-abc")
        self.client._mock_client.submit_quantum_task.assert_called_once_with(
            circuit_qasm=BELL_QASM, shots=256, backend="tianyan-287"
        )

    def test_get_task_status_delegates(self):
        """get_task_status 应委托给 mock 客户端。"""
        self.client._mock_client.get_task_status.return_value = {"status": "COMPLETED"}
        result = self.client.get_task_status("tid-1")
        self.assertEqual(result["status"], "COMPLETED")
        self.client._mock_client.get_task_status.assert_called_once_with("tid-1")

    def test_get_task_result_delegates(self):
        """get_task_result 应委托给 mock 客户端。"""
        self.client._mock_client.get_task_result.return_value = {"counts": {}}
        result = self.client.get_task_result("tid-1")
        self.assertIn("counts", result)
        self.client._mock_client.get_task_result.assert_called_once_with("tid-1")

    def test_list_backends_delegates(self):
        """list_backends 应委托给 mock 客户端。"""
        self.client._mock_client.list_backends.return_value = [{"name": "b1"}]
        backends = self.client.list_backends()
        self.assertEqual(backends, [{"name": "b1"}])

    def test_get_backend_info_delegates(self):
        """get_backend_info 应委托给 mock 客户端。"""
        self.client._mock_client.get_backend_info.return_value = {"name": "tianyan-287"}
        info = self.client.get_backend_info("tianyan-287")
        self.assertEqual(info["name"], "tianyan-287")

    def test_submit_classical_task_delegates(self):
        """submit_classical_task 应委托给 mock 客户端。"""
        self.client._mock_client.submit_classical_task.return_value = "mock-classical-1"
        tid = self.client.submit_classical_task(code="print(1)", language="python3")
        self.assertEqual(tid, "mock-classical-1")

    def test_get_queue_status_delegates(self):
        """get_queue_status 应委托给 mock 客户端。"""
        self.client._mock_client.get_queue_status.return_value = {"total_pending": 3}
        q = self.client.get_queue_status()
        self.assertEqual(q["total_pending"], 3)

    def test_wait_for_task_mock_mode_polls(self):
        """Mock 模式下 wait_for_task 应轮询 mock 客户端直到完成。"""
        self.client._mock_client.get_task_status.return_value = {"status": "COMPLETED"}
        self.client._mock_client.get_task_result.return_value = {"counts": {"00": 10}}
        with patch("time.sleep"):
            result = self.client.wait_for_task("tid", poll_interval=0.1, timeout=2.0)
        self.assertEqual(result["counts"]["00"], 10)

    def test_wait_for_task_mock_failed_raises(self):
        """Mock 模式下任务 FAILED 应抛出 TianyanAPIError(400)。"""
        self.client._mock_client.get_task_status.return_value = {
            "status": "FAILED",
            "error": "boom",
        }
        with patch("time.sleep"), self.assertRaises(TianyanAPIError) as ctx:
            self.client.wait_for_task("tid", poll_interval=0.1, timeout=2.0)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_wait_for_task_mock_timeout_raises(self):
        """Mock 模式下超时应抛出 TianyanAPIError(408)。"""
        self.client._mock_client.get_task_status.return_value = {"status": "PENDING"}
        with patch("time.sleep"), self.assertRaises(TianyanAPIError) as ctx:
            self.client.wait_for_task("tid", poll_interval=0.05, timeout=0.1)
        self.assertEqual(ctx.exception.status_code, 408)


class TestTianyanClientNoCqlibFallback(unittest.TestCase):
    """测试 TianyanClient 在无 cqlib 时的行为（REST 路径已移除，应抛出错误）。"""

    def setUp(self):
        """构造真实模式客户端（提供 api_key 但强制 _cqlib=None 以测试回退路径）。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_MOCK_MODE", "TIANYAN_MACHINE")
        with patch.dict(os.environ, env, clear=True), patch("src.api.tianyan_client.load_dotenv"):
            # 提供假 api_key 以通过凭证校验，随后手动置 _cqlib=None 模拟无 cqlib 场景
            self.client = TianyanClient(api_key="fake-key", mock_mode=False)
        self.client._cqlib = None

    def test_authenticate_no_cqlib_returns_false(self):
        """无 cqlib 时 authenticate 应返回 False。"""
        self.assertFalse(self.client.authenticate())

    def test_submit_quantum_task_no_cqlib_raises(self):
        """无 cqlib 时 submit_quantum_task 应抛出 TianyanAPIError。"""
        with self.assertRaises(TianyanAPIError) as ctx:
            self.client.submit_quantum_task(circuit_qasm="OPENQASM 2.0;")
        self.assertEqual(ctx.exception.status_code, 500)

    def test_submit_classical_task_no_cqlib_raises(self):
        """无 cqlib 时 submit_classical_task 应抛出 TianyanAPIError。"""
        with self.assertRaises(TianyanAPIError) as ctx:
            self.client.submit_classical_task(code="print(1)")
        self.assertEqual(ctx.exception.status_code, 500)

    def test_get_task_status_no_cqlib_raises(self):
        """无 cqlib 时 get_task_status 应抛出 TianyanAPIError。"""
        with self.assertRaises(TianyanAPIError) as ctx:
            self.client.get_task_status("tid-9")
        self.assertEqual(ctx.exception.status_code, 500)

    def test_get_task_result_no_cqlib_raises(self):
        """无 cqlib 时 get_task_result 应抛出 TianyanAPIError。"""
        with self.assertRaises(TianyanAPIError) as ctx:
            self.client.get_task_result("tid-9")
        self.assertEqual(ctx.exception.status_code, 500)

    def test_list_backends_no_cqlib_raises(self):
        """无 cqlib 时 list_backends 应抛出 TianyanAPIError。"""
        with self.assertRaises(TianyanAPIError) as ctx:
            self.client.list_backends()
        self.assertEqual(ctx.exception.status_code, 500)

    def test_get_backend_info_no_cqlib_raises(self):
        """无 cqlib 时 get_backend_info 应抛出 TianyanAPIError。"""
        with self.assertRaises(TianyanAPIError) as ctx:
            self.client.get_backend_info("tianyan-287")
        self.assertEqual(ctx.exception.status_code, 500)

    def test_get_queue_status_no_cqlib_raises(self):
        """无 cqlib 时 get_queue_status 应抛出 TianyanAPIError。"""
        with self.assertRaises(TianyanAPIError) as ctx:
            self.client.get_queue_status()
        self.assertEqual(ctx.exception.status_code, 500)


class TestTianyanClientCqlibDelegation(unittest.TestCase):
    """测试 TianyanClient 真实模式下对 cqlib 客户端的委托。"""

    def setUp(self):
        """构造真实模式客户端（带 api_key），并替换 _cqlib 为 MagicMock。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_MOCK_MODE")
        env["TIANYAN_MACHINE"] = "tianyan_s"
        with patch.dict(os.environ, env, clear=True), patch("src.api.tianyan_client.load_dotenv"):
            self.client = TianyanClient(api_key="fake-key", mock_mode=False)
        self.client._cqlib = MagicMock()

    def test_authenticate_delegates_to_cqlib(self):
        """真实模式 authenticate 应委托给 cqlib。"""
        self.client._cqlib.authenticate.return_value = True
        self.assertTrue(self.client.authenticate())
        self.client._cqlib.authenticate.assert_called_once()

    def test_submit_quantum_task_delegates_to_cqlib(self):
        """真实模式 submit_quantum_task 应委托给 cqlib（使用 qcis）。"""
        self.client._cqlib.submit_quantum_task.return_value = "real-tid"
        tid = self.client.submit_quantum_task(qcis="H Q0\nM Q0", shots=512, task_name="T")
        self.assertEqual(tid, "real-tid")
        self.client._cqlib.submit_quantum_task.assert_called_once_with(
            qcis="H Q0\nM Q0", shots=512, task_name="T"
        )

    def test_submit_quantum_task_empty_input_raises(self):
        """真实模式未提供 qcis/circuit_qasm 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            self.client.submit_quantum_task(qcis="", circuit_qasm="", shots=64)

    def test_submit_quantum_task_none_return_raises(self):
        """cqlib 返回 None 时应抛出 TianyanAPIError(500)。"""
        self.client._cqlib.submit_quantum_task.return_value = None
        with self.assertRaises(TianyanAPIError) as ctx:
            self.client.submit_quantum_task(qcis="H Q0\nM Q0", shots=64)
        self.assertEqual(ctx.exception.status_code, 500)

    def test_get_task_status_delegates_to_cqlib(self):
        """真实模式 get_task_status 应委托给 cqlib。"""
        self.client._cqlib.get_task_status.return_value = {"status": "running"}
        result = self.client.get_task_status("tid")
        self.assertEqual(result["status"], "running")

    def test_get_task_result_delegates_to_cqlib(self):
        """真实模式 get_task_result 应委托给 cqlib。"""
        self.client._cqlib.get_task_result.return_value = {"probability": 0.5}
        result = self.client.get_task_result("tid")
        self.assertEqual(result["probability"], 0.5)

    def test_list_backends_delegates_to_cqlib(self):
        """真实模式 list_backends 应委托给 cqlib。"""
        self.client._cqlib.list_backends.return_value = [{"name": "tianyan_s"}]
        backends = self.client.list_backends()
        self.assertEqual(backends, [{"name": "tianyan_s"}])

    def test_get_backend_info_delegates_to_cqlib(self):
        """真实模式 get_backend_info 应委托给 cqlib。"""
        self.client._cqlib.get_backend_info.return_value = {"name": "tianyan_s"}
        info = self.client.get_backend_info("tianyan_s")
        self.assertEqual(info["name"], "tianyan_s")

    def test_get_queue_status_delegates_to_cqlib(self):
        """真实模式 get_queue_status 应委托给 cqlib。"""
        self.client._cqlib.get_queue_status.return_value = {"total_machines": 8}
        q = self.client.get_queue_status()
        self.assertEqual(q["total_machines"], 8)

    def test_wait_for_task_delegates_to_cqlib(self):
        """真实模式 wait_for_task 应委托给 cqlib.wait_for_task。"""
        self.client._cqlib.wait_for_task.return_value = {"status": "completed"}
        result = self.client.wait_for_task("tid", poll_interval=3.0, timeout=120.0)
        self.assertEqual(result["status"], "completed")
        self.client._cqlib.wait_for_task.assert_called_once_with(
            "tid", timeout=120, poll_interval=3
        )


class TestTianyanAPIError(unittest.TestCase):
    """测试 TianyanAPIError 异常类。"""

    def test_attributes_and_message(self):
        """异常应正确存储状态码、消息与响应体。"""
        err = TianyanAPIError(503, "服务不可用", {"detail": "down"})
        self.assertEqual(err.status_code, 503)
        self.assertEqual(err.message, "服务不可用")
        self.assertEqual(err.response_body, {"detail": "down"})
        self.assertIn("503", str(err))
        self.assertIn("服务不可用", str(err))

    def test_default_response_body(self):
        """未提供 response_body 时应默认为空字典。"""
        err = TianyanAPIError(404, "未找到")
        self.assertEqual(err.response_body, {})

    def test_can_be_raised_and_caught(self):
        """异常应可被 raise 并被 except 捕获。"""
        with self.assertRaises(TianyanAPIError):
            raise TianyanAPIError(500, "boom")


class TestCqlibClient(unittest.TestCase):
    """测试 CqlibTianyanClient（platform 用 mock 替代，避免真实网络）。"""

    def setUp(self):
        """创建 cqlib 客户端并注入 mock platform。"""
        self.client = CqlibTianyanClient(
            login_key="fake-key", machine_name="tianyan_s", auto_retry_machine=True
        )
        self.client._platform = MagicMock()
        # 默认机器列表包含 tianyan_s 且状态为 running
        self.client._platform.query_quantum_computer_list.return_value = [
            ("id1", "superconducting", "running", "tianyan_s"),
            ("id2", "superconducting", "calibration", "tianyan_sw"),
        ]

    def test_real_machines_non_empty(self):
        """REAL_MACHINES 类属性应为非空列表。"""
        self.assertIsInstance(CqlibTianyanClient.REAL_MACHINES, list)
        self.assertGreater(len(CqlibTianyanClient.REAL_MACHINES), 0)
        self.assertIn("tianyan_s", CqlibTianyanClient.REAL_MACHINES)

    def test_init_stores_attributes(self):
        """__init__ 应正确存储 login_key/machine_name/auto_retry_machine 且 _platform 初始为 None。"""
        client = CqlibTianyanClient(
            login_key="k2", machine_name="tianyan_sw", auto_retry_machine=False
        )
        self.assertEqual(client.login_key, "k2")
        self.assertEqual(client.machine_name, "tianyan_sw")
        self.assertFalse(client.auto_retry_machine)
        self.assertIsNone(client._platform)

    def test_platform_lazy_load(self):
        """platform 属性应懒加载 TianYanPlatform。"""
        client = CqlibTianyanClient(
            login_key="k", machine_name="tianyan_sw", auto_retry_machine=False
        )
        client.cqlib = MagicMock()
        client.cqlib.TianYanPlatform.return_value = "PLATFORM_OBJ"
        self.assertEqual(client.platform, "PLATFORM_OBJ")
        client.cqlib.TianYanPlatform.assert_called_once_with(
            login_key="k", machine_name="tianyan_sw"
        )
        # 第二次访问应使用缓存
        _ = client.platform
        self.assertEqual(client.cqlib.TianYanPlatform.call_count, 1)

    def test_authenticate_success(self):
        """platform 可访问时 authenticate 应返回 True。"""
        self.assertTrue(self.client.authenticate())

    def test_authenticate_failure(self):
        """platform 抛异常时 authenticate 应返回 False。"""
        client = CqlibTianyanClient(login_key="k", machine_name="tianyan_s")
        client.cqlib = MagicMock()
        client.cqlib.TianYanPlatform.side_effect = Exception("auth fail")
        self.assertFalse(client.authenticate())

    def test_list_backends_success(self):
        """list_backends 应将元组列表转为字典列表。"""
        backends = self.client.list_backends()
        self.assertEqual(len(backends), 2)
        self.assertEqual(
            backends[0],
            {"id": "id1", "type": "superconducting", "status": "running", "name": "tianyan_s"},
        )

    def test_list_backends_exception_returns_empty(self):
        """查询异常时应返回空列表。"""
        self.client._platform.query_quantum_computer_list.side_effect = Exception("net")
        self.assertEqual(self.client.list_backends(), [])

    def test_get_backend_info_found(self):
        """查询已存在后端应返回其信息。"""
        info = self.client.get_backend_info("tianyan_s")
        self.assertEqual(info["name"], "tianyan_s")

    def test_get_backend_info_not_found_returns_empty(self):
        """查询不存在后端应返回空字典。"""
        self.assertEqual(self.client.get_backend_info("nope"), {})

    def test_get_backend_info_default_machine(self):
        """未指定后端名时应使用 machine_name。"""
        info = self.client.get_backend_info(None)
        self.assertEqual(info["name"], "tianyan_s")

    def test_submit_quantum_task_success_list_result(self):
        """提交成功（结果为列表）应返回 task_id 字符串。"""
        self.client._platform.submit_experiment.return_value = ["cqlib-tid-1"]
        tid = self.client.submit_quantum_task(qcis="H Q0\nM Q0", shots=128, task_name="T")
        self.assertEqual(tid, "cqlib-tid-1")

    def test_submit_quantum_task_success_non_list_result(self):
        """提交成功（结果非列表）应返回 str(result)。"""
        self.client._platform.submit_experiment.return_value = 12345
        tid = self.client.submit_quantum_task(qcis="H Q0\nM Q0", shots=128)
        self.assertEqual(tid, "12345")

    def test_submit_quantum_task_with_circuit_object(self):
        """传入 circuit 对象（带 qcis 属性）应使用其 qcis 字段。"""
        circuit = MagicMock()
        circuit.qcis = "H Q0\nM Q0"
        self.client._platform.submit_experiment.return_value = ["tid-x"]
        tid = self.client.submit_quantum_task(circuit=circuit, shots=64)
        self.assertEqual(tid, "tid-x")

    def test_submit_quantum_task_with_circuit_str(self):
        """传入 circuit 对象（无 qcis 属性）应使用 str(circuit)。"""
        circuit = "RAW_CIRCUIT_TEXT"
        self.client._platform.submit_experiment.return_value = ["tid-y"]
        tid = self.client.submit_quantum_task(circuit=circuit, shots=64)
        self.assertEqual(tid, "tid-y")
        kwargs = self.client._platform.submit_experiment.call_args.kwargs
        self.assertEqual(kwargs["circuit"], "RAW_CIRCUIT_TEXT")

    def test_submit_quantum_task_no_input_raises(self):
        """既未提供 qcis 也未提供 circuit 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            self.client.submit_quantum_task(qcis="", circuit=None, shots=64)

    def test_submit_quantum_task_machine_unavailable_with_retry(self):
        """当前机器不可用且 auto_retry=True 应调用 _retry_other_machine。"""
        with (
            patch.object(self.client, "_is_machine_available", return_value=False),
            patch.object(self.client, "_retry_other_machine", return_value="alt-tid") as mock_retry,
        ):
            tid = self.client.submit_quantum_task(qcis="H Q0\nM Q0", shots=64)
        self.assertEqual(tid, "alt-tid")
        mock_retry.assert_called_once()

    def test_submit_quantum_task_machine_unavailable_no_retry(self):
        """当前机器不可用且 auto_retry=False 应返回 None。"""
        client = CqlibTianyanClient(
            login_key="k", machine_name="tianyan_s", auto_retry_machine=False
        )
        client._platform = MagicMock()
        with patch.object(client, "_is_machine_available", return_value=False):
            tid = client.submit_quantum_task(qcis="H Q0\nM Q0", shots=64)
        self.assertIsNone(tid)

    def test_submit_quantum_task_exception_triggers_retry(self):
        """提交抛异常且 auto_retry=True 应触发 _retry_other_machine。"""
        self.client._platform.submit_experiment.side_effect = Exception("校准中")
        with (
            patch.object(self.client, "_is_machine_available", return_value=True),
            patch.object(self.client, "_retry_other_machine", return_value="alt-tid") as mock_retry,
        ):
            tid = self.client.submit_quantum_task(qcis="H Q0\nM Q0", shots=64)
        self.assertEqual(tid, "alt-tid")
        mock_retry.assert_called_once()

    def test_submit_quantum_task_exception_no_retry_returns_none(self):
        """提交抛异常且 auto_retry=False 应返回 None。"""
        client = CqlibTianyanClient(
            login_key="k", machine_name="tianyan_s", auto_retry_machine=False
        )
        client._platform = MagicMock()
        client._platform.submit_experiment.side_effect = Exception("校准中")
        with patch.object(client, "_is_machine_available", return_value=True):
            tid = client.submit_quantum_task(qcis="H Q0\nM Q0", shots=64)
        self.assertIsNone(tid)

    def test_retry_other_machine_success(self):
        """_retry_other_machine 在备用机提交成功时应返回 task_id。"""
        self.client._platform.query_quantum_computer_list.return_value = [
            ("id", "sc", "running", m) for m in CqlibTianyanClient.REAL_MACHINES
        ]
        alt_platform = MagicMock()
        alt_platform.submit_experiment.return_value = ["alt-tid"]
        self.client.cqlib = MagicMock()
        self.client.cqlib.TianYanPlatform.return_value = alt_platform
        tid = self.client._retry_other_machine("H Q0\nM Q0", 64, "T")
        self.assertEqual(tid, "alt-tid")

    def test_retry_other_machine_all_fail_returns_none(self):
        """所有备用机均失败时应返回 None。"""
        self.client._platform.query_quantum_computer_list.return_value = [
            ("id", "sc", "running", m) for m in CqlibTianyanClient.REAL_MACHINES
        ]
        alt_platform = MagicMock()
        alt_platform.submit_experiment.side_effect = Exception("fail")
        self.client.cqlib = MagicMock()
        self.client.cqlib.TianYanPlatform.return_value = alt_platform
        tid = self.client._retry_other_machine("H Q0\nM Q0", 64, "T")
        self.assertIsNone(tid)

    def test_is_machine_available_running(self):
        """机器状态为 running 时应返回 True。"""
        self.assertTrue(self.client._is_machine_available("tianyan_s"))

    def test_is_machine_available_not_running(self):
        """机器状态非 running 时应返回 False。"""
        self.assertFalse(self.client._is_machine_available("tianyan_sw"))

    def test_is_machine_available_not_found_optimistic(self):
        """未找到机器时应乐观返回 True。"""
        self.assertTrue(self.client._is_machine_available("unknown_machine"))

    def test_is_machine_available_exception_optimistic(self):
        """查询异常时应乐观返回 True。"""
        with patch.object(self.client, "list_backends", side_effect=Exception("net")):
            self.assertTrue(self.client._is_machine_available("tianyan_s"))

    def test_is_unavailable_error_keywords(self):
        """_is_unavailable_error 应识别校准/维护/不可用等关键词。"""
        for msg in (
            "机器校准中",
            "calibration in progress",
            "维护中",
            "machine busy",
            "offline now",
        ):
            self.assertTrue(CqlibTianyanClient._is_unavailable_error(msg), f"failed for {msg}")
        self.assertFalse(CqlibTianyanClient._is_unavailable_error("some random error"))

    def test_get_task_status_completed(self):
        """包含 resultStatus 的结果应返回 completed 状态。"""
        self.client._platform.query_experiment.return_value = [
            {"resultStatus": "done", "probability": {"0": 0.5, "1": 0.5}}
        ]
        status = self.client.get_task_status("tid")
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["result"], {"0": 0.5, "1": 0.5})

    def test_get_task_status_running(self):
        """不含结果字段的结果应返回 running 状态。"""
        self.client._platform.query_experiment.return_value = [{"foo": "bar"}]
        status = self.client.get_task_status("tid")
        self.assertEqual(status["status"], "running")

    def test_get_task_status_unknown(self):
        """非列表或空列表结果应返回 unknown 状态。"""
        self.client._platform.query_experiment.return_value = []
        status = self.client.get_task_status("tid")
        self.assertEqual(status["status"], "unknown")

    def test_get_task_status_error(self):
        """查询异常时应返回 error 状态。"""
        self.client._platform.query_experiment.side_effect = Exception("boom")
        status = self.client.get_task_status("tid")
        self.assertEqual(status["status"], "error")
        self.assertIn("boom", status["error"])

    def test_get_task_result_delegates_to_status(self):
        """get_task_result 应委托给 get_task_status。"""
        with patch.object(
            self.client, "get_task_status", return_value={"status": "completed"}
        ) as mock_s:
            result = self.client.get_task_result("tid")
        self.assertEqual(result["status"], "completed")
        mock_s.assert_called_once_with("tid")

    def test_wait_for_task_completed(self):
        """任务 completed 时应立即返回。"""
        with patch.object(self.client, "get_task_status", return_value={"status": "completed"}):
            result = self.client.wait_for_task("tid", timeout=10, poll_interval=1)
        self.assertEqual(result["status"], "completed")

    def test_wait_for_task_error(self):
        """任务 error 时应立即返回。"""
        with patch.object(self.client, "get_task_status", return_value={"status": "error"}):
            result = self.client.wait_for_task("tid", timeout=10, poll_interval=1)
        self.assertEqual(result["status"], "error")

    def test_wait_for_task_timeout(self):
        """超时应返回 timeout 状态。"""
        with (
            patch.object(self.client, "get_task_status", return_value={"status": "running"}),
            patch("time.sleep"),
        ):
            result = self.client.wait_for_task("tid", timeout=0.1, poll_interval=0.05)
        self.assertEqual(result["status"], "timeout")

    def test_get_queue_status(self):
        """get_queue_status 应基于 list_backends 统计 running 机器数。"""
        with patch.object(
            self.client,
            "list_backends",
            return_value=[
                {"name": "tianyan_s", "status": "running"},
                {"name": "tianyan_sw", "status": "running"},
                {"name": "tianyan_tn", "status": "calibration"},
            ],
        ):
            q = self.client.get_queue_status()
        self.assertEqual(q["total_machines"], 3)
        self.assertEqual(q["running"], 2)
        self.assertEqual(q["available"], ["tianyan_s", "tianyan_sw"])


class TestMultiMachineCoordinator(unittest.TestCase):
    """测试 MultiMachineCqlibCoordinator 与 create_multi_machine_clients 工厂。"""

    def test_create_multi_machine_clients_empty(self):
        """空机器列表应返回空字典。"""
        self.assertEqual(create_multi_machine_clients("key", []), {})

    @unittest.skipUnless(_HAS_CQLIB, "cqlib SDK not installed (CI environment)")
    def test_create_multi_machine_clients_with_machines(self):
        """给定机器列表应返回对应的客户端映射。"""
        clients = create_multi_machine_clients("key", ["tianyan_s", "tianyan_sw"])
        self.assertEqual(set(clients.keys()), {"tianyan_s", "tianyan_sw"})
        for c in clients.values():
            self.assertIsInstance(c, CqlibTianyanClient)
            self.assertFalse(c.auto_retry_machine)

    def test_coordinator_init(self):
        """协调器初始化应正确设置纳管机器与计数器。"""
        coord = MultiMachineCqlibCoordinator(
            login_key="key", machine_names=["tianyan_s", "tianyan_sw"], auto_retry_machine=False
        )
        self.assertEqual(coord.machine_names, ["tianyan_s", "tianyan_sw"])
        self.assertEqual(coord._submit_count, {"tianyan_s": 0, "tianyan_sw": 0})
        self.assertEqual(coord._fail_count, {"tianyan_s": 0, "tianyan_sw": 0})
        self.assertEqual(coord._clients, {})

    @unittest.skipUnless(_HAS_CQLIB, "cqlib SDK not installed (CI environment)")
    def test_get_client_lazy_and_cache(self):
        """_get_client 应懒加载并缓存客户端。"""
        coord = MultiMachineCqlibCoordinator(
            login_key="key", machine_names=["tianyan_s"], auto_retry_machine=False
        )
        c1 = coord._get_client("tianyan_s")
        c2 = coord._get_client("tianyan_s")
        self.assertIs(c1, c2)
        self.assertIsInstance(c1, CqlibTianyanClient)

    def test_get_client_unknown_machine_raises(self):
        """未纳管的机器应抛出 ValueError。"""
        coord = MultiMachineCqlibCoordinator(
            login_key="key", machine_names=["tianyan_s"], auto_retry_machine=False
        )
        with self.assertRaises(ValueError):
            coord._get_client("tianyan_unknown")

    def test_submit_to_machine_success(self):
        """提交成功应递增 submit_count 并返回 task_id。"""
        coord = MultiMachineCqlibCoordinator(
            login_key="key", machine_names=["tianyan_s", "tianyan_sw"], auto_retry_machine=False
        )
        mock_client = MagicMock()
        mock_client.submit_quantum_task.return_value = "tid-1"
        with patch.object(coord, "_get_client", return_value=mock_client):
            tid = coord.submit_to_machine("tianyan_s", "H Q0\nM Q0", shots=64, task_name="T")
        self.assertEqual(tid, "tid-1")
        self.assertEqual(coord._submit_count["tianyan_s"], 1)
        self.assertEqual(coord._fail_count["tianyan_s"], 0)

    def test_submit_to_machine_failure(self):
        """提交抛异常应递增 fail_count 并返回 None。"""
        coord = MultiMachineCqlibCoordinator(
            login_key="key", machine_names=["tianyan_s"], auto_retry_machine=False
        )
        mock_client = MagicMock()
        mock_client.submit_quantum_task.side_effect = Exception("boom")
        with patch.object(coord, "_get_client", return_value=mock_client):
            tid = coord.submit_to_machine("tianyan_s", "H Q0\nM Q0", shots=64)
        self.assertIsNone(tid)
        self.assertEqual(coord._fail_count["tianyan_s"], 1)
        self.assertEqual(coord._submit_count["tianyan_s"], 0)

    def test_get_all_status(self):
        """get_all_status 应聚合所有纳管机器的队列状态。"""
        coord = MultiMachineCqlibCoordinator(
            login_key="key", machine_names=["tianyan_s", "tianyan_sw"], auto_retry_machine=False
        )
        mock_client = MagicMock()
        mock_client.get_queue_status.return_value = {"total_machines": 1}
        with patch.object(coord, "_get_client", return_value=mock_client):
            status = coord.get_all_status()
        self.assertEqual(set(status.keys()), {"tianyan_s", "tianyan_sw"})
        self.assertEqual(status["tianyan_s"], {"total_machines": 1})

    def test_get_all_status_with_error(self):
        """单台机器异常时应记录 error 而不中断聚合。"""
        coord = MultiMachineCqlibCoordinator(
            login_key="key", machine_names=["tianyan_s"], auto_retry_machine=False
        )
        with patch.object(coord, "_get_client", side_effect=Exception("conn fail")):
            status = coord.get_all_status()
        self.assertIn("error", status["tianyan_s"])

    def test_get_submit_stats(self):
        """get_submit_stats 应返回各机器的提交/失败计数。"""
        coord = MultiMachineCqlibCoordinator(
            login_key="key", machine_names=["tianyan_s", "tianyan_sw"], auto_retry_machine=False
        )
        coord._submit_count["tianyan_s"] = 3
        coord._fail_count["tianyan_sw"] = 2
        stats = coord.get_submit_stats()
        self.assertEqual(stats["tianyan_s"]["submit"], 3)
        self.assertEqual(stats["tianyan_sw"]["fail"], 2)
        self.assertEqual(stats["tianyan_s"]["fail"], 0)

    @unittest.skipUnless(_HAS_CQLIB, "cqlib SDK not installed (CI environment)")
    def test_as_client_map_triggers_lazy_load(self):
        """as_client_map 应触发所有纳管机器的懒加载并返回映射。"""
        coord = MultiMachineCqlibCoordinator(
            login_key="key", machine_names=["tianyan_s", "tianyan_sw"], auto_retry_machine=False
        )
        client_map = coord.as_client_map()
        self.assertEqual(len(client_map), 2)
        for c in client_map.values():
            self.assertIsInstance(c, CqlibTianyanClient)
        # 懒加载后应已缓存到内部 _clients
        self.assertIn("tianyan_s", coord._clients)
        self.assertIn("tianyan_sw", coord._clients)


class TestTianyanClientCircuitBreaker(unittest.TestCase):
    """测试 TianyanClient 熔断器集成：连续失败熔断、OPEN 拒绝、状态查询、禁用。"""

    def setUp(self):
        """构造真实模式客户端（cqlib 模式），用于熔断器测试。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_MOCK_MODE")
        env["TIANYAN_MACHINE"] = "tianyan_s"
        with patch.dict(os.environ, env, clear=True), patch("src.api.tianyan_client.load_dotenv"):
            self.client = TianyanClient(api_key="fake-key", mock_mode=False)
        self.client._cqlib = MagicMock()

    def test_initial_circuit_state_is_closed(self):
        """初始熔断器状态应为 closed。"""
        self.assertEqual(self.client.get_circuit_state(), "closed")

    def test_circuit_opens_after_consecutive_failures(self):
        """连续失败达阈值（5 次）后熔断器应进入 OPEN 状态。"""
        self.client._cqlib.get_task_status.side_effect = RuntimeError("cqlib 网络异常")
        for _ in range(5):
            with self.assertRaises(RuntimeError):
                self.client.get_task_status("tid")
        self.assertEqual(self.client.get_circuit_state(), "open")

    def test_circuit_open_raises_circuit_open_error(self):
        """熔断器 OPEN 时后续调用应抛出 CircuitOpenError 且不实际请求。"""
        self.client._cqlib.get_task_status.side_effect = RuntimeError("cqlib 网络异常")
        for _ in range(5):
            with self.assertRaises(RuntimeError):
                self.client.get_task_status("tid")
        # 熔断器已 OPEN，下一次调用应直接抛出 CircuitOpenError
        with self.assertRaises(CircuitOpenError):
            self.client.get_task_status("tid")

    def test_get_circuit_state_reflects_half_open(self):
        """OPEN 状态经过 recovery_timeout 后应转为 HALF_OPEN 并在成功后恢复 CLOSED。"""
        cb = self.client._circuit_breaker
        cb.state = CircuitState.OPEN
        cb.last_failure_time = 0.0
        self.client._cqlib.get_task_status.return_value = {"status": "ok"}
        result = self.client.get_task_status("tid")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(self.client.get_circuit_state(), "closed")

    def test_success_resets_failure_count(self):
        """成功调用应重置失败计数，熔断器保持 CLOSED。"""
        call_count = {"n": 0}

        def _flaky_cqlib(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 3:
                raise RuntimeError("cqlib 网络异常")
            return {"status": "ok"}

        self.client._cqlib.get_task_status.side_effect = _flaky_cqlib
        for _ in range(3):
            with self.assertRaises(RuntimeError):
                self.client.get_task_status("tid")
        result = self.client.get_task_status("tid")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(self.client.get_circuit_state(), "closed")

    def test_circuit_breaker_disabled(self):
        """enable_circuit_breaker=False 时不应熔断，持续透传异常且状态保持 closed。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_MOCK_MODE")
        env["TIANYAN_MACHINE"] = "tianyan_s"
        with patch.dict(os.environ, env, clear=True), patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(
                api_key="fake-key", mock_mode=False, enable_circuit_breaker=False
            )
        client._cqlib = MagicMock()
        client._cqlib.get_task_status.side_effect = RuntimeError("cqlib 网络异常")

        for _ in range(10):
            with self.assertRaises(RuntimeError):
                client.get_task_status("tid")
        self.assertEqual(client.get_circuit_state(), "closed")


class TestTianyanClientConfigAndMetrics(unittest.TestCase):
    """测试 TianyanClient 的可配置参数（Issue #74）与 Prometheus 指标埋点（Issue #73）。"""

    def setUp(self):
        """构造 Mock 模式客户端，避免依赖真实 API。"""
        env = _env_without(
            "TIANYAN_API_KEY",
            "TIANYAN_MOCK_MODE",
            "TIANYAN_MOCK_DELAY",
            "TIANYAN_API_TIMEOUT",
            "TIANYAN_API_MAX_RETRIES",
            "TIANYAN_API_RETRY_DELAY",
        )
        with patch.dict(os.environ, env, clear=True), patch("src.api.tianyan_client.load_dotenv"):
            self.client = TianyanClient(mock_mode=True)
        self.client._mock_client = MagicMock()

    def test_timeout_parameter(self):
        """timeout 参数应可通过构造函数配置。"""
        with patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(mock_mode=True, timeout=45.0)
        self.assertEqual(client.timeout, 45.0)

    def test_retry_parameter(self):
        """max_retries 与 retry_delay 应可通过构造函数配置。"""
        with patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(mock_mode=True, max_retries=10, retry_delay=0.5)
        self.assertEqual(client.max_retries, 10)
        self.assertEqual(client.retry_delay, 0.5)

    def test_env_var_override(self):
        """环境变量应覆盖默认的 timeout/max_retries/retry_delay。"""
        env = _env_without(
            "TIANYAN_API_KEY",
            "TIANYAN_MOCK_MODE",
            "TIANYAN_API_TIMEOUT",
            "TIANYAN_API_MAX_RETRIES",
            "TIANYAN_API_RETRY_DELAY",
        )
        env["TIANYAN_API_TIMEOUT"] = "60.0"
        env["TIANYAN_API_MAX_RETRIES"] = "5"
        env["TIANYAN_API_RETRY_DELAY"] = "2.0"
        env["TIANYAN_MOCK_MODE"] = "true"
        with patch.dict(os.environ, env, clear=True), patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(mock_mode=True)
        self.assertEqual(client.timeout, 60.0)
        self.assertEqual(client.max_retries, 5)
        self.assertEqual(client.retry_delay, 2.0)

    def test_metrics_recorded(self):
        """API 调用应递增 Prometheus 请求计数指标。"""
        from prometheus_client import REGISTRY

        self.client._mock_client.authenticate.return_value = True
        before = REGISTRY.get_sample_value(
            "tianyan_api_requests_total",
            {"method": "authenticate", "endpoint": "auth"},
        )
        before = before if before is not None else 0
        self.client.authenticate()
        after = REGISTRY.get_sample_value(
            "tianyan_api_requests_total",
            {"method": "authenticate", "endpoint": "auth"},
        )
        after = after if after is not None else 0
        self.assertGreater(after, before)

    def test_metrics_error_counter(self):
        """API 调用失败时应递增 Prometheus 错误计数指标。"""
        from prometheus_client import REGISTRY

        self.client._mock_client.list_backends.side_effect = RuntimeError("boom")
        before = REGISTRY.get_sample_value(
            "tianyan_api_errors_total",
            {
                "method": "list_backends",
                "endpoint": "backends",
                "error_type": "RuntimeError",
            },
        )
        before = before if before is not None else 0
        with self.assertRaises(RuntimeError):
            self.client.list_backends()
        after = REGISTRY.get_sample_value(
            "tianyan_api_errors_total",
            {
                "method": "list_backends",
                "endpoint": "backends",
                "error_type": "RuntimeError",
            },
        )
        after = after if after is not None else 0
        self.assertGreater(after, before)

    def test_retry_behavior(self):
        """API 调用失败时应按 max_retries 进行重试。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_MOCK_MODE")
        env["TIANYAN_MACHINE"] = "tianyan_s"
        with patch.dict(os.environ, env, clear=True), patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(
                api_key="fake-key", mock_mode=False, max_retries=2, retry_delay=0.01
            )
        client._cqlib = MagicMock()
        call_count = {"n": 0}

        def _flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise RuntimeError("transient")
            return [{"name": "tianyan_s"}]

        client._cqlib.list_backends.side_effect = _flaky
        with patch("time.sleep"):
            result = client.list_backends()
        self.assertEqual(result, [{"name": "tianyan_s"}])
        self.assertEqual(call_count["n"], 3)


class TestMaskToken(unittest.TestCase):
    """测试 mask_token 令牌脱敏工具函数（Issue #78）。"""

    def test_mask_token_standard(self):
        """标准长度令牌应保留首尾各 4 个字符，中间以 * 填充。"""
        token = "abcd1234wxyz5678"
        masked = mask_token(token)
        self.assertEqual(masked, "abcd********5678")
        # 首尾 4 字符保留
        self.assertTrue(masked.startswith("abcd"))
        self.assertTrue(masked.endswith("5678"))
        # 中间字符全部脱敏
        self.assertNotIn("1234", masked)
        self.assertNotIn("wxyz", masked)

    def test_mask_token_preserves_length(self):
        """脱敏后字符串长度应与原令牌一致。"""
        token = "sk-my-secret-token-12345678"
        self.assertEqual(len(mask_token(token)), len(token))

    def test_mask_token_short(self):
        """短令牌（≤8 字符）应全部以 * 替换，避免泄露任何片段。"""
        for token in ("a", "ab", "abc", "abcd", "abcde", "abcdef", "abcdefg", "abcdefgh"):
            masked = mask_token(token)
            self.assertEqual(masked, "*" * len(token), f"failed for {token!r}")
            self.assertNotIn(token, masked)

    def test_mask_token_exactly_8_chars(self):
        """长度恰好 8 的令牌应全部脱敏（边界条件）。"""
        token = "12345678"
        self.assertEqual(mask_token(token), "********")

    def test_mask_token_exactly_9_chars(self):
        """长度 9 的令牌应保留首尾各 4 字符，中间 1 个 *。"""
        token = "123456789"
        self.assertEqual(mask_token(token), "1234*6789")

    def test_mask_token_empty(self):
        """空字符串应原样返回空字符串。"""
        self.assertEqual(mask_token(""), "")

    def test_mask_token_does_not_leak_middle(self):
        """脱敏结果不应包含原令牌的中间部分。"""
        token = "abcdefghijklmn"
        masked = mask_token(token)
        # 中间字符 'efghij' 不应出现在脱敏结果中
        for ch in "efghij":
            self.assertNotIn(ch, masked)


class TestCredentialSecurity(unittest.TestCase):
    """测试 API 凭证安全管理（Issue #78）。"""

    def test_missing_credentials_raises_error(self):
        """真实模式下未提供 api_key 且环境变量未配置时应抛出 ValueError。"""
        env = _env_without(
            "TIANYAN_API_KEY", "TIANYAN_API_TOKEN", "TIANYAN_MOCK_MODE", "TIANYAN_MACHINE"
        )
        with (
            patch.dict(os.environ, env, clear=True),
            patch("src.api.tianyan_client.load_dotenv"),
            self.assertRaises(ValueError) as ctx,
        ):
            TianyanClient(mock_mode=False)
        # 错误消息应包含环境变量名（中文提示）
        self.assertIn("TIANYAN_API_KEY", str(ctx.exception))
        self.assertIn("TIANYAN_API_TOKEN", str(ctx.exception))

    def test_missing_credentials_raises_error_with_env_only(self):
        """_load_credentials 直接调用时，缺失凭证应抛出 ValueError。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_API_TOKEN")
        with (
            patch.dict(os.environ, env, clear=True),
            self.assertRaises(ValueError) as ctx,
        ):
            TianyanClient._load_credentials(api_key=None)
        self.assertIn("TIANYAN_API_KEY", str(ctx.exception))

    def test_load_credentials_from_explicit_arg(self):
        """显式传入 api_key 应优先于环境变量。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_API_TOKEN")
        with patch.dict(os.environ, env, clear=True):
            creds = TianyanClient._load_credentials(api_key="explicit-key")
        self.assertEqual(creds["api_key"], "explicit-key")

    def test_load_credentials_from_env(self):
        """未显式传入时，应从环境变量 TIANYAN_API_KEY 读取。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_API_TOKEN")
        env["TIANYAN_API_KEY"] = "env-key-12345"
        with patch.dict(os.environ, env, clear=True):
            creds = TianyanClient._load_credentials(api_key=None)
        self.assertEqual(creds["api_key"], "env-key-12345")

    def test_load_credentials_from_alias_env(self):
        """TIANYAN_API_TOKEN 作为别名应被识别。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_API_TOKEN")
        env["TIANYAN_API_TOKEN"] = "alias-key-67890"
        with patch.dict(os.environ, env, clear=True):
            creds = TianyanClient._load_credentials(api_key=None)
        self.assertEqual(creds["api_key"], "alias-key-67890")

    def test_load_credentials_explicit_overrides_env(self):
        """显式传入的 api_key 应优先于环境变量。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_API_TOKEN")
        env["TIANYAN_API_KEY"] = "env-key"
        with patch.dict(os.environ, env, clear=True):
            creds = TianyanClient._load_credentials(api_key="explicit-key")
        self.assertEqual(creds["api_key"], "explicit-key")

    def test_load_credentials_includes_optional_fields_when_set(self):
        """设置可选环境变量时，凭证字典应包含对应字段。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_API_TOKEN")
        env["TIANYAN_API_KEY"] = "key"
        env["TIANYAN_API_SECRET"] = "secret-val"
        env["TIANYAN_APP_ID"] = "app-123"
        with patch.dict(os.environ, env, clear=True):
            creds = TianyanClient._load_credentials(api_key=None)
        self.assertEqual(creds["api_key"], "key")
        self.assertEqual(creds["api_secret"], "secret-val")
        self.assertEqual(creds["app_id"], "app-123")

    def test_load_credentials_omits_optional_fields_when_unset(self):
        """未设置可选环境变量时，凭证字典不应包含对应字段。"""
        env = _env_without(
            "TIANYAN_API_KEY", "TIANYAN_API_TOKEN", "TIANYAN_API_SECRET", "TIANYAN_APP_ID"
        )
        env["TIANYAN_API_KEY"] = "key"
        with patch.dict(os.environ, env, clear=True):
            creds = TianyanClient._load_credentials(api_key=None)
        self.assertNotIn("api_secret", creds)
        self.assertNotIn("app_id", creds)

    def test_credentials_not_logged(self):
        """验证令牌明文不会出现在日志输出中。"""
        from loguru import logger

        test_token = "sk-super-secret-token-1234567890abcdef"
        logs: list[str] = []

        def capture_sink(message: object) -> None:
            logs.append(str(message))

        logger_id = logger.add(capture_sink, level="DEBUG")
        try:
            env = _env_without("TIANYAN_API_KEY", "TIANYAN_API_TOKEN", "TIANYAN_MOCK_MODE")
            env["TIANYAN_API_KEY"] = test_token
            with (
                patch.dict(os.environ, env, clear=True),
                patch("src.api.tianyan_client.load_dotenv"),
            ):
                TianyanClient(mock_mode=False)
        finally:
            logger.remove(logger_id)

        # 完整令牌明文不应出现在任何日志行中
        for log_line in logs:
            self.assertNotIn(test_token, log_line, "令牌明文泄露到日志中")
        # 但脱敏后的令牌应出现（证明凭证已加载且日志脱敏生效）
        masked_appeared = any(mask_token(test_token) in line for line in logs)
        self.assertTrue(masked_appeared, "脱敏后的令牌应出现在日志中以证明凭证已加载")

    def test_credentials_not_logged_with_explicit_key(self):
        """显式传入的 api_key 明文同样不应出现在日志中。"""
        from loguru import logger

        test_token = "explicit-secret-key-xyz-9876543210"
        logs: list[str] = []

        def capture_sink(message: object) -> None:
            logs.append(str(message))

        logger_id = logger.add(capture_sink, level="DEBUG")
        try:
            env = _env_without("TIANYAN_API_KEY", "TIANYAN_API_TOKEN", "TIANYAN_MOCK_MODE")
            with (
                patch.dict(os.environ, env, clear=True),
                patch("src.api.tianyan_client.load_dotenv"),
            ):
                TianyanClient(api_key=test_token, mock_mode=False)
        finally:
            logger.remove(logger_id)

        for log_line in logs:
            self.assertNotIn(test_token, log_line, "令牌明文泄露到日志中")

    def test_mock_mode_does_not_require_credentials(self):
        """Mock 模式下不应调用 _load_credentials，无需凭证。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_API_TOKEN", "TIANYAN_MOCK_MODE")
        with (
            patch.dict(os.environ, env, clear=True),
            patch("src.api.tianyan_client.load_dotenv"),
            patch.object(
                TianyanClient, "_load_credentials", side_effect=AssertionError("不应调用")
            ) as mock_load,
        ):
            client = TianyanClient(mock_mode=True)
        mock_load.assert_not_called()
        self.assertTrue(client.mock_mode)


class TestTokenBucketRateLimiter(unittest.TestCase):
    """测试令牌桶限流器（Issue #84）。"""

    def test_init_capacity_and_rate(self):
        """初始化应正确存储容量与速率，令牌数初始等于容量。"""
        limiter = TokenBucketRateLimiter(capacity=10.0, rate=5.0)
        self.assertEqual(limiter.capacity, 10.0)
        self.assertEqual(limiter.rate, 5.0)
        self.assertEqual(limiter.available_tokens, 10.0)

    def test_acquire_with_sufficient_tokens(self):
        """令牌充足时 acquire 应返回 0 且消费令牌。"""
        limiter = TokenBucketRateLimiter(capacity=5.0, rate=1.0)
        wait = limiter.acquire(1.0)
        self.assertEqual(wait, 0.0)
        self.assertAlmostEqual(limiter.available_tokens, 4.0, places=1)

    def test_acquire_depletes_tokens(self):
        """连续 acquire 应能消费全部令牌。"""
        limiter = TokenBucketRateLimiter(capacity=3.0, rate=1.0)
        for _ in range(3):
            self.assertEqual(limiter.acquire(1.0), 0.0)
        # 令牌已耗尽，下一次 acquire 应返回等待时间 > 0
        wait = limiter.acquire(1.0)
        self.assertGreater(wait, 0.0)

    def test_acquire_returns_wait_time_when_empty(self):
        """令牌不足时 acquire 应返回需要等待的时间。"""
        limiter = TokenBucketRateLimiter(capacity=1.0, rate=2.0)
        limiter.acquire(1.0)  # 耗尽
        wait = limiter.acquire(1.0)
        # rate=2/s，需要 1 个令牌 → 等待约 0.5s
        self.assertGreater(wait, 0.0)
        self.assertLess(wait, 1.0)

    def test_try_acquire_succeeds_with_tokens(self):
        """令牌充足时 try_acquire 应返回 True。"""
        limiter = TokenBucketRateLimiter(capacity=2.0, rate=1.0)
        self.assertTrue(limiter.try_acquire(1.0))
        self.assertTrue(limiter.try_acquire(1.0))

    def test_try_acquire_fails_when_empty(self):
        """令牌不足时 try_acquire 应返回 False 且不消费。"""
        limiter = TokenBucketRateLimiter(capacity=1.0, rate=1.0)
        limiter.acquire(1.0)  # 耗尽
        self.assertFalse(limiter.try_acquire(1.0))

    def test_tokens_refill_over_time(self):
        """令牌应随时间按速率补充。"""
        limiter = TokenBucketRateLimiter(capacity=1.0, rate=10.0)
        limiter.acquire(1.0)  # 耗尽
        # 模拟时间流逝 0.15s → 补充 1.5 个令牌（被 capacity 截断为 1.0）。
        # 用 0.15 而非 0.1 避免 0.1*10.0 的浮点精度边界
        # （0.1*10.0 在 IEEE 754 中可能为 0.9999999999...，导致 try_acquire 失败）。
        with patch("src.api.tianyan_client.monotonic", return_value=limiter._last_refill + 0.15):
            self.assertTrue(limiter.try_acquire(1.0))

    def test_refill_capped_at_capacity(self):
        """令牌补充不应超过容量上限。"""
        limiter = TokenBucketRateLimiter(capacity=5.0, rate=100.0)
        limiter.acquire(5.0)  # 耗尽
        # 即使经过很长时间，也不应超过 capacity
        with patch("src.api.tianyan_client.monotonic", return_value=limiter._last_refill + 100):
            self.assertLessEqual(limiter.available_tokens, 5.0)

    def test_rate_limiter_disabled_in_client_by_default(self):
        """不传 max_requests_per_second 时限流器应为 None。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_MOCK_MODE", "TIANYAN_API_RATE_LIMIT")
        with patch.dict(os.environ, env, clear=True), patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(mock_mode=True)
        self.assertIsNone(client._rate_limiter)

    def test_rate_limiter_enabled_with_param(self):
        """传入 max_requests_per_second 时限流器应被创建。"""
        with patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(mock_mode=True, max_requests_per_second=10.0)
        self.assertIsNotNone(client._rate_limiter)
        self.assertEqual(client._rate_limiter.capacity, 10.0)
        self.assertEqual(client._rate_limiter.rate, 10.0)

    def test_rate_limiter_enabled_via_env(self):
        """环境变量 TIANYAN_API_RATE_LIMIT 应启用限流。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_MOCK_MODE", "TIANYAN_API_RATE_LIMIT")
        env["TIANYAN_API_RATE_LIMIT"] = "8"
        with patch.dict(os.environ, env, clear=True), patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(mock_mode=True)
        self.assertIsNotNone(client._rate_limiter)
        self.assertEqual(client._rate_limiter.rate, 8.0)

    def test_rate_limiter_zero_disables(self):
        """max_requests_per_second=0 时应禁用限流。"""
        with patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(mock_mode=True, max_requests_per_second=0)
        self.assertIsNone(client._rate_limiter)

    def test_client_apply_rate_limit_blocks_when_empty(self):
        """限流器令牌不足时 _apply_rate_limit 应调用 time.sleep。"""
        with patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(mock_mode=True, max_requests_per_second=1.0)
        # 耗尽令牌
        client._rate_limiter.acquire(1.0)
        # 下一次应触发等待
        with patch("time.sleep") as mock_sleep:
            client._apply_rate_limit()
            mock_sleep.assert_called_once()
            wait_time = mock_sleep.call_args[0][0]
            self.assertGreater(wait_time, 0.0)

    def test_client_apply_rate_limit_noop_when_disabled(self):
        """限流未启用时 _apply_rate_limit 应直接返回。"""
        with patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(mock_mode=True)
        with patch("time.sleep") as mock_sleep:
            client._apply_rate_limit()
            mock_sleep.assert_not_called()


class TestQuotaTracker(unittest.TestCase):
    """测试 API 配额追踪器（Issue #84）。"""

    def test_init_zero_counts(self):
        """初始化后计数应为 0。"""
        tracker = QuotaTracker()
        self.assertEqual(tracker.hourly_count, 0)
        self.assertEqual(tracker.daily_count, 0)

    def test_record_increments_counts(self):
        """record 应同时递增小时与日计数。"""
        tracker = QuotaTracker()
        tracker.record()
        tracker.record()
        tracker.record()
        self.assertEqual(tracker.hourly_count, 3)
        self.assertEqual(tracker.daily_count, 3)

    def test_hourly_window_reset(self):
        """超过 1 小时后小时计数应重置。"""
        tracker = QuotaTracker()
        tracker.record()
        tracker.record()
        self.assertEqual(tracker.hourly_count, 2)
        # 模拟时间推移超过 1 小时
        tracker._hourly_window_start -= 3601
        self.assertEqual(tracker.hourly_count, 0)

    def test_daily_window_reset(self):
        """超过 1 天后日计数应重置。"""
        tracker = QuotaTracker()
        for _ in range(5):
            tracker.record()
        self.assertEqual(tracker.daily_count, 5)
        # 模拟时间推移超过 1 天
        tracker._daily_window_start -= 86401
        self.assertEqual(tracker.daily_count, 0)

    def test_record_after_hourly_reset(self):
        """小时窗口重置后 record 应从 1 开始计数。"""
        tracker = QuotaTracker()
        tracker.record()
        tracker.record()
        tracker._hourly_window_start -= 3601
        tracker.record()
        self.assertEqual(tracker.hourly_count, 1)

    def test_reset_clears_counts(self):
        """reset 方法应清零所有计数。"""
        tracker = QuotaTracker()
        tracker.record()
        tracker.record()
        tracker.reset()
        self.assertEqual(tracker.hourly_count, 0)
        self.assertEqual(tracker.daily_count, 0)

    def test_client_quota_tracking(self):
        """TianyanClient 应通过 _track_quota 记录调用。"""
        with patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(mock_mode=True)
        self.assertEqual(client.get_hourly_quota(), 0)
        self.assertEqual(client.get_daily_quota(), 0)
        client._track_quota()
        client._track_quota()
        self.assertEqual(client.get_hourly_quota(), 2)
        self.assertEqual(client.get_daily_quota(), 2)

    def test_client_quota_increments_on_api_call(self):
        """真实模式 API 调用应递增配额计数。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_MOCK_MODE")
        env["TIANYAN_MACHINE"] = "tianyan_s"
        with patch.dict(os.environ, env, clear=True), patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(api_key="fake-key", mock_mode=False)
        client._cqlib = MagicMock()
        client._cqlib.list_backends.return_value = [{"name": "tianyan_s"}]
        before = client.get_daily_quota()
        client.list_backends()
        after = client.get_daily_quota()
        self.assertGreater(after, before)


class TestAdaptiveBackoff(unittest.TestCase):
    """测试 429 自适应退避（Issue #84）。"""

    def test_compute_adaptive_backoff_exponential(self):
        """退避时间应随重试次数指数增长。"""
        with patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(mock_mode=True, retry_delay=1.0)
        # 固定 jitter 为 0 以测试纯指数部分
        with patch("random.uniform", return_value=0.0):
            b0 = client._compute_adaptive_backoff(0)
            b1 = client._compute_adaptive_backoff(1)
            b2 = client._compute_adaptive_backoff(2)
        self.assertAlmostEqual(b0, 1.0, places=2)
        self.assertAlmostEqual(b1, 2.0, places=2)
        self.assertAlmostEqual(b2, 4.0, places=2)

    def test_compute_adaptive_backoff_has_jitter(self):
        """退避时间应包含随机 jitter。"""
        with patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(mock_mode=True, retry_delay=1.0)
        with patch("random.uniform", return_value=0.05):
            backoff = client._compute_adaptive_backoff(0)
        self.assertAlmostEqual(backoff, 1.05, places=2)

    def test_compute_adaptive_backoff_retry_after_override(self):
        """服务端 Retry-After 大于指数退避时应采用 Retry-After。"""
        with patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(mock_mode=True, retry_delay=0.1)
        with patch("random.uniform", return_value=0.0):
            backoff = client._compute_adaptive_backoff(0, retry_after=10.0)
        self.assertGreaterEqual(backoff, 10.0)

    def test_is_rate_limited_detects_429(self):
        """_is_rate_limited 应识别 status_code=429 的异常。"""
        exc = TianyanAPIError(status_code=429, message="Too Many Requests")
        self.assertTrue(TianyanClient._is_rate_limited(exc))

    def test_is_rate_limited_detects_rate_limit_error(self):
        """_is_rate_limited 应识别 RateLimitError。"""
        exc = RateLimitError("限流")
        self.assertTrue(TianyanClient._is_rate_limited(exc))

    def test_is_rate_limited_ignores_other_errors(self):
        """_is_rate_limited 应忽略非 429 异常。"""
        exc = TianyanAPIError(status_code=500, message="内部错误")
        self.assertFalse(TianyanClient._is_rate_limited(exc))
        self.assertFalse(TianyanClient._is_rate_limited(RuntimeError("boom")))

    def test_429_triggers_adaptive_backoff_retry(self):
        """429 响应应触发自适应退避重试并最终成功。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_MOCK_MODE")
        env["TIANYAN_MACHINE"] = "tianyan_s"
        with patch.dict(os.environ, env, clear=True), patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(
                api_key="fake-key", mock_mode=False, max_retries=3, retry_delay=0.01
            )
        client._cqlib = MagicMock()
        call_count = {"n": 0}

        def _flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise TianyanAPIError(status_code=429, message="Too Many Requests")
            return [{"name": "tianyan_s"}]

        client._cqlib.list_backends.side_effect = _flaky
        with patch("time.sleep"), patch("random.uniform", return_value=0.0):
            result = client.list_backends()
        self.assertEqual(result, [{"name": "tianyan_s"}])
        self.assertEqual(call_count["n"], 3)

    def test_429_exhausted_raises_rate_limit_error(self):
        """429 重试耗尽应抛出 RateLimitError。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_MOCK_MODE")
        env["TIANYAN_MACHINE"] = "tianyan_s"
        with patch.dict(os.environ, env, clear=True), patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(
                api_key="fake-key", mock_mode=False, max_retries=1, retry_delay=0.01
            )
        client._cqlib = MagicMock()
        client._cqlib.list_backends.side_effect = TianyanAPIError(
            status_code=429, message="Too Many Requests"
        )
        with (
            patch("time.sleep"),
            patch("random.uniform", return_value=0.0),
            self.assertRaises(RateLimitError),
        ):
            client.list_backends()

    def test_429_converted_to_rate_limit_error(self):
        """429 TianyanAPIError 应被转换为 RateLimitError 抛出。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_MOCK_MODE")
        env["TIANYAN_MACHINE"] = "tianyan_s"
        with patch.dict(os.environ, env, clear=True), patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(
                api_key="fake-key", mock_mode=False, max_retries=0, retry_delay=0.01
            )
        client._cqlib = MagicMock()
        client._cqlib.get_backend_info.side_effect = TianyanAPIError(
            status_code=429, message="Too Many Requests"
        )
        with self.assertRaises(RateLimitError) as ctx:
            client.get_backend_info("tianyan_s")
        self.assertEqual(ctx.exception.code, "RATE_LIMIT")
        self.assertTrue(ctx.exception.retryable)

    def test_non_429_error_uses_fixed_retry_delay(self):
        """非 429 异常应使用固定 retry_delay 重试。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_MOCK_MODE")
        env["TIANYAN_MACHINE"] = "tianyan_s"
        with patch.dict(os.environ, env, clear=True), patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(
                api_key="fake-key", mock_mode=False, max_retries=2, retry_delay=0.05
            )
        client._cqlib = MagicMock()
        call_count = {"n": 0}

        def _flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise RuntimeError("网络错误")
            return [{"name": "tianyan_s"}]

        client._cqlib.list_backends.side_effect = _flaky
        with patch("time.sleep") as mock_sleep:
            result = client.list_backends()
        self.assertEqual(result, [{"name": "tianyan_s"}])
        # 非 429 应使用固定 retry_delay
        for call_args in mock_sleep.call_args_list:
            self.assertEqual(call_args[0][0], 0.05)

    def test_retry_after_from_rate_limit_error(self):
        """RateLimitError 携带的 retry_after 应用于退避计算。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_MOCK_MODE")
        env["TIANYAN_MACHINE"] = "tianyan_s"
        with patch.dict(os.environ, env, clear=True), patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(
                api_key="fake-key", mock_mode=False, max_retries=1, retry_delay=0.01
            )
        client._cqlib = MagicMock()
        client._cqlib.get_queue_status.side_effect = RateLimitError("限流", retry_after=5.0)
        with (
            patch("time.sleep") as mock_sleep,
            patch("random.uniform", return_value=0.0),
            self.assertRaises(RateLimitError),
        ):
            client.get_queue_status()
        # 应使用 retry_after=5.0 而非指数退避
        mock_sleep.assert_called_with(5.0)


class TestRateLimitCircuitBreakerIntegration(unittest.TestCase):
    """测试限流与熔断器联动（Issue #84）：限流失败不触发熔断。"""

    def setUp(self):
        """构造真实模式客户端（带熔断器与限流）。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_MOCK_MODE", "TIANYAN_API_RATE_LIMIT")
        env["TIANYAN_MACHINE"] = "tianyan_s"
        with patch.dict(os.environ, env, clear=True), patch("src.api.tianyan_client.load_dotenv"):
            self.client = TianyanClient(
                api_key="fake-key",
                mock_mode=False,
                max_retries=0,
                retry_delay=0.01,
            )
        self.client._cqlib = MagicMock()

    def test_rate_limit_does_not_open_circuit_breaker(self):
        """429 限流连续失败不应触发熔断器 OPEN。"""
        self.client._cqlib.get_task_status.side_effect = TianyanAPIError(
            status_code=429, message="Too Many Requests"
        )
        # 连续 10 次限流失败（远超熔断阈值 5）
        for _ in range(10):
            with self.assertRaises(RateLimitError):
                self.client.get_task_status("tid")
        # 熔断器应仍为 closed
        self.assertEqual(self.client.get_circuit_state(), "closed")

    def test_rate_limit_error_skips_circuit_breaker_on_failure(self):
        """RateLimitError 不应调用熔断器 on_failure。"""
        self.client._cqlib.get_task_status.side_effect = RateLimitError("限流")
        with self.assertRaises(RateLimitError):
            self.client.get_task_status("tid")
        # 失败计数应仍为 0
        if self.client._circuit_breaker:
            self.assertEqual(self.client._circuit_breaker.failure_count, 0)

    def test_non_rate_limit_error_triggers_circuit_breaker(self):
        """非限流异常仍应正常触发熔断器失败计数。"""
        self.client._cqlib.get_task_status.side_effect = RuntimeError("网络错误")
        with self.assertRaises(RuntimeError):
            self.client.get_task_status("tid")
        if self.client._circuit_breaker:
            self.assertEqual(self.client._circuit_breaker.failure_count, 1)

    def test_circuit_breaker_opens_on_non_rate_limit_errors(self):
        """非限流连续失败达阈值后熔断器应 OPEN。"""
        self.client._cqlib.get_task_status.side_effect = RuntimeError("网络错误")
        for _ in range(5):
            with self.assertRaises(RuntimeError):
                self.client.get_task_status("tid")
        self.assertEqual(self.client.get_circuit_state(), "open")

    def test_mixed_errors_rate_limit_does_not_count(self):
        """限流与普通错误混合时，仅普通错误计入熔断器。"""
        # 3 次限流 + 3 次普通错误 = 熔断器失败计数应为 3（未达阈值 5）
        for _ in range(3):
            self.client._cqlib.get_task_status.side_effect = TianyanAPIError(
                status_code=429, message="限流"
            )
            with self.assertRaises(RateLimitError):
                self.client.get_task_status("tid")

        for _ in range(3):
            self.client._cqlib.get_task_status.side_effect = RuntimeError("网络错误")
            with self.assertRaises(RuntimeError):
                self.client.get_task_status("tid")

        if self.client._circuit_breaker:
            self.assertEqual(self.client._circuit_breaker.failure_count, 3)
        self.assertEqual(self.client.get_circuit_state(), "closed")

    def test_rate_limit_then_success_resets_nothing(self):
        """限流失败后成功调用，熔断器失败计数应保持 0（从未递增）。"""
        self.client._cqlib.get_task_status.side_effect = TianyanAPIError(
            status_code=429, message="限流"
        )
        with self.assertRaises(RateLimitError):
            self.client.get_task_status("tid")

        self.client._cqlib.get_task_status.side_effect = None
        self.client._cqlib.get_task_status.return_value = {"status": "ok"}
        result = self.client.get_task_status("tid")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(self.client.get_circuit_state(), "closed")
        if self.client._circuit_breaker:
            self.assertEqual(self.client._circuit_breaker.failure_count, 0)


class TestComprehensiveApiClient(unittest.TestCase):
    """综合 API 客户端测试（Issue #76）：补全核心方法的 mock 测试。"""

    def setUp(self):
        """构造真实模式客户端（cqlib 委托），替换 _cqlib 为 MagicMock。"""
        env = _env_without("TIANYAN_API_KEY", "TIANYAN_MOCK_MODE")
        env["TIANYAN_MACHINE"] = "tianyan_s"
        with patch.dict(os.environ, env, clear=True), patch("src.api.tianyan_client.load_dotenv"):
            self.client = TianyanClient(
                api_key="fake-key", mock_mode=False, max_retries=1, retry_delay=0.01
            )
        self.client._cqlib = MagicMock()

    # -- submit_quantum_task 综合测试 --

    def test_submit_quantum_task_success(self):
        """submit_quantum_task 成功应返回 task_id。"""
        self.client._cqlib.submit_quantum_task.return_value = "real-tid-001"
        tid = self.client.submit_quantum_task(qcis="H Q0\nM Q0", shots=1024, task_name="Test")
        self.assertEqual(tid, "real-tid-001")
        self.client._cqlib.submit_quantum_task.assert_called_once_with(
            qcis="H Q0\nM Q0", shots=1024, task_name="Test"
        )

    def test_submit_quantum_task_retries_on_failure(self):
        """submit_quantum_task 失败应重试并最终成功。"""
        call_count = {"n": 0}

        def _flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("瞬时错误")
            return "tid-after-retry"

        self.client._cqlib.submit_quantum_task.side_effect = _flaky
        with patch("time.sleep"):
            tid = self.client.submit_quantum_task(qcis="H Q0\nM Q0", shots=64)
        self.assertEqual(tid, "tid-after-retry")
        self.assertEqual(call_count["n"], 2)

    def test_submit_quantum_task_retries_exhausted(self):
        """submit_quantum_task 重试耗尽应抛出最后一次异常。"""
        self.client._cqlib.submit_quantum_task.side_effect = RuntimeError("持续失败")
        with patch("time.sleep"), self.assertRaises(RuntimeError):
            self.client.submit_quantum_task(qcis="H Q0\nM Q0", shots=64)

    def test_submit_quantum_task_with_circuit_qasm_fallback(self):
        """未提供 qcis 时应回退使用 circuit_qasm。"""
        self.client._cqlib.submit_quantum_task.return_value = "tid-qasm"
        tid = self.client.submit_quantum_task(circuit_qasm="OPENQASM 2.0;", shots=128)
        self.assertEqual(tid, "tid-qasm")
        kwargs = self.client._cqlib.submit_quantum_task.call_args.kwargs
        self.assertEqual(kwargs["qcis"], "OPENQASM 2.0;")

    def test_submit_quantum_task_none_return_raises(self):
        """cqlib 返回 None 时应抛出 TianyanAPIError(500)。"""
        self.client._cqlib.submit_quantum_task.return_value = None
        with self.assertRaises(TianyanAPIError) as ctx:
            self.client.submit_quantum_task(qcis="H Q0\nM Q0", shots=64)
        self.assertEqual(ctx.exception.status_code, 500)

    # -- get_task_result 综合测试 --

    def test_get_task_result_success(self):
        """get_task_result 成功应返回结果字典。"""
        self.client._cqlib.get_task_result.return_value = {
            "counts": {"00": 512, "11": 512},
            "shots": 1024,
        }
        result = self.client.get_task_result("tid-100")
        self.assertEqual(result["counts"]["00"], 512)
        self.client._cqlib.get_task_result.assert_called_once_with("tid-100")

    def test_get_task_result_retries_on_failure(self):
        """get_task_result 失败应重试。"""
        call_count = {"n": 0}

        def _flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("超时")
            return {"counts": {"0": 1024}}

        self.client._cqlib.get_task_result.side_effect = _flaky
        with patch("time.sleep"):
            result = self.client.get_task_result("tid-200")
        self.assertEqual(result["counts"]["0"], 1024)
        self.assertEqual(call_count["n"], 2)

    def test_get_task_result_retries_exhausted(self):
        """get_task_result 重试耗尽应抛出异常。"""
        self.client._cqlib.get_task_result.side_effect = RuntimeError("持续失败")
        with patch("time.sleep"), self.assertRaises(RuntimeError):
            self.client.get_task_result("tid-300")

    # -- get_backend_info 综合测试 --

    def test_get_backend_info_tianyan_287(self):
        """get_backend_info 应返回 tianyan-287 后端详情。"""
        self.client._cqlib.get_backend_info.return_value = {
            "name": "tianyan-287",
            "num_qubits": 287,
            "status": "online",
        }
        info = self.client.get_backend_info("tianyan-287")
        self.assertEqual(info["name"], "tianyan-287")
        self.assertEqual(info["num_qubits"], 287)

    def test_get_backend_info_simulator(self):
        """get_backend_info 应返回模拟器后端详情。"""
        self.client._cqlib.get_backend_info.return_value = {
            "name": "tianyan-simulator",
            "num_qubits": 30,
            "status": "online",
        }
        info = self.client.get_backend_info("tianyan-simulator")
        self.assertEqual(info["num_qubits"], 30)

    def test_get_backend_info_retries_on_failure(self):
        """get_backend_info 失败应重试。"""
        call_count = {"n": 0}

        def _flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("网络错误")
            return {"name": "tianyan_s"}

        self.client._cqlib.get_backend_info.side_effect = _flaky
        with patch("time.sleep"):
            info = self.client.get_backend_info("tianyan_s")
        self.assertEqual(info["name"], "tianyan_s")
        self.assertEqual(call_count["n"], 2)

    def test_get_backend_info_nonexistent_raises(self):
        """查询不存在后端应抛出异常（cqlib 侧）。"""
        self.client._cqlib.get_backend_info.side_effect = RuntimeError("后端不存在")
        with patch("time.sleep"), self.assertRaises(RuntimeError):
            self.client.get_backend_info("no-such-backend")

    # -- get_queue_status 综合测试 --

    def test_get_queue_status_normal(self):
        """get_queue_status 应返回正常队列状态。"""
        self.client._cqlib.get_queue_status.return_value = {
            "total_pending": 5,
            "total_running": 2,
            "queue_capacity": 100,
        }
        q = self.client.get_queue_status()
        self.assertEqual(q["total_pending"], 5)
        self.assertEqual(q["total_running"], 2)

    def test_get_queue_status_deep_queue(self):
        """get_queue_status 应能处理深度队列场景。"""
        self.client._cqlib.get_queue_status.return_value = {
            "total_pending": 500,
            "total_running": 10,
            "queue_capacity": 1000,
        }
        q = self.client.get_queue_status()
        self.assertEqual(q["total_pending"], 500)

    def test_get_queue_status_empty_queue(self):
        """get_queue_status 应能处理空队列场景。"""
        self.client._cqlib.get_queue_status.return_value = {
            "total_pending": 0,
            "total_running": 0,
            "queue_capacity": 100,
        }
        q = self.client.get_queue_status()
        self.assertEqual(q["total_pending"], 0)

    def test_get_queue_status_retries_on_failure(self):
        """get_queue_status 失败应重试。"""
        call_count = {"n": 0}

        def _flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("超时")
            return {"total_pending": 1}

        self.client._cqlib.get_queue_status.side_effect = _flaky
        with patch("time.sleep"):
            q = self.client.get_queue_status()
        self.assertEqual(q["total_pending"], 1)
        self.assertEqual(call_count["n"], 2)

    # -- authenticate 综合测试 --

    def test_authenticate_success(self):
        """authenticate 成功应返回 True。"""
        self.client._cqlib.authenticate.return_value = True
        self.assertTrue(self.client.authenticate())
        self.client._cqlib.authenticate.assert_called_once()

    def test_authenticate_failure_returns_false(self):
        """authenticate 失败应返回 False。"""
        self.client._cqlib.authenticate.return_value = False
        self.assertFalse(self.client.authenticate())

    def test_authenticate_retries_on_transient_error(self):
        """authenticate 瞬时错误应重试并最终成功。"""
        call_count = {"n": 0}

        def _flaky():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("连接超时")
            return True

        self.client._cqlib.authenticate.side_effect = _flaky
        with patch("time.sleep"):
            result = self.client.authenticate()
        self.assertTrue(result)
        self.assertEqual(call_count["n"], 2)

    def test_authenticate_retries_exhausted_raises(self):
        """authenticate 重试耗尽应抛出最后一次异常。"""
        self.client._cqlib.authenticate.side_effect = RuntimeError("持续失败")
        with patch("time.sleep"), self.assertRaises(RuntimeError):
            self.client.authenticate()

    # -- 熔断器状态转换综合测试 --

    def test_circuit_breaker_closed_to_open(self):
        """CLOSED → OPEN：连续失败达阈值应熔断。"""
        self.client._cqlib.get_task_status.side_effect = RuntimeError("网络错误")
        for _ in range(5):
            with self.assertRaises(RuntimeError):
                self.client.get_task_status("tid")
        self.assertEqual(self.client.get_circuit_state(), "open")

    def test_circuit_breaker_open_rejects_request(self):
        """OPEN 状态应拒绝请求并抛出 CircuitOpenError。"""
        self.client._cqlib.get_task_status.side_effect = RuntimeError("网络错误")
        for _ in range(5):
            with self.assertRaises(RuntimeError):
                self.client.get_task_status("tid")
        with self.assertRaises(CircuitOpenError):
            self.client.get_task_status("tid")

    def test_circuit_breaker_open_to_half_open_to_closed(self):
        """OPEN → HALF_OPEN → CLOSED：恢复超时后试探成功应恢复。"""
        cb = self.client._circuit_breaker
        self.assertIsNotNone(cb)
        cb.state = CircuitState.OPEN
        cb.last_failure_time = 0.0  # 模拟已过恢复超时
        self.client._cqlib.get_task_status.return_value = {"status": "ok"}
        result = self.client.get_task_status("tid")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(self.client.get_circuit_state(), "closed")

    def test_circuit_breaker_half_open_failure_increments_count(self):
        """HALF_OPEN 状态下试探失败应递增失败计数。"""
        cb = self.client._circuit_breaker
        self.assertIsNotNone(cb)
        cb.state = CircuitState.HALF_OPEN
        cb.failure_count = 0
        self.client._cqlib.get_task_status.side_effect = RuntimeError("仍故障")
        with self.assertRaises(RuntimeError):
            self.client.get_task_status("tid")
        # 失败计数应递增（HALF_OPEN 下 on_failure 仍计数）
        self.assertEqual(cb.failure_count, 1)

    def test_circuit_breaker_success_resets_failure_count(self):
        """成功调用应重置失败计数。"""
        self.client._cqlib.get_task_status.side_effect = [
            RuntimeError("错误1"),
            RuntimeError("错误2"),
            {"status": "ok"},
        ]
        with patch("time.sleep"):
            # 两次失败
            with self.assertRaises(RuntimeError):
                self.client.get_task_status("tid")
            with self.assertRaises(RuntimeError):
                self.client.get_task_status("tid")
            # 第三次成功
            result = self.client.get_task_status("tid")
        self.assertEqual(result["status"], "ok")
        if self.client._circuit_breaker:
            self.assertEqual(self.client._circuit_breaker.failure_count, 0)

    # -- 错误处理路径 --

    def test_submit_quantum_task_no_cqlib_raises(self):
        """无 cqlib 时 submit_quantum_task 应抛出 TianyanAPIError(500)。"""
        self.client._cqlib = None
        with self.assertRaises(TianyanAPIError) as ctx:
            self.client.submit_quantum_task(qcis="H Q0\nM Q0", shots=64)
        self.assertEqual(ctx.exception.status_code, 500)

    def test_get_task_result_no_cqlib_raises(self):
        """无 cqlib 时 get_task_result 应抛出 TianyanAPIError(500)。"""
        self.client._cqlib = None
        with self.assertRaises(TianyanAPIError) as ctx:
            self.client.get_task_result("tid")
        self.assertEqual(ctx.exception.status_code, 500)

    def test_get_backend_info_no_cqlib_raises(self):
        """无 cqlib 时 get_backend_info 应抛出 TianyanAPIError(500)。"""
        self.client._cqlib = None
        with self.assertRaises(TianyanAPIError) as ctx:
            self.client.get_backend_info("tianyan_s")
        self.assertEqual(ctx.exception.status_code, 500)

    def test_get_queue_status_no_cqlib_raises(self):
        """无 cqlib 时 get_queue_status 应抛出 TianyanAPIError(500)。"""
        self.client._cqlib = None
        with self.assertRaises(TianyanAPIError) as ctx:
            self.client.get_queue_status()
        self.assertEqual(ctx.exception.status_code, 500)

    def test_authenticate_no_cqlib_returns_false(self):
        """无 cqlib 时 authenticate 应返回 False。"""
        self.client._cqlib = None
        self.assertFalse(self.client.authenticate())

    def test_submit_quantum_task_empty_input_raises_value_error(self):
        """真实模式未提供 qcis/circuit_qasm 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            self.client.submit_quantum_task(qcis="", circuit_qasm="", shots=64)

    def test_timeout_parameter_configurable(self):
        """timeout 应可通过构造函数配置。"""
        with patch("src.api.tianyan_client.load_dotenv"):
            client = TianyanClient(mock_mode=True, timeout=99.0)
        self.assertEqual(client.timeout, 99.0)


if __name__ == "__main__":
    unittest.main()
