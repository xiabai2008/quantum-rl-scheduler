"""
量子RL调度系统 - 集成测试
Integration Tests for End-to-End Pipelines

测试覆盖：
- Mock 全流程集成测试：任务提交→解析→调度→执行→结果返回
- Mock 客户端集成测试：MockTianyanClient 完整生命周期
- Web API 集成测试：FastAPI TestClient 端到端流程
- 错误恢复集成测试：异常输入与网络错误恢复

所有测试均使用 Mock 客户端，不依赖真实 API 或外部服务。
"""

import copy
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.mock_client import MockTianyanClient
from src.api.tianyan_client import TianyanAPIError
from src.scheduler.env import (
    OBS_DIM,
    QuantumSchedulingEnv,
)
from src.scheduler.env import (
    Task as EnvTask,
)
from src.scheduler.parser import (
    Task as ParserTask,
)
from src.scheduler.parser import (
    TaskParser,
)
from src.visualization.app import app

# 通过 sys.modules 获取 visualization.app 子模块，绕过 __init__.py 的属性遮蔽
# （src/visualization/__init__.py 把 app 属性覆盖为 FastAPI 实例，遮蔽了子模块）
app_module = sys.modules["src.visualization.app"]


# ============================================================
# 测试常量与辅助函数
# ============================================================

# Bell 态 QASM 电路，用于 Mock 客户端提交量子任务测试
BELL_QASM = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
""".strip()


def make_task_dict(task_id: str = "test_task_001", **overrides: Any) -> dict[str, Any]:
    """构造测试用任务字典。

    Args:
        task_id: 任务 ID
        **overrides: 覆盖默认字段的键值对

    Returns:
        符合 TaskParser.parse 输入要求的任务字典
    """
    base: dict[str, Any] = {
        "task_id": task_id,
        "type": "quantum",
        "algorithm": "VQE",
        "qubits_required": 8,
        "circuit_depth": 50,
        "shots": 1024,
        "estimated_time": 120.0,
        "priority": "high",
    }
    base.update(overrides)
    return base


# ============================================================
# 1. Mock 全流程集成测试
# ============================================================


class TestMockFullPipeline(unittest.TestCase):
    """端到端集成测试：任务提交→解析→调度→执行→结果返回"""

    def test_parse_schedule_execute(self):
        """解析任务→创建环境→step调度→验证奖励"""
        # 1. 解析任务
        parser = TaskParser()
        task_dict = make_task_dict(task_id="pipe_001", priority="urgent", qubits_required=10)
        parsed_task = parser.parse(task_dict)
        self.assertEqual(parsed_task.task_id, "pipe_001")
        self.assertEqual(parsed_task.task_type, "quantum")
        self.assertEqual(parsed_task.qubits_required, 10)

        # 2. 创建环境并 reset
        env = QuantumSchedulingEnv(max_steps=50, seed=42)
        obs, info = env.reset(seed=42)
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertIsInstance(info, dict)

        # 3. step 调度（多步随机动作）
        total_reward = 0.0
        steps_taken = 0
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps_taken += 1
            if terminated or truncated:
                break

        # 4. 验证奖励和状态
        self.assertGreater(steps_taken, 0)
        self.assertIsInstance(total_reward, float)
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertTrue(np.all(obs >= 0.0))
        self.assertTrue(np.all(obs <= 1.0))
        env.close()

    def test_batch_parse_and_schedule(self):
        """批量解析→逐个调度→统计完成率"""
        parser = TaskParser()
        # 构造 5 个任务字典
        task_dicts = [
            make_task_dict(f"batch_{i:03d}", qubits_required=4 + i * 2, priority="medium")
            for i in range(5)
        ]

        # 批量解析
        tasks: list[ParserTask] = []
        for td in task_dicts:
            task = parser.parse(td)
            tasks.append(task)
        self.assertEqual(len(tasks), 5)
        for i, task in enumerate(tasks):
            self.assertEqual(task.task_id, f"batch_{i:03d}")

        # 创建环境并调度
        env = QuantumSchedulingEnv(max_steps=30, seed=123)
        env.reset(seed=123)
        scheduled_count = 0
        for _ in range(20):
            action = env.action_space.sample()
            _obs, _reward, terminated, truncated, info = env.step(action)
            scheduled_count = info.get("total_scheduled", 0)
            if terminated or truncated:
                break

        # 验证至少有部分任务被成功调度
        self.assertGreater(scheduled_count, 0)
        env.close()

    def test_parser_to_env_data_flow(self):
        """parser 输出 QuantumTask → env 接收任务→验证状态"""
        # 1. parser 输出 Task
        parser = TaskParser()
        task_dict = make_task_dict(task_id="flow_001", qubits_required=10, priority="urgent")
        parsed_task = parser.parse(task_dict)
        self.assertEqual(parsed_task.task_id, "flow_001")
        self.assertEqual(parsed_task.task_type, "quantum")
        self.assertEqual(parsed_task.qubits_required, 10)

        # 2. 转换为 env Task 并注入环境
        # parser 的 "hybrid" 类型映射到 env 的 "universal" 类型
        env_task_type = "universal" if parsed_task.task_type == "hybrid" else parsed_task.task_type
        env_task = EnvTask(
            task_id=parsed_task.task_id,
            task_type=env_task_type,
            qubit_count=parsed_task.qubits_required,
            wait_steps=0,
            urgency=float(parsed_task.priority) / 4.0,
            priority=min(parsed_task.priority + 1, 5),
            execution_time=max(1, int(parsed_task.estimated_time / 10)),
        )

        env = QuantumSchedulingEnv(max_steps=20, seed=42)
        env.reset(seed=42)
        initial_queue_len = len(env._task_queue)
        env._task_queue.append(env_task)

        # 3. 验证 env 接收了任务
        self.assertEqual(len(env._task_queue), initial_queue_len + 1)
        injected = env._task_queue[-1]
        self.assertEqual(injected.task_id, "flow_001")
        self.assertEqual(injected.qubit_count, 10)
        self.assertEqual(injected.task_type, "quantum")

        # 4. 验证 env 状态向量正常
        obs = env._get_observation()
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertTrue(np.all(obs >= 0.0))
        self.assertTrue(np.all(obs <= 1.0))

        env.close()

    def test_multi_task_scheduling_cycle(self):
        """多任务完整调度周期（提交→排队→执行→完成）"""
        client = MockTianyanClient(mock_delay=0.0, mock_failure_rate=0.0)

        # 提交多个量子任务
        task_ids: list[str] = []
        for _i in range(3):
            tid = client.submit_quantum_task(
                circuit_qasm=BELL_QASM,
                shots=1024,
                backend="tianyan-287",
            )
            task_ids.append(tid)

        self.assertEqual(len(task_ids), 3)
        # 验证 task_id 唯一性
        self.assertEqual(len(set(task_ids)), 3)
        # 验证所有 task_id 以 "mock-" 开头
        for tid in task_ids:
            self.assertTrue(tid.startswith("mock-"))

        # 等待所有任务完成
        for tid in task_ids:
            result = client.wait_for_task(tid, poll_interval=0.01, timeout=30.0)
            self.assertEqual(result["status"], "COMPLETED")
            self.assertIn("counts", result)
            self.assertEqual(result["shots"], 1024)
            self.assertEqual(result["backend"], "tianyan-287")


# ============================================================
# 2. Mock 客户端集成测试
# ============================================================


class TestMockClientIntegration(unittest.TestCase):
    """Mock 客户端集成测试：MockTianyanClient 完整生命周期"""

    def test_mock_client_submit_and_result(self):
        """submit_quantum_task→get_task_result 完整流程"""
        client = MockTianyanClient(mock_delay=0.0, mock_failure_rate=0.0)

        # 提交量子任务
        task_id = client.submit_quantum_task(
            circuit_qasm=BELL_QASM,
            shots=2048,
            backend="tianyan-287",
        )
        self.assertTrue(task_id.startswith("mock-"))

        # 等待任务完成并获取结果
        result = client.wait_for_task(task_id, poll_interval=0.01, timeout=30.0)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["backend"], "tianyan-287")
        self.assertEqual(result["shots"], 2048)
        self.assertIn("counts", result)
        self.assertIsInstance(result["counts"], dict)
        # Bell 态电路应产生 00 和 11 两个测量结果
        self.assertIn("00", result["counts"])
        self.assertIn("11", result["counts"])

    def test_mock_client_authenticate(self):
        """authenticate() 成功流程"""
        client = MockTianyanClient(mock_delay=0.0, mock_failure_rate=0.0)
        result = client.authenticate()
        self.assertTrue(result)

    def test_mock_client_backend_info(self):
        """get_backend_info 返回结构验证"""
        client = MockTianyanClient(mock_delay=0.0, mock_failure_rate=0.0)
        info = client.get_backend_info("tianyan-287")
        self.assertEqual(info["name"], "tianyan-287")
        self.assertEqual(info["type"], "superconducting")
        self.assertEqual(info["num_qubits"], 287)
        self.assertIn("fidelity", info)
        self.assertIn("single_qubit_gate", info["fidelity"])
        self.assertIn("two_qubit_gate", info["fidelity"])
        self.assertIn("readout", info["fidelity"])
        self.assertIn("queue_depth", info)
        self.assertEqual(info["status"], "online")
        self.assertIn("max_shots", info)

    def test_mock_client_queue_status(self):
        """get_queue_status 返回结构验证"""
        client = MockTianyanClient(mock_delay=0.0, mock_failure_rate=0.0)

        # 提交几个量子任务以填充队列
        for _ in range(3):
            client.submit_quantum_task(circuit_qasm=BELL_QASM, shots=1024)

        status = client.get_queue_status()
        self.assertIn("total_pending", status)
        self.assertIn("total_running", status)
        self.assertIn("queue_capacity", status)
        self.assertIn("estimated_wait_time", status)
        self.assertIn("by_backend", status)

        # 验证队列中有待执行任务
        self.assertGreaterEqual(status["total_pending"], 1)
        self.assertEqual(status["queue_capacity"], 1000)

        # 验证 by_backend 结构
        self.assertIn("tianyan-287", status["by_backend"])
        backend_287 = status["by_backend"]["tianyan-287"]
        self.assertIn("pending", backend_287)
        self.assertIn("running", backend_287)
        self.assertIn("capacity", backend_287)


# ============================================================
# 3. Web API 集成测试
# ============================================================


class TestWebAPIIntegration(unittest.TestCase):
    """Web API 端到端集成测试：FastAPI TestClient 完整流程"""

    @staticmethod
    async def _noop_simulate() -> None:
        """空操作后台任务，替代 simulate_scheduler 避免后台任务干扰测试。"""
        return None

    def setUp(self) -> None:
        """保存全局状态并创建 TestClient。"""
        # 快照全局状态，确保测试间隔离
        self._saved_status = copy.deepcopy(app_module.system_status)
        self._saved_queue = copy.deepcopy(app_module.task_queue)
        self._saved_strategy = app_module.system_status.get("current_strategy")

        # 补丁 simulate_scheduler 为空操作，避免后台任务修改全局状态
        self._patcher = patch.object(app_module, "simulate_scheduler", self._noop_simulate)
        self._patcher.start()

        # 使用 TestClient 的 context manager 触发 lifespan（已补丁为 noop）
        self._client_ctx = TestClient(app)
        self.client = self._client_ctx.__enter__()

    def tearDown(self) -> None:
        """恢复全局状态。"""
        try:
            self._client_ctx.__exit__(None, None, None)
        finally:
            self._patcher.stop()
            # 还原全局状态
            app_module.system_status.clear()
            app_module.system_status.update(copy.deepcopy(self._saved_status))
            app_module.task_queue.clear()
            app_module.task_queue.extend(copy.deepcopy(self._saved_queue))
            app_module.system_status["current_strategy"] = self._saved_strategy

    def test_full_web_workflow(self):
        """GET /api/status→POST /api/tasks→GET /api/tasks→POST /api/strategy→GET /metrics"""
        # 1. GET /api/status — 获取初始系统状态
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        status = resp.json()
        self.assertIn("qubit_utilization", status)
        self.assertIn("queue_length", status)
        self.assertIn("strategy_options", status)
        self.assertIn("current_strategy", status)

        # 2. POST /api/tasks — 提交新任务
        task_payload = {
            "user_id": "integration_user",
            "task_type": "quantum",
            "priority": 4,
            "qubit_count": 10,
            "circuit_depth": 100,
            "estimated_time": 30.0,
        }
        resp = self.client.post("/api/tasks", json=task_payload)
        self.assertEqual(resp.status_code, 200)
        task_data = resp.json()
        self.assertIn("task_id", task_data)
        self.assertTrue(task_data["task_id"].startswith("QTASK-"))

        # 3. GET /api/tasks — 查询任务列表
        resp = self.client.get("/api/tasks")
        self.assertEqual(resp.status_code, 200)
        tasks = resp.json()
        self.assertIsInstance(tasks, list)
        self.assertGreaterEqual(len(tasks), 1)
        # 验证新提交的任务在列表中
        submitted = any(t["task_id"] == task_data["task_id"] for t in tasks)
        self.assertTrue(submitted)

        # 4. POST /api/strategy — 切换调度策略
        resp = self.client.post("/api/strategy", params={"strategy": "FCFS"})
        self.assertEqual(resp.status_code, 200)
        strat_data = resp.json()
        self.assertTrue(strat_data["success"])
        self.assertIn("FCFS", strat_data["message"])

        # 5. GET /metrics — 获取 Prometheus 指标
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        content_type = resp.headers.get("content-type", "")
        self.assertIn("text/plain", content_type)

    def test_web_status_update_cycle(self):
        """POST /api/update→GET /api/status 验证更新"""
        update_payload = {
            "qubit_utilization": 0.85,
            "queue_length": 15,
            "completed_tasks": 200,
            "average_wait_time": 7.5,
        }

        # POST /api/update — 更新系统状态
        resp = self.client.post("/api/update", json=update_payload)
        self.assertEqual(resp.status_code, 200)
        update_data = resp.json()
        self.assertIn("status", update_data)

        # GET /api/status — 验证状态已更新
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        status = resp.json()
        self.assertAlmostEqual(status["qubit_utilization"], 0.85)
        self.assertEqual(status["queue_length"], 15)
        self.assertEqual(status["completed_tasks"], 200)
        self.assertAlmostEqual(status["average_wait_time"], 7.5)

    def test_web_task_lifecycle(self):
        """提交任务→查询任务列表→验证队列长度变化"""
        # 获取初始任务列表长度
        resp = self.client.get("/api/tasks")
        self.assertEqual(resp.status_code, 200)
        initial_task_count = len(resp.json())

        # 提交新任务
        task_payload = {
            "user_id": "lifecycle_user",
            "task_type": "hybrid",
            "priority": 3,
            "qubit_count": 5,
            "circuit_depth": 50,
            "estimated_time": 20.0,
        }
        resp = self.client.post("/api/tasks", json=task_payload)
        self.assertEqual(resp.status_code, 200)
        new_task_id = resp.json()["task_id"]

        # 验证任务列表长度增加
        resp = self.client.get("/api/tasks")
        self.assertEqual(resp.status_code, 200)
        updated_task_count = len(resp.json())
        self.assertGreater(updated_task_count, initial_task_count)

        # 验证新任务在 pending 列表中
        resp = self.client.get("/api/tasks", params={"status": "pending"})
        self.assertEqual(resp.status_code, 200)
        pending_tasks = resp.json()
        found = any(t["task_id"] == new_task_id for t in pending_tasks)
        self.assertTrue(found)


# ============================================================
# 4. 错误恢复集成测试
# ============================================================


class TestErrorRecoveryIntegration(unittest.TestCase):
    """错误恢复集成测试：异常输入与网络错误恢复"""

    def test_invalid_task_graceful_handling(self):
        """无效任务输入→解析失败→不影响其他任务"""
        parser = TaskParser()

        # 一批任务：中间夹杂无效任务
        task_dicts = [
            make_task_dict("ok_001", qubits_required=4),
            {"type": "quantum", "qubits_required": 8},  # 缺少 task_id
            make_task_dict("ok_002", qubits_required=6),
            {"task_id": "bad_type", "type": "invalid_type"},  # 非法任务类型
            make_task_dict("ok_003", qubits_required=8),
        ]

        # 逐个解析，捕获错误
        parsed_tasks: list[ParserTask] = []
        errors: list[Exception] = []
        for td in task_dicts:
            try:
                task = parser.parse(td)
                parsed_tasks.append(task)
            except (ValueError, TypeError) as e:
                errors.append(e)

        # 3 个有效任务解析成功
        self.assertEqual(len(parsed_tasks), 3)
        self.assertEqual(len(errors), 2)

        # 验证有效任务的 ID
        parsed_ids = {t.task_id for t in parsed_tasks}
        self.assertEqual(parsed_ids, {"ok_001", "ok_002", "ok_003"})

        # 验证无效任务的错误信息包含关键字段
        error_messages = " ".join(str(e) for e in errors)
        self.assertTrue("task_id" in error_messages or "task_type" in error_messages)

    def test_mock_client_network_error_recovery(self):
        """模拟网络错误→重试→恢复"""
        # 创建始终失败的客户端（模拟网络错误）
        client = MockTianyanClient(mock_delay=0.0, mock_failure_rate=1.0)

        # 验证操作抛出 TianyanAPIError（模拟网络故障）
        with self.assertRaises(TianyanAPIError):
            client.authenticate()

        with self.assertRaises(TianyanAPIError):
            client.submit_quantum_task(circuit_qasm=BELL_QASM)

        # 恢复：将失败率设为 0（网络恢复）
        client.mock_failure_rate = 0.0

        # 验证操作恢复正常
        self.assertTrue(client.authenticate())
        task_id = client.submit_quantum_task(circuit_qasm=BELL_QASM)
        self.assertTrue(task_id.startswith("mock-"))

        # 验证可以正常查询后端信息
        backends = client.list_backends()
        self.assertGreater(len(backends), 0)


if __name__ == "__main__":
    unittest.main()
