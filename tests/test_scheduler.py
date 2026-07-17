"""
量子RL调度系统 - 单元测试
Unit Tests for Quantum RL Scheduling System

测试覆盖：
- 调度环境（QuantumSchedulingEnv）
- RL智能体（SchedulerAgent）
- 任务解析器（TaskParser / LegacyTaskParser / TaskBuilder）
- 量子退火（QuantumAnnealingOptimizer）
- 仿真策略（GreedyStrategy / FCFSStrategy 等）
"""

import os
import sys
import unittest
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.evaluation.run_simulation import (
    ClassicalOnlyStrategy,
    FCFSStrategy,
    GreedyStrategy,
    QuantumOnlyStrategy,
    RandomStrategy,
    ShortestJobFirstStrategy,
)
from src.quantum.annealing import QuantumAnnealingOptimizer
from src.scheduler.agent import DuelingQNetwork, SchedulerAgent
from src.scheduler.env import (
    DEFAULT_MACHINE_CONFIGS,
    OBS_DIM,
    QuantumMachine,
    QuantumSchedulingEnv,
    Task,
)
from src.scheduler.parser import (
    LegacyTaskParser,
    TaskBuilder,
    TaskFeatures,
    TaskParser,
)
from src.scheduler.parser import Task as ParserTask


class TestQuantumSchedulingEnv(unittest.TestCase):
    """测试量子调度环境"""

    def setUp(self):
        """测试初始化"""
        self.env = QuantumSchedulingEnv(
            max_steps=100,
            max_qubits=287,
            seed=42,
        )

    def test_reset(self):
        """测试环境重置"""
        obs, info = self.env.reset(seed=42)

        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertTrue(np.all(obs >= 0.0))
        self.assertTrue(np.all(obs <= 1.0))
        self.assertIsInstance(info, dict)

    def test_step(self):
        """测试环境步进"""
        _obs, info = self.env.reset(seed=42)

        action = self.env.action_space.sample()
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertEqual(next_obs.shape, (OBS_DIM,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_observation_range(self):
        """测试状态向量值域始终在 [0, 1]"""
        obs, _ = self.env.reset(seed=42)

        for _ in range(50):
            action = self.env.action_space.sample()
            obs, _, terminated, truncated, _ = self.env.step(action)

            self.assertTrue(np.all(obs >= 0.0))
            self.assertTrue(np.all(obs <= 1.0))

            if terminated or truncated:
                break

    def test_action_space(self):
        """测试动作空间（3个动作）"""
        self.assertEqual(self.env.action_space.n, 3)

        valid_actions = [0, 1, 2]
        for a in valid_actions:
            self.assertTrue(0 <= a < self.env.action_space.n)

    def test_episode_terminates(self):
        """测试 episode 会在 max_steps 后终止"""
        self.env.reset(seed=42)
        steps = 0
        for _ in range(200):
            action = self.env.action_space.sample()
            _, _, terminated, truncated, _ = self.env.step(action)
            steps += 1
            if terminated or truncated:
                break

        self.assertLessEqual(steps, 100)

    def test_task_dataclass(self):
        """测试 Task 数据类"""
        task = Task(
            task_id="test_001",
            task_type="quantum",
            qubit_count=10,
            wait_steps=0,
            urgency=0.8,
            priority=4,
        )

        self.assertEqual(task.task_id, "test_001")
        self.assertEqual(task.task_type, "quantum")
        self.assertEqual(task.qubit_count, 10)
        self.assertEqual(task.urgency, 0.8)

    def test_render_ansi(self):
        """测试 ANSI 渲染模式"""
        env = QuantumSchedulingEnv(max_steps=20, render_mode="ansi")
        env.reset(seed=42)
        output = env.render()
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)
        env.close()

    def test_info_keys(self):
        """测试 info 字典包含预期的键"""
        _obs, info = self.env.reset(seed=42)
        _, _, _, _, info = self.env.step(0)

        expected_keys = [
            "total_scheduled",
            "quantum_success",
            "classical_success",
            "hybrid_success",
            "mismatch_count",
        ]
        for key in expected_keys:
            self.assertIn(key, info)


class TestMultiMachineScheduling(unittest.TestCase):
    """多机器调度扩展测试"""

    def test_quantum_machine_dataclass(self):
        """测试 QuantumMachine 数据类"""
        m = QuantumMachine(
            name="tianyan_test",
            total_qubits=100,
            available_ratio=0.8,
            fidelity=0.95,
            supported_gates=("H", "CZ", "M"),
        )
        self.assertEqual(m.name, "tianyan_test")
        self.assertEqual(m.total_qubits, 100)
        self.assertTrue(m.available)
        self.assertFalse(m.is_real)

    def test_default_machine_configs_valid(self):
        """默认多机器配置应包含 3 台机器且字段完整"""
        self.assertEqual(len(DEFAULT_MACHINE_CONFIGS), 3)
        for cfg in DEFAULT_MACHINE_CONFIGS:
            self.assertIn("name", cfg)
            self.assertIn("total_qubits", cfg)
            self.assertIn("supported_gates", cfg)
            self.assertGreater(cfg["total_qubits"], 0)

    def test_multi_machine_init(self):
        """多机器环境初始化应创建指定数量的机器"""
        env = QuantumSchedulingEnv(
            max_steps=50,
            machine_configs=DEFAULT_MACHINE_CONFIGS,
        )
        self.assertEqual(env.num_machines, 3)
        self.assertEqual(len(env.machine_names), 3)
        self.assertIn("tianyan_s", env.machine_names)

    def test_obs_action_space_unchanged(self):
        """多机器模式下 obs/action 空间应保持不变（PPO 兼容）"""
        env = QuantumSchedulingEnv(
            max_steps=50,
            machine_configs=DEFAULT_MACHINE_CONFIGS,
        )
        self.assertEqual(env.observation_space.shape, (OBS_DIM,))
        self.assertEqual(env.action_space.n, 3)

    def test_single_machine_backward_compat(self):
        """machine_configs=None 应退化为单机模式（向后兼容）"""
        env = QuantumSchedulingEnv(max_steps=50, machine_configs=None)
        obs, info = env.reset(seed=42)
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertEqual(info["num_machines"], 1)
        # 单机模式不应显示多机器明细（render 不报错即可）
        env.close()

    def test_multi_machine_episode_runs(self):
        """多机器模式下完整 episode 应正常跑完"""
        env = QuantumSchedulingEnv(
            max_steps=80,
            machine_configs=DEFAULT_MACHINE_CONFIGS,
        )
        env.reset(seed=7)
        for _ in range(80):
            _obs, _, term, trunc, _ = env.step(1)  # 全部走量子资源
            if term or trunc:
                break
        # 至少有任务被路由到某台机器
        self.assertGreater(sum(env._machine_schedule_count.values()), 0)

    def test_machine_selection_distribution(self):
        """多机器调度应在多台机器间分布（负载均衡）"""
        env = QuantumSchedulingEnv(
            max_steps=100,
            machine_configs=DEFAULT_MACHINE_CONFIGS,
        )
        env.reset(seed=42)

        # 手动注入一个量子任务确保能被调度
        from src.scheduler.env import Task

        quantum_task = Task(task_id="test_quantum", task_type="quantum", qubit_count=5)
        env._task_queue.insert(0, quantum_task)
        env._current_task = quantum_task

        # 执行一步量子调度
        env.step(1)  # ACTION_QUANTUM

        # 至少有一台机器被调度
        used_machines = sum(1 for c in env._machine_schedule_count.values() if c > 0)
        self.assertGreaterEqual(used_machines, 1)

    def test_gate_set_filtering(self):
        """需要 RX 门的任务不应路由到仅支持 H/CZ/M 的 tianyan_s"""
        env = QuantumSchedulingEnv(
            max_steps=20,
            machine_configs=DEFAULT_MACHINE_CONFIGS,
        )
        env.reset(seed=42)
        # 构造一个需要 RX 门的任务
        task = Task(task_id="gate_test", task_type="quantum", qubit_count=2)
        task.required_gates = ("RX", "H", "M")
        machine = env._select_best_machine(task)
        # tianyan_s 仅支持 H/CZ/M，应被过滤；tianyan_tn 支持 RX
        if machine is not None:
            self.assertNotEqual(machine.name, "tianyan_s")
            self.assertIn("RX", machine.supported_gates)

    def test_select_best_machine_returns_none_when_no_fit(self):
        """无机器能承接超大任务时应返回 None"""
        env = QuantumSchedulingEnv(
            max_steps=20,
            machine_configs=[
                {
                    "name": "small",
                    "total_qubits": 10,
                    "supported_gates": ("H", "CZ", "M"),
                    "is_real": False,
                },
            ],
        )
        env.reset(seed=42)
        # 需要 100 比特，但机器只有 10 比特
        big_task = Task(task_id="big", task_type="quantum", qubit_count=100)
        machine = env._select_best_machine(big_task)
        self.assertIsNone(machine)

    def test_attach_real_clients_sets_is_real(self):
        """attach_real_clients 应将对应机器标记为真机"""
        env = QuantumSchedulingEnv(
            max_steps=20,
            machine_configs=DEFAULT_MACHINE_CONFIGS,
        )
        fake_clients = {
            "tianyan_s": object(),  # 占位客户端
            "tianyan_tn": object(),
        }
        env.attach_real_clients(fake_clients)
        for m in env._machines:
            if m.name in fake_clients:
                self.assertTrue(m.is_real)
            else:
                self.assertFalse(m.is_real)

    def test_info_contains_machine_details(self):
        """info 字典应包含多机器调度详情"""
        env = QuantumSchedulingEnv(
            max_steps=20,
            machine_configs=DEFAULT_MACHINE_CONFIGS,
        )
        _, info = env.reset(seed=42)
        _, _, _, _, info = env.step(1)
        self.assertIn("machines", info)
        self.assertIn("last_selected_machine", info)
        self.assertIn("machine_schedule_count", info)
        self.assertEqual(len(info["machines"]), 3)

    def test_use_real_machine_param_default_false(self):
        """use_real_machine 默认应为 False（向后兼容）"""
        env = QuantumSchedulingEnv(max_steps=20, seed=42)
        self.assertFalse(env.use_real_machine)
        self.assertFalse(env.is_real_machine_degraded())

    def test_use_real_machine_param_enabled(self):
        """启用 use_real_machine 后环境应记录该状态"""
        env = QuantumSchedulingEnv(
            max_steps=20,
            machine_configs=DEFAULT_MACHINE_CONFIGS,
            use_real_machine=True,
            real_submit_probability=0.0,  # 不实际提交
            seed=42,
        )
        self.assertTrue(env.use_real_machine)
        # 未绑定客户端时不应报错
        env.reset(seed=42)
        _obs, _reward, _term, _trunc, info = env.step(1)
        self.assertIn("real_machine_stats", info)
        self.assertIn("real_machine_degraded", info)
        self.assertEqual(info["real_machine_stats"]["pending_count"], 0)

    def test_real_machine_stats_initial_state(self):
        """真机闭环统计初始状态应为全零"""
        env = QuantumSchedulingEnv(max_steps=20, seed=42)
        stats = env.get_real_machine_stats()
        self.assertEqual(stats["pending_count"], 0)
        self.assertEqual(stats["success_count"], 0)
        self.assertEqual(stats["fail_count"], 0)
        self.assertFalse(stats["degraded"])
        self.assertEqual(stats["consecutive_failures"], 0)

    def test_real_machine_degrade_after_consecutive_failures(self):
        """连续失败超过阈值应触发降级"""
        env = QuantumSchedulingEnv(
            max_steps=20,
            machine_configs=DEFAULT_MACHINE_CONFIGS,
            use_real_machine=True,
            real_submit_probability=1.0,
            seed=42,
        )

        # 模拟真机客户端：提交总是抛异常（触发 _record_real_failure）
        class FailingClient:
            machine_name = "tianyan_s"

            def submit_quantum_task(self, **kwargs):
                raise RuntimeError("simulated network failure")

            def get_task_status(self, task_id):
                return {"status": "error"}

        env.attach_real_clients({"tianyan_s": FailingClient()})
        env.reset(seed=42)
        # 直接调用 _submit_to_real_machine 触发失败（绕过任务兼容性检查）
        machine = env._machines[0]
        quantum_task = Task(task_id="T_test", task_type="quantum", qubit_count=2)
        for _ in range(3):
            env._submit_to_real_machine(machine, quantum_task)
            if env.is_real_machine_degraded():
                break
        self.assertTrue(env.is_real_machine_degraded())
        stats = env.get_real_machine_stats()
        self.assertGreaterEqual(stats["fail_count"], 3)
        self.assertTrue(stats["degraded"])

    def test_real_machine_success_feedback_resets_failure_count(self):
        """真机成功反馈应重置连续失败计数"""
        env = QuantumSchedulingEnv(
            max_steps=30,
            machine_configs=DEFAULT_MACHINE_CONFIGS,
            use_real_machine=True,
            real_submit_probability=1.0,
            seed=42,
        )

        # 模拟客户端：提交成功返回 task_id，状态查询返回 completed
        class SuccessClient:
            machine_name = "tianyan_s"
            _counter = 0

            def submit_quantum_task(self, **kwargs):
                SuccessClient._counter += 1
                return f"real_task_{SuccessClient._counter}"

            def get_task_status(self, task_id):
                return {"status": "completed", "result": {"0": 0.5}}

        env.attach_real_clients({"tianyan_s": SuccessClient()})
        env.reset(seed=42)
        # 直接提交一个真机任务（绕过任务兼容性检查）
        machine = env._machines[0]
        quantum_task = Task(task_id="T_test", task_type="quantum", qubit_count=2)
        env._submit_to_real_machine(machine, quantum_task)
        self.assertEqual(len(env._pending_real_tasks), 1)
        # 触发轮询（_poll_pending_real_tasks 不依赖 step 的 action）
        feedback = env._poll_pending_real_tasks()
        self.assertGreater(feedback, 0.0)  # 成功反馈应为正
        stats = env.get_real_machine_stats()
        self.assertGreaterEqual(stats["success_count"], 1)
        self.assertEqual(stats["consecutive_failures"], 0)
        self.assertFalse(stats["degraded"])

    def test_real_machine_degraded_skips_submission(self):
        """降级后应跳过真机提交（fallback 到 Mock）"""
        env = QuantumSchedulingEnv(
            max_steps=20,
            machine_configs=DEFAULT_MACHINE_CONFIGS,
            use_real_machine=True,
            real_submit_probability=1.0,
            seed=42,
        )

        submit_count = {"n": 0}

        class CountingFailingClient:
            machine_name = "tianyan_s"

            def submit_quantum_task(self, **kwargs):
                submit_count["n"] += 1
                raise RuntimeError("fail")

            def get_task_status(self, task_id):
                return {"status": "error"}

        env.attach_real_clients({"tianyan_s": CountingFailingClient()})
        env.reset(seed=42)
        # 触发降级（直接调用 _submit_to_real_machine）
        machine = env._machines[0]
        quantum_task = Task(task_id="T_test", task_type="quantum", qubit_count=2)
        for _ in range(3):
            env._submit_to_real_machine(machine, quantum_task)
            if env.is_real_machine_degraded():
                break
        self.assertTrue(env.is_real_machine_degraded())

        # 降级后继续提交，提交次数不应增长
        submits_before = submit_count["n"]
        for _ in range(5):
            env._submit_to_real_machine(machine, quantum_task)
        self.assertEqual(submit_count["n"], submits_before)

    def test_real_machine_pending_task_polling(self):
        """提交后任务应进入 pending 列表，轮询后移出"""
        env = QuantumSchedulingEnv(
            max_steps=30,
            machine_configs=DEFAULT_MACHINE_CONFIGS,
            use_real_machine=True,
            real_submit_probability=1.0,
            seed=42,
        )

        class PollClient:
            machine_name = "tianyan_s"

            def submit_quantum_task(self, **kwargs):
                return "poll_task_001"

            def get_task_status(self, task_id):
                # 第一次查询返回 running，第二次返回 completed
                PollClient._calls = getattr(PollClient, "_calls", 0) + 1
                if PollClient._calls == 1:
                    return {"status": "running"}
                return {"status": "completed"}

        env.attach_real_clients({"tianyan_s": PollClient()})
        env.reset(seed=42)
        # 直接提交（绕过任务兼容性检查）
        machine = env._machines[0]
        quantum_task = Task(task_id="T_test", task_type="quantum", qubit_count=2)
        env._submit_to_real_machine(machine, quantum_task)
        self.assertEqual(len(env._pending_real_tasks), 1)
        # 第一次轮询：running，任务仍在 pending
        env._poll_pending_real_tasks()
        self.assertEqual(len(env._pending_real_tasks), 1)
        # 第二次轮询：completed，任务移出 pending
        env._poll_pending_real_tasks()
        self.assertEqual(len(env._pending_real_tasks), 0)
        stats = env.get_real_machine_stats()
        self.assertEqual(stats["success_count"], 1)


class TestSchedulerAgent(unittest.TestCase):
    """测试RL智能体"""

    def setUp(self):
        """测试初始化"""
        self.env = QuantumSchedulingEnv(max_steps=50, seed=42)
        self.agent = SchedulerAgent(
            env=self.env,
            learning_rate=1e-3,
            buffer_size=1000,
            batch_size=32,
            verbose=0,
            seed=42,
        )

    def test_initialization(self):
        """测试智能体初始化"""
        self.assertEqual(self.agent.observation_space.shape[0], OBS_DIM)
        self.assertEqual(self.agent.action_space.n, 3)
        self.assertIsNone(self.agent.model)

    def test_get_config(self):
        """测试获取配置信息"""
        config = self.agent.get_config()

        self.assertIn("observation_dim", config)
        self.assertIn("action_dim", config)
        self.assertIn("learning_rate", config)
        self.assertIn("architecture", config)
        self.assertEqual(config["architecture"], "Dueling DQN")

    def test_predict_before_train_raises(self):
        """测试未训练时调用 predict 抛异常"""
        state = np.zeros(OBS_DIM, dtype=np.float32)
        with self.assertRaises(RuntimeError):
            self.agent.predict(state)

    def test_save_model(self):
        """测试模型保存"""
        import os
        import tempfile

        self.agent.model = self.agent._build_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model")
            self.agent.save(save_path)

            zip_path = save_path + ".zip"
            self.assertTrue(os.path.exists(zip_path))

    def test_repr(self):
        """测试智能体的字符串表示"""
        rep = repr(self.agent)
        self.assertIn("SchedulerAgent", rep)
        self.assertIn("Dueling DQN", rep)

    def test_build_model(self):
        """测试模型构建"""
        model = self.agent._build_model()
        self.assertIsNotNone(model)
        self.assertEqual(model.observation_space.shape[0], OBS_DIM)
        self.assertEqual(model.action_space.n, 3)


class TestLegacyTaskParser(unittest.TestCase):
    """测试旧版任务解析器（向后兼容）"""

    def setUp(self):
        self.parser = LegacyTaskParser()

    def test_parse_json(self):
        """测试JSON格式解析"""
        json_str = """{
            "task_id": "task_001",
            "user_id": "user_123",
            "task_type": "quantum",
            "qubit_count": 10,
            "circuit_depth": 50,
            "algorithm": "VQE",
            "estimated_time": 120.0,
            "priority": 4
        }"""

        features = self.parser.parse(json_str, format="json")

        self.assertIsNotNone(features)
        self.assertEqual(features.task_id, "task_001")
        self.assertEqual(features.task_type, "quantum")
        self.assertEqual(features.qubit_count, 10)
        self.assertEqual(features.algorithm, "VQE")

    def test_parse_qasm(self):
        """测试QASM格式解析"""
        qasm_str = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[5];
        creg c[5];
        h q[0];
        cx q[0],q[1];
        measure q[0] -> c[0];
        """

        features = self.parser.parse(qasm_str, format="qasm")

        self.assertIsNotNone(features)
        self.assertEqual(features.qubit_count, 5)
        self.assertGreater(features.gate_count, 0)
        self.assertGreater(features.measurement_count, 0)

    def test_parse_text(self):
        """测试文本格式解析"""
        text = "这是一个量子任务，需要10比特，使用VQE算法"
        features = self.parser.parse(text, format="text")

        self.assertIsNotNone(features)
        self.assertEqual(features.task_type, "quantum")
        self.assertEqual(features.qubit_count, 10)

    def test_to_vector(self):
        """测试特征向量转换"""
        features = TaskFeatures(
            task_id="test",
            user_id="user",
            task_type="quantum",
            qubit_count=10,
            circuit_depth=50,
            algorithm="VQE",
        )

        vector = features.to_vector(feature_dim=20)

        self.assertEqual(len(vector), 20)
        self.assertTrue(all(0.0 <= x <= 1.0 for x in vector))

    def test_batch_parse(self):
        """测试批量解析"""
        descriptions = [
            '{"task_id": "t1", "task_type": "quantum", "qubit_count": 5}',
            '{"task_id": "t2", "task_type": "classical", "qubit_count": 0}',
        ]
        results = self.parser.batch_parse(descriptions, format="json")
        self.assertEqual(len(results), 2)


class TestTaskParser(unittest.TestCase):
    """测试新版任务解析器（Task + TaskParser + TaskBuilder）"""

    def setUp(self):
        self.parser = TaskParser()
        self.sample_dict = {
            "task_id": "task_001",
            "type": "quantum",
            "algorithm": "VQE",
            "qubits_required": 8,
            "circuit_depth": 50,
            "shots": 1024,
            "estimated_time": 120,
            "priority": "high",
            "deadline": "2026-07-01T12:00:00",
        }

    def test_parse_basic(self):
        """测试基本字典解析"""
        task = self.parser.parse(self.sample_dict)
        self.assertIsInstance(task, ParserTask)
        self.assertEqual(task.task_id, "task_001")
        self.assertEqual(task.task_type, "quantum")
        self.assertEqual(task.algorithm, "VQE")
        self.assertEqual(task.qubits_required, 8)
        self.assertEqual(task.priority, 3)
        self.assertIsInstance(task.deadline, datetime)
        self.assertEqual(task.status, "pending")

    def test_parse_with_int_priority(self):
        """测试整型 priority"""
        d = dict(self.sample_dict, priority=4)
        task = self.parser.parse(d)
        self.assertEqual(task.priority, 4)

    def test_parse_classical(self):
        """测试经典任务类型"""
        d = dict(self.sample_dict, type="classical", qubits_required=0, algorithm=None)
        task = self.parser.parse(d)
        self.assertEqual(task.task_type, "classical")
        self.assertIsNone(task.algorithm)

    def test_parse_hybrid(self):
        """测试混合任务类型"""
        d = dict(self.sample_dict, type="hybrid")
        task = self.parser.parse(d)
        self.assertEqual(task.task_type, "hybrid")

    def test_parse_missing_task_id_raises(self):
        """测试缺少 task_id 抛异常"""
        with self.assertRaises(ValueError):
            self.parser.parse({"type": "quantum", "qubits_required": 8})

    def test_parse_invalid_type_raises(self):
        """测试无效 task_type 抛异常"""
        with self.assertRaises(ValueError):
            self.parser.parse({"task_id": "x", "type": "invalid"})

    def test_parse_invalid_priority_raises(self):
        """测试无效 priority 抛异常"""
        with self.assertRaises(ValueError):
            self.parser.parse(dict(self.sample_dict, priority="super_urgent"))

    def test_parse_qubits_exceed_limit_raises(self):
        """测试量子比特超限抛异常"""
        with self.assertRaises(ValueError):
            self.parser.parse(dict(self.sample_dict, qubits_required=999))

    def test_parse_not_dict_raises(self):
        """测试非字典输入抛异常"""
        with self.assertRaises(TypeError):
            self.parser.parse("not a dict")

    def test_validate_valid(self):
        """测试合法任务校验通过"""
        task = self.parser.parse(self.sample_dict)
        self.assertTrue(self.parser.validate(task))

    def test_validate_invalid(self):
        """测试非法任务校验失败"""
        invalid_task = ParserTask(
            task_id="bad",
            task_type="quantum",
            qubits_required=0,
            estimated_time=10,
            priority=1,
        )
        self.assertFalse(self.parser.validate(invalid_task))

    def test_estimate_resources(self):
        """测试资源预估"""
        task = self.parser.parse(self.sample_dict)
        res = self.parser.estimate_resources(task)

        self.assertIn("qubit_hours", res)
        self.assertIn("total_gate_operations", res)
        self.assertIn("memory_mb", res)
        self.assertIn("classical_compute_ratio", res)
        self.assertIn("estimated_queue_time", res)

        self.assertGreater(res["qubit_hours"], 0)
        self.assertEqual(res["total_gate_operations"], 50 * 1024)
        self.assertEqual(res["classical_compute_ratio"], 0.1)

    def test_to_internal_format(self):
        """测试内部格式转换"""
        task = self.parser.parse(self.sample_dict)
        internal = self.parser.to_internal_format(task)

        self.assertEqual(internal["task_id"], "task_001")
        self.assertEqual(internal["priority_label"], "high")
        self.assertIn("resource_estimate", internal)
        self.assertIn("scheduling_weight", internal)
        self.assertIsInstance(internal["scheduling_weight"], float)
        self.assertGreater(internal["scheduling_weight"], 0)
        self.assertIsInstance(internal["deadline"], str)

    def test_builder_basic(self):
        """测试 Builder 基本链式调用"""
        task = (
            TaskBuilder()
            .set_id("b_001")
            .set_type("quantum")
            .set_algorithm("Grover")
            .set_qubits(16)
            .set_circuit_depth(100)
            .set_shots(2048)
            .set_estimated_time(200)
            .set_priority("urgent")
            .set_deadline("2026-08-01T00:00:00")
            .build()
        )
        self.assertEqual(task.task_id, "b_001")
        self.assertEqual(task.priority, 4)

    def test_builder_from_dict(self):
        """测试 Builder.from_dict"""
        task = TaskBuilder.from_dict(self.sample_dict).build()
        self.assertEqual(task.task_id, "task_001")
        self.assertEqual(task.task_type, "quantum")
        self.assertEqual(task.priority, 3)

    def test_builder_empty_id_raises(self):
        """测试 Builder 空 task_id 抛异常"""
        with self.assertRaises(ValueError):
            TaskBuilder().build()

    def test_builder_status(self):
        """测试 Builder 设置状态"""
        task = TaskBuilder().set_id("s_001").set_type("classical").set_status("running").build()
        self.assertEqual(task.status, "running")


class TestQuantumAnnealing(unittest.TestCase):
    """测试量子退火优化器"""

    def setUp(self):
        self.optimizer = QuantumAnnealingOptimizer(
            num_qubits=8,
            annealing_time=20,
            shots=100,
        )

    def test_initialization(self):
        """测试优化器初始化"""
        self.assertEqual(self.optimizer.num_qubits, 8)
        self.assertEqual(self.optimizer.shots, 100)
        self.assertFalse(self.optimizer.use_dw)

    def test_network_to_qubo(self):
        """测试权重到 QUBO 的映射"""
        W1 = np.random.randn(4, 2).astype(np.float32)
        b1 = np.random.randn(2).astype(np.float32)
        weights = [W1, b1]

        qubo = self.optimizer.network_to_qubo(weights)

        self.assertIsInstance(qubo, np.ndarray)
        self.assertEqual(qubo.shape[0], qubo.shape[1])
        self.assertGreater(qubo.shape[0], 0)

    def test_network_to_qubo_with_gradients(self):
        """测试带梯度信息的 QUBO 映射"""
        np.random.seed(42)
        W1 = np.random.randn(4, 2).astype(np.float32)
        b1 = np.random.randn(2).astype(np.float32)
        weights = [W1, b1]

        dW1 = np.random.randn(4, 2).astype(np.float32)
        db1 = np.random.randn(2).astype(np.float32)
        gradients = [dW1, db1]

        qubo = self.optimizer.network_to_qubo(weights, gradients=gradients)

        self.assertIsInstance(qubo, np.ndarray)
        self.assertEqual(qubo.shape[0], qubo.shape[1])

    def test_network_to_qubo_with_td_errors(self):
        """测试带 TD 误差的 QUBO 映射"""
        np.random.seed(42)
        W1 = np.random.randn(4, 2).astype(np.float32)
        b1 = np.random.randn(2).astype(np.float32)
        weights = [W1, b1]

        dW1 = np.random.randn(4, 2).astype(np.float32)
        db1 = np.random.randn(2).astype(np.float32)
        gradients = [dW1, db1]
        td_errors = np.random.randn(32).astype(np.float32)

        qubo = self.optimizer.network_to_qubo(weights, gradients=gradients, td_errors=td_errors)

        self.assertIsInstance(qubo, np.ndarray)
        self.assertEqual(qubo.shape[0], qubo.shape[1])

    def test_anneal(self):
        """测试退火求解"""
        n = 10
        Q = np.random.randn(n, n).astype(np.float64)
        Q = (Q + Q.T) / 2

        bitstring = self.optimizer.anneal(Q)

        self.assertEqual(len(bitstring), n)
        self.assertTrue(all(b in "01" for b in bitstring))

    def test_bitstring_to_weights(self):
        """测试比特串到权重的解码"""
        original_shapes = [(4, 2), (2,)]
        total_params = 4 * 2 + 2
        n_bits = total_params * max(1, self.optimizer.num_qubits // 4)
        bitstring = "1" * n_bits

        weights = self.optimizer.bitstring_to_weights(bitstring, original_shapes)

        self.assertEqual(len(weights), len(original_shapes))
        for w, shape in zip(weights, original_shapes, strict=False):
            self.assertEqual(w.shape, shape)

    def test_bitstring_to_weights_with_current(self):
        """测试带当前权重的解码（权重差模式）"""
        np.random.seed(42)
        W1 = np.random.randn(4, 2).astype(np.float32)
        b1 = np.random.randn(2).astype(np.float32)
        current_weights = [W1, b1]
        original_shapes = [w.shape for w in current_weights]

        total_params = sum(np.prod(s) for s in original_shapes)
        n_bits = int(total_params) * max(1, self.optimizer.num_qubits // 4)
        bitstring = "0" * n_bits

        new_weights = self.optimizer.bitstring_to_weights(
            bitstring, original_shapes, current_weights=current_weights
        )

        self.assertEqual(len(new_weights), len(original_shapes))
        for w, shape in zip(new_weights, original_shapes, strict=False):
            self.assertEqual(w.shape, shape)

        # 全 0 比特串对应 0 更新，所以新权重应该和旧权重相同
        # （符号位 0 = 正，但数值位全 0 → magnitude = 0 → delta = 0）
        for w_old, w_new in zip(current_weights, new_weights, strict=False):
            np.testing.assert_array_almost_equal(w_old, w_new, decimal=5)

    def test_compute_qubo_energy(self):
        """测试 QUBO 能量计算"""
        n = 5
        Q = np.random.randn(n, n)
        solution = np.random.randint(0, 2, n).astype(np.float64)

        energy = QuantumAnnealingOptimizer._compute_qubo_energy(solution, Q)

        expected = float(solution @ Q @ solution)
        self.assertAlmostEqual(energy, expected, places=6)

    def test_anneal_finds_better_than_random(self):
        """测试退火求解结果优于随机解"""
        np.random.seed(42)
        n = 20
        Q = np.random.randn(n, n)
        Q = (Q + Q.T) / 2

        best_bitstring = self.optimizer.anneal(Q)
        best_bits = np.array([int(b) for b in best_bitstring], dtype=np.float64)
        best_energy = QuantumAnnealingOptimizer._compute_qubo_energy(best_bits, Q)

        random_energies = []
        for _ in range(100):
            rand_bits = np.random.randint(0, 2, n).astype(np.float64)
            random_energies.append(QuantumAnnealingOptimizer._compute_qubo_energy(rand_bits, Q))

        avg_random = np.mean(random_energies)
        self.assertLess(best_energy, avg_random)


class TestSchedulingStrategies(unittest.TestCase):
    """测试调度策略"""

    def setUp(self):
        self.obs = np.array(
            [
                0.5,  # qubit_availability
                0.3,  # queue_length
                0.2,  # avg_wait_time
                0.95,  # fidelity
                0.4,  # classical_load
                0.5,  # quantum_queue_ratio
                0.5,  # time_of_day
                0.6,  # urgency_level
            ],
            dtype=np.float32,
        )

    def test_greedy_strategy(self):
        """测试贪心策略"""
        strategy = GreedyStrategy()
        action = strategy.select_action(self.obs)
        self.assertIn(action, [0, 1, 2])

    def test_greedy_high_urgency(self):
        """测试高紧急度时贪心策略优先量子"""
        strategy = GreedyStrategy()
        obs_high_urgency = self.obs.copy()
        obs_high_urgency[7] = 0.9  # high urgency
        obs_high_urgency[0] = 0.8  # high qubit availability
        action = strategy.select_action(obs_high_urgency)
        self.assertEqual(action, 1)  # should choose quantum

    def test_fcfs_strategy(self):
        """测试 FCFS 策略"""
        strategy = FCFSStrategy()
        action = strategy.select_action(self.obs)
        self.assertEqual(action, 2)  # always hybrid

    def test_random_strategy(self):
        """测试随机策略"""
        strategy = RandomStrategy(action_dim=3, seed=42)
        actions = [strategy.select_action(self.obs) for _ in range(100)]
        self.assertTrue(all(0 <= a < 3 for a in actions))

    def test_quantum_only_strategy(self):
        """测试仅量子策略"""
        strategy = QuantumOnlyStrategy()
        action = strategy.select_action(self.obs)
        self.assertEqual(action, 1)

    def test_classical_only_strategy(self):
        """测试仅经典策略"""
        strategy = ClassicalOnlyStrategy()
        action = strategy.select_action(self.obs)
        self.assertEqual(action, 0)

    def test_sjf_strategy(self):
        """测试 SJF 策略"""
        strategy = ShortestJobFirstStrategy()
        action = strategy.select_action(self.obs)
        self.assertIn(action, [0, 1, 2])

    def test_sjf_long_queue(self):
        """测试长队列时 SJF 使用混合执行"""
        strategy = ShortestJobFirstStrategy()
        obs_long_queue = self.obs.copy()
        obs_long_queue[1] = 0.8  # long queue
        action = strategy.select_action(obs_long_queue)
        self.assertEqual(action, 2)  # should choose hybrid


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_env_agent_interaction(self):
        """测试环境和智能体交互"""
        env = QuantumSchedulingEnv(max_steps=50, seed=42)
        agent = SchedulerAgent(env=env, learning_rate=1e-3, verbose=0)

        agent.model = agent._build_model()

        state, _ = env.reset(seed=42)

        for _i in range(20):
            action = agent.predict(state, deterministic=False)
            next_state, reward, terminated, truncated, _info = env.step(action)

            self.assertEqual(next_state.shape, (OBS_DIM,))
            self.assertIsInstance(reward, float)

            state = next_state

            if terminated or truncated:
                break

    def test_agent_short_training(self):
        """测试智能体短时间训练"""
        env = QuantumSchedulingEnv(max_steps=50, seed=42)
        agent = SchedulerAgent(
            env=env,
            learning_rate=1e-3,
            buffer_size=500,
            batch_size=32,
            learning_starts=50,
            verbose=0,
            seed=42,
        )

        model = agent.train(total_timesteps=200, eval_freq=100, n_eval_episodes=2)
        self.assertIsNotNone(model)

    def test_evaluate(self):
        """测试评估方法"""
        env = QuantumSchedulingEnv(max_steps=50, seed=42)
        agent = SchedulerAgent(env=env, verbose=0, seed=42)
        agent.model = agent._build_model()

        result = agent.evaluate(num_episodes=3, deterministic=True)

        self.assertIn("mean_reward", result)
        self.assertIn("std_reward", result)
        self.assertIn("success_rate", result)
        self.assertIn("num_episodes", result)
        self.assertEqual(result["num_episodes"], 3)


def run_tests():
    """运行所有测试"""
    print("=" * 64)
    print("  Quantum RL Scheduling System - Unit Tests")
    print("=" * 64)
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestQuantumSchedulingEnv))
    suite.addTests(loader.loadTestsFromTestCase(TestSchedulerAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestLegacyTaskParser))
    suite.addTests(loader.loadTestsFromTestCase(TestTaskParser))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumAnnealing))
    suite.addTests(loader.loadTestsFromTestCase(TestSchedulingStrategies))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 64)
    print("  Summary")
    print("=" * 64)
    print(f"  Tests run:    {result.testsRun}")
    print(f"  Failures:     {len(result.failures)}")
    print(f"  Errors:       {len(result.errors)}")
    print(f"  Skipped:      {len(result.skipped)}")

    if result.wasSuccessful():
        print("  All tests passed!")
    else:
        print("  Some tests failed!")

    print("=" * 64)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
