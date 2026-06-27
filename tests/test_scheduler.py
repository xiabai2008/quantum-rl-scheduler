"""
量子RL调度系统 - 单元测试
Unit Tests for Quantum RL Scheduling System

测试覆盖：
- 调度环境（SchedulingEnv）
- RL智能体（SchedulerAgent）
- 任务解析器（TaskParser / LegacyTaskParser / TaskBuilder）
- 天衍云客户端（TianyanClient）
"""

import unittest
import numpy as np
from datetime import datetime
import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.scheduler.env import SchedulingEnv, QuantumTask
from src.scheduler.agent import SchedulingAgent
from src.scheduler.parser import (
    TaskParser, LegacyTaskParser, TaskBuilder,
    Task, TaskFeatures,
)


class TestSchedulingEnv(unittest.TestCase):
    """测试调度环境"""

    def setUp(self):
        """测试初始化"""
        self.env = SchedulingEnv(
            max_qubits=287,
            max_queue_size=50,
            simulation_mode=True,
        )

    def test_reset(self):
        """测试环境重置"""
        obs, info = self.env.reset()

        # 检查状态向量维度
        self.assertEqual(obs.shape, (20,))

        # 检查状态值在[0, 1]范围内
        self.assertTrue(np.all(obs >= 0.0))
        self.assertTrue(np.all(obs <= 1.0))

        print("  test_reset passed")

    def test_step(self):
        """测试环境步进"""
        obs, info = self.env.reset()

        # 执行一个随机动作
        action = self.env.action_space.sample()
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # 检查返回值
        self.assertEqual(next_obs.shape, (20,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

        print("  test_step passed")

    def test_observation_range(self):
        """测试状态向量值域"""
        obs, _ = self.env.reset()

        # 多次步进，检查状态值范围
        for _ in range(10):
            action = self.env.action_space.sample()
            obs, _, _, _, _ = self.env.step(action)

            # 检查值域
            self.assertTrue(np.all(obs >= 0.0))
            self.assertTrue(np.all(obs <= 1.0))

        print("  test_observation_range passed")

    def test_task_generation(self):
        """测试任务生成"""
        # 手动生成任务
        task = QuantumTask(
            task_id="test_001",
            user_id="user_001",
            task_type="quantum",
            qubit_count=10,
            circuit_depth=50,
            estimated_time=120.0,
            priority=3,
            arrival_time=datetime.now(),
        )

        # 检查任务属性
        self.assertEqual(task.task_id, "test_001")
        self.assertEqual(task.task_type, "quantum")
        self.assertEqual(task.qubit_count, 10)
        self.assertEqual(task.status, "pending")

        print("  test_task_generation passed")


class TestSchedulerAgent(unittest.TestCase):
    """测试RL智能体"""

    def setUp(self):
        """测试初始化"""
        self.agent = SchedulingAgent(
            state_dim=20,
            action_dim=5,
            algorithm="DQN",
            learning_rate=3e-4,
        )

    def test_initialization(self):
        """测试智能体初始化"""
        self.assertEqual(self.agent.state_dim, 20)
        self.assertEqual(self.agent.action_dim, 5)
        self.assertEqual(self.agent.algorithm, "DQN")
        self.assertEqual(self.agent.epsilon, 1.0)

        print("  test_initialization passed")

    def test_select_action(self):
        """测试动作选择"""
        state = np.random.randn(20)

        # 训练模式
        action = self.agent.select_action(state, training=True)
        self.assertIn(action, range(5))

        # 评估模式
        action = self.agent.select_action(state, training=False)
        self.assertIn(action, range(5))

        print("  test_select_action passed")

    def test_epsilon_decay(self):
        """测试epsilon-贪婪衰减"""
        initial_epsilon = self.agent.epsilon

        # 执行多次更新
        for _ in range(100):
            self.agent.update_epsilon()

        # 检查epsilon值衰减
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_end)

        print("  test_epsilon_decay passed")

    def test_save_load(self):
        """测试模型保存和加载"""
        # 保存模型
        save_path = "tests/test_model.pth"
        self.agent.save(save_path)

        # 检查文件是否存在
        self.assertTrue(os.path.exists(save_path))

        # 加载模型
        new_agent = SchedulingAgent(state_dim=20, action_dim=5)
        new_agent.load(save_path)

        # 检查参数是否一致
        self.assertAlmostEqual(new_agent.epsilon, self.agent.epsilon)

        # 清理
        os.remove(save_path)

        print("  test_save_load passed")


class TestLegacyTaskParser(unittest.TestCase):
    """测试旧版任务解析器（向后兼容）"""

    def setUp(self):
        """测试初始化"""
        self.parser = LegacyTaskParser()

    def test_parse_json(self):
        """测试JSON格式解析"""
        json_str = '''{
            "task_id": "task_001",
            "user_id": "user_123",
            "task_type": "quantum",
            "qubit_count": 10,
            "circuit_depth": 50,
            "algorithm": "VQE",
            "estimated_time": 120.0,
            "priority": 4
        }'''

        features = self.parser.parse(json_str, format="json")

        # 检查结果
        self.assertIsNotNone(features)
        self.assertEqual(features.task_id, "task_001")
        self.assertEqual(features.task_type, "quantum")
        self.assertEqual(features.qubit_count, 10)
        self.assertEqual(features.algorithm, "VQE")

        print("  test_parse_json passed")

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

        # 检查结果
        self.assertIsNotNone(features)
        self.assertEqual(features.qubit_count, 5)
        self.assertGreater(features.gate_count, 0)
        self.assertGreater(features.measurement_count, 0)

        print("  test_parse_qasm passed")

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

        # 检查向量
        self.assertEqual(len(vector), 20)
        self.assertTrue(all(0.0 <= x <= 1.0 for x in vector))

        print("  test_to_vector passed")


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

    # ---- parse ----

    def test_parse_basic(self):
        """测试基本字典解析"""
        task = self.parser.parse(self.sample_dict)
        self.assertIsInstance(task, Task)
        self.assertEqual(task.task_id, "task_001")
        self.assertEqual(task.task_type, "quantum")
        self.assertEqual(task.algorithm, "VQE")
        self.assertEqual(task.qubits_required, 8)
        self.assertEqual(task.circuit_depth, 50)
        self.assertEqual(task.shots, 1024)
        self.assertEqual(task.estimated_time, 120)
        self.assertEqual(task.priority, 3)  # "high" → 3
        self.assertIsInstance(task.deadline, datetime)
        self.assertEqual(task.status, "pending")

        print("  test_parse_basic passed")

    def test_parse_with_int_priority(self):
        """测试整型 priority"""
        d = dict(self.sample_dict, priority=4)
        task = self.parser.parse(d)
        self.assertEqual(task.priority, 4)

        print("  test_parse_with_int_priority passed")

    def test_parse_classical(self):
        """测试经典任务类型"""
        d = dict(self.sample_dict, type="classical", qubits_required=0, algorithm=None)
        task = self.parser.parse(d)
        self.assertEqual(task.task_type, "classical")
        self.assertIsNone(task.algorithm)

        print("  test_parse_classical passed")

    def test_parse_missing_task_id_raises(self):
        """测试缺少 task_id 抛异常"""
        with self.assertRaises(ValueError):
            self.parser.parse({"type": "quantum", "qubits_required": 8})

        print("  test_parse_missing_task_id_raises passed")

    def test_parse_invalid_type_raises(self):
        """测试无效 task_type 抛异常"""
        with self.assertRaises(ValueError):
            self.parser.parse({"task_id": "x", "type": "invalid"})

        print("  test_parse_invalid_type_raises passed")

    def test_parse_invalid_priority_raises(self):
        """测试无效 priority 抛异常"""
        with self.assertRaises(ValueError):
            self.parser.parse(dict(self.sample_dict, priority="super_urgent"))

        print("  test_parse_invalid_priority_raises passed")

    def test_parse_qubits_exceed_limit_raises(self):
        """测试量子比特超限抛异常"""
        with self.assertRaises(ValueError):
            self.parser.parse(dict(self.sample_dict, qubits_required=999))

        print("  test_parse_qubits_exceed_limit_raises passed")

    def test_parse_not_dict_raises(self):
        """测试非字典输入抛异常"""
        with self.assertRaises(TypeError):
            self.parser.parse("not a dict")

        print("  test_parse_not_dict_raises passed")

    # ---- validate ----

    def test_validate_valid(self):
        """测试合法任务校验通过"""
        task = self.parser.parse(self.sample_dict)
        self.assertTrue(self.parser.validate(task))

        print("  test_validate_valid passed")

    def test_validate_invalid(self):
        """通过 Builder 绕过 parse 的校验，手动构造非法 Task"""
        # 直接构造一个 qubits=0 的 quantum task
        invalid_task = Task(
            task_id="bad",
            task_type="quantum",
            qubits_required=0,
            estimated_time=10,
            priority=1,
        )
        self.assertFalse(self.parser.validate(invalid_task))

        print("  test_validate_invalid passed")

    # ---- estimate_resources ----

    def test_estimate_resources(self):
        """测试资源预估"""
        task = self.parser.parse(self.sample_dict)
        res = self.parser.estimate_resources(task)
        self.assertIn("qubit_hours", res)
        self.assertIn("total_gate_operations", res)
        self.assertIn("memory_mb", res)
        self.assertIn("classical_compute_ratio", res)
        self.assertIn("estimated_queue_time", res)

        # 验证基本数值
        self.assertGreater(res["qubit_hours"], 0)
        self.assertEqual(res["total_gate_operations"], 50 * 1024)
        self.assertEqual(res["classical_compute_ratio"], 0.1)  # quantum

        print("  test_estimate_resources passed")

    # ---- to_internal_format ----

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

        # deadline 序列化为字符串
        self.assertIsInstance(internal["deadline"], str)

        print("  test_to_internal_format passed")

    # ---- TaskBuilder ----

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

        print("  test_builder_basic passed")

    def test_builder_from_dict(self):
        """测试 Builder.from_dict"""
        task = TaskBuilder.from_dict(self.sample_dict).build()
        self.assertEqual(task.task_id, "task_001")
        self.assertEqual(task.task_type, "quantum")
        self.assertEqual(task.priority, 3)

        print("  test_builder_from_dict passed")

    def test_builder_empty_id_raises(self):
        """测试 Builder 空 task_id 抛异常"""
        with self.assertRaises(ValueError):
            TaskBuilder().build()

        print("  test_builder_empty_id_raises passed")


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_env_agent_interaction(self):
        """测试环境和智能体交互"""
        env = SchedulingEnv()
        agent = SchedulingAgent(state_dim=20, action_dim=5)

        # 重置环境
        state, _ = env.reset()

        # 执行10步
        for i in range(10):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            # 检查交互
            self.assertEqual(next_state.shape, (20,))
            self.assertIsInstance(reward, float)

            state = next_state

            if terminated or truncated:
                break

        print("  test_env_agent_interaction passed")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Quantum RL Scheduling System - Unit Tests")
    print("=" * 60)
    print()

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestSchedulingEnv))
    suite.addTests(loader.loadTestsFromTestCase(TestSchedulerAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestLegacyTaskParser))
    suite.addTests(loader.loadTestsFromTestCase(TestTaskParser))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 打印摘要
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures:  {len(result.failures)}")
    print(f"Errors:    {len(result.errors)}")

    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print("Some tests failed!")

    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
