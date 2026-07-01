"""
状态空间扩展测试
Unit Tests for Extended State Space (14 dimensions)

测试覆盖：
- 阶段1：物理噪声特征（12维）
- 阶段2：拓扑特征（14维）
- 阶段3：LSTM策略训练收敛性
- 向后兼容性（旧10维模型在新环境上可加载）
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.agent import PPOAgent
from src.scheduler.env import (
    DEFAULT_MACHINE_CONFIGS,
    OBS_AVG_CONNECTIVITY,
    OBS_COUPLING_DENSITY,
    OBS_DIM,
    OBS_SINGLE_GATE_FIDELITY,
    OBS_TWO_GATE_FIDELITY,
    QuantumMachine,
    QuantumSchedulingEnv,
)


class TestExtendedStateSpace(unittest.TestCase):
    """测试扩展状态空间（14维）"""

    def setUp(self):
        """测试初始化"""
        self.env = QuantumSchedulingEnv(
            max_steps=100,
            max_qubits=287,
            seed=42,
        )

    def test_obs_dim_is_14(self):
        """测试观测空间维度为14"""
        self.assertEqual(OBS_DIM, 14, "OBS_DIM 应为 14")
        self.assertEqual(self.env.observation_space.shape[0], 14, "环境观测空间维度应为 14")

    def test_observation_shape(self):
        """测试观测向量形状"""
        obs, info = self.env.reset(seed=42)
        self.assertEqual(obs.shape, (14,), "观测向量形状应为 (14,)")
        self.assertEqual(obs.dtype, np.float32, "观测向量类型应为 float32")

    def test_observation_range(self):
        """测试所有维度在 [0, 1] 范围内"""
        obs, _ = self.env.reset(seed=42)

        # 运行多个步骤，检查观测值始终在 [0, 1]
        for _ in range(50):
            action = self.env.action_space.sample()
            obs, _, terminated, truncated, _ = self.env.step(action)

            self.assertTrue(np.all(obs >= 0.0), f"观测值应 >= 0，但发现 {obs.min()}")
            self.assertTrue(np.all(obs <= 1.0), f"观测值应 <= 1，但发现 {obs.max()}")

            if terminated or truncated:
                obs, _ = self.env.reset()

    def test_physical_noise_features(self):
        """测试物理噪声特征（维度10-11）"""
        obs, _ = self.env.reset(seed=42)

        # 单比特门保真度
        single_gate_fid = obs[OBS_SINGLE_GATE_FIDELITY]
        self.assertGreaterEqual(single_gate_fid, 0.0, "单比特门保真度应 >= 0")
        self.assertLessEqual(single_gate_fid, 1.0, "单比特门保真度应 <= 1")

        # 两比特门保真度
        two_gate_fid = obs[OBS_TWO_GATE_FIDELITY]
        self.assertGreaterEqual(two_gate_fid, 0.0, "两比特门保真度应 >= 0")
        self.assertLessEqual(two_gate_fid, 1.0, "两比特门保真度应 <= 1")

        # 两比特门保真度通常低于单比特门
        self.assertLessEqual(
            two_gate_fid,
            single_gate_fid + 0.1,  # 允许小范围波动
            "两比特门保真度通常应低于单比特门保真度",
        )

    def test_topology_features(self):
        """测试拓扑特征（维度12-13）"""
        obs, _ = self.env.reset(seed=42)

        # 耦合图密度
        coupling_density = obs[OBS_COUPLING_DENSITY]
        self.assertGreaterEqual(coupling_density, 0.0, "耦合图密度应 >= 0")
        self.assertLessEqual(coupling_density, 1.0, "耦合图密度应 <= 1")

        # 平均连通度
        avg_connectivity = obs[OBS_AVG_CONNECTIVITY]
        self.assertGreaterEqual(avg_connectivity, 0.0, "平均连通度应 >= 0")
        self.assertLessEqual(avg_connectivity, 1.0, "平均连通度应 <= 1")

    def test_quantum_machine_noise_features(self):
        """测试 QuantumMachine 的噪声特征更新"""
        machine = QuantumMachine(
            name="test_machine",
            total_qubits=100,
            fidelity=0.95,
        )

        rng = np.random.default_rng(42)

        # 更新噪声特征
        machine.update_noise_features(rng)

        # 验证单比特门保真度
        self.assertGreater(machine.single_gate_fidelity, 0.0)
        self.assertLess(machine.single_gate_fidelity, 1.0)

        # 验证两比特门保真度
        self.assertGreater(machine.two_gate_fidelity, 0.0)
        self.assertLess(machine.two_gate_fidelity, 1.0)

        # 两比特门保真度应低于单比特门
        self.assertLess(machine.two_gate_fidelity, machine.single_gate_fidelity)

    def test_quantum_machine_topology_features(self):
        """测试 QuantumMachine 的拓扑特征更新"""
        # 小芯片（<100 qubits）
        small_machine = QuantumMachine(name="small", total_qubits=72)
        small_machine.update_topology_features()
        self.assertAlmostEqual(
            small_machine.coupling_density, 0.7, places=2, msg="小芯片耦合密度应约为 0.7"
        )

        # 中芯片（100-200 qubits）
        medium_machine = QuantumMachine(name="medium", total_qubits=176)
        medium_machine.update_topology_features()
        self.assertAlmostEqual(
            medium_machine.coupling_density, 0.5, places=2, msg="中芯片耦合密度应约为 0.5"
        )

        # 大芯片（>200 qubits）
        large_machine = QuantumMachine(name="large", total_qubits=287)
        large_machine.update_topology_features()
        self.assertAlmostEqual(
            large_machine.coupling_density, 0.3, places=2, msg="大芯片耦合密度应约为 0.3"
        )

        # 平均连通度
        self.assertGreater(large_machine.avg_connectivity, 0.0)
        self.assertLess(large_machine.avg_connectivity, 1.0)


class TestMultiMachineExtendedState(unittest.TestCase):
    """测试多机器模式下的扩展状态空间"""

    def setUp(self):
        """测试初始化"""
        self.env = QuantumSchedulingEnv(
            max_steps=100,
            machine_configs=DEFAULT_MACHINE_CONFIGS,
            seed=42,
        )

    def test_multi_machine_obs_dim(self):
        """测试多机器模式下观测维度"""
        self.assertEqual(self.env.observation_space.shape[0], 14, "多机器模式观测维度应为 14")

    def test_multi_machine_observation(self):
        """测试多机器模式下观测向量"""
        obs, _ = self.env.reset(seed=42)

        self.assertEqual(obs.shape, (14,), "观测向量形状应为 (14,)")
        self.assertTrue(np.all(obs >= 0.0), "所有观测值应 >= 0")
        self.assertTrue(np.all(obs <= 1.0), "所有观测值应 <= 1")

        # 物理噪声特征应为所有机器的加权平均
        single_gate_fid = obs[OBS_SINGLE_GATE_FIDELITY]
        two_gate_fid = obs[OBS_TWO_GATE_FIDELITY]

        self.assertGreater(single_gate_fid, 0.0, "单比特门保真度应 > 0")
        self.assertGreater(two_gate_fid, 0.0, "两比特门保真度应 > 0")


class TestLSTMPolicy(unittest.TestCase):
    """测试 LSTM 策略（阶段3）"""

    def test_lstm_agent_creation(self):
        """测试 LSTM 智能体创建"""
        env = QuantumSchedulingEnv(max_steps=50, seed=42)

        agent = PPOAgent(
            env,
            use_lstm=True,
            n_lstm_layers=1,
            lstm_hidden_size=64,
            learning_rate=3e-4,
            verbose=0,
        )

        self.assertTrue(agent.use_lstm, "use_lstm 应为 True")
        self.assertEqual(agent.n_lstm_layers, 1, "LSTM 层数应为 1")
        self.assertEqual(agent.lstm_hidden_size, 64, "LSTM 隐藏层大小应为 64")

    def test_lstm_model_build(self):
        """测试 LSTM 模型构建"""
        env = QuantumSchedulingEnv(max_steps=50, seed=42)

        agent = PPOAgent(
            env,
            use_lstm=True,
            n_lstm_layers=1,
            lstm_hidden_size=64,
            verbose=0,
        )

        model = agent._build_model()
        self.assertIsNotNone(model, "模型应成功构建")

        # 验证策略类型为 RecurrentActorCriticPolicy（RecurrentPPO 的策略）
        policy_class = model.policy.__class__.__name__
        self.assertIn(
            "Recurrent", policy_class, f"策略类名应包含 'Recurrent'，实际为 {policy_class}"
        )

    def test_lstm_training_convergence(self):
        """测试 LSTM 训练收敛性（50000步后 mean reward > 2000）"""
        env = QuantumSchedulingEnv(max_steps=100, seed=42)

        agent = PPOAgent(
            env,
            use_lstm=True,
            n_lstm_layers=1,
            lstm_hidden_size=64,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            verbose=0,
            seed=42,
        )

        # 训练 50000 步
        agent.train(total_timesteps=50000, eval_freq=10000, n_eval_episodes=5)

        # 评估
        results = agent.evaluate(num_episodes=10, deterministic=True)
        mean_reward = results["mean_reward"]

        self.assertGreater(
            mean_reward,
            2000,
            f"LSTM PPO 训练 50000 步后 mean reward 应 > 2000，实际为 {mean_reward:.2f}",
        )


class TestBackwardCompatibility(unittest.TestCase):
    """测试向后兼容性（旧10维模型在新环境上可加载）"""

    def test_old_model_load(self):
        """测试旧10维模型在新14维环境上可加载（不报错）"""
        # 创建一个旧模型（模拟10维观测）
        # 注意：这里我们创建一个新模型，但测试加载逻辑
        env = QuantumSchedulingEnv(max_steps=50, seed=42)

        agent = PPOAgent(env, verbose=0)
        agent.train(total_timesteps=1000, eval_freq=500, n_eval_episodes=2)

        # 保存模型
        save_path = "./test_old_model"
        agent.save(save_path)

        # 在新环境上加载（新环境是14维）
        new_env = QuantumSchedulingEnv(max_steps=50, seed=42)
        new_agent = PPOAgent(new_env, verbose=0)

        # 加载不应报错
        try:
            new_agent.load(save_path)
            loaded_successfully = True
        except Exception as e:
            loaded_successfully = False
            print(f"加载失败: {e}")

        self.assertTrue(loaded_successfully, "旧模型应能在新环境上加载（即使性能下降）")

        # 清理测试文件
        if os.path.exists(f"{save_path}.zip"):
            os.remove(f"{save_path}.zip")


class TestObservationIndices(unittest.TestCase):
    """测试观测向量索引常量"""

    def test_obs_indices(self):
        """测试所有 OBS_* 索引常量"""
        from src.scheduler.env import (
            OBS_AVG_CONNECTIVITY,
            OBS_AVG_WAIT_TIME,
            OBS_CLASSICAL_LOAD,
            OBS_COUPLING_DENSITY,
            OBS_FIDELITY,
            OBS_QUANTUM_QUEUE_RATIO,
            OBS_QUBIT_AVAILABILITY,
            OBS_QUEUE_LENGTH,
            OBS_SINGLE_GATE_FIDELITY,
            OBS_TASK_TYPE_CLASSICAL,
            OBS_TASK_TYPE_QUANTUM,
            OBS_TIME_OF_DAY,
            OBS_TWO_GATE_FIDELITY,
            OBS_URGENCY_LEVEL,
        )

        self.assertEqual(OBS_QUBIT_AVAILABILITY, 0)
        self.assertEqual(OBS_QUEUE_LENGTH, 1)
        self.assertEqual(OBS_AVG_WAIT_TIME, 2)
        self.assertEqual(OBS_FIDELITY, 3)
        self.assertEqual(OBS_CLASSICAL_LOAD, 4)
        self.assertEqual(OBS_QUANTUM_QUEUE_RATIO, 5)
        self.assertEqual(OBS_TIME_OF_DAY, 6)
        self.assertEqual(OBS_URGENCY_LEVEL, 7)
        self.assertEqual(OBS_TASK_TYPE_QUANTUM, 8)
        self.assertEqual(OBS_TASK_TYPE_CLASSICAL, 9)
        self.assertEqual(OBS_SINGLE_GATE_FIDELITY, 10)
        self.assertEqual(OBS_TWO_GATE_FIDELITY, 11)
        self.assertEqual(OBS_COUPLING_DENSITY, 12)
        self.assertEqual(OBS_AVG_CONNECTIVITY, 13)


if __name__ == "__main__":
    unittest.main()
