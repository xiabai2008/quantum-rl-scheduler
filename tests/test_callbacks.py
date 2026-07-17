"""
量子RL调度系统 - 自定义回调模块单元测试
Unit Tests for src/scheduler/callbacks.py

测试覆盖：
- EpsilonExplorationCallback: Epsilon 探索率衰减回调
    - 初始化（默认/自定义参数）
    - _on_step 衰减逻辑、logger.record 调用
    - epsilon 不会低于 epsilon_end 下限
- AnnealingCallback: 量子退火优化回调
    - 非 interval 倍数步不触发
    - interval 倍数步触发 optimizer.optimize_policy
    - 异常处理（不中断训练）
    - quality > best_reward 时更新计数
    - 无 _evaluate_network_quality 方法的分支
    - head_only 参数透传
- RealMachineCallback: 真机抽样回调
    - n_calls=0 / 非 interval 倍数 跳过
    - prob<=0 跳过
    - random.random() >= prob 跳过
    - 无 client 且 env._real_clients 为空时降级（仅警告一次）
    - 有显式 client 时正常提交
    - 从 env._real_clients 自动取 client
    - 提交异常时记录 error 状态
    - 从 env.get_random_pending_task 获取任务
    - _on_training_end 保存 JSON / 不保存 / OSError / 创建目录
"""

import itertools
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.callbacks import (
    AnnealingCallback,
    EpsilonExplorationCallback,
    RealMachineCallback,
)


# ============================================================
# 辅助函数：为 BaseCallback 子类注入运行时必需属性
# ============================================================
def _bind_callback(cb, n_calls=1, model=None, logger=None):
    """为回调注入 n_calls / model / logger 等运行时属性，便于直接调用 _on_step。

    SB3 2.9.0 中 ``logger`` 是只读 property，实际返回 ``self.model.logger``，
    因此通过设置 ``cb.model``（必要时覆写 ``cb.model.logger``）来注入 mock logger。

    Args:
        cb: BaseCallback 子类实例
        n_calls: 当前调用次数
        model: mock 的 SB3 模型
        logger: mock 的 logger（将赋给 model.logger）

    Returns:
        注入属性后的回调实例
    """
    cb.n_calls = n_calls
    cb.model = model if model is not None else MagicMock()
    if logger is not None:
        cb.model.logger = logger
    return cb


# ============================================================
# EpsilonExplorationCallback 测试
# ============================================================
class TestEpsilonExplorationCallbackInit(unittest.TestCase):
    """测试 Epsilon 探索回调初始化。"""

    def test_default_init(self):
        """默认参数初始化应设置标准值。"""
        cb = EpsilonExplorationCallback()
        self.assertEqual(cb.epsilon, 1.0)
        self.assertEqual(cb.epsilon_end, 0.05)
        self.assertEqual(cb.epsilon_decay, 0.995)
        self.assertEqual(cb.decay_freq, 1)

    def test_custom_init(self):
        """自定义参数应被正确存储。"""
        cb = EpsilonExplorationCallback(
            epsilon_start=0.8,
            epsilon_end=0.1,
            epsilon_decay=0.9,
            decay_freq=5,
            verbose=1,
        )
        self.assertEqual(cb.epsilon, 0.8)
        self.assertEqual(cb.epsilon_end, 0.1)
        self.assertEqual(cb.epsilon_decay, 0.9)
        self.assertEqual(cb.decay_freq, 5)
        self.assertEqual(cb.verbose, 1)


class TestEpsilonExplorationCallbackOnStep(unittest.TestCase):
    """测试 Epsilon 探索回调 _on_step 行为。"""

    def test_on_step_decays_epsilon(self):
        """_on_step 应按 decay 系数衰减 epsilon。"""
        cb = _bind_callback(EpsilonExplorationCallback(epsilon_start=1.0, epsilon_decay=0.995))
        cb._on_step()
        self.assertAlmostEqual(cb.epsilon, 0.995)

    def test_on_step_records_to_logger(self):
        """_on_step 应调用 logger.record 记录 epsilon 值。"""
        mock_logger = MagicMock()
        cb = _bind_callback(EpsilonExplorationCallback(), logger=mock_logger)
        cb._on_step()
        mock_logger.record.assert_called_once()
        args, _ = mock_logger.record.call_args
        self.assertEqual(args[0], "exploration/epsilon")
        self.assertAlmostEqual(args[1], 0.995)

    def test_on_step_returns_true(self):
        """_on_step 应始终返回 True 以继续训练。"""
        cb = _bind_callback(EpsilonExplorationCallback())
        self.assertTrue(cb._on_step())

    def test_on_step_epsilon_not_below_end(self):
        """多次衰减后 epsilon 不应低于 epsilon_end 下限。"""
        cb = _bind_callback(
            EpsilonExplorationCallback(
                epsilon_start=0.2,
                epsilon_end=0.15,
                epsilon_decay=0.5,
            )
        )
        # 衰减一次：max(0.15, 0.2*0.5=0.1) = 0.15
        cb._on_step()
        self.assertAlmostEqual(cb.epsilon, 0.15)
        # 再次衰减：max(0.15, 0.15*0.5=0.075) = 0.15（保持下限）
        cb._on_step()
        self.assertAlmostEqual(cb.epsilon, 0.15)

    def test_on_step_multiple_decays(self):
        """连续多次衰减应单调递减直至下限。"""
        cb = _bind_callback(
            EpsilonExplorationCallback(
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay=0.9,
            )
        )
        values = []
        for _ in range(50):
            cb._on_step()
            values.append(cb.epsilon)
        # 应单调非递增
        for a, b in itertools.pairwise(values):
            self.assertGreaterEqual(a, b)
        # 最终应收敛到下限
        self.assertAlmostEqual(cb.epsilon, 0.05)


# ============================================================
# AnnealingCallback 测试
# ============================================================
class TestAnnealingCallbackInit(unittest.TestCase):
    """测试退火回调初始化。"""

    def test_init_default(self):
        """默认参数初始化应设置标准值。"""
        opt = MagicMock()
        cb = AnnealingCallback(optimizer=opt)
        self.assertIs(cb.optimizer, opt)
        self.assertEqual(cb.interval, 1000)
        self.assertEqual(cb.best_reward, -float("inf"))
        self.assertEqual(cb.optimized_count, 0)
        self.assertTrue(cb.head_only)

    def test_init_custom(self):
        """自定义参数应被正确存储。"""
        opt = MagicMock()
        cb = AnnealingCallback(
            optimizer=opt,
            interval=500,
            verbose=1,
            head_only=False,
        )
        self.assertEqual(cb.interval, 500)
        self.assertEqual(cb.verbose, 1)
        self.assertFalse(cb.head_only)


class TestAnnealingCallbackOnStep(unittest.TestCase):
    """测试退火回调 _on_step 行为。"""

    def test_on_step_no_trigger_when_not_interval_multiple(self):
        """非 interval 倍数步不应触发 optimize_policy。"""
        opt = MagicMock()
        cb = _bind_callback(AnnealingCallback(optimizer=opt, interval=1000), n_calls=500)
        cb._on_step()
        opt.optimize_policy.assert_not_called()

    def test_on_step_no_trigger_when_n_calls_zero(self):
        """n_calls=0 时不应触发 optimize_policy。"""
        opt = MagicMock()
        cb = _bind_callback(AnnealingCallback(optimizer=opt, interval=1000), n_calls=0)
        cb._on_step()
        opt.optimize_policy.assert_not_called()

    def test_on_step_triggers_at_interval_multiple(self):
        """interval 倍数步应触发 optimize_policy。"""
        opt = MagicMock()
        optimized_agent = MagicMock()
        opt.optimize_policy.return_value = optimized_agent
        # 无 _evaluate_network_quality 方法 -> quality 保持 0.0
        del opt._evaluate_network_quality
        del opt._get_policy_net
        cb = _bind_callback(AnnealingCallback(optimizer=opt, interval=1000), n_calls=1000)
        cb._on_step()
        opt.optimize_policy.assert_called_once_with(cb.model, head_only=True)

    def test_on_step_head_only_param_passthrough(self):
        """head_only 参数应正确透传到 optimize_policy。"""
        opt = MagicMock()
        opt.optimize_policy.return_value = MagicMock()
        del opt._evaluate_network_quality
        del opt._get_policy_net
        cb = _bind_callback(
            AnnealingCallback(optimizer=opt, interval=100, head_only=False),
            n_calls=100,
        )
        cb._on_step()
        opt.optimize_policy.assert_called_once_with(cb.model, head_only=False)

    def test_on_step_updates_count_when_quality_better(self):
        """quality > best_reward 时应更新 best_reward 和 optimized_count。"""
        opt = MagicMock()
        optimized_agent = MagicMock()
        opt.optimize_policy.return_value = optimized_agent
        policy_net = MagicMock()
        opt._get_policy_net.return_value = policy_net
        # loss=0.1 -> quality=-0.1，大于初始 -inf
        opt._evaluate_network_quality.return_value = 0.1
        cb = _bind_callback(
            AnnealingCallback(optimizer=opt, interval=1000, verbose=1),
            n_calls=1000,
        )
        cb._on_step()
        self.assertEqual(cb.optimized_count, 1)
        self.assertAlmostEqual(cb.best_reward, -0.1)

    def test_on_step_no_update_when_quality_not_better(self):
        """quality <= best_reward 时不应更新 optimized_count。"""
        opt = MagicMock()
        opt.optimize_policy.return_value = MagicMock()
        opt._get_policy_net.return_value = MagicMock()
        opt._evaluate_network_quality.return_value = 0.5  # loss=0.5 -> quality=-0.5
        cb = _bind_callback(
            AnnealingCallback(optimizer=opt, interval=1000),
            n_calls=1000,
        )
        # 预设 best_reward 已为较高值
        cb.best_reward = 10.0
        cb._on_step()
        self.assertEqual(cb.optimized_count, 0)
        self.assertEqual(cb.best_reward, 10.0)

    def test_on_step_no_quality_eval_when_policy_net_none(self):
        """_get_policy_net 返回 None 时 quality 保持 0.0。"""
        opt = MagicMock()
        opt.optimize_policy.return_value = MagicMock()
        opt._get_policy_net.return_value = None
        # _evaluate_network_quality 不应被调用
        opt._evaluate_network_quality.return_value = 0.5
        cb = _bind_callback(
            AnnealingCallback(optimizer=opt, interval=1000),
            n_calls=1000,
        )
        cb._on_step()
        opt._evaluate_network_quality.assert_not_called()
        # quality=0.0 > -inf，应更新计数
        self.assertEqual(cb.optimized_count, 1)
        self.assertEqual(cb.best_reward, 0.0)

    def test_on_step_exception_handled(self):
        """optimize_policy 抛异常时应被捕获，不中断训练。"""
        opt = MagicMock()
        opt.optimize_policy.side_effect = RuntimeError("annealing failed")
        cb = _bind_callback(
            AnnealingCallback(optimizer=opt, interval=1000, verbose=1),
            n_calls=1000,
        )
        # 应返回 True，不抛异常
        result = cb._on_step()
        self.assertTrue(result)
        self.assertEqual(cb.optimized_count, 0)

    def test_on_step_returns_true_when_not_triggered(self):
        """未触发时 _on_step 应返回 True。"""
        opt = MagicMock()
        cb = _bind_callback(AnnealingCallback(optimizer=opt, interval=1000), n_calls=1)
        self.assertTrue(cb._on_step())

    def test_on_step_quality_negative_loss_conversion(self):
        """quality 应为 -loss（loss 越小 quality 越大）。"""
        opt = MagicMock()
        opt.optimize_policy.return_value = MagicMock()
        opt._get_policy_net.return_value = MagicMock()
        opt._evaluate_network_quality.return_value = 2.5  # loss=2.5 -> quality=-2.5
        cb = _bind_callback(
            AnnealingCallback(optimizer=opt, interval=100),
            n_calls=100,
        )
        cb._on_step()
        # quality=-2.5 > -inf，应更新
        self.assertEqual(cb.best_reward, -2.5)
        self.assertEqual(cb.optimized_count, 1)


# ============================================================
# RealMachineCallback 测试
# ============================================================
class TestRealMachineCallbackInit(unittest.TestCase):
    """测试真机抽样回调初始化。"""

    def test_init_default(self):
        """默认参数初始化应设置标准值。"""
        env = MagicMock()
        cb = RealMachineCallback(env=env)
        self.assertIs(cb.env, env)
        self.assertEqual(cb.interval, 1000)
        self.assertAlmostEqual(cb.prob, 0.05)
        self.assertIsNone(cb.client)
        self.assertEqual(cb.save_path, "results/real_times.json")
        self.assertEqual(cb.shots, 512)
        self.assertEqual(cb.verbose, 1)
        self.assertEqual(cb.real_times, [])
        self.assertFalse(cb._warned_no_client)

    def test_init_custom(self):
        """自定义参数应被正确存储（含类型强转）。"""
        env = MagicMock()
        client = MagicMock()
        cb = RealMachineCallback(
            env=env,
            interval="2000",
            prob="0.1",
            client=client,
            save_path="/tmp/test_real.json",
            shots="1024",
            verbose=0,
        )
        self.assertEqual(cb.interval, 2000)
        self.assertAlmostEqual(cb.prob, 0.1)
        self.assertIs(cb.client, client)
        self.assertEqual(cb.shots, 1024)
        self.assertEqual(cb.verbose, 0)


class TestRealMachineCallbackOnStepSkip(unittest.TestCase):
    """测试真机回调 _on_step 的跳过逻辑。"""

    def test_skip_when_n_calls_zero(self):
        """n_calls=0 时应跳过。"""
        env = MagicMock()
        cb = _bind_callback(RealMachineCallback(env=env, interval=1000), n_calls=0)
        with patch("src.scheduler.callbacks.random.random", return_value=0.01):
            cb._on_step()
        self.assertEqual(cb.real_times, [])

    def test_skip_when_not_interval_multiple(self):
        """非 interval 倍数步应跳过。"""
        env = MagicMock()
        cb = _bind_callback(RealMachineCallback(env=env, interval=1000), n_calls=500)
        with patch("src.scheduler.callbacks.random.random", return_value=0.01):
            cb._on_step()
        self.assertEqual(cb.real_times, [])

    def test_skip_when_prob_le_zero(self):
        """prob<=0 时应跳过。"""
        env = MagicMock()
        cb = _bind_callback(
            RealMachineCallback(env=env, interval=1000, prob=0.0),
            n_calls=1000,
        )
        with patch("src.scheduler.callbacks.random.random", return_value=0.01):
            cb._on_step()
        self.assertEqual(cb.real_times, [])

    def test_skip_when_random_ge_prob(self):
        """random.random() >= prob 时应跳过。"""
        env = MagicMock()
        cb = _bind_callback(
            RealMachineCallback(env=env, interval=1000, prob=0.05),
            n_calls=1000,
        )
        # random=0.5 >= prob=0.05，应跳过
        with patch("src.scheduler.callbacks.random.random", return_value=0.5):
            cb._on_step()
        self.assertEqual(cb.real_times, [])


class TestRealMachineCallbackDegrade(unittest.TestCase):
    """测试真机回调无 client 时的降级行为。"""

    def test_degrade_when_no_client_and_empty_env_clients(self):
        """无 client 且 env._real_clients 为空时应降级跳过。"""
        env = MagicMock()
        env._real_clients = {}
        cb = _bind_callback(
            RealMachineCallback(env=env, interval=1000, prob=0.5, verbose=0),
            n_calls=1000,
        )
        with patch("src.scheduler.callbacks.random.random", return_value=0.01):
            cb._on_step()
        self.assertEqual(cb.real_times, [])
        self.assertTrue(cb._warned_no_client)

    def test_degrade_when_env_has_no_real_clients_attr(self):
        """env 无 _real_clients 属性时应降级跳过。"""
        env = MagicMock()
        # 删除 _real_clients 属性，使 getattr 返回默认值 {}
        del env._real_clients
        env.configure_mock(**{"_real_clients": {}})
        cb = _bind_callback(
            RealMachineCallback(env=env, interval=1000, prob=0.5, verbose=0),
            n_calls=1000,
        )
        with patch("src.scheduler.callbacks.random.random", return_value=0.01):
            cb._on_step()
        self.assertEqual(cb.real_times, [])

    def test_warned_no_client_only_once(self):
        """无 client 警告标志应只触发一次，后续不再重复警告。"""
        env = MagicMock()
        env._real_clients = {}
        cb = _bind_callback(
            RealMachineCallback(env=env, interval=1000, prob=0.5, verbose=0),
            n_calls=1000,
        )
        with patch("src.scheduler.callbacks.random.random", return_value=0.01):
            cb._on_step()
            self.assertTrue(cb._warned_no_client)
            # 第二次调用，_warned_no_client 应保持 True（已设置）
            cb.n_calls = 2000
            cb._on_step()
        self.assertTrue(cb._warned_no_client)
        self.assertEqual(cb.real_times, [])


class TestRealMachineCallbackSubmit(unittest.TestCase):
    """测试真机回调正常提交逻辑。"""

    def test_submit_with_explicit_client(self):
        """有显式 client 时应正常提交任务并记录。"""
        env = MagicMock()
        # 删除 get_random_pending_task 使 hasattr 返回 False，task 保持 None
        del env.get_random_pending_task
        client = MagicMock()
        client.machine_name = "tianyan-287"
        client.submit_quantum_task.return_value = "real-task-001"
        cb = _bind_callback(
            RealMachineCallback(
                env=env,
                interval=1000,
                prob=0.5,
                client=client,
                verbose=1,
            ),
            n_calls=1000,
        )
        with patch("src.scheduler.callbacks.random.random", return_value=0.01):
            cb._on_step()
        client.submit_quantum_task.assert_called_once()
        self.assertEqual(len(cb.real_times), 1)
        record = cb.real_times[0]
        self.assertEqual(record["step"], 1000)
        self.assertEqual(record["machine"], "tianyan-287")
        self.assertEqual(record["status"], "submitted")
        self.assertEqual(record["real_task_id"], "real-task-001")
        self.assertEqual(record["task_id"], "synthetic")

    def test_submit_with_env_real_clients(self):
        """无显式 client 时应从 env._real_clients 取第一项。"""
        env = MagicMock()
        env._real_clients = {}
        client = MagicMock()
        env._real_clients["tianyan-287"] = client
        client.submit_quantum_task.return_value = "tid-002"
        # env 无 get_random_pending_task 方法 -> task=None
        del env.get_random_pending_task
        cb = _bind_callback(
            RealMachineCallback(env=env, interval=1000, prob=0.5, verbose=0),
            n_calls=1000,
        )
        with patch("src.scheduler.callbacks.random.random", return_value=0.01):
            cb._on_step()
        client.submit_quantum_task.assert_called_once()
        self.assertEqual(len(cb.real_times), 1)
        self.assertEqual(cb.real_times[0]["machine"], "tianyan-287")
        self.assertEqual(cb.real_times[0]["status"], "submitted")

    def test_submit_records_rejected_when_real_tid_falsy(self):
        """真机返回 falsy 值时 status 应为 rejected。"""
        env = MagicMock()
        client = MagicMock()
        client.submit_quantum_task.return_value = None
        cb = _bind_callback(
            RealMachineCallback(env=env, interval=1000, prob=0.5, client=client),
            n_calls=1000,
        )
        with patch("src.scheduler.callbacks.random.random", return_value=0.01):
            cb._on_step()
        self.assertEqual(cb.real_times[0]["status"], "rejected")
        self.assertIsNone(cb.real_times[0]["real_task_id"])

    def test_submit_with_task_from_env(self):
        """从 env.get_random_pending_task 获取任务时应使用任务的 task_id 和 qcis。"""
        env = MagicMock()
        env._real_clients = {}
        client = MagicMock()
        client.machine_name = "tianyan-287"
        client.submit_quantum_task.return_value = "tid-003"
        # 模拟任务对象
        task = MagicMock()
        task.task_id = "task-42"
        task.qcis = "X Q0\nM Q0"
        env.get_random_pending_task.return_value = task
        cb = _bind_callback(
            RealMachineCallback(env=env, interval=1000, prob=0.5, client=client),
            n_calls=1000,
        )
        with patch("src.scheduler.callbacks.random.random", return_value=0.01):
            cb._on_step()
        # 应调用 client.submit_quantum_task，并传入任务的 qcis
        _, kwargs = client.submit_quantum_task.call_args
        self.assertEqual(kwargs["qcis"], "X Q0\nM Q0")
        self.assertEqual(kwargs["shots"], 512)
        self.assertIn("task-42", kwargs["task_name"])
        self.assertEqual(cb.real_times[0]["task_id"], "task-42")

    def test_submit_with_task_no_qcis_uses_placeholder(self):
        """任务无 qcis 字段时应使用占位电路 H Q0\\nM Q0。"""
        env = MagicMock()
        client = MagicMock()
        client.submit_quantum_task.return_value = "tid-004"
        task = MagicMock()
        task.task_id = "task-99"
        task.qcis = None  # qcis 为 None
        env.get_random_pending_task.return_value = task
        cb = _bind_callback(
            RealMachineCallback(env=env, interval=1000, prob=0.5, client=client),
            n_calls=1000,
        )
        with patch("src.scheduler.callbacks.random.random", return_value=0.01):
            cb._on_step()
        _, kwargs = client.submit_quantum_task.call_args
        self.assertEqual(kwargs["qcis"], "H Q0\nM Q0")

    def test_submit_get_pending_task_exception_falls_back(self):
        """get_random_pending_task 抛异常时应降级为无任务（不中断）。"""
        env = MagicMock()
        client = MagicMock()
        client.submit_quantum_task.return_value = "tid-005"
        env.get_random_pending_task.side_effect = RuntimeError("env state error")
        cb = _bind_callback(
            RealMachineCallback(env=env, interval=1000, prob=0.5, client=client),
            n_calls=1000,
        )
        with patch("src.scheduler.callbacks.random.random", return_value=0.01):
            cb._on_step()
        # 异常被捕获，仍应提交（用 synthetic task）
        client.submit_quantum_task.assert_called_once()
        self.assertEqual(cb.real_times[0]["task_id"], "synthetic")

    def test_submit_exception_records_error(self):
        """submit_quantum_task 抛异常时应记录 error 状态。"""
        env = MagicMock()
        client = MagicMock()
        client.machine_name = "tianyan-287"
        client.submit_quantum_task.side_effect = RuntimeError("network timeout")
        cb = _bind_callback(
            RealMachineCallback(env=env, interval=1000, prob=0.5, client=client, verbose=1),
            n_calls=1000,
        )
        with patch("src.scheduler.callbacks.random.random", return_value=0.01):
            cb._on_step()
        self.assertEqual(len(cb.real_times), 1)
        record = cb.real_times[0]
        self.assertTrue(record["status"].startswith("error:"))
        self.assertIn("network timeout", record["status"])
        self.assertIsNone(record["real_task_id"])

    def test_submit_machine_name_unknown_when_client_no_attr(self):
        """client 无 machine_name 属性时 machine 应为 unknown。"""
        env = MagicMock()
        client = MagicMock()
        del client.machine_name  # 删除属性，使 getattr 返回默认
        client.submit_quantum_task.return_value = "tid-006"
        cb = _bind_callback(
            RealMachineCallback(env=env, interval=1000, prob=0.5, client=client),
            n_calls=1000,
        )
        with patch("src.scheduler.callbacks.random.random", return_value=0.01):
            cb._on_step()
        self.assertEqual(cb.real_times[0]["machine"], "unknown")

    def test_submit_task_id_str_conversion(self):
        """task_id 应被转为字符串。"""
        env = MagicMock()
        client = MagicMock()
        client.submit_quantum_task.return_value = "tid-007"
        task = MagicMock()
        task.task_id = 12345  # 整数 task_id
        task.qcis = "H Q0\nM Q0"
        env.get_random_pending_task.return_value = task
        cb = _bind_callback(
            RealMachineCallback(env=env, interval=1000, prob=0.5, client=client),
            n_calls=1000,
        )
        with patch("src.scheduler.callbacks.random.random", return_value=0.01):
            cb._on_step()
        self.assertEqual(cb.real_times[0]["task_id"], "12345")

    def test_submit_task_id_synthetic_when_no_task_id_attr(self):
        """任务无 task_id 属性时 task_id 应为 synthetic。"""
        env = MagicMock()
        client = MagicMock()
        client.submit_quantum_task.return_value = "tid-008"
        task = MagicMock()
        del task.task_id  # 删除 task_id 属性
        task.qcis = "H Q0\nM Q0"
        env.get_random_pending_task.return_value = task
        cb = _bind_callback(
            RealMachineCallback(env=env, interval=1000, prob=0.5, client=client),
            n_calls=1000,
        )
        with patch("src.scheduler.callbacks.random.random", return_value=0.01):
            cb._on_step()
        self.assertEqual(cb.real_times[0]["task_id"], "synthetic")


# ============================================================
# RealMachineCallback._on_training_end 测试
# ============================================================
class TestRealMachineCallbackTrainingEnd(unittest.TestCase):
    """测试真机回调 _on_training_end 保存 JSON 行为。"""

    def test_save_json_writes_records(self):
        """_on_training_end 应将 real_times 写入 JSON 文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "sub", "real_times.json")
            env = MagicMock()
            cb = _bind_callback(
                RealMachineCallback(env=env, save_path=save_path, verbose=1),
                n_calls=1,
            )
            cb.real_times = [
                {"step": 1000, "task_id": "t1", "status": "submitted"},
                {"step": 2000, "task_id": "t2", "status": "error: timeout"},
            ]
            cb._on_training_end()
            self.assertTrue(os.path.exists(save_path))
            with open(save_path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["task_id"], "t1")
            self.assertEqual(data[1]["status"], "error: timeout")

    def test_save_json_empty_records(self):
        """real_times 为空时也应写入空 JSON 数组。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "real_times.json")
            env = MagicMock()
            cb = _bind_callback(RealMachineCallback(env=env, save_path=save_path), n_calls=1)
            cb._on_training_end()
            with open(save_path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data, [])

    def test_no_save_when_save_path_empty(self):
        """save_path 为空字符串时应直接返回不保存。"""
        env = MagicMock()
        cb = _bind_callback(
            RealMachineCallback(env=env, save_path=""),
            n_calls=1,
        )
        cb.real_times = [{"step": 1}]
        # 不应抛异常
        cb._on_training_end()

    def test_save_creates_directory(self):
        """save_path 包含子目录时应自动创建。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "nested", "deep", "real.json")
            env = MagicMock()
            cb = _bind_callback(RealMachineCallback(env=env, save_path=save_path), n_calls=1)
            cb.real_times = [{"step": 1, "status": "submitted"}]
            cb._on_training_end()
            self.assertTrue(os.path.exists(save_path))

    def test_save_oserror_handled(self):
        """保存时 OSError 应被捕获，不中断。"""
        env = MagicMock()
        cb = _bind_callback(
            RealMachineCallback(env=env, save_path="/nonexistent/path/real.json"),
            n_calls=1,
        )
        cb.real_times = [{"step": 1}]
        # 在 Linux 上 /nonexistent 不可创建；Windows 上可能创建失败
        # 用 mock 确保触发 OSError 分支
        with patch("builtins.open", side_effect=OSError("disk full")):
            cb._on_training_end()  # 不应抛异常

    def test_save_no_directory_in_save_path(self):
        """save_path 无目录部分时应正常保存（os.path.dirname 返回空字符串）。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 切到临时目录，使用纯文件名
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                env = MagicMock()
                cb = _bind_callback(
                    RealMachineCallback(env=env, save_path="real_times.json"),
                    n_calls=1,
                )
                cb.real_times = [{"step": 1}]
                cb._on_training_end()
                self.assertTrue(os.path.exists(os.path.join(tmpdir, "real_times.json")))
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
