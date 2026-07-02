"""
量子RL调度系统 - 后台仿真任务单元测试
Unit Tests for src/visualization/simulator.py

测试覆盖：
- simulate_scheduler：PPO 模型可用 / 不可用 / 推理异常 三条路径
- tick % 20 == 0 真机轮询路径（成功 / 异常）
- 任务状态迁移（pending → completed / running）
- 资源历史 _resource_history 记录与修剪（>100 时 pop(0)）
- 决策日志 _decision_log 记录与修剪（>200 时 pop(0)）
- WebSocket broadcast 调用与消息结构
- asyncio.sleep 被 mock 后通过 CancelledError 退出无限循环

测试风格：unittest.IsolatedAsyncioTestCase + unittest.mock
通过 patch("src.visualization.simulator._app") 替换 app 模块全局状态引用，
避免触碰真实 app 模块的全局可变状态，保证测试间隔离。
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# 预先导入 src.quantum.annealing（间接触发 torch/numpy 加载），
# 避免 src/__init__.py 在 numpy 被 pytest 插件重载后再次导入 torch 时
# 出现 "module functions cannot set METH_CLASS" 错误。
# （与 tests/test_scheduler.py 相同的导入顺序约定）
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.quantum.annealing import QuantumAnnealingOptimizer
from src.visualization.simulator import simulate_scheduler


def _build_mock_app():
    """构造一个 mock _app 对象，预置 simulate_scheduler 所需的全部属性。

    Returns:
        MagicMock：system_status / task_queue / _resource_history / _decision_log
        / manager.broadcast (AsyncMock) / _get_ppo_model / _ppo_env / _ppo_model
        / _get_real_machines_status / _load_real_submissions 均已就绪。
    """
    mock_app = MagicMock()
    mock_app.system_status = {
        "current_step": 0,
        "qubit_utilization": 0.5,
        "queue_length": 0,
        "average_wait_time": 1.0,
        "completed_tasks": 0,
        "last_update": "",
    }
    mock_app.task_queue = [
        {
            "task_id": "QTASK-001",
            "status": "pending",
        },
        {
            "task_id": "QTASK-002",
            "status": "pending",
        },
    ]
    mock_app._resource_history = []
    mock_app._decision_log = []
    mock_app._ppo_model = None
    mock_app._ppo_env = None
    mock_app._get_ppo_model = MagicMock(return_value=None)
    mock_app._get_real_machines_status = MagicMock(return_value=[])
    mock_app._load_real_submissions = MagicMock(return_value=[])
    mock_app.manager = MagicMock()
    mock_app.manager.broadcast = AsyncMock()
    return mock_app


def _make_sleep_raising_after(n: int):
    """构造一个 fake asyncio.sleep：第 n 次调用时抛 CancelledError 以退出无限循环。

    Args:
        n: 第 n 次调用时抛 CancelledError（之前 n-1 次正常返回 None）。

    Returns:
        async 函数，可直接替换 asyncio.sleep。
    """
    counter = {"count": 0}

    async def fake_sleep(_seconds):
        counter["count"] += 1
        if counter["count"] >= n:
            raise asyncio.CancelledError()
        return None

    return fake_sleep


class TestSimulateSchedulerPpoPath(unittest.IsolatedAsyncioTestCase):
    """测试 simulate_scheduler 在 PPO 模型可用时的推理路径。"""

    async def test_ppo_model_available_updates_qubit_utilization(self):
        """PPO 模型可用且推理成功时应根据动作更新 qubit_utilization。"""
        mock_app = _build_mock_app()
        # 构造 mock PPO 模型 — 返回 action=1（量子）
        mock_model = MagicMock()
        mock_model.env = MagicMock()
        mock_model.env.reset.return_value = ([0.1] * 14, {})
        mock_model.predict.return_value = (1, None)
        mock_app._get_ppo_model.return_value = mock_model
        mock_app._ppo_env = MagicMock()  # 非 None
        mock_app._ppo_model = mock_model  # 用于 broadcast 时 ppo_active 判断

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.0),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        # PPO 推理分支：action=1 → target=0.45，更新后利用率应为 0.5*0.7 + 0.45*0.3 = 0.485
        self.assertAlmostEqual(
            mock_app.system_status["qubit_utilization"], 0.485, places=4
        )
        # current_step 应递增
        self.assertEqual(mock_app.system_status["current_step"], 1)
        # 决策日志应记录 PPO 动作
        self.assertEqual(len(mock_app._decision_log), 1)
        entry = mock_app._decision_log[0]
        self.assertEqual(entry["action"], 1)
        self.assertEqual(entry["action_label"], "量子")
        self.assertEqual(entry["source"], "PPO")
        # 资源历史应记录一条
        self.assertEqual(len(mock_app._resource_history), 1)
        # broadcast 应被调用
        mock_app.manager.broadcast.assert_awaited()

    async def test_ppo_action_classical_label(self):
        """PPO action=0 时 action_label 应为 '经典'。"""
        mock_app = _build_mock_app()
        mock_model = MagicMock()
        mock_model.env = MagicMock()
        mock_model.env.reset.return_value = ([0.1] * 14, {})
        mock_model.predict.return_value = (0, None)
        mock_app._get_ppo_model.return_value = mock_model
        mock_app._ppo_env = MagicMock()
        mock_app._ppo_model = mock_model

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.0),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        self.assertEqual(mock_app._decision_log[0]["action_label"], "经典")

    async def test_ppo_action_hybrid_label(self):
        """PPO action=2 时 action_label 应为 '混合'，且 qubit_utilization 走 0.40 分支。"""
        mock_app = _build_mock_app()
        mock_model = MagicMock()
        mock_model.env = MagicMock()
        mock_model.env.reset.return_value = ([0.1] * 14, {})
        mock_model.predict.return_value = (2, None)
        mock_app._get_ppo_model.return_value = mock_model
        mock_app._ppo_env = MagicMock()
        mock_app._ppo_model = mock_model

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.0),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        self.assertEqual(mock_app._decision_log[0]["action_label"], "混合")
        # action=2 → target=0.40，0.5*0.7 + 0.40*0.3 = 0.47
        self.assertAlmostEqual(
            mock_app.system_status["qubit_utilization"], 0.47, places=4
        )

    async def test_ppo_predict_exception_fallback_to_random(self):
        """PPO predict 抛 ValueError 时应回退到随机更新路径。"""
        mock_app = _build_mock_app()
        mock_model = MagicMock()
        mock_model.env = MagicMock()
        mock_model.env.reset.return_value = ([0.1] * 14, {})
        mock_model.predict.side_effect = ValueError("infer fail")
        mock_app._get_ppo_model.return_value = mock_model
        mock_app._ppo_env = MagicMock()
        mock_app._ppo_model = mock_model

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.0),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        # 异常回退随机：utilization = 0.5 + 0.0 = 0.5（clamp 后）
        self.assertAlmostEqual(
            mock_app.system_status["qubit_utilization"], 0.5, places=4
        )
        # 异常路径不应记录决策日志（action 仍为 -1）
        self.assertEqual(len(mock_app._decision_log), 0)

    async def test_ppo_predict_runtime_error_fallback(self):
        """PPO predict 抛 RuntimeError 时也应回退到随机更新路径。"""
        mock_app = _build_mock_app()
        mock_model = MagicMock()
        mock_model.env = MagicMock()
        mock_model.env.reset.return_value = ([0.1] * 14, {})
        mock_model.predict.side_effect = RuntimeError("runtime fail")
        mock_app._get_ppo_model.return_value = mock_model
        mock_app._ppo_env = MagicMock()
        mock_app._ppo_model = mock_model

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.0),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        self.assertEqual(len(mock_app._decision_log), 0)
        mock_app.manager.broadcast.assert_awaited()

    async def test_ppo_predict_oserror_fallback(self):
        """PPO predict 抛 OSError 时也应回退到随机更新路径。"""
        mock_app = _build_mock_app()
        mock_model = MagicMock()
        mock_model.env = MagicMock()
        mock_model.env.reset.return_value = ([0.1] * 14, {})
        mock_model.predict.side_effect = OSError("os err")
        mock_app._get_ppo_model.return_value = mock_model
        mock_app._ppo_env = MagicMock()
        mock_app._ppo_model = mock_model

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.0),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        # OSError 被捕获，回退随机后不应崩溃
        self.assertEqual(mock_app.system_status["current_step"], 1)


class TestSimulateSchedulerNoModel(unittest.IsolatedAsyncioTestCase):
    """测试 simulate_scheduler 在 PPO 模型不可用时的随机路径。"""

    async def test_no_model_uses_random_path(self):
        """PPO 模型为 None 时应走随机更新路径且不记录决策日志。"""
        mock_app = _build_mock_app()
        mock_app._get_ppo_model.return_value = None
        mock_app._ppo_env = None
        mock_app._ppo_model = None

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.0),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        # 随机路径：utilization = 0.5 + 0.0 = 0.5
        self.assertAlmostEqual(
            mock_app.system_status["qubit_utilization"], 0.5, places=4
        )
        # 无 PPO 推理 → 不记录决策日志
        self.assertEqual(len(mock_app._decision_log), 0)

    async def test_no_model_with_env_none_uses_random_path(self):
        """PPO 模型存在但 _ppo_env 为 None 时应走随机路径。"""
        mock_app = _build_mock_app()
        mock_model = MagicMock()
        mock_model.env = MagicMock()
        mock_app._get_ppo_model.return_value = mock_model
        mock_app._ppo_env = None  # env 为 None
        mock_app._ppo_model = mock_model

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.0),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        # 应走随机分支
        self.assertEqual(len(mock_app._decision_log), 0)

    async def test_no_model_with_model_env_none_uses_random_path(self):
        """PPO 模型存在但 model.env 为 None 时应走随机路径。"""
        mock_app = _build_mock_app()
        mock_model = MagicMock()
        mock_model.env = None  # model.env 为 None
        mock_app._get_ppo_model.return_value = mock_model
        mock_app._ppo_env = MagicMock()
        mock_app._ppo_model = mock_model

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.0),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        self.assertEqual(len(mock_app._decision_log), 0)

    async def test_qubit_utilization_clamped_to_min(self):
        """qubit_utilization 偏低时应被 clamp 到 0.1 下限。"""
        mock_app = _build_mock_app()
        mock_app.system_status["qubit_utilization"] = 0.05
        mock_app._get_ppo_model.return_value = None

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            # uniform 返回 -0.5，应让利用率被 clamp 到 0.1
            patch("src.visualization.simulator.random.uniform", lambda a, b: -0.5),
            patch("src.visualization.simulator.random.random", lambda: 0.0),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        self.assertAlmostEqual(
            mock_app.system_status["qubit_utilization"], 0.1, places=4
        )

    async def test_qubit_utilization_clamped_to_max(self):
        """qubit_utilization 偏高时应被 clamp 到 1.0 上限。"""
        mock_app = _build_mock_app()
        mock_app.system_status["qubit_utilization"] = 0.99
        mock_app._get_ppo_model.return_value = None

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.5),
            patch("src.visualization.simulator.random.random", lambda: 0.0),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        self.assertAlmostEqual(
            mock_app.system_status["qubit_utilization"], 1.0, places=4
        )

    async def test_average_wait_time_clamped_to_min(self):
        """average_wait_time 应被 clamp 到 0.5 下限。"""
        mock_app = _build_mock_app()
        mock_app.system_status["average_wait_time"] = 0.4
        mock_app._get_ppo_model.return_value = None

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch(
                "src.visualization.simulator.random.uniform", lambda a, b: -1.0
            ),  # wait_time 减少
            patch("src.visualization.simulator.random.random", lambda: 0.0),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        self.assertGreaterEqual(mock_app.system_status["average_wait_time"], 0.5)


class TestSimulateSchedulerTaskMigration(unittest.IsolatedAsyncioTestCase):
    """测试 simulate_scheduler 的任务状态迁移分支。"""

    async def test_task_completed_when_random_below_threshold(self):
        """random.random < 0.35 时应将 pending 任务置为 completed。"""
        mock_app = _build_mock_app()
        mock_app._get_ppo_model.return_value = None
        # 第一次 random.random 控制 PPO 路径（无 PPO 时仅在任务状态分支使用）
        # 实际：random.random 第一次用于任务完成判断 (<0.35 触发)，第二次用于运行判断 (<0.25 触发)
        # 这里让第一次返回 0.0（<0.35 触发完成），第二次返回 1.0（不触发 running）
        random_returns = iter([0.0, 1.0])

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch(
                "src.visualization.simulator.random.random",
                lambda: next(random_returns),
            ),
            patch("src.visualization.simulator.random.choice", lambda seq: seq[0]),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        completed = [t for t in mock_app.task_queue if t["status"] == "completed"]
        self.assertGreaterEqual(len(completed), 1)
        self.assertEqual(mock_app.system_status["completed_tasks"], 1)
        # queue_length 应相应减 1
        self.assertEqual(mock_app.system_status["queue_length"], 1)

    async def test_task_running_when_random_below_threshold(self):
        """random.random < 0.25 时应将 pending 任务置为 running（不增加 completed）。"""
        mock_app = _build_mock_app()
        mock_app._get_ppo_model.return_value = None
        # 第一次 1.0（不触发 completed），第二次 0.0（触发 running）
        random_returns = iter([1.0, 0.0])

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch(
                "src.visualization.simulator.random.random",
                lambda: next(random_returns),
            ),
            patch("src.visualization.simulator.random.choice", lambda seq: seq[0]),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        running = [t for t in mock_app.task_queue if t["status"] == "running"]
        self.assertGreaterEqual(len(running), 1)
        # completed_tasks 不应增加
        self.assertEqual(mock_app.system_status["completed_tasks"], 0)

    async def test_no_task_migration_when_random_above_thresholds(self):
        """random.random 均高于阈值时任务状态不应改变。"""
        mock_app = _build_mock_app()
        mock_app._get_ppo_model.return_value = None

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.99),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        pending = [t for t in mock_app.task_queue if t["status"] == "pending"]
        self.assertEqual(len(pending), 2)
        self.assertEqual(mock_app.system_status["completed_tasks"], 0)

    async def test_empty_task_queue_does_not_raise(self):
        """任务队列为空时 simulate_scheduler 不应抛异常。"""
        mock_app = _build_mock_app()
        mock_app.task_queue = []
        mock_app._get_ppo_model.return_value = None

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.0),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        # queue_length 应为 0
        self.assertEqual(mock_app.system_status["queue_length"], 0)


class TestSimulateSchedulerRealMachinePoll(unittest.IsolatedAsyncioTestCase):
    """测试 simulate_scheduler 在 tick % 20 == 0 时的真机轮询分支。"""

    async def test_real_machine_poll_success(self):
        """tick=20 时应调用 _get_real_machines_status 并更新 real_machines 字段。"""
        mock_app = _build_mock_app()
        mock_app._get_ppo_model.return_value = None
        machines = [
            {"id": "1", "type": "sc", "status": "running", "name": "tianyan_s"}
        ]
        mock_app._get_real_machines_status = MagicMock(return_value=machines)
        mock_app._load_real_submissions = MagicMock(
            return_value=[{"step": 1, "task_id": "t1"}]
        )

        # 让循环跑 21 次后退出：第 21 次 sleep 抛 CancelledError
        # tick 从 1 递增到 20，第 20 次迭代时 tick%20==0 触发轮询
        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(21),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.99),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        # 真机状态应被写入 system_status
        self.assertEqual(
            mock_app.system_status["real_machines"], machines
        )
        # real_submissions 应被写入
        self.assertEqual(
            mock_app.system_status["real_submissions"],
            [{"step": 1, "task_id": "t1"}],
        )
        mock_app._get_real_machines_status.assert_called()
        mock_app._load_real_submissions.assert_called()

    async def test_real_machine_poll_empty_does_not_update(self):
        """tick=20 但真机列表为空时不应更新 system_status['real_machines']。"""
        mock_app = _build_mock_app()
        mock_app._get_ppo_model.return_value = None
        mock_app._get_real_machines_status = MagicMock(return_value=[])

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(21),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.99),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        # 真机列表为空 → real_machines 字段不应被设置
        self.assertNotIn("real_machines", mock_app.system_status)

    async def test_real_machine_poll_oserror_handled(self):
        """tick=20 时 _get_real_machines_status 抛 OSError 应被捕获不崩溃。"""
        mock_app = _build_mock_app()
        mock_app._get_ppo_model.return_value = None
        mock_app._get_real_machines_status = MagicMock(
            side_effect=OSError("net down")
        )
        mock_app._load_real_submissions = MagicMock(return_value=[])

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(21),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.99),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        # 异常被捕获，real_machines 不应被设置
        self.assertNotIn("real_machines", mock_app.system_status)

    async def test_real_machine_poll_runtime_error_handled(self):
        """tick=20 时 _get_real_machines_status 抛 RuntimeError 应被捕获。"""
        mock_app = _build_mock_app()
        mock_app._get_ppo_model.return_value = None
        mock_app._get_real_machines_status = MagicMock(
            side_effect=RuntimeError("rt fail")
        )

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(21),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.99),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        self.assertNotIn("real_machines", mock_app.system_status)

    async def test_real_machine_poll_value_error_handled(self):
        """tick=20 时 _get_real_machines_status 抛 ValueError 应被捕获。"""
        mock_app = _build_mock_app()
        mock_app._get_ppo_model.return_value = None
        mock_app._get_real_machines_status = MagicMock(
            side_effect=ValueError("bad value")
        )

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(21),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.99),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        self.assertNotIn("real_machines", mock_app.system_status)

    async def test_real_submissions_load_oserror_handled(self):
        """tick=20 时 _load_real_submissions 抛 OSError 应被捕获不崩溃。"""
        mock_app = _build_mock_app()
        mock_app._get_ppo_model.return_value = None
        mock_app._get_real_machines_status = MagicMock(return_value=[])
        mock_app._load_real_submissions = MagicMock(side_effect=OSError("io err"))

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(21),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.99),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        # 异常被捕获，real_submissions 不应被设置
        self.assertNotIn("real_submissions", mock_app.system_status)

    async def test_real_submissions_load_value_error_handled(self):
        """tick=20 时 _load_real_submissions 抛 ValueError 应被捕获。"""
        mock_app = _build_mock_app()
        mock_app._get_ppo_model.return_value = None
        mock_app._get_real_machines_status = MagicMock(return_value=[])
        mock_app._load_real_submissions = MagicMock(
            side_effect=ValueError("bad json")
        )

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(21),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.99),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        self.assertNotIn("real_submissions", mock_app.system_status)

    async def test_real_submissions_load_runtime_error_handled(self):
        """tick=20 时 _load_real_submissions 抛 RuntimeError 应被捕获。"""
        mock_app = _build_mock_app()
        mock_app._get_ppo_model.return_value = None
        mock_app._get_real_machines_status = MagicMock(return_value=[])
        mock_app._load_real_submissions = MagicMock(
            side_effect=RuntimeError("rt fail")
        )

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(21),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.99),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        self.assertNotIn("real_submissions", mock_app.system_status)


class TestSimulateSchedulerHistoryAndLog(unittest.IsolatedAsyncioTestCase):
    """测试 _resource_history 与 _decision_log 的记录与修剪逻辑。"""

    async def test_resource_history_recorded_each_iteration(self):
        """每次迭代都应向 _resource_history 追加一条记录。"""
        mock_app = _build_mock_app()
        mock_app._get_ppo_model.return_value = None

        # 跑 5 次迭代后退出
        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(6),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.99),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        self.assertEqual(len(mock_app._resource_history), 5)
        # 每条记录应包含必要字段
        entry = mock_app._resource_history[0]
        self.assertIn("step", entry)
        self.assertIn("qubit_utilization", entry)
        self.assertIn("queue_length", entry)
        self.assertIn("completed_tasks", entry)
        self.assertIn("average_wait_time", entry)
        # 步数应从 1 递增到 5
        steps = [r["step"] for r in mock_app._resource_history]
        self.assertEqual(steps, [1, 2, 3, 4, 5])

    async def test_resource_history_trimmed_to_100(self):
        """_resource_history 超过 100 条时应 pop(0) 保留最近 100 条。"""
        mock_app = _build_mock_app()
        mock_app._get_ppo_model.return_value = None
        # 预填充 99 条，再跑 3 次迭代 → 第 101 条触发 pop(0)
        mock_app._resource_history = [
            {"step": i, "qubit_utilization": 0.5, "queue_length": 0,
             "completed_tasks": 0, "average_wait_time": 1.0}
            for i in range(99)
        ]

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(4),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.99),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        # 99 + 3 = 102 → trim 到 100
        self.assertEqual(len(mock_app._resource_history), 100)

    async def test_decision_log_trimmed_to_200(self):
        """_decision_log 超过 200 条时应 pop(0) 保留最近 200 条。"""
        mock_app = _build_mock_app()
        mock_model = MagicMock()
        mock_model.env = MagicMock()
        mock_model.env.reset.return_value = ([0.1] * 14, {})
        mock_model.predict.return_value = (1, None)
        mock_app._get_ppo_model.return_value = mock_model
        mock_app._ppo_env = MagicMock()
        mock_app._ppo_model = mock_model
        # 预填充 199 条决策日志
        mock_app._decision_log = [
            {
                "step": i,
                "task_id": f"task_{i}",
                "action": 1,
                "action_label": "量子",
                "reward": 5.0,
                "source": "PPO",
            }
            for i in range(199)
        ]

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(4),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.99),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        # 199 + 3 = 202 → trim 到 200
        self.assertEqual(len(mock_app._decision_log), 200)
        # 最新一条应是本次 PPO 推理记录
        last = mock_app._decision_log[-1]
        self.assertEqual(last["action"], 1)
        self.assertEqual(last["source"], "PPO")

    async def test_decision_log_record_structure(self):
        """PPO 推理路径下决策日志记录结构应完整。"""
        mock_app = _build_mock_app()
        mock_model = MagicMock()
        mock_model.env = MagicMock()
        mock_model.env.reset.return_value = ([0.1] * 14, {})
        mock_model.predict.return_value = (1, None)
        mock_app._get_ppo_model.return_value = mock_model
        mock_app._ppo_env = MagicMock()
        mock_app._ppo_model = mock_model

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.99),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        self.assertEqual(len(mock_app._decision_log), 1)
        entry = mock_app._decision_log[0]
        self.assertIn("step", entry)
        self.assertIn("task_id", entry)
        self.assertIn("action", entry)
        self.assertIn("action_label", entry)
        self.assertIn("reward", entry)
        self.assertIn("source", entry)
        self.assertEqual(entry["task_id"], "task_1")
        # reward 应为 qubit_utilization * 10
        expected_reward = round(mock_app.system_status["qubit_utilization"] * 10, 2)
        self.assertEqual(entry["reward"], expected_reward)


class TestSimulateSchedulerBroadcast(unittest.IsolatedAsyncioTestCase):
    """测试 simulate_scheduler 的 WebSocket broadcast 调用与消息结构。"""

    async def test_broadcast_called_with_correct_structure(self):
        """每次迭代都应调用 manager.broadcast，且消息结构包含 type/status/tasks/ppo_active。"""
        mock_app = _build_mock_app()
        mock_app._get_ppo_model.return_value = None

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.99),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        mock_app.manager.broadcast.assert_awaited_once()
        call_args = mock_app.manager.broadcast.await_args
        msg = call_args[0][0]
        self.assertEqual(msg["type"], "status_update")
        self.assertIn("status", msg)
        self.assertIn("tasks", msg)
        self.assertIn("ppo_active", msg)
        # 无 PPO 模型时 ppo_active 应为 False
        self.assertFalse(msg["ppo_active"])
        # status 应就是 system_status 引用
        self.assertIs(msg["status"], mock_app.system_status)
        # tasks 应就是 task_queue 引用
        self.assertIs(msg["tasks"], mock_app.task_queue)

    async def test_broadcast_ppo_active_true_when_model_loaded(self):
        """PPO 模型已加载时 broadcast 消息 ppo_active 应为 True。"""
        mock_app = _build_mock_app()
        mock_model = MagicMock()
        mock_model.env = MagicMock()
        mock_model.env.reset.return_value = ([0.1] * 14, {})
        mock_model.predict.return_value = (0, None)
        mock_app._get_ppo_model.return_value = mock_model
        mock_app._ppo_env = MagicMock()
        mock_app._ppo_model = mock_model  # 非 None

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.99),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        msg = mock_app.manager.broadcast.await_args[0][0]
        self.assertTrue(msg["ppo_active"])

    async def test_broadcast_called_each_iteration(self):
        """多次迭代应多次调用 broadcast。"""
        mock_app = _build_mock_app()
        mock_app._get_ppo_model.return_value = None

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(4),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.99),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        # 3 次迭代 → 3 次 broadcast
        self.assertEqual(mock_app.manager.broadcast.await_count, 3)

    async def test_last_update_isoformat_string_set(self):
        """每次迭代应将 last_update 设置为 ISO 格式时间字符串。"""
        mock_app = _build_mock_app()
        mock_app._get_ppo_model.return_value = None

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.99),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        last_update = mock_app.system_status["last_update"]
        self.assertIsInstance(last_update, str)
        self.assertGreater(len(last_update), 0)


class TestSimulateSchedulerQueueLength(unittest.IsolatedAsyncioTestCase):
    """测试 simulate_scheduler 中 queue_length 的计算逻辑。"""

    async def test_queue_length_counts_pending_only(self):
        """queue_length 应只统计 status=pending 的任务。"""
        mock_app = _build_mock_app()
        mock_app.task_queue = [
            {"task_id": "T1", "status": "pending"},
            {"task_id": "T2", "status": "running"},
            {"task_id": "T3", "status": "completed"},
            {"task_id": "T4", "status": "pending"},
        ]
        mock_app._get_ppo_model.return_value = None

        with (
            patch("src.visualization.simulator._app", mock_app),
            patch(
                "src.visualization.simulator.asyncio.sleep",
                new=_make_sleep_raising_after(2),
            ),
            patch("src.visualization.simulator.random.uniform", lambda a, b: 0.0),
            patch("src.visualization.simulator.random.random", lambda: 0.99),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await simulate_scheduler()

        # 2 个 pending
        self.assertEqual(mock_app.system_status["queue_length"], 2)


if __name__ == "__main__":
    unittest.main()
