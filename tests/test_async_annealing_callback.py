"""
异步量子退火训练回调单元测试
Unit Tests for src/scheduler/async_annealing_callback.py

测试覆盖：
- AsyncAnnealingCallback 初始化与 _init_callback 启动工作线程（含 verbose 日志分支）
- _on_step 自适应触发间隔：
    * _next_trigger_step 为 None 时从 loop 获取间隔
    * 未达阈值不触发提交
    * 达到阈值提交退火任务（成功/队列满）
    * 深拷贝策略网络失败时的异常处理
    * verbose 日志分支
- _on_rollout_start 权重回写：
    * 无待回写结果时直接返回
    * 正常回写已完成的优化权重
    * load_state_dict 异常处理
    * verbose 日志分支
- _on_training_end 关闭工作线程（含 verbose 日志分支）
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.async_annealing_callback import AsyncAnnealingCallback


# ============================================================
# 辅助函数：构造 mock loop 与注入 BaseCallback 运行时属性
# ============================================================
def _make_mock_loop(interval=100, submit_return=True, pending_result=None):
    """构造一个 mock 的 AsyncAnnealingLoop，避免真实退火计算。

    Args:
        interval        : get_current_interval 返回值，默认 100
        submit_return   : submit 返回值，True=成功，False=队列满
        pending_result  : get_pending_result 返回值，None 表示无待回写结果

    Returns:
        配置好的 MagicMock loop 对象
    """
    mock_loop = MagicMock()
    mock_loop.get_current_interval.return_value = int(interval)
    mock_loop.submit.return_value = bool(submit_return)
    mock_loop.get_pending_result.return_value = pending_result
    return mock_loop


def _bind_callback(cb, n_calls=0, model=None):
    """为 BaseCallback 子类注入 n_calls / model 等运行时属性，便于直接调用 _on_step。

    SB3 的 BaseCallback 在训练过程中由外层模型设置这些属性，
    单元测试中需要手动注入。

    Args:
        cb       : BaseCallback 子类实例
        n_calls  : 当前调用次数，默认 0
        model    : mock 的 SB3 模型，默认为 MagicMock

    Returns:
        注入属性后的回调实例
    """
    cb.n_calls = int(n_calls)
    cb.model = model if model is not None else MagicMock()
    return cb


# ============================================================
# AsyncAnnealingCallback 初始化与 _init_callback 测试
# ============================================================
class TestAsyncAnnealingCallbackInit(unittest.TestCase):
    """测试 AsyncAnnealingCallback 初始化与 _init_callback 行为。"""

    def test_init_stores_loop_and_default_trigger_step(self):
        """__init__ 应保存 loop 引用且 _next_trigger_step 初始为 None。"""
        mock_loop = _make_mock_loop()
        cb = AsyncAnnealingCallback(loop=mock_loop, verbose=0)
        self.assertIs(cb.loop, mock_loop)
        self.assertIsNone(cb._next_trigger_step)
        self.assertEqual(cb.verbose, 0)

    def test_init_callback_starts_loop_and_sets_trigger_step(self):
        """_init_callback 应调用 loop.start() 并设置首次触发步数。"""
        mock_loop = _make_mock_loop(interval=100)
        cb = AsyncAnnealingCallback(loop=mock_loop, verbose=0)
        cb._init_callback()
        mock_loop.start.assert_called_once()
        self.assertEqual(cb._next_trigger_step, 100)

    def test_init_callback_verbose_logs(self):
        """verbose=1 时 _init_callback 应记录启动日志（覆盖第 53 行）。"""
        mock_loop = _make_mock_loop(interval=200)
        cb = AsyncAnnealingCallback(loop=mock_loop, verbose=1)
        with patch("src.scheduler.async_annealing_callback.logger") as mock_logger:
            cb._init_callback()
        mock_logger.info.assert_called_once()
        # 日志内容应包含首次触发步数
        args, _ = mock_logger.info.call_args
        self.assertIn("200", args[0])


# ============================================================
# AsyncAnnealingCallback._on_step 测试
# ============================================================
class TestAsyncAnnealingCallbackOnStep(unittest.TestCase):
    """测试 AsyncAnnealingCallback._on_step 的触发与异常处理。"""

    def test_on_step_sets_trigger_when_none(self):
        """_next_trigger_step 为 None 时应从 loop 获取间隔（覆盖第 65 行）。"""
        mock_loop = _make_mock_loop(interval=100)
        cb = _bind_callback(
            AsyncAnnealingCallback(loop=mock_loop, verbose=0),
            n_calls=0,
        )
        # 未调用 _init_callback，_next_trigger_step 仍为 None
        self.assertIsNone(cb._next_trigger_step)
        result = cb._on_step()
        # 应设置触发步数为 100
        self.assertEqual(cb._next_trigger_step, 100)
        # n_calls=0 < 100，不应提交
        mock_loop.submit.assert_not_called()
        self.assertTrue(result)

    def test_on_step_no_trigger_when_below_threshold(self):
        """n_calls 未达触发步数时不应提交退火任务。"""
        mock_loop = _make_mock_loop(interval=100)
        cb = _bind_callback(
            AsyncAnnealingCallback(loop=mock_loop, verbose=0),
            n_calls=50,
        )
        cb._next_trigger_step = 100
        result = cb._on_step()
        self.assertTrue(result)
        mock_loop.submit.assert_not_called()

    def test_on_step_submits_when_reached_threshold(self):
        """n_calls 达到触发步数时应提交退火任务并更新触发步数。"""
        mock_loop = _make_mock_loop(interval=100, submit_return=True)
        cb = _bind_callback(
            AsyncAnnealingCallback(loop=mock_loop, verbose=0),
            n_calls=100,
        )
        cb._next_trigger_step = 100
        cb._on_step()
        mock_loop.submit.assert_called_once()
        # 验证提交参数：第一个是 policy 快照，第二个是步数
        args, _ = mock_loop.submit.call_args
        self.assertEqual(args[1], 100)
        # 触发步数应更新为 100 + 100 = 200
        self.assertEqual(cb._next_trigger_step, 200)

    def test_on_step_submit_success_verbose_logs(self):
        """verbose=1 且提交成功时应记录 info 日志（覆盖第 87-90 行）。"""
        mock_loop = _make_mock_loop(interval=100, submit_return=True)
        cb = _bind_callback(
            AsyncAnnealingCallback(loop=mock_loop, verbose=1),
            n_calls=100,
        )
        cb._next_trigger_step = 100
        with patch("src.scheduler.async_annealing_callback.logger") as mock_logger:
            cb._on_step()
        mock_logger.info.assert_called_once()
        # 触发步数应更新
        self.assertEqual(cb._next_trigger_step, 200)

    def test_on_step_submit_queue_full_updates_trigger(self):
        """队列满（submit 返回 False）时应更新触发步数以便稍后重试（覆盖第 91-93 行）。"""
        mock_loop = _make_mock_loop(interval=100, submit_return=False)
        cb = _bind_callback(
            AsyncAnnealingCallback(loop=mock_loop, verbose=0),
            n_calls=100,
        )
        cb._next_trigger_step = 100
        result = cb._on_step()
        self.assertTrue(result)
        mock_loop.submit.assert_called_once()
        # 即使提交失败，触发步数也应更新为 100 + 100 = 200
        self.assertEqual(cb._next_trigger_step, 200)

    def test_on_step_deepcopy_failure_handled(self):
        """深拷贝策略网络失败时应捕获异常、不调用 submit 并更新触发步数（覆盖第 72-80 行）。"""
        mock_loop = _make_mock_loop(interval=100)
        cb = _bind_callback(
            AsyncAnnealingCallback(loop=mock_loop, verbose=0),
            n_calls=100,
        )
        cb._next_trigger_step = 100
        with patch(
            "src.scheduler.async_annealing_callback.copy.deepcopy",
            side_effect=RuntimeError("deepcopy fail"),
        ):
            result = cb._on_step()
        # 应返回 True，不中断训练
        self.assertTrue(result)
        # 深拷贝失败后不应调用 submit
        mock_loop.submit.assert_not_called()
        # 触发步数应更新为 100 + 100 = 200
        self.assertEqual(cb._next_trigger_step, 200)

    def test_on_step_deepcopy_failure_logs_error(self):
        """深拷贝失败时应记录 error 日志（覆盖第 75-78 行）。"""
        mock_loop = _make_mock_loop(interval=100)
        cb = _bind_callback(
            AsyncAnnealingCallback(loop=mock_loop, verbose=1),
            n_calls=100,
        )
        cb._next_trigger_step = 100
        with patch(
            "src.scheduler.async_annealing_callback.copy.deepcopy",
            side_effect=TypeError("type error"),
        ), patch("src.scheduler.async_annealing_callback.logger") as mock_logger:
            cb._on_step()
        mock_logger.error.assert_called_once()
        # 日志应包含异常类型名
        args, _ = mock_logger.error.call_args
        self.assertIn("TypeError", args[0])

    def test_on_step_returns_true_always(self):
        """_on_step 应始终返回 True 以继续训练。"""
        mock_loop = _make_mock_loop(interval=100)
        cb = _bind_callback(
            AsyncAnnealingCallback(loop=mock_loop, verbose=0),
            n_calls=10,
        )
        cb._next_trigger_step = 100
        self.assertTrue(cb._on_step())


# ============================================================
# AsyncAnnealingCallback._on_rollout_start 测试
# ============================================================
class TestAsyncAnnealingCallbackRolloutStart(unittest.TestCase):
    """测试 AsyncAnnealingCallback._on_rollout_start 的权重回写行为。"""

    def test_on_rollout_start_no_pending_result(self):
        """无待回写结果时应直接返回（覆盖第 105 行）。"""
        mock_loop = _make_mock_loop(pending_result=None)
        cb = _bind_callback(
            AsyncAnnealingCallback(loop=mock_loop, verbose=0),
        )
        # 不应抛异常
        cb._on_rollout_start()
        mock_loop.get_pending_result.assert_called_once()

    def test_on_rollout_start_writes_back_weights(self):
        """有待回写结果时应调用 load_state_dict 回写权重。"""
        pending = {
            "state_dict": {"weight": 1.0},
            "step": 100,
            "delta": 0.5,
        }
        mock_loop = _make_mock_loop(pending_result=pending)
        mock_model = MagicMock()
        cb = _bind_callback(
            AsyncAnnealingCallback(loop=mock_loop, verbose=0),
            model=mock_model,
        )
        cb._on_rollout_start()
        mock_model.policy.load_state_dict.assert_called_once_with(
            {"weight": 1.0}, strict=False
        )

    def test_on_rollout_start_verbose_logs(self):
        """verbose=1 且回写成功时应记录 info 日志（覆盖第 114-117 行）。"""
        pending = {
            "state_dict": {"weight": 1.0},
            "step": 100,
            "delta": 0.5,
        }
        mock_loop = _make_mock_loop(pending_result=pending)
        cb = _bind_callback(
            AsyncAnnealingCallback(loop=mock_loop, verbose=1),
        )
        with patch("src.scheduler.async_annealing_callback.logger") as mock_logger:
            cb._on_rollout_start()
        mock_logger.info.assert_called_once()
        # 日志应包含 step 与 delta
        args, _ = mock_logger.info.call_args
        self.assertIn("100", args[0])

    def test_on_rollout_start_load_state_dict_failure_handled(self):
        """load_state_dict 抛异常时应捕获并记录 error，不中断训练（覆盖第 118-124 行）。"""
        pending = {
            "state_dict": {"weight": 1.0},
            "step": 100,
            "delta": 0.5,
        }
        mock_loop = _make_mock_loop(pending_result=pending)
        mock_model = MagicMock()
        mock_model.policy.load_state_dict.side_effect = RuntimeError("shape mismatch")
        cb = _bind_callback(
            AsyncAnnealingCallback(loop=mock_loop, verbose=1),
            model=mock_model,
        )
        with patch("src.scheduler.async_annealing_callback.logger") as mock_logger:
            # 不应抛异常
            cb._on_rollout_start()
        mock_logger.error.assert_called_once()
        # 日志应包含异常类型名
        args, _ = mock_logger.error.call_args
        self.assertIn("RuntimeError", args[0])


# ============================================================
# AsyncAnnealingCallback._on_training_end 测试
# ============================================================
class TestAsyncAnnealingCallbackTrainingEnd(unittest.TestCase):
    """测试 AsyncAnnealingCallback._on_training_end 的关闭行为。"""

    def test_on_training_end_calls_shutdown(self):
        """_on_training_end 应调用 loop.shutdown(wait=True)（覆盖第 128 行）。"""
        mock_loop = _make_mock_loop()
        cb = AsyncAnnealingCallback(loop=mock_loop, verbose=0)
        cb._on_training_end()
        mock_loop.shutdown.assert_called_once_with(wait=True)

    def test_on_training_end_verbose_logs(self):
        """verbose=1 时 _on_training_end 应记录关闭日志（覆盖第 129-130 行）。"""
        mock_loop = _make_mock_loop()
        cb = AsyncAnnealingCallback(loop=mock_loop, verbose=1)
        with patch("src.scheduler.async_annealing_callback.logger") as mock_logger:
            cb._on_training_end()
        mock_loop.shutdown.assert_called_once_with(wait=True)
        mock_logger.info.assert_called_once()


if __name__ == "__main__":
    unittest.main()
