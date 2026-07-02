"""
量子RL调度系统 - 环境渲染模块单元测试
Unit Tests for src/scheduler/env_render.py

测试覆盖：
- render_env：render_mode=None / "human" / "ansi" 三种模式
- 多机器明细渲染（machine_configs >= 2）
- 当前任务信息（有/无 _current_task）
- 最近日志渲染（_render_log）
- 本步路由（_last_selected_machine）
- close_env：清空 _task_queue / _current_task / _render_log
- 单机模式不渲染机器明细（分支覆盖）

测试风格：unittest.TestCase + 真实 QuantumSchedulingEnv 实例（不 mock 内部状态）
"""

import os
import sys
import unittest

# 预先导入 src.quantum.annealing（间接触发 torch/numpy 加载），
# 避免 src/__init__.py 在 numpy 被 pytest 插件重载后再次导入 torch 时
# 出现 "module functions cannot set METH_CLASS" 错误。
# （与 tests/test_scheduler.py 相同的导入顺序约定）
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.quantum.annealing import QuantumAnnealingOptimizer
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv, Task
from src.scheduler.env_render import close_env, render_env


class TestRenderEnvModes(unittest.TestCase):
    """测试 render_env 在不同 render_mode 下的行为。"""

    def test_render_mode_none_returns_none(self):
        """render_mode=None 时 render_env 应返回 None，不执行渲染。"""
        env = QuantumSchedulingEnv(max_steps=50, render_mode=None)
        env.reset(seed=42)
        self.assertIsNone(env.render_mode)
        self.assertIsNone(render_env(env))
        env.close()

    def test_render_mode_ansi_returns_nonempty_string(self):
        """render_mode="ansi" 时 render_env 应返回非空格式化字符串。"""
        env = QuantumSchedulingEnv(max_steps=50, render_mode="ansi")
        env.reset(seed=42)
        output = render_env(env)
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)
        # 关键字段应出现在输出中
        self.assertIn("量子任务调度环境", output)
        self.assertIn("步骤:", output)
        self.assertIn("累计奖励:", output)
        self.assertIn("可用比率:", output)
        self.assertIn("保真度:", output)
        env.close()

    def test_render_mode_human_returns_none_and_does_not_raise(self):
        """render_mode="human" 时 render_env 通过 logger 打印，返回 None 且不抛异常。"""
        env = QuantumSchedulingEnv(max_steps=50, render_mode="human")
        env.reset(seed=42)
        # 不应抛出异常
        result = render_env(env)
        self.assertIsNone(result)
        env.close()

    def test_render_output_contains_statistics_fields(self):
        """ansi 输出应包含统计字段（量子/经典/混合成功、不兼容计数）。"""
        env = QuantumSchedulingEnv(max_steps=50, render_mode="ansi")
        env.reset(seed=42)
        # 执行若干 step 以触发统计字段更新
        for _ in range(5):
            env.step(env.action_space.sample())
        output = render_env(env)
        self.assertIn("量子成功:", output)
        self.assertIn("经典成功:", output)
        self.assertIn("混合成功:", output)
        self.assertIn("不兼容:", output)
        self.assertIn("已调度:", output)
        env.close()

    def test_render_output_contains_resource_aggregate(self):
        """ansi 输出应包含量子资源聚合视图与经典资源视图。"""
        env = QuantumSchedulingEnv(max_steps=50, render_mode="ansi")
        env.reset(seed=42)
        output = render_env(env)
        self.assertIn("[量子资源(聚合)]", output)
        self.assertIn("[经典资源]", output)
        self.assertIn("[任务队列]", output)
        env.close()

    def test_render_step_and_episode_reward_reflected(self):
        """渲染输出应反映当前步数与累计奖励。"""
        env = QuantumSchedulingEnv(max_steps=30, render_mode="ansi")
        env.reset(seed=7)
        env.step(0)  # 走一步
        step_after = env._current_step
        reward_after = env._episode_reward
        output = render_env(env)
        self.assertIn(f"步骤: {step_after}/30", output)
        self.assertIn(f"累计奖励: {reward_after:.2f}", output)
        env.close()


class TestRenderEnvMultiMachine(unittest.TestCase):
    """测试 render_env 在多机器模式下的渲染行为。"""

    def test_multi_machine_render_includes_machine_detail(self):
        """多机器（>1）模式下渲染输出应包含 [量子机器明细] 段。"""
        env = QuantumSchedulingEnv(
            max_steps=50,
            render_mode="ansi",
            machine_configs=DEFAULT_MACHINE_CONFIGS,
        )
        env.reset(seed=42)
        output = render_env(env)
        self.assertIn("[量子机器明细]", output)
        # 每台机器名都应出现
        for name in env.machine_names:
            self.assertIn(name, output)
        env.close()

    def test_single_machine_render_excludes_machine_detail(self):
        """单机模式下渲染输出不应包含 [量子机器明细] 段。"""
        env = QuantumSchedulingEnv(
            max_steps=50,
            render_mode="ansi",
            machine_configs=None,  # 单机模式
        )
        env.reset(seed=42)
        output = render_env(env)
        self.assertNotIn("[量子机器明细]", output)
        env.close()

    def test_multi_machine_render_shows_status_and_real_tag(self):
        """多机器渲染应显示在线/维护状态及真机/仿真标签。"""
        configs = [
            {
                "name": "machine_a",
                "total_qubits": 100,
                "supported_gates": ("H", "CZ", "M"),
                "is_real": False,
            },
            {
                "name": "machine_b",
                "total_qubits": 50,
                "supported_gates": ("H", "CZ", "M"),
                "is_real": True,
            },
        ]
        env = QuantumSchedulingEnv(
            max_steps=30,
            render_mode="ansi",
            machine_configs=configs,
        )
        env.reset(seed=42)
        # 将 machine_b 置为维护状态以覆盖 "维护" 分支
        env._machines[1].available = False
        output = render_env(env)
        self.assertIn("在线", output)
        self.assertIn("维护", output)
        self.assertIn("真机", output)
        self.assertIn("仿真", output)
        env.close()

    def test_multi_machine_render_shows_last_selected_machine(self):
        """存在 _last_selected_machine 时渲染应包含 [本步路由] 段。"""
        env = QuantumSchedulingEnv(
            max_steps=50,
            render_mode="ansi",
            machine_configs=DEFAULT_MACHINE_CONFIGS,
        )
        env.reset(seed=42)
        # 模拟路由选择 — 直接设置字段以覆盖渲染分支
        env._last_selected_machine = env._machines[0].name
        output = render_env(env)
        self.assertIn("[本步路由]", output)
        self.assertIn(env._machines[0].name, output)
        env.close()

    def test_multi_machine_render_without_last_selected_machine(self):
        """多机器模式下 _last_selected_machine=None 时不应渲染 [本步路由]。"""
        env = QuantumSchedulingEnv(
            max_steps=50,
            render_mode="ansi",
            machine_configs=DEFAULT_MACHINE_CONFIGS,
        )
        env.reset(seed=42)
        env._last_selected_machine = None
        output = render_env(env)
        self.assertNotIn("[本步路由]", output)
        env.close()


class TestRenderEnvCurrentTask(unittest.TestCase):
    """测试 render_env 在有/无当前任务时的渲染分支。"""

    def test_render_with_current_task(self):
        """_current_task 不为 None 时渲染应包含任务信息。"""
        env = QuantumSchedulingEnv(max_steps=50, render_mode="ansi")
        env.reset(seed=42)
        # reset 后通常有 _current_task
        if env._current_task is None:
            # 手动塞一个任务以覆盖该分支
            env._current_task = Task(
                task_id="QT-TEST",
                task_type="quantum",
                qubit_count=4,
                wait_steps=2,
                urgency=0.7,
                priority=4,
            )
        task = env._current_task
        output = render_env(env)
        self.assertIn("[当前任务]", output)
        self.assertIn(f"ID={task.task_id}", output)
        self.assertIn(f"类型={task.task_type}", output)
        self.assertIn(f"紧急={task.urgency:.2f}", output)
        self.assertIn(f"等待={task.wait_steps}步", output)
        env.close()

    def test_render_without_current_task(self):
        """_current_task 为 None 时渲染应显示 [当前任务] 无。"""
        env = QuantumSchedulingEnv(max_steps=50, render_mode="ansi")
        env.reset(seed=42)
        env._current_task = None
        output = render_env(env)
        self.assertIn("[当前任务] 无", output)
        env.close()


class TestRenderEnvRenderLog(unittest.TestCase):
    """测试 render_env 在有/无 _render_log 时的渲染分支。"""

    def test_render_with_render_log(self):
        """存在 _render_log 时渲染应包含 最近日志 段。"""
        env = QuantumSchedulingEnv(max_steps=50, render_mode="ansi")
        env.reset(seed=42)
        # 触发若干步以产生日志
        for _ in range(3):
            env.step(env.action_space.sample())
        # 保证至少有一条日志
        if not env._render_log:
            env._render_log.append("[测试] 模拟日志条目")
        output = render_env(env)
        self.assertIn("最近日志:", output)
        # 最近 5 条中至少有一条出现
        for log in env._render_log[-5:]:
            self.assertIn(log, output)
        env.close()

    def test_render_without_render_log(self):
        """_render_log 为空时渲染不应包含 最近日志 段。"""
        env = QuantumSchedulingEnv(max_steps=50, render_mode="ansi")
        env.reset(seed=42)
        env._render_log.clear()
        output = render_env(env)
        self.assertNotIn("最近日志:", output)
        env.close()

    def test_render_log_truncated_to_last_five(self):
        """渲染最近日志时应只显示最后 5 条。"""
        env = QuantumSchedulingEnv(max_steps=50, render_mode="ansi")
        env.reset(seed=42)
        env._render_log.clear()
        # 写入 6 条日志，第 1 条不应出现在输出中
        for i in range(6):
            env._render_log.append(f"日志_{i}_unique_marker")
        output = render_env(env)
        self.assertNotIn("日志_0_unique_marker", output)
        for i in range(1, 6):
            self.assertIn(f"日志_{i}_unique_marker", output)
        env.close()


class TestCloseEnv(unittest.TestCase):
    """测试 close_env 清空内部状态的行为。"""

    def test_close_clears_task_queue(self):
        """close_env 应清空 _task_queue。"""
        env = QuantumSchedulingEnv(max_steps=50, render_mode="ansi")
        env.reset(seed=42)
        self.assertGreater(len(env._task_queue), 0)
        close_env(env)
        self.assertEqual(len(env._task_queue), 0)

    def test_close_clears_render_log(self):
        """close_env 应清空 _render_log。"""
        env = QuantumSchedulingEnv(max_steps=50, render_mode="ansi")
        env.reset(seed=42)
        for _ in range(3):
            env.step(env.action_space.sample())
        if not env._render_log:
            env._render_log.append("dummy")
        self.assertGreater(len(env._render_log), 0)
        close_env(env)
        self.assertEqual(len(env._render_log), 0)

    def test_close_resets_current_task_to_none(self):
        """close_env 应将 _current_task 置为 None。"""
        env = QuantumSchedulingEnv(max_steps=50, render_mode="ansi")
        env.reset(seed=42)
        # 确保 close 前存在 current_task（若有）
        if env._current_task is None:
            env._current_task = Task(
                task_id="QT-X",
                task_type="classical",
                qubit_count=0,
            )
        close_env(env)
        self.assertIsNone(env._current_task)

    def test_close_via_env_close_method(self):
        """通过 env.close() 调用应等价于 close_env(env)。"""
        env = QuantumSchedulingEnv(max_steps=50, render_mode="ansi")
        env.reset(seed=42)
        env.close()
        self.assertEqual(len(env._task_queue), 0)
        self.assertEqual(len(env._render_log), 0)
        self.assertIsNone(env._current_task)

    def test_close_idempotent(self):
        """多次调用 close_env 不应抛异常。"""
        env = QuantumSchedulingEnv(max_steps=50, render_mode="ansi")
        env.reset(seed=42)
        close_env(env)
        close_env(env)
        close_env(env)
        self.assertEqual(len(env._task_queue), 0)


class TestRenderEnvEdgeCases(unittest.TestCase):
    """render_env 边界与组合场景测试。"""

    def test_render_after_close_does_not_raise(self):
        """close 后再调用 render_env 不应抛异常。"""
        env = QuantumSchedulingEnv(max_steps=50, render_mode="ansi")
        env.reset(seed=42)
        env.close()
        output = render_env(env)
        self.assertIsInstance(output, str)
        # 关闭后任务队列为 0，渲染应反映
        self.assertIn("长度: 0", output)
        self.assertIn("[当前任务] 无", output)

    def test_render_after_step_shows_running_state(self):
        """执行若干 step 后渲染应反映递增的步数与非空统计。"""
        env = QuantumSchedulingEnv(max_steps=20, render_mode="ansi")
        env.reset(seed=11)
        for _ in range(5):
            env.step(env.action_space.sample())
        output = render_env(env)
        # 步数应至少为 5
        self.assertIn(f"步骤: {env._current_step}/20", output)
        # 机器数字段
        self.assertIn(f"机器数: {env.num_machines}", output)
        env.close()

    def test_render_ansi_output_ends_with_separator(self):
        """ansi 输出应以分隔线结尾（覆盖最后 lines.append(sep) 分支）。"""
        env = QuantumSchedulingEnv(max_steps=20, render_mode="ansi")
        env.reset(seed=42)
        output = render_env(env)
        sep = "=" * 64
        self.assertTrue(output.rstrip().endswith(sep))
        env.close()


if __name__ == "__main__":
    unittest.main()
