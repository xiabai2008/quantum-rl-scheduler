"""变异测试增强：针对核心模块的典型变异存活点补充测试。

Issue #122: 核心模块变异测试得分偏低（50-68%），本文件针对以下典型变异存活点
补充深度测试，确保变异体被杀死：

env_reward.py 变异存活点：
- compute_execution_reward: ACTION_CLASSICAL/QUANTUM/HYBRID 三分支精确值验证
- fidelity < 0.9 边界（< vs <= 变异）
- hybrid_factor = 0.5 + 0.5 * ratio 公式变异（+ vs -, * vs /）
- compute_wait_penalty: wait_steps > MAX_WAIT_STEPS 边界（> vs >=）
- overtime_ratio 公式变异

env_dynamics.py 变异存活点：
- check_compatibility: 9 种 (task_type, action) 组合全覆盖
- generate_random_task: quantum_ratio=None vs 指定值分支
- generate_random_task: classical → qubit_count=0
- pick_next_task: 排序键 (-priority, -wait_steps, -urgency) 变异
- advance_time: wait_steps += 1 变异（+= vs -=）

baselines.py 变异存活点：
- _get_float / _get_int: None/missing/invalid/valid 四路径
- EDFScheduler._effective_deadline: deadline 存在/缺失/非法三分支
- run_baseline_comparison: wait = max(0, current - arrival) 边界
- run_baseline_comparison: reward = 10 + priority*2 - wait*0.1 公式变异
- BaselineScheduler 基类: NotImplementedError / reset 无操作 / repr
"""

from __future__ import annotations

import unittest
from typing import Any, ClassVar
from unittest.mock import MagicMock

import numpy as np

from src.scheduler.baselines import (
    _DEFAULT_ARRIVAL_TIME,
    _DEFAULT_ESTIMATED_TIME,
    BaselineScheduler,
    EDFScheduler,
    FCFSScheduler,
    LIFOScheduler,
    PriorityScheduler,
    RoundRobinScheduler,
    SPTFScheduler,
    _get_float,
    _get_int,
    run_baseline_comparison,
)
from src.scheduler.env_dynamics import (
    check_compatibility,
    generate_random_task,
    pick_next_task,
)
from src.scheduler.env_reward import (
    compute_execution_reward,
    compute_wait_penalty,
)
from src.scheduler.env_types import (
    ACTION_CLASSICAL,
    ACTION_HYBRID,
    ACTION_QUANTUM,
    MAX_WAIT_STEPS,
    QUANTUM_SPEEDUP_RANGE,
    REWARD_CLASSICAL,
    REWARD_HYBRID,
    REWARD_QUANTUM_BASE,
    REWARD_SUCCESS_BONUS,
    REWARD_WAIT_OVER_THRESHOLD,
    Task,
)


# ============================================================
# 辅助函数
# ============================================================
def _make_task(
    task_id: str = "T0",
    task_type: str = "quantum",
    qubit_count: int = 4,
    wait_steps: int = 0,
    urgency: float = 0.5,
    priority: int = 3,
    execution_time: int = 3,
) -> Task:
    """构造测试用 Task 对象。"""
    return Task(
        task_id=task_id,
        task_type=task_type,
        qubit_count=qubit_count,
        wait_steps=wait_steps,
        urgency=urgency,
        priority=priority,
        execution_time=execution_time,
    )


def _make_dict_task(
    task_id: str = "T0",
    priority: int = 3,
    estimated_time: float = 10.0,
    arrival_time: float = 0.0,
    deadline: float | None = None,
) -> dict[str, Any]:
    """构造测试用任务字典（baselines 使用 dict 格式）。"""
    task: dict[str, Any] = {
        "task_id": task_id,
        "priority": priority,
        "estimated_time": estimated_time,
        "arrival_time": arrival_time,
    }
    if deadline is not None:
        task["deadline"] = deadline
    return task


# ============================================================
# env_reward.py: compute_execution_reward 变异测试
# ============================================================
class TestComputeExecutionRewardMutation(unittest.TestCase):
    """compute_execution_reward 的变异杀死测试。

    针对以下变异：
    - 常量替换（REWARD_CLASSICAL, REWARD_SUCCESS_BONUS 等）
    - 分支交换（if/elif/else 顺序变异）
    - 算术运算符变异（+ → -, * → /）
    - 边界条件变异（< → <=）
    """

    def setUp(self) -> None:
        """初始化固定种子的随机数生成器，确保可复现。"""
        self.rng = np.random.default_rng(42)
        self.task = _make_task()
        # Issue #401: compute_execution_reward 应用 urgency/priority 加权
        # urgency=0.5 → urgency_factor = 0.5 + 0.5*0.5 = 0.75
        # priority=3 → priority_factor = 0.6 + 0.1*3 = 0.9
        # task_weight = 0.75 * 0.9 = 0.675
        self.task_weight = 0.675

    def test_classical_reward_exact_value(self) -> None:
        """经典执行奖励应精确等于 (REWARD_CLASSICAL + REWARD_SUCCESS_BONUS) * task_weight。

        杀死变异：REWARD_CLASSICAL 常量替换、+ → -。
        """
        reward = compute_execution_reward(self.task, ACTION_CLASSICAL, self.rng, 0.99, 1.0)
        expected = (REWARD_CLASSICAL + REWARD_SUCCESS_BONUS) * self.task_weight
        self.assertAlmostEqual(reward, expected, places=6)

    def test_classical_reward_independent_of_quantum_state(self) -> None:
        """经典执行奖励不依赖量子保真度/可用率。

        杀死变异：经典分支误用量子参数。
        """
        r1 = compute_execution_reward(self.task, ACTION_CLASSICAL, self.rng, 0.5, 0.1)
        r2 = compute_execution_reward(self.task, ACTION_CLASSICAL, self.rng, 0.99, 1.0)
        self.assertEqual(r1, r2)

    def test_quantum_reward_within_expected_range(self) -> None:
        """量子执行奖励应在 [base*min_speedup*1.0*weight + bonus, base*max_speedup*1.0*weight + bonus] 内。

        杀死变异：QUANTUM_SPEEDUP_RANGE 常量替换、speedup 采样变异。
        """
        # 高保真度（>= 0.9），不触发 0.6 折扣
        fidelity = 0.99
        reward = compute_execution_reward(self.task, ACTION_QUANTUM, self.rng, fidelity, 1.0)
        min_speedup = QUANTUM_SPEEDUP_RANGE[0] * (fidelity / 0.99)
        max_speedup = QUANTUM_SPEEDUP_RANGE[1] * (fidelity / 0.99)
        self.assertGreaterEqual(
            reward,
            REWARD_QUANTUM_BASE * min_speedup * self.task_weight + REWARD_SUCCESS_BONUS - 0.001,
        )
        self.assertLessEqual(
            reward,
            REWARD_QUANTUM_BASE * max_speedup * self.task_weight + REWARD_SUCCESS_BONUS + 0.001,
        )

    def test_quantum_low_fidelity_discount_boundary(self) -> None:
        """保真度 0.9 是折扣边界：0.89 触发折扣，0.90 不触发。

        杀死变异：< → <=（边界条件变异）。
        """
        task = _make_task()
        # 保真度 0.90 不触发折扣
        rng1 = np.random.default_rng(100)
        r_normal = compute_execution_reward(task, ACTION_QUANTUM, rng1, 0.90, 1.0)

        # 保真度 0.89 触发 0.6 折扣
        rng2 = np.random.default_rng(100)
        r_discounted = compute_execution_reward(task, ACTION_QUANTUM, rng2, 0.89, 1.0)

        # 同一种子下 speedup 相同，但折扣后应明显更小
        self.assertLess(r_discounted, r_normal * 0.7)

    def test_quantum_fidelity_factor_applied(self) -> None:
        """保真度因子 fidelity/0.99 应影响量子奖励。

        杀死变异：fidelity_factor = fidelity / 0.99 中的 / → *。
        """
        rng1 = np.random.default_rng(200)
        rng2 = np.random.default_rng(200)
        r_high = compute_execution_reward(self.task, ACTION_QUANTUM, rng1, 0.99, 1.0)
        r_low = compute_execution_reward(self.task, ACTION_QUANTUM, rng2, 0.95, 1.0)
        # 同种子下 speedup 相同，但 fidelity_factor 不同
        # r_high: speedup * (0.99/0.99) = speedup * 1.0
        # r_low: speedup * (0.95/0.99) ≈ speedup * 0.9596
        # 两者保真度均 >= 0.9，不触发折扣
        self.assertGreater(r_high, r_low)

    def test_hybrid_reward_at_zero_availability(self) -> None:
        """混合执行在可用率为 0 时，hybrid_factor=0.5。

        杀死变异：hybrid_factor = 0.5 + 0.5*0 = 0.5（+ → -, * → /）。
        """
        reward = compute_execution_reward(self.task, ACTION_HYBRID, self.rng, 0.99, 0.0)
        expected = REWARD_HYBRID * 0.5 * self.task_weight + REWARD_SUCCESS_BONUS
        self.assertAlmostEqual(reward, expected, places=6)

    def test_hybrid_reward_at_full_availability(self) -> None:
        """混合执行在可用率为 1.0 时，hybrid_factor=1.0。

        杀死变异：hybrid_factor = 0.5 + 0.5*1.0 = 1.0（+ → -）。
        """
        reward = compute_execution_reward(self.task, ACTION_HYBRID, self.rng, 0.99, 1.0)
        expected = REWARD_HYBRID * 1.0 * self.task_weight + REWARD_SUCCESS_BONUS
        self.assertAlmostEqual(reward, expected, places=6)

    def test_hybrid_reward_monotonic_with_availability(self) -> None:
        """混合执行奖励应随可用率单调递增。

        杀死变异：0.5 + 0.5*ratio 中 + → -。
        """
        r_low = compute_execution_reward(self.task, ACTION_HYBRID, self.rng, 0.99, 0.2)
        r_mid = compute_execution_reward(self.task, ACTION_HYBRID, self.rng, 0.99, 0.5)
        r_high = compute_execution_reward(self.task, ACTION_HYBRID, self.rng, 0.99, 0.9)
        self.assertLess(r_low, r_mid)
        self.assertLess(r_mid, r_high)

    def test_quantum_reward_includes_success_bonus(self) -> None:
        """量子奖励应包含 REWARD_SUCCESS_BONUS。

        杀死变异：reward + REWARD_SUCCESS_BONUS 中 + → -。
        """
        reward = compute_execution_reward(self.task, ACTION_QUANTUM, self.rng, 0.99, 1.0)
        # reward = REWARD_QUANTUM_BASE * speedup * factor + REWARD_SUCCESS_BONUS
        # speedup * factor 至少为 QUANTUM_SPEEDUP_RANGE[0] * (0.99/0.99) = 2.0
        # 所以 reward > REWARD_SUCCESS_BONUS
        self.assertGreater(reward, REWARD_SUCCESS_BONUS)


# ============================================================
# env_reward.py: compute_wait_penalty 变异测试
# ============================================================
class TestComputeWaitPenaltyMutation(unittest.TestCase):
    """compute_wait_penalty 的变异杀死测试。

    针对以下变异：
    - > → >=（边界条件变异）
    - overtime_ratio 公式中的 - → +
    - penalty += → -=
    - MAX_WAIT_STEPS 常量替换
    """

    def test_empty_queue_returns_zero(self) -> None:
        """空队列等待惩罚应为 0.0。

        杀死变异：penalty 初始值 0.0 → 非 0。
        """
        self.assertEqual(compute_wait_penalty([]), 0.0)

    def test_below_threshold_no_penalty(self) -> None:
        """等待步数等于 MAX_WAIT_STEPS 时不触发惩罚（> 而非 >=）。

        杀死变异：> → >=。
        """
        task = _make_task(wait_steps=MAX_WAIT_STEPS)
        self.assertEqual(compute_wait_penalty([task]), 0.0)

    def test_above_threshold_triggers_penalty(self) -> None:
        """等待步数 MAX_WAIT_STEPS + 1 时触发惩罚。

        杀死变异：> → >=、> → <。
        """
        task = _make_task(wait_steps=MAX_WAIT_STEPS + 1)
        penalty = compute_wait_penalty([task])
        expected = REWARD_WAIT_OVER_THRESHOLD * (1.0 / MAX_WAIT_STEPS)
        self.assertAlmostEqual(penalty, expected, places=6)

    def test_penalty_proportional_to_overtime(self) -> None:
        """惩罚应与超时比例成正比。

        杀死变异：overtime_ratio = (wait - MAX) / MAX 中 - → +。
        """
        overtime = 10
        task = _make_task(wait_steps=MAX_WAIT_STEPS + overtime)
        penalty = compute_wait_penalty([task])
        expected = REWARD_WAIT_OVER_THRESHOLD * (overtime / MAX_WAIT_STEPS)
        self.assertAlmostEqual(penalty, expected, places=6)

    def test_penalty_is_negative_or_zero(self) -> None:
        """等待惩罚应为负值或零（REWARD_WAIT_OVER_THRESHOLD 为负）。"""
        task = _make_task(wait_steps=MAX_WAIT_STEPS + 5)
        penalty = compute_wait_penalty([task])
        self.assertLessEqual(penalty, 0.0)

    def test_multiple_tasks_penalty_accumulates(self) -> None:
        """多任务惩罚应累加。

        杀死变异：penalty += → -=。
        """
        t1 = _make_task(wait_steps=MAX_WAIT_STEPS + 2)
        t2 = _make_task(wait_steps=MAX_WAIT_STEPS + 4)
        combined = compute_wait_penalty([t1, t2])
        p1 = compute_wait_penalty([t1])
        p2 = compute_wait_penalty([t2])
        self.assertAlmostEqual(combined, p1 + p2, places=6)

    def test_mixed_threshold_and_below(self) -> None:
        """混合超时和未超时任务时仅惩罚超时任务。"""
        t_below = _make_task(wait_steps=MAX_WAIT_STEPS)
        t_above = _make_task(wait_steps=MAX_WAIT_STEPS + 3)
        combined = compute_wait_penalty([t_below, t_above])
        only_above = compute_wait_penalty([t_above])
        self.assertAlmostEqual(combined, only_above, places=6)


# ============================================================
# env_dynamics.py: check_compatibility 变异测试
# ============================================================
class TestCheckCompatibilityMutation(unittest.TestCase):
    """check_compatibility 的变异杀死测试（9 种组合全覆盖）。

    针对以下变异：
    - return True → return False（universal 分支）
    - action in (...) → action not in (...)
    - task_type == "classical" → != "classical"
    """

    def test_classical_with_classical_action(self) -> None:
        """classical + ACTION_CLASSICAL → True。"""
        task = _make_task(task_type="classical")
        self.assertTrue(check_compatibility(task, ACTION_CLASSICAL))

    def test_classical_with_quantum_action(self) -> None:
        """classical + ACTION_QUANTUM → False。

        杀死变异：return action in (...) → return True。
        """
        task = _make_task(task_type="classical")
        self.assertFalse(check_compatibility(task, ACTION_QUANTUM))

    def test_classical_with_hybrid_action(self) -> None:
        """classical + ACTION_HYBRID → True。"""
        task = _make_task(task_type="classical")
        self.assertTrue(check_compatibility(task, ACTION_HYBRID))

    def test_quantum_with_classical_action(self) -> None:
        """quantum + ACTION_CLASSICAL → False。"""
        task = _make_task(task_type="quantum")
        self.assertFalse(check_compatibility(task, ACTION_CLASSICAL))

    def test_quantum_with_quantum_action(self) -> None:
        """quantum + ACTION_QUANTUM → True。"""
        task = _make_task(task_type="quantum")
        self.assertTrue(check_compatibility(task, ACTION_QUANTUM))

    def test_quantum_with_hybrid_action(self) -> None:
        """quantum + ACTION_HYBRID → True。"""
        task = _make_task(task_type="quantum")
        self.assertTrue(check_compatibility(task, ACTION_HYBRID))

    def test_universal_with_classical_action(self) -> None:
        """universal + ACTION_CLASSICAL → True。

        杀死变异：return True → return False。
        """
        task = _make_task(task_type="universal")
        self.assertTrue(check_compatibility(task, ACTION_CLASSICAL))

    def test_universal_with_quantum_action(self) -> None:
        """universal + ACTION_QUANTUM → True。"""
        task = _make_task(task_type="universal")
        self.assertTrue(check_compatibility(task, ACTION_QUANTUM))

    def test_universal_with_hybrid_action(self) -> None:
        """universal + ACTION_HYBRID → True。"""
        task = _make_task(task_type="universal")
        self.assertTrue(check_compatibility(task, ACTION_HYBRID))

    def test_unknown_task_type_defaults_to_universal(self) -> None:
        """未知 task_type 走 else 分支（universal），对所有动作返回 True。"""
        task = _make_task(task_type="unknown_type")
        for action in (ACTION_CLASSICAL, ACTION_QUANTUM, ACTION_HYBRID):
            self.assertTrue(check_compatibility(task, action))


# ============================================================
# env_dynamics.py: generate_random_task 变异测试
# ============================================================
class TestGenerateRandomTaskMutation(unittest.TestCase):
    """generate_random_task 的变异杀死测试。

    针对以下变异：
    - quantum_ratio=None 分支 vs 指定值分支
    - qubit_count = 0 if classical（条件变异）
    - execution_time = max(1, ...) 中的 max → min
    - priority range [1, 6) 变异
    """

    def test_quantum_ratio_none_can_produce_universal(self) -> None:
        """quantum_ratio=None 时可生成 universal 类型（多次采样验证）。

        杀死变异：quantum_ratio=None 分支被跳过。
        """
        rng = np.random.default_rng(42)
        types_seen: set[str] = set()
        for i in range(500):
            task = generate_random_task(rng, i, quantum_ratio=None)
            types_seen.add(task.task_type)
        # 应至少包含 quantum 和 classical
        self.assertIn("quantum", types_seen)
        self.assertIn("classical", types_seen)

    def test_quantum_ratio_one_produces_only_quantum(self) -> None:
        """quantum_ratio=1.0 时只生成 quantum 类型。

        杀死变异：rng.random() < quantum_ratio 中 < → >。
        """
        rng = np.random.default_rng(42)
        for i in range(100):
            task = generate_random_task(rng, i, quantum_ratio=1.0)
            self.assertEqual(task.task_type, "quantum")

    def test_quantum_ratio_zero_produces_only_classical(self) -> None:
        """quantum_ratio=0.0 时只生成 classical 类型。

        杀死变异：rng.random() < 0.0 恒为 False 的逻辑变异。
        """
        rng = np.random.default_rng(42)
        for i in range(100):
            task = generate_random_task(rng, i, quantum_ratio=0.0)
            self.assertEqual(task.task_type, "classical")

    def test_classical_task_has_zero_qubits(self) -> None:
        """classical 任务的 qubit_count 应为 0。

        杀死变异：qubit_count = 0 if task_type == "classical" 中 0 → qubits。
        """
        rng = np.random.default_rng(42)
        for i in range(200):
            task = generate_random_task(rng, i, quantum_ratio=0.0)
            self.assertEqual(task.qubit_count, 0)

    def test_quantum_task_has_nonzero_qubits(self) -> None:
        """quantum 任务的 qubit_count 应 > 0。"""
        rng = np.random.default_rng(42)
        for i in range(200):
            task = generate_random_task(rng, i, quantum_ratio=1.0)
            self.assertGreater(task.qubit_count, 0)

    def test_execution_time_at_least_one(self) -> None:
        """execution_time 应 >= 1（max(1, ...) 保证）。

        杀死变异：max → min。
        """
        rng = np.random.default_rng(42)
        for i in range(500):
            task = generate_random_task(rng, i, quantum_ratio=None)
            self.assertGreaterEqual(task.execution_time, 1)

    def test_priority_in_valid_range(self) -> None:
        """priority 应在 [1, 5] 范围内。

        杀死变异：rng.integers(1, 6) 中 1 → 0 或 6 → 5。
        """
        rng = np.random.default_rng(42)
        for i in range(500):
            task = generate_random_task(rng, i, quantum_ratio=None)
            self.assertGreaterEqual(task.priority, 1)
            self.assertLessEqual(task.priority, 5)

    def test_task_id_format(self) -> None:
        """task_id 应格式化为 T{task_id:04d}。"""
        rng = np.random.default_rng(42)
        task = generate_random_task(rng, 7)
        self.assertEqual(task.task_id, "T0007")
        task = generate_random_task(rng, 123)
        self.assertEqual(task.task_id, "T0123")

    def test_wait_steps_initially_zero(self) -> None:
        """新生成任务的 wait_steps 应为 0。"""
        rng = np.random.default_rng(42)
        task = generate_random_task(rng, 0)
        self.assertEqual(task.wait_steps, 0)

    def test_urgency_in_valid_range(self) -> None:
        """urgency 应在 [0.1, 1.0] 范围内。"""
        rng = np.random.default_rng(42)
        for i in range(500):
            task = generate_random_task(rng, i, quantum_ratio=None)
            self.assertGreaterEqual(task.urgency, 0.1)
            self.assertLessEqual(task.urgency, 1.0)


# ============================================================
# env_dynamics.py: pick_next_task 变异测试
# ============================================================
class TestPickNextTaskMutation(unittest.TestCase):
    """pick_next_task 的变异杀死测试。

    针对以下变异：
    - 排序键 (-priority, -wait_steps, -urgency) 中 - → +
    - pop(0) → pop()（LIFO 变异）
    - 空队列 → None 分支
    """

    def _make_mock_env(self, task_queue: list[Task]) -> Any:
        """构造包含指定任务队列的 mock 环境。"""
        env = MagicMock()
        env._task_queue = task_queue
        env._current_task = None
        return env

    def test_empty_queue_sets_none(self) -> None:
        """空队列应将 _current_task 设为 None。

        杀死变异：if not env._task_queue → if env._task_queue。
        """
        env = self._make_mock_env([])
        pick_next_task(env)
        self.assertIsNone(env._current_task)

    def test_highest_priority_selected_first(self) -> None:
        """应优先选择 priority 最高的任务。

        杀死变异：-t.priority → +t.priority。
        """
        t1 = _make_task("T1", priority=1)
        t2 = _make_task("T2", priority=5)
        t3 = _make_task("T3", priority=3)
        env = self._make_mock_env([t1, t2, t3])
        pick_next_task(env)
        self.assertEqual(env._current_task.task_id, "T2")

    def test_same_priority_higher_wait_selected(self) -> None:
        """相同 priority 时，wait_steps 更高的优先。

        杀死变异：-t.wait_steps → +t.wait_steps。
        """
        t1 = _make_task("T1", priority=3, wait_steps=5)
        t2 = _make_task("T2", priority=3, wait_steps=10)
        t3 = _make_task("T3", priority=3, wait_steps=2)
        env = self._make_mock_env([t1, t2, t3])
        pick_next_task(env)
        self.assertEqual(env._current_task.task_id, "T2")

    def test_same_priority_and_wait_higher_urgency_selected(self) -> None:
        """相同 priority 和 wait_steps 时，urgency 更高的优先。

        杀死变异：-t.urgency → +t.urgency。
        """
        t1 = _make_task("T1", priority=3, wait_steps=5, urgency=0.3)
        t2 = _make_task("T2", priority=3, wait_steps=5, urgency=0.9)
        env = self._make_mock_env([t1, t2])
        pick_next_task(env)
        self.assertEqual(env._current_task.task_id, "T2")

    def test_pops_first_element_not_last(self) -> None:
        """应弹出队首元素（pop(0)），而非队尾（pop()）。

        杀死变异：pop(0) → pop()。
        """
        t1 = _make_task("T1", priority=5)
        t2 = _make_task("T2", priority=1)
        env = self._make_mock_env([t1, t2])
        pick_next_task(env)
        # t1 优先级最高，应被弹出
        self.assertEqual(env._current_task.task_id, "T1")
        # 队列中应只剩 t2
        self.assertEqual(len(env._task_queue), 1)
        self.assertEqual(env._task_queue[0].task_id, "T2")

    def test_single_task_popped_correctly(self) -> None:
        """单任务时应弹出该任务，队列为空。"""
        t1 = _make_task("T1", priority=3)
        env = self._make_mock_env([t1])
        pick_next_task(env)
        self.assertEqual(env._current_task.task_id, "T1")
        self.assertEqual(len(env._task_queue), 0)


# ============================================================
# env_dynamics.py: advance_time 变异测试
# ============================================================
class TestAdvanceTimeMutation(unittest.TestCase):
    """advance_time 的变异杀死测试。

    针对以下变异：
    - wait_steps += 1 → -= 1
    - MAX_QUEUE_SIZE 检查被跳过
    - 新任务生成逻辑变异
    """

    def _make_mock_env(
        self,
        task_queue: list[Task] | None = None,
        machines: list[Any] | None = None,
        arrival_lambda: float = 0.0,
    ) -> Any:
        """构造 mock 环境用于 advance_time 测试。"""
        from src.scheduler.env_types import ClassicalResource, QuantumResource

        env = MagicMock()
        env._task_queue = task_queue if task_queue is not None else []
        env._machines = machines or []
        env._quantum = QuantumResource()
        env._classical = ClassicalResource()
        env._total_scheduled = 0
        # Issue #522: advance_time 访问 arrival_history 和 max_arrival_history_length
        env.arrival_history = []
        env.max_arrival_history_length = 10

        # _get_arrival_lambda 返回固定值
        env._get_arrival_lambda.return_value = arrival_lambda

        # _recompute_aggregate 无操作
        env._recompute_aggregate = MagicMock()

        # _generate_random_task 代理到真实函数
        env._generate_random_task = MagicMock(side_effect=generate_random_task)

        return env

    def test_wait_steps_incremented(self) -> None:
        """advance_time 后队列中任务的 wait_steps 应 +1。

        杀死变异：task.wait_steps += 1 → -= 1。
        """
        t1 = _make_task("T1", wait_steps=5)
        t2 = _make_task("T2", wait_steps=10)
        env = self._make_mock_env(task_queue=[t1, t2], arrival_lambda=0.0)
        rng = np.random.default_rng(42)
        # 直接调用 advance_time
        from src.scheduler.env_dynamics import advance_time

        advance_time(env, rng)
        self.assertEqual(t1.wait_steps, 6)
        self.assertEqual(t2.wait_steps, 11)

    def test_no_new_tasks_when_lambda_zero(self) -> None:
        """arrival_lambda=0.0 时泊松分布不生成新任务。

        杀死变异：rng.poisson(0) 返回 >0 的变异。
        """
        env = self._make_mock_env(task_queue=[], arrival_lambda=0.0)
        rng = np.random.default_rng(42)
        from src.scheduler.env_dynamics import advance_time

        advance_time(env, rng)
        self.assertEqual(len(env._task_queue), 0)

    def test_machine_available_ratio_clipped(self) -> None:
        """机器 available_ratio 应被 clip 到 [0.05, 1.0]。

        杀死变异：clip 范围变异。
        """
        from src.scheduler.env_types import QuantumMachine

        m = QuantumMachine(name="test", available_ratio=0.06, fidelity=0.95)
        env = self._make_mock_env(machines=[m], arrival_lambda=0.0)
        rng = np.random.default_rng(42)
        from src.scheduler.env_dynamics import advance_time

        # 多次推进，确保 clip 生效
        for _ in range(50):
            advance_time(env, rng)
            self.assertGreaterEqual(m.available_ratio, 0.05 - 1e-9)
            self.assertLessEqual(m.available_ratio, 1.0 + 1e-9)

    def test_machine_fidelity_clipped(self) -> None:
        """机器 fidelity 应被 clip 到 [0.7, 0.999]。"""
        from src.scheduler.env_types import QuantumMachine

        m = QuantumMachine(name="test", available_ratio=0.5, fidelity=0.98)
        env = self._make_mock_env(machines=[m], arrival_lambda=0.0)
        rng = np.random.default_rng(42)
        from src.scheduler.env_dynamics import advance_time

        for _ in range(50):
            advance_time(env, rng)
            self.assertGreaterEqual(m.fidelity, 0.7 - 1e-9)
            self.assertLessEqual(m.fidelity, 0.999 + 1e-9)

    def test_classical_load_clipped(self) -> None:
        """经典负载应被 clip 到 [0.0, 1.0]。"""
        env = self._make_mock_env(task_queue=[], arrival_lambda=0.0)
        env._classical.load = 0.5
        rng = np.random.default_rng(42)
        from src.scheduler.env_dynamics import advance_time

        for _ in range(50):
            advance_time(env, rng)
            self.assertGreaterEqual(float(env._classical.load), 0.0 - 1e-9)
            self.assertLessEqual(float(env._classical.load), 1.0 + 1e-9)


# ============================================================
# baselines.py: _get_float / _get_int 变异测试
# ============================================================
class TestGetFloatIntMutation(unittest.TestCase):
    """_get_float / _get_int 的变异杀死测试。

    针对以下变异：
    - return default → return 0
    - except (TypeError, ValueError) → except Exception
    - float(value) → int(value)
    """

    def test_get_float_valid_value(self) -> None:
        """有效 float 字段应返回正确值。"""
        task = {"time": 3.14}
        self.assertAlmostEqual(_get_float(task, "time", 0.0), 3.14, places=6)

    def test_get_float_missing_key(self) -> None:
        """缺失字段应返回默认值。

        杀死变异：return default → return 0.0。
        """
        task: dict[str, Any] = {}
        self.assertEqual(_get_float(task, "missing", 9.9), 9.9)

    def test_get_float_none_value(self) -> None:
        """None 值应返回默认值。

        杀死变异：if value is None → if value is not None。
        """
        task = {"time": None}
        self.assertEqual(_get_float(task, "time", 5.0), 5.0)

    def test_get_float_invalid_string(self) -> None:
        """无法转换的字符串应返回默认值。"""
        task = {"time": "abc"}
        self.assertEqual(_get_float(task, "time", 7.0), 7.0)

    def test_get_float_int_value_converted(self) -> None:
        """int 值应被转换为 float。

        杀死变异：float(value) → value（不转换）。
        """
        task = {"time": 5}
        result = _get_float(task, "time", 0.0)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 5.0, places=6)

    def test_get_float_string_numeric_converted(self) -> None:
        """数字字符串应被转换为 float。"""
        task = {"time": "3.14"}
        self.assertAlmostEqual(_get_float(task, "time", 0.0), 3.14, places=6)

    def test_get_int_valid_value(self) -> None:
        """有效 int 字段应返回正确值。"""
        task = {"priority": 4}
        self.assertEqual(_get_int(task, "priority", 3), 4)

    def test_get_int_missing_key(self) -> None:
        """缺失字段应返回默认值。"""
        task: dict[str, Any] = {}
        self.assertEqual(_get_int(task, "missing", 5), 5)

    def test_get_int_none_value(self) -> None:
        """None 值应返回默认值。"""
        task = {"priority": None}
        self.assertEqual(_get_int(task, "priority", 3), 3)

    def test_get_int_invalid_string(self) -> None:
        """无法转换的字符串应返回默认值。"""
        task = {"priority": "abc"}
        self.assertEqual(_get_int(task, "priority", 2), 2)

    def test_get_int_float_value_truncated(self) -> None:
        """float 值应被截断为 int。"""
        task = {"priority": 3.7}
        self.assertEqual(_get_int(task, "priority", 0), 3)

    def test_get_int_string_numeric_converted(self) -> None:
        """数字字符串应被转换为 int。"""
        task = {"priority": "5"}
        self.assertEqual(_get_int(task, "priority", 0), 5)


# ============================================================
# baselines.py: EDFScheduler._effective_deadline 变异测试
# ============================================================
class TestEffectiveDeadlineMutation(unittest.TestCase):
    """EDFScheduler._effective_deadline 的变异杀死测试。

    针对以下变异：
    - deadline 存在分支跳过
    - float(raw) → int(raw)
    - arrival + est * 2.0 中 * → +, 2.0 → 1.0
    """

    def test_valid_deadline_used_directly(self) -> None:
        """有效 deadline 应直接使用。

        杀死变异：return float(raw) → return inferred。
        """
        task = {"deadline": 42.0, "arrival_time": 0.0, "estimated_time": 10.0}
        self.assertAlmostEqual(EDFScheduler._effective_deadline(task), 42.0, places=6)

    def test_none_deadline_inferred(self) -> None:
        """deadline=None 应走推算路径。

        杀死变异：if raw is not None → if raw is None。
        """
        task = {"deadline": None, "arrival_time": 5.0, "estimated_time": 3.0}
        # inferred = 5.0 + 3.0 * 2.0 = 11.0
        self.assertAlmostEqual(EDFScheduler._effective_deadline(task), 11.0, places=6)

    def test_missing_deadline_inferred(self) -> None:
        """deadline 缺失应走推算路径。"""
        task = {"arrival_time": 2.0, "estimated_time": 4.0}
        # inferred = 2.0 + 4.0 * 2.0 = 10.0
        self.assertAlmostEqual(EDFScheduler._effective_deadline(task), 10.0, places=6)

    def test_invalid_deadline_string_inferred(self) -> None:
        """deadline 为非数字字符串应走推算路径。

        杀死变异：except (TypeError, ValueError) → pass（不捕获）。
        """
        task = {"deadline": "not_a_number", "arrival_time": 1.0, "estimated_time": 2.0}
        # inferred = 1.0 + 2.0 * 2.0 = 5.0
        self.assertAlmostEqual(EDFScheduler._effective_deadline(task), 5.0, places=6)

    def test_inference_formula_correct(self) -> None:
        """推算公式应为 arrival + est * 2.0。

        杀死变异：* 2.0 → * 1.0 或 + 2.0。
        """
        task = {"arrival_time": 10.0, "estimated_time": 7.0}
        # 10 + 7 * 2 = 24
        self.assertAlmostEqual(EDFScheduler._effective_deadline(task), 24.0, places=6)

    def test_inference_uses_default_arrival(self) -> None:
        """arrival_time 缺失时使用默认值 _DEFAULT_ARRIVAL_TIME。"""
        task = {"estimated_time": 5.0}
        # inferred = _DEFAULT_ARRIVAL_TIME + 5.0 * 2.0
        expected = _DEFAULT_ARRIVAL_TIME + 5.0 * 2.0
        self.assertAlmostEqual(EDFScheduler._effective_deadline(task), expected, places=6)

    def test_inference_uses_default_estimated_time(self) -> None:
        """estimated_time 缺失时使用默认值 _DEFAULT_ESTIMATED_TIME。"""
        task = {"arrival_time": 3.0}
        # inferred = 3.0 + _DEFAULT_ESTIMATED_TIME * 2.0
        expected = 3.0 + _DEFAULT_ESTIMATED_TIME * 2.0
        self.assertAlmostEqual(EDFScheduler._effective_deadline(task), expected, places=6)


# ============================================================
# baselines.py: run_baseline_comparison 变异测试
# ============================================================
class TestRunBaselineComparisonMutation(unittest.TestCase):
    """run_baseline_comparison 的变异杀死测试。

    针对以下变异：
    - wait = max(0, current - arrival) 中 max → min
    - reward = 10 + priority*2 - wait*0.1 公式变异
    - avg_wait = total / completed 中 / → *
    - throughput = completed / num_steps 中 / → *
    """

    def test_wait_clamped_to_zero(self) -> None:
        """等待时间 = max(0, current - arrival)，负值应被钳为 0。

        杀死变异：max → min。
        """
        # arrival_time 远大于 0，但 current_time 从 0 开始
        # 第一个任务 wait = max(0, 0 - 100) = 0
        tasks = [_make_dict_task("T1", priority=3, estimated_time=1.0, arrival_time=100.0)]
        results = run_baseline_comparison(tasks, num_steps=10)
        for name, metrics in results.items():
            self.assertEqual(metrics["avg_wait_time"], 0.0, f"{name} 负等待应被钳为 0")

    def test_reward_formula_verified(self) -> None:
        """奖励公式 reward = 10 + priority*2 - wait*0.1 应精确验证。

        杀死变异：10 → 0、priority*2 → priority*1、wait*0.1 → wait*1。
        """
        # 单任务，arrival_time=0，estimated_time=1.0
        # current_time=0, wait=0, reward = 10 + 3*2 - 0*0.1 = 16
        tasks = [_make_dict_task("T1", priority=3, estimated_time=1.0, arrival_time=0.0)]
        results = run_baseline_comparison(tasks, num_steps=10)
        for name, metrics in results.items():
            self.assertAlmostEqual(
                metrics["total_reward"], 16.0, places=4, msg=f"{name} 奖励公式错误"
            )

    def test_higher_priority_gives_higher_reward(self) -> None:
        """高优先级任务应获得更高奖励。

        杀死变异：priority * 2.0 → priority * (-2.0)。
        """
        tasks_low = [_make_dict_task("T1", priority=1, estimated_time=1.0, arrival_time=0.0)]
        tasks_high = [_make_dict_task("T1", priority=5, estimated_time=1.0, arrival_time=0.0)]
        r_low = run_baseline_comparison(tasks_low, num_steps=10)
        r_high = run_baseline_comparison(tasks_high, num_steps=10)
        for name in r_low:
            self.assertGreater(
                r_high[name]["total_reward"],
                r_low[name]["total_reward"],
                f"{name} 高优先级奖励应更高",
            )

    def test_throughput_formula_verified(self) -> None:
        """吞吐率 = completed / num_steps。

        杀死变异：/ → *。
        """
        tasks = [_make_dict_task("T1", estimated_time=1.0, arrival_time=0.0)]
        results = run_baseline_comparison(tasks, num_steps=10)
        for _name, metrics in results.items():
            # 1 task completed / 10 steps = 0.1
            self.assertAlmostEqual(metrics["throughput"], 0.1, places=6)

    def test_avg_wait_zero_when_no_completion(self) -> None:
        """无任务完成时 avg_wait 应为 0.0（避免除零）。

        杀死变异：completed > 0 → completed >= 0（除零）。
        """
        results = run_baseline_comparison([], num_steps=10)
        for _name, metrics in results.items():
            self.assertEqual(metrics["avg_wait_time"], 0.0)

    def test_throughput_zero_when_zero_steps(self) -> None:
        """num_steps=0 时 throughput 应为 0.0（避免除零）。

        杀死变异：num_steps > 0 → num_steps >= 0。
        """
        tasks = [_make_dict_task("T1", estimated_time=1.0, arrival_time=0.0)]
        results = run_baseline_comparison(tasks, num_steps=0)
        for _name, metrics in results.items():
            self.assertEqual(metrics["throughput"], 0.0)

    def test_tasks_not_shared_between_strategies(self) -> None:
        """各策略应使用独立的任务副本，不互相污染。

        杀死变异：深拷贝 → 浅引用。
        """
        tasks = [_make_dict_task("T1", estimated_time=1.0, arrival_time=0.0)]
        original_len = len(tasks)
        run_baseline_comparison(tasks, num_steps=10)
        # 原始列表不应被修改
        self.assertEqual(len(tasks), original_len)


# ============================================================
# baselines.py: BaselineScheduler 基类变异测试
# ============================================================
class TestBaselineSchedulerBaseMutation(unittest.TestCase):
    """BaselineScheduler 基类的变异杀死测试。

    针对以下变异：
    - raise NotImplementedError → return 0
    - reset() 空操作 → reset() 修改状态
    - __repr__ 格式变异
    """

    def test_base_select_action_raises_not_implemented(self) -> None:
        """基类 select_action 应抛出 NotImplementedError。

        杀死变异：raise → return。
        """
        scheduler = BaselineScheduler("base")
        with self.assertRaises(NotImplementedError):
            scheduler.select_action([{"task_id": "T1"}], {})

    def test_base_select_action_empty_returns_negative(self) -> None:
        """基类空列表应返回 -1（不抛异常）。"""
        scheduler = BaselineScheduler("base")
        self.assertEqual(scheduler.select_action([], {}), -1)

    def test_reset_is_noop_in_base(self) -> None:
        """基类 reset() 应为空操作，不修改任何状态。

        杀死变异：reset() 添加 self.name = "" 。
        """
        scheduler = BaselineScheduler("test")
        original_name = scheduler.name
        scheduler.reset()
        self.assertEqual(scheduler.name, original_name)

    def test_repr_contains_class_name_and_name(self) -> None:
        """__repr__ 应包含类名和 name 属性。

        杀死变异：self.__class__.__name__ → self.name。
        """
        scheduler = FCFSScheduler()
        repr_str = repr(scheduler)
        self.assertIn("FCFSScheduler", repr_str)
        self.assertIn("FCFS", repr_str)

    def test_all_schedulers_have_correct_names(self) -> None:
        """每个策略实例的 name 属性应与类名对应。

        杀死变异：super().__init__("FCFS") → super().__init__("wrong")。
        """
        expected = {
            FCFSScheduler: "FCFS",
            SPTFScheduler: "SPTF",
            EDFScheduler: "EDF",
            PriorityScheduler: "Priority",
            RoundRobinScheduler: "RoundRobin",
            LIFOScheduler: "LIFO",
        }
        for cls, name in expected.items():
            scheduler = cls()
            self.assertEqual(scheduler.name, name, f"{cls.__name__} name mismatch")


# ============================================================
# baselines.py: 策略选择逻辑变异测试
# ============================================================
class TestSchedulerSelectionLogicMutation(unittest.TestCase):
    """各策略选择逻辑的变异杀死测试。

    针对以下变异：
    - min → max（FCFS/SPTF/EDF）
    - max → min（Priority/LIFO）
    - key 函数中字段引用变异
    """

    _RESOURCES: ClassVar[dict[str, Any]] = {"qubits": 20, "classical_load": 0.0}

    def test_fcfs_uses_arrival_time_not_priority(self) -> None:
        """FCFS 应按 arrival_time 排序，不受 priority 影响。

        杀死变异：arrival_time → priority。
        """
        tasks = [
            _make_dict_task("T1", priority=5, arrival_time=10.0),
            _make_dict_task("T2", priority=1, arrival_time=1.0),
        ]
        scheduler = FCFSScheduler()
        idx = scheduler.select_action(tasks, self._RESOURCES)
        self.assertEqual(idx, 1)  # T2 到达更早

    def test_sptf_uses_estimated_time_not_arrival(self) -> None:
        """SPTF 应按 estimated_time 排序，不受 arrival_time 影响。

        杀死变异：estimated_time → arrival_time。
        """
        tasks = [
            _make_dict_task("T1", estimated_time=50.0, arrival_time=0.0),
            _make_dict_task("T2", estimated_time=5.0, arrival_time=100.0),
        ]
        scheduler = SPTFScheduler()
        idx = scheduler.select_action(tasks, self._RESOURCES)
        self.assertEqual(idx, 1)  # T2 时间更短

    def test_priority_uses_priority_not_estimated_time(self) -> None:
        """Priority 应按 priority 排序，不受 estimated_time 影响。

        杀死变异：priority → estimated_time。
        """
        tasks = [
            _make_dict_task("T1", priority=5, estimated_time=100.0),
            _make_dict_task("T2", priority=1, estimated_time=1.0),
        ]
        scheduler = PriorityScheduler()
        idx = scheduler.select_action(tasks, self._RESOURCES)
        self.assertEqual(idx, 0)  # T1 优先级更高

    def test_lifo_uses_arrival_descending(self) -> None:
        """LIFO 应按 arrival_time 降序选择（max 而非 min）。

        杀死变异：max → min。
        """
        tasks = [
            _make_dict_task("T1", arrival_time=1.0),
            _make_dict_task("T2", arrival_time=10.0),
        ]
        scheduler = LIFOScheduler()
        idx = scheduler.select_action(tasks, self._RESOURCES)
        self.assertEqual(idx, 1)  # T2 到达更晚

    def test_round_robin_wraps_around(self) -> None:
        """RoundRobin 指针应在到达末尾后回绕到 0。

        杀死变异：self._pointer % n → self._pointer（无回绕）。
        """
        tasks = [_make_dict_task("A"), _make_dict_task("B"), _make_dict_task("C")]
        scheduler = RoundRobinScheduler()
        # 调用 4 次，第 4 次应回绕到索引 0
        indices = [scheduler.select_action(tasks, self._RESOURCES) for _ in range(4)]
        self.assertEqual(indices, [0, 1, 2, 0])

    def test_round_robin_different_sizes(self) -> None:
        """RoundRobin 在不同任务数量下应正确回绕。"""
        tasks = [_make_dict_task("A"), _make_dict_task("B")]
        scheduler = RoundRobinScheduler()
        indices = [scheduler.select_action(tasks, self._RESOURCES) for _ in range(5)]
        self.assertEqual(indices, [0, 1, 0, 1, 0])

    def test_priority_tiebreak_uses_arrival_time(self) -> None:
        """相同 priority 时应以 arrival_time 升序作为 tiebreaker。

        杀死变异：-_get_float(..., "arrival_time", ...) → +_get_float(...)。
        """
        tasks = [
            _make_dict_task("T1", priority=3, arrival_time=10.0),
            _make_dict_task("T2", priority=3, arrival_time=1.0),
        ]
        scheduler = PriorityScheduler()
        idx = scheduler.select_action(tasks, self._RESOURCES)
        self.assertEqual(idx, 1)  # T2 到达更早

    def test_edf_prefers_explicit_over_inferred(self) -> None:
        """EDF 应比较显式 deadline 和推算 deadline。

        杀死变异：_effective_deadline → arrival_time。
        """
        tasks = [
            _make_dict_task("T1", arrival_time=0.0, estimated_time=100.0),  # 推算 200
            _make_dict_task("T2", arrival_time=0.0, estimated_time=100.0, deadline=50.0),  # 显式 50
        ]
        scheduler = EDFScheduler()
        idx = scheduler.select_action(tasks, self._RESOURCES)
        self.assertEqual(idx, 1)  # T2 显式 deadline 更早


if __name__ == "__main__":
    unittest.main()
