"""Issue #441: OR-Tools 对比实验脚本修复的单元测试。

验证修复点：
1. _generate_task_trace 生成固定数量的任务（原版 generate_batch 平均只产 0.5 个）
2. solve_cp_sat 返回随规模变化的 makespan（原版恒为 5.0）
3. solve_cp_sat 返回真实求解时间（原版恒为 0.0s）
4. _CompatPPOStrategy 自动截断观测维度（环境 16 维 → 模型 14 维）
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ortools 是可选依赖，跳过未安装时的测试
_ortools_available = importlib.util.find_spec("ortools") is not None
_skip_no_ortools = pytest.mark.skipif(
    not _ortools_available, reason="ortools 未安装（requirements-dev.txt 可选依赖）"
)


@_skip_no_ortools
class TestGenerateTaskTrace:
    """测试任务生成修复（泊松累积）。"""

    def test_generates_exact_count(self):
        """_generate_task_trace 应生成恰好 n_tasks 个任务。"""
        from scripts.evaluation.ortools_comparison import _generate_task_trace

        for n in [5, 20, 50]:
            tasks = _generate_task_trace(n, seed=42)
            assert len(tasks) == n, f"期望 {n} 个任务，实际 {len(tasks)}"

    def test_tasks_have_required_fields(self):
        """任务应包含 task_type 和 qubit_count 字段。"""
        from scripts.evaluation.ortools_comparison import _generate_task_trace

        tasks = _generate_task_trace(10, seed=42)
        for t in tasks:
            assert "task_type" in t
            assert "qubit_count" in t
            assert t["task_type"] in ("quantum", "classical")

    def test_seed_reproducibility(self):
        """相同 seed 应生成相同任务序列。"""
        from scripts.evaluation.ortools_comparison import _generate_task_trace

        t1 = _generate_task_trace(20, seed=42)
        t2 = _generate_task_trace(20, seed=42)
        assert t1 == t2


@_skip_no_ortools
class TestSolveCpSat:
    """测试 CP-SAT 建模修复（按机器分组 NoOverlap + OptionalIntervalVar）。"""

    def test_returns_non_constant_makespan(self):
        """makespan 不应恒为 5.0（原版 bug），应随任务规模变化。"""
        from scripts.evaluation.ortools_comparison import _generate_task_trace, solve_cp_sat

        mk_20 = solve_cp_sat(_generate_task_trace(20, seed=42), time_limit=10)["makespan"]
        mk_50 = solve_cp_sat(_generate_task_trace(50, seed=42), time_limit=10)["makespan"]

        assert mk_20 is not None
        assert mk_50 is not None
        # 50 任务的 makespan 应明显大于 20 任务（原版两者均为 5.0）
        assert mk_50 > mk_20, f"makespan 未随规模变化: mk_20={mk_20}, mk_50={mk_50}"

    def test_returns_nonzero_solve_time(self):
        """求解时间不应为 0.0s（原版 bug），应有真实测量值。"""
        from scripts.evaluation.ortools_comparison import _generate_task_trace, solve_cp_sat

        r = solve_cp_sat(_generate_task_trace(20, seed=42), time_limit=10)
        assert r["wall_time"] > 0.0, f"求解时间为 0.0s: {r}"

    def test_returns_required_fields(self):
        """结果应包含 makespan/avg_flow_time/status/wall_time/n_tasks。"""
        from scripts.evaluation.ortools_comparison import _generate_task_trace, solve_cp_sat

        r = solve_cp_sat(_generate_task_trace(10, seed=42), time_limit=10)
        for key in ("makespan", "avg_flow_time", "status", "wall_time", "n_tasks"):
            assert key in r, f"缺少字段 {key}: {r}"

    def test_empty_tasks(self):
        """空任务列表应返回零值结果。"""
        from scripts.evaluation.ortools_comparison import solve_cp_sat

        r = solve_cp_sat([], time_limit=5)
        assert r["makespan"] == 0.0
        assert r["n_tasks"] == 0
        assert r["status"] == "EMPTY"

    def test_makespan_reasonable_upper_bound(self):
        """makespan 不应超过 sum(durations)（horizon 上界）。"""
        from scripts.evaluation.ortools_comparison import (
            _generate_task_trace,
            _task_duration,
            solve_cp_sat,
        )

        tasks = _generate_task_trace(20, seed=42)
        r = solve_cp_sat(tasks, time_limit=10)
        horizon = sum(_task_duration(t) for t in tasks)
        assert r["makespan"] <= horizon, f"makespan={r['makespan']} 超过 horizon={horizon}"


@_skip_no_ortools
class TestTaskDuration:
    """测试任务时长估算。"""

    def test_quantum_task_duration(self):
        """量子任务 duration = qubit_count * 10 + 5。"""
        from scripts.evaluation.ortools_comparison import _task_duration

        task = {"task_type": "quantum", "qubit_count": 10}
        assert _task_duration(task) == 105

    def test_classical_task_duration(self):
        """经典任务 duration = 5。"""
        from scripts.evaluation.ortools_comparison import _task_duration

        task = {"task_type": "classical", "qubit_count": 0}
        assert _task_duration(task) == 5

    def test_min_duration(self):
        """duration 至少为 1。"""
        from scripts.evaluation.ortools_comparison import _task_duration

        task = {"task_type": "quantum", "qubit_count": 0}
        # qubit_count=0 时 quantum duration = 0*10+5 = 5
        assert _task_duration(task) == 5


@_skip_no_ortools
class TestCompatPPOStrategy:
    """测试 PPO 观测维度兼容层。"""

    def test_truncates_observation_when_model_dim_smaller(self):
        """当模型维度 < 环境维度时，应截断观测。"""
        from scripts.evaluation.ortools_comparison import _CompatPPOStrategy

        mock_model = MagicMock()
        mock_model.observation_space = MagicMock(shape=(14,))
        mock_model.predict.return_value = (np.array([2]), None)

        strategy = _CompatPPOStrategy(mock_model)
        assert strategy._model_obs_dim == 14

        obs_16 = np.random.rand(16)
        action = strategy.select_action(obs_16)
        # 应截断为 14 维
        called_obs = mock_model.predict.call_args[0][0]
        assert called_obs.shape[0] == 14
        np.testing.assert_array_equal(called_obs, obs_16[:14])
        assert action == 2

    def test_no_truncation_when_dims_match(self):
        """当模型维度 == 环境维度时，不应截断。"""
        from scripts.evaluation.ortools_comparison import _CompatPPOStrategy

        mock_model = MagicMock()
        mock_model.observation_space = MagicMock(shape=(14,))
        mock_model.predict.return_value = (np.array([1]), None)

        strategy = _CompatPPOStrategy(mock_model)
        obs_14 = np.random.rand(14)
        strategy.select_action(obs_14)
        called_obs = mock_model.predict.call_args[0][0]
        assert called_obs.shape[0] == 14

    def test_handles_model_without_observation_space(self):
        """模型无 observation_space 时应回退到不截断。"""
        from scripts.evaluation.ortools_comparison import _CompatPPOStrategy

        mock_model = MagicMock()
        mock_model.observation_space = None
        mock_model.predict.return_value = (np.array([0]), None)

        strategy = _CompatPPOStrategy(mock_model)
        assert strategy._model_obs_dim is None

        obs = np.random.rand(16)
        strategy.select_action(obs)
        called_obs = mock_model.predict.call_args[0][0]
        assert called_obs.shape[0] == 16


@_skip_no_ortools
class TestRlAvgFlowTime:
    """测试 RL 侧 flow_time 计算。"""

    def test_sum_of_wait_and_execution(self):
        """avg_flow_time = avg_wait_time + avg_execution_time。"""
        from scripts.evaluation.ortools_comparison import _rl_avg_flow_time

        summary = {"avg_wait_time": 5.5, "avg_execution_time": 2.5}
        assert _rl_avg_flow_time(summary) == 8.0

    def test_handles_missing_fields(self):
        """缺少字段时应返回 0.0。"""
        from scripts.evaluation.ortools_comparison import _rl_avg_flow_time

        assert _rl_avg_flow_time({}) == 0.0
        assert _rl_avg_flow_time({"avg_wait_time": 3.0}) == 3.0
