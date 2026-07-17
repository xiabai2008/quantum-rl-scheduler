"""
Issue #45: QUBO 矩阵构建性能剖析与加速 — 测试模块

测试内容：
    - TestProfileQuboConstruction  : 剖析函数返回字段与计时正确性
    - TestBuildQuboMatrixOptimized : 向量化构建正确性、与原版一致
    - TestBenchmarkQuboVersions    : 性能对比、加速比、结果一致性
    - TestFindOptimalQuboParams    : 最优 penalty 网格搜索
    - TestQuboEdgeCases            : 单任务、空任务、大任务数、形状校验
    - TestQuboMatrixProperties     : 对称性、对角线值、非负性、非对角公式
"""

import numpy as np
import pytest

from src.quantum.annealing import (
    benchmark_qubo_versions,
    build_qubo_matrix,
    build_qubo_matrix_optimized,
    find_optimal_qubo_params,
    profile_qubo_construction,
)


@pytest.fixture
def random_tasks() -> tuple[np.ndarray, np.ndarray]:
    """生成随机任务优先级与处理时间（固定种子，可复现）"""
    rng = np.random.default_rng(seed=2024)
    priorities = rng.uniform(1.0, 10.0, size=10)
    times = rng.uniform(1.0, 20.0, size=10)
    return priorities, times


class TestProfileQuboConstruction:
    """剖析 QUBO 矩阵构建性能"""

    def test_returns_complete_fields(self) -> None:
        result = profile_qubo_construction(n_tasks=10, n_iterations=20)
        expected_keys = {
            "mean_time_ms",
            "std_time_ms",
            "min_time_ms",
            "max_time_ms",
            "matrix_size",
            "n_tasks",
        }
        assert expected_keys.issubset(result.keys())

    def test_timings_positive(self) -> None:
        result = profile_qubo_construction(n_tasks=10, n_iterations=20)
        assert result["mean_time_ms"] > 0
        assert result["min_time_ms"] > 0
        assert result["max_time_ms"] > 0

    def test_matrix_size_correct(self) -> None:
        result = profile_qubo_construction(n_tasks=10, n_iterations=20)
        assert result["matrix_size"] == 10
        assert result["n_tasks"] == 10

    def test_min_le_mean_le_max(self) -> None:
        result = profile_qubo_construction(n_tasks=8, n_iterations=15)
        assert result["min_time_ms"] <= result["mean_time_ms"] <= result["max_time_ms"]

    def test_std_non_negative(self) -> None:
        result = profile_qubo_construction(n_tasks=10, n_iterations=20)
        assert result["std_time_ms"] >= 0


class TestBuildQuboMatrixOptimized:
    """向量化 QUBO 构建正确性"""

    def test_shape(self, random_tasks: tuple[np.ndarray, np.ndarray]) -> None:
        priorities, times = random_tasks
        qubo = build_qubo_matrix_optimized(priorities, times)
        assert qubo.shape == (10, 10)

    def test_matches_original(self, random_tasks: tuple[np.ndarray, np.ndarray]) -> None:
        priorities, times = random_tasks
        qubo_orig = build_qubo_matrix(priorities, times)
        qubo_opt = build_qubo_matrix_optimized(priorities, times)
        assert np.allclose(qubo_orig, qubo_opt)

    def test_matches_original_custom_penalty(self) -> None:
        priorities = np.array([1.0, 2.0, 3.0, 4.0])
        times = np.array([4.0, 3.0, 2.0, 1.0])
        for penalty in [0.0, 1.0, 5.0, 10.0, 100.0]:
            qubo_orig = build_qubo_matrix(priorities, times, penalty=penalty)
            qubo_opt = build_qubo_matrix_optimized(priorities, times, penalty=penalty)
            assert np.allclose(qubo_orig, qubo_opt), f"penalty={penalty} 时两版结果不一致"

    def test_returns_float64(self, random_tasks: tuple[np.ndarray, np.ndarray]) -> None:
        priorities, times = random_tasks
        qubo = build_qubo_matrix_optimized(priorities, times)
        assert qubo.dtype == np.float64


class TestBenchmarkQuboVersions:
    """原版 vs 优化版性能对比"""

    def test_returns_complete_fields(self) -> None:
        result = benchmark_qubo_versions(n_tasks=10, n_iterations=20)
        expected_keys = {"original_mean_ms", "optimized_mean_ms", "speedup", "results_match"}
        assert expected_keys.issubset(result.keys())

    def test_results_match(self) -> None:
        result = benchmark_qubo_versions(n_tasks=10, n_iterations=20)
        assert result["results_match"] is True

    def test_speedup_positive(self) -> None:
        result = benchmark_qubo_versions(n_tasks=10, n_iterations=20)
        assert result["speedup"] > 0

    def test_speedup_gt_one_large_n(self) -> None:
        # 大规模下 numpy 向量化应快于 Python 双重循环
        result = benchmark_qubo_versions(n_tasks=100, n_iterations=30)
        assert result["speedup"] > 1.0


class TestFindOptimalQuboParams:
    """最优 penalty 网格搜索"""

    def test_returns_best_penalty(self) -> None:
        rng = np.random.default_rng(seed=7)
        priorities = rng.uniform(1.0, 10.0, size=8)
        times = rng.uniform(1.0, 20.0, size=8)
        result = find_optimal_qubo_params(priorities, times)
        assert "best_penalty" in result
        assert "best_energy" in result
        assert "all_results" in result

    def test_best_penalty_in_grid(self) -> None:
        rng = np.random.default_rng(seed=7)
        priorities = rng.uniform(1.0, 10.0, size=8)
        times = rng.uniform(1.0, 20.0, size=8)
        grid = {"penalty": [1.0, 5.0, 10.0, 50.0, 100.0]}
        result = find_optimal_qubo_params(priorities, times, param_grid=grid)
        assert result["best_penalty"] in grid["penalty"]

    def test_all_results_length(self) -> None:
        rng = np.random.default_rng(seed=7)
        priorities = rng.uniform(1.0, 10.0, size=8)
        times = rng.uniform(1.0, 20.0, size=8)
        grid = {"penalty": [1.0, 5.0, 10.0]}
        result = find_optimal_qubo_params(priorities, times, param_grid=grid)
        assert len(result["all_results"]) == 3
        for item in result["all_results"]:
            assert "penalty" in item
            assert "energy" in item

    def test_best_energy_is_min(self) -> None:
        rng = np.random.default_rng(seed=7)
        priorities = rng.uniform(1.0, 10.0, size=8)
        times = rng.uniform(1.0, 20.0, size=8)
        grid = {"penalty": [1.0, 5.0, 10.0, 50.0]}
        result = find_optimal_qubo_params(priorities, times, param_grid=grid)
        energies = [r["energy"] for r in result["all_results"]]
        assert result["best_energy"] == pytest.approx(min(energies))

    def test_default_grid(self) -> None:
        # 不传 param_grid 时使用默认网格（5 个值）
        rng = np.random.default_rng(seed=7)
        priorities = rng.uniform(1.0, 10.0, size=5)
        times = rng.uniform(1.0, 20.0, size=5)
        result = find_optimal_qubo_params(priorities, times)
        assert len(result["all_results"]) == 5


class TestQuboEdgeCases:
    """边界情况"""

    def test_single_task(self) -> None:
        priorities = np.array([5.0])
        times = np.array([3.0])
        qubo_orig = build_qubo_matrix(priorities, times)
        qubo_opt = build_qubo_matrix_optimized(priorities, times)
        assert qubo_orig.shape == (1, 1)
        assert qubo_opt.shape == (1, 1)
        assert np.allclose(qubo_orig, qubo_opt)
        # 单任务：Q[0,0] = priority * time = 15.0
        assert qubo_opt[0, 0] == pytest.approx(15.0)

    def test_empty_tasks(self) -> None:
        priorities = np.array([], dtype=np.float64)
        times = np.array([], dtype=np.float64)
        qubo_orig = build_qubo_matrix(priorities, times)
        qubo_opt = build_qubo_matrix_optimized(priorities, times)
        assert qubo_orig.shape == (0, 0)
        assert qubo_opt.shape == (0, 0)
        assert np.allclose(qubo_orig, qubo_opt)

    def test_large_n_tasks_100(self) -> None:
        rng = np.random.default_rng(seed=99)
        n = 100
        priorities = rng.uniform(1.0, 10.0, size=n)
        times = rng.uniform(1.0, 20.0, size=n)
        qubo_orig = build_qubo_matrix(priorities, times)
        qubo_opt = build_qubo_matrix_optimized(priorities, times)
        assert qubo_opt.shape == (100, 100)
        assert np.allclose(qubo_orig, qubo_opt)

    def test_shape_mismatch_raises(self) -> None:
        priorities = np.array([1.0, 2.0, 3.0])
        times = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            build_qubo_matrix(priorities, times)
        with pytest.raises(ValueError):
            build_qubo_matrix_optimized(priorities, times)

    def test_2d_input_raises(self) -> None:
        priorities = np.array([[1.0, 2.0], [3.0, 4.0]])
        times = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError):
            build_qubo_matrix(priorities, times)
        with pytest.raises(ValueError):
            build_qubo_matrix_optimized(priorities, times)


class TestQuboMatrixProperties:
    """矩阵属性：对称性、对角线、非负性、非对角公式"""

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(seed=55)
        priorities = rng.uniform(1.0, 10.0, size=12)
        times = rng.uniform(1.0, 20.0, size=12)
        qubo = build_qubo_matrix_optimized(priorities, times, penalty=7.0)
        assert np.allclose(qubo, qubo.T)

    def test_diagonal_values(self) -> None:
        rng = np.random.default_rng(seed=55)
        priorities = rng.uniform(1.0, 10.0, size=12)
        times = rng.uniform(1.0, 20.0, size=12)
        qubo = build_qubo_matrix_optimized(priorities, times, penalty=7.0)
        expected_diag = priorities * times
        assert np.allclose(np.diag(qubo), expected_diag)

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(seed=55)
        priorities = rng.uniform(1.0, 10.0, size=12)
        times = rng.uniform(1.0, 20.0, size=12)
        qubo = build_qubo_matrix_optimized(priorities, times, penalty=10.0)
        assert np.all(qubo >= 0)

    def test_off_diagonal_formula(self) -> None:
        priorities = np.array([1.0, 2.0, 3.0])
        times = np.array([4.0, 5.0, 6.0])
        penalty = 10.0
        qubo = build_qubo_matrix_optimized(priorities, times, penalty=penalty)
        # Q[0,1] = 0.5 * penalty * (p0*t1 + p1*t0)
        expected_01 = 0.5 * penalty * (1.0 * 5.0 + 2.0 * 4.0)
        assert qubo[0, 1] == pytest.approx(expected_01)
        # 对称性
        assert qubo[0, 1] == pytest.approx(qubo[1, 0])
