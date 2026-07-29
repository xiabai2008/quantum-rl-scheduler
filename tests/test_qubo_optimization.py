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
from hypothesis import given, settings
from hypothesis import strategies as st

from src.quantum.annealing import (
    QuantumAnnealingOptimizer,
    benchmark_qubo_versions,
    build_qubo_matrix,
    build_qubo_matrix_optimized,
    find_optimal_qubo_params,
    profile_qubo_construction,
)


def _exact_minimum_bitstring(qubo: np.ndarray) -> str:
    """穷举小型 QUBO，返回确定性的最低能量比特串。"""
    n_variables = qubo.shape[0]
    candidates = (
        np.array(list(np.binary_repr(value, width=n_variables)), dtype=np.float64)
        for value in range(2**n_variables)
    )
    best = min(candidates, key=lambda bits: float(bits @ qubo @ bits))
    return "".join(str(int(bit)) for bit in best)


@st.composite
def weight_gradient_pairs(
    draw: st.DrawFn,
    *,
    min_size: int = 1,
    max_size: int = 4,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """生成有限、非零梯度的随机权重与梯度。"""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    finite_float = st.floats(
        min_value=-1.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
    nonzero_gradient = st.one_of(
        st.floats(
            min_value=-2.0,
            max_value=-0.25,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        st.floats(
            min_value=0.25,
            max_value=2.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
    )
    weights = draw(st.lists(finite_float, min_size=size, max_size=size))
    gradients = draw(st.lists(nonzero_gradient, min_size=size, max_size=size))
    return (
        [np.asarray(weights, dtype=np.float64)],
        [np.asarray(gradients, dtype=np.float64)],
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


# =============================================================================
# Issue #253: QUBO 形式化 / 属性验证（CI 通过 -k "formal or property" 选中）
# 注意：pytest 的 -k 在本仓库区分大小写，因此测试“方法名”中显式包含
# 小写子串 "formal" / "property"，确保 CI 步骤能稳定命中。
# 本组历史测试仅使用 numpy + 标准库；Issue #252 的 Hypothesis 测试见文件末尾。
# =============================================================================


class TestQuboFormalProperties:
    """QUBO 矩阵的数学形式化性质（对称性、能量公式）"""

    def test_qubo_formal_matrix_symmetry(self) -> None:
        # QUBO 矩阵必须对称且为实矩阵（对任意随机输入）。
        rng = np.random.default_rng(seed=2024)
        for _ in range(25):
            n = int(rng.integers(4, 16))
            priorities = rng.uniform(1.0, 10.0, size=n)
            times = rng.uniform(1.0, 20.0, size=n)
            penalty = float(rng.uniform(1.0, 50.0))
            qubo = build_qubo_matrix_optimized(priorities, times, penalty=penalty)
            assert np.all(np.isreal(qubo)), "QUBO 矩阵应为实数矩阵"
            assert np.allclose(qubo, qubo.T, atol=1e-12), "QUBO 矩阵必须对称"

    def test_qubo_formal_energy_formula(self) -> None:
        # 对二进制向量 x，能量 x^T Q x 必须等于
        # sum_i Q[i,i]*x[i] + 2 * sum_{i<j} Q[i,j]*x[i]*x[j]。
        rng = np.random.default_rng(seed=99)
        for _ in range(15):
            n = int(rng.integers(4, 12))
            priorities = rng.uniform(1.0, 10.0, size=n)
            times = rng.uniform(1.0, 20.0, size=n)
            penalty = float(rng.uniform(1.0, 50.0))
            qubo = build_qubo_matrix_optimized(priorities, times, penalty=penalty)
            for _ in range(10):
                x = rng.integers(0, 2, size=n).astype(np.float64)
                energy = float(x @ qubo @ x)
                diag = sum(qubo[i, i] * x[i] for i in range(n))
                off = 2.0 * sum(qubo[i, j] * x[i] * x[j] for i in range(n) for j in range(i + 1, n))
                expected = diag + off
                assert abs(energy - expected) < 1e-9, f"能量公式不匹配: {energy} vs {expected}"


class TestQuboPropertyBased:
    """QUBO 的属性测试（随机大量赋值的能量有限性 / 对称性）"""

    def test_qubo_property_energy_finite_for_valid_assignment(self) -> None:
        # 对大量随机 0/1 向量，能量应保持有限且为实数。
        rng = np.random.default_rng(seed=7)
        n = 12
        priorities = rng.uniform(1.0, 10.0, size=n)
        times = rng.uniform(1.0, 20.0, size=n)
        qubo = build_qubo_matrix_optimized(priorities, times, penalty=5.0)
        for _ in range(50):
            x = rng.integers(0, 2, size=n).astype(np.float64)
            energy = float(x @ qubo @ x)
            assert np.isfinite(energy), "能量必须为有限值"
            assert np.isreal(energy), "能量必须为实数"

    def test_qubo_property_network_to_qubo_symmetric(self) -> None:
        """验证 QuantumAnnealingOptimizer.network_to_qubo 生成的 QUBO 矩阵对称。

        network_to_qubo 是实例方法，需实例化 QuantumAnnealingOptimizer 后调用。
        """
        from src.quantum.annealing import QuantumAnnealingOptimizer

        optimizer = QuantumAnnealingOptimizer(simulation_mode=True)
        rng = np.random.default_rng(seed=2024)
        for _ in range(20):
            n = int(rng.integers(4, 14))
            # network_to_qubo 接收权重列表，生成 QUBO 矩阵
            weights = [rng.uniform(-1.0, 1.0, size=n) for _ in range(2)]
            qubo = optimizer.network_to_qubo(weights)
            assert np.all(np.isreal(qubo)), "QUBO 矩阵应为实数矩阵"
            assert np.allclose(qubo, qubo.T, atol=1e-12), "QUBO 矩阵必须对称"


class TestNetworkQuboHypothesisProperties:
    """Issue #252：用 100 个随机用例验证网络权重 QUBO 性质。"""

    @given(weight_gradient_pairs(max_size=1))
    @settings(max_examples=100, deadline=None)
    def test_property_optimum_follows_descent_direction(
        self,
        problem: tuple[list[np.ndarray], list[np.ndarray]],
    ) -> None:
        weights, gradients = problem
        optimizer = QuantumAnnealingOptimizer(
            n_bits_per_weight=4,
            simulation_mode=True,
        )
        # 注意：main 分支的 network_to_qubo 使用硬编码 reg_lambda=0.1，
        # 不再暴露 regularization_strength 参数；默认正则化下梯度项仍主导最优方向。
        qubo = optimizer.network_to_qubo(
            weights,
            gradients=gradients,
        )

        decoded = optimizer.bitstring_to_weights(
            _exact_minimum_bitstring(qubo),
            [weights[0].shape],
        )[0]
        descent_direction = -gradients[0]

        assert float(np.dot(decoded, descent_direction)) > 0

    @given(
        deltas=st.lists(
            st.floats(
                min_value=-0.1,
                max_value=0.1,
                allow_nan=False,
                allow_infinity=False,
                width=64,
            ),
            min_size=1,
            max_size=16,
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_encoding_roundtrip_within_precision(
        self,
        deltas: list[float],
    ) -> None:
        optimizer = QuantumAnnealingOptimizer(n_bits_per_weight=8)
        original = np.asarray(deltas, dtype=np.float64)
        max_delta = 0.1

        encoded = optimizer.weight_deltas_to_bitstring([original], max_delta=max_delta)
        decoded = optimizer.bitstring_to_weights(encoded, [original.shape])[0]
        encoding_precision = max_delta / (2 ** (optimizer.n_bits_per_weight - 1))

        assert np.all(np.abs(decoded - original) <= encoding_precision + 1e-12)

    # 注：原 Issue #252 还设计了 test_property_regularization_energy_is_nonnegative，
    # 通过对比 regularization_strength=0 与 regularization_strength=λ 的 QUBO 能量差来
    # 验证 L2 正则化贡献非负。main 分支已将 reg_lambda 硬编码为 0.1（不再暴露该参数），
    # 无法构建零正则化基线，故移除该用例。正则化能量 λ·||Δw||²≥0 由构造保证。


# =============================================================================
# Issue #251: QUBO 形式化验证（梯度一致性/能量单调性/往返一致性/正则化效果）
# =============================================================================


def _build_qubo_with_lambda(
    weights: np.ndarray,
    gradients: np.ndarray,
    n_bits_per_weight: int,
    reg_lambda: float,
) -> np.ndarray:
    """使用指定正则化系数 λ 手动构建 QUBO 矩阵。

    遵循 ``network_to_qubo`` 的数学公式（不含跨权重耦合项）：
        QUBO 目标: min_x  g^T Δw(x) + λ * ||Δw(x)||^2

    用于验证 λ 增大时更新幅度减小的正则化效果。
    注意：``network_to_qubo`` 中 ``reg_lambda`` 硬编码为 0.1，无法通过公开接口调整，
    故此处手动构建 QUBO 以测试不同 λ 值的效果。
    """
    flat_weights = weights.flatten().astype(np.float64)
    flat_gradients = gradients.flatten().astype(np.float64)
    num_weights = flat_weights.size

    grad_abs_max = np.max(np.abs(flat_gradients)) + 1e-8
    flat_grad_norm = flat_gradients / grad_abs_max

    weight_std = np.std(flat_weights) + 1e-8
    max_delta = weight_std * 0.1

    total_bits = num_weights * n_bits_per_weight
    qubo = np.zeros((total_bits, total_bits), dtype=np.float64)
    magnitude_bits = n_bits_per_weight - 1

    for i in range(num_weights):
        g_norm = flat_grad_norm[i]
        target = -g_norm
        base_idx = i * n_bits_per_weight

        # 对角项 + 符号-数值耦合项
        for bit_k in range(n_bits_per_weight):
            global_idx = base_idx + bit_k
            if bit_k == 0:
                # 符号位对角项为 0
                qubo[global_idx, global_idx] = 0.0
            else:
                mag_idx = bit_k - 1
                bit_val = max_delta / (2 ** (mag_idx + 1))
                qubo[global_idx, global_idx] = -target * bit_val + reg_lambda * bit_val * bit_val

        # 符号位与数值位的耦合
        sign_idx = base_idx
        for mag_idx in range(magnitude_bits):
            bit_k = 1 + mag_idx
            bit_val = max_delta / (2 ** (mag_idx + 1))
            qubo[sign_idx, bit_k] = 2.0 * target * bit_val
            qubo[bit_k, sign_idx] = qubo[sign_idx, bit_k]

        # 数值位之间的耦合（L2 正则化二次项）
        for mk1 in range(magnitude_bits):
            for mk2 in range(mk1 + 1, magnitude_bits):
                b1 = 1 + mk1
                b2 = 1 + mk2
                val1 = max_delta / (2 ** (mk1 + 1))
                val2 = max_delta / (2 ** (mk2 + 1))
                coupling = 2.0 * reg_lambda * val1 * val2
                qubo[b1 + base_idx, b2 + base_idx] = coupling
                qubo[b2 + base_idx, b1 + base_idx] = coupling

    return qubo


class TestQuboFormalVerification:
    """Issue #251: QUBO 形式化验证测试。

    验证 QUBO 优化的数学正确性：
    - 梯度下降一致性：最优解码方向与梯度下降一致
    - 能量单调性：沿梯度下降方向 QUBO 能量单调递减
    - 往返一致性：权重→比特串→解码权重的往返一致
    - 正则化效果：λ 增大时更新幅度减小
    """

    def test_qubo_optimal_matches_gradient_descent(self) -> None:
        """构造已知梯度，验证 QUBO 最优解码方向与梯度下降一致。

        QUBO 目标函数: min_x  g^T Δw(x) + λ * ||Δw(x)||^2
        最优解的 Δw 应与 -g（梯度下降方向）同向，即 <Δw, -g> > 0。
        覆盖全正/全负/混合梯度方向多组场景。
        """
        optimizer = QuantumAnnealingOptimizer(
            n_bits_per_weight=4,
            simulation_mode=True,
        )
        rng = np.random.default_rng(seed=42)

        test_cases = [
            # (权重, 梯度) — 覆盖正/负/混合梯度方向
            (np.array([0.5, -0.3, 0.8, 0.1]), np.array([1.0, 1.0, 1.0, 1.0])),
            (np.array([0.5, -0.3, 0.8, 0.1]), np.array([-1.0, -1.0, -1.0, -1.0])),
            (np.array([0.5, -0.3, 0.8, 0.1]), np.array([1.0, -1.0, 1.0, -1.0])),
            (np.array([0.2, 0.4, -0.1, 0.3]), rng.uniform(-1.0, 1.0, size=4)),
        ]

        for weights_arr, gradient_arr in test_cases:
            weights = [weights_arr.astype(np.float64)]
            gradients = [gradient_arr.astype(np.float64)]

            qubo = optimizer.network_to_qubo(weights, gradients=gradients)
            optimal_bitstring = _exact_minimum_bitstring(qubo)
            decoded_delta = optimizer.bitstring_to_weights(
                optimal_bitstring,
                [weights[0].shape],
            )[0]

            descent_direction = -gradients[0]
            dot_product = float(np.dot(decoded_delta, descent_direction))
            assert dot_product > 0, (
                f"梯度={gradient_arr}, 解码Δw={decoded_delta}, 与下降方向点积={dot_product} 应为正"
            )

    def test_qubo_energy_monotonicity(self) -> None:
        """验证沿梯度下降方向 QUBO 能量单调递减。

        构造一系列在下降方向上幅度递增的比特串，
        验证 QUBO 能量随幅度增大而递减（线性项主导区域）。
        """
        optimizer = QuantumAnnealingOptimizer(
            n_bits_per_weight=4,
            simulation_mode=True,
        )

        # 4 个权重（非零方差），强正梯度 -> 下降方向为负更新
        weights = [np.array([0.5, 0.1, 0.9, 0.3], dtype=np.float64)]
        gradients = [np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)]

        qubo = optimizer.network_to_qubo(weights, gradients=gradients)

        num_weights = weights[0].size  # 4

        # 构造下降方向（sign=1，负更新）上幅度递增的 per-weight 比特模式：
        # "1000"(mag=0) < "1001"(mag=1/8) < "1011"(mag=3/8) < "1111"(mag=7/8)
        per_weight_patterns = ["1000", "1001", "1011", "1111"]

        # 将 per-weight 模式拼接为全权重比特串
        bitstrings = ["".join(p) * num_weights for p in per_weight_patterns]

        energies = []
        for bs in bitstrings:
            bits = np.array([int(b) for b in bs], dtype=np.float64)
            energy = optimizer.compute_qubo_energy(bits, qubo)
            energies.append(energy)

        # 验证能量单调递减
        for i in range(len(energies) - 1):
            assert energies[i] >= energies[i + 1], (
                f"能量未单调递减: E({per_weight_patterns[i]})={energies[i]:.6f} "
                f"< E({per_weight_patterns[i + 1]})={energies[i + 1]:.6f}"
            )

        # 最大幅度下降能量应严格小于零更新能量
        assert energies[-1] < energies[0], "最大幅度下降能量应小于零更新能量"

    def test_qubo_encoding_roundtrip(self) -> None:
        """验证权重→比特串→解码权重的往返一致性（在编码精度范围内）。

        使用 weight_deltas_to_bitstring 编码，bitstring_to_weights 解码，
        验证往返误差不超过 1 LSB 对应的精度。
        """
        optimizer = QuantumAnnealingOptimizer(n_bits_per_weight=8)
        max_delta = 0.1

        # 构造多种 delta 值：正/负/零/边界/小值/非整除值
        test_deltas = np.array(
            [
                0.0,  # 零
                0.05,  # 正中等
                -0.05,  # 负中等
                0.1,  # 正最大（边界）
                -0.1,  # 负最大（边界）
                0.001,  # 正小值
                -0.001,  # 负小值
                0.099,  # 接近最大
                -0.099,  # 接近负最大
                0.033,  # 非整除值
                -0.067,  # 非整除值
            ],
            dtype=np.float64,
        )

        encoded = optimizer.weight_deltas_to_bitstring([test_deltas], max_delta=max_delta)
        decoded = optimizer.bitstring_to_weights(encoded, [test_deltas.shape])[0]

        # 编码精度 = max_delta / 2^(magnitude_bits)
        magnitude_bits = optimizer.n_bits_per_weight - 1
        encoding_precision = max_delta / (2**magnitude_bits)

        errors = np.abs(decoded - test_deltas)
        assert np.all(errors <= encoding_precision + 1e-12), (
            f"往返误差 {errors.max():.8f} 超过编码精度 {encoding_precision:.8f}"
        )

    def test_qubo_regularization_effect(self) -> None:
        """验证 λ 增大时更新幅度减小。

        注意: ``network_to_qubo`` 中 ``reg_lambda`` 硬编码为 0.1，无法通过公开接口调整。
        本测试使用与 ``network_to_qubo`` 相同的数学公式手动构建 QUBO（不含跨权重耦合），
        对比不同 λ 值下最优解的更新幅度，验证 L2 正则化的抑制效果。
        限制说明详见 PR body。
        """
        optimizer = QuantumAnnealingOptimizer(
            n_bits_per_weight=4,
            simulation_mode=True,
        )

        # 非零方差权重 + 强梯度，确保线性项和正则化项都可观测
        weights = np.array([0.5, 0.1, 0.9, 0.3], dtype=np.float64)
        gradients = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        # 弱正则化 vs 强正则化
        lambda_weak = 0.001
        lambda_strong = 1000.0

        qubo_weak = _build_qubo_with_lambda(weights, gradients, 4, lambda_weak)
        qubo_strong = _build_qubo_with_lambda(weights, gradients, 4, lambda_strong)

        # 暴力搜索最优比特串（16 bits = 65536 种组合，可接受）
        optimal_weak = _exact_minimum_bitstring(qubo_weak)
        optimal_strong = _exact_minimum_bitstring(qubo_strong)

        # 解码为权重更新量（传入 current_weights 以使用与 QUBO 一致的 max_delta）
        decoded_weak = optimizer.bitstring_to_weights(
            optimal_weak, [weights.shape], current_weights=[weights]
        )[0]
        decoded_strong = optimizer.bitstring_to_weights(
            optimal_strong, [weights.shape], current_weights=[weights]
        )[0]

        delta_weak = decoded_weak - weights
        delta_strong = decoded_strong - weights

        magnitude_weak = float(np.linalg.norm(delta_weak))
        magnitude_strong = float(np.linalg.norm(delta_strong))

        # 强正则化下更新幅度应严格小于弱正则化
        assert magnitude_strong < magnitude_weak, (
            f"λ增大时更新幅度应减小: "
            f"||Δw(λ={lambda_weak})||={magnitude_weak:.8f}, "
            f"||Δw(λ={lambda_strong})||={magnitude_strong:.8f}"
        )


# =============================================================================
# Issue #362: 大规模 QUBO formal/property 测试（n≥64 不变量）
# =============================================================================


class TestQuboFormalLargeScale:
    """Issue #362: n≥64 规模 QUBO 形式化不变量验证。

    覆盖三类不变量：
    - 能量下界：任意 0/1 向量的 QUBO 能量 ≥ sum(min(0, Q[i,i]))
    - penalty 单调性：penalty 增大时，约束违反的能量惩罚增大
    - QUBO↔分配对应关系：x^T Q x 与逐元素求和一致
    """

    def test_formal_energy_lower_bound_n64(self) -> None:
        """n=64 时，任意 0/1 向量的 QUBO 能量 ≥ 对角负项之和（下界）。

        对于对称 QUBO 矩阵 Q，能量 x^T Q x = sum_i Q[i,i] x_i + 2 sum_{i<j} Q[i,j] x_i x_j。
        当所有 x_i=1 时取得对角项总和；下界为 sum_i min(0, Q[i,i])
        （选 x_i=1 当 Q[i,i]<0，否则 x_i=0）。
        """
        rng = np.random.default_rng(seed=362)
        n = 64
        priorities = rng.uniform(1.0, 10.0, size=n)
        times = rng.uniform(1.0, 20.0, size=n)
        penalty = float(rng.uniform(1.0, 50.0))
        qubo = build_qubo_matrix_optimized(priorities, times, penalty=penalty)

        # 理论下界：sum of min(0, diagonal)
        diagonal = np.diag(qubo)
        lower_bound = float(np.sum(np.minimum(0.0, diagonal)))

        for _ in range(20):
            x = rng.integers(0, 2, size=n).astype(np.float64)
            energy = float(x @ qubo @ x)
            assert energy >= lower_bound - 1e-9, f"能量 {energy} 低于理论下界 {lower_bound}"

    def test_formal_penalty_monotonicity_n64(self) -> None:
        """n=64 时，penalty 增大 → 约束违反的 QUBO 能量增大。

        固定一个违反约束的分配 x（如全 1，即所有任务同时调度），
        增大 penalty 时该分配的能量应单调递增。
        """
        rng = np.random.default_rng(seed=3621)
        n = 64
        priorities = rng.uniform(1.0, 10.0, size=n)
        times = rng.uniform(1.0, 20.0, size=n)

        # 全 1 分配（违反"同时只能选一个"约束）
        x = np.ones(n, dtype=np.float64)

        penalties = [1.0, 5.0, 10.0, 50.0, 100.0]
        energies = []
        for p in penalties:
            qubo = build_qubo_matrix_optimized(priorities, times, penalty=p)
            energy = float(x @ qubo @ x)
            energies.append(energy)

        # 验证能量随 penalty 增大而单调递增
        for i in range(len(energies) - 1):
            assert energies[i + 1] >= energies[i] - 1e-9, (
                f"penalty {penalties[i]}→{penalties[i + 1]}: "
                f"能量 {energies[i]}→{energies[i + 1]} 未单调递增"
            )

    def test_property_qubo_assignment_correspondence_n128(self) -> None:
        """n=128 时，x^T Q x 与逐元素求和一致（QUBO↔分配对应关系）。

        验证向量化能量计算 (x @ Q @ x) 与显式公式
        sum_i Q[i,i]*x_i + 2*sum_{i<j} Q[i,j]*x_i*x_j 数值一致。
        """
        rng = np.random.default_rng(seed=3622)
        n = 128
        priorities = rng.uniform(1.0, 10.0, size=n)
        times = rng.uniform(1.0, 20.0, size=n)
        penalty = float(rng.uniform(1.0, 50.0))
        qubo = build_qubo_matrix_optimized(priorities, times, penalty=penalty)

        for _ in range(5):
            x = rng.integers(0, 2, size=n).astype(np.float64)
            # 向量化计算
            energy_vec = float(x @ qubo @ x)
            # 显式公式计算
            diag_sum = float(np.sum(np.diag(qubo) * x))
            # 上三角部分 × 2
            upper = np.triu(qubo, k=1)
            off_diag_sum = 2.0 * float(x @ upper @ x)
            energy_explicit = diag_sum + off_diag_sum
            assert abs(energy_vec - energy_explicit) < 1e-6, (
                f"n={n}: 向量化能量 {energy_vec} 与显式公式 {energy_explicit} 不一致"
            )

    def test_formal_matrix_symmetry_n64(self) -> None:
        """n=64 时，QUBO 矩阵保持对称（大规模不变量）。"""
        rng = np.random.default_rng(seed=3623)
        n = 64
        priorities = rng.uniform(1.0, 10.0, size=n)
        times = rng.uniform(1.0, 20.0, size=n)
        penalty = float(rng.uniform(1.0, 50.0))
        qubo = build_qubo_matrix_optimized(priorities, times, penalty=penalty)
        assert np.allclose(qubo, qubo.T, atol=1e-12), "n=64 QUBO 矩阵不对称"
        assert np.all(np.isreal(qubo)), "n=64 QUBO 矩阵含非实数"

    def test_property_energy_finite_n64(self) -> None:
        """n=64 时，任意 0/1 分配的 QUBO 能量为有限实数（大规模属性）。"""
        rng = np.random.default_rng(seed=3624)
        n = 64
        priorities = rng.uniform(1.0, 10.0, size=n)
        times = rng.uniform(1.0, 20.0, size=n)
        penalty = float(rng.uniform(1.0, 50.0))
        qubo = build_qubo_matrix_optimized(priorities, times, penalty=penalty)
        for _ in range(50):
            x = rng.integers(0, 2, size=n).astype(np.float64)
            energy = float(x @ qubo @ x)
            assert np.isfinite(energy), "能量必须为有限值"
            assert np.isreal(energy), "能量必须为实数"
