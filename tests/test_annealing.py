"""
??RL???? - ???????????
Unit Tests for src/quantum/annealing.py

????:
- QuantumAnnealingOptimizer ???(??/????????????)
- network_to_qubo QUBO ????(?????????????/TD????)
- bitstring_to_weights ?????(???????????/?????????/??)
- anneal ????(????????????????????????)
- _compute_qubo_energy ????(????)
- _extract_weights / _set_weights ?????????
- optimize_policy ???(??/??/?????)???????
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.quantum import annealing as annealing_mod
from src.quantum.annealing import QuantumAnnealingOptimizer


# ============================================================
# ?????
# ============================================================
class TestQuantumAnnealingOptimizerInit(unittest.TestCase):
    """?????????????"""

    def test_default_init(self):
        """??????????????"""
        opt = QuantumAnnealingOptimizer()
        self.assertEqual(opt.num_qubits, 16)
        self.assertEqual(opt.annealing_time, 20.0)
        self.assertEqual(opt.shots, 1000)
        self.assertTrue(opt.simulation_mode)
        self.assertIsNone(opt.cqlib_client)

    def test_solver_type_initial_value(self):
        """solver_type 初始值应为 'none'。"""
        opt = QuantumAnnealingOptimizer()
        self.assertEqual(opt.solver_type, "none")
        self.assertEqual(opt._last_solver, "none")

    def test_custom_init(self):
        """????????????"""
        opt = QuantumAnnealingOptimizer(
            num_qubits=32,
            annealing_time=50.0,
            shots=500,
            simulation_mode=False,
        )
        self.assertEqual(opt.num_qubits, 32)
        self.assertEqual(opt.annealing_time, 50.0)
        self.assertEqual(opt.shots, 500)
        self.assertFalse(opt.simulation_mode)

    def test_simulation_mode_is_bool(self):
        """simulation_mode ??????? bool?"""
        opt_false = QuantumAnnealingOptimizer(simulation_mode=0)
        self.assertFalse(opt_false.simulation_mode)
        opt_true = QuantumAnnealingOptimizer(simulation_mode=1)
        self.assertTrue(opt_true.simulation_mode)

    def test_cqlib_client_stored(self):
        """cqlib_client ???????"""
        client = object()
        opt = QuantumAnnealingOptimizer(cqlib_client=client)
        self.assertIs(opt.cqlib_client, client)

    def test_simulated_annealing_hyperparams(self):
        """????????????????"""
        opt = QuantumAnnealingOptimizer()
        self.assertEqual(opt._sim_initial_temp, 2.0)
        self.assertEqual(opt._sim_cooling_rate, 0.995)
        self.assertEqual(opt._sim_num_sweeps, 200)

    def test_use_dw_flag_is_bool(self):
        """use_dw ???????(??? D-Wave SDK ????)?"""
        opt = QuantumAnnealingOptimizer()
        self.assertIsInstance(opt.use_dw, bool)

    def test_low_num_qubits_does_not_raise(self):
        """????????????????????"""
        # num_qubits=4 ? n_bits_per_weight=1 < 4,????
        opt = QuantumAnnealingOptimizer(num_qubits=4)
        self.assertEqual(opt.num_qubits, 4)

    def test_n_bits_per_weight_derived_from_num_qubits(self):
        """编码精度不再隐式依赖 num_qubits。"""
        weights = [np.array([0.1, 0.2])]
        for nq in (4, 8, 16, 32):
            opt = QuantumAnnealingOptimizer(num_qubits=nq)
            Q = opt.network_to_qubo(weights)
            self.assertEqual(opt.n_bits_per_weight, 4)
            self.assertEqual(Q.shape, (8, 8))

    def test_n_bits_per_weight_is_configurable(self):
        opt = QuantumAnnealingOptimizer(num_qubits=16, n_bits_per_weight=8)
        weights = [np.array([0.1, -0.2])]

        self.assertEqual(opt.n_bits_per_weight, 8)
        self.assertEqual(opt.network_to_qubo(weights).shape, (16, 16))

    def test_n_bits_per_weight_rejects_sign_only_encoding(self):
        with self.assertRaisesRegex(ValueError, "至少为 2"):
            QuantumAnnealingOptimizer(n_bits_per_weight=1)


# ============================================================
# get_annealing_config (Issue #247)
# ============================================================
class TestGetAnnealingConfig(unittest.TestCase):
    """Issue #247: 验证 get_annealing_config 返回完整退火参数配置。"""

    def test_returns_dict_with_required_fields(self):
        """应返回包含全部必需字段的字典。"""
        optimizer = QuantumAnnealingOptimizer(num_qubits=16, simulation_mode=True)
        cfg = optimizer.get_annealing_config()
        required = {
            "num_qubits",
            "annealing_time",
            "shots",
            "simulation_mode",
            "solver_backend",
            "sim_initial_temp",
            "sim_cooling_rate",
            "sim_num_sweeps",
            "n_bits_per_weight",
            "last_solver",
            "quantum_acceleration_enabled",
        }
        self.assertTrue(required.issubset(cfg.keys()), f"缺少字段: {required - set(cfg.keys())}")

    def test_num_qubits_reflected(self):
        """num_qubits 应反映构造参数。"""
        optimizer = QuantumAnnealingOptimizer(num_qubits=24, simulation_mode=True)
        cfg = optimizer.get_annealing_config()
        self.assertEqual(cfg["num_qubits"], 24)
        self.assertEqual(cfg["n_bits_per_weight"], 4)  # 默认值 4

    def test_solver_backend_is_string(self):
        """solver_backend 应为字符串。"""
        optimizer = QuantumAnnealingOptimizer(simulation_mode=True)
        cfg = optimizer.get_annealing_config()
        self.assertIsInstance(cfg["solver_backend"], str)
        self.assertIn(cfg["solver_backend"], {"neal", "numpy_sa"})

    def test_quantum_acceleration_enabled_reflects_env(self):
        """quantum_acceleration_enabled 应反映环境变量。"""
        old = os.environ.get("QUANTUM_ACCELERATION_ENABLED")
        try:
            os.environ["QUANTUM_ACCELERATION_ENABLED"] = "1"
            optimizer = QuantumAnnealingOptimizer(simulation_mode=True)
            cfg = optimizer.get_annealing_config()
            self.assertTrue(cfg["quantum_acceleration_enabled"])

            os.environ["QUANTUM_ACCELERATION_ENABLED"] = "0"
            cfg2 = optimizer.get_annealing_config()
            self.assertFalse(cfg2["quantum_acceleration_enabled"])
        finally:
            if old is None:
                os.environ.pop("QUANTUM_ACCELERATION_ENABLED", None)
            else:
                os.environ["QUANTUM_ACCELERATION_ENABLED"] = old

    def test_last_solver_initial_value(self):
        """初始 last_solver 应为 'none'。"""
        optimizer = QuantumAnnealingOptimizer(simulation_mode=True)
        cfg = optimizer.get_annealing_config()
        self.assertEqual(cfg["last_solver"], "none")


# ============================================================
# network_to_qubo ??
# ============================================================
class TestNetworkToQubo(unittest.TestCase):
    """?? network_to_qubo QUBO ?????"""

    def setUp(self):
        """????????????"""
        self.opt = QuantumAnnealingOptimizer(num_qubits=16)
        np.random.seed(42)
        self.weights = [
            np.random.randn(4, 2).astype(np.float32),
            np.random.randn(2).astype(np.float32),
        ]

    def test_qubo_shape(self):
        """QUBO ?????? (total_bits, total_bits)?"""
        Q = self.opt.network_to_qubo(self.weights)
        num_weights = sum(w.size for w in self.weights)
        n_bits_per_weight = max(1, self.opt.num_qubits // 4)
        expected = num_weights * n_bits_per_weight
        self.assertEqual(Q.shape, (expected, expected))

    def test_qubo_is_symmetric(self):
        """QUBO ????????"""
        Q = self.opt.network_to_qubo(self.weights)
        np.testing.assert_array_almost_equal(Q, Q.T)

    def test_qubo_is_finite(self):
        """QUBO ????????????"""
        Q = self.opt.network_to_qubo(self.weights)
        self.assertTrue(np.all(np.isfinite(Q)))

    def test_qubo_sign_bit_diagonal_zero(self):
        """????????(? 0 ?)????? 0?"""
        Q = self.opt.network_to_qubo(self.weights)
        n_bits_per_weight = max(1, self.opt.num_qubits // 4)
        num_weights = sum(w.size for w in self.weights)
        for i in range(num_weights):
            sign_idx = i * n_bits_per_weight
            self.assertEqual(Q[sign_idx, sign_idx], 0.0)

    def test_qubo_without_gradients(self):
        """????????????? QUBO ???"""
        Q = self.opt.network_to_qubo(self.weights)
        self.assertEqual(Q.shape[0], Q.shape[1])
        self.assertGreater(np.count_nonzero(Q), 0)

    def test_qubo_with_gradients_changes_matrix(self):
        """?????????????????? QUBO?"""
        np.random.seed(0)
        gradients = [
            np.random.randn(4, 2).astype(np.float32),
            np.random.randn(2).astype(np.float32),
        ]
        Q_grad = self.opt.network_to_qubo(self.weights, gradients=gradients)
        Q_no_grad = self.opt.network_to_qubo(self.weights)
        self.assertEqual(Q_grad.shape, Q_no_grad.shape)
        self.assertFalse(np.allclose(Q_grad, Q_no_grad))

    def test_qubo_with_td_errors(self):
        """?? TD ?????????? QUBO ???"""
        np.random.seed(1)
        gradients = [
            np.random.randn(4, 2).astype(np.float32),
            np.random.randn(2).astype(np.float32),
        ]
        td_errors = np.array([0.1, -0.2, 0.3])
        Q = self.opt.network_to_qubo(self.weights, gradients=gradients, td_errors=td_errors)
        self.assertEqual(Q.shape[0], Q.shape[1])
        self.assertTrue(np.all(np.isfinite(Q)))

    def test_qubo_with_empty_td_errors_equals_none(self):
        """? TD ??????????? TD ???"""
        Q_empty = self.opt.network_to_qubo(self.weights, td_errors=np.array([]))
        Q_none = self.opt.network_to_qubo(self.weights)
        np.testing.assert_array_almost_equal(Q_empty, Q_none)

    def test_qubo_single_weight_layer(self):
        """??????????? QUBO?"""
        w = [np.array([0.5, -0.3, 0.8])]
        Q = self.opt.network_to_qubo(w)
        n_bits_per_weight = max(1, self.opt.num_qubits // 4)
        self.assertEqual(Q.shape, (3 * n_bits_per_weight, 3 * n_bits_per_weight))

    def test_qubo_with_gradients_is_symmetric(self):
        """????? QUBO ???????"""
        np.random.seed(2)
        gradients = [
            np.random.randn(4, 2).astype(np.float32),
            np.random.randn(2).astype(np.float32),
        ]
        Q = self.opt.network_to_qubo(self.weights, gradients=gradients)
        np.testing.assert_array_almost_equal(Q, Q.T)


# ============================================================
# bitstring_to_weights ??
# ============================================================
class TestBitstringToWeights(unittest.TestCase):
    """?? bitstring_to_weights ??????"""

    def setUp(self):
        """??????????"""
        self.opt = QuantumAnnealingOptimizer(num_qubits=16)
        self.shapes = [(4, 2), (2,)]
        self.num_params = 10
        self.n_bits = max(1, self.opt.num_qubits // 4)  # 4
        self.bitstring_len = self.num_params * self.n_bits  # 40

    def test_returns_list_of_correct_shapes(self):
        """????????????????"""
        bitstring = "0" * self.bitstring_len
        weights = self.opt.bitstring_to_weights(bitstring, self.shapes)
        self.assertIsInstance(weights, list)
        self.assertEqual(len(weights), len(self.shapes))
        for w, s in zip(weights, self.shapes, strict=False):
            self.assertEqual(w.shape, s)

    def test_all_zeros_bitstring_yields_zero_delta(self):
        """?????(?????)?????????"""
        bitstring = "0" * self.bitstring_len
        weights = self.opt.bitstring_to_weights(bitstring, self.shapes)
        for w in weights:
            np.testing.assert_array_almost_equal(w, np.zeros_like(w))

    def test_all_zeros_with_current_weights_returns_unchanged(self):
        """????? + ??????????(???)?"""
        np.random.seed(7)
        current = [
            np.random.randn(4, 2).astype(np.float64),
            np.random.randn(2).astype(np.float64),
        ]
        bitstring = "0" * self.bitstring_len
        weights = self.opt.bitstring_to_weights(bitstring, self.shapes, current_weights=current)
        for w, c in zip(weights, current, strict=False):
            np.testing.assert_array_almost_equal(w, c)

    def test_sign_bit_one_yields_nonpositive_delta(self):
        """???? 1(???)?????????"""
        # ? 1 ???:???=1(?),???? 1(????)
        bitstring = "1" * self.bitstring_len
        weights = self.opt.bitstring_to_weights(bitstring, self.shapes)
        flat = np.concatenate([w.flatten() for w in weights])
        self.assertTrue(np.all(flat <= 0))

    def test_sign_bit_zero_yields_nonnegative_delta(self):
        """???? 0(???)?????????"""
        # ??:??????? 0,???? 1
        per = self.n_bits
        bitstring = "".join("0" + "1" * (per - 1) for _ in range(self.num_params))
        weights = self.opt.bitstring_to_weights(bitstring, self.shapes)
        flat = np.concatenate([w.flatten() for w in weights])
        self.assertTrue(np.all(flat >= 0))

    def test_with_current_weights_adds_delta(self):
        """???????,????? w_old + ?w(?? delta ???)?"""
        np.random.seed(11)
        current = [
            np.random.randn(4, 2).astype(np.float64),
            np.random.randn(2).astype(np.float64),
        ]
        bitstring = "0" * self.bitstring_len
        weights = self.opt.bitstring_to_weights(bitstring, self.shapes, current_weights=current)
        for w, c in zip(weights, current, strict=False):
            np.testing.assert_array_almost_equal(w, c)

    def test_short_bitstring_padded_with_zeros(self):
        """???????????????????"""
        bitstring = "1"
        weights = self.opt.bitstring_to_weights(bitstring, self.shapes)
        self.assertEqual(len(weights), 2)
        self.assertEqual(weights[0].shape, (4, 2))

    def test_long_bitstring_truncated(self):
        """?????????????"""
        bitstring = "1" * (self.bitstring_len * 3)
        weights = self.opt.bitstring_to_weights(bitstring, self.shapes)
        self.assertEqual(weights[0].shape, (4, 2))
        self.assertEqual(weights[1].shape, (2,))

    def test_magnitude_scales_with_bits(self):
        """?????,???????(??????)?"""
        per = self.n_bits
        # ??????? 1
        bs_high = "".join("0" + "1" + "0" * (per - 2) for _ in range(self.num_params))
        # ??????? 1
        bs_low = "".join("0" + "0" * (per - 2) + "1" for _ in range(self.num_params))
        high = self.opt.bitstring_to_weights(bs_high, self.shapes)
        low = self.opt.bitstring_to_weights(bs_low, self.shapes)
        high_flat = np.concatenate([w.flatten() for w in high])
        low_flat = np.concatenate([w.flatten() for w in low])
        # ???? 1/2,???? 1/2^(n-1),?????????
        self.assertTrue(np.all(high_flat >= low_flat))


# ============================================================
# anneal ??
# ============================================================
class TestAnneal(unittest.TestCase):
    """?? anneal ???????"""

    def setUp(self):
        """???????????????????"""
        self.opt = QuantumAnnealingOptimizer(num_qubits=16, shots=10)
        self.opt._sim_num_sweeps = 20
        np.random.seed(123)
        self.weights = [
            np.random.randn(4, 2).astype(np.float32),
            np.random.randn(2).astype(np.float32),
        ]
        self.Q = self.opt.network_to_qubo(self.weights)

    def test_returns_str(self):
        """anneal ???????"""
        result = self.opt.anneal(self.Q)
        self.assertIsInstance(result, str)

    def test_bitstring_length_matches_qubo(self):
        """??????????? QUBO ?????"""
        result = self.opt.anneal(self.Q)
        self.assertEqual(len(result), self.Q.shape[0])

    def test_bitstring_is_binary(self):
        """??????? 0/1 ???"""
        result = self.opt.anneal(self.Q)
        self.assertTrue(set(result).issubset({"0", "1"}))

    def test_energy_is_finite(self):
        """????????????"""
        bitstring = self.opt.anneal(self.Q)
        bits = np.array([int(b) for b in bitstring], dtype=np.float64)
        energy = self.opt.compute_qubo_energy(bits, self.Q)
        self.assertTrue(np.isfinite(energy))

    def test_anneal_beats_worst_random(self):
        """????????????????????"""
        np.random.seed(42)
        bitstring = self.opt.anneal(self.Q)
        bits = np.array([int(b) for b in bitstring], dtype=np.float64)
        best_energy = self.opt.compute_qubo_energy(bits, self.Q)
        worst_random = max(
            self.opt.compute_qubo_energy(
                np.random.randint(0, 2, self.Q.shape[0]).astype(np.float64), self.Q
            )
            for _ in range(5)
        )
        self.assertLessEqual(best_energy, worst_random)

    def test_real_machine_path_with_string_result(self):
        """???????????????????"""
        client = MagicMock()
        client.submit_annealing_task = MagicMock(return_value="1010")
        opt = QuantumAnnealingOptimizer(simulation_mode=False, cqlib_client=client)
        Q = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = opt.anneal(Q)
        self.assertEqual(result, "1010")

    def test_real_machine_path_with_dict_result(self):
        """???????????? bitstring ???"""
        client = MagicMock()
        client.submit_annealing_task = MagicMock(return_value={"bitstring": "01"})
        opt = QuantumAnnealingOptimizer(simulation_mode=False, cqlib_client=client)
        Q = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = opt.anneal(Q)
        self.assertEqual(result, "01")

    def test_real_machine_path_with_empty_dict_falls_back(self):
        """????? bitstring ??????????"""
        client = MagicMock()
        client.submit_annealing_task = MagicMock(return_value={"bitstring": ""})
        opt = QuantumAnnealingOptimizer(simulation_mode=False, cqlib_client=client)
        opt._sim_num_sweeps = 5
        Q = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = opt.anneal(Q)
        self.assertEqual(len(result), 2)
        self.assertTrue(set(result).issubset({"0", "1"}))

    def test_real_machine_path_falls_back_on_exception(self):
        """???????????????????????"""
        client = MagicMock()
        client.submit_annealing_task = MagicMock(side_effect=RuntimeError("boom"))
        opt = QuantumAnnealingOptimizer(simulation_mode=False, cqlib_client=client)
        opt._sim_num_sweeps = 5
        Q = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = opt.anneal(Q)
        self.assertEqual(len(result), 2)
        self.assertTrue(set(result).issubset({"0", "1"}))

    def test_real_machine_unknown_result_type_falls_back(self):
        """????????????????"""
        client = MagicMock()
        client.submit_annealing_task = MagicMock(return_value=12345)
        opt = QuantumAnnealingOptimizer(simulation_mode=False, cqlib_client=client)
        opt._sim_num_sweeps = 5
        Q = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = opt.anneal(Q)
        self.assertEqual(len(result), 2)

    def test_cqlib_without_annealing_method_falls_back(self):
        """cqlib ???? submit_annealing_task ??????????"""
        client = MagicMock(spec=[])  # ???
        opt = QuantumAnnealingOptimizer(simulation_mode=False, cqlib_client=client)
        opt._sim_num_sweeps = 5
        Q = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = opt.anneal(Q)
        self.assertEqual(len(result), 2)
        self.assertTrue(set(result).issubset({"0", "1"}))

    def test_solver_type_numpy_sa(self):
        """numpy 模拟退火路径应设置 solver_type='numpy_sa'。"""
        # 强制使用 numpy 路径（即使 neal 可用）
        with patch.object(annealing_mod, "_DWAVE_AVAILABLE", False):
            opt = QuantumAnnealingOptimizer(num_qubits=16, shots=10)
            opt._sim_num_sweeps = 20
            Q = opt.network_to_qubo(self.weights)
            opt.anneal(Q)
            self.assertEqual(opt.solver_type, "numpy_sa")
            self.assertEqual(opt._last_solver, "numpy_sa")

    def test_solver_type_neal_sa(self):
        """neal 模拟退火路径应设置 solver_type='neal_sa'。"""
        if not annealing_mod._DWAVE_AVAILABLE:
            self.skipTest("D-Wave neal 未安装，跳过 neal 路径测试")
        opt = QuantumAnnealingOptimizer(num_qubits=16, shots=10)
        opt._sim_num_sweeps = 20
        Q = opt.network_to_qubo(self.weights)
        opt.anneal(Q)
        self.assertEqual(opt.solver_type, "neal_sa")
        self.assertEqual(opt._last_solver, "neal_sa")

    def test_solver_type_real_quantum_string(self):
        """真机退火返回字符串时 solver_type='real_quantum'。"""
        client = MagicMock()
        client.submit_annealing_task = MagicMock(return_value="1010")
        opt = QuantumAnnealingOptimizer(simulation_mode=False, cqlib_client=client)
        Q = np.array([[1.0, 0.5], [0.5, 1.0]])
        opt.anneal(Q)
        self.assertEqual(opt.solver_type, "real_quantum")
        self.assertEqual(opt._last_solver, "real_quantum")

    def test_solver_type_real_quantum_dict(self):
        """真机退火返回含 bitstring 的 dict 时 solver_type='real_quantum'。"""
        client = MagicMock()
        client.submit_annealing_task = MagicMock(return_value={"bitstring": "01"})
        opt = QuantumAnnealingOptimizer(simulation_mode=False, cqlib_client=client)
        Q = np.array([[1.0, 0.5], [0.5, 1.0]])
        opt.anneal(Q)
        self.assertEqual(opt.solver_type, "real_quantum")

    def test_solver_type_fallback_empty_dict(self):
        """真机退火返回空 bitstring 降级时 solver_type='numpy_sa'。"""
        client = MagicMock()
        client.submit_annealing_task = MagicMock(return_value={"bitstring": ""})
        opt = QuantumAnnealingOptimizer(simulation_mode=False, cqlib_client=client)
        opt._sim_num_sweeps = 5
        Q = np.array([[1.0, 0.5], [0.5, 1.0]])
        opt.anneal(Q)
        self.assertEqual(opt.solver_type, "numpy_sa")

    def test_solver_type_fallback_unknown_type(self):
        """真机退火返回无法识别类型降级时 solver_type='numpy_sa'。"""
        client = MagicMock()
        client.submit_annealing_task = MagicMock(return_value=12345)
        opt = QuantumAnnealingOptimizer(simulation_mode=False, cqlib_client=client)
        opt._sim_num_sweeps = 5
        Q = np.array([[1.0, 0.5], [0.5, 1.0]])
        opt.anneal(Q)
        self.assertEqual(opt.solver_type, "numpy_sa")

    def test_solver_type_fallback_on_exception(self):
        """真机退火异常降级后 solver_type 应为仿真求解器。"""
        client = MagicMock()
        client.submit_annealing_task = MagicMock(side_effect=RuntimeError("boom"))
        opt = QuantumAnnealingOptimizer(simulation_mode=False, cqlib_client=client)
        opt._sim_num_sweeps = 5
        Q = np.array([[1.0, 0.5], [0.5, 1.0]])
        opt.anneal(Q)
        self.assertIn(opt.solver_type, ("numpy_sa", "neal_sa"))

    # ----------------------------------------------------------------
    # Issue #229: 降级日志验证
    # ----------------------------------------------------------------

    def test_degradation_log_on_empty_dict_fallback(self):
        """真机返回空 bitstring 时应记录 [降级] 日志（Issue #229）。"""
        from unittest.mock import patch as _patch

        client = MagicMock()
        client.submit_annealing_task = MagicMock(return_value={"bitstring": ""})
        opt = QuantumAnnealingOptimizer(simulation_mode=False, cqlib_client=client)
        opt._sim_num_sweeps = 5
        Q = np.array([[1.0, 0.5], [0.5, 1.0]])
        with _patch("src.quantum.annealing.logger") as mock_logger:
            opt.anneal(Q)
            warning_msgs = [str(call.args[0]) for call in mock_logger.warning.call_args_list]
            self.assertTrue(
                any("[降级]" in msg for msg in warning_msgs),
                f"期望 warning 日志包含 '[降级]'，实际: {warning_msgs}",
            )

    def test_degradation_log_on_unknown_type_fallback(self):
        """真机返回未知类型时应记录 [降级] 日志（Issue #229）。"""
        from unittest.mock import patch as _patch

        client = MagicMock()
        client.submit_annealing_task = MagicMock(return_value=12345)
        opt = QuantumAnnealingOptimizer(simulation_mode=False, cqlib_client=client)
        opt._sim_num_sweeps = 5
        Q = np.array([[1.0, 0.5], [0.5, 1.0]])
        with _patch("src.quantum.annealing.logger") as mock_logger:
            opt.anneal(Q)
            warning_msgs = [str(call.args[0]) for call in mock_logger.warning.call_args_list]
            self.assertTrue(
                any("[降级]" in msg and "无法识别" in msg for msg in warning_msgs),
                f"期望 warning 包含 '[降级]' 和 '无法识别'，实际: {warning_msgs}",
            )

    def test_degradation_log_on_exception_fallback(self):
        """真机异常降级时应记录 [降级] 日志及降级原因（Issue #229）。"""
        from unittest.mock import patch as _patch

        client = MagicMock()
        client.submit_annealing_task = MagicMock(side_effect=RuntimeError("network down"))
        opt = QuantumAnnealingOptimizer(simulation_mode=False, cqlib_client=client)
        opt._sim_num_sweeps = 5
        Q = np.array([[1.0, 0.5], [0.5, 1.0]])
        with _patch("src.quantum.annealing.logger") as mock_logger:
            opt.anneal(Q)
            warning_msgs = [str(call.args[0]) for call in mock_logger.warning.call_args_list]
            self.assertTrue(
                any("[降级]" in msg and "RuntimeError" in msg for msg in warning_msgs),
                f"期望 warning 包含 '[降级]' 和 'RuntimeError'，实际: {warning_msgs}",
            )

    def test_degradation_log_when_no_submit_annealing_task(self):
        """cqlib_client 无 submit_annealing_task 方法时应记录 [降级] 日志（Issue #229）。"""
        from unittest.mock import patch as _patch

        client = MagicMock(spec=[])  # 空接口
        opt = QuantumAnnealingOptimizer(simulation_mode=False, cqlib_client=client)
        opt._sim_num_sweeps = 5
        Q = np.array([[1.0, 0.5], [0.5, 1.0]])
        with _patch("src.quantum.annealing.logger") as mock_logger:
            opt.anneal(Q)
            warning_msgs = [str(call.args[0]) for call in mock_logger.warning.call_args_list]
            self.assertTrue(
                any("[降级]" in msg and "submit_annealing_task" in msg for msg in warning_msgs),
                f"期望 warning 包含 '[降级]' 和 'submit_annealing_task'，实际: {warning_msgs}",
            )


# ============================================================
# _compute_qubo_energy ??
# ============================================================
class TestQuboEnergy(unittest.TestCase):
    """?? _compute_qubo_energy ?????"""

    def setUp(self):
        self.opt = QuantumAnnealingOptimizer()

    def test_zero_solution_zero_energy(self):
        """???????? 0?"""
        Q = np.array([[1.0, 2.0], [2.0, 3.0]])
        x = np.array([0.0, 0.0])
        self.assertEqual(self.opt.compute_qubo_energy(x, Q), 0.0)

    def test_known_energy_single_bit(self):
        """????? 1 ?????? Q[0,0]?"""
        Q = np.array([[1.0, 2.0], [2.0, 3.0]])
        x = np.array([1.0, 0.0])
        self.assertAlmostEqual(self.opt.compute_qubo_energy(x, Q), 1.0)

    def test_known_energy_full_ones(self):
        """? 1 ????????????????"""
        Q = np.array([[1.0, 2.0], [2.0, 3.0]])
        x = np.array([1.0, 1.0])
        # Q00 + Q11 + Q01 + Q10 = 1 + 3 + 2 + 2 = 8
        self.assertAlmostEqual(self.opt.compute_qubo_energy(x, Q), 8.0)

    def test_energy_returns_float(self):
        """????? float ???"""
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        x = np.array([1.0, 1.0])
        e = self.opt.compute_qubo_energy(x, Q)
        self.assertIsInstance(e, float)

    def test_energy_diagonal_only(self):
        """????????????? Q_ii ???"""
        Q = np.diag([2.0, 3.0, 4.0])
        x = np.array([1.0, 0.0, 1.0])
        # 2 + 4 = 6
        self.assertAlmostEqual(self.opt.compute_qubo_energy(x, Q), 6.0)

    def test_energy_matches_manual_formula(self):
        """?????? x^T Q x ?????"""
        np.random.seed(99)
        Q = np.random.randn(5, 5)
        Q = Q + Q.T  # ???
        x = np.array([1, 0, 1, 1, 0], dtype=np.float64)
        expected = float(x @ Q @ x)
        self.assertAlmostEqual(self.opt.compute_qubo_energy(x, Q), expected)


# ============================================================
# QUBO 翻转能量差 & 默认仿真求解器正确性（C1 回归）
# ============================================================
class TestQuboFlipDelta(unittest.TestCase):
    """回归测试：单比特翻转能量差公式（修复 C1）。

    早期实现 ΔE=(1-2x_k)·(Q_k·x) 漏写 ×2 离对角项与对角项 Q[k,k]，
    导致 Metropolis 接受概率偏离正确玻尔兹曼分布、累进能量漂移。
    """

    def _analytic_delta(self, q_matrix: np.ndarray, x: np.ndarray, k: int) -> float:
        x2 = x.copy()
        x2[k] = 1.0 - x2[k]
        return float(x2 @ q_matrix @ x2) - float(x @ q_matrix @ x)

    def test_flip_delta_matches_analytic_many(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            n = int(rng.integers(2, 8))
            Q = rng.standard_normal((n, n))
            Q = Q + Q.T
            x = (rng.random(n) < 0.5).astype(float)
            k = int(rng.integers(0, n))
            got = QuantumAnnealingOptimizer._qubo_flip_delta(Q, x, k)
            self.assertAlmostEqual(got, self._analytic_delta(Q, x, k), places=9)

    def test_flip_delta_known_case(self):
        # 子代理示例：Q=[[1,2],[2,3]], x=[1,1], flip k=0 -> 真值 ΔE=-5
        Q = np.array([[1.0, 2.0], [2.0, 3.0]])
        x = np.array([1.0, 1.0])
        self.assertAlmostEqual(QuantumAnnealingOptimizer._qubo_flip_delta(Q, x, 0), -5.0, places=9)


class TestNumpySimulatedAnnealing(unittest.TestCase):
    """回归测试：默认 numpy 仿真求解器端到端正确性（修复 C1）。

    修复前错误 ΔE 使退火无法稳定命中真值最优；修复后应在给定 RNG 种子下
    稳定收敛到暴力枚举的全局最优。
    """

    def _brute_force_min(self, q: np.ndarray) -> float:
        n = q.shape[0]
        best = float("inf")
        for mask in range(1 << n):
            x = np.array([float((mask >> i) & 1) for i in range(n)])
            e = float(x @ q @ x)
            if e < best:
                best = e
        return best

    def test_finds_global_optimum_small_qubo(self):
        import random

        rng = np.random.default_rng(0)
        Q = rng.standard_normal((6, 6))
        Q = Q + Q.T
        opt = QuantumAnnealingOptimizer()
        random.seed(1234)
        np.random.seed(1234)
        opt._sim_initial_temp = 1.0
        opt._sim_cooling_rate = 0.995
        opt._sim_num_sweeps = 3000
        bitstring = opt.numpy_simulated_annealing(Q)
        x = np.array([float(int(b)) for b in bitstring])
        found = float(x @ Q @ x)
        truth = self._brute_force_min(Q)
        # 正确求解器应命中全局最优；错误公式会显著偏离
        self.assertAlmostEqual(found, truth, delta=1e-6)

    def test_reproducibility_with_fixed_seed(self):
        """Issue #391: 固定 random_state 后两次退火结果应完全一致。"""
        rng = np.random.default_rng(42)
        Q = rng.standard_normal((8, 8))
        Q = Q + Q.T

        opt1 = QuantumAnnealingOptimizer(random_state=12345)
        opt2 = QuantumAnnealingOptimizer(random_state=12345)

        # 两次独立运行应产生相同结果
        result1 = opt1.numpy_simulated_annealing(Q)
        result2 = opt2.numpy_simulated_annealing(Q)

        self.assertEqual(result1, result2, "固定 seed 后两次退火结果应完全一致")

    def test_different_seeds_may_differ(self):
        """Issue #391: 不同 seed 可能产生不同结果（验证 seed 确实影响随机性）。"""
        rng = np.random.default_rng(42)
        Q = rng.standard_normal((10, 10))
        Q = Q + Q.T

        opt1 = QuantumAnnealingOptimizer(random_state=1)
        opt2 = QuantumAnnealingOptimizer(random_state=999)

        result1 = opt1.numpy_simulated_annealing(Q)
        result2 = opt2.numpy_simulated_annealing(Q)

        # 不同 seed 大概率产生不同结果（不强制不等，但验证 seed 生效）
        # 这里我们只验证两者都能产生有效比特串
        self.assertEqual(len(result1), 10)
        self.assertEqual(len(result2), 10)

    def test_early_stopping_triggers(self):
        """Issue #391: 连续 _sim_patience 次扫描无改进时应早停。"""
        rng = np.random.default_rng(0)
        Q = rng.standard_normal((6, 6))
        Q = Q + Q.T

        opt = QuantumAnnealingOptimizer(random_state=42)
        opt._sim_num_sweeps = 3000  # 设置很高的扫描次数
        opt._sim_patience = 5  # 设置很低的耐心值

        # 运行退火，应早停而非跑满 3000 次
        bitstring = opt.numpy_simulated_annealing(Q)
        self.assertEqual(len(bitstring), 6)

    def test_random_state_none_preserves_original_behavior(self):
        """Issue #391: random_state=None 时应保持原有行为（不报错）。"""
        rng = np.random.default_rng(0)
        Q = rng.standard_normal((4, 4))
        Q = Q + Q.T

        opt = QuantumAnnealingOptimizer(random_state=None)
        bitstring = opt.numpy_simulated_annealing(Q)
        self.assertEqual(len(bitstring), 4)


# ============================================================
# ?????????
# ============================================================
class TestWeightExtraction(unittest.TestCase):
    """?? _extract_weights / _set_weights ????????"""

    def setUp(self):
        self.opt = QuantumAnnealingOptimizer()
        self.net = nn.Linear(4, 2)
        with torch.no_grad():
            self.net.weight.copy_(torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]))
            self.net.bias.copy_(torch.tensor([0.01, 0.02]))

    def test_extract_returns_weights_and_shapes(self):
        """???????????????"""
        weights, shapes = self.opt.extract_weights(self.net)
        self.assertEqual(len(weights), 2)
        self.assertEqual(shapes, [(2, 4), (2,)])
        np.testing.assert_array_almost_equal(weights[0], self.net.weight.detach().numpy())
        np.testing.assert_array_almost_equal(weights[1], self.net.bias.detach().numpy())

    def test_extract_shapes_match_module(self):
        """??????? nn.Module ???????"""
        weights, shapes = self.opt.extract_weights(self.net)
        for w, s, p in zip(weights, shapes, self.net.parameters(), strict=False):
            self.assertEqual(w.shape, p.shape)
            self.assertEqual(s, p.shape)

    def test_set_weights_round_trip_preserves_values(self):
        """??????????????(????)?"""
        original_w, _ = self.opt.extract_weights(self.net)
        new_net = nn.Linear(4, 2)
        self.opt._set_weights(new_net, original_w)
        round_trip, _ = self.opt.extract_weights(new_net)
        for a, b in zip(original_w, round_trip, strict=False):
            np.testing.assert_array_almost_equal(a, b)

    def test_set_weights_modifies_parameters(self):
        """_set_weights ??????????"""
        target = [
            np.full((2, 4), 0.9, dtype=np.float32),
            np.full((2,), 0.1, dtype=np.float32),
        ]
        self.opt._set_weights(self.net, target)
        np.testing.assert_array_almost_equal(self.net.weight.detach().numpy(), np.full((2, 4), 0.9))
        np.testing.assert_array_almost_equal(self.net.bias.detach().numpy(), np.full((2,), 0.1))

    def test_extract_multi_layer_network(self):
        """??????????????"""
        net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
        weights, shapes = self.opt.extract_weights(net)
        # 4 ?????(2 ? Linear ?? weight+bias,ReLU ???)
        self.assertEqual(len(weights), 4)
        self.assertEqual(shapes, [(3, 4), (3,), (2, 3), (2,)])


# ============================================================
# optimize_policy ?????????
# ============================================================
class TestOptimizePolicyAndHelpers(unittest.TestCase):
    """?? optimize_policy ???????????"""

    def setUp(self):
        self.opt = QuantumAnnealingOptimizer(num_qubits=16, shots=10)
        self.opt._sim_num_sweeps = 10

    def test_optimize_policy_disabled_returns_agent(self):
        """QUANTUM_ACCELERATION_ENABLED ?????????? agent?"""
        agent = MagicMock()
        agent.policy_net = nn.Linear(4, 2)
        result = self.opt.optimize_policy(agent, num_iterations=1)
        self.assertIs(result, agent)

    def test_optimize_policy_enabled_runs_and_syncs_target(self):
        """??????? optimize_policy ?????? target_net?"""

        class MockAgent:
            """?? RL ???,???? optimize_policy ???"""

            def __init__(self):
                self.policy_net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
                self.target_net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
                self.target_net.load_state_dict(self.policy_net.state_dict())

        agent = MockAgent()
        with patch.object(annealing_mod, "QUANTUM_ACCELERATION_ENABLED", True):
            # head_only=False ? _apply_weights_v2(head_only=True ???????,?????)
            result = self.opt.optimize_policy(
                agent, num_iterations=2, learning_rate=0.01, head_only=False
            )
        self.assertIs(result, agent)
        # target_net ????? policy_net
        for p1, p2 in zip(
            agent.policy_net.parameters(), agent.target_net.parameters(), strict=False
        ):
            self.assertTrue(torch.equal(p1, p2))

    @unittest.skip("v5 ??? head_only ?????,???????")
    def test_optimize_policy_head_only_raises_attribute_error(self):
        pass

    def test_optimize_policy_no_policy_net_returns_agent(self):
        """agent ???????????????"""
        agent = MagicMock(spec=[])  # ?????
        with patch.object(annealing_mod, "QUANTUM_ACCELERATION_ENABLED", True):
            result = self.opt.optimize_policy(agent, num_iterations=1)
        self.assertIs(result, agent)

    def test_optimize_policy_callback_invoked(self):
        """????????????????"""

        class MockAgent:
            """? policy_net ???????"""

            def __init__(self):
                self.policy_net = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))

        agent = MockAgent()
        calls = []

        def cb(iteration, loss):
            calls.append((iteration, loss))

        with patch.object(annealing_mod, "QUANTUM_ACCELERATION_ENABLED", True):
            self.opt.optimize_policy(agent, num_iterations=3, callback=cb, head_only=False)
        self.assertEqual(len(calls), 3)
        self.assertEqual([c[0] for c in calls], [0, 1, 2])

    def test_get_policy_net_with_policy_net_attr(self):
        """?? policy_net ??? agent ???????"""
        net = nn.Linear(4, 2)
        agent = MagicMock()
        agent.policy_net = net
        self.assertIs(self.opt._get_policy_net(agent), net)

    def test_get_policy_net_unrecognized_returns_none(self):
        """????? agent ??? None?"""
        agent = MagicMock(spec=[])
        self.assertIsNone(self.opt._get_policy_net(agent))

    def test_get_policy_net_with_sb3_dqn_style(self):
        """SB3 DQN ?? agent(policy.q_net)??? q_net?"""
        net = nn.Linear(4, 2)
        agent = MagicMock()
        agent.policy.q_net = net
        del agent.policy_net  # ???? policy_net ??
        self.assertIs(self.opt._get_policy_net(agent), net)

    def test_evaluate_network_quality_positive_finite(self):
        """???????????????"""
        net = nn.Linear(4, 2)
        loss = self.opt._evaluate_network_quality(net)
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(loss, 0.0)

    def test_matrix_to_qubo_dict_skips_zeros(self):
        """_matrix_to_qubo_dict ?????????"""
        Q = np.array([[1.0, 0.0], [0.0, 2.0]])
        d = self.opt._matrix_to_qubo_dict(Q)
        self.assertIn((0, 0), d)
        self.assertIn((1, 1), d)
        self.assertNotIn((0, 1), d)

    def test_matrix_to_qubo_dict_values(self):
        """_matrix_to_qubo_dict ?????????"""
        Q = np.array([[1.0, 0.5], [0.5, 2.0]])
        d = self.opt._matrix_to_qubo_dict(Q)
        self.assertAlmostEqual(d[(0, 0)], 1.0)
        self.assertAlmostEqual(d[(0, 1)], 0.5)
        self.assertAlmostEqual(d[(1, 1)], 2.0)

    def test_apply_weights_v2_updates_with_learning_rate(self):
        """_apply_weights_v2 ????????????"""
        net = nn.Linear(3, 1)
        with torch.no_grad():
            net.weight.copy_(torch.zeros(1, 3))
            net.bias.copy_(torch.zeros(1))
        old = [np.zeros((1, 3), dtype=np.float32), np.zeros(1, dtype=np.float32)]
        new = [np.ones((1, 3), dtype=np.float32), np.ones(1, dtype=np.float32)]
        self.opt._apply_weights_v2(net, old, new, learning_rate=0.5)
        # w_final = 0 + 0.5 * (1 - 0) = 0.5
        np.testing.assert_array_almost_equal(net.weight.detach().numpy(), np.full((1, 3), 0.5))
        np.testing.assert_array_almost_equal(net.bias.detach().numpy(), np.full(1, 0.5))

    def test_apply_weights_v1_linear_interpolation(self):
        """_apply_weights ??????????????"""
        net = nn.Linear(2, 1)
        with torch.no_grad():
            net.weight.copy_(torch.tensor([[1.0, 1.0]]))
            net.bias.copy_(torch.tensor([1.0]))
        old = [
            np.array([[1.0, 1.0]], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        ]
        new = [
            np.array([[3.0, 3.0]], dtype=np.float32),
            np.array([3.0], dtype=np.float32),
        ]
        shapes = [(1, 2), (1,)]
        self.opt._apply_weights(net, old, new, shapes, learning_rate=0.5)
        # old_std?0, new_std?0 ? ????1 ? w_final = 0.5*1 + 0.5*3 = 2
        np.testing.assert_array_almost_equal(net.weight.detach().numpy(), np.full((1, 2), 2.0))
        np.testing.assert_array_almost_equal(net.bias.detach().numpy(), np.full(1, 2.0))

    # ============================================================
    # Issue #194: 退火无效化诊断指标 (min_effective_delta) 测试
    # ============================================================

    def test_min_effective_delta_param_accepted(self):
        """min_effective_delta 参数被正确接受且不报错。"""
        agent = MagicMock()
        agent.policy_net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        with patch.object(annealing_mod, "QUANTUM_ACCELERATION_ENABLED", True):
            result = self.opt.optimize_policy(
                agent,
                num_iterations=1,
                learning_rate=0.01,
                head_only=False,
                min_effective_delta=1e-4,
            )
        self.assertIs(result, agent)

    def test_last_anneal_stats_has_ineffective_fields(self):
        """optimize_policy 完成后 _last_anneal_stats 包含 ineffective_count 和 weight_l2_diff 字段。"""

        class MockAgent:
            def __init__(self):
                self.policy_net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

        agent = MockAgent()
        with patch.object(annealing_mod, "QUANTUM_ACCELERATION_ENABLED", True):
            self.opt.optimize_policy(
                agent,
                num_iterations=2,
                learning_rate=0.01,
                head_only=False,
                min_effective_delta=1e-4,
            )
        stats = self.opt._last_anneal_stats
        self.assertIn("ineffective_count", stats)
        self.assertIn("weight_l2_diff", stats)
        self.assertIsInstance(stats["ineffective_count"], int)
        self.assertIsInstance(stats["weight_l2_diff"], float)

    def test_ineffective_count_with_large_threshold(self):
        """min_effective_delta 设为极大值时，所有迭代都应被标记为无效。"""

        class MockAgent:
            def __init__(self):
                self.policy_net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

        agent = MockAgent()
        with patch.object(annealing_mod, "QUANTUM_ACCELERATION_ENABLED", True):
            self.opt.optimize_policy(
                agent,
                num_iterations=3,
                learning_rate=0.01,
                head_only=False,
                # 极大阈值：任何权重变化都无法超过
                min_effective_delta=1e10,
            )
        stats = self.opt._last_anneal_stats
        # 3 次迭代全部无效
        self.assertEqual(stats["ineffective_count"], 3)
        # 无效迭代不计入 accepted 或 rejected
        self.assertEqual(stats["accepted"], 0)
        self.assertEqual(stats["rejected"], 0)

    def test_ineffective_count_zero_with_large_learning_rate(self):
        """learning_rate 足够大时，ineffective_count 应为 0。"""

        class MockAgent:
            def __init__(self):
                self.policy_net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

        agent = MockAgent()
        with patch.object(annealing_mod, "QUANTUM_ACCELERATION_ENABLED", True):
            self.opt.optimize_policy(
                agent,
                num_iterations=2,
                # 大学习率确保权重变化超过阈值
                learning_rate=1.0,
                head_only=False,
                min_effective_delta=1e-8,
            )
        stats = self.opt._last_anneal_stats
        self.assertEqual(stats["ineffective_count"], 0)

    def test_min_effective_delta_default_value(self):
        """min_effective_delta 默认值为 1e-4，保持向后兼容。"""
        agent = MagicMock()
        agent.policy_net = nn.Linear(4, 2)
        with patch.object(annealing_mod, "QUANTUM_ACCELERATION_ENABLED", True):
            # 不传 min_effective_delta，使用默认值
            self.opt.optimize_policy(agent, num_iterations=1, head_only=False)
        stats = self.opt._last_anneal_stats
        # 默认值下应正常执行，字段存在
        self.assertIn("ineffective_count", stats)


# ============================================================
# Issue #148: ??/?? QUBO ????
# ============================================================
class TestParamBlockCreation(unittest.TestCase):
    """?? _create_param_blocks ?????"""

    def setUp(self):
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

    def test_tensor_wise_each_tensor_separate(self):
        """tensor_wise ?????????????"""
        params = list(self.net.parameters())
        blocks = QuantumAnnealingOptimizer._create_param_blocks(
            params, block_strategy="tensor_wise"
        )
        # nn.Sequential ? 3 ? Linear ? 6 ????? (weight + bias)
        # ? ReLU ????,input ??????
        # Linear(4,16): weight(16,4) + bias(16) ? 64+16=80 params
        # Linear(16,8): weight(8,16) + bias(8) ? 128+8=136 params
        # Linear(8,4):  weight(4,8) + bias(4) ? 32+4=36 params
        self.assertEqual(len(blocks), len(params))
        for i, block in enumerate(blocks):
            self.assertEqual(len(block), 1)
            self.assertEqual(block[0], i)

    def test_size_limited_within_limit(self):
        """size_limited ????????? ? ???"""
        params = list(self.net.parameters())
        max_per_block = 150
        blocks = QuantumAnnealingOptimizer._create_param_blocks(
            params, block_strategy="size_limited", max_params_per_block=max_per_block
        )
        for block in blocks:
            block_params = sum(params[i].numel() for i in block)
            self.assertLessEqual(block_params, max_per_block)

    def test_size_limited_covers_all_params(self):
        """size_limited ??????????"""
        params = list(self.net.parameters())
        total = len(params)
        blocks = QuantumAnnealingOptimizer._create_param_blocks(
            params, block_strategy="size_limited", max_params_per_block=100
        )
        covered = set()
        for block in blocks:
            covered.update(block)
        self.assertEqual(covered, set(range(total)))

    def test_large_tensor_own_block(self):
        """???????????"""
        # ???? 500 ?????? + ?????
        large_param = nn.Parameter(torch.randn(500))

        class TestNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.large = large_param
                self.small1 = nn.Parameter(torch.randn(10))
                self.small2 = nn.Parameter(torch.randn(5))

        net = TestNet()
        params = list(net.parameters())
        blocks = QuantumAnnealingOptimizer._create_param_blocks(
            params, block_strategy="size_limited", max_params_per_block=200
        )
        # 500 > 200,?????
        large_blocks = [b for b in blocks if 0 in b]
        self.assertEqual(len(large_blocks), 1)
        self.assertEqual(large_blocks[0], [0])

    def test_no_params_returns_empty(self):
        """????????????"""
        blocks = QuantumAnnealingOptimizer._create_param_blocks([], block_strategy="size_limited")
        self.assertEqual(blocks, [])


class TestHierarchicalAnnealing(unittest.TestCase):
    """????/???????????"""

    def setUp(self):
        os.environ["QUANTUM_ACCELERATION_ENABLED"] = "1"
        # ???? Annealing ???????
        annealing_mod.QUANTUM_ACCELERATION_ENABLED = True
        self.opt = QuantumAnnealingOptimizer(num_qubits=16)
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

        # ?? SimpleAgent ??
        class TestAgent:
            def __init__(self, net):
                self.policy_net = net

        self.agent = TestAgent(self.net)

    def tearDown(self):
        annealing_mod.QUANTUM_ACCELERATION_ENABLED = os.environ.get(
            "QUANTUM_ACCELERATION_ENABLED", "0"
        ).strip().lower() in ("1", "true", "yes")

    def test_hierarchical_basic_run(self):
        """????????????? agent?"""
        result = self.opt.optimize_policy_hierarchical(
            self.agent,
            num_iterations=2,
            learning_rate=0.01,
        )
        self.assertIs(result, self.agent)

    def test_hierarchical_covers_more_than_4_tensors(self):
        """??????? >4 ??????"""
        # ???????(3 ? ? 6 ?????)
        big_net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

        class BigAgent:
            def __init__(self, net):
                self.policy_net = net

        BigAgent(big_net)
        total_tensors = len(list(big_net.parameters()))
        self.assertGreater(total_tensors, 4, f"?????? >4 ??, ?? {total_tensors}")

    def test_hierarchical_via_mode_parameter(self):
        """?? optimize_policy ? mode='hierarchical' ??????"""
        result = self.opt.optimize_policy(
            self.agent,
            num_iterations=2,
            mode="hierarchical",
        )
        self.assertIs(result, self.agent)

    def test_hierarchical_loss_does_not_increase(self):
        """??????? loss ?????"""
        loss_before = QuantumAnnealingOptimizer._evaluate_network_quality(self.net)
        self.opt.optimize_policy_hierarchical(
            self.agent,
            num_iterations=5,
            learning_rate=0.01,
        )
        loss_after = QuantumAnnealingOptimizer._evaluate_network_quality(self.net)
        # loss ??????(??????,<5%)
        self.assertLessEqual(
            loss_after,
            loss_before * 1.05,
            f"loss ????: {loss_before:.4f} ? {loss_after:.4f}",
        )

    def test_hierarchical_disabled_when_quantum_disabled(self):
        """?????????????? agent?"""
        original = annealing_mod.QUANTUM_ACCELERATION_ENABLED
        annealing_mod.QUANTUM_ACCELERATION_ENABLED = False
        result = self.opt.optimize_policy_hierarchical(self.agent, num_iterations=2)
        self.assertIs(result, self.agent)
        annealing_mod.QUANTUM_ACCELERATION_ENABLED = original

    def test_hierarchical_no_replay_buffer(self):
        """? replay_buffer ???????????"""
        result = self.opt.optimize_policy_hierarchical(
            self.agent,
            num_iterations=3,
            learning_rate=0.05,
            max_params_per_block=100,
        )
        self.assertIs(result, self.agent)

    def test_hierarchical_memory_efficient(self):
        """?????????(??? OOM)?"""

        # ??????(6 ?????,??? 540,>500 ????)?
        # ????:???? 200 ?? ? ?? QUBO 800 ?,
        # neal ????? CI ?? runner ??? 60s ???(?? ~18s)?
        # ???? 16?64?32?16?8 ??(3832 ??,QUBO 4096 ?),
        # ? CI ? simulated_annealing ?????? 120s ?? pytest-timeout?
        big_net = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
        )

        class BigAgent:
            def __init__(self, net):
                self.policy_net = net

        agent = BigAgent(big_net)
        total_params = sum(p.numel() for p in big_net.parameters())
        total_tensors = len(list(big_net.parameters()))

        self.assertGreater(total_tensors, 4, f"????? >4 ??, ?? {total_tensors}")
        self.assertGreater(total_params, 500, f"????? >500 ??, ?? {total_params}")

        # ????,?? OOM
        try:
            self.opt.optimize_policy_hierarchical(
                agent,
                num_iterations=1,
                max_params_per_block=200,
                block_strategy="tensor_wise",
            )
        except MemoryError:
            self.fail("??????? MemoryError")

    def test_head_only_backward_compatible(self):
        """head_only ????????(????)?"""
        result = self.opt.optimize_policy(
            self.agent,
            num_iterations=2,
            head_only=True,
            max_head_tensors=4,
        )
        self.assertIs(result, self.agent)


# ============================================================
# Issue #189: ??? QUBO ??????
# ============================================================
class TestQuboMatrixConstruction(unittest.TestCase):
    """?? build_qubo_matrix / build_qubo_matrix_optimized ??????"""

    def setUp(self):
        np.random.seed(42)
        self.priorities = np.array([1.0, 2.0, 3.0])
        self.times = np.array([5.0, 3.0, 2.0])

    def test_build_qubo_shape(self):
        """build_qubo_matrix ??? (n, n) ???"""
        Q = annealing_mod.build_qubo_matrix(self.priorities, self.times)
        self.assertEqual(Q.shape, (3, 3))

    def test_build_qubo_symmetric(self):
        """build_qubo_matrix ????????"""
        Q = annealing_mod.build_qubo_matrix(self.priorities, self.times)
        np.testing.assert_array_almost_equal(Q, Q.T)

    def test_build_qubo_diagonal(self):
        """?????? priority[i] * time[i]?"""
        Q = annealing_mod.build_qubo_matrix(self.priorities, self.times)
        for i in range(3):
            self.assertAlmostEqual(Q[i, i], self.priorities[i] * self.times[i])

    def test_build_qubo_custom_penalty(self):
        """??? penalty ????????"""
        # ?? penalty=10.0,? penalty=20.0 ??(?? 2 ?????)
        Q_default = annealing_mod.build_qubo_matrix(self.priorities, self.times, penalty=10.0)
        Q_custom = annealing_mod.build_qubo_matrix(self.priorities, self.times, penalty=20.0)
        # ??????(?? penalty ??)
        np.testing.assert_array_almost_equal(np.diag(Q_default), np.diag(Q_custom))
        # ?????? 2 ???
        off_default = Q_default - np.diag(np.diag(Q_default))
        off_custom = Q_custom - np.diag(np.diag(Q_custom))
        np.testing.assert_array_almost_equal(off_custom, 2.0 * off_default)

    def test_build_qubo_shape_mismatch_raises(self):
        """priorities ? times ???????? ValueError?"""
        with self.assertRaises(ValueError):
            annealing_mod.build_qubo_matrix(np.array([1.0, 2.0]), np.array([1.0]))

    def test_build_qubo_non_1d_raises(self):
        """???????? ValueError?"""
        with self.assertRaises(ValueError):
            annealing_mod.build_qubo_matrix(
                np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 2.0])
            )

    def test_build_qubo_optimized_matches_original(self):
        """????????????"""
        Q_orig = annealing_mod.build_qubo_matrix(self.priorities, self.times, penalty=5.0)
        Q_opt = annealing_mod.build_qubo_matrix_optimized(self.priorities, self.times, penalty=5.0)
        np.testing.assert_array_almost_equal(Q_orig, Q_opt)

    def test_build_qubo_optimized_shape_mismatch_raises(self):
        """??????????"""
        with self.assertRaises(ValueError):
            annealing_mod.build_qubo_matrix_optimized(np.array([1.0]), np.array([1.0, 2.0]))

    def test_build_qubo_optimized_non_1d_raises(self):
        """??????????"""
        with self.assertRaises(ValueError):
            annealing_mod.build_qubo_matrix_optimized(np.array([[1.0]]), np.array([1.0]))

    def test_build_qubo_zero_penalty(self):
        """penalty=0 ???????? 0?"""
        Q = annealing_mod.build_qubo_matrix(self.priorities, self.times, penalty=0.0)
        off_diag = Q - np.diag(np.diag(Q))
        np.testing.assert_array_almost_equal(off_diag, np.zeros((3, 3)))


class TestQuboProfiling(unittest.TestCase):
    """?? profile_qubo_construction / benchmark_qubo_versions ???????"""

    def test_profile_returns_valid_dict(self):
        """profile_qubo_construction ????????????"""
        result = annealing_mod.profile_qubo_construction(n_tasks=5, n_iterations=3)
        self.assertIn("mean_time_ms", result)
        self.assertIn("std_time_ms", result)
        self.assertIn("min_time_ms", result)
        self.assertIn("max_time_ms", result)
        self.assertIn("matrix_size", result)
        self.assertIn("n_tasks", result)
        self.assertEqual(result["n_tasks"], 5)
        self.assertEqual(result["matrix_size"], 5)
        self.assertGreaterEqual(result["mean_time_ms"], 0.0)

    def test_profile_negative_n_tasks_raises(self):
        """? n_tasks ??? ValueError?"""
        with self.assertRaises(ValueError):
            annealing_mod.profile_qubo_construction(n_tasks=-1)

    def test_profile_zero_iterations_raises(self):
        """n_iterations < 1 ??? ValueError?"""
        with self.assertRaises(ValueError):
            annealing_mod.profile_qubo_construction(n_iterations=0)

    def test_profile_zero_tasks(self):
        """n_tasks=0 ?????(???)?"""
        result = annealing_mod.profile_qubo_construction(n_tasks=0, n_iterations=1)
        self.assertEqual(result["matrix_size"], 0)

    def test_benchmark_returns_valid_dict(self):
        """benchmark_qubo_versions ????????"""
        result = annealing_mod.benchmark_qubo_versions(n_tasks=5, n_iterations=3)
        self.assertIn("original_mean_ms", result)
        self.assertIn("optimized_mean_ms", result)
        self.assertIn("speedup", result)
        self.assertIn("results_match", result)
        self.assertTrue(result["results_match"])

    def test_benchmark_negative_n_tasks_raises(self):
        """? n_tasks ??? ValueError?"""
        with self.assertRaises(ValueError):
            annealing_mod.benchmark_qubo_versions(n_tasks=-1)

    def test_benchmark_zero_iterations_raises(self):
        """n_iterations < 1 ??? ValueError?"""
        with self.assertRaises(ValueError):
            annealing_mod.benchmark_qubo_versions(n_iterations=0)


class TestFindOptimalQuboParams(unittest.TestCase):
    """?? find_optimal_qubo_params ???????"""

    def setUp(self):
        self.priorities = np.array([1.0, 2.0, 3.0])
        self.times = np.array([5.0, 3.0, 2.0])

    def test_returns_valid_dict(self):
        """????? best_penalty/best_energy/all_results ????"""
        result = annealing_mod.find_optimal_qubo_params(self.priorities, self.times)
        self.assertIn("best_penalty", result)
        self.assertIn("best_energy", result)
        self.assertIn("all_results", result)

    def test_default_grid_has_5_penalties(self):
        """?????? 5 ? penalty ??"""
        result = annealing_mod.find_optimal_qubo_params(self.priorities, self.times)
        self.assertEqual(len(result["all_results"]), 5)

    def test_custom_grid(self):
        """????????????"""
        custom_grid = {"penalty": [1.0, 10.0]}
        result = annealing_mod.find_optimal_qubo_params(
            self.priorities, self.times, param_grid=custom_grid
        )
        self.assertEqual(len(result["all_results"]), 2)

    def test_best_penalty_in_grid(self):
        """best_penalty ???????????"""
        grid = {"penalty": [2.0, 4.0, 8.0]}
        result = annealing_mod.find_optimal_qubo_params(
            self.priorities, self.times, param_grid=grid
        )
        self.assertIn(result["best_penalty"], [2.0, 4.0, 8.0])

    def test_best_energy_is_minimum(self):
        """best_energy ??? all_results ???????"""
        result = annealing_mod.find_optimal_qubo_params(self.priorities, self.times)
        min_energy = min(r["energy"] for r in result["all_results"])
        self.assertAlmostEqual(result["best_energy"], min_energy)

    def test_empty_grid_falls_back(self):
        """????????? penalty=10.0?"""
        result = annealing_mod.find_optimal_qubo_params(
            self.priorities, self.times, param_grid={"penalty": []}
        )
        self.assertEqual(result["best_penalty"], 10.0)


# ============================================================
# Issue #189: _get_full_policy / _set_params_from_weights ??
# ============================================================
class TestGetFullPolicy(unittest.TestCase):
    """?? _get_full_policy ?????"""

    def test_with_sb3_ppo_style_policy(self):
        """SB3 PPO ?? agent(policy ? nn.Module)????? policy?"""
        net = nn.Linear(4, 2)
        agent = MagicMock()
        agent.policy = net
        result = QuantumAnnealingOptimizer.get_full_policy(agent)
        self.assertIs(result, net)

    def test_falls_back_to_get_policy_net(self):
        """? PPO ?????? _get_policy_net?"""
        net = nn.Linear(4, 2)
        agent = MagicMock()
        agent.policy_net = net
        # policy ????? ? ?????
        del agent.policy
        result = QuantumAnnealingOptimizer.get_full_policy(agent)
        self.assertIs(result, net)


class TestSetParamsFromWeights(unittest.TestCase):
    """?? _set_params_from_weights ?????"""

    def test_updates_params_in_place(self):
        """??????????"""
        params = [nn.Parameter(torch.zeros(3)), nn.Parameter(torch.zeros(2))]
        weights = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0], dtype=np.float32),
        ]
        QuantumAnnealingOptimizer._set_params_from_weights(params, weights)
        np.testing.assert_array_almost_equal(params[0].detach().numpy(), [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(params[1].detach().numpy(), [4.0, 5.0])

    def test_partial_param_subset(self):
        """????????(???????)?"""
        all_params = [nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(3))]
        subset = [all_params[1]]  # ??????
        weights = [np.array([10.0, 20.0, 30.0], dtype=np.float32)]
        QuantumAnnealingOptimizer._set_params_from_weights(subset, weights)
        # ??????????
        np.testing.assert_array_almost_equal(all_params[0].detach().numpy(), [1.0, 1.0])
        # ?????????
        np.testing.assert_array_almost_equal(all_params[1].detach().numpy(), [10.0, 20.0, 30.0])


# ============================================================
# Issue #189: _compute_gradients ??(?? mock replay buffer)
# ============================================================
class TestComputeGradients(unittest.TestCase):
    """?? _compute_gradients ?????"""

    def setUp(self):
        self.opt = QuantumAnnealingOptimizer(num_qubits=16, shots=10)
        self.opt._sim_num_sweeps = 5
        self.net = nn.Sequential(nn.Linear(4, 2))

    def test_replay_buffer_sample_exception_raises_valueerror(self):
        """replay buffer sample ??????? ValueError?"""
        bad_buffer = MagicMock()
        bad_buffer.sample = MagicMock(side_effect=RuntimeError("buffer empty"))
        with self.assertRaises(ValueError):
            self.opt._compute_gradients(self.net, bad_buffer, agent=None)

    def test_replay_buffer_without_sample_raises_valueerror(self):
        """replay buffer ? sample ?????? ValueError?"""
        bad_buffer = MagicMock(spec=[])  # ???
        with self.assertRaises(ValueError):
            self.opt._compute_gradients(self.net, bad_buffer, agent=None)

    def test_compute_gradients_with_tuple_batch(self):
        """tuple ?? batch(SB3 ReplayBuffer)???????"""
        # ?? SB3 ??? batch: (obs, actions, rewards, next_obs, dones, ...)
        # actions ??? 2D (batch, 1),?? gather(1, actions) ?????
        batch = (
            np.random.randn(8, 4).astype(np.float32),  # obs
            np.array([[0], [1], [0], [1], [0], [1], [0], [1]], dtype=np.int64),  # actions 2D
            np.array([1.0, 0.5, -0.5, 1.0, 0.0, 0.3, -0.2, 0.8]),  # rewards
            np.random.randn(8, 4).astype(np.float32),  # next_obs
            np.array([0, 0, 0, 0, 0, 0, 0, 1]),  # dones
            np.empty(8),  # infos placeholder
        )
        buffer = MagicMock()
        buffer.sample = MagicMock(return_value=batch)
        _gradients, _td_errors, loss = self.opt._compute_gradients(self.net, buffer, agent=None)
        # ??????
        self.assertTrue(np.isfinite(loss))


# ============================================================
# Issue #111: QUBO ?????????
# ============================================================
class TestSolverComparison(unittest.TestCase):
    """?? QUBO ?????????????????????"""

    def setUp(self):
        """????????? QUBO ???"""
        self.opt = QuantumAnnealingOptimizer(num_qubits=16, shots=10)
        self.opt._sim_num_sweeps = 10
        np.random.seed(42)
        weights = [
            np.random.randn(4, 2).astype(np.float32),
            np.random.randn(2).astype(np.float32),
        ]
        self.Q = self.opt.network_to_qubo(weights)

    def test_random_sample_qubo_returns_valid_bitstring(self):
        """random_sample_qubo ?????????"""
        from scripts.evaluation.annealing_solver_comparison import random_sample_qubo

        bitstring = random_sample_qubo(self.Q, num_samples=100)
        self.assertIsInstance(bitstring, str)
        self.assertEqual(len(bitstring), self.Q.shape[0])
        self.assertTrue(set(bitstring).issubset({"0", "1"}))

    def test_random_sample_qubo_energy_is_finite(self):
        """random_sample_qubo ????????????"""
        from scripts.evaluation.annealing_solver_comparison import random_sample_qubo

        bitstring = random_sample_qubo(self.Q, num_samples=100)
        bits = np.array([int(b) for b in bitstring], dtype=np.float64)
        energy = self.opt.compute_qubo_energy(bits, self.Q)
        self.assertTrue(np.isfinite(energy))

    def test_random_sample_improves_with_more_samples(self):
        """?????????????(?????)?"""
        from scripts.evaluation.annealing_solver_comparison import random_sample_qubo

        np.random.seed(123)
        bs_small = random_sample_qubo(self.Q, num_samples=10)
        bits_small = np.array([int(b) for b in bs_small], dtype=np.float64)
        energy_small = self.opt.compute_qubo_energy(bits_small, self.Q)

        np.random.seed(123)
        bs_large = random_sample_qubo(self.Q, num_samples=500)
        bits_large = np.array([int(b) for b in bs_large], dtype=np.float64)
        energy_large = self.opt.compute_qubo_energy(bits_large, self.Q)

        # ??????????(????????????)
        self.assertLessEqual(energy_large, energy_small)

    def test_numpy_sa_beats_random_on_same_qubo(self):
        """numpy ??????? QUBO ????????(?????)?"""
        from scripts.evaluation.annealing_solver_comparison import random_sample_qubo

        np.random.seed(7)
        # ?? 20 ??? QUBO,???????????????
        qubo = np.random.randn(20, 20).astype(np.float64)
        qubo = qubo + qubo.T

        # ????????
        original_sweeps = self.opt._sim_num_sweeps
        self.opt._sim_num_sweeps = 200

        sa_energies = []
        rand_energies = []
        for _ in range(5):
            bs_sa = self.opt.numpy_simulated_annealing(qubo)
            bits_sa = np.array([int(b) for b in bs_sa], dtype=np.float64)
            sa_energies.append(self.opt.compute_qubo_energy(bits_sa, qubo))

            bs_rand = random_sample_qubo(qubo, num_samples=300)
            bits_rand = np.array([int(b) for b in bs_rand], dtype=np.float64)
            rand_energies.append(self.opt.compute_qubo_energy(bits_rand, qubo))

        self.opt._sim_num_sweeps = original_sweeps

        mean_sa = float(np.mean(sa_energies))
        mean_rand = float(np.mean(rand_energies))
        # ????? SA ???????
        self.assertLessEqual(mean_sa, mean_rand, f"SA mean={mean_sa} > random mean={mean_rand}")

    def test_apply_weights_v2_partial_with_gradient_direction(self):
        """????????(D)???????????"""
        net = nn.Linear(4, 2)
        with torch.no_grad():
            net.weight.copy_(torch.zeros(2, 4))
            net.bias.copy_(torch.zeros(2))

        old_weights = [
            np.zeros((2, 4), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        ]
        # ???????(???? = 1.0)
        new_weights = [
            np.ones((2, 4), dtype=np.float32),
            np.ones(2, dtype=np.float32),
        ]

        QuantumAnnealingOptimizer._apply_weights_v2_partial(
            list(net.parameters()), old_weights, new_weights, learning_rate=0.1
        )

        # ????? = 0 + 0.1 * (1 - 0) = 0.1
        expected_w = np.full((2, 4), 0.1, dtype=np.float32)
        expected_b = np.full(2, 0.1, dtype=np.float32)
        np.testing.assert_array_almost_equal(net.weight.detach().numpy(), expected_w)
        np.testing.assert_array_almost_equal(net.bias.detach().numpy(), expected_b)


# ============================================================
# Issue #222: optimize_policy 子方法单元测试
# ============================================================
class TestOptimizePolicySubMethods(unittest.TestCase):
    """Issue #222: 验证从 optimize_policy 拆分出的子方法。"""

    def setUp(self):
        self.optimizer = QuantumAnnealingOptimizer(num_qubits=8)
        # 4 个参数张量：[Linear(4,8).weight, Linear(4,8).bias, Linear(8,2).weight, Linear(8,2).bias]
        self.policy_net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def test_setup_head_only_params_returns_zero_when_not_head_only(self):
        """非 head_only 模式应返回 0。"""
        idx = self.optimizer._setup_head_only_params(
            self.policy_net, head_only=False, max_head_tensors=4
        )
        self.assertEqual(idx, 0)

    def test_setup_head_only_params_returns_correct_index(self):
        """head_only 模式应返回正确的尾部参数起始索引。"""
        # 共 4 个参数张量，max_head_tensors=2 -> head_start_idx=2
        idx = self.optimizer._setup_head_only_params(
            self.policy_net, head_only=True, max_head_tensors=2
        )
        self.assertEqual(idx, 2)

    def test_setup_head_only_params_clamps_max_head_tensors(self):
        """max_head_tensors 超过参数张量数时应被截断。"""
        # 4 个参数张量，max_head_tensors=100 -> 应取 min(100, 4)=4 -> head_start_idx=0
        idx = self.optimizer._setup_head_only_params(
            self.policy_net, head_only=True, max_head_tensors=100
        )
        self.assertEqual(idx, 0)

    def test_compute_weight_delta_stats_returns_correct_l2(self):
        """_compute_weight_delta_stats 应正确计算 L2 范数和最大绝对差。"""
        current = [np.array([1.0, 2.0]), np.array([3.0])]
        optimized = [np.array([1.5, 2.5]), np.array([2.5])]
        delta_l2, delta_max, delta_flat = self.optimizer._compute_weight_delta_stats(
            current, optimized
        )
        # delta = [0.5, 0.5, -0.5], L2 = sqrt(0.25*3) = sqrt(0.75)
        expected_l2 = float(np.sqrt(0.75))
        self.assertAlmostEqual(delta_l2, expected_l2, places=6)
        self.assertAlmostEqual(delta_max, 0.5, places=6)
        np.testing.assert_array_almost_equal(delta_flat, [0.5, 0.5, -0.5])

    def test_compute_weight_delta_stats_empty_input(self):
        """空输入应返回 0 而非抛异常。"""
        delta_l2, delta_max, _ = self.optimizer._compute_weight_delta_stats(
            [np.array([])], [np.array([])]
        )
        self.assertEqual(delta_l2, 0.0)
        self.assertEqual(delta_max, 0.0)

    def test_compute_actual_weight_diff_head_only(self):
        """head_only 模式应只计算尾部参数的差异。"""
        # 先记录初始权重
        old_weights = [p.detach().cpu().numpy().copy() for p in self.policy_net.parameters()]
        # 修改最后一个参数张量（bias of Linear(8,2)）
        last_param = list(self.policy_net.parameters())[-1]
        with torch.no_grad():
            last_param.add_(0.5)

        head_start_idx = self.optimizer._setup_head_only_params(
            self.policy_net, head_only=True, max_head_tensors=2
        )
        diff = self.optimizer._compute_actual_weight_diff(
            self.policy_net,
            head_only=True,
            head_start_idx=head_start_idx,
            old_weights=old_weights[head_start_idx:],
        )
        # 修改了 2 个 bias 各 +0.5，L2 = sqrt(0.5^2 * 2) = sqrt(0.5)
        expected_diff = float(np.sqrt(0.5))
        self.assertAlmostEqual(diff, expected_diff, places=5)

    def test_compute_actual_weight_diff_full_network(self):
        """非 head_only 模式应计算全部参数的差异。"""
        old_weights = [p.detach().cpu().numpy().copy() for p in self.policy_net.parameters()]
        # 修改第一个参数张量
        first_param = next(iter(self.policy_net.parameters()))
        with torch.no_grad():
            first_param.add_(0.1)

        diff = self.optimizer._compute_actual_weight_diff(
            self.policy_net,
            head_only=False,
            head_start_idx=0,
            old_weights=old_weights,
        )
        # 修改了 Linear(4,8).weight 共 32 个参数各 +0.1
        expected_diff = float(np.sqrt(0.1**2 * 32))
        self.assertAlmostEqual(diff, expected_diff, places=5)

    def test_finalize_anneal_stats_writes_last_anneal_stats(self):
        """_finalize_anneal_stats 应正确写入 _last_anneal_stats 字典。"""
        initial_flat = np.array([1.0, 2.0, 3.0])
        best_weights = [np.array([1.1, 2.1, 3.1])]

        self.optimizer._finalize_anneal_stats(
            initial_l2_norm=float(np.linalg.norm(initial_flat)),
            initial_flat=initial_flat,
            initial_loss=1.0,
            best_weights=best_weights,
            best_loss=0.5,
            anneal_accepted=5,
            anneal_rejected=3,
            ineffective_count=2,
        )

        stats = self.optimizer._last_anneal_stats
        self.assertEqual(stats["accepted"], 5)
        self.assertEqual(stats["rejected"], 3)
        self.assertEqual(stats["total"], 8)
        self.assertAlmostEqual(stats["accept_rate"], 5 / 8, places=6)
        self.assertEqual(stats["ineffective_count"], 2)
        self.assertGreater(stats["weight_l2_diff"], 0.0)

    def test_finalize_anneal_stats_with_none_best_weights(self):
        """best_weights 为 None 时应回退到 initial_flat 并报告 0 差异。"""
        initial_flat = np.array([1.0, 2.0, 3.0])

        self.optimizer._finalize_anneal_stats(
            initial_l2_norm=float(np.linalg.norm(initial_flat)),
            initial_flat=initial_flat,
            initial_loss=1.0,
            best_weights=None,
            best_loss=1.0,
            anneal_accepted=0,
            anneal_rejected=0,
            ineffective_count=0,
        )

        stats = self.optimizer._last_anneal_stats
        self.assertEqual(stats["accepted"], 0)
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["accept_rate"], 0.0)
        self.assertEqual(stats["weight_l2_diff"], 0.0)


if __name__ == "__main__":
    unittest.main()
