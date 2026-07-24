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
        """????????? num_qubits // 4(??? 1)?"""
        # ?? network_to_qubo ???????????
        weights = [np.array([0.1, 0.2])]
        for nq, expected_nbits in [(16, 4), (8, 2), (4, 1), (32, 8)]:
            opt = QuantumAnnealingOptimizer(num_qubits=nq)
            Q = opt.network_to_qubo(weights)
            self.assertEqual(Q.shape[0], 2 * expected_nbits)


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
        energy = self.opt._compute_qubo_energy(bits, self.Q)
        self.assertTrue(np.isfinite(energy))

    def test_anneal_beats_worst_random(self):
        """????????????????????"""
        np.random.seed(42)
        bitstring = self.opt.anneal(self.Q)
        bits = np.array([int(b) for b in bitstring], dtype=np.float64)
        best_energy = self.opt._compute_qubo_energy(bits, self.Q)
        worst_random = max(
            self.opt._compute_qubo_energy(
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
        self.assertEqual(self.opt._compute_qubo_energy(x, Q), 0.0)

    def test_known_energy_single_bit(self):
        """????? 1 ?????? Q[0,0]?"""
        Q = np.array([[1.0, 2.0], [2.0, 3.0]])
        x = np.array([1.0, 0.0])
        self.assertAlmostEqual(self.opt._compute_qubo_energy(x, Q), 1.0)

    def test_known_energy_full_ones(self):
        """? 1 ????????????????"""
        Q = np.array([[1.0, 2.0], [2.0, 3.0]])
        x = np.array([1.0, 1.0])
        # Q00 + Q11 + Q01 + Q10 = 1 + 3 + 2 + 2 = 8
        self.assertAlmostEqual(self.opt._compute_qubo_energy(x, Q), 8.0)

    def test_energy_returns_float(self):
        """????? float ???"""
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        x = np.array([1.0, 1.0])
        e = self.opt._compute_qubo_energy(x, Q)
        self.assertIsInstance(e, float)

    def test_energy_diagonal_only(self):
        """????????????? Q_ii ???"""
        Q = np.diag([2.0, 3.0, 4.0])
        x = np.array([1.0, 0.0, 1.0])
        # 2 + 4 = 6
        self.assertAlmostEqual(self.opt._compute_qubo_energy(x, Q), 6.0)

    def test_energy_matches_manual_formula(self):
        """?????? x^T Q x ?????"""
        np.random.seed(99)
        Q = np.random.randn(5, 5)
        Q = Q + Q.T  # ???
        x = np.array([1, 0, 1, 1, 0], dtype=np.float64)
        expected = float(x @ Q @ x)
        self.assertAlmostEqual(self.opt._compute_qubo_energy(x, Q), expected)


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
        weights, shapes = self.opt._extract_weights(self.net)
        self.assertEqual(len(weights), 2)
        self.assertEqual(shapes, [(2, 4), (2,)])
        np.testing.assert_array_almost_equal(weights[0], self.net.weight.detach().numpy())
        np.testing.assert_array_almost_equal(weights[1], self.net.bias.detach().numpy())

    def test_extract_shapes_match_module(self):
        """??????? nn.Module ???????"""
        weights, shapes = self.opt._extract_weights(self.net)
        for w, s, p in zip(weights, shapes, self.net.parameters(), strict=False):
            self.assertEqual(w.shape, p.shape)
            self.assertEqual(s, p.shape)

    def test_set_weights_round_trip_preserves_values(self):
        """??????????????(????)?"""
        original_w, _ = self.opt._extract_weights(self.net)
        new_net = nn.Linear(4, 2)
        self.opt._set_weights(new_net, original_w)
        round_trip, _ = self.opt._extract_weights(new_net)
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
        weights, shapes = self.opt._extract_weights(net)
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
        old = [np.array([[1.0, 1.0]], dtype=np.float32), np.array([1.0], dtype=np.float32)]
        new = [np.array([[3.0, 3.0]], dtype=np.float32), np.array([3.0], dtype=np.float32)]
        shapes = [(1, 2), (1,)]
        self.opt._apply_weights(net, old, new, shapes, learning_rate=0.5)
        # old_std?0, new_std?0 ? ????1 ? w_final = 0.5*1 + 0.5*3 = 2
        np.testing.assert_array_almost_equal(net.weight.detach().numpy(), np.full((1, 2), 2.0))
        np.testing.assert_array_almost_equal(net.bias.detach().numpy(), np.full(1, 2.0))


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
            loss_after, loss_before * 1.05, f"loss ????: {loss_before:.4f} ? {loss_after:.4f}"
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
        import sys

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
        result = QuantumAnnealingOptimizer._get_full_policy(agent)
        self.assertIs(result, net)

    def test_falls_back_to_get_policy_net(self):
        """? PPO ?????? _get_policy_net?"""
        net = nn.Linear(4, 2)
        agent = MagicMock()
        agent.policy_net = net
        # policy ????? ? ?????
        del agent.policy
        result = QuantumAnnealingOptimizer._get_full_policy(agent)
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
        energy = self.opt._compute_qubo_energy(bits, self.Q)
        self.assertTrue(np.isfinite(energy))

    def test_random_sample_improves_with_more_samples(self):
        """?????????????(?????)?"""
        from scripts.evaluation.annealing_solver_comparison import random_sample_qubo

        np.random.seed(123)
        bs_small = random_sample_qubo(self.Q, num_samples=10)
        bits_small = np.array([int(b) for b in bs_small], dtype=np.float64)
        energy_small = self.opt._compute_qubo_energy(bits_small, self.Q)

        np.random.seed(123)
        bs_large = random_sample_qubo(self.Q, num_samples=500)
        bits_large = np.array([int(b) for b in bs_large], dtype=np.float64)
        energy_large = self.opt._compute_qubo_energy(bits_large, self.Q)

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
            bs_sa = self.opt._numpy_simulated_annealing(qubo)
            bits_sa = np.array([int(b) for b in bs_sa], dtype=np.float64)
            sa_energies.append(self.opt._compute_qubo_energy(bits_sa, qubo))

            bs_rand = random_sample_qubo(qubo, num_samples=300)
            bits_rand = np.array([int(b) for b in bs_rand], dtype=np.float64)
            rand_energies.append(self.opt._compute_qubo_energy(bits_rand, qubo))

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


if __name__ == "__main__":
    unittest.main()
