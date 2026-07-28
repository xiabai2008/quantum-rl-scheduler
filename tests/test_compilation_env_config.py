"""
编译环境可配置规模测试（Issue #594）
Unit Tests for Configurable Compilation Environment

测试覆盖：
- TestConfigurablePhysicalQubits : n_physical 参数配置
- TestCustomCouplingGraph         : 自定义耦合图输入
- TestTianyan287Preset            : 天衍-287 10x11 网格预设
- TestGridParameters              : grid_rows/grid_cols 参数
- TestBackwardCompatibility       : 向后兼容性（默认行为不变）
- TestConfigurableEnvBehavior     : 配置后环境行为正确性
"""

import os
import sys
import unittest

import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.quantum.compilation_env import (
    COUPLING_GRAPH,
    GRID_COLS,
    GRID_ROWS,
    PHYSICAL_QUBITS,
    TIANYAN_287_COLS,
    TIANYAN_287_COUPLING_GRAPH,
    TIANYAN_287_QUBITS,
    TIANYAN_287_ROWS,
    QuantumCompilationEnv,
    _build_2d_grid_coupling,
)


# ============================================================
# TestConfigurablePhysicalQubits
# ============================================================
class TestConfigurablePhysicalQubits(unittest.TestCase):
    """测试 n_physical 参数配置。"""

    def test_default_n_physical_is_16(self):
        """默认 n_physical 应为 16（向后兼容）。"""
        env = QuantumCompilationEnv()
        self.assertEqual(env.n_physical, 16)

    def test_custom_n_physical_9(self):
        """n_physical=9 应使用 3x3 网格。"""
        env = QuantumCompilationEnv(n_physical=9)
        self.assertEqual(env.n_physical, 9)
        self.assertEqual(env.action_space.n, 9)
        self.assertEqual(len(env.coupling_graph), 9)

    def test_custom_n_physical_25(self):
        """n_physical=25 应使用 5x5 网格。"""
        env = QuantumCompilationEnv(n_physical=25)
        self.assertEqual(env.n_physical, 25)
        self.assertEqual(env.action_space.n, 25)

    def test_custom_n_physical_changes_action_space(self):
        """n_physical 改变后 action_space.n 应同步变化。"""
        env_default = QuantumCompilationEnv()
        env_custom = QuantumCompilationEnv(n_physical=9)
        self.assertEqual(env_default.action_space.n, 16)
        self.assertEqual(env_custom.action_space.n, 9)

    def test_non_square_n_physical_falls_back(self):
        """非完全平方数 n_physical 应退化为默认 16。"""
        env = QuantumCompilationEnv(n_physical=10)
        self.assertEqual(env.n_physical, 16)

    def test_n_physical_with_circuit(self):
        """n_physical 与 circuit 参数应可同时使用。"""
        env = QuantumCompilationEnv(circuit=None, max_steps=50, n_physical=9)
        self.assertEqual(env.n_physical, 9)
        self.assertEqual(env.n_logical, 8)

    def test_n_physical_none_uses_default(self):
        """n_physical=None 应使用默认值 16。"""
        env = QuantumCompilationEnv(n_physical=None)
        self.assertEqual(env.n_physical, 16)


# ============================================================
# TestCustomCouplingGraph
# ============================================================
class TestCustomCouplingGraph(unittest.TestCase):
    """测试自定义耦合图输入。"""

    def test_custom_graph_sets_n_physical(self):
        """自定义耦合图应自动设置 n_physical。"""
        graph = _build_2d_grid_coupling(3, 4)  # 12 qubits
        env = QuantumCompilationEnv(coupling_graph=graph)
        self.assertEqual(env.n_physical, 12)
        self.assertEqual(env.action_space.n, 12)

    def test_custom_graph_used_in_observation(self):
        """自定义耦合图应在观测计算中使用。"""
        graph = {0: {1}, 1: {0, 2}, 2: {1}}  # 线性链 0-1-2
        env = QuantumCompilationEnv(coupling_graph=graph)
        env.reset(seed=42)
        env.step(0)
        env.step(1)
        # 0 和 1 在自定义图中相邻
        obs = env._get_obs()
        # conn 维度（索引 3）应 > 0，因为 0-1 相邻
        self.assertGreater(obs[3], 0.0)

    def test_custom_graph_distance_cache(self):
        """自定义图的距离缓存应正确计算。"""
        graph = {0: {1}, 1: {0, 2}, 2: {1}}  # 0-1-2
        env = QuantumCompilationEnv(coupling_graph=graph)
        # 0 到 2 的距离应为 2
        self.assertEqual(env._distance_cache[(0, 2)], 2)
        self.assertEqual(env._distance_cache[(0, 1)], 1)
        self.assertEqual(env._distance_cache[(1, 2)], 1)

    def test_custom_graph_overrides_n_physical(self):
        """自定义耦合图应覆盖 n_physical 参数。"""
        graph = _build_2d_grid_coupling(3, 3)  # 9 qubits
        env = QuantumCompilationEnv(n_physical=16, coupling_graph=graph)
        self.assertEqual(env.n_physical, 9)

    def test_custom_graph_reset_step_works(self):
        """自定义耦合图环境下 reset/step 应正常工作。"""
        graph = _build_2d_grid_coupling(3, 3)
        env = QuantumCompilationEnv(coupling_graph=graph, max_steps=20)
        obs, _ = env.reset(seed=42)
        self.assertEqual(obs.shape, (14,))
        obs, reward, _terminated, _truncated, _ = env.step(0)
        self.assertIsInstance(reward, float)
        self.assertEqual(obs.shape, (14,))


# ============================================================
# TestTianyan287Preset
# ============================================================
class TestTianyan287Preset(unittest.TestCase):
    """测试天衍-287 10x11 网格预设。"""

    def test_tianyan_287_qubits_is_110(self):
        """TIANYAN_287_QUBITS 应为 110（10x11）。"""
        self.assertEqual(TIANYAN_287_QUBITS, 110)

    def test_tianyan_287_grid_dimensions(self):
        """天衍-287 网格应为 10 行 x 11 列。"""
        self.assertEqual(TIANYAN_287_ROWS, 10)
        self.assertEqual(TIANYAN_287_COLS, 11)

    def test_tianyan_287_coupling_graph_has_110_nodes(self):
        """天衍-287 耦合图应有 110 个节点。"""
        self.assertEqual(len(TIANYAN_287_COUPLING_GRAPH), 110)

    def test_tianyan_287_coupling_graph_structure(self):
        """天衍-287 耦合图应为 2D 网格拓扑。"""
        graph = TIANYAN_287_COUPLING_GRAPH
        rows, cols = TIANYAN_287_ROWS, TIANYAN_287_COLS
        for r in range(rows):
            for c in range(cols):
                q = r * cols + c
                neighbors = graph[q]
                if c + 1 < cols:
                    self.assertIn(q + 1, neighbors)
                if r + 1 < rows:
                    self.assertIn(q + cols, neighbors)

    def test_tianyan_287_env_creation(self):
        """n_physical=110 应创建天衍-287 环境。"""
        env = QuantumCompilationEnv(n_physical=TIANYAN_287_QUBITS)
        self.assertEqual(env.n_physical, 110)
        self.assertEqual(env.action_space.n, 110)
        self.assertEqual(len(env.coupling_graph), 110)

    def test_tianyan_287_env_reset_step(self):
        """天衍-287 环境应可正常 reset/step。"""
        env = QuantumCompilationEnv(n_physical=TIANYAN_287_QUBITS, max_steps=50)
        obs, _ = env.reset(seed=42)
        self.assertEqual(obs.shape, (14,))
        self.assertTrue(np.all(obs >= 0.0))
        self.assertTrue(np.all(obs <= 1.0))
        obs, reward, _, _, _ = env.step(0)
        self.assertEqual(obs.shape, (14,))
        self.assertIsInstance(reward, float)

    def test_tianyan_287_corner_node_degree(self):
        """天衍-287 角落节点的度应为 2。"""
        graph = TIANYAN_287_COUPLING_GRAPH
        cols = TIANYAN_287_COLS
        corners = [0, cols - 1, (TIANYAN_287_ROWS - 1) * cols, TIANYAN_287_QUBITS - 1]
        for corner in corners:
            self.assertEqual(len(graph[corner]), 2, f"角落 {corner} 度应为 2")

    def test_tianyan_287_edge_node_degree(self):
        """天衍-287 边缘节点（非角落）的度应为 3。"""
        graph = TIANYAN_287_COUPLING_GRAPH
        cols = TIANYAN_287_COLS
        # 第一行中间节点
        for c in range(1, cols - 1):
            self.assertEqual(len(graph[c]), 3, f"第一行节点 {c} 度应为 3")
        # 第一列中间节点
        for r in range(1, TIANYAN_287_ROWS - 1):
            self.assertEqual(len(graph[r * cols]), 3, f"第一列节点 {r * cols} 度应为 3")

    def test_tianyan_287_interior_node_degree(self):
        """天衍-287 内部节点的度应为 4。"""
        graph = TIANYAN_287_COUPLING_GRAPH
        cols = TIANYAN_287_COLS
        # 内部节点（不在边缘）
        for r in range(1, TIANYAN_287_ROWS - 1):
            for c in range(1, cols - 1):
                q = r * cols + c
                self.assertEqual(len(graph[q]), 4, f"内部节点 {q} 度应为 4")


# ============================================================
# TestGridParameters
# ============================================================
class TestGridParameters(unittest.TestCase):
    """测试 grid_rows/grid_cols 参数。"""

    def test_grid_3x5_creates_15_qubits(self):
        """grid_rows=3, grid_cols=5 应创建 15 个物理比特。"""
        env = QuantumCompilationEnv(grid_rows=3, grid_cols=5)
        self.assertEqual(env.n_physical, 15)
        self.assertEqual(env.action_space.n, 15)

    def test_grid_5x5_creates_25_qubits(self):
        """grid_rows=5, grid_cols=5 应创建 25 个物理比特。"""
        env = QuantumCompilationEnv(grid_rows=5, grid_cols=5)
        self.assertEqual(env.n_physical, 25)

    def test_grid_overrides_n_physical(self):
        """grid_rows/grid_cols 应覆盖 n_physical。"""
        env = QuantumCompilationEnv(n_physical=16, grid_rows=3, grid_cols=4)
        self.assertEqual(env.n_physical, 12)

    def test_grid_coupling_graph_correct(self):
        """grid 参数构建的耦合图应正确。"""
        env = QuantumCompilationEnv(grid_rows=3, grid_cols=4)
        graph = env.coupling_graph
        self.assertEqual(len(graph), 12)
        # 检查角节点 0 的邻居
        self.assertIn(1, graph[0])
        self.assertIn(4, graph[0])
        self.assertEqual(len(graph[0]), 2)

    def test_grid_only_rows_falls_back(self):
        """仅提供 grid_rows（不提供 grid_cols）应走 n_physical 推导路径。"""
        env = QuantumCompilationEnv(grid_rows=3)
        # grid_cols=None，不走 grid 路径，走默认
        self.assertEqual(env.n_physical, 16)


# ============================================================
# TestBackwardCompatibility
# ============================================================
class TestBackwardCompatibility(unittest.TestCase):
    """测试向后兼容性。"""

    def test_default_env_uses_16_qubits(self):
        """默认环境应使用 16 物理比特。"""
        env = QuantumCompilationEnv()
        self.assertEqual(env.n_physical, 16)
        self.assertEqual(env.action_space.n, 16)

    def test_default_env_uses_default_coupling_graph(self):
        """默认环境应使用模块级 COUPLING_GRAPH。"""
        env = QuantumCompilationEnv()
        self.assertEqual(env.coupling_graph, COUPLING_GRAPH)

    def test_module_level_constants_unchanged(self):
        """模块级常量应保持不变。"""
        self.assertEqual(PHYSICAL_QUBITS, 16)
        self.assertEqual(GRID_ROWS, 4)
        self.assertEqual(GRID_COLS, 4)
        self.assertEqual(len(COUPLING_GRAPH), 16)

    def test_default_env_observation_shape(self):
        """默认环境观测形状应为 (14,)。"""
        env = QuantumCompilationEnv()
        obs, _ = env.reset(seed=42)
        self.assertEqual(obs.shape, (14,))

    def test_existing_test_no_circuit_still_works(self):
        """无电路的默认环境应正常工作（模拟现有测试 fixture）。"""
        env = QuantumCompilationEnv(circuit=None, max_steps=200)
        self.assertEqual(env.n_logical, 8)
        self.assertEqual(env.n_physical, 16)
        obs, _ = env.reset(seed=42)
        self.assertEqual(obs.shape, (14,))
        obs, _reward, _terminated, _truncated, _ = env.step(5)
        self.assertEqual(env._mapping[0], 5)


# ============================================================
# TestConfigurableEnvBehavior
# ============================================================
class TestConfigurableEnvBehavior(unittest.TestCase):
    """测试配置后环境行为正确性。"""

    def test_larger_env_step_returns_valid_obs(self):
        """更大规模环境的 step 应返回有效观测。"""
        env = QuantumCompilationEnv(n_physical=25, max_steps=50)
        env.reset(seed=42)
        obs, reward, _terminated, _truncated, _ = env.step(10)
        self.assertEqual(obs.shape, (14,))
        self.assertTrue(np.all(obs >= 0.0))
        self.assertTrue(np.all(obs <= 1.0))
        self.assertIsInstance(reward, float)

    def test_larger_env_full_mapping_terminates(self):
        """更大规模环境完成全部映射应 terminated。"""
        env = QuantumCompilationEnv(circuit=None, max_steps=200, n_physical=25)
        env.reset(seed=42)
        terminated = False
        for i in range(env.n_logical):
            _, _, terminated, _, _ = env.step(i)
        self.assertTrue(terminated)

    def test_custom_graph_swap_uses_correct_distances(self):
        """自定义图上的 SWAP 应使用正确的图距离。"""
        # 线性链 0-1-2-3-4
        graph = _build_2d_grid_coupling(1, 5)
        env = QuantumCompilationEnv(coupling_graph=graph, max_steps=50)
        env.reset(seed=42)
        env.step(0)  # logical 0 -> physical 0
        # 冲突：physical 0 已占用，最近空闲是 1（距离=1）
        _, _reward, _, _, _ = env.step(0)
        # base -2 + dist*2 + 1 = -2 + 2 + 1 = 1... but wait, reward should include +1 for mapping
        # Actually: -2 (base swap) + 1*2 (distance penalty) + 1 (mapping success) = -2+2+1 = 1
        # Hmm, actually the swap_count should be 2 (1 base + 1 distance)
        self.assertEqual(env._swap_count, 2)

    def test_tianyan_287_distance_cache_completeness(self):
        """天衍-287 距离缓存应覆盖所有节点对。"""
        env = QuantumCompilationEnv(n_physical=TIANYAN_287_QUBITS)
        # 检查部分距离对
        self.assertEqual(env._distance_cache[(0, 0)], 0)
        self.assertEqual(env._distance_cache[(0, 1)], 1)
        self.assertEqual(env._distance_cache[(0, 11)], 1)  # 下一行同列
        self.assertEqual(env._distance_cache[(0, 12)], 2)  # 对角线
        # 最大距离（0 到 109 = 9*11+10）
        max_dist = env._distance_cache[(0, 109)]
        self.assertEqual(max_dist, 9 + 10)  # 10 行 + 11 列的曼哈顿距离 = 19

    def test_get_stats_reflects_n_physical(self):
        """get_stats 应反映配置的 n_physical。"""
        env = QuantumCompilationEnv(n_physical=9)
        env.reset(seed=42)
        env.step(0)
        stats = env.get_stats()
        self.assertEqual(stats["n_physical"], 9)

    def test_observation_all_in_range_for_custom_graph(self):
        """自定义图环境的所有观测值应在 [0, 1]。"""
        graph = _build_2d_grid_coupling(5, 5)
        env = QuantumCompilationEnv(coupling_graph=graph, max_steps=100)
        env.reset(seed=42)
        for _ in range(10):
            obs, _, terminated, truncated, _ = env.step(0)
            self.assertTrue(np.all(obs >= 0.0))
            self.assertTrue(np.all(obs <= 1.0))
            if terminated or truncated:
                break

    def test_grid_3x4_distance_correct(self):
        """3x4 网格的距离应正确。"""
        env = QuantumCompilationEnv(grid_rows=3, grid_cols=4)
        # 0 到 5 的距离：0->1->2->3->7->6->5 = 3 步？不对
        # 3x4 网格：0(0,0) -> 1(0,1) -> 2(0,2) -> 3(0,3)
        #           4(1,0) -> 5(1,1) -> 6(1,2) -> 7(1,3)
        # 0 到 5 = (0,0)->(1,1) = 2 步（0->4->5 或 0->1->5）
        self.assertEqual(env._distance_cache[(0, 5)], 2)
        # 0 到 11 = (0,0)->(2,3) = 5 步
        self.assertEqual(env._distance_cache[(0, 11)], 5)


if __name__ == "__main__":
    unittest.main()
