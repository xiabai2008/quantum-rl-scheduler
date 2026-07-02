"""
量子RL调度系统 - 随机种子管理器单元测试
Unit Tests for src/utils/seeds.py

测试覆盖：
- set_seed 返回值（显式参数、默认值）
- Python random 模块确定性
- NumPy 随机数生成器确定性（numpy 不可用时跳过）
- 环境变量 QUANTUM_RL_SEED 覆盖参数
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.seeds import set_seed

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class TestSetSeed(unittest.TestCase):
    """测试 set_seed 随机种子管理器。"""

    def test_set_seed_returns_seed(self):
        """set_seed(42) 应返回 42。"""
        self.assertEqual(set_seed(42), 42)

    def test_set_seed_default(self):
        """set_seed() 无参数时应使用默认值 42。"""
        # 确保环境变量未污染本测试
        self.assertNotIn("QUANTUM_RL_SEED", os.environ)
        self.assertEqual(set_seed(), 42)

    def test_set_seed_sets_python_random(self):
        """set_seed(42) 后 random.random() 应可复现。"""
        set_seed(42)
        v1 = random.random()
        set_seed(42)
        v2 = random.random()
        self.assertEqual(v1, v2)

    @unittest.skipUnless(
        NUMPY_AVAILABLE,
        "NumPy 未安装，跳过 numpy 确定性测试",
    )
    def test_set_seed_sets_numpy(self):
        """set_seed(42) 后 np.random.rand() 应可复现。"""
        set_seed(42)
        v1 = np.random.rand()
        set_seed(42)
        v2 = np.random.rand()
        self.assertEqual(v1, v2)

    def test_set_seed_env_override(self):
        """QUANTUM_RL_SEED 环境变量应覆盖传入的 seed 参数。"""
        os.environ["QUANTUM_RL_SEED"] = "123"
        try:
            self.assertEqual(set_seed(42), 123)
        finally:
            del os.environ["QUANTUM_RL_SEED"]

    def test_set_seed_env_invalid_falls_back_to_param(self):
        """QUANTUM_RL_SEED 非整数时应回退到传入参数并发出警告。"""
        os.environ["QUANTUM_RL_SEED"] = "not-a-number"
        try:
            self.assertEqual(set_seed(42), 42)
        finally:
            del os.environ["QUANTUM_RL_SEED"]


if __name__ == "__main__":
    unittest.main()
