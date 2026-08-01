"""
量子RL调度系统 - 随机种子管理器单元测试
Unit Tests for src/utils/seeds.py

测试覆盖：
- set_seed 返回值（显式参数、默认值）
- Python random 模块确定性
- NumPy 随机数生成器确定性（numpy 不可用时跳过）
- Gymnasium 随机源设置（Issue #879）
- PyTorch CUDA 种子设置（Issue #879）
- 环境变量 QUANTUM_RL_SEED 覆盖参数
"""

import os
import random
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.seeds import set_seed

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import gymnasium.utils.seeding as gym_seeding

    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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

    @unittest.skipUnless(GYM_AVAILABLE, "Gymnasium 未安装")
    def test_set_seed_sets_gymnasium(self):
        """set_seed(42) 后 gymnasium np_random 应可复现（Issue #879）。"""
        set_seed(42)
        rng1, _ = gym_seeding.np_random(42)
        v1 = rng1.random()
        set_seed(42)
        rng2, _ = gym_seeding.np_random(42)
        v2 = rng2.random()
        self.assertEqual(v1, v2)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch 未安装")
    def test_set_seed_sets_torch_cpu(self):
        """set_seed(42) 后 torch CPU 随机数应可复现（Issue #879）。"""
        set_seed(42)
        v1 = torch.rand(1).item()
        set_seed(42)
        v2 = torch.rand(1).item()
        self.assertEqual(v1, v2)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch 未安装")
    def test_set_seed_invokes_cuda_manual_seed_all(self):
        """set_seed 应调用 torch.cuda.manual_seed_all（Issue #879）。"""
        with patch("torch.cuda.manual_seed_all") as mock_cuda:
            set_seed(99)
            mock_cuda.assert_called_with(99)


if __name__ == "__main__":
    unittest.main()
