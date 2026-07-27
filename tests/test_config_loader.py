"""
配置加载单元测试 - annealing 配置节读取
Unit Tests for load_annealing_config (src/utils/helpers.py)

验证：
- load_annealing_config 消费 main 上已有的 ``annealing:`` 配置节
- 文件缺失时返回空字典（定义契约）
"""

import unittest

from src.utils.helpers import load_annealing_config

CONFIG_PATH = "config/config.yaml"


class TestLoadAnnealingConfig(unittest.TestCase):
    """Issue #246: 验证 load_annealing_config 读取既有 annealing: 配置节。"""

    def test_load_annealing_config_reads_keys(self):
        """返回值应包含取自 config.yaml 的退火参数键。"""
        cfg = load_annealing_config(CONFIG_PATH)
        self.assertEqual(cfg.get("num_qubits"), 16)
        self.assertEqual(cfg.get("shots"), 1000)
        self.assertEqual(cfg.get("annealing_time"), 20.0)
        self.assertEqual(cfg.get("sim_cooling_rate"), 0.995)

    def test_load_annealing_config_missing_file(self):
        """path 指向不存在文件时应返回空字典 {}（契约）。"""
        cfg = load_annealing_config("path/that/does/not/exist.yaml")
        self.assertEqual(cfg, {})


if __name__ == "__main__":
    unittest.main()
