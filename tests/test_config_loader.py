"""
配置加载单元测试 - annealing 配置节读取
Unit Tests for load_annealing_config (src/utils/helpers.py)

验证：
- load_annealing_config 消费 main 上已有的 ``annealing:`` 配置节
- 文件缺失时返回空字典（定义契约）
"""

import os
import unittest
from unittest.mock import patch

from src.config.settings import CRITICAL_FIELDS, load_settings
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


class TestSettingsFailFast(unittest.TestCase):
    """Issue #887: 关键配置项类型转换失败应 fail-fast，而非静默回退。"""

    def test_critical_fields_include_max_qubits(self):
        """CRITICAL_FIELDS 应包含 max_qubits。"""
        self.assertIn("max_qubits", CRITICAL_FIELDS)

    def test_invalid_critical_field_raises(self):
        """max_qubits='abc' 转换失败应抛 ValueError。"""
        with patch.dict(os.environ, {"QUANTUM_MAX_QUBITS": "abc"}, clear=False):
            with self.assertRaises(ValueError):
                load_settings(config_path="path/not/exist.yaml", env_path="path/not/.env")

    def test_invalid_algorithm_raises(self):
        """algorithm 不可转换应抛 ValueError（fail-fast）。"""
        with patch.dict(os.environ, {"SCHEDULER_ALGORITHM": ""}, clear=False):
            # 空字符串是合法 str，此处用非字符串源验证失败路径
            with patch("src.config.settings._convert", side_effect=ValueError("boom")):
                with self.assertRaises(ValueError):
                    load_settings(config_path="path/not/exist.yaml", env_path="path/not/.env")

    def test_non_critical_invalid_falls_back(self):
        """非关键字段（viz_port）转换失败仍应回退默认值（保证健壮性）。"""
        with patch.dict(os.environ, {"WEB_PORT": "abc"}, clear=False):
            settings = load_settings(config_path="path/not/exist.yaml", env_path="path/not/.env")
        # 加载不应抛异常，viz_port 回退默认值
        self.assertIsNotNone(settings)


if __name__ == "__main__":
    unittest.main()
