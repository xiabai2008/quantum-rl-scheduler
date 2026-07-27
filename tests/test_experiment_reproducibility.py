"""测试实验可复现性校验模块。"""

import json
import tempfile
from pathlib import Path

import pytest

from scripts.ci.validate_experiment_reproducibility import _compute_config_hash, validate


class TestComputeConfigHash:
    """测试 _compute_config_hash 函数。"""

    def test_deterministic_hash(self) -> None:
        """相同配置应产生相同哈希。"""
        config = {"a": 1, "b": 2}
        h1 = _compute_config_hash(config)
        h2 = _compute_config_hash(config)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex = 64 chars

    def test_key_order_independence(self) -> None:
        """键的顺序不应影响哈希结果。"""
        config1 = {"arrival_lambda": 0.5, "quantum_ratio": 0.7, "seeds": [42]}
        config2 = {"seeds": [42], "arrival_lambda": 0.5, "quantum_ratio": 0.7}
        assert _compute_config_hash(config1) == _compute_config_hash(config2)

    def test_different_configs_different_hashes(self) -> None:
        """不同配置应产生不同哈希。"""
        config1 = {"arrival_lambda": 0.5}
        config2 = {"arrival_lambda": 0.6}
        assert _compute_config_hash(config1) != _compute_config_hash(config2)

    def test_nested_dict_hash(self) -> None:
        """嵌套字典应能被正确哈希。"""
        config = {"outer": {"inner": 1}, "list": [1, 2, 3]}
        h = _compute_config_hash(config)
        assert len(h) == 64


class TestValidate:
    """测试 validate 函数。"""

    def _make_valid_data(self, config: dict | None = None) -> dict:
        """构造带正确 config_hash 的测试数据。"""
        default_config = {
            "seeds": [42, 179],
            "episodes_per_seed": 5,
            "tasks_per_episode": 200,
            "total_episodes": 10,
            "ppo_model": "models/ppo.zip",
            "dqn_model": "models/dqn.zip",
            "observation_dim": 14,
            "wrapper": "原生 14 维环境",
            "arrival_lambda": 0.5,
            "quantum_ratio": 0.7,
            "n_workers": 1,
            "use_cache": False,
            "timestamp": "20260101_000000",
        }
        cfg = config if config is not None else default_config
        cfg_copy = dict(cfg)
        cfg_copy["config_hash"] = _compute_config_hash(cfg_copy)
        return {"config": cfg_copy, "rewards": {}}

    def test_valid_file_passes(self, tmp_path: Path) -> None:
        """有效文件应通过验证。"""
        data = self._make_valid_data()
        path = tmp_path / "rewards.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        assert validate(path) is True

    def test_missing_config_hash_fails(self, tmp_path: Path) -> None:
        """缺少 config_hash 应失败。"""
        data = self._make_valid_data()
        del data["config"]["config_hash"]
        path = tmp_path / "rewards.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        assert validate(path) is False

    def test_tampered_config_fails(self, tmp_path: Path) -> None:
        """篡改配置值后应失败。"""
        data = self._make_valid_data()
        data["config"]["arrival_lambda"] = 0.99  # 篡改
        path = tmp_path / "rewards.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        assert validate(path) is False

    def test_missing_file_fails(self, tmp_path: Path) -> None:
        """不存在的文件应失败。"""
        path = tmp_path / "nonexistent.json"
        assert validate(path) is False

    def test_model_missing_warns_but_passes(self, tmp_path: Path) -> None:
        """模型文件缺失应警告但验证仍通过。"""
        data = self._make_valid_data()
        # 指向不存在的模型路径
        data["config"]["ppo_model"] = "nonexistent/path/model.zip"
        # 重新计算 hash（因为修改了配置）
        cfg = dict(data["config"])
        del cfg["config_hash"]
        data["config"]["config_hash"] = _compute_config_hash(cfg)
        path = tmp_path / "rewards.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        assert validate(path) is True
