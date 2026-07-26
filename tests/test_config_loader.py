"""
config_loader 单元测试（Issue #245 / #246）

本测试不导入 torch / annealing.py，仅验证配置加载逻辑，可在无 GPU/无 torch 环境运行。
"""

import os
import sys

import pytest

# 确保仓库根目录在 sys.path（config_loader.py 位于根目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import DEFAULT_ANNEALING_CONFIG, load_annealing_config  # noqa: E402


def test_defaults_complete():
    """load_annealing_config 在缺少文件时返回完整默认配置。"""
    cfg = load_annealing_config(path="non_existent_path.yaml")
    assert cfg == DEFAULT_ANNEALING_CONFIG
    # 13 个退火参数齐全
    for key in [
        "simulation_mode",
        "num_qubits",
        "shots",
        "annealing_time",
        "sim_initial_temp",
        "sim_cooling_rate",
        "sim_num_sweeps",
        "reg_lambda",
        "max_delta_ratio",
        "accept_threshold_ratio",
        "head_only",
        "max_params_per_block",
        "block_strategy",
    ]:
        assert key in cfg


def test_loads_from_config_yaml():
    """从实际 config/config.yaml 读取 annealing 节（默认值与 annealing.py 保持一致）。"""
    cfg = load_annealing_config()
    assert cfg["num_qubits"] == 16
    assert cfg["simulation_mode"] is True
    assert cfg["shots"] == 1000
    assert cfg["annealing_time"] == 20.0
    assert cfg["reg_lambda"] == 0.1
    assert cfg["max_delta_ratio"] == 0.1
    assert cfg["accept_threshold_ratio"] == 0.01
    assert cfg["block_strategy"] == "tensor_wise"
    assert cfg["max_params_per_block"] == 200
    assert cfg["head_only"] is True


def test_env_override(monkeypatch):
    """环境变量 ANNEALING_ 可覆盖配置。"""
    monkeypatch.setenv("ANNEALING_NUM_QUBITS", "32")
    monkeypatch.setenv("ANNEALING_HEAD_ONLY", "false")
    monkeypatch.setenv("ANNEALING_SIM_COOLING_RATE", "0.9")
    cfg = load_annealing_config()
    assert cfg["num_qubits"] == 32
    assert cfg["head_only"] is False
    assert cfg["sim_cooling_rate"] == 0.9
