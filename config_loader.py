"""
量子退火配置加载器（Issue #245 / #246）

从 ``config/config.yaml`` 读取 ``annealing`` 配置节，并以安全默认值兜底。
支持通过环境变量（前缀 ``ANNEALING_``）覆盖，便于通过 ``.env`` 进行配置。

典型用法::

    from config_loader import load_annealing_config
    cfg = load_annealing_config()
    optimizer = QuantumAnnealingOptimizer(config=cfg)

注意：本模块刻意保持轻量、不依赖 torch，便于在测试与配置校验场景中快速导入。
"""

from __future__ import annotations

import os
from typing import Any

import yaml

# 与 annealing.py 硬编码默认值保持一致（config 缺失或字段缺失时兜底）
DEFAULT_ANNEALING_CONFIG: dict[str, Any] = {
    "simulation_mode": True,
    "num_qubits": 16,
    "shots": 1000,
    "annealing_time": 20.0,
    "sim_initial_temp": 2.0,
    "sim_cooling_rate": 0.995,
    "sim_num_sweeps": 200,
    "reg_lambda": 0.1,
    "max_delta_ratio": 0.1,
    "accept_threshold_ratio": 0.01,
    "head_only": True,
    "max_params_per_block": 200,
    "block_strategy": "tensor_wise",
}

# 环境变量 -> 配置键（仅当环境变量存在时覆盖）
_ENV_MAP: dict[str, str] = {
    "ANNEALING_SIMULATION_MODE": "simulation_mode",
    "ANNEALING_NUM_QUBITS": "num_qubits",
    "ANNEALING_SHOTS": "shots",
    "ANNEALING_ANNEALING_TIME": "annealing_time",
    "ANNEALING_SIM_INITIAL_TEMP": "sim_initial_temp",
    "ANNEALING_SIM_COOLING_RATE": "sim_cooling_rate",
    "ANNEALING_SIM_NUM_SWEEPS": "sim_num_sweeps",
    "ANNEALING_REG_LAMBDA": "reg_lambda",
    "ANNEALING_MAX_DELTA_RATIO": "max_delta_ratio",
    "ANNEALING_ACCEPT_THRESHOLD_RATIO": "accept_threshold_ratio",
    "ANNEALING_HEAD_ONLY": "head_only",
    "ANNEALING_MAX_PARAMS_PER_BLOCK": "max_params_per_block",
    "ANNEALING_BLOCK_STRATEGY": "block_strategy",
}

_TYPE_COERCION: dict[str, Any] = {
    "simulation_mode": lambda v: str(v).strip().lower() in ("1", "true", "yes"),
    "num_qubits": int,
    "shots": int,
    "annealing_time": float,
    "sim_initial_temp": float,
    "sim_cooling_rate": float,
    "sim_num_sweeps": int,
    "reg_lambda": float,
    "max_delta_ratio": float,
    "accept_threshold_ratio": float,
    "head_only": lambda v: str(v).strip().lower() in ("1", "true", "yes"),
    "max_params_per_block": int,
    "block_strategy": str,
}


def _find_config_path() -> str:
    """在常见位置查找 config.yaml，返回首个存在的路径。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.getcwd(), "config", "config.yaml"),
        os.path.join(here, "config", "config.yaml"),
        os.path.join(here, "..", "config", "config.yaml"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    # 兜底：返回最可能的相对路径（调用方读取失败时会回退到默认配置）
    return candidates[0]


def load_annealing_config(path: str | None = None) -> dict[str, Any]:
    """
    读取 config.yaml 的 ``annealing`` 配置节，并以安全默认值兜底。

    Args:
        path: config.yaml 路径；默认自动在常见位置查找。

    Returns:
        合并后的 annealing 配置字典（始终包含所有 ``DEFAULT_ANNEALING_CONFIG`` 键）。
    """
    cfg: dict[str, Any] = dict(DEFAULT_ANNEALING_CONFIG)

    config_path = path or _find_config_path()
    try:
        if os.path.isfile(config_path):
            with open(config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            section = raw.get("annealing", {}) if isinstance(raw, dict) else {}
            if isinstance(section, dict):
                for key in DEFAULT_ANNEALING_CONFIG:
                    if key in section and section[key] is not None:
                        cfg[key] = section[key]
    except (OSError, yaml.YAMLError) as exc:
        # 配置读取失败时使用默认配置，保证退火优化器可正常构造（Issue #246 健壮性）
        from loguru import logger

        logger.warning(f"[config_loader] 读取退火配置失败，使用默认配置: {exc}")

    # 环境变量覆盖（可选）
    for env_key, cfg_key in _ENV_MAP.items():
        if env_key in os.environ:
            raw_val = os.environ[env_key]
            coerce = _TYPE_COERCION.get(cfg_key, str)
            try:
                cfg[cfg_key] = coerce(raw_val)
            except (ValueError, TypeError):
                from loguru import logger

                logger.warning(
                    f"[config_loader] 环境变量 {env_key}={raw_val} 无法解析为 "
                    f"{cfg_key}，已忽略"
                )
    return cfg
