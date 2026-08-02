"""统一的随机种子管理器，确保实验可复现性。"""

import os
import random

from loguru import logger


def set_seed(seed: int = 42) -> int:
    """一键固定所有随机源（Python/numpy/torch/gymnasium），确保实验可复现。

    依次设置以下随机源：
    - Python 内置 random 模块
    - NumPy 随机数生成器
    - PyTorch CPU 和 CUDA 随机数生成器
    - 环境变量 PYTHONHASHSEED（影响哈希随机化）

    Args:
        seed: 随机种子值，默认 42

    Returns:
        实际使用的种子值（可能与输入不同，如从环境变量覆盖）

    Example:
        >>> from src.utils.seeds import set_seed
        >>> set_seed(42)  # 一键固定所有随机源
        42
    """
    # 环境变量 QUANTUM_RL_SEED 优先级最高，可在运行时覆盖传入参数
    env_seed = os.environ.get("QUANTUM_RL_SEED")
    if env_seed is not None:
        try:
            seed = int(env_seed)
        except ValueError:
            logger.warning(
                f"环境变量 QUANTUM_RL_SEED={env_seed!r} 不是有效整数，使用传入参数 seed={seed}"
            )

    # 设置 PYTHONHASHSEED（影响 dict/set 哈希顺序）
    # Issue #708: PYTHONHASHSEED 仅在解释器启动时读取，运行时设置对当前进程无效，
    # 仅影响子进程。此处保留设置以确保 spawn 的子进程可复现，并记录警告。
    os.environ["PYTHONHASHSEED"] = str(seed)
    current_hashseed = os.environ.get("PYTHONHASHSEED", "")
    if current_hashseed != str(seed):
        logger.warning(
            "PYTHONHASHSEED 运行时设置对当前进程无效（需启动前设置），已写入环境变量供子进程使用"
        )

    # Python 内置 random 模块
    random.seed(seed)

    # NumPy 随机数生成器（如可用）
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        logger.debug("NumPy 未安装，跳过 np.random.seed 设置")

    # PyTorch 随机数生成器（如可用）
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # CUDA 确定性设置，保证多seed实验可复现
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.debug("PyTorch 未安装，跳过 torch.manual_seed 设置")

    # Gymnasium 随机数生成器（如可用）
    # Issue #879: gymnasium.utils.seeding 用于环境 reset(seed=...) 的底层随机源
    try:
        import gymnasium.utils.seeding as gym_seeding

        gym_seeding.np_random(seed)
    except ImportError:
        logger.debug("Gymnasium 未安装，跳过 gymnasium 随机源设置")

    logger.info(f"随机种子已设置为 {seed}")
    return seed
