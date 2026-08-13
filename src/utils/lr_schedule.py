"""
学习率调度工具模块（Issue #403）

提供统一的学习率调度函数，支持三种调度策略：
    - ``"linear"``  : 线性衰减到 0（SB3 PPO 默认推荐的调度方式）
    - ``"cosine"``  : 余弦退火衰减到 0
    - ``"constant"``: 保持恒定不变（向后兼容旧行为）

调度函数签名与 SB3 兼容：``Callable[[float], float]``，
输入为 ``progress_remaining``（1.0 → 0.0），输出为当前学习率。

典型用法（SB3 系智能体）::

    from src.utils.lr_schedule import create_lr_schedule

    lr_fn = create_lr_schedule(base_lr=3e-4, schedule_type="linear")
    model = PPO(..., learning_rate=lr_fn)

典型用法（自定义 PyTorch 优化器，如 MAPPO）::

    from src.utils.lr_schedule import create_lr_schedule, compute_lr_at_progress

    lr_fn = create_lr_schedule(base_lr=3e-4, schedule_type="cosine")
    # 在训练循环中手动更新优化器学习率
    progress_remaining = 1.0 - current_step / total_steps
    new_lr = lr_fn(progress_remaining)
    for pg in optimizer.param_groups:
        pg["lr"] = new_lr
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

LRScheduleType = Literal["linear", "cosine", "constant"]

__all__ = [
    "SUPPORTED_SCHEDULE_TYPES",
    "LRScheduleType",
    "compute_lr_at_progress",
    "constant_lr_fn",
    "cosine_lr_fn",
    "create_lr_schedule",
    "linear_lr_fn",
]

SUPPORTED_SCHEDULE_TYPES: tuple[str, ...] = ("linear", "cosine", "constant")


def constant_lr_fn(base_lr: float) -> Callable[[float], float]:
    """恒定学习率调度函数工厂（模块级具名函数，可被 cloudpickle 稳定序列化）。

    Args:
        base_lr: 恒定学习率。

    Returns:
        恒返回 ``base_lr`` 的调度函数。
    """

    def fn(progress_remaining: float) -> float:
        return float(base_lr)

    return fn


def linear_lr_fn(base_lr: float) -> Callable[[float], float]:
    """线性衰减学习率调度函数工厂（模块级具名函数，可被 cloudpickle 稳定序列化）。

    Args:
        base_lr: 初始学习率。

    Returns:
        调度函数：``lr = base_lr * max(progress_remaining, 0.0)``。
    """

    def fn(progress_remaining: float) -> float:
        return float(base_lr * max(progress_remaining, 0.0))

    return fn


def cosine_lr_fn(base_lr: float) -> Callable[[float], float]:
    """余弦退火学习率调度函数工厂（模块级具名函数，可被 cloudpickle 稳定序列化）。

    Args:
        base_lr: 初始学习率。

    Returns:
        调度函数：``lr = base_lr * 0.5 * (1 + cos(pi * progress))``。
    """

    def fn(progress_remaining: float) -> float:
        progress = 1.0 - max(min(progress_remaining, 1.0), 0.0)
        return float(base_lr * 0.5 * (1.0 + np.cos(np.pi * progress)))

    return fn


def create_lr_schedule(
    base_lr: float,
    schedule_type: LRScheduleType = "linear",
) -> Callable[[float], float]:
    """创建学习率调度函数。

    Args:
        base_lr: 基础（初始）学习率，必须为正数。
        schedule_type: 调度类型，可选 ``"linear"`` / ``"cosine"`` / ``"constant"``。

    Returns:
        调度函数 ``lr_fn(progress_remaining: float) -> float``，
        输入 ``progress_remaining`` 从 1.0（训练开始）递减到 0.0（训练结束）。

    Raises:
        ValueError: ``base_lr`` 非正或 ``schedule_type`` 不受支持时。
    """
    if base_lr <= 0:
        raise ValueError(f"base_lr 必须为正数，实际传入: {base_lr}")
    if schedule_type not in SUPPORTED_SCHEDULE_TYPES:
        raise ValueError(
            f"不支持的 schedule_type={schedule_type!r}，可选: {SUPPORTED_SCHEDULE_TYPES}"
        )

    # 8.13 round10 审查（P1-3）：此前返回内联 lambda（constant/linear 分支），
    # cloudpickle 序列化 lambda 字节码嵌入模型 zip，跨 Python 版本反序列化
    # 触发 SIGSEGV（3.11 实测 0xC0000005）。改用模块级具名函数工厂——
    # 具名函数按"模块路径+函数名"序列化，跨版本安全。行为完全不变。
    if schedule_type == "constant":
        return constant_lr_fn(base_lr)

    if schedule_type == "linear":
        # 线性衰减：lr = base_lr * progress_remaining
        # progress_remaining: 1.0 → 0.0，lr: base_lr → 0
        return linear_lr_fn(base_lr)

    # cosine: 余弦退火
    # progress = 1 - progress_remaining (0 → 1)
    # lr = base_lr * 0.5 * (1 + cos(pi * progress))
    # progress=0: lr = base_lr, progress=1: lr = 0
    return cosine_lr_fn(base_lr)


def compute_lr_at_progress(
    base_lr: float,
    schedule_type: LRScheduleType,
    progress_remaining: float,
) -> float:
    """直接计算指定进度处的学习率（无需创建调度函数）。

    便捷方法，适用于单次查询场景。内部调用 :func:`create_lr_schedule`。

    Args:
        base_lr: 基础（初始）学习率。
        schedule_type: 调度类型。
        progress_remaining: 剩余进度，1.0 = 训练开始，0.0 = 训练结束。

    Returns:
        当前学习率。
    """
    lr_fn = create_lr_schedule(base_lr, schedule_type)
    return lr_fn(progress_remaining)
