"""
训练检查点自动恢复模块
Training Checkpoint Auto-Resume Module

提供训练过程的检查点管理能力，支持从已保存的检查点恢复训练，
避免训练中断后从头开始。

模块内容：
    - find_latest_checkpoint: 递归查找最新检查点文件
    - resume_training: 从指定检查点恢复训练（PPO/DQN 自动识别）
    - auto_resume_train: 自动检查点恢复训练（无检查点时从头训练）
"""

import os
from typing import Any

import gymnasium as gym
from loguru import logger
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)

# ---------------------------------------------------------------------------
# 检查点自动恢复（Issue #43）
# ---------------------------------------------------------------------------


def find_latest_checkpoint(checkpoint_dir: str = "models/") -> str | None:
    """
    递归查找检查点目录下最新的 .zip 文件（按修改时间排序）。

    遍历 checkpoint_dir 及其所有子目录，收集 .zip 文件并按修改时间
    （mtime）降序排序，返回最新（mtime 最大）的文件路径。

    Args:
        checkpoint_dir: 检查点目录路径，默认 "models/"

    Returns:
        最新检查点文件的完整路径；目录不存在或无 .zip 文件时返回 None
    """
    if not os.path.isdir(checkpoint_dir):
        logger.debug(f"[FindCheckpoint] 目录不存在: {checkpoint_dir}")
        return None

    # 递归收集所有 .zip 文件及其修改时间
    checkpoint_files: list[tuple[float, str]] = []
    for root, _dirs, files in os.walk(checkpoint_dir):
        for fname in files:
            if fname.lower().endswith(".zip"):
                fpath = os.path.join(root, fname)
                try:
                    mtime = os.path.getmtime(fpath)
                    checkpoint_files.append((mtime, fpath))
                except OSError as e:
                    # 文件可能在扫描过程中被删除，跳过
                    logger.debug(f"[FindCheckpoint] 跳过不可访问文件 {fpath}: {e}")
                    continue

    if not checkpoint_files:
        logger.debug(f"[FindCheckpoint] 目录 {checkpoint_dir} 中无 .zip 检查点")
        return None

    # 按修改时间降序排序，返回最新的文件路径
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
    latest_path = checkpoint_files[0][1]
    logger.info(f"[FindCheckpoint] 找到最新检查点: {latest_path}")
    return latest_path


def resume_training(
    model_path: str,
    env: gym.Env,
    total_timesteps: int,
    additional_timesteps: int = 0,
    checkpoint_callback: BaseCallback | None = None,
) -> Any:
    """
    从已保存的检查点恢复训练。

    自动识别模型类型（PPO/DQN），加载后继续训练。通过
    ``reset_num_timesteps=False`` 保持训练步数连续，使 TensorBoard
    曲线无缝衔接。

    模型类型识别策略：
        1. 优先依据 model_path 中包含的 "ppo" / "dqn" 关键字
        2. 关键字不明确时依次尝试 PPO.load / DQN.load

    Args:
        model_path: 已保存模型文件路径（.zip），可带或不带扩展名
        env: 训练环境（必须与原训练环境兼容）
        total_timesteps: 原始计划的总训练步数
        additional_timesteps: 额外训练步数；为 0 时自动计算
                              （total_timesteps - 已训练步数）
        checkpoint_callback: 可选的检查点回调，训练时定期保存

    Returns:
        训练后的模型实例（PPO 或 DQN）

    Raises:
        FileNotFoundError: 模型文件无法加载为 PPO 或 DQN
    """
    logger.info(f"[ResumeTraining] 从检查点恢复: {model_path}")

    model: Any = None
    load_errors: list[str] = []

    # 规范化路径：补全 .zip 扩展名（若用户传入不带扩展名的路径）
    normalized_path = model_path if model_path.lower().endswith(".zip") else model_path + ".zip"

    # 依据文件名关键字识别算法类型
    path_lower = normalized_path.lower()
    if "ppo" in path_lower:
        try:
            model = PPO.load(normalized_path, env=env)
            logger.info("[ResumeTraining] 已加载 PPO 模型")
        except Exception as e:
            load_errors.append(f"PPO: {e}")
    elif "dqn" in path_lower:
        try:
            model = DQN.load(normalized_path, env=env)
            logger.info("[ResumeTraining] 已加载 DQN 模型")
        except Exception as e:
            load_errors.append(f"DQN: {e}")
    else:
        # 文件名无明确算法标识，依次尝试 PPO / DQN
        for algo_name, algo_cls in [("PPO", PPO), ("DQN", DQN)]:
            try:
                model = algo_cls.load(normalized_path, env=env)
                logger.info(f"[ResumeTraining] 已加载 {algo_name} 模型")
                break
            except Exception as e:
                load_errors.append(f"{algo_name}: {e}")

    if model is None:
        raise FileNotFoundError(
            f"无法从 {model_path} 加载模型（尝试 PPO/DQN 均失败）。"
            f"错误详情: {load_errors}"
        )

    # 计算还需训练的步数
    trained_steps = int(getattr(model, "num_timesteps", 0))
    if additional_timesteps > 0:
        steps_to_train = additional_timesteps
    else:
        steps_to_train = max(0, total_timesteps - trained_steps)

    logger.info(
        f"[ResumeTraining] 已训练 {trained_steps} 步，本次再训练 {steps_to_train} 步"
    )

    if steps_to_train <= 0:
        logger.warning("[ResumeTraining] 已达到总训练步数，无需继续训练")
        return model

    # 继续训练（reset_num_timesteps=False 保持步数连续）
    model.learn(
        total_timesteps=steps_to_train,
        callback=checkpoint_callback,
        reset_num_timesteps=False,
    )

    logger.info(
        f"[ResumeTraining] 恢复训练完成，累计训练步数: {model.num_timesteps}"
    )
    return model


def auto_resume_train(
    algorithm: str = "ppo",
    env: gym.Env | None = None,
    total_timesteps: int = 50000,
    checkpoint_dir: str = "models/",
    checkpoint_freq: int = 5000,
) -> Any:
    """
    自动检查点恢复训练。

    工作流程：
        1. 检查 checkpoint_dir 下是否存在检查点（.zip 文件）
        2. 存在检查点：加载最新检查点并继续训练
        3. 不存在检查点：从头开始训练
        4. 自动设置 CheckpointCallback 定期保存检查点

    Args:
        algorithm: 算法类型，"ppo" 或 "dqn"，默认 "ppo"
        env: 训练环境；None 时使用默认 QuantumSchedulingEnv
        total_timesteps: 总训练步数
        checkpoint_dir: 检查点保存目录
        checkpoint_freq: 检查点保存频率（步数）

    Returns:
        训练后的模型实例（PPO 或 DQN）

    Raises:
        ValueError: algorithm 不在 {"ppo", "dqn"} 中
    """
    algorithm = algorithm.lower().strip()
    if algorithm not in ("ppo", "dqn"):
        raise ValueError(
            f"不支持的算法类型: {algorithm}，仅支持 'ppo' 或 'dqn'"
        )

    # 默认环境：延迟导入避免循环依赖
    if env is None:
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv()
        logger.info("[AutoResume] 未提供环境，使用默认 QuantumSchedulingEnv")

    # 确保检查点目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 设置定期检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix=f"{algorithm}_checkpoint",
        verbose=0,
    )

    # 查找最新检查点
    latest_ckpt = find_latest_checkpoint(checkpoint_dir)

    if latest_ckpt is not None:
        # 检查点存在，恢复训练
        logger.info(
            f"[AutoResume] 发现检查点 {latest_ckpt}，从检查点恢复训练"
        )
        return resume_training(
            model_path=latest_ckpt,
            env=env,
            total_timesteps=total_timesteps,
            checkpoint_callback=checkpoint_callback,
        )
    else:
        # 无检查点，从头训练
        logger.info(
            f"[AutoResume] 检查点目录 {checkpoint_dir} 无检查点，从头开始训练"
        )
        if algorithm == "ppo":
            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                tensorboard_log="./logs/",
                policy_kwargs={"net_arch": [64, 64]},
            )
        else:
            model = DQN(
                "MlpPolicy",
                env,
                verbose=0,
                learning_starts=50,
                buffer_size=1000,
                policy_kwargs={"net_arch": [64, 64]},
            )

        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=True,
        )
        logger.info(
            f"[AutoResume] 从头训练完成，累计训练步数: {model.num_timesteps}"
        )
        return model


__all__ = [
    "auto_resume_train",
    "find_latest_checkpoint",
    "resume_training",
]
