"""
RL 训练循环单元测试（Issue #361）
Unit Tests for src/scheduler/training.py

测试覆盖：
- find_latest_checkpoint: 递归查找最新 .zip 检查点（空目录、多文件 mtime、非 .zip 忽略、子目录递归）
- resume_training: 从检查点恢复训练（PPO/DQN 自动识别、文件不存在异常）
- auto_resume_train: 自动恢复训练（算法校验、从头训练、检查点恢复）

使用 pytest + gymnasium(CartPole-v1) + stable_baselines3(PPO/DQN)，
训练步数保持极小（64-128）以加速测试。
"""

import os
import sys
import time

import gymnasium as gym
import pytest
from stable_baselines3 import DQN, PPO

# 将项目根目录加入 sys.path，确保 src.scheduler.training 可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.training import (
    auto_resume_train,
    find_latest_checkpoint,
    resume_training,
)

# ---------------------------------------------------------------------------
# 公共 fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def cartpole_env():
    """创建 CartPole-v1 测试环境，测试结束后自动关闭。

    Returns:
        gymnasium.Env: CartPole-v1 环境实例
    """
    env = gym.make("CartPole-v1")
    yield env
    env.close()


# ---------------------------------------------------------------------------
# TestFindLatestCheckpoint
# ---------------------------------------------------------------------------


class TestFindLatestCheckpoint:
    """测试 find_latest_checkpoint 函数。"""

    def test_no_directory_returns_none(self, tmp_path):
        """不存在的目录应返回 None。"""
        nonexistent_dir = tmp_path / "nonexistent_dir"
        assert find_latest_checkpoint(str(nonexistent_dir)) is None

    def test_empty_directory_returns_none(self, tmp_path):
        """空目录（无任何文件）应返回 None。"""
        assert find_latest_checkpoint(str(tmp_path)) is None

    def test_finds_latest_by_mtime(self, tmp_path):
        """多个 .zip 文件时应返回修改时间最新的一个。

        创建三个 .zip 文件并设置递增的 mtime，验证返回 mtime 最大的文件。
        """
        names = ["model_a.zip", "model_b.zip", "model_c.zip"]
        for idx, name in enumerate(names):
            fpath = tmp_path / name
            fpath.write_text(f"checkpoint_{idx}", encoding="utf-8")
            # 显式设置递增的 mtime，避免文件系统时间精度问题
            mtime = time.time() + idx * 100
            os.utime(str(fpath), (mtime, mtime))

        result = find_latest_checkpoint(str(tmp_path))
        assert result is not None
        assert result.endswith("model_c.zip")

    def test_ignores_non_zip_files(self, tmp_path):
        """非 .zip 文件应被忽略，目录中仅有非 .zip 文件时返回 None。"""
        for name in ["readme.txt", "config.json", "model.pt", "notes.md", "data.csv"]:
            (tmp_path / name).write_text("dummy", encoding="utf-8")

        assert find_latest_checkpoint(str(tmp_path)) is None

    def test_recursive_search(self, tmp_path):
        """子目录中的 .zip 文件也应被找到，且按 mtime 返回最新。"""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        root_file = tmp_path / "root.zip"
        sub_file = subdir / "sub.zip"
        root_file.write_text("root", encoding="utf-8")
        sub_file.write_text("sub", encoding="utf-8")

        # 子目录文件 mtime 更新，应为最新
        old_mtime = time.time() - 1000
        new_mtime = time.time()
        os.utime(str(root_file), (old_mtime, old_mtime))
        os.utime(str(sub_file), (new_mtime, new_mtime))

        result = find_latest_checkpoint(str(tmp_path))
        assert result is not None
        assert result.endswith("sub.zip")


# ---------------------------------------------------------------------------
# TestAutoResumeTrain
# ---------------------------------------------------------------------------


class TestAutoResumeTrain:
    """测试 auto_resume_train 函数。"""

    def test_invalid_algorithm_raises_valueerror(self, cartpole_env, tmp_path):
        """不支持的算法类型应抛出 ValueError。"""
        with pytest.raises(ValueError):
            auto_resume_train(
                algorithm="invalid",
                env=cartpole_env,
                total_timesteps=64,
                checkpoint_dir=str(tmp_path),
            )

    def test_train_from_scratch(self, cartpole_env, tmp_path):
        """无检查点时应从头训练 PPO 模型。"""
        model = auto_resume_train(
            algorithm="ppo",
            env=cartpole_env,
            total_timesteps=64,
            checkpoint_dir=str(tmp_path),
            checkpoint_freq=32,
        )
        assert model is not None
        assert model.num_timesteps >= 64

    def test_resume_from_checkpoint(self, cartpole_env, tmp_path):
        """有检查点时应从检查点恢复训练。

        先手动创建 PPO 检查点（训练 64 步），再调用 auto_resume_train
        应加载该检查点并继续训练。
        """
        # 创建并训练 PPO 模型，保存为检查点（文件名含 "ppo" 以便自动识别算法）
        model = PPO(
            "MlpPolicy",
            cartpole_env,
            verbose=0,
            n_steps=64,
            batch_size=32,
            policy_kwargs={"net_arch": [32]},
        )
        model.learn(total_timesteps=64, reset_num_timesteps=True)
        ckpt_path = tmp_path / "ppo_checkpoint.zip"
        model.save(str(ckpt_path))
        # 确保 mtime 足够新，使 find_latest_checkpoint 能找到它
        mtime = time.time()
        os.utime(str(ckpt_path), (mtime, mtime))

        # 调用 auto_resume_train 应发现检查点并从检查点恢复训练
        resumed = auto_resume_train(
            algorithm="ppo",
            env=cartpole_env,
            total_timesteps=128,
            checkpoint_dir=str(tmp_path),
            checkpoint_freq=64,
        )
        assert resumed is not None
        assert resumed.num_timesteps >= 64


# ---------------------------------------------------------------------------
# TestResumeTraining
# ---------------------------------------------------------------------------


class TestResumeTraining:
    """测试 resume_training 函数。"""

    def test_loads_ppo_model(self, cartpole_env, tmp_path):
        """应正确加载 PPO 模型并继续训练。

        文件名含 "ppo" 关键字，resume_training 应自动识别为 PPO 算法。
        """
        # 创建并保存 PPO 检查点
        model = PPO(
            "MlpPolicy",
            cartpole_env,
            verbose=0,
            n_steps=64,
            batch_size=32,
            policy_kwargs={"net_arch": [32]},
        )
        model.learn(total_timesteps=64, reset_num_timesteps=True)
        ckpt_path = str(tmp_path / "ppo_model")
        model.save(ckpt_path)

        # 通过 resume_training 加载并继续训练
        resumed = resume_training(
            model_path=ckpt_path,
            env=cartpole_env,
            total_timesteps=128,
            additional_timesteps=64,
        )
        assert resumed is not None
        assert isinstance(resumed, PPO)
        assert resumed.num_timesteps >= 64

    def test_loads_dqn_model(self, cartpole_env, tmp_path):
        """应正确加载 DQN 模型并继续训练。

        文件名含 "dqn" 关键字，resume_training 应自动识别为 DQN 算法。
        """
        # 创建并保存 DQN 检查点
        model = DQN(
            "MlpPolicy",
            cartpole_env,
            verbose=0,
            learning_starts=10,
            buffer_size=500,
            batch_size=32,
            policy_kwargs={"net_arch": [32]},
        )
        model.learn(total_timesteps=64, reset_num_timesteps=True)
        ckpt_path = str(tmp_path / "dqn_model")
        model.save(ckpt_path)

        # 通过 resume_training 加载并继续训练
        resumed = resume_training(
            model_path=ckpt_path,
            env=cartpole_env,
            total_timesteps=128,
            additional_timesteps=64,
        )
        assert resumed is not None
        assert isinstance(resumed, DQN)
        assert resumed.num_timesteps >= 64

    def test_file_not_found(self, cartpole_env, tmp_path):
        """不存在的模型路径应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            resume_training(
                model_path=str(tmp_path / "nonexistent_ppo_model.zip"),
                env=cartpole_env,
                total_timesteps=64,
            )
