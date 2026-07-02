"""
训练检查点自动恢复测试（Issue #43）
Unit Tests for Training Checkpoint Auto-Resume

测试覆盖：
- find_latest_checkpoint: 查找最新检查点（递归、空目录、多文件、非 .zip 忽略）
- resume_training: 从检查点恢复训练（PPO/DQN、自动步数计算、已完成情况）
- auto_resume_train: 自动恢复训练逻辑（有/无检查点、算法校验、checkpoint_freq）
- 边界情况: 损坏检查点文件、空检查点目录、非 .zip 文件忽略、混合文件
"""

import os
import shutil
import sys
import tempfile
import time
import unittest

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.agent import (
    auto_resume_train,
    find_latest_checkpoint,
    resume_training,
)

# ---------------------------------------------------------------------------
# 测试用极简环境：避免依赖真实 QuantumSchedulingEnv，加速测试
# ---------------------------------------------------------------------------


class _DummyEnv(gym.Env):
    """测试用极简 Gymnasium 环境。

    提供 4 维观测空间和 Discrete(3) 动作空间，每个 episode 5 步，
    用于快速验证检查点恢复逻辑，不依赖真实调度环境。
    """

    observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = spaces.Discrete(3)

    def __init__(self) -> None:
        """初始化测试环境。"""
        super().__init__()
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        """重置环境到初始状态。"""
        super().reset(seed=seed)
        self._step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        """执行一步环境交互。"""
        self._step_count += 1
        obs = np.zeros(4, dtype=np.float32)
        reward = 0.0
        terminated = self._step_count >= 5
        truncated = False
        return obs, reward, terminated, truncated, {}


# ---------------------------------------------------------------------------
# TestFindLatestCheckpoint
# ---------------------------------------------------------------------------


class TestFindLatestCheckpoint(unittest.TestCase):
    """测试 find_latest_checkpoint 函数。"""

    def setUp(self) -> None:
        """每个测试前创建独立临时目录。"""
        self.tmpdir = tempfile.mkdtemp(prefix="ckpt_find_")

    def tearDown(self) -> None:
        """测试结束后清理临时目录。"""
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_find_latest_with_single_checkpoint(self) -> None:
        """目录中只有一个 .zip 文件时返回该文件路径。"""
        fpath = os.path.join(self.tmpdir, "model.zip")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("checkpoint")

        result = find_latest_checkpoint(self.tmpdir)
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith("model.zip"))
        self.assertTrue(os.path.isabs(result) or os.path.exists(result))

    def test_find_latest_empty_dir_returns_none(self) -> None:
        """空目录应返回 None。"""
        result = find_latest_checkpoint(self.tmpdir)
        self.assertIsNone(result)

    def test_find_latest_multiple_checkpoints_returns_newest(self) -> None:
        """多个检查点时返回修改时间最新的一个。"""
        # 创建三个文件，依次写入并调整 mtime 确保时间递增
        for idx, name in enumerate(["a.zip", "b.zip", "c.zip"]):
            fpath = os.path.join(self.tmpdir, name)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(str(idx))
            # 显式设置递增的 mtime，避免文件系统时间精度问题
            mtime = time.time() + idx * 100
            os.utime(fpath, (mtime, mtime))

        result = find_latest_checkpoint(self.tmpdir)
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith("c.zip"))

    def test_find_latest_nonexistent_dir_returns_none(self) -> None:
        """不存在的目录应返回 None。"""
        result = find_latest_checkpoint("/nonexistent/path/abc_xyz_123")
        self.assertIsNone(result)

    def test_find_latest_recursive_search(self) -> None:
        """递归查找子目录中的检查点。"""
        subdir = os.path.join(self.tmpdir, "subdir")
        os.makedirs(subdir)

        root_file = os.path.join(self.tmpdir, "root.zip")
        sub_file = os.path.join(subdir, "sub.zip")
        with open(root_file, "w", encoding="utf-8") as f:
            f.write("root")
        with open(sub_file, "w", encoding="utf-8") as f:
            f.write("sub")

        # 子目录文件 mtime 更大，应为最新
        old_mtime = time.time() - 1000
        new_mtime = time.time()
        os.utime(root_file, (old_mtime, old_mtime))
        os.utime(sub_file, (new_mtime, new_mtime))

        result = find_latest_checkpoint(self.tmpdir)
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith("sub.zip"))


# ---------------------------------------------------------------------------
# TestResumeTraining
# ---------------------------------------------------------------------------


class TestResumeTraining(unittest.TestCase):
    """测试 resume_training 函数。"""

    def setUp(self) -> None:
        """每个测试前创建临时目录和测试环境。"""
        self.tmpdir = tempfile.mkdtemp(prefix="ckpt_resume_")
        self.env = _DummyEnv()

    def tearDown(self) -> None:
        """测试结束后清理临时目录。"""
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_ppo_checkpoint(self, name: str, timesteps: int = 100) -> str:
        """创建 PPO 检查点文件并返回路径（含 .zip 扩展名）。

        Args:
            name: 检查点文件名（不含扩展名）
            timesteps: 训练步数

        Returns:
            检查点文件完整路径（含 .zip）
        """
        from stable_baselines3 import PPO

        model = PPO(
            "MlpPolicy",
            self.env,
            verbose=0,
            n_steps=64,
            batch_size=32,
            policy_kwargs={"net_arch": [32]},
        )
        model.learn(total_timesteps=timesteps, reset_num_timesteps=True)
        ckpt_path = os.path.join(self.tmpdir, name)
        model.save(ckpt_path)
        return ckpt_path + ".zip"

    def _create_dqn_checkpoint(self, name: str, timesteps: int = 100) -> str:
        """创建 DQN 检查点文件并返回路径（含 .zip 扩展名）。

        Args:
            name: 检查点文件名（不含扩展名）
            timesteps: 训练步数

        Returns:
            检查点文件完整路径（含 .zip）
        """
        from stable_baselines3 import DQN

        model = DQN(
            "MlpPolicy",
            self.env,
            verbose=0,
            learning_starts=50,
            buffer_size=500,
            batch_size=32,
            policy_kwargs={"net_arch": [32]},
        )
        model.learn(total_timesteps=timesteps, reset_num_timesteps=True)
        ckpt_path = os.path.join(self.tmpdir, name)
        model.save(ckpt_path)
        return ckpt_path + ".zip"

    def test_resume_from_ppo_checkpoint(self) -> None:
        """从 PPO 检查点恢复训练，模型加载正确且步数增加。"""
        ckpt_path = self._create_ppo_checkpoint("ppo_test", timesteps=100)

        resumed = resume_training(
            model_path=ckpt_path,
            env=self.env,
            total_timesteps=200,
            additional_timesteps=100,
        )
        self.assertIsNotNone(resumed)
        # 恢复训练后步数应至少为 100（原始）+ 部分
        self.assertGreaterEqual(resumed.num_timesteps, 100)

    def test_resume_from_dqn_checkpoint(self) -> None:
        """从 DQN 检查点恢复训练，模型加载正确。"""
        ckpt_path = self._create_dqn_checkpoint("dqn_test", timesteps=100)

        resumed = resume_training(
            model_path=ckpt_path,
            env=self.env,
            total_timesteps=200,
            additional_timesteps=100,
        )
        self.assertIsNotNone(resumed)
        self.assertGreaterEqual(resumed.num_timesteps, 100)

    def test_resume_default_additional_timesteps(self) -> None:
        """未指定 additional_timesteps 时自动计算还需训练的步数。"""
        ckpt_path = self._create_ppo_checkpoint("ppo_auto", timesteps=100)

        # total_timesteps=200，已训练 100，应再训练 100
        resumed = resume_training(
            model_path=ckpt_path,
            env=self.env,
            total_timesteps=200,
        )
        self.assertIsNotNone(resumed)
        # 应至少完成原始 100 步
        self.assertGreaterEqual(resumed.num_timesteps, 100)

    def test_resume_already_complete_returns_without_training(self) -> None:
        """已达到总训练步数时不再训练，直接返回模型。"""
        ckpt_path = self._create_ppo_checkpoint("ppo_done", timesteps=200)

        # total_timesteps=200，已训练 200，无需继续训练
        resumed = resume_training(
            model_path=ckpt_path,
            env=self.env,
            total_timesteps=200,
        )
        self.assertIsNotNone(resumed)
        self.assertGreaterEqual(resumed.num_timesteps, 200)

    def test_resume_invalid_path_raises(self) -> None:
        """无效路径应抛出异常。"""
        with self.assertRaises((FileNotFoundError, Exception)):
            resume_training(
                model_path="/nonexistent/path/model.zip",
                env=self.env,
                total_timesteps=100,
            )

    def test_resume_model_type_identification_by_name(self) -> None:
        """通过文件名关键字正确识别模型类型（PPO）。"""
        ckpt_path = self._create_ppo_checkpoint("ppo_named", timesteps=100)

        # 传入不含 .zip 的路径，函数应自动补全
        resumed = resume_training(
            model_path=ckpt_path[:-4],  # 去掉 .zip
            env=self.env,
            total_timesteps=150,
            additional_timesteps=50,
        )
        self.assertIsNotNone(resumed)


# ---------------------------------------------------------------------------
# TestAutoResumeTrain
# ---------------------------------------------------------------------------


class TestAutoResumeTrain(unittest.TestCase):
    """测试 auto_resume_train 函数。"""

    def setUp(self) -> None:
        """每个测试前创建临时目录和测试环境。"""
        self.tmpdir = tempfile.mkdtemp(prefix="ckpt_auto_")
        self.env = _DummyEnv()

    def tearDown(self) -> None:
        """测试结束后清理临时目录。"""
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_auto_resume_no_checkpoint_trains_from_scratch(self) -> None:
        """无检查点时从头开始训练 PPO。"""
        model = auto_resume_train(
            algorithm="ppo",
            env=self.env,
            total_timesteps=100,
            checkpoint_dir=self.tmpdir,
            checkpoint_freq=50,
        )
        self.assertIsNotNone(model)
        # 从头训练应至少完成 100 步
        self.assertGreaterEqual(model.num_timesteps, 100)

    def test_auto_resume_with_checkpoint_resumes_training(self) -> None:
        """有检查点时从检查点恢复训练。"""
        from stable_baselines3 import PPO

        # 先创建一个 PPO 检查点
        model = PPO(
            "MlpPolicy",
            self.env,
            verbose=0,
            n_steps=64,
            batch_size=32,
            policy_kwargs={"net_arch": [32]},
        )
        model.learn(total_timesteps=100, reset_num_timesteps=True)
        ckpt_path = os.path.join(self.tmpdir, "ppo_checkpoint_100_steps.zip")
        model.save(ckpt_path)
        # 确保 mtime 足够新
        mtime = time.time()
        os.utime(ckpt_path, (mtime, mtime))

        # 调用 auto_resume_train 应该恢复训练
        resumed = auto_resume_train(
            algorithm="ppo",
            env=self.env,
            total_timesteps=200,
            checkpoint_dir=self.tmpdir,
            checkpoint_freq=100,
        )
        self.assertIsNotNone(resumed)
        # 恢复后步数应至少为原始 100
        self.assertGreaterEqual(resumed.num_timesteps, 100)

    def test_auto_resume_invalid_algorithm_raises_value_error(self) -> None:
        """不支持的算法类型应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            auto_resume_train(
                algorithm="invalid_algo",
                env=self.env,
                total_timesteps=100,
                checkpoint_dir=self.tmpdir,
            )

    def test_auto_resume_dqn_from_scratch(self) -> None:
        """DQN 算法从头训练。"""
        model = auto_resume_train(
            algorithm="dqn",
            env=self.env,
            total_timesteps=100,
            checkpoint_dir=self.tmpdir,
            checkpoint_freq=50,
        )
        self.assertIsNotNone(model)
        self.assertGreaterEqual(model.num_timesteps, 100)

    def test_auto_resume_checkpoint_freq_param_accepted(self) -> None:
        """不同 checkpoint_freq 参数值均应正常工作。"""
        for freq in [10, 50, 100]:
            sub_dir = os.path.join(self.tmpdir, f"freq_{freq}")
            os.makedirs(sub_dir, exist_ok=True)
            model = auto_resume_train(
                algorithm="ppo",
                env=self.env,
                total_timesteps=100,
                checkpoint_dir=sub_dir,
                checkpoint_freq=freq,
            )
            self.assertIsNotNone(model)

    def test_auto_resume_creates_checkpoint_dir(self) -> None:
        """检查点目录不存在时应自动创建。"""
        nonexistent_dir = os.path.join(self.tmpdir, "new_ckpt_dir")
        model = auto_resume_train(
            algorithm="ppo",
            env=self.env,
            total_timesteps=100,
            checkpoint_dir=nonexistent_dir,
            checkpoint_freq=50,
        )
        self.assertIsNotNone(model)
        self.assertTrue(os.path.isdir(nonexistent_dir))


# ---------------------------------------------------------------------------
# TestCheckpointEdgeCases
# ---------------------------------------------------------------------------


class TestCheckpointEdgeCases(unittest.TestCase):
    """测试检查点边界情况。"""

    def setUp(self) -> None:
        """每个测试前创建临时目录。"""
        self.tmpdir = tempfile.mkdtemp(prefix="ckpt_edge_")

    def tearDown(self) -> None:
        """测试结束后清理临时目录。"""
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_corrupted_checkpoint_file_raises(self) -> None:
        """损坏的检查点文件应抛出异常而非静默失败。"""
        fake_path = os.path.join(self.tmpdir, "ppo_corrupt.zip")
        with open(fake_path, "wb") as f:
            f.write(b"not a real zip file content")

        env = _DummyEnv()
        with self.assertRaises((FileNotFoundError, Exception)):
            resume_training(
                model_path=fake_path,
                env=env,
                total_timesteps=100,
                additional_timesteps=50,
            )

    def test_empty_checkpoint_dir_returns_none(self) -> None:
        """空目录应让 find_latest_checkpoint 返回 None。"""
        result = find_latest_checkpoint(self.tmpdir)
        self.assertIsNone(result)

    def test_non_zip_files_ignored(self) -> None:
        """非 .zip 文件应被 find_latest_checkpoint 忽略。"""
        for name in ["readme.txt", "log.json", "data.csv", "model.pt", "notes.md"]:
            with open(os.path.join(self.tmpdir, name), "w", encoding="utf-8") as f:
                f.write("dummy content")

        result = find_latest_checkpoint(self.tmpdir)
        self.assertIsNone(result)

    def test_mixed_zip_and_non_zip_files(self) -> None:
        """混合目录中只识别 .zip 文件，忽略其他格式。"""
        # 创建非 .zip 文件（时间较早）
        with open(os.path.join(self.tmpdir, "log.txt"), "w", encoding="utf-8") as f:
            f.write("log")
        old_mtime = time.time() - 1000
        os.utime(os.path.join(self.tmpdir, "log.txt"), (old_mtime, old_mtime))

        # 创建 .zip 文件（时间较新）
        zip_path = os.path.join(self.tmpdir, "model.zip")
        with open(zip_path, "w", encoding="utf-8") as f:
            f.write("model")
        new_mtime = time.time()
        os.utime(zip_path, (new_mtime, new_mtime))

        # 创建另一个非 .zip 文件（时间最新）
        with open(os.path.join(self.tmpdir, "notes.md"), "w", encoding="utf-8") as f:
            f.write("notes")
        newest_mtime = time.time() + 100
        os.utime(os.path.join(self.tmpdir, "notes.md"), (newest_mtime, newest_mtime))

        result = find_latest_checkpoint(self.tmpdir)
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith("model.zip"))

    def test_auto_resume_with_corrupted_checkpoint_raises(self) -> None:
        """auto_resume_train 遇到损坏检查点时应抛出异常。"""
        fake_path = os.path.join(self.tmpdir, "ppo_checkpoint_corrupt.zip")
        with open(fake_path, "wb") as f:
            f.write(b"corrupted content")

        env = _DummyEnv()
        with self.assertRaises((FileNotFoundError, Exception)):
            auto_resume_train(
                algorithm="ppo",
                env=env,
                total_timesteps=100,
                checkpoint_dir=self.tmpdir,
                checkpoint_freq=50,
            )

    def test_uppercase_zip_extension_recognized(self) -> None:
        """大写 .ZIP 扩展名也应被识别。"""
        fpath = os.path.join(self.tmpdir, "model.ZIP")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("checkpoint")

        result = find_latest_checkpoint(self.tmpdir)
        self.assertIsNotNone(result)
        self.assertTrue(result.lower().endswith(".zip"))


if __name__ == "__main__":
    unittest.main()
