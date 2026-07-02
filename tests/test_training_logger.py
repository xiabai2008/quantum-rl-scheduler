"""
TrainingMetricsLogger / TensorboardCallback 单元测试

覆盖：
    - TestTrainingMetricsLogger: 标量/直方图/文本/超参数/Episode 记录、flush/close
    - TestTensorboardCallback  : _on_step 回调、log_freq 控制、Episode 检测
    - TestCreateTrainingLogger : 工厂函数返回正确类型
    - TestJsonlFallback        : 无 tensorboard 时降级到 JSONL、文件内容正确
    - TestGetSummary           : 摘要返回结构、多指标聚合
    - TestEdgeCases            : 空日志目录、重复 close、大量步数、空直方图
"""

import json
import os
import tempfile
from typing import Any

import numpy as np
import pytest

from src.scheduler import training_logger as tl_module
from src.scheduler.training_logger import (
    TensorboardCallback,
    TrainingMetricsLogger,
    create_training_logger,
)

# ---------------------------------------------------------------------------
# 辅助：伪 SB3 模型（用于回调初始化）
# ---------------------------------------------------------------------------


class FakeSB3Model:
    """伪 SB3 模型，提供 init_callback 所需的最小接口。"""

    def __init__(self, num_timesteps: int = 0) -> None:
        self.num_timesteps = num_timesteps
        self.logger: Any = None

    def get_env(self) -> None:
        return None


def _make_callback(
    logger: TrainingMetricsLogger,
    log_freq: int = 100,
    num_timesteps: int = 0,
) -> TensorboardCallback:
    """构造一个已初始化的 TensorboardCallback（含 model/locals 设置）。"""
    callback = TensorboardCallback(logger, log_freq=log_freq)
    callback.init_callback(FakeSB3Model(num_timesteps=num_timesteps))
    return callback


# ---------------------------------------------------------------------------
# TestTrainingMetricsLogger
# ---------------------------------------------------------------------------


class TestTrainingMetricsLogger:
    """TrainingMetricsLogger 各记录方法测试。"""

    def test_log_scalar(self, tmp_path: str) -> None:
        """标量记录后应出现在摘要中。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="t1")
        logger_obj.log_scalar("train/reward", 1.5, step=10)
        summary = logger_obj.get_summary()
        assert len(summary["scalars"]) == 1
        rec = summary["scalars"][0]
        assert rec["tag"] == "train/reward"
        assert rec["value"] == pytest.approx(1.5)
        assert rec["step"] == 10
        logger_obj.close()

    def test_log_histogram(self, tmp_path: str) -> None:
        """直方图记录后摘要应包含统计量。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="t2")
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        logger_obj.log_histogram("dist/rewards", values, step=20)
        summary = logger_obj.get_summary()
        assert len(summary["histograms"]) == 1
        rec = summary["histograms"][0]
        assert rec["tag"] == "dist/rewards"
        assert rec["step"] == 20
        assert rec["count"] == 5
        assert rec["mean"] == pytest.approx(3.0)
        assert rec["min"] == pytest.approx(1.0)
        assert rec["max"] == pytest.approx(5.0)
        logger_obj.close()

    def test_log_text(self, tmp_path: str) -> None:
        """文本记录后应出现在摘要中。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="t3")
        logger_obj.log_text("config/info", "PPO lr=3e-4", step=0)
        summary = logger_obj.get_summary()
        assert len(summary["texts"]) == 1
        rec = summary["texts"][0]
        assert rec["tag"] == "config/info"
        assert rec["text"] == "PPO lr=3e-4"
        assert rec["step"] == 0
        logger_obj.close()

    def test_log_hyperparams(self, tmp_path: str) -> None:
        """超参数记录后应出现在摘要中。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="t4")
        params = {"learning_rate": 3e-4, "batch_size": 64}
        metrics = {"final_reward": 250.0}
        logger_obj.log_hyperparams(params, metrics)
        summary = logger_obj.get_summary()
        assert len(summary["hyperparams"]) == 1
        rec = summary["hyperparams"][0]
        assert rec["params"]["learning_rate"] == 3e-4
        assert rec["params"]["batch_size"] == 64
        assert rec["metrics"]["final_reward"] == pytest.approx(250.0)
        logger_obj.close()

    def test_log_hyperparams_without_metrics(self, tmp_path: str) -> None:
        """无 metrics 时应正常记录空指标。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="t4b")
        logger_obj.log_hyperparams({"lr": 0.001})
        summary = logger_obj.get_summary()
        assert len(summary["hyperparams"]) == 1
        assert summary["hyperparams"][0]["metrics"] == {}
        logger_obj.close()

    def test_log_episode(self, tmp_path: str) -> None:
        """Episode 记录应包含 reward/length/info。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="t5")
        logger_obj.log_episode(episode=3, reward=42.0, length=100, info={"extra": 1})
        summary = logger_obj.get_summary()
        assert len(summary["episodes"]) == 1
        rec = summary["episodes"][0]
        assert rec["episode"] == 3
        assert rec["reward"] == pytest.approx(42.0)
        assert rec["length"] == 100
        assert rec["info"]["extra"] == 1
        logger_obj.close()

    def test_log_episode_without_info(self, tmp_path: str) -> None:
        """无 info 时应记录空字典。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="t5b")
        logger_obj.log_episode(episode=0, reward=1.0, length=10)
        summary = logger_obj.get_summary()
        assert summary["episodes"][0]["info"] == {}
        logger_obj.close()

    def test_flush(self, tmp_path: str) -> None:
        """flush 不应抛出异常。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="t6")
        logger_obj.log_scalar("x", 1.0, step=1)
        logger_obj.flush()  # 不应抛异常
        logger_obj.close()

    def test_close(self, tmp_path: str) -> None:
        """close 后 writer 应被关闭。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="t7")
        logger_obj.log_scalar("x", 1.0, step=1)
        logger_obj.close()
        # close 后内部 _closed 标志应为 True
        assert logger_obj._closed is True


# ---------------------------------------------------------------------------
# TestTensorboardCallback
# ---------------------------------------------------------------------------


class TestTensorboardCallback:
    """TensorboardCallback 回调逻辑测试。"""

    def test_on_step_logs_at_log_freq(self, tmp_path: str) -> None:
        """n_calls 为 log_freq 倍数时应记录标量。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="cb1")
        callback = _make_callback(logger_obj, log_freq=100, num_timesteps=500)
        callback.n_calls = 100
        callback.num_timesteps = 500
        callback.locals = {"dones": [], "infos": []}
        result = callback._on_step()
        assert result is True
        summary = logger_obj.get_summary()
        assert any(
            s["tag"] == "train/n_steps" and s["step"] == 100 and s["value"] == 500.0
            for s in summary["scalars"]
        )
        logger_obj.close()

    def test_on_step_no_log_between_freq(self, tmp_path: str) -> None:
        """n_calls 非 log_freq 倍数时不应记录标量。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="cb2")
        callback = _make_callback(logger_obj, log_freq=100)
        callback.n_calls = 50
        callback.num_timesteps = 50
        callback.locals = {"dones": [], "infos": []}
        callback._on_step()
        summary = logger_obj.get_summary()
        # 不应有 train/n_steps 标量
        assert not any(s["tag"] == "train/n_steps" for s in summary["scalars"])
        logger_obj.close()

    def test_on_episode_end_direct_call(self, tmp_path: str) -> None:
        """直接调用 on_episode_end 应记录 Episode。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="cb3")
        callback = _make_callback(logger_obj)
        callback.on_episode_end(episode=0, reward=10.0, length=50)
        summary = logger_obj.get_summary()
        assert len(summary["episodes"]) == 1
        assert summary["episodes"][0]["reward"] == pytest.approx(10.0)
        logger_obj.close()

    def test_episode_detection_from_dones(self, tmp_path: str) -> None:
        """_on_step 应从 dones/infos 检测 Episode 结束并记录。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="cb4")
        callback = _make_callback(logger_obj, log_freq=1000)
        callback.n_calls = 1
        callback.num_timesteps = 1
        callback.locals = {
            "dones": [False, True, False],
            "infos": [
                {},
                {"episode": {"r": 15.0, "l": 30}},
                {},
            ],
        }
        callback._on_step()
        summary = logger_obj.get_summary()
        assert len(summary["episodes"]) == 1
        rec = summary["episodes"][0]
        assert rec["reward"] == pytest.approx(15.0)
        assert rec["length"] == 30
        logger_obj.close()

    def test_episode_count_increments(self, tmp_path: str) -> None:
        """多次 Episode 结束应递增 episode 计数。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="cb5")
        callback = _make_callback(logger_obj, log_freq=1000)
        # 第一次 Episode 结束
        callback.n_calls = 1
        callback.num_timesteps = 1
        callback.locals = {
            "dones": [True],
            "infos": [{"episode": {"r": 1.0, "l": 10}}],
        }
        callback._on_step()
        # 第二次 Episode 结束
        callback.n_calls = 2
        callback.locals = {
            "dones": [True],
            "infos": [{"episode": {"r": 2.0, "l": 20}}],
        }
        callback._on_step()
        summary = logger_obj.get_summary()
        assert len(summary["episodes"]) == 2
        assert summary["episodes"][0]["episode"] == 0
        assert summary["episodes"][1]["episode"] == 1
        logger_obj.close()

    def test_log_freq_clamped_to_minimum(self, tmp_path: str) -> None:
        """log_freq 小于 1 时应被钳制为 1。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="cb6")
        callback = TensorboardCallback(logger_obj, log_freq=0)
        assert callback.log_freq == 1
        logger_obj.close()


# ---------------------------------------------------------------------------
# TestCreateTrainingLogger
# ---------------------------------------------------------------------------


class TestCreateTrainingLogger:
    """工厂函数测试。"""

    def test_returns_correct_types(self, tmp_path: str) -> None:
        """工厂函数应返回 (TrainingMetricsLogger, TensorboardCallback)。"""
        logger_obj, callback = create_training_logger(
            log_dir=str(tmp_path), experiment_name="factory1"
        )
        assert isinstance(logger_obj, TrainingMetricsLogger)
        assert isinstance(callback, TensorboardCallback)
        logger_obj.close()

    def test_callback_references_logger(self, tmp_path: str) -> None:
        """回调应引用传入的 logger 实例。"""
        logger_obj, callback = create_training_logger(
            log_dir=str(tmp_path), experiment_name="factory2"
        )
        assert callback.metrics_logger is logger_obj
        logger_obj.close()


# ---------------------------------------------------------------------------
# TestJsonlFallback
# ---------------------------------------------------------------------------


class TestJsonlFallback:
    """tensorboard 不可用时的 JSONL 降级测试。"""

    def test_fallback_when_tensorboard_unavailable(
        self, tmp_path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """monkeypatch 关闭 tensorboard 后应降级到 JSONL。"""
        monkeypatch.setattr(tl_module, "_TENSORBOARD_AVAILABLE", False)
        logger_obj = TrainingMetricsLogger(
            log_dir=str(tmp_path), experiment_name="fallback1"
        )
        assert logger_obj.use_tensorboard is False
        assert logger_obj._writer is None
        logger_obj.close()

    def test_jsonl_file_created(
        self, tmp_path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """降级模式下记录后应生成 JSONL 文件。"""
        monkeypatch.setattr(tl_module, "_TENSORBOARD_AVAILABLE", False)
        logger_obj = TrainingMetricsLogger(
            log_dir=str(tmp_path), experiment_name="fallback2"
        )
        logger_obj.log_scalar("train/reward", 1.0, step=5)
        logger_obj.close()
        jsonl_path = os.path.join(str(tmp_path), "fallback2.jsonl")
        assert os.path.exists(jsonl_path)

    def test_jsonl_scalar_content(
        self, tmp_path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """JSONL 标量记录内容应正确。"""
        monkeypatch.setattr(tl_module, "_TENSORBOARD_AVAILABLE", False)
        logger_obj = TrainingMetricsLogger(
            log_dir=str(tmp_path), experiment_name="fallback3"
        )
        logger_obj.log_scalar("a/b", 2.5, step=3)
        logger_obj.close()
        jsonl_path = os.path.join(str(tmp_path), "fallback3.jsonl")
        with open(jsonl_path, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert record["type"] == "scalar"
        assert record["tag"] == "a/b"
        assert record["value"] == pytest.approx(2.5)
        assert record["step"] == 3

    def test_jsonl_episode_content(
        self, tmp_path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """JSONL Episode 记录内容应正确。"""
        monkeypatch.setattr(tl_module, "_TENSORBOARD_AVAILABLE", False)
        logger_obj = TrainingMetricsLogger(
            log_dir=str(tmp_path), experiment_name="fallback4"
        )
        logger_obj.log_episode(episode=2, reward=5.0, length=20, info={"k": "v"})
        logger_obj.close()
        jsonl_path = os.path.join(str(tmp_path), "fallback4.jsonl")
        with open(jsonl_path, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert record["type"] == "episode"
        assert record["episode"] == 2
        assert record["reward"] == pytest.approx(5.0)
        assert record["length"] == 20
        assert record["info"]["k"] == "v"

    def test_jsonl_histogram_content(
        self, tmp_path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """JSONL 直方图记录应包含 values 和统计量。"""
        monkeypatch.setattr(tl_module, "_TENSORBOARD_AVAILABLE", False)
        logger_obj = TrainingMetricsLogger(
            log_dir=str(tmp_path), experiment_name="fallback5"
        )
        values = np.array([1.0, 2.0, 3.0])
        logger_obj.log_histogram("dist/q", values, step=7)
        logger_obj.close()
        jsonl_path = os.path.join(str(tmp_path), "fallback5.jsonl")
        with open(jsonl_path, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert record["type"] == "histogram"
        assert record["tag"] == "dist/q"
        assert record["step"] == 7
        assert record["count"] == 3
        assert record["values"] == [1.0, 2.0, 3.0]
        assert record["mean"] == pytest.approx(2.0)

    def test_jsonl_multiple_records(
        self, tmp_path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """多条 JSONL 记录应按行追加。"""
        monkeypatch.setattr(tl_module, "_TENSORBOARD_AVAILABLE", False)
        logger_obj = TrainingMetricsLogger(
            log_dir=str(tmp_path), experiment_name="fallback6"
        )
        logger_obj.log_scalar("a", 1.0, step=1)
        logger_obj.log_scalar("b", 2.0, step=2)
        logger_obj.log_text("t", "hello", step=3)
        logger_obj.close()
        jsonl_path = os.path.join(str(tmp_path), "fallback6.jsonl")
        with open(jsonl_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3
        assert json.loads(lines[0])["tag"] == "a"
        assert json.loads(lines[1])["tag"] == "b"
        assert json.loads(lines[2])["type"] == "text"


# ---------------------------------------------------------------------------
# TestGetSummary
# ---------------------------------------------------------------------------


class TestGetSummary:
    """get_summary 摘要结构与聚合测试。"""

    def test_summary_structure(self, tmp_path: str) -> None:
        """摘要应包含全部 5 个类型的键。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="sum1")
        summary = logger_obj.get_summary()
        assert set(summary.keys()) == {
            "scalars",
            "histograms",
            "texts",
            "hyperparams",
            "episodes",
        }
        logger_obj.close()

    def test_summary_empty(self, tmp_path: str) -> None:
        """无记录时摘要各列表应为空。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="sum2")
        summary = logger_obj.get_summary()
        for key in ("scalars", "histograms", "texts", "hyperparams", "episodes"):
            assert summary[key] == []
        logger_obj.close()

    def test_summary_multi_metric_aggregation(self, tmp_path: str) -> None:
        """多条标量记录应全部出现在摘要中。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="sum3")
        for i in range(5):
            logger_obj.log_scalar("train/reward", float(i), step=i)
        summary = logger_obj.get_summary()
        assert len(summary["scalars"]) == 5
        values = [r["value"] for r in summary["scalars"]]
        assert values == [0.0, 1.0, 2.0, 3.0, 4.0]
        logger_obj.close()

    def test_summary_from_jsonl(
        self, tmp_path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """JSONL 降级模式下摘要应从文件读取。"""
        monkeypatch.setattr(tl_module, "_TENSORBOARD_AVAILABLE", False)
        logger_obj = TrainingMetricsLogger(
            log_dir=str(tmp_path), experiment_name="sum4"
        )
        logger_obj.log_scalar("x", 10.0, step=1)
        logger_obj.log_episode(episode=0, reward=5.0, length=10)
        summary = logger_obj.get_summary()
        assert len(summary["scalars"]) == 1
        assert summary["scalars"][0]["value"] == pytest.approx(10.0)
        assert len(summary["episodes"]) == 1
        assert summary["episodes"][0]["reward"] == pytest.approx(5.0)
        logger_obj.close()

    def test_summary_mixed_types(
        self, tmp_path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """JSONL 模式下多种记录类型应正确分类。"""
        monkeypatch.setattr(tl_module, "_TENSORBOARD_AVAILABLE", False)
        logger_obj = TrainingMetricsLogger(
            log_dir=str(tmp_path), experiment_name="sum5"
        )
        logger_obj.log_scalar("s", 1.0, step=1)
        logger_obj.log_text("t", "txt", step=2)
        logger_obj.log_hyperparams({"lr": 0.1}, {"reward": 1.0})
        logger_obj.log_episode(0, 1.0, 5)
        logger_obj.log_histogram("h", np.array([1.0, 2.0]), step=3)
        summary = logger_obj.get_summary()
        assert len(summary["scalars"]) == 1
        assert len(summary["texts"]) == 1
        assert len(summary["hyperparams"]) == 1
        assert len(summary["episodes"]) == 1
        assert len(summary["histograms"]) == 1
        logger_obj.close()


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """边界情况测试。"""

    def test_empty_log_dir_created(self, tmp_path: str) -> None:
        """传入不存在的子目录时应自动创建。"""
        nested = os.path.join(str(tmp_path), "nested", "deep", "path")
        logger_obj = TrainingMetricsLogger(
            log_dir=nested, experiment_name="edge1"
        )
        assert os.path.isdir(nested)
        logger_obj.close()

    def test_duplicate_close(self, tmp_path: str) -> None:
        """重复 close 不应抛出异常。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="edge2")
        logger_obj.close()
        # 第二次 close 应安全
        logger_obj.close()
        assert logger_obj._closed is True

    def test_large_step_count(self, tmp_path: str) -> None:
        """大步数值不应导致错误。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="edge3")
        large_step = 1_000_000
        logger_obj.log_scalar("train/n_steps", float(large_step), step=large_step)
        summary = logger_obj.get_summary()
        assert summary["scalars"][0]["step"] == large_step
        logger_obj.close()

    def test_empty_histogram(self, tmp_path: str) -> None:
        """空数组的直方图应记录零值统计量。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="edge4")
        logger_obj.log_histogram("empty", np.array([]), step=0)
        summary = logger_obj.get_summary()
        rec = summary["histograms"][0]
        assert rec["count"] == 0
        assert rec["mean"] == 0.0
        assert rec["min"] == 0.0
        assert rec["max"] == 0.0
        logger_obj.close()

    def test_negative_reward(self, tmp_path: str) -> None:
        """负奖励应正确记录。"""
        logger_obj = TrainingMetricsLogger(log_dir=str(tmp_path), experiment_name="edge5")
        logger_obj.log_episode(episode=0, reward=-5.5, length=10)
        summary = logger_obj.get_summary()
        assert summary["episodes"][0]["reward"] == pytest.approx(-5.5)
        logger_obj.close()

    def test_tempdir_isolation(self) -> None:
        """使用 TemporaryDirectory 隔离测试环境。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger_obj = TrainingMetricsLogger(
                log_dir=tmpdir, experiment_name="edge6"
            )
            logger_obj.log_scalar("isolated", 1.0, step=1)
            summary = logger_obj.get_summary()
            assert len(summary["scalars"]) == 1
            logger_obj.close()
        # TemporaryDirectory 退出后目录应被清理
        assert not os.path.exists(tmpdir)

    def test_log_freq_minimum_in_factory(self, tmp_path: str) -> None:
        """工厂函数创建的回调 log_freq 应为正数。"""
        logger_obj, callback = create_training_logger(
            log_dir=str(tmp_path), experiment_name="edge7"
        )
        assert callback.log_freq >= 1
        logger_obj.close()
