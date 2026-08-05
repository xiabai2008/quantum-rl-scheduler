"""validate_effect_size.py 单元测试 (Issue #355)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.ci.validate_effect_size import (
    compute_improvement,
    load_rewards,
    main,
    validate_effect_size,
)


# ============================================================
# compute_improvement 单元测试
# ============================================================
class TestComputeImprovement:
    """测试提升百分比计算。"""

    def test_basic_improvement(self) -> None:
        """正常情况：target > baseline。"""
        result = compute_improvement([200.0, 180.0], [100.0, 100.0])
        # mean(target)=190, mean(baseline)=100, improvement=90%
        assert result["improvement_pct"] == pytest.approx(90.0, abs=0.01)
        assert result["target_mean"] == pytest.approx(190.0)
        assert result["baseline_mean"] == pytest.approx(100.0)
        assert result["n_target"] == 2
        assert result["n_baseline"] == 2

    def test_negative_improvement(self) -> None:
        """target < baseline 时返回负值。"""
        result = compute_improvement([50.0], [100.0])
        # (50 - 100) / 100 * 100 = -50%
        assert result["improvement_pct"] == pytest.approx(-50.0)

    def test_zero_baseline(self) -> None:
        """baseline_mean=0 时返回 inf 或 0。"""
        result = compute_improvement([100.0], [0.0])
        assert result["improvement_pct"] == float("inf")

    def test_zero_baseline_zero_target(self) -> None:
        """baseline=0 且 target=0 时返回 0。"""
        result = compute_improvement([0.0], [0.0])
        assert result["improvement_pct"] == 0.0

    def test_single_sample_std_zero(self) -> None:
        """单样本时 std=0。"""
        result = compute_improvement([100.0], [50.0])
        assert result["target_std"] == 0.0
        assert result["baseline_std"] == 0.0


# ============================================================
# load_rewards 单元测试
# ============================================================
class TestLoadRewards:
    """测试 JSON 数据加载。"""

    def test_load_wrapped_format(self, tmp_path: Path) -> None:
        """包装格式：{config: ..., rewards: {...}}。"""
        data = {
            "config": {"seeds": [42]},
            "rewards": {
                "PPO": [100.0, 200.0],
                "FCFS": [50.0, 60.0],
            },
        }
        fpath = tmp_path / "rewards.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        result = load_rewards(fpath)
        assert "PPO" in result
        assert "FCFS" in result
        assert result["PPO"] == [100.0, 200.0]
        assert result["FCFS"] == [50.0, 60.0]

    def test_load_direct_format(self, tmp_path: Path) -> None:
        """直接格式：{策略名: [奖励列表]}。"""
        data = {
            "PPO": [100.0],
            "FCFS": [50.0],
        }
        fpath = tmp_path / "rewards.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        result = load_rewards(fpath)
        assert result["PPO"] == [100.0]
        assert result["FCFS"] == [50.0]

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """文件不存在时抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            load_rewards(tmp_path / "nonexistent.json")

    def test_load_invalid_json_type(self, tmp_path: Path) -> None:
        """JSON 顶层非对象时抛出 ValueError。"""
        fpath = tmp_path / "rewards.json"
        fpath.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="顶层必须是对象"):
            load_rewards(fpath)

    def test_load_skips_non_list_entries(self, tmp_path: Path) -> None:
        """非列表条目被跳过。"""
        data = {
            "PPO": [100.0],
            "FCFS": "not_a_list",  # type: ignore[dict-item]
            "config": {"seeds": 42},
        }
        fpath = tmp_path / "rewards.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        result = load_rewards(fpath)
        assert "PPO" in result
        assert "FCFS" not in result
        assert "config" not in result


# ============================================================
# validate_effect_size 单元测试
# ============================================================
class TestValidateEffectSize:
    """测试效果量校验主函数。"""

    def test_pass_when_above_threshold(self, tmp_path: Path) -> None:
        """提升百分比 >= 阈值时通过。"""
        data = {
            "rewards": {
                "PPO": [200.0, 190.0],  # mean=195
                "FCFS": [100.0, 100.0],  # mean=100, improvement=95%
            }
        }
        fpath = tmp_path / "rewards.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        passed, details = validate_effect_size(fpath, threshold=0.80)
        assert passed is True
        assert details["improvement_pct"] == pytest.approx(95.0, abs=0.01)
        assert details["threshold_pct"] == pytest.approx(80.0)

    def test_fail_when_below_threshold(self, tmp_path: Path) -> None:
        """提升百分比 < 阈值时失败。"""
        data = {
            "rewards": {
                "PPO": [110.0],  # mean=110
                "FCFS": [100.0],  # mean=100, improvement=10%
            }
        }
        fpath = tmp_path / "rewards.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        passed, details = validate_effect_size(fpath, threshold=0.80)
        assert passed is False
        assert details["improvement_pct"] == pytest.approx(10.0)

    def test_pass_at_exact_threshold(self, tmp_path: Path) -> None:
        """提升百分比恰好等于阈值时通过（>=）。"""
        data = {
            "rewards": {
                "PPO": [180.0],  # mean=180
                "FCFS": [100.0],  # mean=100, improvement=80%
            }
        }
        fpath = tmp_path / "rewards.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        passed, _ = validate_effect_size(fpath, threshold=0.80)
        assert passed is True

    def test_missing_target_strategy_raises(self, tmp_path: Path) -> None:
        """目标策略不存在时抛出 ValueError。"""
        data = {"rewards": {"FCFS": [100.0]}}
        fpath = tmp_path / "rewards.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ValueError, match="目标策略"):
            validate_effect_size(fpath, threshold=0.80, target_strategy="PPO")

    def test_missing_baseline_strategy_raises(self, tmp_path: Path) -> None:
        """基线策略不存在时抛出 ValueError。"""
        data = {"rewards": {"PPO": [100.0]}}
        fpath = tmp_path / "rewards.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ValueError, match="基线策略"):
            validate_effect_size(fpath, threshold=0.80, baseline_strategy="FCFS")

    def test_custom_strategy_names(self, tmp_path: Path) -> None:
        """支持自定义策略名。"""
        data = {
            "rewards": {
                "DQN": [200.0],
                "SJF": [100.0],
            }
        }
        fpath = tmp_path / "rewards.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        passed, details = validate_effect_size(
            fpath, threshold=0.50, target_strategy="DQN", baseline_strategy="SJF"
        )
        assert passed is True
        assert details["target_strategy"] == "DQN"
        assert details["baseline_strategy"] == "SJF"


# ============================================================
# main CLI 单元测试
# ============================================================
class TestMainCLI:
    """测试命令行入口。"""

    def test_main_pass_exit_zero(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """通过时退出码 0。"""
        data = {
            "rewards": {
                "PPO": [200.0, 190.0],
                "FCFS": [100.0, 100.0],
            }
        }
        fpath = tmp_path / "rewards.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        exit_code = main(["--data", str(fpath), "--threshold", "0.80"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "[PASS]" in captured.out

    def test_main_fail_exit_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """失败时退出码 1。"""
        data = {
            "rewards": {
                "PPO": [110.0],
                "FCFS": [100.0],
            }
        }
        fpath = tmp_path / "rewards.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        exit_code = main(["--data", str(fpath), "--threshold", "0.80"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "[FAIL]" in captured.out

    def test_main_missing_file_exit_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """数据文件缺失时退出码 2（不阻断）。"""
        exit_code = main(["--data", str(tmp_path / "nonexistent.json")])
        assert exit_code == 2
        captured = capsys.readouterr()
        assert "[WARN]" in captured.out

    def test_main_invalid_json_exit_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """JSON 格式错误时退出码 2（不阻断）。"""
        fpath = tmp_path / "rewards.json"
        fpath.write_text("[1, 2, 3]", encoding="utf-8")

        exit_code = main(["--data", str(fpath)])
        assert exit_code == 2

    def test_main_default_data_path(self, capsys: pytest.CaptureFixture[str]) -> None:
        """使用默认数据路径（仓库权威数据）应通过。"""
        # 默认路径为 results/multiseed_evaluation/rewards_multiseed.json
        # 权威数据 PPO=1982.69, 真实FCFS=1648.91, improvement=20.2% > 10%
        exit_code = main(["--threshold", "0.10"])
        # 如果数据文件存在则应返回 0，否则返回 2（环境差异）
        assert exit_code in (0, 2)
        if exit_code == 0:
            captured = capsys.readouterr()
            assert "20.2" in captured.out or "20.2%" in captured.out
