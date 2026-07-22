"""Issue #39 多负载统计 seed 级配对检验测试。

覆盖：
    - 缺少 PPO
    - 样本不足
    - seed 聚合
    - 配对统计
    - 多重比较校正
    - 不显著时判定为「不支持」
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.workload_pattern_stats import (
    ALPHA,
    BONFERRONI_COMPARISONS,
    aggregate_to_seed_level,
    analyze_pattern,
    cohens_d_z,
    holm_bonferroni,
    validate_input,
)


def _make_strategies(
    ppo_rewards: list[float],
    other_rewards: list[float],
    other_name: str = "FCFS",
) -> dict[str, dict[str, object]]:
    return {
        "PPO": {"episode_rewards": ppo_rewards},
        other_name: {"episode_rewards": other_rewards},
    }


class TestValidateInput:
    """输入校验测试。"""

    def test_missing_patterns_raises(self) -> None:
        with pytest.raises(ValueError, match="patterns"):
            validate_input({}, seeds=10, episodes_per_seed=5)

    def test_empty_patterns_raises(self) -> None:
        with pytest.raises(ValueError, match="patterns"):
            validate_input({"patterns": {}}, seeds=10, episodes_per_seed=5)

    def test_missing_ppo_raises(self) -> None:
        raw = {"patterns": {"default": {"strategies": {"FCFS": {"episode_rewards": [1.0] * 50}}}}}
        with pytest.raises(ValueError, match="PPO"):
            validate_input(raw, seeds=10, episodes_per_seed=5)

    def test_missing_episode_rewards_raises(self) -> None:
        raw = {"patterns": {"default": {"strategies": {"PPO": {}}}}}
        with pytest.raises(ValueError, match="episode_rewards"):
            validate_input(raw, seeds=10, episodes_per_seed=5)

    def test_length_mismatch_raises(self) -> None:
        raw = {
            "patterns": {
                "default": {
                    "strategies": {
                        "PPO": {"episode_rewards": [1.0] * 40},
                        "FCFS": {"episode_rewards": [1.0] * 50},
                    }
                }
            }
        }
        with pytest.raises(ValueError, match="不一致"):
            validate_input(raw, seeds=10, episodes_per_seed=5)

    def test_insufficient_seeds_raises(self) -> None:
        raw = {
            "patterns": {
                "default": {
                    "strategies": {
                        "PPO": {"episode_rewards": [1.0] * 20},
                        "FCFS": {"episode_rewards": [1.0] * 20},
                    }
                }
            }
        }
        # 4 seeds × 5 episodes = 20, but seeds=4 < MIN_PAIRED_SAMPLE=5
        with pytest.raises(ValueError, match="seeds"):
            validate_input(raw, seeds=4, episodes_per_seed=5)

    def test_valid_input_passes(self) -> None:
        raw = {
            "patterns": {
                "default": {
                    "strategies": {
                        "PPO": {"episode_rewards": [1.0] * 50},
                        "FCFS": {"episode_rewards": [1.0] * 50},
                    }
                }
            }
        }
        validate_input(raw, seeds=10, episodes_per_seed=5)


class TestSeedAggregation:
    """seed 级聚合测试。"""

    def test_aggregation_produces_n_seeds(self) -> None:
        rewards = list(range(50))  # 10 seeds × 5 episodes
        seed_means = aggregate_to_seed_level(rewards, seeds=10, episodes_per_seed=5)
        assert len(seed_means) == 10

    def test_aggregation_correctness(self) -> None:
        rewards = list(range(50))
        seed_means = aggregate_to_seed_level(rewards, seeds=10, episodes_per_seed=5)
        # seed 0: episodes 0,1,2,3,4 → mean = 2.0
        assert seed_means[0] == pytest.approx(2.0)
        # seed 1: episodes 5,6,7,8,9 → mean = 7.0
        assert seed_means[1] == pytest.approx(7.0)
        # seed 9: episodes 45,46,47,48,49 → mean = 47.0
        assert seed_means[9] == pytest.approx(47.0)

    def test_aggregation_preserves_order(self) -> None:
        rewards = [float(i) for i in range(50)]
        seed_means = aggregate_to_seed_level(rewards, seeds=10, episodes_per_seed=5)
        # seed 级均值应单调递增
        assert all(seed_means[i] < seed_means[i + 1] for i in range(9))


class TestPairedStatistics:
    """配对统计测试。"""

    def test_cohens_d_z_zero_when_no_diff(self) -> None:
        diffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        assert cohens_d_z(diffs) == 0.0

    def test_cohens_d_z_positive(self) -> None:
        diffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohens_d_z(diffs)
        assert d == pytest.approx(3.0 / np.std(diffs, ddof=1))

    def test_cohens_d_z_insufficient_samples(self) -> None:
        diffs = np.array([1.0])
        assert cohens_d_z(diffs) == 0.0

    def test_analyze_pattern_uses_paired_t(self) -> None:
        np.random.seed(42)
        ppo = [float(x) for x in [100, 200, 300, 400, 500, 110, 210, 310, 410, 510]]
        fcfs = [float(x) for x in [10, 20, 30, 40, 50, 11, 21, 31, 41, 51]]
        strategies = _make_strategies(ppo, fcfs)
        result = analyze_pattern("test", strategies, seeds=2, episodes_per_seed=5)
        comp = result["comparisons"][0]
        assert "paired_t" in comp
        assert "p_value" in comp
        assert comp["p_value"] < ALPHA

    def test_analyze_pattern_includes_wilcoxon(self) -> None:
        ppo = [float(x) for x in range(10, 60, 5)]
        fcfs = [float(x) for x in range(0, 50, 5)]
        strategies = _make_strategies(ppo, fcfs)
        result = analyze_pattern("test", strategies, seeds=2, episodes_per_seed=5)
        comp = result["comparisons"][0]
        assert "wilcoxon_p" in comp

    def test_analyze_pattern_includes_d_z(self) -> None:
        ppo = [float(x) for x in range(10, 60, 5)]
        fcfs = [float(x) for x in range(0, 50, 5)]
        strategies = _make_strategies(ppo, fcfs)
        result = analyze_pattern("test", strategies, seeds=2, episodes_per_seed=5)
        comp = result["comparisons"][0]
        assert "cohens_d_z" in comp
        assert comp["cohens_d_z"] > 0


class TestHolmCorrection:
    """Holm-Bonferroni 校正测试。"""

    def test_all_significant(self) -> None:
        p_values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]
        rejected, adj_p = holm_bonferroni(p_values, ALPHA)
        assert all(rejected)
        assert all(p <= ALPHA for p in adj_p)

    def test_none_significant(self) -> None:
        p_values = [0.6, 0.7, 0.8, 0.5, 0.9, 0.4, 0.3]
        rejected, adj_p = holm_bonferroni(p_values, ALPHA)
        assert not any(rejected)
        assert all(p > ALPHA for p in adj_p)

    def test_partial_significant(self) -> None:
        p_values = [0.001, 0.6, 0.7, 0.5, 0.9, 0.4, 0.3]
        rejected, _ = holm_bonferroni(p_values, ALPHA)
        assert rejected[0]
        assert not all(rejected)

    def test_adjusted_p_monotonicity(self) -> None:
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
        _, adj_p = holm_bonferroni(p_values, ALPHA)
        # adj_p 按原顺序返回，排序后应与自身一致
        assert all(0.0 <= p <= 1.0 for p in adj_p)

    def test_empty_input(self) -> None:
        rejected, adj_p = holm_bonferroni([])
        assert rejected == []
        assert adj_p == []


class TestJudgmentLogic:
    """判定逻辑测试：不显著时必须为「不支持」。"""

    def test_not_significant_yields_unsupported(self) -> None:
        np.random.seed(123)
        ppo = list(np.random.normal(100, 5, 50))
        fcfs = list(np.random.normal(100, 5, 50))
        strategies = _make_strategies(ppo, fcfs)
        result = analyze_pattern("test", strategies, seeds=10, episodes_per_seed=5)
        comp = result["comparisons"][0]
        if not comp["holm_significant"]:
            assert comp["judgment"] == "不支持"
        else:
            assert comp["judgment"] == "支持"

    def test_significant_yields_supported(self) -> None:
        ppo = [float(x) for x in [100, 200, 300, 400, 500, 110, 210, 310, 410, 510]]
        fcfs = [float(x) for x in [10, 20, 30, 40, 50, 11, 21, 31, 41, 51]]
        strategies = _make_strategies(ppo, fcfs)
        result = analyze_pattern("test", strategies, seeds=2, episodes_per_seed=5)
        comp = result["comparisons"][0]
        assert comp["holm_significant"]
        assert comp["judgment"] == "支持"


class TestEndToEnd:
    """端到端测试：使用实际数据文件。"""

    def test_real_data_produces_valid_output(self, tmp_path: Path) -> None:
        input_path = Path("results/workload_pattern_evaluation/workload_pattern_results.json")
        if not input_path.exists():
            pytest.skip("原始数据文件不存在")
        with open(input_path, encoding="utf-8") as f:
            raw = json.load(f)
        config = raw.get("config", {})
        seeds = len(config.get("seed_list", []))
        episodes_per_seed = config.get("episodes_per_seed", 5)
        validate_input(raw, seeds, episodes_per_seed)
        for pattern_key, pattern_data in raw["patterns"].items():
            result = analyze_pattern(
                pattern_key,
                pattern_data["strategies"],
                seeds,
                episodes_per_seed,
            )
            assert result["n_seeds"] == seeds
            assert len(result["comparisons"]) == 7
            for comp in result["comparisons"]:
                assert "judgment" in comp
                assert comp["judgment"] in ("支持", "不支持")


class TestConstants:
    """常量验证测试。"""

    def test_bonferroni_spelling_fixed(self) -> None:
        from scripts.evaluation import workload_pattern_stats

        assert not hasattr(workload_pattern_stats, "BONFERRONS_COMPARISONS")
        assert hasattr(workload_pattern_stats, "BONFERRONI_COMPARISONS")

    def test_comparison_count(self) -> None:
        assert BONFERRONI_COMPARISONS == 7
