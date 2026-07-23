"""多seed真机实验统计分析脚本测试（Issue #58 重构版）

测试 scripts/evaluation/multiseed_real_machine_analysis.py 的核心功能：
- _effect_level：Cohen's d 效应量等级划分
- _welch_t_test：Welch t 检验主分析 + 均值差 95% CI
- _paired_t_test：配对 t 检验敏感性分析 + Cohen's d_z
- _validate_data_provenance：数据来源验证（机器/shots 一致性）
- analyze：完整统计分析流程
- generate_report：Markdown 报告生成
- judgment 严格规则：bonferroni_significant=false → "不支持"
"""

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.multiseed_real_machine_analysis import (
    ALPHA_BONFERRONI,
    COMPARISONS,
    EXPECTED_MACHINE,
    EXPECTED_SHOTS,
    N_COMPARISONS,
    STRATEGIES,
    _effect_level,
    _paired_t_test,
    _validate_data_provenance,
    _welch_t_test,
    analyze,
    generate_report,
)

# ── 测试数据（统一协议：tianyan-287 + shots=32） ──

SAMPLE_DATA_VALID = {
    "experiment": "tianyan-287_multiseed_10seeds",
    "timestamp": "2026-07-22T03:06:25",
    "config": {
        "seeds": [42, 123, 456, 789, 1024],
        "strategies": ["ppo", "fcfs", "sjf"],
        "num_tasks": 32,
        "max_real_tasks_per_run": 1,
        "shots": 32,
        "machine": "tianyan-287",
    },
    "total_elapsed_seconds": 536.16,
    "results": [
        {
            "strategy": "PPO",
            "seed": 42,
            "machine": "tianyan-287",
            "shots": 32,
            "metrics": {"total_reward": 1560.86},
        },
        {
            "strategy": "FCFS",
            "seed": 42,
            "machine": "tianyan-287",
            "shots": 32,
            "metrics": {"total_reward": 410.23},
        },
        {
            "strategy": "SJF",
            "seed": 42,
            "machine": "tianyan-287",
            "shots": 32,
            "metrics": {"total_reward": 704.32},
        },
        {
            "strategy": "PPO",
            "seed": 123,
            "machine": "tianyan-287",
            "shots": 32,
            "metrics": {"total_reward": 1224.13},
        },
        {
            "strategy": "FCFS",
            "seed": 123,
            "machine": "tianyan-287",
            "shots": 32,
            "metrics": {"total_reward": 288.77},
        },
        {
            "strategy": "SJF",
            "seed": 123,
            "machine": "tianyan-287",
            "shots": 32,
            "metrics": {"total_reward": 502.91},
        },
        {
            "strategy": "PPO",
            "seed": 456,
            "machine": "tianyan-287",
            "shots": 32,
            "metrics": {"total_reward": 1615.17},
        },
        {
            "strategy": "FCFS",
            "seed": 456,
            "machine": "tianyan-287",
            "shots": 32,
            "metrics": {"total_reward": 317.66},
        },
        {
            "strategy": "SJF",
            "seed": 456,
            "machine": "tianyan-287",
            "shots": 32,
            "metrics": {"total_reward": 854.43},
        },
        {
            "strategy": "PPO",
            "seed": 789,
            "machine": "tianyan-287",
            "shots": 32,
            "metrics": {"total_reward": 2097.05},
        },
        {
            "strategy": "FCFS",
            "seed": 789,
            "machine": "tianyan-287",
            "shots": 32,
            "metrics": {"total_reward": 404.57},
        },
        {
            "strategy": "SJF",
            "seed": 789,
            "machine": "tianyan-287",
            "shots": 32,
            "metrics": {"total_reward": 390.39},
        },
        {
            "strategy": "PPO",
            "seed": 1024,
            "machine": "tianyan-287",
            "shots": 32,
            "metrics": {"total_reward": 1828.90},
        },
        {
            "strategy": "FCFS",
            "seed": 1024,
            "machine": "tianyan-287",
            "shots": 32,
            "metrics": {"total_reward": 344.85},
        },
        {
            "strategy": "SJF",
            "seed": 1024,
            "machine": "tianyan-287",
            "shots": 32,
            "metrics": {"total_reward": 383.93},
        },
    ],
}

# 混合机器/shots 的无效数据（模拟旧 10seeds_merged.json）
SAMPLE_DATA_INVALID = {
    "experiment": "tianyan-287_multiseed_10seeds",
    "timestamp": "2026-07-22T03:06:25",
    "config": {
        "seeds": [42, 123, 456, 789, 1024],
        "strategies": ["ppo", "fcfs", "sjf"],
        "shots": "32 (new5) / 1024 (old5)",
        "machine": "mixed",
    },
    "total_elapsed_seconds": 795.78,
    "results": [
        {
            "strategy": "PPO",
            "seed": 42,
            "machine": "tianyan-287",
            "shots": 1024,
            "metrics": {"total_reward": 1560.86},
        },
        {
            "strategy": "FCFS",
            "seed": 42,
            "machine": "tianyan-287",
            "shots": 1024,
            "metrics": {"total_reward": 410.23},
        },
        {
            "strategy": "PPO",
            "seed": 456,
            "machine": "tianyan176",
            "shots": 32,
            "metrics": {"total_reward": 1615.17},
        },
        {
            "strategy": "FCFS",
            "seed": 456,
            "machine": "tianyan176",
            "shots": 32,
            "metrics": {"total_reward": 317.66},
        },
    ],
}


@pytest.fixture
def tmp_data_file(tmp_path: Path) -> Path:
    """创建临时数据文件。"""
    data_file = tmp_path / "test_multiseed_data.json"
    with data_file.open("w", encoding="utf-8") as f:
        json.dump(SAMPLE_DATA_VALID, f)
    return data_file


# ── 常量测试 ──


class TestConstants:
    """测试常量定义。"""

    def test_strategies_order(self) -> None:
        """策略顺序应为 PPO, SJF, FCFS。"""
        assert STRATEGIES == ["PPO", "SJF", "FCFS"]

    def test_comparisons_count(self) -> None:
        """比较对数量应为 3。"""
        assert len(COMPARISONS) == 3

    def test_bonferroni_alpha(self) -> None:
        """Bonferroni 校正 α 应为 0.05/3 ≈ 0.0167。"""
        assert N_COMPARISONS == 3
        assert pytest.approx(0.01667, rel=0.01) == ALPHA_BONFERRONI

    def test_expected_machine_constant(self) -> None:
        """Issue #58：期望机器必须为 tianyan-287。"""
        assert EXPECTED_MACHINE == "tianyan-287"

    def test_expected_shots_constant(self) -> None:
        """Issue #58：期望 shots 必须为 32。"""
        assert EXPECTED_SHOTS == 32


# ── 效应量等级测试 ──


class TestEffectLevel:
    """测试 Cohen's d 效应量等级划分。"""

    def test_negligible(self) -> None:
        """d < 0.2 为可忽略。"""
        assert _effect_level(0.1) == "可忽略"

    def test_small(self) -> None:
        """0.2 ≤ d < 0.5 为小效应。"""
        assert _effect_level(0.3) == "小效应"

    def test_medium(self) -> None:
        """0.5 ≤ d < 0.8 为中效应。"""
        assert _effect_level(0.6) == "中效应"

    def test_large(self) -> None:
        """d ≥ 0.8 为大效应。"""
        assert _effect_level(0.8) == "大效应"
        assert _effect_level(1.0) == "大效应"
        assert _effect_level(5.0) == "大效应"


# ── Welch t 检验测试 ──


class TestWelchTTest:
    """测试 _welch_t_test 函数。"""

    def test_returns_dict_with_required_fields(self) -> None:
        """应返回包含所有必需字段的字典。"""
        result = _welch_t_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        required_fields = {"mean_diff", "t_stat", "p_value", "ci_95", "ci_crosses_zero", "df", "se"}
        assert required_fields.issubset(result.keys())

    def test_identical_groups_mean_diff_zero(self) -> None:
        """相同组均值差应为 0。"""
        a = [100.0, 200.0, 300.0]
        b = [100.0, 200.0, 300.0]
        result = _welch_t_test(a, b)
        assert result["mean_diff"] == pytest.approx(0.0)

    def test_different_groups_positive_mean_diff(self) -> None:
        """不同组均值差应为正且显著。"""
        a = [1000.0, 1100.0, 1200.0]
        b = [100.0, 200.0, 300.0]
        result = _welch_t_test(a, b)
        assert result["mean_diff"] == pytest.approx(900.0)
        assert result["t_stat"] > 0
        assert result["p_value"] < 0.05
        assert result["ci_95"][0] > 0
        assert result["ci_95"][1] > result["ci_95"][0]
        assert not result["ci_crosses_zero"]

    def test_ci_contains_mean_diff(self) -> None:
        """均值差应在 CI 范围内。"""
        a = [10.0, 20.0, 30.0, 40.0, 50.0]
        b = [5.0, 15.0, 25.0, 35.0, 45.0]
        result = _welch_t_test(a, b)
        ci_lo, ci_hi = result["ci_95"]
        mean_diff = result["mean_diff"]
        assert ci_lo < mean_diff < ci_hi

    def test_ci_crosses_zero_for_similar_groups(self) -> None:
        """相似组的 CI 应跨 0。"""
        a = [10.0, 20.0, 30.0]
        b = [11.0, 19.0, 31.0]
        result = _welch_t_test(a, b)
        assert result["ci_crosses_zero"]

    def test_empty_input_returns_neutral_result(self) -> None:
        """空输入应返回中性结果，不崩溃。"""
        result = _welch_t_test([], [])
        assert result["mean_diff"] == 0.0
        assert result["p_value"] == 1.0
        assert result["ci_crosses_zero"] is True
        assert result["se"] == 0.0

    def test_one_empty_input_returns_neutral_result(self) -> None:
        """一侧空输入应返回中性结果。"""
        result = _welch_t_test([1.0, 2.0, 3.0], [])
        assert result["p_value"] == 1.0
        assert result["ci_crosses_zero"] is True


# ── 配对 t 检验测试 ──


class TestPairedTTest:
    """测试 _paired_t_test 函数（敏感性分析）。"""

    def test_returns_unavailable_for_mismatched_lengths(self) -> None:
        """长度不一致应返回 paired_available=False。"""
        result = _paired_t_test([1.0, 2.0], [1.0])
        assert result["paired_available"] is False
        assert "reason" in result

    def test_returns_unavailable_for_insufficient_pairs(self) -> None:
        """少于 2 对样本应返回 paired_available=False。"""
        result = _paired_t_test([1.0], [1.0])
        assert result["paired_available"] is False

    def test_returns_available_for_matched_pairs(self) -> None:
        """配对样本充足时应返回完整结果。"""
        a = [10.0, 20.0, 30.0, 40.0, 50.0]
        b = [5.0, 15.0, 25.0, 35.0, 45.0]
        result = _paired_t_test(a, b)
        assert result["paired_available"] is True
        required = {
            "mean_diff",
            "t_stat",
            "p_value",
            "cohens_d_z",
            "effect_level",
            "ci_95",
            "ci_crosses_zero",
            "n_pairs",
        }
        assert required.issubset(result.keys())
        assert result["n_pairs"] == 5

    def test_cohens_d_z_sign_matches_diff(self) -> None:
        """Cohen's d_z 符号应与均值差符号一致。"""
        a = [100.0, 200.0, 300.0, 400.0, 500.0]
        b = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = _paired_t_test(a, b)
        assert result["mean_diff"] > 0
        assert result["cohens_d_z"] > 0


# ── 数据来源验证测试 ──


class TestValidateDataProvenance:
    """测试 _validate_data_provenance 函数。"""

    def test_valid_unified_protocol(self) -> None:
        """统一协议数据应标记为 valid。"""
        result = _validate_data_provenance(SAMPLE_DATA_VALID)
        assert result["valid"] is True
        assert result["invalid_for_formal_comparison"] is False
        assert result["reason"] == "unified_protocol"
        assert result["machines_found"] == ["tianyan-287"]
        assert result["shots_found"] == ["32"]

    def test_invalid_mixed_machines(self) -> None:
        """混合机器数据应标记为 invalid_for_formal_comparison。"""
        result = _validate_data_provenance(SAMPLE_DATA_INVALID)
        assert result["valid"] is False
        assert result["invalid_for_formal_comparison"] is True
        assert result["reason"] == "mixed_machines_or_shots"
        assert "tianyan-287" in result["machines_found"]
        assert "tianyan176" in result["machines_found"]

    def test_invalid_wrong_machine(self) -> None:
        """单机器但非 tianyan-287 应标记为 invalid。"""
        data = {
            "config": {"machine": "tianyan176", "shots": 32},
            "results": [
                {
                    "strategy": "PPO",
                    "seed": 42,
                    "machine": "tianyan176",
                    "shots": 32,
                    "metrics": {"total_reward": 1000.0},
                },
            ],
        }
        result = _validate_data_provenance(data)
        assert result["valid"] is False

    def test_invalid_wrong_shots(self) -> None:
        """单机器但 shots 不为 32 应标记为 invalid。"""
        data = {
            "config": {"machine": "tianyan-287", "shots": 1024},
            "results": [
                {
                    "strategy": "PPO",
                    "seed": 42,
                    "machine": "tianyan-287",
                    "shots": 1024,
                    "metrics": {"total_reward": 1000.0},
                },
            ],
        }
        result = _validate_data_provenance(data)
        assert result["valid"] is False

    def test_skips_smoke_test_records(self) -> None:
        """冒烟记录应被跳过。"""
        data = {
            "config": {"machine": "tianyan-287", "shots": 32},
            "results": [
                {
                    "strategy": "PPO",
                    "seed": 42,
                    "machine": "tianyan-287",
                    "shots": 32,
                    "metrics": {"total_reward": 1000.0},
                },
                {"smoke_test": True, "machine": "mock_machine", "shots": 999},
            ],
        }
        result = _validate_data_provenance(data)
        assert result["valid"] is True
        assert "mock_machine" not in result["machines_found"]


# ── 分析函数测试 ──


class TestAnalyze:
    """测试 analyze 函数。"""

    def test_descriptive_stats(self) -> None:
        """描述性统计应正确计算。"""
        analysis = analyze(SAMPLE_DATA_VALID)
        desc = analysis["descriptive"]
        assert desc["PPO"]["n"] == 5
        assert desc["PPO"]["mean"] == pytest.approx(1665.22, rel=0.01)
        assert desc["FCFS"]["mean"] == pytest.approx(353.22, rel=0.01)
        assert desc["SJF"]["mean"] == pytest.approx(567.20, rel=0.01)

    def test_data_provenance_present(self) -> None:
        """分析结果应包含数据来源验证。"""
        analysis = analyze(SAMPLE_DATA_VALID)
        assert "data_provenance" in analysis
        assert analysis["data_provenance"]["valid"] is True

    def test_pairwise_comparisons(self) -> None:
        """两两比较应包含所有比较对。"""
        analysis = analyze(SAMPLE_DATA_VALID)
        pairwise = analysis["pairwise"]
        assert "PPO_vs_FCFS" in pairwise
        assert "PPO_vs_SJF" in pairwise
        assert "SJF_vs_FCFS" in pairwise

    def test_pairwise_has_welch_fields(self) -> None:
        """每个比较应包含 Welch t 检验字段。"""
        analysis = analyze(SAMPLE_DATA_VALID)
        p = analysis["pairwise"]["PPO_vs_FCFS"]
        required = {
            "welch_t",
            "welch_p",
            "welch_df",
            "welch_se",
            "mean_diff_ci_95",
            "ci_crosses_zero",
            "cohens_d",
            "effect_level",
            "rank_biserial",
            "paired_analysis",
            "bonferroni_significant",
            "judgment",
        }
        assert required.issubset(p.keys())

    def test_pairwise_has_paired_analysis(self) -> None:
        """同 seed 设计应包含配对敏感性分析。"""
        analysis = analyze(SAMPLE_DATA_VALID)
        p = analysis["pairwise"]["PPO_vs_FCFS"]
        paired = p["paired_analysis"]
        assert paired["paired_available"] is True
        assert paired["n_pairs"] == 5
        assert "cohens_d_z" in paired

    def test_ppo_dominates_fcfs(self) -> None:
        """PPO 应在 vs FCFS 比较中胜出（大效应 + 显著）。"""
        analysis = analyze(SAMPLE_DATA_VALID)
        ppo_fcfs = analysis["pairwise"]["PPO_vs_FCFS"]
        assert ppo_fcfs["cohens_d"] > 0.8
        assert ppo_fcfs["bonferroni_significant"] is True
        assert ppo_fcfs["judgment"] == "支持"

    def test_analysis_protocol_present(self) -> None:
        """分析结果应包含统计协议描述。"""
        analysis = analyze(SAMPLE_DATA_VALID)
        protocol = analysis["analysis_protocol"]
        assert protocol["main_test"] == "Welch t-test"
        assert protocol["sensitivity_test"] == "paired t-test + Cohen's d_z"
        assert protocol["correction"] == "Bonferroni"
        assert protocol["n_comparisons"] == 3
        assert "judgment_rule" in protocol

    def test_no_duplicate_full_stats(self) -> None:
        """Issue #58：应删除重复的 full_stats 区块。"""
        analysis = analyze(SAMPLE_DATA_VALID)
        assert "full_stats" not in analysis
        assert "normality" not in analysis

    def test_invalid_data_provenance_flagged(self) -> None:
        """混合机器数据应被标记为 invalid_for_formal_comparison。"""
        analysis = analyze(SAMPLE_DATA_INVALID)
        assert analysis["data_provenance"]["valid"] is False
        assert analysis["data_provenance"]["invalid_for_formal_comparison"] is True


# ── judgment 严格规则测试 ──


class TestJudgmentRule:
    """Issue #58：judgment 严格规则测试。"""

    def test_judgment_support_when_significant(self) -> None:
        """Bonferroni 显著 + CI 不跨0 + d≥0.5 → 支持。"""
        analysis = analyze(SAMPLE_DATA_VALID)
        ppo_fcfs = analysis["pairwise"]["PPO_vs_FCFS"]
        assert ppo_fcfs["bonferroni_significant"] is True
        assert not ppo_fcfs["ci_crosses_zero"]
        assert ppo_fcfs["cohens_d"] >= 0.5
        assert ppo_fcfs["judgment"] == "支持"

    def test_judgment_not_support_when_not_significant(self) -> None:
        """bonferroni_significant=False 时 judgment 必须为「不支持」。

        Issue #58 严格规则，不存在「不确定」中间态。
        """
        # 构造不显著的数据：PPO 和 FCFS 的 reward 接近
        data = {
            "config": {"machine": "tianyan-287", "shots": 32, "seeds": [1, 2, 3, 4, 5]},
            "results": [
                {
                    "strategy": "PPO",
                    "seed": i,
                    "machine": "tianyan-287",
                    "shots": 32,
                    "metrics": {"total_reward": 1000.0 + i * 10},
                }
                for i in range(1, 6)
            ]
            + [
                {
                    "strategy": "FCFS",
                    "seed": i,
                    "machine": "tianyan-287",
                    "shots": 32,
                    "metrics": {"total_reward": 1000.0 + i * 5},
                }
                for i in range(1, 6)
            ]
            + [
                {
                    "strategy": "SJF",
                    "seed": i,
                    "machine": "tianyan-287",
                    "shots": 32,
                    "metrics": {"total_reward": 1000.0 + i * 7},
                }
                for i in range(1, 6)
            ],
        }
        analysis = analyze(data)
        for key in ("PPO_vs_FCFS", "PPO_vs_SJF", "SJF_vs_FCFS"):
            p = analysis["pairwise"][key]
            if not p["bonferroni_significant"]:
                assert p["judgment"] == "不支持", (
                    f"Issue #58: bonferroni_significant=False 时 judgment 必须为「不支持」"
                    f"，实际为 {p['judgment']}（{key}）"
                )


# ── 报告生成测试 ──


class TestGenerateReport:
    """测试报告生成。"""

    def test_report_contains_sections(self) -> None:
        """报告应包含 Issue #58 统一口径的所有章节。"""
        analysis = analyze(SAMPLE_DATA_VALID)
        report = generate_report(analysis, Path("test_data.json"))
        assert "# 天衍-287 多seed真机实验统计分析报告（Issue #58 统一口径）" in report
        assert "## 0. 数据来源验证" in report
        assert "## 1. 描述性统计" in report
        assert "## 2. 两两比较（Welch t-test 主分析）" in report
        assert "## 3. 汇总表" in report
        assert "## 4. 结论" in report

    def test_report_contains_key_numbers(self) -> None:
        """报告应包含关键数字。"""
        analysis = analyze(SAMPLE_DATA_VALID)
        report = generate_report(analysis, Path("test_data.json"))
        assert "Cohen's d" in report
        assert "Bonferroni" in report
        assert "95% CI" in report
        assert "Welch" in report

    def test_report_contains_provenance_warning_for_invalid(self) -> None:
        """无效数据报告应包含数据来源警告。"""
        analysis = analyze(SAMPLE_DATA_INVALID)
        report = generate_report(analysis, Path("test_data.json"))
        assert "invalid_for_formal_comparison" in report
        assert "数据来源警告" in report

    def test_report_contains_judgment(self) -> None:
        """报告应包含判定结果。"""
        analysis = analyze(SAMPLE_DATA_VALID)
        report = generate_report(analysis, Path("test_data.json"))
        assert "判定" in report

    def test_report_contains_paired_analysis(self) -> None:
        """报告应包含配对敏感性分析。"""
        analysis = analyze(SAMPLE_DATA_VALID)
        report = generate_report(analysis, Path("test_data.json"))
        assert "配对敏感性" in report
        assert "d_z" in report

    def test_report_contains_judgment_rule(self) -> None:
        """报告应包含判定规则说明。"""
        analysis = analyze(SAMPLE_DATA_VALID)
        report = generate_report(analysis, Path("test_data.json"))
        assert "bonferroni_significant=false" in report
