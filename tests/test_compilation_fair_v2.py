"""
Issue #451: 编译层公平对比脚本单元测试

测试 scripts/evaluation/compilation_fair_v2.py 的核心函数：
- generate_circuit_pool: 电路池生成可复现性
- compute_statistics: 统计计算正确性
- compute_category_breakdown: 类别统计
- compute_subset_analysis: 子集分析
- generate_report: 报告生成
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.evaluation.compilation_fair_v2 import (
    CATEGORIES,
    N_PER_CATEGORY,
    SEED,
    compute_category_breakdown,
    compute_statistics,
    compute_subset_analysis,
    generate_circuit_pool,
    generate_report,
)

# ---------------------------------------------------------------------------
# 1. 电路池生成测试
# ---------------------------------------------------------------------------


class TestCircuitPoolGeneration:
    """测试电路池生成的可复现性与结构。"""

    def test_pool_size_is_60(self) -> None:
        """电路池总数应为 60（3 类别 × 20）。"""
        pool = generate_circuit_pool(SEED)
        assert len(pool) == 60

    def test_pool_reproducible_with_same_seed(self) -> None:
        """相同种子应生成相同电路池（qubits/gates 一致）。"""
        pool1 = generate_circuit_pool(SEED)
        pool2 = generate_circuit_pool(SEED)
        for item1, item2 in zip(pool1, pool2, strict=False):
            assert item1["qubits"] == item2["qubits"]
            assert item1["gates"] == item2["gates"]
            assert item1["category"] == item2["category"]

    def test_pool_categories_distribution(self) -> None:
        """每个类别应有 N_PER_CATEGORY 个电路。"""
        pool = generate_circuit_pool(SEED)
        for cat_name in CATEGORIES:
            cat_count = sum(1 for item in pool if item["category"] == cat_name)
            assert cat_count == N_PER_CATEGORY

    def test_pool_qubits_within_category_range(self) -> None:
        """电路比特数应在类别配置范围内。

        注：random_circuit 的 n_gates 参数是期望门数，实际 qc.size() 可能不同，
        因此只验证 qubits 范围。
        """
        pool = generate_circuit_pool(SEED)
        for item in pool:
            cfg = CATEGORIES[item["category"]]
            assert cfg["qubits"][0] <= item["qubits"] <= cfg["qubits"][1]

    def test_pool_indices_unique_and_sequential(self) -> None:
        """电路索引应唯一且按序。"""
        pool = generate_circuit_pool(SEED)
        indices = [item["index"] for item in pool]
        assert indices == list(range(60))


# ---------------------------------------------------------------------------
# 2. 统计计算测试
# ---------------------------------------------------------------------------


class TestComputeStatistics:
    """测试统计分析函数。"""

    def test_statistics_basic_fields(self) -> None:
        """统计结果应包含所有必需字段。"""
        sabre = [10, 20, 30, 40, 50]
        ppo = [5, 10, 15, 20, 25]
        stats = compute_statistics(sabre, ppo)
        required_fields = [
            "n_pairs",
            "sabre_mean",
            "ppo_mean",
            "sabre_std",
            "ppo_std",
            "improvement_pct",
            "wilcoxon_w",
            "p_value",
            "rank_biserial",
            "cohen_d",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "significant",
        ]
        for field in required_fields:
            assert field in stats, f"缺少字段: {field}"

    def test_improvement_pct_positive_when_ppo_lower(self) -> None:
        """PPO SWAP 数低于 SABRE 时改进率应为正。"""
        sabre = [100, 200, 300]
        ppo = [50, 100, 150]
        stats = compute_statistics(sabre, ppo)
        assert stats["improvement_pct"] == pytest.approx(50.0, abs=0.01)

    def test_n_pairs_correct(self) -> None:
        """配对数应等于输入长度。"""
        sabre = [10, 20, 30, 40]
        ppo = [5, 10, 15, 20]
        stats = compute_statistics(sabre, ppo)
        assert stats["n_pairs"] == 4

    def test_significant_flag_correct(self) -> None:
        """significant 字段应基于 p_value < 0.05。"""
        sabre = [100, 200, 300, 400, 500]
        ppo = [1, 2, 3, 4, 5]
        stats = compute_statistics(sabre, ppo)
        assert stats["significant"] is True
        assert stats["p_value"] < 0.05

    def test_bootstrap_ci_contains_improvement(self) -> None:
        """Bootstrap CI 应包含实际改进率。"""
        sabre = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        ppo = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        stats = compute_statistics(sabre, ppo)
        assert stats["bootstrap_ci_low"] <= stats["improvement_pct"]
        assert stats["improvement_pct"] <= stats["bootstrap_ci_high"]

    def test_cohen_d_positive_when_ppo_better(self) -> None:
        """PPO 更优时 Cohen's d 应为正。"""
        sabre = [100, 200, 300, 400, 500]
        ppo = [10, 20, 30, 40, 50]
        stats = compute_statistics(sabre, ppo)
        assert stats["cohen_d"] > 0


# ---------------------------------------------------------------------------
# 3. 类别统计测试
# ---------------------------------------------------------------------------


class TestCategoryBreakdown:
    """测试类别统计函数。"""

    def test_breakdown_has_all_categories(self) -> None:
        """分类别统计应包含所有 3 个类别。"""
        per_circuit = [
            {"category": "shallow", "sabre_swap": 1, "ppo_swap": 3},
            {"category": "medium", "sabre_swap": 20, "ppo_swap": 10},
            {"category": "deep", "sabre_swap": 70, "ppo_swap": 30},
        ]
        breakdown = compute_category_breakdown(per_circuit)
        for cat_name in CATEGORIES:
            assert cat_name in breakdown

    def test_breakdown_improvement_calculation(self) -> None:
        """改进率计算应正确。"""
        per_circuit = [
            {"category": "shallow", "sabre_swap": 10, "ppo_swap": 5},
        ]
        breakdown = compute_category_breakdown(per_circuit)
        assert breakdown["shallow"]["improvement_pct"] == pytest.approx(50.0, abs=0.01)

    def test_breakdown_empty_category_handled(self) -> None:
        """空类别应被跳过。"""
        per_circuit = [{"category": "shallow", "sabre_swap": 10, "ppo_swap": 5}]
        breakdown = compute_category_breakdown(per_circuit)
        assert "shallow" in breakdown
        assert "medium" not in breakdown
        assert "deep" not in breakdown


# ---------------------------------------------------------------------------
# 4. 子集分析测试
# ---------------------------------------------------------------------------


class TestSubsetAnalysis:
    """测试子集分析函数。"""

    def test_subset_excludes_shallow(self) -> None:
        """子集应排除浅电路，仅包含中电路+深电路。"""
        per_circuit = [
            {"category": "shallow", "sabre_swap": 1, "ppo_swap": 3},
            {"category": "medium", "sabre_swap": 20, "ppo_swap": 10},
            {"category": "deep", "sabre_swap": 70, "ppo_swap": 30},
            {"category": "medium", "sabre_swap": 25, "ppo_swap": 15},
            {"category": "deep", "sabre_swap": 80, "ppo_swap": 40},
        ]
        result = compute_subset_analysis(per_circuit)
        assert result["n_pairs"] == 4
        assert result["subset"] == "medium+deep"

    def test_subset_stats_complete(self) -> None:
        """子集统计应包含所有统计字段。"""
        per_circuit = [
            {"category": "medium", "sabre_swap": 20, "ppo_swap": 10},
            {"category": "deep", "sabre_swap": 70, "ppo_swap": 30},
        ]
        result = compute_subset_analysis(per_circuit)
        stats = result["stats"]
        assert "improvement_pct" in stats
        assert "p_value" in stats
        assert "wilcoxon_w" in stats

    def test_subset_rationale_present(self) -> None:
        """子集分析应包含科学依据说明。"""
        per_circuit = [
            {"category": "medium", "sabre_swap": 20, "ppo_swap": 10},
            {"category": "deep", "sabre_swap": 70, "ppo_swap": 30},
        ]
        result = compute_subset_analysis(per_circuit)
        assert "rationale" in result
        assert len(result["rationale"]) > 0


# ---------------------------------------------------------------------------
# 5. 报告生成测试
# ---------------------------------------------------------------------------


class TestReportGeneration:
    """测试报告生成函数。"""

    def test_report_contains_required_sections(self) -> None:
        """报告应包含所有必需章节。"""
        stats = {
            "n_pairs": 60,
            "sabre_mean": 31.32,
            "ppo_mean": 18.43,
            "sabre_std": 55.33,
            "ppo_std": 10.89,
            "improvement_pct": 41.1,
            "wilcoxon_w": 906.0,
            "p_value": 0.652,
            "rank_biserial": -0.059,
            "cohen_d": 0.263,
            "per_circuit_improvement_mean": -559.7,
            "per_circuit_improvement_median": -200.0,
            "per_circuit_improvement_ci_low": -2857.5,
            "per_circuit_improvement_ci_high": 83.4,
            "bootstrap_ci_low": 7.0,
            "bootstrap_ci_high": 58.4,
            "significant": False,
        }
        breakdown = {
            "shallow": {
                "n": 20,
                "sabre_mean": 3.4,
                "sabre_std": 5.93,
                "ppo_mean": 6.7,
                "ppo_std": 2.27,
                "improvement_pct": -97.1,
            },
            "medium": {
                "n": 20,
                "sabre_mean": 22.55,
                "sabre_std": 33.54,
                "ppo_mean": 17.45,
                "ppo_std": 5.12,
                "improvement_pct": 22.6,
            },
            "deep": {
                "n": 20,
                "sabre_mean": 69.2,
                "sabre_std": 77.98,
                "ppo_mean": 31.15,
                "ppo_std": 4.56,
                "improvement_pct": 55.0,
            },
        }
        subset_analysis = {
            "subset": "medium+deep",
            "n_pairs": 40,
            "rationale": "测试依据",
            "stats": {
                "n_pairs": 40,
                "sabre_mean": 45.88,
                "ppo_mean": 24.30,
                "sabre_std": 60.0,
                "ppo_std": 8.0,
                "improvement_pct": 47.0,
                "wilcoxon_w": 200.0,
                "p_value": 0.204,
                "rank_biserial": 0.1,
                "cohen_d": 0.5,
                "per_circuit_improvement_mean": 30.0,
                "per_circuit_improvement_median": 25.0,
                "per_circuit_improvement_ci_low": -50.0,
                "per_circuit_improvement_ci_high": 80.0,
                "bootstrap_ci_low": 20.0,
                "bootstrap_ci_high": 60.0,
                "significant": False,
            },
        }
        per_circuit = [
            {
                "index": 0,
                "category": "shallow",
                "qubits": 5,
                "depth": 3,
                "gates": 7,
                "sabre_swap": 0,
                "ppo_swap": 3,
            }
        ]
        config = {"seed": 42, "model_path": "test_model.zip", "n_circuits": 60}

        report = generate_report(stats, breakdown, subset_analysis, per_circuit, config)

        assert "# 编译层 PPO SWAP 公平对比报告 v2" in report
        assert "## 一、公平对比设计" in report
        assert "## 二、总体对比" in report
        assert "## 三、分类别对比" in report
        assert "## 四、统计显著性检验" in report
        assert "## 五、子集分析" in report
        assert "## 六、逐电路明细" in report
        assert "## 七、与原 76.4% 数字对比" in report
        assert "## 八、结论" in report
        assert "## 九、复现方法" in report

    def test_report_contains_fair_number_not_76(self) -> None:
        """报告应包含新公平数字而非 76.4%。"""
        stats = {
            "n_pairs": 60,
            "sabre_mean": 31.32,
            "ppo_mean": 18.43,
            "sabre_std": 55.33,
            "ppo_std": 10.89,
            "improvement_pct": 41.1,
            "wilcoxon_w": 906.0,
            "p_value": 0.652,
            "rank_biserial": -0.059,
            "cohen_d": 0.263,
            "per_circuit_improvement_mean": -559.7,
            "per_circuit_improvement_median": -200.0,
            "per_circuit_improvement_ci_low": -2857.5,
            "per_circuit_improvement_ci_high": 83.4,
            "bootstrap_ci_low": 7.0,
            "bootstrap_ci_high": 58.4,
            "significant": False,
        }
        breakdown = {
            cat: {
                "n": 20,
                "sabre_mean": 30.0,
                "sabre_std": 10.0,
                "ppo_mean": 15.0,
                "ppo_std": 5.0,
                "improvement_pct": 50.0,
            }
            for cat in CATEGORIES
        }
        subset_analysis = {
            "subset": "medium+deep",
            "n_pairs": 40,
            "rationale": "测试",
            "stats": {
                "n_pairs": 40,
                "sabre_mean": 45.0,
                "ppo_mean": 22.0,
                "sabre_std": 30.0,
                "ppo_std": 6.0,
                "improvement_pct": 51.0,
                "wilcoxon_w": 150.0,
                "p_value": 0.05,
                "rank_biserial": 0.2,
                "cohen_d": 0.6,
                "per_circuit_improvement_mean": 40.0,
                "per_circuit_improvement_median": 35.0,
                "per_circuit_improvement_ci_low": -20.0,
                "per_circuit_improvement_ci_high": 70.0,
                "bootstrap_ci_low": 30.0,
                "bootstrap_ci_high": 65.0,
                "significant": True,
            },
        }
        per_circuit: list[dict[str, int]] = []
        config = {"seed": 42, "model_path": "test_model.zip", "n_circuits": 60}

        report = generate_report(stats, breakdown, subset_analysis, per_circuit, config)

        assert "41.1%" in report
        assert "76.4%" in report  # 在对比表中应出现原数字作为对照


# ---------------------------------------------------------------------------
# 6. 集成测试（使用真实统计计算）
# ---------------------------------------------------------------------------


class TestIntegration:
    """集成测试：验证统计计算端到端正确性。"""

    def test_end_to_end_statistics_with_known_data(self) -> None:
        """使用已知数据验证统计计算端到端正确。"""
        # PPO 显著低于 SABRE 的场景
        sabre = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        ppo = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        stats = compute_statistics(sabre, ppo)

        assert stats["sabre_mean"] == pytest.approx(550.0)
        assert stats["ppo_mean"] == pytest.approx(55.0)
        assert stats["improvement_pct"] == pytest.approx(90.0, abs=0.1)
        assert stats["significant"] is True
        assert stats["p_value"] < 0.05
        assert stats["rank_biserial"] > 0  # PPO 更优

    def test_per_circuit_data_structure(self) -> None:
        """验证逐电路数据结构（来自实际脚本输出的 JSON 格式）。"""
        summary_path = "results/compilation/fair_v2_summary.json"
        if not os.path.exists(summary_path):
            pytest.skip("公平对比 v2 汇总 JSON 不存在，跳过")

        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        assert "config" in summary
        assert "stats" in summary
        assert "category_breakdown" in summary
        assert "subset_analysis" in summary
        assert summary["config"]["seed"] == SEED
        assert summary["stats"]["n_pairs"] == 60
