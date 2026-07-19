"""权威实验数字审计测试。"""

from pathlib import Path

from scripts.ci.audit_authoritative_metrics import (
    audit_repository,
    find_forbidden,
    validate_canonical_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_find_forbidden_reports_line_and_value() -> None:
    """旧版比例和奖励应被精确定位。"""
    findings = find_forbidden("当前 +92.4%\n旧版 +95.4% 和 2864.26")

    assert findings == [
        (2, "旧提升比例", "+95.4%"),
        (2, "旧 PPO 奖励", "2864.26"),
    ]


def test_canonical_report_requires_complete_ranking() -> None:
    """只写核心数字但缺少排名时不应通过。"""
    errors = validate_canonical_report("2723.0 1457.0 +86.9% Obs10Wrapper 14 维")

    assert "权威报告缺少完整的八策略排名" in errors


def test_repository_authoritative_metrics_are_consistent() -> None:
    """仓库当前文档、代码和 Office 材料应使用同一口径。"""
    assert audit_repository(PROJECT_ROOT) == []
