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
    findings = find_forbidden("当前 +86.9%\n旧版 +95.4% 和 2723.0")

    assert findings == [
        (2, "旧提升比例", "+95.4%"),
        (2, "旧 PPO 奖励", "2723.0"),
    ]


def test_find_forbidden_skips_audit_exempt_lines() -> None:
    """带 audit-exempt 标记的历史数据行应被跳过，不报禁止数字。

    豁免仅对带标记的当前行生效；同一文件中其他不带标记的行仍应被审计。
    """
    text = (
        "当前 +88.3%\n"
        "历史 2723.0 ± 138.2 <!-- audit-exempt: historical 10-seed -->\n"
        "旧版 2723.0 不带豁免标记"
    )
    findings = find_forbidden(text)

    # 第 1 行无禁止数字；第 2 行带豁免标记跳过；第 3 行仍应报错
    assert findings == [(3, "旧 PPO 奖励", "2723.0")]


def test_canonical_report_requires_complete_ranking() -> None:
    """只写核心数字但缺少排名时不应通过。"""
    errors = validate_canonical_report("2746.94 1458.77 +88.3% Obs10Wrapper 14 维")

    assert "权威报告缺少完整的八策略排名" in errors


def test_repository_authoritative_metrics_are_consistent() -> None:
    """仓库当前文档、代码和 Office 材料应使用同一口径。"""
    assert audit_repository(PROJECT_ROOT) == []
