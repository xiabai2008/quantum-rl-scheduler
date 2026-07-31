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
        "当前 +123.4%\n"
        "历史 2723.0 ± 138.2 <!-- audit-exempt: historical 10-seed -->\n"
        "旧版 2723.0 不带豁免标记"
    )
    findings = find_forbidden(text)

    # 第 1 行无禁止数字；第 2 行带豁免标记跳过；第 3 行仍应报错
    assert findings == [(3, "旧 PPO 奖励", "2723.0")]


def test_find_forbidden_no_false_positive_without_context() -> None:
    """#173：同名合法数字不应因缺少上下文而误报。

    例如 ``duration_sec = 2864 秒`` 中的 2864 只是合法训练/计时数字，
    附近无 PPO/奖励/提升/旧版 等关键词，不应被判定为旧 PPO 奖励。
    """
    findings = find_forbidden("duration_sec = 2864 秒")
    assert findings == []


def test_find_forbidden_catches_old_metric_with_context() -> None:
    """#173：真实旧指标行（带上下文关键词）仍应被捕获。

    验证文本上下文锚定没有削弱对真实旧指标行的捕获能力。
    """
    findings = find_forbidden("旧版 PPO 奖励 2864")
    assert findings == [(1, "旧 PPO 奖励", "2864")]


def test_find_forbidden_collects_all_matches_on_line() -> None:
    """#188：一行内同一禁用值重复出现应报告全部匹配（finditer）。

    这里关闭上下文要求以隔离验证多匹配收集能力；真实审计默认要求上下文。
    """
    findings = find_forbidden("出现了 95.4% 这里 95.4% 又出现", require_context=False)
    assert len(findings) == 2
    assert all(label == "旧提升比例" for _, label, _ in findings)


def test_canonical_report_requires_complete_ranking() -> None:
    """只写核心数字但缺少排名时不应通过。"""
    errors = validate_canonical_report("2348.91 1051.59 +123.4% OBS_DIM=16 16 维")

    assert "权威报告缺少完整的八策略排名" in errors


def test_repository_authoritative_metrics_are_consistent() -> None:
    """仓库当前文档、代码和 Office 材料应使用同一口径。"""
    assert audit_repository(PROJECT_ROOT) == []
