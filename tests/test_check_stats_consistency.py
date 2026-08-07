"""tests/test_check_stats_consistency.py — Issue #691 统计口径一致性检查 BLACKLIST 单元测试"""

from __future__ import annotations

import re

from scripts.ci.check_stats_consistency import (
    BLACKLIST_PATTERNS,
    _is_honest_disclosure,
    _normalize_chinese_numerals,
)


def _run_blacklist_check_on_text(text: str) -> list[tuple[str, str]]:
    """对文本逐行运行 BLACKLIST 检查，返回命中的 (pattern, message) 列表。

    8.7-v2：与 scan_markdown_file 一致，先对中文数字归一化，确保"百分之X"
    中文表述的废弃值也能被检测到。
    """
    violations: list[tuple[str, str]] = []
    for line in text.splitlines():
        if _is_honest_disclosure(line):
            continue
        norm_line = _normalize_chinese_numerals(line)
        for pattern, message in BLACKLIST_PATTERNS:
            if re.search(pattern, norm_line, re.IGNORECASE):
                violations.append((pattern, message))
    return violations


# ---------------------------------------------------------------------------
# 正例测试：应该命中旧值 BLACKLIST 规则
# ---------------------------------------------------------------------------


class TestOldRealMachinePPOBlacklistPositive:
    """PPO 真机 1665.22 旧值 BLACKLIST 正例测试"""

    def test_detects_ppo_real_machine_1665_old_value(self) -> None:
        """PPO+真机+1665.22 三者同时出现应触发 BLACKLIST。"""
        text = "### 真机结果\nPPO 真机奖励 1665.22\n"
        violations = _run_blacklist_check_on_text(text)
        assert any("1736.32" in msg for _pat, msg in violations), (
            "应检测到真机PPO旧值1665.22并提示升级为1736.32"
        )

    def test_detects_real_machine_ppo_1665_order_variant(self) -> None:
        """关键词顺序变化（真机+PPO+1665.22）也应触发。"""
        text = "真机实验结果：PPO策略均值=1665.22\n"
        violations = _run_blacklist_check_on_text(text)
        assert any("1736.32" in msg for _pat, msg in violations), (
            "真机+PPO+1665.22顺序变体应触发BLACKLIST"
        )


class TestOldRealMachineFCFSBlacklistPositive:
    """FCFS 真机 353.22 旧值 BLACKLIST 正例测试"""

    def test_detects_fcfs_real_machine_353_old_value(self) -> None:
        """FCFS+真机+353.22 三者同时出现应触发 BLACKLIST。"""
        text = "## 真机性能\nFCFS 真机平均奖励 353.22\n"
        violations = _run_blacklist_check_on_text(text)
        assert any("383.00" in msg for _pat, msg in violations), (
            "应检测到真机FCFS旧值353.22并提示升级为383.00"
        )

    def test_detects_353_with_fcfs_real_machine_prefix(self) -> None:
        """353.22+FCFS+真机 前缀变体也应触发。"""
        text = "基准策略结果：353.22 为 FCFS 真机均值\n"
        violations = _run_blacklist_check_on_text(text)
        assert any("383.00" in msg for _pat, msg in violations), (
            "353.22+FCFS+真机变体应触发BLACKLIST"
        )


# ---------------------------------------------------------------------------
# 反例测试：不应误报
# ---------------------------------------------------------------------------


class TestOldRealMachinePPOBlacklistNegative:
    """PPO 1665.22 孤立出现不误报"""

    def test_no_false_positive_isolated_1665_without_real_machine(self) -> None:
        """单独出现 1665.22 但无「真机」关键词时，不应触发。"""
        text = "某非真机仿真实验奖励 1665.22，效果良好\n"
        violations = _run_blacklist_check_on_text(text)
        assert not any("1736.32" in msg for _pat, msg in violations), (
            "孤立1665.22不含真机关键词时不应误报"
        )

    def test_no_false_positive_ppo_without_real_machine_keyword(self) -> None:
        """PPO+1665.22 但无真机时，不应触发。"""
        text = "仿真基线：PPO 奖励 1665.22（仅仿真参考）\n"
        violations = _run_blacklist_check_on_text(text)
        assert not any("1736.32" in msg for _pat, msg in violations), (
            "PPO+1665.22无真机关键词不应触发BLACKLIST"
        )


class TestOldRealMachineFCFSBlacklistNegative:
    """FCFS 353.22 孤立出现不误报"""

    def test_no_false_positive_isolated_353_without_real_machine(self) -> None:
        """单独出现 353.22 但无「真机」关键词时，不应触发。"""
        text = "本地调试输出：353.22 为某中间结果\n"
        violations = _run_blacklist_check_on_text(text)
        assert not any("383.00" in msg for _pat, msg in violations), (
            "孤立353.22不含真机关键词时不应误报"
        )

    def test_no_false_positive_fcfs_without_real_machine_keyword(self) -> None:
        """FCFS+353.22 但无真机时，不应触发。"""
        text = "高负载场景：FCFS奖励 353.22（纯仿真）\n"
        violations = _run_blacklist_check_on_text(text)
        assert not any("383.00" in msg for _pat, msg in violations), (
            "FCFS+353.22无真机关键词不应触发BLACKLIST"
        )


# ---------------------------------------------------------------------------
# 诚实披露豁免测试：N=5/已废弃标注时旧值应豁免
# ---------------------------------------------------------------------------


class TestHonestDisclosureExemption:
    """旧值在诚实披露上下文中应被豁免"""

    def test_n5_annotation_exempts_old_ppo_1665(self) -> None:
        """包含「5 seeds」诚实披露标注的旧PPO值应豁免。"""
        text = "（历史参考，N=5旧实验已废弃）PPO真机值为1665.22\n"
        violations = _run_blacklist_check_on_text(text)
        assert not any("1736.32" in msg for _pat, msg in violations), (
            "含N=5诚实披露时旧值1665.22应豁免"
        )

    def test_deprecated_annotation_exempts_old_fcfs_353(self) -> None:
        """包含「已废弃」标注的旧FCFS值应豁免。"""
        text = "注：以下值已废弃（N=5旧版）：FCFS真机 353.22\n"
        violations = _run_blacklist_check_on_text(text)
        assert not any("383.00" in msg for _pat, msg in violations), (
            "含已废弃诚实披露时旧值353.22应豁免"
        )


# ---------------------------------------------------------------------------
# 中文数字归一化测试（8.7-v2 门禁加固）
# 背景：演示脚本曾用中文数字（"百分之一百二十三点四"）表达废弃统计量，
# 绕过了 ASCII 黑名单。归一化后应能被 BLACKLIST 捕获。
# ---------------------------------------------------------------------------


class TestChineseNumeralNormalization:
    """中文数字归一化函数正确性"""

    def test_normalizes_percentage(self) -> None:
        assert "123.4%" in _normalize_chinese_numerals("百分之一百二十三点四")
        assert "20.2%" in _normalize_chinese_numerals("百分之二十点二")
        assert "84.6%" in _normalize_chinese_numerals("百分之八十四点六")

    def test_normalizes_integer(self) -> None:
        assert "2349" in _normalize_chinese_numerals("两千三百四十九")
        assert "1983" in _normalize_chinese_numerals("一千九百八十三")

    def test_normalizes_decimal(self) -> None:
        assert "0.246" in _normalize_chinese_numerals("零点二四六")

    def test_plain_ascii_unchanged(self) -> None:
        assert _normalize_chinese_numerals("avg=1982.69 +20.2%") == "avg=1982.69 +20.2%"


class TestChineseDeprecatedValueCaughtByBlacklist:
    """中文废弃值经归一化后应触发 BLACKLIST"""

    def test_chinese_123_percent_triggered(self) -> None:
        """中文"百分之一百二十三点四"归一化后应触发 +123.4% 废弃黑名单。"""
        text = "PPO 相对 FCFS 提升百分之一百二十三点四\n"
        violations = _run_blacklist_check_on_text(text)
        assert any("123.4" in msg for _pat, msg in violations), (
            "中文废弃值+123.4%经归一化后应被BLACKLIST捕获"
        )

    def test_chinese_2349_not_blacklisted_by_rate_pattern(self) -> None:
        """中文"两千三百四十九"正常归一化，且不误触发百分比黑名单。"""
        assert "2349" in _normalize_chinese_numerals("奖励达到两千三百四十九")


class TestNewDeprecatedValuesPositive:
    """8.7-v2 新增：提交物文档残留旧口径的 BLACKLIST 正例"""

    def test_mappo_reward_improvement_863_triggers(self) -> None:
        """将 +86.3% 直接表述为'奖励提升'应触发（权威协同优势为 +84.6%）。"""
        text = "MAPPO 多智能体协同调度，奖励提升 86.3%\n"
        violations = _run_blacklist_check_on_text(text)
        assert any("84.6" in msg for _pat, msg in violations), (
            "直接表述'奖励提升86.3%'应被BLACKLIST捕获并提示权威+84.6%"
        )

    def test_p_less_than_1e66_triggers(self) -> None:
        """p<10⁻⁶⁶ 旧 p 值应触发（权威为 p=7.56e-12）。"""
        text = "PPO vs FCFS +20.2%（N=250, p<10⁻⁶⁶）\n"
        violations = _run_blacklist_check_on_text(text)
        assert any("7.56e-12" in msg for _pat, msg in violations), (
            "p<10⁻⁶⁶ 旧弱基线 p 值应被BLACKLIST捕获"
        )

    def test_significantly_better_than_sabre_triggers(self) -> None:
        """'显著优于 SABRE' 定论表述应触发（编译层为事后子集方向性证据）。"""
        text = "深电路上 PPO 显著优于 SABRE\n"
        violations = _run_blacklist_check_on_text(text)
        assert any("不构成" in msg for _pat, msg in violations), (
            "'显著优于SABRE'定论表述应被BLACKLIST捕获"
        )

    def test_robustness_improvement_triggers(self) -> None:
        """'鲁棒性提升'正向措辞应触发（噪声反馈为负向证据）。"""
        text = "量子赋能AI：真机噪声反馈优化PPO鲁棒性提升\n"
        violations = _run_blacklist_check_on_text(text)
        assert any("噪声敏感性" in msg for _pat, msg in violations), (
            "'鲁棒性提升'正向措辞应被BLACKLIST捕获"
        )


class TestNewDeprecatedValuesNegative:
    """8.7-v2 新增：诚实拆解/合法表述不应误触发"""

    def test_863_as_total_vs_single_machine_not_triggered(self) -> None:
        """+86.3% 作为'总提升 vs 单机'的诚实拆解不应触发。"""
        text = "协同优势 +84.6%；叠加规模扩展效应后总提升 vs 单机 +86.3%\n"
        violations = _run_blacklist_check_on_text(text)
        assert not any("84.6" in msg for _pat, msg in violations), (
            "诚实拆解为总提升vs单机的+86.3%不应被误判"
        )

    def test_ppo_significantly_better_than_random_not_triggered(self) -> None:
        """调度层'PPO 优于 Random/FCFS'为合法显著结论，不应误触发编译层 SABRE 规则。"""
        text = "PPO 显著优于 Random（p<0.001）\n"
        violations = _run_blacklist_check_on_text(text)
        assert not any("不构成" in msg for _pat, msg in violations), (
            "'PPO显著优于Random'非SABRE定论表述，不应命中编译层规则"
        )


class TestOrphanDeprecatedUtilization:
    """8.7-v3 红队审查新增：孤立废弃利用率值的 A9 检测"""

    def test_orphan_489_registered(self) -> None:
        """+48.9% 应被 A9 孤立废弃值表收录（权威 -3.3%）。"""
        import scripts.ci.check_stats_consistency as mod

        orphan = {old: new for old, new, _ in mod._ORPHAN_DEPRECATED}
        assert orphan["+48.9%"] == "-3.3%", "+48.9%（N=1 单次运行，已证伪）应映射到权威 -3.3%"

    def test_orphan_336_registered(self) -> None:
        """旧口径 33.6% 应被 A9 孤立废弃值表收录（权威 -3.3%）。"""
        import scripts.ci.check_stats_consistency as mod

        orphan = {old: new for old, new, _ in mod._ORPHAN_DEPRECATED}
        assert orphan["33.6%"] == "-3.3%", "旧口径 33.6%→50%（N=1）应映射到权威 -3.3%"


class TestUtilization72AndDemo65Positive:
    """8.7-v3 红队审查新增：利用率 72% 荒谬数字与演示 65% 的 BLACKLIST 正例"""

    def test_utilization_72_triggers(self) -> None:
        """'利用率提升至72%' 应触发（权威 -3.3%）。"""
        text = "异质化调度 + 多机器协同，利用率提升至72%\n"
        violations = _run_blacklist_check_on_text(text)
        assert any("-3.3%" in msg for _pat, msg in violations), (
            "利用率72%荒谬数字应被BLACKLIST捕获并提示权威-3.3%"
        )

    def test_utilization_72_plain_triggers(self) -> None:
        """'利用率 72%' 应触发。"""
        text = "多机器协同，利用率 72%\n"
        violations = _run_blacklist_check_on_text(text)
        assert any("-3.3%" in msg for _pat, msg in violations), (
            "利用率 72% 应被BLACKLIST捕获"
        )

    def test_demo_quantum_util_65_to_78_triggers(self) -> None:
        """演示脚本'量子利用率从 65% → 78%' 应触发（暗示提升，与 -3.3% 矛盾）。"""
        text = "量子利用率从 65% → 78%\n"
        violations = _run_blacklist_check_on_text(text)
        assert any("65%" in msg or "~46%" in msg for _pat, msg in violations), (
            "演示量子利用率65%→78%误导性绝对值应被BLACKLIST捕获"
        )

    def test_demo_quantum_util_65_plain_triggers(self) -> None:
        """'量子利用率 65%' 单独出现也应触发。"""
        text = "资源状态（量子利用率 65%）\n"
        violations = _run_blacklist_check_on_text(text)
        assert any("65%" in msg or "~46%" in msg for _pat, msg in violations), (
            "量子利用率65%孤立绝对值应被BLACKLIST捕获"
        )


class TestUtilization72Negative:
    """8.7-v3 红队审查新增：72%/65% 不应误伤覆盖率等其他百分比"""

    def test_coverage_72_not_triggered(self) -> None:
        """覆盖率 72%（非利用率）不应触发利用率 72% 规则。"""
        text = "代码覆盖率 72%\n"
        violations = _run_blacklist_check_on_text(text)
        assert not any("-3.3%" in msg for _pat, msg in violations), (
            "覆盖率72%不应被误判为利用率口径"
        )

    def test_cpu_util_65_without_quantum_not_triggered(self) -> None:
        """经典/CPU 利用率 65%（无'量子利用率'前缀）不应触发演示 65% 规则。"""
        text = "CPU 利用率 65%\n"
        violations = _run_blacklist_check_on_text(text)
        assert not any("~46%" in msg for _pat, msg in violations), (
            "经典利用率65%不应被误判为量子利用率"
        )
