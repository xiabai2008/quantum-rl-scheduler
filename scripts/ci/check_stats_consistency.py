#!/usr/bin/env python
"""
统计口径一致性检查脚本 (Statistical Consistency Checker)

Issue #141: 消除4套p值混用，建立单一权威统计源

扫描所有 .md 文件及演示链路 .py 文件中的统计数字（p值、效应量、N值），
与 config/statistics.yaml 权威源比对，报告不一致告警。

用法:
    python scripts/ci/check_stats_consistency.py
    python scripts/ci/check_stats_consistency.py --strict  # CI模式：有不一致则退出码1
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_YAML = _PROJECT_ROOT / "config" / "statistics.yaml"
AUTHORITATIVE_NUMBERS_MD = _PROJECT_ROOT / "docs" / "authoritative_numbers.md"

# 修复 Windows GBK 终端下 emoji 字符导致的 UnicodeEncodeError 崩溃
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 需要扫描的文件模式
SCAN_GLOBS = ["*.md", "**/*.md"]

# 演示链路 .py 文件（Web面板模板/一键演示/PPT生成脚本的展示字符串
# 曾多次出现废弃数字，同样纳入口径审计）
DEMO_CHAIN_FILES = [
    "src/visualization/fallback_template.py",
    "scripts/demo_one_click.py",
    "scripts/generate_defense_ppt.py",
    "scripts/demo/demo.py",
    "scripts/demo/demo_multi_machine.py",
]

# 8.5 审查：权威数值内部冲突检查（A8）
# 同一文件内禁止并存两组互斥的"权威"数值——评委可当场发现。
# 豁免行必须含诚实披露标记（废弃/错误数据/中间产物/审查修正等）。
INTERNAL_CONFLICTS: list[tuple[list[str], str]] = [
    (["+20.2%", "+123.4%"], "综合收益 +20.2%（主口径）与 +123.4%（历史 vs Hybrid-Default）并存"),
    (["-3.3%", "+7.9%"], "利用率 -3.3%（N=250 真实）与 +7.9%（错误旧数据）并存"),
    (["25.46", "57.27"], "等待时间 25.46（N=250）与 57.27（N=4 中间品）并存"),
    (["+38.5%", "+33.3%"], "编译层深电路 +38.5%（N=80 显著）与 +33.3%（N=20 旧样本）并存"),
]
_DISCLOSURE_MARK = ("废弃", "错误数据", "中间产物", "审查修正", "8.5 审查", "已废弃", "无法支撑", "非统计证据", "历史")

# 排除的文件/目录（权威报告本身，不检查自身）
EXCLUDE_PATHS = {
    "config/statistics.yaml",
    "docs/authoritative_numbers.md",  # 权威数字事实源本身
    "results/reports/statistical_validation.md",  # 权威数据源本身
    "results/reports/multiseed_real_machine_report.md",
    "results/reports/multiseed_real_machine_report_10seeds.md",
    "results/reports/multiseed_real_machine_report_10seeds_v2.md",
    "results/reports/head_only_validation.md",
    "results/reports/real_machine_statistical_significance.md",
    "results/reports/dqn_ppo_fcfs_comparison.md",
    "results/reports/annealing_ablation_20seeds_report.md",
    "results/reports/real_machine_boundary_statement.md",
    "results/reports/roi_analysis.md",
    "results/reports/utilization_multiseed_report.md",
    # 历史实验报告（14维旧模型口径，已加废弃横幅冻结，禁止直接引用）
    "results/reports/ablation_report.md",
    "results/reports/issue_457_105_qubits_validation_report.md",
    "results/reports/power_analysis.md",
    "results/reports/quantum_ratio_sensitivity.md",
    "results/reports/real_machine_closed_loop.md",
    "results/reports/real_machine_validation.md",
    "results/reports/tradeoff_analysis.md",
}

# 排除的文件名模式（内部协作文档，不提交到仓库，不需检查）
EXCLUDE_PATTERNS = [
    "teammate_",  # docs/teammate_*.md
    "PR审查",  # PR审查临时报告
    "_pr",  # PR审查临时报告前缀
    "pr_patrol_",  # docs/pr_patrol_*.md
    "project_dashboard_",  # docs/project_dashboard_*.html
]

# p值正则模式：匹配 p=0.031, p=1.032e-42, p<0.001, p=3.04×10⁻¹¹ 等
P_VALUE_PATTERN = re.compile(
    r"p\s*[=<>]\s*"
    r"(?:"
    r"(?:\d+\.?\d*(?:[eE][+-]?\d+)?)"  # 标准科学计数法: 1.032e-42
    r"|"
    r"(?:\d+\.?\d*\s*[×x]\s*10[⁻⁺\-+]?[⁰¹²³⁴⁵⁶⁷⁸⁹0-9]+)"  # Unicode: 1.032×10⁻⁴²
    r"|"
    r"(?:<\s*10[⁻⁺\-+]?[⁰¹²³⁴⁵⁶⁷⁸⁹0-9]+)"  # p<10⁻⁴²
    r")",
    re.IGNORECASE,
)

# 已知的权威p值集合（从YAML加载后动态构建）
AUTHORITATIVE_P_VALUES: dict[str, str] = {}

# 已知废弃/错误p值 → 应替换为的权威值
DEPRECATED_P_VALUES: dict[str, str] = {
    "4.92e-55": "MISATTRIBUTED: p=4.92e-55 是 Random vs PPO 的p值，不是 PPO vs FCFS。应使用 p=1.449e-66 (Welch t)",
}

# 检验方法混用检查
WELCH_T_FOR_1032E42 = "ERROR: p=1.449e-66 对应 Welch t 检验，不是 Mann-Whitney U 检验"

# Issue #446: 严禁表述黑名单（来自 docs/authoritative_numbers.md 第六节）
# 注意：黑名单匹配后，若行内包含以下"诚实披露"关键词则豁免（避免对已修正的诚实标注误报）
HONEST_DISCLOSURE_KEYWORDS = [
    "单seed",
    "探索性",
    "诚实披露",
    "10seeds",
    "10 seeds",
    "20seeds",
    "20 seeds",
    "5seeds",
    "5 seeds",
    "旧实验",
    "注：",
    "注:",
    "边界",
    "cherry-pick",
    "已降级",
    "已废弃",
]


def _is_honest_disclosure(line: str) -> bool:
    """判断行内是否包含诚实披露限定词（豁免黑名单检测）。"""
    lower = line.lower()
    return any(kw in line or kw.lower() in lower for kw in HONEST_DISCLOSURE_KEYWORDS)


BLACKLIST_PATTERNS: list[tuple[str, str]] = [
    (
        r"284\s*次(SDK)?(真机)?调用",
        "BLACKLIST: 真机调用次数已统一为315次（284+31审计口径），淘汰284旧表述",
    ),
    (
        r"284\s*次真机",
        "BLACKLIST: 真机调用次数已统一为315次，淘汰284旧表述",
    ),
    (
        r"v5已完成|答辩PPT.*v5|技术白皮书.*v5.*11章",
        "BLACKLIST: v5白皮书/PPT从未存在，实际为docs/technical_whitepaper.pdf（7章），PPT制作中",
    ),
    (
        r"利用率(提升)?\s*[≥>=]\s*30%.*已?达成|资源利用率.*≥30%.*(?<!未)达标",
        "BLACKLIST: N=250权威数据利用率仅+7.9%（未达30%目标），需改为部分达成口径",
    ),
    (
        r"等待时间.*-5\.7%",
        "BLACKLIST: -5.7%为单seed乐观结果，10seeds严谨实验显示等待时间+6.1%，需诚实呈现",
    ),
    (
        r"退火.*p\s*=\s*0\.190(?!.*(20seeds|不显著.*0\.94|已降级.*探索))",
        "BLACKLIST: 退火p=0.190对应5seeds旧实验，20seeds新实验p=0.9430（不显著），以新数据为准",
    ),
    (
        r"PPO.*真机.*1665\.22|真机.*PPO.*1665\.22|1665\.22.*PPO.*真机",
        "BLACKLIST: 真机PPO均值已升级为N=10 v2权威值1736.32，旧N=5值1665.22严禁作为对外基准（若为历史引用请显式标注N=5并说明已废弃）",
    ),
    (
        r"FCFS.*真机.*353\.22|真机.*FCFS.*353\.22|353\.22.*FCFS.*真机",
        "BLACKLIST: 真机FCFS均值已升级为N=10 v2权威值383.00，旧N=5值353.22严禁作为对外基准（若为历史引用请显式标注N=5并说明已废弃）",
    ),
]


def load_statistics_yaml() -> dict[str, Any]:
    """加载权威统计源 YAML 文件。"""
    with open(STATS_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_authoritative_coverage(stats: dict[str, Any]) -> list[str]:
    """检查权威源是否遗漏已发布的关键实验（Issue #536）。

    确保已发布报告中的关键实验都被 statistics.yaml 收录，
    避免单一权威源机制自身失守。
    """
    warnings = []
    # 已发布的关键实验及其应有的 p 值记录
    required_experiments = [
        (
            "annealing_ablation_head_only",
            "with_vs_without_annealing",
            0.9430,
            "退火消融 20seeds 权威验证（p=0.9430）",
        ),
        (
            "real_machine_10seed_v2",
            "ppo_vs_fcfs",
            0.001,
            "真机 10seeds v2 实验（p<0.001）",
        ),
        (
            "simulation_8strategy_50seed",
            "ppo_vs_fcfs",
            1.032e-42,
            "8 策略 50seed 仿真（p=1.032e-42）",
        ),
    ]
    for exp_key, comp_key, expected_p, desc in required_experiments:
        exp = stats.get(exp_key)
        if not exp:
            warnings.append(f"权威源遗漏实验: {exp_key}（{desc}）未在 statistics.yaml 中收录")
            continue
        comp = exp.get(comp_key, {})
        actual_p = comp.get("p_value")
        if actual_p is None:
            warnings.append(f"权威源遗漏 p 值: {exp_key}.{comp_key}（{desc}）未记录 p_value")
            continue
        # 数值比较（允许格式差异）
        try:
            if abs(float(actual_p) - float(expected_p)) > 1e-6:
                warnings.append(
                    f"权威源 p 值不一致: {exp_key}.{comp_key} 期望 p={expected_p}, 实际 p={actual_p}"
                )
        except (TypeError, ValueError):
            warnings.append(f"权威源 p 值格式异常: {exp_key}.{comp_key} p_value={actual_p}")
    return warnings


def build_authoritative_p_values(stats: dict[str, Any]) -> dict[str, str]:
    """从YAML构建权威p值映射表。

    Returns:
        {p_value_string: "实验名.对比名 (检验方法, N=xxx)"}
    """
    p_map = {}

    # 实验1: 8策略50seed仿真
    sim = stats.get("simulation_8strategy_50seed", {})
    for comp_name in [
        "ppo_vs_fcfs",
        "ppo_vs_dqn",
        "ppo_vs_sjf",
        "ppo_vs_random",
        "dqn_vs_fcfs",
        "fcfs_vs_sjf",
    ]:
        comp = sim.get(comp_name, {})
        if "p_value" in comp:
            p_val = format_p_value(comp["p_value"])
            ctx = f"8策略50seed仿真.{comp_name} ({comp.get('test_method', '?')}, N={comp.get('n', '?')})"
            p_map[p_val] = ctx

    # 实验2: 多seed真机
    real = stats.get("real_machine_5seed", {})
    for comp_name in ["ppo_vs_fcfs", "ppo_vs_sjf", "sjf_vs_fcfs"]:
        comp = real.get(comp_name, {})
        if "p_value" in comp:
            p_val = format_p_value(comp["p_value"])
            ctx = f"多seed真机.{comp_name} ({comp.get('test_method', '?')}, N={comp.get('n', '?')})"
            p_map[p_val] = ctx

    # 实验3: 退火消融
    annealing = stats.get("annealing_ablation_head_only", {})
    comp = annealing.get("with_vs_without_annealing", {})
    if "p_value" in comp:
        p_val = format_p_value(comp["p_value"])
        ctx = (
            f"退火消融 ({comp.get('test_method', '?')}, n={comp.get('n', annealing.get('n', '?'))})"
        )
        p_map[p_val] = ctx

    # 实验4: 真机闭环
    closed_loop = stats.get("real_machine_closed_loop", {})
    comp = closed_loop.get("simulation_vs_mixed_real", {})
    if "p_value" in comp:
        p_val = format_p_value(comp["p_value"])
        ctx = f"真机闭环 ({comp.get('test_method', '?')}, N={comp.get('n', closed_loop.get('n', '?'))})"
        p_map[p_val] = ctx

    # 实验5: 3策略对比
    sim3 = stats.get("simulation_3strategy_10seed", {})
    for comp_name in ["ppo_vs_fcfs", "ppo_vs_dqn"]:
        comp = sim3.get(comp_name, {})
        if "p_value" in comp:
            p_val = format_p_value(comp["p_value"])
            ctx = (
                f"3策略10seed.{comp_name} ({comp.get('test_method', '?')}, N={comp.get('n', '?')})"
            )
            p_map[p_val] = ctx

    return p_map


def build_deprecated_p_values(stats: dict[str, Any]) -> dict[str, str]:
    """从YAML构建废弃p值映射表。"""
    dep_map = {}
    for _key, info in stats.get("deprecated", {}).items():
        p_val = format_p_value(info.get("value", 0))
        status = info.get("status", "deprecated")
        reason = info.get("reason", "").strip().split("\n")[0]
        replacement = info.get("replacement", "")
        dep_map[p_val] = f"[{status.upper()}] {reason} → 替换为: {replacement}"
    return dep_map


def format_p_value(value: Any) -> str:
    """将p值格式化为标准字符串。"""
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, (int, float)):
        if value == 0:
            return "0"
        # 保留科学计数法格式
        return f"{value:.3e}".lower().replace("e+0", "e+").replace("e-0", "e-")
    return str(value).lower()


def normalize_p_value_text(text: str) -> str:
    """将文本中的p值标准化以便比较。"""
    text = text.lower().strip()
    # 移除空格
    text = text.replace(" ", "")
    # Unicode上标转ASCII
    superscript_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
    text = text.translate(superscript_map)
    # ×10 转换为 e
    text = re.sub(r"[×x]10\^?", "e", text)
    # 10^-42 → e-42 (for p<10⁻⁴² style)
    text = re.sub(r"10\^?(-?\d+)", lambda m: f"e{m.group(1)}" if "." in text else text, text)
    return text


def extract_p_values_from_line(line: str) -> list[tuple[str, str]]:
    """从一行文本中提取所有p值引用。

    Returns:
        [(raw_match, normalized_p_value), ...]
    """
    results = []
    for match in P_VALUE_PATTERN.finditer(line):
        raw = match.group(0)
        # 提取数值部分
        val_match = re.search(r"(?:\d+\.?\d*(?:[eE][+-]?\d+)?)", raw)
        if val_match:
            val_str = val_match.group(0).lower()
            results.append((raw, val_str))
        else:
            # 尝试匹配 Unicode 科学计数法
            unicode_match = re.search(r"(\d+\.?\d*)\s*[×x]\s*10[⁻⁺\-+]([⁰¹²³⁴⁵⁶⁷⁸⁹0-9]+)", raw)
            if unicode_match:
                base = float(unicode_match.group(1))
                val_str = f"{base:.3e}".lower()
                results.append((raw, val_str))

    return results


def _is_deprecation_notice(line: str) -> bool:
    """判断行内是否为废弃声明上下文（在废弃声明中提及旧p值属合规引用）。"""
    lower = line.lower()
    return any(kw in line or kw in lower for kw in ("已废弃", "已淘汰", "deprecated"))


def check_welch_t_misattribution(line: str) -> str | None:
    """检查 Welch t + p=1.032e-42 的错误搭配。"""
    line_lower = line.lower()
    has_1032e42 = (
        "1.032e-42" in line_lower or "1.032×10⁻⁴²" in line_lower or "1.03×10⁻⁴²" in line_lower
    )
    # 行中同时包含 Mann-Whitney 或正确新p值 1.449e-66 或废弃声明时，
    # 说明是对比/废弃说明文本，不报错
    has_context = (
        "mann-whitney" in line_lower or "1.449e-66" in line_lower or _is_deprecation_notice(line)
    )
    if has_1032e42 and "welch" in line_lower and not has_context:
        return WELCH_T_FOR_1032E42
    return None


def scan_markdown_file(
    filepath: Path,
    authoritative_p: dict[str, str],
    deprecated_p: dict[str, str],
) -> list[str]:
    """扫描单个Markdown文件，返回告警列表。"""
    warnings = []

    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return warnings

    text = "".join(lines)

    for line_num, line in enumerate(lines, 1):
        # 检查 Welch t 错误搭配
        welch_err = check_welch_t_misattribution(line)
        if welch_err:
            warnings.append(f"  L{line_num}: {welch_err}")
            warnings.append(f"    > {line.strip()[:120]}")

        # 检查黑名单表述（Issue #446）—— 诚实披露上下文豁免
        if not _is_honest_disclosure(line):
            for pattern, message in BLACKLIST_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    warnings.append(f"  L{line_num}: {message}")
                    warnings.append(f"    > {line.strip()[:120]}")

        # 提取p值
        p_values = extract_p_values_from_line(line)
        for _raw, normalized in p_values:
            # 检查是否为废弃p值（废弃声明上下文中提及旧p值属合规引用，豁免）
            if _is_deprecation_notice(line):
                break
            for dep_val, dep_msg in deprecated_p.items():
                if normalized == dep_val or normalized.startswith(dep_val[:8]):
                    # 检查上下文：如果是在正确的实验上下文中使用则跳过
                    # 例如 p=4.92e-55 在 "Random vs PPO" 上下文中是正确的
                    line_lower = line.lower()
                    is_4_92e55 = "4.92e-55" in normalized or "4.92e-55" in dep_val
                    if is_4_92e55 and "random" in line_lower and "ppo" in line_lower:
                        continue  # 在Random vs PPO上下文中使用是正确的
                    warnings.append(f"  L{line_num}: 废弃/错误p值 p={normalized}")
                    warnings.append(f"    {dep_msg}")
                    warnings.append(f"    > {line.strip()[:120]}")
                    break

    # 8.5 审查：权威数值内部冲突（A8）
    for vals, conflict_msg in INTERNAL_CONFLICTS:
        if not vals[0] in text and not vals[1] in text:
            continue
        hits = [v for v in vals if v in text]
        if len(hits) >= 2 and not any(any(mark in line for mark in _DISCLOSURE_MARK) for line in lines):
            warnings.append(f"  文件级冲突: {conflict_msg}")
            for v in hits:
                warn_lines = [i for i, l in enumerate(lines, 1) if v in l]
                warnings.append(f"    -> '{v}' 出现于 L{warn_lines[:3]}")

    return warnings


def main() -> int:
    """主入口：扫描所有Markdown文件并报告统计口径不一致。"""
    import argparse

    parser = argparse.ArgumentParser(description="统计口径一致性检查")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：发现不一致时退出码1（用于CI）",
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default=str(STATS_YAML),
        help="权威统计源YAML文件路径",
    )
    args = parser.parse_args()

    stats_path = Path(args.stats_file)
    if not stats_path.exists():
        print(f"ERROR: 权威统计源文件不存在: {stats_path}")
        return 1

    # 加载权威源
    stats = load_statistics_yaml()
    authoritative_p = build_authoritative_p_values(stats)
    deprecated_p = build_deprecated_p_values(stats)

    print("=" * 70)
    print("  统计口径一致性检查 (Issue #141)")
    print("=" * 70)
    print(f"  权威源: {stats_path}")
    print(f"  权威p值数: {len(authoritative_p)}")
    print(f"  废弃p值数: {len(deprecated_p)}")
    print("=" * 70)

    # Issue #536: 权威源覆盖度检查（确保已发布实验不遗漏）
    coverage_warnings = check_authoritative_coverage(stats)
    if coverage_warnings:
        print("\n[Issue #536] 权威源覆盖度检查:")
        for w in coverage_warnings:
            print(f"  [ERROR] {w}")
        print()
    else:
        print("[Issue #536] 权威源覆盖度检查: ✅ 关键实验均已收录")
        print()

    # 收集所有Markdown文件（使用 os.walk 跳过 node_modules 等目录）
    md_files = set()
    # 演示链路 .py 文件纳入扫描（与 .md 共用同一套逐行检查逻辑）
    for rel in DEMO_CHAIN_FILES:
        demo_path = _PROJECT_ROOT / rel
        if demo_path.exists():
            md_files.add(demo_path)
    # Issue #174: 排除非交付目录（AI 工作目录、数据转储、临时文件等），
    # 这些目录可能包含历史数字/草稿，不应参与口径审计，避免假失败
    exclude_dirs = {
        "node_modules",
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "mutants",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
        # 非交付目录：AI 工作目录、数据转储、临时文件等
        ".workbuddy",
        ".trae",
        ".trae-cn",
        "project-review",
        ".trae-html-share-packages",
        "tmp",
        "temp",
        "data",
        "datasets",
        "build",
        "dist",
        "downloads",
    }
    for root, dirs, files in os.walk(_PROJECT_ROOT, topdown=True):
        # 原地修改 dirs 跳过排除目录，阻止 os.walk 递归进入
        # 同时跳过所有 .venv* 开头的目录（各种虚拟环境）和 site-packages
        dirs[:] = [
            d
            for d in dirs
            if d not in exclude_dirs
            and not d.startswith(".venv")
            and d not in ("site-packages", "lib", "lib64")
        ]
        for fname in files:
            if not fname.endswith(".md"):
                continue
            fpath = Path(root) / fname
            rel = str(fpath.relative_to(_PROJECT_ROOT)).replace("\\", "/")
            if rel not in EXCLUDE_PATHS and not any(pat in rel for pat in EXCLUDE_PATTERNS):
                md_files.add(fpath)

    md_files = sorted(md_files, key=lambda f: str(f.relative_to(_PROJECT_ROOT)))

    print(f"  扫描文件数: {len(md_files)}")
    print()

    total_warnings = 0
    files_with_warnings = 0

    for filepath in md_files:
        rel_path = str(filepath.relative_to(_PROJECT_ROOT)).replace("\\", "/")
        warnings = scan_markdown_file(filepath, authoritative_p, deprecated_p)
        if warnings:
            files_with_warnings += 1
            total_warnings += len(warnings) // 3  # 每个告警约3行
            print(f"[WARN] {rel_path}")
            for w in warnings:
                print(w)
            print()

    # 汇总
    print("=" * 70)
    if total_warnings == 0:
        print("  ✅ 统计口径一致性检查通过，无不一致告警。")
    else:
        print(f"  ⚠️  发现 {total_warnings} 处不一致告警（{files_with_warnings} 个文件）。")
    print("=" * 70)

    # 打印权威p值参考表
    print("\n权威p值参考表:")
    print(f"  {'p值':<20} {'实验上下文'}")
    print("  " + "-" * 66)
    for p_val, ctx in sorted(authoritative_p.items()):
        print(f"  {p_val:<20} {ctx}")

    if args.strict and total_warnings > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
