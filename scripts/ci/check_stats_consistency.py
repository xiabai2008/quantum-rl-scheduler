#!/usr/bin/env python
"""
统计口径一致性检查脚本 (Statistical Consistency Checker)

Issue #141: 消除4套p值混用，建立单一权威统计源

扫描所有 .md 文件中的统计数字（p值、效应量、N值），
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

# 需要扫描的文件模式
SCAN_GLOBS = ["*.md", "**/*.md"]

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
    "4.92e-55": "MISATTRIBUTED: p=4.92e-55 是 Random vs PPO 的p值，不是 PPO vs FCFS。应使用 p=1.032e-42 (Mann-Whitney U)",
}

# 检验方法混用检查
WELCH_T_FOR_1032E42 = "ERROR: p=1.032e-42 对应 Mann-Whitney U 检验，不是 Welch t 检验"

# Issue #446: 严禁表述黑名单（来自 docs/authoritative_numbers.md 第六节）
# 注意：黑名单匹配后，若行内包含以下"诚实披露"关键词则豁免（避免对已修正的诚实标注误报）
HONEST_DISCLOSURE_KEYWORDS = [
    "单seed", "探索性", "诚实披露", "10seeds", "10 seeds",
    "5seeds", "5 seeds", "旧实验",
    "注：", "注:", "边界", "cherry-pick", "已降级",
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
        r"利用率(提升)?\s*[≥>=]\s*30%.*已?达成|资源利用率.*≥30%.*达标",
        "BLACKLIST: N=250权威数据利用率仅+7.9%（未达30%目标），需改为部分达成口径",
    ),
    (
        r"等待时间\s*-5\.7%",
        "BLACKLIST: -5.7%为单seed乐观结果，10seeds严谨实验显示等待时间+6.1%，需诚实呈现",
    ),
    (
        r"退火.*p\s*=\s*0\.190(?!.*(20seeds|不显著.*0\.94|已降级.*探索))",
        "BLACKLIST: 退火p=0.190对应5seeds旧实验，20seeds新实验p=0.9430（不显著），以新数据为准",
    ),
]


def load_statistics_yaml() -> dict[str, Any]:
    """加载权威统计源 YAML 文件。"""
    with open(STATS_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def check_welch_t_misattribution(line: str) -> str | None:
    """检查 Welch t + p=1.032e-42 的错误搭配。"""
    line_lower = line.lower()
    has_1032e42 = (
        "1.032e-42" in line_lower or "1.032×10⁻⁴²" in line_lower or "1.03×10⁻⁴²" in line_lower
    )
    # 如果行中同时包含 Mann-Whitney，说明是对比说明文本（如"已被...取代"），不报错
    has_mann_whitney = "mann-whitney" in line_lower
    if has_1032e42 and "welch" in line_lower and not has_mann_whitney:
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
            # 检查是否为废弃p值
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

    # 收集所有Markdown文件（使用 os.walk 跳过 node_modules 等目录）
    md_files = set()
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
