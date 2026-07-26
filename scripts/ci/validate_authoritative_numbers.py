#!/usr/bin/env python
"""权威数字一致性校验脚本 (Authoritative Numbers Validator).

Issue #158: 修复全项目权威数字不一致问题

扫描所有 .md 文件中的统计数字，与权威值比对，
输出不一致报告，返回非零退出码如果有不一致。

权威数字（50seed 仿真, N=250）:
    - PPO: 2746.94 ± 1160.72
    - FCFS: 1458.77 ± 60.47
    - p = 1.032e-42 (Mann-Whitney U 检验, Bonferroni 校正后)
    - rank-biserial = -0.71 (大效应量)

不应出现的旧数字:
    - ±1121.19 (旧 PPO 标准差, 应为 ±1160.72)
    - ±55.85 (旧 FCFS 标准差, 应为 ±60.47)
    - p=3.04e-11 / p=3.04×10⁻¹¹ (旧 p 值, 应为 p=1.032e-42)
    - Cohen's d=-1.70 (旧效应量, 应为 rank-biserial=-0.71)
    - "Welch t" + p=1.032e-42 (错误搭配, 应为 Mann-Whitney U)

用法:
    python scripts/ci/validate_authoritative_numbers.py
    python scripts/ci/validate_authoritative_numbers.py --strict  # CI 模式
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 权威数字（50seed 仿真, N=250）
AUTHORITATIVE = {
    "ppo_mean": "2746.94",
    "ppo_std": "1160.72",
    "fcfs_mean": "1458.77",
    "fcfs_std": "60.47",
    "p_value": "1.032e-42",
    "effect_size": "-0.71",
    "effect_size_type": "rank-biserial",
    "test_method": "Mann-Whitney U",
    "n": "250",
    "improvement": "+88.3%",
}

# 旧数字模式 → 应替换为的权威值
# (编译后的正则, 旧数字描述, 应替换为的权威值)
DEPRECATED_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"1121\.19"),
        "旧 PPO 标准差 ±1121.19",
        "±1160.72",
    ),
    (
        re.compile(r"55\.85"),
        "旧 FCFS 标准差 ±55.85",
        "±60.47",
    ),
    (
        re.compile(r"3\.04[eE][-]?11|3\.04\s*[×x]\s*10[⁻-]1[¹1]"),
        "旧 p 值 p=3.04e-11",
        "p=1.032e-42",
    ),
]

# Cohen's d=-1.70 旧效应量（需要更精确的匹配以避免误报 1.70% 等数字）
DEPRECATED_EFFECT_SIZE = re.compile(r"[dD]\s*=\s*-?1\.70\b|Cohen's\s+[dD]\s*[=＝]\s*-?1\.70\b")

# Welch t + p=1.032e-42 错误搭配
WELCH_WITH_1032E42 = re.compile(
    r"Welch\s*t.*1\.032[eE][-]?42|1\.032[eE][-]?42.*Welch\s*t",
    re.IGNORECASE,
)

# 跳过的目录
SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "venv",
    ".venv",
    "node_modules",
    "mutants",
    # 实验数据目录：历史快照，不参与口径审计
    "fair_comparison",
    "issue_experiments",
    "multiseed_evaluation",
    "real_machine",
    "gradient_stress",
    "ablation_d3_training",
    "models",
    # Issue #174: 非交付目录 — AI 工作目录、数据转储、临时文件等
    # 这些目录可能包含历史数字/草稿/数据转储，不应参与口径审计，避免假失败
    ".workbuddy",  # AI agent 工作目录
    ".trae",  # TRAE 工作目录
    ".trae-cn",  # TRAE 工作目录（中文版）
    ".hypothesis",  # property-based testing 数据库（生成的样例）
    "project-review",  # 项目评审临时文件
    ".trae-html-share-packages",  # HTML 分享包临时目录
    "tmp",  # 临时文件
    "temp",  # 临时文件
    "data",  # 数据转储
    "datasets",  # 数据集
    "build",  # 构建产物
    "dist",  # 发布产物
    "downloads",  # 下载目录
}

# 排除的文件（权威数据源本身，不检查自身）
EXCLUDE_PATHS = {
    "config/statistics.yaml",
    "results/reports/statistical_validation.md",
    "results/reports/multiseed_real_machine_report.md",
    "results/reports/multiseed_real_machine_report_10seeds.md",
    "results/reports/head_only_validation.md",
    "results/reports/real_machine_statistical_significance.md",
    "results/reports/dqn_ppo_fcfs_comparison.md",
    "scripts/ci/validate_authoritative_numbers.py",
}

# 排除的文件名模式（历史巡逻记录，只更新引用数字但不改变叙述）
EXCLUDE_PATTERNS = [
    "pr_patrol_",
]

# 行级豁免标记：包含此标记的行视为明确标注的历史/废弃数据，跳过检查
EXEMPT_MARKERS = (
    "deprecated",
    "历史",
    "取代",
    "早期",
    "旧版",
    "已废弃",
    "audit-exempt",
    "已被",
    "曾错误",
    "曾使用",
    "来源不明",
)


@dataclass
class Violation:
    """单条违规记录。"""

    file_path: str
    line_number: int
    line_content: str
    issue: str
    fix: str


@dataclass
class ScanResult:
    """扫描结果汇总。"""

    violations: list[Violation] = field(default_factory=list)
    files_scanned: int = 0
    files_with_violations: int = 0

    @property
    def passed(self) -> bool:
        """是否通过校验。"""
        return len(self.violations) == 0


def is_exempt_line(line: str) -> bool:
    """判断一行是否为豁免行（废弃说明、历史标注等）。

    包含 EXEMPT_MARKERS 中任一关键词的行视为明确标注的历史数据行，
    跳过该行的旧数字检查。豁免仅对当前行生效。
    """
    line_lower = line.lower()
    return any(marker.lower() in line_lower for marker in EXEMPT_MARKERS)


def is_patrol_finding(line: str) -> bool:
    """判断一行是否为巡逻记录中的发现描述。

    巡逻记录中引用旧数字是为了描述"发现了什么问题"，
    例如 "P0: 标准差数字不一致（±1121.19 vs ±1160.72）"，
    这类引用不需要修改。
    """
    line_lower = line.lower()
    patrol_indicators = (
        "不一致",
        "错误",
        "引用",
        "提醒",
        "修正",
        "应为",
        "权威口径",
        "但agents",
        "但 agents",
        "vs",
    )
    return any(indicator in line_lower for indicator in patrol_indicators)


def check_line(line: str, filepath: str, line_num: int) -> list[Violation]:
    """检查单行文本中的旧数字。

    返回该行中发现的所有违规。
    """
    violations: list[Violation] = []

    # 豁免行：废弃说明、历史标注
    if is_exempt_line(line):
        return violations

    # 巡逻记录中的发现描述：引用旧数字是为了说明问题
    is_patrol = "pr_patrol" in filepath.lower() or is_patrol_finding(line)

    # 检查旧 PPO/FCFS 标准差和旧 p 值
    for pattern, issue_desc, fix_desc in DEPRECATED_PATTERNS:
        if pattern.search(line):
            # 在巡逻记录中，旧数字与权威数字同时出现用于对比，不报错
            if is_patrol and any(
                auth_val in line for auth_val in ("1160.72", "60.47", "1.032e-42")
            ):
                continue
            violations.append(
                Violation(
                    file_path=filepath,
                    line_number=line_num,
                    line_content=line.strip()[:120],
                    issue=issue_desc,
                    fix=f"应替换为 {fix_desc}",
                )
            )

    # 检查 Cohen's d=-1.70 旧效应量
    # 巡逻记录中旧数字与权威数字同时出现用于对比，不报错
    if DEPRECATED_EFFECT_SIZE.search(line) and not (is_patrol and "1.032e-42" in line):
        violations.append(
            Violation(
                file_path=filepath,
                line_number=line_num,
                line_content=line.strip()[:120],
                issue="旧效应量 Cohen's d=-1.70",
                fix="应替换为 rank-biserial=-0.71",
            )
        )

    # 检查 Welch t + p=1.032e-42 错误搭配
    # 如果行中同时包含 Mann-Whitney，说明是对比说明文本，不报错
    if WELCH_WITH_1032E42.search(line) and "mann-whitney" not in line.lower():
        violations.append(
            Violation(
                file_path=filepath,
                line_number=line_num,
                line_content=line.strip()[:120],
                issue="Welch t 与 p=1.032e-42 错误搭配",
                fix="p=1.032e-42 对应 Mann-Whitney U 检验",
            )
        )

    return violations


def scan_markdown_file(filepath: Path) -> list[Violation]:
    """扫描单个 Markdown 文件，返回违规列表。"""
    violations: list[Violation] = []
    rel_path = str(filepath.relative_to(_PROJECT_ROOT)).replace("\\", "/")

    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return violations

    for line_num, line in enumerate(lines, 1):
        violations.extend(check_line(line, rel_path, line_num))

    return violations


def iter_markdown_files() -> list[Path]:
    """遍历所有需要检查的 Markdown 文件。

    使用 os.walk 并在遍历时跳过 SKIP_DIRS，
    同时排除 EXCLUDE_PATHS 和 EXCLUDE_PATTERNS。
    """
    md_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(_PROJECT_ROOT, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".venv")]
        for filename in filenames:
            if not filename.endswith(".md"):
                continue
            fpath = Path(dirpath) / filename
            rel = str(fpath.relative_to(_PROJECT_ROOT)).replace("\\", "/")
            if rel in EXCLUDE_PATHS:
                continue
            if any(pat in rel for pat in EXCLUDE_PATTERNS):
                continue
            md_files.append(fpath)

    return sorted(md_files, key=lambda f: str(f.relative_to(_PROJECT_ROOT)))


def print_authoritative_reference() -> None:
    """打印权威数字参考表。"""
    print("\n权威数字参考表（50seed 仿真, N=250）:")
    print(f"  {'指标':<25} {'权威值':<25} {'说明'}")
    print("  " + "-" * 68)
    print(f"  {'PPO 平均奖励':<25} {AUTHORITATIVE['ppo_mean']:<25} 14维原生环境")
    print(f"  {'PPO 标准差':<25} ±{AUTHORITATIVE['ppo_std']:<24} 50 seeds × 5 episodes")
    print(f"  {'FCFS 平均奖励':<25} {AUTHORITATIVE['fcfs_mean']:<25} 基线策略")
    print(f"  {'FCFS 标准差':<25} ±{AUTHORITATIVE['fcfs_std']:<24} 同上")
    print(f"  {'p 值':<25} {AUTHORITATIVE['p_value']:<25} Mann-Whitney U 检验")
    print(f"  {'效应量':<25} {AUTHORITATIVE['effect_size']:<25} rank-biserial（大效应）")
    print(f"  {'提升百分比':<25} {AUTHORITATIVE['improvement']:<25} PPO vs FCFS")
    print()
    print("不应出现的旧数字:")
    print("  ±1121.19 → ±1160.72 (PPO 标准差)")
    print("  ±55.85   → ±60.47   (FCFS 标准差)")
    print("  p=3.04e-11 → p=1.032e-42 (p 值)")
    print("  Cohen's d=-1.70 → rank-biserial=-0.71 (效应量)")
    print("  Welch t + p=1.032e-42 → Mann-Whitney U (检验方法)")


def main() -> int:
    """主入口：扫描所有 Markdown 文件并报告权威数字不一致。"""
    parser = argparse.ArgumentParser(
        description="权威数字一致性校验 (Issue #158)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：发现不一致时退出码 1（用于 CI）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细模式：打印所有扫描文件",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  权威数字一致性校验 (Issue #158)")
    print("=" * 70)

    md_files = iter_markdown_files()
    print(f"  扫描文件数: {len(md_files)}")
    print("=" * 70)
    print()

    result = ScanResult()

    for filepath in md_files:
        rel_path = str(filepath.relative_to(_PROJECT_ROOT)).replace("\\", "/")
        result.files_scanned += 1

        if args.verbose:
            print(f"  扫描: {rel_path}")

        violations = scan_markdown_file(filepath)
        if violations:
            result.files_with_violations += 1
            result.violations.extend(violations)
            print(f"[FAIL] {rel_path}")
            for v in violations:
                print(f"  L{v.line_number}: {v.issue}")
                print(f"    {v.fix}")
                print(f"    > {v.line_content}")
            print()

    # 汇总
    print("=" * 70)
    if result.passed:
        print("  PASS: 权威数字一致性校验通过，无不一致。")
    else:
        print(
            f"  FAIL: 发现 {len(result.violations)} 处不一致"
            f"（{result.files_with_violations} 个文件）。"
        )
    print(f"  扫描文件: {result.files_scanned}")
    print("=" * 70)

    print_authoritative_reference()

    if args.strict and not result.passed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
