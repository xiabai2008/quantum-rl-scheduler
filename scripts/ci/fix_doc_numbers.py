#!/usr/bin/env python
"""批量修复文档中的禁用表述 - 第二版：更精准的替换。"""

from __future__ import annotations

import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 只处理核心文档（不处理results/下的历史报告、不处理临时PR审查文件）
TARGET_FILES = [
    "README.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "参赛总结报告-草案.md",
    "docs/award_roadmap.md",
    "docs/bidirectional_empowerment.md",
    "docs/Code_Wiki.md",
    "docs/defense_ppt_outline.md",
    "docs/defense_qa_handbook.md",
    "docs/demo_script.md",
    "docs/demo_video_script.md",
    "docs/novelty_statement.md",
    "docs/sota_comparison.md",
    "docs/value_quantification.md",
    "docs/platform_landing_value.md",
    "docs/production_roadmap.md",
    "docs/dual_empowerment_asymmetry_analysis.md",
    "docs/real_machine_annealing_research.md",
    "docs/adr/ADR-003-cqlib-sdk-for-real-machine-integration.md",
    "docs/real_machine_verification_boundary.md",
    "docs/annealing_significance-defense.md",
    "docs/technical_whitepaper.md",
    "docs/requirements_traceability.md",
]


# 按顺序的替换规则
def apply_replacements(content: str, filepath: str) -> tuple[str, int, list[str]]:
    changes = []
    total = 0

    # ============ 1. 284次 -> 315次真机调用 ============
    patterns_284 = [
        (r"284\s*次SDK调用", "315次SDK调用", "284次SDK调用->315次"),
        (r"284\s*次(真机)?调用(?!100)", "315次调用", "284次调用->315次"),
        (r"284\s*次真机", "315次真机", "284次真机->315次"),
    ]
    for pat, rep, desc in patterns_284:
        content, n = re.subn(pat, rep, content)
        if n > 0:
            total += n
            changes.append(f"  - {desc}: {n}处")

    # ============ 2. 退火p值 0.190 -> 0.9430 ============
    # 注意：要保留在"5 seeds下"这样的历史说明语境中的0.190
    patterns_p = [
        (r"p=0\.190（不显著）", "p=0.9430（20seeds，统计不显著）", "退火p值(带不显著)"),
        (r"p=0\.190 不显著", "p=0.9430（20seeds，不显著）", "退火p值2"),
        (r"p=0\.190，不显著", "p=0.9430（20seeds，不显著）", "退火p值3"),
        (r"p=0\.190\)", "p=0.9430)", "退火p值4（括弧）"),
        (r"p=0\.190,", "p=0.9430,", "退火p值5（逗号）"),
        (r"p=0\.190（", "p=0.9430（", "退火p值6（中文括号）"),
        (r"\| 0\.190 \|", "| 0.9430 |", "退火p值表格"),
        (
            r"p\s*=\s*0\.190(?!\s*(在|下|对应|旧|5))",
            "p=0.9430",
            "退火p值默认替换（排除历史说明语境）",
        ),
    ]
    for pat, rep, desc in patterns_p:
        content, n = re.subn(pat, rep, content)
        if n > 0:
            total += n
            changes.append(f"  - {desc}: {n}处")

    # ============ 3. v5虚报相关 ============
    # 这些已在AGENTS/README中修复，检查其他文档
    patterns_v5 = [
        (r"v5已完成", "制作中", "删除v5已完成"),
        (r"技术白皮书.*v5.*11章", "技术白皮书（7章）", "v5白皮书11章修正"),
        (r"答辩PPT.*v5", "答辩PPT（制作中）", "v5 PPT修正"),
    ]
    for pat, rep, desc in patterns_v5:
        content, n = re.subn(pat, rep, content)
        if n > 0:
            total += n
            changes.append(f"  - {desc}: {n}处")

    # ============ 4. 利用率30%目标修正（在defense_ppt/award_roadmap等中） ============
    patterns_util = [
        (
            r"资源利用率(提升)?\s*[≥>=]\s*30%.*已?达成",
            "资源利用率+7.9%（p=0.0046，高负载场景显著优于FCFS），30%目标部分达成",
            "利用率30%达成修正",
        ),
        (r"利用率.*≥30%.*达标", "利用率+7.9%，30%目标部分达成", "利用率达标修正"),
    ]
    for pat, rep, desc in patterns_util:
        content, n = re.subn(pat, rep, content)
        if n > 0:
            total += n
            changes.append(f"  - {desc}: {n}处")

    # ============ 5. defense_ppt/defense_qa中的特定问题 ============
    if "defense_ppt" in filepath or "defense_qa" in filepath or "award_roadmap" in filepath:
        # 等待时间-5.7%在PPT大纲/QA手册中
        content, n = re.subn(
            r"等待时间-5\.7%",
            "单seed等待时间-5.7%（探索性，10seeds+6.1%）",
            content,
        )
        if n > 0:
            total += n
            changes.append(f"  - 等待时间-5.7%诚实披露: {n}处")

        # 284 次真机调用在QA手册中
        content, n = re.subn(r"284 次调用", "315次调用", content)
        if n > 0:
            total += n
            changes.append(f"  - 284次调用修正: {n}处")

        content, n = re.subn(r"284 次真机", "315次真机", content)
        if n > 0:
            total += n
            changes.append(f"  - 284次真机修正: {n}处")

    # ============ 5. 所有目标文件中的通用修复 ============
    # 各种284变体
    content, n = re.subn(r"284次调用", "315次调用", content)
    if n > 0:
        total += n
        changes.append(f"  - 284次调用(无空格)修正: {n}处")

    content, n = re.subn(r"284 次(正式)?(SDK)?调用", "315次SDK调用", content)
    if n > 0:
        total += n
        changes.append(f"  - 284次(带空格)修正: {n}处")

    content, n = re.subn(r"284 次真机", "315次真机", content)
    if n > 0:
        total += n
        changes.append(f"  - 284次真机(带空格)修正: {n}处")

    content, n = re.subn(r"284\s*次(可用性)?验证", "315次可用性验证", content)
    if n > 0:
        total += n
        changes.append(f"  - 284次验证修正: {n}处")

    content, n = re.subn(r"284\s*次成功", "315次调用100%成功", content)
    if n > 0:
        total += n
        changes.append(f"  - 284次成功修正: {n}处")

    content, n = re.subn(r"284\s*次\s*100%", "315次调用100%", content)
    if n > 0:
        total += n
        changes.append(f"  - 284次100%修正: {n}处")

    content, n = re.subn(r"100%（284 次）", "100%（315次）", content)
    if n > 0:
        total += n
        changes.append(f"  - 100%(284次)修正: {n}处")

    content, n = re.subn(r"100%（284次）", "100%（315次）", content)
    if n > 0:
        total += n
        changes.append(f"  - 100%(284次无空格)修正: {n}处")

    content, n = re.subn(r"284\s*次，", "315次，", content)
    if n > 0:
        total += n
        changes.append(f"  - 284次逗号修正: {n}处")

    # real_machine_verification_boundary 特殊情况
    content, n = re.subn(
        r"284 次正式 SDK 调用，全部获得 task ID",
        "315次SDK调用全部获得task ID（284次主验证+31次Issue #244补充）",
        content,
    )
    if n > 0:
        total += n
        changes.append(f"  - 真机验证边界文档修正: {n}处")

    # technical_whitepaper SDK调用总次数
    content, n = re.subn(r"\| SDK调用总次数 \| 284次 \|", "| SDK调用总次数 | 315次 |", content)
    if n > 0:
        total += n
        changes.append(f"  - 白皮书SDK总次数修正: {n}处")

    # defense_qa 中大量 p=0.190 的引用
    content, n = re.subn(r"p=0\.190，", "p=0.9430（20seeds），", content)
    if n > 0:
        total += n
        changes.append(f"  - p=0.190中文逗号修正: {n}处")

    content, n = re.subn(r"p=0\.190。", "p=0.9430（20seeds）。", content)
    if n > 0:
        total += n
        changes.append(f"  - p=0.190句号修正: {n}处")

    content, n = re.subn(r"p=0\.190；", "p=0.9430（20seeds）；", content)
    if n > 0:
        total += n
        changes.append(f"  - p=0.190分号修正: {n}处")

    content, n = re.subn(r"p=0\.190（不", "p=0.9430（20seeds，统计不", content)
    if n > 0:
        total += n
        changes.append(f"  - p=0.190(不显著)修正: {n}处")

    content, n = re.subn(r"p=0\.190，未", "p=0.9430（20seeds），未", content)
    if n > 0:
        total += n
        changes.append(f"  - p=0.190(未达)修正: {n}处")

    # 覆盖率94.2% -> 93.58%
    content, n = re.subn(r"94\.2%", "93.58%", content)
    if n > 0:
        total += n
        changes.append(f"  - 覆盖率94.2%->93.58%: {n}处")

    return content, total, changes


def main():
    print("=" * 70)
    print("  批量文档口径修复（第二轮：核心文档精准替换）")
    print("=" * 70)

    total_replacements = 0
    files_changed = 0

    for rel_path in TARGET_FILES:
        fpath = _PROJECT_ROOT / rel_path
        if not fpath.exists():
            continue

        try:
            content = fpath.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        new_content, count, changes = apply_replacements(content, rel_path)

        if count > 0 and new_content != content:
            fpath.write_text(new_content, encoding="utf-8")
            files_changed += 1
            total_replacements += count
            print(f"\n[FIXED] {rel_path} ({count}处)")
            for c in changes:
                print(c)

    print("\n" + "=" * 70)
    print(f"  完成：修改了 {files_changed} 个文件，共 {total_replacements} 处替换")
    print("=" * 70)


if __name__ == "__main__":
    main()
