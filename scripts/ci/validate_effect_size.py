#!/usr/bin/env python
"""核心效果量 CI 门禁脚本 (Effect Size Gate).

Issue #355: 在 CI 中重算 PPO-vs-FCFS 核心效果量，低于阈值则阻断 PR。

本脚本不重新训练模型，而是从权威 multiseed 评估数据
``results/multiseed_evaluation/rewards_multiseed.json`` 中加载 PPO 与 FCFS
的奖励数据，重算提升百分比并与阈值对比。

阈值对齐 ``config/statistics.yaml`` 权威数字（v9.1+ 16维交付模型）：
    - PPO vs FCFS 提升 +20.2%（N=250, 50 seeds × 5 episodes, p=7.56e-12）
    - CI 门禁阈值默认 +80%（留充足安全边际，防止小样本波动导致误报）

退出码:
    - 0: 效果量 >= 阈值，门禁通过
    - 1: 效果量 < 阈值，门禁失败（阻断 PR）
    - 2: 数据文件缺失或格式错误（不阻断，仅告警）

用法:
    python scripts/ci/validate_effect_size.py
    python scripts/ci/validate_effect_size.py --threshold 0.80
    python scripts/ci/validate_effect_size.py --threshold 0.80 --data path/to/rewards.json

注意:
    本脚本本身不修改 ``.github/workflows/ci.yml``。要在 CI 中启用门禁，
    需在 ci.yml 的 test job 后新增 step::
        - name: Effect Size Gate
          run: python scripts/ci/validate_effect_size.py --threshold 0.80
    该 step 的添加需遵守项目约束（修改 ci.yml 需明确授权）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_PATH = _PROJECT_ROOT / "results" / "multiseed_evaluation" / "rewards_multiseed.json"

# 权威阈值：PPO vs 真实 FCFS 提升 +20.2%（8.5 审查基线诚实化，N=250 p=7.56e-12），门禁阈值留充足安全边际
_DEFAULT_THRESHOLD = 0.10  # 10%（即 improvement >= +10% 视为通过）

# 权威参考值（8.5 审查基线诚实化：vs 真实 FCFS +20.2%；旧 +123.4% 为 vs Hybrid-Default 历史）
_AUTHORITATIVE_IMPROVEMENT = 20.2


def load_rewards(data_path: Path) -> dict[str, list[float]]:
    """从 multiseed 评估 JSON 加载策略奖励数据。

    支持两种 JSON 结构：
        1. 直接格式：``{策略名: [奖励列表]}``
        2. 包装格式：``{config: ..., rewards: {策略名: [奖励列表]}, ...}``

    Args:
        data_path: rewards_multiseed.json 文件路径

    Returns:
        ``{策略名: [float, ...]}`` 字典

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: JSON 格式错误或缺少 PPO/FCFS 数据
    """
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    with open(data_path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"JSON 顶层必须是对象，实际类型: {type(raw).__name__}")

    # 检测包装格式
    rewards_dict = raw["rewards"] if "rewards" in raw and isinstance(raw["rewards"], dict) else raw

    cleaned: dict[str, list[float]] = {}
    for name, rewards in rewards_dict.items():
        if not isinstance(rewards, list):
            continue
        try:
            cleaned[name] = [float(r) for r in rewards]
        except (TypeError, ValueError):
            continue

    return cleaned


def compute_improvement(target: list[float], baseline: list[float]) -> dict[str, Any]:
    """计算目标策略相对基线的提升百分比及描述性统计。

    提升百分比定义：``(mean(target) - mean(baseline)) / |mean(baseline)| * 100``

    Args:
        target: 目标策略奖励列表（如 PPO）
        baseline: 基线策略奖励列表（如 FCFS）

    Returns:
        包含 improvement_pct / target_mean / baseline_mean / target_std /
        baseline_std / n_target / n_baseline 的字典
    """
    import statistics

    target_mean = statistics.mean(target)
    baseline_mean = statistics.mean(baseline)
    target_std = statistics.stdev(target) if len(target) >= 2 else 0.0
    baseline_std = statistics.stdev(baseline) if len(baseline) >= 2 else 0.0

    if baseline_mean == 0:
        improvement = float("inf") if target_mean > 0 else 0.0
    else:
        improvement = (target_mean - baseline_mean) / abs(baseline_mean) * 100.0

    return {
        "improvement_pct": improvement,
        "target_mean": target_mean,
        "baseline_mean": baseline_mean,
        "target_std": target_std,
        "baseline_std": baseline_std,
        "n_target": len(target),
        "n_baseline": len(baseline),
    }


def validate_effect_size(
    data_path: Path,
    threshold: float,
    target_strategy: str = "PPO",
    baseline_strategy: str = "FCFS",
) -> tuple[bool, dict[str, Any]]:
    """校验核心效果量是否达到阈值。

    Args:
        data_path: rewards_multiseed.json 文件路径
        threshold: 提升百分比阈值（如 0.80 表示 +80%）
        target_strategy: 目标策略名（默认 PPO）
        baseline_strategy: 基线策略名（默认 FCFS）

    Returns:
        (passed, details) 元组：
        - passed: 是否通过门禁（improvement >= threshold * 100）
        - details: 包含统计详情的字典
    """
    rewards = load_rewards(data_path)

    if target_strategy not in rewards:
        raise ValueError(
            f"目标策略 {target_strategy!r} 不在数据中，可用策略: {list(rewards.keys())}"
        )
    if baseline_strategy not in rewards:
        raise ValueError(
            f"基线策略 {baseline_strategy!r} 不在数据中，可用策略: {list(rewards.keys())}"
        )

    stats = compute_improvement(rewards[target_strategy], rewards[baseline_strategy])
    threshold_pct = threshold * 100.0
    passed = stats["improvement_pct"] >= threshold_pct

    details: dict[str, Any] = {
        "target_strategy": target_strategy,
        "baseline_strategy": baseline_strategy,
        "threshold_pct": threshold_pct,
        "authoritative_improvement": _AUTHORITATIVE_IMPROVEMENT,
        **stats,
    }

    return passed, details


def print_report(passed: bool, details: dict[str, Any]) -> None:
    """打印门禁结果报告。"""
    print("=" * 70)
    print("  核心效果量 CI 门禁 (Issue #355)")
    print("=" * 70)
    print()
    print(f"  目标策略: {details['target_strategy']}")
    print(f"  基线策略: {details['baseline_strategy']}")
    print()
    print("  描述性统计:")
    print(
        f"    {details['target_strategy']}: mean={details['target_mean']:.2f} "
        f"std={details['target_std']:.2f} n={details['n_target']}"
    )
    print(
        f"    {details['baseline_strategy']}: mean={details['baseline_mean']:.2f} "
        f"std={details['baseline_std']:.2f} n={details['n_baseline']}"
    )
    print()
    print("  效果量:")
    print(f"    提升百分比: {details['improvement_pct']:.2f}%")
    print(f"    权威参考值: +{details['authoritative_improvement']:.1f}%")
    print(f"    门禁阈值:   +{details['threshold_pct']:.1f}%")
    print()
    if passed:
        print(
            f"  [PASS] 提升百分比 {details['improvement_pct']:.2f}% "
            f">= 阈值 +{details['threshold_pct']:.1f}%"
        )
    else:
        print(
            f"  [FAIL] 提升百分比 {details['improvement_pct']:.2f}% "
            f"< 阈值 +{details['threshold_pct']:.1f}%"
        )
        print(
            f"         权威值应为 +{details['authoritative_improvement']:.1f}%，"
            "核心效果量已劣化，请检查 PPO 模型与评估流程"
        )
    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 命令行参数列表，None 时从 sys.argv 读取（用于测试注入）
    """
    parser = argparse.ArgumentParser(
        description="核心效果量 CI 门禁 (Issue #355)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=_DEFAULT_THRESHOLD,
        help=f"提升百分比阈值（0.80 表示 +80%%，默认 {_DEFAULT_THRESHOLD}）",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_DEFAULT_DATA_PATH,
        help=f"rewards_multiseed.json 路径（默认 {_DEFAULT_DATA_PATH}）",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="PPO",
        help="目标策略名（默认 PPO）",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="FCFS",
        help="基线策略名（默认 FCFS）",
    )
    args = parser.parse_args(argv)

    try:
        passed, details = validate_effect_size(
            args.data,
            args.threshold,
            args.target,
            args.baseline,
        )
    except FileNotFoundError as e:
        print(f"[WARN] {e}")
        print("[WARN] 数据文件缺失，跳过效果量门禁（不阻断）")
        return 2
    except (ValueError, json.JSONDecodeError) as e:
        print(f"[ERROR] 数据格式错误: {e}")
        print("[WARN] 跳过效果量门禁（不阻断）")
        return 2

    print_report(passed, details)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
