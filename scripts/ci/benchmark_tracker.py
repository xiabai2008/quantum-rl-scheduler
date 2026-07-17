#!/usr/bin/env python3
"""
Benchmark 结果跨版本追踪与性能回归检测工具

功能：
1. 读取 pytest-benchmark 生成的 JSON 结果文件
2. 将结果追加到 JSONL 历史记录文件（包含时间戳、git commit、分支信息）
3. 对比当前结果与上一次历史记录，检测性能回归
4. 当任何 benchmark 的平均执行时间增长超过阈值时输出 WARNING
5. 支持生成对比报告表格

使用示例：
    # 基本用法：追踪 benchmark 结果
    python scripts/ci/benchmark_tracker.py benchmark_results.json
    
    # 自定义历史记录路径和回归阈值
    python scripts/ci/benchmark_tracker.py benchmark_results.json \
        --history-path results/benchmark_history.jsonl \
        --threshold 15.0
    
    # 生成对比报告
    python scripts/ci/benchmark_tracker.py benchmark_results.json --report

作者：量子RL调度系统团队
日期：2026-07-02
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class BenchmarkTracker:
    """Benchmark 结果追踪器"""

    def __init__(
        self, history_path: str = "results/benchmark_history.jsonl"
    ) -> None:
        """初始化追踪器

        Args:
            history_path: 历史记录文件路径
        """
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def load_current_results(self, json_path: str) -> List[Dict[str, Any]]:
        """加载当前 benchmark 结果

        Args:
            json_path: pytest-benchmark JSON 文件路径

        Returns:
            benchmark 结果列表
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        benchmarks = []
        for bench in data.get("benchmarks", []):
            benchmarks.append(
                {
                    "name": bench["name"],
                    "mean": bench["stats"]["mean"],
                    "stddev": bench["stats"]["stddev"],
                    "rounds": bench["stats"]["rounds"],
                }
            )
        return benchmarks

    def get_git_info(self) -> Dict[str, str]:
        """获取 Git 信息

        Returns:
            包含 commit 和 branch 的字典
        """
        try:
            commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
        except Exception:
            commit = "unknown"

        try:
            branch = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except Exception:
            branch = "unknown"

        return {"commit": commit, "branch": branch}

    def append_to_history(self, benchmarks: List[Dict[str, Any]]) -> None:
        """追加结果到历史记录

        Args:
            benchmarks: benchmark 结果列表
        """
        git_info = self.get_git_info()
        record = {
            "timestamp": datetime.now().isoformat(),
            "commit": git_info["commit"],
            "branch": git_info["branch"],
            "benchmarks": benchmarks,
        }

        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"✅ 结果已追加到: {self.history_path}")
        print(f"   Commit: {git_info['commit']}")
        print(f"   Branch: {git_info['branch']}")

    def load_previous_results(self) -> Optional[Dict[str, Any]]:
        """加载上一次的历史记录

        Returns:
            上一次的历史记录，如果不存在则返回 None
        """
        if not self.history_path.exists():
            return None

        with open(self.history_path, encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            return None

        # 返回最后一条记录
        return json.loads(lines[-1])

    def detect_regressions(
        self,
        current: List[Dict[str, Any]],
        previous: List[Dict[str, Any]],
        threshold: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """检测性能回归

        Args:
            current: 当前结果
            previous: 上一次结果
            threshold: 回归阈值（百分比）

        Returns:
            回归列表
        """
        regressions = []
        prev_map = {b["name"]: b for b in previous}

        for curr in current:
            name = curr["name"]
            if name in prev_map:
                prev = prev_map[name]
                prev_mean = prev["mean"]
                curr_mean = curr["mean"]

                if prev_mean > 0:
                    change_pct = ((curr_mean - prev_mean) / prev_mean) * 100

                    if change_pct > threshold:
                        regressions.append(
                            {
                                "name": name,
                                "previous": prev_mean,
                                "current": curr_mean,
                                "change_pct": change_pct,
                            }
                        )

        return regressions

    def generate_report(
        self,
        current: List[Dict[str, Any]],
        previous: Optional[Dict[str, Any]],
        threshold: float = 10.0,
    ) -> None:
        """生成对比报告

        Args:
            current: 当前结果
            previous: 上一次结果
            threshold: 回归阈值
        """
        print("\n" + "=" * 80)
        print("Benchmark 对比报告")
        print("=" * 80)

        if not previous:
            print("\n⚠️  无历史记录，跳过对比")
            return

        print(f"\n上次运行: {previous['timestamp']}")
        print(f"上次 Commit: {previous['commit']}")
        print(f"上次 Branch: {previous['branch']}")

        prev_benchmarks = previous.get("benchmarks", [])
        regressions = self.detect_regressions(current, prev_benchmarks, threshold)

        if regressions:
            print(f"\n❌ 检测到 {len(regressions)} 个性能回归 (阈值: {threshold}%):\n")
            print(f"{'Benchmark':<50} {'上次 (s)':<12} {'当前 (s)':<12} {'变化':<10}")
            print("-" * 80)
            for reg in regressions:
                print(
                    f"{reg['name']:<50} {reg['previous']:<12.6f} "
                    f"{reg['current']:<12.6f} +{reg['change_pct']:.1f}%"
                )
        else:
            print(f"\n✅ 无性能回归 (阈值: {threshold}%)")

        print("=" * 80)


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Benchmark 结果追踪与回归检测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 追踪 benchmark 结果
  python scripts/ci/benchmark_tracker.py benchmark_results.json

  # 自定义历史记录路径和阈值
  python scripts/ci/benchmark_tracker.py benchmark_results.json \\
      --history-path results/benchmark_history.jsonl \\
      --threshold 15.0

  # 生成对比报告
  python scripts/ci/benchmark_tracker.py benchmark_results.json --report
        """,
    )

    parser.add_argument("json_file", type=str, help="pytest-benchmark JSON 文件路径")
    parser.add_argument(
        "--history-path",
        type=str,
        default="results/benchmark_history.jsonl",
        help="历史记录文件路径 (默认: results/benchmark_history.jsonl)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="回归检测阈值（百分比，默认: 10.0）",
    )
    parser.add_argument(
        "--report", action="store_true", help="生成对比报告"
    )

    args = parser.parse_args()

    tracker = BenchmarkTracker(args.history_path)

    # 加载当前结果
    print(f"📊 加载 benchmark 结果: {args.json_file}")
    current = tracker.load_current_results(args.json_file)
    print(f"   找到 {len(current)} 个 benchmarks")

    # 加载历史记录用于对比
    previous = tracker.load_previous_results()

    # 生成报告
    if args.report:
        tracker.generate_report(current, previous, args.threshold)

    # 追加到历史记录
    tracker.append_to_history(current)

    # 检测回归
    if previous:
        regressions = tracker.detect_regressions(
            current, previous["benchmarks"], args.threshold
        )
        if regressions:
            print(f"\n⚠️  检测到 {len(regressions)} 个性能回归")
            sys.exit(1)

    print("\n✅ Benchmark 追踪完成")


if __name__ == "__main__":
    main()
