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
版本：v1.0 (2026-07-02)
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class BenchmarkTracker:
    """Benchmark 结果追踪器，用于跨版本性能回归检测"""

    def __init__(
        self,
        history_path: Path = Path("results/benchmark_history.jsonl"),
        threshold: float = 10.0
    ) -> None:
        """
        初始化追踪器

        Args:
            history_path: 历史记录文件路径（JSONL 格式）
            threshold: 性能回归阈值（百分比），默认 10%
        """
        self.history_path = history_path
        self.threshold = threshold

        # 确保历史记录目录存在
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_git_info(self) -> dict[str, str]:
        """
        获取当前 git 信息（commit hash 和分支名）

        Returns:
            包含 commit 和 branch 的字典
        """
        try:
            # 获取短 commit hash
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()

            # 获取分支名
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()

            return {"commit": commit, "branch": branch}
        except subprocess.CalledProcessError:
            # 如果不在 git 仓库中，返回默认值
            return {"commit": "unknown", "branch": "unknown"}

    def read_benchmark_results(self, json_path: Path) -> list[dict[str, Any]]:
        """
        读取 pytest-benchmark 生成的 JSON 结果文件

        Args:
            json_path: pytest-benchmark --benchmark-json 输出的文件路径

        Returns:
            benchmark 结果列表，每项包含 name, mean, stddev, rounds
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # pytest-benchmark JSON 格式：benchmarks 数组包含所有测试结果
        benchmarks = []
        for bench in data.get("benchmarks", []):
            benchmarks.append({
                "name": bench["name"],
                "mean": bench["stats"]["mean"],
                "stddev": bench["stats"]["stddev"],
                "rounds": bench["stats"]["rounds"]
            })

        return benchmarks

    def append_to_history(
        self,
        benchmarks: list[dict[str, Any]],
        timestamp: datetime | None = None
    ) -> None:
        """
        将 benchmark 结果追加到历史记录文件

        Args:
            benchmarks: benchmark 结果列表
            timestamp: 时间戳，默认使用当前时间
        """
        if timestamp is None:
            timestamp = datetime.now()

        # 获取 git 信息
        git_info = self._get_git_info()

        # 构建历史记录条目
        entry = {
            "timestamp": timestamp.isoformat(timespec="seconds"),
            "commit": git_info["commit"],
            "branch": git_info["branch"],
            "benchmarks": benchmarks
        }

        # 追加到 JSONL 文件（每行一个 JSON 对象）
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_previous_entry(self) -> dict[str, Any] | None:
        """
        从历史记录中加载上一次的结果

        Returns:
            上一次的历史记录条目，如果历史记录为空则返回 None
        """
        if not self.history_path.exists():
            return None

        # 读取所有历史记录
        entries = []
        with open(self.history_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        # 返回最后一条记录（最新的一条）
        if entries:
            return entries[-1]
        return None

    def detect_regressions(
        self,
        current: list[dict[str, Any]],
        previous: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        检测性能回归

        Args:
            current: 当前 benchmark 结果
            previous: 上一次 benchmark 结果

        Returns:
            回归列表，每项包含 name, previous_mean, current_mean, change_percent
        """
        # 构建 previous 的 name -> mean 映射
        prev_map = {bench["name"]: bench["mean"] for bench in previous}

        regressions = []
        for curr_bench in current:
            name = curr_bench["name"]
            curr_mean = curr_bench["mean"]

            if name not in prev_map:
                # 新增的 benchmark，跳过
                continue

            prev_mean = prev_map[name]

            # 计算变化百分比：(当前 - 上次) / 上次 * 100
            change_percent = (curr_mean - prev_mean) / prev_mean * 100.0 if prev_mean > 0 else 0.0

            # 如果变化超过阈值，记录为回归
            if change_percent > self.threshold:
                regressions.append({
                    "name": name,
                    "previous_mean": prev_mean,
                    "current_mean": curr_mean,
                    "change_percent": change_percent
                })

        return regressions

    def generate_report(
        self,
        current: list[dict[str, Any]],
        previous: list[dict[str, Any]] | None
    ) -> str:
        """
        生成对比报告表格

        Args:
            current: 当前 benchmark 结果
            previous: 上一次 benchmark 结果（可选）

        Returns:
            格式化的报告字符串
        """
        lines = []
        lines.append("=" * 80)
        lines.append("Benchmark 性能对比报告")
        lines.append("=" * 80)
        lines.append("")

        if previous is None:
            lines.append("【首次运行】无历史数据可供对比")
            lines.append("")
            lines.append("当前 Benchmark 结果：")
            lines.append(f"{'名称':<50} {'平均时间(s)':<15} {'标准差':<15} {'轮次':<10}")
            lines.append("-" * 80)
            for bench in current:
                lines.append(
                    f"{bench['name']:<50} "
                    f"{bench['mean']:<15.6f} "
                    f"{bench['stddev']:<15.6f} "
                    f"{bench['rounds']:<10}"
                )
        else:
            # 构建 previous 的 name -> bench 映射
            prev_map = {bench["name"]: bench for bench in previous}

            lines.append(f"{'名称':<40} {'上次(s)':<12} {'当前(s)':<12} {'变化(%)':<12} {'状态':<10}")
            lines.append("-" * 80)

            for curr_bench in current:
                name = curr_bench["name"]
                curr_mean = curr_bench["mean"]

                if name not in prev_map:
                    # 新增的 benchmark
                    lines.append(f"{name:<40} {'N/A':<12} {curr_mean:<12.6f} {'N/A':<12} {'新增':<10}")
                    continue

                prev_bench = prev_map[name]
                prev_mean = prev_bench["mean"]

                # 计算变化百分比
                if prev_mean > 0:
                    change_percent = ((curr_mean - prev_mean) / prev_mean) * 100.0
                else:
                    change_percent = 0.0

                # 判断状态
                if change_percent > self.threshold:
                    status = "⚠️ 回归"
                elif change_percent < -self.threshold:
                    status = "✅ 提升"
                else:
                    status = "✅ 正常"

                lines.append(
                    f"{name:<40} "
                    f"{prev_mean:<12.6f} "
                    f"{curr_mean:<12.6f} "
                    f"{change_percent:<+12.2f} "
                    f"{status:<10}"
                )

        lines.append("=" * 80)
        return "\n".join(lines)


def main() -> int:
    """
    主函数：解析命令行参数并执行 benchmark 追踪

    Returns:
        退出码：0 表示成功，1 表示检测到性能回归
    """
    parser = argparse.ArgumentParser(
        description="Benchmark 结果跨版本追踪与性能回归检测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 基本用法
  python scripts/ci/benchmark_tracker.py benchmark_results.json

  # 自定义历史记录路径和阈值
  python scripts/ci/benchmark_tracker.py results.json \\
      --history-path custom_history.jsonl \\
      --threshold 15.0

  # 生成对比报告
  python scripts/ci/benchmark_tracker.py results.json --report
        """
    )

    parser.add_argument(
        "benchmark_json",
        type=Path,
        help="pytest-benchmark --benchmark-json 输出的 JSON 文件路径"
    )

    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path("results/benchmark_history.jsonl"),
        help="历史记录文件路径（默认：results/benchmark_history.jsonl）"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="性能回归阈值（百分比，默认：10.0）"
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="生成并打印对比报告"
    )

    args = parser.parse_args()

    # 验证输入文件存在
    if not args.benchmark_json.exists():
        print(f"错误：benchmark JSON 文件不存在：{args.benchmark_json}", file=sys.stderr)
        return 1

    # 创建追踪器
    tracker = BenchmarkTracker(
        history_path=args.history_path,
        threshold=args.threshold
    )

    # 读取当前 benchmark 结果
    print(f"读取 benchmark 结果：{args.benchmark_json}")
    current_benchmarks = tracker.read_benchmark_results(args.benchmark_json)
    print(f"找到 {len(current_benchmarks)} 个 benchmark")

    # 加载上一次的历史记录
    previous_entry = tracker.load_previous_entry()

    # 生成报告（如果请求）
    if args.report:
        previous_benchmarks = previous_entry["benchmarks"] if previous_entry else None
        report = tracker.generate_report(current_benchmarks, previous_benchmarks)
        print(report)

    # 追加当前结果到历史记录
    print(f"追加结果到历史记录：{args.history_path}")
    tracker.append_to_history(current_benchmarks)

    # 检测性能回归
    if previous_entry:
        previous_benchmarks = previous_entry["benchmarks"]
        regressions = tracker.detect_regressions(current_benchmarks, previous_benchmarks)

        if regressions:
            print(f"\n⚠️ 检测到 {len(regressions)} 个性能回归（阈值：{args.threshold}%）：")
            for reg in regressions:
                print(
                    f"  - {reg['name']}: "
                    f"{reg['previous_mean']:.6f}s → {reg['current_mean']:.6f}s "
                    f"({reg['change_percent']:+.2f}%)"
                )
            return 1
        else:
            print("\n✅ 未检测到性能回归")
    else:
        print("\n【首次运行】无历史数据可供对比")

    return 0


if __name__ == "__main__":
    sys.exit(main())
