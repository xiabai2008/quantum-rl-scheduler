#!/usr/bin/env python
"""
tianyan176 成功率窗口探针 + 自动触发实验（方案 A）

背景：tianyan176 免费机执行成功率分钟级波动（实测 15:09 窗口 5/5 成功，
其他时段 ~15% 失败率）。硬跑会浪费大量机时且数据不可信。

策略：
  1. 周期性探测窗口：每轮提交 3 个 H 门任务（间隔 5s），轮询等待结果
  2. 判定标准：completed ≥ 2/3 视为"高成功率窗口"
  3. 命中后：自动启动预注册三条件实验（run_prereg_three_conditions.py）
  4. 实验完成后：再次探测，循环直到任务完成或手动停止

用法：
    python scripts/real_machine/window_probe_runner.py --rounds 20
    python scripts/real_machine/window_probe_runner.py --probe-only   # 只探测不触发
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from src.api.tianyan_cqlib import CqlibTianyanClient

PROBE_Q_CIRCUIT = "H Q1\nM Q1"
PROBE_SHOTS = 32
PROBE_COUNT = 3
PROBE_INTERVAL = 5.0
WAIT_BETWEEN_ROUNDS = 60.0
TARGET_MACHINE = "tianyan176"


def probe_once(client) -> tuple[int, int]:
    """探测一轮：提交 3 个任务并等待结果，返回 (成功数, 提交数)。"""
    tids = []
    for i in range(PROBE_COUNT):
        try:
            tid = client.submit_quantum_task(
                qcis=PROBE_Q_CIRCUIT, shots=PROBE_SHOTS, task_name=f"winprobe_{i}"
            )
            if tid:
                tids.append(tid)
        except Exception as e:
            print(f"  probe submit EXC: {str(e)[:80]}")
        time.sleep(PROBE_INTERVAL)

    if not tids:
        return 0, 0

    completed = 0
    for tid in tids:
        try:
            tr = client.wait_for_task(tid, timeout=120, poll_interval=5)
            if tr.status == "completed" and tr.probability:
                completed += 1
        except Exception as e:
            print(f"  probe poll EXC: {str(e)[:80]}")
    return completed, len(tids)


def main():
    parser = argparse.ArgumentParser(description="tianyan176 成功率窗口探针 + 自动触发实验")
    parser.add_argument("--rounds", type=int, default=20, help="最大探测轮数（默认 20）")
    parser.add_argument("--probe-only", action="store_true", help="只探测不触发实验")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="实验 seeds（默认 18）")
    args = parser.parse_args()

    api_key = os.environ.get("TIANYAN_API_KEY", "")
    if not api_key:
        print("错误: 未设置 TIANYAN_API_KEY")
        sys.exit(1)

    client = CqlibTianyanClient(
        login_key=api_key, machine_name=TARGET_MACHINE, auto_retry_machine=False
    )

    for rnd in range(1, args.rounds + 1):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n=== 轮次 {rnd}/{args.rounds} @ {ts} ===", flush=True)
        try:
            completed, total = probe_once(client)
            rate = completed / total if total else 0.0
            print(f"探测结果: {completed}/{total} completed (成功率 {rate:.0%})", flush=True)
            if rate >= 0.67 and completed >= 2:
                print("✅ 高成功率窗口命中！开始实验...", flush=True)
                seeds = args.seeds or [42, 123, 456, 789, 1024, 2026, 314, 271, 828, 5566,
                                       7788, 1234, 2345, 3456, 4567, 5678, 6789, 7890]
                cmd = [
                    sys.executable,
                    str(_PROJECT_ROOT / "scripts/real_machine/run_prereg_three_conditions.py"),
                    "--seeds",
                ] + [str(s) for s in seeds]
                if args.probe_only:
                    print("  (probe-only 模式，不执行实验)")
                else:
                    print(f"  启动: {' '.join(cmd[:3])} --seeds {len(seeds)}个")
                    subprocess.run(cmd, check=False)
                    print("\n实验完成，继续探测下一窗口...")
            else:
                print(f"窗口不佳，等待 {WAIT_BETWEEN_ROUNDS}s 后重试...", flush=True)
        except Exception as e:
            print(f"探测异常: {type(e).__name__}: {str(e)[:100]}")

        if rnd < args.rounds:
            time.sleep(WAIT_BETWEEN_ROUNDS)

    print(f"\n探针结束（{args.rounds} 轮）")


if __name__ == "__main__":
    main()
