#!/usr/bin/env python
"""
一次性审计脚本 v2（实验① cost 评估配套，不修改 src/）：
对 prereg_high_ratio_eval.py 的 JSON 中所有 real_records[].real_task_id 做单次
get_task_status 查询，并捕获平台原始错误（"Run failure"/query_error/完成概率），
报告每条真机任务的实际终态。不做长轮询（failed 任务用单次查询即可判定）。

背景：实验 @prob 0.5 中 env 因 query_error 连败 3 次自动降级 Mock，
real_tasks_completed=0，但任务实际已提交平台。本脚本按 real_task_id 复查。

用法:
    python scripts/real_machine/audit_real_task_ids.py <实验JSON路径>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from src.api.tianyan_cqlib import CqlibTianyanClient

TARGET_MACHINE = "tianyan-287"


def _prob_of(st: Any) -> dict[str, float] | None:
    if isinstance(st, dict):
        prob = st.get("probability") or st.get("result")
    else:
        prob = getattr(st, "probability", None) or getattr(st, "result", None)
    if isinstance(prob, str) and prob:
        with __import__("contextlib").suppress(json.JSONDecodeError, ValueError):
            prob = json.loads(prob)
    return prob if isinstance(prob, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser(description="按 real_task_id 单次复查实验 JSON 中真机任务")
    parser.add_argument("json_path", type=Path)
    args = parser.parse_args()

    if not args.json_path.exists():
        print(f"❌ 文件不存在: {args.json_path}")
        return 1
    data = json.loads(args.json_path.read_text(encoding="utf-8"))

    tasks: list[dict[str, Any]] = []
    if "results" in data:
        for r in data["results"]:
            seed = r.get("seed")
            strat = r.get("strategy")
            for rr in r.get("real_records") or []:
                tid = rr.get("real_task_id") or rr.get("task_id")
                # 仅真机 platform id（数字长串），跳过内部 T0001 等符号 id
                if tid and str(tid).isdigit():
                    tasks.append(
                        {
                            "seed": seed,
                            "strategy": strat,
                            "task_name": rr.get("task_id"),
                            "tid": str(tid),
                        }
                    )
    else:
        print("❌ 非预期的 JSON 结构（无 results 列表）")
        return 1

    if not tasks:
        print("✅ JSON 中无数值 real_task_id（无真机任务）")
        return 0

    api_key = os.environ.get("TIANYAN_API_KEY", "")
    if not api_key:
        print("❌ 未设置 TIANYAN_API_KEY")
        return 1
    client = CqlibTianyanClient(
        login_key=api_key, machine_name=TARGET_MACHINE, auto_retry_machine=False
    )

    print(f"待复查: {len(tasks)} 个真机任务（单次查询）")
    completed = 0
    failed = 0
    other = 0
    for i, t in enumerate(tasks):
        st = None
        try:
            st = client.get_task_status(t["tid"])
        except Exception as e:
            print(
                f"  [{i + 1}/{len(tasks)}] {t['seed']}/{t['strategy']} "
                f"{t['task_name']} real_task={t['tid']} → EXC:{str(e)[:90]}"
            )
            other += 1
            continue
        status = str(st.get("status", "unknown"))
        prob = _prob_of(st)
        if status == "completed":
            completed += 1
        elif "failed" in status or "error" in status:
            failed += 1
        else:
            other += 1
        print(
            f"  [{i + 1}/{len(tasks)}] {t['seed']}/{t['strategy']} "
            f"{t['task_name']} real_task={t['tid']} → status={status} prob={prob}"
        )
        time.sleep(0.5)

    print(f"\n汇总: completed={completed}, failed={failed}, other={other}, 共 {len(tasks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
