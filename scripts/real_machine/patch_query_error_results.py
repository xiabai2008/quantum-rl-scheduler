#!/usr/bin/env python
"""
真机结果复查补录脚本（2026-08-14 平台行为适配，实验③配套）

背景：
    天衍-287 平台行为变化（8/14 实测）：任务完成前 get_task_status 对查询返回
    "Failed to query the experimental result."（query_error），任务实际需 ~5 分钟
    完成；而 tianyan287_multiseed.py 的轮询（wait_for_task 连续 3 次 query_error
    快速终止，Issue #407）会把执行中的任务误判失败 → 实验 JSON 中记录为
    query_error，但任务实际已完成。

方案（零代码改动纪律）：
    - tianyan287_multiseed.py 保持原样（52 单元测试全绿，不改提交协议/统计口径）
    - 本脚本对实验 JSON 中 status != completed 且 task_id 非空的记录，
      用 get_task_status 复查（query_error 容忍轮询至 completed/error/超时），
      completed 则补录 probability / measurement_balance_score / 完成时间。
    - 输出修正副本 <原名>_patched.json，**不修改原始 JSON**（审计轨迹保留）。

用法：
    python scripts/real_machine/patch_query_error_results.py <实验JSON路径>
    # 输出: <实验JSON路径> 同目录 *_patched.json + 控制台补录摘要

退出码：0 = 全部可补录记录已复查（剩余非 completed 数量打印在摘要）；
         1 = 无有效 task_id 可复查 / 输入文件缺失
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from src.api.tianyan_cqlib import CqlibTianyanClient

TARGET_MACHINE = "tianyan-287"
REVIEW_TIMEOUT = 600  # 单任务复查轮询上限（与平台 8/14 实测 ~5 分钟完成对齐）
REVIEW_INTERVAL = 10


def _status_of(st: Any) -> str:
    if isinstance(st, dict):
        return str(st.get("status", "unknown"))
    return str(getattr(st, "status", "unknown"))


def _prob_of(st: Any) -> dict[str, float] | None:
    if isinstance(st, dict):
        prob = st.get("probability") or st.get("result")
    else:
        prob = getattr(st, "probability", None) or getattr(st, "result", None)
    if isinstance(prob, str) and prob:
        with __import__("contextlib").suppress(json.JSONDecodeError, ValueError):
            prob = json.loads(prob)
    return prob if isinstance(prob, dict) else None


def review_task(client: CqlibTianyanClient, task_id: str) -> dict[str, Any]:
    """query_error 容忍复查：轮询至 completed/error 或超时。"""
    start = time.time()
    while time.time() - start < REVIEW_TIMEOUT:
        st = client.get_task_status(task_id)
        status = _status_of(st)
        if status in ("completed", "error"):
            return {
                "status": status,
                "probability": _prob_of(st),
                "error": (st.get("error") if isinstance(st, dict) else getattr(st, "error", None)),
            }
        time.sleep(REVIEW_INTERVAL)
    return {"status": "timeout", "probability": None, "error": f"复查超时（{REVIEW_TIMEOUT}s）"}


def patch_record(record: dict[str, Any], review: dict[str, Any]) -> bool:
    """补录单条记录；返回是否补录成功（completed）。"""
    if review["status"] != "completed" or not review.get("probability"):
        return False
    prob = review["probability"]
    p0 = float(prob.get("0", 0.0))
    p1 = float(prob.get("1", 0.0))
    score = max(0.0, min(1.0, 1.0 - abs(p0 - 0.5) - abs(p1 - 0.5)))
    record["status"] = "completed"
    record["probability"] = prob
    record["measurement_balance_score"] = round(score, 4)
    record["error"] = None
    record["completed_at"] = datetime.now().isoformat()
    if record.get("submitted_at"):
        with __import__("contextlib").suppress(ValueError, TypeError):
            record["elapsed_seconds"] = round(
                (datetime.fromisoformat(record["completed_at"])
                 - datetime.fromisoformat(str(record["submitted_at"]))).total_seconds(),
                2,
            )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="真机 query_error 记录复查补录（实验③配套）")
    parser.add_argument("json_path", type=Path, help="实验 JSON（multiseed_data_<ts>.json 或 smoke JSON）")
    args = parser.parse_args()

    if not args.json_path.exists():
        print(f"❌ 文件不存在: {args.json_path}")
        return 1
    data = json.loads(args.json_path.read_text(encoding="utf-8"))

    # 收集待复查记录（smoke 单条结构 or multiseed results 列表）
    records: list[dict[str, Any]] = []
    if "results" in data:
        records = [r for r in data["results"] if r.get("task_id") and r.get("status") != "completed"]
    elif data.get("task_id"):
        records = [data]

    pending = [r for r in records if r.get("task_id")]
    if not pending:
        print("✅ 无待复查记录（全部 completed 或无 task_id）")
        return 0

    api_key = os.environ.get("TIANYAN_API_KEY", "")
    if not api_key:
        print("❌ 未设置 TIANYAN_API_KEY")
        return 1
    client = CqlibTianyanClient(login_key=api_key, machine_name=TARGET_MACHINE,
                                auto_retry_machine=False)
    if getattr(client, "machine_name", TARGET_MACHINE) != TARGET_MACHINE:
        print("❌ 客户端机器不一致（禁止回退）")
        return 1

    print(f"待复查: {len(pending)} 条（query_error/失败记录）")
    patched = 0
    for i, rec in enumerate(pending):
        task_id = str(rec["task_id"])
        review = review_task(client, task_id)
        if patch_record(rec, review):
            patched += 1
            print(f"  [{i+1}/{len(pending)}] task {task_id}: 补录 completed "
                  f"(prob={review['probability']})")
        else:
            print(f"  [{i+1}/{len(pending)}] task {task_id}: 仍为 {review['status']}"
                  f"（{str(review.get('error'))[:60]}）")

    # 保存修正副本
    out_path = args.json_path.with_name(args.json_path.stem + "_patched.json")
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    print(f"\n补录完成: {patched}/{len(pending)} | 修正副本: {out_path}")

    # 汇总（对齐 analyze_multiseed_v3 审计口径）
    if "results" in data:
        all_recs = [r for r in data["results"] if not r.get("smoke_test")]
        completed = sum(1 for r in all_recs if r.get("metrics", {}).get("real_tasks_completed", 0))
        # 冒烟/单条结构下 metrics 不存在，直接按 status 统计
        status_counts: dict[str, int] = {}
        for r in data["results"]:
            s = r.get("status") or r.get("metrics", {}).get("real_tasks_completed", 0) or "unknown"
            status_counts[str(s)] = status_counts.get(str(s), 0) + 1
        print("记录状态分布:", status_counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
