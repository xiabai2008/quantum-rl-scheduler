#!/usr/bin/env python
"""round5-D: tianyan176 噪声分布扩充（MBS 保真度测量）。

目标：把"量子→AI"噪声建模的数据源从 tianyan-287 10seeds MBS 分布
扩充到 20+ 数据点（176 上测量 H 门，不依赖统计显著，每成功 1 次 1 数据点）。

用法:
    python scripts/real_machine/mbs_distribution_expansion.py --target 15 --wait-hours 2
"""

from __future__ import annotations

import argparse
import json
import os
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

TARGET_MACHINE = "tianyan176"
SHOTS = 1024
QCIS = "H Q1\nM Q1"
RESULTS_PATH = _PROJECT_ROOT / "results" / "real_machine" / "mbs_expansion_20260810.json"
PROBE_INTERVAL = 8.0


def machine_status(client: CqlibTianyanClient) -> str:
    try:
        for b in client.list_backends():
            name = b.get("name") or b.get("machine_name") or ""
            if name == TARGET_MACHINE:
                return str(b.get("status", "unknown"))
    except Exception:
        pass
    return "unknown"


def measure_once(client: CqlibTianyanClient, idx: int) -> dict | None:
    try:
        tid = client.submit_quantum_task(qcis=QCIS, shots=SHOTS, task_name=f"mbs_exp_{idx}")
        if not tid:
            return None
        tr = client.wait_for_task(tid, timeout=180, poll_interval=5)
        prob = getattr(tr, "probability", None) or {}
        if tr.status == "completed" and prob:
            p0 = float(prob.get("0", 0.0))
            p1 = float(prob.get("1", 0.0))
            return {
                "task_id": str(tid),
                "timestamp": datetime.now().isoformat(),
                "p0": p0,
                "p1": p1,
                "fidelity": round(1 - abs(p0 - p1), 6),
                "mbs": round(1 - 2 * abs(p0 - 0.5), 6),
            }
        return None
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="176 噪声分布扩充（MBS 测量）")
    parser.add_argument("--target", type=int, default=15, help="目标成功测量数（含已有 10）")
    parser.add_argument("--wait-hours", type=float, default=2.0)
    args = parser.parse_args()

    api_key = os.environ.get("TIANYAN_API_KEY", "")
    if not api_key:
        print("错误: 未设置 TIANYAN_API_KEY")
        return 1

    client = CqlibTianyanClient(
        login_key=api_key, machine_name=TARGET_MACHINE, auto_retry_machine=False
    )

    deadline = time.time() + args.wait_hours * 3600
    results: list[dict] = []
    idx = 0
    while len(results) < args.target and time.time() < deadline:
        st = machine_status(client)
        if st != "running":
            print(f"[{datetime.now():%H:%M:%S}] 176 status={st}，60s 后重试...", flush=True)
            time.sleep(60)
            continue
        m = measure_once(client, idx)
        idx += 1
        if m:
            results.append(m)
            print(
                f"[{datetime.now():%H:%M:%S}] 成功 #{len(results)}/{args.target} "
                f"mbs={m['mbs']:.4f} fid={m['fidelity']:.4f}",
                flush=True,
            )
        else:
            print(f"[{datetime.now():%H:%M:%S}] 测量失败（窗口/执行），继续...", flush=True)
        time.sleep(PROBE_INTERVAL)

    data = {
        "experiment": "mbs_expansion_20260810",
        "machine": TARGET_MACHINE,
        "shots": SHOTS,
        "qcis": QCIS,
        "timestamp": datetime.now().isoformat(),
        "success_count": len(results),
        "target": args.target,
        "results": results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {RESULTS_PATH}（成功 {len(results)}/{args.target}）", flush=True)
    return 0 if len(results) >= args.target else 2


if __name__ == "__main__":
    sys.exit(main())
