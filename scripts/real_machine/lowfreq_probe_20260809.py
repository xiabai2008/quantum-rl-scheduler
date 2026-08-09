#!/usr/bin/env python
"""tianyan176 低频窗口探测：单任务间隔提交，评估当前成功率（2026-08-09 round3-B）。"""

from __future__ import annotations

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

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from src.api.tianyan_cqlib import CqlibTianyanClient

PROBE_Q_CIRCUIT = "H Q1\nM Q1"
PROBE_SHOTS = 32
TARGET_MACHINE = "tianyan176"
INTERVAL = 10.0
ROUNDS = 6
WAIT_TIMEOUT = 180


def probe_once(client, i: int) -> tuple[bool, str]:
    try:
        tid = client.submit_quantum_task(
            qcis=PROBE_Q_CIRCUIT, shots=PROBE_SHOTS, task_name=f"lprobe_{i}"
        )
        if not tid:
            return False, "no-tid"
        tr = client.wait_for_task(tid, timeout=WAIT_TIMEOUT, poll_interval=5)
        prob = getattr(tr, "probability", None)
        if tr.status == "completed" and prob:
            return True, "completed fid_ok"
        return False, f"status={tr.status} prob={'yes' if prob else 'no'}"
    except Exception as e:
        return False, f"EXC: {str(e)[:60]}"


def main() -> int:
    api_key = os.environ.get("TIANYAN_API_KEY", "")
    if not api_key:
        print("错误: 未设置 TIANYAN_API_KEY")
        return 1
    client = CqlibTianyanClient(
        login_key=api_key, machine_name=TARGET_MACHINE, auto_retry_machine=False
    )
    ok = 0
    for i in range(ROUNDS):
        ts = datetime.now().strftime("%H:%M:%S")
        success, note = probe_once(client, i)
        ok += int(success)
        print(f"[{ts}] probe#{i}: {'OK' if success else 'FAIL'} ({note})", flush=True)
        time.sleep(INTERVAL)
    print(f"\nRESULT: {ok}/{ROUNDS} 成功率 {ok / ROUNDS:.0%}", flush=True)
    return 0 if ok >= 2 else 2


if __name__ == "__main__":
    sys.exit(main())
