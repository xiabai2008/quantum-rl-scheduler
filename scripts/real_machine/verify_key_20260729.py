"""验证真机密钥并提交最小任务（2026-07-29）。

用法：
    python scripts/real_machine/verify_key_20260729.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    """验证真机密钥并提交最小任务，返回退出码。"""
    from dotenv import load_dotenv

    load_dotenv()

    from src.api.tianyan_cqlib import CqlibTianyanClient

    api_key = os.environ.get("TIANYAN_API_KEY", "")
    if not api_key:
        print("[FAIL] TIANYAN_API_KEY 未设置")
        return 1

    print("=== 真机密钥验证 ===")
    print(f"API Key 长度: {len(api_key)}")

    client = CqlibTianyanClient(
        login_key=api_key,
        machine_name="tianyan-287",
        auto_retry_machine=True,
    )
    print("客户端初始化成功")

    # 列出可用后端
    print("\n=== 查询可用量子计算机 ===")
    machines = client.list_backends()
    print(f"可用机器数: {len(machines)}")
    t287 = next((m for m in machines if m.get("name") == "tianyan-287"), None)
    if t287:
        print(f"tianyan-287 状态: {t287.get('status')} (type={t287.get('type')})")
    else:
        print("[WARN] 未找到 tianyan-287")

    # 提交最小任务
    # 已核实：天衍-287 物理比特 Q1～Q105，没有 Q0
    # H Q0/M Q0 QCIS 校验 false；H Q1/M Q1 QCIS 校验 true
    print("\n=== 提交最小真机任务（H Q1 + M Q1, shots=128）===")
    task_id = client.submit_quantum_task(
        qcis="H Q1\nM Q1",
        shots=128,
        task_name="verify_real_machine_20260729",
    )
    print(f"task_id = {task_id}")

    if not task_id:
        print("[FAIL] 任务提交失败")
        return 1

    print("\n=== 等待任务结果（timeout=300s, poll=5s）===")
    result = client.wait_for_task(task_id, timeout=300, poll_interval=5)
    status = result.get("status")
    print(f"状态: {status}")
    print(f"结果: {result.get('result')}")

    if status == "completed":
        print("\n=== 真机闭环验证成功！密钥有效，tianyan-287 可用。===")
        return 0
    print(f"\n[FAIL] 任务未完成: {result}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
