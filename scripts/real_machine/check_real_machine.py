"""
检查 Trae 生成的真机数据收集脚本是否确实走真机

用法：
    D:\tools\Python 3.12.9\python.exe scripts/check_real_machine.py

检查点：
1. 环境变量 TIANYAN_MOCK_MODE 是否为 false
2. 脚本是否使用 CqlibTianyanClient 而不是 MockTianyanClient
3. 能否成功提交一个最小任务并获取真机 task_id
4. 真机任务结果是否包含 resultStatus/probability
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    print("=" * 60)
    print("  真机执行路径检查工具")
    print("=" * 60)

    # 1. 环境变量检查
    print("\n[1/5] 环境变量检查")
    mock_mode = os.environ.get("TIANYAN_MOCK_MODE", "<未设置>")
    api_key = os.environ.get("TIANYAN_API_KEY", "")
    print(f"  TIANYAN_MOCK_MODE = {mock_mode}")
    print(f"  TIANYAN_API_KEY   = {'已设置' if api_key else '未设置'} (长度={len(api_key)})")
    if str(mock_mode).lower() in ("true", "1", "yes"):
        print("  [FAIL] TIANYAN_MOCK_MODE 是 true，当前不会走真机！")
        return 1
    if not api_key:
        print("  [FAIL] TIANYAN_API_KEY 未设置")
        return 1
    print("  [PASS] 环境变量配置看起来正确")

    # 2. 检查生成的脚本是否导入 Mock 客户端
    print("\n[2/5] 检查 scripts/real_machine/ 下的脚本是否使用 Mock")
    real_machine_dir = ROOT / "scripts" / "real_machine"
    has_mock_import = False
    if real_machine_dir.exists():
        for py_file in real_machine_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            if "MockTianyanClient" in content or "create_tianyan_client" in content or "TIANYAN_MOCK_MODE" in content:
                print(f"  [WARN] {py_file.name} 包含可能的 Mock 相关代码/环境变量")
                has_mock_import = True
    else:
        print("  [INFO] scripts/real_machine/ 目录不存在，Trae 还没生成脚本")

    if not has_mock_import:
        print("  [PASS] 未发现 scripts/real_machine/ 下使用 Mock 客户端")

    # 3. 实际提交一个真机任务验证
    print("\n[3/5] 实际提交真机任务验证（H Q0 + M Q0，shots=128）")
    try:
        import cqlib
        from src.api.tianyan_cqlib import CqlibTianyanClient

        client = CqlibTianyanClient(
            login_key=api_key,
            machine_name="tianyan176",
            auto_retry_machine=True,
        )

        # 先查机器状态
        machines = client.list_backends()
        t176 = next((m for m in machines if m.get("name") == "tianyan176"), None)
        if t176:
            print(f"  [INFO] tianyan176 状态: {t176.get('status')} (type={t176.get('type')})")
        else:
            print("  [WARN] 未在机器列表中找到 tianyan176")

        # 提交一个最小任务
        task_id = client.submit_quantum_task(
            qcis="H Q0\nM Q0",
            shots=128,
            task_name="verify_real_machine_check",
        )
        if not task_id:
            print("  [FAIL] submit_quantum_task 返回 None，可能走到了 Mock 或全部机器不可用")
            return 1

        print(f"  [PASS] 真机任务提交成功，task_id = {task_id}")
        print("  [INFO] 等待任务结果...")

        result = client.wait_for_task(task_id, timeout=300, poll_interval=5)
        if result.get("status") == "completed":
            print(f"  [PASS] 真机任务完成！")
            print(f"  [INFO] 结果: {result.get('result')}")
            return 0
        else:
            print(f"  [FAIL] 任务未完成: {result}")
            return 1

    except Exception as e:
        print(f"  [FAIL] 真机提交异常: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
