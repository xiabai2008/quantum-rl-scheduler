"""
天衍-176 真机连通性测试脚本

步骤：
1. 验证 API Key 有效性
2. 列出所有可用量子计算机
3. 检查天衍-176 机器状态
4. 尝试提交简单量子电路（H Q0 + M Q0）
5. 查询任务结果
"""

import os
import sys
import time

# --- 设置环境变量（必须在导入前） ---
os.environ["TIANYAN_MOCK_MODE"] = "false"
os.environ["TIANYAN_API_KEY"] = "qCNQVWtZacuH6XLWz9O/Ngqqps8AZU6HrlUfDzqjsc7hkowBxbdjuZnSUxaiO6v4srso4Q=="

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEPARATOR = "=" * 60


def test_tianyan176():
    machine_name = "tianyan176"
    api_key = os.environ["TIANYAN_API_KEY"]

    print(SEPARATOR)
    print("  天衍-176 真机连通性测试")
    print(SEPARATOR)

    # ==================== 步骤 1: 导入 cqlib ====================
    print("\n[Step 1] 导入 cqlib 库...")
    try:
        import cqlib
        print(f"  [PASS] cqlib 版本: {cqlib.__version__ if hasattr(cqlib, '__version__') else 'OK'}")
    except ImportError as e:
        print(f"  [FAIL] cqlib 未安装: {e}")
        print("  请运行: pip install cqlib")
        return False

    # ==================== 步骤 2: 建立天衍云平台连接 ====================
    print(f"\n[Step 2] 建立天衍云平台连接 -> {machine_name}...")
    try:
        platform = cqlib.TianYanPlatform(
            login_key=api_key,
            machine_name=machine_name,
        )
        print(f"  [PASS] 平台连接成功，目标机器: {machine_name}")
    except Exception as e:
        print(f"  [FAIL] 平台连接失败: {e}")
        return False

    # ==================== 步骤 3: 列出所有可用量子计算机 ====================
    print("\n[Step 3] 列出所有可用量子计算机...")
    try:
        machines = platform.query_quantum_computer_list()
        if machines:
            print(f"  [PASS] 共 {len(machines)} 台机器:")
            for m in machines:
                mid, mtype, status, name = m[0], m[1], m[2], m[3]
                status_icon = "[ON]" if status == "running" else "[OFF]"
                print(f"    {status_icon} {name:20s} | ID={mid:12s} | type={mtype:10s} | status={status}")
        else:
            print("  [WARN] 机器列表为空")
    except Exception as e:
        print(f"  [FAIL] 获取机器列表失败: {e}")

    # ==================== 步骤 4: 检查天衍-176 状态 ====================
    print(f"\n[Step 4] 检查 {machine_name} 状态...")
    try:
        machines = platform.query_quantum_computer_list()
        t176_info = None
        for m in machines:
            if m[3] == machine_name:
                t176_info = m
                break

        if t176_info:
            mid, mtype, status, name = t176_info
            print(f"  [PASS] 找到 {name}: ID={mid}, type={mtype}, status={status}")
            if status == "running":
                print(f"  [INFO] 机器状态正常，可以提交任务")
            else:
                print(f"  [WARN] 机器当前状态为 '{status}'，可能无法提交任务")
        else:
            print(f"  [WARN] 未在机器列表中找到 {machine_name}")
            # 尝试查找类似名称
            similar = [m for m in machines if "176" in m[3]]
            for s in similar:
                print(f"    找到类似机器: {s[3]} (ID={s[0]}, status={s[2]})")
    except Exception as e:
        print(f"  [FAIL] 查询失败: {e}")

    # ==================== 步骤 5: 提交简单量子电路 ====================
    print(f"\n[Step 5] 向 {machine_name} 提交简单量子电路...")
    task_id = None
    try:
        qcis = "H Q0\nM Q0"
        print(f"  QCIS: {qcis}")
        result = platform.submit_experiment(
            circuit=qcis,
            name="Test_Tianyan176",
            num_shots=1024,
            is_verify=False,
        )
        if isinstance(result, list) and len(result) > 0:
            task_id = str(result[0])
        else:
            task_id = str(result)
        print(f"  [PASS] 任务已提交! task_id = {task_id}")
    except Exception as e:
        print(f"  [FAIL] 任务提交失败: {e}")

    # ==================== 步骤 6: 查询任务结果 ====================
    if task_id:
        print(f"\n[Step 6] 等待任务完成并获取结果 (最多等待 120 秒)...")
        max_wait = 120
        poll_interval = 5
        waited = 0

        while waited < max_wait:
            try:
                status_result = platform.query_experiment(task_id)
                # 尝试解析状态
                if isinstance(status_result, dict):
                    task_status = status_result.get("status", "unknown")
                elif isinstance(status_result, list) and len(status_result) > 0:
                    task_status = str(status_result)
                else:
                    task_status = str(status_result)

                print(f"  [{waited}s] 状态: {task_status}")

                # 检查是否完成
                status_lower = str(task_status).lower()
                if any(kw in status_lower for kw in ["completed", "finished", "done", "success"]):
                    print(f"  [PASS] 任务执行完成!")
                    print(f"  详细结果: {status_result}")
                    break
                elif any(kw in status_lower for kw in ["failed", "error", "cancelled"]):
                    print(f"  [FAIL] 任务执行失败: {status_result}")
                    break
                # 还在运行中
                time.sleep(poll_interval)
                waited += poll_interval
            except Exception as e:
                print(f"  [{waited}s] 查询异常: {e}")
                time.sleep(poll_interval)
                waited += poll_interval
        else:
            print(f"  [WARN] 超时 {max_wait}s，任务可能仍在执行中。可稍后手动查询 task_id={task_id}")

    # ==================== 总结 ====================
    print(f"\n{SEPARATOR}")
    print("  测试完成")
    print(SEPARATOR)
    if task_id:
        print(f"  task_id: {task_id} (可用于后续追踪)")
    return True


if __name__ == "__main__":
    success = test_tianyan176()
    sys.exit(0 if success else 1)
