"""查询天衍云真机任务结果"""
import time

import cqlib

API_KEY = "u9ViyEDXlTzYmYkSpzdP+WZ68JavOEkH+PWuy0GwTxKsM66Y8Ud1nelj+ebKcQQUyBRazg=="
TASK_ID = "2071927047586058241"

platform = cqlib.TianYanPlatform(login_key=API_KEY, machine_name="tianyan_s")

print("=== 查询任务结果 ===")
for i in range(30):
    result = platform.query_experiment(TASK_ID)
    status = result.get("status", "unknown") if isinstance(result, dict) else "processing"
    print(f"  [{i+1}s] status={status}")
    if status in ("completed", "finished", "done"):
        print("\n✅ 任务完成！")
        print(f"Result: {result}")
        break
    time.sleep(5)
else:
    print("\n⏰ 超时，最后状态：", result)
