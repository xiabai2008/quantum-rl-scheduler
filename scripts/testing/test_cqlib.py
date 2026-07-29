"""查询天衍云真机任务结果"""

import os
import sys
import time

import cqlib

# 修复 Windows GBK 终端下 emoji 字符导致的 UnicodeEncodeError 崩溃
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

API_KEY = os.environ.get("TIANYAN_API_KEY", "")
TASK_ID = os.environ.get("TIANYAN_TASK_ID", "")

if not API_KEY:
    raise RuntimeError(
        "请设置环境变量 TIANYAN_API_KEY 后再运行此脚本，禁止硬编码API密钥。\n"
        "示例: $env:TIANYAN_API_KEY='your_key_here'; python scripts/testing/test_cqlib.py"
    )
if not TASK_ID:
    raise RuntimeError("请设置环境变量 TIANYAN_TASK_ID（要查询的任务ID）")

platform = cqlib.TianYanPlatform(login_key=API_KEY, machine_name="tianyan_s")

print("=== 查询任务结果 ===")
for i in range(30):
    result = platform.query_experiment(TASK_ID)
    status = result.get("status", "unknown") if isinstance(result, dict) else "processing"
    print(f"  [{i + 1}s] status={status}")
    if status in ("completed", "finished", "done"):
        print("\n✅ 任务完成！")
        print(f"Result: {result}")
        break
    time.sleep(5)
else:
    print("\n⏰ 超时，最后状态：", result)
