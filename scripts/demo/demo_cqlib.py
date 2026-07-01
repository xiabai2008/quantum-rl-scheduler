"""天衍云真机调度演示 — 一键跑通"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv()

from src.api.tianyan_cqlib import CqlibTianyanClient

API_KEY = os.getenv("TIANYAN_API_KEY", "")

print("=== 天衍云真机调度演示 ===")
print(f"API Key: {API_KEY[:10]}...\n")

# 1. 创建客户端
client = CqlibTianyanClient(login_key=API_KEY, machine_name="tianyan_s")

# 2. 列出所有可用机器
print("📡 可用量子计算机:")
for m in client.list_backends():
    free = "🆓" if m["type"] == "free" else "💰"
    print(f"  {free} {m['name']:15s} | {m['status']}")

# 3. 提交量子任务
import cqlib

circuit = cqlib.Circuit(1)
circuit.h(0)
circuit.measure_all()

print("\n🚀 提交任务: H|0> → |+>")
task_id = client.submit_quantum_task(circuit=circuit, shots=1024, task_name="Demo_Bell")
print(f"   任务 ID: {task_id}")

# 4. 等待结果
print("\n⏳ 等待结果...")
result = client.wait_for_task(task_id, timeout=300)
print(f"   结果: {result}")
