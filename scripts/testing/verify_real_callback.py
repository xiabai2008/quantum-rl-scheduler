"""真机闭环任务二验证：训练循环接入真机抽样。

验收标准（来自 TRAE任务清单_真机闭环.md 任务二）：
    [x] 训练 5000 步，至少 1 个真机任务被提交
    [x] 真机耗时记录保存到 results/real_times.json

用法：
    python scripts/verify_real_callback.py
"""

import json
import os
import sys
from pathlib import Path

# 必须先关闭 Mock 模式，再加载 .env
os.environ["TIANYAN_MOCK_MODE"] = "false"
os.environ["TIANYAN_MOCK_DELAY"] = "0"

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv()  # 读取 TIANYAN_API_KEY

from src.api.tianyan_cqlib import CqlibTianyanClient
from src.scheduler.agent import PPOAgent
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv


def main():
    api_key = os.getenv("TIANYAN_API_KEY", "")
    if not api_key:
        print("[FAIL] 未设置 TIANYAN_API_KEY，无法走真机验证")
        sys.exit(1)

    print("=" * 60)
    print("  任务二验收：训练循环接入真机抽样")
    print(f"  API Key 长度: {len(api_key)}")
    print("=" * 60)

    # 1) 构造多机器环境
    env = QuantumSchedulingEnv(machine_configs=DEFAULT_MACHINE_CONFIGS, seed=42)

    # 2) 创建真机客户端（启用 Task 5 的故障自动切换）
    #    用 tianyan_s 作为首选，校准中会自动切换到备用机
    real_client = CqlibTianyanClient(
        login_key=api_key,
        machine_name="tianyan_s",
        auto_retry_machine=True,
    )
    env.attach_real_clients({"tianyan_s": real_client})
    print("[Setup] 真机客户端已绑定（tianyan_s，auto_retry=True）")

    # 3) 训练 PPO 5000 步，interval=2000, prob=1.0
    #    → 触发点 step=2000, 4000 → 至少 2 次真机提交
    agent = PPOAgent(env, learning_rate=3e-4, n_steps=256, verbose=0, seed=42)
    save_path = "results/real_times.json"
    if os.path.exists(save_path):
        os.remove(save_path)

    print("[Train] 开始训练 5000 步（interval=2000, prob=1.0）...")
    agent.train(
        total_timesteps=5000,
        eval_freq=100000,  # 关闭评估回调，避免占用步数与产生额外日志
        n_eval_episodes=1,
        real_callback_interval=2000,
        real_callback_prob=1.0,
        real_callback_client=real_client,
        real_callback_save_path=save_path,
    )

    # 4) 校验验收标准
    print("\n" + "=" * 60)
    print("  验收检查")
    print("=" * 60)

    if not os.path.exists(save_path):
        print(f"[FAIL] JSON 未生成: {save_path}")
        sys.exit(1)

    with open(save_path, encoding="utf-8") as f:
        records = json.load(f)

    submitted = [r for r in records if r.get("status") == "submitted"]
    print(f"总记录数: {len(records)}")
    print(f"成功提交: {len(submitted)}")
    for r in records:
        print(
            f"  step={r['step']} machine={r['machine']} status={r['status']} "
            f"tid={r.get('real_task_id')} latency={r['latency_s']}s"
        )

    # 验收：至少 1 个真机任务被提交 + JSON 已保存
    ok_submit = len(submitted) >= 1
    ok_json = os.path.exists(save_path) and len(records) > 0

    print("\n--- 验收结论 ---")
    print(f"  [{'x' if ok_submit else ' '}] 至少 1 个真机任务被提交 (实际: {len(submitted)})")
    print(f"  [{'x' if ok_json else ' '}] 真机耗时记录保存到 {save_path} (共 {len(records)} 条)")

    if ok_submit and ok_json:
        print("\n=== 任务二验收通过 ===")
    else:
        print("\n=== 任务二验收未通过 ===")
        sys.exit(2)


if __name__ == "__main__":
    main()
