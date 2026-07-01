"""
一键演示脚本 — 量子RL调度系统完整展示
One-Click Demo Script

用法:
    python scripts/demo.py              # 完整演示
    python scripts/demo.py --skip-web   # 跳过 Web 界面（仅仿真）
    python scripts/demo.py --skip-train # 跳过训练（仅仿真+Web）

演示流程:
    1. PPO 快速训练（5000步热身）
    2. 8种策略仿真对比
    3. 生成对比报告
    4. 启动 Web 监控界面
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    print("\n" + "=" * 60)
    print("  量子RL调度系统 — 一键演示")
    print("  Quantum RL Scheduler — Demo")
    print("=" * 60 + "\n")


def step(title):
    print(f"\n{'─' * 60}")
    print(f"  [{title}]")
    print(f"{'─' * 60}")


def run_command(cmd, env=None):
    """运行命令并打印输出"""
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=str(PROJECT_ROOT),
        env=env or os.environ,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ⚠️ 命令返回非零: {result.returncode}")
        if result.stderr:
            print(f"  stderr: {result.stderr[:300]}")
    else:
        if result.stdout:
            # 只打印最后几行
            lines = result.stdout.strip().split("\n")
            for line in lines[-5:]:
                print(f"  {line}")
    return result.returncode == 0


def demo_train(args):
    """步骤1: PPO 快速训练"""
    step("1/4 PPO 快速训练（5000步热身）")
    cmd = "python scripts/quick_train.py"
    success = run_command(cmd)
    if success:
        print("  ✅ PPO 训练完成")
    else:
        print("  ⚠️ 训练未完全成功，但演示继续")
    return success


def demo_simulation(args):
    """步骤2: 8种策略仿真对比"""
    step("2/4 8种策略仿真对比（200任务）")
    cmd = "python scripts/run_simulation.py --mock-mode --num-tasks 200 --output-dir ./results"
    success = run_command(cmd)
    if success:
        print("  ✅ 仿真对比完成")
    return success


def demo_report(args):
    """步骤3: 生成对比报告"""
    step("3/4 生成策略对比报告")

    try:

        report_dir = os.path.join(PROJECT_ROOT, "results")
        json_files = sorted(
            [
                f
                for f in os.listdir(report_dir)
                if f.startswith("simulation_results_") and f.endswith(".json")
            ],
            reverse=True,
        )

        if json_files:
            import json

            with open(os.path.join(report_dir, json_files[0]), encoding="utf-8") as f:
                data = json.load(f)

            sorted_items = sorted(
                data.items(), key=lambda x: x[1].get("avg_reward", -9999), reverse=True
            )

            print("\n  📊 策略排名：")
            for rank, (name, metrics) in enumerate(sorted_items, 1):
                emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
                print(
                    f"  {emoji} {rank}. {name:20s} | reward={metrics['avg_reward']:8.1f} | "
                    f"wait={metrics['avg_wait_time']:6.1f}s | "
                    f"completion={metrics['completion_rate']:.0%}"
                )

            ppo_item = next((v for k, v in sorted_items if "PPO" in k.upper()), None)
            random_item = data.get("Random", {})
            if ppo_item and random_item:
                vs_random = ppo_item["avg_reward"] - random_item.get("avg_reward", 0)
                print(
                    f"\n  🏆 PPO vs Random: {vs_random:+.1f} ({vs_random /abs(random_item.get('avg_reward', 1)) *100:.1f}%)"
                )

            print("  ✅ 报告生成完成")
        else:
            print("  ⚠️ 未找到仿真结果文件")
    except Exception as e:
        print(f"  ⚠️ 报告生成失败: {e}")


def demo_web(args):
    """步骤4: 启动 Web 界面"""
    step("4/4 启动 Web 监控界面")
    print("\n  🌐 访问地址: http://localhost:8000")
    print("  📊 API 端点:")
    print("     GET  /api/ppo/stats       — PPO 排名数据")
    print("     GET  /api/ppo/comparison  — 8策略完整对比")
    print("     GET  /api/ppo/predict     — PPO 实时推理")
    print("     GET  /api/status          — 系统状态")
    print("\n  💡 按 Ctrl+C 停止服务器")
    print(f"{'─' * 60}\n")

    import uvicorn

    from src.visualization.app import app

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


def main():
    parser = argparse.ArgumentParser(description="量子RL调度系统一键演示")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练步骤")
    parser.add_argument("--skip-simulation", action="store_true", help="跳过仿真步骤")
    parser.add_argument("--skip-web", action="store_true", help="跳过 Web 界面")
    parser.add_argument("--port", type=int, default=8000, help="Web 端口（默认 8000）")
    args = parser.parse_args()

    print_banner()

    if not args.skip_train:
        demo_train(args)

    if not args.skip_simulation:
        demo_simulation(args)

    demo_report(args)

    if not args.skip_web:
        demo_web(args)
    else:
        print("\n  ✅ 演示完成（已跳过 Web 界面）\n")


if __name__ == "__main__":
    main()
