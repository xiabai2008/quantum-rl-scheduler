"""
超参数搜索脚本 — 网格搜索最优 DQN 配置
Hyperparameter Search Script - Grid Search for Optimal DQN Configuration

搜索参数：
    - learning_rate: [1e-4, 5e-4, 1e-3]
    - gamma: [0.95, 0.99, 0.999]
    - epsilon_decay: [0.99, 0.995, 0.999]
    - batch_size: [32, 64, 128]
    - buffer_size: [5000, 10000, 20000]

结果输出：
    - CSV 文件：./results/hyperparameter_search.csv
    - 最佳参数组合 + 性能指标
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduler.agent import SchedulerAgent
from src.scheduler.env import QuantumSchedulingEnv


def parse_args():
    parser = argparse.ArgumentParser(description="超参数网格搜索")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20000,
        help="每个参数组合的训练步数（默认: 20000）",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="评估时的 episode 数（默认: 10）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/",
        help="结果输出目录（默认: ./results/）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )
    return parser.parse_args()


def run_search(
    args: argparse.Namespace,
    param_grid: dict[str, list],
) -> list[dict]:
    """执行网格搜索"""
    results = []
    total_combinations = np.prod([len(v) for v in param_grid.values()])

    print(f"{'=' *60}")
    print("超参数网格搜索")
    print(f"{'=' *60}")
    print(f"参数组合数: {total_combinations}")
    print(f"每组合训练步数: {args.timesteps}")
    print(f"评估 episode 数: {args.eval_episodes}")
    print(f"{'=' *60}")

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    from itertools import product

    for combination_count, params in enumerate(product(*values), start=1):
        param_dict = dict(zip(keys, params, strict=False))

        print(f"\n--- [{combination_count}/{total_combinations}] 参数组合 ---")
        for k, v in param_dict.items():
            print(f"  {k}: {v}")

        try:
            np.random.seed(args.seed)
            import torch

            torch.manual_seed(args.seed)

            env = QuantumSchedulingEnv(
                max_steps=500,
                max_qubits=287,
                seed=args.seed,
            )

            agent = SchedulerAgent(
                env=env,
                learning_rate=param_dict["learning_rate"],
                buffer_size=param_dict["buffer_size"],
                batch_size=param_dict["batch_size"],
                gamma=param_dict["gamma"],
                epsilon_decay=param_dict["epsilon_decay"],
                seed=args.seed,
                verbose=0,
            )

            start_time = time.time()
            agent.train(
                total_timesteps=args.timesteps,
                eval_freq=args.timesteps // 5,
                log_dir=None,
            )
            elapsed = time.time() - start_time

            eval_result = agent.evaluate(num_episodes=args.eval_episodes)

            result = {
                "combination": combination_count,
                "learning_rate": param_dict["learning_rate"],
                "gamma": param_dict["gamma"],
                "epsilon_decay": param_dict["epsilon_decay"],
                "batch_size": param_dict["batch_size"],
                "buffer_size": param_dict["buffer_size"],
                "avg_reward": eval_result["mean_reward"],
                "success_rate": eval_result["success_rate"],
                "avg_wait_time": eval_result.get("avg_wait_time", 0),
                "qubit_utilization": eval_result.get("qubit_utilization", 0),
                "classical_utilization": eval_result.get("classical_utilization", 0),
                "elapsed_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            }

            results.append(result)

            print(
                f"  ✓ 完成 | Reward: {result['avg_reward']:.2f} | "
                f"Success: {result['success_rate']:.2%} | "
                f"耗时: {result['elapsed_seconds']:.1f}s"
            )

        except Exception as e:
            print(f"  ✗ 失败: {e!s}")
            result = {
                "combination": combination_count,
                "learning_rate": param_dict["learning_rate"],
                "gamma": param_dict["gamma"],
                "epsilon_decay": param_dict["epsilon_decay"],
                "batch_size": param_dict["batch_size"],
                "buffer_size": param_dict["buffer_size"],
                "avg_reward": float("nan"),
                "success_rate": float("nan"),
                "avg_wait_time": float("nan"),
                "qubit_utilization": float("nan"),
                "classical_utilization": float("nan"),
                "elapsed_seconds": float("nan"),
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            results.append(result)

    return results


def save_results(results: list[dict], output_dir: str):
    """保存搜索结果"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"hyperparameter_search_{timestamp}.csv")

    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n[保存] 搜索结果 CSV: {csv_path}")

        valid_results = [
            r
            for r in results
            if not isinstance(r["avg_reward"], float) or not np.isnan(r["avg_reward"])
        ]
        if valid_results:
            best_result = max(valid_results, key=lambda r: r["avg_reward"])
            print(f"\n{'=' *60}")
            print("最佳参数组合")
            print(f"{'=' *60}")
            print(f"学习率: {best_result['learning_rate']}")
            print(f"折扣因子: {best_result['gamma']}")
            print(f"探索率衰减: {best_result['epsilon_decay']}")
            print(f"批次大小: {best_result['batch_size']}")
            print(f"缓冲区大小: {best_result['buffer_size']}")
            print(f"{'=' *60}")
            print(f"平均奖励: {best_result['avg_reward']:.2f}")
            print(f"成功率: {best_result['success_rate']:.2%}")
            print(f"平均等待时间: {best_result['avg_wait_time']:.2f}")
            print(f"量子利用率: {best_result['qubit_utilization']:.2%}")
            print(f"经典利用率: {best_result['classical_utilization']:.2%}")
            print(f"{'=' *60}")

            best_path = os.path.join(output_dir, f"best_hyperparameters_{timestamp}.json")
            with open(best_path, "w", encoding="utf-8") as f:
                json.dump(best_result, f, indent=2, ensure_ascii=False)
            print(f"\n[保存] 最佳参数 JSON: {best_path}")


def main():
    args = parse_args()

    param_grid = {
        "learning_rate": [3e-4],
        "gamma": [0.99],
        "epsilon_decay": [0.995],
        "batch_size": [64],
        "buffer_size": [10000],
    }

    results = run_search(args, param_grid)
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
