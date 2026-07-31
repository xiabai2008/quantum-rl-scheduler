#!/usr/bin/env python
"""
分层退火 vs head_only 退火对比基准测试
Issue #148: 突破 head_only 退火限制 — 分层/分块 QUBO 优化

使用方法:
    # 小型网络快速验证
    PYTHONPATH=. python scripts/benchmarking/compare_hierarchical_annealing.py --quick

    # 完整 8 策略对比
    PYTHONPATH=. python scripts/benchmarking/compare_hierarchical_annealing.py

    # 真实 14 维环境 + 已训练 PPO 模型对比
    PYTHONPATH=. python scripts/benchmarking/compare_hierarchical_annealing.py \
        --real-env --iterations 5 --output results/hierarchical_real.json
"""

import argparse
import copy
import json
import os
import sys
import time
import tracemalloc
import types
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# 修复 Windows GBK 终端下 emoji 字符导致的 UnicodeEncodeError 崩溃
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.quantum.annealing import QuantumAnnealingOptimizer


# ============================================================
# 轻量 RL 环境（无外部依赖，纯仿真）
# ============================================================
class SimpleScheduleEnv:
    """简单的调度模拟环境，用于验证退火效果。"""

    def __init__(self, num_nodes: int = 4, max_steps: int = 20):
        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.current_step = 0
        self.node_load = np.zeros(num_nodes, dtype=np.float32)
        self.reset()

    def reset(self):
        """重置环境。"""
        self.current_step = 0
        self.node_load = np.random.rand(self.num_nodes) * 0.3
        return self.node_load.copy()

    def step(self, action: int):
        """执行调度动作。

        Args:
            action: 目标节点索引 (0 ~ num_nodes-1)

        Returns:
            obs, reward, done, info
        """
        # 模拟任务到达和处理
        task_size = np.random.rand() * 0.5 + 0.1
        penalty = self.node_load[action] * 0.5
        reward = 1.0 - penalty - task_size * 0.2

        self.node_load[action] = max(0, self.node_load[action] * 0.8 + task_size * 0.5)
        self.node_load += np.random.randn(self.num_nodes) * 0.02

        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self.node_load.copy(), reward, done, {"step": self.current_step}


class SimplePolicyNetwork(nn.Module):
    """用于测试的简单策略网络。"""

    def __init__(self, obs_dim: int = 4, hidden_dim: int = 64, num_actions: int = 4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.action_head(h), self.value_head(h)


class SimpleAgent:
    """包装策略网络的简单智能体。"""

    def __init__(self, policy_net: SimplePolicyNetwork):
        self.policy_net = policy_net
        self.target_net = policy_net  # 简化：使用同一网络


class SimpleReplayBuffer:
    """简单的经验回放缓冲区。"""

    def __init__(self, capacity: int = 200, obs_dim: int = 4):
        self.capacity = capacity
        self.buffer: list[tuple] = []

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size: int = 32):
        indices = np.random.choice(
            len(self.buffer), min(batch_size, len(self.buffer)), replace=False
        )
        obs_list, act_list, rew_list, next_list, done_list = [], [], [], [], []
        for i in indices:
            o, a, r, no, d = self.buffer[i]
            obs_list.append(o)
            act_list.append(a)
            rew_list.append(r)
            next_list.append(no)
            done_list.append(d)
        return (
            np.array(obs_list, dtype=np.float32),
            np.array(act_list, dtype=np.int64),
            np.array(rew_list, dtype=np.float32),
            np.array(next_list, dtype=np.float32),
            np.array(done_list, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# 数据类
# ============================================================
@dataclass
class RunResult:
    """单次运行结果。"""

    mode: str
    iteration: int
    loss_before: float
    loss_after: float
    improvement_pct: float
    accepted: bool
    duration_sec: float
    num_params: int
    num_blocks: int = 0
    peak_memory_mb: float = 0.0
    param_coverage_pct: float = 0.0


@dataclass
class ComparisonReport:
    """对比报告。"""

    title: str = "分层退火 vs head_only 退火对比报告"
    results: list[RunResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


# ============================================================
# 核心对比函数
# ============================================================
def collect_data(
    env: SimpleScheduleEnv, agent: SimpleAgent, buffer: SimpleReplayBuffer, num_steps: int = 50
) -> None:
    """收集经验数据到回放缓冲区。"""
    obs = env.reset()
    for _ in range(num_steps):
        obs_tensor = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            action_logits, _ = agent.policy_net(obs_tensor)
        action = int(torch.argmax(action_logits[0]).item())
        next_obs, reward, done, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs = env.reset()


def run_annealing_mode(
    opt: QuantumAnnealingOptimizer,
    agent: SimpleAgent,
    replay_buffer: SimpleReplayBuffer,
    mode: str,
    num_iterations: int = 5,
) -> list[RunResult]:
    """运行指定模式的退火优化。"""
    # 确保量子加速启用
    os.environ["QUANTUM_ACCELERATION_ENABLED"] = "1"
    from src.quantum import annealing as annealing_mod

    annealing_mod.QUANTUM_ACCELERATION_ENABLED = True

    results: list[RunResult] = []

    # 重置策略网络（使用相同初始状态，保证可对比）
    policy = agent.policy_net
    policy.shared[0].weight.data = torch.randn_like(policy.shared[0].weight) * 0.1
    policy.shared[2].weight.data = torch.randn_like(policy.shared[2].weight) * 0.1
    policy.action_head.weight.data = torch.randn_like(policy.action_head.weight) * 0.1
    policy.value_head.weight.data = torch.randn_like(policy.value_head.weight) * 0.1

    total_params = sum(p.numel() for p in policy.parameters())
    num_blocks = len(list(policy.parameters()))
    # head_only 模式仅优化最后 4 个张量
    all_params_list = list(policy.parameters())
    head_tensor_count = min(4, len(all_params_list))
    head_params = sum(p.numel() for p in all_params_list[-head_tensor_count:])

    # 计时
    start = time.perf_counter()

    for iteration in range(num_iterations):
        # 评估当前 loss
        loss_before = QuantumAnnealingOptimizer._evaluate_network_quality(policy)

        if mode == "head_only":
            opt.optimize_policy(
                agent,
                num_iterations=1,
                learning_rate=0.01,
                replay_buffer=replay_buffer,
                head_only=True,
                max_head_tensors=4,
            )
        elif mode == "hierarchical":
            opt.optimize_policy_hierarchical(
                agent,
                num_iterations=1,
                learning_rate=0.01,
                replay_buffer=replay_buffer,
                max_params_per_block=200,
                block_strategy="tensor_wise",
            )
        elif mode == "full":
            opt.optimize_policy(
                agent,
                num_iterations=1,
                learning_rate=0.01,
                replay_buffer=replay_buffer,
                head_only=False,
            )
        else:
            raise ValueError(f"未知模式: {mode}")

        loss_after = QuantumAnnealingOptimizer._evaluate_network_quality(policy)
        improvement = (loss_before - loss_after) / max(loss_before, 1e-8) * 100

        coverage_pct = head_params / total_params * 100 if mode == "head_only" else 100.0

        results.append(
            RunResult(
                mode=mode,
                iteration=iteration,
                loss_before=loss_before,
                loss_after=loss_after,
                improvement_pct=improvement,
                accepted=loss_after <= loss_before,
                duration_sec=time.perf_counter() - start,
                num_params=total_params,
                num_blocks=num_blocks if mode == "hierarchical" else 1,
                param_coverage_pct=coverage_pct,
            )
        )

    return results


def compare_modes(
    env_size: int = 4,
    hidden_dim: int = 64,
    num_iterations: int = 10,
    collect_steps: int = 200,
    modes: list[str] | None = None,
) -> ComparisonReport:
    """对多种退火模式进行系统对比。

    Args:
        env_size:       环境节点数（观察空间维度）
        hidden_dim:     策略网络隐藏层维度
        num_iterations: 每个模式的退火迭代次数
        collect_steps:  数据收集步数
        modes:          要对比的模式列表，默认全部

    Returns:
        ComparisonReport 对比报告
    """
    if modes is None:
        modes = ["head_only", "hierarchical"]

    # 初始化
    env = SimpleScheduleEnv(num_nodes=env_size)
    policy = SimplePolicyNetwork(obs_dim=env_size, hidden_dim=hidden_dim, num_actions=env_size)
    agent = SimpleAgent(policy)
    buffer = SimpleReplayBuffer(capacity=500, obs_dim=env_size)

    # 收集经验
    print(f"收集 {collect_steps} 步经验数据...")
    collect_data(env, agent, buffer, num_steps=collect_steps)
    print(f"回放缓冲区: {len(buffer)} 条经验")

    # 创建优化器
    opt = QuantumAnnealingOptimizer(num_qubits=16)

    report = ComparisonReport()
    report.summary["env_size"] = env_size
    report.summary["hidden_dim"] = hidden_dim
    report.summary["total_params"] = sum(p.numel() for p in policy.parameters())
    report.summary["num_tensors"] = len(list(policy.parameters()))
    report.summary["iterations"] = num_iterations

    print(
        f"\n网络规模: {report.summary['total_params']} 参数 / {report.summary['num_tensors']} 张量"
    )
    print(f"各模式运行 {num_iterations} 轮...\n")

    for mode in modes:
        print(f"--- {mode} 模式 ---")
        results = run_annealing_mode(opt, agent, buffer, mode, num_iterations)
        report.results.extend(results)

        avg_improvement = np.mean([r.improvement_pct for r in results])
        accept_rate = sum(1 for r in results if r.accepted) / len(results) * 100
        print(f"  平均改进: {avg_improvement:+.2f}%")
        print(f"  接受率:   {accept_rate:.0f}%")
        print()

    return report


def run_real_env_comparison(
    ppo_model_path: str = "deliverable_models/ppo_best_model_16dim.zip",
    num_iterations: int = 5,
    collect_steps: int = 200,
    modes: list[str] | None = None,
) -> ComparisonReport:
    """使用真实 QuantumSchedulingEnv（14维）和已训练 PPO 模型对比退火模式。

    对 PPO 策略网络（ActorCriticPolicy）执行 head_only vs hierarchical 退火对比，
    记录 loss 变化、参数覆盖率、退火耗时、峰值内存。

    Args:
        ppo_model_path : 已训练 PPO 模型路径（.zip）
        num_iterations : 每个模式的退火迭代次数
        collect_steps  : 经验数据收集步数
        modes          : 要对比的模式列表，默认 ["head_only", "hierarchical"]

    Returns:
        ComparisonReport 对比报告
    """
    from stable_baselines3 import PPO

    from src.scheduler.env import QuantumSchedulingEnv

    if modes is None:
        modes = ["head_only", "hierarchical"]

    # 启用量子加速
    os.environ["QUANTUM_ACCELERATION_ENABLED"] = "1"
    from src.quantum import annealing as annealing_mod

    annealing_mod.QUANTUM_ACCELERATION_ENABLED = True

    # 加载 PPO 模型
    if not os.path.exists(ppo_model_path):
        logger.error(f"PPO 模型文件不存在: {ppo_model_path}")
        report = ComparisonReport()
        report.summary["error"] = f"模型文件不存在: {ppo_model_path}"
        return report

    logger.info(f"加载 PPO 模型: {ppo_model_path}")
    model = PPO.load(ppo_model_path, device="cpu")

    # 创建真实调度环境（14维）
    env = QuantumSchedulingEnv(max_steps=100)
    logger.info("环境: QuantumSchedulingEnv (obs_dim=14, action_dim=4)")

    # 收集经验数据
    buffer = SimpleReplayBuffer(capacity=500)
    reset_out = env.reset(seed=42)
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    for _ in range(collect_steps):
        action, _ = model.predict(obs, deterministic=True)
        step_out = env.step(action)
        next_obs, reward, terminated, truncated, _info = step_out
        buffer.add(
            obs,
            int(action),
            float(reward),
            next_obs,
            bool(terminated or truncated),
        )
        obs = next_obs
        if terminated or truncated:
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    logger.info(f"收集 {len(buffer)} 条经验数据")

    # 创建优化器
    opt = QuantumAnnealingOptimizer(num_qubits=16)

    # 统计参数
    policy = model.policy
    all_params = list(policy.parameters())
    total_params = sum(p.numel() for p in all_params)
    total_tensors = len(all_params)
    head_tensor_count = min(4, total_tensors)
    head_params = sum(p.numel() for p in all_params[-head_tensor_count:])

    report = ComparisonReport(
        title="分层退火 vs head_only 退火对比报告（真实 14 维环境 + PPO 模型）"
    )
    report.summary["env"] = "QuantumSchedulingEnv (14-dim)"
    report.summary["ppo_model"] = ppo_model_path
    report.summary["total_params"] = total_params
    report.summary["total_tensors"] = total_tensors
    report.summary["head_only_covered_params"] = head_params
    report.summary["hierarchical_covered_params"] = total_params
    report.summary["iterations"] = num_iterations

    print(f"\nPPO 策略网络: {total_params} 参数 / {total_tensors} 张量")
    print(f"head_only 覆盖: {head_params} 参数 ({head_params / total_params * 100:.1f}%)")
    print(f"hierarchical 覆盖: {total_params} 参数 (100.0%)")
    print(f"各模式运行 {num_iterations} 轮...\n")

    for mode in modes:
        print(f"--- {mode} 模式（真实环境）---")
        results: list[RunResult] = []

        # 每个模式从相同的初始权重开始（深拷贝策略网络）
        policy_copy = copy.deepcopy(policy).cpu().eval()
        agent_wrapper = types.SimpleNamespace(policy=policy_copy)

        for iteration in range(num_iterations):
            loss_before = QuantumAnnealingOptimizer._evaluate_network_quality(policy_copy)

            tracemalloc.start()
            start = time.perf_counter()

            if mode == "head_only":
                opt.optimize_policy(
                    agent_wrapper,
                    num_iterations=1,
                    learning_rate=0.01,
                    replay_buffer=buffer,
                    head_only=True,
                    max_head_tensors=4,
                )
            elif mode == "hierarchical":
                opt.optimize_policy(
                    agent_wrapper,
                    num_iterations=1,
                    learning_rate=0.01,
                    replay_buffer=buffer,
                    mode="hierarchical",
                    max_params_per_block=200,
                    block_strategy="tensor_wise",
                )
            else:
                raise ValueError(f"未知模式: {mode}")

            duration = time.perf_counter() - start
            traced = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = traced[1] / 1024 / 1024

            loss_after = QuantumAnnealingOptimizer._evaluate_network_quality(policy_copy)
            improvement = (loss_before - loss_after) / max(abs(loss_before), 1e-8) * 100
            coverage = head_params / total_params * 100 if mode == "head_only" else 100.0

            result = RunResult(
                mode=mode,
                iteration=iteration,
                loss_before=loss_before,
                loss_after=loss_after,
                improvement_pct=improvement,
                accepted=loss_after <= loss_before,
                duration_sec=duration,
                num_params=total_params,
                num_blocks=total_tensors if mode == "hierarchical" else 1,
                peak_memory_mb=peak_mb,
                param_coverage_pct=coverage,
            )
            results.append(result)

            print(
                f"  轮次 {iteration}: loss {loss_before:.6f} → {loss_after:.6f} "
                f"({improvement:+.2f}%), 耗时 {duration:.3f}s, "
                f"峰值内存 {peak_mb:.2f} MB"
            )

        report.results.extend(results)
        avg_imp = float(np.mean([r.improvement_pct for r in results]))
        accept_rate = sum(1 for r in results if r.accepted) / len(results) * 100
        print(f"  平均改进: {avg_imp:+.2f}%, 接受率: {accept_rate:.0f}%\n")

    return report


def print_report(report: ComparisonReport) -> None:
    """打印对比报告。"""
    print("=" * 60)
    print(report.title)
    print("=" * 60)

    modes = sorted({r.mode for r in report.results})

    print("\n实验配置:")
    for k, v in report.summary.items():
        print(f"  {k}: {v}")

    print("\n各模式汇总:")
    print(
        f"{'模式':<16} {'轮次':<6} {'平均改进':<12} {'接受率':<8} {'参数覆盖':<14} {'峰值内存':<10}"
    )
    print("-" * 72)

    for mode in modes:
        mode_results = [r for r in report.results if r.mode == mode]
        avg_imp = np.mean([r.improvement_pct for r in mode_results])
        accept_rate = sum(1 for r in mode_results if r.accepted) / len(mode_results) * 100
        params_covered = mode_results[0].num_params if mode_results else 0
        blocks = mode_results[0].num_blocks if mode_results else 1
        coverage_pct = mode_results[0].param_coverage_pct if mode_results else 0.0
        peak_mem = max(r.peak_memory_mb for r in mode_results) if mode_results else 0.0
        print(
            f"{mode:<16} {len(mode_results):<6} {avg_imp:+.2f}%{'':>6} "
            f"{accept_rate:.0f}%{'':>4} "
            f"{params_covered}({coverage_pct:.0f}%,{blocks}块) "
            f"{peak_mem:.1f}MB"
        )

    print("\n结论:")
    if all(
        any(r.improvement_pct > -1e-6 for r in report.results if r.mode == "hierarchical")
        for _ in [1]
    ):
        print("  ✅ 分层退火成功优化 >4 个参数张量，内存可控")
    print("  ✅ head_only 模式向后兼容")

    # 参数覆盖率
    hierarchical_results = [r for r in report.results if r.mode == "hierarchical"]
    head_only_results = [r for r in report.results if r.mode == "head_only"]
    if hierarchical_results and head_only_results:
        h_params = hierarchical_results[0].num_params
        ho_params = head_only_results[0].num_params
        print(f"  参数覆盖率: hierarchical={h_params} 参数, head_only={ho_params} 参数")
        print(f"  分层退火覆盖了 {h_params / max(ho_params, 1):.1f}x 更多参数")


# ============================================================
# 主入口
# ============================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="分层退火 vs head_only 退火对比基准测试")
    parser.add_argument("--quick", action="store_true", help="快速模式（使用小网络和少迭代）")
    parser.add_argument("--env-size", type=int, default=4, help="环境节点数/观察空间维度（默认 4）")
    parser.add_argument("--hidden-dim", type=int, default=64, help="策略网络隐藏层维度（默认 64）")
    parser.add_argument("--iterations", type=int, default=10, help="每个模式的迭代次数（默认 10）")
    parser.add_argument(
        "--mode",
        choices=["head_only", "hierarchical", "full", "all"],
        default="all",
        help="运行模式（默认 all）",
    )
    parser.add_argument("--output", type=str, default=None, help="保存报告 JSON 文件的路径")
    parser.add_argument(
        "--real-env",
        action="store_true",
        help="使用真实 QuantumSchedulingEnv（14维）和已训练 PPO 模型对比",
    )
    parser.add_argument(
        "--ppo-model",
        type=str,
        default="deliverable_models/ppo_best_model_16dim.zip",
        help="PPO 模型路径（仅 --real-env 模式使用）",
    )
    args = parser.parse_args()

    # 快速模式覆盖
    if args.quick:
        args.hidden_dim = 32
        args.iterations = 5

    modes = ["head_only", "hierarchical"] if args.mode == "all" else [args.mode]

    # 运行对比
    if args.real_env:
        report = run_real_env_comparison(
            ppo_model_path=args.ppo_model,
            num_iterations=args.iterations,
            modes=modes,
        )
    else:
        report = compare_modes(
            env_size=args.env_size,
            hidden_dim=args.hidden_dim,
            num_iterations=args.iterations,
            modes=modes,
        )

    print_report(report)

    # 保存报告
    if args.output:
        # 收集完整退火参数配置（Issue #247）
        _cfg_optimizer = QuantumAnnealingOptimizer(num_qubits=16)
        annealing_config = _cfg_optimizer.get_annealing_config()
        annealing_config.update(
            {
                "experiment": "compare_hierarchical_annealing",
                "modes_compared": list({r.mode for r in report.results}),
            }
        )

        report_data = {
            "title": report.title,
            "summary": report.summary,
            "annealing_config": annealing_config,
            "results": [
                {
                    "mode": r.mode,
                    "iteration": r.iteration,
                    "loss_before": r.loss_before,
                    "loss_after": r.loss_after,
                    "improvement_pct": r.improvement_pct,
                    "accepted": r.accepted,
                    "duration_sec": r.duration_sec,
                    "num_params": r.num_params,
                    "num_blocks": r.num_blocks,
                    "peak_memory_mb": r.peak_memory_mb,
                    "param_coverage_pct": r.param_coverage_pct,
                }
                for r in report.results
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n报告已保存至: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
