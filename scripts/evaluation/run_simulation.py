"""
仿真测试脚本 — 调度策略对比实验
Simulation Benchmark for Quantum-Classical Hybrid Task Scheduling Strategies

在没有真实量子硬件的情况下，使用仿真环境对比 7 种调度策略：
    A. DQN（训练好的深度 Q 网络）
    B. FCFS（先来先服务）
    C. Random（随机分配）
    D. Quantum-Only（仅量子资源）
    E. Classical-Only（仅经典资源）
    F. Greedy（贪心调度，基于资源利用率和紧急程度）
    G. SJF（最短作业优先，基于队列长度动态调整）

评估指标：
    - 平均任务等待时间
    - 任务完成率
    - 量子比特平均利用率
    - 经典资源平均利用率
    - 平均任务执行时间

用法示例：
    python scripts/run_simulation.py --episodes 50 --model-path ./models/dqn_scheduler.zip
    python scripts/run_simulation.py --episodes 100 --tasks-per-episode 200 --output-dir ./results/
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 延迟导入：确保从项目根目录运行
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _setup_matplotlib_font():
    """配置 matplotlib 中文字体，避免方块乱码。"""
    import matplotlib

    matplotlib.use("Agg")  # 无头后端
    import matplotlib.pyplot as plt

    try:
        plt.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC",
            "WenQuanYi Micro Hei",
            "Microsoft YaHei",
            "SimHei",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    return plt


# ---------------------------------------------------------------------------
# 仿真任务生成器（泊松到达 + 类型/比特/优先级随机）
# ---------------------------------------------------------------------------


class SimulationTaskGenerator:
    """
    泊松分布任务生成器。

    按泊松过程（lambda=0.5，即平均每 2 步一个任务）生成新任务。
    任务类型：70% 量子 / 30% 经典。
    量子任务所需比特数：3-20 随机。
    优先级：均匀分布 1-5。
    """

    def __init__(
        self,
        arrival_lambda: float = 0.5,
        quantum_ratio: float = 0.7,
        qubit_range: tuple[int, int] = (3, 20),
        seed: int | None = None,
    ):
        self.arrival_lambda = arrival_lambda
        self.quantum_ratio = quantum_ratio
        self.qubit_range = qubit_range
        self.rng = np.random.default_rng(seed)
        self._task_counter = 0

    def generate_batch(self, max_batch: int = 30) -> list[dict]:
        """
        按泊松分布生成一批新任务。

        Returns:
            任务字典列表，每个包含 task_id/task_type/qubit_count/priority/wait_steps/urgency
        """
        n_new = int(self.rng.poisson(self.arrival_lambda))
        n_new = min(n_new, max_batch)
        tasks = []
        for _ in range(n_new):
            self._task_counter += 1
            is_quantum = self.rng.random() < self.quantum_ratio
            task_type = "quantum" if is_quantum else "classical"
            qubit_count = (
                int(self.rng.integers(self.qubit_range[0], self.qubit_range[1] + 1))
                if is_quantum
                else 0
            )
            priority = int(self.rng.integers(1, 6))
            urgency = float(self.rng.uniform(0.0, 1.0))
            tasks.append(
                {
                    "task_id": f"SIM{self._task_counter:05d}",
                    "task_type": task_type,
                    "qubit_count": qubit_count,
                    "priority": priority,
                    "urgency": urgency,
                    "wait_steps": 0,
                    "execution_time": 0.0,
                }
            )
        return tasks


# ---------------------------------------------------------------------------
# 仿真环境（封装 QuantumSchedulingEnv 并注入自定义任务生成）
# ---------------------------------------------------------------------------


class SimulationEnv:
    """
    仿真调度环境。

    封装 QuantumSchedulingEnv，在每个 step 后收集等待时间、资源利用率等指标。
    """

    def __init__(
        self,
        env,
        task_generator: SimulationTaskGenerator | None = None,
        seed: int | None = None,
    ):
        self.env = env
        self.task_gen = task_generator or SimulationTaskGenerator(seed=seed)
        self.seed = seed

        # 跨 episode 累计指标
        self._total_tasks_arrived: int = 0
        self._total_tasks_completed: int = 0
        self._episode_count: int = 0

        # 逐步采样
        self._wait_time_samples: list[float] = []
        self._qubit_util_samples: list[float] = []
        self._classical_util_samples: list[float] = []
        self._execution_time_samples: list[float] = []

        # 当前 episode 的调度计数（用于估算执行时间）
        self._ep_scheduled: int = 0

    def reset(self, **kwargs):
        self._ep_scheduled = 0
        return self.env.reset(**kwargs)

    def step(self, action: int):
        """执行一步并采集统计指标。"""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 采样资源利用率
        qubit_avail = info.get("qubit_availability", 0.0)
        classical_load = info.get("classical_load", 0.0)
        self._qubit_util_samples.append(1.0 - qubit_avail)
        self._classical_util_samples.append(classical_load)

        # 从环境内部任务队列采样等待时间
        queue = self.env._task_queue
        if queue:
            avg_wait = sum(t.wait_steps for t in queue) / len(queue)
            self._wait_time_samples.append(float(avg_wait))

        # 估算执行时间：每次成功调度按 1 步计算
        total_sched = info.get("total_scheduled", 0)
        if total_sched > self._ep_scheduled:
            new_completions = total_sched - self._ep_scheduled
            self._execution_time_samples.extend([1.0] * new_completions)
            self._ep_scheduled = total_sched

        return obs, reward, terminated, truncated, info

    def record_episode_stats(self, info: dict):
        """记录单个 episode 结束时的统计信息。"""
        self._episode_count += 1
        self._total_tasks_arrived += info.get("total_scheduled", 0)
        self._total_tasks_completed += (
            info.get("quantum_success", 0)
            + info.get("classical_success", 0)
            + info.get("hybrid_success", 0)
        )

    def get_summary(self) -> dict[str, float]:
        """计算并返回汇总指标。"""
        total = max(self._total_tasks_arrived, 1)
        completed = max(self._total_tasks_completed, 0)
        return {
            "avg_wait_time": round(
                float(np.mean(self._wait_time_samples)) if self._wait_time_samples else 0.0, 4
            ),
            "completion_rate": round(completed / total, 4),
            "qubit_utilization": round(
                float(np.mean(self._qubit_util_samples)) if self._qubit_util_samples else 0.0, 4
            ),
            "classical_utilization": round(
                (
                    float(np.mean(self._classical_util_samples))
                    if self._classical_util_samples
                    else 0.0
                ),
                4,
            ),
            "avg_execution_time": round(
                (
                    float(np.mean(self._execution_time_samples))
                    if self._execution_time_samples
                    else 0.0
                ),
                4,
            ),
        }


# ---------------------------------------------------------------------------
# 调度策略基类与具体实现
# ---------------------------------------------------------------------------


class BaseStrategy:
    """调度策略基类。"""

    name: str = "base"

    def select_action(self, obs: np.ndarray) -> int:
        raise NotImplementedError


class DQNStrategy(BaseStrategy):
    """策略 A：基于 SchedulerAgent 的 DQN 调度策略。"""

    name = "DQN"

    def __init__(self, agent):
        self.agent = agent

    def select_action(self, obs: np.ndarray) -> int:
        return self.agent.predict(obs, deterministic=True)


class DQNModelStrategy(BaseStrategy):
    """策略 A（SB3 原生）：直接使用 SB3 DQN 模型进行推理。"""

    name = "DQN"

    def __init__(self, model):
        self.model = model

    def select_action(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action.item())


class FCFSStrategy(BaseStrategy):
    """策略 B：先来先服务。"""

    name = "FCFS"

    def select_action(self, obs: np.ndarray) -> int:
        # FCFS：总是选择混合执行（action=2），让环境内部的优先级排序生效
        # 因为环境已经按 priority/wait_steps 排序取出队首任务
        # FCFS 策略将当前任务送到最合适的资源（混合=最高兼容性）
        return 2


class RandomStrategy(BaseStrategy):
    """策略 C：随机分配。"""

    name = "Random"

    def __init__(self, action_dim: int = 3, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.action_dim = action_dim

    def select_action(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(0, self.action_dim))


class QuantumOnlyStrategy(BaseStrategy):
    """策略 D：仅使用量子资源。"""

    name = "Quantum-Only"

    def select_action(self, obs: np.ndarray) -> int:
        return 1  # ACTION_QUANTUM


class ClassicalOnlyStrategy(BaseStrategy):
    """策略 E：仅使用经典资源。"""

    name = "Classical-Only"

    def select_action(self, obs: np.ndarray) -> int:
        return 0  # ACTION_CLASSICAL


class PPOStrategy(BaseStrategy):
    """策略 H：PPO 强化学习调度策略。"""

    name = "PPO"

    def __init__(self, model):
        self.model = model

    def select_action(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action.item())


class GreedyStrategy(BaseStrategy):
    """策略 F：贪心调度策略。

    贪心策略逻辑：
    1. 如果量子资源充足（可用率 > 30%）且任务是量子类型，优先分配到量子资源
    2. 如果经典资源负载低（< 70%），分配到经典资源
    3. 否则使用混合执行
    4. 同时考虑任务紧急程度：紧急任务优先量子资源
    """

    name = "Greedy"

    def __init__(self, qubit_threshold: float = 0.3, classical_threshold: float = 0.7):
        self.qubit_threshold = qubit_threshold
        self.classical_threshold = classical_threshold

    def select_action(self, obs: np.ndarray) -> int:
        # 解析观测向量
        qubit_availability = obs[0]  # 量子比特可用率
        classical_load = obs[4]  # 经典资源负载
        urgency = obs[7]  # 任务紧急程度

        # 贪心决策逻辑
        # 1. 紧急任务 + 量子资源充足 → 量子资源
        if urgency > 0.7 and qubit_availability > self.qubit_threshold:
            return 1  # ACTION_QUANTUM

        # 2. 量子资源充足 + 经典资源负载高 → 量子资源
        if qubit_availability > self.qubit_threshold and classical_load > self.classical_threshold:
            return 1  # ACTION_QUANTUM

        # 3. 经典资源负载低 + 量子资源紧张 → 经典资源
        if classical_load < self.classical_threshold and qubit_availability <= self.qubit_threshold:
            return 0  # ACTION_CLASSICAL

        # 4. 默认使用混合执行（最通用）
        return 2  # ACTION_HYBRID


class ShortestJobFirstStrategy(BaseStrategy):
    """策略 G：最短作业优先（SJF）。

    倾向于将任务分配到执行速度更快的资源：
    - 量子任务在量子资源上执行更快
    - 经典任务在经典资源上执行更快
    - 通用任务根据当前负载选择
    """

    name = "SJF"

    def select_action(self, obs: np.ndarray) -> int:
        qubit_availability = obs[0]
        classical_load = obs[4]
        queue_length = obs[1]

        # 队列很长时，用混合执行提高吞吐量
        if queue_length > 0.6:
            return 2  # ACTION_HYBRID

        # 量子资源充足时优先量子
        if qubit_availability > 0.5:
            return 1  # ACTION_QUANTUM

        # 经典资源空闲时用经典
        if classical_load < 0.5:
            return 0  # ACTION_CLASSICAL

        # 否则混合
        return 2  # ACTION_HYBRID


# ---------------------------------------------------------------------------
# 单策略仿真运行
# ---------------------------------------------------------------------------


def run_strategy(
    env: SimulationEnv,
    strategy: BaseStrategy,
    num_episodes: int,
    tasks_per_episode: int,
    max_steps: int = 500,
    verbose: bool = False,
) -> dict[str, float]:
    """
    使用指定策略运行仿真，返回汇总指标。

    Args:
        env: 仿真环境
        strategy: 调度策略
        num_episodes: 运行的 episode 数
        tasks_per_episode: 每 episode 的任务数目标
        max_steps: 每 episode 最大步数
        verbose: 是否打印详细日志

    Returns:
        指标字典
    """
    all_rewards = []
    total_real_submits = 0  # 跨 episode 累计真机提交数（env.reset 会清零单 episode 计数）

    for ep in range(num_episodes):
        obs, info = env.reset(seed=None)
        ep_reward = 0.0
        step = 0

        while step < max_steps:
            action = strategy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1

            if terminated or truncated:
                break

        # 本 episode 的真机提交数（reset 前累计，避免下个 episode reset 清零）
        inner_env = getattr(env, "env", env)
        ep_real = int(sum(getattr(inner_env, "_machine_real_submits", {}).values()))
        total_real_submits += ep_real

        all_rewards.append(ep_reward)
        env.record_episode_stats(info)

        if verbose and (ep + 1) % max(1, num_episodes // 10) == 0:
            print(
                f"  [{strategy.name}] Episode {ep + 1}/{num_episodes} "
                f"| reward={ep_reward:.2f} | avg_reward={np.mean(all_rewards[-10:]):.2f} "
                f"| real_submits(ep)={ep_real}"
            )

    summary = env.get_summary()
    summary["avg_reward"] = round(float(np.mean(all_rewards)), 4)
    summary["real_submits"] = total_real_submits
    return summary


# ---------------------------------------------------------------------------
# 可视化：生成对比柱状图
# ---------------------------------------------------------------------------


def plot_comparison(results: dict[str, dict[str, float]], output_path: str):
    """
    使用 matplotlib 生成策略对比柱状图，保存到 output_path。

    包含 5 个子图，分别展示 5 个评估指标。

    Args:
        results: 策略名 -> 指标字典
        output_path: 图片保存路径
    """
    plt = _setup_matplotlib_font()

    strategies = list(results.keys())
    metrics = [
        ("avg_wait_time", "平均任务等待时间 (步)"),
        ("completion_rate", "任务完成率"),
        ("qubit_utilization", "量子比特平均利用率"),
        ("classical_utilization", "经典资源平均利用率"),
        ("avg_execution_time", "平均任务执行时间 (步)"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle("调度策略对比实验", fontsize=16, fontweight="bold", y=1.02)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4", "#8BC34A"]

    for ax_idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[ax_idx]
        values = [results[s].get(metric_key, 0.0) for s in strategies]
        bars = ax.bar(
            strategies, values, color=colors[: len(strategies)], edgecolor="white", linewidth=0.5
        )

        # 在柱子上方标注数值
        for bar, val in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        ax.set_title(metric_label, fontsize=10, pad=10)
        ax.set_ylabel("值" if ax_idx == 0 else "")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[可视化] 对比图已保存至: {output_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run_simulation(
    episodes: int = 100,
    tasks_per_episode: int = 100,
    model_path: str | None = None,
    ppo_model_path: str | None = None,
    output_dir: str = "./results/",
    verbose: bool = False,
    real_prob: float = 0.0,
    real_machine: str = "tianyan_s",
):
    """
    运行完整的仿真对比实验。

    Args:
        episodes: 仿真 episode 数
        tasks_per_episode: 每个 episode 的任务数目标
        model_path: 训练好的 DQN 模型路径（.zip），为 None 则使用未训练的随机 DQN
        ppo_model_path: 训练好的 PPO 模型路径（.zip），为 None 则不包含 PPO 策略
        output_dir: 结果输出目录
        verbose: 是否打印详细日志
        real_prob: 真机抽样概率（0.0=纯仿真，>0 时量子任务以此概率提交真机）。
                   需配置 TIANYAN_API_KEY；建议 0.01-0.05 控制机时成本。
        real_machine: 真机抽样目标机器名（默认 tianyan_s，校准中自动切换备用机）
    """
    print("=" * 64)
    print("  量子-经典混合调度系统 — 仿真对比实验")
    print("=" * 64)
    print(f"  Episodes:           {episodes}")
    print(f"  Tasks/Episode:      {tasks_per_episode}")
    print(f"  DQN Model Path:     {model_path or '(无，使用随机 DQN)'}")
    print(f"  PPO Model Path:     {ppo_model_path or '(无，不包含 PPO)'}")
    print(
        f"  Real Prob:          {real_prob} (机器={real_machine})"
        if real_prob > 0
        else "  Real Prob:          0 (纯仿真)"
    )
    print(f"  Output Dir:         {output_dir}")
    print("=" * 64)

    # ---- 导入模块 ----
    try:
        from src.scheduler.env import QuantumSchedulingEnv

        print("[导入] 环境模块加载成功")
    except ImportError as e:
        print(f"[错误] 环境模块导入失败: {e}")
        print("  请确保从项目根目录运行脚本，且已安装所有依赖。")
        sys.exit(1)

    try:
        from stable_baselines3 import DQN

        print("[导入] Stable-Baselines3 DQN 加载成功")
    except ImportError as e:
        print(f"[错误] stable_baselines3 未安装: {e}")
        print("  请运行: pip install stable-baselines3")
        sys.exit(1)

    # ---- 真机抽样客户端（real_prob>0 时启用，需 TIANYAN_API_KEY）----
    real_client = None
    if real_prob > 0:
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("TIANYAN_API_KEY", "")
        if not api_key:
            print("[警告] 未设置 TIANYAN_API_KEY，真机抽样已禁用，退化为纯仿真")
            real_prob = 0.0
        else:
            try:
                from src.api.tianyan_cqlib import CqlibTianyanClient

                real_client = CqlibTianyanClient(
                    login_key=api_key,
                    machine_name=real_machine,
                    auto_retry_machine=True,  # 启用 Task 5 故障自动切换
                )
                print(f"[真机] 客户端已创建: {real_machine} (prob={real_prob}, auto_retry=True)")
            except Exception as e:
                print(f"[警告] 真机客户端创建失败 ({e})，退化为纯仿真")
                real_prob = 0.0

    # ---- 创建共享的基础环境配置 ----
    base_env_kwargs = {
        "max_steps": tasks_per_episode,
        "max_qubits": 287,
    }

    # ---- 创建策略 ----
    strategies: list[BaseStrategy] = []

    # 策略 A：DQN
    if model_path and os.path.isfile(model_path):
        print(f"[DQN] 加载已训练模型: {model_path}")
        dqn_env = QuantumSchedulingEnv(**base_env_kwargs)
        from src.scheduler.agent import SchedulerAgent

        agent = SchedulerAgent(env=dqn_env)
        agent.load(model_path)
        dqn_model = agent.model
    else:
        print("[DQN] 未提供模型路径，使用未训练的 DQN（用于演示）")
        dqn_env = QuantumSchedulingEnv(**base_env_kwargs)
        dqn_model = DQN(
            policy="MlpPolicy",
            env=dqn_env,
            learning_rate=3e-4,
            buffer_size=10000,
            batch_size=64,
            gamma=0.99,
            verbose=0,
        )
        dqn_model.learn(total_timesteps=1000)
        print("[DQN] 快速预训练完成（1000 步）")
    strategies.append(DQNModelStrategy(dqn_model))

    # 策略 B：FCFS
    strategies.append(FCFSStrategy())

    # 策略 C：随机
    strategies.append(RandomStrategy(action_dim=3, seed=42))

    # 策略 D：仅量子
    strategies.append(QuantumOnlyStrategy())

    # 策略 E：仅经典
    strategies.append(ClassicalOnlyStrategy())

    # 策略 F：贪心
    strategies.append(GreedyStrategy())

    # 策略 G：最短作业优先
    strategies.append(ShortestJobFirstStrategy())

    # 策略 H：PPO
    if ppo_model_path and os.path.isfile(ppo_model_path):
        print(f"[PPO] 加载已训练模型: {ppo_model_path}")
        from stable_baselines3 import PPO

        ppo_env = QuantumSchedulingEnv(**base_env_kwargs)
        ppo_model = PPO.load(ppo_model_path, env=ppo_env)
        strategies.append(PPOStrategy(ppo_model))

    # ---- 逐策略运行仿真 ----
    results: dict[str, dict[str, float]] = {}

    for strategy in strategies:
        print(f"\n--- 运行策略: {strategy.name} ({episodes} episodes) ---")
        start_time = time.time()

        env = QuantumSchedulingEnv(**base_env_kwargs)
        # 真机抽样：绑定客户端 + 设置抽样概率（env 内部 _route_to_machine 自动触发）
        if real_prob > 0 and real_client is not None:
            env.attach_real_clients({real_machine: real_client})
            env.real_submit_probability = float(real_prob)
        sim_env = SimulationEnv(
            env=env,
            task_generator=SimulationTaskGenerator(seed=42),
        )
        summary = run_strategy(
            env=sim_env,
            strategy=strategy,
            num_episodes=episodes,
            tasks_per_episode=tasks_per_episode,
            max_steps=tasks_per_episode,
            verbose=verbose,
        )

        elapsed = time.time() - start_time
        # real_submits 已由 run_strategy 跨 episode 累计（env.reset 每集清零，故不能直接读 env）
        real_submits = int(summary.get("real_submits", 0)) if real_prob > 0 else 0
        results[strategy.name] = summary
        summary["elapsed_seconds"] = round(elapsed, 2)

        print(f"  完成 | 耗时 {elapsed:.1f}s | 真机提交: {real_submits}")
        for k, v in summary.items():
            print(f"    {k}: {v}")

    # ---- 打印汇总表格 ----
    print("\n" + "=" * 64)
    print("  实验结果汇总")
    print("=" * 64)

    metric_names = [
        "avg_wait_time",
        "completion_rate",
        "qubit_utilization",
        "classical_utilization",
        "avg_execution_time",
        "avg_reward",
    ]
    metric_labels = [
        "平均等待时间",
        "完成率",
        "量子利用率",
        "经典利用率",
        "平均执行时间",
        "平均奖励",
    ]
    # 真机抽样启用时，额外展示各策略的真机提交数
    if real_prob > 0:
        metric_names.append("real_submits")
        metric_labels.append("真机提交")

    header = f"  {'策略':<16}" + "".join(f"{label:>12}" for label in metric_labels)
    print(header)
    print("  " + "-" * (16 + 12 * len(metric_labels)))

    for sname in results:
        row = f"  {sname:<16}"
        for mk in metric_names:
            val = results[sname].get(mk, 0.0)
            # real_submits 是整数计数，其余为浮点指标
            if isinstance(val, int):
                row += f"{val:>12d}"
            else:
                row += f"{val:>12.4f}"
        print(row)
    print("=" * 64)

    # ---- 保存结果 JSON ----
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"simulation_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[保存] 详细结果 JSON: {json_path}")

    # ---- 生成可视化图 ----
    png_path = os.path.join(output_dir, "comparison.png")
    plot_comparison(results, png_path)

    print("\n仿真对比实验完成！")
    return results


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="量子-经典混合调度系统仿真对比测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python scripts/run_simulation.py --episodes 50 --model-path ./models/dqn_scheduler.zip
  python scripts/run_simulation.py --episodes 100 --tasks-per-episode 200 --output-dir ./results/
  python scripts/run_simulation.py --episodes 20 --verbose
  python scripts/run_simulation.py --real-prob 0.05 --tasks-per-episode 200   # 8 策略对比 + 真机抽样
        """,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="仿真 episode 数（默认 100）",
    )
    parser.add_argument(
        "--tasks-per-episode",
        type=int,
        default=100,
        help="每个 episode 的任务数（默认 100）",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="训练好的 DQN 模型路径（.zip 文件）",
    )
    parser.add_argument(
        "--ppo-model-path",
        type=str,
        default=None,
        help="训练好的 PPO 模型路径（.zip 文件）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/",
        help="结果输出目录（默认 ./results/）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印详细的逐 episode 日志",
    )
    parser.add_argument(
        "--real-prob",
        type=float,
        default=0.0,
        help="真机抽样概率（0.0=纯仿真，>0 时量子任务以此概率提交真机；建议 0.01-0.05）",
    )
    parser.add_argument(
        "--real-machine",
        type=str,
        default="tianyan_s",
        help="真机抽样目标机器名（默认 tianyan_s，校准中自动切换备用机）",
    )

    args = parser.parse_args()

    run_simulation(
        episodes=args.episodes,
        tasks_per_episode=args.tasks_per_episode,
        model_path=args.model_path,
        ppo_model_path=args.ppo_model_path,
        output_dir=args.output_dir,
        verbose=args.verbose,
        real_prob=args.real_prob,
        real_machine=args.real_machine,
    )


if __name__ == "__main__":
    main()
