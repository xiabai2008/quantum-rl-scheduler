"""
消融实验框架
Ablation Study Framework

提供系统化的消融实验工具，用于量化各核心组件对调度系统性能的贡献。
通过定义标准消融配置（5 维：算法 / 状态空间 / 奖励函数 / 多机调度 / 量子退火），
在受控环境下运行简化策略（随机或 FCFS），收集指标并生成对比报告。

设计说明：
    - run_single 使用简化策略（随机 / FCFS），不启动真实 RL 训练，
      适用于框架自检与快速对比；完整组件贡献量化需接入训练好的智能体。
    - 组件开关记录在 AblationConfig.components 中，便于复现与扩展。
    - 持久化采用 JSON 格式，保证可读与可移植。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.scheduler.env import (
    DEFAULT_MACHINE_CONFIGS,
    MAX_WAIT_STEPS,
    OBS_AVG_WAIT_TIME,
    OBS_TASK_TYPE_QUANTUM,
    QuantumSchedulingEnv,
)

# 多目标奖励包装器为可选依赖：不可用时退化为原始环境
try:
    from src.scheduler.multi_objective_env import MultiObjectiveRewardWrapper
except ImportError:  # pragma: no cover - 依赖缺失时的安全回退
    MultiObjectiveRewardWrapper = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------

# 简化运行使用的默认环境参数（小规模、快速）
_DEFAULT_ENV_PARAMS: dict[str, Any] = {
    "max_steps": 150,
    "max_qubits": 50,
    "seed": 42,
}

# 完整系统的组件开关（消融实验的"全量基线"）
_FULL_COMPONENTS: dict[str, bool] = {
    "rl": True,
    "annealing": True,
    "multi_machine": True,
    "multi_objective": True,
    "state_14dim": True,
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class AblationConfig:
    """
    消融实验配置。

    Attributes:
        name        : 配置名，如 "D1_algorithm_fcfs"
        description : 配置的人类可读描述
        components  : 各组件开关，如 {"rl": True, "annealing": False, ...}
        env_params  : 环境参数覆盖，如 {"max_steps": 200, "seed": 42}
    """

    name: str
    description: str
    components: dict[str, bool]
    env_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationResult:
    """
    消融实验结果。

    Attributes:
        config               : 对应的消融配置
        mean_reward          : 平均回合奖励
        std_reward           : 回合奖励标准差
        completion_rate      : 任务完成率（成功执行数 / 调度数）
        avg_wait_time        : 平均等待步数（回合内逐步平均）
        resource_utilization : 资源利用率（吞吐量代理：调度数 / 最大步数）
        n_episodes           : 运行回合数
        timestamp            : 结果生成时间（UTC ISO 8601）
    """

    config: AblationConfig
    mean_reward: float
    std_reward: float
    completion_rate: float
    avg_wait_time: float
    resource_utilization: float
    n_episodes: int
    timestamp: str


# ---------------------------------------------------------------------------
# 核心执行器
# ---------------------------------------------------------------------------


class AblationRunner:
    """
    消融实验执行器。

    负责定义标准消融配置、运行单个/全部配置、对比结果、生成报告与持久化。

    Args:
        output_dir : 报告与结果默认输出目录
    """

    def __init__(self, output_dir: str = "results/ablation") -> None:
        """初始化执行器，并确保输出目录存在。"""
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 配置定义
    # ------------------------------------------------------------------

    def define_configs(self) -> list[AblationConfig]:
        """
        定义标准 5 维消融配置。

        每个配置在"全量基线"基础上关闭一个组件，用于量化该组件的边际贡献：
            D1_algorithm_fcfs     : 关闭 RL，使用 FCFS 基线策略
            D2_state_simplified   : 关闭 14 维扩展状态（使用 8 维简化）
            D3_reward_single      : 关闭多目标奖励（使用单目标）
            D4_single_machine     : 关闭多机调度（单机模式）
            D5_no_annealing       : 关闭量子退火加速

        Returns:
            list[AblationConfig]: 5 个标准消融配置
        """
        configs: list[AblationConfig] = []

        # D1 — 算法消融：关闭 RL，回退到 FCFS 基线
        d1 = dict(_FULL_COMPONENTS)
        d1["rl"] = False
        configs.append(
            AblationConfig(
                name="D1_algorithm_fcfs",
                description="算法消融：关闭 RL，使用 FCFS 基线策略",
                components=d1,
                env_params={},
            )
        )

        # D2 — 状态空间消融：使用 8 维简化状态
        d2 = dict(_FULL_COMPONENTS)
        d2["state_14dim"] = False
        configs.append(
            AblationConfig(
                name="D2_state_simplified",
                description="状态空间消融：关闭 14 维扩展状态（使用 8 维简化）",
                components=d2,
                env_params={},
            )
        )

        # D3 — 奖励函数消融：关闭多目标奖励
        d3 = dict(_FULL_COMPONENTS)
        d3["multi_objective"] = False
        configs.append(
            AblationConfig(
                name="D3_reward_single",
                description="奖励函数消融：关闭多目标奖励（使用单目标）",
                components=d3,
                env_params={},
            )
        )

        # D4 — 多机调度消融：回退到单机模式
        d4 = dict(_FULL_COMPONENTS)
        d4["multi_machine"] = False
        configs.append(
            AblationConfig(
                name="D4_single_machine",
                description="多机调度消融：关闭多机协同（单机模式）",
                components=d4,
                env_params={},
            )
        )

        # D5 — 量子退火消融：关闭量子退火加速
        d5 = dict(_FULL_COMPONENTS)
        d5["annealing"] = False
        configs.append(
            AblationConfig(
                name="D5_no_annealing",
                description="量子退火消融：关闭量子退火加速",
                components=d5,
                env_params={},
            )
        )

        return configs

    # ------------------------------------------------------------------
    # 运行实验
    # ------------------------------------------------------------------

    def run_single(
        self,
        config: AblationConfig,
        n_episodes: int = 10,
    ) -> AblationResult:
        """
        运行单个消融配置，收集指标并返回结果。

        根据配置构建环境（多机 / 单机）、选择策略（RL→随机代理 / FCFS），
        运行 n_episodes 个回合后聚合指标。使用简化策略避免真实 RL 训练开销。

        Args:
            config     : 消融配置
            n_episodes : 运行回合数，默认 10

        Returns:
            AblationResult: 包含聚合指标的结果对象
        """
        # 边界保护：非正回合数返回零值结果
        if n_episodes <= 0:
            return AblationResult(
                config=config,
                mean_reward=0.0,
                std_reward=0.0,
                completion_rate=0.0,
                avg_wait_time=0.0,
                resource_utilization=0.0,
                n_episodes=0,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # 合并环境参数（默认值 + 配置覆盖）
        env_params: dict[str, Any] = dict(_DEFAULT_ENV_PARAMS)
        env_params.update(config.env_params)
        seed = int(env_params.get("seed", 42))

        # 根据组件开关构建环境
        env = self._build_env(config, env_params)
        # 从参数读取最大步数（包装器不直接暴露 max_steps 属性）
        max_steps_value = int(env_params.get("max_steps", 150))

        # 选择策略
        use_rl = bool(config.components.get("rl", True))
        rng = np.random.default_rng(seed)
        policy = self._make_random_policy(rng) if use_rl else self._fcfs_action

        # 运行多个回合并收集指标
        episode_rewards: list[float] = []
        completion_rates: list[float] = []
        avg_waits: list[float] = []
        resource_utils: list[float] = []

        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed + ep)
            total_reward = 0.0
            step_wait_sum = 0.0
            step_count = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = policy(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                # 逐步累积平均等待步数（归一化值还原为步数）
                step_wait_sum += float(obs[OBS_AVG_WAIT_TIME]) * MAX_WAIT_STEPS
                step_count += 1

            # 回合级指标
            episode_rewards.append(total_reward)
            total_scheduled = int(info.get("total_scheduled", 0))
            successes = (
                int(info.get("quantum_success", 0))
                + int(info.get("classical_success", 0))
                + int(info.get("hybrid_success", 0))
            )
            completion_rates.append(
                successes / total_scheduled if total_scheduled > 0 else 0.0
            )
            avg_waits.append(step_wait_sum / step_count if step_count > 0 else 0.0)
            resource_utils.append(
                total_scheduled / float(max_steps_value)
                if max_steps_value > 0
                else 0.0
            )

        # 关闭环境释放资源
        env.close()

        return AblationResult(
            config=config,
            mean_reward=float(np.mean(episode_rewards)),
            std_reward=float(np.std(episode_rewards)),
            completion_rate=float(np.mean(completion_rates)),
            avg_wait_time=float(np.mean(avg_waits)),
            resource_utilization=float(np.mean(resource_utils)),
            n_episodes=n_episodes,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def run_all(
        self,
        configs: list[AblationConfig] | None = None,
        n_episodes: int = 10,
    ) -> list[AblationResult]:
        """
        运行所有消融配置。

        Args:
            configs    : 待运行配置列表，None 时使用 define_configs() 的标准配置
            n_episodes : 每个配置运行的回合数

        Returns:
            list[AblationResult]: 与 configs 一一对应的结果列表
        """
        if configs is None:
            configs = self.define_configs()
        return [self.run_single(cfg, n_episodes=n_episodes) for cfg in configs]

    # ------------------------------------------------------------------
    # 对比分析
    # ------------------------------------------------------------------

    def compare(self, results: list[AblationResult]) -> dict[str, Any]:
        """
        对比所有消融结果，以最优结果为基线计算各项指标的差异。

        基线选取策略：取 mean_reward 最高的结果作为基线（代理"全量系统"），
        其余配置的 delta = 该配置值 - 基线值。

        Args:
            results: 消融结果列表

        Returns:
            dict: {
                "baseline": {name, mean_reward, completion_rate, ...},
                "deltas": {config_name: {reward_delta, reward_pct, ...}}
            }
            空列表时返回 {"baseline": None, "deltas": {}}
        """
        if not results:
            return {"baseline": None, "deltas": {}}

        baseline = max(results, key=lambda r: r.mean_reward)

        baseline_summary: dict[str, Any] = {
            "name": baseline.config.name,
            "mean_reward": baseline.mean_reward,
            "completion_rate": baseline.completion_rate,
            "avg_wait_time": baseline.avg_wait_time,
            "resource_utilization": baseline.resource_utilization,
        }

        deltas: dict[str, dict[str, float]] = {}
        for r in results:
            if r.config.name == baseline.config.name:
                continue
            reward_delta = r.mean_reward - baseline.mean_reward
            reward_pct = (
                reward_delta / abs(baseline.mean_reward) * 100.0
                if baseline.mean_reward != 0
                else 0.0
            )
            deltas[r.config.name] = {
                "reward_delta": float(reward_delta),
                "reward_pct": float(reward_pct),
                "completion_delta": float(r.completion_rate - baseline.completion_rate),
                "wait_delta": float(r.avg_wait_time - baseline.avg_wait_time),
                "utilization_delta": float(
                    r.resource_utilization - baseline.resource_utilization
                ),
            }

        return {"baseline": baseline_summary, "deltas": deltas}

    # ------------------------------------------------------------------
    # 报告生成
    # ------------------------------------------------------------------

    def generate_report(
        self,
        results: list[AblationResult],
        output_path: str = "results/ablation/report.md",
    ) -> str:
        """
        生成 Markdown 消融实验报告。

        报告包含：结果汇总表、相对基线的改进百分比表、关键结论。

        Args:
            results     : 消融结果列表
            output_path : 报告输出路径

        Returns:
            str: 生成的 Markdown 报告内容
        """
        comparison = self.compare(results)
        lines: list[str] = []
        lines.append("# 消融实验报告")
        lines.append("")
        lines.append(
            f"> 生成时间: {datetime.now(timezone.utc).isoformat()}  "
            f"| 配置数: {len(results)}"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

        # 结果汇总表
        lines.append("## 结果汇总")
        lines.append("")
        lines.append(
            "| 配置名 | 描述 | 平均奖励 | 标准差 | 完成率 | "
            "平均等待(步) | 资源利用率 | 回合数 |"
        )
        lines.append("|:--|:--|--:|--:|--:|--:|--:|--:|")
        for r in results:
            lines.append(
                f"| {r.config.name} | {r.config.description} "
                f"| {r.mean_reward:.2f} | {r.std_reward:.2f} "
                f"| {r.completion_rate:.4f} | {r.avg_wait_time:.2f} "
                f"| {r.resource_utilization:.4f} | {r.n_episodes} |"
            )
        lines.append("")

        # 相对基线的改进表
        baseline = comparison.get("baseline")
        deltas = comparison.get("deltas", {})
        lines.append("## 相对基线对比")
        lines.append("")
        if baseline is None:
            lines.append("无可用结果。")
        else:
            lines.append(f"**基线**: {baseline['name']}（平均奖励 {baseline['mean_reward']:.2f}）")
            lines.append("")
            lines.append("| 配置名 | 奖励差值 | 奖励变化% | 完成率差值 | 等待差值 | 利用率差值 |")
            lines.append("|:--|--:|--:|--:|--:|--:|")
            # 基线自身行
            lines.append(
                f"| {baseline['name']} (基线) | 0.00 | 0.00% | 0.0000 | 0.00 | 0.0000 |"
            )
            for name, d in deltas.items():
                lines.append(
                    f"| {name} | {d['reward_delta']:.2f} | {d['reward_pct']:.2f}% "
                    f"| {d['completion_delta']:.4f} | {d['wait_delta']:.2f} "
                    f"| {d['utilization_delta']:.4f} |"
                )
        lines.append("")

        # 结论
        lines.append("## 结论")
        lines.append("")
        if baseline is not None and deltas:
            # 找出奖励下降最多的配置（贡献最大的组件）
            worst = min(deltas.items(), key=lambda kv: kv[1]["reward_delta"])
            best = max(deltas.items(), key=lambda kv: kv[1]["reward_delta"])
            lines.append(
                f"- 关闭 **{worst[0]}** 造成奖励下降最多（"
                f"{worst[1]['reward_delta']:.2f}，{worst[1]['reward_pct']:.2f}%），"
                f"提示该组件对系统性能贡献最大。"
            )
            lines.append(
                f"- 关闭 **{best[0]}** 对奖励影响最小（"
                f"{best[1]['reward_delta']:.2f}，{best[1]['reward_pct']:.2f}%）。"
            )
        else:
            lines.append("- 结果不足，无法计算组件贡献度。")
        lines.append("")

        report = "\n".join(lines)

        # 写入文件
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")

        return report

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save_results(
        self,
        results: list[AblationResult],
        path: str,
    ) -> None:
        """
        将结果列表序列化为 JSON 文件。

        Args:
            results: 消融结果列表
            path   : JSON 文件路径
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(r) for r in results]
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_results(self, path: str) -> list[AblationResult]:
        """
        从 JSON 文件加载结果列表。

        Args:
            path: JSON 文件路径

        Returns:
            list[AblationResult]: 反序列化的结果列表
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        results: list[AblationResult] = []
        for item in data:
            cfg_dict = item["config"]
            config = AblationConfig(
                name=cfg_dict["name"],
                description=cfg_dict["description"],
                components=cfg_dict["components"],
                env_params=cfg_dict.get("env_params", {}),
            )
            results.append(
                AblationResult(
                    config=config,
                    mean_reward=float(item["mean_reward"]),
                    std_reward=float(item["std_reward"]),
                    completion_rate=float(item["completion_rate"]),
                    avg_wait_time=float(item["avg_wait_time"]),
                    resource_utilization=float(item["resource_utilization"]),
                    n_episodes=int(item["n_episodes"]),
                    timestamp=item["timestamp"],
                )
            )
        return results

    # ------------------------------------------------------------------
    # 私有辅助
    # ------------------------------------------------------------------

    def _build_env(
        self,
        config: AblationConfig,
        env_params: dict[str, Any],
    ) -> QuantumSchedulingEnv:
        """
        根据配置构建调度环境。

        多机调度开关决定机器配置；多目标奖励开关决定是否包装奖励。

        Args:
            config     : 消融配置
            env_params : 环境参数

        Returns:
            QuantumSchedulingEnv: 构建好的环境（可能被多目标包装器包裹）
        """
        multi_machine = bool(config.components.get("multi_machine", True))
        machine_configs = DEFAULT_MACHINE_CONFIGS if multi_machine else None

        env = QuantumSchedulingEnv(
            max_steps=int(env_params.get("max_steps", 150)),
            max_qubits=int(env_params.get("max_qubits", 50)),
            seed=int(env_params.get("seed", 42)),
            machine_configs=machine_configs,
        )

        # 多目标奖励包装（依赖可用时）
        if (
            bool(config.components.get("multi_objective", True))
            and MultiObjectiveRewardWrapper is not None
        ):
            env = MultiObjectiveRewardWrapper(env)  # type: ignore[assignment]

        return env

    @staticmethod
    def _fcfs_action(obs: np.ndarray) -> int:
        """
        FCFS 基线策略：按任务类型路由到兼容资源。

        quantum 任务 → 量子资源；classical / universal → 经典资源。

        Args:
            obs: 环境观测向量

        Returns:
            int: 动作索引（0=经典，1=量子，2=混合）
        """
        is_quantum = float(obs[OBS_TASK_TYPE_QUANTUM]) > 0.5
        if is_quantum:
            return 1  # ACTION_QUANTUM
        return 0  # ACTION_CLASSICAL（含 universal 默认走经典）

    @staticmethod
    def _make_random_policy(rng: np.random.Generator) -> Callable[[np.ndarray], int]:
        """
        构造随机策略（RL 的代理），从 {0,1,2} 均匀采样。

        Args:
            rng: 已设种子的随机数生成器

        Returns:
            Callable[[np.ndarray], int]: 接收观测、返回动作的策略函数
        """

        def _policy(_obs: np.ndarray) -> int:
            return int(rng.integers(0, 3))

        return _policy
