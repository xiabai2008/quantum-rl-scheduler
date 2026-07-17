"""
自定义回调模块
Custom Callbacks Module

包含训练过程中使用的各类回调：
    - EpsilonExplorationCallback: Epsilon-Greedy 探索率衰减回调
    - AnnealingCallback: 量子退火优化 PPO 网络权重回调
    - RealMachineCallback: 真机抽样回调（向天衍云真机提交任务）

设计原则：
    - 所有回调继承自 stable_baselines3.common.callbacks.BaseCallback
    - 不依赖 agent.py（避免循环导入）
    - 通过参数注入所需依赖（optimizer / client / env）
"""

import json
import os
import random
import time
from typing import Any

from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

# ---------------------------------------------------------------------------
# 自定义回调：记录探索率衰减
# ---------------------------------------------------------------------------


class EpsilonExplorationCallback(BaseCallback):
    """
    Epsilon-Greedy 探索率回调

    在训练过程中监控并衰减探索率 epsilon：
        - 初始 epsilon = 1.0（完全随机探索）
        - 最终 epsilon = 0.05（保持少量探索）
        - 每次回调触发时：epsilon *= 0.995

    同时将 epsilon 值记录到 TensorBoard 供可视化分析。
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        decay_freq: int = 1,
        verbose: int = 0,
    ) -> None:
        """
        初始化 Epsilon 探索回调。

        Args:
            epsilon_start: 初始探索率，默认 1.0
            epsilon_end: 最终探索率下限，默认 0.05
            epsilon_decay: 衰减系数，默认 0.995
            decay_freq: 衰减频率（步数），默认 1（每步衰减）
            verbose: 日志详细程度，默认 0
        """
        super().__init__(verbose)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.decay_freq = decay_freq

    def _on_step(self) -> bool:
        """每步触发：衰减 epsilon 并记录到 TensorBoard。"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        # 记录到 TensorBoard
        self.logger.record("exploration/epsilon", self.epsilon)
        return True


# ---------------------------------------------------------------------------
# 量子退火回调
# ---------------------------------------------------------------------------


class AnnealingCallback(BaseCallback):
    """
    每 N 步用量子退火优化 PPO 网络权重的回调。

    量子退火可以加速 PPO 的策略优化，通过在退火过程中探索
    更优的权重组合来提升策略性能。

    Attributes:
        optimizer: 量子退火优化器
        interval: 退火间隔（步数）
        best_reward: 最佳奖励值
        optimized_count: 累计优化次数
        head_only: 是否仅优化网络输出头权重（避免全量参数 OOM）
    """

    def __init__(
        self,
        optimizer: Any,
        interval: int = 1000,
        verbose: int = 0,
        head_only: bool = True,
    ) -> None:
        """
        初始化退火回调。

        Args:
            optimizer: 量子退火优化器实例
            interval: 退火触发间隔（步数），默认 1000
            verbose: 日志详细程度，默认 0
            head_only: 是否仅优化输出头权重，默认 True
        """
        super().__init__(verbose)
        self.optimizer = optimizer
        self.interval = interval
        self.best_reward = -float("inf")
        self.optimized_count = 0
        self.head_only = head_only

    def _on_step(self) -> bool:
        """每步检查是否需要触发退火优化。"""
        if self.n_calls % self.interval == 0 and self.n_calls > 0:
            try:
                optimized_agent = self.optimizer.optimize_policy(
                    self.model,
                    head_only=self.head_only,
                )

                quality = 0.0
                if hasattr(self.optimizer, "_evaluate_network_quality"):
                    policy_net = self.optimizer._get_policy_net(optimized_agent)
                    if policy_net is not None:
                        loss = self.optimizer._evaluate_network_quality(policy_net)
                        quality = -loss

                if quality > self.best_reward:
                    self.best_reward = quality
                    self.optimized_count += 1

                    if self.verbose:
                        logger.info(
                            f"[退火] 步数{self.n_calls}: 优化完成 (质量={quality:.4f}, "
                            f"累计优化{self.optimized_count}次)"
                        )
            except Exception as e:
                # 量子退火优化可能抛出多种异常（dimod/neal/torch），无法精确收窄
                if self.verbose:
                    logger.warning(f"[退火] 步数{self.n_calls}: 退火跳过 ({e})")
        return True


# ---------------------------------------------------------------------------
# 真机抽样回调：训练过程中按概率向天衍云真机提交任务
# ---------------------------------------------------------------------------


class RealMachineCallback(BaseCallback):
    """每 N 步抽样 1 个任务提交真机，记录真实耗时。

    在 PPO 训练过程中，每隔 ``interval`` 步以概率 ``prob`` 从当前任务队列
    中随机抽取一个任务，构建最小 QCIS 电路并提交到天衍云真机，记录提交
    耗时与 task_id。训练结束时自动保存记录到 ``save_path``（默认
    ``results/real_times.json``）。

    若环境未绑定真机客户端（``env._real_clients`` 为空且未显式传入
    ``client``），回调自动降级为 no-op，仅打印一次告警，不影响训练流程。

    典型用法（PPO 训练时启用真机抽样）::

        env = QuantumSchedulingEnv(machine_configs=DEFAULT_MACHINE_CONFIGS)
        env.attach_real_clients(real_clients)   # 绑定 cqlib 客户端
        agent = PPOAgent(env)
        agent.train(
            total_timesteps=5000,
            real_callback_interval=1000,  # 每 1000 步抽样一次
            real_callback_prob=0.5,       # 抽样时 50% 概率提交真机
        )
        # 训练结束后 results/real_times.json 已生成

    Attributes:
        env        : 训练环境（需已 attach_real_clients）
        interval   : 抽样间隔（步数）
        prob       : 每次触发的提交概率（控制机时消耗，建议 0.01-0.05）
        client     : 显式指定的真机客户端；None 时自动取 env._real_clients 第一项
        save_path  : 真机提交记录 JSON 保存路径
        real_times : 真机提交记录列表 [{step, task_id, machine, latency_s, status, real_task_id}]
    """

    def __init__(
        self,
        env: Any,
        interval: int = 1000,
        prob: float = 0.05,
        client: Any = None,
        save_path: str = "results/real_times.json",
        shots: int = 512,
        verbose: int = 1,
    ) -> None:
        """
        初始化真机抽样回调。

        Args:
            env: 训练环境（需已 attach_real_clients）
            interval: 抽样间隔（步数），默认 1000
            prob: 每次触发的提交概率，默认 0.05
            client: 显式指定的真机客户端；None 时自动取 env._real_clients 第一项
            save_path: 真机提交记录 JSON 保存路径，默认 "results/real_times.json"
            shots: 真机任务 shots，默认 512
            verbose: 日志详细程度，默认 1
        """
        super().__init__(verbose)
        self.env = env
        self.interval = int(interval)
        self.prob = float(prob)
        self.client = client
        self.save_path = save_path
        self.shots = int(shots)
        self.real_times: list[dict[str, Any]] = []
        self._warned_no_client = False

    def _on_step(self) -> bool:
        """每步触发：达到 interval 时按 prob 概率提交真机任务。"""
        # 仅在 interval 倍数步触发；跳过第 0 步（环境尚未 reset 完成）
        if self.n_calls == 0 or self.n_calls % self.interval != 0:
            return True
        if self.prob <= 0.0:
            return True
        # 概率门控：未命中则跳过本次
        if random.random() >= self.prob:
            return True

        # 解析可用的真机客户端（显式传入优先；否则从 env._real_clients 取第一项）
        client = self.client
        machine_name = getattr(client, "machine_name", "unknown") if client else "unknown"
        if client is None:
            real_clients = getattr(self.env, "_real_clients", {}) or {}
            if not real_clients:
                if not self._warned_no_client:
                    logger.warning(
                        f"[RealCallback] env 未绑定真机客户端，真机抽样已禁用 "
                        f"(step={self.n_calls})"
                    )
                    self._warned_no_client = True
                return True
            machine_name = next(iter(real_clients.keys()))
            client = real_clients[machine_name]

        # 从环境取一个待处理任务（队列空时退化为当前任务 / None）
        task = None
        if hasattr(self.env, "get_random_pending_task"):
            try:
                task = self.env.get_random_pending_task()
            except Exception as e:
                # 防御性捕获：env 内部状态访问可能抛出多种异常，降级为无任务
                logger.debug(f"[RealCallback] 获取待处理任务失败: {e}")
                task = None

        # 构造 QCIS（Task 无 qcis 字段时用最小占位电路保证可执行）
        qcis = "H Q0\nM Q0"
        task_id_str = "synthetic"
        if task is not None:
            task_id_str = str(getattr(task, "task_id", "synthetic"))
            qcis = getattr(task, "qcis", None) or "H Q0\nM Q0"

        # 提交并计时（异常安全，失败仅记录，不中断训练）
        t0 = time.time()
        record: dict[str, Any] = {
            "step": int(self.n_calls),
            "task_id": task_id_str,
            "machine": machine_name,
            "latency_s": 0.0,
            "status": "failed",
            "real_task_id": None,
        }
        try:
            real_tid = client.submit_quantum_task(
                qcis=qcis,
                shots=self.shots,
                task_name=f"RLCallback_{task_id_str}_step{self.n_calls}",
            )
            record["latency_s"] = round(time.time() - t0, 3)
            record["real_task_id"] = str(real_tid) if real_tid else None
            record["status"] = "submitted" if real_tid else "rejected"
            if self.verbose:
                logger.info(
                    f"[RealCallback] step={self.n_calls} machine={machine_name} "
                    f"tid={real_tid} latency={record['latency_s']}s "
                    f"task={task_id_str}"
                )
        except Exception as e:
            # 真机 API 提交可能因网络/认证/服务端等多种原因失败，无法精确收窄
            record["latency_s"] = round(time.time() - t0, 3)
            record["status"] = f"error: {str(e)[:80]}"
            if self.verbose:
                logger.error(f"[RealCallback] step={self.n_calls} 提交失败: {e}")

        self.real_times.append(record)
        return True

    def _on_training_end(self) -> None:
        """训练结束时保存真机提交记录到 JSON 文件。"""
        if not self.save_path:
            return
        save_dir = os.path.dirname(self.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        try:
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(self.real_times, f, ensure_ascii=False, indent=2)
            if self.verbose:
                logger.info(
                    f"[RealCallback] 真机提交记录已保存: {self.save_path} "
                    f"(共 {len(self.real_times)} 条)"
                )
        except OSError as e:
            logger.error(f"[RealCallback] 保存记录失败: {e}")


__all__ = [
    "AnnealingCallback",
    "EpsilonExplorationCallback",
    "RealMachineCallback",
]
