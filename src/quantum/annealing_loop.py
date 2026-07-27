"""
量子启发式退火异步闭环训练模块（探索性功能）

.. deprecated:: 2026-07-27
    退火已降级为探索性功能，默认关闭，不再投入开发。
    量子赋能AI主方向为真机噪声反馈优化PPO鲁棒性。
    详见 src/quantum/annealing.py 的废弃说明。

实现 "RL 训练 → 周期性触发退火优化 → 反馈权重 → 继续训练" 的全自动异步流程：
    - 训练线程通过 queue.Queue 提交退火任务，不被退火求解阻塞
    - 工作线程在后台完成 QUBO 退火、验证集评估、效果追踪
    - 优化后的权重在下一个 RL rollout 开始前回写到训练模型
    - 根据退火效果自适应调整触发频率
    - 真机退火失败时自动重试并降级为模拟退火
"""

import copy
import json
import os
import queue
import threading
import time
import types
from typing import Any

import numpy as np
from loguru import logger

from src.utils.alerts import alert_error


class AsyncAnnealingLoop:
    """
    异步量子退火闭环控制器

    以生产者-消费者模式运行：
        - 生产者：RL 训练回调（AsyncAnnealingCallback）在训练步达到触发条件时，
          将当前模型引用提交到任务队列
        - 消费者：独立工作线程从队列取出任务，复制策略网络进行退火优化，
          并在验证环境上比较退火前后的平均奖励，最后将优化权重暂存到 pending_result

    Attributes:
        optimizer          : 量子退火优化器（需实现 optimize_policy 方法）
        validation_env     : 用于评估退火效果的 Gymnasium 环境
        eval_episodes      : 每次评估的回合数
        eval_deterministic : 评估时是否使用确定性策略
        initial_interval   : 初始退火触发间隔（步数）
        min_interval       : 最小触发间隔
        max_interval       : 最大触发间隔
        improvement_threshold: 判断退火有效的奖励提升阈值
        retry_delays       : 真机失败后的重试等待时间（秒）
        log_path           : 退火效果日志保存路径（JSON）
        annealing_mode     : 退火模式 ("head_only" / "hierarchical")
    """

    # 支持的退火模式
    _SUPPORTED_ANNEALING_MODES: tuple[str, ...] = ("head_only", "hierarchical")

    def __init__(
        self,
        optimizer: Any,
        validation_env: Any,
        eval_episodes: int = 3,
        eval_deterministic: bool = True,
        initial_interval: int = 5000,
        min_interval: int = 1000,
        max_interval: int = 20000,
        improvement_threshold: float = 0.0,
        retry_delays: list[float] | None = None,
        log_path: str = "results/annealing_loop_log.json",
        queue_maxsize: int = 1,
        annealing_mode: str = "head_only",
        min_effective_reward_delta: float = 1.0,
        config: dict[str, Any] | None = None,
    ):
        """
        初始化异步退火闭环

        Args:
            optimizer           : 量子退火优化器实例
            validation_env      : 验证环境，用于计算退火前后的奖励变化
            eval_episodes       : 每次评估运行几个回合，默认 3
            eval_deterministic  : 评估是否使用确定性策略，默认 True
            initial_interval    : 初始退火触发间隔，默认 5000 步
            min_interval        : 最小触发间隔，默认 1000 步
            max_interval        : 最大触发间隔，默认 20000 步
            improvement_threshold: 奖励提升阈值，默认 0.0
            retry_delays        : 真机失败重试等待时间列表，默认 [5.0, 15.0]
            log_path            : 效果日志保存路径
            queue_maxsize       : 任务队列最大长度，默认 1（避免堆积）
            annealing_mode      : 退火模式，"head_only"（仅尾部参数，向后兼容）
                                  或 "hierarchical"（分层分块全量退火，突破 OOM 限制）。
                                  默认 "head_only" 保持向后兼容。
            min_effective_reward_delta: 介入率诊断阈值（默认 1.0）。仅当退火后奖励变化
                                  delta > 该阈值时才视为"有效介入"，计入 effective_triggers。
                                  impact_rate = effective_triggers / total_triggers 用于
                                  诊断退火是否对 RL 训练产生实质影响（Issue #194）。
            config              : 配置字典（Issue #246）。当提供时，从字典中读取闭环参数，
                                  覆盖构造函数默认值；为 None 时使用原始默认值（向后兼容）。
                                  支持的 key 包括：eval_episodes、eval_deterministic、
                                  initial_interval、min_interval、max_interval、
                                  improvement_threshold、retry_delays、log_path、
                                  queue_maxsize、annealing_mode、min_effective_reward_delta。

        Raises:
            ValueError: 当 annealing_mode 不在支持列表中时
        """
        # Issue #246: config 驱动的参数初始化
        # 当 config 提供时从字典读取参数，否则使用构造函数默认值（向后兼容）
        self._config: dict[str, Any] | None = config
        _cfg: dict[str, Any] = config or {}

        _annealing_mode = str(_cfg.get("annealing_mode", annealing_mode))
        if _annealing_mode not in self._SUPPORTED_ANNEALING_MODES:
            raise ValueError(
                f"annealing_mode 必须是 {self._SUPPORTED_ANNEALING_MODES} 之一，"
                f"得到 {_annealing_mode!r}"
            )
        self.optimizer = optimizer
        self.validation_env = validation_env
        self.eval_episodes = int(_cfg.get("eval_episodes", eval_episodes))
        self.eval_deterministic = bool(
            _cfg.get("eval_deterministic", eval_deterministic)
        )
        self.min_interval = int(_cfg.get("min_interval", min_interval))
        self.max_interval = int(_cfg.get("max_interval", max_interval))
        self.improvement_threshold = float(
            _cfg.get("improvement_threshold", improvement_threshold)
        )
        self.min_effective_reward_delta = float(
            _cfg.get("min_effective_reward_delta", min_effective_reward_delta)
        )
        _cfg_retry_delays = _cfg.get("retry_delays", retry_delays)
        self.retry_delays = (
            _cfg_retry_delays if _cfg_retry_delays is not None else [5.0, 15.0]
        )
        self.log_path = str(_cfg.get("log_path", log_path))
        self.annealing_mode = _annealing_mode

        self._current_interval = int(_cfg.get("initial_interval", initial_interval))
        self._consecutive_good = 0
        self._consecutive_bad = 0

        # 介入率统计（Issue #194）
        self._total_triggers = 0
        self._effective_triggers = 0

        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=int(_cfg.get("queue_maxsize", queue_maxsize))
        )
        self._pending_result: dict[str, Any] | None = None
        self._history: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def get_annealing_config(self) -> dict[str, Any]:
        """返回异步退火闭环的完整参数配置（Issue #247）。

        合并底层优化器参数与闭环控制参数，用于实验脚本输出
        ``annealing_config`` 字段，确保退火配置可追溯、可复现。

        Returns:
            包含闭环参数与底层优化器参数的合并字典：
            - ``annealing_mode``: 退火模式（head_only / hierarchical）
            - ``initial_interval``: 初始触发间隔
            - ``min_interval`` / ``max_interval``: 触发间隔范围
            - ``eval_episodes`` / ``eval_deterministic``: 评估参数
            - ``improvement_threshold``: 奖励提升阈值
            - ``min_effective_reward_delta``: 介入率诊断阈值
            - ``retry_delays``: 真机失败重试延迟
            - ``optimizer_config``: 底层 QuantumAnnealingOptimizer 的参数（来自其 get_annealing_config）
        """
        optimizer_config: dict[str, Any] = {}
        if hasattr(self.optimizer, "get_annealing_config"):
            optimizer_config = self.optimizer.get_annealing_config()
        return {
            "annealing_mode": self.annealing_mode,
            "initial_interval": self._current_interval,
            "min_interval": self.min_interval,
            "max_interval": self.max_interval,
            "eval_episodes": self.eval_episodes,
            "eval_deterministic": self.eval_deterministic,
            "improvement_threshold": self.improvement_threshold,
            "min_effective_reward_delta": self.min_effective_reward_delta,
            "retry_delays": list(self.retry_delays),
            "optimizer_config": optimizer_config,
        }

    def start(self) -> None:
        """启动异步退火工作线程。"""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("异步退火工作线程已启动，跳过重复启动")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        logger.info("异步退火工作线程已启动")

    def shutdown(self, wait: bool = True, timeout: float | None = 300.0) -> None:
        """
        关闭异步退火工作线程

        关闭时输出介入率总结日志（Issue #194），包含总触发数、有效触发数和介入率。

        Args:
            wait   : 是否等待工作线程结束，默认 True
            timeout: 等待超时时间（秒），默认 300 秒（覆盖一次完整退火优化）
        """
        self._stop_event.set()
        if wait and self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("异步退火工作线程未能在超时时间内结束")

        # 介入率总结日志（Issue #194）
        with self._lock:
            total = self._total_triggers
            effective = self._effective_triggers
        impact_rate = effective / total if total > 0 else 0.0
        logger.info(
            f"[退火闭环] 介入率总结: 总触发={total}, 有效触发={effective}, 介入率={impact_rate:.1%}"
        )
        logger.info("异步退火工作线程已关闭")

    def submit(self, policy: Any, step: int) -> bool:
        """
        向退火任务队列提交一个优化请求

        该方法只把策略网络快照放入队列，不做退火计算，因此不会阻塞 RL 训练。
        调用方应确保传入的 policy 是训练模型权重的独立副本（深拷贝），
        避免工作线程与训练线程竞争同一组参数。

        Args:
            policy: 策略网络快照（需实现 predict / state_dict / load_state_dict）
            step  : 当前训练步数

        Returns:
            是否成功提交；队列满时返回 False
        """
        try:
            self._queue.put_nowait({"policy": policy, "step": int(step)})
            logger.info(f"[退火闭环] 步数 {step}: 已提交退火任务到异步队列")
            return True
        except queue.Full:
            logger.warning(f"[退火闭环] 步数 {step}: 退火任务队列已满，跳过本次提交")
            return False

    def get_pending_result(self) -> dict[str, Any] | None:
        """获取并清空当前待回写的优化结果（非线程安全调用需自行保证在主线程）。"""
        with self._lock:
            result = self._pending_result
            self._pending_result = None
            return result

    def peek_pending_result(self) -> dict[str, Any] | None:
        """查看当前待回写的优化结果，但不清空。

        性能优化（Issue #220）：
            原实现在 ``self._lock`` 锁内执行 ``copy.deepcopy()``，深拷贝包含
            完整 ``state_dict``（神经网络权重）的字典可能耗时数十到数百毫秒，
            持锁期间会阻塞 ``get_pending_result()`` 和 ``_update_interval()``。

            改为在锁内只获取引用，在锁外执行深拷贝，显著降低锁持有时间。
            引用本身在 CPython 中是原子操作，锁外深拷贝期间即使有其他线程
            修改 ``_pending_result``，也只是产生两个独立的快照，不影响正确性。
        """
        with self._lock:
            ref = self._pending_result
        # 深拷贝在锁外执行，避免长时间持锁阻塞并发访问
        return copy.deepcopy(ref) if ref is not None else None

    def get_current_interval(self) -> int:
        """获取当前自适应退火触发间隔。"""
        with self._lock:
            return self._current_interval

    def get_history(self) -> list[dict[str, Any]]:
        """获取退火效果历史记录（深拷贝，避免外部修改）。"""
        with self._lock:
            return copy.deepcopy(self._history)

    def _worker_loop(self) -> None:
        """退火工作线程主循环：消费队列任务并完成优化、评估、记录。

        性能优化（Issue #220）：
            支持接收 ``PolicySnapshot`` 快照（仅含 state_dict），避免训练线程
            深拷贝整个 policy 对象。worker 线程首次收到快照时通过 deepcopy
            创建持久化 eval_policy 实例，后续仅 load_state_dict 更新权重。
        """
        # 持久化 eval_policy 实例（Issue #220）
        # 首次收到 PolicySnapshot 时通过 deepcopy(policy_ref) 创建，
        # 后续快照仅通过 load_state_dict 更新权重，避免重复深拷贝。
        eval_policy: Any = None

        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            policy_or_snapshot = task["policy"]
            step = task["step"]

            try:
                eval_policy = self._prepare_eval_policy(policy_or_snapshot, eval_policy)
            except (AttributeError, RuntimeError) as e:
                logger.error(f"[退火闭环] 步数 {step}: 准备策略网络失败 ({type(e).__name__}: {e})")
                continue

            agent_wrapper = types.SimpleNamespace(policy=eval_policy)

            try:
                baseline_evaluation = self._evaluate_policy(eval_policy)
                old_reward = baseline_evaluation["reward"]
                natural_evaluation = self._evaluate_policy(
                    eval_policy,
                    baseline_reward=old_reward,
                )
                natural_delta = natural_evaluation["counterfactual_delta"]
                optimized_wrapper = self._run_annealing_with_retries(agent_wrapper, step)
                optimized_evaluation = self._evaluate_policy(
                    optimized_wrapper.policy,
                    baseline_reward=old_reward,
                    natural_delta=natural_delta,
                )
                new_reward = optimized_evaluation["reward"]
            except Exception as e:
                # 退火与评估涉及优化器、网络推理、环境交互，异常类型无法穷举，保留宽捕获并记录日志
                logger.error(f"[退火闭环] 步数 {step}: 退火或评估失败 ({type(e).__name__}: {e})")
                alert_error("annealing", f"退火或评估失败: {type(e).__name__}: {e}", step=step)
                continue

            delta = new_reward - old_reward
            counterfactual_delta = optimized_evaluation["counterfactual_delta"]
            attribution = optimized_evaluation["attribution"]
            attribution_ratio = optimized_evaluation["attribution_ratio"]
            attribution_status = "退火无效" if attribution < 0.0 else "退火有效"
            is_effective = self._update_interval(delta)
            impact_rate = self.get_impact_rate()

            # 获取本次退火实际使用的求解器类型（Issue #226）
            solver_type = getattr(self.optimizer, "solver_type", "unknown")

            record = {
                "step": step,
                "timestamp": time.time(),
                "old_reward": old_reward,
                "new_reward": new_reward,
                "delta": delta,
                "counterfactual_delta": counterfactual_delta,
                "natural_delta": natural_delta,
                "attribution": attribution,
                "attribution_ratio": attribution_ratio,
                "attribution_status": attribution_status,
                "interval": self.get_current_interval(),
                "effective": is_effective,
                "impact_rate": impact_rate,
                "solver_type": solver_type,
            }

            with self._lock:
                self._pending_result = {
                    "step": step,
                    "state_dict": copy.deepcopy(optimized_wrapper.policy.state_dict()),
                    "delta": delta,
                    "counterfactual_delta": counterfactual_delta,
                    "natural_delta": natural_delta,
                    "attribution": attribution,
                    "attribution_ratio": attribution_ratio,
                    "attribution_status": attribution_status,
                    "timestamp": record["timestamp"],
                }
                self._history.append(record)

            self._save_log()

            logger.info(
                f"[退火闭环] 步数 {step}: 旧奖励={old_reward:.4f}, "
                f"新奖励={new_reward:.4f}, delta={delta:.4f}, "
                f"counterfactual_delta={counterfactual_delta:.4f}, "
                f"natural_delta={natural_delta:.4f}, "
                f"attribution={attribution:.4f}, "
                f"attribution_ratio={attribution_ratio:.1%}, "
                f"归因诊断={attribution_status}, "
                f"当前间隔={self.get_current_interval()}, "
                f"有效={'是' if is_effective else '否'}, "
                f"介入率={impact_rate:.1%}, "
                f"求解器={solver_type}"
            )

    def _prepare_eval_policy(
        self,
        policy_or_snapshot: Any,
        existing_eval_policy: Any,
    ) -> Any:
        """根据传入的 policy 或 PolicySnapshot 准备 eval_policy 实例（Issue #220）。

        两种模式：
        - **旧模式**（直接传 policy 对象）：``.cpu().eval()`` 后返回，与原行为一致
        - **新模式**（传 PolicySnapshot）：
            - 首次（``existing_eval_policy is None``）：通过 ``deepcopy(policy_ref)``
              创建持久化实例，``.cpu().eval()`` 后 ``load_state_dict`` 加载快照权重
            - 后续：直接 ``load_state_dict`` 更新 ``existing_eval_policy`` 权重，
              避免重复深拷贝

        Args:
            policy_or_snapshot: policy 对象或 PolicySnapshot 实例
            existing_eval_policy: worker_loop 中持久化的 eval_policy（首次为 None）

        Returns:
            准备好的 eval_policy 实例
        """
        # 检测是否为 PolicySnapshot（避免 import 循环，使用鸭子类型）
        if hasattr(policy_or_snapshot, "state_dict") and hasattr(policy_or_snapshot, "policy_ref"):
            # 新模式：PolicySnapshot
            if existing_eval_policy is None:
                # 首次：deepcopy policy_ref 创建持久化实例
                eval_policy = copy.deepcopy(policy_or_snapshot.policy_ref).cpu().eval()
            else:
                # 后续：复用持久化实例，仅更新权重
                eval_policy = existing_eval_policy.cpu().eval()
            eval_policy.load_state_dict(policy_or_snapshot.state_dict)
            return eval_policy

        # 旧模式：直接传 policy 对象（向后兼容）
        return policy_or_snapshot.cpu().eval()

    def _optimize_policy_call(self, agent_wrapper: Any) -> Any:
        """
        根据 annealing_mode 路由到对应的退火优化调用

        - "head_only": 仅优化网络尾部参数张量（向后兼容）
        - "hierarchical": 分层/分块退火，逐块 QUBO 求解，覆盖全量网络参数

        Args:
            agent_wrapper: 包装了待优化策略网络的简单对象

        Returns:
            优化后的 agent_wrapper
        """
        if self.annealing_mode == "hierarchical":
            return self.optimizer.optimize_policy(
                agent_wrapper,
                mode="hierarchical",
                max_params_per_block=200,
                block_strategy="tensor_wise",
            )
        return self.optimizer.optimize_policy(agent_wrapper, head_only=True)

    def _run_annealing_with_retries(self, agent_wrapper: Any, step: int) -> Any:
        """
        执行退火优化，并处理真机失败重试与降级

        重试策略：
            - 第一次在真机模式下失败，等待 retry_delays[0] 秒后重试
            - 第二次失败，等待 retry_delays[1] 秒后重试
            - 第三次失败，将优化器切换到仿真模式并最后尝试一次
            - 若仍失败，则抛出异常由工作线程记录

        退火模式由 self.annealing_mode 决定（head_only / hierarchical），
        两种模式共享相同的重试与降级逻辑。

        Args:
            agent_wrapper: 包装了待优化策略网络的简单对象
            step         : 当前训练步数，仅用于日志

        Returns:
            优化后的 agent_wrapper
        """
        for attempt, delay in enumerate(self.retry_delays):
            try:
                return self._optimize_policy_call(agent_wrapper)
            except Exception as e:
                # 优化器内部涉及退火与权重更新，异常类型无法穷举，保留宽捕获并记录日志
                if getattr(self.optimizer, "simulation_mode", True):
                    raise
                # Issue #229: 显式记录降级上下文
                qubo_n = getattr(self.optimizer, "num_qubits", "unknown")
                solver = getattr(self.optimizer, "_last_solver", "unknown")
                logger.warning(
                    f"[退火闭环][降级] 步数 {step}: 真机退火失败（第 {attempt + 1} 次），"
                    f"{delay}s 后重试。"
                    f"降级原因={type(e).__name__}: {e}, "
                    f"当前求解器={solver}, QUBO 比特数={qubo_n}"
                )
                time.sleep(delay)

        # 重试次数耗尽，降级为仿真退火
        try:
            # Issue #229: 首次降级时记录完整降级上下文
            prev_solver = getattr(self.optimizer, "_last_solver", "unknown")
            target_solver = "neal_sa" if getattr(self.optimizer, "use_dw", False) else "numpy_sa"
            logger.warning(
                f"[退火闭环][降级] 步数 {step}: 真机退火重试耗尽，"
                f"降级为仿真退火。"
                f"降级原因=retries_exhausted (max={len(self.retry_delays)}), "
                f"前求解器={prev_solver}, 目标求解器={target_solver}, "
                f"QUBO 比特数={getattr(self.optimizer, 'num_qubits', 'unknown')}"
            )
            self.optimizer.simulation_mode = True
            return self._optimize_policy_call(agent_wrapper)
        except Exception as e:
            # 仿真退火仍可能失败（权重更新/张量运算），保留宽捕获并记录日志
            logger.error(f"[退火闭环][降级] 步数 {step}: 仿真退火也失败 ({type(e).__name__}: {e})")
            raise

    def _evaluate_policy(
        self,
        policy: Any,
        *,
        baseline_reward: float | None = None,
        natural_delta: float = 0.0,
    ) -> dict[str, float]:
        """
        在验证环境上评估策略网络的平均回合奖励

        使用固定种子 (seed=42) 确保每次评估的环境初始化一致，
        减少评估噪声对退火效果判断的干扰。

        Args:
            policy: 策略网络（需实现 predict 方法）
            baseline_reward: 反事实比较的基准奖励；省略时以本次奖励为基准
            natural_delta: 未应用退火时重复评估得到的自然奖励变化

        Returns:
            包含平均奖励、反事实增量、自然增量、退火归因及归因占比的评估结果
        """
        episode_rewards: list[float] = []
        for ep_idx in range(self.eval_episodes):
            # 使用固定种子确保评估可复现，减少环境随机性对退火效果比较的干扰
            seed_value = 42 + ep_idx
            reset_output = self.validation_env.reset(seed=seed_value)
            if isinstance(reset_output, tuple):
                obs, _info = reset_output
            else:
                obs = reset_output

            done = False
            total_reward = 0.0
            while not done:
                action, _ = policy.predict(obs, deterministic=self.eval_deterministic)
                step_output = self.validation_env.step(action)
                obs, reward, terminated, truncated, _info = step_output
                total_reward += float(reward)
                done = bool(terminated or truncated)
            episode_rewards.append(total_reward)

        reward = float(np.mean(episode_rewards))
        baseline = reward if baseline_reward is None else float(baseline_reward)
        counterfactual_delta = reward - baseline
        attribution = counterfactual_delta - float(natural_delta)
        attribution_ratio = (
            attribution / counterfactual_delta
            if abs(counterfactual_delta) > np.finfo(float).eps
            else 0.0
        )
        return {
            "reward": reward,
            "counterfactual_delta": counterfactual_delta,
            "natural_delta": float(natural_delta),
            "attribution": attribution,
            "attribution_ratio": attribution_ratio,
        }

    def _update_interval(self, delta: float) -> bool:
        """
        根据退火效果自适应调整触发间隔并统计介入率

        介入率诊断 (Issue #194)：
            仅当 delta > min_effective_reward_delta 时才视为"有效介入"，
            计入 effective_triggers。impact_rate = effective / total 用于诊断
            退火是否对 RL 训练产生实质影响。

        间隔调整规则（保持向后兼容，基于 improvement_threshold）：
            - 连续 3 次 delta > threshold：触发间隔减半（不低于 min_interval）
            - 连续 3 次 delta < threshold：触发间隔加倍（不高于 max_interval）

        Args:
            delta: 退火后奖励 - 退火前奖励

        Returns:
            is_effective: 本次退火是否为有效介入（delta > min_effective_reward_delta）
        """
        with self._lock:
            # 介入率统计：每次调用 _update_interval 代表一次退火触发
            self._total_triggers += 1
            is_effective = delta > self.min_effective_reward_delta
            if is_effective:
                self._effective_triggers += 1

            # 间隔调整（基于 improvement_threshold，保持向后兼容）
            if delta > self.improvement_threshold:
                self._consecutive_good += 1
                self._consecutive_bad = 0
                if self._consecutive_good >= 3:
                    self._current_interval = max(self.min_interval, self._current_interval // 2)
                    self._consecutive_good = 0
                    logger.info(
                        f"[退火闭环] 连续 3 次有效，触发间隔缩短为 {self._current_interval}"
                    )
            elif delta < self.improvement_threshold:
                self._consecutive_bad += 1
                self._consecutive_good = 0
                if self._consecutive_bad >= 3:
                    self._current_interval = min(self.max_interval, self._current_interval * 2)
                    self._consecutive_bad = 0
                    logger.info(
                        f"[退火闭环] 连续 3 次无效，触发间隔延长为 {self._current_interval}"
                    )

            return is_effective

    def get_impact_rate(self) -> float:
        """
        获取当前介入率（有效触发数 / 总触发数）

        Returns:
            impact_rate: 介入率，0.0 ~ 1.0。总触发数为 0 时返回 0.0
        """
        with self._lock:
            if self._total_triggers == 0:
                return 0.0
            return self._effective_triggers / self._total_triggers

    def _save_log(self) -> None:
        """将退火效果历史保存为 JSON 日志。"""
        try:
            log_dir = os.path.dirname(self.log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with self._lock:
                history = copy.deepcopy(self._history)
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except (OSError, TypeError) as e:
            # OSError: 文件读写失败；TypeError: history 含不可 JSON 序列化的对象
            logger.error(f"[退火闭环] 保存日志失败 ({type(e).__name__}: {e})")


if __name__ == "__main__":
    # loguru 已在模块顶部导入，无需 basicConfig
    logger.info("AsyncAnnealingLoop 模块已加载，请通过 train_with_annealing_loop.py 使用")
