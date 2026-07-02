"""
量子退火异步闭环训练模块

实现 "RL 训练 → 周期性触发退火优化 → 反馈权重 → 继续训练" 的全自动异步流程：
    - 训练线程通过 queue.Queue 提交退火任务，不被退火求解阻塞
    - 工作线程在后台完成 QUBO 退火、验证集评估、效果追踪
    - 优化后的权重在下一个 RL rollout 开始前回写到训练模型
    - 根据退火效果自适应调整触发频率
    - 真机退火失败时自动重试并降级为模拟退火
"""

import copy
import json
import logging
import os
import queue
import threading
import time
import types
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


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
    """

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
        """
        self.optimizer = optimizer
        self.validation_env = validation_env
        self.eval_episodes = int(eval_episodes)
        self.eval_deterministic = bool(eval_deterministic)
        self.min_interval = int(min_interval)
        self.max_interval = int(max_interval)
        self.improvement_threshold = float(improvement_threshold)
        self.retry_delays = retry_delays if retry_delays is not None else [5.0, 15.0]
        self.log_path = str(log_path)

        self._current_interval = int(initial_interval)
        self._consecutive_good = 0
        self._consecutive_bad = 0

        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=queue_maxsize)
        self._pending_result: dict[str, Any] | None = None
        self._history: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

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

        Args:
            wait   : 是否等待工作线程结束，默认 True
            timeout: 等待超时时间（秒），默认 300 秒（覆盖一次完整退火优化）
        """
        self._stop_event.set()
        if wait and self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("异步退火工作线程未能在超时时间内结束")
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
        """查看当前待回写的优化结果，但不清空。"""
        with self._lock:
            return copy.deepcopy(self._pending_result) if self._pending_result is not None else None

    def get_current_interval(self) -> int:
        """获取当前自适应退火触发间隔。"""
        with self._lock:
            return self._current_interval

    def get_history(self) -> list[dict[str, Any]]:
        """获取退火效果历史记录（深拷贝，避免外部修改）。"""
        with self._lock:
            return copy.deepcopy(self._history)

    def _worker_loop(self) -> None:
        """退火工作线程主循环：消费队列任务并完成优化、评估、记录。"""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            eval_policy = task["policy"]
            step = task["step"]

            try:
                # 确保策略网络在 CPU 评估模式，避免影响训练设备上的张量
                eval_policy = eval_policy.cpu().eval()
            except (AttributeError, RuntimeError) as e:
                logger.error(f"[退火闭环] 步数 {step}: 准备策略网络失败 ({type(e).__name__}: {e})")
                continue

            agent_wrapper = types.SimpleNamespace(policy=eval_policy)

            try:
                old_reward = self._evaluate_policy(eval_policy)
                optimized_wrapper = self._run_annealing_with_retries(agent_wrapper, step)
                new_reward = self._evaluate_policy(optimized_wrapper.policy)
            except Exception as e:
                # 退火与评估涉及优化器、网络推理、环境交互，异常类型无法穷举，保留宽捕获并记录日志
                logger.error(f"[退火闭环] 步数 {step}: 退火或评估失败 ({type(e).__name__}: {e})")
                continue

            delta = new_reward - old_reward
            self._update_interval(delta)

            record = {
                "step": step,
                "timestamp": time.time(),
                "old_reward": old_reward,
                "new_reward": new_reward,
                "delta": delta,
                "interval": self.get_current_interval(),
            }

            with self._lock:
                self._pending_result = {
                    "step": step,
                    "state_dict": copy.deepcopy(optimized_wrapper.policy.state_dict()),
                    "delta": delta,
                    "timestamp": record["timestamp"],
                }
                self._history.append(record)

            self._save_log()

            logger.info(
                f"[退火闭环] 步数 {step}: 旧奖励={old_reward:.4f}, "
                f"新奖励={new_reward:.4f}, delta={delta:.4f}, "
                f"当前间隔={self.get_current_interval()}"
            )

    def _run_annealing_with_retries(self, agent_wrapper: Any, step: int) -> Any:
        """
        执行退火优化，并处理真机失败重试与降级

        重试策略：
            - 第一次在真机模式下失败，等待 retry_delays[0] 秒后重试
            - 第二次失败，等待 retry_delays[1] 秒后重试
            - 第三次失败，将优化器切换到仿真模式并最后尝试一次
            - 若仍失败，则抛出异常由工作线程记录

        Args:
            agent_wrapper: 包装了待优化策略网络的简单对象
            step         : 当前训练步数，仅用于日志

        Returns:
            优化后的 agent_wrapper
        """
        for attempt, delay in enumerate(self.retry_delays):
            try:
                return self.optimizer.optimize_policy(agent_wrapper, head_only=True)
            except Exception as e:
                # 优化器内部涉及退火与权重更新，异常类型无法穷举，保留宽捕获并记录日志
                if getattr(self.optimizer, "simulation_mode", True):
                    raise
                logger.warning(
                    f"[退火闭环] 步数 {step}: 真机退火失败（第 {attempt + 1} 次），"
                    f"{delay}s 后重试 ({type(e).__name__}: {e})"
                )
                time.sleep(delay)

        # 重试次数耗尽，降级为仿真退火
        try:
            logger.warning(f"[退火闭环] 步数 {step}: 真机退火重试耗尽，降级为仿真退火")
            self.optimizer.simulation_mode = True
            return self.optimizer.optimize_policy(agent_wrapper, head_only=True)
        except Exception as e:
            # 仿真退火仍可能失败（权重更新/张量运算），保留宽捕获并记录日志
            logger.error(f"[退火闭环] 步数 {step}: 仿真退火也失败 ({type(e).__name__}: {e})")
            raise

    def _evaluate_policy(self, policy: Any) -> float:
        """
        在验证环境上评估策略网络的平均回合奖励

        Args:
            policy: 策略网络（需实现 predict 方法）

        Returns:
            平均回合奖励
        """
        episode_rewards: list[float] = []
        for _ in range(self.eval_episodes):
            reset_output = self.validation_env.reset()
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

        return float(np.mean(episode_rewards))

    def _update_interval(self, delta: float) -> None:
        """
        根据退火效果自适应调整触发间隔

        规则：
            - 连续 3 次 delta > threshold：触发间隔减半（不低于 min_interval）
            - 连续 3 次 delta < threshold：触发间隔加倍（不高于 max_interval）

        Args:
            delta: 退火后奖励 - 退火前奖励
        """
        with self._lock:
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("AsyncAnnealingLoop 模块已加载，请通过 train_with_annealing_loop.py 使用")
