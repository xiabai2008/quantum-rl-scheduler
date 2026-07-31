"""
异步量子退火训练回调

替代原有的同步 AnnealingCallback，将退火优化放到独立工作线程中执行，
使 RL 训练不被退火求解阻塞，并在每个 rollout 开始前将优化后的权重回写到模型。
"""

import atexit
from typing import Any

from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from src.quantum.annealing_loop import AsyncAnnealingLoop


def _clone_tensor(value: Any) -> Any:
    """创建张量或普通值的独立副本（Issue #220）。

    对于 PyTorch 张量：调用 ``detach().clone().cpu()`` 创建独立副本，
    避免共享内存且不携带梯度信息。
    对于非张量值（如 float/int/None，常见于测试用的 FakePolicy）：
    直接返回原值（不可变类型无需复制）。

    Args:
        value: state_dict 中的值

    Returns:
        value 的独立副本（张量）或原值（非张量）
    """
    # PyTorch 张量检测：通过 hasattr 鸭子类型判断，避免硬依赖 torch
    if hasattr(value, "detach") and hasattr(value, "clone") and hasattr(value, "cpu"):
        return value.detach().clone().cpu()
    # 非张量值（float/int/None 等）：不可变类型直接返回
    return value


class PolicySnapshot:
    """策略网络权重快照（Issue #220）。

    使用 ``state_dict()`` + ``clone()`` 替代 ``copy.deepcopy``，避免复制
    计算图、optimizer state 等冗余数据，显著降低训练线程中的拷贝耗时。

    快照包含：
    - ``state_dict``：权重张量的独立副本（detach + clone + cpu）
    - ``policy_ref``：原始 policy 的弱引用，用于 worker 线程中重建实例

    worker 线程首次收到快照时，通过 ``copy.deepcopy(policy_ref)`` 创建一个
    持久化的 eval_policy 实例；后续快照仅通过 ``load_state_dict`` 更新权重，
    避免重复深拷贝整个 policy 对象。
    """

    def __init__(self, state_dict: dict[str, Any], policy_ref: Any) -> None:
        """
        Args:
            state_dict: 策略网络权重快照（已 detach + clone + cpu）
            policy_ref: 原始策略网络引用（用于 worker 线程首次重建实例）
        """
        self.state_dict = state_dict
        self.policy_ref = policy_ref


class AsyncAnnealingCallback(BaseCallback):
    """
    异步量子退火触发与权重回写回调

    工作流程：
        1. _on_step: 每步检查是否达到自适应触发间隔，达到则向 AsyncAnnealingLoop
           提交一个退火任务（仅放入队列，不阻塞训练）
        2. _on_rollout_start: 在每个 rollout 收集数据前，检查是否有已完成并暂存的
           优化权重，若有则通过 model.policy.load_state_dict 回写
        3. _on_training_end: 关闭异步退火工作线程

    Attributes:
        loop            : 异步退火闭环控制器
        verbose         : 日志详细程度
        annealing_mode  : 退火模式（透传给 loop），"head_only" / "hierarchical"
    """

    def __init__(
        self,
        loop: AsyncAnnealingLoop,
        verbose: int = 0,
        annealing_mode: str = "head_only",
    ):
        """
        初始化异步退火回调

        Args:
            loop           : AsyncAnnealingLoop 实例
            verbose        : 日志详细程度，0=静默，1=打印关键事件
            annealing_mode : 退火模式，透传给 AsyncAnnealingLoop。
                             "head_only"（默认，仅尾部参数）或
                             "hierarchical"（分层分块全量退火）。
                             在 _init_callback 中应用到 loop，启动工作线程前生效。
        """
        super().__init__(verbose)
        self.loop = loop
        self.annealing_mode = str(annealing_mode)
        self._next_trigger_step: int | None = None
        # Issue #694: 标记 atexit 回调是否已注册，避免重复注册
        self._atexit_registered: bool = False

    def _init_callback(self) -> None:
        """回调初始化：透传退火模式、启动异步退火工作线程并设置首次触发步数。"""
        # 透传退火模式到 loop（在启动工作线程前生效）
        self.loop.annealing_mode = self.annealing_mode
        self.loop.start()
        self._next_trigger_step = self.loop.get_current_interval()
        # Issue #694: 注册 atexit 回调确保 worker 线程在异常中断时被清理。
        # 训练被 Ctrl+C 或异常中断时 _on_training_end 不会被调用，
        # 通过 atexit 兜底调用 loop.shutdown 避免 worker 线程泄漏。
        if not self._atexit_registered:
            atexit.register(self.loop.shutdown)
            self._atexit_registered = True
        if self.verbose:
            logger.info(
                f"[AsyncAnnealingCallback] 异步退火回调已启动，"
                f"退火模式={self.annealing_mode}, "
                f"首次触发步数={self._next_trigger_step}"
            )

    def _on_step(self) -> bool:
        """
        每步触发：到达自适应间隔时提交退火任务

        性能优化（Issue #220）：
            原实现使用 ``copy.deepcopy(self.model.policy)`` 深拷贝整个策略网络
            （含计算图、optimizer state 等），在训练线程中同步执行，可能达到
            百毫秒级阻塞。

            改为使用 ``state_dict()`` + ``clone()`` 创建权重快照（PolicySnapshot），
            仅复制权重张量，不复制计算图等冗余数据。worker 线程首次收到快照时
            通过 ``deepcopy(policy_ref)`` 重建实例，后续仅通过 ``load_state_dict``
            更新权重，避免重复深拷贝。
        """
        if self._next_trigger_step is None:
            self._next_trigger_step = self.loop.get_current_interval()

        if self.n_calls >= self._next_trigger_step:
            # 使用 state_dict() + clone() 创建权重快照（轻量）
            # 相比 copy.deepcopy，避免了复制计算图、optimizer state 等冗余数据
            try:
                snapshot_dict = {
                    k: _clone_tensor(v) for k, v in self.model.policy.state_dict().items()
                }
                policy_snapshot = PolicySnapshot(
                    state_dict=snapshot_dict,
                    policy_ref=self.model.policy,
                )
            except Exception as e:  # noqa: BLE001
                # PyTorch state_dict/clone/cpu 转换可能抛出多种异常
                # （RuntimeError/TypeError/MemoryError 等），无法精确收窄
                logger.error(
                    f"[AsyncAnnealingCallback] 步数 {self.n_calls}: "
                    f"创建策略网络快照失败 ({type(e).__name__}: {e})"
                )
                self._next_trigger_step = self.n_calls + self.loop.get_current_interval()
                return True

            submitted = self.loop.submit(policy_snapshot, self.n_calls)
            if submitted:
                interval = self.loop.get_current_interval()
                self._next_trigger_step = self.n_calls + interval
                if self.verbose:
                    logger.info(
                        f"[AsyncAnnealingCallback] 步数 {self.n_calls}: "
                        f"已提交退火任务，下次触发={self._next_trigger_step}"
                    )
            else:
                # 队列满时，稍后再试（下一个间隔再次尝试）
                self._next_trigger_step = self.n_calls + self.loop.get_current_interval()

        return True

    def _on_rollout_start(self) -> None:
        """
        每个 rollout 开始前触发：回写已完成的优化权重

        训练在 rollout 之间自然存在同步点，此时加载权重不会与梯度更新冲突。
        """
        result = self.loop.get_pending_result()
        if result is None:
            return

        state_dict = result["state_dict"]
        step = result["step"]
        delta = result["delta"]

        try:
            # Issue #694: 使用 strict=True 替代 strict=False。
            # strict=False 会静默跳过不匹配的键，导致退火结果部分丢失且无感知。
            # strict=True 在键不匹配时抛出 RuntimeError，由下方 except 捕获并记录
            # 缺失/多余键的详细信息，便于排查退火权重结构不一致的问题。
            self.model.policy.load_state_dict(state_dict, strict=True)
            if self.verbose:
                logger.info(
                    f"[AsyncAnnealingCallback] rollout 开始前回写退火权重 "
                    f"(step={step}, delta={delta:.4f})"
                )
        except RuntimeError as e:
            # strict=True 在键不匹配或形状不一致时抛出 RuntimeError。
            # 计算缺失键与多余键并记录警告，便于诊断退火权重结构变化。
            model_keys = set(self.model.policy.state_dict().keys())
            incoming_keys = set(state_dict.keys())
            missing_keys = sorted(model_keys - incoming_keys)
            unexpected_keys = sorted(incoming_keys - model_keys)
            if missing_keys:
                logger.warning(
                    f"[AsyncAnnealingCallback] 回写退火权重缺失键 "
                    f"{len(missing_keys)} 个: {missing_keys[:10]}"
                )
            if unexpected_keys:
                logger.warning(
                    f"[AsyncAnnealingCallback] 回写退火权重多余键 "
                    f"{len(unexpected_keys)} 个: {unexpected_keys[:10]}"
                )
            logger.error(
                f"[AsyncAnnealingCallback] 回写退火权重失败 "
                f"(step={step}, 键不匹配或形状不一致: {type(e).__name__}: {e})"
            )
        except Exception as e:  # noqa: BLE001
            # 其他异常（如 CPU-GPU 不匹配等）仍需捕获，避免中断训练
            logger.error(
                f"[AsyncAnnealingCallback] 回写退火权重失败 (step={step}, {type(e).__name__}: {e})"
            )

    def _on_training_end(self) -> None:
        """训练结束时关闭异步退火工作线程。"""
        self.loop.shutdown(wait=True)
        if self.verbose:
            logger.info("[AsyncAnnealingCallback] 异步退火工作线程已关闭")


if __name__ == "__main__":
    logger.info("AsyncAnnealingCallback 模块已加载，请配合 AsyncAnnealingLoop 使用")
