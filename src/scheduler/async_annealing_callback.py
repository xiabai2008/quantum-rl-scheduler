"""
异步量子退火训练回调

替代原有的同步 AnnealingCallback，将退火优化放到独立工作线程中执行，
使 RL 训练不被退火求解阻塞，并在每个 rollout 开始前将优化后的权重回写到模型。
"""

import copy
import logging

from stable_baselines3.common.callbacks import BaseCallback

from src.quantum.annealing_loop import AsyncAnnealingLoop

logger = logging.getLogger(__name__)


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
        loop          : 异步退火闭环控制器
        verbose       : 日志详细程度
    """

    def __init__(
        self,
        loop: AsyncAnnealingLoop,
        verbose: int = 0,
    ):
        """
        初始化异步退火回调

        Args:
            loop   : AsyncAnnealingLoop 实例
            verbose: 日志详细程度，0=静默，1=打印关键事件
        """
        super().__init__(verbose)
        self.loop = loop
        self._next_trigger_step: int | None = None

    def _init_callback(self) -> None:
        """回调初始化：启动异步退火工作线程并设置首次触发步数。"""
        self.loop.start()
        self._next_trigger_step = self.loop.get_current_interval()
        if self.verbose:
            print(
                f"[AsyncAnnealingCallback] 异步退火回调已启动，"
                f"首次触发步数={self._next_trigger_step}"
            )

    def _on_step(self) -> bool:
        """
        每步触发：到达自适应间隔时提交退火任务

        提交操作只把模型引用放入队列，耗时在毫秒级，不会阻塞训练。
        """
        if self._next_trigger_step is None:
            self._next_trigger_step = self.loop.get_current_interval()

        if self.n_calls >= self._next_trigger_step:
            # 在主线程中快速复制一份策略网络快照，再提交到异步队列
            # 这样工作线程不需要访问正在前向传播的训练模型，避免竞争
            try:
                policy_snapshot = copy.deepcopy(self.model.policy).cpu().eval()
            except Exception as e:
                logger.error(
                    f"[AsyncAnnealingCallback] 步数 {self.n_calls}: "
                    f"复制策略网络快照失败 ({type(e).__name__}: {e})"
                )
                self._next_trigger_step = self.n_calls + self.loop.get_current_interval()
                return True

            submitted = self.loop.submit(policy_snapshot, self.n_calls)
            if submitted:
                interval = self.loop.get_current_interval()
                self._next_trigger_step = self.n_calls + interval
                if self.verbose:
                    print(
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
            self.model.policy.load_state_dict(state_dict, strict=False)
            if self.verbose:
                print(
                    f"[AsyncAnnealingCallback] rollout 开始前回写退火权重 "
                    f"(step={step}, delta={delta:.4f})"
                )
        except Exception as e:
            logger.error(
                f"[AsyncAnnealingCallback] 回写退火权重失败 "
                f"(step={step}, {type(e).__name__}: {e})"
            )

    def _on_training_end(self) -> None:
        """训练结束时关闭异步退火工作线程。"""
        self.loop.shutdown(wait=True)
        if self.verbose:
            print("[AsyncAnnealingCallback] 异步退火工作线程已关闭")


if __name__ == "__main__":
    print("AsyncAnnealingCallback 模块已加载，请配合 AsyncAnnealingLoop 使用")
