"""
异步量子退火闭环单元测试

覆盖：
    - 退火任务异步提交不阻塞训练
    - 退火前后奖励变化（delta）记录
    - 自适应触发间隔调整
    - 真机失败重试与降级
    - 回调在 rollout 开始前回写权重
"""

import json
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

from src.quantum.annealing_loop import AsyncAnnealingLoop
from src.scheduler.async_annealing_callback import AsyncAnnealingCallback


class FakePolicy:
    """用于测试的伪策略网络，支持 predict / state_dict / load_state_dict。"""

    def __init__(self, weight: float = 0.0):
        self.weight = float(weight)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[Any, Any | None]:
        """根据当前权重选择动作：weight>=1 时返回动作 1，否则返回动作 0。"""
        action = 1 if self.weight >= 1.0 else 0
        return np.array(action), None

    def state_dict(self) -> dict[str, Any]:
        return {"weight": self.weight}

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        self.weight = float(state_dict.get("weight", self.weight))

    def eval(self) -> "FakePolicy":
        return self

    def cpu(self) -> "FakePolicy":
        return self


class FakeModel:
    """用于测试的伪 RL 模型，仅包含 policy 属性。"""

    def __init__(self, weight: float = 0.0):
        self.policy = FakePolicy(weight)


class FakeEnv:
    """用于测试的伪 Gymnasium 环境，奖励等于动作索引。"""

    def __init__(self, max_steps: int = 3):
        self.max_steps = int(max_steps)
        self.step_count = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """重置环境。"""
        self.step_count = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """执行一步，奖励为动作值，到达 max_steps 后终止。"""
        self.step_count += 1
        reward = float(action)
        terminated = self.step_count >= self.max_steps
        return np.zeros(2, dtype=np.float32), reward, terminated, False, {}


class FakeOptimizer:
    """用于测试的伪退火优化器。"""

    def __init__(
        self,
        sleep: float = 0.0,
        fail_count: int = 0,
        weight_boost: float = 1.0,
        simulation_mode: bool = True,
    ):
        self.sleep = float(sleep)
        self.fail_count = int(fail_count)
        self.weight_boost = float(weight_boost)
        self.simulation_mode = bool(simulation_mode)

    def optimize_policy(self, agent: Any, **kwargs: Any) -> Any:
        """模拟退火优化：增加 policy.weight，支持按次数失败。"""
        if self.sleep > 0:
            time.sleep(self.sleep)
        if self.fail_count > 0:
            self.fail_count -= 1
            raise RuntimeError("真机退火失败")
        agent.policy.weight += self.weight_boost
        return agent


def test_async_submit_does_not_block():
    """验证退火任务提交不会阻塞 RL 训练主线程。"""
    optimizer = FakeOptimizer(sleep=0.5)
    env = FakeEnv()
    loop = AsyncAnnealingLoop(
        optimizer,
        env,
        initial_interval=100,
        retry_delays=[0.0, 0.0],
    )
    loop.start()

    model = FakeModel()
    t0 = time.time()
    submitted = loop.submit(model.policy, step=1)
    elapsed = time.time() - t0

    loop.shutdown()

    assert submitted is True
    assert elapsed < 0.1, f"提交操作耗时过长: {elapsed:.3f}s"


def test_effect_tracking(tmp_path):
    """验证退火完成后会记录 old/new reward 和 delta。"""
    optimizer = FakeOptimizer(weight_boost=1.0)
    env = FakeEnv()
    log_path = tmp_path / "annealing_log.json"
    loop = AsyncAnnealingLoop(
        optimizer,
        env,
        eval_episodes=2,
        initial_interval=100,
        retry_delays=[0.0, 0.0],
        log_path=str(log_path),
    )
    loop.start()

    model = FakeModel(weight=0.0)
    loop.submit(model.policy, step=10)
    loop.shutdown()

    history = loop.get_history()
    assert len(history) == 1
    record = history[0]
    assert record["step"] == 10
    assert record["old_reward"] == 0.0
    assert record["new_reward"] == 3.0
    assert record["delta"] == 3.0

    # 验证 JSON 日志已写入
    assert log_path.exists()
    with open(log_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded) == 1
    assert loaded[0]["delta"] == 3.0


def test_adaptive_interval():
    """验证自适应频率：连续 3 次有效减半，连续 3 次无效加倍。"""
    optimizer = FakeOptimizer()
    env = FakeEnv()
    loop = AsyncAnnealingLoop(
        optimizer,
        env,
        initial_interval=4000,
        min_interval=1000,
        max_interval=16000,
        improvement_threshold=0.0,
        retry_delays=[0.0, 0.0],
    )

    # 连续 3 次 delta > 0 -> 减半
    loop._update_interval(1.0)
    loop._update_interval(0.5)
    loop._update_interval(2.0)
    assert loop.get_current_interval() == 2000

    # 连续 3 次 delta < 0 -> 加倍
    loop._update_interval(-1.0)
    loop._update_interval(-0.5)
    loop._update_interval(-2.0)
    assert loop.get_current_interval() == 4000

    # 边界：不应低于 min_interval
    loop._current_interval = 1500
    loop._consecutive_good = 0
    loop._consecutive_bad = 0
    loop._update_interval(1.0)
    loop._update_interval(1.0)
    loop._update_interval(1.0)
    assert loop.get_current_interval() == 1000


def test_real_machine_fallback(tmp_path):
    """验证真机退火失败并经过两次重试后自动降级为模拟退火。"""
    optimizer = FakeOptimizer(fail_count=2, simulation_mode=False, weight_boost=1.0)
    env = FakeEnv()
    log_path = tmp_path / "fallback_log.json"
    loop = AsyncAnnealingLoop(
        optimizer,
        env,
        initial_interval=100,
        retry_delays=[0.0, 0.0],
        log_path=str(log_path),
    )
    loop.start()

    model = FakeModel(weight=0.0)
    loop.submit(model.policy, step=20)
    loop.shutdown()

    assert optimizer.simulation_mode is True, "失败 2 次重试后应降级为仿真模式"
    history = loop.get_history()
    assert len(history) == 1
    assert history[0]["delta"] == 3.0


def test_callback_triggers_submit():
    """验证回调在达到间隔时正确提交退火任务。"""
    optimizer = FakeOptimizer()
    env = FakeEnv()
    loop = AsyncAnnealingLoop(
        optimizer,
        env,
        initial_interval=10,
        retry_delays=[0.0, 0.0],
    )

    callback = AsyncAnnealingCallback(loop, verbose=0)
    callback._init_callback()
    callback.model = FakeModel()

    callback.n_calls = 10
    callback._on_step()

    assert loop._queue.qsize() == 1
    loop.shutdown()


def test_callback_writes_back_pending_weights():
    """验证回调在 rollout 开始前回写已完成的优化权重。"""
    optimizer = FakeOptimizer(weight_boost=1.0)
    env = FakeEnv()
    loop = AsyncAnnealingLoop(
        optimizer,
        env,
        initial_interval=10,
        retry_delays=[0.0, 0.0],
    )

    callback = AsyncAnnealingCallback(loop, verbose=0)
    callback._init_callback()
    callback.model = FakeModel(weight=0.0)

    loop.submit(callback.model.policy, step=10)
    loop.shutdown()

    # 此时 pending_result 应包含优化后的权重
    assert loop.peek_pending_result() is not None

    callback._on_rollout_start()
    assert callback.model.policy.weight == 1.0

    # 回写后 pending_result 应被清空
    assert loop.peek_pending_result() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
