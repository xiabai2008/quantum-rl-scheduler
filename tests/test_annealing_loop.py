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
import types
from typing import Any, Optional
from unittest.mock import patch

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
        simulation_fail: bool = False,
    ):
        self.sleep = float(sleep)
        self.fail_count = int(fail_count)
        self.weight_boost = float(weight_boost)
        self.simulation_mode = bool(simulation_mode)
        self.simulation_fail = bool(simulation_fail)
        self.last_kwargs: dict[str, Any] = {}
        self.solver_type: str = "numpy_sa"

    def optimize_policy(self, agent: Any, **kwargs: Any) -> Any:
        """模拟退火优化：增加 policy.weight，支持按次数/模式失败。"""
        self.last_kwargs = dict(kwargs)
        if self.sleep > 0:
            time.sleep(self.sleep)
        if self.simulation_mode:
            if self.simulation_fail:
                raise RuntimeError("仿真退火失败")
        else:
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


def test_attribution_metrics_in_evaluation_history_and_log(tmp_path):
    """成对评估应在返回值、历史和 JSON 日志中输出退火归因。"""
    optimizer = FakeOptimizer(weight_boost=1.0)
    env = FakeEnv()
    log_path = tmp_path / "attribution_log.json"
    loop = AsyncAnnealingLoop(
        optimizer,
        env,
        eval_episodes=2,
        retry_delays=[0.0, 0.0],
        log_path=str(log_path),
    )

    evaluation = loop._evaluate_policy(
        FakePolicy(weight=1.0),
        baseline_reward=0.0,
        natural_delta=0.5,
    )
    assert evaluation == {
        "reward": 3.0,
        "counterfactual_delta": 3.0,
        "natural_delta": 0.5,
        "attribution": 2.5,
        "attribution_ratio": pytest.approx(2.5 / 3.0),
    }

    loop.start()
    loop.submit(FakePolicy(weight=0.0), step=10)
    loop.shutdown()

    record = loop.get_history()[0]
    assert record["counterfactual_delta"] == 3.0
    assert record["natural_delta"] == 0.0
    assert record["attribution"] == 3.0
    assert record["attribution_ratio"] == 1.0
    assert record["attribution_status"] == "退火有效"

    with open(log_path, encoding="utf-8") as f:
        logged = json.load(f)[0]
    assert logged["attribution"] == 3.0
    assert logged["attribution_ratio"] == 1.0


def test_negative_attribution_is_marked_ineffective(tmp_path):
    """归因度量为负时，诊断结果必须明确标记“退火无效”并随结果回写。"""
    optimizer = FakeOptimizer(weight_boost=-1.0)
    loop = AsyncAnnealingLoop(
        optimizer,
        FakeEnv(),
        eval_episodes=2,
        retry_delays=[0.0, 0.0],
        log_path=str(tmp_path / "negative_attribution.json"),
    )
    loop.start()
    loop.submit(FakePolicy(weight=1.0), step=20)
    loop.shutdown()

    record = loop.get_history()[0]
    assert record["attribution"] == -3.0
    assert record["attribution_ratio"] == 1.0
    assert record["attribution_status"] == "退火无效"

    pending = loop.peek_pending_result()
    assert pending is not None
    assert pending["attribution"] == -3.0
    assert pending["attribution_status"] == "退火无效"


def test_attribution_ratio_is_zero_when_counterfactual_delta_is_zero():
    """反事实增量为零时归因占比应为零，避免除零或无穷值。"""
    loop = AsyncAnnealingLoop(FakeOptimizer(), FakeEnv(), retry_delays=[0.0, 0.0])
    evaluation = loop._evaluate_policy(
        FakePolicy(weight=0.0),
        baseline_reward=0.0,
        natural_delta=1.0,
    )
    assert evaluation["counterfactual_delta"] == 0.0
    assert evaluation["attribution"] == -1.0
    assert evaluation["attribution_ratio"] == 0.0


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
    """验证回调在达到间隔时正确提交退火任务。

    用 mock 替代直接读队列大小，避免与后台工作线程消费队列的时序竞争
    (Issue #65: Python 3.10 CI 上 qsize()==0 而非 1 的 flaky 失败)。
    """
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
    # 用 mock 替换 loop.submit，验证回调确实调用了提交，且不依赖
    # 后台工作线程是否已消费队列的时序
    with patch.object(loop, "submit", return_value=True) as mock_submit:
        callback._on_step()
        assert mock_submit.call_count == 1
        # 校验提交参数：policy 快照 + 当前步数
        _submitted_policy, submitted_step = mock_submit.call_args.args
        assert submitted_step == 10

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


# ============================================================
# Issue #148: 分层退火模式（annealing_mode）路由测试
# ============================================================
def test_annealing_mode_default_is_head_only():
    """验证 annealing_mode 默认值为 'head_only'，保持向后兼容。"""
    optimizer = FakeOptimizer()
    env = FakeEnv()
    loop = AsyncAnnealingLoop(optimizer, env, retry_delays=[0.0, 0.0])
    assert loop.annealing_mode == "head_only"


def test_annealing_mode_invalid_raises_value_error():
    """验证传入无效的 annealing_mode 会抛出 ValueError。"""
    optimizer = FakeOptimizer()
    env = FakeEnv()
    with pytest.raises(ValueError, match="annealing_mode"):
        AsyncAnnealingLoop(optimizer, env, annealing_mode="invalid_mode")


def test_annealing_mode_head_only_routing():
    """验证 head_only 模式调用 optimize_policy 时传 head_only=True。"""
    optimizer = FakeOptimizer(weight_boost=1.0)
    env = FakeEnv()
    loop = AsyncAnnealingLoop(optimizer, env, initial_interval=100, retry_delays=[0.0, 0.0])
    loop.start()

    model = FakeModel(weight=0.0)
    loop.submit(model.policy, step=10)
    loop.shutdown()

    # head_only 模式应传 head_only=True，不传 mode
    assert optimizer.last_kwargs.get("head_only") is True
    assert "mode" not in optimizer.last_kwargs


def test_annealing_mode_hierarchical_routing():
    """验证 hierarchical 模式调用 optimize_policy 时传 mode='hierarchical'。"""
    optimizer = FakeOptimizer(weight_boost=1.0)
    env = FakeEnv()
    loop = AsyncAnnealingLoop(
        optimizer,
        env,
        initial_interval=100,
        retry_delays=[0.0, 0.0],
        annealing_mode="hierarchical",
    )
    loop.start()

    model = FakeModel(weight=0.0)
    loop.submit(model.policy, step=10)
    loop.shutdown()

    # hierarchical 模式应传 mode='hierarchical' 及分块参数
    assert optimizer.last_kwargs.get("mode") == "hierarchical"
    assert optimizer.last_kwargs.get("max_params_per_block") == 200
    assert optimizer.last_kwargs.get("block_strategy") == "tensor_wise"
    # 不应传 head_only
    assert "head_only" not in optimizer.last_kwargs


def test_callback_passes_annealing_mode_to_loop():
    """验证 AsyncAnnealingCallback 将 annealing_mode 透传给 loop。"""
    optimizer = FakeOptimizer()
    env = FakeEnv()
    loop = AsyncAnnealingLoop(optimizer, env, initial_interval=100, retry_delays=[0.0, 0.0])
    # loop 默认 head_only，callback 传 hierarchical 应覆盖
    callback = AsyncAnnealingCallback(loop, verbose=0, annealing_mode="hierarchical")
    callback._init_callback()
    assert loop.annealing_mode == "hierarchical"
    loop.shutdown()


# ============================================================
# Issue #194: 介入率统计 (impact_rate) 测试
# ============================================================
def test_min_effective_reward_delta_param_accepted():
    """验证 min_effective_reward_delta 参数被正确接受。"""
    optimizer = FakeOptimizer()
    env = FakeEnv()
    loop = AsyncAnnealingLoop(
        optimizer,
        env,
        retry_delays=[0.0, 0.0],
        min_effective_reward_delta=2.0,
    )
    assert loop.min_effective_reward_delta == 2.0


def test_min_effective_reward_delta_default_value():
    """验证 min_effective_reward_delta 默认值为 1.0。"""
    optimizer = FakeOptimizer()
    env = FakeEnv()
    loop = AsyncAnnealingLoop(optimizer, env, retry_delays=[0.0, 0.0])
    assert loop.min_effective_reward_delta == 1.0


def test_update_interval_returns_effectiveness():
    """验证 _update_interval 返回布尔值表示是否有效介入。"""
    optimizer = FakeOptimizer()
    env = FakeEnv()
    loop = AsyncAnnealingLoop(
        optimizer,
        env,
        retry_delays=[0.0, 0.0],
        min_effective_reward_delta=1.0,
    )
    # delta=2.0 > 1.0 -> 有效
    assert loop._update_interval(2.0) is True
    # delta=0.5 < 1.0 -> 无效
    assert loop._update_interval(0.5) is False
    # delta=1.0 不大于 1.0 -> 无效（严格大于）
    assert loop._update_interval(1.0) is False


def test_get_impact_rate_zero_when_no_triggers():
    """验证无触发时 get_impact_rate 返回 0.0。"""
    optimizer = FakeOptimizer()
    env = FakeEnv()
    loop = AsyncAnnealingLoop(optimizer, env, retry_delays=[0.0, 0.0])
    assert loop.get_impact_rate() == 0.0


# ============================================================
# Issue #519: 仿真降级失败路径测试
# ============================================================
def test_run_annealing_both_real_and_simulation_fail_raises(tmp_path):
    """验证真机和仿真退火都失败时异常正确传播（Issue #519）。

    _run_annealing_with_retries 在真机重试耗尽后降级为仿真，
    若仿真也失败，异常应被抛出而非静默吞掉。
    """
    optimizer = FakeOptimizer(
        fail_count=10,
        simulation_mode=False,
        simulation_fail=True,
    )
    env = FakeEnv()
    loop = AsyncAnnealingLoop(
        optimizer,
        env,
        initial_interval=100,
        retry_delays=[0.0, 0.0],
        log_path=str(tmp_path / "fail_log.json"),
    )

    agent_wrapper = types.SimpleNamespace(policy=FakePolicy(weight=0.0))

    with pytest.raises(RuntimeError, match="仿真退火失败"):
        loop._run_annealing_with_retries(agent_wrapper, step=1)

    assert optimizer.simulation_mode is True, "重试耗尽后应降级为仿真模式"


def test_both_fail_graceful_in_worker(tmp_path):
    """验证退火完全失败时 worker 线程优雅处理，不崩溃（Issue #519）。

    worker_loop 应捕获 _run_annealing_with_retries 抛出的异常，
    记录日志后继续运行，不产生 history 记录。
    """
    optimizer = FakeOptimizer(
        fail_count=10,
        simulation_mode=False,
        simulation_fail=True,
    )
    env = FakeEnv()
    loop = AsyncAnnealingLoop(
        optimizer,
        env,
        initial_interval=100,
        retry_delays=[0.0, 0.0],
        log_path=str(tmp_path / "graceful_log.json"),
    )
    loop.start()

    model = FakeModel(weight=0.0)
    loop.submit(model.policy, step=20)
    loop.shutdown()

    history = loop.get_history()
    assert len(history) == 0, "退火完全失败时应跳过 history 记录"
    assert optimizer.simulation_mode is True


def test_get_impact_rate_after_triggers():
    """验证多次触发后 get_impact_rate 计算正确。"""
    optimizer = FakeOptimizer()
    env = FakeEnv()
    loop = AsyncAnnealingLoop(
        optimizer,
        env,
        retry_delays=[0.0, 0.0],
        min_effective_reward_delta=1.0,
    )
    # 3 次触发：2 次有效（delta=2.0, 3.0），1 次无效（delta=0.5）
    loop._update_interval(2.0)  # effective
    loop._update_interval(0.5)  # ineffective
    loop._update_interval(3.0)  # effective
    assert loop.get_impact_rate() == pytest.approx(2.0 / 3.0)


def test_impact_rate_in_history_records(tmp_path):
    """验证退火历史记录中包含 impact_rate 和 effective 字段。"""
    optimizer = FakeOptimizer(weight_boost=1.0)
    env = FakeEnv()
    log_path = tmp_path / "impact_log.json"
    loop = AsyncAnnealingLoop(
        optimizer,
        env,
        eval_episodes=2,
        initial_interval=100,
        retry_delays=[0.0, 0.0],
        log_path=str(log_path),
        min_effective_reward_delta=1.0,
    )
    loop.start()

    model = FakeModel(weight=0.0)
    loop.submit(model.policy, step=10)
    loop.shutdown()

    history = loop.get_history()
    assert len(history) == 1
    record = history[0]
    assert "impact_rate" in record
    assert "effective" in record
    assert isinstance(record["impact_rate"], float)
    assert isinstance(record["effective"], bool)
    # FakeOptimizer weight_boost=1.0，FakeEnv max_steps=3
    # delta = new_reward(3.0) - old_reward(0.0) = 3.0 > 1.0 -> effective
    assert record["effective"] is True
    assert record["impact_rate"] == 1.0


def test_impact_rate_in_log_file(tmp_path):
    """验证 JSON 日志文件中每条记录包含 impact_rate 字段。"""
    optimizer = FakeOptimizer(weight_boost=1.0)
    env = FakeEnv()
    log_path = tmp_path / "impact_file_log.json"
    loop = AsyncAnnealingLoop(
        optimizer,
        env,
        eval_episodes=2,
        initial_interval=100,
        retry_delays=[0.0, 0.0],
        log_path=str(log_path),
        min_effective_reward_delta=1.0,
    )
    loop.start()

    model = FakeModel(weight=0.0)
    loop.submit(model.policy, step=10)
    loop.shutdown()

    assert log_path.exists()
    with open(log_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded) == 1
    assert "impact_rate" in loaded[0]
    assert "effective" in loaded[0]


def test_impact_rate_low_when_delta_below_threshold(tmp_path):
    """验证当退火奖励变化低于阈值时，介入率为 0。"""
    # weight_boost=0.1 -> delta 很小，低于 min_effective_reward_delta=1.0
    optimizer = FakeOptimizer(weight_boost=0.1)
    env = FakeEnv()
    loop = AsyncAnnealingLoop(
        optimizer,
        env,
        eval_episodes=2,
        initial_interval=100,
        retry_delays=[0.0, 0.0],
        min_effective_reward_delta=1.0,
    )
    loop.start()

    model = FakeModel(weight=0.0)
    loop.submit(model.policy, step=10)
    loop.shutdown()

    history = loop.get_history()
    assert len(history) == 1
    # delta = 0.3 (0.1 * 3 steps) < 1.0 -> 无效
    assert history[0]["effective"] is False
    assert history[0]["impact_rate"] == 0.0


# ============================================================
# Issue #226: solver_type 追踪测试
# ============================================================


def test_solver_type_in_history_records(tmp_path):
    """验证退火历史记录中包含 solver_type 字段（Issue #226）。"""
    optimizer = FakeOptimizer(weight_boost=1.0)
    env = FakeEnv()
    loop = AsyncAnnealingLoop(
        optimizer,
        env,
        eval_episodes=2,
        initial_interval=100,
        retry_delays=[0.0, 0.0],
        log_path=str(tmp_path / "solver_type_log.json"),
    )
    loop.start()

    model = FakeModel(weight=0.0)
    loop.submit(model.policy, step=10)
    loop.shutdown()

    history = loop.get_history()
    assert len(history) == 1
    assert "solver_type" in history[0]
    assert history[0]["solver_type"] == "numpy_sa"


# ============================================================
# Issue #220: 锁内深拷贝优化测试
# ============================================================


def test_peek_pending_result_deepcopy_outside_lock():
    """验证 peek_pending_result 在锁外执行深拷贝。

    Issue #220 要求：原实现在 self._lock 锁内执行 copy.deepcopy()，
    修改后应在锁内只获取引用，在锁外执行深拷贝。
    """
    optimizer = FakeOptimizer()
    env = FakeEnv()
    loop = AsyncAnnealingLoop(optimizer, env, retry_delays=[0.0, 0.0])

    # 设置一个 pending_result
    test_result = {"step": 10, "state_dict": {"weight": 1.0}, "delta": 0.5}
    with loop._lock:
        loop._pending_result = test_result

    # 调用 peek_pending_result
    peeked = loop.peek_pending_result()
    assert peeked is not None
    assert peeked["step"] == 10
    # 应返回深拷贝，修改不影响原对象
    peeked["step"] = 999
    assert loop._pending_result["step"] == 10  # 原对象未被修改


def test_peek_pending_result_returns_none_when_empty():
    """验证 pending_result 为 None 时 peek_pending_result 返回 None。"""
    optimizer = FakeOptimizer()
    env = FakeEnv()
    loop = AsyncAnnealingLoop(optimizer, env, retry_delays=[0.0, 0.0])
    assert loop._pending_result is None
    assert loop.peek_pending_result() is None


def test_policy_snapshot_class():
    """验证 PolicySnapshot 类的基本功能。"""
    from src.scheduler.async_annealing_callback import PolicySnapshot

    state_dict = {"weight": 0.5}
    policy_ref = FakePolicy(weight=0.5)
    snapshot = PolicySnapshot(state_dict=state_dict, policy_ref=policy_ref)

    assert snapshot.state_dict == state_dict
    assert snapshot.policy_ref is policy_ref


def test_callback_submits_policy_snapshot():
    """验证 callback 提交的是 PolicySnapshot 而非 deepcopy 后的 policy。

    Issue #220 要求：使用 state_dict() + clone() 替代 copy.deepcopy。
    """
    optimizer = FakeOptimizer()
    env = FakeEnv()
    loop = AsyncAnnealingLoop(optimizer, env, initial_interval=10, retry_delays=[0.0, 0.0])

    callback = AsyncAnnealingCallback(loop, verbose=0)
    callback._init_callback()
    callback.model = FakeModel(weight=0.0)

    callback.n_calls = 10
    # 用 mock 替换 loop.submit，验证提交的是 PolicySnapshot
    with patch.object(loop, "submit", return_value=True) as mock_submit:
        callback._on_step()
        assert mock_submit.call_count == 1
        submitted_policy, submitted_step = mock_submit.call_args.args
        # 应提交 PolicySnapshot 实例，而非 FakePolicy 实例
        from src.scheduler.async_annealing_callback import PolicySnapshot

        assert isinstance(submitted_policy, PolicySnapshot)
        # state_dict 应包含权重
        assert "weight" in submitted_policy.state_dict
        assert submitted_step == 10

    loop.shutdown()


def test_worker_loop_handles_policy_snapshot():
    """验证 worker_loop 能正确处理 PolicySnapshot（首次 deepcopy，后续 load_state_dict）。"""
    from src.scheduler.async_annealing_callback import PolicySnapshot

    optimizer = FakeOptimizer(weight_boost=1.0)
    env = FakeEnv()
    loop = AsyncAnnealingLoop(optimizer, env, initial_interval=10, retry_delays=[0.0, 0.0])
    loop.start()

    # 首次提交 PolicySnapshot
    model = FakeModel(weight=0.0)
    snapshot = PolicySnapshot(
        state_dict=model.policy.state_dict(),
        policy_ref=model.policy,
    )
    loop.submit(snapshot, step=10)
    loop.shutdown()

    # worker_loop 应正确处理 PolicySnapshot，产生 history 记录
    history = loop.get_history()
    assert len(history) == 1
    assert history[0]["step"] == 10
    # FakeOptimizer weight_boost=1.0，FakeEnv max_steps=3
    # delta = 3.0 (new) - 0.0 (old) = 3.0
    assert history[0]["delta"] == 3.0


def test_worker_loop_handles_legacy_policy_object():
    """验证 worker_loop 仍兼容旧模式（直接传 policy 对象）。"""
    optimizer = FakeOptimizer(weight_boost=1.0)
    env = FakeEnv()
    loop = AsyncAnnealingLoop(optimizer, env, initial_interval=10, retry_delays=[0.0, 0.0])
    loop.start()

    # 旧模式：直接传 policy 对象
    model = FakeModel(weight=0.0)
    loop.submit(model.policy, step=10)
    loop.shutdown()

    history = loop.get_history()
    assert len(history) == 1
    assert history[0]["step"] == 10
    assert history[0]["delta"] == 3.0


def test_worker_loop_reuses_eval_policy_for_multiple_snapshots():
    """验证 worker_loop 在多次提交 PolicySnapshot 时复用 eval_policy 实例。

    Issue #220 优化点：首次 deepcopy 创建持久化 eval_policy，
    后续仅 load_state_dict 更新权重，避免重复深拷贝。
    """
    from src.scheduler.async_annealing_callback import PolicySnapshot

    optimizer = FakeOptimizer(weight_boost=0.0)  # 不改变权重，便于复用
    env = FakeEnv()
    loop = AsyncAnnealingLoop(optimizer, env, initial_interval=10, retry_delays=[0.0, 0.0])
    loop.start()

    # 第一次提交
    model1 = FakeModel(weight=0.0)
    snapshot1 = PolicySnapshot(
        state_dict=model1.policy.state_dict(),
        policy_ref=model1.policy,
    )
    loop.submit(snapshot1, step=10)

    # 等待 worker 处理完第一个任务
    import time as _time

    _time.sleep(0.3)

    # 第二次提交
    model2 = FakeModel(weight=2.0)
    snapshot2 = PolicySnapshot(
        state_dict=model2.policy.state_dict(),
        policy_ref=model2.policy,
    )
    loop.submit(snapshot2, step=20)
    loop.shutdown()

    # 两次任务都应正常处理
    history = loop.get_history()
    assert len(history) == 2
    assert {h["step"] for h in history} == {10, 20}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
