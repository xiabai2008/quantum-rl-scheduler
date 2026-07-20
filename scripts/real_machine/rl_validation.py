"""RL 调度器真机闭环验证（阶段 1）。

PPO 训练过程中按概率抽样上真机，对比真机 vs 模拟器决策差异。
约 20 个真机任务。

实验设计:
    - 任务规模: 200 个调度任务 (total_timesteps=600)
    - 真机抽样: interval=30, prob=1.0 → 20 个真机任务
    - 训练算法: PPO
    - 种子: seed=42（确保可复现）

用法:
    # Mock dry-run（不消耗真机机时）
    python scripts/real_machine/rl_validation.py --mock

    # 真机执行
    python scripts/real_machine/rl_validation.py

    # 指定机器
    python scripts/real_machine/rl_validation.py --machine tianyan176
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 环境变量设置（必须在 import 项目模块之前）
# ---------------------------------------------------------------------------
os.environ.setdefault("TIANYAN_API_KEY", "")
os.environ.setdefault("TIANYAN_MOCK_MODE", "false")
os.environ.setdefault("TIANYAN_MACHINE", "tianyan176")
os.environ.setdefault("QUANTUM_ACCELERATION_ENABLED", "1")

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import contextlib

from loguru import logger

# 复用 smoke_test.py 中的工具函数
from smoke_test import (  # type: ignore[import-not-found]
    MockSmokeClient,
    compute_fidelity,
    compute_measurement_error,
    compute_probability_from_shots,
    parse_probability,
    poll_task_result,
)
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.api.tianyan_cqlib import CqlibTianyanClient
from src.scheduler.agent import PPOAgent
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
RESULTS_DIR = _PROJECT_ROOT / "results" / "real_machine"

# 训练参数
TOTAL_TIMESTEPS = 600  # 200 任务 × 3 步/任务
REAL_CALLBACK_INTERVAL = 30  # 每 30 步触发一次 → 600/30 = 20 次真机提交
REAL_CALLBACK_PROB = 1.0  # 每次触发 100% 提交 → 20 个真机任务
REAL_CALLBACK_SHOTS = 1024
SEED = 42

# RL 动作含义映射
ACTION_MEANINGS: dict[int, str] = {
    0: "classical",
    1: "quantum",
    2: "hybrid",
}


# ---------------------------------------------------------------------------
# 增强真机回调：记录 RL 上下文 + 提交真机任务
# ---------------------------------------------------------------------------


class EnhancedRealCallback(BaseCallback):
    """增强真机回调：在 PPO 训练中按间隔提交真机任务。

    与 src/scheduler/agent.py 的 RealMachineCallback 相比，本回调额外记录：
        - rl_action: 提交时的 RL 动作
        - rl_action_meaning: 动作含义
        - reward: 提交时的即时奖励
        - quantum_speedup: 量子加速比（从环境获取）

    提交采用非阻塞模式（仅 submit，不 wait），训练结束后统一轮询结果。

    Attributes:
        env: 训练环境
        interval: 抽样间隔（步数）
        prob: 每次触发的提交概率
        client: 真机客户端
        shots: 真机任务 shots 数
        records: 真机提交记录列表
    """

    def __init__(
        self,
        env: QuantumSchedulingEnv,
        interval: int = 30,
        prob: float = 1.0,
        client: Any = None,
        shots: int = 1024,
        verbose: int = 1,
    ) -> None:
        """初始化增强真机回调。

        Args:
            env: 训练环境（QuantumSchedulingEnv）
            interval: 真机抽样间隔步数
            prob: 每次触发的提交概率
            client: 真机客户端（CqlibTianyanClient 或 MockSmokeClient）
            shots: 真机任务 shots 数
            verbose: 日志详细程度
        """
        super().__init__(verbose)
        self.env = env
        self.interval = int(interval)
        self.prob = float(prob)
        self.client = client
        self.shots = int(shots)
        self.records: list[dict[str, Any]] = []
        self._warned_no_client = False

    def _on_step(self) -> bool:
        """每步触发：达到 interval 时按 prob 概率提交真机任务。

        Returns:
            bool: 始终返回 True（不中断训练）
        """
        # 跳过第 0 步和非间隔步
        if self.n_calls == 0 or self.n_calls % self.interval != 0:
            return True
        if self.prob <= 0.0:
            return True
        # 概率门控
        if random.random() >= self.prob:
            return True

        # 客户端检查
        if self.client is None:
            if not self._warned_no_client:
                logger.warning(f"[EnhancedCB] 无客户端，真机抽样已禁用 (step={self.n_calls})")
                self._warned_no_client = True
            return True

        machine_name = getattr(self.client, "machine_name", "unknown")

        # 从环境获取待处理任务
        task = None
        if hasattr(self.env, "get_random_pending_task"):
            try:
                task = self.env.get_random_pending_task()
            except Exception:
                task = None

        # 构造 QCIS（Task 无 qcis 字段时用 H 门占位电路）
        qcis = "H Q0\nM Q0"
        task_id_str = "synthetic"
        if task is not None:
            task_id_str = str(getattr(task, "task_id", "synthetic"))
            qcis = getattr(task, "qcis", None) or "H Q0\nM Q0"

        # 获取 RL 上下文（动作、奖励）
        rl_action = -1
        rl_action_meaning = "unknown"
        reward = 0.0
        quantum_speedup = 0.0

        try:
            # 从 SB3 locals 获取当前动作和奖励
            actions = self.locals.get("actions", None)
            rewards = self.locals.get("rewards", None)
            if actions is not None and len(actions) > 0:
                rl_action = int(actions[0])
                rl_action_meaning = ACTION_MEANINGS.get(rl_action, "unknown")
            if rewards is not None and len(rewards) > 0:
                reward = float(rewards[0])
        except Exception:
            pass

        # 从环境获取量子加速比
        with contextlib.suppress(Exception):
            quantum_speedup = float(getattr(self.env, "_last_quantum_speedup", 0.0))

        # 提交真机任务（非阻塞）
        t0 = time.time()
        record: dict[str, Any] = {
            "step": int(self.n_calls),
            "task_id": task_id_str,
            "machine": machine_name,
            "qcis": qcis,
            "rl_action": rl_action,
            "rl_action_meaning": rl_action_meaning,
            "reward": round(reward, 4),
            "quantum_speedup": round(quantum_speedup, 4),
            "real_task_id": None,
            "submit_status": "pending",
            "submit_latency_s": 0.0,
            "real_probability": {},
            "mock_probability": {"0": 0.5, "1": 0.5},  # H 门理论值
            "probability_diff": None,
            "duration_sec": None,
            "fidelity": None,
            "measurement_error": None,
        }

        try:
            real_tid = self.client.submit_quantum_task(
                qcis=qcis,
                shots=self.shots,
                task_name=f"RLVal_step{self.n_calls}_{task_id_str}",
            )
            record["submit_latency_s"] = round(time.time() - t0, 3)
            record["real_task_id"] = str(real_tid) if real_tid else None
            record["submit_status"] = "submitted" if real_tid else "rejected"
            logger.info(
                f"[EnhancedCB] step={self.n_calls} machine={machine_name} "
                f"tid={real_tid} action={rl_action_meaning} "
                f"latency={record['submit_latency_s']}s"
            )
        except Exception as e:
            record["submit_latency_s"] = round(time.time() - t0, 3)
            record["submit_status"] = f"error: {str(e)[:80]}"
            logger.error(f"[EnhancedCB] step={self.n_calls} 提交失败: {e}")

        self.records.append(record)
        return True


# ---------------------------------------------------------------------------
# 奖励追踪回调
# ---------------------------------------------------------------------------


class RewardTrackerCallback(BaseCallback):
    """追踪训练过程中的奖励曲线。

    记录每个 episode 的总奖励和步数，用于真机 vs Mock 对照。

    Attributes:
        episode_rewards: 每个 episode 的总奖励列表
        episode_lengths: 每个 episode 的步数列表
        step_rewards: 每步的即时奖励列表
    """

    def __init__(self, verbose: int = 0) -> None:
        """初始化奖励追踪回调。

        Args:
            verbose: 日志详细程度
        """
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.step_rewards: list[dict[str, float]] = []
        self._current_reward = 0.0
        self._current_length = 0

    def _on_step(self) -> bool:
        """每步记录奖励，episode 结束时汇总。

        Returns:
            bool: 始终返回 True
        """
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])

        reward = float(rewards[0]) if len(rewards) > 0 else 0.0
        done = bool(dones[0]) if len(dones) > 0 else False

        self._current_reward += reward
        self._current_length += 1
        self.step_rewards.append({"step": float(self.n_calls), "reward": reward})

        if done:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            self._current_reward = 0.0
            self._current_length = 0

        return True


# ---------------------------------------------------------------------------
# 训练函数
# ---------------------------------------------------------------------------


def run_training(
    client: Any,
    machine_name: str,
    seed: int = SEED,
    mock_mode: bool = False,
) -> tuple[list[dict[str, Any]], list[float], list[int], str]:
    """执行一次 PPO 训练，返回真机记录和奖励曲线。

    Args:
        client: 真机客户端（CqlibTianyanClient 或 MockSmokeClient）
        machine_name: 机器名称
        seed: 随机种子
        mock_mode: 是否 Mock 模式（影响日志标签）

    Returns:
        tuple: (真机提交记录, episode奖励列表, episode长度列表, 日志目录)
    """
    tag = "Mock" if mock_mode else "Real"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = str(RESULTS_DIR / f"ppo_logs_{tag.lower()}_{timestamp}")

    logger.info(f"[{tag}] 创建环境 (seed={seed})")
    env = QuantumSchedulingEnv(
        machine_configs=DEFAULT_MACHINE_CONFIGS,
        seed=seed,
        real_submit_probability=0.0,  # 环境不上真机，仅回调上真机
    )

    # 绑定真机客户端（仅用于回调，不影响环境逻辑）
    env.attach_real_clients({machine_name: client})

    logger.info(f"[{tag}] 创建 PPOAgent")
    agent = PPOAgent(
        env,
        learning_rate=3e-4,
        n_steps=256,
        verbose=0,
        seed=seed,
        log_dir=log_dir,
    )

    # 手动构建模型
    if agent.model is None:
        agent.model = agent._build_model()

    # 创建回调
    enhanced_cb = EnhancedRealCallback(
        env=env,
        interval=REAL_CALLBACK_INTERVAL,
        prob=REAL_CALLBACK_PROB,
        client=client,
        shots=REAL_CALLBACK_SHOTS,
        verbose=1,
    )
    reward_cb = RewardTrackerCallback(verbose=0)

    # 评估回调（关闭评估，避免消耗步数）
    eval_env = Monitor(env)
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=None,
        log_path=None,
        eval_freq=100000,  # 实际关闭
        n_eval_episodes=1,
        deterministic=True,
    )

    callback = CallbackList([enhanced_cb, reward_cb, eval_cb])

    # 训练
    logger.info(
        f"[{tag}] 开始训练: timesteps={TOTAL_TIMESTEPS}, "
        f"interval={REAL_CALLBACK_INTERVAL}, prob={REAL_CALLBACK_PROB}"
    )
    print(f"\n{'=' * 60}")
    print(f"  [{tag}] RL 调度器真机闭环验证")
    print(
        f"  总步数: {TOTAL_TIMESTEPS} | 真机间隔: {REAL_CALLBACK_INTERVAL} | "
        f"概率: {REAL_CALLBACK_PROB}"
    )
    print(f"  预计真机任务数: {TOTAL_TIMESTEPS // REAL_CALLBACK_INTERVAL}")
    print(f"{'=' * 60}")

    t0 = time.time()
    agent.model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        tb_log_name=f"ppo_{tag.lower()}_{timestamp}",
        reset_num_timesteps=True,
    )
    elapsed = round(time.time() - t0, 1)
    logger.info(f"[{tag}] 训练完成，耗时 {elapsed}s")

    return enhanced_cb.records, reward_cb.episode_rewards, reward_cb.episode_lengths, log_dir


# ---------------------------------------------------------------------------
# 真机结果轮询
# ---------------------------------------------------------------------------


def poll_all_real_results(
    client: Any,
    records: list[dict[str, Any]],
    per_task_timeout: int = 60,
) -> list[dict[str, Any]]:
    """轮询所有真机任务的结果，补充概率和耗时数据。

    Args:
        client: 真机客户端
        records: 真机提交记录列表
        per_task_timeout: 单任务轮询超时秒数

    Returns:
        更新后的记录列表（含概率、耗时、保真度）
    """
    submitted = [r for r in records if r.get("real_task_id")]
    total = len(submitted)
    logger.info(f"[Poll] 开始轮询 {total} 个真机任务结果")

    for i, record in enumerate(records):
        task_id = record.get("real_task_id")
        if not task_id:
            continue

        print(f"  [{i + 1}/{total}] 轮询 {task_id} ...", end=" ", flush=True)

        result = poll_task_result(
            client=client,
            task_id=task_id,
            timeout=per_task_timeout,
            poll_interval=3,
            max_unknown=3,
            per_poll_timeout=15,
        )

        if result.get("status") == "completed":
            # 解析概率
            probability = {}
            raw_data = result.get("raw", {})

            if isinstance(raw_data, dict):
                raw_prob = raw_data.get("probability")
                if raw_prob:
                    probability = parse_probability(raw_prob)

                if not probability:
                    result_status = raw_data.get("resultStatus")
                    if result_status:
                        probability = compute_probability_from_shots(result_status)

            if not probability and result.get("result"):
                probability = parse_probability(result["result"])

            # H 门理论分布
            mock_prob = record.get("mock_probability", {"0": 0.5, "1": 0.5})

            # 计算差异和保真度
            prob_diff = compute_measurement_error(probability, mock_prob)
            fidelity = compute_fidelity(probability, mock_prob)

            record["real_probability"] = probability
            record["probability_diff"] = round(prob_diff, 4)
            record["fidelity"] = round(fidelity, 4)
            record["measurement_error"] = round(prob_diff, 4)
            record["duration_sec"] = (
                round(result.get("duration_sec", 0.0), 2) if result.get("duration_sec") else None
            )
            record["poll_status"] = "completed"
            print(f"[PASS] fidelity={record['fidelity']}")
        else:
            record["poll_status"] = result.get("status", "unknown")
            record["real_probability"] = {}
            print(f"[FAIL] {record['poll_status']}")

    return records


# ---------------------------------------------------------------------------
# 结果保存
# ---------------------------------------------------------------------------


def save_results(
    real_records: list[dict[str, Any]],
    mock_records: list[dict[str, Any]],
    real_episode_rewards: list[float],
    mock_episode_rewards: list[float],
    real_episode_lengths: list[int],
    mock_episode_lengths: list[int],
    output_dir: Path | None = None,
) -> str:
    """保存 RL 验证结果到 JSON 文件。

    Args:
        real_records: 真机提交记录
        mock_records: Mock 提交记录
        real_episode_rewards: 真机训练 episode 奖励
        mock_episode_rewards: Mock 训练 episode 奖励
        real_episode_lengths: 真机训练 episode 长度
        mock_episode_lengths: Mock 训练 episode 长度
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rl_validation_{timestamp}.json"
    filepath = output_dir / filename

    # 统计
    real_submitted = [r for r in real_records if r.get("real_task_id")]
    real_completed = [r for r in real_records if r.get("poll_status") == "completed"]
    real_fidelities = [r["fidelity"] for r in real_completed if r.get("fidelity") is not None]
    real_prob_diffs = [
        r["probability_diff"] for r in real_completed if r.get("probability_diff") is not None
    ]

    summary = {
        "test_type": "rl_validation",
        "timestamp": datetime.now().astimezone().isoformat(),
        "config": {
            "total_timesteps": TOTAL_TIMESTEPS,
            "real_callback_interval": REAL_CALLBACK_INTERVAL,
            "real_callback_prob": REAL_CALLBACK_PROB,
            "real_callback_shots": REAL_CALLBACK_SHOTS,
            "seed": SEED,
        },
        "real_machine": {
            "total_submitted": len(real_submitted),
            "completed": len(real_completed),
            "failed": len(real_submitted) - len(real_completed),
            "avg_fidelity": (
                round(sum(real_fidelities) / max(len(real_fidelities), 1), 4)
                if real_fidelities
                else None
            ),
            "avg_probability_diff": (
                round(sum(real_prob_diffs) / max(len(real_prob_diffs), 1), 4)
                if real_prob_diffs
                else None
            ),
            "episode_rewards": real_episode_rewards,
            "episode_lengths": real_episode_lengths,
            "avg_episode_reward": (
                round(sum(real_episode_rewards) / max(len(real_episode_rewards), 1), 4)
                if real_episode_rewards
                else None
            ),
        },
        "mock_control": {
            "total_submitted": len([r for r in mock_records if r.get("real_task_id")]),
            "episode_rewards": mock_episode_rewards,
            "episode_lengths": mock_episode_lengths,
            "avg_episode_reward": (
                round(sum(mock_episode_rewards) / max(len(mock_episode_rewards), 1), 4)
                if mock_episode_rewards
                else None
            ),
        },
        "real_task_records": real_records,
        "mock_task_records": mock_records,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"[RL Val] 结果已保存: {filepath}")
    return str(filepath)


def print_summary(
    real_records: list[dict[str, Any]],
    mock_records: list[dict[str, Any]],
    real_rewards: list[float],
    mock_rewards: list[float],
) -> None:
    """打印汇总报告。

    Args:
        real_records: 真机提交记录
        mock_records: Mock 提交记录
        real_rewards: 真机训练奖励
        mock_rewards: Mock 训练奖励
    """
    real_submitted = [r for r in real_records if r.get("real_task_id")]
    real_completed = [r for r in real_records if r.get("poll_status") == "completed"]
    real_fidelities = [r["fidelity"] for r in real_completed if r.get("fidelity") is not None]
    real_diffs = [
        r["probability_diff"] for r in real_completed if r.get("probability_diff") is not None
    ]

    print(f"\n{'=' * 60}")
    print("  RL 调度器真机闭环验证 - 汇总报告")
    print(f"{'=' * 60}")

    print("\n  [真机]")
    print(f"  提交任务数: {len(real_submitted)}")
    print(f"  成功获取结果: {len(real_completed)}")
    print(f"  失败: {len(real_submitted) - len(real_completed)}")
    if real_fidelities:
        print(f"  平均保真度: {sum(real_fidelities) / len(real_fidelities):.4f}")
    if real_diffs:
        print(f"  平均概率差异: {sum(real_diffs) / len(real_diffs):.4f}")
    if real_rewards:
        print(f"  平均 episode 奖励: {sum(real_rewards) / len(real_rewards):.2f}")
        print(f"  episode 数: {len(real_rewards)}")

    print("\n  [Mock 对照]")
    print(f"  episode 数: {len(mock_rewards)}")
    if mock_rewards:
        print(f"  平均 episode 奖励: {sum(mock_rewards) / len(mock_rewards):.2f}")

    # 奖励对比
    if real_rewards and mock_rewards:
        real_avg = sum(real_rewards) / len(real_rewards)
        mock_avg = sum(mock_rewards) / len(mock_rewards)
        diff = real_avg - mock_avg
        print("\n  [对比]")
        print(f"  真机平均奖励: {real_avg:.2f}")
        print(f"  Mock 平均奖励: {mock_avg:.2f}")
        print(f"  差异: {diff:+.2f} ({'真机优' if diff > 0 else 'Mock优' if diff < 0 else '相同'})")

    # 真机任务详情
    print(
        f"\n  {'Step':<6s} {'Machine':<12s} {'Action':<10s} {'Reward':>8s} "
        f"{'Fidelity':>10s} {'ProbDiff':>10s}"
    )
    print(f"  {'-' * 6} {'-' * 12} {'-' * 10} {'-' * 8} {'-' * 10} {'-' * 10}")
    for r in real_records:
        step = r.get("step", "?")
        machine = r.get("machine", "?")[:11]
        action = r.get("rl_action_meaning", "?")[:9]
        reward = f"{r.get('reward', 0):.2f}"
        fid = f"{r['fidelity']:.4f}" if r.get("fidelity") is not None else "N/A"
        diff = f"{r['probability_diff']:.4f}" if r.get("probability_diff") is not None else "N/A"
        print(f"  {step:<6} {machine:<12s} {action:<10s} {reward:>8s} {fid:>10s} {diff:>10s}")

    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> None:
    """RL 验证主入口。

    支持命令行参数:
        --mock: 使用 Mock 客户端 dry-run
        --machine: 指定首选机器
        --skip-mock: 跳过 Mock 对照组（真机执行时可加速）
    """
    import argparse

    parser = argparse.ArgumentParser(description="RL 调度器真机闭环验证")
    parser.add_argument("--mock", action="store_true", help="Mock dry-run")
    parser.add_argument("--machine", default="tianyan176", help="首选机器")
    parser.add_argument("--skip-mock", action="store_true", help="跳过 Mock 对照组")
    parser.add_argument("--verbose", action="store_true", help="DEBUG 日志")
    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    # 创建客户端
    if args.mock:
        print("[Mode] Mock dry-run")
        real_client: Any = MockSmokeClient(machine_name=args.machine, mock_delay=0.02)
    else:
        print("[Mode] 真机执行")
        api_key = os.environ.get("TIANYAN_API_KEY", "")
        if not api_key:
            print("[FAIL] 未设置 TIANYAN_API_KEY")
            sys.exit(1)
        real_client = CqlibTianyanClient(
            login_key=api_key,
            machine_name=args.machine,
            auto_retry_machine=True,
        )
        print(f"[Setup] 真机客户端已创建: {args.machine}")

    # 1. 真机训练
    print("\n--- 阶段 1a: 真机训练 + 真机抽样 ---")
    real_records, real_rewards, real_lengths, _ = run_training(
        client=real_client,
        machine_name=args.machine,
        seed=SEED,
        mock_mode=args.mock,
    )
    logger.info(f"[Real] 真机提交记录: {len(real_records)} 条")
    logger.info(f"[Real] episode 奖励: {real_rewards}")

    # 2. 轮询真机结果（Mock 模式也走此流程验证逻辑）
    if any(r.get("real_task_id") for r in real_records):
        print("\n--- 阶段 1b: 轮询真机任务结果 ---")
        real_records = poll_all_real_results(real_client, real_records)

    # 3. Mock 对照训练
    mock_records: list[dict[str, Any]] = []
    mock_rewards: list[float] = []
    mock_lengths: list[int] = []

    if not args.skip_mock:
        print("\n--- 阶段 1c: Mock 对照训练 ---")
        mock_client: Any = MockSmokeClient(machine_name=args.machine, mock_delay=0.01)
        mock_records, mock_rewards, mock_lengths, _ = run_training(
            client=mock_client,
            machine_name=args.machine,
            seed=SEED,
            mock_mode=True,
        )
        # 轮询 Mock 结果
        if any(r.get("real_task_id") for r in mock_records):
            mock_records = poll_all_real_results(mock_client, mock_records)
    else:
        logger.info("[Mock] 跳过 Mock 对照组")

    # 4. 保存结果
    filepath = save_results(
        real_records=real_records,
        mock_records=mock_records,
        real_episode_rewards=real_rewards,
        mock_episode_rewards=mock_rewards,
        real_episode_lengths=real_lengths,
        mock_episode_lengths=mock_lengths,
    )

    # 5. 打印汇总
    print_summary(real_records, mock_records, real_rewards, mock_rewards)
    print(f"\n  结果文件: {filepath}")


if __name__ == "__main__":
    main()
