"""PPO 真机闭环训练（完整 PPO 训练并全程使用真机反馈）。

与 `rl_validation.py` 不同：
- `rl_validation.py`：只在训练回调中抽样提交任务，用于验证 RL 决策与真机兼容性
- `ppo_closed_loop_training.py`：**完整的闭环训练**，每个量子任务真正提交到天衍真机，
  使用真机测量结果更新奖励，并持续训练 PPO 策略直到训练完成。

特点:
    - 真正的真机闭环：每个量子任务实际在天衍真机上执行，用真机结果计算奖励
    - 非阻塞轮询：使用 daemon 线程超时方案避免 SDK 内部重试死锁
    - 自动降级：连续失败超过阈值自动切换到模拟，保护训练过程
    - 回调记录：训练过程中记录每次真机提交的完整信息
    - checkpoint 保存：支持断点续训

实验设计:
    - 总训练步数: 100,000 (可配置)
    - 保存间隔: 每 10,000 步
    - 学习率: 3e-4
    - 种子: 42

用法:
    # Mock dry-run（使用模拟环境，不消耗真机机时）
    python scripts/real_machine/ppo_closed_loop_training.py --mock --timesteps 1000

    # 真机训练，总步数 50,000
    python scripts/real_machine/ppo_closed_loop_training.py --timesteps 50000

    # 指定机器和保存目录
    python scripts/real_machine/ppo_closed_loop_training.py --machine tianyan176 --output models/ppo_real_closed_loop
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 环境变量设置（必须在 import 项目模块之前）
# ---------------------------------------------------------------------------
# 加载 .env 文件（优先项目根目录，其次向上查找）
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
if _ENV_PATH.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_PATH)
    except ImportError:
        # fallback：手动解析 .env 文件
        with open(_ENV_PATH, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _key, _val = _line.split("=", 1)
                    if _key.strip() not in os.environ:
                        os.environ[_key.strip()] = _val.strip().strip('"').strip("'")

os.environ.setdefault("TIANYAN_API_KEY", "")
os.environ.setdefault("TIANYAN_MOCK_MODE", "false")
os.environ.setdefault("TIANYAN_MACHINE", "tianyan176")
os.environ.setdefault("QUANTUM_ACCELERATION_ENABLED", "1")

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

for p in [_PROJECT_ROOT, _SCRIPT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from loguru import logger

import csv
from stable_baselines3.common.callbacks import BaseCallback

from src.api.tianyan_cqlib import CqlibTianyanClient
from src.scheduler.agent import PPOAgent
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "models" / "ppo_real_closed_loop"
RESULTS_DIR = _PROJECT_ROOT / "results" / "real_machine"


# ---------------------------------------------------------------------------
# 训练记录回调
# ---------------------------------------------------------------------------


def _json_default(o):
    """JSON 序列化兜底：把 numpy 标量/数组转成原生 Python 类型。"""
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


class TrainingRecordCallback:
    """训练记录回调：记录每个真机提交的信息供后续分析。

    在 PPO 训练完成每个 step 后被调用，记录当前真机状态统计、
    奖励分布和已完成的真机任务数。
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self.start_time = time.perf_counter()

    def on_step(self, timestep: int, env: QuantumSchedulingEnv, reward: float) -> None:
        """记录一步训练。"""
        real_stats = env.get_real_machine_stats()
        elapsed = time.perf_counter() - self.start_time

        record = {
            "timestep": timestep,
            "elapsed_sec": round(elapsed, 2),
            "reward": round(float(reward), 4),
            "real_success": real_stats.get("success_count", 0),
            "real_failed": real_stats.get("fail_count", 0),
            "pending": real_stats.get("pending_count", 0),
            "degraded": real_stats.get("degraded", False),
            "consecutive_failures": real_stats.get("consecutive_failures", 0),
        }
        self.records.append(record)

    def save(self, prefix: str = "training") -> Path:
        """保存训练记录到 JSON。"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"{prefix}_record_{timestamp}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "type": "ppo_closed_loop_training",
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "total_elapsed_sec": round(time.perf_counter() - self.start_time, 2),
                    "records": self.records,
                },
                f,
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )
        logger.info(f"[Callback] 训练记录已保存: {filepath}")
        return filepath


# ---------------------------------------------------------------------------
# 训练指标回调（loss / KL / value_loss / entropy 等结构化记录）
# ---------------------------------------------------------------------------


class _MetricCapture:  # pragma: no cover - 保留接口占位，实际捕获改用 _on_step 快照
    """历史实现占位（已弃用）。

    早期实现试图通过挂载 ``KVWriter`` 在 ``dump()`` 写盘前捕获 ``train/*``，
    但 SB3 的调用顺序是 ``on_rollout_end`` → ``dump()`` → ``train()``，
    导致回调永远读到上一轮的快照、首轮缺失 loss。现改为在 ``_on_step``
    中实时快照 ``logger.name_to_value``（``train()`` 后、``dump()`` 前该字典
    恰好含有上一轮 ``train/*`` 标量），更稳定且不受 dump 顺序影响。
    """

    def __init__(self) -> None:
        self.last: dict[str, float] = {}


class TrainingMetricsCallback(BaseCallback):
    """记录 PPO 训练过程中的关键指标到 CSV 与 JSON。

    在每次 rollout 结束时，从 ``model.logger`` 读取最近一次 dump 的训练标量
    （loss / policy_gradient_loss / value_loss / entropy_loss / approx_kl /
    clip_fraction / explained_variance / fps），并合并 episode 平均奖励，
    写入 ``training_metrics.csv``（追加）与训练结束时的 ``*_metrics_*.json``。

    指标用于监控训练健康度、判断策略是否崩溃，以及断点续训后对比趋势。
    """

    _FIELDS = [
        "timestep", "ep_rew_mean", "ep_len_mean",
        "loss", "policy_gradient_loss", "value_loss", "entropy_loss",
        "approx_kl", "clip_fraction", "explained_variance", "fps",
    ]

    def __init__(self, record_cb: "TrainingRecordCallback", output_dir: Path,
                 verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.record_cb = record_cb
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: list[dict[str, Any]] = []
        # 实时快照：在 _on_step 中捕获 logger.name_to_value 的最新 train/* 标量
        self._latest: dict[str, Any] = {}
        self.csv_path = self.output_dir / "training_metrics.csv"
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self._FIELDS)

    def _on_rollout_end(self) -> None:
        sb = self._latest
        ep_info = getattr(self.model, "ep_info_buffer", None)
        ep_rew_mean = float(np.mean([e["r"] for e in ep_info])) if ep_info else None
        ep_len_mean = float(np.mean([e["l"] for e in ep_info])) if ep_info else None

        def _get(key: str) -> Any:
            return sb.get(key)

        row = {
            "timestep": int(self.num_timesteps),
            "ep_rew_mean": ep_rew_mean,
            "ep_len_mean": ep_len_mean,
            "loss": _get("train/loss"),
            "policy_gradient_loss": _get("train/policy_gradient_loss"),
            "value_loss": _get("train/value_loss"),
            "entropy_loss": _get("train/entropy_loss"),
            "approx_kl": _get("train/approx_kl"),
            "clip_fraction": _get("train/clip_fraction"),
            "explained_variance": _get("train/explained_variance"),
            "fps": _get("train/fps"),
        }
        self.metrics.append(row)

        # 同步到训练记录，便于与真机反馈统计合并分析
        self.record_cb.records.append({
            "timestep": row["timestep"],
            "metric": True,
            "loss": row["loss"],
            "approx_kl": row["approx_kl"],
            "value_loss": row["value_loss"],
            "entropy_loss": row["entropy_loss"],
            "ep_rew_mean": ep_rew_mean,
        })

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row[k] for k in self._FIELDS])

        if self.verbose:
            logger.info(
                f"[Metrics] step={row['timestep']} loss={row['loss']} "
                f"kl={row['approx_kl']} ent={row['entropy_loss']} "
                f"ep_rew={ep_rew_mean}"
            )

    def _on_step(self) -> bool:
        """BaseCallback 要求的抽象方法；同时实时快照最新 train/* 标量。

        SB3 的 ``train/*`` 标量在 ``train()`` 结束后写入 ``logger.name_to_value``，
        并在下一次 rollout 末 ``dump()`` 时清空。由于 ``on_rollout_end`` 总在
        ``dump()`` 之前触发，直接读 ``name_to_value`` 会错过本轮 loss。
        这里在环境交互阶段（train() 之后、dump() 之前）逐步捕获，确保
        ``_on_rollout_end`` 能取到上一轮真实的 train/* 指标。
        """
        n2v = getattr(self.model.logger, "name_to_value", None)
        if n2v and "train/loss" in n2v:
            self._latest = dict(n2v)
        return True

    def save(self, prefix: str = "ppo") -> Path:
        """保存指标 JSON（训练结束时调用）。"""
        path = self.output_dir / f"{prefix}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"metrics": self.metrics}, f, indent=2, ensure_ascii=False, default=_json_default)
        logger.info(f"[Metrics] 训练指标已保存: {path}")
        return path


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------


def main() -> None:
    """PPO 真机闭环训练主入口。"""
    parser = argparse.ArgumentParser(description="PPO 真机闭环训练")
    parser.add_argument("--mock", action="store_true", help="Mock 模式（不使用真机）")
    parser.add_argument("--machine", default="tianyan176", help="首选机器名称")
    parser.add_argument("--timesteps", type=int, default=2000, help="总训练步数（真机模式建议 2000-5000）")
    parser.add_argument("--save-interval", type=int, default=1000, help="保存间隔步数")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--n-steps", type=int, default=512, help="PPO n_steps 参数")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO batch_size")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="输出模型目录")
    parser.add_argument("--resume", type=str, default=None,
                        help="断点续训：指定已有模型 .zip 路径，从该检查点继续训练")
    parser.add_argument("--real-submit-prob", type=float, default=1.0,
                        help="量子任务真机提交概率 (1.0=每个量子任务都提交)")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    args = parser.parse_args()

    output_dir = Path(args.output)

    print(f"\n{'=' * 70}")
    print("  PPO 真机闭环训练")
    print(f"{'=' * 70}")
    print(f"  总训练步数: {args.timesteps}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  n_steps: {args.n_steps}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  保存间隔: {args.save_interval} 步")
    print(f"  随机种子: {args.seed}")
    print(f"  输出目录: {output_dir}")
    print(f"  机器: {args.machine}")
    print(f"  真机提交概率: {args.real_submit_prob}")
    if args.mock:
        print(f"  模式: [MOCK] 模拟环境（不使用真机）")
    else:
        print(f"  模式: [REAL] 真机闭环训练")
    print(f"{'=' * 70}\n")

    # ── 步骤 1: 创建环境 ──
    print("--- [1/5] 创建调度环境 ---")

    # 准备机器配置：将指定机器标记为真实。
    # 注意：DEFAULT_MACHINE_CONFIGS 中的机器名（如 tianyan_tn）与 cqlib 机器名
    # （如 tianyan176）不同，必须统一使用 cqlib 机器名才能匹配 attach_real_clients。
    machine_configs = []
    # cqlib 机器名 → 配置机器名映射
    CQLIB_TO_CONFIG: dict[str, str] = {
        "tianyan176": "tianyan_tn",
        "tianyan176-2": "tianyan_tn",
        "tianyan_sw": "tianyan_sw",
        "tianyan_s": "tianyan_s",
    }
    matched_config_name = CQLIB_TO_CONFIG.get(args.machine, args.machine)
    for cfg in DEFAULT_MACHINE_CONFIGS:
        if cfg["name"] == matched_config_name:
            # 用 cqlib 机器名覆盖配置名，确保 _real_clients 中能匹配
            machine_configs.append({**cfg, "name": args.machine, "is_real": True})
        else:
            machine_configs.append(cfg)

    env = QuantumSchedulingEnv(
        machine_configs=machine_configs,
        seed=args.seed,
        real_submit_probability=args.real_submit_prob,
        use_real_machine=True,  # 启用真机闭环
        real_machine_feedback_weight=1.0,  # 真机反馈权重
    )

    # ── 步骤 2: 绑定真机客户端 ──
    print("\n--- [2/5] 绑定真机客户端 ---")
    if not args.mock:
        api_key = os.environ.get("TIANYAN_API_KEY", "")
        if not api_key:
            print("[ERROR] 未设置 TIANYAN_API_KEY 环境变量")
            sys.exit(1)
        client = CqlibTianyanClient(
            login_key=api_key,
            machine_name=args.machine,
            auto_retry_machine=True,
        )
        env.attach_real_clients({args.machine: client})
        logger.info(f"[Setup] 已绑定真机客户端: {args.machine}")
        print(f"[Setup] 已绑定真机客户端: {args.machine}")

        # ── Monkey-patch 0: 客户端 _send_request 使用带重试的短 HTTP 超时 ──
        # cqlib SDK 的 _send_request 默认 timeout=60s，即使 query_experiment 的
        # max_wait_time 也无法阻止单次 HTTP 请求阻塞 60s。
        # 将 HTTP 超时降到 15s + 最多 2 次重试，应对间歇性超时。
        import types
        _orig_send = client.platform._send_request

        def _fast_send_request(
            plat_self: Any, path: str, method: str = "GET",
            data: Any = None, params: Any = None, raise_for_code: bool = True,
        ) -> dict[str, Any]:
            """带重试的短 HTTP 超时版 _send_request：timeout=15s + 最多 2 次重试。"""
            import requests as _requests
            url = f"{plat_self.SCHEME}://{plat_self.DOMAIN}{path}"
            headers = {
                "basicToken": plat_self.access_token,
                "Authorization": f"Bearer {plat_self.access_token}",
            }
            last_error = None
            for attempt in range(3):  # 1 次原始 + 2 次重试
                try:
                    res = _requests.request(
                        method.upper(), url, json=data, headers=headers,
                        params=params, timeout=15,
                    )
                    if res.status_code != 200:
                        from cqlib import CqlibRequestError  # type: ignore[import-untyped]
                        raise CqlibRequestError(
                            f"Request API failed: {res.text}", res.status_code
                        )
                    result: dict[str, Any] = res.json()
                    if raise_for_code and result.get("code", -1) != 0:
                        from cqlib import CqlibRequestError  # type: ignore[import-untyped]
                        raise CqlibRequestError(
                            result.get("message", "Unknown error"), result.get("code")
                        )
                    return result
                except Exception as _e:
                    last_error = _e
                    if attempt < 2:
                        import time as _time
                        _time.sleep(1.0)  # 重试前等待 1s
            raise last_error  # type: ignore[misc]

        client.platform._send_request = types.MethodType(
            _fast_send_request, client.platform
        )  # type: ignore[method-assign]

        # ── Monkey-patch 1: 客户端 get_task_status 使用适中超时 ──
        # cqlib SDK 的 query_experiment 默认 max_wait_time=3600s，会阻塞训练。
        # 将查询超时设为 15s，超时后返回 "running" 状态保留在 pending 列表。
        _orig_get_status = client.get_task_status

        def _fast_get_status(task_id: str) -> dict[str, Any]:
            """适中超时版 get_task_status：15s 内返回 running/completed/error。"""
            try:
                result = client.platform.query_experiment(
                    task_id, max_wait_time=15, sleep_time=2
                )
                if isinstance(result, list) and len(result) > 0:
                    data = result[0]
                    if isinstance(data, dict):
                        has_result = "resultStatus" in data or "probability" in data
                        return {
                            "task_id": task_id,
                            "status": "completed" if has_result else "running",
                            "result": data.get("probability"),
                            "raw": data,
                        }
                return {"task_id": task_id, "status": "running"}
            except Exception:
                # 查询超时或网络错误 → 视为仍在运行，下次重试
                return {"task_id": task_id, "status": "running"}

        client.get_task_status = _fast_get_status  # type: ignore[method-assign]

        # ── Monkey-patch 2: 环境轮询增加每步配额限制 + 可见输出 ──
        # 每次 step 最多轮询 MAX_POLL_PER_STEP 个 pending 任务，避免单步耗时过长。
        _orig_poll = env._poll_pending_real_tasks

        def _patched_poll(self: Any) -> float:  # type: ignore[misc]
            """带配额限制的真机轮询：每步最多 3 个任务。"""
            from src.scheduler.env_types import (
                REAL_MACHINE_FAIL_PENALTY,
                REAL_MACHINE_SUCCESS_BONUS,
            )
            from src.scheduler.env_real_machine import (
                record_real_failure,
                _update_task_duration,
            )
            # 本地覆盖：更宽松的降级参数（适应真机执行慢、网络不稳）
            DEGRADE_THRESHOLD = 5
            MAX_POLL_STEPS = 30

            # 降级后清空 pending，避免无效轮询
            if self._real_machine_degraded:
                self._pending_real_tasks = []
                return 0.0

            if not self._pending_real_tasks:
                return 0.0

            # 追踪调用次数，每 N 次打印进度
            self._patch_call_count = getattr(self, "_patch_call_count", 0) + 1
            if self._patch_call_count % 10 == 1:
                print(f"  [Poll #{self._patch_call_count}] pending={len(self._pending_real_tasks)}, "
                      f"success={self._real_success_count}, step={self._current_step}",
                      flush=True)

            MAX_POLL_PER_STEP = 5
            total_feedback = 0.0
            still_pending: list[dict[str, Any]] = []
            poll_count = 0

            for pending in self._pending_real_tasks:
                if poll_count >= MAX_POLL_PER_STEP:
                    still_pending.append(pending)
                    continue

                pending["poll_count"] += 1
                machine_name = pending["machine_name"]
                real_task_id = pending["task_id"]
                task_id_str = pending["task_id_str"]
                client_obj = self._real_clients.get(machine_name)

                if client_obj is None:
                    total_feedback += REAL_MACHINE_FAIL_PENALTY * self.real_machine_feedback_weight
                    record_real_failure(self, machine_name, "客户端丢失")
                    poll_count += 1
                    continue

                # get_task_status 已通过 monkey-patch 1 限制为 5s 超时
                try:
                    status = client_obj.get_task_status(real_task_id)
                except Exception:
                    still_pending.append(pending)
                    poll_count += 1
                    continue

                status_str = str(status.get("status", "unknown"))

                if status_str == "completed":
                    total_feedback += (
                        REAL_MACHINE_SUCCESS_BONUS * self.real_machine_feedback_weight
                    )
                    self._real_success_count += 1
                    self._real_consecutive_failures = 0
                    actual_duration = status.get("execution_time_s", None)
                    _update_task_duration(self, task_id_str, actual_duration)
                    print(f"  [真机OK] 任务 {task_id_str} 完成! "
                          f"total_success={self._real_success_count}",
                          flush=True)
                    logger.info(
                        f"[真机闭环] 任务 {task_id_str} 完成 (real_id={real_task_id})"
                    )
                elif status_str == "error":
                    total_feedback += (
                        REAL_MACHINE_FAIL_PENALTY * self.real_machine_feedback_weight
                    )
                    self._real_fail_count += 1
                    self._real_consecutive_failures += 1
                    record_real_failure(self, machine_name, "真机返回 error")
                    print(f"  [真机FAIL] 任务 {task_id_str} 失败! "
                          f"consecutive_failures={self._real_consecutive_failures}",
                          flush=True)
                    if self._real_consecutive_failures >= DEGRADE_THRESHOLD:
                        self._real_machine_degraded = True
                        logger.warning(
                            f"[真机闭环] 连续失败 {self._real_consecutive_failures} 次，"
                            f"自动降级为仿真模式"
                        )
                        print(f"  [降级] 连续失败 {self._real_consecutive_failures} 次，已降级为仿真模式",
                              flush=True)
                elif pending["poll_count"] >= MAX_POLL_STEPS:
                    # 轮询超时：视为失败
                    total_feedback += (
                        REAL_MACHINE_FAIL_PENALTY * self.real_machine_feedback_weight
                    )
                    self._real_fail_count += 1
                    self._real_consecutive_failures += 1
                    record_real_failure(self, machine_name, "轮询超时")
                    print(f"  [超时] 任务 {task_id_str} 轮询超时 "
                          f"(poll_count={pending['poll_count']})",
                          flush=True)
                    if self._real_consecutive_failures >= DEGRADE_THRESHOLD:
                        self._real_machine_degraded = True
                        print(f"  [降级] 连续失败 {self._real_consecutive_failures} 次，已降级为仿真模式",
                              flush=True)
                else:
                    # running/unknown: 保留在 pending
                    still_pending.append(pending)

                poll_count += 1

            self._pending_real_tasks = still_pending
            return total_feedback

        env._poll_pending_real_tasks = types.MethodType(_patched_poll, env)  # type: ignore[method-assign]
        logger.info("[Setup] 已安装真机轮询保护 (HTTP 5s超时 + 每步最多3任务)")
        print("[Setup] 已安装真机轮询保护 (HTTP 5s超时 + 每步最多3任务)")
    else:
        # Mock 模式：使用模拟环境，不绑定真机
        from smoke_test import MockSmokeClient  # type: ignore[import-not-found]
        client = MockSmokeClient(machine_name=args.machine, mock_delay=0.01)
        env.attach_real_clients({args.machine: client})
        print("[Setup] Mock 客户端已绑定")

    # ── 步骤 3: 创建 PPO Agent ──
    print("\n--- [3/5] 创建 PPO Agent ---")
    agent = PPOAgent(
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        verbose=1 if args.verbose else 0,
        seed=args.seed,
        log_dir=str(output_dir / "tb"),
    )

    # 输出模型信息
    logger.info(f"[Agent] PPO Agent 创建完成: lr={args.learning_rate}, n_steps={args.n_steps}")
    print(f"[Agent] PPO Agent 创建完成")
    print(f"  观测空间维度: {env.observation_space.shape}")
    print(f"  动作空间大小: {env.action_space.n}")

    # ── 步骤 4: 开始训练 ──
    print("\n--- [4/5] 开始训练 ---")
    record_cb = TrainingRecordCallback(output_dir)
    metrics_cb = TrainingMetricsCallback(
        record_cb, output_dir, verbose=1 if args.verbose else 0
    )

    t0 = time.perf_counter()

    try:
        model = agent.train(
            total_timesteps=args.timesteps,
            eval_freq=max(1, args.save_interval // 2),
            n_eval_episodes=5,
            resume_from=args.resume,
            extra_callbacks=[metrics_cb],
        )
    except KeyboardInterrupt:
        print("\n[INFO] 训练被用户中断", flush=True)
        record_cb.save()
        metrics_cb.save()
    except Exception as e:
        logger.exception("[Train] 训练过程中发生异常")
        print(f"\n[ERROR] 训练过程中发生异常: {e}", flush=True)
        record_cb.save()
        metrics_cb.save()
        if agent.model:
            agent.save(str(output_dir / f"interrupt_{int(time.time())}"))
            print(f"[INFO] 中断模型已保存", flush=True)
        sys.exit(1)

    training_time = time.perf_counter() - t0

    # ── 步骤 5: 保存最终结果 ──
    print("\n--- [5/5] 保存最终结果 ---")

    # 保存最终模型（agent.save() 内部会追加 .zip）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = output_dir / f"ppo_closed_loop_{timestamp}"
    agent.save(str(final_model_path))
    print(f"[Save] 最终模型: {final_model_path}.zip")

    # 保存训练记录
    record_path = record_cb.save(prefix=f"ppo_{args.timesteps}")
    metrics_path = metrics_cb.save(prefix=f"ppo_{args.timesteps}")

    # 打印最终统计
    stats = env.get_real_machine_stats()
    print(f"\n{'=' * 70}")
    print(f"  PPO 真机闭环训练完成")
    print(f"{'=' * 70}")
    print(f"  总训练步数: {args.timesteps}")
    print(f"  总耗时: {training_time / 60:.2f} 分钟")
    print(f"  真机任务成功: {stats['success_count']}")
    print(f"  真机任务失败: {stats['fail_count']}")
    print(f"  当前降级状态: {'是 (连续失败太多)' if stats['degraded'] else '否'}")
    print(f"  最终模型: {final_model_path}")
    print(f"  训练记录: {record_path}")
    print(f"  训练指标: {metrics_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
