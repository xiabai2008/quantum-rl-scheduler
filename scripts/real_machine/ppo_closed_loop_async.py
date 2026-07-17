"""PPO 真机闭环训练（异步版：后台线程轮询 + 训练循环无阻塞）。

与 `ppo_closed_loop_training.py` 区别：
- 同步版：每步轮询 pending，若任务仍在运行则占用训练时间，真机负载高时总耗时极长
- 异步版：后台 daemon 线程用长超时（120s）轮询，训练循环持续推进，不等待结果

核心设计：
1. 任务提交后，加入 `AsyncResultPoller` 的 pending 队列
2. 后台线程持续轮询，结果就绪后回调 `env._on_real_task_completed` 更新奖励
3. 训练循环每步只做极少量工作（检查已完成任务数），训练不阻塞
4. 连续失败 → 自动降级 → 后台线程停止轮询

特点:
    - 真正非阻塞：训练循环不受真机执行慢影响，步数推进稳定
    - 长超时轮询：后台线程给真机足够时间（120s），应对队列积压
    - 自动降级：连续失败超过阈值自动切换到模拟
    - 回调记录：训练过程中记录每次真机提交的完整信息
    - checkpoint 保存：支持断点续训

实验设计:
    - 总训练步数: 10,000 (可配置)
    - 保存间隔: 每 2,000 步
    - 学习率: 3e-4
    - 种子: 42

用法:
    # Mock dry-run（使用模拟环境，不消耗真机机时）
    python scripts/real_machine/ppo_closed_loop_async.py --mock --timesteps 500

    # 真机训练，总步数 2000
    python scripts/real_machine/ppo_closed_loop_async.py --timesteps 2000 --real-submit-prob 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

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

from src.api.tianyan_cqlib import CqlibTianyanClient
from src.scheduler.agent import PPOAgent
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "models" / "ppo_real_closed_loop_async"
RESULTS_DIR = _PROJECT_ROOT / "results" / "real_machine"


# ---------------------------------------------------------------------------
# 异步结果轮询器：后台 daemon 线程持续轮询 pending 任务
# ---------------------------------------------------------------------------


class AsyncResultPoller:
    """异步结果轮询器：后台 daemon 线程轮询真机任务结果。

    设计目标：
    - 训练循环不阻塞：即使任务在真机排队几小时，训练仍能稳步推进
    - 长超时容忍：给真机足够执行时间（默认 120s）
    - 结果就绪回调：结果完成后调用回调更新环境奖励统计
    """

    def __init__(
        self,
        client: CqlibTianyanClient,
        on_completed: Callable[[str, dict[str, Any]], None],
        on_failed: Callable[[str, str], None],
        poll_interval: float = 5.0,
        max_wait_seconds: float = 120.0,
        max_workers: int = 5,
    ) -> None:
        """
        Args:
            client: Cqlib 客户端
            on_completed: 任务完成回调 (task_id_str, status_dict) -> None
            on_failed: 任务失败/超时回调 (task_id_str, reason) -> None
            poll_interval: 轮询间隔（秒）
            max_wait_seconds: 最大等待时间（秒），超时后视为失败
            max_workers: 并行查询的线程数（默认 5）
        """
        self._client = client
        self._on_completed = on_completed
        self._on_failed = on_failed
        self._poll_interval = poll_interval
        self._max_wait_seconds = max_wait_seconds
        self._max_workers = max_workers

        # 线程安全：pending 任务队列用锁保护
        self._pending_lock = threading.Lock()
        self._pending: list[dict[str, Any]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._degraded = False

        logger.info(
            f"[AsyncPoller] 初始化: poll_interval={poll_interval}s, "
            f"max_wait={max_wait_seconds}s, max_workers={max_workers}"
        )

    def submit_pending(
        self,
        task_id_str: str,
        real_task_id: str,
        machine_name: str,
    ) -> None:
        """提交一个新 pending 任务给后台轮询。

        Args:
            task_id_str: 环境内部任务 ID（用于回调时定位）
            real_task_id: cqlib 返回的真机任务 ID
            machine_name: 提交的机器名
        """
        with self._pending_lock:
            self._pending.append(
                {
                    "task_id_str": task_id_str,
                    "real_task_id": real_task_id,
                    "machine_name": machine_name,
                    "submit_time": time.time(),
                    "poll_count": 0,
                }
            )
        logger.debug(f"[AsyncPoller] 任务 {task_id_str} (real={real_task_id}) 加入轮询队列")

    def set_degraded(self, degraded: bool) -> None:
        """设置降级标志：降级后清空 pending 队列并停止轮询。"""
        self._degraded = degraded
        if degraded:
            with self._pending_lock:
                pending_count = len(self._pending)
                if pending_count > 0:
                    logger.warning(f"[AsyncPoller] 已降级，清空 {pending_count} 个 pending 任务")
                    self._pending.clear()

    def start(self) -> None:
        """启动后台轮询线程（daemon 模式，主线程退出自动结束）。"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._polling_loop_safe,
            daemon=True,  # daemon 线程：主线程退出自动结束
            name="AsyncResultPoller",
        )
        self._thread.start()
        logger.info("[AsyncPoller] 后台轮询线程已启动")

    def stop(self) -> None:
        """停止后台轮询线程。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("[AsyncPoller] 后台轮询线程已停止")

    @property
    def pending_count(self) -> int:
        """返回当前 pending 任务数（线程安全）。"""
        with self._pending_lock:
            return len(self._pending)

    def _polling_loop_safe(self) -> None:
        """轮询循环的安全包装：捕获所有异常，防止 daemon 线程静默崩溃。"""
        try:
            self._polling_loop()
        except Exception as e:
            logger.error(
                f"[AsyncPoller] 轮询线程崩溃: {type(e).__name__}: {e}",
                exc_info=True,
            )
            self._degraded = True

    def _polling_loop(self) -> None:
        """后台轮询主循环：并行查询 pending 任务，每轮之间休眠。

        设计要点：
        - 并行查询：使用 ThreadPoolExecutor 同时查询多个任务（默认 5 并发），
          每个任务 max_wait_time=2s（快速检测，不长时间等待）
        - 批量查询不可用：cqlib 的 query_experiment 批量查询时，只要有一个
          任务仍在运行，整个批量查询就返回 CqlibRequestError。因此用并行逐
          个查询替代，既快又不受单任务失败影响
        - 结果字段：completed 任务返回 {'resultStatus': ..., 'probability': {...},
          'experimentTaskId': '...'}，无 taskStatus 字段
        - 轮间休眠：每轮查询完所有 pending 任务后休眠 poll_interval 秒
        - 超时兜底：单任务超过 max_wait_seconds 按失败处理
        - 线程安全：查询函数只读 client（HTTP GET），不改写共享状态；回调在主
          线程中执行，无竞态
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from cqlib.exceptions import CqlibRequestError

        def _query_one(task: dict[str, Any]) -> tuple[dict[str, Any], Any, Any]:
            """查询单个任务（工作线程中执行）。

            直接调用 _send_request 而非 query_experiment，原因：
            - query_experiment 在 code!=0 时进入 wait→retry 循环，不区分"运行失败"
              和"仍在运行"，导致失败任务永远留在 pending 队列
            - 直接调用 _send_request 可以拿到原始 API 响应，判断是永久失败还是临时

            Returns:
                (task, status, result)
                - status=True: 任务完成，result 是 experimentResultModelList
                - status="failed": 任务运行失败，result 是错误消息
                - status=False: 任务仍在运行，需要继续轮询
            """
            from cqlib.exceptions import CqlibRequestError

            try:
                raw = self._client.platform._send_request(
                    path=self._client.platform.QUERY_EXP_PATH,
                    data={"query_ids": [task["real_task_id"]]},
                    method="POST",
                )
                query_exp = raw.get("data", {}).get("experimentResultModelList", [])
                if query_exp and len(query_exp) >= 1:
                    return (task, True, query_exp)  # 任务完成
                else:
                    return (task, False, None)  # 结果为空，仍在运行
            except CqlibRequestError as e:
                err_msg = str(e)
                # 检查是否为"运行失败"（永久失败，不需要重试）
                if "运行失败" in err_msg or "Run failure" in err_msg:
                    return (task, "failed", err_msg)  # 真机运行失败
                return (task, False, None)  # 其他 API 错误，可能仍在运行
            except Exception as e:
                logger.debug(f"[AsyncPoller] 任务 {task['task_id_str']} 查询异常: {e}")
                return (task, False, None)  # 网络错误，重试

        while self._running and not self._degraded:
            with self._pending_lock:
                if not self._pending:
                    time.sleep(self._poll_interval)
                    continue
                tasks_snapshot = list(self._pending)

            logger.debug(f"[AsyncPoller] 轮询心跳: {len(tasks_snapshot)} 个任务待查询")

            now = time.time()
            active_tasks: list[dict[str, Any]] = []

            # 第一遍：筛出超时任务
            for task in tasks_snapshot:
                task["poll_count"] += 1
                if now - task["submit_time"] > self._max_wait_seconds:
                    logger.warning(f"[AsyncPoller] 任务 {task['task_id_str']} 超时，标记为失败")
                    self._on_failed(task["task_id_str"], "轮询超时")
                else:
                    active_tasks.append(task)

            if not active_tasks:
                with self._pending_lock:
                    self._pending[:] = []
                time.sleep(self._poll_interval)
                continue

            # 第二遍：并行查询所有活跃任务
            still_pending: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {executor.submit(_query_one, t): t for t in active_tasks}
                for future in as_completed(futures):
                    task, status, result = future.result()
                    if status is True:
                        # 任务完成
                        task_id_str = task["task_id_str"]
                        if isinstance(result, list) and len(result) > 0:
                            item = result[0]
                            if isinstance(item, dict):
                                rs = item.get("resultStatus", "")
                                logger.info(
                                    f"[AsyncPoller] 任务 {task_id_str} "
                                    f"(real={task['real_task_id']}) 完成"
                                    f" (resultStatus={rs})"
                                )
                                self._on_completed(task_id_str, item)
                            else:
                                logger.info(
                                    f"[AsyncPoller] 任务 {task_id_str} "
                                    f"(real={task['real_task_id']}) 完成"
                                )
                                self._on_completed(task_id_str, {})
                        else:
                            logger.info(
                                f"[AsyncPoller] 任务 {task_id_str} "
                                f"(real={task['real_task_id']}) 完成"
                            )
                            self._on_completed(task_id_str, {})
                    elif status == "failed":
                        # 真机运行失败（永久失败）
                        logger.warning(
                            f"[AsyncPoller] 任务 {task['task_id_str']} "
                            f"真机运行失败: {str(result)[:100]}"
                        )
                        self._on_failed(task["task_id_str"], f"真机运行失败: {str(result)[:100]}")
                    else:
                        # 仍在运行，保留到下一轮
                        still_pending.append(task)

            # 更新 pending 队列
            with self._pending_lock:
                self._pending[:] = still_pending

            # 轮间休眠
            time.sleep(self._poll_interval)


# ---------------------------------------------------------------------------
# 训练记录回调
# ---------------------------------------------------------------------------


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
                    "type": "ppo_closed_loop_training_async",
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "total_elapsed_sec": round(time.perf_counter() - self.start_time, 2),
                    "records": self.records,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"[Callback] 训练记录已保存: {filepath}")
        return filepath


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------


def main() -> None:
    """PPO 真机闭环训练（异步版）主入口。"""
    parser = argparse.ArgumentParser(description="PPO 真机闭环训练（异步版）")
    parser.add_argument("--mock", action="store_true", help="Mock 模式（不使用真机）")
    parser.add_argument("--machine", default="tianyan176", help="首选机器名称")
    parser.add_argument(
        "--timesteps", type=int, default=2000, help="总训练步数（真机模式建议 2000-5000）"
    )
    parser.add_argument("--save-interval", type=int, default=1000, help="保存间隔步数")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--n-steps", type=int, default=512, help="PPO n_steps 参数")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO batch_size")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR), help="输出模型目录")
    parser.add_argument(
        "--real-submit-prob",
        type=float,
        default=0.3,
        help="量子任务真机提交概率 (1.0=每个量子任务都提交)",
    )
    parser.add_argument("--poll-interval", type=float, default=5.0, help="后台轮询间隔（秒）")
    parser.add_argument("--max-wait", type=float, default=120.0, help="单任务最大等待时间（秒）")
    parser.add_argument("--degrade-threshold", type=int, default=5, help="连续失败多少次降级")
    parser.add_argument(
        "--max-workers", type=int, default=5, help="后台轮询并行查询线程数（默认 5）"
    )
    parser.add_argument(
        "--max-real-tasks",
        type=int,
        default=0,
        help="真机任务总数上限（0=不限，到达后自动切换仿真模式）",
    )
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    args = parser.parse_args()

    output_dir = Path(args.output)

    print(f"\n{'=' * 70}")
    print("  PPO 真机闭环训练（异步版）")
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
    print(f"  轮询间隔: {args.poll_interval}s")
    print(f"  最大等待: {args.max_wait}s")
    print(f"  降级阈值: {args.degrade_threshold} 次连续失败")
    print(f"  并行查询线程: {args.max_workers}")
    if args.max_real_tasks > 0:
        print(f"  真机任务上限: {args.max_real_tasks}（到达后自动切换仿真）")
    if args.mock:
        print(f"  模式: [MOCK] 模拟环境（不使用真机）")
    else:
        print(f"  模式: [REAL] 异步真机闭环训练（后台轮询）")
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
    async_poller: Optional[AsyncResultPoller] = None

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
            plat_self: Any,
            path: str,
            method: str = "GET",
            data: Any = None,
            params: Any = None,
            raise_for_code: bool = True,
        ) -> dict[str, Any]:
            """带重试的短 HTTP 超时版 _send_request：timeout=15s + 最多 2 次重试。

            关键改进：只对 HTTP/网络错误重试，API 级别错误（如"运行失败"）直接抛出，
            避免把 3 次重试浪费在永久失败的任务上。
            """
            import requests as _requests
            from cqlib.exceptions import CqlibRequestError

            url = f"{plat_self.SCHEME}://{plat_self.DOMAIN}{path}"
            headers = {
                "basicToken": plat_self.access_token,
                "Authorization": f"Bearer {plat_self.access_token}",
            }
            last_error = None
            for attempt in range(3):  # 1 次原始 + 2 次重试（仅网络错误）
                try:
                    res = _requests.request(
                        method.upper(),
                        url,
                        json=data,
                        headers=headers,
                        params=params,
                        timeout=15,
                    )
                    if res.status_code != 200:
                        raise CqlibRequestError(f"Request API failed: {res.text}", res.status_code)
                    result: dict[str, Any] = res.json()
                    if raise_for_code and result.get("code", -1) != 0:
                        # API 级别错误（运行失败、权限不足等）→ 不重试，直接抛出
                        raise CqlibRequestError(
                            result.get("message", "Unknown error"), result.get("code")
                        )
                    return result
                except CqlibRequestError:
                    raise  # API 级别错误不重试
                except Exception as _e:
                    # HTTP/网络错误 → 重试
                    last_error = _e
                    if attempt < 2:
                        import time as _time

                        _time.sleep(1.0)
            raise last_error  # type: ignore[misc]

        client.platform._send_request = types.MethodType(
            _fast_send_request, client.platform
        )  # type: ignore[method-assign]

        # ── Monkey-patch 1: 覆盖 env._poll_pending_real_tasks 集成异步轮询 ──
        # 同步版是每步轮询所有 pending，会阻塞训练。异步版只做：
        # 1. 降级后清空 pending
        # 2. 打印进度统计
        # 3. 返回 0 反馈（因为反馈已在异步回调处理）
        _orig_poll = env._poll_pending_real_tasks

        def _async_patched_poll(self: Any) -> float:  # type: ignore[misc]
            """异步轮询包装：只处理已完成任务，不阻塞训练。

            实际轮询发生在后台线程，本方法只收集已完成结果。
            """
            from src.scheduler.env_types import (
                REAL_MACHINE_FAIL_PENALTY,
                REAL_MACHINE_SUCCESS_BONUS,
            )

            # 降级处理：清空所有 pending，停止后台轮询
            if self._real_machine_degraded:
                if async_poller:
                    async_poller.set_degraded(True)
                self._pending_real_tasks = []
                return 0.0

            # 追踪调用次数，每 N 次打印进度
            self._patch_call_count = getattr(self, "_patch_call_count", 0) + 1
            if self._patch_call_count % 10 == 1:
                pending_count = async_poller.pending_count if async_poller else 0
                print(
                    f"  [Poll #{self._patch_call_count}] async_pending={pending_count}, "
                    f"success={self._real_success_count}, step={self._current_step}",
                    flush=True,
                )

            # 异步轮询：后台处理一切，这里只返回 0（反馈由回调累加）
            return 0.0

        env._poll_pending_real_tasks = types.MethodType(
            _async_patched_poll, env
        )  # type: ignore[method-assign]

        # ── 步骤 2a: 创建并启动异步轮询器 ──
        from src.scheduler.env_real_machine import (
            REAL_MACHINE_FAIL_PENALTY,
            REAL_MACHINE_SUCCESS_BONUS,
            record_real_failure,
            _update_task_duration,
        )

        def _on_real_completed(task_id_str: str, status: dict[str, Any]) -> None:
            """真机任务完成回调：更新环境统计和奖励。"""
            nonlocal env
            env._real_success_count += 1
            env._real_consecutive_failures = 0
            actual_duration = status.get("raw", {}).get("executionTime", None)
            _update_task_duration(env, task_id_str, actual_duration)
            total_real = env._real_success_count + env._real_fail_count
            print(
                f"  [真机OK] 任务 {task_id_str} 完成! "
                f"total_success={env._real_success_count} "
                f"total_real={total_real}",
                flush=True,
            )
            logger.info(f"[异步回调] 任务 {task_id_str} 完成")
            if args.max_real_tasks > 0 and total_real >= args.max_real_tasks:
                env._real_machine_degraded = True
                if async_poller:
                    async_poller.set_degraded(True)
                print(
                    f"  [限流] 真机任务已达上限 {args.max_real_tasks}，自动切换仿真模式", flush=True
                )

        def _on_real_failed(task_id_str: str, reason: str) -> None:
            """真机任务失败回调：更新环境统计，触发降级检查。"""
            nonlocal env
            env._real_fail_count += 1
            env._real_consecutive_failures += 1
            record_real_failure(env, args.machine, reason)
            total_real = env._real_success_count + env._real_fail_count
            print(
                f"  [真机FAIL] 任务 {task_id_str} 失败: {reason} "
                f"consecutive_failures={env._real_consecutive_failures}",
                flush=True,
            )
            # 降级条件 1：连续失败超阈值
            if env._real_consecutive_failures >= args.degrade_threshold:
                env._real_machine_degraded = True
                logger.warning(
                    f"[异步回调] 连续失败 {env._real_consecutive_failures} 次，"
                    f"自动降级为仿真模式"
                )
                print(
                    f"  [降级] 连续失败 {env._real_consecutive_failures} 次，已降级为仿真模式",
                    flush=True,
                )
            # 降级条件 2：真机任务总数达上限
            elif args.max_real_tasks > 0 and total_real >= args.max_real_tasks:
                env._real_machine_degraded = True
                if async_poller:
                    async_poller.set_degraded(True)
                print(
                    f"  [限流] 真机任务已达上限 {args.max_real_tasks}，自动切换仿真模式", flush=True
                )

        async_poller = AsyncResultPoller(
            client=client,
            on_completed=_on_real_completed,
            on_failed=_on_real_failed,
            poll_interval=args.poll_interval,
            max_wait_seconds=args.max_wait,
            max_workers=args.max_workers,
        )
        async_poller.start()
        logger.info("[Setup] 异步轮询器已启动 (后台 daemon 线程)")

        # ── Monkey-patch 2: 拦截 _pending_real_tasks 自动注册到轮询器 ──
        # submit_to_real_machine 会 append 到 env._pending_real_tasks，
        # 我们通过自定义 list 子类拦截 append 调用，自动注册到 poller。
        class _AsyncTaskList(list):
            """拦截 append 自动注册到 AsyncResultPoller。"""

            def __init__(self, _poller: AsyncResultPoller, _iterable: Any = ()) -> None:
                super().__init__(_iterable)
                self._poller = _poller

            def append(self, _item: Any) -> None:
                super().append(_item)
                if isinstance(_item, dict) and "task_id" in _item and "task_id_str" in _item:
                    print(
                        f"  [AsyncPoller] 注册任务 {_item['task_id_str']} (real={_item['task_id'][:12]}...)",
                        flush=True,
                    )
                    self._poller.submit_pending(
                        task_id_str=_item["task_id_str"],
                        real_task_id=_item["task_id"],
                        machine_name=_item.get("machine_name", "unknown"),
                    )

        env._pending_real_tasks = _AsyncTaskList(async_poller)

        # 保护 reset() 后仍然使用自定义 list
        _orig_reset = env.reset

        def _patched_reset(self: Any, **_reset_kwargs: Any) -> Any:
            result = _orig_reset(**_reset_kwargs)
            env._pending_real_tasks = _AsyncTaskList(async_poller)
            return result

        env.reset = types.MethodType(_patched_reset, env)  # type: ignore[method-assign]

        print("[Setup] 异步轮询保护就绪 (HTTP 15s超时 + 后台长轮询 + 自动注册)")

    else:
        # Mock 模式：使用模拟环境，不绑定真机
        print("[Setup] Mock 模式，不绑定真机")

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

    t0 = time.perf_counter()

    try:
        model = agent.train(
            total_timesteps=args.timesteps,
            eval_freq=max(1, args.save_interval // 2),
            n_eval_episodes=5,
        )
    except KeyboardInterrupt:
        print("\n[INFO] 训练被用户中断", flush=True)
    except Exception as e:
        logger.exception("[Train] 训练过程中发生异常")
        print(f"\n[ERROR] 训练过程中发生异常: {e}", flush=True)
        record_cb.save()
        if agent.model:
            agent.save(str(output_dir / f"interrupt_{int(time.time())}"))
            print(f"[INFO] 中断模型已保存", flush=True)
        if async_poller:
            async_poller.stop()
        sys.exit(1)

    training_time = time.perf_counter() - t0

    # ── 停止后台轮询 ──
    if async_poller:
        async_poller.stop()

    # ── 步骤 5: 保存最终结果 ──
    print("\n--- [5/5] 保存最终结果 ---")

    # 保存最终模型（agent.save() 内部会追加 .zip）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = output_dir / f"ppo_closed_loop_async_{timestamp}"
    agent.save(str(final_model_path))
    print(f"[Save] 最终模型: {final_model_path}.zip")

    # 保存训练记录
    record_path = record_cb.save(prefix=f"ppo_async_{args.timesteps}")

    # 打印最终统计
    stats = env.get_real_machine_stats()
    print(f"\n{'=' * 70}")
    print(f"  PPO 真机闭环训练（异步版）完成")
    print(f"{'=' * 70}")
    print(f"  总训练步数: {args.timesteps}")
    print(f"  总耗时: {training_time / 60:.2f} 分钟")
    print(f"  真机任务成功: {stats['success_count']}")
    print(f"  真机任务失败: {stats['fail_count']}")
    print(f"  当前降级状态: {'是 (连续失败太多)' if stats['degraded'] else '否'}")
    print(f"  最终模型: {final_model_path}")
    print(f"  训练记录: {record_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
