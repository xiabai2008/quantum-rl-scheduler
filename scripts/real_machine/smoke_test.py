"""天衍-176 真机冒烟测试 + 基线数据收集（阶段 0）。

验证天衍云真机连通性，采集单/双/多比特门基线性能。
共 4 类实验 x 3 次重复 = 12 个真机任务。

实验设计:
    S1: H门     -- 单比特门基线，理论 P(0)=P(1)=0.5
    S2: Bell态  -- 双比特纠缠，理论 P(00)=P(11)=0.5
    S3: GHZ态   -- 三比特纠缠，理论 P(000)=P(111)=0.5
    S4: T门链   -- 相位累积精度，理论 P(0)=0.85, P(1)=0.15

用法:
    # Mock dry-run（不消耗真机机时）
    python scripts/real_machine/smoke_test.py --mock

    # 真机执行
    python scripts/real_machine/smoke_test.py

    # 指定首选机器
    python scripts/real_machine/smoke_test.py --machine tianyan176
"""

from __future__ import annotations

import json
import os
import sys
import threading
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
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from loguru import logger

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 机器故障切换链（按优先级）
MACHINE_CHAIN: list[str] = [
    "tianyan176",
    "tianyan176-2",
    "tianyan_sw",
    "tianyan_s",
]

# 实验定义
EXPERIMENTS: list[dict[str, Any]] = [
    {
        "id": "S1_H_gate",
        "name": "H门基线",
        "qcis": "H Q0\nM Q0",
        "shots": 1024,
        "repeats": 3,
        "theoretical": {"0": 0.5, "1": 0.5},
        "description": "单比特 Hadamard 门，验证测量随机性",
    },
    {
        "id": "S2_Bell_state",
        "name": "Bell态纠缠",
        "qcis": "H Q0\nCZ Q0 Q1\nM Q0 Q1",
        "shots": 1024,
        "repeats": 3,
        "theoretical": {"00": 0.5, "11": 0.5},
        "description": "双比特 Bell 态，验证纠缠保真度",
    },
    {
        "id": "S3_GHZ_state",
        "name": "GHZ态纠缠",
        "qcis": "H Q0\nCZ Q0 Q1\nCZ Q1 Q2\nM Q0 Q1 Q2",
        "shots": 1024,
        "repeats": 3,
        "theoretical": {"000": 0.5, "111": 0.5},
        "description": "三比特 GHZ 态，验证多体纠缠",
    },
    {
        "id": "S4_T_gate_chain",
        "name": "T门相位累积",
        "qcis": "H Q0\nT Q0\nT Q0\nT Q0\nM Q0",
        "shots": 1024,
        "repeats": 3,
        "theoretical": {"0": 0.8536, "1": 0.1464},
        "description": "三次 T 门相位累积 (3*pi/4)，验证相位门精度",
    },
]

# 结果输出目录
RESULTS_DIR = _PROJECT_ROOT / "results" / "real_machine"


# ---------------------------------------------------------------------------
# Mock 客户端（用于 dry-run，接口与 CqlibTianyanClient 一致）
# ---------------------------------------------------------------------------


class MockSmokeClient:
    """模拟真机客户端，接口与 CqlibTianyanClient 一致。

    用于 Mock dry-run，不消耗真机机时。
    根据实验类型返回接近理论值的模拟概率分布。

    Attributes:
        machine_name: 当前机器名称
        mock_delay: 模拟延迟（秒）
    """

    def __init__(self, machine_name: str = "tianyan176", mock_delay: float = 0.1) -> None:
        """初始化 Mock 客户端。

        Args:
            machine_name: 机器名称
            mock_delay: 模拟延迟秒数
        """
        self.machine_name = machine_name
        self.mock_delay = mock_delay
        logger.info(f"[Mock] 冒烟测试 Mock 客户端初始化, machine={machine_name}")

    def submit_quantum_task(
        self,
        qcis: str = "",
        circuit: Any = None,
        shots: int = 1024,
        task_name: str = "SmokeTest",
    ) -> str:
        """Mock 提交量子任务，返回虚拟 task_id。

        Args:
            qcis: QCIS 指令字符串
            circuit: Circuit 对象（与 qcis 二选一）
            shots: 测量次数
            task_name: 任务名称

        Returns:
            虚拟 task_id 字符串
        """
        time.sleep(self.mock_delay)
        task_id = f"mock-{int(time.time() * 1000) % 1000000}"
        logger.debug(f"[Mock] 提交任务: {task_name}, task_id={task_id}")
        return task_id

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Mock 查询任务状态，直接返回完成。

        返回 resultStatus 格式（与真机一致），用于验证概率解析逻辑。

        Args:
            task_id: 任务 ID

        Returns:
            模拟状态字典
        """
        # 模拟 1024 次测量，50/50 分布
        import random

        random.seed(hash(task_id) % 2**32)
        shots = 1024
        result_status = [[random.randint(0, 1)] for _ in range(shots)]
        return {
            "task_id": task_id,
            "status": "completed",
            "result": None,
            "raw": {
                "resultStatus": result_status,
                "probability": '{"0": 0.5, "1": 0.5}',
            },
        }

    def wait_for_task(
        self, task_id: str, timeout: int = 300, poll_interval: int = 5
    ) -> dict[str, Any]:
        """Mock 等待任务完成，返回模拟结果。

        Args:
            task_id: 任务 ID
            timeout: 超时秒数
            poll_interval: 轮询间隔

        Returns:
            模拟结果字典
        """
        time.sleep(self.mock_delay)
        return self.get_task_status(task_id)

    def list_backends(self) -> list[dict[str, Any]]:
        """Mock 查询机器列表。"""
        return [
            {"id": "1", "type": "superconducting", "status": "running", "name": "tianyan176"},
        ]


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------


def parse_probability(raw_result: Any) -> dict[str, float]:
    """从真机返回结果中解析测量概率分布。

    天衍云返回格式可能为:
        - dict: {"0": 0.5, "1": 0.5}
        - list: [["0", 0.5], ["1", 0.5]]
        - str (JSON): '{"0": 0.5, "1": 0.5}'
        - str (CSV): "0:0.5,1:0.5"

    Args:
        raw_result: 原始结果数据

    Returns:
        概率分布字典 {bitstring: probability}
    """
    if raw_result is None:
        return {}

    # 已经是 dict
    if isinstance(raw_result, dict):
        result: dict[str, float] = {}
        for k, v in raw_result.items():
            try:
                result[str(k)] = float(v)
            except (ValueError, TypeError):
                continue
        return result

    # list 格式 [["0", 0.5], ["1", 0.5]]
    if isinstance(raw_result, list):
        result = {}
        for item in raw_result:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    result[str(item[0])] = float(item[1])
                except (ValueError, TypeError):
                    continue
            elif isinstance(item, dict):
                key = item.get("key", item.get("state", ""))
                val = item.get("value", item.get("probability", 0))
                if key:
                    try:
                        result[str(key)] = float(val)
                    except (ValueError, TypeError):
                        continue
        return result

    # 字符串格式
    if isinstance(raw_result, str):
        # 尝试 JSON 解析
        raw_str = raw_result.strip()
        if raw_str.startswith("{"):
            try:
                parsed = json.loads(raw_str)
                if isinstance(parsed, dict):
                    return {
                        str(k): float(v) for k, v in parsed.items() if isinstance(v, (int, float))
                    }
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # 尝试 CSV 格式 "0:0.5,1:0.5"
        result = {}
        for pair in raw_str.split(","):
            pair = pair.strip()
            if ":" in pair:
                k, v = pair.split(":", 1)
                try:
                    result[k.strip()] = float(v.strip())
                except ValueError:
                    continue
        return result

    return {}


def compute_probability_from_shots(result_status: Any) -> dict[str, float]:
    """从 resultStatus 原始测量数据统计概率分布。

    天衍云返回的 resultStatus 格式为每次测量的结果列表:
        单比特: [[0], [1], [0], [0], [1], ...]
        双比特: [[0, 1], [1, 0], [0, 0], ...]
        三比特: [[0, 0, 1], [1, 1, 0], ...]

    Args:
        result_status: resultStatus 字段的值（列表的列表）

    Returns:
        概率分布字典 {bitstring: probability}
    """
    if not isinstance(result_status, list) or len(result_status) == 0:
        return {}

    counts: dict[str, int] = {}
    total = 0

    for shot in result_status:
        if isinstance(shot, list):
            # 将 [0, 1, 0] 转为 "010"
            bitstring = "".join(str(int(bit)) for bit in shot)
        elif isinstance(shot, (int, float)):
            bitstring = str(int(shot))
        else:
            continue
        counts[bitstring] = counts.get(bitstring, 0) + 1
        total += 1

    if total == 0:
        return {}

    return {k: round(v / total, 6) for k, v in counts.items()}


def compute_measurement_error(measured: dict[str, float], theoretical: dict[str, float]) -> float:
    """计算测量误差。

    测量误差 = sum(|P_measured(k) - P_theoretical(k)|) / 2

    Args:
        measured: 实测概率分布
        theoretical: 理论概率分布

    Returns:
        测量误差值（0-1）
    """
    all_keys = set(list(measured.keys()) + list(theoretical.keys()))
    total_diff = 0.0
    for k in all_keys:
        m_val = measured.get(k, 0.0)
        t_val = theoretical.get(k, 0.0)
        total_diff += abs(m_val - t_val)
    return total_diff / 2.0


def compute_fidelity(measured: dict[str, float], theoretical: dict[str, float]) -> float:
    """计算保真度。

    保真度 = 1 - measurement_error

    对于 Bell/GHZ 态，保真度 = (P(00) + P(11)) / 2 或 (P(000) + P(111)) / 2

    Args:
        measured: 实测概率分布
        theoretical: 理论概率分布

    Returns:
        保真度值（0-1）
    """
    error = compute_measurement_error(measured, theoretical)
    return max(0.0, 1.0 - error)


# ---------------------------------------------------------------------------
# 自定义轮询（绕过 CqlibTianyanClient.wait_for_task 对失败任务的卡死问题）
# ---------------------------------------------------------------------------

# 失败任务错误关键词
_FAILURE_KEYWORDS = ("failed", "失败", "error", 'code":1', "code':1")


def _is_task_failed(status_result: dict[str, Any]) -> bool:
    """检测任务是否在真机上执行失败。

    cqlib SDK 的 query_experiment 对失败任务返回错误 dict 而非抛异常，
    导致 get_task_status 返回 "unknown" 状态。此函数检查 raw 结果中
    是否包含失败关键词。

    Args:
        status_result: get_task_status 返回的状态字典

    Returns:
        bool: 检测到失败返回 True
    """
    if status_result.get("status") == "error":
        return True

    raw = status_result.get("raw")
    if raw is None:
        return False

    raw_str = str(raw).lower()
    for kw in _FAILURE_KEYWORDS:
        if kw.lower() in raw_str:
            return True

    # 检查 error 字段
    if status_result.get("error"):
        return True

    return False


def _get_status_with_timeout(client: Any, task_id: str, timeout: int = 15) -> dict[str, Any] | None:
    """在单独 daemon 线程中调用 get_task_status，带超时。

    cqlib SDK 的 query_experiment 对失败任务会进入无限内部重试，
    无法通过信号或线程 API 中断。使用 daemon 线程 + join(timeout)
    可以让主线程在超时后继续执行，旧线程在后台自动消亡。

    Args:
        client: 真机客户端
        task_id: 任务 ID
        timeout: 超时秒数

    Returns:
        状态字典，超时返回 None
    """
    result: list[Any] = [None]

    def worker() -> None:
        """工作线程函数。"""
        try:
            result[0] = client.get_task_status(task_id)
        except Exception as e:
            result[0] = {"status": "error", "error": str(e)}

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        return None
    return result[0]  # type: ignore[return-value]


def poll_task_result(
    client: Any,
    task_id: str,
    timeout: int = 180,
    poll_interval: int = 3,
    max_unknown: int = 5,
    per_poll_timeout: int = 15,
) -> dict[str, Any]:
    """自定义轮询等待任务完成，正确处理失败任务。

    与 CqlibTianyanClient.wait_for_task 不同，此函数能检测真机上
    执行失败的任务（cqlib SDK 返回错误 dict 而非抛异常的情况）。
    使用 daemon 线程超时包裹 get_task_status，防止 SDK 内部无限重试
    阻塞后续任务的状态查询。

    Args:
        client: 真机客户端
        task_id: 任务 ID
        timeout: 总超时秒数
        poll_interval: 轮询间隔秒数
        max_unknown: 连续 unknown 状态最大次数，超过视为失败
        per_poll_timeout: 单次 get_task_status 调用超时秒数

    Returns:
        任务结果字典
    """
    start = time.time()
    unknown_count = 0

    while time.time() - start < timeout:
        status = _get_status_with_timeout(client, task_id, per_poll_timeout)

        if status is None:
            # SDK 卡在内部重试，任务大概率已失败
            logger.warning(
                f"[Poll] get_task_status 超时 ({per_poll_timeout}s)，SDK 可能卡在内部重试，视为失败"
            )
            return {
                "task_id": task_id,
                "status": "failed",
                "error": f"Poll timeout ({per_poll_timeout}s) - SDK stuck",
            }

        if status.get("status") == "completed":
            return status

        if _is_task_failed(status):
            return {
                "task_id": task_id,
                "status": "failed",
                "error": f"Task failed on quantum machine: {str(status.get('raw', ''))[:200]}",
                "raw": status.get("raw"),
            }

        if status.get("status") == "unknown":
            unknown_count += 1
            if unknown_count >= max_unknown:
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": f"Max unknown status ({max_unknown}) exceeded",
                    "raw": status.get("raw"),
                }
        else:
            unknown_count = 0

        time.sleep(poll_interval)

    return {"task_id": task_id, "status": "timeout"}


# ---------------------------------------------------------------------------
# 核心实验逻辑
# ---------------------------------------------------------------------------


def run_single_experiment(
    client: Any,
    experiment: dict[str, Any],
    repeat_index: int,
    machine_name: str,
) -> dict[str, Any]:
    """执行单次真机实验并记录完整元数据。

    Args:
        client: 真机客户端（CqlibTianyanClient 或 MockSmokeClient）
        experiment: 实验配置字典
        repeat_index: 重复序号（1-based）
        machine_name: 使用的机器名称

    Returns:
        实验记录字典（符合 JSON schema）
    """
    exp_id = experiment["id"]
    qcis = experiment["qcis"]
    shots = experiment["shots"]
    theoretical = experiment["theoretical"]
    task_name = f"{exp_id}_run_{repeat_index}"

    submit_time = datetime.now().astimezone().isoformat()
    start = time.time()

    record: dict[str, Any] = {
        "experiment_id": task_name,
        "experiment_type": exp_id,
        "machine": machine_name,
        "qcis": qcis,
        "shots": shots,
        "task_id": None,
        "submit_time": submit_time,
        "complete_time": None,
        "duration_sec": None,
        "probability": {},
        "raw_result": None,
        "theoretical": theoretical,
        "measurement_error": None,
        "fidelity": None,
        "status": "pending",
        "error": None,
    }

    try:
        # 提交任务
        logger.info(f"[Smoke] 提交: {task_name} -> {machine_name}")
        task_id = client.submit_quantum_task(
            qcis=qcis,
            shots=shots,
            task_name=task_name,
        )

        if task_id is None:
            record["status"] = "submit_failed"
            record["error"] = "submit returned None (machine unavailable)"
            logger.error(f"[Smoke] {task_name} 提交失败: 机器不可用")
            return record

        record["task_id"] = task_id

        # 自定义轮询等待结果（正确处理失败任务）
        result = poll_task_result(client, task_id, timeout=180, poll_interval=3, max_unknown=5)
        elapsed = round(time.time() - start, 2)
        complete_time = datetime.now().astimezone().isoformat()

        record["complete_time"] = complete_time
        record["duration_sec"] = elapsed

        if result.get("status") == "completed":
            # 解析概率分布 -- 多层尝试
            probability = {}
            raw_data = result.get("raw", {})

            # 尝试1: result["raw"]["probability"] (可能是 JSON 字符串)
            if isinstance(raw_data, dict):
                raw_prob = raw_data.get("probability")
                if raw_prob:
                    probability = parse_probability(raw_prob)

            # 尝试2: result["result"] 字段 (get_task_status 返回的)
            if not probability:
                raw_result = result.get("result")
                if raw_result:
                    probability = parse_probability(raw_result)

            # 尝试3: 从 resultStatus 原始测量数据统计
            if not probability and isinstance(raw_data, dict):
                result_status = raw_data.get("resultStatus")
                if result_status:
                    probability = compute_probability_from_shots(result_status)
                    logger.debug(f"[Smoke] 从 resultStatus 统计概率: {probability}")

            # 尝试4: result["raw"] 本身就是概率 dict
            if not probability and isinstance(raw_data, dict):
                if "probability" not in raw_data and "resultStatus" not in raw_data:
                    probability = parse_probability(raw_data)

            record["probability"] = probability
            record["raw_result"] = str(raw_data)[:500]
            record["measurement_error"] = round(
                compute_measurement_error(probability, theoretical), 4
            )
            record["fidelity"] = round(compute_fidelity(probability, theoretical), 4)
            record["status"] = "completed"

            # 打印原始结果用于调试
            logger.info(
                f"[Smoke] [PASS] {task_name} 完成: "
                f"耗时={elapsed}s, 保真度={record['fidelity']}, "
                f"概率={probability}"
            )
            logger.debug(
                f"[Smoke] raw_data keys={list(raw_data.keys()) if isinstance(raw_data, dict) else type(raw_data)}"
            )
        else:
            record["status"] = result.get("status", "unknown")
            record["error"] = str(result.get("error", result))[:200]
            record["raw_result"] = str(result.get("raw", ""))[:500]
            logger.error(
                f"[Smoke] [FAIL] {task_name} 状态={record['status']}, "
                f"耗时={elapsed}s, error={record['error'][:100]}"
            )

    except Exception as e:
        elapsed = round(time.time() - start, 2)
        record["duration_sec"] = elapsed
        record["status"] = "exception"
        record["error"] = str(e)[:200]
        logger.error(f"[Smoke] [FAIL] {task_name} 异常: {e}")

    return record


def run_smoke_test(
    client: Any,
    machine_name: str,
    experiments: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """执行完整冒烟测试（4 类实验 x 3 次 = 12 个任务）。

    Args:
        client: 真机客户端
        machine_name: 首选机器名称
        experiments: 实验列表，None 时使用默认 EXPERIMENTS

    Returns:
        所有实验记录列表
    """
    if experiments is None:
        experiments = EXPERIMENTS

    all_records: list[dict[str, Any]] = []
    total = sum(exp["repeats"] for exp in experiments)
    done = 0

    print(f"\n{'=' * 60}")
    print(f"  天衍-176 冒烟测试 | 机器: {machine_name}")
    print(f"  总任务数: {total} | 预计耗时: ~{total * 2:.0f}min (真机)")
    print(f"{'=' * 60}")

    for exp in experiments:
        print(f"\n--- {exp['id']}: {exp['name']} ({exp['repeats']}次) ---")
        for i in range(1, exp["repeats"] + 1):
            done += 1
            print(f"  [{done}/{total}] {exp['id']}_run_{i} ...", end=" ", flush=True)
            record = run_single_experiment(client, exp, i, machine_name)
            all_records.append(record)
            status_tag = "PASS" if record["status"] == "completed" else "FAIL"
            dur = record.get("duration_sec", "?")
            fid = record.get("fidelity", "?")
            print(f"[{status_tag}] {dur}s fidelity={fid}")

    return all_records


# ---------------------------------------------------------------------------
# 结果保存
# ---------------------------------------------------------------------------


def save_results(records: list[dict[str, Any]], output_dir: Path | None = None) -> str:
    """保存实验结果到 JSON 文件。

    Args:
        records: 实验记录列表
        output_dir: 输出目录，None 时使用默认 RESULTS_DIR

    Returns:
        保存的文件路径
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"smoke_test_{timestamp}.json"
    filepath = output_dir / filename

    # 汇总统计
    total = len(records)
    completed = sum(1 for r in records if r["status"] == "completed")
    avg_duration = sum(r["duration_sec"] for r in records if r["duration_sec"] is not None) / max(
        completed, 1
    )
    avg_fidelity = sum(r["fidelity"] for r in records if r["fidelity"] is not None) / max(
        completed, 1
    )

    summary = {
        "test_type": "smoke_test",
        "timestamp": datetime.now().astimezone().isoformat(),
        "total_tasks": total,
        "completed": completed,
        "failed": total - completed,
        "success_rate": round(completed / max(total, 1), 4),
        "avg_duration_sec": round(avg_duration, 2),
        "avg_fidelity": round(avg_fidelity, 4),
        "experiments": records,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"[Smoke] 结果已保存: {filepath}")
    return str(filepath)


def print_summary(records: list[dict[str, Any]]) -> None:
    """打印汇总报告。

    Args:
        records: 实验记录列表
    """
    total = len(records)
    completed = sum(1 for r in records if r["status"] == "completed")
    failed = total - completed
    durations = [r["duration_sec"] for r in records if r["duration_sec"] is not None]
    fidelities = [r["fidelity"] for r in records if r["fidelity"] is not None]

    avg_dur = sum(durations) / max(len(durations), 1)
    avg_fid = sum(fidelities) / max(len(fidelities), 1)

    print(f"\n{'=' * 60}")
    print("  冒烟测试汇总报告")
    print(f"{'=' * 60}")
    print(f"  总任务数: {total}")
    print(f"  成功: {completed} | 失败: {failed} | 成功率: {completed / max(total, 1):.1%}")
    print(f"  平均耗时: {avg_dur:.1f}s | 平均保真度: {avg_fid:.4f}")

    # 按实验类型分组
    print(f"\n  {'实验':<25s} {'次数':>4s} {'平均耗时':>8s} {'平均保真度':>10s}")
    print(f"  {'-' * 25} {'-' * 4} {'-' * 8} {'-' * 10}")
    exp_types: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        exp_types.setdefault(r["experiment_type"], []).append(r)

    for exp_type, exp_records in exp_types.items():
        exp_durations = [r["duration_sec"] for r in exp_records if r["duration_sec"] is not None]
        exp_fidelities = [r["fidelity"] for r in exp_records if r["fidelity"] is not None]
        cnt = len(exp_records)
        avg_d = sum(exp_durations) / max(len(exp_durations), 1)
        avg_f = sum(exp_fidelities) / max(len(exp_fidelities), 1)
        print(f"  {exp_type:<25s} {cnt:>4d} {avg_d:>7.1f}s {avg_f:>10.4f}")

    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> None:
    """冒烟测试主入口。

    支持命令行参数:
        --mock: 使用 Mock 客户端 dry-run
        --machine: 指定首选机器（默认 tianyan176）
        --shots: 覆盖默认 shots 数（默认 1024）
    """
    import argparse

    parser = argparse.ArgumentParser(description="天衍-176 真机冒烟测试")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用 Mock 客户端 dry-run（不消耗真机机时）",
    )
    parser.add_argument(
        "--machine",
        default="tianyan176",
        help="首选机器名称（默认 tianyan176）",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="每次测量 shots 数（默认 1024）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示 DEBUG 级别日志（查看原始返回数据）",
    )
    args = parser.parse_args()

    # 日志级别
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    # 覆盖实验 shots
    experiments = []
    for exp in EXPERIMENTS:
        exp_copy = dict(exp)
        exp_copy["shots"] = args.shots
        experiments.append(exp_copy)

    # 创建客户端
    if args.mock:
        print("[Mode] Mock dry-run（不消耗真机机时）")
        client: Any = MockSmokeClient(machine_name=args.machine, mock_delay=0.05)
    else:
        print("[Mode] 真机执行")
        api_key = os.environ.get("TIANYAN_API_KEY", "")
        if not api_key:
            print("[FAIL] 未设置 TIANYAN_API_KEY 环境变量")
            sys.exit(1)

        from src.api.tianyan_cqlib import CqlibTianyanClient

        client = CqlibTianyanClient(
            login_key=api_key,
            machine_name=args.machine,
            auto_retry_machine=True,
        )
        print(f"[Setup] 真机客户端已创建: {args.machine} (auto_retry=True)")

    # 执行冒烟测试
    records = run_smoke_test(
        client=client,
        machine_name=args.machine,
        experiments=experiments,
    )

    # 保存结果
    filepath = save_results(records)

    # 打印汇总
    print_summary(records)
    print(f"\n  结果文件: {filepath}")


if __name__ == "__main__":
    main()
