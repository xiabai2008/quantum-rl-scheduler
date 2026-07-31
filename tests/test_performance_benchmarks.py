"""性能基准测试 — Issue #521

覆盖关键性能路径基准：
    - PPO 模型单次推理延迟（加载 ppo_best_model_16dim.zip，100 次 obs->action）
    - DAG 调度器操作吞吐量（build_scheduling_qubo + 求解，不同任务数 QPS）
    - 环境 step() 单次执行延迟
    - SchedulerCache 命中率与延迟改善

运行方式：
    # 使用 pytest-benchmark（推荐）：
    python -m pytest tests/test_performance_benchmarks.py -m benchmark --benchmark-only

    # 简单计时模式（pytest-benchmark 不可用时自动降级）：
    python -m pytest tests/test_performance_benchmarks.py -m benchmark -v
"""

import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.cache import SchedulerCache
from src.scheduler.dag_scheduler import DAGScheduler, DAGTask
from src.scheduler.env import OBS_DIM, QuantumSchedulingEnv


def _check_pytest_benchmark() -> bool:
    try:
        import pytest_benchmark

        return True
    except ImportError:
        return False


HAS_BENCHMARK = _check_pytest_benchmark()


class SimpleTimer:
    """pytest-benchmark 不可用时的简单计时器，提供类似的 stats 接口。"""

    def __init__(self) -> None:
        self._latencies_sec: list[float] = []

    def __call__(self, function_to_benchmark: callable, *args: Any, **kwargs: Any) -> Any:
        warmup = min(3, max(1, len(self._latencies_sec)))
        for _ in range(warmup):
            function_to_benchmark(*args, **kwargs)

        self._latencies_sec = []
        last_result = None
        rounds = getattr(self, "_rounds", 100)
        for _ in range(rounds):
            t0 = time.perf_counter()
            last_result = function_to_benchmark(*args, **kwargs)
            t1 = time.perf_counter()
            self._latencies_sec.append(t1 - t0)
        return last_result

    @property
    def stats(self) -> "SimpleTimerStats":
        return SimpleTimerStats(self._latencies_sec)


class SimpleTimerStats:
    """SimpleTimer 的统计结果，提供 .mean/.median/.min/.max/.iterations 属性。"""

    def __init__(self, latencies_sec: list[float]) -> None:
        self._latencies_sec = latencies_sec

    @property
    def mean(self) -> float:
        return statistics.mean(self._latencies_sec) if self._latencies_sec else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self._latencies_sec) if self._latencies_sec else 0.0

    @property
    def min(self) -> float:
        return min(self._latencies_sec) if self._latencies_sec else 0.0

    @property
    def max(self) -> float:
        return max(self._latencies_sec) if self._latencies_sec else 0.0

    @property
    def iterations(self) -> list[float]:
        return list(self._latencies_sec)


def _percentile(data: list[float], q: float) -> float:
    """计算百分位数（线性插值）。"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * q
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return sorted_data[f]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


if not HAS_BENCHMARK:

    @pytest.fixture(name="benchmark")
    def _fallback_benchmark(request: pytest.FixtureRequest) -> SimpleTimer:
        """pytest-benchmark 不可用时提供 SimpleTimer 作为 benchmark fixture。"""
        timer = SimpleTimer()
        marker = request.node.get_closest_marker("benchmark_rounds")
        timer._rounds = marker.args[0] if marker else 100
        return timer


def _get_stat(stats: Any, key: str, default: float = 0.0) -> float:
    """兼容 pytest-benchmark 4.x/5.x 的 stats 属性访问。

    4.x: stats.mean / stats.median / stats.iterations（list）
    5.x: stats 是 Metadata 对象，需通过 stats['mean'] 或 stats.stats.mean 访问
    """
    # 4.x: 直接属性访问
    val = getattr(stats, key, None)
    if val is not None:
        return val
    # 5.x: dict-like 访问
    try:
        return stats[key]
    except (KeyError, TypeError, IndexError):
        pass
    # 5.x: 嵌套 Stats 对象
    inner = getattr(stats, "stats", None)
    if inner is not None:
        val = getattr(inner, key, None)
        if val is not None:
            return val
    return default


def _run_benchmark(benchmark: Any, func: callable, name: str) -> dict[str, float]:
    """运行基准测试并返回统计指标字典（单位：毫秒）。"""
    result = benchmark(func)
    stats = benchmark.stats
    # pytest-benchmark 4.x/5.x 兼容：iterations 可能是 list（每次耗时）或 int（次数）
    iterations_raw = _get_stat(stats, "iterations", [])
    if isinstance(iterations_raw, (list, tuple)):
        iterations_ms = [t * 1000.0 for t in iterations_raw]
        median_val = _get_stat(stats, "median", 0.0)
        p95_ms = _percentile(iterations_ms, 0.95) if iterations_ms else median_val * 1000.0
    else:
        # iterations 是 int（迭代次数），无法计算 p95，退化为 median
        median_val = _get_stat(stats, "median", 0.0)
        p95_ms = median_val * 1000.0

    metrics = {
        "mean_ms": _get_stat(stats, "mean", 0.0) * 1000.0,
        "median_ms": median_val * 1000.0,
        "p95_ms": p95_ms,
        "min_ms": _get_stat(stats, "min", 0.0) * 1000.0,
        "max_ms": _get_stat(stats, "max", 0.0) * 1000.0,
        "_result": result,
    }

    if not HAS_BENCHMARK:
        print(
            f"\n  [simple-timer] {name}: "
            f"mean={metrics['mean_ms']:.3f}ms, median={metrics['median_ms']:.3f}ms, "
            f"p95={metrics['p95_ms']:.3f}ms"
        )

    return metrics


def _find_ppo_model() -> str | None:
    """在标准候选路径中查找 ppo_best_model_16dim.zip。"""
    project_root = Path(__file__).parent.parent
    candidates = [
        project_root / "deliverable_models" / "ppo_best_model_16dim.zip",
        project_root / "models" / "ppo_seed_42" / "best_model.zip",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _load_ppo_predictor(model_path: str) -> tuple[callable, QuantumSchedulingEnv]:
    """加载 PPO 模型，返回 (predict_fn, env)。直接用 stable_baselines3 加载。"""
    from stable_baselines3 import PPO

    env = QuantumSchedulingEnv(max_steps=100, seed=42)
    model = PPO.load(model_path, env=env)

    def predict(obs: np.ndarray) -> int:
        model_state = obs.reshape(1, -1) if obs.ndim == 1 else obs
        action, _ = model.predict(model_state, deterministic=True)
        return int(action.item())

    return predict, env


def _generate_random_dag(
    num_tasks: int,
    max_qubits: int = 50,
    max_duration: float = 5.0,
    seed: int = 42,
) -> DAGScheduler:
    """生成随机无环 DAG。仅允许高 id 任务依赖低 id 任务以保证无环。"""
    rng = np.random.default_rng(seed)
    tasks: list[DAGTask] = []
    for i in range(num_tasks):
        task_id = f"task_{i}"
        task_type = "quantum" if rng.random() < 0.7 else "classical"
        qubits_required = int(rng.integers(1, max_qubits + 1)) if task_type == "quantum" else 0
        estimated_time = float(rng.uniform(0.5, max_duration))
        priority = int(rng.integers(1, 6))

        possible_deps = [f"task_{j}" for j in range(max(0, i - 5), i)]
        n_deps = min(len(possible_deps), int(rng.integers(0, min(3, len(possible_deps) + 1))))
        deps = (
            list(rng.choice(possible_deps, size=n_deps, replace=False))
            if n_deps > 0 and possible_deps
            else []
        )

        tasks.append(
            DAGTask(
                task_id=task_id,
                task_type=task_type,
                qubits_required=qubits_required,
                estimated_time=estimated_time,
                priority=priority,
                dependencies=deps,
            )
        )

    return DAGScheduler(tasks=tasks, max_qubits=max_qubits, seed=seed)


@pytest.mark.benchmark
@pytest.mark.benchmark_rounds(100)
class TestPPOInferenceLatency:
    """PPO 模型单次推理延迟基准（Issue #521）。

    加载 deliverable_models/ppo_best_model_16dim.zip，运行 100 次 obs->action 推理，
    计算平均/中位/p95 延迟。模型文件不存在时 skip。
    """

    def test_ppo_inference_latency(self, benchmark: Any) -> None:
        """PPO 模型单次推理延迟（100 次 obs->action）。"""
        model_path = _find_ppo_model()
        if model_path is None:
            pytest.skip("PPO 模型文件 (ppo_best_model_16dim.zip) 未找到")

        try:
            predict_fn, env = _load_ppo_predictor(model_path)
        except Exception as exc:
            pytest.skip(f"PPO 模型加载失败: {exc}")

        obs, _info = env.reset(seed=42)

        def inference_once() -> int:
            return predict_fn(obs)

        stats = _run_benchmark(benchmark, inference_once, "PPO inference latency")

        print(
            f"\n  PPO 推理延迟: mean={stats['mean_ms']:.3f}ms, "
            f"median={stats['median_ms']:.3f}ms, p95={stats['p95_ms']:.3f}ms"
        )

        assert stats["mean_ms"] < 100.0, (
            f"PPO 平均推理延迟 {stats['mean_ms']:.3f}ms 超过 100ms 阈值"
        )
        assert isinstance(inference_once(), int)


@pytest.mark.benchmark
class TestDAGSchedulerThroughput:
    """DAG 调度器操作吞吐量基准（Issue #521）。

    对 build_scheduling_qubo + NumPy 模拟退火求解在不同任务数量下计时，
    计算每秒可完成调度次数（QPS）。
    """

    @pytest.mark.parametrize("num_tasks", [5, 10, 20])
    def test_dag_scheduler_throughput(self, benchmark: Any, num_tasks: int) -> None:
        """DAG 调度 QUBO 构建 + 求解吞吐量。"""
        if not HAS_BENCHMARK and hasattr(benchmark, "_rounds"):
            rounds_map = {5: 30, 10: 20, 20: 10}
            benchmark._rounds = rounds_map.get(num_tasks, 10)

        time_horizon = max(5, num_tasks)
        scheduler = _generate_random_dag(num_tasks=num_tasks, seed=42)

        def schedule_once() -> list[dict[str, Any]]:
            return scheduler.schedule_with_annealing(
                time_horizon=time_horizon,
                num_reads=20,
                fallback=True,
            )

        stats = _run_benchmark(
            benchmark,
            schedule_once,
            f"DAG scheduler throughput (N={num_tasks})",
        )

        qps = 1000.0 / stats["mean_ms"] if stats["mean_ms"] > 0 else float("inf")
        print(
            f"\n  DAG 调度 (N={num_tasks}, T={time_horizon}): "
            f"mean={stats['mean_ms']:.2f}ms, QPS≈{qps:.1f}"
        )

        assert isinstance(stats["_result"], list)


@pytest.mark.benchmark
@pytest.mark.benchmark_rounds(200)
class TestEnvironmentStepLatency:
    """环境 step() 单次执行延迟基准（Issue #521）。"""

    def test_environment_step_latency(self, benchmark: Any) -> None:
        """QuantumSchedulingEnv.step() 单次执行延迟。"""
        env = QuantumSchedulingEnv(max_steps=200, seed=42)
        env.reset(seed=42)

        def step_once() -> np.ndarray:
            action = int(env.action_space.sample())
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                env.reset(seed=42)
            return obs

        stats = _run_benchmark(benchmark, step_once, "Environment step latency")

        print(
            f"\n  环境 step() 延迟: mean={stats['mean_ms']:.3f}ms, "
            f"median={stats['median_ms']:.3f}ms, p95={stats['p95_ms']:.3f}ms"
        )

        assert stats["mean_ms"] < 50.0, (
            f"环境 step 平均延迟 {stats['mean_ms']:.3f}ms 超过 50ms 阈值"
        )
        assert step_once().shape == (OBS_DIM,)


@pytest.mark.benchmark
@pytest.mark.benchmark_rounds(100)
class TestCacheHitRate:
    """SchedulerCache 命中率与延迟改善基准（Issue #521）。

    对比三种场景：
        1. 无缓存（直接 PPO 推理，baseline）
        2. 100% 命中（全部精确命中快速路径）
        3. ~80% 命中（混合命中与未命中，模拟典型生产场景）

    验证缓存命中率提升与延迟改善的关系。
    """

    def test_cache_hit_rate_and_latency(self, benchmark: Any) -> None:
        """SchedulerCache 命中率-延迟曲线基准。"""
        model_path = _find_ppo_model()
        if model_path is None:
            pytest.skip("PPO 模型文件未找到，跳过缓存命中率基准")

        try:
            predict_fn, _env = _load_ppo_predictor(model_path)
        except Exception as exc:
            pytest.skip(f"PPO 模型加载失败: {exc}")

        rng = np.random.default_rng(42)
        dim = OBS_DIM
        n_entries = 200
        n_queries = 100

        cached_states = [rng.standard_normal(dim) for _ in range(n_entries)]
        fresh_states = [rng.standard_normal(dim) for _ in range(n_entries)]

        cache = SchedulerCache(max_size=n_entries, similarity_threshold=0.999, ttl_seconds=1e9)
        for state in cached_states:
            cache.put(state, predict_fn(state))

        for state in cached_states[:10]:
            predict_fn(state)
            cache.get(state)

        mixed_queries: list[np.ndarray] = []
        for i in range(n_queries):
            if rng.random() < 0.8:
                mixed_queries.append(cached_states[int(rng.integers(0, n_entries))])
            else:
                mixed_queries.append(fresh_states[i % n_entries])

        scenarios: list[tuple[str, list[np.ndarray], bool]] = [
            ("no_cache (baseline)", cached_states[:n_queries], False),
            ("cache_hit_100pct", cached_states[:n_queries], True),
            ("cache_hit_~80pct", mixed_queries, True),
        ]

        results: dict[str, dict[str, float]] = {}
        for name, queries, use_cache in scenarios:

            def make_fn(qlist: list[np.ndarray], uc: bool) -> callable:
                idx = [0]

                def query_once() -> int | None:
                    q = qlist[idx[0] % len(qlist)]
                    idx[0] += 1
                    if uc:
                        cached_action = cache.get(q)
                        if cached_action is not None:
                            return cached_action
                    return predict_fn(q)

                return query_once

            fn = make_fn(queries, use_cache)
            # pytest-benchmark 5.x: 每个测试只能调用 benchmark() 一次
            # 仅对关键场景（100% 命中）使用 benchmark fixture，其余用简单计时
            if name == "cache_hit_100pct":
                stats = _run_benchmark(benchmark, fn, f"Cache {name}")
            else:
                # 简单计时（不使用 benchmark fixture）
                import time as _time

                timings = []
                for _ in range(50):
                    fn()  # warmup
                for _ in range(100):
                    t0 = _time.perf_counter()
                    fn()
                    timings.append((_time.perf_counter() - t0) * 1000.0)
                stats = {
                    "mean_ms": sum(timings) / len(timings),
                    "median_ms": sorted(timings)[len(timings) // 2],
                    "p95_ms": _percentile(timings, 0.95),
                    "min_ms": min(timings),
                    "max_ms": max(timings),
                    "_result": None,
                }
                print(
                    f"\n  [simple-timer] Cache {name}: "
                    f"mean={stats['mean_ms']:.3f}ms"
                )
            results[name] = stats

            if use_cache:
                cs = cache.stats()
                print(f"\n  {name}: mean={stats['mean_ms']:.3f}ms, hit_rate={cs['hit_rate']:.2%}")
            else:
                print(f"\n  {name}: mean={stats['mean_ms']:.3f}ms (baseline)")

        baseline = results["no_cache (baseline)"]["mean_ms"]
        hit_100 = results["cache_hit_100pct"]["mean_ms"]
        speedup_100 = baseline / hit_100 if hit_100 > 0 else float("inf")

        print(
            f"\n  缓存 100% 命中加速比: {speedup_100:.1f}x "
            f"(baseline={baseline:.3f}ms -> cached={hit_100:.3f}ms)"
        )

        assert hit_100 < baseline, "100% 缓存命中延迟应低于无缓存基线"
        assert len(cache) <= n_entries
