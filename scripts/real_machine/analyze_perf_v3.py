#!/usr/bin/env python
"""perf_v3 真机实验结果分析：效应量 + 95% CI（预注册口径，p 值辅助）。

用法:
    python scripts/real_machine/analyze_perf_v3.py <json文件>
"""

import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(os.getcwd())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from scipy import stats


def summarize(results: list[dict]) -> dict:
    data: dict[str, list[float]] = {}
    mock_counts: dict[str, int] = {}
    completed_counts: dict[str, int] = {}
    for r in results:
        if "error" in r:
            print(f"  [SKIP] seed={r['seed']} policy={r['policy']} error={r['error']}")
            continue
        data.setdefault(r["policy"], []).append(r["total_reward"])
        mock_counts[r["policy"]] = mock_counts.get(r["policy"], 0) + sum(
            1 for rec in r.get("real_records", []) if rec.get("mock")
        )
        completed_counts[r["policy"]] = completed_counts.get(r["policy"], 0) + r.get(
            "real_tasks_completed", 0
        )

    print("\n=== 描述性统计 ===")
    for p, vals in data.items():
        arr = np.array(vals)
        print(
            f"  {p.upper()}: N={len(arr)} mean={arr.mean():.2f}±{arr.std(ddof=1):.2f} "
            f"median={np.median(arr):.2f} min={arr.min():.2f} max={arr.max():.2f}"
        )

    print("\n=== 成对比较（效应量 + 95% CI，p 辅助） ===")
    pairs = [("ppo", "fcfs"), ("ppo", "sjf"), ("sjf", "fcfs")]
    for a, b in pairs:
        if a not in data or b not in data or len(data[a]) < 2 or len(data[b]) < 2:
            print(f"  {a} vs {b}: 数据不足")
            continue
        x, y = np.array(data[a]), np.array(data[b])
        sp = np.sqrt(
            ((len(x) - 1) * x.std(ddof=1) ** 2 + (len(y) - 1) * y.std(ddof=1) ** 2)
            / (len(x) + len(y) - 2)
        )
        d = (x.mean() - y.mean()) / sp if sp > 0 else 0.0
        t, p = stats.ttest_ind(x, y, equal_var=False)
        se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
        ci_low, ci_high = (x.mean() - y.mean()) + np.array([-1, 1]) * stats.t.ppf(
            0.975, len(x) + len(y) - 2
        ) * se
        print(
            f"  {a} vs {b}: d={d:.4f} (等级={effect_level(d)}), "
            f"mean_diff={x.mean() - y.mean():.2f}, 95%CI=[{ci_low:.2f}, {ci_high:.2f}], "
            f"Welch t={t:.2f} p={p:.2e}"
        )

    print("\n=== 真机参与率审计 ===")
    for p, _vals in data.items():
        print(
            f"  {p.upper()}: real_task_ids={completed_counts.get(p, 0)} mock={mock_counts.get(p, 0)}"
        )
    return {"policies": {p: {"n": len(v), "mean": float(np.mean(v))} for p, v in data.items()}}


def effect_level(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "可忽略"
    if ad < 0.5:
        return "小"
    if ad < 0.8:
        return "中"
    return "大"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    print(f"实验: {payload.get('experiment')} @ {payload.get('timestamp')}")
    print(f"配置: {json.dumps(payload.get('config', {}), ensure_ascii=False)}")
    summarize(payload.get("results", []))
