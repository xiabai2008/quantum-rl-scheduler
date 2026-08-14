# 实验①执行清单：真机高占比调度闭环（Runbook）

> **目标**：`real_submit_probability` 提到主导档位（0.5/0.8），验证 **H1（PPO 真机奖励
> 维度优于 FCFS）** 与 **H2（真机参与度占比）**，消解"真机只是装饰（1/96 步）"的质疑。
> 预注册方案：`docs/prereg_real_machine_high_ratio_20260814.md` 实验①。
> **脚本**：`scripts/real_machine/prereg_high_ratio_eval.py`（新写，已 mock 实测通过；
> env 原生概率提交 + result_aware 反馈，与 v2 协议除 prob/steps 外零差异）。
> **零改动纪律**：不修改 src/ 任何行为或默认参数；`REAL_SUBMIT_PROBABILITY_DEFAULT` 不动。

> **📌 决策记录（2026-08-14，冻结终检）：选 B —— 放弃实跑，本文档转为"可恢复执行预案"**
>
> **决策理由**：
> 1. 真机策略对比证据缺口已收窄：v3 权威（N=20/组，8.14 扩样）已达 N≥18 功效线
>    （PPO vs FCFS d=2.11, p=1.22e-07），评委"真机样本量不足"质疑已被主动扩样回应；
> 2. 当日真机平台 ~30% 真实失败率 + 单任务 ~5 分钟排队 + 仅支持单比特 H 电路——
>    高占比档位（0.5/0.8）下失败任务将大量进入 reward，实验结果可比性风险高；
> 3. 新档位引入的 reward 口径与 v3（1/96 步参与）不一致，需全仓同步成本高，
>    与"冻结后口径稳定"原则冲突。
>
> **恢复执行条件（满足任一才重启）**：① 平台多比特电路支持稳定且失败率 <10%；
> ② 机时包升级且评审阶段需要补充真机高占比证据；③ 队长明确指示重启。
> **预案完备性**：脚本（`prereg_high_ratio_eval.py`）已 mock 实测通过；本手册
> §0-§9（前置检查/参数/种子/序列/判定口径/入库衔接/预算矩阵/决策门槛）可直接执行；
> 预注册假设 H1/H2/H3 已锁定（`docs/prereg_real_machine_high_ratio_20260814.md` 实验①）。

---

## 0. 前置检查（全部 ✅ 才继续）

| # | 检查项 | 命令/标准 |
|:--|:--|:--|
| 0.1 | 并发冲突检查 | `git status --short`：`prereg_high_ratio_eval.py`、`.env`、`src/scheduler/env*.py` 不在其他会话 M 列表 |
| 0.2 | 凭证 | `.env` 中 `TIANYAN_API_KEY` 非空（脚本自动加载 .env） |
| 0.3 | 模型可加载 | `python -c "from stable_baselines3 import PPO; PPO.load('deliverable_models/ppo_best_model_16dim.zip')"` 无 SIGSEGV |
| 0.4 | 干跑通过 | `python scripts/real_machine/prereg_high_ratio_eval.py --mock --seeds 42 --prob 0.5 --strategy ppo --strategy fcfs --steps 20 --output %TEMP%\smoke_mock.json` → exit 0 |
| 0.5 | 机时额度确认 | 按 §7 预算表反推可承受规模（**先跑成本评估再定正式规模**） |
| 0.6 | 预注册记录 | 在预注册文档记录：启动时间、所选规模、seeds 清单、预算 |

## 1. 参数表

| 参数 | 值 | 说明 |
|:--|:--|:--|
| 脚本 | `scripts/real_machine/prereg_high_ratio_eval.py` | 新文件，已实测 |
| `--machine` | `tianyan-287`（默认） | 强校验不得回退（脚本内 :280） |
| `--shots` | `32`（默认） | 强校验，与 v2 一致（:283） |
| `--steps` | 成本评估 **50**；正式 **100** | 100 步=权威步数；50 步用于成本评估 |
| `--prob` | 成本评估 **0.5**；正式 **0.5**（0.8 视额度） | 0.15=现状对照（已有 v2 数据），0.5 主导档 |
| `--strategy` | `--strategy ppo --strategy fcfs` | **注意：必须传两次**（append 语义）；加 `--strategy sjf` 可扩 |
| `--seeds` | 成本评估 2 个；正式 8-10 个 | 配对设计种子（见 §2） |
| `--max-real-tasks` | 默认 = ceil(steps×prob)+1（自动机时保护） | 显式收紧可传更小值 |
| 电路/反馈 | `H Q1/M Q1`、result_aware（脚本内置） | 与 v2 一致 |
| 超时 | 180s/任务、5s 轮询（内置） | 与 v2 一致 |

## 2. 种子清单（配对设计，与 v2 风格一致）

```
成本评估（N=2）:  42  123
正式（N=8，推荐）:  42  123  456  789  1024  2025  3141  5678
正式（N=10）:     42  123  456  789  1024  2025  3141  5678  8765  9999
```

配对方式：同 seed 下 PPO 与 FCFS 各跑 1 episode（脚本按 `--seeds` × `--strategy` 全组合跑，
调用一次即可得到同 seed 配对数据）。

## 3. 执行序列

```powershell
# ① 冒烟（1 个真机任务，验证可用性）
python scripts/real_machine/prereg_high_ratio_eval.py --smoke

# ② 成本评估（50 个真机任务：2 seeds × 2 策略 × ~25/run）
python scripts/real_machine/prereg_high_ratio_eval.py `
  --seeds 42 123 --prob 0.5 --steps 50 `
  --strategy ppo --strategy fcfs

# ③ 依据成本评估的单任务耗时 + 额度余量，决定正式规模（§7 预算矩阵）
#    正式（8 seeds × 2 策略 × 0.5 档 × 100 步 ≈ 800 任务上限，实际 ~50 任务/run）
python scripts/real_machine/prereg_high_ratio_eval.py `
  --seeds 42 123 456 789 1024 2025 3141 5678 --prob 0.5 --steps 100 `
  --strategy ppo --strategy fcfs

# ④（额度充裕时）0.8 档补充：先 N=4 再决定是否扩
python scripts/real_machine/prereg_high_ratio_eval.py `
  --seeds 42 123 456 789 --prob 0.8 --steps 100 `
  --strategy ppo --strategy fcfs
```

**每次运行输出**：`results/real_machine/prereg_high_ratio_<策略>_p<档位>_<时间戳>.json`
（记录文件名）；日志含逐 seed 的 reward/真机数/参与度。

**执行中注意**：
- 若冒烟失败/机器不可用 → 脚本 exit 1，**不重试不换机器**，记录后隔日再跑
- 单 run 真机任务上限由 `--max-real-tasks` 保护；预计单任务耗时 = 提交+排队+轮询（v2 实测 ~8s，排队高峰可达 180s 超时）
- 中断恢复：按 seed×策略 分批重跑，JSON 用 `--output` 指定独立文件名

## 4. 数据分析（H1/H2 判定）

对每档位（0.5/0.8）独立分析，PPO vs FCFS 配对（同 seed）：

```python
# 粘贴为 %TEMP%\analyze_high_ratio.py 运行（勿写仓库）
import json, sys
import numpy as np
from scipy import stats
files = sys.argv[1:]  # 同一档位的多个 JSON
rows = []
for f in files:
    d = json.load(open(f, encoding="utf-8"))
    rows += [r for r in d["results"] if "total_reward" in r]
by = {}
for r in rows:
    by.setdefault(r["strategy"], {})[r["seed"]] = r["total_reward"]
ppo = np.array([v for k, v in sorted(by.get("ppo", {}).items())])
fcfs = np.array([v for k, v in sorted(by.get("fcfs", {}).items())])
print("N=", len(ppo), len(fcfs))
print("PPO mean=%.1f  FCFS mean=%.1f  diff=%+.1f (%+.1f%%)" % (
    ppo.mean(), fcfs.mean(), ppo.mean()-fcfs.mean(), (ppo.mean()/fcfs.mean()-1)*100))
w = stats.wilcoxon(ppo, fcfs, alternative="greater")  # H1 单侧: PPO > FCFS
t = stats.ttest_rel(ppo, fcfs)
print("配对 Wilcoxon p=%.4g (W=%.0f) | 配对 t p=%.4g" % (w.pvalue, w.statistic, t.pvalue))
share = np.mean([r["real_task_share"] for r in rows if "real_task_share" in r])
print("真机参与度（任务数/步数）: %.2f" % share)
# H2: 参与度 ≥0.30 即真机主导
print("H2 (参与度≥0.30):", "达标" if share >= 0.30 else "未达标")
```

**H1/H2 判定口径（预注册）**：α=0.05 单侧；任一档位 p<0.05 且方向正确 → H1 支持；
p≥0.05 或方向为负 → **如实报告**（负向结论也是有效证据，预注册诚实披露条款）；
H2 用参与度代理（真机 reward 占比的直接度量在 env 层无累计字段，报告注明代理口径）。

## 5. 入库与门禁衔接

| 步骤 | 动作 |
|:--|:--|
| 5.1 | 结果 JSON 入库（`results/real_machine/prereg_high_ratio_*.json`） |
| 5.2 | 分析报告入库：`results/reports/real_machine_high_ratio_report.md`（模板参考 v2 报告；含 H1/H2 判定、负向如实披露、代理指标说明） |
| 5.3 | `config/statistics.yaml` 新增段 `real_machine_high_ratio`（每档位 n/p/结论）——**独立 key，不覆盖任何现有段** |
| 5.4 | `scripts/ci/check_stats_consistency.py` 登记新 p 值（`_KNOWN_LEGIT_P_VALUES`） |
| 5.5 | 回归验证三连：`reproduce_authoritative.py`（exit 0）+ `check_stats_consistency.py --strict`（exit 0）+ `pytest tests/test_env_real_machine.py -q` |
| 5.6 | 冻结时统一重打包（`validate_submission.py --pack`） |

## 6. 风险与预案

| 风险 | 表现 | 预案 |
|:--|:--|:--|
| 机时消耗超预期 | 高占比任务数多（§7 矩阵） | **必须先跑成本评估**；正式规模按额度反推；`--max-real-tasks` 收紧 |
| 排队高峰 | 单任务 180s 超时 | 超时计 failed 入库；完成率 <100% 如实披露（v2 同规则） |
| 降级触发 | env 连续失败自动降级仿真 | JSON 中 degraded 字段记录；降级后的 run 标记并从主分析剔除（单独披露） |
| H1 不显著/负向 | p≥0.05 | 预注册条款：如实报告 + 置信区间 + 功效说明；诚实负结果不掉分 |
| 并行会话冲突 | 其他会话改 statistics.yaml/门禁 | 5.3/5.4 前先 `git status` 确认；新段独立 key 不覆盖 |

## 7. 机时预算矩阵（启动前必读）

单任务真机调用 ≈ 1 次；每 run 任务数 ≈ `steps × prob`（受 `--max-real-tasks` 封顶）：

| 规模 | 档位 | steps | 任务/run | 总任务（seeds×策略×任务/run） |
|:--|:--|:--|:--|:--|
| 冒烟 | — | — | 1 | 1 |
| 成本评估（N=2） | 0.5 | 50 | 25 | 2×2×25 = **100** |
| 正式（N=8） | 0.5 | 100 | 50 | 8×2×50 = **800**（上限） |
| 正式（N=8） | 0.8 | 100 | 80 | 8×2×80 = **1280**（上限） |
| 保守替代（N=8） | 0.5 | 50 | 25 | 8×2×25 = **400** |

> 建议：若额度紧张，正式规模用 **N=8 × 0.5 档 × 50 步（400 任务）** 起步；
> 结果显著则足以支撑 H1，无需 0.8 档。0.8 档仅在额度充裕且 0.5 档方向为正时补跑。

## 8. 耗时预算

- 冒烟：~1-2 分钟
- 成本评估（100 任务）：v2 实测单任务 ~8s → ~15-30 分钟（不含排队）
- 正式（800 任务）：~2-6 小时（排队高峰可达 180s/任务 × 800 = 40 小时上限，**强烈建议分 2-3 批执行**，每批用独立 `--output`）
- 分析 + 入库 + 门禁：~30 分钟

## 9. 决策门槛

| 阶段 | 门槛 |
|:--|:--|
| 冒烟 | passed=true 才继续 |
| 成本评估 | 单任务平均耗时 <60s 且完成率 ≥90% → 正式规模按预算矩阵；否则降档（0.5→50 步）或缩减 N |
| 正式 0.5 档 | H1 判定后：方向为正且 p<0.05 → 报告 H1 支持；p≥0.05 → 如实披露，**不补跑 0.8 档**（避免 p-hacking 观感） |
| 0.8 档（可选） | 仅当 0.5 档 H1 支持且额度充裕 |
| 入库 | 全部 JSON + 报告 + yaml 新段 + 门禁登记 + 回归三连通过 |
