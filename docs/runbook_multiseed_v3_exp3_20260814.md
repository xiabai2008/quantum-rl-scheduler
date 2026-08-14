# 实验③执行清单：真机 v2 扩样 N=10 → N=20（Runbook）

> **目标**：把真机多seed权威实验从 N=10/组 扩到 N=20/组，收紧 CI、消解"d=5.33 效应量
> 过大可疑"的质疑（预注册方案 `docs/prereg_real_machine_high_ratio_20260814.md` 实验③）。
> **协议零改动**：与 v2 权威（`multiseed_data_20260727_005558.json`）完全一致——
> tianyan-287（带连字符）、shots=32、`H Q1/M Q1`、96 步/episode、1 真机任务/run、
> unified_protocol=true、Welch t + Bonferroni(α=0.0167)。
> **脚本零改动**：复用 `scripts/real_machine/tianyan287_multiseed.py`（原样，含
> HARD_LIMIT_TOTAL=31 硬上限——只跑新增 10 seeds 时 1 冒烟 + 30 正式 = 31，刚好够用）。
> **只读数据**：v2 权威 JSON 不修改；新增数据独立文件，分析时合并。

---

## 0. 前置检查（每项必须 ✅ 才继续）

| # | 检查项 | 命令 | 通过标准 |
|:--|:--|:--|:--|
| 0.1 | 仓库状态（防并发冲突） | `git -C <repo> status --short` | 确认 `scripts/real_machine/tianyan287_multiseed.py`、`.env` 不在其他会话的 M 列表中 |
| 0.2 | 凭证存在 | `python -c "import os; from dotenv import load_dotenv; load_dotenv('.env'); k=os.environ.get('TIANYAN_API_KEY',''); print('OK' if k else 'MISSING')"` | 输出 OK（空 key 脚本会 exit 1） |
| 0.3 | 环境可用 | `python -c "import loguru, numpy, scipy, stable_baselines3; print('deps ok')"` | deps ok（Python 3.10-3.12 均可，3.11 实测） |
| 0.4 | 模型可加载 | `python -c "from stable_baselines3 import PPO; PPO.load('deliverable_models/ppo_best_model_16dim.zip'); print('model ok')"` | model ok |
| 0.5 | v2 权威数据在位 | `Test-Path results/real_machine/tianyan287_multiseed/multiseed_data_20260727_005558.json` | True |
| 0.6 | 预注册记录 | 在 `docs/prereg_real_machine_high_ratio_20260814.md` 实验③节记录：启动时间、新增 seeds 清单、机时预算（30 任务） | 已记录 |

## 1. 参数表（正式扩样运行）

**新增 10 个 seeds**（与 v2 原 10 个 [42, 123, 456, 789, 1024, 2025, 3141, 5678, 8765, 9999] 无重叠，简单可审计）：

```
11, 22, 33, 44, 55, 66, 77, 88, 99, 110
```

| 参数 | 值 | 说明 |
|:--|:--|:--|
| 脚本 | `scripts/real_machine/tianyan287_multiseed.py` | 原样，不改 |
| 模式 | `--formal` | 自动先冒烟 1 次再正式（:637） |
| `--seeds` | 上述 10 个 | nargs="+"，只传新增 seeds |
| `--machine` | `tianyan-287`（默认） | 强校验不得回退（:576） |
| `--shots` | `32`（默认） | 强校验（:579） |
| 真机任务数 | 10 seeds × 3 策略 × 1 = **30 次** | +1 冒烟 = 31 = 硬上限，刚好 |
| 硬上限 | HARD_LIMIT_TOTAL=31（默认） | 不需要 --hard-limit（无此参数，勿传） |
| 电路/步数 | `H Q1/M Q1`、96 步（NUM_TASKS=32×3 上限） | 与 v2 一致 |
| 超时 | 180s/任务、轮询 5s | 与 v2 一致 |
| 重试 | 提交重试 3 次（with_retry，:330） | 内置 |

## 2. 执行序列

```powershell
# ① 冒烟（1 个真机任务，验证可用性——正式前自动执行，也可手动先验）
python scripts/real_machine/tianyan287_multiseed.py --smoke

# ② 正式扩样（30 个真机任务；预计 0.5-2 小时含排队波动）
python scripts/real_machine/tianyan287_multiseed.py --formal `
  --seeds 11 22 33 44 55 66 77 88 99 110
```

**执行中注意**：
- 输出自动保存：`results/real_machine/tianyan287_multiseed/multiseed_data_<时间戳>.json`（**记录该文件名**，下一步要用）
- 日志含逐 seed 汇总表（奖励/测量平衡分/耗时），结束后核对：30 条记录全部 `real_tasks_completed=1`、无 error
- 若中途机器不可用/冒烟失败，脚本自动停止（Issue #58 纪律），**不要手动重试或换机器**，把失败记录入库后按风险预案处理（见 §6）

## 3. 合并分析（新增脚本，已就绪并测试通过）

```powershell
python scripts/real_machine/analyze_multiseed_v3.py `
  --primary  results/real_machine/tianyan287_multiseed/multiseed_data_20260727_005558.json `
  --extension results/real_machine/tianyan287_multiseed/multiseed_data_<②的新时间戳>.json `
  --output   results/reports/multiseed_real_machine_report_20seeds_v3.md
```

**脚本自动完成**（`scripts/real_machine/analyze_multiseed_v3.py`）：
- 数据质量审计（exit 1 拦截）：unified_protocol、machine=tianyan-287、shots=32、mock/degraded=False、real_tasks_completed=记录数、task_id 100% 留档
- 合并 20 seeds：均值/std(ddof=1)/min/max（对齐 v2 报告 §6.1 格式）
- 两两对比：Welch t + 95% CI + Cohen's d + Bonferroni 判定（§6.2-6.3 格式）
- 同 seed 配对敏感性（PPO vs FCFS 配对 t，按 seed 字典对齐）
- 预注册检查项：**任一 |d|>3 输出效应量异常预警**（v2 曾 d=5.33）

## 4. 数据质量验收（v3 报告生成后人工核对）

| 项 | 标准 |
|:--|:--|
| 样本数 | 每策略 N=20（脚本强制，否则 exit 1） |
| 完成率 | 60/60 真机任务 completed（2 文件合计） |
| task_id 留档 | 100%（审计输出） |
| 效应量变化 | v2 d=5.33 → v3 d 若明显收缩（如 <3）：正常，属小样本回归，报告已自动预警；若仍 >3：按预注册条款补充效应量异常分析 |
| 结论 | 无论显著与否如实呈现（预注册诚实披露条款） |

## 5. 入库与门禁衔接（v3 报告确认后）

| 步骤 | 动作 | 文件 |
|:--|:--|:--|
| 5.1 | 新增数据 JSON 入库（git add） | `results/real_machine/tianyan287_multiseed/multiseed_data_<ts>.json` |
| 5.2 | v3 报告入库 | `results/reports/multiseed_real_machine_report_20seeds_v3.md` |
| 5.3 | statistics.yaml 新增段 `real_machine_20seed_v3`（n=20、新 p 值、supersedes: v2） | `config/statistics.yaml` |
| 5.4 | 门禁登记新 p 值 | `scripts/ci/check_stats_consistency.py` `_KNOWN_LEGIT_P_VALUES` / 权威表 |
| 5.5 | AGENTS.md 真机 v2 表旁注/更新引用（指向 v3） | `AGENTS.md` |
| 5.6 | 回归验证三连 | ① `python scripts/ci/reproduce_authoritative.py`（exit 0）② `python scripts/ci/check_stats_consistency.py --strict`（exit 0）③ `python -m pytest tests/test_tianyan287_multiseed.py tests/test_env_real_machine.py -q`（全过） |
| 5.7 | 重打包（冻结时统一做） | `python scripts/ci/validate_submission.py --pack` + dist 旧包清理 |

> ⚠️ 5.3/5.4 涉及 statistics.yaml 与门禁文件——若另一会话正在改它们，先 `git status` 确认再动；新增段用独立 key（`real_machine_20seed_v3`），不覆盖 v2 段（保留审计轨迹）。

## 6. 风险与预案

| 风险 | 表现 | 预案 |
|:--|:--|:--|
| 排队高峰超时 | 任务 timeout（180s） | 脚本记录 timeout 不重试；数据照常入库，报告如实披露完成率；不影响统计有效性（v2 同规则） |
| 机器不可用 | 冒烟失败 / 客户端机器不一致 | 脚本自动停止（Issue #58 禁止回退）；保留失败记录，隔日重试；不换机器 |
| 部分任务失败 | completed < 60 | 完成率 <100% 时 v3 报告如实标注；若失败集中在某策略（系统偏差信号），在报告中分析并考虑补跑该策略 |
| 结果不显著 / 方向为负 | N=20 下 p 变大 | 预注册条款：如实报告。诚实负结果不掉分（严谨性护城河），且 N=20 扩样本身回应了"效应量可疑"质疑 |
| 扩样后 d 仍 >3 | 效应量异常预警触发 | 按报告 §四 提示补充分析（reward 分布、真机占比 1/96 机制），答辩主动说明 |

## 7. 耗时预算

- 冒烟：~1-2 分钟
- 正式 30 任务：每任务 v2 实测 ~8s（7.89s）+ 排队波动，乐观 ~5-10 分钟，排队高峰可达 1-2 小时（180s 超时上限内）
- 分析 + 入库 + 门禁：~15 分钟
- **建议安排在不影响其他会话文件的时段执行**（如晚间），并在实验期间不与其他会话同时 git add 同一目录
