# 真机提分实验总执行手册（2026-08-14）

> **一句话**：三个真机实验（③ v2 扩样 → ① 高占比闭环 → ② 编译真机 smoke）是冻结延期
> 窗口内"方案可行性 7.5→8.5、主题契合度 8.0→8.5"的核心杠杆，全部使用冻结代码 +
> 显式参数执行，预注册先行、负结果照常入库。
> **配套**：预注册总纲 `docs/prereg_real_machine_high_ratio_20260814.md`；
> 权威数字自检 `python scripts/ci/reproduce_authoritative.py`。
> 三份分册 runbook 保留作详细参考，本手册为唯一执行入口。

---

## 一、总览

| 实验 | 验证的问题（评委质疑） | 脚本（新文件） | 真机任务量 | 技术风险 | 优先级 |
|:--|:--|:--|:--|:--|:--|
| **③ v2 扩样 N=20** | "d=5.33 效应量大到可疑（N=10 幸存者偏差）" | 原脚本零改动 + `analyze_multiseed_v3.py` | ~31（新增 30+冒烟） | 极低（协议完全复用） | **1（先跑）** |
| **① 高占比闭环** | "真机 reward 仅占 1/96 步，真机只是装饰" | `prereg_high_ratio_eval.py`（mock 实测） | 成本评估 ~100，正式 400-800 | 中（机时消耗大，排队） | **2** |
| **② 编译真机 smoke** | "AI→量子只有 SWAP 仿真，无真机证据" | `compilation_real_smoke.py`（离线 6/6） | ~11 | **高**（F1-F4 四风险，前置到预检 A） | **3** |

**决策依赖**：③ 独立可先跑；① 先成本评估再定规模；② smoke 是正式实验②的门槛
（不过则实验②终止，如实记录"多比特/显式布局不可行"——该发现本身是有效证据）。

## 二、公共纪律（三个实验一律适用，违反即废）

1. **零改动**：只用冻结代码 + 显式参数；**禁止修改任何 src/ 行为或默认参数**
   （`REAL_SUBMIT_PROBABILITY_DEFAULT` 等常量一动，全部权威仿真数字作废）
2. **预注册先行**：实验启动前在预注册文档记录 seeds/参数/预算/假设；结果无论正负入库
3. **不换机器、不降级、不重试到通过为止**：方法学失败不重试（网络/排队原因可重试 1 次）
4. **防 p-hacking**：不显著就如实报告；**不为让结果好看而修改方法**（严谨性 9 分是
   最大竞争资产，负结果全披露正是它的来源）
5. **机时保护**：每个实验启动前按本手册预算矩阵反推规模；`--max-real-tasks` 收紧
6. **并发协作**：执行/入库前 `git status --short` 确认目标文件不在其他会话 M 列表；
   statistics.yaml 用**独立新段**（不覆盖 v2 段）；门禁文件先确认再动
7. **回归三连**（每个实验入库后必跑）：
   ```powershell
   python scripts/ci/reproduce_authoritative.py              # exit 0（权威数字零污染）
   python scripts/ci/check_stats_consistency.py --strict     # exit 0
   python -m pytest tests/test_env_real_machine.py tests/test_tianyan287_multiseed.py -q
   ```
8. **协议常量**（所有实验统一）：machine=`tianyan-287`（强校验不得回退）、
   shots=32（v2 协议；实验②多比特分布用 1024 例外，理由见分册）、电路=`H Q1/M Q1`、
   timeout=180s（②为 300s）、提交重试 3 次

## 三、机时总预算与排期（建议）

| 阶段 | 时间 | 实验 | 任务量（上限） | 完成标志 |
|:--|:--|:--|:--|:--|
| P0 | 8/14-8/16 | ③ 冒烟 + 正式（新增 10 seeds） | 31 | JSON 入库，30/30 完成 |
| P1 | 8/16-8/18 | ③ 合并分析 + 入库 | 0 | v3 报告 + yaml 新段 + 门禁登记 |
| P2 | 8/18-8/20 | ① 冒烟 + 成本评估（N=2×0.5×50 步） | ~100 | 单任务耗时 <60s 且完成率 ≥90% |
| P3 | 8/20-8/28 | ① 正式（N=8×0.5 档，分 2-3 批） | 400-800 | H1/H2 判定入库 |
| P4 | 8/22-8/25 | ② smoke（A→B→C） | ~11 | 门槛判定（通过→预注册正式②；不通过→终止记录） |
| P5 | 8/28-9/5 | ② 正式（若 smoke 通过）+ ① 0.8 档（仅当 0.5 档正向且额度充裕） | 视额度 | 全部入库 |
| P6 | 9/5-9/10 | 冻结收尾：回归三连 → 重打 tag → `validate_submission.py --pack` → dist 旧包清理 | 0 | validate 0 错误 |

**总任务量估计**：~550-950（③ 31 + ① 500-900 + ② 11；不含可选 0.8 档）。
**执行顺序理由**：③ 成本最低成功率最高先做；① 需成本评估控制机时；② 风险最高
且依赖最少，可并行推进，但 smoke 先行（11 任务探明平台能力，避免正式投入）。

---

## 四、实验③执行：真机 v2 扩样 N=10 → N=20（优先级 1）

### 4.1 目标与协议
- 目标：N=20 收紧 CI，消解"d=5.33 效应量可疑"质疑；**协议与 v2 权威完全一致**
- 脚本：`scripts/real_machine/tianyan287_multiseed.py` **原样零改动**
  （HARD_LIMIT_TOTAL=31：只跑新增 10 seeds = 1 冒烟 + 30 正式，刚好够用）
- v2 权威数据只读：`results/real_machine/tianyan287_multiseed/multiseed_data_20260727_005558.json`

### 4.2 参数表
| 参数 | 值 | 说明 |
|:--|:--|:--|
| 新增 seeds | `11 22 33 44 55 66 77 88 99 110` | 与原 10 个无重叠 |
| 模式 | `--formal` | 自动先冒烟再正式 |
| 协议 | tianyan-287 / shots=32 / `H Q1/M Q1` / 96 步 / 1 真机任务/run / unified_protocol=true | 与 v2 一致（脚本强校验） |

### 4.3 执行序列
```powershell
# ① 冒烟（1 任务）
python scripts/real_machine/tianyan287_multiseed.py --smoke
# ② 正式扩样（30 任务；记录输出文件名 multiseed_data_<ts>.json）
python scripts/real_machine/tianyan287_multiseed.py --formal `
  --seeds 11 22 33 44 55 66 77 88 99 110
# ③ 合并分析（20 seeds → v3 报告）
python scripts/real_machine/analyze_multiseed_v3.py `
  --primary  results/real_machine/tianyan287_multiseed/multiseed_data_20260727_005558.json `
  --extension results/real_machine/tianyan287_multiseed/multiseed_data_<②的时间戳>.json `
  --output   results/reports/multiseed_real_machine_report_20seeds_v3.md
```

### 4.4 验收与入库
- 数据质量（脚本自动审计，不过 exit 1）：unified_protocol、mock/degraded=False、
  完成率 60/60、task_id 100% 留档
- 每策略 N=20 强制校验；**|d|>3 自动输出效应量异常预警**（预注册检查项）
- 入库：新 JSON + v3 报告 → `statistics.yaml` 新增段 `real_machine_20seed_v3`
  （supersedes v2，保留审计轨迹）→ `check_stats_consistency.py` 登记新 p 值
  → AGENTS.md 真机表指向 v3 → 回归三连

### 4.5 风险预案
- 排队超时（180s）：记录 timeout 照常入库，报告披露完成率
- 冒烟失败/机器不可用：脚本自动停止，隔日重试，不换机器
- 扩样后 d 仍 >3：按预注册条款补效应量异常分析（分布双峰/真机占比 1/96 机制）

---

## 五、实验①执行：真机高占比调度闭环（优先级 2）

### 5.1 目标与协议
- 目标：`real_submit_probability` 提到 0.5 主导档，验证 H1（PPO 真机奖励维度 > FCFS）
  与 H2（真机参与度占比）
- 脚本：`scripts/real_machine/prereg_high_ratio_eval.py`（已 mock 实测）
- env 原生概率提交 + result_aware 反馈（与 v2 除 prob/steps 外零差异）

### 5.2 参数表
| 参数 | 值 | 说明 |
|:--|:--|:--|
| `--machine` / `--shots` | tianyan-287 / 32 | 强校验（脚本内） |
| `--steps` | 成本评估 50；正式 100 | — |
| `--prob` | 成本评估 0.5；正式 0.5（0.8 视额度） | 0.15=现状对照（已有 v2 数据） |
| `--strategy` | **`--strategy ppo --strategy fcfs`（必须传两次，append 语义）** | 可加 sjf |
| `--seeds` | 成本评估 `42 123`；正式 8-10 个 | 配对设计 |
| `--max-real-tasks` | 默认 = ceil(steps×prob)+1 | 机时保护，可收紧 |

### 5.3 执行序列（三段式，先成本评估再定规模）
```powershell
# ① 冒烟（1 任务）
python scripts/real_machine/prereg_high_ratio_eval.py --smoke
# ② 成本评估（2 seeds × 2 策略 × 0.5 × 50 步 ≈ 100 任务）
python scripts/real_machine/prereg_high_ratio_eval.py `
  --seeds 42 123 --prob 0.5 --steps 50 --strategy ppo --strategy fcfs
# ③ 正式（8 seeds × 0.5 × 100 步 ≈ 800 任务上限；建议分 2-3 批，各用独立 --output）
python scripts/real_machine/prereg_high_ratio_eval.py `
  --seeds 42 123 456 789 1024 2025 3141 5678 --prob 0.5 --steps 100 `
  --strategy ppo --strategy fcfs
# ④（可选）0.8 档：仅当 0.5 档 H1 支持且额度充裕
```

### 5.4 机时预算矩阵（启动前必读）
| 规模 | 档位 | steps | 任务/run | 总任务 |
|:--|:--|:--|:--|:--|
| 冒烟 | — | — | 1 | 1 |
| 成本评估（N=2） | 0.5 | 50 | 25 | ~100 |
| 正式（N=8） | 0.5 | 100 | 50 | ~800（上限） |
| 保守（N=8） | 0.5 | 50 | 25 | ~400 |
| 可选（N=8） | 0.8 | 100 | 80 | ~1280（上限） |

> 额度紧张时用 **N=8 × 0.5 × 50 步（400 任务）** 起步；结果显著即可支撑 H1。

### 5.5 分析判定（H1/H2，预注册口径）
- H1：配对 Wilcoxon 单侧（PPO > FCFS，同 seed），α=0.05；p<0.05 且方向正确 → 支持
- H2：`real_task_share`（任务数/步数）≥0.30 → 真机主导（代理指标，报告注明）
- 分析代码见分册 `docs/runbook_high_ratio_exp1_20260814.md` §4（粘贴 %TEMP% 运行）
- **决策门槛**：0.5 档不显著 → 如实披露，**不补跑 0.8 档**（防 p-hacking 观感）

### 5.6 入库
结果 JSON → 分析报告 `results/reports/real_machine_high_ratio_report.md`
→ `statistics.yaml` 新段 `real_machine_high_ratio`（每档位独立）→ 门禁登记 → 回归三连

---

## 六、实验②执行：编译层真机验证 Smoke Test（优先级 3，正式实验②的门槛）

### 6.1 目标与门槛
- 目标：验证"编译布局 → 物理比特映射 → 真机执行 → fidelity 对比"全链路可行性
- 脚本：`scripts/real_machine/compilation_real_smoke.py`（核心链路离线实测 6/6）
- **smoke 通过（A+B+C 全过）才启动正式实验②；不通过则实验②终止并如实记录**

### 6.2 四个调研发现（决定方案，必须知晓）
| # | 发现 | 处理 |
|:--|:--|:--|
| F1 | 原仿真电路池（random_circuit 全门集）无法转 QCIS（3-4 qubit 门） | 受控门集生成器（QAOA 风格 h/rx/ry/rz/cx，seed 派生可复现） |
| F2 | 编译 env 只输出映射不生成电路 | SetLayout 固定 PPO 映射 + SabreSwap 标准插入（唯一差异=初始布局） |
| F3 | SABRE 高成本电路 SWAP 落在 measure 后 | 理论分布过滤重建；QCIS 丢弃测量后门 |
| F4 | MBS 不适配多比特、32 shots 不够 | fidelity（vs qiskit statevector 理论）+ shots=1024 |

### 6.3 执行序列（11 任务）
```powershell
python scripts/real_machine/compilation_real_smoke.py --check-a   # 2 任务：多比特可执行性
python scripts/real_machine/compilation_real_smoke.py --check-b   # 2 任务：邻接/布局忠实度
python scripts/real_machine/compilation_real_smoke.py --check-c   # 6 任务：3 深电路×SABRE/PPO fidelity
# 或：python scripts/real_machine/compilation_real_smoke.py --all（任一失败即停）
```

### 6.4 通过/失败标准
| 预检 | 通过 | 失败预案 |
|:--|:--|:--|
| A | 2/2 completed + probability 非空 | 平台不支持多比特/CNOT → 实验②终止，记录"多比特电路不可用" |
| B | 2/2 completed（显式布局被接受） | B2 failed → 平台强制物理邻接 → 正式需先做真实拓扑探测（风险标记） |
| C | 6/6 completed + 有效 fidelity | 部分失败如实披露；fidelity 全低 → 比特序校准后重跑 C |
| 总体 | A+B+C | 不换机器不降级；方向性判定见分册 §5（smoke 验可行性，非验 PPO 优势） |

### 6.5 诚实披露红线（最重要）
SetLayout 方案下 PPO 2q 门数可能不低于 SABRE（离线实测 27-31 vs 15-17）——PPO
真机 fidelity 是否占优是**开放问题**。不占优就如实报告（"编译差异在真机噪声/标准
SWAP 插入下不带来保真度优势"是有效证据）；**禁止为结果好看修改方法**。

### 6.6 入库
smoke JSON + 报告 `results/reports/compilation_real_smoke_report.md`
（含 F1-F4 披露 + 门槛判定）→ smoke 通过后另行预注册正式实验② → 回归三连

---

## 七、结果决策树（三实验结果 → 下一步）

```
实验③：N=20 入库
  ├─ d 明显收缩（<3）→ 效应量回归正常，答辩主动说明 ✓
  └─ d 仍 >3 → 补异常分析（分布双峰/真机占比机制），答辩预案
实验① 0.5 档：
  ├─ H1 支持（p<0.05 方向正）→ 可行性 7.5→8.5 的核心证据；额度充裕才跑 0.8 档
  ├─ H1 不显著 → 如实披露 + CI/功效说明；不补跑 0.8 档
  └─ 方向为负 → 报告"高占比下 PPO 无优势"（仍有价值：证明差异确实由仿真驱动）
实验② smoke：
  ├─ 全过 → 预注册正式②（20-30 电路配对 Wilcoxon，独立预注册文档）
  ├─ A 失败 → 实验②终止，记录平台能力边界
  └─ B/C 失败 → 按 6.4 预案
```

## 八、冻结前收尾（9/5-9/10）

1. 回归三连全绿（reproduce_authoritative + check_stats_consistency --strict + 相关 pytest）
2. `python scripts/ci/validate_submission.py --pack` 重打包（新 HEAD）
3. `git tag -f v9.1-submission <新HEAD>`（在 `qrl-fix-732` 工作树操作）
4. **dist 旧包清理**（20260808/10/11/13 已证过期）
5. 白皮书 §12 等残留清零（另一会话处理中；`reproduce_authoritative.py` exit 0 为验收器）
6. 提交包核对：`docs/` 内新 runbook 若需排除出包，在 `config/submission_manifest.yaml`
   exclude 列表补充（内部操作文档不入提交包）
