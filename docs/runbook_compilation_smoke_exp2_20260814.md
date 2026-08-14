# 实验②执行清单：编译层真机验证 Smoke Test（Runbook）

> **目标**：验证"PPO/SABRE 编译布局 → 物理比特映射 → 天衍-287 真机执行 → 保真度对比"
> 全链路可行性。**通过门槛后才启动正式实验②；不通过则实验②终止并如实记录**（预注册条款，
> 见 `docs/prereg_real_machine_high_ratio_20260814.md` 实验②）。
> **脚本**：`scripts/real_machine/compilation_real_smoke.py`（新写，核心链路已离线实测 6/6 通过）。
> **零改动纪律**：不修改 src/ 任何行为；本实验只用现成模型与脚本。

---

## 0. 调研关键结论（2026-08-14，决定本方案设计的四个发现）

| # | 发现 | 影响 |
|:--|:--|:--|
| F1 | **compilation_deep_scale/fair_v2 的 random_circuit 电路池无法转 QCIS**：qiskit 全门集（50+ 种门，100% 电路含 ccx/cswap/c3sx 等 3-4 qubit 门），天衍仅支持 H/X/Y/Z/RX/RY/RZ/CNOT/CZ/M | smoke 与正式实验②必须用**受控门集电路**（QAOA 风格 h/rx/ry/rz/cx，脚本已内置生成器 `generate_qaoa_style_circuits`，seed 派生可复现） |
| F2 | **QuantumCompilationEnv 只输出映射决策（`_mapping`），从不生成最终电路**；仿真报告的 "ppo_swap" 是理论距离成本 | PPO 物理电路须重建。**方法学修正：用 qiskit SetLayout 固定 PPO 映射为初始布局 + SabreSwap 标准插入**——SABRE/PPO 流程唯一差异 = 初始布局，对比干净 |
| F3 | **SABRE 高成本电路 transpile 后 SWAP 可能落在 measure 之后**（remove_final_measurements 无法清理） | 理论分布显式过滤 measure/barrier 重建；QCIS 转换丢弃测量后的门（无物理意义，脚本含 warn 日志） |
| F4 | **MBS（测量平衡分）只适用单比特 50/50 态**，16 比特电路不可用；v2 的 shots=32 不足以分辨多比特分布 | 指标改用 **fidelity = 真机测量分布 vs qiskit statevector 无噪声理论分布**（复用 smoke_test.compute_fidelity）；shots=1024 |

**诚实披露预案（重要）**：SetLayout 方案下 PPO 布局经标准 SWAP 插入后 2q 门数可能
**不低于** SABRE（离线实测 top3 电路：SABRE 15-17 个 2q 门 vs PPO 27-31 个）——仿真中
PPO 的优势来自理论距离成本，真机保真度是否占优是**开放问题**。若 smoke/正式结果为
PPO 不占优或不可分辨：如实报告（这是"编译差异在真机噪声/标准 SWAP 插入下不带来
保真度优势"的有效证据），**不修改方法直到结果"好看"**（p-hacking 红线）。

## 1. 前置检查

| # | 检查项 | 标准 |
|:--|:--|:--|
| 1.1 | 并发冲突 | `git status --short`：`compilation_real_smoke.py`、`src/quantum/compilation_env.py` 不在其他会话 M 列表 |
| 1.2 | 凭证 | `.env` 中 `TIANYAN_API_KEY` 非空 |
| 1.3 | 依赖 | `python -c "import qiskit, stable_baselines3; print('ok')"` |
| 1.4 | 编译模型可加载 | `python -c "from stable_baselines3 import PPO; PPO.load('deliverable_models/ppo_compilation_agent.zip')"` |
| 1.5 | 离线链路自检 | `python scripts/real_machine/compilation_real_smoke.py --help`（无 import 错误）；6/6 离线验证已通过 |
| 1.6 | 预注册记录 | 在预注册文档记录启动时间、预算（~11 任务） |

## 2. 执行序列（11 个真机任务）

```powershell
# ① 预检 A：多比特电路可执行性（2 任务：4 比特 CZ 链 + CNOT 对）
python scripts/real_machine/compilation_real_smoke.py --check-a

# ② 预检 B：物理比特邻接/布局忠实度（2 任务：CZ Q1Q2 近邻 vs CZ Q1Q3 跨比特）
python scripts/real_machine/compilation_real_smoke.py --check-b

# ③ 预检 C：3 深电路 × SABRE/PPO 布局（6 任务，fidelity 对比）
python scripts/real_machine/compilation_real_smoke.py --check-c

# 或全流程串行（任一失败即停）
python scripts/real_machine/compilation_real_smoke.py --all
```

每次运行输出：`results/real_machine/compilation_real_smoke_<时间戳>.json`
（含 check_a/check_b/check_c 全记录：task_id/status/probability/fidelity + 电路参数 + QCIS）。

## 3. 通过/失败标准（预注册门槛）

| 预检 | 通过标准 | 失败预案 |
|:--|:--|:--|
| A | 2/2 completed 且 probability 非空（多比特电路可执行） | 平台不支持多比特/CNOT → **实验②终止**，如实记录"天衍-287 多比特电路不可用"（v2 仅验证过单比特 H Q1/M Q1，此发现本身有价值） |
| B | 2/2 completed（无论 fidelity，关键是"显式比特布局被接受"） | 若 B2（远比特 CZ）failed → 平台要求物理邻接 → 布局映射改为"先做真实拓扑探测"（复杂度↑，风险标记） |
| C | 6/6 completed 且拿到有效 fidelity | 部分失败 → 按完成率如实披露；若 fidelity 全部异常低 → 检查比特序约定（§5 校准）后重跑 C（不重跑 A/B） |
| 总体 | A+B+C 全过 | 任一 fail → exit 1，按预案处理；**不换机器、不降级** |

**方向性判定（C 结果）**：3 电路 × 2 布局的 fidelity 配对对比——无论 PPO 是否占优，
只要数据有效（fidelity ∈ (0,1] 且可复现方向），smoke 即通过（smoke 验证的是**可行性**，
不是 PPO 优势）。正式实验的假设检验在 smoke 通过后另行预注册。

## 4. 机时与耗时预算

- 任务数：2(A) + 2(B) + 6(C) = 10 + 可选 1 次比特序校准 = ~11
- 每任务：提交+排队+轮询（v2 实测 ~8s，排队高峰 300s 超时）
- 预计总耗时：30 分钟 ~ 2 小时
- 电路规模：A/B 2-4 比特（小），C 14-16 比特 × 1024 shots（大，但每个电路只有 1 次）

## 5. 结果判定细则

- **fidelity 有效性**：真机 1024 shots 的分布 vs 理论分布（512/256/64 态）；若所有
  fidelity < 0.1 且概率分布接近均匀 → **比特序约定不一致**（qiskit 序 vs 真机序）：
  用 B1（单比特 H 态 50/50，序无关）确认链路，再用 C 数据做"反转序"对照计算；
  校准一次后正式实验固定该约定（这是约定校准，不是选择性报告——正式实验的配对
  对比在同一约定下公平）
- **B 的忠实度判断**：B1（近邻）与 B2（跨比特）都 completed → 平台接受显式布局
  （忠实度由 C 的 fidelity 体现）；B2 failed → 平台强制物理邻接，正式实验需先做
  真实 coupling map 探测（风险预案）
- **QCIS 行数合理性**：C 的 QCIS 40-140 行（离线实测）——若平台拒绝长电路，
  在 C 记录中可见（status=failed + error），按 A 的预案处理

## 6. 入库与门禁衔接（smoke 通过后）

| 步骤 | 动作 |
|:--|:--|
| 6.1 | JSON 入库：`results/real_machine/compilation_real_smoke_*.json` |
| 6.2 | smoke 报告入库：`results/reports/compilation_real_smoke_report.md`（含 F1-F4 方法学披露 + 门槛判定） |
| 6.3 | 正式实验②预注册（电路集扩大至 20-30 个、配对 Wilcoxon、预注册 H）——**独立于本 runbook 的另一份文档**，在 smoke 通过后起草 |
| 6.4 | 回归验证：`reproduce_authoritative.py` exit 0 + `check_stats_consistency.py --strict` exit 0 |
| 6.5 | 冻结时统一重打包 |

## 7. 红线（违反即废）

1. **不修改任何编译方法直到结果"好看"**——PPO 不占优就如实报告，这是严谨性资产
2. **不换机器、不降级、不重试到通过为止**——每预检最多重试 1 次（网络/排队原因），
   方法学失败不重试
3. **不修改 src/ 行为**——本实验全部基于现有模型与标准 qiskit 流程
4. 与另一会话并行时先 `git status` 确认 `compilation_env.py`/`smoke_test.py` 未被改动
