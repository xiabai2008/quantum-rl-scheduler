# 真机验证结论边界声明（Issue #204）

> **生成时间**: 2026-07-21
> **适用范围**: 所有涉及真机实验的文档（README、白皮书、答辩PPT、答辩QA手册、技术报告）

---

## 一、结论边界定义

真机实验结论严格区分为两类：

### 1. 可用性验证（✅ 可宣称）

| 验证项 | 证据 | 数据来源 |
|:--|:--|:--|
| SDK 认证 | `authenticate()` 成功，`list_backends()` 返回 `tianyan176` 状态 `running` | Issue #165 |
| 任务提交 | 284 次正式 SDK 调用，全部获得 task ID | `results/real_machine/issue165_ablation.json` |
| 状态轮询 | `get_task_status()` 正确返回 `completed`/`running` 状态 | Issue #165, #192 |
| 结果获取 | 测量概率分布成功回传，`probability` 字段可解析 | Issue #235 |
| task ID 审计 | 284 条完整审计记录（task ID、状态、概率、耗时），无 Mock 调用 | `results/real_machine/issue165_ablation.json` |
| 成功率 | 284/284 = 100% | Issue #165, #192 |
| 降级机制 | 连续失败自动降级到 Mock，验证通过 | Issue #64 |

### 2. 性能验证（⚠️ 不宣称）

| 指标 | 现状 | 原因 |
|:--|:--|:--|
| 真机 vs 仿真奖励对比 | mixed_real vs simulation p=0.344，不显著 | N=5-10 seeds，统计功效不足 |
| 真机任务规模 | 1-3 qubit 单比特门电路 | 免费机时包限制（`FREE_TIER_MAX_QUBITS=1`） |
| 真机参与率 | 1.70%（mixed_real 条件） | `real_submit_probability=0.05`, `cap=10/seed` |
| 完成率 | 66.67%（混合/纯真机条件） | 真机延迟导致部分 episode 未完成 |
| 纯真机 reward | -298.77 ± 1164.07 | 真机延迟降低训练效率，不代表策略质量 |

**核心声明**：

> 真机实验证明系统已完成天衍云平台接入和端到端任务闭环（284 次调用 100% 成功）；
> 当前性能提升主要由仿真实验支撑（PPO vs FCFS +88.3%, N=250, p<0.001），
> 真机大规模性能验证受机时和硬件排队约束，是后续扩展方向。

---

## 二、文档更新清单

以下文档已按边界声明更新：

| 文档 | 更新内容 | 位置 |
|:--|:--|:--|
| `results/reports/real_machine_validation.md` | 顶部边界声明 | L3-11 |
| `results/reports/real_machine_ablation.md` | 顶部边界声明 | L3 |
| `results/reports/real_machine_statistical_significance.md` | 结论重写：移除"真机对性能有影响"表述 | L51-65 |
| `docs/defense_qa_handbook.md` Q8 | 移除"PPO 执行时间优于 FCFS 证明迁移有效" | L175 |
| `docs/defense_qa_handbook.md` Q21 | 移除"真机环境下可能有更大提升空间" | L420 |
| `docs/defense_qa_handbook.md` Q25 | "真机验证" → "真机可用性验证" | L484, L490 |
| `docs/sota_comparison.md` | "真机验证" → "真机可用性验证（可用性验证，非性能验证）" | L57, L67 |

---

## 三、答辩口径标准答案

### 问：真机实验证明了什么？

**答**：真机实验证明了系统已完成天衍云平台接入和端到端任务闭环：
- SDK 认证、任务提交、状态轮询、结果获取全链路验证通过
- 284 次真机调用 100% 成功，287 量子比特超导处理器
- 完整 task ID 审计记录，无 Mock 调用

### 问：真机实验是否证明了性能提升？

**答**：没有。真机实验定位为**平台可用性验证**，不作为性能提升证据：
- 样本量小（N=5-10 seeds），统计功效不足
- 任务规模小（1-3 qubit），受免费机时包限制
- mixed_real vs simulation 统计不显著（p=0.344）

性能提升结论由仿真实验支撑：PPO vs FCFS +88.3%（N=250, p=1.032e-42, rank-biserial=-0.71）。

### 问：为什么不直接在真机上做大规模性能验证？

**答**：受以下约束：
- 免费机时包限制：单任务最大 1 qubit（`FREE_TIER_MAX_QUBITS=1`）
- 硬件排队：天衍-176 为共享资源，单任务平均等待 10-30 秒
- 训练需要数千步交互，真机延迟（~10s/task）使训练周期过长
- 量子比特退相干限制电路深度

大规模真机性能验证是后续扩展方向，需申请更多机时配额和专用硬件通道。

---

## 四、数据完整性声明

### 可用性数据（可信）

- 284 条真机 task ID 审计记录
- 100% 成功率（completed 状态）
- 0 次 Mock 调用（纯真机验证）
- 0 次降级触发（连续失败 < 3）

### 性能数据（不可作为性能证据）

- mixed_real reward 异常：6/10 seeds 与 simulation/pure_real 15 位有效数字完全一致
- 有效分叉样本仅 4/10 seeds
- pure_real 仅 3 seeds，无法计算 95% CI
- 真机参与率极低（1.70%），真机 reward 在总 reward 中占比极小

详见 `results/reports/real_machine_validation.md` 数据质量说明章节。

---

## 五、后续验证路线图

| 阶段 | 目标 | 前置条件 |
|:--|:--|:--|
| 当前 | 平台可用性验证 ✅ | — |
| 短期 | 真机反馈语义验证（#235） | result_aware 模式消融实验 |
| 中期 | 真机小规模性能验证 | 申请 50+ qubit 机时包 |
| 长期 | 真机大规模性能验证 | 专用硬件通道 + 1000+ seeds |

---

*Issue #204 验收文件 | 2026-07-21*
