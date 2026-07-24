# 自主巡查发现汇总（2026-07-25）

> 巡查方向：落地价值（heka-ky 主攻方向）
> 巡查范围：docs/value_*.md、docs/platform_landing_value.md、results/reports/、scripts/evaluation/
> 巡查人：heka-ky 自主巡查

---

## 发现清单

### F1: value_quantification.md 统计数字张冠李戴 [P0]

**文件**: `docs/value_quantification.md` 第 182-183 行

**问题**: 汇总表第 9、10 行将 **PPO vs Random** 的统计数字误标为 PPO vs FCFS：
- `p=4.92e-55` → 实际是 PPO vs Random 的 p 值
- `Cohen's d=-1.73` → 实际是 PPO vs Random 的效应量

**正确数字**（来源: AGENTS.md 权威锁定 + strategy_comparison.md）：
- PPO vs FCFS: Welch t p=3.04e-11, Cohen's d=-1.70（AGENTS.md 口径）
- 或 Mann-Whitney U p=1.03e-42, rank-biserial r=-0.708（strategy_comparison.md 口径）

**影响**: 答辩时被问到统计数字来源，如果与 AGENTS.md 不一致会被质疑数据可信度。

**修复方案**: 统一使用 AGENTS.md 权威数字。

---

### F2: value_quantification.md 真机数据过时 [P1]

**文件**: `docs/value_quantification.md` 第 176 行

**问题**: 写的是 "真机调用成功率 100%（284 次）| 天衍-176 真机验证"
但最新数据是 **天衍-287 多seed真机实验**（2026-07-24）：
- PPO=1665.22±324.51 vs FCFS=353.22±53.33
- Cohen's d=5.64, Welch p=6.83e-04（Bonferroni校正后显著）

**影响**: 未利用最新的、统计显著的多seed真机数据来支撑落地价值叙事。

**修复方案**: 更新为天衍-287多seed数据，补充真机统计显著性结论。

---

### F3: 三份价值文档内容高度重叠且数字不一致 [P1]

**文件**: 
- `docs/value_quantification.md`（基础价值量化）
- `docs/value_deep_quantification.md`（深度量化）
- `docs/platform_landing_value.md`（平台落地价值）

**问题**: 三份文档在 ROI 分析、经济价值估算、行业场景方面大量重复，但使用不同版本的数字：
- value_quantification.md: p=4.92e-55, d=-1.73（错误）
- value_deep_quantification.md: p=1.03e-42, r=-0.708（不同口径）
- platform_landing_value.md: 未引用具体 p 值

**影响**: 答辩准备时信息碎片化，难以快速定位权威数字。

**修复方案**: 创建一份统一的答辩用价值摘要，锁定一套权威数字。

---

### F4: ROI 分析脚本未实现（Issue #138 已建但未修） [P1]

**文件**: `scripts/evaluation/roi_analysis.py`（不存在）

**问题**: Issue #138 要求创建自动化 ROI 分析脚本，让商业价值数字可追溯到实验数据。当前所有经济价值估算均为手工填写，标注"低（估算）"置信度。

**影响**: 答辩时无法回答"这些价值数字是怎么算出来的"。

**修复方案**: 实现脚本，从 results/ 数据自动计算 ROI。

---

### F5: 无 VQE 场景演示脚本 [P2]

**文件**: `scripts/demo/` 目录下无 VQE 相关 demo

**问题**: 价值文档反复引用 VQE 场景（50电路×1024 shots），但没有可执行的演示脚本。现有 demo 只有 demo.py、demo_cqlib.py、demo_multi_machine.py。

**影响**: 答辩演示时无法展示具体的 VQE 调度优化效果。

**修复方案**: 创建 VQE 场景模拟脚本，展示 PPO 调度对 VQE 工作流的优化。

---

### F6: multiseed_real_machine_report_20260724.md 不完整 [P2]

**文件**: `results/reports/multiseed_real_machine_report_20260724.md`

**问题**: 
1. "实验时间"字段为 "N/A"
2. 报告在第 89 行被截断，缺少第 4 节结论的完整内容（SJF vs FCFS 结论缺失）

**影响**: 报告不完整，引用时缺乏完整性。

**修复方案**: 补全报告缺失部分。

---

## 优先级排序

| 优先级 | 编号 | 问题 | 预计工时 |
|:--:|:--:|:--|:--:|
| P0 | F1 | 统计数字张冠李戴 | 10min |
| P1 | F2 | 真机数据过时 | 15min |
| P1 | F4 | ROI脚本未实现 | 30min |
| P1 | F3 | 三份文档重叠 | 20min |
| P2 | F5 | VQE演示缺失 | 30min |
| P2 | F6 | 报告不完整 | 5min |

---

## 关联

项目权威数字（不可篡改）：
- 50seed仿真：PPO=2746.94±1121.19 vs FCFS=1458.77±55.85, 提升+88.3%, Welch t检验 p=3.04e-11, Cohen's d=-1.70
- 多seed真机：PPO=1665.22±324.51 vs FCFS=353.22±53.33, Cohen's d=5.64, p=6.83e-04（Bonferroni校正后显著）
- 所有实验数据和PR必须与以上数字一致
