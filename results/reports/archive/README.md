# 归档报告说明（Archive）

> 本目录存放 `results/reports/` 中已被取代的中间版本/重复版本报告。
> 这些报告保留用于历史追溯与问题复盘，**禁止在答辩 / PPT / 白皮书中引用**。
> 归档操作对应 Issue #544（实验报告冗余 - 多版本并存应归档）。

---

## 归档原则

仅以下两类报告会被归档：

1. **明确标注为 SUPERSEDED / DEPRECATED / INVALID 的报告**：报告头部已声明被新版取代。
2. **带日期戳的旧版本，且已有更新的 v2 版本存在**：日期戳版本为中间产物。

对于仅有一个版本的同类报告，即使内容较旧，也保留在 `results/reports/` 中不归档。

---

## 归档文件清单

| 归档文件 | 类型 | 取代它的最新版本 | 归档原因 |
|:--|:--|:--|:--|
| `multiseed_real_machine_report.md` | 5-seed 探索性结果 | `../multiseed_real_machine_report_10seeds_v2.md` | 报告头部明确标注 SUPERSEDED，权威数字以 10-seed v2 为准 |
| `multiseed_real_machine_report_10seeds.md` | 旧 10-seed 报告 | `../multiseed_real_machine_report_10seeds_v2.md` | 报告头部明确标注 INVALID（机器名/shots 混用、真机任务完成数为 0、统计方法不统一） |
| `multiseed_real_machine_report_20260724.md` | 日期戳中间版本 | `../multiseed_real_machine_report_10seeds_v2.md` | 2026-07-24 生成的中间版本，已被 2026-07-27 生成的 v2 取代 |
| `industry_case_vqe_20260727_103454.md` | 单次运行（N=1） | `../industry_case_vqe_v2.md` | 原报告仅单次运行无显著性检验，v2 通过 10 seeds × 5 episodes（N=50）修复（Issue #462） |
| `compilation_report.md` | 不公平对比设计 | `../compilation_fair_v2_report.md` | 报告头部明确标注 DEPRECATED（Issue #560），-76.4% 数字已废弃 |

---

## 引用指引

- 需要引用真机多 seed 结论时，请使用 `results/reports/multiseed_real_machine_report_10seeds_v2.md`。
- 需要引用 VQE 行业场景结论时，请使用 `results/reports/industry_case_vqe_v2.md`。
- 需要引用编译层 SWAP 优化结论时，请使用 `results/reports/compilation_fair_v2_report.md`。

如需查阅历史演进过程或问题复盘，可在本目录中查阅对应归档文件，但不得将其数字作为权威结论引用。
