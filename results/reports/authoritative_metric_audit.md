# 最终数字一致性审计报告

> **⚠️ 历史冻结声明（2026-08-01）**：本审计记录于 2026-07-17，锁定的是 14 维旧口径（+88.3%, p=1.032e-42），**已废弃**。
> 现行权威口径为 16 维交付模型（PPO vs 真实 FCFS **+20.2%**, N=250, Welch t p=7.56e-12, 8.5 基线诚实化；旧 +123.4% 为 vs Hybrid-Default 弱基线），见 `config/statistics.yaml`。
> 本报告仅保留审计过程追溯，对外数字禁止直接使用。

**审计日期**：2026-07-17  
**权威来源（当时）**：`results/reports/strategy_comparison.md`

## 锁定口径（14维旧口径，已废弃）

- PPO 平均奖励：**2746.94**
- FCFS 平均奖励：**1458.77**
- PPO 相对 FCFS 提升：**88.3%**
- 排名：PPO > SJF > FCFS > Random > Greedy > DQN / Quantum-Only > Classical-Only
- 环境口径：`QuantumSchedulingEnv` 原生 14 维（不使用 10 维截断包装），主公平对比在 14 维原生环境下进行

## 本次修正

- 修正答辩 PPT 大纲中 SJF、FCFS 的名次和单次运行等待时间
- 更新 Code Wiki 中残留的 v4 排名
- 修正统计显著性脚本示例中的旧奖励数字
- 为历史 v4 报告增加“已被取代”提示，保留原始实验可追溯性
- 统一 README、AGENTS、项目记忆和答辩手册中的消融贡献排序
- 修正将负数 DQN 奖励直接换算百分比的误导性表述
- 明确 `agent.py` 中 14 维原生环境与 10 维兼容包装器的关系

## 文件范围

审计覆盖仓库内 Markdown、Python、YAML、JSON、TXT、RST，以及存在时的 DOCX/PPTX XML 文本。本次仓库中未发现 DOCX、PPTX 或 PDF 交付物，因此没有遗漏待修改的 Office 文件。

## 自动验证

```bash
venv\Scripts\python.exe scripts\ci\audit_authoritative_metrics.py
venv\Scripts\python.exe -m pytest tests\test_metric_audit.py -v
```

审计脚本会阻止已废弃的核心数字重新进入文档，并验证权威报告的八策略完整排名和 14→10 维说明。
