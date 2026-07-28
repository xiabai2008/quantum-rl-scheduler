# Changelog

本项目所有重要变更记录于本文件中。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [v8.0] - 2026-07-27

### 统计数字迁移与权威源建立（Issue #141 / #431 / #434）
- 建立 `config/statistics.yaml` 单一权威统计源，消除 4 套 p 值混用问题
- PPO vs FCFS 仿真 p 值统一为 `1.032e-42`（Mann-Whitney U 检验），淘汰 `1.0e-42`、`1.03×10⁻⁴²`、`p<0.001` 等不精确表述
- Random 基线均值由旧值 `1247.17` 修正为权威值 `1217.08`（源自 `results/multiseed_evaluation/rewards_multiseed.json`，N=250）
- CI 新增 `scripts/ci/check_stats_consistency.py` + `validate_authoritative_numbers.py` 自动校验文档数字一致性

### 叙事文档重写
- 技术白皮书 / 答辩 PPT 大纲 / 参赛总结报告全面对齐权威数字（+88.3%，N=250，p=1.032e-42）
- 真机实验边界严格区分"可用性验证"与"性能验证"两级（Issue #128）

### 退火模块优雅降级（2026-07-27）
- 退火默认关闭（`annealing.enabled=false`），dimod/dwave-neal 降级为可选依赖
- 退火方法标注 `deprecated`，探索性功能不再投入开发（统计不显著 p=0.9430）
- 量子赋能 AI 主方向由 QUBO 退火转向真机噪声反馈

### 新增能力
- 编译 AI（PPO 替代 SABRE，公平对比v2同池配对60电路，4×4 2D网格拓扑；深电路(14-16q)SWAP减少约33%；Issue #451修复原-76.4%不公平对比）
- VQE 行业场景（10 分子×100 任务，PPO +97.5% vs FCFS）
- OR-Tools 对比（CP-SAT 静态最优，PPO 动态实时优势）
- 多 seed 真机实验 v2（N=10 per group，Cohen's d=5.33，p<0.001）

### 工程清理
- 删除根目录 0 字节垃圾文件与临时杂物文件
- `requirements.lock` 重生成，移除 plotly/redis 等已移除包
- 文档计数同步（src 67 py / tests / reports）

## [Unreleased]

### Removed（Issue #255）

删除以下 4 个已弃用别名方法，降低 `src/quantum/annealing.py` 文件阅读负担。
所有调用方应改用对应的公共接口：

| 已删除方法 | 公共接口替代 | 弃用原因 |
|------------|-------------|----------|
| `QuantumAnnealingOptimizer._get_full_policy(agent)` | `QuantumAnnealingOptimizer.get_full_policy(agent)` | 别名重复，公共接口已稳定 |
| `QuantumAnnealingOptimizer._extract_weights(network)` | `QuantumAnnealingOptimizer.extract_weights(network)` | 别名重复，公共接口已稳定 |
| `QuantumAnnealingOptimizer._numpy_simulated_annealing(qubo_matrix)` | `QuantumAnnealingOptimizer.numpy_simulated_annealing(qubo_matrix)` | 别名重复，公共接口已稳定 |
| `QuantumAnnealingOptimizer._compute_qubo_energy(solution, qubo_matrix)` | `QuantumAnnealingOptimizer.compute_qubo_energy(solution, qubo_matrix)` | 别名重复，公共接口已稳定 |

**迁移指引**：

调用方只需将方法名前的下划线前缀去掉即可，参数与返回值完全一致：

```python
# 旧（已删除）
opt._numpy_simulated_annealing(qubo_matrix)
opt._compute_qubo_energy(solution, qubo_matrix)
QuantumAnnealingOptimizer._get_full_policy(agent)
QuantumAnnealingOptimizer._extract_weights(network)

# 新（公共接口，自 v8 起为推荐用法）
opt.numpy_simulated_annealing(qubo_matrix)
opt.compute_qubo_energy(solution, qubo_matrix)
QuantumAnnealingOptimizer.get_full_policy(agent)
QuantumAnnealingOptimizer.extract_weights(network)
```

**调研依据**：详见 `docs/deprecated_methods_audit.md`（PR #332），
确认 `src/`、`scripts/`、`tests/` 中无生产代码调用以上弃用方法。

**保留方法**：`_apply_weights`（v1，旧版本权重应用）虽标注为"向后兼容保留"，
但有专门的单元测试 `test_apply_weights_v1_linear_interpolation` 覆盖，
本次不删除，留待后续 v9 与 `_apply_weights_v2` 合并重构时统一处理。
