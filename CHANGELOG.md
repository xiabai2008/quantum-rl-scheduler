# Changelog

本项目所有重要变更记录于本文件中。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

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
