# Changelog

## [Unreleased] 退火模块现代化（#245, #246, #229, #254, #255）

### 新增
- `config/config.yaml` 新增完整 `annealing:` 配置节，覆盖 13 个退火参数
  （`simulation_mode`, `num_qubits`, `shots`, `annealing_time`, `sim_initial_temp`,
  `sim_cooling_rate`, `sim_num_sweeps`, `reg_lambda`, `max_delta_ratio`,
  `accept_threshold_ratio`, `head_only`, `max_params_per_block`, `block_strategy`），
  每项均附默认值与有效范围注释；`.env.example` 同步以 `ANNEALING_` 前缀镜像。
- 新增 `config_loader.py: load_annealing_config()`，读取 `annealing` 配置节并兜底默认值，
  支持环境变量（`ANNEALING_` 前缀）覆盖。
- `QuantumAnnealingOptimizer` 与 `AsyncAnnealingLoop` 的 `__init__` 新增 `config` 参数，
  退火参数可由配置驱动（向后兼容：`config=None` 时与原有硬编码默认值完全一致）。
- 量子加速降级日志（Issue #229）：`anneal()` 在降级时通过 `logger.warning` 输出
  降级原因（`simulation_mode=True` / `cqlib_client is None` / `无 submit_annealing_task 接口` /
  真机退火失败）、降级求解器（`neal_sa` / `numpy_sa`）与 QUBO 矩阵规模；
  `optimize_policy()` 在 `QUANTUM_ACCELERATION_ENABLED` 禁用时也记录降级。
  降级事件聚合于 `optimizer._degradation_log`，并由 `AsyncAnnealingLoop` 在实验报告导出中
  通过每条记录的 `degradation_log` 字段输出（仅首次出现某原因时记录，避免刷屏）。

### 清理
- 删除已弃用的向后兼容别名方法：`_get_full_policy`、`_extract_weights`、
  `_numpy_simulated_annealing`、`_compute_qubo_energy`。
  经排查（Issue #254）这 4 个方法在仓库内均无生产调用方（仅测试 docstring 提及公共接口），
  移除后无 `DeprecationWarning` 来源；相关断言已由公共接口（`get_full_policy` /
  `extract_weights` / `compute_qubo_energy` / `numpy_simulated_annealing`）覆盖。

### 注意事项（建议后续处理）
- `src/config/schema.py` 的 `AnnealingConfig` 默认值（`num_qubits=10`、`annealing_time=1.0`）
  与本次 `config/config.yaml` 更新后的取值（`16` / `20.0`）不一致，建议在后续 PR 中同步
  schema 默认值（非本次强制范围，未改动 schema 以免超出本次改动面）。
- 本报告未改动 `README.md`（由其他维护者负责）。
