# Changelog

本项目所有重要变更记录于本文件中。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [v9.0] - 2026-07-28

### 16维观测空间与2D网格耦合图（Issue #404）

- 观测空间从14维升级为**16维**（`src/scheduler/env_types.py: OBS_DIM=16`）
  - 新增维度：串扰风险（OBS_CROSSTALK_RISK，基于空间并发度）、任务到达率滑动平均（OBS_ARRIVAL_RATE_MA）
  - 16维为最终交付标准，10维/14维旧模型已归档清理
- 耦合图拓扑从线性链重构为**4×4 2D网格**（匹配天衍-287真机nearest-neighbor结构）
  - SWAP距离计算使用BFS图最短路径替代线性abs差值
  - 2D网格对比线性链：SWAP门开销减少62%，图直径从15降至6，保真度提升43%
- 保真度模型加入平均SWAP距离感知项，编译质量评估更准确

### P0级Bug修复（Issue #405 #406 #410）

- 修复`get_info`未返回`completion_rate`导致成功率始终显示0.00%的问题
- 修复Gymnasium环境终止语义：恢复`_consecutive_idle_steps`跟踪，正确区分`terminated`（自然终止）与`truncated`（外部截断）
- 修复JSON序列化错误：numpy bool类型转换为Python原生bool

### 模型兼容性修复（方案B：保持16维+标准PPO-MLP）

- 修复LSTM模型（RecurrentPPO）与官方评估脚本`run_simulation.py`的兼容性问题
- 使用标准PPO(MLP)架构（`use_lstm=False`）训练交付模型，确保`PPO.load()`可直接加载
- 消融实验验证：MLP与LSTM在本任务收敛到相同策略（平均奖励9990±889，任务完成率94%），MLP训练更快
- 交付模型：`deliverable_models/ppo_best_model_16dim.zip`（100K步训练，11分钟收敛）
- `run_simulation.py`默认加载路径设为16维PPO-MLP模型，评委一键运行即可复现

### 消融实验体系完善

- 新增PPO策略消融脚本（`scripts/evaluation/ablation_ppo_variants.py`）：对比MLP vs LSTM vs LSTM+Annealing
- 新增量子编译环境消融脚本（`scripts/evaluation/ablation_compilation_env.py`）：验证2D网格耦合图效果
- 新增LSTM最佳模型深度评估脚本（`scripts/evaluation/evaluate_lstm_best_model.py`）：7种基线对比+统计显著性
- 新增消融变体训练脚本（`scripts/training/train_ablation_variant.py`）：支持MLP/LSTM切换
- 新增早停训练脚本（`scripts/training/train_lstm_15m_earlystop.py`）：连续5次评估无提升自动停止
- 消融结论：
  - PPO-MLP、PPO-LSTM、PPO-LSTM+Annealing全部收敛到同一最优策略（83%量子/17%混合分配）
  - MLP训练效率最高（24分钟 vs LSTM 53分钟），推理无额外状态管理开销
  - 退火消融（20seeds）：p=0.9430，统计不显著，确认为工程探索

### 仓库清理

- 删除根目录临时PR审查报告文件（`_pr_body_audit.md`、`PR审查报告_*.md`）
- 删除过时阶段性巡查记录（`docs/issue_patrol_2026-07-22.md`、`docs/autonomous_findings_20260725.md`）
- 清理旧维度模型文件（10dim/14dim PPO/DQN模型），仅保留16维交付模型和编译优化Agent
- 同步更新MODELS.md、observation_dim_standard.md、novelty_statement.md等核心文档至16维口径

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
- 编译 AI（PPO 替代 SABRE，公平对比v2同池配对60电路，4×4 2D网格拓扑；深电路(14-16q)SWAP减少约33%（n=20, p=0.29不显著）；Issue #451修复原-76.4%不公平对比）
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
