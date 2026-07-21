# 真机测量结果反馈闭环验证报告（Issue #235）

> **生成时间**: 2026-07-21
> **关联 Issue**: #235
> **验证目标**: 证明真机测量结果（而非仅 completed 状态）实际进入 RL reward

## 1. 问题背景

### 评委指出的核心缺陷

> 当前所谓"真机闭环"的反馈语义不足以证明量子计算结果参与了 RL 学习：
> - `poll_pending_real_tasks()` 在任务状态为 `completed` 时加入固定 `REAL_MACHINE_SUCCESS_BONUS`
> - 代码没有把真机测量分布、保真度、任务目标值或解质量纳入 reward
> - 如果评委追问"去掉真机测量结果，只保留 completed 状态，训练是否完全一样"，当前答案接近"是"

### 修复前的代码行为

```python
# 旧行为（status_only 模式）
if status_str == "completed":
    total_feedback += REAL_MACHINE_SUCCESS_BONUS * env.real_machine_feedback_weight
    # 测量结果被完全忽略
```

## 2. 修复方案

### 2.1 三种反馈模式

| 模式 | 行为 | 用途 |
|:--|:--|:--|
| `status_only` | 固定 bonus=2.0，不解析测量结果 | 旧行为，向后兼容 |
| `result_aware` | 解析测量分布，按保真度计算 reward [0.5, 5.0] | **语义闭环** |
| `shuffled` | 打乱测量结果后按保真度计算 | **消融对照组** |

### 2.2 Reward 计算公式

```
quality = fidelity(measured, theoretical)
reward = 0.5 + quality * (5.0 - 0.5)
       = 0.5 + quality * 4.5
```

- `fidelity = 1.0`（完美匹配）→ `reward = 5.0`
- `fidelity = 0.0`（完全偏离）→ `reward = 0.5`
- `fidelity = 0.97`（接近理论）→ `reward ≈ 4.87`

### 2.3 保真度计算

经典保真度（classical fidelity）：

```
F(p, q) = (Σᵢ √(pᵢ · qᵢ))²
```

其中 `p` 是真机测量分布，`q` 是理论分布。

### 2.4 可追溯性记录

每条真机结果记录包含：

| 字段 | 说明 |
|:--|:--|
| `task_id` | 任务标识 |
| `real_task_id` | 真机任务 ID |
| `machine` | 执行机器 |
| `shots` | 采样次数 |
| `feedback_mode` | 反馈模式 |
| `probability` | 测量概率分布 |
| `fidelity` | 保真度 |
| `reward_delta` | 实际 reward 增量 |
| `formula` | 计算公式描述 |
| `result_valid` | 结果是否有效 |

## 3. 新增代码文件

### 3.1 核心函数（`src/scheduler/env_real_machine.py`）

| 函数 | 功能 |
|:--|:--|
| `parse_measurement_result(status)` | 从真机状态解析概率分布（支持 probability/resultStatus/result 三路径） |
| `compute_theoretical_distribution(qcis)` | 根据 QCIS 电路计算理论分布（H 门均匀/X 门确定态） |
| `compute_result_fidelity(measured, theoretical)` | 经典保真度计算 |
| `compute_real_result_reward(measured, theoretical)` | 质量感知 reward 计算 |
| `shuffle_measurement(measured)` | 打乱测量结果（消融对照） |
| `_compute_real_feedback(env, pending, status)` | 根据模式计算反馈（内部） |
| `_record_real_result(env, pending, status, ...)` | 记录结果元数据（内部） |

### 3.2 类型常量（`src/scheduler/env_types.py`）

```python
REAL_FEEDBACK_STATUS_ONLY = "status_only"
REAL_FEEDBACK_RESULT_AWARE = "result_aware"
REAL_FEEDBACK_SHUFFLED = "shuffled"
REAL_RESULT_REWARD_MAX = 5.0
REAL_RESULT_REWARD_MIN = 0.5
```

### 3.3 环境参数（`src/scheduler/env.py`）

```python
QuantumSchedulingEnv(
    real_feedback_mode="result_aware",  # 默认 status_only，可切换
)
```

## 4. 测试覆盖

### 4.1 单元测试（`tests/test_real_result_feedback.py`）

34 个测试，覆盖 7 个维度：

| 测试类 | 测试数 | 覆盖内容 |
|:--|:--:|:--|
| TestParseMeasurementResult | 7 | probability/resultStatus/result/空/无效值 |
| TestComputeTheoreticalDistribution | 4 | H 门/X 门/多比特/无测量 |
| TestComputeResultFidelity | 6 | 完美/完全不匹配/部分匹配/空/clamp |
| TestComputeRealResultReward | 5 | 最大/最小/空/公式可追溯/线性映射 |
| TestShuffleMeasurement | 4 | 改变分布/保持值集/单结果/空 |
| TestRealFeedbackModes | 6 | 三种模式/无效模式/初始化/reset 清空 |
| TestPollPendingRealTasksFeedback | 2 | result_aware 集成/status_only 集成 |

### 4.2 验证结果

```
tests/test_real_result_feedback.py: 34 passed
tests/test_env_real_machine.py: 12 passed (无回归)
tests/test_scheduler.py: 74 passed (无回归)
总计: 120 passed
```

## 5. 消融实验设计

### 5.1 四组对比

| 组别 | 反馈模式 | 说明 |
|:--|:--|:--|
| Simulation | 无真机 | 纯仿真训练 |
| Real-status-only | `status_only` | 仅 completed bonus（旧行为） |
| Real-result-aware | `result_aware` | 测量结果按保真度计算 reward |
| Shuffled-result control | `shuffled` | 打乱测量结果（检验是否只是噪声注入） |

### 5.2 预期结论

1. **Real-result-aware vs Real-status-only**: 如果真机测量质量有信息量，result_aware 应优于 status_only
2. **Shuffled-result vs Real-status-only**: 如果 shuffled 不优于 status_only，说明不是单纯奖励注入在起作用
3. **Real-result-aware vs Shuffled-result**: 差异应来自测量结果与任务目标的语义关联

### 5.3 运行方式

```python
# status_only 模式（旧行为）
env = QuantumSchedulingEnv(real_feedback_mode="status_only", ...)

# result_aware 模式（语义闭环）
env = QuantumSchedulingEnv(real_feedback_mode="result_aware", ...)

# shuffled 模式（消融对照）
env = QuantumSchedulingEnv(real_feedback_mode="shuffled", ...)
```

## 6. 评委追问回答

### Q: "去掉真机测量结果，只保留 completed 状态，训练是否完全一样？"

**修复前**：是。测量结果被完全忽略，只有 `completed` 状态给固定 bonus。

**修复后**：否。`result_aware` 模式下，真机测量分布通过保真度直接影响 reward：
- 高保真度测量 → reward=5.0
- 低保真度测量 → reward=0.5
- 这使得 PPO 能学习到"选择能产生高质量量子结果的调度策略"

### Q: "如何手算一次真机 reward？"

给定 H 门电路（理论 50/50），真机返回 `{"0": 0.52, "1": 0.48}`：

1. 保真度: F = (√(0.52×0.5) + √(0.48×0.5))² = (0.5099 + 0.4899)² = 0.9998² ≈ 0.9996
2. reward = 0.5 + 0.9996 × 4.5 = 0.5 + 4.498 = 4.998

可追溯公式记录在 `_real_result_records` 中。

### Q: "Shuffled-result control 怎么验证不是噪声注入？"

如果 shuffled 组与 result_aware 组表现相同，说明 reward 提升仅来自奖励注入而非测量质量。如果 result_aware 显著优于 shuffled，说明测量结果与任务目标的语义关联是关键。

## 7. 代码质量验证

- ruff check: **通过**（0 errors）
- ruff format: **通过**
- mypy: **通过**（0 errors）
- 测试: **120 passed**（34 新增 + 86 回归）

## 8. 待后续工作

- [ ] 真机执行四组消融实验（需瑞哥申请真机要时）
- [ ] 支持更多电路类型（Bell 态、QAOA、VQE）的理论分布计算
- [ ] 将 `_real_result_records` 导出为 JSON 供分析
