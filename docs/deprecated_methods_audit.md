# 弃用方法调用点调研报告

> **Issue**: #254
> **调研日期**: 2026-07-27
> **调研范围**: `src/`、`scripts/`、`tests/`、`docs/`
> **调研目标**: 确认 5 个弃用方法在删除前无外部生产代码调用

## 调研方法

使用 `grep -rn` 全局搜索以下 5 个弃用方法的调用点：

- `_get_full_policy`
- `_extract_weights`
- `_apply_weights`（注意：不含 `_apply_weights_v2` 和 `_apply_weights_v2_partial`，这两个是当前使用的接口）
- `_numpy_simulated_annealing`
- `_compute_qubo_energy`

## 弃用方法清单与调用点分析

### 1. `_get_full_policy`（src/quantum/annealing.py:1461）

| 调用位置 | 文件 | 行号 | 调用类型 | 结论 |
|----------|------|------|----------|------|
| 定义 | src/quantum/annealing.py | 1461 | 方法定义 | 弃用，含 `warnings.warn` |
| 测试 | tests/test_annealing.py | 1316, 1319 | 单元测试 | 测试弃用方法本身 |

**结论**：✅ **可安全删除**（仅测试调用，无生产代码依赖）

### 2. `_extract_weights`（src/quantum/annealing.py:1491）

| 调用位置 | 文件 | 行号 | 调用类型 | 结论 |
|----------|------|------|----------|------|
| 定义 | src/quantum/annealing.py | 1491 | 方法定义 | 弃用，含 `warnings.warn` |
| 测试 | tests/test_annealing.py | 11, 617 | 单元测试 | 测试弃用方法本身 |

**结论**：✅ **可安全删除**（仅测试调用，无生产代码依赖）

### 3. `_apply_weights`（src/quantum/annealing.py:1524）

| 调用位置 | 文件 | 行号 | 调用类型 | 结论 |
|----------|------|------|----------|------|
| 定义 | src/quantum/annealing.py | 1524 | 方法定义 | 弃用，含 `warnings.warn` |
| 测试 | tests/test_annealing.py | 797, 798, 812 | 单元测试 | 测试弃用方法本身 |

**结论**：✅ **可安全删除**（仅测试调用，无生产代码依赖）

> **注意**：`_apply_weights_v2`（L1557）和 `_apply_weights_v2_partial`（L1603）不是弃用方法，它们是当前使用的接口，在 `annealing.py:998, 1005, 1239` 中被生产代码调用，**不可删除**。

### 4. `_numpy_simulated_annealing`（src/quantum/annealing.py:1819）

| 调用位置 | 文件 | 行号 | 调用类型 | 结论 |
|----------|------|------|----------|------|
| 定义 | src/quantum/annealing.py | 1819 | 方法定义 | 弃用，含 `warnings.warn` |
| 文档引用 | docs/real_machine_annealing_research.md | 108 | 文档说明 | 仅作为历史说明引用 |

**结论**：✅ **可安全删除**（无代码调用，仅文档历史引用）

### 5. `_compute_qubo_energy`（src/quantum/annealing.py:1847）

| 调用位置 | 文件 | 行号 | 调用类型 | 结论 |
|----------|------|------|----------|------|
| 定义 | src/quantum/annealing.py | 1847 | 方法定义 | 弃用，含 `warnings.warn` |
| 测试 | tests/test_annealing.py | 10, 562, 565 | 单元测试 | 测试弃用方法本身 |
| 测试 | tests/test_scheduler.py | 1038 | 单元测试 | 测试弃用方法本身 |

**结论**：✅ **可安全删除**（仅测试调用，无生产代码依赖）

## 调研汇总表

| 弃用方法 | 定义行号 | 生产代码调用 | 测试调用 | 文档引用 | 删除结论 |
|----------|----------|:----------:|:--------:|:--------:|----------|
| `_get_full_policy` | L1461 | ❌ 无 | ✅ tests/test_annealing.py | ❌ 无 | ✅ 可安全删除 |
| `_extract_weights` | L1491 | ❌ 无 | ✅ tests/test_annealing.py | ❌ 无 | ✅ 可安全删除 |
| `_apply_weights` | L1524 | ❌ 无 | ✅ tests/test_annealing.py | ❌ 无 | ✅ 可安全删除 |
| `_numpy_simulated_annealing` | L1819 | ❌ 无 | ❌ 无 | ✅ docs/（历史引用） | ✅ 可安全删除 |
| `_compute_qubo_energy` | L1847 | ❌ 无 | ✅ tests/test_annealing.py, test_scheduler.py | ❌ 无 | ✅ 可安全删除 |

## 删除操作建议

### 步骤 1：删除弃用方法定义

在 `src/quantum/annealing.py` 中删除以下 5 个方法：

- `_get_full_policy`（L1461-1489）
- `_extract_weights`（L1491-1522）
- `_apply_weights`（L1524-1555）
- `_numpy_simulated_annealing`（L1819-1845）
- `_compute_qubo_energy`（L1847-1860）

### 步骤 2：删除或迁移对应测试

在 `tests/test_annealing.py` 中删除或迁移以下测试：

- `test_compute_qubo_energy`（L562-565）
- `_extract_weights` / `_set_weights` 相关测试（L617）
- `_apply_weights` 相关测试（L797-812）
- `_get_full_policy` 相关测试（L1316-1319）

在 `tests/test_scheduler.py` 中删除或迁移：

- `test_compute_qubo_energy`（L1038）

### 步骤 3：更新文档引用

在 `docs/real_machine_annealing_research.md:108` 中更新对 `_numpy_simulated_annealing` 的引用，改为说明已迁移到 `dimod` / `neal` 求解器。

### 步骤 4：更新 CHANGELOG

在 `CHANGELOG.md` 中记录删除的弃用方法列表。

## 验收标准

- [x] 调用点清单完成（5 个方法，共 35 处匹配）
- [x] 每个弃用方法标注"可安全删除"或"需先迁移调用"
- [x] 所有 5 个弃用方法均标注为"✅ 可安全删除"

## 相关文件

- [src/quantum/annealing.py](src/quantum/annealing.py)（弃用方法定义）
- [tests/test_annealing.py](tests/test_annealing.py)（弃用方法测试）
- [tests/test_scheduler.py](tests/test_scheduler.py)（弃用方法测试）
- [docs/real_machine_annealing_research.md](docs/real_machine_annealing_research.md)（文档引用）
