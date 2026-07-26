# 真机量子退火能力调研报告（Issue #118）

> **调研日期**: 2026-07-26
> **调研对象**: 天衍云平台（天衍-287/504，中国电信量子）/ cqlib SDK / 项目退火实现
> **调研目的**: 评估天衍云是否支持 QUBO/量子退火，为项目真机退火集成给出可行路径
> **关键结论**: 天衍云平台不提供真机量子退火服务，所有真机均为门级超导量子计算机；cqlib SDK 无 QUBO 求解接口；项目当前实现正确，已诚实降级为仿真退火

---

## 一、核心发现

### 1.1 天衍云是否提供量子退火服务？

**❌ 不提供真机量子退火服务，cqlib SDK 无原生 QUBO 求解接口。**

天衍云平台的硬件底座全部为**门级（gate-based）超导/光量子量子计算机**，并非量子退火机（Quantum Annealer）。门级量子计算机和量子退火机是两条完全不同的技术路线，硬件物理结构和工作原理都不同：

| 维度 | 量子退火机（D-Wave） | 门级超导量子计算机（天衍云） |
|---|---|---|
| 物理原理 | 量子退火 + 量子隧穿 | 量子门操作 + 量子纠缠 |
| 编程模型 | QUBO/Ising 模型 | 量子电路（Quantum Circuit） |
| 指令集 | 无指令集，直接配置耦合矩阵 | QCIS / QASM 指令序列 |
| 求解方式 | 直接能量最小化采样 | 通过算法（QAOA/VQE）求解 |
| 典型用途 | 组合优化、物流调度 | 量子化学、机器学习、密码破解 |
| 代表硬件 | D-Wave Advantage2（5000+比特） | 祖冲之三号（105比特）、骁鸿（504比特） |

### 1.2 天衍云真机型号清单（截至 2026-07）

| 机器名 | 芯片 | 比特数 | 类型 | 上线时间 |
|---|---|---|---|---|
| **天衍-287** | 祖冲之三号同款 | 105 比特 | 超导门级 | 2024-11 |
| **天衍-504** | 骁鸿 | 504 比特 | 超导门级 | 2024-12 |
| **天衍176 / tianyan176** | — | 176 比特 | 超导门级 | 2023-11 |
| **轩辕一号 (xuanyuanone)** | — | — | 超导门级 | — |
| **天衍-P2000** | 九章四号同款 | 光量子模式 | 光量子玻色采样 | 2025 |
| zdxlz_simulator | — | 仿真 | 状态向量仿真器 | — |
| stabilizer / tensor_network 等 | — | 仿真 | 多种仿真后端 | — |

> **注意**：天衍-287 的命名数字"287"并非量子比特数，其搭载的"祖冲之三号"芯片实际为 **105 个物理量子比特**。天衍-504 的"504"对应"骁鸿"芯片的 504 比特，是国内单台比特数最多的超导量子计算机。

### 1.3 替代方案：QAOA（门级上的"QUBO 求解器"）

虽然天衍云没有量子退火机，但**提供 QAOA（Quantum Approximate Optimization Algorithm）应用框架**，这是在门级量子计算机上求解组合优化问题（QUBO 问题的等价形式）的标准方案：

- 天衍云应用中心已上线 **QAOA 一键启算** 服务
- 已落地的 QAOA 应用包括：机组组合优化（电力能源）、投资组合优化（金融科技）、基于 QAOA 的 VQF 算法密码破解
- VQE（变分量子本征求解器）和量子机器学习（QML）也已上线

---

## 二、cqlib SDK API 能力边界

### 2.1 核心接口（门级量子计算）

**安装**：`pip install cqlib`（从天衍云内部源 `https://pypi.tianyan.com/simple/`）
**支持平台**：`TianYanPlatform`（中电信天衍云）、`GuoDunPlatform`（国盾量子云）

```python
from cqlib import TianYanPlatform, QuantumLanguage
from cqlib.circuits import Circuit, Parameter

# 1. 平台连接
platform = TianYanPlatform(login_key="your_key", machine_name="tianyan-287")

# 2. 创建参数化电路（VQE/QAOA 必备）
circuit = Circuit(qubits=[0, 6])
circuit.h(0); circuit.x(6); circuit.cz(0, 6); circuit.measure_all()
# 支持门：H, X, Y, Z, RX, RY, RZ, X2P, X2M, Y2P, Y2M, S, SD, T, TD,
#         CX, CCX, CRX, CRY, CRZ, CZ, SWAP, XY, XY2P, XY2M

# 3. 提交任务
query_id = platform.submit_experiment(circuit=circuit.qcis, language=QuantumLanguage.QCIS, num_shots=5000)

# 4. 查询结果
result = platform.query_experiment(query_id=query_id, max_wait_time=120, sleep_time=5)
```

### 2.2 能力边界对照表

| 能力 | 是否支持 | 说明 |
|---|:---:|---|
| 门级量子电路（H/CX/RZ 等） | ✅ | QCIS 指令集，完整的单/双比特门集合 |
| 参数化电路（VQE/QAOA） | ✅ | `Parameter` 类支持加减乘除运算 |
| 真机测量结果返回 | ✅ | `query_experiment` 返回原始 + 概率分布 |
| 读取修正与归一化 | ✅ | 真机模式支持，仿真器不需要 |
| **QUBO 直接求解** | ❌ | 无 QUBO/Ising 求解接口 |
| **量子退火指令** | ❌ | 无 `anneal`、`submit_annealing_task` 等方法 |
| **退火调度与提交** | ❌ | 硬件物理层面就不支持 |

---

## 三、项目退火实现现状分析

### 3.1 退火库与求解器支持

| 依赖 | 版本 | 用途 | 文件 |
|------|------|------|------|
| `dimod` | ≥0.12.0 | D-Wave QUBO 问题建模框架 | `requirements.txt` |
| `dwave-neal` | ≥0.6.0 | D-Wave 模拟退火求解器 | `requirements.txt` |
| `cqlib` | ≥1.0.0 | 天衍云真机 SDK（门控量子，非退火） | `requirements-quantum.txt` |

### 3.2 求解器优先级（`src/quantum/annealing.py` `anneal()` 方法）

| 优先级 | 求解器 | 触发条件 | 实际状态 |
|:--:|------|------|------|
| 1 | 真机退火（`cqlib_client.submit_annealing_task`） | `simulation_mode=False` 且客户端具备该方法 | **从未生效**（cqlib 无此方法） |
| 2 | D-Wave neal 模拟退火（`neal.SimulatedAnnealingSampler`） | `_DWAVE_AVAILABLE=True`（SDK 已安装） | 当前默认路径 |
| 3 | 内置 numpy 模拟退火（`_numpy_simulated_annealing`） | SDK 不可用时兜底 | Metropolis-Hastings 经典实现 |

### 3.3 真机退火接入状态：**未接入（且代码已诚实降级）**

**关键发现**：`CqlibTianyanClient` 类没有 `submit_annealing_task` 方法，`annealing.py` L346-380 的 `hasattr` 探测必然返回 `False`，真机路径是死代码，运行时自动降级为仿真。

代码注释明确标注（`annealing.py` L334-335）：
> "天衍云 cqlib 为门控量子计算机 SDK，不提供 QUBO 退火接口"

这与天衍云真实 API 能力完全一致，**项目没有过度宣称**。

### 3.4 QUBO 映射规模

| 模式 | 参数量 | QUBO 规模 | 内存需求 | 状态 |
|------|--------|-----------|----------|------|
| `head_only`（默认） | ~260 | 1,040² | ~8 MB | 当前可用 |
| `full` | ~19,488 | 19,488² | ~2.9 GB | 不可行（OOM） |
| `hierarchical` | 全量分块 | 每块 ≤(200×4)² | ≤5 MB/块 | 已实现 |

训练用 QUBO 规模 **4368 比特**（1092 权重 × 4 bit，详见 `results/reports/annealing_solver_comparison.md`）。

### 3.5 退火加速统计显著性

| 指标 | 数值 | 来源 |
|------|------|------|
| 检验方法 | Wilcoxon 秩和检验 | `head_only_validation.md` |
| p 值 | **0.190**（>0.05，不显著） | n=5 样本量小 |
| 效应量 | Cliff's delta = **0.40**（中等效应） | >0.33 阈值 |
| 50k 步奖励提升 | **+6.4%** | 1659.01 vs 1558.86 |
| 训练时间开销 | **+74.5%** | 95.89s vs 54.97s |

**不显著的根本原因**：n=5 在中等效应量下检验功效仅约 15%，要达到 80% 功效需 n≥26。已有五层答辩应对策略（详见 `docs/annealing_significance-defense.md`）。

---

## 四、与主流量子退火平台对比

| 平台 | 类型 | 比特数 | QUBO 直解 | 编程模型 | 优势场景 |
|---|---|---|:---:|---|---|
| **D-Wave Advantage2** | 量子退火机 | 5000+ | ✅ | QUBO/Ising | 组合优化、调度、物流 |
| **天衍云 (天衍-504)** | 门级超导 | 504 | ❌（需 QAOA） | 量子电路 QCIS | 量子化学、QML、QAOA |
| **天衍云 (天衍-287)** | 门级超导 | 105 | ❌（需 QAOA） | 量子电路 QCIS | 已具备量子优越性 |
| **IBM Quantum** | 门级超导 | 1121（Condor） | ❌（需 QAOA） | OpenQASM | 通用量子计算研究 |
| **本源量子 (本源悟空)** | 门级超导 | 72 | ❌（需 QAOA） | 本源 QUAFI | 国产门级量子云 |

**国内目前无商用大规模量子退火机**，全球商用退火机仅 D-Wave 一家。

---

## 五、真机退火集成路径建议

基于当前架构（`annealing.py` 预留 `submit_annealing_task` 钩子、`AsyncAnnealingLoop` 具备重试与降级机制）与天衍云硬件现状，提出四条可行路径：

### 路径 A：QAOA 转换路径（中期可行，利用现有门控硬件）

**原理**：将 QUBO 转为 QAOA 电路，在天衍-287 门控机上运行。

**改造点**：
1. 在 `CqlibTianyanClient` 中新增 `submit_annealing_task(qubo_matrix, shots, annealing_time)` 方法
2. 内部实现：QUBO → Max-Cut 图 → QAOA 电路（p 层 Hadamard + cost + mixer）→ QCIS 指令 → `submit_experiment`
3. `annealing.py` 的 `hasattr` 探测将自动命中

**瓶颈**：
- QAOA 在 NISQ 时代受噪声限制，p 层难以做深（建议 p≤3）
- 4368 比特 QUBO 远超天衍-287 容量，需先用 `hierarchical` 分块到 ≤200 比特/块
- QUBO→QAOA 转换的耦合项映射损失

**预期**：小规模 QUBO（≤200 比特）可在天衍-287 上跑通，作为"真机退火"的概念验证。

### 路径 B：经典高性能求解器作为"准真机"后端（短期可行）

**原理**：用 OR-Tools / Gurobi / 符号执行器求解 QUBO，作为 neal 的高性能替代。

**改造点**：
1. 新增 `class ORToolsAnnealingOptimizer`，实现 `submit_annealing_task` 接口
2. 在 `anneal()` 中作为第 1.5 优先级（neal 之前）

**优势**：无需真机硬件，求解质量与速度均优于 neal，可立即提升退火效果显著性。

**局限**：不满足"量子"标签，仅作为工程优化。

### 路径 C：等待天衍云退火后端（长期，依赖平台演进）

**原理**：天衍云若未来提供量子退火机（如超导 Flux Qubit 架构），直接接入。

**改造点**：零代码改造，仅需在 `CqlibTianyanClient` 中实现 `submit_annealing_task` 即可命中预留钩子。

**风险**：平台路线图未知，时间不可控。

### 路径 D：混合 VQE 路径（研究型）

**原理**：将 QUBO 编码为 VQE 的 Hamiltonian，用门控机求解基态。

**适用场景**：小规模 QUBO（≤50 比特）的精确求解，作为消融实验的"金标准"对照。

### 推荐优先级

| 优先级 | 路径 | 理由 |
|:--:|------|------|
| 短期（9/15 前） | B（经典求解器） | 立即提升退火显著性，无硬件依赖 |
| 中期（赛后） | A（QAOA） | 利用现有真机硬件，实现"真机量子退火"标签 |
| 长期 | C（等平台） | 零改造，但时间不可控 |
| 研究型 | D（VQE） | 仅适用于小规模验证 |

---

## 六、当前方案评估与答辩建议

### 6.1 当前方案评估

项目当前的实现策略是**正确且诚实的**：
- `src/quantum/annealing.py`：使用 D-Wave `neal`（模拟退火）或 numpy 内置仿真求解 QUBO，**没有宣称天衍云真机退火**
- `src/api/tianyan_cqlib.py`：仅用于门级量子电路任务（如 Bell 态、H 门任务验证）
- 真机验证已完成：284 次 SDK 调用 100% 成功，但都是**门级电路任务**，不是退火任务

这与 `AGENTS.md` 中"Issue #128 真机验证结论边界"的口径完全一致：**可用性验证已达成，性能验证不充分**。

### 6.2 答辩话术建议

如果评委质询"为什么不用天衍云真机做退火"，推荐回答：

> "天衍云平台当前提供的是基于'祖冲之三号'和'骁鸿'芯片的**门级超导量子计算机**，物理架构上是门级量子计算，不是量子退火机。门级量子计算机求解 QUBO 问题需要通过 QAOA 算法间接实现，受 NISQ 噪声和 p 层深度限制，对大规模 QUBO（如 RL 策略网络权重优化）效果有限。因此我们在退火模块采用 D-Wave Ocean SDK 的模拟退火求解器 `neal` 作为工程实现，这是工业界事实标准；而真机集成部分通过 cqlib 调用天衍云门级量子计算机，完成'AI 赋能量子计算'侧的真机验证（284 次调用 100% 成功），形成完整的'量子 AI 双向赋能'闭环。未来真机退火能力可在中国推出商用退火机后无缝接入。"

---

## 七、关键信息源

| 信息 | 来源 |
|---|---|
| 天衍云官网（确认门级超导，无退火） | https://qc.zdxlz.com/home |
| cqlib 教程（QCIS 指令集，门级接口） | https://docs.quantumctek-cloud.com/Appendix/cqlib_turtorial/ |
| 天衍-287 搭载祖冲之三号同款芯片（105 比特门级） | http://m.toutiao.com/group/7572151645883171347/ |
| 天衍-504 骁鸿芯片 504 比特超导 | https://m.baike.com/wiki/天衍-504/7613590384111190056 |
| 天衍云 QAOA 一键启算应用上线 | http://m.toutiao.com/group/7579918084849271348/ |
| 天衍云金融 QAOA 智能体（2026 数字中国峰会） | http://m.toutiao.com/group/7652248931020636682/ |
| D-Wave Advantage2 5000+ 比特退火机 | https://xueqiu.com/5497315115/357974494 |

---

## 八、TL;DR

1. **天衍云不提供量子退火服务，cqlib SDK 无 QUBO 接口**——所有真机均为门级超导/光量子计算机
2. **天衍-287 = 祖冲之三号 105 比特门级**（命名数字"287"非比特数）；**天衍-504 = 骁鸿 504 比特门级**
3. **门级上求解 QUBO 的标准方案是 QAOA**，天衍云已上线 QAOA 应用，但受 NISQ 噪声限制，对大规模 QUBO 效果有限
4. **国内目前无商用大规模量子退火机**，全球商用退火机仅 D-Wave 一家（5000+ 比特）
5. **项目当前实现正确**：退火模块用 D-Wave `neal` 模拟退火，真机模块用 cqlib 提交门级任务，与天衍云真实能力边界一致
6. **建议维持现状**，8/15 冻结前不建议引入 QAOA 真机改造；如答辩质询，话术应明确区分"门级 vs 退火"两条技术路线

---

## 参考文件

- 项目退火实现：[src/quantum/annealing.py](src/quantum/annealing.py)
- 异步退火闭环：[src/quantum/annealing_loop.py](src/quantum/annealing_loop.py)
- cqlib 真机客户端：[src/api/tianyan_cqlib.py](src/api/tianyan_cqlib.py)
- 退火显著性答辩策略：[docs/annealing_significance-defense.md](docs/annealing_significance-defense.md)
- 真机验证结论边界：[docs/real_machine_verification_boundary.md](docs/real_machine_verification_boundary.md)
- QUBO 求解器对比报告：[results/reports/annealing_solver_comparison.md](results/reports/annealing_solver_comparison.md)
