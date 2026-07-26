# PR 审查报告：4 个文档类 PR

> 审查日期：2026-07-26
> 审查员：AI Code Reviewer
> 仓库：xiabai2008/quantum-rl-scheduler

---

## 总览

| PR | 标题 | 作者 | CI 状态 | 建议 |
|:--:|:--|:--:|:--:|:--:|
| #224 | docs(#205,#209,#210): 文档一致性修复与碎片化整理 | NN2914 | Lint FAIL | **需修改** |
| #278 | docs: 统一修改比赛材料中的"量子退火"措辞 (Issue #228) | qpqpalalzmzm112 | 全通过 | **需修改** |
| #276 | docs: 移除 anneal() 文档字符串中"量子退火器"绝对表述 (Issue #227) | qpqpalalzmzm112 | 全通过 | **需修改** |
| #198 | docs(#118): 真机量子退火能力调研报告 | Jackhock-1 | 全通过 | **需修改** |

### 共性问题（4 个 PR 通用）

**所有 4 个 PR 均标记为 "docs" 类型，但都包含大量非文档变更：**

1. **ci.yml 修改**：4 个 PR 都修改了 `.github/workflows/ci.yml`，添加了 `continue-on-error: true`（mutation-testing job）和 `pip install -r requirements-dev.txt`。这些变更与各 PR 声明的文档 Issue 无关。
2. **代码格式化变更**：4 个 PR 都包含对 `src/quantum/annealing.py`、`src/quantum/annealing_loop.py`、`scripts/evaluation/annealing_lr_sweep.py`、`tests/test_annealing.py`、`tests/test_annealing_loop.py` 的相同格式化修改（多行压缩为单行）。
3. **marl.py 变更**：4 个 PR 都修改了 `src/scheduler/marl.py`，添加 BOM 字符（`﻿`）并移除 `cast` 导入和使用。
4. **PR 范围蔓延**：每个 PR 都将文档修改、CI 配置修改、代码格式化、甚至新功能混在一起，违反了"一个 PR 只做一件事"的原则。

**建议**：先将共有的 ci.yml + 格式化 + marl.py 变更合并到 main（它们已通过 CI），然后各 PR rebase 后只保留各自的文档变更。

---

## PR #224: docs(#205,#209,#210): 文档一致性修复与碎片化整理

### CI 状态
- **Lint (ruff format + ruff check + bandit)**: **FAIL** (11s)
- Test / Type Check / Security Audit: 全部 SKIPPED（因 Lint 是前置依赖）
- Auto Label PR / Check Commit Format: PASS

### 核心修改内容

| 类别 | 修改项 | 说明 |
|:--:|:--|:--|
| Issue #205 | README.md Issue 表格清理 | 删除已关闭的 #142-#152 表格，替换为 GitHub Issues 页面指引 |
| Issue #209 | 覆盖率门槛统一为 80% | README/AGENTS/CONTRIBUTING/code_freeze_checklist 等文档：60/70% → 80% |
| Issue #210 | 文档碎片化交叉引用 | 为代码冻结/价值量化/跨硬件/部署 4 类重叠文档建立主从关系引用 |
| 路径修复 | statistical_validation.md | 绝对路径 `C:\Users\HZR\...` → 相对路径 |
| 非文档变更 | ci.yml | 添加 mutation-testing `continue-on-error`、`requirements-dev.txt` |
| 非文档变更 | 代码格式化 | annealing.py / marl.py / tests 等（与其他 3 个 PR 相同） |

### 文档正确性分析

1. **覆盖率门槛 80% 的声明是正确的**：`pyproject.toml` 中 `fail_under = 80` 已验证。文档从 60/70% 更新为 80% 与代码配置一致。
2. **但 ci.yml 仍为 `--cov-fail-under=70`**：PR 修改了文档说 80%，但 ci.yml 中的实际门槛仍是 70%。PR body 承认"需手动更新 ci.yml"，但 diff 显示 PR 确实修改了 ci.yml（添加 continue-on-error），却未更新覆盖率门槛。这是自相矛盾的。
3. **Lint 失败原因推测**：lint job 包含 `python scripts/ci/check_stats_consistency.py --strict`，该脚本可能检测到文档中的 80% 与 ci.yml 中的 70% 不一致，导致失败。
4. **文档碎片化交叉引用**：为多组重叠文档添加了"主文档/分册"关系说明，不删除文件（避免破坏引用），这是合理的处理方式。
5. **value_summary_for_defense.md 的完整重写**：diff 显示删除 135 行 + 新增 138 行，但内容几乎相同（只是添加了 BOM 和 Issue #210 标注）。这种全文件重写可能由行尾符变化导致，增加了 review 难度。

### 项目价值评估

- **高价值**：覆盖率门槛统一和文档碎片化整理都是比赛前急需完成的工作。README Issue 表格清理避免了文档与仓库实际状态不同步的问题。
- **中价值**：统计验证报告的绝对路径修复（`C:\Users\HZR\...`）对可移植性很重要。

### 潜在问题

1. **CI Lint 失败**：必须修复后才能合并。最可能的原因是 `check_stats_consistency.py` 检测到文档(80%)与ci.yml(70%)不一致。
2. **PR body 误导**：声称"本 PAT 无 workflow scope，无法修改 workflow 文件"，但 diff 实际包含 ci.yml 修改。应直接将 `--cov-fail-under=70` 改为 `80`。
3. **混入非文档变更**：ci.yml 修改、代码格式化、marl.py BOM 变更与 Issue #205/#209/#210 无关。

### 处理建议：**需修改**

- 修复 Lint 失败：将 ci.yml 中 `--cov-fail-under=70` 同步更新为 `80`
- 移除非文档变更（ci.yml 的 continue-on-error/requirements-dev.txt、代码格式化、marl.py BOM），或拆分为独立 PR
- 验证 `check_stats_consistency.py --strict` 通过

---

## PR #278: docs: 统一修改比赛材料中的"量子退火"措辞 (Issue #228)

### CI 状态
- Lint / Test (3 versions) / Type Check / Security Audit: **全部 PASS**
- Benchmark / Mutation: FAIL（`continue-on-error: true`，非阻断）

### 核心修改内容

| 文件 | 原文 | 修改后 |
|:--|:--|:--|
| AGENTS.md | 量子退火（requirements.txt） | QUBO退火求解（仿真模拟退火，requirements.txt） |
| README.md | 利用量子退火算法加速 RL 策略搜索过程 | 利用量子退火算法（仿真模拟退火）加速 RL 策略搜索过程 |
| README.md | QUBO求解 | QUBO退火求解（仿真模拟退火） |
| README.md | QUBO映射 + 退火求解 + 异步闭环 | QUBO映射 + 退火求解（仿真） + 异步闭环 |
| api_reference.md | 量子退火器 | 退火求解器（仿真模拟退火） |
| bidirectional_empowerment.md | QUBO → 量子退火 (Simulated Annealing) / D-Wave 经典算法，真机验证通过 | QUBO → 退火求解（Simulated Annealing）/ D-Wave neal 仿真，QUBO 兼容真机量子退火 |

非文档变更：ci.yml 完整重写（BOM）、代码格式化、marl.py BOM + cast 移除（与 PR #276/#198/#224 相同）。

### 文档正确性分析

1. **技术准确性**：项目实际使用 D-Wave `neal` 模拟退火求解器（已验证 `src/quantum/annealing.py` 源码），不是真机量子退火。将"量子退火"修改为"仿真模拟退火"是**正确且必要的**。
2. **bidirectional_empowerment.md 的修改值得商榷**：原文"D-Wave 经典算法，真机验证通过"被改为"D-Wave neal 仿真"。原表述虽然不够精确，但"真机验证通过"指的是 cqlib 门级任务的真机验证（284 次成功），不是退火真机验证。修改后更准确，但需确认上下文不会产生新的歧义。
3. **措辞冗余**："QUBO退火求解（仿真模拟退火）"表述略显冗长。"QUBO退火求解"已隐含退火过程，"仿真模拟退火"进一步限定方法。建议简化为"QUBO退火求解（仿真）"。
4. **api_reference.md 修改准确**：`tianyan_annealer` 机器的类型从"量子退火器"改为"退火求解器（仿真模拟退火）"，与实际实现一致。

### 项目价值评估

- **高价值**：比赛材料中的措辞准确性直接影响评委印象。明确区分"仿真退火"和"真机量子退火"可避免被评委质疑过度宣称。
- **与 PR #198 互补**：PR #198 的调研报告详细解释了为什么天衍云不支持量子退火，而 PR #278 在比赛材料层面落实了这一结论。

### 潜在问题

1. **混入大量非文档变更**：ci.yml 完整重写（315→319 行）、代码格式化、marl.py 修改，均与 Issue #228 无关。
2. **ci.yml 重写引入 BOM**：diff 显示整个 ci.yml 被删除重写，可能引入 BOM 字符导致行尾符变化。虽然 CI 通过，但增加了不必要的 diff 噪声。
3. **与 PR #276 的重叠**：两个 PR 解决相关问题（"量子退火"措辞），修改不同文件（#278 改外部文档，#276 改代码文档字符串），本身不重复。但共享的 ci.yml/格式化/marl.py 变更完全重复。

### 处理建议：**需修改**

- 移除非文档变更（ci.yml、格式化、marl.py），仅保留 4 个文档文件的措辞修改
- 考虑简化措辞："QUBO退火求解（仿真模拟退火）" → "QUBO退火求解（仿真）"
- 与 PR #276 协调合并顺序，避免非文档变更冲突

---

## PR #276: docs: 移除 anneal() 文档字符串中"量子退火器"绝对表述 (Issue #227)

### CI 状态
- Lint / Test (3 versions) / Type Check / Security Audit: **全部 PASS**
- Benchmark / Mutation: FAIL（非阻断）

### 核心修改内容

| 文件 | 修改类型 | 说明 |
|:--|:--|:--|
| `src/quantum/annealing.py` 模块 docstring | 措辞修改 | "利用量子退火器（或仿真模拟退火）" → "利用量子退火或仿真模拟退火" |
| `src/quantum/annealing.py` anneal() docstring | 措辞修改 | "调用量子退火器（或仿真）" → "调用退火求解器（真机/仿真）" |
| `src/quantum/annealing.py` anneal() docstring | 措辞修改 | "提交QUBO到天衍云量子退火器" → "提交QUBO到天衍云真机退火接口" |
| `src/quantum/annealing.py` | **新功能（Issue #226）** | 添加 `self.solver_type` 公开属性，跟踪实际求解器类型 |
| `src/quantum/annealing.py` | **代码变更** | anneal() 方法内添加 solver_type 赋值逻辑（4 处） |
| `src/quantum/annealing.py` | **代码变更** | optimize_policy/optimize_policy_hierarchical 返回字典添加 solver_type |
| `src/scheduler/ppo_agent.py` | 措辞修改 | 注释/日志："量子退火器" → "退火优化器"（883 行 diff，可能为行尾符变化） |
| 非文档变更 | ci.yml / 格式化 / marl.py | 与其他 3 个 PR 相同 |

### 文档正确性分析

1. **docstring 修改准确且必要**：已验证当前源码 `annealing.py` 第 7、324、328 行确实包含"量子退火器"表述。项目使用 neal/numpy 模拟退火，cqlib 无 `submit_annealing_task` 方法（已验证 `tianyan_cqlib.py` 仅有 `submit_quantum_task`/`submit_and_get_task_id`/`submit_to_machine`），真机退火路径是死代码。修改为"退火求解器"更准确。
2. **solver_type 功能是代码变更，非文档**：PR 标题声称是 "docs" 类型，但实际添加了 `self.solver_type` 公开属性和 4 处赋值逻辑。这是一个新功能（关联 Issue #226），应该单独提交。
3. **solver_type 实现质量**：新增的 solver_type 属性在每条退火路径上正确赋值（`real_quantum`/`neal_sa`/`numpy_sa`），并同步更新 `_last_solver` 向后兼容别名。同时将 solver_type 加入 optimize_policy 的返回字典。实现逻辑正确。
4. **ppo_agent.py 大量 diff**：883 行的 diff 对于仅修改 2 处注释/日志来说过大，很可能由行尾符变化（CRLF→LF）或 BOM 引起。虽然不影响功能，但增加了 review 难度。

### 项目价值评估

- **高价值**：代码文档字符串的准确性直接影响代码可维护性和答辩可信度。移除"量子退火器"的绝对表述是诚信修复。
- **中价值**：solver_type 属性对退火诊断有实际帮助，可追踪实际使用的求解器类型。
- **低价值**：ppo_agent.py 的注释修改价值较低，但无害。

### 潜在问题

1. **PR 类型不匹配**：标题为 "docs" 但包含新功能代码（solver_type）。应拆分为两个 PR：一个 docs PR（docstring 修改）+ 一个 feat PR（solver_type 功能）。
2. **混入非文档变更**：ci.yml 重写、格式化、marl.py 变更与 Issue #227 无关。
3. **ppo_agent.py 全文件 diff**：883 行 diff 中实际有效修改仅 2 行（注释/日志措辞），应避免行尾符变化导致的全文 diff。

### 处理建议：**需修改**

- 将 solver_type 功能（Issue #226）拆分为独立 feat PR
- 将 docstring 措辞修改保留为本 docs PR
- 移除非文档变更（ci.yml、格式化、marl.py）
- 修复 ppo_agent.py 的行尾符问题，使 diff 只显示实际修改的 2 行

---

## PR #198: docs(#118): 真机量子退火能力调研报告

### CI 状态
- Lint / Test (3 versions) / Type Check / Security Audit: **全部 PASS**
- Benchmark / Mutation: FAIL（非阻断）

### 核心修改内容

| 修改项 | 说明 |
|:--|:--|
| **新增 `docs/real_machine_annealing_research.md`（265 行）** | 完整的真机量子退火能力调研报告 |
| 非文档变更 | ci.yml（与 #224 相同的 continue-on-error + requirements-dev.txt） |
| 非文档变更 | 代码格式化（与其他 3 个 PR 相同） |
| 非文档变更 | marl.py 全文件重写（1181→1179 行，可能为行尾符变化） |

### 调研报告内容分析

报告分为 8 个章节，核心结论为：**天衍云平台不提供真机量子退火服务，所有真机均为门级超导量子计算机；cqlib SDK 无 QUBO 求解接口；项目当前实现正确，已诚实降级为仿真退火。**

#### 正确性验证

1. **cqlib 无 QUBO 接口**：已验证 `src/api/tianyan_cqlib.py` 中 `CqlibTianyanClient` 类仅有 `submit_quantum_task`、`submit_and_get_task_id`、`submit_to_machine` 方法，无 `submit_annealing_task`。报告结论正确。
2. **项目退火实现现状描述准确**：报告描述的三级求解路径与 `src/quantum/annealing.py` 的 `anneal()` 方法实现完全一致。
3. **退火统计显著性数据**：报告引用 p=0.190、Cliff's delta=0.40、+6.4% 等数据，与 AGENTS.md 和 `docs/annealing_significance-defense.md` 中的口径一致。
4. **QUBO 规模数据**：报告提到"训练用 QUBO 规模 4368 比特"，引用 `results/reports/annealing_solver_comparison.md` 作为来源。未直接验证，但数据合理。

#### 需要关注的事实性争议

5. **天衍-287 量子比特数争议**：报告称"天衍-287 的命名数字'287'并非量子比特数，其搭载的'祖冲之三号'芯片实际为 105 个物理量子比特"。但 AGENTS.md 多处写"287量子比特超导量子计算机"。如果报告正确，则 AGENTS.md 存在事实性错误。**建议团队核实此问题**——若确实为 105 比特，需修正 AGENTS.md 和所有比赛材料。

#### 问题

6. **file:// 链接使用绝对路径**：报告末尾的"参考文件"部分使用了 `file:///c:/Users/WJH/Desktop/2026年度中国青年科技创新"揭榜挂帅"擂台赛/quantum-rl-scheduler/...` 格式的链接。这些链接：
   - 指向另一个开发者（WJH）的本地路径，在其他机器上无法访问
   - 应改为相对路径（如 `[src/quantum/annealing.py](../../src/quantum/annealing.py)`）
7. **信息源质量**：7 条信息源中有 4 条来自今日头条（toutiao.com）链接，权威性一般。天衍云官网和 cqlib 文档链接较权威。

### 项目价值评估

- **极高价值**：这份调研报告对比赛答辩具有直接的战略价值：
  - 明确了天衍云的硬件能力边界（门级 vs 退火）
  - 为"为什么不用天衍云真机做退火"提供了有据可查的回答
  - 提出了 4 条可行的真机退火集成路径（QAOA/经典求解器/等待平台/VQE）
  - 提供了答辩话术建议
- **与 PR #278 互补**：#198 提供调研依据，#278 在比赛材料层面落实结论。
- **与 AGENTS.md Issue #128 一致**：报告结论与 AGENTS.md 中"真机验证结论边界"的口径完全一致。

### 潜在问题

1. **混入非文档变更**：ci.yml 修改、代码格式化、marl.py 全文件重写与 Issue #118 无关。
2. **file:// 绝对路径链接**：必须修复，改为 Markdown 相对路径。
3. **天衍-287 比特数争议**：需团队核实并统一口径。
4. **部分信息源权威性不足**：建议补充更权威的来源（如学术论文、官方白皮书）。

### 处理建议：**需修改**

- 修复 file:// 绝对路径链接为 Markdown 相对路径
- 移除非文档变更（ci.yml、格式化、marl.py）
- 团队核实天衍-287 的实际比特数，若为 105 比特则需修正全仓库相关描述
- 考虑补充更权威的信息源

---

## PR #278 与 #276 重叠分析

### 结论：文档变更不重复，但非文档变更完全重复

| 维度 | PR #278 | PR #276 | 是否重复 |
|:--|:--|:--|:--:|
| 修改目标 | 外部文档（AGENTS/README/api_reference/bidirectional） | 代码内文档字符串（annealing.py/ppo_agent.py） | 否（互补） |
| Issue | #228 | #227 | 不同 Issue |
| ci.yml 变更 | 完整重写（BOM） | 完整重写（BOM） | **完全重复** |
| 代码格式化 | 相同 | 相同 | **完全重复** |
| marl.py 变更 | BOM + cast 移除 | BOM + cast 移除 | **完全重复** |

### 建议

1. **不需要合并为一个 PR**：两个 PR 解决不同 Issue，修改不同文件，合并反而增加复杂度。
2. **应先合并共有变更**：将 ci.yml + 格式化 + marl.py 变更作为独立 PR（或直接 push 到 main），然后两个 docs PR rebase 后自动只保留各自的文档变更。
3. **合并顺序**：建议先合并 #276（代码 docstring 修改），再合并 #278（外部文档修改），避免 docstring 与外部文档措辞暂时不一致。

---

## 综合建议

### 优先合并顺序

1. **第一步**：将 4 个 PR 共有的非文档变更（ci.yml continue-on-error + requirements-dev.txt + 代码格式化 + marl.py BOM/cast 修复）提取为独立 PR，先合并到 main。这些变更已通过 CI 验证。
2. **第二步**：合并 **PR #198**（调研报告）——修复 file:// 链接后。这是答辩准备的高优先级材料。
3. **第三步**：合并 **PR #276**（代码 docstring 修改）——拆出 solver_type 功能后。确保代码内描述准确。
4. **第四步**：合并 **PR #278**（外部文档措辞修改）——简化措辞后。确保比赛材料不误导评委。
5. **第五步**：合并 **PR #224**（覆盖率门槛 + 碎片化整理）——修复 Lint 失败后。确保文档与 ci.yml 一致。

### 需要团队核实的事实性问题

- **天衍-287 实际比特数**：PR #198 报告称 105 比特（祖冲之三号），AGENTS.md 称 287 比特。需核实并统一全仓库口径。
