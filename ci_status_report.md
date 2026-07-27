# Quantum RL Scheduler — CI/CD & 仓库状态报告

> 检查时间：2026-07-24
> 仓库地址：https://github.com/xiabai2008/quantum-rl-scheduler
> 注意：git remote 配置为 `xiabai2004/quantum-rl-scheduler`，但实际仓库在 `xiabai2008/quantum-rl-scheduler` 下

---

## 一、CI/CD 状态总览

### 核心结论：CI 未通过（main 分支连续 3 次失败）

| 指标 | 结果 |
|------|------|
| **main 分支最近 CI 状态** | **FAILURE（连续 3 次 push 失败）** |
| PR 级别 CI 状态 | 全部通过（最近 4 个 PR 均 success） |
| 测试覆盖率门槛 | **80%**（`--cov-fail-under=80`） |
| Release Tag | **无**（无 v* 标签，无 GitHub Release） |
| Open Issues 数量 | **12 个** |
| 最近 PR 状态 | **#89 已合并**（MERGED） |

### 最近 10 次 Workflow Run 详情

| Run ID | 状态 | 触发方式 | 分支 | Workflow | 耗时 | 时间 |
|--------|------|----------|------|----------|------|------|
| 30009506897 | **failure** | push | main | ci.yml | 0s | 2026-07-23 13:02 |
| 30008857806 | **failure** | push | main | ci.yml | 10m30s | 2026-07-23 12:53 |
| 30008094821 | **failure** | push | main | ci.yml | 6m38s | 2026-07-23 12:42 |
| 30008086613 | success | pull_request | feature/issue-87 | PR Automation | 12s | 2026-07-23 12:42 |
| 30008086602 | success | pull_request | feature/issue-87 | ci.yml | 7m26s | 2026-07-23 12:42 |
| 30007991878 | cancelled | push | main | ci.yml | 1m38s | 2026-07-23 12:41 |
| 30007795734 | cancelled | push | main | ci.yml | 3m2s | 2026-07-23 12:38 |
| 30007784697 | success | pull_request | feature/issues-10-16-17-25 | ci.yml | 6m5s | 2026-07-23 12:38 |
| 30007784674 | success | pull_request | feature/issues-10-16-17-25 | PR Automation | 9s | 2026-07-23 12:38 |
| 30004480002 | success | pull_request | feat/issue-46-dqn-14dim | ci.yml | 7m42s | 2026-07-23 11:47 |

**失败原因分析：**
- **Run 30009506897**（最新）：workflow 文件配置错误（0 秒即失败，"This run likely failed because of a workflow file issue"）
- **Run 30008857806**：14 维 DQN 提交（Issue #46），运行 10 分 30 秒后失败
- **Run 30008094821**：websocket_handler 覆盖率提升提交（#88），运行 6 分 38 秒后失败
- PR 合并前的 CI 全部通过，但 push 到 main 后失败（可能是合并冲突或 main 分支独有条件触发）

---

## 二、测试覆盖率门槛和 CI 阻断条件

配置文件：`.github/workflows/ci.yml`

### 覆盖率门槛
```yaml
--cov-fail-under=80
```
- **覆盖率低于 80% 将直接阻断 CI**（pytest 返回非零退出码）
- 覆盖率报告格式：term-missing（终端显示）+ xml（Codecov 上传）
- Codecov 上传仅在 `ubuntu-latest + Python 3.11` 条件下执行，且 `fail_ci_if_error: false`（上传失败不阻断）

### CI 流水线 6 个 Job 及阻断策略

| Job | 名称 | 运行环境 | 依赖 | 阻断条件 | 超时 |
|-----|------|----------|------|----------|------|
| **lint** | Lint (ruff format + ruff check + bandit) | ubuntu-latest | 无 | **严格阻断** | 10min |
| **test** | Test (Python 3.10/3.11/3.12 矩阵) | windows-latest | lint | **严格阻断**（含覆盖率<80%） | 30min |
| **typecheck** | Type Check (mypy) | ubuntu-latest | lint | **严格阻断** | 10min |
| **security-audit** | Dependency Security Audit (pip-audit) | ubuntu-latest | lint | **不阻断**（continue-on-error: true） | 10min |
| **mutation-testing** | Mutation Testing (mutmut) | ubuntu-latest | test | **不阻断**（continue-on-error: true，仅 main push） | 30min |
| **benchmark** | Benchmark Regression Check | ubuntu-latest | test | 严格阻断（仅 main push） | 15min |

### 严格阻断项清单
1. **ruff format --check**：代码格式不符合规范 → 阻断
2. **ruff check**：代码静态检查发现错误 → 阻断
3. **bandit -ll**：安全扫描发现中高危漏洞 → 阻断
4. **pytest 测试失败**：任何测试用例失败 → 阻断
5. **覆盖率 < 80%**：`--cov-fail-under=80` → 阻断
6. **单测超时 > 120s**：`--timeout=120` → 阻断
7. **mypy 类型检查失败**（strict mode）→ 阻断

### 非阻断项（仅告警）
- **pip-audit**：依赖漏洞扫描（初期允许失败，待修复后改为阻断）
- **mutmut 变异测试**：测试质量评估（初期允许失败，仅收集基线数据，仅 main push 触发）
- **ffmpeg 安装（Windows）**：`continue-on-error: true`

---

## 三、Open Issues 列表（共 12 个）

> **注意**：仓库中不存在 #200-#209 的 issues。当前最高 issue 编号为 **#100**。

| 编号 | 标题 | 标签 | 创建时间 |
|------|------|------|----------|
| **#92** | 【P0证据】真机性能实验：天衍-287/176上 PPO vs FCFS vs SJF 对比 | `优先级-高`, `algorithm`, `real-machine` | 2026-07-23 13:18 |
| #93 | [P0证据] 真机测量结果闭环：poll_pending_real_tasks返回真实量子测量值 | `algorithm`, `real-machine` | 2026-07-23 13:22 |
| #94 | [P0证据] 退火统计扩展：N=500 seed检验p值能否跌破0.05 | `algorithm` | 2026-07-23 13:22 |
| #95 | [P1补全] 消融实验D3：奖励函数消融（4种设计对比） | `algorithm` | 2026-07-23 13:22 |
| #96 | [P1补全] 14维DQN三策略完整对比：DQN vs PPO vs FCFS | `algorithm` | 2026-07-23 13:23 |
| #97 | [P1质量] env_real_machine.py测试覆盖率 29%->60% | `testing` | 2026-07-23 13:23 |
| #98 | [P1质量] marl.py MAPPO多机测试覆盖率 64%->80% | `testing` | 2026-07-23 13:23 |
| #99 | [P1质量] ppo_agent.py测试覆盖率 72%->80% | `testing` | 2026-07-23 13:23 |
| #100 | [P1补全] 跨硬件兼容路线图文档（超导/离子阱/光量子） | `documentation` | 2026-07-23 13:23 |
| #91 | 【可视化改进】实时调度过程可视化效果优化 | `优先级-中`, `ai-ready` | 2026-07-23 13:17 |
| #90 | 【可视化改进】实时调度过程可视化增强 | `优先级-中`, `ai-ready` | 2026-07-23 13:14 |
| #28 | 【评委审查P1】补齐超导/离子阱/光量子跨硬件兼容证据 | `algorithm`, `优先级-中` | 2026-07-21 13:04 |

### 按标签分类统计
- **P0 高优先级**：#92, #93, #94（3 个，均为真机/统计证据类）
- **P1 补全/质量**：#95-#100（6 个，含算法补全3个+测试覆盖率3个+文档1个）
- **可视化改进**：#90, #91（2 个，均为 ai-ready）
- **评委审查**：#28（1 个，跨硬件兼容证据）

---

## 四、最近关闭的 Issues（最近 20 个）

| 编号 | 标题 | 标签 | 关闭时间 |
|------|------|------|----------|
| #87 | 【测试覆盖】提升websocket_handler.py覆盖率（78%->80%+） | `优先级-中`, `ai-ready` | 2026-07-23 12:42 |
| #85 | 【每周检查】2026-07-23 CI/覆盖率问题 | - | 2026-07-23 12:43 |
| #84 | 【代码质量】修复test_metric_audit测试与FORBIDDEN_PATTERNS不一致 | `优先级-中`, `ai-ready` | 2026-07-23 12:43 |
| #83 | 【代码质量】统一熔断器实现，消除重复代码 | `优先级-中`, `cleanup`, `ai-ready` | 2026-07-23 12:41 |
| #82 | [自动化检查] FCFS标准差与AGENTS.md权威数字偏差超过5% | `bug`, `ai-ready` | 2026-07-23 12:43 |
| #81 | 【每周检查】2026-07-23 CI失败问题 | - | 2026-07-23 12:43 |
| #76 | 【技术创新】引入强化学习策略优化量子计算资源分配 | `优先级-中`, `ai-ready` | 2026-07-23 12:52 |
| #74 | 【实验严谨】补齐调度环境seed可复现性验证测试 | `优先级-中`, `ai-ready` | 2026-07-23 07:03 |
| #73 | 【竞赛对齐】将可解释性模块集成到可视化仪表盘 | `enhancement`, `优先级-中`, `ai-ready` | 2026-07-23 11:40 |
| #72 | 【实验严谨】修复文档中PPO vs FCFS统计数字引用错误 | `documentation`, `优先级-中`, `ai-ready` | 2026-07-23 12:39 |
| #66 | 【Issue巡检】2026-07-22 高优Issue未分配预警 | `优先级-高` | 2026-07-23 12:39 |
| #65 | fix: test_annealing_loop.py::test_callback_triggers_submit 在 Python 3.10 上持续失败 | `优先级-高`, `ai-ready` | 2026-07-23 07:02 |
| #58 | 【PR修复通知】5个PR审查反馈汇总 | `优先级-高`, `testing` | 2026-07-23 07:10 |
| #50 | CI流水线优化：合并冗余Job、加速构建 | `enhancement`, `ci`, `优先级-低` | 2026-07-23 13:02 |
| #49 | Docker Compose健康检查与多服务编排完善 | `enhancement`, `优先级-中`, `backend` | 2026-07-22 05:13 |
| #48 | 真机验证结论边界重写：区分可用性vs性能验证 | `documentation`, `优先级-高` | 2026-07-22 06:00 |
| #47 | 量子退火统计证据补强：提升统计说服力 | `优先级-高`, `algorithm` | 2026-07-23 13:04 |
| #46 | 14维DQN模型重训：修复退化问题 | `优先级-高`, `algorithm` | 2026-07-23 12:53 |
| #45 | 真机闭环验证10seeds扩展：提升统计说服力 | `优先级-高`, `algorithm` | 2026-07-23 07:03 |
| #44 | 多用户公平性调度：Jain Index+多租户对比 | `algorithm`, `优先级-中` | 2026-07-23 13:03 |

---

## 五、Release Tag 状态

| 检查项 | 结果 |
|--------|------|
| `git tag -l`（所有标签） | `backup/branch-pre-merge-20260717`, `backup/main-pre-merge-20260717` |
| `git tag -l "v*"`（版本标签） | **无** |
| `gh release list`（GitHub Release） | **无** |

**结论：项目尚未创建任何 release tag 或 GitHub Release。** 仅有 2 个合并前的备份标签（2026-07-17 创建）。

---

## 六、最近 PR 状态

| 编号 | 标题 | 状态 | 源分支 | 目标分支 | 合并时间 |
|------|------|------|--------|----------|----------|
| **#89** | feat: unify circuit breaker, eliminate duplicate code in tianyan_client.py | **MERGED** | feature/issue-83 | main | 2026-07-23 12:41 |
| #88 | test: 提升websocket_handler.py覆盖率（78%->100%） | MERGED | feature/issue-87 | main | 2026-07-23 12:42 |
| #86 | feat: 将可解释性模块集成到可视化仪表盘 | MERGED | feature/issue-73 | main | 2026-07-23 11:40 |
| #80 | docs(#10,#16,#17,#25): 文档收敛与价值量化路线图 | MERGED | feature/issues-10-16-17-25-doc-convergence | main | 2026-07-23 12:38 |
| #79 | feat: 可解释性模块深度应用 (Closes #31) | MERGED | feature/issue-31-explainability | main | 2026-07-23 11:40 |

**最近 5 个 PR 全部已合并。** 但注意：PR 合并后 push 到 main 分支的 CI 均失败（见第一节），存在"PR CI 通过但 main CI 失败"的问题。

---

## 七、风险提示与建议

1. **main 分支 CI 红（高优）**：连续 3 次 push 失败，最新一次是 workflow 文件配置错误（0s 失败），需立即排查 ci.yml 语法/配置问题
2. **PR 与 main CI 不一致**：PR 上 CI 通过但合并到 main 后失败，可能存在并发合并冲突或 main 分支保护规则问题
3. **无 Release Tag**：项目尚未打版本标签，建议在代码冻结前创建 v1.0.0 标签
4. **测试覆盖率债务**：env_real_machine.py 仅 29% 覆盖率（#97），marl.py 64%（#98），距 80% 目标有差距
5. **P0 证据类 Issues 未关闭**：#92（真机性能实验）、#93（真机测量闭环）、#94（N=500统计检验）均为 OPEN 状态
