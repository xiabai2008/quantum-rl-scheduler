# 代码冻结统一规范（合并自 code_freeze_checklist.md / code_freeze_policy.md / 代码冻结流程.md）

> 本文档由三份碎片文档合并统一：冻结流程与政策规范、冻结操作流程、以及检查清单（见附录A）。

## 一、冻结流程与政策规范（原 code_freeze_policy.md）


> **文档版本**: v1.0
> **创建日期**: 2026-07-02
> **适用阶段**: 2026-08-15 代码冻结至 2026-09-15 作品提交

---

## 一、代码冻结概述

### 1.1 冻结目标

确保比赛提交前代码稳定性，避免引入新缺陷，保证演示效果和答辩数据一致性。

### 1.2 冻结时间线

| 阶段 | 日期 | 说明 |
|:--|:--|:--|
| **预冻结期** | 2026-08-01 ~ 2026-08-14 | 完成所有功能开发，开始逐步收敛变更 |
| **正式冻结期** | 2026-08-15 ~ 2026-09-15 | 仅允许 P0/P1 Bug 修复，禁止新功能 |
| **解冻期** | 2026-09-16 之后 | 比赛提交完成后恢复正常开发 |

### 1.3 冻结范围

**禁止的变更类型**：
- 新功能开发（feature）
- 代码重构（refactor）
- 性能优化（除非修复 P0 性能回退）
- 文档更新（除非修正明显错误）
- 依赖升级（除非修复安全漏洞）

**允许的变更类型**：
- P0 Bug 修复（系统崩溃、数据丢失、安全漏洞）
- P1 Bug 修复（核心功能失效、演示失败）
- 配置调整（仅限 `config/` 目录）
- 测试用例补充（提高覆盖率）

---

## 二、冻结检查清单

### 2.1 预冻结期检查（2026-08-14 前完成）

- [ ] 所有计划功能已合并到 main 分支
- [ ] CI 流水线全部通过（lint + test + typecheck + benchmarks）
- [x] 测试覆盖率 ≥ 80%（当前 93.58%）
- [ ] mypy 类型检查通过（无新增豁免）
- [ ] 所有 P0/P1 Bug 已修复
- [ ] 演示视频录制完成（使用冻结前代码）
- [ ] 实验数据已固化（`results/reports/` 目录）
- [ ] 技术白皮书终稿完成
- [ ] 答辩 PPT 终稿完成

### 2.2 正式冻结期检查（每周一次）

- [ ] 无新功能代码合入
- [ ] 所有 Bug 修复经过 Code Review
- [ ] 回归测试通过
- [ ] 演示环境可正常启动
- [ ] 真机接口连通性验证（每周一次）

---

## 三、例外审批流程

### 3.1 例外申请条件

仅在以下情况下可申请例外：

1. **P0 Bug**：系统崩溃、数据丢失、安全漏洞
2. **演示失败**：答辩演示时出现致命错误
3. **数据不一致**：实验数据与报告不符
4. **安全漏洞**：依赖库爆出高危 CVE

### 3.2 例外申请模板

```markdown
## 代码冻结例外申请

**申请日期**: YYYY-MM-DD
**申请人**: [GitHub 用户名]
**紧急程度**: P0 / P1

### 问题描述
[简要描述问题，包括复现步骤、影响范围]

### 修复方案
[说明需要修改的文件、修改内容、预期效果]

### 风险评估
[说明修改可能引入的风险]

### 回滚方案
[如果修改失败，如何回滚到冻结状态]

### 审批人
- [ ] 算法组负责人
- [ ] 工程组负责人
- [ ] 项目负责人（瑞哥）
```

### 3.3 审批角色定义

| 角色 | 人员 | 职责 |
|:--|:--|:--|
| **算法组负责人** | [待指定] | 评估修改对实验数据的影响 |
| **工程组负责人** | [待指定] | 评估修改对系统稳定性的影响 |
| **项目负责人** | 瑞哥 | 最终审批权，承担风险 |

### 3.4 审批流程

1. **申请人**填写例外申请模板，提交到 GitHub Issue（标签：`freeze-exception`）
2. **算法组负责人**在 4 小时内评估数据影响
3. **工程组负责人**在 4 小时内评估稳定性影响
4. **项目负责人**在 8 小时内做出最终决定
5. **审批通过**后，申请人创建功能分支，修复后提交 PR
6. **PR 需经 2 人 Review**（包括至少一名审批人）
7. **合并后**立即运行完整回归测试

---

## 四、回滚机制

### 4.1 回滚触发条件

- 修复引入新 Bug
- 测试覆盖率下降
- 演示环境无法启动
- 实验数据发生变化

### 4.2 回滚步骤

```bash
# 1. 查看冻结点 commit
git log --oneline | grep "freeze: 2026-08-15"

# 2. 回滚到冻结点
git revert <commit-hash>

# 3. 强制推送（仅管理员）
git push origin main --force-with-lease

# 4. 通知团队
echo "已回滚到冻结状态，请检查系统"
```

### 4.3 回滚验证

- [ ] CI 流水线通过
- [ ] 演示环境可正常启动
- [ ] 实验数据与冻结前一致
- [ ] 测试覆盖率未下降

---

## 五、冻结期工作规范

### 5.1 分支管理

- **禁止**创建新功能分支
- **仅允许**创建 `fix/issue-<编号>-<简述>` 格式的 Bug 修复分支
- 所有 PR 必须关联 GitHub Issue

### 5.2 Commit 规范

```
fix: 修复 <问题描述> (#<issue编号>)

[冻结期例外审批：@审批人1 @审批人2]
```

### 5.3 Code Review 要求

- 至少 2 人 Review
- 必须包含一名冻结审批人
- Review 重点：
  - 是否仅修复指定问题
  - 是否引入新依赖
  - 是否影响实验数据
  - 是否通过所有测试

### 5.4 测试要求

- 所有现有测试必须通过
- 新增测试用例需覆盖修复场景
- 测试覆盖率不得下降
- 演示脚本需手动验证

---

## 六、冻结期日历

| 日期 | 事项 | 负责人 |
|:--|:--|:--|
| 2026-08-01 | 预冻结期开始 | 全体 |
| 2026-08-14 | 预冻结检查完成 | 工程组 |
| 2026-08-15 | 正式冻结开始 | 瑞哥 |
| 2026-08-22 | 第一次冻结期检查 | 全体 |
| 2026-08-29 | 第二次冻结期检查 | 全体 |
| 2026-09-05 | 第三次冻结期检查 | 全体 |
| 2026-09-12 | 最终检查 | 全体 |
| 2026-09-15 | 作品提交 | 瑞哥 |
| 2026-09-16 | 解冻 | 瑞哥 |

---

## 七、常见问题

### Q1: 冻结期间发现文档错误怎么办？

**A**: 轻微错误（错别字、格式）可直接修正，无需申请例外。重大错误（数据、结论）需走例外审批流程。

### Q2: 冻结期间依赖库爆出安全漏洞怎么办？

**A**: 立即走例外审批流程，标注为 P0 级别。修复后需运行完整安全扫描（bandit）。

### Q3: 冻结期间发现性能回退怎么办？

**A**: 如果性能回退导致演示失败或指标不达标，走例外审批流程。否则记录为 Issue，解冻后处理。

### Q4: 冻结期间可以更新 README 吗？

**A**: 仅限修正明显错误（链接失效、版本号错误）。新增内容需走例外审批。

### Q5: 冻结期间可以合并其他分支的代码吗？

**A**: 禁止。冻结期仅允许 Bug 修复分支合入 main。

---

## 八、附录

### 8.1 冻结点 Commit 信息模板

```
chore: 代码冻结 2026-08-15

冻结范围：禁止新功能、重构、性能优化
允许变更：P0/P1 Bug 修复、配置调整
解冻日期：2026-09-16

冻结检查清单：
- [x] 所有功能已合并
- [x] CI 全部通过
- [x] 测试覆盖率 ≥ 80%
- [x] 演示视频录制完成
- [x] 实验数据已固化

审批人：@xiabai2008
```

### 8.2 冻结期联系渠道

- **紧急问题**：微信群 + 电话
- **一般问题**：GitHub Issue（标签：`freeze`）
- **例外申请**：GitHub Issue（标签：`freeze-exception`）

---

*本文档为比赛提交前的代码冻结规范，全体团队成员必须严格遵守。*

---

## 二、冻结操作流程（原 代码冻结流程.md）


> **冻结日期**: 2026-08-15 | **提交截止**: 2026-09-15
>
> 8/15 后不再合并新功能，仅允许修复致命 Bug 和文档更新。

---

## 冻结前检查清单（8/14 完成）

### 1. 拉取最新代码

```bash
git checkout main
git pull origin main
```

### 2. 运行预冻结检查脚本

```bash
bash scripts/ci/pre_freeze_check.sh
```

脚本自动检查 8 项：

| # | 检查项 | 说明 |
|---|--------|------|
| 1 | 代码同步 | 本地与 origin/main 一致 |
| 2 | 分支状态 | 在 main 分支，工作区干净 |
| 3 | CI 状态 | 所有 required checks 通过 |
| 4 | 代码格式 | ruff format + ruff check 通过 |
| 5 | 单元测试 | 全部通过，覆盖率 ≥ 80% |
| 6 | 提交物校验 | submission_manifest.yaml 各项齐备 |
| 7 | 数字审计 | 无旧值残留（运行 audit_authoritative_metrics.py 通过） |
| 8 | 文件确认 | README / 模型 / 报告等存在 |

`--quick` 模式跳过慢速检查（CI 状态、全量测试）：

```bash
bash scripts/ci/pre_freeze_check.sh --quick
```

### 3. 手动确认以下事项

- [ ] 所有 open PR 已合并或关闭
- [ ] `config/submission_manifest.yaml` 版本号正确 (`v9.1`)
- [ ] 技术白皮书 v3 已导出 PDF
- [ ] 答辩 PPT 终稿已保存
- [ ] 演示视频已就位 (1080p, 4-5min, ≤500MB)
- [ ] PPO/DQN 权威模型在 `deliverable_models/`
- [ ] `README.md` 包含复现步骤
- [ ] `.env.example` 与实际配置一致
- [ ] 没有提交 `.env`、`models/`、`logs/` 等

---

## 冻结日操作（8/15）

### Step 1: 最终确认

```bash
# 确保所有检查通过
bash scripts/ci/pre_freeze_check.sh
```

### Step 2: 打标签

```bash
# 创建 annotated tag（推荐，可包含详细说明）
git tag -a v9.1-submission -m "v9.1 提交版本 (2026-08-15)

八策略对比权威数字: PPO (16维) vs FCFS +20.2%
量子占比敏感性: 50% 时 PPO 提升最高 +94.1%
50 seed 统计显著 p<0.001

提交物:
- 技术白皮书 v3 (PDF)
- 答辩 PPT 终稿
- 演示视频 (1080p 5min)
- 5 份实验报告
- PPO/DQN 权威模型
- 代码仓库 + 压缩包
"
```

### Step 3: 推送标签

```bash
# 只推送标签（不推送其他未提交的更改）
git push origin v9.1-submission
```

### Step 4: 等待自动发布

推送标签后，GitHub Actions 自动执行 `release.yml` 流水线：

```
git push origin v9.1-submission
        │
        ▼
┌─────────────────────────────────┐
│  Quality Gate                   │
│  ├─ 全量单元测试 (cov ≥ 80%)    │
│  ├─ 提交物校验 (--check)        │
│  └─ 权威数字审计                │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  Package Submission             │
│  └─ 打包提交物 zip              │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  Create GitHub Release          │
│  ├─ 自动生成 Release Notes      │
│  └─ 上传提交物 zip 附件          │
└─────────────────────────────────┘
```

### Step 5: 下载最终提交物

1. 打开 https://github.com/xiabai2008/quantum-rl-scheduler/releases
2. 找到 `v9.1-submission` Release
3. 下载 `submission_v9.1_*.zip` 附件
4. 将 zip 中的内容与仓库文件一起打包提交

---

## 冻结后紧急修复流程

冻结后如发现致命 Bug，需遵循以下流程：

1. **创建 hotfix 分支**:
   ```bash
   git checkout -b hotfix/<简短描述> v9.1-submission
   ```

2. **修复 + 提交**:
   ```bash
   git add <修改的文件>
   git commit -m "hotfix: <描述>"
   ```

3. **创建 PR → main**（需要队友 Review）

4. **合并后重新打标签**:
   ```bash
   git checkout main
   git pull origin main
   git tag -a v9.1-submission-hotfix1 -m "v9.1 提交版本 (hotfix 1)"
   git push origin v9.1-submission-hotfix1
   ```

5. **通知团队**：在新标签的 Release 中说明变更内容。

---

## 常见问题

### Q: 推送标签后 CI 失败怎么办？

查看失败日志 → 修复问题 → 重新打标签（删除旧标签 + 创建新标签）：

```bash
# 删除本地和远程旧标签
git tag -d v9.1-submission
git push origin :refs/tags/v9.1-submission

# 修复代码后重新打标签
git tag -a v9.1-submission -m "v9.1 提交版本"
git push origin v9.1-submission
```

### Q: 冻结日期可以推迟吗？

由瑞哥决策。如推迟，需同步更新 `submission_manifest.yaml` 中的 `deadline` 字段。

### Q: pre_freeze_check.sh 中某些检查失败但不影响提交怎么办？

例如演示视频临时缺失（其他队友负责），可以手动跳过特定检查。但必须确保 `submission_manifest.yaml` 中的所有强制项就位。

---

## 相关文件

| 文件 | 用途 |
|------|------|
| `.github/workflows/release.yml` | 自动发布流水线 |
| `scripts/ci/pre_freeze_check.sh` | 冻结前检查脚本 |
| `scripts/ci/validate_submission.py` | 提交物校验与打包 |
| `scripts/ci/audit_authoritative_metrics.py` | 权威数字一致性审计 |
| `config/submission_manifest.yaml` | 最终提交物清单 |

---

## 附录A：代码冻结检查清单（原 code_freeze_checklist.md，25 项）


> **冻结日期**: 2026-08-15
> **提交截止**: 2026-09-15
> **版本标签**: v9.1-submission
> **负责人**: 瑞哥（xiabai2008）
> **最后更新**: 2026-07-25

本文档是 8/15 代码冻结前的完整检查清单，覆盖 6 大类共 25 项检查。每项须由指定负责人执行验证命令并确认通过后方可打标签冻结。

---

## 一、CI/CD（5 项）

### 1.1 Lint 检查全绿

- [ ] **ruff format 检查通过**（零格式错误）
  - 负责人：工程组
  - 验证命令：`ruff format --check src/ scripts/ tests/`

### 1.2 Lint 规则检查全绿

- [ ] **ruff check 零错误**（历史遗留 142→0 已完成，不可回退）
  - 负责人：工程组
  - 验证命令：`ruff check src/ scripts/ tests/`

### 1.3 类型检查全绿

- [ ] **mypy 严格模式零错误**（26→0 已完成，不可回退）
  - 负责人：工程组
  - 验证命令：`mypy src/`

### 1.4 安全扫描全绿

- [ ] **bandit 安全扫描通过**（无高危/中危漏洞）
  - 负责人：工程组
  - 验证命令：`bandit -r src/ -c pyproject.toml -ll`

### 1.5 CI 流水线全绿

- [ ] **GitHub Actions CI 4 Job 全部通过**（lint→test→typecheck→benchmarks）
  - 负责人：瑞哥
  - 验证命令：`gh run list --branch main --limit 1 --json conclusion,name` 或在 GitHub Actions 页面确认

---

## 二、代码质量（4 项）

### 2.1 测试通过

- [ ] **全部测试用例通过**（3717 用例，0 失败）
- [ ] **全部测试用例通过**（3717 用例，0 失败）
  - 负责人：工程组
  - 验证命令：`pytest tests/ -q`

### 2.2 测试覆盖率达标

- [ ] **测试覆盖率 >= 80%**（当前实际 93.58%，门槛不可低于 80%，与 pyproject.toml `fail_under=80` 一致）
  - 负责人：工程组
  - 验证命令：`pytest tests/ --cov=src --cov-fail-under=80 --cov-report=term-missing`

### 2.3 ruff 零错误确认

- [ ] **ruff check 输出 "All checks passed!"**（确认无 --exit-zero 回退）
  - 负责人：工程组
  - 验证命令：`ruff check src/ scripts/ tests/ 2>&1 | grep -c "All checks passed"`

### 2.4 mypy 零错误确认

- [ ] **mypy 输出 "Success: no issues found"**（确认严格模式无豁免新增）
  - 负责人：工程组
  - 验证命令：`mypy src/ 2>&1 | tail -1`

---

## 三、文档（4 项）

### 3.1 AGENTS.md 已更新

- [ ] **AGENTS.md "最后更新"日期为冻结当天**，版本号、进度条、文件结构均与实际一致
  - 负责人：瑞哥
  - 验证命令：`head -10 AGENTS.md | grep "最后更新"`

### 3.2 README 准确

- [ ] **README.md 中的快速开始命令可执行**，依赖安装说明准确，文件数/结构描述与实际一致
  - 负责人：工程组
  - 验证命令：`python scripts/cli.py demo --multi-machine`（验证 demo 可运行）

### 3.3 实验报告已定稿

- [ ] **results/reports/ 目录下所有报告已定稿**，至少包含 18 份报告（统计验证、多seed真机、公平调度、D3消融、高负载公平调度等）
  - 负责人：算法组
  - 验证命令：`ls results/reports/*.md | wc -l`（确认 >= 18）

### 3.4 提交校验通过

- [ ] **提交清单校验脚本通过**（17 项检查中通过项 >= 12）
  - 负责人：瑞哥
  - 验证命令：`python scripts/ci/validate_submission.py --check`

---

## 四、交付物（4 项）

### 4.1 答辩 PPT

- [ ] **答辩 PPT（17页）已定稿**，数字与权威数字一致（+20.2%，p=7.56e-12，N=250）
  - 负责人：瑞哥
  - 验证命令：`python scripts/ci/validate_authoritative_numbers.py --strict`（全项目数字一致性）

### 4.2 技术白皮书

- [ ] **技术白皮书（11章）已定稿**，含第10章落地与价值量化，数字与权威数字一致
  - 负责人：瑞哥
  - 验证命令：`python scripts/ci/validate_authoritative_numbers.py --strict`

### 4.3 演示视频

- [ ] **演示视频（4-5分钟，1080p）已录制完成**，使用冻结前代码录制，分镜脚本已对齐
  - 负责人：瑞哥
  - 验证命令：人工确认视频文件存在且可播放

### 4.4 源代码包

- [ ] **源代码打包通过**（validate_submission.py --pack 生成完整交付包）
  - 负责人：瑞哥
  - 验证命令：`python scripts/ci/validate_submission.py --pack`

---

## 五、统计数字一致性（4 项）

> **权威数字来源**: AGENTS.md 第6节（50seed N=250 验证）

### 5.1 PPO 与 FCFS 奖励数字

- [ ] **全项目 PPO/FCFS 均值与标准差一致**：PPO=1982.69±557.25，FCFS=1648.91±502.95（不可出现旧值 2348.91 / 857.25 / 1051.59 / 58.34 / 2746.94 / 1160.72 / 1458.77 等）<!-- audit-exempt: 历史旧值参考 -->
  - 负责人：算法组
  - 验证命令：`python scripts/ci/validate_authoritative_numbers.py`

### 5.2 统计检验数字

- [ ] **全项目 p 值与效应量一致**：p=7.56e-12（Welch t 检验），rank-biserial=-0.3642（中效应）（不可出现旧值 1.449e-66 / Cohen's d=-2.1353 / 1.032e-42 / 3.5e-8 / rank-biserial=-0.71 / rank-biserial=-0.7081 / Cohen's d=4.09 / Mann-Whitney U 等）<!-- audit-exempt: 历史旧值参考 -->
  - 负责人：算法组
  - 验证命令：`python scripts/ci/validate_authoritative_numbers.py --strict`

### 5.3 提升百分比

- [ ] **全项目提升百分比一致**：+20.2%（不可出现旧值 +123.4% / +88.3% / +86.9% / +102.3% / +95.4%）<!-- audit-exempt: 历史旧值参考 -->
  - 负责人：算法组
  - 验证命令：`grep -rn "86\.9%\|95\.4%\|102\.3%" docs/ results/reports/ --include="*.md" | grep -v "10 seed\|旧\|旧版\|对比\|变化\|audit-exempt"`

### 5.4 样本量

- [ ] **全项目样本量一致**：N=250（50 seeds x 5 episodes，不可出现 N=50 作为权威样本量）
  - 负责人：算法组
  - 验证命令：`grep -rn "N=50\b" docs/ results/reports/ --include="*.md" | grep -v "10 seed\|N=50 seeds\|seeds"`（不应有将 N=50 作为权威样本量的表述）

---

## 六、Git（4 项）

### 6.1 工作树干净

- [ ] **git 工作树无未提交变更**（所有变更已 commit）
  - 负责人：瑞哥
  - 验证命令：`git status --porcelain`（输出应为空）

### 6.2 所有 PR 已合并

- [ ] **所有功能分支 PR 已合并到 main**（无未合并的功能分支 PR）
  - 负责人：瑞哥
  - 验证命令：`gh pr list --state open --base main`（输出应为空或仅文档类 PR）

### 6.3 权威模型已入库

- [ ] **权威模型文件已入库**（deliverable_models/ppo_best_model_16dim.zip 和 ppo_compilation_agent.zip 存在）
  - 负责人：算法组
  - 验证命令：`ls -la deliverable_models/ppo_best_model_16dim.zip deliverable_models/ppo_compilation_agent.zip`

### 6.4 版本标签已打

- [ ] **已打 v9.1-submission 标签并推送**（annotated tag，含提交版本说明）
  - 负责人：瑞哥
  - 验证命令：
    ```bash
    git tag -a v9.1-submission -m "v9.1 提交版本"
    git push origin v9.1-submission
    git tag -l v9.1-submission
    ```

---

## 冻结确认

| 类别 | 检查项数 | 通过 | 未通过 |
|:--|:--:|:--:|:--:|
| CI/CD | 5 | | |
| 代码质量 | 4 | | |
| 文档 | 4 | | |
| 交付物 | 4 | | |
| 统计数字 | 4 | | |
| Git | 4 | | |
| **合计** | **25** | | |

> **冻结条件**: 25 项全部通过方可打标签冻结。如有未通过项，须走例外审批流程（详见 `docs/code_freeze.md`）。

**冻结人签字**: _______________  **日期**: _______________

---

*本文档（检查清单）与 `docs/code_freeze.md`（统一规范，含冻结流程与例外审批）配合使用。*
