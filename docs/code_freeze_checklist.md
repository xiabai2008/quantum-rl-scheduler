# 代码冻结检查清单（Issue #125）

> **冻结日期**: 2026-08-15
> **提交截止**: 2026-09-15
> **版本标签**: v8.0-submission
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

- [ ] **全部测试用例通过**（2285+ 用例，0 失败）
  - 负责人：工程组
  - 验证命令：`pytest tests/ -q`

### 2.2 测试覆盖率达标

- [ ] **测试覆盖率 >= 70%**（当前实际 91%，门槛不可低于 70%）
  - 负责人：工程组
  - 验证命令：`pytest tests/ --cov=src --cov-fail-under=70 --cov-report=term-missing`

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

- [ ] **提交清单校验脚本通过**（13 项检查中通过项 >= 12）
  - 负责人：瑞哥
  - 验证命令：`python scripts/ci/validate_submission.py --check`

---

## 四、交付物（4 项）

### 4.1 答辩 PPT

- [ ] **答辩 PPT（17页）已定稿**，数字与权威数字一致（+88.3%，p=1.032e-42，N=250）
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

- [ ] **全项目 PPO/FCFS 均值与标准差一致**：PPO=2746.94±1160.72，FCFS=1458.77±60.47（不可出现旧值 1121.19 / 55.85 / 2966.17 等）<!-- audit-exempt: 历史旧值参考 -->
  - 负责人：算法组
  - 验证命令：`python scripts/ci/validate_authoritative_numbers.py`

### 5.2 统计检验数字

- [ ] **全项目 p 值与效应量一致**：p=1.032e-42（Mann-Whitney U 检验），rank-biserial=-0.71（大效应量）（不可出现旧值 3.04e-11 / 3.5e-8 / Cohen's d=-1.70 / Cohen's d=4.09 / Welch t）<!-- audit-exempt: 历史旧值参考 -->
  - 负责人：算法组
  - 验证命令：`python scripts/ci/validate_authoritative_numbers.py --strict`

### 5.3 提升百分比

- [ ] **全项目提升百分比一致**：+88.3%（不可出现旧值 +86.9% / +102.3% / +95.4%）<!-- audit-exempt: 历史旧值参考 -->
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

- [ ] **权威模型文件已入库**（deliverable_models/ppo_best_model_14dim.zip 和 dqn_best_model_10dim.zip 存在）
  - 负责人：算法组
  - 验证命令：`ls -la deliverable_models/ppo_best_model_14dim.zip deliverable_models/dqn_best_model_10dim.zip`

### 6.4 版本标签已打

- [ ] **已打 v8.0-submission 标签并推送**（annotated tag，含提交版本说明）
  - 负责人：瑞哥
  - 验证命令：
    ```bash
    git tag -a v8.0-submission -m "v8.0 提交版本"
    git push origin v8.0-submission
    git tag -l v8.0-submission
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

> **冻结条件**: 25 项全部通过方可打标签冻结。如有未通过项，须走例外审批流程（详见 `docs/code_freeze_policy.md`）。

**冻结人签字**: _______________  **日期**: _______________

---

*本文档与 `docs/code_freeze_policy.md`（冻结流程规范）配合使用。*
