> ⚠️ **已废弃文档** — 本文档中的 +88.3%/76.4% 等数字为旧版 14 维模型数据，
> 已被 v9.1 16 维模型的 +123.4% 取代。详见 `config/statistics.yaml`。
> 保留本文档仅用于历史追溯，不作为数据基准。

# zyhsga 定时任务提示词（4个任务完整版）

> zyhsga 新加入项目，需要一次性创建4个定时任务。
> 复制下面代码块全部内容，粘贴到 TRAE 发送即可。
> 4个任务时间与其他队友完全错开，不冲突。

---

## 通用说明

zyhsga，欢迎加入项目。我在 TRAE 里配置了自动定时任务系统，每天会自动在 GitHub 上提醒你的任务进度、发现改进点、自动完成 Issue，你不用主动登录查看，GitHub 会发邮件通知你。

**使用方法：**
1. 打开 TRAE，打开项目仓库文件夹
2. 找到下面的提示词，完整复制代码块
3. 粘贴到 TRAE 对话框中发送
4. TRAE 会自动帮你创建4个定时任务
5. 之后自动运行，你会收到 GitHub 邮件通知

**前提条件（只需做一次）：**
- TRAE 需要打开项目仓库文件夹，且工作时间内保持 TRAE 后台运行（定时任务需要 TRAE 在线才能执行）
- 配置 GitHub CLI 认证：在 TRAE 终端中执行：
```
$token = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'
[System.Environment]::SetEnvironmentVariable('GH_TOKEN', $token, 'User')
```
- 执行后重启 TRAE

**注意事项：**
- 如果 TRAE 没有打开，定时任务不会执行，所以请工作时间内保持 TRAE 运行
- 如果 GitHub 邮件通知突然停止，可能是 Token 过期，在 TRAE 终端重新执行上面的认证命令
- 在 TRAE 中说"列出我的定时任务"可以管理（暂停/修改/删除）

---

## zyhsga 的提示词

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建4个定时任务来辅助我的项目开发工作。

=========================================
任务1：每日Issue和PR进度提醒
=========================================

每天上午11:00执行。

执行步骤：
1. 设置GitHub CLI认证：$env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 查询分配给我的所有open issues：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee zyhsga --limit 50 --json number,title,labels,updatedAt

3. 查询我提交的所有open PRs：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --author zyhsga --limit 20 --json number,title,headRefName,updatedAt

4. 对每个PR检查最新审查状态：
   gh pr view <PR号> -R xiabai2008/quantum-rl-scheduler --json reviews,state,statusCheckRollup

5. 对有review反馈的PR，检查是否有新commit（是否已修复）：
   gh pr view <PR号> -R xiabai2008/quantum-rl-scheduler --json commits

6. 对我的每个Issue，判断是否有关联PR（检查Issue评论或PR标题中是否含"Closes #Issue号"）

7. 在需要提醒的Issue下发布评论（@zyhsga），内容包括：
   - 距上次更新天数
   - 有关联PR：PR审查状态、是否需要修复
   - 无关联PR：提醒开始开发
   - 距代码冻结(2026-08-15)剩余天数
   - 按优先级标签排序建议优先处理哪个

8. 有review反馈但未修复的PR：在PR下评论提醒我修复
9. 已修复(有新commit)但未被re-review的PR：在PR下@xiabai2008请求重新审查

约束：不对2天内更新过的Issue催促；无事项则不操作。

cron表达式：0 11 * * *
时区：Asia/Shanghai
任务名称：zyhsga每日进度提醒

=========================================
任务2：每日自动提Issue（AI可执行任务）
=========================================

每天下午16:00执行。

1. 设置GitHub CLI认证：$env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 审视项目（必做）：
   - 读取 AGENTS.md 了解项目全貌
   - 运行 ruff check src/ scripts/ tests/ --statistics 2>&1
   - 运行 pytest tests/ -n auto --dist loadscope --cov=src --cov-report=term-missing --tb=no -q 2>&1
   - 查看现有open issues避免重复：gh issue list -R xiabai2008/quantum-rl-scheduler --state open --limit 100 --json number,title
   - 检查 docs/ 目录文档完整性
   - 检查最近1小时内创建的Issue避免与相邻时段队友重复：gh issue list -R xiabai2008/quantum-rl-scheduler --state open --limit 10 --json number,title,createdAt

3. 比赛背景：量子AI双向赋能（RL调度+量子退火），评审标准：主题契合度/技术创新性/方案可行性/落地价值/验证严谨性，代码冻结2026-08-15

4. 从以下维度发现1个AI可完成的改进任务：
   主攻方向：文档完善（API文档完善、架构文档更新、Code Wiki维护、答辩QA手册补充、README优化、依赖管理文档更新、需求追溯矩阵更新、贡献指南完善）
   备选方向1：测试覆盖（覆盖率低于80%的模块、缺少边界/异常测试、property-based testing补充）
   备选方向2：功能增强（强化量子AI双向赋能故事线的新功能，如新的量子退火应用场景、多目标调度策略扩展、DAG调度增强）
   不涉及方向：实验严谨性、落地价值、竞赛对齐、性能优化、可解释性/可视化（由其他队友负责，避免重复）
   注意：代码质量方向（ruff/mypy/docstring）已完成，不要再创建此类Issue
   优先从主攻方向找改进点，主攻方向无改进点时再从备选中选1个

5. 严格约束：
   - 只创建1个Issue，宁缺毋滥，无改进点则不创建
   - 不创建比赛交付物Issue（PPT/白皮书/演示视频）
   - 不创建需要真机操作的Issue
   - 不与现有open issue重复，不与最近1小时内创建的Issue重复
   - 不连续两次创建相同改进类型
   - 先检查我名下已有多少个ai-ready标签的open issue，如果已有3个以上则不创建新Issue（先完成现有的）
   - 检查我名下总open issue数（含所有标签），超过6个则不创建新Issue
   - 不活跃成员 K1660729 现阶段不参与项目，不要将Issue分配给他
   - 宁缺毋滥，无改进点则不创建，不要为了凑数而创建低价值Issue
   - 预计工作量1-4小时

6. 创建Issue：
   gh issue create -R xiabai2008/quantum-rl-scheduler --title "【改进类型】简短描述" --body "正文" --assignee "zyhsga" --label "优先级-中" --label "ai-ready"

   Issue正文必须包含：
   ## 背景
   ## 具体任务（列出文件和代码位置）
   ## AI执行指引（打开TRAE就能做的步骤）
   ## 验收标准
   ## 关联（对齐哪个评审标准，预计工作量）
   
   项目权威数字（不可篡改，写入Issue正文）：
   - 50seed仿真：PPO=2746.94±1160.72 vs FCFS=1458.77±60.47, 提升+88.3%, Mann-Whitney U检验 p=1.032e-42, rank-biserial=-0.71
   - 多seed真机：PPO=1665.22±324.51 vs FCFS=353.22±53.33, Cohen's d=5.33, p=6.83e-04（Bonferroni校正后显著）
   - 所有实验数据和PR必须与以上数字一致

cron表达式：0 16 * * *
时区：Asia/Shanghai
任务名称：zyhsga每日自动提Issue

=========================================
任务3：每周文档完整性检查
=========================================

每周日上午10:00执行。

1. 验证GH_TOKEN：gh auth status 2>&1，如果失败则在Issue #58下评论"@zyhsga GH_TOKEN已过期，请重新执行认证命令"

2. 检查文档完整性：
   - docs/ 目录下所有文档是否存在且非空
   - README.md 是否与当前代码结构一致
   - docs/api_reference.md 是否与 src/ 中的实际函数签名一致
   - docs/Code_Wiki.md 是否与最新代码结构匹配
   - docs/requirements_traceability.md 需求追溯矩阵是否完整
   - AGENTS.md 中的代码结构树是否与实际目录一致
   - CONTRIBUTING.md 贡献指南是否最新

3. 检查文档与权威数字一致性：
   - 搜索 docs/ 和 README.md 中的实验数字
   - 与权威数字对比（PPO=2746.94±1160.72, FCFS=1458.77±60.47, +88.3%；多seed真机 PPO=1665.22±324.51 vs FCFS=353.22±53.33, d=5.33, p=6.83e-04）
   - 如果偏差超过5%或数字不一致，创建GitHub Issue报告文档不一致，分配给zyhsga

4. 获取本周我的commit：git log --author="zyhsga" --since="7 days ago" --oneline

5. 如果发现文档问题，创建Issue：
   标题："【每周文档检查】YYYY-MM-DD 文档问题"
   分配给 zyhsga
   内容：问题列表、涉及文件、修复建议

6. 如果本周无commit，在我的一个open issue下评论："本周暂无代码提交，建议本周至少完成一个Issue。"

7. 全部正常且有提交则不操作（静默）。

已知flaky test `test_annealing_loop.py::test_callback_triggers_submit` 在Python 3.10/3.11上偶发失败（竞态条件，Issue #65已由PR#71修复）。如果仅此一个测试失败，不要创建Issue，属正常现象。

cron表达式：0 10 * * 0
时区：Asia/Shanghai
任务名称：zyhsga每周文档完整性检查

=========================================
任务4：每日自动认领并完成 ai-ready Issue
=========================================

每天下午19:00执行。

你是一个自主开发助手。你的任务是：在项目仓库 xiabai2008/quantum-rl-scheduler 中，找到分配给我的、带有 ai-ready 标签的未完成 Issue，按照 Issue 中的"AI执行指引"完成代码实现，通过所有质量检查后提交 PR。

项目背景：量子AI双向赋能调度系统（RL调度+量子退火），Python 3.10+，使用 ruff/mypy/pytest 质量门禁，代码冻结日期 2026-08-15。

权威数字（禁止修改）：PPO=2746.94±1160.72, FCFS=1458.77±60.47, 提升+88.3%, Mann-Whitney U检验 p=1.032e-42, rank-biserial=-0.71；多seed真机 PPO=1665.22±324.51 vs FCFS=353.22±53.33, d=5.33, p=6.83e-04（Bonferroni校正后显著）。

执行步骤：

1. 设置GitHub CLI认证：
   $env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 同步最新代码：
   git checkout main
   git pull origin main
   （如果当前分支有未提交的改动，先 git stash）

3. 查询分配给我的 ai-ready 标签 open Issues：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee zyhsga --label "ai-ready" --limit 20 --json number,title,body,labels,updatedAt

4. 如果没有分配给我的 ai-ready Issue，再查未分配的：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --label "ai-ready" --limit 20 --json number,title,body,updatedAt,assignees
   筛选 assignees 为空的 Issue，认领一个：
   gh issue edit <issue号> -R xiabai2008/quantum-rl-scheduler --add-assignee zyhsga

5. 如果步骤3和4都没有可用 Issue，直接结束（无需操作）。

6. 查询我名下所有 open PR，记录已有关联 Issue 号：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --author zyhsga --limit 20 --json number,title,body
   对每个 PR 的 body 检查是否包含 "Closes #<issue号>"，记录已被处理的 Issue 号。

7. 从第3步（或第4步认领）的 Issue 列表中，排除第6步已有关联 PR 的 Issue。
   如果全部已有关联 PR，直接结束（无需操作）。

8. 从剩余 Issue 中选择1个优先处理：选择 updatedAt 最早的（最久未更新的）。
   如果有"优先级-高"标签的Issue，优先选择。

9. 读取选中 Issue 的完整正文：
   gh issue view <issue号> -R xiabai2008/quantum-rl-scheduler --json body
   找到 "## AI执行指引" 章节理解任务，同时阅读 "## 具体任务" 和 "## 验收标准"。

10. 创建功能分支：
    git checkout -b feature/issue-<issue号>
    （如果分支已存在，先 git branch -D 删除旧分支再创建）

11. 按 Issue 的"AI执行指引"实现代码修改：
    - 仔细阅读指引中提到的源文件和代码位置
    - 严格遵循项目编码规范：
      * Python 3.10+ 语法
      * 函数必须有 docstring
      * 函数/变量用 snake_case，类名用 PascalCase
      * 类型注解必须完整（mypy strict 模式）
      * ruff format 格式化（行宽88）
    - 禁止修改以下内容：
      * AGENTS.md 中的权威数字
      * models/ 和 deliverable_models/ 目录下的模型文件
      * results/ 目录下的实验数据文件
    - 已知 flaky 测试：test_annealing_loop.py::test_callback_triggers_submit 在 Python 3.10 上可能随机失败，如果只有这个测试失败可以忽略

12. 代码质量检查（必须全部通过才能提交）：
    ruff format src/ scripts/ tests/
    ruff check src/ scripts/ tests/
    mypy src/
    pytest tests/ -n auto --dist loadscope --tb=short -q

13. 如果检查失败：
    a. 分析错误原因，修复代码
    b. 重新运行全部质量检查
    c. 最多重试3轮
    d. 3轮后仍有失败：在 Issue 下评论说明情况，然后清理分支结束
       gh issue comment <issue号> -R xiabai2008/quantum-rl-scheduler --body "@zyhsga 自动执行此Issue时遇到问题，需要人工介入：\n\n**失败步骤**：...\n**错误信息**：...\n**已尝试的修复**：..."
       git checkout main
       git branch -D feature/issue-<issue号>

14. 如果质量检查全部通过：
    git add -A
    git commit -m "feat: <简短描述> (Closes #<issue号>)"
    git push origin feature/issue-<issue号>

15. 创建 Pull Request：
    gh pr create -R xiabai2008/quantum-rl-scheduler --base main --head feature/issue-<issue号> --title "feat: <简短描述>" --body "## 关联Issue\nCloses #<issue号>\n\n## 改动说明\n<描述具体改动内容>\n\n## 验证\n- [x] ruff check 通过\n- [x] ruff format 通过\n- [x] mypy 通过\n- [x] pytest 通过\n\n## 权威数字一致性\n- [x] 未修改 PPO/FCFS 权威数字\n- [x] 未修改实验数据文件" --assignee "zyhsga"

16. 在 Issue 下评论通知：
    gh issue comment <issue号> -R xiabai2008/quantum-rl-scheduler --body "已创建 PR 处理此Issue，等待 @xiabai2008 审查。"

约束：
- 每天最多完成1个 Issue
- 只处理 ai-ready 标签的 open Issue
- 不直接 push 到 main 分支
- 不修改权威实验数字和实验数据文件
- 不创建比赛交付物（PPT/白皮书/视频）
- 如果 Issue 的"AI执行指引"不清晰或任务超出AI能力，在 Issue 下评论说明情况
- commit 格式遵循：feat/fix/docs/test/refactor: <描述>
- PR body 必须包含 "Closes #<issue号>" 以自动关联 Issue

cron表达式：0 19 * * *
时区：Asia/Shanghai
任务名称：zyhsga每日自动完成Issue
```

---

## zyhsga 4个任务时间总览

| 任务 | 执行时间 | cron表达式 | 说明 |
|:--|:--|:--|:--|
| 任务1 每日进度提醒 | 11:00 | `0 11 * * *` | 在DUMNOX(10:30)之后，不与其他人冲突 |
| 任务2 每日自动提Issue | 16:00 | `0 16 * * *` | 在Izzro(15:30)之后，不与其他人冲突 |
| 任务3 每周文档检查 | 周日10:00 | `0 10 * * 0` | 新slot，不与其他人的周六检查冲突 |
| 任务4 每日自动完成Issue | 19:00 | `0 19 * * *` | 在Izzro(18:30)之后，不与其他人冲突 |

## 方向分工

| 队友 | 主攻方向 | 对齐评审标准 |
|:--|:--|:--|
| DUMNOX | 功能增强 | 技术创新性 |
| heka-ky | 落地价值 | 落地价值 |
| NN2914 | 实验严谨性 | 验证严谨性 |
| Jackhock-1 | 竞赛对齐 | 主题契合度/方案可行性 |
| qpqpalalzmzm112 | 性能优化 | 方案可行性 |
| Izzro | 可解释性/可视化 | 方案可行性/落地价值 |
| **zyhsga** | **文档完善** | **方案可行性** |

zyhsga 主攻文档完善，聚焦：API文档、架构文档、Code Wiki、答辩QA手册、README、依赖管理文档、需求追溯矩阵。这是其他6人未覆盖的方向，直接支撑方案可行性评审标准。

## 工作流程闭环

1. **11:00 任务1**：查询我的Issue/PR → 没有PR的提醒开发 → 有review反馈的提醒修复
2. **16:00 任务2**：TRAE审视项目 → 发现文档相关改进点 → 创建带"AI执行指引"的Issue
3. **19:00 任务4**：自动认领ai-ready Issue → 按指引写代码 → 通过质量检查 → 提交PR
4. **周日10:00 任务3**：检查文档完整性/权威数字一致性 → 发现问题自动创建Issue

## 注意事项

- TRAE 需要工作时间内保持运行，否则定时任务不会执行
- GitHub 邮件通知突然停止 = Token可能过期，重新执行认证命令
- 每天最多完成1个Issue，不贪多
- 不修改权威数字相关代码
- 在TRAE中说"列出我的定时任务"可以管理（暂停/修改/删除）
