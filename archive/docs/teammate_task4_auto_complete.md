> **⚠️ DEPRECATED (Issue #792)**: This document contains outdated statistics (+88.3%/76.4%). Current verified values: +123.4% (PPO vs FCFS, N=250, p=1.032e-42). Refer to `results/reports/` for current data.

# 第4个定时任务：每日自动认领并完成 ai-ready Issue

> 瑞哥发给各队友。每人找到自己的部分，复制代码块粘贴到 TRAE 发送即可。
> 这是新增的第4个定时任务，与你已有的3个任务并行运行，互不影响。
> 这个任务会自动认领分配给你的 ai-ready Issue，按 Issue 中的「AI执行指引」完成代码实现，提交PR。

---

## 通用说明

**这个任务做什么：**
1. 查询分配给你的、带 `ai-ready` 标签的 open Issue
2. 跳过已有关联 PR 的 Issue（避免重复）
3. 选1个最久未更新的 Issue，读取其中的「## AI执行指引」
4. 拉取最新代码，创建功能分支
5. 按「AI执行指引」完成代码实现
6. 运行 ruff format / ruff check / mypy / pytest 质量门禁
7. 提交、推送、创建 PR（标题含 `Closes #Issue号`）

**前提条件（已有3个任务的队友已配置，无需重复）：**
- TRAE 打开项目仓库文件夹，工作时间保持运行
- GH_TOKEN 已配置

**执行时间排布（与已有3个任务不冲突）：**

| 顺序 | 队友 | 执行时间 | cron表达式 |
|:--:|:--|:--|:--|
| 1 | DUMNOX | 16:00 | `0 16 * * *` |
| 2 | heka-ky | 16:30 | `30 16 * * *` |
| 3 | NN2914 | 17:00 | `0 17 * * *` |
| 4 | Jackhock-1 | 17:30 | `30 17 * * *` |
| 5 | qpqpalalzmzm112 | 18:00 | `0 18 * * *` |
| 6 | Izzro | 18:30 | `30 18 * * *` |

- 6人每人每天1次，相邻间隔30分钟
- 与任务1（09:30/10:30）和任务2（13:00-15:30）不冲突
- 每人每天最多完成1个 Issue，不会过载

---

## DUMNOX

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建1个定时任务。

任务：每日自动认领并完成 ai-ready Issue

每天下午16:00执行。

执行步骤：

1. 设置GitHub CLI认证：
   $env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 同步最新代码：
   git checkout main
   git pull origin main

3. 查询分配给我的、带ai-ready标签的open Issue：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee DUMNOX --label "ai-ready" --limit 20 --json number,title,body,labels,updatedAt

4. 如果没有ai-ready的open Issue，则停止（无事项，静默）。

5. 对每个Issue，检查是否已有关联PR（搜索PR标题或body中是否含"Closes #Issue号"或"Issue号"）：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --search "Issue号" --limit 10 --json number,title,body

6. 过滤掉已有关联PR的Issue。如果全部Issue都已有PR，则停止（静默）。

7. 从剩余Issue中选择updatedAt最早的那个（最久未更新的优先处理）。
   如果有"优先级-高"标签的Issue，优先选择。

8. 读取选中Issue的body，找到「## AI执行指引」部分，按其中的步骤完成代码实现：
   - 创建功能分支：git checkout -b feature/issue-<Issue号>-<简短描述>
   - 按AI执行指引修改代码
   - 实现时必须遵守项目规范：Python 3.10+、snake_case函数名、PascalCase类名、函数必须有docstring、类型注解

9. 权威数字保护（不可篡改，修改代码时不得改动以下数字相关内容）：
   - 50seed仿真：PPO=2746.94±1160.72 vs FCFS=1458.77±60.47, 提升+88.3%, Mann-Whitney U检验 p=1.032e-42, rank-biserial=-0.71
   - 多seed真机：PPO=1665.22±324.51 vs FCFS=353.22±53.33, Cohen's d=5.33, p=6.83e-04（Bonferroni校正后显著）

10. 质量门禁（必须全部通过才能提交PR）：
    - ruff format src/ scripts/ tests/
    - ruff check src/ scripts/ tests/
    - mypy src/
    - pytest tests/ -n auto --dist loadscope --tb=short -q

    如果有错误，修复后重新运行，直到全部通过。
    已知flaky test `test_annealing_loop.py::test_callback_triggers_submit` 偶发失败属正常，忽略此单个失败。

11. 提交并推送：
    git add -A
    git commit -m "feat: 完成Issue #<Issue号> <简短描述>"
    git push origin feature/issue-<Issue号>-<简短描述>

12. 创建PR：
    gh pr create -R xiabai2008/quantum-rl-scheduler --base main --head feature/issue-<Issue号>-<简短描述> --title "feat: 完成Issue #<Issue号> <简短描述>" --body "## 关联Issue\nCloses #<Issue号>\n\n## 改动说明\n<简述改动内容>\n\n## 验收标准\n<对照Issue中的验收标准逐项说明>\n\n## 质量检查\n- [x] ruff format\n- [x] ruff check\n- [x] mypy\n- [x] pytest" --assignee "DUMNOX"

13. 在Issue下评论：
    gh issue comment <Issue号> -R xiabai2008/quantum-rl-scheduler --body "已通过PR #<PR号> 完成此Issue，请@xiabai2008 审查。"

约束：
- 每天只完成1个Issue，不贪多
- 无ai-ready Issue则不操作（静默）
- 所有Issue都已有PR则不操作（静默）
- 不修改权威数字相关代码
- 不创建比赛交付物（PPT/白皮书/演示视频）
- 不执行真机操作

cron表达式：0 16 * * *
时区：Asia/Shanghai
任务名称：DUMNOX每日自动完成Issue
```

---

## heka-ky

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建1个定时任务。

任务：每日自动认领并完成 ai-ready Issue

每天下午16:30执行。

执行步骤：

1. 设置GitHub CLI认证：
   $env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 同步最新代码：
   git checkout main
   git pull origin main

3. 查询分配给我的、带ai-ready标签的open Issue：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee heka-ky --label "ai-ready" --limit 20 --json number,title,body,labels,updatedAt

4. 如果没有ai-ready的open Issue，则停止（无事项，静默）。

5. 对每个Issue，检查是否已有关联PR（搜索PR标题或body中是否含"Closes #Issue号"或"Issue号"）：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --search "Issue号" --limit 10 --json number,title,body

6. 过滤掉已有关联PR的Issue。如果全部Issue都已有PR，则停止（静默）。

7. 从剩余Issue中选择updatedAt最早的那个（最久未更新的优先处理）。
   如果有"优先级-高"标签的Issue，优先选择。

8. 读取选中Issue的body，找到「## AI执行指引」部分，按其中的步骤完成代码实现：
   - 创建功能分支：git checkout -b feature/issue-<Issue号>-<简短描述>
   - 按AI执行指引修改代码
   - 实现时必须遵守项目规范：Python 3.10+、snake_case函数名、PascalCase类名、函数必须有docstring、类型注解

9. 权威数字保护（不可篡改，修改代码时不得改动以下数字相关内容）：
   - 50seed仿真：PPO=2746.94±1160.72 vs FCFS=1458.77±60.47, 提升+88.3%, Mann-Whitney U检验 p=1.032e-42, rank-biserial=-0.71
   - 多seed真机：PPO=1665.22±324.51 vs FCFS=353.22±53.33, Cohen's d=5.33, p=6.83e-04（Bonferroni校正后显著）

10. 质量门禁（必须全部通过才能提交PR）：
    - ruff format src/ scripts/ tests/
    - ruff check src/ scripts/ tests/
    - mypy src/
    - pytest tests/ -n auto --dist loadscope --tb=short -q

    如果有错误，修复后重新运行，直到全部通过。
    已知flaky test `test_annealing_loop.py::test_callback_triggers_submit` 偶发失败属正常，忽略此单个失败。

11. 提交并推送：
    git add -A
    git commit -m "feat: 完成Issue #<Issue号> <简短描述>"
    git push origin feature/issue-<Issue号>-<简短描述>

12. 创建PR：
    gh pr create -R xiabai2008/quantum-rl-scheduler --base main --head feature/issue-<Issue号>-<简短描述> --title "feat: 完成Issue #<Issue号> <简短描述>" --body "## 关联Issue\nCloses #<Issue号>\n\n## 改动说明\n<简述改动内容>\n\n## 验收标准\n<对照Issue中的验收标准逐项说明>\n\n## 质量检查\n- [x] ruff format\n- [x] ruff check\n- [x] mypy\n- [x] pytest" --assignee "heka-ky"

13. 在Issue下评论：
    gh issue comment <Issue号> -R xiabai2008/quantum-rl-scheduler --body "已通过PR #<PR号> 完成此Issue，请@xiabai2008 审查。"

约束：
- 每天只完成1个Issue，不贪多
- 无ai-ready Issue则不操作（静默）
- 所有Issue都已有PR则不操作（静默）
- 不修改权威数字相关代码
- 不创建比赛交付物（PPT/白皮书/演示视频）
- 不执行真机操作

cron表达式：30 16 * * *
时区：Asia/Shanghai
任务名称：heka-ky每日自动完成Issue
```

---

## NN2914

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建1个定时任务。

任务：每日自动认领并完成 ai-ready Issue

每天下午17:00执行。

执行步骤：

1. 设置GitHub CLI认证：
   $env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 同步最新代码：
   git checkout main
   git pull origin main

3. 查询分配给我的、带ai-ready标签的open Issue：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee NN2914 --label "ai-ready" --limit 20 --json number,title,body,labels,updatedAt

4. 如果没有ai-ready的open Issue，则停止（无事项，静默）。

5. 对每个Issue，检查是否已有关联PR（搜索PR标题或body中是否含"Closes #Issue号"或"Issue号"）：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --search "Issue号" --limit 10 --json number,title,body

6. 过滤掉已有关联PR的Issue。如果全部Issue都已有PR，则停止（静默）。

7. 从剩余Issue中选择updatedAt最早的那个（最久未更新的优先处理）。
   如果有"优先级-高"标签的Issue，优先选择。

8. 读取选中Issue的body，找到「## AI执行指引」部分，按其中的步骤完成代码实现：
   - 创建功能分支：git checkout -b feature/issue-<Issue号>-<简短描述>
   - 按AI执行指引修改代码
   - 实现时必须遵守项目规范：Python 3.10+、snake_case函数名、PascalCase类名、函数必须有docstring、类型注解

9. 权威数字保护（不可篡改，修改代码时不得改动以下数字相关内容）：
   - 50seed仿真：PPO=2746.94±1160.72 vs FCFS=1458.77±60.47, 提升+88.3%, Mann-Whitney U检验 p=1.032e-42, rank-biserial=-0.71
   - 多seed真机：PPO=1665.22±324.51 vs FCFS=353.22±53.33, Cohen's d=5.33, p=6.83e-04（Bonferroni校正后显著）

10. 质量门禁（必须全部通过才能提交PR）：
    - ruff format src/ scripts/ tests/
    - ruff check src/ scripts/ tests/
    - mypy src/
    - pytest tests/ -n auto --dist loadscope --tb=short -q

    如果有错误，修复后重新运行，直到全部通过。
    已知flaky test `test_annealing_loop.py::test_callback_triggers_submit` 偶发失败属正常，忽略此单个失败。

11. 提交并推送：
    git add -A
    git commit -m "feat: 完成Issue #<Issue号> <简短描述>"
    git push origin feature/issue-<Issue号>-<简短描述>

12. 创建PR：
    gh pr create -R xiabai2008/quantum-rl-scheduler --base main --head feature/issue-<Issue号>-<简短描述> --title "feat: 完成Issue #<Issue号> <简短描述>" --body "## 关联Issue\nCloses #<Issue号>\n\n## 改动说明\n<简述改动内容>\n\n## 验收标准\n<对照Issue中的验收标准逐项说明>\n\n## 质量检查\n- [x] ruff format\n- [x] ruff check\n- [x] mypy\n- [x] pytest" --assignee "NN2914"

13. 在Issue下评论：
    gh issue comment <Issue号> -R xiabai2008/quantum-rl-scheduler --body "已通过PR #<PR号> 完成此Issue，请@xiabai2008 审查。"

约束：
- 每天只完成1个Issue，不贪多
- 无ai-ready Issue则不操作（静默）
- 所有Issue都已有PR则不操作（静默）
- 不修改权威数字相关代码
- 不创建比赛交付物（PPT/白皮书/演示视频）
- 不执行真机操作

cron表达式：0 17 * * *
时区：Asia/Shanghai
任务名称：NN2914每日自动完成Issue
```

---

## Jackhock-1

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建1个定时任务。

任务：每日自动认领并完成 ai-ready Issue

每天下午17:30执行。

执行步骤：

1. 设置GitHub CLI认证：
   $env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 同步最新代码：
   git checkout main
   git pull origin main

3. 查询分配给我的、带ai-ready标签的open Issue：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee Jackhock-1 --label "ai-ready" --limit 20 --json number,title,body,labels,updatedAt

4. 如果没有ai-ready的open Issue，则停止（无事项，静默）。

5. 对每个Issue，检查是否已有关联PR（搜索PR标题或body中是否含"Closes #Issue号"或"Issue号"）：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --search "Issue号" --limit 10 --json number,title,body

6. 过滤掉已有关联PR的Issue。如果全部Issue都已有PR，则停止（静默）。

7. 从剩余Issue中选择updatedAt最早的那个（最久未更新的优先处理）。
   如果有"优先级-高"标签的Issue，优先选择。

8. 读取选中Issue的body，找到「## AI执行指引」部分，按其中的步骤完成代码实现：
   - 创建功能分支：git checkout -b feature/issue-<Issue号>-<简短描述>
   - 按AI执行指引修改代码
   - 实现时必须遵守项目规范：Python 3.10+、snake_case函数名、PascalCase类名、函数必须有docstring、类型注解

9. 权威数字保护（不可篡改，修改代码时不得改动以下数字相关内容）：
   - 50seed仿真：PPO=2746.94±1160.72 vs FCFS=1458.77±60.47, 提升+88.3%, Mann-Whitney U检验 p=1.032e-42, rank-biserial=-0.71
   - 多seed真机：PPO=1665.22±324.51 vs FCFS=353.22±53.33, Cohen's d=5.33, p=6.83e-04（Bonferroni校正后显著）

10. 质量门禁（必须全部通过才能提交PR）：
    - ruff format src/ scripts/ tests/
    - ruff check src/ scripts/ tests/
    - mypy src/
    - pytest tests/ -n auto --dist loadscope --tb=short -q

    如果有错误，修复后重新运行，直到全部通过。
    已知flaky test `test_annealing_loop.py::test_callback_triggers_submit` 偶发失败属正常，忽略此单个失败。

11. 提交并推送：
    git add -A
    git commit -m "feat: 完成Issue #<Issue号> <简短描述>"
    git push origin feature/issue-<Issue号>-<简短描述>

12. 创建PR：
    gh pr create -R xiabai2008/quantum-rl-scheduler --base main --head feature/issue-<Issue号>-<简短描述> --title "feat: 完成Issue #<Issue号> <简短描述>" --body "## 关联Issue\nCloses #<Issue号>\n\n## 改动说明\n<简述改动内容>\n\n## 验收标准\n<对照Issue中的验收标准逐项说明>\n\n## 质量检查\n- [x] ruff format\n- [x] ruff check\n- [x] mypy\n- [x] pytest" --assignee "Jackhock-1"

13. 在Issue下评论：
    gh issue comment <Issue号> -R xiabai2008/quantum-rl-scheduler --body "已通过PR #<PR号> 完成此Issue，请@xiabai2008 审查。"

约束：
- 每天只完成1个Issue，不贪多
- 无ai-ready Issue则不操作（静默）
- 所有Issue都已有PR则不操作（静默）
- 不修改权威数字相关代码
- 不创建比赛交付物（PPT/白皮书/演示视频）
- 不执行真机操作

cron表达式：30 17 * * *
时区：Asia/Shanghai
任务名称：Jackhock-1每日自动完成Issue
```

---

## qpqpalalzmzm112

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建1个定时任务。

任务：每日自动认领并完成 ai-ready Issue

每天下午18:00执行。

执行步骤：

1. 设置GitHub CLI认证：
   $env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 同步最新代码：
   git checkout main
   git pull origin main

3. 查询分配给我的、带ai-ready标签的open Issue：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee qpqpalalzmzm112 --label "ai-ready" --limit 20 --json number,title,body,labels,updatedAt

4. 如果没有ai-ready的open Issue，则停止（无事项，静默）。

5. 对每个Issue，检查是否已有关联PR（搜索PR标题或body中是否含"Closes #Issue号"或"Issue号"）：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --search "Issue号" --limit 10 --json number,title,body

6. 过滤掉已有关联PR的Issue。如果全部Issue都已有PR，则停止（静默）。

7. 从剩余Issue中选择updatedAt最早的那个（最久未更新的优先处理）。
   如果有"优先级-高"标签的Issue，优先选择。

8. 读取选中Issue的body，找到「## AI执行指引」部分，按其中的步骤完成代码实现：
   - 创建功能分支：git checkout -b feature/issue-<Issue号>-<简短描述>
   - 按AI执行指引修改代码
   - 实现时必须遵守项目规范：Python 3.10+、snake_case函数名、PascalCase类名、函数必须有docstring、类型注解

9. 权威数字保护（不可篡改，修改代码时不得改动以下数字相关内容）：
   - 50seed仿真：PPO=2746.94±1160.72 vs FCFS=1458.77±60.47, 提升+88.3%, Mann-Whitney U检验 p=1.032e-42, rank-biserial=-0.71
   - 多seed真机：PPO=1665.22±324.51 vs FCFS=353.22±53.33, Cohen's d=5.33, p=6.83e-04（Bonferroni校正后显著）

10. 质量门禁（必须全部通过才能提交PR）：
    - ruff format src/ scripts/ tests/
    - ruff check src/ scripts/ tests/
    - mypy src/
    - pytest tests/ -n auto --dist loadscope --tb=short -q

    如果有错误，修复后重新运行，直到全部通过。
    已知flaky test `test_annealing_loop.py::test_callback_triggers_submit` 偶发失败属正常，忽略此单个失败。

11. 提交并推送：
    git add -A
    git commit -m "feat: 完成Issue #<Issue号> <简短描述>"
    git push origin feature/issue-<Issue号>-<简短描述>

12. 创建PR：
    gh pr create -R xiabai2008/quantum-rl-scheduler --base main --head feature/issue-<Issue号>-<简短描述> --title "feat: 完成Issue #<Issue号> <简短描述>" --body "## 关联Issue\nCloses #<Issue号>\n\n## 改动说明\n<简述改动内容>\n\n## 验收标准\n<对照Issue中的验收标准逐项说明>\n\n## 质量检查\n- [x] ruff format\n- [x] ruff check\n- [x] mypy\n- [x] pytest" --assignee "qpqpalalzmzm112"

13. 在Issue下评论：
    gh issue comment <Issue号> -R xiabai2008/quantum-rl-scheduler --body "已通过PR #<PR号> 完成此Issue，请@xiabai2008 审查。"

约束：
- 每天只完成1个Issue，不贪多
- 无ai-ready Issue则不操作（静默）
- 所有Issue都已有PR则不操作（静默）
- 不修改权威数字相关代码
- 不创建比赛交付物（PPT/白皮书/演示视频）
- 不执行真机操作

cron表达式：0 18 * * *
时区：Asia/Shanghai
任务名称：qpqpalalzmzm112每日自动完成Issue
```

---

## Izzro

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建1个定时任务。

任务：每日自动认领并完成 ai-ready Issue

每天下午18:30执行。

执行步骤：

1. 设置GitHub CLI认证：
   $env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 同步最新代码：
   git checkout main
   git pull origin main

3. 查询分配给我的、带ai-ready标签的open Issue：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee Izzro --label "ai-ready" --limit 20 --json number,title,body,labels,updatedAt

4. 如果没有ai-ready的open Issue，则停止（无事项，静默）。

5. 对每个Issue，检查是否已有关联PR（搜索PR标题或body中是否含"Closes #Issue号"或"Issue号"）：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --search "Issue号" --limit 10 --json number,title,body

6. 过滤掉已有关联PR的Issue。如果全部Issue都已有PR，则停止（静默）。

7. 从剩余Issue中选择updatedAt最早的那个（最久未更新的优先处理）。
   如果有"优先级-高"标签的Issue，优先选择。

8. 读取选中Issue的body，找到「## AI执行指引」部分，按其中的步骤完成代码实现：
   - 创建功能分支：git checkout -b feature/issue-<Issue号>-<简短描述>
   - 按AI执行指引修改代码
   - 实现时必须遵守项目规范：Python 3.10+、snake_case函数名、PascalCase类名、函数必须有docstring、类型注解

9. 权威数字保护（不可篡改，修改代码时不得改动以下数字相关内容）：
   - 50seed仿真：PPO=2746.94±1160.72 vs FCFS=1458.77±60.47, 提升+88.3%, Mann-Whitney U检验 p=1.032e-42, rank-biserial=-0.71
   - 多seed真机：PPO=1665.22±324.51 vs FCFS=353.22±53.33, Cohen's d=5.33, p=6.83e-04（Bonferroni校正后显著）

10. 质量门禁（必须全部通过才能提交PR）：
    - ruff format src/ scripts/ tests/
    - ruff check src/ scripts/ tests/
    - mypy src/
    - pytest tests/ -n auto --dist loadscope --tb=short -q

    如果有错误，修复后重新运行，直到全部通过。
    已知flaky test `test_annealing_loop.py::test_callback_triggers_submit` 偶发失败属正常，忽略此单个失败。

11. 提交并推送：
    git add -A
    git commit -m "feat: 完成Issue #<Issue号> <简短描述>"
    git push origin feature/issue-<Issue号>-<简短描述>

12. 创建PR：
    gh pr create -R xiabai2008/quantum-rl-scheduler --base main --head feature/issue-<Issue号>-<简短描述> --title "feat: 完成Issue #<Issue号> <简短描述>" --body "## 关联Issue\nCloses #<Issue号>\n\n## 改动说明\n<简述改动内容>\n\n## 验收标准\n<对照Issue中的验收标准逐项说明>\n\n## 质量检查\n- [x] ruff format\n- [x] ruff check\n- [x] mypy\n- [x] pytest" --assignee "Izzro"

13. 在Issue下评论：
    gh issue comment <Issue号> -R xiabai2008/quantum-rl-scheduler --body "已通过PR #<PR号> 完成此Issue，请@xiabai2008 审查。"

约束：
- 每天只完成1个Issue，不贪多
- 无ai-ready Issue则不操作（静默）
- 所有Issue都已有PR则不操作（静默）
- 不修改权威数字相关代码
- 不创建比赛交付物（PPT/白皮书/演示视频）
- 不执行真机操作

cron表达式：30 18 * * *
时区：Asia/Shanghai
任务名称：Izzro每日自动完成Issue
```

---

## 4个任务时间总览

| 队友 | 任务1 进度提醒 | 任务2 每日提Issue | 任务3 每周检查 | 任务4 每日完成Issue |
|:--|:--|:--|:--|:--|
| DUMNOX | 10:30 | 13:00 | 周日15:00 | 16:00 |
| heka-ky | 09:30 | 13:30 | 周六10:00 | 16:30 |
| NN2914 | 09:30 | 14:00 | 周六11:00 | 17:00 |
| Jackhock-1 | 09:30 | 14:30 | 周六14:00 | 17:30 |
| qpqpalalzmzm112 | 09:30 | 15:00 | 周六15:00 | 18:00 |
| Izzro | 09:30 | 15:30 | 周六16:00 | 18:30 |

工作流闭环：
1. 09:30/10:30 任务1提醒进度（催你干活）
2. 13:00-15:30 任务2创建新Issue（发现要干的事）
3. 16:00-18:30 任务4自动完成Issue（AI帮你干活）
4. 周六/周日 任务3检查代码质量（体检）

## 注意事项

- TRAE 需要工作时间内保持运行，否则定时任务不会执行
- 任务4执行时间较长（需要写代码+跑测试），每次约5-15分钟
- 每天每人最多完成1个Issue，6人每天最多6个PR，不会造成审查瓶颈
- PR创建后瑞哥会收到GitHub邮件通知，进行代码审查后合并
- 如果Token过期，任务1会在Issue #58下提醒你重新配置
