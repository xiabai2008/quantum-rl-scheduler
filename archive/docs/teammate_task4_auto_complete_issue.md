# 队友第4个定时任务：每日自动认领并完成 ai-ready Issue

> **此文件为独立的第4个任务，与之前3个任务文件分开。**
> 队友已实施前3个任务（每日提醒 + 自动提Issue + 每周检查），此文件仅包含新增的第4个任务。
> 每人找到自己的部分，复制代码块粘贴到 TRAE 发送即可。

---

## 通用说明

各位，这是在之前3个定时任务基础上新增的第4个任务：**每日自动认领并完成 ai-ready Issue**。

这个任务会让 TRAE 每天自动：
1. 查找分配给你的 ai-ready 标签 Issue
2. 按 Issue 中的"AI执行指引"自动编写代码
3. 通过 ruff/mypy/pytest 质量检查后提交 PR
4. 在 Issue 下评论通知

**使用方法：**
1. 打开 TRAE，确保已打开项目仓库文件夹
2. 找到下面属于你的提示词，完整复制代码块
3. 粘贴到 TRAE 对话框中发送
4. TRAE 会自动帮你创建第4个定时任务
5. 之后每天自动运行，你会收到 GitHub PR 通知

**前提条件（已在前3个任务中配置，无需重复）：**
- TRAE 工作时间内保持运行
- GitHub CLI 认证已配置（GH_TOKEN 环境变量）

---

## DUMNOX 的提示词

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建1个定时任务来辅助我的项目开发工作。

=========================================
任务4：每日自动认领并完成 ai-ready Issue
=========================================

每天下午14:00执行。

你是一个自主开发助手。你的任务是：在项目仓库 xiabai2008/quantum-rl-scheduler 中，找到分配给我的、带有 ai-ready 标签的未完成 Issue，按照 Issue 中的"AI执行指引"完成代码实现，通过所有质量检查后提交 PR。

项目背景：量子AI双向赋能调度系统（RL调度+量子退火），Python 3.10+，使用 ruff/mypy/pytest 质量门禁，代码冻结日期 2026-08-15。

权威数字（禁止修改）：PPO=2746.94±1160.72, FCFS=1458.77±60.47, 提升+88.3%, Mann-Whitney U检验 p=1.032e-42, rank-biserial=-0.71；多seed真机 PPO=1736.32 vs FCFS=383.00, d=5.33, p<0.001。

执行步骤：

1. 设置GitHub CLI认证：
   $env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 同步最新代码：
   git checkout main
   git pull origin main
   （如果当前分支有未提交的改动，先 git stash）

3. 查询分配给我的 ai-ready 标签 open Issues：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee DUMNOX --label ai-ready --limit 20 --json number,title,body,updatedAt

4. 如果没有分配给我的 ai-ready Issue，再查未分配的：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --label ai-ready --limit 20 --json number,title,body,updatedAt,assignees
   筛选 assignees 为空的 Issue，认领一个：
   gh issue edit <issue号> -R xiabai2008/quantum-rl-scheduler --add-assignee DUMNOX

5. 如果步骤3和4都没有可用 Issue，直接结束（无需操作）。

6. 查询我名下所有 open PR，记录已有关联 Issue 号：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --author DUMNOX --limit 20 --json number,title,body
   对每个 PR 的 body 检查是否包含 "Closes #<issue号>"，记录已被处理的 Issue 号。

7. 从第3步（或第4步认领）的 Issue 列表中，排除第6步已有关联 PR 的 Issue。
   如果全部已有关联 PR，直接结束（无需操作）。

8. 从剩余 Issue 中选择1个优先处理：选择 updatedAt 最早的（最久未更新的）。

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
    pytest tests/ --tb=short -q

13. 如果检查失败：
    a. 分析错误原因，修复代码
    b. 重新运行全部质量检查
    c. 最多重试3轮
    d. 3轮后仍有失败：在 Issue 下评论说明情况，然后清理分支结束
       gh issue comment <issue号> -R xiabai2008/quantum-rl-scheduler --body "@DUMNOX 自动执行此Issue时遇到问题，需要人工介入：\n\n**失败步骤**：...\n**错误信息**：...\n**已尝试的修复**：..."
       git checkout main
       git branch -D feature/issue-<issue号>

14. 如果质量检查全部通过：
    git add -A
    git commit -m "feat: <简短描述> (Closes #<issue号>)"
    git push origin feature/issue-<issue号>

15. 创建 Pull Request：
    gh pr create -R xiabai2008/quantum-rl-scheduler --base main --head feature/issue-<issue号> --title "feat: <简短描述>" --body "## 关联Issue\nCloses #<issue号>\n\n## 改动说明\n<描述具体改动内容>\n\n## 验证\n- [x] ruff check 通过\n- [x] ruff format 通过\n- [x] mypy 通过\n- [x] pytest 通过\n\n## 权威数字一致性\n- [x] 未修改 PPO/FCFS 权威数字\n- [x] 未修改实验数据文件" --assignee "DUMNOX"

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

cron表达式：0 14 * * *
时区：Asia/Shanghai
任务名称：DUMNOX每日自动完成Issue
```

---

## heka-ky 的提示词

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建1个定时任务来辅助我的项目开发工作。

=========================================
任务4：每日自动认领并完成 ai-ready Issue
=========================================

每天下午14:30执行。

你是一个自主开发助手。你的任务是：在项目仓库 xiabai2008/quantum-rl-scheduler 中，找到分配给我的、带有 ai-ready 标签的未完成 Issue，按照 Issue 中的"AI执行指引"完成代码实现，通过所有质量检查后提交 PR。

项目背景：量子AI双向赋能调度系统（RL调度+量子退火），Python 3.10+，使用 ruff/mypy/pytest 质量门禁，代码冻结日期 2026-08-15。

权威数字（禁止修改）：PPO=2746.94±1160.72, FCFS=1458.77±60.47, 提升+88.3%, Mann-Whitney U检验 p=1.032e-42, rank-biserial=-0.71；多seed真机 PPO=1736.32 vs FCFS=383.00, d=5.33, p<0.001。

执行步骤：

1. 设置GitHub CLI认证：
   $env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 同步最新代码：
   git checkout main
   git pull origin main
   （如果当前分支有未提交的改动，先 git stash）

3. 查询分配给我的 ai-ready 标签 open Issues：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee heka-ky --label ai-ready --limit 20 --json number,title,body,updatedAt

4. 如果没有分配给我的 ai-ready Issue，再查未分配的：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --label ai-ready --limit 20 --json number,title,body,updatedAt,assignees
   筛选 assignees 为空的 Issue，认领一个：
   gh issue edit <issue号> -R xiabai2008/quantum-rl-scheduler --add-assignee heka-ky

5. 如果步骤3和4都没有可用 Issue，直接结束（无需操作）。

6. 查询我名下所有 open PR，记录已有关联 Issue 号：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --author heka-ky --limit 20 --json number,title,body
   对每个 PR 的 body 检查是否包含 "Closes #<issue号>"，记录已被处理的 Issue 号。

7. 从第3步（或第4步认领）的 Issue 列表中，排除第6步已有关联 PR 的 Issue。
   如果全部已有关联 PR，直接结束（无需操作）。

8. 从剩余 Issue 中选择1个优先处理：选择 updatedAt 最早的（最久未更新的）。

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
    pytest tests/ --tb=short -q

13. 如果检查失败：
    a. 分析错误原因，修复代码
    b. 重新运行全部质量检查
    c. 最多重试3轮
    d. 3轮后仍有失败：在 Issue 下评论说明情况，然后清理分支结束
       gh issue comment <issue号> -R xiabai2008/quantum-rl-scheduler --body "@heka-ky 自动执行此Issue时遇到问题，需要人工介入：\n\n**失败步骤**：...\n**错误信息**：...\n**已尝试的修复**：..."
       git checkout main
       git branch -D feature/issue-<issue号>

14. 如果质量检查全部通过：
    git add -A
    git commit -m "feat: <简短描述> (Closes #<issue号>)"
    git push origin feature/issue-<issue号>

15. 创建 Pull Request：
    gh pr create -R xiabai2008/quantum-rl-scheduler --base main --head feature/issue-<issue号> --title "feat: <简短描述>" --body "## 关联Issue\nCloses #<issue号>\n\n## 改动说明\n<描述具体改动内容>\n\n## 验证\n- [x] ruff check 通过\n- [x] ruff format 通过\n- [x] mypy 通过\n- [x] pytest 通过\n\n## 权威数字一致性\n- [x] 未修改 PPO/FCFS 权威数字\n- [x] 未修改实验数据文件" --assignee "heka-ky"

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

cron表达式：30 14 * * *
时区：Asia/Shanghai
任务名称：heka-ky每日自动完成Issue
```

---

## NN2914 的提示词

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建1个定时任务来辅助我的项目开发工作。

=========================================
任务4：每日自动认领并完成 ai-ready Issue
=========================================

每天下午15:00执行。

你是一个自主开发助手。你的任务是：在项目仓库 xiabai2008/quantum-rl-scheduler 中，找到分配给我的、带有 ai-ready 标签的未完成 Issue，按照 Issue 中的"AI执行指引"完成代码实现，通过所有质量检查后提交 PR。

项目背景：量子AI双向赋能调度系统（RL调度+量子退火），Python 3.10+，使用 ruff/mypy/pytest 质量门禁，代码冻结日期 2026-08-15。

权威数字（禁止修改）：PPO=2746.94±1160.72, FCFS=1458.77±60.47, 提升+88.3%, Mann-Whitney U检验 p=1.032e-42, rank-biserial=-0.71；多seed真机 PPO=1736.32 vs FCFS=383.00, d=5.33, p<0.001。

执行步骤：

1. 设置GitHub CLI认证：
   $env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 同步最新代码：
   git checkout main
   git pull origin main
   （如果当前分支有未提交的改动，先 git stash）

3. 查询分配给我的 ai-ready 标签 open Issues：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee NN2914 --label ai-ready --limit 20 --json number,title,body,updatedAt

4. 如果没有分配给我的 ai-ready Issue，再查未分配的：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --label ai-ready --limit 20 --json number,title,body,updatedAt,assignees
   筛选 assignees 为空的 Issue，认领一个：
   gh issue edit <issue号> -R xiabai2008/quantum-rl-scheduler --add-assignee NN2914

5. 如果步骤3和4都没有可用 Issue，直接结束（无需操作）。

6. 查询我名下所有 open PR，记录已有关联 Issue 号：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --author NN2914 --limit 20 --json number,title,body
   对每个 PR 的 body 检查是否包含 "Closes #<issue号>"，记录已被处理的 Issue 号。

7. 从第3步（或第4步认领）的 Issue 列表中，排除第6步已有关联 PR 的 Issue。
   如果全部已有关联 PR，直接结束（无需操作）。

8. 从剩余 Issue 中选择1个优先处理：选择 updatedAt 最早的（最久未更新的）。

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
    pytest tests/ --tb=short -q

13. 如果检查失败：
    a. 分析错误原因，修复代码
    b. 重新运行全部质量检查
    c. 最多重试3轮
    d. 3轮后仍有失败：在 Issue 下评论说明情况，然后清理分支结束
       gh issue comment <issue号> -R xiabai2008/quantum-rl-scheduler --body "@NN2914 自动执行此Issue时遇到问题，需要人工介入：\n\n**失败步骤**：...\n**错误信息**：...\n**已尝试的修复**：..."
       git checkout main
       git branch -D feature/issue-<issue号>

14. 如果质量检查全部通过：
    git add -A
    git commit -m "feat: <简短描述> (Closes #<issue号>)"
    git push origin feature/issue-<issue号>

15. 创建 Pull Request：
    gh pr create -R xiabai2008/quantum-rl-scheduler --base main --head feature/issue-<issue号> --title "feat: <简短描述>" --body "## 关联Issue\nCloses #<issue号>\n\n## 改动说明\n<描述具体改动内容>\n\n## 验证\n- [x] ruff check 通过\n- [x] ruff format 通过\n- [x] mypy 通过\n- [x] pytest 通过\n\n## 权威数字一致性\n- [x] 未修改 PPO/FCFS 权威数字\n- [x] 未修改实验数据文件" --assignee "NN2914"

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

cron表达式：0 15 * * *
时区：Asia/Shanghai
任务名称：NN2914每日自动完成Issue
```

---

## Jackhock-1 的提示词

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建1个定时任务来辅助我的项目开发工作。

=========================================
任务4：每日自动认领并完成 ai-ready Issue
=========================================

每天下午15:30执行。

你是一个自主开发助手。你的任务是：在项目仓库 xiabai2008/quantum-rl-scheduler 中，找到分配给我的、带有 ai-ready 标签的未完成 Issue，按照 Issue 中的"AI执行指引"完成代码实现，通过所有质量检查后提交 PR。

项目背景：量子AI双向赋能调度系统（RL调度+量子退火），Python 3.10+，使用 ruff/mypy/pytest 质量门禁，代码冻结日期 2026-08-15。

权威数字（禁止修改）：PPO=2746.94±1160.72, FCFS=1458.77±60.47, 提升+88.3%, Mann-Whitney U检验 p=1.032e-42, rank-biserial=-0.71；多seed真机 PPO=1736.32 vs FCFS=383.00, d=5.33, p<0.001。

执行步骤：

1. 设置GitHub CLI认证：
   $env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 同步最新代码：
   git checkout main
   git pull origin main
   （如果当前分支有未提交的改动，先 git stash）

3. 查询分配给我的 ai-ready 标签 open Issues：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee Jackhock-1 --label ai-ready --limit 20 --json number,title,body,updatedAt

4. 如果没有分配给我的 ai-ready Issue，再查未分配的：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --label ai-ready --limit 20 --json number,title,body,updatedAt,assignees
   筛选 assignees 为空的 Issue，认领一个：
   gh issue edit <issue号> -R xiabai2008/quantum-rl-scheduler --add-assignee Jackhock-1

5. 如果步骤3和4都没有可用 Issue，直接结束（无需操作）。

6. 查询我名下所有 open PR，记录已有关联 Issue 号：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --author Jackhock-1 --limit 20 --json number,title,body
   对每个 PR 的 body 检查是否包含 "Closes #<issue号>"，记录已被处理的 Issue 号。

7. 从第3步（或第4步认领）的 Issue 列表中，排除第6步已有关联 PR 的 Issue。
   如果全部已有关联 PR，直接结束（无需操作）。

8. 从剩余 Issue 中选择1个优先处理：选择 updatedAt 最早的（最久未更新的）。

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
    pytest tests/ --tb=short -q

13. 如果检查失败：
    a. 分析错误原因，修复代码
    b. 重新运行全部质量检查
    c. 最多重试3轮
    d. 3轮后仍有失败：在 Issue 下评论说明情况，然后清理分支结束
       gh issue comment <issue号> -R xiabai2008/quantum-rl-scheduler --body "@Jackhock-1 自动执行此Issue时遇到问题，需要人工介入：\n\n**失败步骤**：...\n**错误信息**：...\n**已尝试的修复**：..."
       git checkout main
       git branch -D feature/issue-<issue号>

14. 如果质量检查全部通过：
    git add -A
    git commit -m "feat: <简短描述> (Closes #<issue号>)"
    git push origin feature/issue-<issue号>

15. 创建 Pull Request：
    gh pr create -R xiabai2008/quantum-rl-scheduler --base main --head feature/issue-<issue号> --title "feat: <简短描述>" --body "## 关联Issue\nCloses #<issue号>\n\n## 改动说明\n<描述具体改动内容>\n\n## 验证\n- [x] ruff check 通过\n- [x] ruff format 通过\n- [x] mypy 通过\n- [x] pytest 通过\n\n## 权威数字一致性\n- [x] 未修改 PPO/FCFS 权威数字\n- [x] 未修改实验数据文件" --assignee "Jackhock-1"

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

cron表达式：30 15 * * *
时区：Asia/Shanghai
任务名称：Jackhock-1每日自动完成Issue
```

---

## qpqpalalzmzm112 的提示词

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建1个定时任务来辅助我的项目开发工作。

=========================================
任务4：每日自动认领并完成 ai-ready Issue
=========================================

每天下午16:30执行。

你是一个自主开发助手。你的任务是：在项目仓库 xiabai2008/quantum-rl-scheduler 中，找到分配给我的、带有 ai-ready 标签的未完成 Issue，按照 Issue 中的"AI执行指引"完成代码实现，通过所有质量检查后提交 PR。

项目背景：量子AI双向赋能调度系统（RL调度+量子退火），Python 3.10+，使用 ruff/mypy/pytest 质量门禁，代码冻结日期 2026-08-15。

权威数字（禁止修改）：PPO=2746.94±1160.72, FCFS=1458.77±60.47, 提升+88.3%, Mann-Whitney U检验 p=1.032e-42, rank-biserial=-0.71；多seed真机 PPO=1736.32 vs FCFS=383.00, d=5.33, p<0.001。

执行步骤：

1. 设置GitHub CLI认证：
   $env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 同步最新代码：
   git checkout main
   git pull origin main
   （如果当前分支有未提交的改动，先 git stash）

3. 查询分配给我的 ai-ready 标签 open Issues：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee qpqpalalzmzm112 --label ai-ready --limit 20 --json number,title,body,updatedAt

4. 如果没有分配给我的 ai-ready Issue，再查未分配的：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --label ai-ready --limit 20 --json number,title,body,updatedAt,assignees
   筛选 assignees 为空的 Issue，认领一个：
   gh issue edit <issue号> -R xiabai2008/quantum-rl-scheduler --add-assignee qpqpalalzmzm112

5. 如果步骤3和4都没有可用 Issue，直接结束（无需操作）。

6. 查询我名下所有 open PR，记录已有关联 Issue 号：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --author qpqpalalzmzm112 --limit 20 --json number,title,body
   对每个 PR 的 body 检查是否包含 "Closes #<issue号>"，记录已被处理的 Issue 号。

7. 从第3步（或第4步认领）的 Issue 列表中，排除第6步已有关联 PR 的 Issue。
   如果全部已有关联 PR，直接结束（无需操作）。

8. 从剩余 Issue 中选择1个优先处理：选择 updatedAt 最早的（最久未更新的）。

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
    pytest tests/ --tb=short -q

13. 如果检查失败：
    a. 分析错误原因，修复代码
    b. 重新运行全部质量检查
    c. 最多重试3轮
    d. 3轮后仍有失败：在 Issue 下评论说明情况，然后清理分支结束
       gh issue comment <issue号> -R xiabai2008/quantum-rl-scheduler --body "@qpqpalalzmzm112 自动执行此Issue时遇到问题，需要人工介入：\n\n**失败步骤**：...\n**错误信息**：...\n**已尝试的修复**：..."
       git checkout main
       git branch -D feature/issue-<issue号>

14. 如果质量检查全部通过：
    git add -A
    git commit -m "feat: <简短描述> (Closes #<issue号>)"
    git push origin feature/issue-<issue号>

15. 创建 Pull Request：
    gh pr create -R xiabai2008/quantum-rl-scheduler --base main --head feature/issue-<issue号> --title "feat: <简短描述>" --body "## 关联Issue\nCloses #<issue号>\n\n## 改动说明\n<描述具体改动内容>\n\n## 验证\n- [x] ruff check 通过\n- [x] ruff format 通过\n- [x] mypy 通过\n- [x] pytest 通过\n\n## 权威数字一致性\n- [x] 未修改 PPO/FCFS 权威数字\n- [x] 未修改实验数据文件" --assignee "qpqpalalzmzm112"

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

cron表达式：30 16 * * *
时区：Asia/Shanghai
任务名称：qpqpalalzmzm112每日自动完成Issue
```

---

## Izzro 的提示词

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建1个定时任务来辅助我的项目开发工作。

=========================================
任务4：每日自动认领并完成 ai-ready Issue
=========================================

每天下午13:30执行。

你是一个自主开发助手。你的任务是：在项目仓库 xiabai2008/quantum-rl-scheduler 中，找到分配给我的、带有 ai-ready 标签的未完成 Issue，按照 Issue 中的"AI执行指引"完成代码实现，通过所有质量检查后提交 PR。

项目背景：量子AI双向赋能调度系统（RL调度+量子退火），Python 3.10+，使用 ruff/mypy/pytest 质量门禁，代码冻结日期 2026-08-15。

权威数字（禁止修改）：PPO=2746.94±1160.72, FCFS=1458.77±60.47, 提升+88.3%, Mann-Whitney U检验 p=1.032e-42, rank-biserial=-0.71；多seed真机 PPO=1736.32 vs FCFS=383.00, d=5.33, p<0.001。

执行步骤：

1. 设置GitHub CLI认证：
   $env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 同步最新代码：
   git checkout main
   git pull origin main
   （如果当前分支有未提交的改动，先 git stash）

3. 查询分配给我的 ai-ready 标签 open Issues：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee Izzro --label ai-ready --limit 20 --json number,title,body,updatedAt

4. 如果没有分配给我的 ai-ready Issue，再查未分配的：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --label ai-ready --limit 20 --json number,title,body,updatedAt,assignees
   筛选 assignees 为空的 Issue，认领一个：
   gh issue edit <issue号> -R xiabai2008/quantum-rl-scheduler --add-assignee Izzro

5. 如果步骤3和4都没有可用 Issue，直接结束（无需操作）。

6. 查询我名下所有 open PR，记录已有关联 Issue 号：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --author Izzro --limit 20 --json number,title,body
   对每个 PR 的 body 检查是否包含 "Closes #<issue号>"，记录已被处理的 Issue 号。

7. 从第3步（或第4步认领）的 Issue 列表中，排除第6步已有关联 PR 的 Issue。
   如果全部已有关联 PR，直接结束（无需操作）。

8. 从剩余 Issue 中选择1个优先处理：选择 updatedAt 最早的（最久未更新的）。

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
    pytest tests/ --tb=short -q

13. 如果检查失败：
    a. 分析错误原因，修复代码
    b. 重新运行全部质量检查
    c. 最多重试3轮
    d. 3轮后仍有失败：在 Issue 下评论说明情况，然后清理分支结束
       gh issue comment <issue号> -R xiabai2008/quantum-rl-scheduler --body "@Izzro 自动执行此Issue时遇到问题，需要人工介入：\n\n**失败步骤**：...\n**错误信息**：...\n**已尝试的修复**：..."
       git checkout main
       git branch -D feature/issue-<issue号>

14. 如果质量检查全部通过：
    git add -A
    git commit -m "feat: <简短描述> (Closes #<issue号>)"
    git push origin feature/issue-<issue号>

15. 创建 Pull Request：
    gh pr create -R xiabai2008/quantum-rl-scheduler --base main --head feature/issue-<issue号> --title "feat: <简短描述>" --body "## 关联Issue\nCloses #<issue号>\n\n## 改动说明\n<描述具体改动内容>\n\n## 验证\n- [x] ruff check 通过\n- [x] ruff format 通过\n- [x] mypy 通过\n- [x] pytest 通过\n\n## 权威数字一致性\n- [x] 未修改 PPO/FCFS 权威数字\n- [x] 未修改实验数据文件" --assignee "Izzro"

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

cron表达式：30 13 * * *
时区：Asia/Shanghai
任务名称：Izzro每日自动完成Issue
```

---

## 第4个任务时间排布表

| 队友 | 第4任务时间 | cron表达式 | 与自身现有任务冲突检查 |
|:--|:--|:--|:--|
| Izzro | 13:30 | `30 13 * * *` | 无冲突（现有：09:30/16:00/16:00） |
| DUMNOX | 14:00 | `0 14 * * *` | 无冲突（现有：10:30/09:00/15:00） |
| heka-ky | 14:30 | `30 14 * * *` | 无冲突（现有：09:30/16:00/10:00） |
| NN2914 | 15:00 | `0 15 * * *` | 无冲突（现有：09:30/09:00/11:00） |
| Jackhock-1 | 15:30 | `30 15 * * *` | 无冲突（现有：09:30/16:00/14:00） |
| qpqpalalzmzm112 | 16:30 | `30 16 * * *` | 无冲突（现有：09:30/09:00/15:00） |

- 6人每日错开30分钟，避免同时占用计算资源
- 每人时间均与自身前3个任务无冲突
- 均在下午工作时段，确保 TRAE 在线运行

## 完整任务总览（4个任务）

| 队友 | 任务1 每日提醒 | 任务2 自动提Issue | 任务3 每周检查 | 任务4 自动完成Issue |
|:--|:--|:--|:--|:--|
| DUMNOX | 10:30 | 周一/四 09:00 | 周日 15:00 | 14:00 |
| heka-ky | 09:30 | 周一/四 16:00 | 周六 10:00 | 14:30 |
| NN2914 | 09:30 | 周二/五 09:00 | 周六 11:00 | 15:00 |
| Jackhock-1 | 09:30 | 周二/五 16:00 | 周六 14:00 | 15:30 |
| qpqpalalzmzm112 | 09:30 | 周三/六 09:00 | 周六 15:00 | 16:30 |
| Izzro | 09:30 | 周三/六 16:00 | 周六 16:00 | 13:30 |

## 工作流程闭环（完整版）

1. **进度提醒**（任务1）：查询我的Issue/PR → 没有PR的提醒开发 → 有review反馈的提醒修复
2. **自动提Issue**（任务2）：TRAE审视项目 → 发现AI可完成的改进点 → 创建带"AI执行指引"的Issue
3. **自动完成Issue**（任务4）：TRAE自动认领ai-ready Issue → 按指引编写代码 → 质量检查通过 → 提交PR ← **新增**
4. **每周自检**（任务3）：ruff/mypy/pytest检查 → 发现问题自动创建Issue → 零提交提醒

> 第4个任务填补了之前工作流的缺口：Issue被创建后无人自动完成。现在形成了"发现 → 创建 → 完成 → 审查"的完整闭环。

## 注意事项

- TRAE 需要工作时间内保持运行，否则定时任务不会执行
- GitHub 邮件通知突然停止 = Token可能过期，重新执行认证命令
- 每天最多自动完成1个 Issue，避免PR洪水
- 只处理 ai-ready 标签的 Issue（非 ai-ready 的不自动处理）
- 不直接 push 到 main，必须通过 PR 流程
- 不修改权威数字和实验数据，确保比赛可信度
- 已知 flaky 测试可忽略，不影响PR提交
- 在TRAE中说"列出我的定时任务"可以管理（暂停/修改/删除）
