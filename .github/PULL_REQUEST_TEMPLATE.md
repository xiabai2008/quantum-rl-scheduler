## 概述
<!-- 简要描述这个 PR 做了什么改动，以及为什么 -->


## 关联 Issue
<!-- 链接到相关 Issue，如 Closes #92 -->
Closes #

## 改动类型
<!-- 勾选对应的类型 -->
- [ ] feat: 新功能
- [ ] fix: 修复 Bug
- [ ] docs: 文档
- [ ] test: 测试
- [ ] refactor: 重构
- [ ] chore: 杂项
- [ ] perf: 性能优化

## 改动内容
<!-- 列出主要改动点 -->
-
-
-

## 验证
<!-- 附上测试命令和结果，确保 CI 能通过 -->
```bash
# 测试命令
pytest tests/ --cov=src --cov-fail-under=80
ruff check src/ scripts/ tests/
mypy src/
```

- [ ] 单元测试全部通过
- [ ] 覆盖率 ≥ 80%（与 `pyproject.toml` `fail_under=80` 一致）
- [ ] ruff check 通过
- [ ] mypy 通过

## 检查清单
<!-- 提交前请确认 -->
- [ ] 代码通过 `ruff check src/ scripts/ tests/`
- [ ] 代码通过 `ruff format src/ scripts/ tests/`（已格式化）
- [ ] 测试通过 `pytest tests/ --cov=src --cov-fail-under=80`
- [ ] 添加了必要的测试用例
- [ ] 更新了相关文档（如需要）
- [ ] Commit 格式符合 Conventional Commits（`<type>: <描述>`）
- [ ] 没有提交 `.env`、`models/`、`logs/` 等不该提交的文件

## Review 要点
<!-- 告诉 Reviewer 重点看哪些地方 -->
-
-

## 截图/演示
<!-- 如果涉及 UI 变化，拖拽截图到此处 -->
