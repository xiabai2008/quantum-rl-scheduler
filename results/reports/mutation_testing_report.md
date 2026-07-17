# 变异测试报告 (Mutation Testing Report)

- **日期**: 2026-07-11
- **工具**: mutmut 3.6.0（运行于 WSL，Python 3.14 venv）
- **目标**: 验证项目宣称的 **93.78% 行覆盖率** 的"含金量"——覆盖率只说明代码被执行过，变异测试才说明测试能否抓住行为变更。

---

## 1. 背景

项目此前只统计行覆盖率（仓库内真实值有多个写法：87.38% / 87.5% / 91.4% / 93.78%，最新提交已推到 **93.78%**），但从未做变异测试。变异测试（mutation testing）通过故意篡改源码（如 `+`→`-`、`>`→`>=`、`True`→`False`）生成"变异体"，再跑测试套件：若测试失败说明变异被"杀死"（测试有效），若测试仍通过说明变异"存活"（测试有漏洞）。这是测试质量的试金石。

> 结论先行：**行覆盖率 93.78% ≠ 测试质量高**。三个核心轻逻辑模块的 mutation score 仅 50%–68%，意味着约 1/3 到 1/2 的行为变更测试根本抓不住。

---

## 2. 环境与方法

### 关键工程障碍（已逐一解决）
1. **mutmut 不能在原生 Windows 运行**——官方要求用 WSL。
2. **依赖链**：`src/__init__.py` 在包初始化时急切 `import src.quantum`（torch）+ `src.scheduler`（gym/sb3），导致任何测试都需 torch/gym/sb3。在 WSL 新建 venv 安装 CPU-only torch + gymnasium + stable_baselines3 + sb3_contrib + scipy 解决。
3. **mutmut 3.x 不用 `runner_command`**（已废弃），测试选择靠 `pytest_add_cli_args_test_selection`；且只复制 `source_paths` 下文件到 `mutants/`，故必须按模块缩小范围。
4. **src-layout 不兼容**：mutmut 的 trampoline 机制 `assert not name.startswith("src.")`，而项目用 `from src.utils.seeds import ...` → 模块名 `src.utils.seeds` 直接触发崩溃。
   - **workaround**：临时将对应测试的 import 从 `from src.X import` 改为 `from X import`（并 `sys.path.insert` 指向 `src` 子目录，使模块名变为 `utils.X`/`scheduler.X`/`api.X`），跑完 `git checkout` 恢复。

### 运行命令（单模块逐跑）
```bash
# WSL 内，激活 .venv-mutmut-wsl 后
python -m mutmut run   # 读取 pyproject [tool.mutmut] 的 source_paths + pytest_add_cli_args_test_selection
```

---

## 3. 结果

| 模块 | 变异总数 | killed ✅ | suspicious ⚠️ | survived ❌ | Mutation Score* |
|------|---------|---------|-------------|-----------|----------------|
| `src/utils/seeds.py` | 26 | 13 | 0 | 13 | **50.0%** |
| `src/utils/helpers.py` | 348 | 183 | 57 | 108 | **62.9%**（原始 52.6%） |
| `src/scheduler/parser.py` | 826 | 552 | 10 | 264 | **67.6%**（原始 66.8%） |
| `src/api/circuit_breaker.py` | — | — | — | — | **未跑**（见 §4） |

> \* Mutation Score = killed / (total − suspicious)，排除可疑变异（多为随机性/计时相关）。
> 变异速率约 11–13 mutations/second。

### 逐模块分析
- **seeds.py（最弱，50%）**：13 个存活变异**全部集中在 `set_seed` 函数**（变异点 9–26）。测试只验证了返回值（默认 42、显式参数），未覆盖其分支与算术变更——如环境变量 `QUANTUM_RL_SEED` 覆盖逻辑、边界值、随机性确定性断言。典型的"覆盖到了但没断言行为"。
- **helpers.py（居中，62.9%）**：57 个 suspicious 多为 `MetricsCalculator` 涉及随机性/计时的分支；108 个 survived 需加固，重点在文件 IO 错误路径（`load_config`/`save_json` 的非法/缺失文件）、`normalize_vector` 的常量/负值边界。
- **parser.py（最强，67.6%）**：解析器分支覆盖较全（TaskBuilder 校验、TaskParser 各错误收集、LegacyTaskParser 多格式、QASM 边界），但仍有 264 个存活变异，多为容错/异常路径未断言。

---

## 4. circuit_breaker.py 未跑原因

`circuit_breaker.py` 在**模块顶层** `import src.exceptions` 与 `src.utils.alerts`，且其测试 `test_circuit_breaker.py` 含有 **~50 处** `patch("src.api.circuit_breaker.alert_critical")` / `patch("src.api.circuit_breaker.time.monotonic")` 字符串。在 src-layout workaround 下，需机械修改大量字符串（`src.` → 空前缀）才能让模块名变为 `api.circuit_breaker`，成本高、易错，故本轮跳过。

**后续处理建议（二选一）**：
- **方案 A（快）**：将 `test_circuit_breaker.py` 的 import 与全部 `patch("src.api...")` 字符串批量 `src.` → ``（去掉前缀），临时跑变异后 `git checkout` 恢复。
- **方案 B（治本）**：等项目迁移为非 src-layout（或将 `src/` 作为可安装包 `pip install -e .`），或待 mutmut 修复对 `src.` 前缀模块名的支持。

---

## 5. 结论与行动建议（对接 v2 方案「任务 #1：修复并跑通变异测试」）

1. **核心结论**：93.78% 行覆盖率存在"水分"——三个核心模块 mutation score 仅 50%–68%。覆盖率数字对评委好看，但测试对行为变更的捕获能力不足，属真实的代码质量短板。
2. **优先加固**：
   - `seeds.set_seed`：补充环境变量覆盖、边界值、确定性断言；
   - `helpers` 的 IO 错误路径与 `MetricsCalculator` 各分支；
   - `parser` 的 264 个容错/异常路径。
   - 目标将各模块 mutation score 提升到 **80%+**。
3. **纳入 CI 门禁**：将 mutation score（目标 >80%）与行覆盖率并列作为质量门禁，二者互补——覆盖率防"没测到"，变异测试防"测了但没用"。
4. **诚实披露**：答辩材料中若引用"测试充分"，应辅以 mutation score 而非仅覆盖率，避免被追问时穿帮（呼应《答辩诚信清单》）。

---

*本报告由变异测试实际运行得出，非估算。环境：WSL + Python 3.14 + mutmut 3.6.0 + CPU torch/gym/sb3。配置固化于 `pyproject.toml [tool.mutmut]`，含 src-layout workaround 说明。*
