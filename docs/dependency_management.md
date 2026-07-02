# 依赖管理指南

本文档说明如何管理项目的 Python 依赖，包括版本锁定、更新策略和冲突解决。

## 文件说明

| 文件 | 用途 |
|------|------|
| `requirements.txt` | 人类可读的依赖声明，使用版本范围（如 `numpy>=1.24.0`） |
| `requirements.lock` | 机器可读的精确版本锁定，由 `pip-compile` 自动生成 |
| `.github/dependabot.yml` | Dependabot 配置，自动检测依赖更新并创建 PR |
| `.github/workflows/dependency-update.yml` | 每周自动检查依赖更新的 GitHub Actions 工作流 |

## 重新生成锁定文件

当修改 `requirements.txt` 后，需要重新生成锁定文件：

```bash
# 安装 pip-tools（如果尚未安装）
pip install pip-tools

# 生成锁定文件
pip-compile requirements.txt -o requirements.lock --allow-unsafe

# 安装锁定版本
pip install -r requirements.lock
```

**注意**：`--allow-unsafe` 参数允许锁定 setuptools、pkg_resources 等"不安全"包。

## 添加新依赖

### 方法 1：直接编辑 requirements.txt（推荐）

1. 在 `requirements.txt` 的对应分类下添加依赖：
   ```
   # === 核心依赖 ===
   numpy>=1.24.0
   pandas>=2.0.0
   scipy>=1.10.0
   your-new-package>=1.0.0  # 新增
   ```

2. 重新生成锁定文件：
   ```bash
   pip-compile requirements.txt -o requirements.lock --allow-unsafe
   ```

3. 安装新依赖：
   ```bash
   pip install -r requirements.lock
   ```

4. 提交更改：
   ```bash
   git add requirements.txt requirements.lock
   git commit -m "deps: add your-new-package for feature X"
   ```

### 方法 2：使用 pip-compile 的 --generate-hashes 选项（可选）

如果需要增强安全性，可以生成哈希值：

```bash
pip-compile requirements.txt -o requirements.lock --generate-hashes --allow-unsafe
```

**注意**：启用哈希后，所有依赖（包括间接依赖）都必须有哈希值，会增加维护成本。

## 更新策略

### 自动化更新

项目配置了两层自动化更新机制：

#### 1. Dependabot（每月）

- **频率**：每月检查一次
- **范围**：`requirements.txt` 和 `requirements.lock`
- **行为**：自动创建 PR 更新依赖
- **限制**：每次最多 3 个 PR
- **忽略规则**：
  - `numpy`：忽略主版本更新（避免破坏性变更）
  - `black`：忽略主版本更新

#### 2. GitHub Actions 工作流（每周）

- **频率**：每周一 09:00 UTC
- **行为**：
  - 检查过期包并生成报告
  - 尝试生成新的锁定文件
  - 上传结果作为 Artifact
- **手动触发**：可在 GitHub Actions 页面手动运行

### 手动更新流程

#### 小版本更新（安全补丁）

```bash
# 重新编译锁定文件（自动获取兼容的最新版本）
pip-compile requirements.txt -o requirements.lock --allow-unsafe --upgrade

# 测试
pytest tests/

# 提交
git add requirements.lock
git commit -m "deps: update dependencies for security patches"
```

#### 主版本更新（谨慎操作）

1. 修改 `requirements.txt` 中的版本范围：
   ```
   numpy>=1.24.0  →  numpy>=2.0.0
   ```

2. 重新编译锁定文件：
   ```bash
   pip-compile requirements.txt -o requirements.lock --allow-unsafe --upgrade
   ```

3. **全面测试**：
   ```bash
   pytest tests/ --cov=src --cov-fail-under=60
   mypy src/
   ruff check src/
   ```

4. 检查 breaking changes 并修复代码

5. 提交 PR 并等待 review

### 季度人工审查

每季度（3 个月）进行一次人工审查：

- [ ] 检查 `outdated_packages.txt`（从 GitHub Actions Artifact 下载）
- [ ] 评估主版本更新的必要性
- [ ] 移除未使用的依赖
- [ ] 更新 `docs/dependency_management.md`

## PyTorch 与 CUDA 版本冲突解决

PyTorch 对 CUDA 版本有严格要求，容易出现冲突。

### 问题场景

```
ERROR: Could not find a version that satisfies the requirement torch==2.2.1+cu121
```

### 解决方案

#### 方案 1：使用官方 PyTorch 索引

```bash
# CPU 版本
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
```

#### 方案 2：在 requirements.txt 中指定索引

```txt
# === 强化学习 ===
--extra-index-url https://download.pytorch.org/whl/cu121
torch>=2.0.0
stable-baselines3>=2.0.0
```

**注意**：这会影响所有后续依赖的解析。

#### 方案 3：分离 PyTorch 安装

1. 先手动安装 PyTorch：
   ```bash
   pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
   ```

2. 再安装其他依赖：
   ```bash
   pip install -r requirements.lock
   ```

#### 方案 4：使用 pip-compile 的 --find-links

```bash
pip-compile requirements.txt -o requirements.lock \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --allow-unsafe
```

### 验证 CUDA 可用性

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## 常见问题

### Q: 为什么使用 pip-compile 而不是 pip freeze？

**A**：`pip freeze` 会列出所有已安装包的精确版本（包括间接依赖），而 `pip-compile` 只锁定 `requirements.txt` 中声明的依赖及其直接依赖，更清晰且易于维护。

### Q: 锁定文件冲突怎么办？

**A**：
1. 接受上游的 `requirements.lock` 更改
2. 重新运行 `pip-compile requirements.txt -o requirements.lock --allow-unsafe`
3. 测试并提交

### Q: 如何只更新某个特定包？

**A**：
```bash
pip-compile requirements.txt -o requirements.lock --allow-unsafe --upgrade-package numpy
```

### Q: 间接依赖有安全漏洞怎么办？

**A**：
1. 检查是否有新版本的间接依赖修复了漏洞
2. 运行 `pip-compile --upgrade` 重新编译
3. 如果没有修复，考虑在 `requirements.txt` 中显式声明该间接依赖以强制升级

## 参考资源

- [pip-tools 文档](https://github.com/jazzband/pip-tools)
- [Dependabot 文档](https://docs.github.com/en/code-security/dependabot)
- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
- [Python 依赖管理最佳实践](https://docs.python.org/3/tutorial/venv.html)
