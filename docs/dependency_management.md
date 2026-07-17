# 依赖管理指南

本文档说明如何管理项目的 Python 依赖。

## 文件说明

- `requirements.txt`: 人类可读的依赖列表，包含版本范围约束
- `requirements.lock`: 精确版本锁定的依赖列表，由 `pip-compile` 生成
- `pyproject.toml`: 项目配置和开发工具配置

## 重新生成锁定文件

当需要更新依赖版本时：

```bash
# 安装 pip-tools
pip install pip-tools

# 重新生成锁定文件
pip-compile requirements.txt -o requirements.lock

# 或者允许不安全包（如 setuptools）
pip-compile requirements.txt -o requirements.lock --allow-unsafe
```

## 添加新依赖

### 方法 1: 直接编辑 requirements.txt

1. 在 `requirements.txt` 中添加依赖（使用版本范围）：
   ```
   new-package>=1.0.0
   ```

2. 重新生成锁定文件：
   ```bash
   pip-compile requirements.txt -o requirements.lock
   ```

3. 提交两个文件的更改

### 方法 2: 使用 pip-compile

```bash
pip-compile --append requirements.txt -o requirements.lock
```

## 更新策略

### 自动化更新

- **Dependabot**: 每月自动检查更新，创建 PR
- **GitHub Actions**: 每周一 09:00 UTC 自动运行依赖检查

### 手动更新

#### 小版本更新（推荐）

```bash
pip-compile --upgrade-package "numpy<1.27" requirements.txt -o requirements.lock
```

#### 主版本更新（谨慎）

1. 修改 `requirements.txt` 中的版本约束
2. 重新生成锁定文件
3. 运行完整测试套件
4. 检查是否有破坏性变更

## PyTorch 与 CUDA 版本冲突

### 问题

PyTorch 的 CUDA 版本可能与系统 CUDA 版本不匹配。

### 解决方案

#### 方案 1: 使用 CPU 版本（推荐用于开发）

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### 方案 2: 指定 CUDA 版本

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### 方案 3: 使用 conda

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 方案 4: 在 requirements.txt 中分离 PyTorch

```txt
# requirements.txt
torch>=2.0.0  # 不锁定具体版本

# requirements.lock
torch==2.2.1+cu118  # 锁定具体版本和 CUDA 版本
```

## 常见问题

### Q: 为什么有两个 requirements 文件？

A: `requirements.txt` 用于人类阅读和维护，包含版本范围；`requirements.lock` 用于精确复现环境，包含所有依赖（包括间接依赖）的精确版本。

### Q: 何时需要重新生成锁定文件？

A:
- 添加新依赖时
- 更新依赖版本时
- 发现安全漏洞时
- CI 测试失败且怀疑是依赖问题时

### Q: 锁定文件会导致冲突吗？

A: 如果团队成员使用不同的 Python 版本或操作系统，可能会有冲突。建议在 CI 中测试多个平台。

### Q: 如何处理可选依赖？

A: 在 `requirements.txt` 中使用注释标记可选依赖组：
```txt
# === 测试依赖（可选） ===
pytest>=7.4.0
```

## CI 集成

CI 使用 `requirements.lock` 确保环境一致性：

```yaml
- name: Install dependencies
  run: |
    pip install -r requirements.lock
```

## 参考资料

- [pip-tools 文档](https://github.com/jazzband/pip-tools)
- [Dependabot 文档](https://docs.github.com/en/code-security/dependabot)
- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
