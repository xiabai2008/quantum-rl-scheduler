# 量子RL驱动的天衍云平台智能调度系统

> 2026年度"揭榜挂帅"擂台赛参赛项目  
> 选题编号：XA-202609  
> 发榜单位：中国电信集团有限公司

## 项目简介

本项目面向"量子+AI双向赋能"核心命题，构建基于强化学习（RL）的天衍云平台智能调度系统。

**双向赋能机制：**
- 🤖 AI赋能量子：RL Agent 实时决策任务在量子/经典资源间的最优分流
- ⚛️ 量子赋能AI：利用量子退火算法加速 RL 策略搜索过程

## 项目架构

```
quantum-rl-scheduler/
├── src/                  # 源代码
│   ├── scheduler/        # RL调度引擎核心模块
│   ├── api/             # 天衍云平台API封装
│   ├── quantum/         # 量子计算相关模块
│   ├── visualization/    # Web可视化界面
│   └── utils/          # 工具函数
├── tests/               # 单元测试
├── docs/                # 技术文档
├── config/              # 配置文件
├── scripts/             # 部署脚本
└── notebooks/           # Jupyter实验记录
```

## 核心功能

| 模块 | 功能 | 状态 |
|------|------|------|
| 任务解析器 | 解析量子任务描述，提取特征 | 🚧 开发中 |
| RL决策引擎 | 基于DQN/A3C的智能调度决策 | 🚧 开发中 |
| 天衍API封装 | 任务提交、状态查询、结果获取 | 🚧 开发中 |
| 量子退火加速 | 加速RL策略搜索 | 📋 规划中 |
| Web可视化 | 调度监控界面 | 📋 规划中 |

## 技术栈

| 层级 | 技术选型 |
|-------|----------|
| 编程语言 | Python 3.10+ |
| RL框架 | Stable-Baselines3 |
| 量子模拟 | Qiskit / Pennylane |
| Web框架 | FastAPI（后端）+ Vue3（前端） |
| 数据可视化 | Echarts / Plotly |

## 🚀 快速开始（Mock 模式，无需真实平台）

> **重要**：开发阶段使用 Mock 模式，无需申请天衍云平台权限即可完整开发！

### 1. 克隆仓库

```bash
git clone https://github.com/xiabai2004/quantum-rl-scheduler.git
cd quantum-rl-scheduler
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 无需修改！默认已启用 Mock 模式
# TIANYAN_MOCK_MODE=true
```

### 4. 验证 Mock API

```bash
# 运行 Mock API 测试脚本
python scripts/test_mock_api.py
```

预期输出：
```
✅ MockTianyanClient 导入成功
✅ Mock 客户端创建成功
✅ 认证验证通过（Mock 模式始终返回 True）
✅ 量子任务提交成功
✅ 任务结果获取成功
✅ Mock API 客户端功能完整
```

### 5. 开始开发！

```python
# 示例代码：使用 Mock 客户端开发
from src.api.mock_client import MockTianyanClient

# 创建 Mock 客户端（模拟天衍云平台）
client = MockTianyanClient(mock_delay=0.5)

# 验证认证（Mock 模式始终通过）
if client.authenticate():
    print("✅ 认证通过")

# 提交量子任务（Mock 模式返回虚拟 task_id）
qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""
task_id = client.submit_quantum_task(circuit_qasm=qasm, shots=1024)
print(f"任务提交成功，task_id={task_id}")

# 等待任务完成（Mock 模式自动模拟状态轮转）
result = client.wait_for_task(task_id, poll_interval=0.5, timeout=10.0)
print(f"任务结果: {result}")
```

### 6. Mock 模式配置说明

| 配置项 | 位置 | 说明 |
|--------|------|------|
| `TIANYAN_MOCK_MODE=true` | `.env` 或环境变量 | 启用 Mock 模式（默认开启） |
| `TIANYAN_MOCK_DELAY=1.0` | `.env` 或环境变量 | 模拟网络延迟（秒） |
| `TIANYAN_MOCK_FAILURE_RATE=0.0` | `.env` 或环境变量 | 模拟失败率（0-1） |
| `tianyan.mock_mode: true` | `config/config.yaml` | 配置文件中的 Mock 开关 |

### 7. 切换到真实 API（获得平台权限后）

```bash
# 方法 1：修改 .env
TIANYAN_MOCK_MODE=false
TIANYAN_API_KEY=你的真实API密钥

# 方法 2：修改 config/config.yaml
tianyan:
  mock_mode: false

# 方法 3：代码中显式指定
from src.api.tianyan_client import TianyanClient
client = TianyanClient(mock_mode=False)
```

## 📚 Mock API 功能列表

`MockTianyanClient` 完全模拟天衍云平台 API，支持：

| 方法 | 功能 | 说明 |
|------|------|------|
| `authenticate()` | 认证验证 | Mock 模式始终返回 `True` |
| `submit_quantum_task()` | 提交量子任务 | 返回 `mock-xxxxxxxxxxxx` 格式 ID |
| `get_task_status()` | 查询任务状态 | 自动轮转：PENDING → RUNNING → COMPLETED |
| `get_task_result()` | 获取任务结果 | 返回模拟测量计数 |
| `list_backends()` | 列出可用后端 | 返回 2 个模拟后端（tianyan-287, tianyan-simulator） |
| `get_backend_info()` | 获取后端详情 | 返回模拟的后端信息 |
| `submit_classical_task()` | 提交经典任务 | 立即返回完成状态 |
| `get_queue_status()` | 获取队列状态 | 返回模拟队列统计 |
| `wait_for_task()` | 等待任务完成 | 自动轮询直到完成 |

## 🧪 运行测试

```bash
# 运行 Mock API 测试
python scripts/test_mock_api.py

# 运行单元测试（需先安装 pytest）
pytest tests/ -v
```
| 数据存储 | SQLite（开发）/ Redis（缓存） |

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/your-team/quantum-rl-scheduler.git
cd quantum-rl-scheduler

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp config/.env.example config/.env
# 编辑 config/.env，填入天衍云平台API密钥

# 5. 运行仿真测试
python scripts/run_simulation.py

# 6. 启动Web界面（开发完成后）
python src/visualization/app.py
```

## 开发计划

详见 [docs/开发计划.md](docs/开发计划.md) 和 [Gantt图](docs/Gantt图.png)

| 阶段 | 时间 | 关键交付物 |
|------|------|------------|
| 准备阶段 | W1-W3 | 需求分析、技术选型、架构设计 |
| 核心开发 | W3-W8 | 任务解析器、RL引擎、天衍API集成 |
| 算法实现 | W4-W10 | RL决策引擎优化、量子退火模块 |
| 测试验证 | W10-W12 | 系统测试、真机验证、作品提交 |

## 团队分工

| 角色 | 姓名 | 职责 |
|------|------|------|
| 项目经理 | 待定 | 进度协调、文档撰写 |
| 算法工程师 | 待定 | RL决策引擎、量子退火模块 |
| 后端工程师 | 待定 | 天衍API集成、任务解析器 |
| 前端工程师 | 待定 | Web可视化界面 |
| 测试工程师 | 待定 | 系统测试、性能优化 |

## 验证方案

**验证场景：** 组合优化典型问题（旅行商问题TSP、车辆路径问题VRP）

**对比实验：**
- 基线1：先来先服务（FCFS）调度
- 基线2：经典启发式调度
- 本文方法：RL智能调度

**评估指标：**
- 量子比特日利用率（目标：提升30%+）
- 用户平均等待时间（目标：降低40%+）
- 任务完成率
- 系统吞吐量

## 项目文档

| 文档 | 路径 | 说明 |
|------|------|------|
| 技术方案（简述） | docs/技术方案简述.pdf | 天衍云资源申请用 |
| 技术方案（完整版） | docs/技术方案完整版.pdf | 团队开发参考 |
| 系统架构图 | docs/系统架构图.png | 高清架构图 |
| 算法流程图 | docs/算法流程图.png | RL算法流程 |
| 开发计划 | docs/开发计划.md | 详细开发计划 |
| API接口文档 | docs/API接口文档.md | 天衍云集成规范 |

## 联系我们

- 📧 参赛邮箱：[待填写]
- 🏫 所属院校：[待填写]
- 👨🏫 指导老师：[待填写]
- 🔗 比赛官网：https://2026.tiaozhanbei.net/d51/article/715/

## 许可证

项目知识产权按比赛规则执行（团队与发榜单位共同拥有）  
代码开源协议：[待确定]

---

> ⚛️ 量子+AI双向赋能 · 助力量子计算实用化进程
