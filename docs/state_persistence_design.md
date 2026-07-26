# 任务状态持久化设计文档（MVP 方案）

> **Issue**: #114 — 任务状态持久化设计文档（MVP 方案）
> **项目**: 量子RL驱动的天衍云平台智能调度系统
> **文档类型**: 架构设计文档
> **创建时间**: 2026-07-25
> **状态**: 设计阶段（MVP 路线图，未在 8/15 代码冻结前实现完整持久化）

---

## 0. 文档目的

本文档回答一个关键的工程可信度问题：**"系统宕机后，调度状态能否恢复？"**

当前系统（研究原型阶段）的所有运行时状态均保存在**进程内存**中，进程重启后状态丢失。
本文档设计一套从"内存态"过渡到"持久化态"的 MVP 方案，明确：

1. 当前状态管理架构的边界与局限
2. SQLite 持久化方案（MVP 阶段 1，单机轻量）
3. Redis 持久化方案（MVP 阶段 2，生产级高可用）
4. 分阶段实现路线图
5. 答辩标准回答

> **与现有文档关系**：本文档是 `docs/production_roadmap.md`（生产落地路径）
> 阶段 2「试点部署」中"状态持久化 + Redis 接入"任务的细化设计；
> 也是 `src/visualization/state.py` 模块注释中"生产环境应替换为 Redis 等外部存储"
> 这句话的落地说明。

---

## 1. 当前状态管理架构

### 1.1 状态定义点

可视化层的所有全局可变状态集中定义在 **`src/visualization/state.py`**（Issue #179 重构产物），
这是状态的**唯一定义点**，解决了 `app.py` 与 `routes.py` / `simulator.py` /
`websocket_handler.py` 之间的循环依赖问题。

当前在内存中维护的状态对象：

| 状态对象 | 类型 | 含义 | 容量上限 |
|:--|:--|:--|:--|
| `system_status` | `dict[str, Any]` | 系统综合指标（量子比特利用率、队列长度、平均等待时间、已完成任务数、当前步数、当前策略、真机列表、真机提交记录等） | 单实例 |
| `task_queue` | `list[dict[str, Any]]` | 任务队列（任务ID、用户ID、任务类型、状态、优先级、量子比特数、电路深度、估计时间、到达时间） | 无上限（动态增长） |
| `_resource_history` | `list[dict[str, Any]]` | 资源利用率历史数据点 | `MAX_RESOURCE_HISTORY = 100` |
| `_decision_log` | `list[dict[str, Any]]` | 调度决策日志 | `MAX_DECISION_LOG = 200` |
| `_battle_state` | `dict[str, Any]` | PPO vs FCFS 对战状态（运行标志、步数、双方奖励、历史轨迹、env/obs 引用） | 单实例 |
| `manager` | `ConnectionManager` | WebSocket 连接管理器（连接是瞬态的，无需持久化） | 运行时 |

### 1.2 状态访问模式

`state.py` 提供了两类访问器：

- **线程安全访问器**（`get_system_status` / `update_system_status` / `append_task` /
  `append_resource_history` / `append_decision_log` 等）：基于 `threading.RLock`，
  新代码优先使用。
- **引用访问器**（`get_system_status_ref` / `get_task_queue_ref` / `get_battle_state_ref`）：
  返回对象引用，用于需要原地修改的场景。

状态更新由 `src/visualization/simulator.py` 中的后台 asyncio 任务
`simulate_scheduler()` 驱动：每 3 秒一个 tick，通过 PPO 模型推理执行 `env.step(action)`，
将真实调度指标写回 `system_status` / `task_queue` / `_resource_history` / `_decision_log`。

### 1.3 当前架构的局限（为什么需要持久化）

| 局限 | 影响 | 严重程度 |
|:--|:--|:--|
| **进程重启即丢失全部状态** | 宕机/发版后任务队列清空，进行中的 episode 中断无法续跑 | 高（生产不可接受） |
| **内存历史数据有上限** | `_resource_history` 仅保留 100 条、`_decision_log` 仅 200 条，超限自动裁剪 | 中（监控盲区） |
| **单实例无法水平扩展** | 多 worker 部署时各 worker 状态不一致 | 高（阻碍横向扩容） |
| **对战状态含运行时对象引用** | `_battle_state` 持有 `ppo_env`/`fcfs_env` 等 Python 对象引用，无法直接序列化 | 中（需特殊处理） |

> **结论**：当前架构适用于**研究原型 / 单机演示**，不满足生产环境对**宕机恢复**
> 和**水平扩展**的要求。生产部署必须引入外部持久化存储。

---

## 2. SQLite 持久化方案设计（MVP 阶段 1）

### 2.1 设计目标

- **零外部依赖**：SQLite 是 Python 标准库（`sqlite3`），无需额外服务，适合单机试点
- **WAL 模式**：启用 Write-Ahead Logging，支持读写并发，避免读写互斥
- **增量持久化**：状态变更时异步写入，不阻塞主调度循环
- **崩溃一致性**：利用事务保证状态一致性，宕机后可恢复到最后一次提交点

### 2.2 表结构设计

数据库文件路径：`data/scheduler_state.db`（可通过环境变量 `SCHEDULER_STATE_DB` 覆盖）

```sql
-- ============================================================
-- 表 1: system_status（系统综合指标，单行表）
-- ============================================================
CREATE TABLE IF NOT EXISTS system_status (
    id              INTEGER PRIMARY KEY CHECK (id = 1),  -- 单行约束
    qubit_utilization   REAL    NOT NULL DEFAULT 0.0,
    queue_length        INTEGER NOT NULL DEFAULT 0,
    average_wait_time   REAL    NOT NULL DEFAULT 0.0,
    completed_tasks     INTEGER NOT NULL DEFAULT 0,
    current_step        INTEGER NOT NULL DEFAULT 0,
    current_strategy    TEXT    NOT NULL DEFAULT 'PPO',
    real_machines       TEXT    NOT NULL DEFAULT '[]',   -- JSON 数组
    real_submissions    TEXT    NOT NULL DEFAULT '[]',   -- JSON 数组
    last_update         TEXT    NOT NULL                 -- ISO8601 时间戳
);

-- ============================================================
-- 表 2: task_queue（任务队列）
-- ============================================================
CREATE TABLE IF NOT EXISTS task_queue (
    task_id         TEXT    PRIMARY KEY,
    user_id         TEXT    NOT NULL,
    task_type       TEXT    NOT NULL,            -- quantum / classical / hybrid
    status          TEXT    NOT NULL,            -- pending / running / completed / failed
    priority        INTEGER NOT NULL,
    qubit_count     INTEGER NOT NULL DEFAULT 0,
    circuit_depth   INTEGER NOT NULL DEFAULT 0,
    estimated_time  REAL    NOT NULL DEFAULT 0.0,
    arrival_time    TEXT    NOT NULL,            -- ISO8601
    updated_at      TEXT    NOT NULL             -- ISO8601，用于恢复时排序
);
CREATE INDEX IF NOT EXISTS idx_task_status ON task_queue(status);
CREATE INDEX IF NOT EXISTS idx_task_priority ON task_queue(priority DESC);

-- ============================================================
-- 表 3: resource_history（资源利用率历史）
-- ============================================================
CREATE TABLE IF NOT EXISTS resource_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    payload         TEXT    NOT NULL             -- JSON：完整数据点
);
CREATE INDEX IF NOT EXISTS idx_resource_ts ON resource_history(timestamp);

-- ============================================================
-- 表 4: decision_log（调度决策日志）
-- ============================================================
CREATE TABLE IF NOT EXISTS decision_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    step            INTEGER NOT NULL,
    strategy        TEXT    NOT NULL,
    action          INTEGER NOT NULL,
    payload         TEXT    NOT NULL             -- JSON：完整决策记录
);
CREATE INDEX IF NOT EXISTS idx_decision_ts ON decision_log(timestamp);

-- ============================================================
-- 表 5: episode_checkpoint（episode 检查点，支持续跑）
-- ============================================================
CREATE TABLE IF NOT EXISTS episode_checkpoint (
    episode_id      TEXT    PRIMARY KEY,         -- 如 "ppo_main" / "battle_ppo"
    strategy        TEXT    NOT NULL,
    step            INTEGER NOT NULL,
    cumulative_reward REAL  NOT NULL,
    obs_vector      TEXT    NOT NULL,            -- JSON：观测向量（续跑用）
    env_state       TEXT    NOT NULL,            -- JSON：可序列化的环境状态
    updated_at      TEXT    NOT NULL
);

-- ============================================================
-- 表 6: persistence_meta（持久化元数据，幂等控制）
-- ============================================================
CREATE TABLE IF NOT EXISTS persistence_meta (
    key             TEXT    PRIMARY KEY,
    value           TEXT    NOT NULL
);
```

### 2.3 序列化策略

| 字段类型 | 序列化方式 | 说明 |
|:--|:--|:--|
| 标量（int/float/str） | 原生 SQLite 类型 | 直接存取，无开销 |
| `real_machines` / `real_submissions` | `json.dumps()` → TEXT | 列表内元素为简单 dict |
| `resource_history` / `decision_log` 的 payload | `json.dumps()` → TEXT | 完整数据点 JSON |
| `obs_vector`（numpy ndarray） | `json.dumps(vec.tolist())` → TEXT | 14 维观测向量 |
| `env_state`（环境状态） | 自定义 `to_dict()` / `from_dict()` | 仅序列化可恢复字段（队列、计数器、rng 状态） |

**不可序列化对象的处理**：
- `_battle_state` 中的 `ppo_env` / `fcfs_env` / `ppo_obs` / `fcfs_obs` 是运行时 Python 对象，
  持久化时只保存可恢复的标量字段（`running` / `step` / `ppo_reward` / `fcfs_reward` /
  `ppo_history` / `fcfs_history`），env 对象在恢复时**重建**（基于 `episode_checkpoint`）。
- WebSocket `ConnectionManager` 不持久化（连接是瞬态的，恢复后客户端重连）。

### 2.4 写入策略

采用 **Write-Behind（异步批量写入）** 模式，避免阻塞主调度循环：

```
主调度循环（asyncio）
   │
   ├── 1. 更新内存状态（state.py 访问器）
   │
   └── 2. 投递变更事件到异步队列（asyncio.Queue）
            │
            └── 后台持久化协程（单消费者）
                   ├── 批量取出事件（最多 50 条或 200ms 超时）
                   ├── 单事务批量写入 SQLite（WAL 模式）
                   └── 提交事务
```

**写入频率**：跟随 simulator tick（每 3 秒），关键事件（任务状态变更、episode 边界）立即写入。

**WAL 模式配置**：
```python
conn = sqlite3.connect("data/scheduler_state.db", isolation_level=None)
conn.execute("PRAGMA journal_mode=WAL")       # 写前日志，读写并发
conn.execute("PRAGMA synchronous=NORMAL")     # 平衡安全与性能
conn.execute("PRAGMA wal_autocheckpoint=1000") # 每 1000 页自动 checkpoint
```

### 2.5 恢复流程

系统启动时执行 `restore_state()`：

```
1. 打开 SQLite 连接，启用 WAL 模式
2. 读取 system_status 单行表 → 覆盖内存 system_status
3. 读取 task_queue WHERE status IN ('pending','running') → 重建 task_queue
4. 读取 resource_history 最近 MAX_RESOURCE_HISTORY 条 → 重建 _resource_history
5. 读取 decision_log 最近 MAX_DECISION_LOG 条 → 重建 _decision_log
6. 读取 episode_checkpoint → 重建未完成 episode 的 env 与 obs
   ├── 若 checkpoint 存在且未结束：env = from_dict(env_state); obs = obs_vector
   └── 否则：env.reset()，开启新 episode
7. 记录恢复日志，启动后台持久化协程
8. 启动 simulator 后台任务，从恢复点继续调度
```

**恢复语义**：
- 任务状态为 `running` 的任务，恢复时回退为 `pending`（无法确定宕机时是否真正在跑），重新入队
- episode 检查点存在则续跑，否则从头开始
- 资源历史/决策日志按时间戳补齐，保证监控连续性

---

## 3. Redis 持久化方案设计（MVP 阶段 2）

### 3.1 设计目标

- **高可用**：Redis 主从 + Sentinel，单点故障自动切换
- **水平扩展**：多 worker 共享状态，支持 K8s 多副本部署
- **低延迟**：内存数据库，亚毫秒级读写，适合高频状态更新
- **事件溯源**：保留状态变更历史，支持审计与回放

### 3.2 数据结构设计

Redis Key 命名规范：`scheduler:{namespace}:{key}`

| Redis Key | 数据结构 | 含义 | TTL |
|:--|:--|:--|:--|
| `scheduler:status:main` | Hash | 系统综合指标（字段对应 system_status 各键） | 无（持久） |
| `scheduler:tasks:queue` | Sorted Set | 任务队列，score=优先级×1e10+到达时间戳，member=task_id | 无 |
| `scheduler:tasks:{task_id}` | Hash | 单个任务详情 | 任务完成后 24h |
| `scheduler:history:resource` | List（LPUSH+LTRIM） | 资源利用率历史 | 保留最新 1000 条 |
| `scheduler:history:decisions` | List（LPUSH+LTRIM） | 决策日志 | 保留最新 2000 条 |
| `scheduler:battle:state` | Hash | 对战状态标量字段 | 无 |
| `scheduler:episode:{id}` | Hash + String | episode 检查点（obs 存 String） | episode 结束后 1h |
| `scheduler:lock:{resource}` | String（NX EX） | 分布式锁（多 worker 协调） | 30s 自动释放 |

**任务队列设计要点**：
- 使用 Sorted Set 而非 List，支持按优先级 + 到达时间排序，天然适配 FCFS/SJF/PPO 决策
- `score = priority * 1e10 + unix_timestamp`，保证高优先级先出、同优先级 FIFO
- 任务详情独立存 Hash，避免大 member 拖慢 ZSet 操作

### 3.3 TTL 策略

| 数据类别 | TTL 策略 | 理由 |
|:--|:--|:--|
| 系统状态 / 任务队列 / 对战状态 | 无 TTL（持久） | 核心运行时状态，需长期保留 |
| 已完成任务详情 | 完成后 24h | 兼顾审计与存储成本 |
| 资源历史 / 决策日志 | 通过 LTRIM 保留固定条数 | 时序数据，仅近期有价值 |
| episode 检查点 | episode 结束后 1h | 防止残留检查点干扰新 episode |
| 分布式锁 | 30s（EX） | 防止持有者宕机导致死锁 |

### 3.4 事件溯源（Event Sourcing）

在阶段 2 引入**事件日志流**，记录所有状态变更事件，支持审计与回放：

```
事件流 Key: scheduler:events:{YYYYMMDD}  (Redis Stream)
事件格式（XADD）:
  ts:        1690287123.456
  event:     TASK_ASSIGNED | TASK_COMPLETED | STRATEGY_SWITCHED | EPISODE_RESET
  actor:     ppo_agent | simulator | api
  payload:   <JSON>
```

**事件类型**：

| 事件 | 触发场景 | 携带数据 |
|:--|:--|:--|
| `TASK_ARRIVED` | 新任务到达 | task_id, user_id, type, priority |
| `TASK_ASSIGNED` | 调度器分配任务到后端 | task_id, strategy, action, machine |
| `TASK_COMPLETED` | 任务执行完成 | task_id, reward, latency |
| `STRATEGY_SWITCHED` | 切换调度策略 | old_strategy, new_strategy |
| `EPISODE_RESET` | episode 结束重置 | episode_id, total_reward, steps |
| `STATE_RESTORED` | 系统从持久化恢复 | restore_point_ts, recovered_items |

**消费者组**：使用 `XGROUP CREATE` 创建消费者组，支持多个 worker 并行消费事件、
审计服务独立消费、以及状态回放（按时间点重建任意时刻状态）。

### 3.5 缓存击穿/雪崩防护

- **缓存击穿**：热点 Key（如 `scheduler:status:main`）使用 `SETNX` 加逻辑过期，避免同时回源
- **缓存雪崩**：TTL 添加随机抖动（±10%），防止大量 Key 同时过期
- **降级策略**：Redis 不可用时降级到内存态（当前架构），保证调度不中断（与 `circuit_breaker.py` 复用熔断模式）

---

## 4. MVP 实现路线图

### 阶段 1：SQLite WAL 模式（试点部署，2026 年 8-10 月）

**目标**：单机部署支持宕机恢复，验证持久化语义正确性。

| 任务 | 产出 | 工作量 |
|:--|:--|:--|
| 新建 `src/visualization/persistence_sqlite.py` | SQLite 持久化适配器（实现统一 `StateBackend` 协议） | 2 人日 |
| 定义 `StateBackend` 抽象协议 | `save_status` / `load_status` / `save_task` / `load_tasks` / `save_checkpoint` 等 | 0.5 人日 |
| 改造 `state.py` 访问器 | 写操作同时投递到持久化队列（开关：`PERSISTENCE_ENABLED` 环境变量） | 1 人日 |
| 实现 `restore_state()` 启动恢复 | 读取 SQLite 重建内存状态 | 1 人日 |
| `env.py` 增加 `to_dict()` / `from_dict()` | 环境状态序列化（队列、计数器、rng 种子） | 1.5 人日 |
| 测试：宕机恢复集成测试 | 模拟进程重启，验证状态一致性 | 2 人日 |
| 文档与配置 | `.env.example` 增加 `SCHEDULER_STATE_DB`、`PERSISTENCE_ENABLED` | 0.5 人日 |

**验收标准**：
1. kill 进程后重启，任务队列与系统指标恢复到宕机前最后一次提交点
2. 持久化开销 < 5%（对比无持久化基线的 tick 处理延迟）
3. 恢复测试覆盖率 ≥ 90%

### 阶段 2：Redis 集成（生产部署，2026 年 10 月-2027 年 1 月）

**目标**：多 worker 水平扩展，高可用，事件溯源审计。

| 任务 | 产出 | 工作量 |
|:--|:--|:--|
| 新建 `src/visualization/persistence_redis.py` | Redis 持久化适配器（实现同一 `StateBackend` 协议） | 3 人日 |
| 引入 `redis-py`（async）依赖 | `requirements.txt` 增加 `redis>=5.0` | 0.5 人日 |
| 分布式锁实现 | 多 worker 并发调度协调 | 1.5 人日 |
| 事件流（Redis Stream） | 事件溯源 + 消费者组 | 2 人日 |
| Redis 不可用降级 | 复用 `circuit_breaker.py`，降级到内存态或 SQLite | 1 人日 |
| K8s 部署配置 | Redis StatefulSet + Sentinel，见 `docs/deployment.md` | 2 人日 |
| 压测：多 worker 一致性 | 4 worker × 1000 tasks，验证无丢失/重复 | 2 人日 |

**验收标准**：
1. 4 worker 并发调度，任务无丢失、无重复分配
2. Redis 主节点故障，Sentinel 自动切换，调度无中断（< 3s 切换时间）
3. 事件流可回放重建任意时间点状态

### 路线图时间轴

```
2026-08 ─── 代码冻结（当前，研究原型，纯内存态）
   │
2026-09 ─── 终审提交
   │
2026-10 ─── 阶段1启动：SQLite WAL 持久化
   │           └── 宕机恢复 MVP
2026-12 ─── 阶段1验收：单机持久化稳定
   │
2027-01 ─── 阶段2启动：Redis 集成
   │           └── 多 worker + 事件溯源
2027-03 ─── 阶段2验收：生产级高可用
```

> 与 `docs/production_roadmap.md` 对应：阶段 1 对应「试点部署」，
> 阶段 2 对应「生产部署」。

---

## 5. 架构示意

### 5.1 当前架构（研究原型）

```
┌─────────────────────────────────────────┐
│            单进程 (FastAPI + asyncio)            │
│  ┌─────────────────────────────────────┐ │
│  │      state.py (内存全局状态)        │ │
│  │  system_status / task_queue / ...   │ │
│  └──────────────┬──────────────────────┘ │
│                 │ 读写                      │
│  ┌──────────────▼──────────────────────┐ │
│  │   simulator.py (后台调度协程)       │ │
│  │   PPO 推理 → env.step → 更新状态    │ │
│  └─────────────────────────────────────┘ │
│  ※ 进程重启 → 状态全部丢失                │
└─────────────────────────────────────────┘
```

### 5.2 阶段 1 架构（SQLite WAL）

```
┌─────────────────────────────────────────┐
│            单进程 (FastAPI + asyncio)            │
│  ┌─────────────────────────────────────┐ │
│  │      state.py (内存全局状态)        │ │
│  └──────────────┬──────────────────────┘ │
│                 │ 异步投递变更事件          │
│  ┌──────────────▼──────────────────────┐ │
│  │  persistence_sqlite.py (后台协程)   │ │
│  └──────────────┬──────────────────────┘ │
└─────────────────┼───────────────────────┘
                  │ WAL 写入
        ┌─────────▼─────────┐
        │  scheduler_state.db │  ← 宕机后据此恢复
        └───────────────────┘
```

### 5.3 阶段 2 架构（Redis）

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Worker 1    │  │  Worker 2    │  │  Worker N    │   (K8s 多副本)
│  state.py    │  │  state.py    │  │  state.py    │
│  + Redis适配 │  │  + Redis适配 │  │  + Redis适配 │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
              ┌──────────▼──────────┐
              │   Redis (主从+Sentinel)   │
              │  Hash/ZSet/Stream   │
              │  + 事件溯源日志流    │
              └─────────────────────┘
```

---

## 6. 答辩标准回答

> **评委提问**："系统宕机后调度状态怎么办？任务会不会丢？"

**标准回答**：

当前系统定位为**研究原型**，运行时状态保存在进程内存中
（`src/visualization/state.py`），进程重启后状态会丢失——这在原型验证阶段是可接受的，
因为我们聚焦于验证 RL 调度算法的有效性（PPO vs FCFS +88.3%，N=250，p=1.032e-42）。

对于生产环境，我们已设计完整的**状态持久化 MVP 方案**（见 `docs/state_persistence_design.md`），
分两阶段落地：

1. **阶段 1（SQLite WAL 模式）**：零外部依赖，单机宕机恢复。任务队列、系统指标、
   episode 检查点持久化到 SQLite，进程重启后从最后一次提交点恢复，运行中的任务回退为
   pending 重新入队，未完成 episode 基于检查点续跑。

2. **阶段 2（Redis 集成）**：支持多 worker 水平扩展与高可用。任务队列用 Sorted Set
   按优先级排序，引入事件溯源（Redis Stream）记录所有状态变更，支持审计与任意时间点
   状态回放；Redis 故障时降级到内存态保证调度不中断。

这套方案与我们的生产落地路径（`docs/production_roadmap.md`）一致：竞赛交付后，
试点部署阶段实现 SQLite 持久化，生产部署阶段接入 Redis。**持久化的工程框架已设计完成，
不是技术风险，而是工程进度问题。**

---

## 7. 相关文件索引

| 文件 | 作用 |
|:--|:--|
| `src/visualization/state.py` | 当前内存状态定义点（唯一定义点，Issue #179） |
| `src/visualization/simulator.py` | 后台调度协程，驱动状态更新 |
| `src/visualization/app.py` | FastAPI 入口，从 state.py 再导出状态 |
| `src/api/circuit_breaker.py` | 熔断器（阶段 2 Redis 降级可复用） |
| `docs/production_roadmap.md` | 生产落地路径（阶段 2/3 对应本文档阶段 1/2） |
| `docs/deployment.md` | 部署架构（K8s + Redis StatefulSet 配置） |
| `docs/technical_bottlenecks.md` | 技术瓶颈分析（含状态持久化瓶颈） |

---

*本文档为 Issue #114 设计产出，对应 MVP 路线图，未在 8/15 代码冻结前实现完整持久化。*
*版本：v1.0 | 创建时间：2026-07-25*
