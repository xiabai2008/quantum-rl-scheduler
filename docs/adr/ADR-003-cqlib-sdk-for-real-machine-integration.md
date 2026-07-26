# ADR-003: 使用 cqlib SDK 进行真机集成

## Status

Accepted

## Context

本项目需要与天衍云量子计算平台（天衍-287，105数据比特+182耦合比特超导量子计算机）集成，以实现真机任务提交、结果查询和机器状态监控。有两种集成路径可选：

### 候选方案

1. **cqlib SDK**：中电信量子提供的 Python SDK，封装了天衍云量子计算平台的底层接口
2. **REST API**：通过 HTTP 请求直接调用天衍云平台的 REST API 端点

### 决策驱动因素

- **WAF 拦截问题**：天衍云平台的 REST API 受 Web Application Firewall（WAF）保护，直接 HTTP 请求被拦截或限速，导致连接不稳定
- **认证复杂度**：REST API 需要手动管理 Token 刷新和会话保持；cqlib SDK 内置认证流程
- **功能完整性**：cqlib SDK 封装了任务提交、结果轮询、机器状态查询、退火任务提交等全套功能
- **官方支持**：cqlib 是中电信量子官方维护的 SDK，与天衍云平台版本同步更新
- **依赖管理**：cqlib 引入额外依赖（qiskit 等），需要与核心调度逻辑解耦

### 实验验证

真机实验验证了 cqlib SDK 的可用性：
- 284 次真机 SDK 调用 100% 成功
- Issue #128 新增 tianyan176 H 门任务成功（P(0)=50.9%, P(1)=49.1%）
- 全链路验证通过：SDK 认证 -> 任务提交 -> 状态轮询 -> 结果获取

## Decision

**采用 cqlib SDK 作为真机集成方案，REST API 方案因 WAF 拦截问题不采用。**

具体决策：

1. 真机集成通过 `src/api/tianyan_cqlib.py`（`TianyanCqlibClient`）实现
2. cqlib 作为可选依赖，通过 `requirements-quantum.txt` 安装，不进入核心依赖
3. 运行时通过 `TIANYAN_MODE` 环境变量控制真机/Mock 模式切换
4. 无 cqlib 时自动降级到 Mock 模式（`src/api/mock_client.py`），保证核心功能可用
5. cqlib SDK 封装在 API 层，与调度引擎解耦，通过 `QuantumAPIClient` 抽象接口交互

### 依赖管理策略

```
requirements.txt           # 核心依赖（含 dimod/dwave-neal 退火模拟）
requirements-quantum.txt   # 真机可选依赖（cqlib）
```

- cqlib 不在 `requirements.txt` 中，避免核心功能依赖重量级量子计算库
- 真机实验环境单独安装 `pip install -r requirements-quantum.txt`

## Consequences

### 正面影响

- 绕过 WAF 拦截问题，真机调用稳定（284 次调用 100% 成功）
- SDK 内置认证和重试机制，降低工程复杂度
- 支持退火任务提交（QUBO 问题求解），与量子退火模块（`src/quantum/annealing.py`）集成
- 官方维护，与平台版本同步

### 负面影响

- cqlib 引入 qiskit 等重量级依赖，安装包体积增大
- 真机功能为可选依赖，CI 环境不安装 cqlib，真机相关测试使用 Mock 模式
- SDK 版本更新可能引入不兼容变更，需关注官方 changelog
- 依赖中电信量子的持续维护，存在供应链风险

### 后续约束

- 真机集成代码必须通过 `QuantumAPIClient` 抽象接口，不可在调度引擎中直接引用 cqlib
- cqlib 版本锁定在 `requirements-quantum.txt` 中，避免自动升级引入不兼容
- 新增真机功能时须同步更新 Mock 实现，保证测试覆盖率
- 真机/Mock 模式切换通过环境变量，不可硬编码

## Alternatives Considered

### 方案 A：REST API 直接调用

- 优势：无额外依赖，实现轻量
- 劣势：WAF 拦截导致连接不稳定；需手动管理 Token 和会话；功能需自行封装
- 结论：不采用。WAF 拦截问题在实验中反复出现，无法可靠解决

### 方案 B：混合方案（REST API 优先 + cqlib 降级）

- 优势：可在无 cqlib 环境下使用 REST API
- 劣势：两套实现增加维护成本；REST API 的 WAF 问题未解决，降级意义有限
- 结论：不采用。维护两套实现不符合工程效率原则

### 方案 C：仅 Mock 模式（不集成真机）

- 优势：零外部依赖，完全自包含
- 劣势：无法验证真机可用性，比赛"双向赋能"叙事缺乏真机证据
- 结论：不采用。真机验证是比赛核心要求之一

---

*ADR 编号: ADR-003 | 创建日期: 2026-07-25 | 决策者: 瑞哥 + 工程组*
