# 官网免费仿真机使用指引（8.11）

> 平台：天衍量子云（qc.zdxlz.com）｜控制台：`https://qc.zdxlz.com/laboratory/#/computerManage?lang=zh`
> 结论：12 台 free 类型机器（含 5 台仿真机）均可免费使用，但**免费机时有限时额度**；
> 8/11 团队 4 账号额度已被 176 实验耗尽，仿真机提交报"剩余机时不足"。

## 免费机清单（type=free）

| 机器 | 类型 | 说明 |
|:--|:--|:--|
| tianyan_s / sw / tn / tnn / sa / swn | **仿真机** | 官网免费仿真（本报告重点） |
| tianyan176 / 176-2 | 真机（free） | 已大量使用 |
| tianyan-p2000 | 光量子（free） | 九章四号同款 |
| tianyan-ion12 | 离子阱（free） | 12 比特 |
| tianyan24 | 真机（free） | 校准中 |
| supremacy_sample | 演示机（free） | — |

## 额度机制

- 免费机时**限时刷新**（推测按日/时段），耗尽后需等待刷新或网页领取
- SDK 提交与网页控制台共用额度（8/11 实测：SDK 侧全部机时不足）
- 网页控制台可能有独立的"免费试用"入口（未验证，需浏览器登录）

## 等额度刷新后（或网页端）可做的事

### 1. 仿真机跑实验（SDK）
```bash
# 用任一未耗尽账号
python scripts/real_machine/mbs_distribution_expansion.py --target 10 --wait-hours 1
# 或直接提交仿真电路（多比特验证）
python scripts/real_machine/multi_qubit_validation.py
```

### 2. 网页控制台截图留档（审计 P2 补缺）
- 打开 `https://qc.zdxlz.com/laboratory/#/computerManage?lang=zh`
- 截图：①机器列表页（显示 free 仿真机）②任一仿真任务结果页（P(0)/P(1) 分布）
- 打码敏感信息后存 `results/real_machine/platform_screenshots/`
- 详见 `results/reports/platform_screenshot_guide.md`

### 3. 仿真机特性验证（若额度恢复）
- tianyan_s（32 比特仿真）/ sw / tn 的**多比特电路**能力（编译层/噪声建模可用）
- 仿真机保真度分布 → 扩充跨机器噪声分析（287/176 之外第三类噪声源）

## 复测命令（额度恢复后快速验证）
```python
# 任一账号 + tianyan_s 提交 H 门
# SUCCESS P0≈0.5 = 可用；"机时不足" = 额度未恢复
```
