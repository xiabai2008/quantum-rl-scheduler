# Issue #164 真机闭环训练报告

生成时间：2026-07-17 23:17:49 +08:00

## 状态：真机预检阻塞，正式训练未执行

- 已依次加载 dotenv、调用 authenticate，并通过 list_backends 查询后端。
- `tianyan_s` 返回 `running` / `free`。
- cqlib 未暴露平台权威余额接口；仓库本地 QuotaTracker 显示 10,000 shots / 200 tasks，但这不能代表平台余额。
- 1-qubit QCIS（`H Q0` / `M Q0`）、64 shots 最小冒烟被平台拒绝：剩余机时不足；没有 task ID。
- 已在 Issue #164 @xiabai2004 请求补充或确认机时额度。

## 实验结果

纯仿真 PPO 与仿真+真机混合 PPO 的 10,000-step 正式对比没有开始。因此本报告不提供 reward、参与率、收敛曲线或真机成功数，也没有生成对比图。

这次失败提交没有被记录为真实真机成功、Mock 成功或降级成功。实现已设置训练真机提交硬上限 8 次；加冒烟总计最多 9 次、576 shots。额度恢复后可用同一命令重跑并覆盖本报告为正式结果。

结构化预检证据见 `results/real_machine/issue164_closed_loop.json`，其中不含 API Key。
