# exp2 (engineer-exp2) — 实验②编译层 smoke 执行前分析笔记 (2026-08-14)

## 任务
t2: 实验②smoke预检A/B（真机4任务）。命令：
- `--check-a` (A1_cz_chain_4q, A2_cnot_pair)
- `--check-b` (B1_near_Q1_Q2, B2_far_Q1_Q3)
等待 t1 (engineer-exp1 成本评估) 完成后启动（串行纪律，避免队列干扰）。

## 脚本/环境确认（已完成，不耗真机）
- Python 3.11.9, compilation_real_smoke.py --help 正常（qiskit/cqlib/compilation_env/dotenv imports OK）
- check_a 判定：all(status==completed 且 probability 非空) —— A 通过标准 2/2 completed + probability 非空
- check_b 判定：all(status==completed)
- 输出 JSON 到 results/real_machine/compilation_real_smoke_<ts>.json

## 关键发现 1：平台时序（8/14 实测，exp1 数据佐证）
- 平台对执行的单比特 H 任务约 5 分钟完成；完成前 get_task_status 返回 query_error
  （"Failed to query the experimental result."），exp1 smoke_patched 示例：
  task 2088104706111127554 初判 error → 复查后 completed，probability {0:0.335, 1:0.665}, MBS 0.6706
- CqlibTianyanClient.wait_for_task (tianyan_cqlib.py:712)：连续 3 次 query_error 即返回 status="error"（Issue #407/#719）
  → 对长时间运行的任务会误判失败。→ 需按 captain 规则用 patch_query_error_results.py 按 task_id 复查。
- get_task_status (tianyan_cqlib.py:597)：终态失败特征 "运行失败"/"run failure"/"tasks have failed" → status="error"（不可补录）

## 关键发现 2：smoke 脚本 wait_for_task 参数签名不匹配（潜在 bug，只记录不改）
- compilation_real_smoke.py submit_and_poll:
  `client.wait_for_task(task_id, max_wait_time=TASK_TIMEOUT_SECONDS, sleep_time=TASK_POLL_INTERVAL)`
  但 CqlibTianyanClient.wait_for_task 签名为 `(task_id, timeout=300, poll_interval=5)`
  → 传入 max_wait_time/sleep_time 关键字会抛 TypeError（被 submit_and_poll 的 except 捕获，
  status→"timeout"，但 task_id 已保留）。
- 不修改 src/ 或 smoke 脚本（纪律）。按 captain 流程：task_id 存在则用 patch_query_error_results.py 复查。

## 关键发现 3：patch_query_error_results.py 对 smoke 复合 JSON 的结构匹配
- 生成的 smoke JSON 顶层键为 check_a/check_b（列表），无顶层 task_id，也无 "results" 键
- patcher 匹配分支：`data["results"]` / `data.get("task_id")` / `data["result"].task_id`
  → 对复合 smoke JSON 直接跑会"✅ 无待复查记录"
- 对策：把 check_a/check_b 中 status!=completed 且有 task_id 的单条记录抽成单条 JSON
  （顶层 task_id 结构），逐个跑 patcher，再合并回主 JSON 语义。

## 团队变更（2026-08-14 电时段化）
- 团队 real-machine-upgrade-v4 已归档 → archive/real-machine-upgrade-v4
- 新团队 real-machine-upgrade-v5（.agent-teams/real-machine-upgrade-v5/team.json）：
  - t1 实验①成本评估 Q1修复后重跑 = in_progress (engineer-exp1)
  - t2 实验②smoke预检A/B = claimed (engineer-exp2)
- 关键：Q0 电路问题已修复，env 自动提交电路改 Q1 起（tianyan-287 无 Q0）。
  我的 smoke 电路用 Q1..Q4（物理比特 Q1 起），与新 env 一致。
- 保留 v4 全部发现（平台时序/patcher 结构匹配/wait_for_task 签名）。

## 执行前需确认（v5 口径）
- v5 team.json 中 t1 == completed（等待中，exp1 正在跑成本评估）
- 若 A 失败（平台拒绝多比特/CNOT）→ 实验②终止，记录平台能力边界，t2 仍标 completed（带终止结论）
