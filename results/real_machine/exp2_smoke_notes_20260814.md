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

## 执行结果（2026-08-14 12:08-12:09，真机 tianyan-287）—— 实验② 终止（平台能力边界）
- captain 已批准启动 t2（06cff...08e，ts=1786680360565）。wait_for_task 参数 bug 已由 captain 修复（timeout/poll_interval），脚本正常。
- --check-a：A1_cz_chain_4q (task_id=None, QCIS 预校验失败)、A2_cnot_pair (task_id=None, QCIS 预校验失败)
  → results/real_machine/compilation_real_smoke_20260814_120833.json；预检 A ❌ 失败
- --check-b：B1_near_Q1_Q2 (task_id=None, QCIS 预校验失败)、B2_far_Q1_Q3 (task_id=None, QCIS 预校验失败)
  → results/real_machine/compilation_real_smoke_20260814_120951.json；预检 B ❌ 失败
- 因无 task_id，无需/无法 patch 补录。

### 决定性取证：qcis_check_regular 平台 QCIS 校验（直接探针，12:09）
| 电路 | 有效? |
|---|---|
| A1 cz 链 (H Q1..4, CZ Q1Q2/Q2Q3/Q3Q4, M) | False |
| A2 cnot (H Q1 Q2, CNOT Q1Q2, M) | False |
| A2b CZ Q1Q2 (直连) | False |
| A2c CZ Q1Q2 + H Q2 (circuit_templates H-CZ-H 风格) | False |
| single_H2q (H Q1, H Q2, M) | True |
| single_H1q (H Q1, M) | True |
| RX Q1 0.5 (参数门) | True |

控制隔离：唯一差异是插入 "CZ Q1 Q2" 行（single_H2q True → A2b False），证明拒绝源于 CZ 2量子门本身（CNOT 同理）。

### 结论
tianyan-287 平台 qcis_check_regular（平台 QCIS 校验）**拒绝所有含 CZ/CNOT（2量子纠缠门）电路**；单比特门（H 等）与参数门 RX/RY/RZ 通过。
→ 实验②（编译层 PPO/SABRE 电路 → 真机保真度对比）所需的多比特门电路**无法在本平台执行**，实验② 终止。平台能力边界已记录。

## 复核性复跑（2026-08-14 12:14，wait_for_task 修复后，captain 批准）
- captain 已修复 compilation_real_smoke.py wait_for_task 参数 bug（max_wait_time/sleep_time → timeout/poll_interval，与 CqlibTianyanClient 签名一致，py_compile 通过）。
- --check-a 复跑：A1_cz_chain_4q / A2_cnot_pair 均 status=failed, task_id=None, "QCIS 预校验失败"
  → results/real_machine/compilation_real_smoke_20260814_121408.json（预检 A ❌）
- --check-b 复跑：B1_near_Q1_Q2 / B2_far_Q1_Q3 均 status=failed, task_id=None, "QCIS 预校验失败"
  → results/real_machine/compilation_real_smoke_20260814_121420.json（预检 B ❌）
- 判定：修复 wait_for_task 后结果与初跑（12:08/12:09）一致——失败发生在提交前平台 QCIS 预校验（qcis_check_regular 拒绝 CZ/CNOT），未产生 task_id，非 TypeError/时序/query_error，亦无可补录。
- 确认：平台能力边界结论稳健，实验②终止。相同结论在两次独立运行中复现。
