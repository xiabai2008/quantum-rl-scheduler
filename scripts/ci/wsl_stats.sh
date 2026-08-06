#!/bin/bash
cd "$HOME/qrl"
.venv/bin/python -c "
import json, statistics as st
d = json.load(open('results/noise_feedback_v2/noise_feedback_v2_results.json', encoding='utf-8'))
print('seeds:', len(d.get('seeds',[])), '| timesteps:', d.get('train_timesteps'))
res = d['results']
for cond in ['standard','noise']:
    r = res[cond]
    means = [x['mean_reward'] for x in r]
    succ = [x['success_rate'] for x in r]
    print(f'{cond}: n={len(r)} mean={st.mean(means):.1f} std={st.stdev(means):.1f} 成功率={st.mean(succ)*100:.1f}%')
s = d.get('statistics',{})
rc = s.get('reward_comparison',{})
print()
print('Mann-Whitney p:', rc.get('p_value'), '| Cliff δ:', rc.get('cliffs_delta'), '| 显著:', rc.get('significant_005'))
sc = s.get('success_rate_comparison',{})
print('成功率 p:', sc.get('p_value'), '| 显著:', sc.get('significant_005'))
"
