import { describe, it, expect } from 'vitest'
import type { DecisionRecord } from '../src/components/../types'
import {
  mergeDecisions,
  buildRewardOption,
  buildActionPieOption
} from '../src/components/decisionTrendOptions'

function makeRecord(step: number, reward: number, actionLabel: string): DecisionRecord {
  return {
    step,
    task_id: 't' + step,
    action: step,
    action_label: actionLabel,
    reward,
    source: 'ppo'
  }
}

describe('mergeDecisions', () => {
  it('按 step 去重并按升序合并增量决策', () => {
    const existing = [makeRecord(1, 0.1, 'A'), makeRecord(2, 0.2, 'B')]
    const incoming = [makeRecord(2, 0.99, 'B'), makeRecord(3, 0.3, 'A')]
    const merged = mergeDecisions(existing, incoming)
    // step=2 已存在，incoming 的重复记录应被忽略
    expect(merged).toHaveLength(3)
    expect(merged.map((d) => d.step)).toEqual([1, 2, 3])
    expect(merged[1].reward).toBe(0.2) // 未覆盖原值
  })
})

describe('buildRewardOption', () => {
  const decisions = [
    makeRecord(1, 1, 'A'),
    makeRecord(2, -1, 'B'),
    makeRecord(3, 0.5, 'A')
  ]
  const option = buildRewardOption(decisions) as any

  it('生成两条折线系列：决策奖励 与 回合平均奖励', () => {
    const names = option.series.map((s: any) => s.name)
    expect(names).toContain('决策奖励')
    expect(names).toContain('回合平均奖励')
  })

  it('将 decisions 的 reward 正确映射为决策奖励 series 的 data', () => {
    const rewardSeries = option.series.find((s: any) => s.name === '决策奖励')
    expect(rewardSeries.data).toEqual([1, -1, 0.5])
  })

  it('xAxis 类别为每步的「步N」标签', () => {
    expect(option.xAxis.data).toEqual(['步1', '步2', '步3'])
  })

  it('回合平均奖励对前 windowSize 步等于累计均值', () => {
    const ep = option.series.find((s: any) => s.name === '回合平均奖励')
    // 第3步窗口均值 = (1 + (-1) + 0.5) / 3
    expect(ep.data[2]).toBeCloseTo((1 - 1 + 0.5) / 3, 4)
  })

  it('空输入不产生 series 数据', () => {
    const empty = buildRewardOption([]) as any
    const rewardSeries = empty.series.find((s: any) => s.name === '决策奖励')
    expect(rewardSeries.data).toEqual([])
  })
})

describe('buildActionPieOption', () => {
  it('按 action_label 聚合动作分布计数', () => {
    const decisions = [
      makeRecord(1, 1, '调度到机器A'),
      makeRecord(2, 1, '调度到机器B'),
      makeRecord(3, 1, '调度到机器A'),
      makeRecord(4, 1, '等待')
    ]
    const option = buildActionPieOption(decisions) as any
    const pie = option.series[0]
    expect(pie.type).toBe('pie')
    const counts: Record<string, number> = {}
    for (const d of pie.data) counts[d.name] = d.value
    expect(counts['调度到机器A']).toBe(2)
    expect(counts['调度到机器B']).toBe(1)
    expect(counts['等待']).toBe(1)
  })

  it('action_label 缺失时回退到 action 字段', () => {
    const rec = makeRecord(1, 1, '')
    rec.action_label = ''
    const option = buildActionPieOption([rec]) as any
    expect(option.series[0].data[0].name).toBe('1')
  })
})
