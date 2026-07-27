// 决策趋势图（DecisionTrend）的纯函数 option 构造器
// 与 Echarts 运行时解耦，便于单元测试，且零后端改动（仅消费既有 /api/decision-log 契约）。
import type { EChartsOption } from 'echarts'
import type { DecisionRecord } from '../types'

/**
 * 增量合并决策记录：把 incoming 中 step 尚未出现的记录追加到 existing，
 * 并最终按 step 升序返回。以 step 去重，匹配 simulator 每步一决策的语义。
 */
export function mergeDecisions(
  existing: DecisionRecord[],
  incoming: DecisionRecord[]
): DecisionRecord[] {
  const seen = new Set(existing.map((d) => d.step))
  const merged = [...existing]
  for (const d of incoming) {
    if (!seen.has(d.step)) {
      seen.add(d.step)
      merged.push(d)
    }
  }
  merged.sort((a, b) => a.step - b.step)
  return merged
}

/**
 * 构造「决策奖励 / 回合平均奖励 vs step」实时折线图 option。
 * - 决策奖励：每个 step 的原始 reward
 * - 回合平均奖励：以滑动窗口（最近 windowSize 步）均值平滑，体现“回合奖励”整体趋势
 */
export function buildRewardOption(
  decisions: DecisionRecord[],
  windowSize = 20
): EChartsOption {
  const steps = decisions.map((d) => '步' + d.step)
  const rewards = decisions.map((d) => +Number(d.reward ?? 0).toFixed(4))

  const episodeRewards = decisions.map((_, i) => {
    const start = Math.max(0, i - windowSize + 1)
    const slice = decisions.slice(start, i + 1)
    const avg = slice.reduce((sum, d) => sum + (d.reward ?? 0), 0) / slice.length
    return +Number(avg).toFixed(4)
  })

  const option: EChartsOption = {
    backgroundColor: 'transparent',
    title: {
      text: '决策奖励 / 回合平均奖励趋势',
      left: 'center',
      textStyle: { color: '#e2e8f0', fontSize: 13, fontWeight: 600 }
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#1e293b',
      borderColor: '#334155',
      textStyle: { color: '#e2e8f0' }
    },
    legend: {
      top: 28,
      textStyle: { color: '#94a3b8' },
      data: ['决策奖励', '回合平均奖励']
    },
    grid: { left: '6%', right: '4%', bottom: '12%', top: '22%', containLabel: true },
    xAxis: {
      type: 'category',
      data: steps,
      boundaryGap: false,
      axisLine: { lineStyle: { color: '#334155' } },
      axisLabel: { color: '#94a3b8' }
    },
    yAxis: {
      type: 'value',
      name: '奖励',
      nameTextStyle: { color: '#94a3b8' },
      axisLine: { lineStyle: { color: '#334155' } },
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#1e293b' } }
    },
    series: [
      {
        name: '决策奖励',
        type: 'line',
        smooth: true,
        showSymbol: false,
        data: rewards,
        lineStyle: { color: '#60a5fa', width: 2 },
        itemStyle: { color: '#60a5fa' },
        areaStyle: { color: 'rgba(96, 165, 250, 0.12)' }
      },
      {
        name: '回合平均奖励',
        type: 'line',
        smooth: true,
        showSymbol: false,
        data: episodeRewards,
        lineStyle: { color: '#a78bfa', width: 2, type: 'dashed' },
        itemStyle: { color: '#a78bfa' }
      }
    ]
  }
  return option
}

/**
 * 构造「动作分布」饼图 option。按 action_label 聚合计数（缺失时回退到 action）。
 */
export function buildActionPieOption(decisions: DecisionRecord[]): EChartsOption {
  const counts = new Map<string, number>()
  for (const d of decisions) {
    const label = d.action_label || String(d.action)
    counts.set(label, (counts.get(label) ?? 0) + 1)
  }
  const data = Array.from(counts.entries()).map(([name, value]) => ({ name, value }))

  const palette = ['#60a5fa', '#a78bfa', '#22d3ee', '#4ade80', '#fbbf24', '#f472b6', '#94a3b8']
  const option: EChartsOption = {
    backgroundColor: 'transparent',
    title: {
      text: '动作分布',
      left: 'center',
      textStyle: { color: '#e2e8f0', fontSize: 13, fontWeight: 600 }
    },
    tooltip: {
      trigger: 'item',
      backgroundColor: '#1e293b',
      borderColor: '#334155',
      textStyle: { color: '#e2e8f0' },
      formatter: '{b}: {c} ({d}%)'
    },
    legend: {
      bottom: 0,
      type: 'scroll',
      textStyle: { color: '#94a3b8' }
    },
    color: palette,
    series: [
      {
        name: '动作分布',
        type: 'pie',
        radius: ['38%', '66%'],
        center: ['50%', '52%'],
        avoidLabelOverlap: true,
        itemStyle: { borderColor: '#0f172a', borderWidth: 2 },
        label: { color: '#cbd5e1', fontSize: 11 },
        data
      }
    ]
  }
  return option
}
