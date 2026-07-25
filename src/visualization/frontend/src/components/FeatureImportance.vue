<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'

// 全局特征重要性排名（来自 /api/explainability/summary）
interface ImportanceItem {
  feature: string
  importance: number
}

const importance = ref<ImportanceItem[]>([])
const totalDecisions = ref(0)
const loading = ref(false)
const chartRef = ref<HTMLElement | null>(null)
let chart: echarts.ECharts | null = null

const fetchSummary = async () => {
  loading.value = true
  try {
    const res = await fetch('/api/explainability/summary')
    if (!res.ok) return
    const data = (await res.json()) as {
      feature_importance?: ImportanceItem[]
      total_decisions?: number
    }
    importance.value = data.feature_importance ?? []
    totalDecisions.value = data.total_decisions ?? 0
    renderChart()
  } catch (err) {
    console.debug('explainability/summary 接口暂不可用:', err)
  } finally {
    loading.value = false
  }
}

const renderChart = () => {
  if (!chartRef.value) return
  if (!chart) {
    chart = echarts.init(chartRef.value)
  }
  const items = importance.value
  if (items.length === 0) {
    chart.clear()
    return
  }
  // 升序排列，使最重要的特征显示在图表顶部
  const ordered = [...items].sort((a, b) => a.importance - b.importance)
  chart.setOption({
    title: {
      text: `全局特征重要性（基于 ${totalDecisions.value} 条决策）`,
      left: 'center',
      textStyle: { color: '#e2e8f0', fontSize: 13, fontWeight: 600 }
    },
    grid: { left: 130, right: 60, top: 44, bottom: 20 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: {
      type: 'value',
      axisLabel: { color: '#94a3b8', fontSize: 10 },
      splitLine: { lineStyle: { color: 'rgba(148,163,184,0.15)' } }
    },
    yAxis: {
      type: 'category',
      data: ordered.map((i) => i.feature),
      axisLabel: { color: '#cbd5e1', fontSize: 11 },
      axisLine: { lineStyle: { color: 'rgba(148,163,184,0.3)' } }
    },
    series: [
      {
        type: 'bar',
        data: ordered.map((i) => Number(i.importance.toFixed(4))),
        barWidth: '62%',
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
            { offset: 0, color: '#0e7490' },
            { offset: 1, color: '#22d3ee' }
          ])
        },
        label: {
          show: true,
          position: 'right',
          color: '#cbd5e1',
          fontSize: 10,
          formatter: '{c}'
        }
      }
    ]
  })
}

const onResize = () => {
  chart?.resize()
}

onMounted(() => {
  fetchSummary()
  window.addEventListener('resize', onResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', onResize)
  chart?.dispose()
  chart = null
})
</script>

<template>
  <div class="panel">
    <div class="panel-header">
      <h2>特征重要性排名</h2>
      <span class="badge">{{ importance.length }} 维特征</span>
    </div>
    <div class="panel-body">
      <div v-if="loading" class="empty-hint">加载中...</div>
      <div v-else-if="importance.length === 0" class="empty-hint">
        暂无特征贡献度数据<br />
        （需先运行调度并产生决策记录）
      </div>
      <div v-else ref="chartRef" class="chart-box"></div>
    </div>
  </div>
</template>

<style scoped>
.chart-box {
  width: 100%;
  height: 440px;
}
</style>
