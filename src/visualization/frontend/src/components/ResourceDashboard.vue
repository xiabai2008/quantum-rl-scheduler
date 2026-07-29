<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, inject, nextTick } from 'vue'
import * as echarts from 'echarts'
import type { SystemStatus, ResourceHistoryPoint } from '../types'

const status = inject<SystemStatus>('status')

// ============ 响应式数据 ============
const chartRef = ref<HTMLDivElement | null>(null)
let chartInstance: echarts.ECharts | null = null

const history = ref<ResourceHistoryPoint[]>([])
const autoRefresh = ref(true)
const refreshInterval = ref(5000) // 5 秒
let refreshTimer: number | null = null

// ============ 方法 ============
const fetchHistory = async () => {
  try {
    const res = await fetch('/api/resource-history')
    if (!res.ok) return
    const data = (await res.json()) as ResourceHistoryPoint[]
    if (Array.isArray(data)) {
      history.value = data
    }
  } catch (err) {
    // 接口可能尚未实现，静默忽略
    console.debug('resource-history 接口暂不可用:', err)
  }
}

const initChart = () => {
  if (!chartRef.value) return
  chartInstance = echarts.init(chartRef.value)
  chartInstance.setOption({
    backgroundColor: 'transparent',
    title: {
      text: '资源利用率历史趋势',
      textStyle: { color: '#e2e8f0', fontSize: 14 },
      left: 'center'
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#1e293b',
      borderColor: '#334155',
      textStyle: { color: '#e2e8f0' }
    },
    legend: {
      top: 30,
      textStyle: { color: '#94a3b8' },
      data: ['量子比特利用率', '队列长度', '完成数', '平均等待时间']
    },
    grid: { left: '6%', right: '4%', bottom: '12%', top: '25%', containLabel: true },
    xAxis: {
      type: 'category',
      data: [],
      axisLine: { lineStyle: { color: '#334155' } },
      axisLabel: { color: '#94a3b8' },
      boundaryGap: false
    },
    yAxis: [
      {
        type: 'value',
        name: '利用率/百分比',
        position: 'left',
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94a3b8', formatter: '{value}' },
        splitLine: { lineStyle: { color: '#1e293b' } }
      },
      {
        type: 'value',
        name: '数量/时间',
        position: 'right',
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94a3b8' },
        splitLine: { show: false }
      }
    ],
    series: [
      {
        name: '量子比特利用率',
        type: 'line',
        smooth: true,
        yAxisIndex: 0,
        symbol: 'circle',
        symbolSize: 6,
        data: [],
        lineStyle: { color: '#60a5fa', width: 2 },
        itemStyle: { color: '#60a5fa' },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(96, 165, 250, 0.3)' },
            { offset: 1, color: 'rgba(96, 165, 250, 0.05)' }
          ])
        }
      },
      {
        name: '队列长度',
        type: 'line',
        smooth: true,
        yAxisIndex: 1,
        symbol: 'circle',
        symbolSize: 6,
        data: [],
        lineStyle: { color: '#a78bfa', width: 2 },
        itemStyle: { color: '#a78bfa' }
      },
      {
        name: '完成数',
        type: 'line',
        smooth: true,
        yAxisIndex: 1,
        symbol: 'circle',
        symbolSize: 6,
        data: [],
        lineStyle: { color: '#4ade80', width: 2 },
        itemStyle: { color: '#4ade80' }
      },
      {
        name: '平均等待时间',
        type: 'line',
        smooth: true,
        yAxisIndex: 1,
        symbol: 'circle',
        symbolSize: 6,
        data: [],
        lineStyle: { color: '#fbbf24', width: 2, type: 'dashed' },
        itemStyle: { color: '#fbbf24' }
      }
    ]
  })
}

const updateChart = () => {
  if (!chartInstance) return
  const steps = history.value.map((p) => '步' + p.step)
  const util = history.value.map((p) => +(p.qubit_utilization * 100).toFixed(2))
  const queue = history.value.map((p) => p.queue_length)
  const completed = history.value.map((p) => p.completed_tasks)
  const wait = history.value.map((p) => +p.average_wait_time.toFixed(2))

  chartInstance.setOption({
    xAxis: { data: steps },
    series: [
      { data: util },
      { data: queue },
      { data: completed },
      { data: wait }
    ]
  })
}

const handleResize = () => {
  chartInstance?.resize()
}

const startRefresh = () => {
  stopRefresh()
  if (!autoRefresh.value) return
  refreshTimer = window.setInterval(() => {
    fetchHistory()
  }, refreshInterval.value)
}

const stopRefresh = () => {
  if (refreshTimer !== null) {
    window.clearInterval(refreshTimer)
    refreshTimer = null
  }
}

const toggleRefresh = () => {
  if (autoRefresh.value) {
    startRefresh()
  } else {
    stopRefresh()
  }
}

const manualRefresh = () => {
  fetchHistory()
}

// ============ 监听数据变化 ============
watch(history, () => {
  updateChart()
}, { deep: true })

// 同时融合实时 WS 状态作为最新点（如后端历史接口未实现，前端用 WS 数据兜底）
watch(
  () => status?.current_step,
  (newStep) => {
    if (!status) return
    // 仅在没有历史数据或最新步数超出历史时追加
    const last = history.value[history.value.length - 1]
    if (last && last.step === newStep) {
      // 更新最后一点
      last.qubit_utilization = status.qubit_utilization
      last.queue_length = status.queue_length
      last.completed_tasks = status.completed_tasks
      last.average_wait_time = status.average_wait_time
    } else if (newStep !== undefined) {
      history.value.push({
        step: newStep,
        qubit_utilization: status.qubit_utilization,
        queue_length: status.queue_length,
        completed_tasks: status.completed_tasks,
        average_wait_time: status.average_wait_time
      })
      // 限制长度
      if (history.value.length > 50) history.value.shift()
    }
    updateChart()
  }
)

// ============ 生命周期 ============
onMounted(async () => {
  await nextTick()
  initChart()
  await fetchHistory()
  // 若后端历史接口未实现，图表为空；watch 会在 WS 状态更新时填充
  startRefresh()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  stopRefresh()
  window.removeEventListener('resize', handleResize)
  chartInstance?.dispose()
  chartInstance = null
})
</script>

<template>
  <div class="panel">
    <div class="panel-header">
      <h2>资源利用率历史趋势</h2>
      <div style="display: flex; align-items: center; gap: 12px;">
        <label class="switch">
          <input v-model="autoRefresh" type="checkbox" @change="toggleRefresh" />
          <span class="switch-slider"></span>
          <span>自动刷新</span>
        </label>
        <button class="btn btn-secondary btn-sm" @click="manualRefresh">立即刷新</button>
        <span class="badge">{{ history.length }} 个采样点</span>
      </div>
    </div>
    <div class="panel-body">
      <div ref="chartRef" class="chart-container" style="height: 360px;"></div>
      <p class="dash-tip">
        数据来源：/api/resource-history（若接口未实现，自动从 WebSocket 状态更新实时累积）
      </p>
    </div>
  </div>
</template>

<style scoped>
.dash-tip {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 10px;
  text-align: center;
}
</style>
