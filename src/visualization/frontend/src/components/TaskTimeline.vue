<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'
import * as echarts from 'echarts'
import type { Task, TaskStatus } from '../types'

interface Props {
  /** 来自 App.vue 的实时任务列表（WebSocket status_update / task_added 推送） */
  tasks: Task[]
}
const props = withDefaults(defineProps<Props>(), {
  tasks: () => []
})

// ============ 响应式数据 ============
const chartRef = ref<HTMLDivElement | null>(null)
let chartInstance: echarts.ECharts | null = null

// 时间轴上最多展示的任务数（最近 N 个），避免过于拥挤
const MAX_ROWS = 30
// 运行中任务条每 2s 刷新一次，使时间条随时间增长（近实时观感）
const RUNNING_REFRESH_MS = 2000
let refreshTimer: number | null = null

// ============ 状态颜色映射 ============
const statusColor: Record<TaskStatus, string> = {
  pending: '#64748b', // 等待中（排队，灰色）
  running: '#60a5fa', // 运行中（蓝色）
  completed: '#4ade80', // 已完成（绿色）
  failed: '#f87171' // 失败（红色）
}
const statusLabelCN: Record<TaskStatus, string> = {
  pending: '等待',
  running: '运行',
  completed: '完成',
  failed: '失败'
}
const taskTypeCN: Record<string, string> = {
  quantum: '量子',
  classical: '经典',
  hybrid: '混合'
}

// ============ 工具方法 ============
const parseTime = (iso?: string): number | null => {
  if (!iso) return null
  const t = Date.parse(iso)
  return isNaN(t) ? null : t
}

const taskRowLabel = (id: string): string =>
  id.length > 14 ? id.slice(0, 12) + '…' : id

const fmtClock = (ms: number): string => {
  const d = new Date(ms)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

// ============ 图表构建 ============
const buildOption = (): echarts.EChartsOption => {
  // 按提交时间升序，取最近 MAX_ROWS 个任务（最新在最上方展示）
  const sorted = [...props.tasks]
    .filter((t) => t.task_id)
    .sort((a, b) => (parseTime(a.submitted_at) ?? 0) - (parseTime(b.submitted_at) ?? 0))
  const recent = sorted.slice(-MAX_ROWS)
  const now = Date.now()

  const labels: string[] = []
  const offsetData: number[] = [] // 透明偏移条（定位到任务开始时刻）
  const durationData: any[] = [] // 可见执行条（颜色随状态）

  for (const t of recent) {
    const startMs = parseTime(t.submitted_at) ?? now
    const status = (t.status || 'pending') as TaskStatus
    let durationMs: number
    if (status === 'running') {
      // 运行中：从提交时刻延伸到现在，随时间增长
      durationMs = Math.max(now - startMs, 1000)
    } else if (status === 'pending') {
      // 等待中：尚未开始执行，用极短灰条标示“已入队”位置
      durationMs = 1500
    } else {
      // 已完成 / 失败：用真实执行耗时（无则给最小可见宽度）
      durationMs = Math.max((t.execution_time || 0) * 1000, 1000)
    }

    labels.push(taskRowLabel(t.task_id))
    offsetData.push(startMs)
    durationData.push({
      value: durationMs,
      itemStyle: { color: statusColor[status], borderRadius: [3, 3, 3, 3] },
      task: t,
      _startMs: startMs,
      _endMs: status === 'running' ? now : startMs + durationMs
    })
  }

  return {
    backgroundColor: 'transparent',
    title: {
      text: '任务执行时间线（实时调度甘特图）',
      textStyle: { color: '#e2e8f0', fontSize: 14 },
      left: 'center'
    },
    tooltip: {
      trigger: 'item',
      backgroundColor: '#1e293b',
      borderColor: '#334155',
      textStyle: { color: '#e2e8f0' },
      formatter: (p: any) => {
        const d = p.data
        if (!d || !d.task) return ''
        const t = d.task as Task
        const st = (t.status || 'pending') as TaskStatus
        const startStr = fmtClock(d._startMs)
        const endStr = fmtClock(d._endMs)
        const durSec = ((d._endMs - d._startMs) / 1000).toFixed(1)
        return [
          `<b>${t.task_id}</b>`,
          `类型：${taskTypeCN[t.task_type] || t.task_type}`,
          `状态：<span style="color:${statusColor[st]}">${statusLabelCN[st]}</span>`,
          `优先级：${t.priority}　比特数：${t.qubit_count}`,
          `起：${startStr}　止：${endStr}`,
          `时长：${durSec}s`
        ].join('<br/>')
      }
    },
    legend: {
      top: 28,
      textStyle: { color: '#94a3b8' },
      data: ['等待', '运行', '完成', '失败']
    },
    grid: { left: '2%', right: '3%', bottom: '6%', top: '18%', containLabel: true },
    xAxis: {
      type: 'time',
      axisLine: { lineStyle: { color: '#334155' } },
      axisLabel: { color: '#94a3b8', formatter: (v: number) => fmtClock(v) },
      splitLine: { lineStyle: { color: '#1e293b' } }
    },
    yAxis: {
      type: 'category',
      data: labels,
      inverse: true,
      axisLine: { lineStyle: { color: '#334155' } },
      axisLabel: { color: '#cbd5e1', fontSize: 11 },
      splitLine: { show: true, lineStyle: { color: '#1e293b' } }
    },
    series: [
      {
        name: 'offset',
        type: 'bar',
        stack: 'gantt',
        silent: true,
        itemStyle: { color: 'transparent' },
        emphasis: { itemStyle: { color: 'transparent' } },
        data: offsetData
      },
      {
        name: 'duration',
        type: 'bar',
        stack: 'gantt',
        barWidth: '55%',
        data: durationData,
        markLine: {
          silent: true,
          symbol: 'none',
          lineStyle: { color: '#fbbf24', type: 'dashed' },
          label: { formatter: '现在', color: '#fbbf24', position: 'insideEndTop' },
          data: [{ xAxis: now }]
        }
      }
    ]
  }
}

const updateChart = () => {
  if (!chartInstance) return
  chartInstance.setOption(buildOption())
}

const handleResize = () => chartInstance?.resize()

// 运行中任务条随时间推进：每 2s 重绘一次（近实时）
const startRunningRefresh = () => {
  stopRunningRefresh()
  refreshTimer = window.setInterval(() => {
    const hasRunning = props.tasks.some((t) => t.status === 'running')
    if (hasRunning) updateChart()
  }, RUNNING_REFRESH_MS)
}
const stopRunningRefresh = () => {
  if (refreshTimer !== null) {
    window.clearInterval(refreshTimer)
    refreshTimer = null
  }
}

// ============ 监听数据变化 ============
watch(
  () => props.tasks,
  () => updateChart(),
  { deep: true }
)

// ============ 生命周期 ============
onMounted(async () => {
  await nextTick()
  if (chartRef.value) {
    chartInstance = echarts.init(chartRef.value)
    updateChart()
  }
  startRunningRefresh()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  stopRunningRefresh()
  window.removeEventListener('resize', handleResize)
  chartInstance?.dispose()
  chartInstance = null
})
</script>

<template>
  <div class="panel">
    <div class="panel-header">
      <h2>任务执行时间线</h2>
      <div style="display: flex; align-items: center; gap: 10px;">
        <span class="badge">实时甘特图</span>
        <span class="badge">{{ tasks.length }} 个任务</span>
      </div>
    </div>
    <div class="panel-body">
      <div v-if="tasks.length === 0" class="empty-hint">
        暂无任务数据<br />
        （数据来源：WebSocket 推送的实时任务列表）
      </div>
      <div
        v-else
        ref="chartRef"
        class="chart-container"
        style="height: 380px;"
      ></div>
      <p class="timeline-tip">
        蓝色=运行中（随时间延伸），绿色=完成，红色=失败，灰色=等待中。
        数据随 WebSocket status_update / task_added 实时刷新。
      </p>
    </div>
  </div>
</template>

<style scoped>
.timeline-tip {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 10px;
  text-align: center;
}
</style>
