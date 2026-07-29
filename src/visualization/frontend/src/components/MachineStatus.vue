<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import * as echarts from 'echarts'
import type { MachineInfo, MachineComparisonItem } from '../types'

interface Props {
  // 来自 App.vue 的真机列表（旧字段结构）
  machines?: MachineInfo[]
}
const props = withDefaults(defineProps<Props>(), {
  machines: () => []
})

// ============ 响应式数据 ============
const radarChartRef = ref<HTMLDivElement | null>(null)
const barChartRef = ref<HTMLDivElement | null>(null)
let radarChart: echarts.ECharts | null = null
let barChart: echarts.ECharts | null = null

const comparisonData = ref<MachineComparisonItem[]>([])
const loading = ref(false)

// ============ 计算属性 ============
// 合并 props.machines 与 comparisonData（优先使用 comparisonData，若接口未实现则用 props.machines 推导）
const effectiveData = computed<MachineComparisonItem[]>(() => {
  if (comparisonData.value.length > 0) return comparisonData.value
  // 从 props.machines 推导（兼容旧版字段）
  return props.machines.map((m) => ({
    name: m.name,
    total_qubits: m.total_qubits ?? 0,
    available_ratio: m.available_ratio ?? (m.status === 'running' || m.status === 'idle' ? 1 : 0),
    fidelity: m.fidelity ?? 0.99,
    queue_depth: m.queue_depth ?? 0,
    status: m.status,
    single_gate_fidelity: m.single_gate_fidelity ?? 0.99,
    two_gate_fidelity: m.two_gate_fidelity ?? 0.95
  }))
})

const machineCount = computed(() => effectiveData.value.length)

// ============ 方法 ============
const fetchComparison = async () => {
  loading.value = true
  try {
    const res = await fetch('/api/machines-comparison')
    if (!res.ok) return
    const data = (await res.json()) as MachineComparisonItem[]
    if (Array.isArray(data)) {
      comparisonData.value = data
    }
  } catch (err) {
    console.debug('machines-comparison 接口暂不可用:', err)
  } finally {
    loading.value = false
  }
}

const machineStatusLabel = (s: string): string => {
  const map: Record<string, string> = {
    running: '运行中',
    calibration: '校准中',
    calibrating: '校准中',
    maintenance: '维护中',
    offline: '离线',
    busy: '忙碌',
    idle: '空闲'
  }
  return map[(s || '').toLowerCase()] || s || '未知'
}

const machineStatusClass = (s: string): string => {
  const v = (s || '').toLowerCase()
  if (v === 'running' || v === 'idle') return 'badge-success'
  if (v === 'calibration' || v === 'calibrating' || v === 'busy') return 'badge-warning'
  if (v === 'maintenance' || v === 'offline') return 'badge-danger'
  return 'badge-secondary'
}

const percent = (v: number): string => (v * 100).toFixed(2) + '%'

const initRadarChart = () => {
  if (!radarChartRef.value) return
  radarChart = echarts.init(radarChartRef.value)
  updateRadarChart()
}

const initBarChart = () => {
  if (!barChartRef.value) return
  barChart = echarts.init(barChartRef.value)
  updateBarChart()
}

const updateRadarChart = () => {
  if (!radarChart) return
  const names = effectiveData.value.map((m) => m.name)
  // 雷达图指标：可用率、整体保真度、单比特门保真度、双比特门保真度、(队列深度倒置归一)
  const indicators = [
    { name: '可用率', max: 1 },
    { name: '整体保真度', max: 1 },
    { name: '单比特门保真度', max: 1 },
    { name: '双比特门保真度', max: 1 },
    { name: '空闲度', max: 1 }
  ]

  const colors = ['#60a5fa', '#a78bfa', '#4ade80', '#fbbf24', '#22d3ee', '#f87171', '#94a3b8']
  const series = [
    {
      type: 'radar' as const,
      data: effectiveData.value.map((m, i) => ({
        value: [
          m.available_ratio,
          m.fidelity,
          m.single_gate_fidelity,
          m.two_gate_fidelity,
          // 队列深度倒置：队列越短，空闲度越高（假设队列上限 50）
          Math.max(0, 1 - m.queue_depth / 50)
        ],
        name: m.name,
        lineStyle: { color: colors[i % colors.length], width: 2 },
        itemStyle: { color: colors[i % colors.length] },
        areaStyle: { opacity: 0.15 }
      }))
    }
  ]

  radarChart.setOption({
    backgroundColor: 'transparent',
    title: {
      text: '多机器指标雷达对比',
      textStyle: { color: '#e2e8f0', fontSize: 14 },
      left: 'center'
    },
    tooltip: {
      trigger: 'item',
      backgroundColor: '#1e293b',
      borderColor: '#334155',
      textStyle: { color: '#e2e8f0' }
    },
    legend: {
      bottom: 0,
      textStyle: { color: '#94a3b8' },
      data: names
    },
    radar: {
      indicator: indicators,
      center: ['50%', '55%'],
      radius: '60%',
      axisName: { color: '#94a3b8', fontSize: 12 },
      splitLine: { lineStyle: { color: '#334155' } },
      splitArea: { areaStyle: { color: ['#1e293b', '#0f172a'] } },
      axisLine: { lineStyle: { color: '#334155' } }
    },
    series
  })
}

const updateBarChart = () => {
  if (!barChart) return
  const names = effectiveData.value.map((m) => m.name)
  const colors = ['#60a5fa', '#a78bfa', '#4ade80', '#fbbf24', '#22d3ee', '#f87171', '#94a3b8']

  barChart.setOption({
    backgroundColor: 'transparent',
    title: {
      text: '多机器指标柱状对比',
      textStyle: { color: '#e2e8f0', fontSize: 14 },
      left: 'center'
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: '#1e293b',
      borderColor: '#334155',
      textStyle: { color: '#e2e8f0' }
    },
    legend: {
      top: 30,
      textStyle: { color: '#94a3b8' }
    },
    grid: { left: '6%', right: '6%', bottom: '8%', top: '25%', containLabel: true },
    xAxis: {
      type: 'category',
      data: names,
      axisLine: { lineStyle: { color: '#334155' } },
      axisLabel: { color: '#94a3b8' }
    },
    yAxis: [
      {
        type: 'value',
        name: '保真度/可用率',
        min: 0,
        max: 1,
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94a3b8', formatter: '{value}' },
        splitLine: { lineStyle: { color: '#1e293b' } }
      },
      {
        type: 'value',
        name: '量子比特数/队列',
        position: 'right',
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94a3b8' },
        splitLine: { show: false }
      }
    ],
    series: [
      {
        name: '可用率',
        type: 'bar',
        yAxisIndex: 0,
        data: effectiveData.value.map((m) => +m.available_ratio.toFixed(3)),
        itemStyle: { color: colors[0], borderRadius: [4, 4, 0, 0] },
        barGap: '10%'
      },
      {
        name: '整体保真度',
        type: 'bar',
        yAxisIndex: 0,
        data: effectiveData.value.map((m) => +m.fidelity.toFixed(3)),
        itemStyle: { color: colors[1], borderRadius: [4, 4, 0, 0] }
      },
      {
        name: '单比特门保真度',
        type: 'bar',
        yAxisIndex: 0,
        data: effectiveData.value.map((m) => +m.single_gate_fidelity.toFixed(3)),
        itemStyle: { color: colors[2], borderRadius: [4, 4, 0, 0] }
      },
      {
        name: '双比特门保真度',
        type: 'bar',
        yAxisIndex: 0,
        data: effectiveData.value.map((m) => +m.two_gate_fidelity.toFixed(3)),
        itemStyle: { color: colors[3], borderRadius: [4, 4, 0, 0] }
      },
      {
        name: '量子比特总数',
        type: 'bar',
        yAxisIndex: 1,
        data: effectiveData.value.map((m) => m.total_qubits),
        itemStyle: { color: colors[4], borderRadius: [4, 4, 0, 0] }
      },
      {
        name: '队列深度',
        type: 'bar',
        yAxisIndex: 1,
        data: effectiveData.value.map((m) => m.queue_depth),
        itemStyle: { color: colors[5], borderRadius: [4, 4, 0, 0] }
      }
    ]
  })
}

const handleResize = () => {
  radarChart?.resize()
  barChart?.resize()
}

// ============ 监听数据变化 ============
watch(
  effectiveData,
  () => {
    updateRadarChart()
    updateBarChart()
  },
  { deep: true }
)

// ============ 生命周期 ============
onMounted(async () => {
  await nextTick()
  initRadarChart()
  initBarChart()
  await fetchComparison()
  // 接口未实现时，依赖 props.machines 推导的数据已通过 watch 更新
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  radarChart?.dispose()
  barChart?.dispose()
  radarChart = null
  barChart = null
})
</script>

<template>
  <div class="panel">
    <div class="panel-header">
      <h2>多机器对比视图</h2>
      <div style="display: flex; align-items: center; gap: 10px;">
        <span class="badge">{{ machineCount }} 台机器</span>
        <button class="btn btn-secondary btn-sm" :disabled="loading" @click="fetchComparison">
          {{ loading ? '加载中...' : '刷新' }}
        </button>
      </div>
    </div>
    <div class="panel-body">
      <div v-if="machineCount === 0" class="empty-hint">
        暂无真机数据<br />
        （数据来源：/api/machines-comparison 或 WebSocket real_machines）
      </div>

      <div v-else>
        <!-- 图表对比 -->
        <div class="charts-grid" style="margin-bottom: 16px;">
          <div ref="radarChartRef" class="chart-container" style="height: 340px;"></div>
          <div ref="barChartRef" class="chart-container" style="height: 340px;"></div>
        </div>

        <!-- 详细表格 -->
        <table class="compare-table">
          <thead>
            <tr>
              <th>机器名称</th>
              <th>状态</th>
              <th>量子比特数</th>
              <th>可用率</th>
              <th>整体保真度</th>
              <th>单比特门保真度</th>
              <th>双比特门保真度</th>
              <th>队列深度</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in effectiveData" :key="m.name">
              <td>{{ m.name }}</td>
              <td>
                <span class="status-badge" :class="machineStatusClass(m.status)">
                  {{ machineStatusLabel(m.status) }}
                </span>
              </td>
              <td>{{ m.total_qubits }}</td>
              <td>{{ percent(m.available_ratio) }}</td>
              <td>{{ percent(m.fidelity) }}</td>
              <td>{{ percent(m.single_gate_fidelity) }}</td>
              <td>{{ percent(m.two_gate_fidelity) }}</td>
              <td>{{ m.queue_depth }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 继承全局样式 */
</style>
