<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, inject, nextTick } from 'vue'
import * as echarts from 'echarts'
import type { SystemStatus, MachineInfo } from '../types'

// 复用 App.vue provide 的实时系统状态（WebSocket 推送）
const status = inject<SystemStatus>('status')

// ============ 响应式数据 ============
const chartRef = ref<HTMLDivElement | null>(null)
let chartInstance: echarts.ECharts | null = null

// 由 real_machines 推导出的可用资源数据（兼容 available_ratio 缺失场景）
const machineData = computed(() => {
  const list: MachineInfo[] = status?.real_machines ?? []
  return list.map((m) => ({
    name: m.name,
    total: m.total_qubits ?? 0,
    // 可用比例：未提供时按机器状态估算
    availableRatio:
      m.available_ratio ??
      (m.status === 'running' || m.status === 'idle' ? 1 : 0)
  }))
})

const hasMachines = computed(() => machineData.value.length > 0)

// ============ 图表构建 ============
const buildOption = (): echarts.EChartsOption => {
  if (hasMachines.value) {
    const md = machineData.value
    const names = md.map((m) => m.name)
    const allocated = md.map((m) =>
      Math.round(m.total * (1 - m.availableRatio))
    )
    const available = md.map((m) => Math.round(m.total * m.availableRatio))
    const utilPct = md.map((m) =>
      m.total > 0 ? +((1 - m.availableRatio) * 100).toFixed(1) : 0
    )

    return {
      backgroundColor: 'transparent',
      title: {
        text: '各真机量子比特资源分配',
        textStyle: { color: '#e2e8f0', fontSize: 14 },
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: '#1e293b',
        borderColor: '#334155',
        textStyle: { color: '#e2e8f0' },
        formatter: (params: any) => {
          const idx = params[0]?.dataIndex ?? 0
          const m = md[idx]
          return [
            `<b>${m.name}</b>`,
            `总比特数：${m.total}`,
            `已分配：${allocated[idx]}　空闲：${available[idx]}`,
            `利用率：${utilPct[idx]}%`
          ].join('<br/>')
        }
      },
      legend: {
        top: 28,
        textStyle: { color: '#94a3b8' },
        data: ['已分配量子比特', '空闲量子比特']
      },
      grid: { left: '2%', right: '4%', bottom: '4%', top: '20%', containLabel: true },
      xAxis: {
        type: 'value',
        name: '量子比特数',
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94a3b8' },
        splitLine: { lineStyle: { color: '#1e293b' } }
      },
      yAxis: {
        type: 'category',
        data: names,
        inverse: true,
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#cbd5e1', fontSize: 12 },
        splitLine: { show: true, lineStyle: { color: '#1e293b' } }
      },
      series: [
        {
          name: '已分配量子比特',
          type: 'bar',
          stack: 'qubits',
          data: allocated,
          itemStyle: { color: '#60a5fa', borderRadius: [4, 0, 0, 4] }
        },
        {
          name: '空闲量子比特',
          type: 'bar',
          stack: 'qubits',
          data: available,
          itemStyle: { color: '#4ade80', borderRadius: [0, 4, 4, 0] }
        }
      ]
    }
  }

  // 无真机数据时：退回展示整体量子比特利用率仪表盘
  const util = (status?.qubit_utilization ?? 0) * 100
  return {
    backgroundColor: 'transparent',
    title: {
      text: '整体量子比特利用率（实时）',
      textStyle: { color: '#e2e8f0', fontSize: 14 },
      left: 'center'
    },
    tooltip: {
      formatter: () => `利用率：${util.toFixed(1)}%`,
      backgroundColor: '#1e293b',
      borderColor: '#334155',
      textStyle: { color: '#e2e8f0' }
    },
    series: [
      {
        type: 'gauge',
        startAngle: 210,
        endAngle: -30,
        min: 0,
        max: 100,
        progress: { show: true, width: 18, itemStyle: { color: '#60a5fa' } },
        axisLine: {
          lineStyle: {
            width: 18,
            color: [
              [0.6, '#4ade80'],
              [0.85, '#fbbf24'],
              [1, '#f87171']
            ]
          }
        },
        axisTick: { show: false },
        splitLine: { length: 12, lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94a3b8', distance: 22 },
        pointer: { width: 5, itemStyle: { color: '#e2e8f0' } },
        detail: {
          valueAnimation: true,
          formatter: '{value}%',
          color: '#e2e8f0',
          fontSize: 26,
          offsetCenter: [0, '55%']
        },
        data: [{ value: +util.toFixed(1) }]
      }
    ]
  }
}

const updateChart = () => {
  if (!chartInstance) return
  // notMerge=true：在“按机器”与“整体仪表盘”两种模式间切换时彻底替换配置
  chartInstance.setOption(buildOption(), true)
}

const handleResize = () => chartInstance?.resize()

// ============ 监听实时状态变化 ============
watch(
  () => status?.real_machines,
  () => updateChart(),
  { deep: true }
)
watch(
  () => status?.qubit_utilization,
  () => updateChart()
)

// ============ 生命周期 ============
onMounted(async () => {
  await nextTick()
  if (chartRef.value) {
    chartInstance = echarts.init(chartRef.value)
    updateChart()
  }
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  chartInstance?.dispose()
  chartInstance = null
})
</script>

<template>
  <div class="panel">
    <div class="panel-header">
      <h2>资源分配</h2>
      <span class="badge">
        {{ hasMachines ? machineData.length + ' 台真机' : '整体利用率' }}
      </span>
    </div>
    <div class="panel-body">
      <div
        ref="chartRef"
        class="chart-container"
        style="height: 360px;"
      ></div>
      <p class="alloc-tip">
        按机器展示已分配 / 空闲量子比特；无真机数据时退回整体利用率仪表盘。
        数据来源：WebSocket 推送的 real_machines / qubit_utilization。
      </p>
    </div>
  </div>
</template>

<style scoped>
.alloc-tip {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 10px;
  text-align: center;
}
</style>
