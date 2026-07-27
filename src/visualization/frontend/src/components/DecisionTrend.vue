<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import type { DecisionRecord } from '../types'
import {
  mergeDecisions,
  buildRewardOption,
  buildActionPieOption
} from './decisionTrendOptions'

// ============ 响应式数据 ============
const decisions = ref<DecisionRecord[]>([])
const loading = ref(false)
const lastUpdated = ref<string>('')

const rewardChartRef = ref<HTMLElement | null>(null)
const pieChartRef = ref<HTMLElement | null>(null)
let rewardChart: echarts.ECharts | null = null
let pieChart: echarts.ECharts | null = null

// 轮询节奏：约 3s，匹配 simulator tick 节奏
const POLL_INTERVAL = 3000
let pollTimer: number | null = null

// ============ 方法 ============
const fetchDecisionLog = async () => {
  try {
    const res = await fetch('/api/decision-log')
    if (!res.ok) return
    const data = (await res.json()) as
      | { decisions?: DecisionRecord[] }
      | DecisionRecord[]
    const incoming: DecisionRecord[] = Array.isArray(data)
      ? data
      : (data as { decisions?: DecisionRecord[] }).decisions ?? []
    if (incoming.length === 0) return
    // 增量合并到本地缓冲（零后端改动：仅消费既有 REST 契约）
    decisions.value = mergeDecisions(decisions.value, incoming)
    lastUpdated.value = new Date().toISOString()
    renderCharts()
  } catch (err) {
    // 接口可能尚未实现，静默忽略（与现有看板组件一致）
    console.debug('decision-log 接口暂不可用:', err)
  } finally {
    loading.value = false
  }
}

const renderCharts = () => {
  if (rewardChart) {
    rewardChart.setOption(buildRewardOption(decisions.value))
  }
  if (pieChart) {
    pieChart.setOption(buildActionPieOption(decisions.value))
  }
}

const initCharts = () => {
  if (rewardChartRef.value && !rewardChart) {
    rewardChart = echarts.init(rewardChartRef.value)
  }
  if (pieChartRef.value && !pieChart) {
    pieChart = echarts.init(pieChartRef.value)
  }
  renderCharts()
}

const onResize = () => {
  rewardChart?.resize()
  pieChart?.resize()
}

const startPolling = () => {
  stopPolling()
  pollTimer = window.setInterval(fetchDecisionLog, POLL_INTERVAL)
}

const stopPolling = () => {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
}

// ============ 生命周期 ============
onMounted(async () => {
  initCharts()
  loading.value = true
  await fetchDecisionLog()
  startPolling()
  window.addEventListener('resize', onResize)
})

onUnmounted(() => {
  stopPolling()
  window.removeEventListener('resize', onResize)
  rewardChart?.dispose()
  pieChart?.dispose()
  rewardChart = null
  pieChart = null
})
</script>

<template>
  <div class="panel">
    <div class="panel-header">
      <h2>调度决策动态趋势</h2>
      <div style="display: flex; align-items: center; gap: 12px;">
        <button class="btn btn-secondary btn-sm" :disabled="loading" @click="fetchDecisionLog">
          {{ loading ? '加载中...' : '立即刷新' }}
        </button>
        <span class="badge">{{ decisions.length }} 条决策</span>
      </div>
    </div>
    <div class="panel-body">
      <div v-if="decisions.length === 0" class="empty-hint">
        暂无决策记录<br />
        （数据来源：/api/decision-log，可能尚未实现或调度未启动）
      </div>
      <div v-else class="trend-grid">
        <div ref="rewardChartRef" class="trend-chart"></div>
        <div ref="pieChartRef" class="trend-chart trend-chart--pie"></div>
      </div>
      <p v-if="lastUpdated" class="trend-tip">
        最后更新：{{ lastUpdated }}（每 {{ POLL_INTERVAL / 1000 }}s 自动轮询 /api/decision-log）
      </p>
    </div>
  </div>
</template>

<style scoped>
.trend-grid {
  display: grid;
  grid-template-columns: 3fr 2fr;
  gap: 16px;
}
.trend-chart {
  width: 100%;
  height: 360px;
}
.trend-chart--pie {
  height: 360px;
}
.trend-tip {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 10px;
  text-align: center;
}
@media (max-width: 900px) {
  .trend-grid {
    grid-template-columns: 1fr;
  }
}
</style>
