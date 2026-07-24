<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'

// ============ 类型 ============
interface BattleStep {
  step: number
  reward: number
  cumulative: number
  action: number
  util: number
}

interface BattleStatus {
  running: boolean
  step: number
  ppo_total: number
  fcfs_total: number
  gap: number
  ppo_history: BattleStep[]
  fcfs_history: BattleStep[]
}

// ============ 响应式数据 ============
const status = ref<BattleStatus | null>(null)
const isRunning = ref(false)
const autoStep = ref(false)
const stepTimer = ref<number | null>(null)
const loading = ref(false)
const chartCanvas = ref<HTMLCanvasElement | null>(null)

// ============ 计算属性 ============
const ppoLeading = computed(() => {
  if (!status.value) return false
  return status.value.ppo_total > status.value.fcfs_total
})

const gapPercent = computed(() => {
  if (!status.value || status.value.fcfs_total === 0) return 0
  return (status.value.gap / Math.abs(status.value.fcfs_total) * 100).toFixed(1)
})

const maxSteps = computed(() => {
  if (!status.value) return 0
  return Math.max(status.value.ppo_history.length, status.value.fcfs_history.length)
})

// ============ 方法 ============
const startBattle = async () => {
  loading.value = true
  try {
    await fetch('/api/battle/reset', { method: 'POST' })
    const res = await fetch('/api/battle/start', { method: 'POST' })
    const data = await res.json()
    if (data.success) {
      isRunning.value = true
      await fetchStatus()
    }
  } catch (err) {
    console.error('对战启动失败:', err)
  } finally {
    loading.value = false
  }
}

const stepBattle = async () => {
  if (!isRunning.value) return
  try {
    const res = await fetch('/api/battle/step', { method: 'POST' })
    const data = await res.json()
    if (data.error) {
      console.error('对战步进错误:', data.error)
      return
    }
    await fetchStatus()
  } catch (err) {
    console.error('对战步进失败:', err)
  }
}

const fetchStatus = async () => {
  try {
    const res = await fetch('/api/battle/status')
    if (!res.ok) return
    status.value = await res.json()
    drawChart()
  } catch (err) {
    console.debug('battle/status 接口暂不可用:', err)
  }
}

const toggleAutoStep = () => {
  autoStep.value = !autoStep.value
  if (autoStep.value) {
    startAutoStep()
  } else {
    stopAutoStep()
  }
}

const startAutoStep = () => {
  stopAutoStep()
  stepTimer.value = window.setInterval(() => {
    if (isRunning.value) {
      stepBattle()
    } else {
      stopAutoStep()
    }
  }, 1000)
}

const stopAutoStep = () => {
  autoStep.value = false
  if (stepTimer.value !== null) {
    window.clearInterval(stepTimer.value)
    stepTimer.value = null
  }
}

const resetBattle = async () => {
  stopAutoStep()
  await fetch('/api/battle/reset', { method: 'POST' })
  isRunning.value = false
  status.value = null
}

const drawChart = () => {
  if (!chartCanvas.value || !status.value) return
  const canvas = chartCanvas.value
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const dpr = window.devicePixelRatio || 1
  const rect = canvas.getBoundingClientRect()
  canvas.width = rect.width * dpr
  canvas.height = rect.height * dpr
  ctx.scale(dpr, dpr)

  const w = rect.width
  const h = rect.height
  const padding = { top: 20, right: 20, bottom: 30, left: 50 }
  const chartW = w - padding.left - padding.right
  const chartH = h - padding.top - padding.bottom

  // 清空
  ctx.clearRect(0, 0, w, h)

  // 背景网格
  ctx.strokeStyle = 'rgba(255,255,255,0.05)'
  ctx.lineWidth = 1
  for (let i = 0; i <= 4; i++) {
    const y = padding.top + (chartH / 4) * i
    ctx.beginPath()
    ctx.moveTo(padding.left, y)
    ctx.lineTo(padding.left + chartW, y)
    ctx.stroke()
  }

  const ppoHist = status.value.ppo_history
  const fcfsHist = status.value.fcfs_history
  const n = Math.max(ppoHist.length, fcfsHist.length)
  if (n === 0) return

  // 计算Y轴范围
  const allValues = [...ppoHist.map(d => d.cumulative), ...fcfsHist.map(d => d.cumulative), 0]
  const yMin = Math.min(...allValues)
  const yMax = Math.max(...allValues, 1)
  const yRange = yMax - yMin || 1

  // Y轴标签
  ctx.fillStyle = 'rgba(255,255,255,0.4)'
  ctx.font = '10px sans-serif'
  ctx.textAlign = 'right'
  for (let i = 0; i <= 4; i++) {
    const val = yMin + (yRange / 4) * (4 - i)
    const y = padding.top + (chartH / 4) * i
    ctx.fillText(val.toFixed(0), padding.left - 8, y + 3)
  }

  // X轴标签
  ctx.textAlign = 'center'
  ctx.fillText('0', padding.left, h - 10)
  ctx.fillText(String(n), padding.left + chartW, h - 10)

  // 绘制函数
  const drawLine = (hist: BattleStep[], color: string, label: string) => {
    if (hist.length === 0) return
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.beginPath()
    hist.forEach((d, i) => {
      const x = padding.left + (chartW / Math.max(n - 1, 1)) * i
      const y = padding.top + chartH - ((d.cumulative - yMin) / yRange) * chartH
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // 填充区域
    ctx.lineTo(padding.left + (chartW / Math.max(n - 1, 1)) * (hist.length - 1), padding.top + chartH)
    ctx.lineTo(padding.left, padding.top + chartH)
    ctx.closePath()
    ctx.fillStyle = color + '15'
    ctx.fill()

    // 图例
    ctx.fillStyle = color
    ctx.fillRect(padding.left + 10, padding.top - 12, 12, 3)
    ctx.fillStyle = 'rgba(255,255,255,0.7)'
    ctx.textAlign = 'left'
    ctx.font = '11px sans-serif'
    ctx.fillText(label, padding.left + 26, padding.top - 8)
  }

  drawLine(ppoHist, '#00d4ff', 'PPO')
  drawLine(fcfsHist, '#f59e0b', 'FCFS')
}

// ============ 生命周期 ============
onMounted(() => {
  fetchStatus()
  window.addEventListener('resize', drawChart)
})

onUnmounted(() => {
  stopAutoStep()
  window.removeEventListener('resize', drawChart)
})
</script>

<template>
  <div class="panel">
    <div class="panel-header">
      <h2>PPO vs FCFS 实时对战</h2>
      <div style="display: flex; align-items: center; gap: 8px;">
        <span v-if="status" class="badge">Step {{ status.step }}</span>
      </div>
    </div>
    <div class="panel-body">
      <!-- 控制按钮 -->
      <div class="battle-controls">
        <button
          class="btn btn-primary btn-sm"
          :disabled="loading || isRunning"
          @click="startBattle"
        >
          {{ loading ? '启动中...' : '开始对战' }}
        </button>
        <button
          class="btn btn-secondary btn-sm"
          :disabled="!isRunning"
          @click="stepBattle"
        >
          单步
        </button>
        <button
          :class="['btn', 'btn-sm', autoStep ? 'btn-primary' : 'btn-secondary']"
          :disabled="!isRunning"
          @click="toggleAutoStep"
        >
          {{ autoStep ? '停止自动' : '自动步进' }}
        </button>
        <button
          class="btn btn-secondary btn-sm"
          @click="resetBattle"
        >
          重置
        </button>
      </div>

      <!-- 累积奖励对比 -->
      <div v-if="status" class="score-board">
        <div class="score-card ppo-card">
          <div class="score-label">PPO 累积奖励</div>
          <div class="score-value">{{ status.ppo_total.toFixed(2) }}</div>
        </div>
        <div class="score-card fcfs-card">
          <div class="score-label">FCFS 累积奖励</div>
          <div class="score-value">{{ status.fcfs_total.toFixed(2) }}</div>
        </div>
        <div class="score-card gap-card" :class="{ 'ppo-leading': ppoLeading }">
          <div class="score-label">奖励差距</div>
          <div class="score-value" :style="{ color: ppoLeading ? 'var(--accent-green)' : 'var(--accent-red)' }">
            {{ status.gap > 0 ? '+' : '' }}{{ status.gap.toFixed(2) }}
          </div>
          <div class="score-sub" v-if="status.fcfs_total !== 0">
            ({{ gapPercent }}%)
          </div>
        </div>
      </div>

      <!-- 奖励曲线图 -->
      <div class="chart-container">
        <canvas ref="chartCanvas" class="battle-chart"></canvas>
        <div v-if="!status || maxSteps === 0" class="chart-empty">
          点击"开始对战"启动 PPO vs FCFS 对比
        </div>
      </div>

      <!-- 最新步数据 -->
      <div v-if="status && status.ppo_history.length > 0" class="latest-step">
        <div class="step-detail">
          <span class="detail-label">PPO</span>
          <span class="detail-value">
            Action={{ status.ppo_history[status.ppo_history.length - 1].action }}
            | Reward={{ status.ppo_history[status.ppo_history.length - 1].reward.toFixed(2) }}
            | Util={{ (status.ppo_history[status.ppo_history.length - 1].util * 100).toFixed(1) }}%
          </span>
        </div>
        <div class="step-detail">
          <span class="detail-label">FCFS</span>
          <span class="detail-value">
            Action={{ status.fcfs_history[status.fcfs_history.length - 1].action }}
            | Reward={{ status.fcfs_history[status.fcfs_history.length - 1].reward.toFixed(2) }}
            | Util={{ (status.fcfs_history[status.fcfs_history.length - 1].util * 100).toFixed(1) }}%
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.battle-controls {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.score-board {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 12px;
  margin-bottom: 16px;
}

.score-card {
  padding: 12px 16px;
  border-radius: 8px;
  text-align: center;
}

.ppo-card {
  background: rgba(0, 212, 255, 0.1);
  border: 1px solid rgba(0, 212, 255, 0.3);
}

.fcfs-card {
  background: rgba(245, 158, 11, 0.1);
  border: 1px solid rgba(245, 158, 11, 0.3);
}

.gap-card {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--border-color);
}

.score-label {
  font-size: 11px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}

.score-value {
  font-size: 24px;
  font-weight: 700;
}

.score-sub {
  font-size: 11px;
  color: var(--text-secondary);
  margin-top: 2px;
}

.chart-container {
  position: relative;
  height: 200px;
  background: var(--bg-secondary);
  border-radius: 8px;
  padding: 8px;
}

.battle-chart {
  width: 100%;
  height: 100%;
}

.chart-empty {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: var(--text-secondary);
  font-size: 14px;
}

.latest-step {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 12px;
  padding: 10px 14px;
  background: var(--bg-secondary);
  border-radius: 8px;
}

.step-detail {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 12px;
}

.detail-label {
  font-weight: 700;
  width: 40px;
}

.detail-value {
  color: var(--text-secondary);
}
</style>
