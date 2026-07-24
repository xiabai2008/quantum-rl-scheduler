<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'

// ============ 响应式数据 ============
const latest = ref<any>(null)
const loading = ref(false)
const refreshInterval = ref<number | null>(null)
const autoRefresh = ref(true)

// ============ 计算属性 ============
const topFeatures = computed(() => {
  if (!latest.value?.feature_contributions) return []
  return Object.entries(latest.value.feature_contributions)
    .map(([name, value]) => ({ name, value: value as number }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 8)
})

const maxValue = computed(() => {
  if (topFeatures.value.length === 0) return 1
  return Math.max(...topFeatures.value.map(f => f.value), 0.01)
})

const actionLabel = computed(() => {
  if (!latest.value) return '-'
  const map: Record<number, string> = { 0: '经典资源', 1: '量子资源', 2: '混合执行' }
  return map[latest.value.action] || latest.value.action_label || '-'
})

const actionColor = computed(() => {
  if (!latest.value) return '#94a3b8'
  const map: Record<number, string> = { 0: '#60a5fa', 1: '#22d3ee', 2: '#a78bfa' }
  return map[latest.value.action] || '#94a3b8'
})

const barColor = (value: number) => {
  const ratio = value / maxValue.value
  if (ratio > 0.66) return 'var(--accent-cyan)'
  if (ratio > 0.33) return 'var(--accent-blue)'
  return 'var(--accent-purple)'
}

// ============ 方法 ============
const fetchData = async () => {
  loading.value = true
  try {
    const res = await fetch('/api/explainability/latest')
    if (!res.ok) return
    const data = await res.json()
    if (!data.empty && data.latest) {
      latest.value = data.latest
    }
  } catch (err) {
    console.debug('explainability/latest 接口暂不可用:', err)
  } finally {
    loading.value = false
  }
}

const toggleAutoRefresh = () => {
  autoRefresh.value = !autoRefresh.value
  if (autoRefresh.value) {
    startAutoRefresh()
  } else {
    stopAutoRefresh()
  }
}

const startAutoRefresh = () => {
  stopAutoRefresh()
  refreshInterval.value = window.setInterval(fetchData, 3000)
}

const stopAutoRefresh = () => {
  if (refreshInterval.value !== null) {
    window.clearInterval(refreshInterval.value)
    refreshInterval.value = null
  }
}

// ============ 生命周期 ============
onMounted(() => {
  fetchData()
  if (autoRefresh.value) startAutoRefresh()
})

onUnmounted(() => {
  stopAutoRefresh()
})
</script>

<template>
  <div class="panel">
    <div class="panel-header">
      <h2>决策放大镜</h2>
      <div style="display: flex; align-items: center; gap: 10px;">
        <span v-if="latest" class="badge">{{ latest.step }}</span>
        <button class="btn btn-secondary btn-sm" :disabled="loading" @click="fetchData">
          {{ loading ? '...' : '刷新' }}
        </button>
        <button
          :class="['btn', 'btn-sm', autoRefresh ? 'btn-primary' : 'btn-secondary']"
          @click="toggleAutoRefresh"
        >
          {{ autoRefresh ? '自动' : '手动' }}
        </button>
      </div>
    </div>
    <div class="panel-body">
      <div v-if="!latest" class="empty-hint">
        暂无决策数据<br />
        （等待 PPO 推理生成决策记录）
      </div>

      <div v-else class="magnifier-content">
        <!-- 决策摘要 -->
        <div class="decision-summary">
          <div class="summary-item">
            <span class="summary-label">步数</span>
            <span class="summary-value">{{ latest.step }}</span>
          </div>
          <div class="summary-item">
            <span class="summary-label">动作</span>
            <span class="summary-value action-badge" :style="{ background: actionColor + '22', color: actionColor, border: '1px solid ' + actionColor + '66' }">
              {{ actionLabel }}
            </span>
          </div>
          <div class="summary-item">
            <span class="summary-label">奖励</span>
            <span class="summary-value" :style="{ color: latest.reward > 0 ? 'var(--accent-green)' : latest.reward < 0 ? 'var(--accent-red)' : 'var(--text-secondary)' }">
              {{ latest.reward?.toFixed(4) ?? '-' }}
            </span>
          </div>
          <div v-if="latest.episode_reward !== undefined" class="summary-item">
            <span class="summary-label">Episode奖励</span>
            <span class="summary-value">{{ latest.episode_reward?.toFixed(2) }}</span>
          </div>
        </div>

        <!-- 特征贡献度条形图 -->
        <div class="contributions-section">
          <h3 class="section-title">特征贡献度分析</h3>
          <div class="bar-chart">
            <div v-for="feature in topFeatures" :key="feature.name" class="bar-row">
              <div class="bar-label" :title="feature.name">{{ feature.name }}</div>
              <div class="bar-track">
                <div
                  class="bar-fill"
                  :style="{
                    width: (feature.value / maxValue * 100) + '%',
                    background: barColor(feature.value)
                  }"
                >
                  <span class="bar-value">{{ (feature.value * 100).toFixed(1) }}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 解释文本 -->
        <div v-if="latest.explanation_text" class="explanation-section">
          <h3 class="section-title">决策解释</h3>
          <div class="explanation-text">{{ latest.explanation_text }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.magnifier-content {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.decision-summary {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  padding: 12px 16px;
  background: var(--bg-secondary);
  border-radius: 8px;
}

.summary-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.summary-label {
  font-size: 11px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.summary-value {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
}

.action-badge {
  display: inline-block;
  padding: 2px 12px;
  border-radius: 10px;
  font-size: 14px;
  font-weight: 600;
}

.section-title {
  font-size: 13px;
  color: var(--text-secondary);
  margin-bottom: 10px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.bar-chart {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.bar-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.bar-label {
  width: 120px;
  font-size: 12px;
  color: var(--text-secondary);
  text-align: right;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex-shrink: 0;
}

.bar-track {
  flex: 1;
  height: 24px;
  background: var(--bg-secondary);
  border-radius: 4px;
  overflow: hidden;
  position: relative;
}

.bar-fill {
  height: 100%;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 8px;
  transition: width 0.5s ease;
  min-width: 40px;
}

.bar-value {
  font-size: 11px;
  color: #fff;
  font-weight: 600;
}

.explanation-section {
  padding: 12px 16px;
  background: var(--bg-secondary);
  border-radius: 8px;
  border-left: 3px solid var(--accent-cyan);
}

.explanation-text {
  font-size: 13px;
  color: var(--text-primary);
  line-height: 1.6;
}
</style>
