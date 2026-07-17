<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import type { DecisionRecord } from '../types'

// ============ 响应式数据 ============
const records = ref<DecisionRecord[]>([])
const currentIndex = ref(0) // 时间轴当前位置（数组索引）
const isPlaying = ref(false)
const playSpeed = ref(1000) // 每步间隔 ms
let playTimer: number | null = null
const loading = ref(false)

// ============ 计算属性 ============
const totalSteps = computed(() => records.value.length)
const currentRecord = computed<DecisionRecord | null>(() => {
  if (records.value.length === 0) return null
  return records.value[currentIndex.value] ?? null
})

const sourceLabel = (source: string): string => {
  const map: Record<string, string> = {
    ppo: 'PPO',
    dqn: 'DQN',
    qaoa: 'QAOA',
    fcfs: 'FCFS',
    annealing: '量子退火'
  }
  return map[(source || '').toLowerCase()] || source || '-'
}

const sourceColor = (source: string): string => {
  const map: Record<string, string> = {
    ppo: '#60a5fa',
    dqn: '#a78bfa',
    qaoa: '#22d3ee',
    fcfs: '#94a3b8',
    annealing: '#fbbf24'
  }
  return map[(source || '').toLowerCase()] || '#e2e8f0'
}

const rewardColor = (reward: number): string => {
  if (reward > 0) return 'var(--accent-green)'
  if (reward < 0) return 'var(--accent-red)'
  return 'var(--text-secondary)'
}

// ============ 方法 ============
const fetchRecords = async () => {
  loading.value = true
  try {
    const res = await fetch('/api/decision-log')
    if (!res.ok) return
    const data = (await res.json()) as DecisionRecord[]
    if (Array.isArray(data)) {
      records.value = data
      if (data.length > 0) {
        currentIndex.value = Math.min(currentIndex.value, data.length - 1)
      } else {
        currentIndex.value = 0
      }
    }
  } catch (err) {
    console.debug('decision-log 接口暂不可用:', err)
  } finally {
    loading.value = false
  }
}

const onSliderChange = (e: Event) => {
  const target = e.target as HTMLInputElement
  currentIndex.value = Math.max(0, Math.min(records.value.length - 1, +target.value))
}

const stepPrev = () => {
  if (currentIndex.value > 0) currentIndex.value -= 1
}

const stepNext = () => {
  if (currentIndex.value < records.value.length - 1) currentIndex.value += 1
}

const jumpToStart = () => {
  currentIndex.value = 0
}

const jumpToEnd = () => {
  if (records.value.length > 0) currentIndex.value = records.value.length - 1
}

const togglePlay = () => {
  if (records.value.length === 0) return
  if (isPlaying.value) {
    pause()
  } else {
    play()
  }
}

const play = () => {
  isPlaying.value = true
  // 如果到末尾，从头开始
  if (currentIndex.value >= records.value.length - 1) {
    currentIndex.value = 0
  }
  playTimer = window.setInterval(() => {
    if (currentIndex.value < records.value.length - 1) {
      currentIndex.value += 1
    } else {
      pause()
    }
  }, playSpeed.value)
}

const pause = () => {
  isPlaying.value = false
  if (playTimer !== null) {
    window.clearInterval(playTimer)
    playTimer = null
  }
}

const changeSpeed = (e: Event) => {
  playSpeed.value = +(e.target as HTMLSelectElement).value
  if (isPlaying.value) {
    pause()
    play()
  }
}

// ============ 生命周期 ============
onMounted(() => {
  fetchRecords()
})

onUnmounted(() => {
  pause()
})
</script>

<template>
  <div class="panel">
    <div class="panel-header">
      <h2>决策过程回放</h2>
      <div style="display: flex; align-items: center; gap: 10px;">
        <span class="badge">{{ totalSteps }} 条决策</span>
        <select
          :value="playSpeed"
          class="speed-select"
          @change="changeSpeed"
        >
          <option :value="2000">0.5x</option>
          <option :value="1000">1x</option>
          <option :value="500">2x</option>
          <option :value="250">4x</option>
        </select>
        <button class="btn btn-secondary btn-sm" :disabled="loading" @click="fetchRecords">
          {{ loading ? '加载中...' : '刷新' }}
        </button>
      </div>
    </div>
    <div class="panel-body">
      <div v-if="totalSteps === 0" class="empty-hint">
        暂无决策记录<br />
        （数据来源：/api/decision-log，可能尚未实现或调度未启动）
      </div>

      <div v-else>
        <!-- 时间轴控制条 -->
        <div class="timeline-bar">
          <button class="btn btn-secondary btn-sm" title="跳到开头" @click="jumpToStart">⏮</button>
          <button class="btn btn-secondary btn-sm" title="上一步" @click="stepPrev">◀</button>
          <button
            class="btn btn-primary btn-sm"
            style="min-width: 70px;"
            :title="isPlaying ? '暂停' : '播放'"
            @click="togglePlay"
          >
            {{ isPlaying ? '⏸ 暂停' : '▶ 播放' }}
          </button>
          <button class="btn btn-secondary btn-sm" title="下一步" @click="stepNext">▶</button>
          <button class="btn btn-secondary btn-sm" title="跳到末尾" @click="jumpToEnd">⏭</button>

          <input
            type="range"
            class="timeline-range"
            min="0"
            :max="Math.max(totalSteps - 1, 0)"
            :value="currentIndex"
            @input="onSliderChange"
          />

          <div class="timeline-step">
            <span class="step-current">{{ currentIndex + 1 }}</span>
            / {{ totalSteps }}
          </div>
        </div>

        <!-- 当前决策详情 -->
        <div v-if="currentRecord" class="decision-detail">
          <div class="decision-row">
            <span class="label">步数</span>
            <span class="value">{{ currentRecord.step }}</span>
          </div>
          <div class="decision-row">
            <span class="label">任务 ID</span>
            <span class="value">{{ currentRecord.task_id }}</span>
          </div>
          <div class="decision-row">
            <span class="label">动作 (Action)</span>
            <span class="value">{{ currentRecord.action }}</span>
          </div>
          <div class="decision-row">
            <span class="label">动作含义</span>
            <span class="value">{{ currentRecord.action_label }}</span>
          </div>
          <div class="decision-row">
            <span class="label">决策来源</span>
            <span class="value">
              <span
                class="source-tag"
                :style="{
                  background: sourceColor(currentRecord.source) + '22',
                  color: sourceColor(currentRecord.source),
                  border: '1px solid ' + sourceColor(currentRecord.source) + '66'
                }"
              >
                {{ sourceLabel(currentRecord.source) }}
              </span>
            </span>
          </div>
          <div class="decision-row">
            <span class="label">奖励 (Reward)</span>
            <span class="value" :style="{ color: rewardColor(currentRecord.reward) }">
              {{ currentRecord.reward?.toFixed(4) ?? '-' }}
            </span>
          </div>
          <div v-if="currentRecord.q_value !== undefined" class="decision-row">
            <span class="label">Q 值</span>
            <span class="value">{{ currentRecord.q_value?.toFixed(4) ?? '-' }}</span>
          </div>
          <div v-if="currentRecord.confidence !== undefined" class="decision-row">
            <span class="label">置信度</span>
            <span class="value">
              {{ (currentRecord.confidence * 100).toFixed(1) }}%
            </span>
          </div>
          <div v-if="currentRecord.timestamp" class="decision-row">
            <span class="label">时间戳</span>
            <span class="value">{{ currentRecord.timestamp }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.speed-select {
  padding: 4px 8px;
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  color: var(--text-primary);
  font-size: 12px;
}
.source-tag {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 10px;
  font-size: 12px;
  font-weight: 600;
}
</style>
