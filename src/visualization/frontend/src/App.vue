<script setup lang="ts">
import { ref, reactive, onMounted, onUnmounted, computed, provide } from 'vue'
import type { Task, SystemStatus, QuotaStatus, WSMessage, Toast, NewTaskForm } from './types'
import TaskQueue from './components/TaskQueue.vue'
import ResourceDashboard from './components/ResourceDashboard.vue'
import DecisionLog from './components/DecisionLog.vue'
import MachineStatus from './components/MachineStatus.vue'
import DecisionMagnifier from './components/DecisionMagnifier.vue'
import BattlePanel from './components/BattlePanel.vue'

// ============ 响应式数据 ============
const loading = ref(true)
const wsConnected = ref(false)
const ws = ref<WebSocket | null>(null)
const wsReconnectTimer = ref<number | null>(null)
const reconnectAttempts = ref(0)

const status = reactive<SystemStatus>({
  qubit_utilization: 0.65,
  queue_length: 5,
  average_wait_time: 12.3,
  completed_tasks: 42,
  current_step: 1024,
  current_strategy: 'DQN-Reward',
  strategy_options: ['DQN-Reward', 'DQN-Latency', 'PPO-Balanced', 'QAOA-Hybrid', 'FCFS'],
  real_machines: [],
  real_submissions: [],
  last_update: new Date().toISOString()
})

const tasks = ref<Task[]>([])
const quota = ref<QuotaStatus | null>(null)
const selectedStrategy = ref('DQN-Reward')

const newTask = reactive<NewTaskForm>({
  user_id: 'user_001',
  task_type: 'quantum',
  priority: 3,
  qubit_count: 10,
  circuit_depth: 100
})

const toast = reactive<Toast>({ show: false, message: '', type: 'success' })

// ============ 计算属性 ============
const realMachineCount = computed(() => status.real_machines.length)
const realSubmissionCount = computed(() => status.real_submissions.length)

// ============ 方法 ============
const showToast = (message: string, type: Toast['type'] = 'success') => {
  toast.message = message
  toast.type = type
  toast.show = true
  window.setTimeout(() => {
    toast.show = false
  }, 3000)
}

const taskTypeLabel = (type: string): string => {
  const map: Record<string, string> = { quantum: '量子', classical: '经典', hybrid: '混合' }
  return map[type] || type
}

const statusLabel = (s: string): string => {
  const map: Record<string, string> = { pending: '等待', running: '运行', completed: '完成', failed: '失败' }
  return map[s] || s
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

// ============ WebSocket 连接（含重连逻辑） ============
const connectWS = () => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = protocol + '//' + window.location.host + '/ws'

  try {
    ws.value = new WebSocket(wsUrl)
  } catch (err) {
    console.error('WebSocket 创建失败:', err)
    scheduleReconnect()
    return
  }

  ws.value.onopen = () => {
    console.log('WebSocket connected')
    wsConnected.value = true
    loading.value = false
    reconnectAttempts.value = 0
  }

  ws.value.onmessage = (event: MessageEvent) => {
    try {
      const msg = JSON.parse(event.data) as WSMessage
      handleWSMessage(msg)
    } catch (err) {
      console.error('解析 WS 消息失败:', err)
    }
  }

  ws.value.onclose = () => {
    console.log('WebSocket disconnected')
    wsConnected.value = false
    scheduleReconnect()
  }

  ws.value.onerror = (error) => {
    console.error('WebSocket error:', error)
  }
}

const scheduleReconnect = () => {
  if (wsReconnectTimer.value !== null) return
  // 指数退避：5s, 10s, 20s ... 上限 60s
  const delay = Math.min(5000 * Math.pow(2, reconnectAttempts.value), 60000)
  reconnectAttempts.value += 1
  wsReconnectTimer.value = window.setTimeout(() => {
    wsReconnectTimer.value = null
    connectWS()
  }, delay)
}

const handleWSMessage = (msg: WSMessage) => {
  switch (msg.type) {
    case 'init':
      if (msg.status) Object.assign(status, msg.status)
      if (msg.tasks) tasks.value = msg.tasks
      // 同步策略
      if (status.current_strategy) selectedStrategy.value = status.current_strategy
      // 拉取配额
      fetchQuota()
      break
    case 'status_update':
      if (msg.status) Object.assign(status, msg.status)
      if (msg.tasks) tasks.value = msg.tasks
      break
    case 'task_added':
      if (msg.task) tasks.value.push(msg.task)
      showToast('新任务已提交: ' + (msg.task?.task_id ?? ''))
      break
    case 'strategy_changed':
      showToast('策略切换: ' + (msg.old_strategy ?? '') + ' → ' + (msg.new_strategy ?? ''))
      if (msg.new_strategy) {
        status.current_strategy = msg.new_strategy
        selectedStrategy.value = msg.new_strategy
      }
      break
  }
}

const switchStrategy = () => {
  if (!wsConnected.value || !ws.value) {
    showToast('WebSocket 未连接，无法切换策略', 'error')
    return
  }
  ws.value.send(
    JSON.stringify({
      action: 'switch_strategy',
      strategy: selectedStrategy.value
    })
  )
}

const submitTask = async () => {
  try {
    const res = await fetch('/api/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newTask)
    })
    if (!res.ok) throw new Error('HTTP ' + res.status)
    const data = await res.json()
    showToast('任务 ' + data.task_id + ' 提交成功')
  } catch (err) {
    showToast('任务提交失败', 'error')
    console.error(err)
  }
}

const fetchQuota = async () => {
  try {
    const res = await fetch('/api/quota')
    if (!res.ok) return
    quota.value = (await res.json()) as QuotaStatus
  } catch (err) {
    // /api/quota 可能尚未实现（由主线程实现），静默忽略
    console.debug('quota 接口暂不可用:', err)
  }
}

// ============ 生命周期 ============
onMounted(() => {
  connectWS()
  // 启动时尝试拉取一次配额状态
  fetchQuota()
})

onUnmounted(() => {
  if (wsReconnectTimer.value !== null) {
    window.clearTimeout(wsReconnectTimer.value)
    wsReconnectTimer.value = null
  }
  if (ws.value) {
    ws.value.onclose = null
    ws.value.close()
  }
})

// ============ provide 给子组件 ============
provide('status', status)
provide('quota', quota)
provide('showToast', showToast)

// 暴露给模板使用
defineExpose({
  taskTypeLabel,
  statusLabel,
  machineStatusLabel,
  machineStatusClass
})
</script>

<template>
  <!-- 加载状态 -->
  <div v-if="loading" class="loading-overlay">
    <div class="loading-spinner"></div>
  </div>

  <!-- Toast 通知 -->
  <Transition name="toast">
    <div v-if="toast.show" :class="['toast', toast.type]">
      {{ toast.message }}
    </div>
  </Transition>

  <!-- 顶部标题栏 -->
  <header class="header">
    <h1>量子RL调度系统 - 监控面板</h1>
    <div :class="['ws-status', wsConnected ? 'connected' : 'disconnected']">
      {{ wsConnected ? '实时连接' : '断开连接' }}
    </div>
  </header>

  <!-- 系统状态卡片 -->
  <div class="status-cards">
    <div class="status-card card-blue">
      <div class="card-label">量子比特利用率</div>
      <div class="card-value">{{ (status.qubit_utilization * 100).toFixed(1) }}%</div>
      <div class="card-sub">实时资源使用</div>
    </div>
    <div class="status-card card-purple">
      <div class="card-label">任务队列长度</div>
      <div class="card-value">{{ status.queue_length }}</div>
      <div class="card-sub">等待调度任务</div>
    </div>
    <div class="status-card card-amber">
      <div class="card-label">平均等待时间</div>
      <div class="card-value">{{ status.average_wait_time.toFixed(1) }}s</div>
      <div class="card-sub">任务延迟指标</div>
    </div>
    <div class="status-card card-green">
      <div class="card-label">已完成任务</div>
      <div class="card-value">{{ status.completed_tasks }}</div>
      <div class="card-sub">总处理数量</div>
    </div>
    <div class="status-card card-cyan">
      <div class="card-label">当前步数</div>
      <div class="card-value">{{ status.current_step.toLocaleString() }}</div>
      <div class="card-sub">调度决策次数</div>
    </div>
  </div>

  <!-- 主内容区域 -->
  <div class="main-content">
    <!-- 资源仪表盘（Issue #22：历史趋势） -->
    <ResourceDashboard />

    <!-- 多机器对比视图（Issue #22） -->
    <MachineStatus :machines="status.real_machines" />

    <!-- 决策过程回放（Issue #22：时间轴滑动） -->
    <DecisionLog />

    <!-- 决策放大镜（Day2-3-10：特征贡献度分析） -->
    <DecisionMagnifier />

    <!-- PPO vs FCFS 实时对战面板（Day4-7-11） -->
    <BattlePanel />

    <!-- 真机状态 + 真机提交记录（天衍云 cqlib 真实数据） -->
    <div class="control-grid">
      <!-- 真机状态卡片 -->
      <div class="panel">
        <div class="panel-header">
          <h2>天衍云真机状态</h2>
          <span class="badge">{{ realMachineCount }} 台</span>
        </div>
        <div class="panel-body" style="max-height: 320px; overflow-y: auto;">
          <div v-if="realMachineCount === 0" class="empty-hint">
            未配置 TIANYAN_API_KEY 或查询失败<br />
            （Mock 模式下不显示真机）
          </div>
          <div v-for="m in status.real_machines" :key="m.name" class="machine-card">
            <div>
              <div class="machine-name">{{ m.name }}</div>
              <div class="machine-meta">类型: {{ m.type }} | ID: {{ m.id }}</div>
            </div>
            <span class="status-badge" :class="machineStatusClass(m.status)">
              {{ machineStatusLabel(m.status) }}
            </span>
          </div>
        </div>
      </div>

      <!-- 真机提交记录 -->
      <div class="panel">
        <div class="panel-header">
          <h2>真机提交记录</h2>
          <span class="badge">{{ realSubmissionCount }} 条</span>
        </div>
        <div class="panel-body" style="max-height: 320px; overflow-y: auto;">
          <div v-if="realSubmissionCount === 0" class="empty-hint">
            暂无真机提交记录<br />
            （训练回调 RealMachineCallback 触发后显示）
          </div>
          <table v-else class="real-table">
            <thead>
              <tr>
                <th>步数</th>
                <th>任务ID</th>
                <th>机器</th>
                <th>耗时(s)</th>
                <th>状态</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(r, i) in status.real_submissions" :key="i">
                <td>{{ r.step }}</td>
                <td>{{ r.task_id }}</td>
                <td>{{ r.machine }}</td>
                <td>{{ r.latency_s }}</td>
                <td>{{ r.status }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- 控制面板和任务队列 -->
    <div class="control-grid">
      <!-- 控制面板 -->
      <div class="panel">
        <div class="panel-header">
          <h2>控制面板</h2>
          <span v-if="quota" class="badge">
            配额: {{ quota.used_quota }}/{{ quota.total_quota }}
          </span>
        </div>
        <div class="panel-body">
          <div class="form-group">
            <label>调度策略</label>
            <select v-model="selectedStrategy">
              <option v-for="s in status.strategy_options" :key="s" :value="s">{{ s }}</option>
            </select>
          </div>
          <button class="btn btn-primary" :disabled="!wsConnected" @click="switchStrategy">
            切换策略
          </button>

          <hr style="border: none; border-top: 1px solid var(--border-color); margin: 20px 0;" />

          <h3 style="font-size: 14px; margin-bottom: 12px; color: var(--text-secondary);">
            提交新任务
          </h3>
          <div class="form-group">
            <label>用户ID</label>
            <input v-model="newTask.user_id" type="text" placeholder="user_001" />
          </div>
          <div class="form-group">
            <label>任务类型</label>
            <select v-model="newTask.task_type">
              <option value="quantum">量子任务</option>
              <option value="classical">经典任务</option>
              <option value="hybrid">混合任务</option>
            </select>
          </div>
          <div class="form-group">
            <label>优先级 (1-5)</label>
            <input v-model.number="newTask.priority" type="number" min="1" max="5" />
          </div>
          <div class="form-group">
            <label>所需量子比特数</label>
            <input v-model.number="newTask.qubit_count" type="number" min="1" />
          </div>
          <div class="form-group">
            <label>电路深度</label>
            <input v-model.number="newTask.circuit_depth" type="number" min="1" />
          </div>
          <button class="btn btn-primary" :disabled="!wsConnected" @click="submitTask">
            提交任务
          </button>
        </div>
      </div>

      <!-- 任务队列（Issue #22：筛选+详情弹窗） -->
      <TaskQueue :tasks="tasks" />
    </div>
  </div>
</template>

<style scoped>
.toast-enter-active,
.toast-leave-active {
  transition: all 0.3s ease;
}
.toast-enter-from,
.toast-leave-to {
  transform: translateX(100%);
  opacity: 0;
}
</style>
