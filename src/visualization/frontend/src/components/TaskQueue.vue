<script setup lang="ts">
import { ref, computed } from 'vue'
import type { Task, TaskStatus } from '../types'

interface Props {
  tasks: Task[]
}
const props = defineProps<Props>()

// ============ 筛选 ============
type FilterKey = 'all' | TaskStatus
const filter = ref<FilterKey>('all')

const filterOptions: { key: FilterKey; label: string }[] = [
  { key: 'all', label: '全部' },
  { key: 'pending', label: '等待中' },
  { key: 'running', label: '运行中' },
  { key: 'completed', label: '已完成' },
  { key: 'failed', label: '失败' }
]

const filteredTasks = computed<Task[]>(() => {
  if (filter.value === 'all') return props.tasks
  return props.tasks.filter((t) => t.status === filter.value)
})

const setFilter = (key: FilterKey) => {
  filter.value = key
}

// ============ 任务详情弹窗 ============
const selectedTask = ref<Task | null>(null)
const showModal = ref(false)

const openDetail = (task: Task) => {
  selectedTask.value = task
  showModal.value = true
}

const closeDetail = () => {
  showModal.value = false
  selectedTask.value = null
}

// ============ 标签映射 ============
const taskTypeLabel = (type: string): string => {
  const map: Record<string, string> = { quantum: '量子', classical: '经典', hybrid: '混合' }
  return map[type] || type
}

const statusLabel = (s: string): string => {
  const map: Record<string, string> = { pending: '等待', running: '运行', completed: '完成', failed: '失败' }
  return map[s] || s
}

const formatTime = (iso: string): string => {
  if (!iso) return '-'
  try {
    return new Date(iso).toLocaleString('zh-CN')
  } catch {
    return iso
  }
}

// ============ 弹窗背景点击关闭 ============
const onOverlayClick = (e: MouseEvent) => {
  if (e.target === e.currentTarget) closeDetail()
}
</script>

<template>
  <div class="panel">
    <div class="panel-header">
      <h2>任务队列</h2>
      <span class="badge">{{ filteredTasks.length }} / {{ tasks.length }} 个任务</span>
    </div>
    <div class="panel-body">
      <!-- 筛选按钮组 -->
      <div class="filter-group" style="margin-bottom: 14px;">
        <button
          v-for="opt in filterOptions"
          :key="opt.key"
          :class="['filter-btn', { active: filter === opt.key }]"
          @click="setFilter(opt.key)"
        >
          {{ opt.label }}
        </button>
      </div>

      <div style="max-height: 360px; overflow-y: auto;">
        <table class="task-table">
          <thead>
            <tr>
              <th>任务ID</th>
              <th>类型</th>
              <th>状态</th>
              <th>优先级</th>
              <th>比特数</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="task in filteredTasks" :key="task.task_id" @click="openDetail(task)">
              <td>{{ task.task_id }}</td>
              <td>{{ taskTypeLabel(task.task_type) }}</td>
              <td>
                <span :class="['status-tag', task.status]">{{ statusLabel(task.status) }}</span>
              </td>
              <td>{{ task.priority }}</td>
              <td>{{ task.qubit_count }}</td>
            </tr>
            <tr v-if="filteredTasks.length === 0">
              <td colspan="5" class="empty-hint">暂无匹配任务</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p class="task-tip">提示：点击任意一行查看任务完整详情</p>
    </div>

    <!-- 详情弹窗 -->
    <Transition name="modal">
      <div v-if="showModal && selectedTask" class="modal-overlay" @click="onOverlayClick">
        <div class="modal">
          <div class="modal-header">
            <h3>任务详情 - {{ selectedTask.task_id }}</h3>
            <button class="modal-close" aria-label="关闭" @click="closeDetail">×</button>
          </div>
          <div class="modal-body">
            <div class="detail-grid">
              <div class="detail-item">
                <div class="detail-label">任务 ID</div>
                <div class="detail-value">{{ selectedTask.task_id }}</div>
              </div>
              <div class="detail-item">
                <div class="detail-label">任务类型</div>
                <div class="detail-value">{{ taskTypeLabel(selectedTask.task_type) }}</div>
              </div>
              <div class="detail-item">
                <div class="detail-label">状态</div>
                <div class="detail-value">
                  <span :class="['status-tag', selectedTask.status]">
                    {{ statusLabel(selectedTask.status) }}
                  </span>
                </div>
              </div>
              <div class="detail-item">
                <div class="detail-label">优先级</div>
                <div class="detail-value">{{ selectedTask.priority }} / 5</div>
              </div>
              <div class="detail-item">
                <div class="detail-label">量子比特数</div>
                <div class="detail-value">{{ selectedTask.qubit_count }}</div>
              </div>
              <div class="detail-item">
                <div class="detail-label">电路深度</div>
                <div class="detail-value">{{ selectedTask.circuit_depth }}</div>
              </div>
              <div class="detail-item">
                <div class="detail-label">紧急度</div>
                <div class="detail-value">{{ selectedTask.urgency?.toFixed(3) ?? '-' }}</div>
              </div>
              <div class="detail-item">
                <div class="detail-label">已等待步数</div>
                <div class="detail-value">{{ selectedTask.wait_steps ?? '-' }}</div>
              </div>
              <div class="detail-item">
                <div class="detail-label">执行耗时(s)</div>
                <div class="detail-value">{{ selectedTask.execution_time?.toFixed(2) ?? '-' }}</div>
              </div>
              <div class="detail-item">
                <div class="detail-label">用户 ID</div>
                <div class="detail-value">{{ selectedTask.user_id ?? '-' }}</div>
              </div>
              <div class="detail-item" style="grid-column: 1 / -1;">
                <div class="detail-label">提交时间</div>
                <div class="detail-value">{{ formatTime(selectedTask.submitted_at) }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.task-tip {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 10px;
  text-align: center;
}
.modal-enter-active,
.modal-leave-active {
  transition: opacity 0.2s ease;
}
.modal-enter-from,
.modal-leave-to {
  opacity: 0;
}
</style>
