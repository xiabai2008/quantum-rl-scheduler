// 量子RL调度系统 - TypeScript 类型定义

/** 任务状态 */
export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed'

/** 任务类型 */
export type TaskType = 'quantum' | 'classical' | 'hybrid'

/** 任务对象 */
export interface Task {
  task_id: string
  task_type: TaskType
  status: TaskStatus
  priority: number
  qubit_count: number
  circuit_depth: number
  urgency: number
  wait_steps: number
  execution_time: number
  submitted_at: string
  user_id?: string
}

/** 真机信息 */
export interface MachineInfo {
  name: string
  type?: string
  id?: string
  status: string
  total_qubits?: number
  available_ratio?: number
  fidelity?: number
  queue_depth?: number
  single_gate_fidelity?: number
  two_gate_fidelity?: number
}

/** 真机提交记录 */
export interface RealSubmission {
  step: number
  task_id: string
  machine: string
  latency_s: number
  status: string
}

/** 系统状态（来自 /api/status 或 WS init） */
export interface SystemStatus {
  qubit_utilization: number
  queue_length: number
  average_wait_time: number
  completed_tasks: number
  current_step: number
  current_strategy: string
  strategy_options: string[]
  real_machines: MachineInfo[]
  real_submissions: RealSubmission[]
  last_update: string
}

/** 配额状态（来自即将新增的 /api/quota） */
export interface QuotaStatus {
  total_quota: number
  used_quota: number
  remaining_quota: number
  reset_at?: string
  daily_limit?: number
  daily_used?: number
}

/** 资源历史趋势单点（来自即将新增的 /api/resource-history） */
export interface ResourceHistoryPoint {
  step: number
  qubit_utilization: number
  queue_length: number
  completed_tasks: number
  average_wait_time: number
}

/** 决策来源标识 */
export type DecisionSource = 'ppo' | 'dqn' | 'qaoa' | 'fcfs' | 'annealing' | string

/** 决策记录（来自即将新增的 /api/decision-log） */
export interface DecisionRecord {
  step: number
  task_id: string
  action: string | number
  action_label: string
  reward: number
  source: DecisionSource
  q_value?: number
  confidence?: number
  timestamp?: string
}

/** 多机器对比项（来自即将新增的 /api/machines-comparison） */
export interface MachineComparisonItem {
  name: string
  total_qubits: number
  available_ratio: number
  fidelity: number
  queue_depth: number
  status: string
  single_gate_fidelity: number
  two_gate_fidelity: number
}

/** WebSocket 消息类型 */
export type WSMessageType =
  | 'init'
  | 'status_update'
  | 'task_added'
  | 'strategy_changed'

/** WebSocket 消息载荷 */
export interface WSMessage {
  type: WSMessageType
  status?: Partial<SystemStatus>
  tasks?: Task[]
  task?: Task
  old_strategy?: string
  new_strategy?: string
}

/** Toast 通知 */
export interface Toast {
  show: boolean
  message: string
  type: 'success' | 'error' | 'info'
}

/** 提交新任务表单 */
export interface NewTaskForm {
  user_id: string
  task_type: TaskType
  priority: number
  qubit_count: number
  circuit_depth: number
}
