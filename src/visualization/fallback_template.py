"""
内置回退 HTML 模板 (v2 增强版)

当 frontend/dist/index.html 不存在时，FastAPI 会回退到本模块的 HTML_TEMPLATE。
这是一个原生 HTML/CSS/JS 实现的完整监控面板，零外部依赖，包含：
- 系统状态卡片（6个）
- PPO vs FCFS 实时对比面板（Canvas 累积奖励曲线 + 自动对战）
- 资源利用率趋势图（Canvas 双Y轴折线图）
- 决策日志面板（最近决策时间轴）
- 特征贡献度（最新决策可解释性条形图）
- 任务队列表格
- 控制面板（提交任务 + 策略切换）
- WebSocket 实时推送 + HTTP 轮询兜底
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>量子RL智能调度系统 - Demo Dashboard</title>
    <style>
        /* ===== 量子深空主题 (Quantum Deep Space) - Cherenkov Design ===== */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            /* 品牌主色 - Cherenkov量子青 */
            --brand: #3bc9db;
            --brand-strong: #22a8b8;
            --brand-dim: rgba(59, 201, 219, 0.14);
            --brand-glow: rgba(59, 201, 219, 0.32);
            --brand-fg: #04161a;

            /* 深空底色层级 - 面层叠而非靠阴影 */
            --bg: #070b14;
            --surface: #0e1422;
            --surface-2: #161f30;
            --surface-3: #1d2840;

            /* 文字层级 */
            --ink: #e6edf3;
            --ink-2: rgba(230, 237, 243, 0.72);
            --ink-3: rgba(230, 237, 243, 0.46);
            --ink-4: rgba(230, 237, 243, 0.28);

            /* 分隔线/边框 */
            --line: rgba(148, 184, 220, 0.10);
            --line-strong: rgba(148, 184, 220, 0.18);

            /* 状态语义色（仅用于真实状态） */
            --success: #34d399;
            --warning: #fbbf24;
            --error: #f87171;
            --info: #3bc9db;

            /* 圆角阶梯 */
            --r-sm: 4px;
            --r-md: 8px;
            --r-lg: 14px;
            --r-pill: 999px;

            /* 阴影（深色模式专用，克制） */
            --shadow-1: 0 1px 2px rgba(0, 0, 0, 0.4), 0 1px 1px rgba(0, 0, 0, 0.2);
            --shadow-2: 0 10px 28px -10px rgba(0, 0, 0, 0.6);

            /* 字体 */
            --font-sans: "Inter", "HarmonyOS Sans SC", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
            --font-mono: "JetBrains Mono", "SF Mono", "Cascadia Code", Consolas, monospace;

            /* 兼容旧CSS变量名 */
            --bg-primary: var(--bg);
            --bg-secondary: var(--surface);
            --bg-card: var(--surface);
            --bg-card-alt: var(--surface-2);
            --bg-input: var(--surface-2);
            --border: var(--line);
            --border-strong: var(--line-strong);
            --text-primary: var(--ink);
            --text-secondary: var(--ink-2);
            --text-muted: var(--ink-3);
            --accent: var(--brand);
            --accent-dark: var(--brand-strong);
            --accent-light: var(--brand);
            --green: var(--success);
            --green-light: var(--success);
            --amber-warn: var(--warning);
            --red: var(--error);
            --cyan: var(--brand);
            --cyan-light: var(--brand);
            --blue-slate: var(--brand-dim);
        }
        body {
            font-family: var(--font-sans);
            background: var(--bg);
            color: var(--ink);
            min-height: 100vh;
            overflow-x: hidden;
            font-size: 13px;
            -webkit-font-smoothing: antialiased;
            position: relative;
        }

        /* ===== 量子粒子背景 ===== */
        #quantum-bg {
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            z-index: 0;
            pointer-events: none;
            opacity: 0.6;
        }
        .bg-vignette {
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            z-index: 0;
            pointer-events: none;
            background:
                radial-gradient(ellipse 80% 50% at 50% -20%, rgba(59,201,219,0.06) 0%, transparent 60%),
                radial-gradient(ellipse 60% 40% at 80% 80%, rgba(167,139,250,0.04) 0%, transparent 50%);
        }

        /* ===== 顶部标题栏 ===== */
        .header {
            position: relative;
            z-index: 10;
            background: linear-gradient(180deg, rgba(14,20,34,0.98) 0%, rgba(14,20,34,0.92) 100%);
            border-bottom: 1px solid var(--line);
            padding: 14px 28px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            backdrop-filter: blur(12px);
        }
        .header::after {
            content: '';
            position: absolute;
            bottom: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, rgba(59,201,219,0.3) 50%, transparent 100%);
        }
        .header-left { display: flex; align-items: center; gap: 14px; }
        .logo {
            width: 42px; height: 42px;
            background: linear-gradient(135deg, rgba(59,201,219,0.2) 0%, rgba(59,201,219,0.05) 100%);
            border: 1px solid rgba(59,201,219,0.3);
            border-radius: var(--r-md);
            display: flex; align-items: center; justify-content: center;
            position: relative;
            box-shadow: 0 0 16px rgba(59,201,219,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
        }
        .logo svg {
            width: 24px; height: 24px;
            color: var(--brand);
            animation: logoPulse 4s ease-in-out infinite;
        }
        @keyframes logoPulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.75; }
        }
        .logo .electron {
            transform-origin: 12px 12px;
            animation: electronOrbit 6s linear infinite;
        }
        .logo .electron-2 {
            animation-duration: 8s;
            animation-direction: reverse;
        }
        @keyframes electronOrbit {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        .header-titles h1 {
            font-size: 18px; font-weight: 600;
            font-family: var(--font-sans);
            color: var(--ink);
            line-height: 1.3;
            letter-spacing: -0.2px;
        }
        .header-titles .subtitle {
            font-size: 10px; color: var(--ink-3);
            margin-top: 2px;
            font-family: var(--font-mono);
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .header-right { display: flex; align-items: center; gap: 10px; }
        .ws-status {
            font-size: 11px; padding: 5px 12px;
            border-radius: var(--r-pill);
            background: var(--surface-2);
            border: 1px solid var(--line);
            display: flex; align-items: center; gap: 6px;
            font-family: var(--font-mono);
            letter-spacing: 0.3px;
            color: var(--ink-3);
            position: relative;
        }
        .ws-status::before {
            content: ''; width: 6px; height: 6px;
            border-radius: 50%;
            background: var(--error);
            flex-shrink: 0;
        }
        .ws-status::after {
            content: '';
            position: absolute;
            left: 12px; top: 50%;
            width: 6px; height: 6px;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
        }
        .ws-status.connected::before { background: var(--success); }
        .ws-status.connected::after {
            background: var(--success);
            animation: statusPulse 2s ease-out infinite;
        }
        @keyframes statusPulse {
            0% { transform: translate(-50%, -50%) scale(1); opacity: 0.6; }
            100% { transform: translate(-50%, -50%) scale(2.5); opacity: 0; }
        }
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        .ws-status.disconnected::before { background: var(--error); }
        .ws-status.connected { color: var(--success); border-color: rgba(52,211,153,0.25); background: rgba(52,211,153,0.08); }
        .ws-status.disconnected { color: var(--error); border-color: rgba(248,113,113,0.2); }
        .model-badge {
            font-size: 10px; padding: 5px 12px;
            border-radius: var(--r-pill);
            background: linear-gradient(135deg, rgba(59,201,219,0.15) 0%, rgba(59,201,219,0.05) 100%);
            border: 1px solid rgba(59,201,219,0.25);
            color: var(--brand);
            font-family: var(--font-mono);
            font-weight: 600;
            letter-spacing: 0.5px;
            box-shadow: 0 0 10px rgba(59,201,219,0.08);
        }

        /* ===== 状态卡片区域 ===== */
        .status-cards {
            position: relative; z-index: 1;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            padding: 20px 28px 0;
            background: var(--bg);
        }
        .status-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: var(--r-lg);
            padding: 18px 20px;
            transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.25s ease;
            position: relative;
            overflow: hidden;
        }
        .status-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
            opacity: 0;
            transition: opacity 0.25s;
        }
        .status-card:hover {
            border-color: var(--line-strong);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px -8px rgba(0,0,0,0.4), 0 0 0 1px var(--line-strong);
        }
        .status-card:hover::before { opacity: 1; }
        /* 激活/高亮卡片（量子比特利用率）使用品牌青边框+微glow */
        .card-blue { border-color: rgba(59,201,219,0.3); box-shadow: 0 0 0 1px rgba(59,201,219,0.1), 0 0 20px rgba(59,201,219,0.06); }
        .card-blue:hover { box-shadow: 0 8px 24px -8px rgba(0,0,0,0.4), 0 0 0 1px rgba(59,201,219,0.3), 0 0 24px rgba(59,201,219,0.12); }
        .card-purple { border-color: var(--line); }
        .card-green { border-color: var(--line); }
        .card-amber { border-color: var(--line); }
        .card-cyan { border-color: var(--line); }
        .card-pink { border-color: var(--line); }
        .card-label {
            font-size: 10px; color: var(--ink-3);
            margin-bottom: 10px;
            font-family: var(--font-mono);
            letter-spacing: 0.8px;
            text-transform: uppercase;
            display: flex; align-items: center; gap: 6px;
        }
        .card-label .card-icon {
            width: 14px; height: 14px;
            display: flex; align-items: center; justify-content: center;
            flex-shrink: 0;
        }
        .card-label .card-icon svg {
            width: 12px; height: 12px;
            stroke: currentColor;
            fill: none;
            stroke-width: 1.8;
            stroke-linecap: round;
            stroke-linejoin: round;
        }
        .card-blue .card-label { color: var(--brand); }
        .card-blue .card-label .card-icon { color: var(--brand); }
        .card-purple .card-label .card-icon { color: #a78bfa; }
        .card-amber .card-label .card-icon { color: var(--warning); }
        .card-green .card-label .card-icon { color: var(--success); }
        .card-cyan .card-label .card-icon { color: #67e8f9; }
        .card-pink .card-label .card-icon { color: #f472b6; }
        .card-value {
            font-size: 32px; font-weight: 700;
            line-height: 1.1;
            font-variant-numeric: tabular-nums;
            font-family: var(--font-mono);
            color: var(--ink);
            letter-spacing: -1px;
            transition: color 0.3s ease;
        }
        .card-value.flash {
            animation: valueFlash 0.5s ease-out;
        }
        @keyframes valueFlash {
            0% { text-shadow: 0 0 0 transparent; }
            50% { text-shadow: 0 0 12px currentColor; }
            100% { text-shadow: 0 0 0 transparent; }
        }
        .card-unit {
            font-size: 14px; font-weight: 400;
            color: var(--ink-3);
            font-family: var(--font-mono);
            margin-left: 2px;
        }
        /* 高亮卡片的数字用量子青 */
        .card-blue .card-value { color: var(--brand); }
        .card-sub {
            font-size: 11px; color: var(--ink-3);
            margin-top: 8px;
            font-family: var(--font-mono);
        }
        .card-trend {
            display: inline-flex; align-items: center; gap: 3px;
            font-size: 11px; font-weight: 600;
            padding: 2px 8px; border-radius: var(--r-pill);
            margin-top: 6px;
            font-family: var(--font-mono);
        }
        .trend-up { background: rgba(52,211,153,0.1); color: var(--success); }
        .trend-down { background: rgba(248,113,113,0.1); color: var(--error); }

        /* ===== 主内容区域 ===== */
        .main-content {
            position: relative; z-index: 1;
            padding: 16px 28px 28px;
            display: grid;
            grid-template-columns: 1fr 360px;
            gap: 16px;
        }
        .main-col { display: flex; flex-direction: column; gap: 16px; }
        .side-col { display: flex; flex-direction: column; gap: 16px; }

        /* ===== 通用面板样式 ===== */
        .panel {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: var(--r-lg);
            overflow: hidden;
            transition: border-color 0.25s ease, box-shadow 0.25s ease;
        }
        .panel:hover {
            border-color: var(--line-strong);
            box-shadow: 0 4px 20px -8px rgba(0,0,0,0.3);
        }
        .panel-header {
            padding: 14px 18px;
            border-bottom: 1px solid var(--line);
            display: flex; align-items: center; justify-content: space-between;
            background: var(--surface);
            position: relative;
        }
        .panel-header::after {
            content: '';
            position: absolute;
            bottom: 0; left: 18px; right: 18px;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--line-strong), transparent);
            opacity: 0;
            transition: opacity 0.25s;
        }
        .panel:hover .panel-header::after { opacity: 1; }
        .panel-header h2 {
            font-size: 14px; font-weight: 600;
            display: flex; align-items: center; gap: 8px;
            font-family: var(--font-sans);
            color: var(--ink);
            letter-spacing: -0.2px;
        }
        .panel-header h2 .icon {
            width: 28px; height: 28px;
            border-radius: 7px;
            display: flex; align-items: center; justify-content: center;
            flex-shrink: 0;
            position: relative;
            border: 1px solid transparent;
            transition: all 0.2s ease;
        }
        .panel-header h2 .icon::before {
            content: '';
            position: absolute;
            inset: 0;
            border-radius: inherit;
            opacity: 0.5;
        }
        .panel-header h2 .icon svg {
            width: 16px; height: 16px;
            stroke: currentColor;
            fill: none;
            stroke-width: 1.6;
            stroke-linecap: round;
            stroke-linejoin: round;
            position: relative;
            z-index: 1;
        }
        .icon-battle {
            background: linear-gradient(135deg, rgba(59,201,219,0.18) 0%, rgba(59,201,219,0.06) 100%) !important;
            color: var(--brand) !important;
            border-color: rgba(59,201,219,0.25) !important;
            box-shadow: 0 0 12px rgba(59,201,219,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
        }
        .icon-chart {
            background: linear-gradient(135deg, rgba(59,201,219,0.18) 0%, rgba(59,201,219,0.06) 100%) !important;
            color: var(--brand) !important;
            border-color: rgba(59,201,219,0.25) !important;
            box-shadow: 0 0 12px rgba(59,201,219,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
        }
        .icon-log {
            background: linear-gradient(135deg, var(--surface-2) 0%, var(--surface-3) 100%) !important;
            color: var(--ink-3) !important;
            border-color: var(--line-strong) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        .icon-explain {
            background: linear-gradient(135deg, rgba(251,191,36,0.18) 0%, rgba(251,191,36,0.06) 100%) !important;
            color: var(--warning) !important;
            border-color: rgba(251,191,36,0.25) !important;
            box-shadow: 0 0 12px rgba(251,191,36,0.12), inset 0 1px 0 rgba(255,255,255,0.05);
        }
        .icon-queue {
            background: linear-gradient(135deg, rgba(52,211,153,0.18) 0%, rgba(52,211,153,0.06) 100%) !important;
            color: var(--success) !important;
            border-color: rgba(52,211,153,0.25) !important;
            box-shadow: 0 0 12px rgba(52,211,153,0.12), inset 0 1px 0 rgba(255,255,255,0.05);
        }
        .icon-control {
            background: linear-gradient(135deg, var(--surface-2) 0%, var(--surface-3) 100%) !important;
            color: var(--ink-2) !important;
            border-color: var(--line-strong) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        .icon-rank {
            background: linear-gradient(135deg, rgba(59,201,219,0.18) 0%, rgba(59,201,219,0.06) 100%) !important;
            color: var(--brand) !important;
            border-color: rgba(59,201,219,0.25) !important;
            box-shadow: 0 0 12px rgba(59,201,219,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
        }
        .icon-system {
            background: linear-gradient(135deg, var(--surface-2) 0%, var(--surface-3) 100%) !important;
            color: var(--ink-2) !important;
            border-color: var(--line-strong) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        .panel-header .badge {
            font-size: 10px; padding: 3px 10px;
            border-radius: var(--r-pill);
            background: var(--surface-2);
            color: var(--ink-3);
            border: 1px solid var(--line);
            font-family: var(--font-mono);
            letter-spacing: 0.3px;
        }
        .panel-header .badge.live {
            background: rgba(52,211,153,0.1);
            color: var(--success);
            border-color: rgba(52,211,153,0.2);
            position: relative;
            padding-left: 20px;
        }
        .panel-header .badge.live::before {
            content: '';
            position: absolute;
            left: 7px; top: 50%;
            width: 6px; height: 6px;
            background: var(--success);
            border-radius: 50%;
            transform: translateY(-50%);
            box-shadow: 0 0 6px rgba(52,211,153,0.5);
            animation: pulse-dot 2s infinite;
        }
        @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .panel-body { padding: 16px 18px; }

        /* ===== PPO vs FCFS 对战面板 ===== */
        .battle-panel { position: relative; }
        .battle-canvas-wrap {
            position: relative;
            width: 100%;
            background: var(--bg);
            border: 1px solid var(--line);
            border-radius: var(--r-md);
            overflow: hidden;
        }
        #battle-canvas { width: 100%; display: block; }
        .battle-stats {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 1px;
            margin-top: 0;
            background: var(--line);
            border: 1px solid var(--line);
            border-top: none;
            border-radius: 0 0 var(--r-md) var(--r-md);
            overflow: hidden;
        }
        .battle-stat {
            background: var(--surface-2);
            padding: 12px 14px;
            text-align: center;
        }
        .battle-stat .label { font-size: 10px; color: var(--ink-3); margin-bottom: 4px; font-family:var(--font-mono); text-transform:uppercase; letter-spacing:0.8px; }
        .battle-stat .value { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; font-family:var(--font-mono); color: var(--ink); }
        .battle-stat.ppo .value { color: var(--brand); }
        .battle-stat.fcfs .value { color: var(--ink-2); }
        .battle-stat.diff .value { color: var(--success); }
        .battle-controls {
            display: flex; gap: 0; margin-top: 12px;
            align-items: center;
        }
        .battle-controls .btn-sm {
            padding: 6px 14px; font-size: 11px;
            border-radius: var(--r-md); border: 1px solid var(--line);
            background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface-3) 100%);
            color: var(--ink-2);
            cursor: pointer; transition: all 0.2s ease;
            font-weight: 500;
            font-family: var(--font-mono);
            margin-right: 6px;
            position: relative;
        }
        .battle-controls .btn-sm:hover {
            border-color: rgba(59,201,219,0.4);
            color: var(--brand);
            box-shadow: 0 0 10px rgba(59,201,219,0.1);
        }
        .battle-controls .btn-sm.active {
            background: linear-gradient(180deg, var(--brand) 0%, var(--brand-strong) 100%);
            border-color: var(--brand);
            color: var(--brand-fg);
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(59,201,219,0.25), inset 0 1px 0 rgba(255,255,255,0.15);
        }
        .battle-controls .battle-info {
            font-size: 11px; color: var(--ink-3);
            margin-left: auto;
            font-family: var(--font-mono);
        }
        .battle-legend {
            display: flex; gap: 16px; margin-top: 10px;
            font-size: 11px; color: var(--ink-2);
            font-family: var(--font-mono);
        }
        .legend-item { display: flex; align-items: center; gap: 6px; }
        .legend-line { width: 20px; height: 2px; border-radius: 2px; }
        .legend-ppo { background: var(--brand); box-shadow: 0 0 6px rgba(59,201,219,0.3); }
        .legend-fcfs { background: var(--ink-3); }
        .legend-baseline { background: var(--ink-3); border-top: 1px dashed var(--ink-3); height: 0; }

        /* ===== 资源趋势图 ===== */
        .chart-canvas-wrap {
            position: relative; width: 100%;
            background: var(--bg);
            border: 1px solid var(--line);
            border-radius: var(--r-md); overflow: hidden;
        }
        #resource-canvas { width: 100%; display: block; }
        .chart-legend {
            display: flex; flex-wrap: wrap; gap: 16px;
            margin-top: 10px;
            font-size: 11px; color: var(--ink-2);
            font-family: var(--font-mono);
        }

        /* ===== 双栏小面板布局 ===== */
        .two-col-panels {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        /* ===== 饼图图例 ===== */
        .pie-legend-item {
            display: flex; align-items: center; gap: 10px;
            font-size: 12px;
            padding: 6px 0;
        }
        .pie-dot {
            width: 10px; height: 10px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .pie-label {
            flex: 1;
            color: var(--ink-2);
            font-family: var(--font-sans);
        }
        .pie-value {
            font-family: var(--font-mono);
            color: var(--ink);
            font-size: 12px;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
        }

        /* ===== 决策日志 ===== */
        .decision-list {
            max-height: 260px;
            overflow-y: auto;
            padding-right: 4px;
        }
        .decision-list::-webkit-scrollbar { width: 4px; }
        .decision-list::-webkit-scrollbar-track { background: transparent; }
        .decision-list::-webkit-scrollbar-thumb { background: var(--line-strong); border-radius: 2px; }
        .decision-item {
            display: flex; gap: 10px;
            padding: 8px 0;
            border-bottom: 1px solid var(--line);
            font-size: 12px;
        }
        .decision-item:last-child { border-bottom: none; }
        .decision-step {
            min-width: 36px;
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--ink-3);
            padding-top: 2px;
        }
        .decision-content { flex: 1; }
        .decision-action {
            display: inline-block;
            padding: 2px 8px;
            border-radius: var(--r-pill);
            font-weight: 600;
            font-size: 10px;
            margin-right: 6px;
            font-family: var(--font-mono);
        }
        .action-quantum { background: var(--brand-dim); color: var(--brand); }
        .action-classical { background: var(--surface-2); color: var(--ink-2); border: 1px solid var(--line); }
        .action-hybrid { background: rgba(52,211,153,0.1); color: var(--success); }
        .decision-reward {
            font-weight: 600;
            color: var(--success);
            font-variant-numeric: tabular-nums;
            font-family: var(--font-mono);
            font-size: 11px;
        }
        .decision-reward.negative { color: var(--error); }
        .decision-meta {
            color: var(--ink-3);
            font-size: 10px;
            margin-top: 3px;
            font-family: var(--font-mono);
        }

        /* ===== 特征贡献度 ===== */
        .feature-bars { padding: 4px 0; }
        .feature-bar-row {
            display: flex; align-items: center; gap: 10px;
            margin-bottom: 8px;
            font-size: 11px;
        }
        .feature-name {
            min-width: 80px;
            color: var(--ink-2);
            font-size: 10px;
            text-align: right;
            font-family: var(--font-mono);
        }
        .feature-bar-bg {
            flex: 1; height: 14px;
            background: var(--bg);
            border-radius: var(--r-sm);
            overflow: hidden;
            position: relative;
            border: 1px solid var(--line);
        }
        .feature-bar-fill {
            height: 100%;
            border-radius: var(--r-sm);
            transition: width 0.4s ease;
            position: relative;
        }
        .feature-bar-fill.positive {
            background: var(--success);
        }
        .feature-bar-fill.negative {
            background: var(--brand);
        }
        .feature-value {
            min-width: 48px;
            text-align: right;
            font-size: 10px;
            font-variant-numeric: tabular-nums;
            color: var(--ink-3);
            font-family: var(--font-mono);
        }

        /* ===== 任务队列表格 ===== */
        .task-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        .task-table th {
            text-align: left;
            padding: 10px 14px;
            color: var(--ink-3);
            font-weight: 600;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            border-bottom: 1px solid var(--line-strong);
            white-space: nowrap;
            font-family: var(--font-mono);
            background: var(--surface-2);
        }
        .task-table th:first-child { border-radius: var(--r-md) 0 0 0; }
        .task-table th:last-child { border-radius: 0 var(--r-md) 0 0; }
        .task-table td {
            padding: 10px 14px;
            border-bottom: 1px solid var(--line);
            font-family: var(--font-mono);
            font-size: 12px;
            color: var(--ink);
        }
        .task-table tbody tr { transition: background 0.15s; }
        .task-table tbody tr:hover { background: var(--surface-2); }
        .task-table-wrap {
            max-height: 300px;
            overflow-y: auto;
            border-radius: var(--r-md);
        }
        .task-table-wrap::-webkit-scrollbar { width: 4px; }
        .task-table-wrap::-webkit-scrollbar-track { background: transparent; }
        .task-table-wrap::-webkit-scrollbar-thumb { background: var(--line-strong); border-radius: 2px; }
        .status-tag {
            display: inline-block;
            padding: 2px 10px;
            border-radius: var(--r-pill);
            font-size: 10px;
            font-weight: 600;
            font-family: var(--font-mono);
        }
        .status-tag.pending { background: rgba(251,191,36,0.1); color: var(--warning); }
        .status-tag.running { background: var(--brand-dim); color: var(--brand); }
        .status-tag.completed { background: rgba(52,211,153,0.1); color: var(--success); }
        .status-tag.failed { background: rgba(248,113,113,0.1); color: var(--error); }
        .priority-high { color: var(--error); font-weight: 700; }
        .priority-medium { color: var(--warning); font-weight: 600; }
        .priority-low { color: var(--success); }

        /* ===== 控制面板 ===== */
        .control-grid {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .control-section h3 {
            font-size: 10px;
            color: var(--ink-3);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            font-weight: 600;
            font-family: var(--font-mono);
            padding-bottom: 8px;
            border-bottom: 1px solid var(--line);
        }
        .form-group { margin-bottom: 10px; }
        .form-group label {
            display: block;
            font-size: 11px;
            color: var(--ink-3);
            margin-bottom: 5px;
            font-family: var(--font-sans);
        }
        .form-group input, .form-group select {
            width: 100%;
            padding: 8px 12px;
            background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface) 100%);
            border: 1px solid var(--line);
            border-radius: var(--r-md);
            color: var(--ink);
            font-size: 13px;
            outline: none;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
            font-family: var(--font-mono);
        }
        .form-group input:focus, .form-group select:focus {
            border-color: rgba(59,201,219,0.5);
            box-shadow: 0 0 0 3px rgba(59,201,219,0.1), inset 0 1px 0 rgba(255,255,255,0.03);
        }
        .form-group input::placeholder { color: var(--ink-4); }
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .btn {
            padding: 10px 20px;
            border: 1px solid transparent;
            border-radius: var(--r-md);
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            font-family: var(--font-sans);
            letter-spacing: 0.2px;
            position: relative;
        }
        .btn-primary {
            background: linear-gradient(180deg, var(--brand) 0%, var(--brand-strong) 100%);
            color: var(--brand-fg);
            width: 100%;
            box-shadow: 0 2px 8px rgba(59,201,219,0.2), inset 0 1px 0 rgba(255,255,255,0.15);
        }
        .btn-primary:hover {
            background: linear-gradient(180deg, #4dd5e7 0%, var(--brand) 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(59,201,219,0.3), inset 0 1px 0 rgba(255,255,255,0.2);
        }
        .btn-primary:active { transform: translateY(0); box-shadow: 0 1px 4px rgba(59,201,219,0.2); }
        .strategy-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        .strategy-btn {
            padding: 6px 14px;
            background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface-3) 100%);
            border: 1px solid var(--line);
            border-radius: var(--r-md);
            color: var(--ink-2);
            font-size: 11px;
            cursor: pointer;
            transition: all 0.2s ease;
            font-weight: 500;
            font-family: var(--font-mono);
        }
        .strategy-btn:hover {
            border-color: rgba(59,201,219,0.4);
            color: var(--brand);
            box-shadow: 0 0 8px rgba(59,201,219,0.1);
        }
        .strategy-btn.active {
            background: linear-gradient(180deg, var(--brand) 0%, var(--brand-strong) 100%);
            border-color: var(--brand);
            color: var(--brand-fg);
            font-weight: 700;
            box-shadow: 0 2px 8px rgba(59,201,219,0.25), inset 0 1px 0 rgba(255,255,255,0.15);
        }
        .strategy-btn.recommended {
            border-color: rgba(59,201,219,0.3);
            position: relative;
        }

        /* ===== 策略排名条 ===== */
        .ranking-bar {
            display: flex; align-items: center; gap: 10px;
            margin-bottom: 8px;
            font-size: 11px;
        }
        .ranking-pos {
            width: 22px; text-align: center;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            font-family: var(--font-mono);
            font-size: 12px;
            color: var(--ink-3);
        }
        .ranking-pos.gold { color: var(--brand); font-weight: 800; }
        .ranking-name { min-width: 60px; color: var(--ink-2); font-family:var(--font-mono); font-size:11px; }
        .ranking-bar-bg {
            flex: 1; height: 16px;
            background: var(--bg);
            border-radius: var(--r-sm);
            overflow: hidden;
            border: 1px solid var(--line);
        }
        .ranking-bar-fill {
            height: 100%;
            border-radius: var(--r-sm);
            background: var(--surface-3);
            transition: width 0.5s;
        }
        .ranking-bar-fill.best {
            background: var(--brand);
            box-shadow: 0 0 8px rgba(59,201,219,0.2);
        }
        .ranking-score {
            min-width: 52px; text-align: right;
            font-variant-numeric: tabular-nums;
            color: var(--ink);
            font-family: var(--font-mono);
            font-weight: 600;
            font-size: 11px;
        }

        /* ===== Toast ===== */
        .toast-container {
            position: fixed;
            top: 70px; right: 24px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .toast {
            padding: 12px 16px;
            background: var(--surface);
            border: 1px solid var(--line);
            border-left: 3px solid var(--brand);
            border-radius: var(--r-md);
            font-size: 12px;
            box-shadow: var(--shadow-2);
            max-width: 300px;
            font-family: var(--font-mono);
            color: var(--ink);
        }
        .toast.success { border-left-color: var(--success); }
        .toast.info { border-left-color: var(--brand); }
        .toast.warn { border-left-color: var(--warning); }

        /* ===== 加载动画 ===== */
        .loading-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: var(--bg);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 2000;
            transition: opacity 0.4s;
        }
        .loading-overlay.hidden { opacity: 0; pointer-events: none; }
        .loader {
            width: 36px; height: 36px;
            border: 2px solid var(--surface-2);
            border-top-color: var(--brand);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading-text {
            margin-top: 16px;
            color: var(--ink);
            font-size: 14px;
            font-family: var(--font-sans);
            font-weight: 600;
        }
        .loading-sub {
            margin-top: 6px;
            color: var(--ink-3);
            font-size: 11px;
            font-family: var(--font-mono);
        }

        /* ===== 空状态 ===== */
        .empty-state {
            text-align: center;
            padding: 24px;
            color: var(--ink-3);
            font-size: 12px;
            font-family: var(--font-mono);
        }

        /* ===== 场景-算法决策树样式 ===== */
        .dt-node {
            display: inline-flex;
            align-items: center;
            padding: 8px 16px;
            border-radius: var(--r-pill);
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.25s ease;
            font-family: var(--font-sans);
        }
        .dt-root {
            background: linear-gradient(135deg, rgba(167,139,250,0.2), rgba(59,201,219,0.15));
            border: 1px solid rgba(167,139,250,0.35);
            color: #c4b5fd;
            box-shadow: 0 0 16px rgba(167,139,250,0.1);
        }
        .dt-root:hover {
            border-color: rgba(167,139,250,0.5);
            box-shadow: 0 0 24px rgba(167,139,250,0.2);
            transform: translateY(-1px);
        }
        .dt-sub {
            background: var(--surface-2);
            border: 1px solid var(--line);
            color: var(--ink-2);
            font-size: 11px;
            padding: 6px 12px;
        }
        .dt-branch {
            padding: 10px 12px;
            border-radius: var(--r-md);
            cursor: pointer;
            transition: all 0.2s ease;
            border: 1px solid;
        }
        .dt-branch-warning {
            background: rgba(251,191,36,0.06);
            border-color: rgba(251,191,36,0.2);
        }
        .dt-branch-warning:hover {
            background: rgba(251,191,36,0.1);
            border-color: rgba(251,191,36,0.35);
        }
        .dt-branch-label {
            font-weight: 600;
            font-size: 12px;
            color: var(--warning);
            margin-bottom: 2px;
        }
        .dt-branch-desc {
            font-size: 10px;
            color: var(--ink-3);
            margin-bottom: 4px;
        }
        .dt-recommend {
            font-size: 11px;
            font-family: var(--font-mono);
        }
        .dt-scenario {
            padding: 10px 12px;
            border-radius: var(--r-md);
            background: var(--surface-2);
            border: 1px solid var(--line);
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .dt-scenario:hover {
            border-color: rgba(59,201,219,0.3);
            background: rgba(59,201,219,0.06);
            transform: translateY(-1px);
        }
        .dt-scenario.active {
            border-color: rgba(59,201,219,0.5);
            background: rgba(59,201,219,0.1);
            box-shadow: 0 0 12px rgba(59,201,219,0.12);
        }
        .dt-sc-title {
            font-weight: 600;
            font-size: 12px;
            color: var(--ink);
            margin-bottom: 2px;
        }
        .dt-sc-desc {
            font-size: 10px;
            color: var(--ink-3);
            margin-bottom: 6px;
        }
        .dt-sc-recommend {
            font-size: 11px;
            font-family: var(--font-mono);
            color: var(--ink-2);
        }
        .dt-up {
            color: var(--success);
            font-weight: 600;
        }
        .dt-detail-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }
        .dt-detail-table th, .dt-detail-table td {
            padding: 6px 8px;
            text-align: left;
            border-bottom: 1px solid var(--line);
        }
        .dt-detail-table th {
            color: var(--ink-3);
            font-weight: 500;
            font-family: var(--font-mono);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .dt-detail-table td {
            color: var(--ink);
        }
        .dt-detail-table .rank-1 { color: #fbbf24; font-weight: 600; }
        .dt-detail-table .rank-2 { color: #94a3b8; font-weight: 500; }
        .dt-detail-table .rank-3 { color: #d97706; }
        .dt-detail-conclusion {
            margin-top: 10px;
            padding: 8px 10px;
            background: rgba(59,201,219,0.06);
            border-left: 2px solid var(--brand);
            border-radius: 0 4px 4px 0;
            font-size: 11px;
            color: var(--ink-2);
            line-height: 1.6;
        }

        /* ===== 真机Demo面板样式 ===== */
        .real-demo-btn {
            width: 100%;
            padding: 8px 12px;
            margin-top: 10px;
            background: linear-gradient(135deg, rgba(59,201,219,0.15), rgba(167,139,250,0.1));
            border: 1px solid rgba(59,201,219,0.3);
            border-radius: var(--r-md);
            color: var(--brand);
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            font-family: var(--font-sans);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }
        .real-demo-btn:hover {
            background: linear-gradient(135deg, rgba(59,201,219,0.25), rgba(167,139,250,0.15));
            border-color: rgba(59,201,219,0.5);
            box-shadow: 0 0 16px rgba(59,201,219,0.15);
        }
        .real-demo-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .real-demo-btn svg {
            width: 14px;
            height: 14px;
        }
        .real-submission-item {
            padding: 6px 0;
            border-bottom: 1px solid var(--line);
            font-size: 10px;
            font-family: var(--font-mono);
        }
        .real-submission-item:last-child {
            border-bottom: none;
        }
        .real-sub-id {
            color: var(--brand);
            font-weight: 600;
        }
        .real-sub-status {
            float: right;
            padding: 1px 6px;
            border-radius: var(--r-pill);
            font-size: 9px;
        }
        .real-sub-status.success {
            background: rgba(52,211,153,0.1);
            color: var(--success);
        }
        .real-sub-status.pending {
            background: rgba(251,191,36,0.1);
            color: var(--warning);
        }
        .real-sub-meta {
            color: var(--ink-3);
            margin-top: 2px;
        }

        /* ===== 响应式 ===== */
        @media (max-width: 1100px) {
            .main-content { grid-template-columns: 1fr; }
            .two-col-panels { grid-template-columns: 1fr; }
        }
        @media (max-width: 768px) {
            .header { padding: 10px 14px; }
            .header-titles h1 { font-size: 15px; }
            .status-cards { padding: 12px; gap: 8px; grid-template-columns: repeat(2, 1fr); }
            .main-content { padding: 10px 14px 14px; }
            .card-value { font-size: 24px; }
            .status-card { padding: 14px; }
        }
    </style>
</head>
<body>

    <!-- 量子粒子背景 -->
    <canvas id="quantum-bg"></canvas>
    <div class="bg-vignette"></div>

    <!-- 加载遮罩 -->
    <div class="loading-overlay" id="loading">
        <div class="loader"></div>
        <div class="loading-text">量子RL调度系统启动中...</div>
        <div class="loading-sub">正在加载PPO模型并初始化仿真环境</div>
    </div>

    <!-- 顶部标题栏 -->
    <div class="header">
        <div class="header-left">
            <div class="logo">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <circle cx="12" cy="12" r="2" fill="currentColor" stroke="none"/>
                    <g class="electron">
                        <ellipse cx="12" cy="12" rx="9" ry="3.5"/>
                    </g>
                    <g class="electron electron-2">
                        <ellipse cx="12" cy="12" rx="9" ry="3.5" transform="rotate(60 12 12)"/>
                    </g>
                    <g style="transform-origin:12px 12px; animation: electronOrbit 10s linear infinite reverse;">
                        <ellipse cx="12" cy="12" rx="9" ry="3.5" transform="rotate(-60 12 12)"/>
                    </g>
                </svg>
            </div>
            <div class="header-titles">
                <h1>量子RL智能调度系统</h1>
                <div class="subtitle">AI赋能量子计算 · PPO强化学习 · 14维状态空间 · 8策略对比</div>
            </div>
        </div>
        <div class="header-right">
            <span class="model-badge" id="model-badge">PPO 14dim</span>
            <span id="ws-status" class="ws-status disconnected">连接中...</span>
        </div>
    </div>

    <!-- 系统状态卡片 -->
    <div class="status-cards">
        <div class="status-card card-blue">
            <div class="card-label"><span class="card-icon"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/><circle cx="12" cy="12" r="7" opacity="0.3"/></svg></span>量子比特利用率</div>
            <div class="card-value" id="val-qubit">0%</div>
            <div class="card-sub">实时资源占用</div>
        </div>
        <div class="status-card card-purple">
            <div class="card-label"><span class="card-icon"><svg viewBox="0 0 24 24"><rect x="3" y="5" width="5" height="5" rx="1" fill="currentColor" stroke="none"/><rect x="10" y="5" width="5" height="5" rx="1" fill="currentColor" stroke="none" opacity="0.6"/><rect x="17" y="5" width="4" height="5" rx="1" fill="currentColor" stroke="none" opacity="0.3"/><rect x="3" y="14" width="5" height="5" rx="1" fill="currentColor" stroke="none"/><rect x="10" y="14" width="5" height="5" rx="1" fill="currentColor" stroke="none" opacity="0.6"/></svg></span>任务队列长度</div>
            <div class="card-value" id="val-queue">0</div>
            <div class="card-sub">等待调度执行</div>
        </div>
        <div class="status-card card-amber">
            <div class="card-label"><span class="card-icon"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg></span>平均等待时间</div>
            <div class="card-value" id="val-wait">0s</div>
            <div class="card-sub">最近100个任务</div>
        </div>
        <div class="status-card card-green">
            <div class="card-label"><span class="card-icon"><svg viewBox="0 0 24 24"><path d="M5 12l5 5L20 7"/></svg></span>已完成任务</div>
            <div class="card-value" id="val-completed">0</div>
            <div class="card-sub">累计吞吐量</div>
        </div>
        <div class="status-card card-cyan">
            <div class="card-label"><span class="card-icon"><svg viewBox="0 0 24 24"><path d="M13 2L4 14h7l-1 8 9-12h-7l1-8z" fill="currentColor" stroke="none"/></svg></span>吞吐量</div>
            <div class="card-value" id="val-throughput">0</div>
            <div class="card-sub">任务/分钟</div>
        </div>
        <div class="status-card card-pink">
            <div class="card-label"><span class="card-icon"><svg viewBox="0 0 24 24"><path d="M12 2l3 7h7l-5.5 4.5L18 21l-6-4-6 4 1.5-7.5L2 9h7z" fill="currentColor" stroke="none"/></svg></span>当前策略</div>
            <div class="card-value" id="val-strategy" style="font-size:22px;">-</div>
            <div class="card-sub" id="val-step">Step: 0</div>
        </div>
    </div>

    <!-- 主内容区域 -->
    <div class="main-content">

        <!-- 左侧主区域 -->
        <div class="main-col">

            <!-- PPO vs FCFS 实时对比面板 -->
            <div class="panel battle-panel">
                <div class="panel-header">
                    <h2><span class="icon icon-battle"><svg viewBox="0 0 24 24"><circle cx="6" cy="12" r="2.2"/><circle cx="18" cy="12" r="2.2"/><path d="M8.2 12h7.6M12 8.5l2 3.5-2 3.5"/></svg></span> PPO vs FCFS 实时策略对比</h2>
                    <span class="badge live" id="battle-status">● 自动运行中</span>
                </div>
                <div class="panel-body">
                    <div class="battle-canvas-wrap">
                        <canvas id="battle-canvas" height="220"></canvas>
                    </div>
                    <div class="battle-legend">
                        <div class="legend-item"><div class="legend-line legend-ppo"></div> PPO (强化学习)</div>
                        <div class="legend-item"><div class="legend-line legend-fcfs"></div> FCFS (先来先服务)</div>
                        <div class="legend-item"><div class="legend-line legend-baseline"></div> 零奖励基线</div>
                    </div>
                    <div class="battle-stats">
                        <div class="battle-stat ppo">
                            <div class="label">PPO 累积奖励</div>
                            <div class="value" id="battle-ppo">0</div>
                        </div>
                        <div class="battle-stat fcfs">
                            <div class="label">FCFS 累积奖励</div>
                            <div class="value" id="battle-fcfs">0</div>
                        </div>
                        <div class="battle-stat diff">
                            <div class="label">PPO 优势</div>
                            <div class="value" id="battle-diff">+0%</div>
                        </div>
                    </div>
                    <div class="battle-controls">
                        <button class="btn-sm active" id="btn-battle-auto" onclick="setBattleAuto(true)">自动对战</button>
                        <button class="btn-sm" id="btn-battle-step" onclick="battleStep()">单步推进</button>
                        <button class="btn-sm" onclick="battleReset()">重置</button>
                        <span class="battle-info" id="battle-step-info">Step: 0</span>
                    </div>
                </div>
            </div>

            <!-- 资源利用率趋势图 -->
            <div class="panel">
                <div class="panel-header">
                    <h2><span class="icon icon-chart"><svg viewBox="0 0 24 24"><path d="M3 17c2-4 4-8 6-8s2 6 4 6 2-5 4-5 2 3 4 3"/><circle cx="3" cy="17" r="1" fill="currentColor" stroke="none"/><circle cx="21" cy="13" r="1" fill="currentColor" stroke="none"/><path d="M3 20h18" opacity="0.3"/></svg></span> 资源利用率趋势</h2>
                    <span class="badge live">● 实时</span>
                </div>
                <div class="panel-body">
                    <div class="chart-canvas-wrap">
                        <canvas id="resource-canvas" height="200"></canvas>
                    </div>
                    <div class="chart-legend">
                        <div class="legend-item"><div class="legend-line" style="background:#3bc9db"></div> 量子比特利用率</div>
                        <div class="legend-item"><div class="legend-line" style="background:rgba(230,237,243,0.5)"></div> 队列长度</div>
                        <div class="legend-item"><div class="legend-line" style="background:#34d399"></div> 完成速率</div>
                    </div>
                </div>
            </div>

            <!-- 决策日志 + 特征贡献度 双栏 -->
            <div class="two-col-panels">
                <!-- 决策日志 -->
                <div class="panel">
                    <div class="panel-header">
                        <h2><span class="icon icon-log"><svg viewBox="0 0 24 24"><rect x="4" y="3" width="16" height="18" rx="2" fill="none"/><path d="M8 8h8M8 12h8M8 16h5"/><circle cx="17" cy="16" r="1" fill="currentColor" stroke="none"/></svg></span> 最近决策</h2>
                        <span class="badge" id="decision-count">0 条</span>
                    </div>
                    <div class="panel-body" style="padding:10px 14px;">
                        <div class="decision-list" id="decision-list">
                            <div class="empty-state">等待决策数据...</div>
                        </div>
                    </div>
                </div>

                <!-- 特征贡献度 -->
                <div class="panel">
                    <div class="panel-header">
                        <h2><span class="icon icon-explain"><svg viewBox="0 0 24 24"><circle cx="6" cy="7" r="1.5"/><circle cx="6" cy="17" r="1.5"/><circle cx="12" cy="12" r="2"/><circle cx="18" cy="7" r="1.5"/><circle cx="18" cy="17" r="1.5"/><path d="M7.3 8l3 2.7M7.3 16l3-2.7M16.7 8l-3 2.7M16.7 16l-3-2.7"/></svg></span> 决策可解释性</h2>
                        <span class="badge">SHAP-like</span>
                    </div>
                    <div class="panel-body" style="padding:12px 14px;">
                        <div style="font-size:11px; color:var(--text-muted); margin-bottom:10px;" id="explain-header">
                            最新决策的特征贡献度
                        </div>
                        <div class="feature-bars" id="feature-bars">
                            <div class="empty-state" style="padding:20px;">等待特征数据...</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 调度决策分布饼图 -->
            <div class="panel">
                <div class="panel-header">
                    <h2><span class="icon icon-explain"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 3v9l6 4" stroke-width="2"/><path d="M12 3a9 9 0 0 1 6.5 15.5" stroke-dasharray="2 2" opacity="0.5"/></svg></span> 调度决策分布</h2>
                    <span class="badge" id="pie-total">0 次决策</span>
                </div>
                <div class="panel-body" style="padding:16px;">
                    <div style="display:flex;align-items:center;gap:24px;">
                        <div style="flex-shrink:0;position:relative;">
                            <canvas id="decision-pie-canvas" width="160" height="160" style="display:block;"></canvas>
                        </div>
                        <div style="flex:1;display:flex;flex-direction:column;gap:10px;" id="pie-legend">
                            <div class="pie-legend-item">
                                <span class="pie-dot" style="background:#3bc9db;box-shadow:0 0 8px rgba(59,201,219,0.4);"></span>
                                <span class="pie-label">量子资源</span>
                                <span class="pie-value" id="pie-quantum">0 (0%)</span>
                            </div>
                            <div class="pie-legend-item">
                                <span class="pie-dot" style="background:#a78bfa;box-shadow:0 0 8px rgba(167,139,250,0.4);"></span>
                                <span class="pie-label">经典资源</span>
                                <span class="pie-value" id="pie-classical">0 (0%)</span>
                            </div>
                            <div class="pie-legend-item">
                                <span class="pie-dot" style="background:#fbbf24;box-shadow:0 0 8px rgba(251,191,36,0.4);"></span>
                                <span class="pie-label">混合执行</span>
                                <span class="pie-value" id="pie-hybrid">0 (0%)</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 策略排名 -->
            <div class="panel">
                <div class="panel-header">
                    <h2><span class="icon icon-rank"><svg viewBox="0 0 24 24"><path d="M6 20h12M8 20v-5l4-7 4 7v5"/><circle cx="12" cy="6" r="2" fill="currentColor" stroke="none"/><path d="M10 11l2 2 2-2" opacity="0.5"/></svg></span> 8种策略性能排名</h2>
                    <span class="badge">14维公平重训</span>
                </div>
                <div class="panel-body" id="ranking-body">
                    <div class="empty-state">加载排名数据...</div>
                </div>
            </div>

            <!-- 场景-算法决策树（交互式） -->
            <div class="panel">
                <div class="panel-header">
                    <h2><span class="icon" style="background:linear-gradient(135deg,rgba(167,139,250,0.2),rgba(59,201,219,0.1));border-color:rgba(167,139,250,0.25);color:#a78bfa;"><svg viewBox="0 0 24 24"><circle cx="12" cy="4" r="2"/><circle cx="5" cy="12" r="2"/><circle cx="19" cy="12" r="2"/><circle cx="8" cy="20" r="2"/><circle cx="16" cy="20" r="2"/><path d="M12 6v2M6.5 10.5L10 8M17.5 10.5L14 8M6 14v3M18 14v3M9 18h6"/></svg></span> 场景-算法决策树</h2>
                    <span class="badge" style="background:rgba(167,139,250,0.1);color:#a78bfa;border-color:rgba(167,139,250,0.2);">压力测试结论</span>
                </div>
                <div class="panel-body" style="padding:16px;">
                    <div style="font-size:11px;color:var(--ink-3);margin-bottom:14px;line-height:1.6;">
                        基于5场景×6策略压力测试（N=250）生成的调度策略推荐。点击场景卡片查看详细分析。
                    </div>
                    <div id="decision-tree-container">
                        <!-- 决策树交互式节点 -->
                        <div style="display:flex;flex-direction:column;gap:10px;">
                            <!-- 根节点 -->
                            <div style="display:flex;justify-content:center;">
                                <div class="dt-node dt-root" onclick="dtSelectRoot()">
                                    <svg width="16" height="16" viewBox="0 0 24 24" style="margin-right:6px;vertical-align:-2px;"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 8v4l3 2" stroke="currentColor" stroke-width="2"/></svg>
                                    系统状态判断
                                </div>
                            </div>
                            <!-- 连接线 -->
                            <div style="display:flex;justify-content:center;height:20px;position:relative;">
                                <div style="width:1px;height:100%;background:linear-gradient(180deg,rgba(167,139,250,0.5),rgba(167,139,250,0.2));"></div>
                            </div>
                            <!-- 分支1：量子资源波动 -->
                            <div style="display:flex;gap:12px;align-items:flex-start;">
                                <div style="flex:1;display:flex;flex-direction:column;gap:8px;">
                                    <div class="dt-branch dt-branch-warning" onclick="dtSelectScenario('volatile')">
                                        <div class="dt-branch-label">量子资源波动</div>
                                        <div class="dt-branch-desc">校准/维护/保真度下降</div>
                                        <div class="dt-recommend" style="color:var(--brand);">→ 推荐 PPO <span style="color:var(--success);font-weight:600;">+91.4%</span></div>
                                    </div>
                                </div>
                                <div style="display:flex;align-items:center;padding-top:12px;color:var(--ink-4);font-family:var(--font-mono);font-size:10px;">OR</div>
                                <div style="flex:2;">
                                    <!-- 子节点：负载模式判断 -->
                                    <div class="dt-node dt-sub" style="margin-bottom:8px;">
                                        <svg width="14" height="14" viewBox="0 0 24 24" style="margin-right:5px;vertical-align:-1px;"><rect x="4" y="4" width="16" height="16" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M9 9h6M9 12h6M9 15h4" stroke="currentColor" stroke-width="1.5"/></svg>
                                        负载模式判断
                                    </div>
                                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                                        <div class="dt-scenario" onclick="dtSelectScenario('balanced')" id="dt-sc-balanced">
                                            <div class="dt-sc-title">均衡负载</div>
                                            <div class="dt-sc-desc">日常稳态运行</div>
                                            <div class="dt-sc-recommend">PPO <span class="dt-up">+88.3%</span></div>
                                        </div>
                                        <div class="dt-scenario" onclick="dtSelectScenario('high_load')" id="dt-sc-high_load">
                                            <div class="dt-sc-title">高负载积压</div>
                                            <div class="dt-sc-desc">批量任务/截止日</div>
                                            <div class="dt-sc-recommend">PPO* <span class="dt-up">+156.5%</span></div>
                                        </div>
                                        <div class="dt-scenario" onclick="dtSelectScenario('tidal')" id="dt-sc-tidal">
                                            <div class="dt-sc-title">潮汐波动</div>
                                            <div class="dt-sc-desc">昼夜模式切换</div>
                                            <div class="dt-sc-recommend">PPO <span class="dt-up">+34.0%</span></div>
                                        </div>
                                        <div class="dt-scenario" onclick="dtSelectScenario('burst')" id="dt-sc-burst">
                                            <div class="dt-sc-title">突发洪峰</div>
                                            <div class="dt-sc-desc">集体提交/作业洪峰</div>
                                            <div class="dt-sc-recommend">PPO+限流</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <!-- 场景详情展示区 -->
                    <div id="dt-detail-panel" style="display:none;margin-top:14px;padding:12px 14px;background:var(--surface-2);border:1px solid var(--line);border-radius:var(--r-md);">
                        <div id="dt-detail-content"></div>
                    </div>
                    <div style="margin-top:12px;padding:8px 10px;background:rgba(59,201,219,0.05);border:1px solid rgba(59,201,219,0.1);border-radius:6px;font-size:10px;color:var(--ink-3);line-height:1.6;">
                        <strong style="color:var(--brand);">决策原则：</strong>
                        PPO综合排名1.8/6，是唯一在所有5场景均进入前三的策略。Greedy在高负载场景领先4.9%但波动场景崩溃(-95.3%)。
                    </div>
                </div>
            </div>

        </div>

        <!-- 右侧栏 -->
        <div class="side-col">

            <!-- 任务队列 -->
            <div class="panel">
                <div class="panel-header">
                    <h2><span class="icon icon-queue"><svg viewBox="0 0 24 24"><rect x="3" y="5" width="4" height="4" rx="1" fill="currentColor" stroke="none"/><rect x="10" y="5" width="4" height="4" rx="1" fill="currentColor" stroke="none"/><rect x="17" y="5" width="4" height="4" rx="1" fill="currentColor" stroke="none" opacity="0.35"/><rect x="3" y="15" width="4" height="4" rx="1" fill="currentColor" stroke="none"/><rect x="10" y="15" width="4" height="4" rx="1" fill="currentColor" stroke="none" opacity="0.35"/><path d="M5 9v2M12 9v2M5 13v-2M12 13v-2" stroke-dasharray="1 1"/></svg></span> 任务队列</h2>
                    <span class="badge" id="task-count">0 个任务</span>
                </div>
                <div class="panel-body" style="padding:0;">
                    <div class="task-table-wrap">
                        <table class="task-table">
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>类型</th>
                                    <th>比特</th>
                                    <th>优先级</th>
                                    <th>状态</th>
                                </tr>
                            </thead>
                            <tbody id="task-tbody"></tbody>
                        </table>
                    </div>
                    <div id="task-empty" class="empty-state" style="display:none; padding:20px;">
                        暂无任务
                    </div>
                </div>
            </div>

            <!-- 控制面板 -->
            <div class="panel">
                <div class="panel-header">
                    <h2><span class="icon icon-control"><svg viewBox="0 0 24 24"><circle cx="7" cy="7" r="2.5"/><circle cx="17" cy="7" r="2.5"/><circle cx="12" cy="17" r="2.5"/><path d="M7 9.5v3M17 9.5v3M12 14.5v-3"/><circle cx="7" cy="7" r="1" fill="currentColor" stroke="none"/><circle cx="17" cy="7" r="1" fill="currentColor" stroke="none"/><circle cx="12" cy="17" r="1" fill="currentColor" stroke="none"/></svg></span> 控制面板</h2>
                </div>
                <div class="panel-body">
                    <div class="control-grid">
                        <!-- 提交新任务 -->
                        <div class="control-section">
                            <h3>提交新任务</h3>
                            <div class="form-group">
                                <label>用户ID</label>
                                <input type="text" id="input-user" value="user_001">
                            </div>
                            <div class="form-row">
                                <div class="form-group">
                                    <label>类型</label>
                                    <select id="input-type">
                                        <option value="quantum">量子</option>
                                        <option value="classical">经典</option>
                                        <option value="hybrid">混合</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label>优先级</label>
                                    <select id="input-priority">
                                        <option value="1">1 最低</option>
                                        <option value="2">2 低</option>
                                        <option value="3" selected>3 中</option>
                                        <option value="4">4 高</option>
                                        <option value="5">5 最高</option>
                                    </select>
                                </div>
                            </div>
                            <div class="form-row">
                                <div class="form-group">
                                    <label>比特数</label>
                                    <input type="number" id="input-qubits" value="10" min="1" max="287">
                                </div>
                                <div class="form-group">
                                    <label>电路深度</label>
                                    <input type="number" id="input-depth" value="100" min="1">
                                </div>
                            </div>
                            <div class="form-group">
                                <label>预计执行时间(秒)</label>
                                <input type="number" id="input-time" value="60" min="1" step="1">
                            </div>
                            <button class="btn btn-primary" onclick="submitTask()">提交任务</button>
                        </div>

                        <!-- 调度策略切换 -->
                        <div class="control-section">
                            <h3>调度策略</h3>
                            <div class="strategy-buttons" id="strategy-buttons"></div>
                            <div style="font-size:10px; color:var(--text-muted); margin-top:8px; line-height:1.5;">
                                * PPO为仿真验证最优策略<br>
                                PPO vs FCFS: <b style="color:var(--green-light);">+88.3%</b> (N=250, p&lt;0.001)
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 天衍云真机状态 -->
            <div class="panel">
                <div class="panel-header">
                    <h2><span class="icon" style="background:linear-gradient(135deg,rgba(59,201,219,0.2),rgba(167,139,250,0.1));border-color:rgba(167,139,250,0.25);color:#a78bfa;"><svg viewBox="0 0 24 24"><path d="M18 10a6 6 0 0 0-12 0c-2.2 0-4 1.8-4 4s1.8 4 4 4h12c2.2 0 4-1.8 4-4s-1.8-4-4-4z"/><path d="M12 14v4M12 10l-2 2M12 10l2 2" stroke-width="2"/></svg></span> 天衍量子云</h2>
                    <span class="badge" id="real-machine-badge" style="background:rgba(52,211,153,0.1);color:var(--success);border-color:rgba(52,211,153,0.2);">284次·100%</span>
                </div>
                <div class="panel-body" style="font-size:12px;line-height:1.8;">
                    <div id="real-machines-list">
                        <div style="padding:8px 0;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;">
                            <div>
                                <div style="font-weight:600;color:var(--ink);font-size:12px;">天衍-S (tianyan_s)</div>
                                <div style="font-size:10px;color:var(--ink-3);font-family:var(--font-mono);">287 qubit · cqlib SDK</div>
                            </div>
                            <span id="machine-tianyan_s-status" style="font-size:10px;padding:2px 8px;border-radius:999px;background:rgba(52,211,153,0.1);color:var(--success);font-family:var(--font-mono);">✓ 已验证</span>
                        </div>
                        <div style="padding:8px 0;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;">
                            <div>
                                <div style="font-weight:600;color:var(--ink);font-size:12px;">天衍-SW (tianyan_sw)</div>
                                <div style="font-size:10px;color:var(--ink-3);font-family:var(--font-mono);">72 qubit · cqlib SDK</div>
                            </div>
                            <span id="machine-tianyan_sw-status" style="font-size:10px;padding:2px 8px;border-radius:999px;background:rgba(52,211,153,0.1);color:var(--success);font-family:var(--font-mono);">✓ 已验证</span>
                        </div>
                        <div style="padding:8px 0;display:flex;justify-content:space-between;align-items:center;">
                            <div>
                                <div style="font-weight:600;color:var(--ink);font-size:12px;">天衍-TN (tianyan_tn)</div>
                                <div style="font-size:10px;color:var(--ink-3);font-family:var(--font-mono);">176 qubit · cqlib SDK</div>
                            </div>
                            <span id="machine-tianyan_tn-status" style="font-size:10px;padding:2px 8px;border-radius:999px;background:rgba(52,211,153,0.1);color:var(--success);font-family:var(--font-mono);">✓ 已验证</span>
                        </div>
                    </div>

                    <!-- 真机Demo提交按钮 -->
                    <button class="real-demo-btn" id="btn-real-demo" onclick="submitRealDemo()">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                        提交真机Demo任务
                    </button>

                    <!-- 最近真机提交记录 -->
                    <div style="margin-top:12px;">
                        <div style="font-size:10px;color:var(--ink-3);font-family:var(--font-mono);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">最近提交记录</div>
                        <div id="real-submissions-list" style="max-height:120px;overflow-y:auto;">
                            <div style="font-size:10px;color:var(--ink-4);text-align:center;padding:8px;">加载中...</div>
                        </div>
                    </div>

                    <div style="margin-top:10px;padding:8px 10px;background:rgba(59,201,219,0.06);border:1px solid rgba(59,201,219,0.12);border-radius:6px;font-size:11px;color:var(--ink-2);line-height:1.6;">
                        <strong style="color:var(--brand);">真机验证说明：</strong>
                        已完成284次SDK全链路调用（认证→提交→轮询→结果），成功率100%。
                        真机性能数据为探索性（5 seeds），核心性能claim基于N=250仿真统计结果。
                        <span id="api-key-status" style="display:block;margin-top:4px;color:var(--warning);font-size:10px;">
                            ⚠ 未配置TIANYAN_API_KEY，真机提交将使用Mock模式演示
                        </span>
                    </div>
                </div>
            </div>

            <!-- 系统信息 -->
            <div class="panel">
                <div class="panel-header">
                    <h2><span class="icon icon-system"><svg viewBox="0 0 24 24"><rect x="5" y="5" width="14" height="14" rx="2" fill="none"/><rect x="9" y="9" width="6" height="6" rx="1" fill="currentColor" stroke="none" opacity="0.25"/><path d="M9 3v2M15 3v2M9 19v2M15 19v2M3 9h2M3 15h2M19 9h2M19 15h2"/></svg></span> 系统信息</h2>
                </div>
                <div class="panel-body" style="font-size:12px; line-height:1.8; color:var(--text-secondary);">
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:var(--text-muted);">核心方向</span>
                        <span>AI赋能量子调度</span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:var(--text-muted);">PPO模型</span>
                        <span>14维观测空间</span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:var(--text-muted);">量子比特</span>
                        <span>287 qubit</span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:var(--text-muted);">退火模块</span>
                        <span style="color:var(--amber-warn);">探索性方向</span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:var(--text-muted);">真机验证</span>
                        <span>SDK可用性验证</span>
                    </div>
                    <div style="margin-top:8px; padding-top:8px; border-top:1px solid var(--border); font-size:11px; color:var(--text-muted);">
                        注："量子赋能AI"为探索性研究方向，当前使用经典模拟退火，训练开销+74.5%，奖励提升统计不显著(p=0.190)
                    </div>
                </div>
            </div>

        </div>
    </div>

    <!-- Toast 通知容器 -->
    <div class="toast-container" id="toast-container"></div>

    <script>
    // ================================================================
    // 全局状态
    // ================================================================
    let ws = null;
    let currentStatus = {};
    let currentTasks = [];
    let strategyOptions = [];
    let reconnectTimer = null;
    let resourceHistory = [];  // {step, utilization, queue, throughput}
    let decisionLog = [];      // 最近决策
    let battleState = null;    // PPO vs FCFS 对战状态
    let battleAutoMode = true;
    let battleAutoTimer = null;
    let httpPollTimer = null;
    let ppoStats = {};
    let chartAnimFrame = null;

    // ================================================================
    // 工具函数
    // ================================================================
    function showToast(msg, type) {
        type = type || 'info';
        var c = document.getElementById('toast-container');
        var t = document.createElement('div');
        t.className = 'toast ' + type;
        t.textContent = msg;
        c.appendChild(t);
        setTimeout(function() { if (t.parentNode) t.parentNode.removeChild(t); }, 3500);
    }

    function fmtTime(isoStr) {
        if (!isoStr) return '-';
        try {
            var d = new Date(isoStr);
            return String(d.getHours()).padStart(2,'0')+':'+String(d.getMinutes()).padStart(2,'0')+':'+String(d.getSeconds()).padStart(2,'0');
        } catch(e) { return '-'; }
    }

    function escapeHtml(str) {
        // Issue #381: 防止存储型 XSS，对所有来自 API 的字符串字段进行 HTML 转义
        if (str === null || str === undefined) return '';
        return String(str).replace(/[&<>"'/]/g, function(c) {
            return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;','/':'&#x2F;'}[c];
        });
    }

    function prioClass(p) { return p>=4?'priority-high':p>=3?'priority-medium':'priority-low'; }
    function statusText(s) { return {pending:'等待中',running:'运行中',completed:'已完成',failed:'失败'}[s]||s; }
    function actionText(a) {
        if (a===0||a==='classical') return {cls:'action-classical', text:'经典'};
        if (a===1||a==='quantum') return {cls:'action-quantum', text:'量子'};
        return {cls:'action-hybrid', text:'混合'};
    }

    // ================================================================
    // Canvas 图表绘制
    // ================================================================

    function setupHiDPI(canvas) {
        var ctx = canvas.getContext('2d');
        var dpr = window.devicePixelRatio || 1;
        var rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
        return { ctx: ctx, w: rect.width, h: rect.height };
    }

    function drawBattleChart() {
        var canvas = document.getElementById('battle-canvas');
        var {ctx, w, h} = setupHiDPI(canvas);
        var pad = {top:15, right:15, bottom:25, left:55};
        var cw = w - pad.left - pad.right;
        var ch = h - pad.top - pad.bottom;

        ctx.clearRect(0,0,w,h);

        // 获取数据 - API返回history是对象数组，提取cumulative字段
        var ppoHist = [];
        var fcfsHist = [];
        if (battleState) {
            if (battleState.ppo_history) {
                ppoHist = battleState.ppo_history.map(function(d){return typeof d==='number'?d:d.cumulative;});
            }
            if (battleState.fcfs_history) {
                fcfsHist = battleState.fcfs_history.map(function(d){return typeof d==='number'?d:d.cumulative;});
            }
        }
        var maxLen = Math.max(ppoHist.length, fcfsHist.length, 50);
        if (maxLen < 20) maxLen = 50;

        // 计算Y轴范围
        var allVals = ppoHist.concat(fcfsHist);
        var yMin = Math.min(0, Math.min.apply(null, allVals.concat([0])));
        var yMax = Math.max(100, Math.max.apply(null, allVals.concat([100])));
        var yRange = yMax - yMin;
        yMin -= yRange * 0.1;
        yMax += yRange * 0.1;
        yRange = yMax - yMin;

        function xPos(i) { return pad.left + (i / (maxLen-1)) * cw; }
        function yPos(v) { return pad.top + ch - ((v - yMin) / yRange) * ch; }

        // 网格线 - 量子深空主题
        ctx.strokeStyle = 'rgba(148,184,220,0.08)';
        ctx.lineWidth = 1;
        for (var gi=0; gi<=4; gi++) {
            var gy = pad.top + (gi/4)*ch;
            ctx.beginPath(); ctx.moveTo(pad.left, gy); ctx.lineTo(w-pad.right, gy); ctx.stroke();
            var gv = yMax - (gi/4)*yRange;
            ctx.fillStyle = 'rgba(230,237,243,0.35)';
            ctx.font = '10px "JetBrains Mono", Consolas, monospace';
            ctx.textAlign = 'right';
            ctx.fillText(Math.round(gv), pad.left-6, gy+3);
        }

        // 零基线
        var zeroY = yPos(0);
        ctx.strokeStyle = 'rgba(148,184,220,0.15)';
        ctx.setLineDash([4,4]);
        ctx.beginPath(); ctx.moveTo(pad.left, zeroY); ctx.lineTo(w-pad.right, zeroY); ctx.stroke();
        ctx.setLineDash([]);

        // 绘制曲线函数
        function drawLine(data, color, fillColor) {
            if (data.length < 2) return;
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.lineJoin = 'round';
            ctx.beginPath();
            for (var i=0; i<data.length; i++) {
                var px = xPos(i);
                var py = yPos(data[i]);
                if (i===0) ctx.moveTo(px,py);
                else ctx.lineTo(px,py);
            }
            ctx.stroke();

            // 填充区域
            var lastX = xPos(data.length-1);
            ctx.fillStyle = fillColor;
            ctx.beginPath();
            ctx.moveTo(xPos(0), zeroY);
            for (var i=0; i<data.length; i++) ctx.lineTo(xPos(i), yPos(data[i]));
            ctx.lineTo(lastX, zeroY);
            ctx.closePath();
            ctx.fill();
        }

        drawLine(fcfsHist, 'rgba(230,237,243,0.35)', 'rgba(230,237,243,0.03)');
        drawLine(ppoHist, '#3bc9db', 'rgba(59,201,219,0.08)');

        // 端点
        if (ppoHist.length > 0) {
            ctx.fillStyle = '#3bc9db';
            ctx.strokeStyle = '#070b14';
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(xPos(ppoHist.length-1), yPos(ppoHist[ppoHist.length-1]), 4, 0, Math.PI*2); ctx.fill(); ctx.stroke();
        }
        if (fcfsHist.length > 0) {
            ctx.fillStyle = 'rgba(230,237,243,0.35)';
            ctx.strokeStyle = '#070b14';
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(xPos(fcfsHist.length-1), yPos(fcfsHist[fcfsHist.length-1]), 4, 0, Math.PI*2); ctx.fill(); ctx.stroke();
        }
    }

    function drawResourceChart() {
        var canvas = document.getElementById('resource-canvas');
        var {ctx, w, h} = setupHiDPI(canvas);
        var pad = {top:15, right:50, bottom:25, left:45};
        var cw = w - pad.left - pad.right;
        var ch = h - pad.top - pad.bottom;

        ctx.clearRect(0,0,w,h);

        var data = resourceHistory.slice(-80);
        if (data.length < 2) {
            ctx.fillStyle = 'rgba(230,237,243,0.35)';
            ctx.font = '12px "JetBrains Mono", Consolas, monospace';
            ctx.textAlign = 'center';
            ctx.fillText('正在收集数据...', w/2, h/2);
            return;
        }

        var maxPoints = Math.max(80, data.length);

        // 左Y轴: 利用率 0-100%
        // 右Y轴: 队列长度
        var utils = data.map(function(d){return d.utilization*100;});
        var queues = data.map(function(d){return d.queue;});
        var throughs = data.map(function(d){return d.throughput;});

        var yLeftMax = 100;
        var yRightMax = Math.max(10, Math.max.apply(null, queues) * 1.2);

        function xPos(i) { return pad.left + (i/(Math.max(maxPoints-1,1))) * cw; }
        function yLeft(v) { return pad.top + ch - (v/yLeftMax)*ch; }
        function yRight(v) { return pad.top + ch - (v/yRightMax)*ch; }

        // 网格 - 量子深空主题
        ctx.strokeStyle = 'rgba(148,184,220,0.08)';
        ctx.lineWidth = 1;
        for (var gi=0; gi<=4; gi++) {
            var gy = pad.top + (gi/4)*ch;
            ctx.beginPath(); ctx.moveTo(pad.left, gy); ctx.lineTo(w-pad.right, gy); ctx.stroke();
            ctx.fillStyle = 'rgba(230,237,243,0.35)'; ctx.font='10px "JetBrains Mono", Consolas, monospace'; ctx.textAlign='right';
            ctx.fillText(Math.round(yLeftMax - (gi/4)*yLeftMax)+'%', pad.left-6, gy+3);
        }
        // 右Y轴标签
        ctx.textAlign = 'left';
        for (var gi=0; gi<=4; gi++) {
            var gy = pad.top + (gi/4)*ch;
            var gv = Math.round(yRightMax - (gi/4)*yRightMax);
            ctx.fillStyle = 'rgba(230,237,243,0.35)';
            ctx.fillText(gv, w-pad.right+6, gy+3);
        }

        // 绘制折线
        function drawLine(values, yFn, color, yFnCtx) {
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.lineJoin = 'round';
            ctx.beginPath();
            var offset = maxPoints - values.length;
            for (var i=0; i<values.length; i++) {
                var px = xPos(i + offset);
                var py = yFn(values[i]);
                if (i===0) ctx.moveTo(px,py);
                else ctx.lineTo(px,py);
            }
            ctx.stroke();
        }

        drawLine(utils, yLeft, '#3bc9db');
        drawLine(queues, yRight, 'rgba(230,237,243,0.5)');
        drawLine(throughs, function(v){return yLeft(Math.min(v*5, 100));}, '#34d399');

        // 图例补充
        ctx.textAlign = 'left';
        ctx.font = '10px sans-serif';
    }

    function redrawCharts() {
        drawBattleChart();
        drawResourceChart();
        chartAnimFrame = requestAnimationFrame(redrawCharts);
    }

    // ================================================================
    // 页面渲染
    // ================================================================
    // 数值更新闪光辅助函数
    function flashValue(elId, newText) {
        var el = document.getElementById(elId);
        if (!el) return;
        var oldText = el.textContent;
        if (oldText !== newText) {
            el.textContent = newText;
            el.classList.remove('flash');
            void el.offsetWidth; // 强制重排触发动画
            el.classList.add('flash');
        }
    }

    function renderStatus(status) {
        if (!status) return;
        flashValue('val-qubit', ((status.qubit_utilization||0)*100).toFixed(1)+'%');
        flashValue('val-queue', String(status.queue_length||0));
        flashValue('val-wait', (status.average_wait_time||0).toFixed(1)+'s');
        flashValue('val-completed', String(status.completed_tasks||0));
        // 吞吐量：如果API没有提供，则根据已完成任务和步数估算
        var throughput = status.throughput;
        if (throughput === undefined || throughput === null) {
            var steps = status.current_step || 1;
            var completed = status.completed_tasks || 0;
            // 假设每步约3秒，换算为任务/分钟
            throughput = steps > 0 ? (completed / (steps * 3 / 60)) : 0;
        }
        flashValue('val-throughput', throughput.toFixed(1));
        flashValue('val-strategy', (status.current_strategy||'-').toUpperCase());
        var stepEl = document.getElementById('val-step');
        if (stepEl) stepEl.textContent = 'Step: '+(status.current_step||0);

        // 记录历史
        resourceHistory.push({
            step: status.current_step||0,
            utilization: status.qubit_utilization||0,
            queue: status.queue_length||0,
            throughput: throughput
        });
        if (resourceHistory.length > 200) resourceHistory.shift();
    }

    function renderTasks(tasks) {
        var tbody = document.getElementById('task-tbody');
        if (!tbody) return; // 新模板中已移除任务面板
        var empty = document.getElementById('task-empty');
        var badge = document.getElementById('task-count');
        if (badge) badge.textContent = (tasks?tasks.length:0)+' 个任务';
        if (!tasks || tasks.length === 0) {
            tbody.innerHTML = '';
            if (empty) empty.style.display = 'block';
            return;
        }
        if (empty) empty.style.display = 'none';
        var sorted = tasks.slice().sort(function(a,b){
            if (a.status==='pending'&&b.status!=='pending') return -1;
            if (a.status!=='pending'&&b.status==='pending') return 1;
            return (b.priority||0)-(a.priority||0);
        });
        var html = '';
        for (var i=0;i<Math.min(sorted.length,20);i++){
            var t = sorted[i];
            html += '<tr>'+
                '<td style="font-family:monospace;color:var(--text-muted);font-size:11px;">'+escapeHtml((t.task_id||'-').slice(-6))+'</td>'+
                '<td><span style="font-size:11px;">'+escapeHtml(t.task_type||'-')+'</span></td>'+
                '<td>'+escapeHtml(t.qubit_count||'-')+'</td>'+
                '<td><span class="'+prioClass(t.priority)+'">'+escapeHtml(t.priority||'-')+'</span></td>'+
                '<td><span class="status-tag '+escapeHtml(t.status||'pending')+'">'+escapeHtml(statusText(t.status||'pending'))+'</span></td>'+
                '</tr>';
        }
        tbody.innerHTML = html;
    }

    function renderStrategies(strategies, current) {
        var c = document.getElementById('strategy-buttons');
        if (!c) return; // 新模板中策略按钮在顶部header中
        var html = '';
        var recommended = ['ppo','PPO'];
        for (var i=0;i<strategies.length;i++){
            var s = strategies[i];
            var isRec = recommended.indexOf(s) >= 0;
            var active = s===current ? ' active' : '';
            var recCls = isRec ? ' recommended' : '';
            html += '<button class="strategy-btn'+active+recCls+'" onclick="switchStrategy(\\''+s+'\\')">'+s+'</button>';
        }
        c.innerHTML = html;
    }

    function updateControls(status) {
        // 更新顶部策略选择器
        var sel = document.getElementById('strategy-select');
        if (!sel || !status) return;
        var opts = status.strategy_options || ['PPO','FCFS','SJF','Random','DQN'];
        var cur = (status.current_strategy||'PPO').toUpperCase();
        sel.innerHTML = '';
        for (var i=0;i<opts.length;i++) {
            var opt = document.createElement('option');
            opt.value = opts[i];
            opt.textContent = opts[i];
            if (opts[i].toUpperCase() === cur) opt.selected = true;
            sel.appendChild(opt);
        }
    }

    function renderBattle() {
        if (!battleState) return;
        var ppoR = battleState.ppo_total || battleState.ppo_reward || 0;
        var fcfsR = battleState.fcfs_total || battleState.fcfs_reward || 0;
        document.getElementById('battle-ppo').textContent = Math.round(ppoR);
        document.getElementById('battle-fcfs').textContent = Math.round(fcfsR);
        var diff = 0;
        if (fcfsR !== 0) {
            diff = ((ppoR - fcfsR) / Math.abs(fcfsR)) * 100;
        } else if (ppoR > 0) {
            diff = 100;
        }
        var diffEl = document.getElementById('battle-diff');
        diffEl.textContent = (diff>=0?'+':'')+diff.toFixed(1)+'%';
        diffEl.style.color = diff>=0 ? '#34d399' : '#f87171';
        document.getElementById('battle-step-info').textContent = 'Step: '+(battleState.step||0);
    }

    function renderDecisions(logs) {
        var list = document.getElementById('decision-list');
        var badge = document.getElementById('decision-count');
        // 适配API返回格式：可能是数组或{decisions: [...]}
        var arr = logs;
        if (logs && logs.decisions) arr = logs.decisions;
        if (!arr || arr.length === 0) return;
        badge.textContent = arr.length+' 条';
        decisionLog = arr.slice(-20);
        var html = '';
        // 统计决策类型用于饼图
        var counts = {0:0, 1:0, 2:0};
        for (var i=decisionLog.length-1; i>=Math.max(0,decisionLog.length-15); i--) {
            var d = decisionLog[i];
            var actLabel = d.action_label || actionText(d.action).text;
            var actCls = d.action === 0 ? 'action-classical' : d.action === 1 ? 'action-quantum' : 'action-hybrid';
            var rwd = d.reward||0;
            counts[d.action] = (counts[d.action]||0) + 1;
            html += '<div class="decision-item">'+
                '<div class="decision-step">#'+escapeHtml(d.step||i)+'</div>'+
                '<div class="decision-content">'+
                    '<span class="decision-action '+actCls+'">'+escapeHtml(actLabel)+'</span>'+
                    '<span class="decision-reward '+(rwd<0?'negative':'')+'">'+(rwd>=0?'+':'')+rwd.toFixed(1)+'</span>'+
                    '<div class="decision-meta">'+(d.task_id?'任务 '+escapeHtml((d.task_id+'').slice(-6)):'')+(d.source?' · '+escapeHtml(d.source):'')+' · ep_r='+((d.episode_reward||0).toFixed(0))+'</div>'+
                '</div></div>';
        }
        list.innerHTML = html;

        // 更新决策分布饼图（统计全部决策，不只最近15条）
        var allCounts = {0:0, 1:0, 2:0};
        for (var j=0; j<arr.length; j++) {
            var a = arr[j].action;
            if (a === 0 || a === 1 || a === 2) allCounts[a]++;
        }
        updateDecisionPie(allCounts, arr.length);
    }

    // 绘制调度决策分布饼图
    function updateDecisionPie(counts, total) {
        var canvas = document.getElementById('decision-pie-canvas');
        if (!canvas) return;
        var ctx = canvas.getContext('2d');
        var dpr = window.devicePixelRatio || 1;
        var size = 160;
        canvas.width = size * dpr;
        canvas.height = size * dpr;
        canvas.style.width = size + 'px';
        canvas.style.height = size + 'px';
        ctx.scale(dpr, dpr);

        var cx = size/2, cy = size/2, r = size/2 - 8;
        var innerR = r * 0.58; // 环形图

        var colors = [
            {main: '#a78bfa', glow: 'rgba(167,139,250,0.3)'}, // 0=经典-紫
            {main: '#3bc9db', glow: 'rgba(59,201,219,0.3)'}, // 1=量子-青
            {main: '#fbbf24', glow: 'rgba(251,191,36,0.3)'}   // 2=混合-琥珀
        ];
        var labels = ['经典资源', '量子资源', '混合执行'];
        var values = [counts[0]||0, counts[1]||0, counts[2]||0];
        var sum = values[0]+values[1]+values[2];
        if (sum === 0) sum = 1;

        // 更新图例数字
        document.getElementById('pie-total').textContent = total + ' 次决策';
        document.getElementById('pie-quantum').textContent = values[1]+' ('+((values[1]/sum)*100).toFixed(0)+'%)';
        document.getElementById('pie-classical').textContent = values[0]+' ('+((values[0]/sum)*100).toFixed(0)+'%)';
        document.getElementById('pie-hybrid').textContent = values[2]+' ('+((values[2]/sum)*100).toFixed(0)+'%)';

        ctx.clearRect(0, 0, size, size);

        // 绘制发光外环
        var grd = ctx.createRadialGradient(cx, cy, innerR, cx, cy, r+4);
        grd.addColorStop(0, 'rgba(59,201,219,0.02)');
        grd.addColorStop(1, 'rgba(59,201,219,0.08)');
        ctx.fillStyle = grd;
        ctx.beginPath();
        ctx.arc(cx, cy, r+4, 0, Math.PI*2);
        ctx.fill();

        var startAngle = -Math.PI/2;
        for (var i=0; i<3; i++) {
            if (values[i] === 0) continue;
            var sliceAngle = (values[i]/sum) * Math.PI*2;
            var endAngle = startAngle + sliceAngle;

            // 扇区发光
            ctx.shadowColor = colors[i].glow;
            ctx.shadowBlur = 8;

            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.arc(cx, cy, r, startAngle, endAngle);
            ctx.closePath();
            ctx.fillStyle = colors[i].main;
            ctx.fill();

            ctx.shadowBlur = 0;
            startAngle = endAngle;
        }

        // 内圆（挖空成环形）- 所有扇区绘制完后再挖
        ctx.beginPath();
        ctx.arc(cx, cy, innerR, 0, Math.PI*2);
        ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--surface').trim() || '#0e1422';
        ctx.fill();

        // 中心圆
        ctx.beginPath();
        ctx.arc(cx, cy, innerR-2, 0, Math.PI*2);
        ctx.fillStyle = 'rgba(7,11,20,0.8)';
        ctx.fill();

        // 中心文字
        ctx.fillStyle = 'rgba(230,237,243,0.9)';
        ctx.font = '600 18px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(total, cx, cy-4);
        ctx.fillStyle = 'rgba(230,237,243,0.4)';
        ctx.font = '400 9px Inter, sans-serif';
        ctx.fillText('DECISIONS', cx, cy+12);
    }

    function renderExplainability(data) {
        var bars = document.getElementById('feature-bars');
        var header = document.getElementById('explain-header');
        // 适配API返回格式：{latest: {...}} 或直接是特征数据
        var d = data;
        if (data && data.latest) d = data.latest;
        if (!d) {
            bars.innerHTML = '<div class="empty-state" style="padding:20px;">等待特征数据...</div>';
            return;
        }
        var feats = d.feature_contributions || d.features;
        header.textContent = 'Step '+(d.step||'?')+' 决策: '+(d.action_label||'');
        if (!feats || Object.keys(feats).length === 0) {
            bars.innerHTML = '<div class="empty-state" style="padding:20px;">暂无特征贡献数据</div>';
            return;
        }
        // 将对象转为数组
        var featArr = [];
        for (var k in feats) {
            featArr.push({name: k, contribution: feats[k]});
        }
        // 按绝对值排序取前6
        featArr.sort(function(a,b){return Math.abs(b.contribution)-Math.abs(a.contribution);});
        featArr = featArr.slice(0, 6);
        var maxAbs = 1;
        for (var i=0;i<featArr.length;i++) maxAbs = Math.max(maxAbs, Math.abs(featArr[i].contribution||0));
        var html = '';
        for (var i=0;i<featArr.length;i++) {
            var f = featArr[i];
            var v = f.contribution||0;
            var pct = (Math.abs(v)/maxAbs)*100;
            var cls = v>=0?'positive':'negative';
            var shortName = f.name.length > 8 ? f.name.slice(0,7)+'…' : f.name;
            html += '<div class="feature-bar-row">'+
                '<div class="feature-name" title="'+f.name+'">'+shortName+'</div>'+
                '<div class="feature-bar-bg"><div class="feature-bar-fill '+cls+'" style="width:'+pct+'%"></div></div>'+
                '<div class="feature-value">'+v.toFixed(2)+'</div>'+
                '</div>';
        }
        bars.innerHTML = html;
    }

    // 内置权威策略排名（当API不可用时使用，基于14维公平重训结果）
    var BUILTIN_RANKINGS = [
        {name: 'PPO', score: 2374},
        {name: 'DQN', score: 1510},
        {name: 'SJF', score: 1462},
        {name: 'FCFS', score: 1261},
        {name: 'Random', score: 1247},
        {name: 'MAPPO', score: 1200},
        {name: 'Greedy', score: -26},
        {name: 'Quantum-Only', score: -920}
    ];

    function renderRanking(stats) {
        var body = document.getElementById('ranking-body');
        var ranks = null;
        if (stats && stats.rankings) {
            ranks = stats.rankings;
        } else if (stats && !stats.error) {
            // 可能直接是ranking数组
            ranks = Array.isArray(stats) ? stats : null;
        }
        if (!ranks) {
            ranks = BUILTIN_RANKINGS;
        }
        ranks.sort(function(a,b){return (b.score||0)-(a.score||0);});
        var maxScore = Math.max.apply(null, ranks.map(function(r){return r.score||0;}));
        var minScore = Math.min.apply(null, ranks.map(function(r){return r.score||0;}));
        var range = maxScore - minScore || 1;
        var html = '';
        for (var i=0;i<ranks.length;i++) {
            var r = ranks[i];
            var pct = Math.max(2, ((r.score - minScore)/range)*100);
            var isBest = i===0;
            var posClass = i===0?'gold':'';
            html += '<div class="ranking-bar">'+
                '<div class="ranking-pos '+posClass+'">'+(i+1)+'</div>'+
                '<div class="ranking-name">'+r.name+'</div>'+
                '<div class="ranking-bar-bg"><div class="ranking-bar-fill '+(isBest?'best':'')+'" style="width:'+pct+'%"></div></div>'+
                '<div class="ranking-score">'+Math.round(r.score)+'</div>'+
                '</div>';
        }
        body.innerHTML = html;
    }

    // ================================================================
    // API 调用
    // ================================================================
    async function fetchInitial() {
        try {
            var [sResp, tResp] = await Promise.all([
                fetch('/api/status'),
                fetch('/api/tasks').catch(function(){return {json:function(){return [];}};})
            ]);
            currentStatus = await sResp.json();
            currentTasks = await tResp.json();
            strategyOptions = currentStatus.strategy_options || ['PPO','FCFS','SJF','Random','DQN'];

            renderStatus(currentStatus);
            renderTasks(currentTasks);
            renderStrategies(strategyOptions, currentStatus.current_strategy);
            updateControls(currentStatus);

            // 拉取battle、决策日志、可解释性、资源历史
            fetch('/api/battle/status').then(function(r){return r.json();}).then(function(d){battleState=d;renderBattle();}).catch(function(){});
            fetch('/api/decision-log').then(function(r){return r.json();}).then(renderDecisions).catch(function(){});
            fetch('/api/explainability/latest').then(function(r){return r.json();}).then(renderExplainability).catch(function(){});
            fetch('/api/resource-history').then(function(r){return r.json();}).then(function(d){
                if (Array.isArray(d)) resourceHistory = d.slice(-200);
            }).catch(function(){});
            fetch('/api/ppo/stats').then(function(r){return r.json();}).then(renderRanking).catch(function(){renderRanking(null);});
        } catch(e) { console.error('初始加载失败:', e); renderRanking(null); }
    }

    async function submitTask() {
        var payload = {
            user_id: document.getElementById('input-user').value||'user_001',
            task_type: document.getElementById('input-type').value,
            priority: parseInt(document.getElementById('input-priority').value),
            qubit_count: parseInt(document.getElementById('input-qubits').value)||10,
            circuit_depth: parseInt(document.getElementById('input-depth').value)||100,
            estimated_time: parseFloat(document.getElementById('input-time').value)||60.0
        };
        try {
            var resp = await fetch('/api/tasks', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
            var r = await resp.json();
            showToast(r.task_id?'任务已提交: '+r.task_id:(r.message||'已提交'), r.task_id?'success':'info');
        } catch(e) { showToast('提交失败: '+e.message, 'warn'); }
    }

    async function switchStrategy(s) {
        try {
            var resp = await fetch('/api/strategy?strategy='+encodeURIComponent(s), {method:'POST'});
            var r = await resp.json();
            showToast(r.message||('已切换到 '+s), r.success?'success':'warn');
            // 更新按钮状态
            var btns = document.querySelectorAll('.strategy-btn');
            for (var i=0;i<btns.length;i++) btns[i].classList.remove('active');
            var all = document.querySelectorAll('.strategy-btn');
            for (var i=0;i<all.length;i++){
                if (all[i].textContent.trim()===s) all[i].classList.add('active');
            }
        } catch(e) { showToast('切换失败: '+e.message, 'warn'); }
    }

    // ---- Battle API ----
    async function battleStart() {
        try {
            await fetch('/api/battle/start', {method:'POST'});
            await fetchBattleStatus();
        } catch(e){}
    }
    async function battleStep() {
        try {
            await fetch('/api/battle/step', {method:'POST'});
            await fetchBattleStatus();
        } catch(e){}
    }
    async function battleReset() {
        try {
            await fetch('/api/battle/reset', {method:'POST'});
            battleState = null;
            renderBattle();
        } catch(e){}
    }
    async function fetchBattleStatus() {
        try {
            var r = await fetch('/api/battle/status');
            battleState = await r.json();
            renderBattle();
        } catch(e){}
    }
    function setBattleAuto(auto) {
        battleAutoMode = auto;
        document.getElementById('btn-battle-auto').classList.toggle('active', auto);
        document.getElementById('btn-battle-step').classList.toggle('active', !auto);
        document.getElementById('battle-status').textContent = auto ? '● 自动运行中' : '○ 手动模式';
        document.getElementById('battle-status').className = 'badge '+(auto?'live':'');
        if (battleAutoTimer) { clearInterval(battleAutoTimer); battleAutoTimer = null; }
        if (auto) {
            battleAutoTimer = setInterval(function(){ battleStep(); }, 1500);
        }
    }

    // HTTP 轮询兜底（WebSocket 挂掉时）
    function startHttpPolling() {
        if (httpPollTimer) return;
        httpPollTimer = setInterval(async function(){
            try {
                var s = await (await fetch('/api/status')).json();
                currentStatus = s;
                renderStatus(s);
                var t = await (await fetch('/api/tasks')).json();
                currentTasks = t;
                renderTasks(t);
                fetchBattleStatus();
                // 每5秒拉一次决策和可解释性
                if (Math.random() < 0.3) {
                    fetch('/api/decision-log').then(function(r){return r.json();}).then(renderDecisions).catch(function(){});
                    fetch('/api/explainability/latest').then(function(r){return r.json();}).then(renderExplainability).catch(function(){});
                }
            } catch(e){}
        }, 2000);
    }

    // ================================================================
    // WebSocket
    // ================================================================
    function connectWS() {
        var proto = location.protocol==='https:'?'wss:':'ws:';
        ws = new WebSocket(proto+'//'+location.host+'/ws');
        var sEl = document.getElementById('ws-status');

        ws.onopen = function() {
            sEl.textContent = 'WebSocket 已连接';
            sEl.className = 'ws-status connected';
            if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer=null; }
        };
        ws.onmessage = function(ev) {
            var msg = JSON.parse(ev.data);
            if (msg.type==='init') {
                currentStatus = msg.status;
                currentTasks = msg.tasks||[];
                strategyOptions = currentStatus.strategy_options||strategyOptions;
                renderStatus(currentStatus); renderTasks(currentTasks);
                renderStrategies(strategyOptions, currentStatus.current_strategy);
                if (msg.ppo_stats) { ppoStats = msg.ppo_stats; renderRanking(ppoStats); }
            } else if (msg.type==='status_update') {
                if (msg.status) { currentStatus=msg.status; renderStatus(currentStatus); }
                if (msg.tasks) { currentTasks=msg.tasks; renderTasks(currentTasks); }
                if (msg.decision) { decisionLog.push(msg.decision); if(decisionLog.length>50)decisionLog.shift(); renderDecisions(decisionLog); }
                if (msg.explainability) { renderExplainability(msg.explainability); }
                if (msg.ppo_stats) { ppoStats = msg.ppo_stats; renderRanking(ppoStats); }
            } else if (msg.type==='task_added') {
                if (msg.status) { currentStatus=msg.status; renderStatus(currentStatus); }
                fetch('/api/tasks').then(function(r){return r.json();}).then(function(t){currentTasks=t;renderTasks(t);});
            } else if (msg.type==='strategy_changed') {
                if (msg.status) { currentStatus=msg.status; renderStatus(currentStatus); }
                renderStrategies(currentStatus.strategy_options||strategyOptions, msg.new_strategy);
                showToast('策略已切换: '+msg.new_strategy, 'info');
            }
        };
        ws.onclose = function() {
            sEl.textContent = 'WebSocket 重连中...';
            sEl.className = 'ws-status disconnected';
            startHttpPolling();
            reconnectTimer = setTimeout(connectWS, 3000);
        };
        ws.onerror = function() { ws.close(); };

        setInterval(function(){
            if (ws && ws.readyState===WebSocket.OPEN) ws.send(JSON.stringify({action:'ping'}));
        }, 30000);
    }

    // ================================================================
    // 场景-算法决策树交互
    // ================================================================
    var dtScenarioData = {
        balanced: {
            title: "均衡负载场景",
            desc: "任务到达率稳定，量子/经典任务比例均衡（日常科研计算、稳态运行）",
            params: "arrival_rate=0.5, quantum_ratio=0.7, max_steps=200",
            strategies: [
                {rank:1, name:"PPO", reward:2746.94, lift:"+88.3%", note:"综合最优"},
                {rank:2, name:"DQN", reward:1527.65, lift:"+4.7%", note:""},
                {rank:3, name:"SJF", reward:1462.39, lift:"+0.2%", note:""},
                {rank:4, name:"FCFS", reward:1458.77, lift:"基线", note:"工业界默认"},
                {rank:5, name:"Random", reward:1247.17, lift:"-14.5%", note:""},
                {rank:6, name:"Greedy", reward:-25.95, lift:"-101.8%", note:"此场景崩溃"}
            ],
            conclusion: "均衡负载下PPO大幅领先所有基线（+88.3%），是稳态运行的最优选择。统计显著性：Mann-Whitney U p=1.03×10⁻⁴²，效应量r=-0.71（大效应）。",
            recommend: "PPO"
        },
        high_load: {
            title: "高负载积压场景",
            desc: "任务密集到达，队列持续积压（项目截止日前集中提交、批量任务）",
            params: "arrival_rate=2.5, max_steps=80",
            strategies: [
                {rank:1, name:"Greedy", reward:1712.99, lift:"+169.6%", note:"极致吞吐但风险大"},
                {rank:2, name:"PPO", reward:1628.74, lift:"+156.5%", note:"仅落后4.9%，不崩溃"},
                {rank:3, name:"Random", reward:602.55, lift:"-5.1%", note:""},
                {rank:4, name:"FCFS", reward:635.07, lift:"基线", note:""},
                {rank:5, name:"SJF", reward:541.77, lift:"-14.7%", note:""}
            ],
            conclusion: "高负载下Greedy以4.9%微弱优势领先PPO，但Greedy在量子波动场景会崩溃（-95.3%）。PPO是稳定高吞吐的安全选择。生产建议：以PPO为主，监控到持续高负载时可临时切换Greedy，资源波动时立即切回。",
            recommend: "PPO（稳定首选）/ Greedy（极致吞吐，需监控）"
        },
        tidal: {
            title: "潮汐波动场景",
            desc: "量子/经典任务比例周期性切换（白天量子实验多/夜间经典数据处理多）",
            params: "前100步量子密集(90%)→后100步经典密集(90%)",
            strategies: [
                {rank:1, name:"Greedy", reward:2351.40, lift:"+65.9%", note:"响应最快"},
                {rank:2, name:"SJF", reward:1662.66, lift:"+17.3%", note:"周期可预测时有效"},
                {rank:3, name:"PPO", reward:1898.92, lift:"+34.0%", note:"自适应强，未专门调优"},
                {rank:4, name:"FCFS", reward:1417.52, lift:"基线", note:""},
                {rank:5, name:"Random", reward:1077.01, lift:"-24.0%", note:""}
            ],
            conclusion: "潮汐负载下Greedy响应最快，但PPO仍比FCFS高34%。若波动周期可预测，SJF/Greedy有优势；若波动模式不可预测，PPO的自适应能力更可靠。",
            recommend: "已知周期→SJF/Greedy；未知模式→PPO"
        },
        burst: {
            title: "突发洪峰场景",
            desc: "短时间大量任务涌入后恢复平静（论文投稿前集体实验、课程作业集中提交）",
            params: "脉冲式到达率（短期峰值3.0+）",
            strategies: [
                {rank:1, name:"FCFS", reward:"⭐⭐⭐⭐⭐", lift:"最稳定", note:"公平性保证，不会崩溃"},
                {rank:2, name:"PPO", reward:"⭐⭐⭐⭐", lift:"吞吐高", note:"适应性强，配合限流更优"},
                {rank:3, name:"Greedy", reward:"⭐⭐", lift:"初期快", note:"易陷入局部最优导致积压"},
                {rank:4, name:"SJF", reward:"⭐⭐", lift:"需预知时长", note:"突发场景信息不足"}
            ],
            conclusion: "突发型任务到达时，FCFS提供最稳定的表现（公平性保证无饥饿）。PPO在吞吐量上仍有优势，但需要配合队列限流机制防止过载。",
            recommend: "追求稳定→FCFS；追求消化速度→PPO+队列限流"
        },
        volatile: {
            title: "量子资源波动场景",
            desc: "量子计算机校准/维护/保真度下降导致资源可用性动态变化",
            params: "模拟量子比特保真度波动、校准窗口、维护时段",
            strategies: [
                {rank:1, name:"PPO", reward:2997.68, lift:"+91.4%", note:"优势最大场景"},
                {rank:2, name:"FCFS", reward:1566.39, lift:"基线", note:""},
                {rank:3, name:"SJF", reward:1231.45, lift:"-21.4%", note:""},
                {rank:4, name:"Random", reward:926.30, lift:"-40.9%", note:""},
                {rank:5, name:"Greedy", reward:141.47, lift:"-91.0%", note:"此场景崩溃(-95.3%)"}
            ],
            conclusion: "PPO在量子资源波动场景优势最大（比FCFS高91.4%），验证了RL策略对动态资源状态的强适应性。Greedy在此场景崩溃（-95.3%），说明简单贪心规则无法应对资源可用性变化。这是PPO最具价值的应用场景。",
            recommend: "PPO（强烈推荐）"
        }
    };

    var dtActiveScenario = null;

    function dtSelectRoot() {
        // 点击根节点，收起详情
        document.getElementById('dt-detail-panel').style.display = 'none';
        document.querySelectorAll('.dt-scenario').forEach(function(el){ el.classList.remove('active'); });
        document.querySelectorAll('.dt-branch').forEach(function(el){ el.style.borderColor = ''; });
        dtActiveScenario = null;
    }

    function dtSelectScenario(key) {
        var data = dtScenarioData[key];
        if (!data) return;

        // 高亮选中
        document.querySelectorAll('.dt-scenario').forEach(function(el){ el.classList.remove('active'); });
        document.querySelectorAll('.dt-branch').forEach(function(el){ el.style.borderColor = ''; });
        var el = document.getElementById('dt-sc-' + key);
        if (el) el.classList.add('active');

        // 构建详情表格
        var html = '<div style="font-weight:600;font-size:13px;color:var(--ink);margin-bottom:6px;">' + data.title + '</div>';
        html += '<div style="font-size:11px;color:var(--ink-3);margin-bottom:10px;">' + data.desc + '</div>';
        html += '<div style="font-size:10px;font-family:var(--font-mono);color:var(--ink-4);margin-bottom:12px;">参数: ' + data.params + '</div>';
        html += '<table class="dt-detail-table"><thead><tr><th>排名</th><th>策略</th><th>平均奖励</th><th>相对FCFS</th><th>备注</th></tr></thead><tbody>';
        data.strategies.forEach(function(s){
            var rankClass = s.rank===1?'rank-1':s.rank===2?'rank-2':s.rank===3?'rank-3':'';
            html += '<tr><td class="' + rankClass + '">' + (s.rank===1?'🥇':s.rank===2?'🥈':s.rank===3?'🥉':s.rank) + '</td>';
            html += '<td>' + s.name + '</td>';
            html += '<td style="font-family:var(--font-mono);">' + (typeof s.reward==='number'?s.reward.toFixed(2):s.reward) + '</td>';
            html += '<td style="font-family:var(--font-mono);">' + s.lift + '</td>';
            html += '<td style="color:var(--ink-3);font-size:10px;">' + s.note + '</td></tr>';
        });
        html += '</tbody></table>';
        html += '<div class="dt-detail-conclusion"><strong style="color:var(--brand);">结论：</strong>' + data.conclusion + '<br><strong style="color:var(--success);margin-top:4px;display:inline-block;">推荐策略：' + data.recommend + '</strong></div>';

        document.getElementById('dt-detail-content').innerHTML = html;
        document.getElementById('dt-detail-panel').style.display = 'block';
        dtActiveScenario = key;
    }

    // ================================================================
    // 真机Demo功能
    // ================================================================
    var realDemoSubmitting = false;

    function loadRealSubmissions() {
        fetch('/api/real-submissions').then(function(r){return r.json();}).then(function(data){
            var listEl = document.getElementById('real-submissions-list');
            if (!listEl) return;
            var subs = data.submissions || [];
            if (subs.length === 0) {
                listEl.innerHTML = '<div style="font-size:10px;color:var(--ink-4);text-align:center;padding:8px;">暂无真机提交记录</div>';
                return;
            }
            var html = '';
            subs.slice(0, 8).forEach(function(s){
                var statusCls = s.status === 'completed' || s.status === 'success' ? 'success' : 'pending';
                var statusText = s.status === 'completed' || s.status === 'success' ? '✓ 完成' : '⏳ 等待';
                var time = s.timestamp || s.submit_time || '';
                if (time) {
                    try { time = new Date(time).toLocaleTimeString('zh-CN', {hour:'2-digit',minute:'2-digit',second:'2-digit'}); } catch(e){}
                }
                var machine = s.machine || s.backend || 'tianyan_s';
                var taskId = s.task_id || s.id || '-';
                html += '<div class="real-submission-item">';
                html += '<span class="real-sub-id">' + escapeHtml(taskId.substring(0,12)) + '...</span>';
                html += '<span class="real-sub-status ' + escapeHtml(statusCls) + '">' + escapeHtml(statusText) + '</span>';
                html += '<div class="real-sub-meta">' + escapeHtml(machine) + ' · ' + escapeHtml(time) + '</div>';
                html += '</div>';
            });
            listEl.innerHTML = html;
        }).catch(function(){});
    }

    function loadRealMachines() {
        fetch('/api/real-machines').then(function(r){return r.json();}).then(function(data){
            var machines = data.machines || [];
            var apiKeyStatus = document.getElementById('api-key-status');
            var badge = document.getElementById('real-machine-badge');

            if (machines.length > 0) {
                // 有真机数据，更新状态
                if (apiKeyStatus) {
                    apiKeyStatus.style.display = 'none';
                }
                machines.forEach(function(m){
                    var mid = m.id || m.name || '';
                    var statusEl = document.getElementById('machine-' + mid + '-status');
                    if (statusEl) {
                        var isRunning = m.status === 'running' || m.status === 'online' || m.status === 'available';
                        if (isRunning) {
                            statusEl.style.background = 'rgba(52,211,153,0.1)';
                            statusEl.style.color = 'var(--success)';
                            statusEl.textContent = '● ' + (m.status || '运行中');
                        } else if (m.status === 'calibrating') {
                            statusEl.style.background = 'rgba(251,191,36,0.1)';
                            statusEl.style.color = 'var(--warning)';
                            statusEl.textContent = '◐ 校准中';
                        } else if (m.status === 'maintenance') {
                            statusEl.style.background = 'rgba(248,113,113,0.1)';
                            statusEl.style.color = 'var(--error)';
                            statusEl.textContent = '⚠ 维护中';
                        } else {
                            statusEl.textContent = m.status || '未知';
                        }
                    }
                });
                if (badge) {
                    badge.textContent = machines.length + '台在线';
                }
            } else {
                // 无真机连接（无API Key），保持Mock状态
                if (apiKeyStatus) {
                    apiKeyStatus.style.display = 'block';
                }
            }
        }).catch(function(){});
    }

    function submitRealDemo() {
        if (realDemoSubmitting) return;
        realDemoSubmitting = true;
        var btn = document.getElementById('btn-real-demo');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation:spin 1s linear infinite;"><circle cx="12" cy="12" r="10" stroke-dasharray="40" stroke-dashoffset="10"/></svg> 提交中...';
        }

        // 提交一个Demo任务到真机（Mock模式：走仿真任务提交接口模拟真机流程）
        var demoTask = {
            user_id: 'demo_user',
            task_type: 'quantum',
            priority: 3,
            qubit_count: 5,
            circuit_depth: 20,
            estimated_time: 30
        };

        fetch('/api/tasks', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(demoTask)
        }).then(function(r){return r.json();}).then(function(data){
            showToast('Demo任务已提交: ' + (data.task_id || '成功'), 'success');
            // 添加一条Mock的真机提交记录
            var listEl = document.getElementById('real-submissions-list');
            if (listEl) {
                var mockHtml = '<div class="real-submission-item" style="border-color:rgba(59,201,219,0.2);">';
                mockHtml += '<span class="real-sub-id">' + (data.task_id||'DEMO-'+Date.now()).substring(0,12) + '...</span>';
                mockHtml += '<span class="real-sub-status pending">⏳ Mock演示</span>';
                mockHtml += '<div class="real-sub-meta">tianyan_s · ' + new Date().toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit',second:'2-digit'}) + ' (Demo模式)</div>';
                mockHtml += '</div>';
                listEl.insertAdjacentHTML('afterbegin', mockHtml);
            }
        }).catch(function(e){
            showToast('提交失败: ' + e.message, 'error');
        }).finally(function(){
            realDemoSubmitting = false;
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> 提交真机Demo任务';
            }
        });
    }

    // ================================================================
    // 初始化
    // ================================================================
    // ================================================================
    // 量子粒子背景动画
    // ================================================================
    (function initQuantumBg(){
        var canvas = document.getElementById('quantum-bg');
        if (!canvas) return;
        var ctx = canvas.getContext('2d');
        var particles = [];
        var PARTICLE_COUNT = 60;
        var CONNECTION_DIST = 140;
        var w, h;

        function resize() {
            w = canvas.width = window.innerWidth * window.devicePixelRatio;
            h = canvas.height = window.innerHeight * window.devicePixelRatio;
            canvas.style.width = window.innerWidth + 'px';
            canvas.style.height = window.innerHeight + 'px';
        }
        resize();
        window.addEventListener('resize', resize);

        // 创建粒子
        for (var i = 0; i < PARTICLE_COUNT; i++) {
            particles.push({
                x: Math.random() * w,
                y: Math.random() * h,
                vx: (Math.random() - 0.5) * 0.3 * window.devicePixelRatio,
                vy: (Math.random() - 0.5) * 0.3 * window.devicePixelRatio,
                r: (Math.random() * 1.2 + 0.4) * window.devicePixelRatio,
                alpha: Math.random() * 0.5 + 0.2,
                pulse: Math.random() * Math.PI * 2,
                pulseSpeed: Math.random() * 0.01 + 0.005,
                color: Math.random() > 0.7 ? 'a78bfa' : '3bc9db'
            });
        }

        function draw() {
            ctx.clearRect(0, 0, w, h);
            var dpr = window.devicePixelRatio;

            // 更新和绘制粒子
            for (var i = 0; i < particles.length; i++) {
                var p = particles[i];
                p.x += p.vx;
                p.y += p.vy;
                p.pulse += p.pulseSpeed;

                // 边界环绕
                if (p.x < -10) p.x = w + 10;
                if (p.x > w + 10) p.x = -10;
                if (p.y < -10) p.y = h + 10;
                if (p.y > h + 10) p.y = -10;

                var pulseAlpha = p.alpha * (0.7 + 0.3 * Math.sin(p.pulse));
                var rgb = p.color === 'a78bfa' ? '167,139,250' : '59,201,219';

                // 发光
                var grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 4);
                grd.addColorStop(0, 'rgba(' + rgb + ',' + pulseAlpha * 0.8 + ')');
                grd.addColorStop(1, 'rgba(' + rgb + ',0)');
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r * 4, 0, Math.PI * 2);
                ctx.fillStyle = grd;
                ctx.fill();

                // 核心点
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(' + rgb + ',' + pulseAlpha + ')';
                ctx.fill();
            }

            // 绘制连接线（量子纠缠效果）
            for (var i = 0; i < particles.length; i++) {
                for (var j = i + 1; j < particles.length; j++) {
                    var dx = particles[i].x - particles[j].x;
                    var dy = particles[i].y - particles[j].y;
                    var dist = Math.sqrt(dx * dx + dy * dy);
                    var maxDist = CONNECTION_DIST * dpr;
                    if (dist < maxDist) {
                        var alpha = (1 - dist / maxDist) * 0.12;
                        var sameColor = particles[i].color === particles[j].color;
                        var rgb = sameColor
                            ? (particles[i].color === 'a78bfa' ? '167,139,250' : '59,201,219')
                            : '148,184,220';
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.strokeStyle = 'rgba(' + rgb + ',' + alpha + ')';
                        ctx.lineWidth = 0.5 * dpr;
                        ctx.stroke();
                    }
                }
            }

            requestAnimationFrame(draw);
        }
        draw();
    })();

    // ================================================================
    // 入口
    // ================================================================
    (function init(){
        // 启动Canvas动画循环
        redrawCharts();

        // 加载数据
        fetchInitial().then(function(){
            // 隐藏loading
            setTimeout(function(){
                var ld = document.getElementById('loading');
                ld.classList.add('hidden');
                setTimeout(function(){ ld.style.display='none'; }, 500);
            }, 800);
        });

        // 启动对战
        battleStart().then(function(){
            setBattleAuto(true);
        });

        // 连接WS
        connectWS();

        // 加载真机状态和提交记录
        loadRealMachines();
        loadRealSubmissions();
        setInterval(loadRealMachines, 15000);  // 每15秒刷新一次真机状态
        setInterval(loadRealSubmissions, 10000);

        // 兜底HTTP轮询
        setTimeout(startHttpPolling, 5000);
    })();
    </script>
</body>
</html>"""
