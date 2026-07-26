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
        /* ===== 量子实验室控制台主题 (Quantum Lab Console) ===== */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg-primary: #0c0e13;
            --bg-secondary: #11141a;
            --bg-card: #151820;
            --bg-card-alt: #1a1e28;
            --bg-input: #12151c;
            --border: #252a35;
            --border-strong: #353c4a;
            --text-primary: #e4e0d8;
            --text-secondary: #9a958c;
            --text-muted: #5c5850;
            --accent: #d4943a;
            --accent-dark: #b07828;
            --accent-light: #e8b060;
            --green: #4e8a5c;
            --green-light: #6aab7a;
            --amber-warn: #c49040;
            --red: #a04840;
            --cyan: #4a9e9e;
            --cyan-light: #68bcbc;
            --blue-slate: #5a7080;
            --shadow-card: 0 1px 4px rgba(0,0,0,0.3);
            --radius: 0;
            --font-mono: "JetBrains Mono", Consolas, "Courier New", "SF Mono", monospace;
            --font-sans: "PingFang SC", "Microsoft YaHei", "Noto Sans SC", -apple-system, sans-serif;
        }
        body {
            font-family: var(--font-sans);
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
            font-size: 13px;
        }
        /* CRT扫描线质感 - 极淡 */
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: repeating-linear-gradient(
                0deg,
                transparent,
                transparent 2px,
                rgba(0,0,0,0.03) 2px,
                rgba(0,0,0,0.03) 4px
            );
            pointer-events: none;
            z-index: 9999;
            opacity: 0.4;
        }
        /* 微弱暗角 */
        body::after {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(ellipse at center, transparent 60%, rgba(0,0,0,0.3) 100%);
            pointer-events: none;
            z-index: 0;
        }

        /* ===== 顶部标题栏 ===== */
        .header {
            position: relative;
            z-index: 1;
            background: var(--bg-card);
            border-bottom: 1px solid var(--border-strong);
            padding: 14px 28px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header::after {
            content: '';
            position: absolute;
            bottom: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--accent), var(--cyan), transparent);
            opacity: 0.5;
        }
        .header-left { display: flex; align-items: center; gap: 14px; }
        .logo {
            width: 40px; height: 40px;
            background: var(--accent);
            border-radius: 0;
            display: flex; align-items: center; justify-content: center;
            font-size: 20px; font-weight: 700; color: #0c0e13;
            font-family: var(--font-mono);
            letter-spacing: -1px;
            position: relative;
        }
        .logo::after {
            content: '';
            position: absolute;
            inset: -2px;
            border: 1px solid var(--accent);
            opacity: 0.3;
        }
        .header-titles h1 {
            font-size: 18px; font-weight: 600;
            font-family: var(--font-sans);
            color: var(--text-primary);
            line-height: 1.3;
            letter-spacing: 1px;
        }
        .header-titles .subtitle {
            font-size: 10px; color: var(--text-muted);
            margin-top: 2px;
            font-family: var(--font-mono);
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .header-right { display: flex; align-items: center; gap: 10px; }
        .ws-status {
            font-size: 10px; padding: 5px 12px;
            border-radius: 0;
            background: var(--bg-input);
            border: 1px solid var(--border);
            display: flex; align-items: center; gap: 6px;
            font-family: var(--font-mono);
            letter-spacing: 0.5px;
            color: var(--text-muted);
        }
        .ws-status::before {
            content: ''; width: 6px; height: 6px;
            border-radius: 50%;
            background: var(--red);
            box-shadow: 0 0 4px var(--red);
        }
        .ws-status.connected::before { background: var(--green); box-shadow: 0 0 6px var(--green); }
        .ws-status.disconnected::before { background: var(--red); box-shadow: 0 0 4px var(--red); }
        .ws-status.connected { color: var(--green-light); border-color: rgba(78,138,92,0.3); }
        .ws-status.disconnected { color: var(--red); border-color: var(--border); }
        .model-badge {
            font-size: 10px; padding: 4px 10px;
            border-radius: 0;
            background: var(--bg-input);
            border: 1px solid var(--accent);
            color: var(--accent);
            font-family: var(--font-mono);
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        /* ===== 状态卡片区域 ===== */
        .status-cards {
            position: relative; z-index: 1;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1px;
            padding: 20px 28px 0;
            background: var(--bg-secondary);
        }
        .status-card {
            background: var(--bg-card);
            border: none;
            border-top: 2px solid var(--border);
            border-radius: 0;
            padding: 16px 18px;
            transition: border-color 0.2s;
            position: relative;
        }
        .status-card:hover {
            border-top-color: var(--accent);
            background: #181c25;
        }
        .card-blue { border-top-color: var(--cyan); }
        .card-purple { border-top-color: var(--accent); }
        .card-green { border-top-color: var(--green); }
        .card-amber { border-top-color: var(--amber-warn); }
        .card-cyan { border-top-color: var(--blue-slate); }
        .card-pink { border-top-color: #8a6050; }
        .card-label {
            font-size: 9px; color: var(--text-muted);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            font-weight: 600;
            font-family: var(--font-mono);
        }
        .card-value {
            font-size: 26px; font-weight: 700;
            line-height: 1.1;
            font-variant-numeric: tabular-nums;
            font-family: var(--font-mono);
            color: var(--text-primary);
        }
        .card-blue .card-value { color: var(--cyan-light); }
        .card-purple .card-value { color: var(--accent-light); }
        .card-green .card-value { color: var(--green-light); }
        .card-amber .card-value { color: var(--accent); }
        .card-cyan .card-value { color: var(--blue-slate); }
        .card-pink .card-value { color: #b07868; }
        .card-sub {
            font-size: 10px; color: var(--text-muted);
            margin-top: 6px;
            font-family: var(--font-mono);
        }
        .card-trend {
            display: inline-flex; align-items: center; gap: 3px;
            font-size: 10px; font-weight: 600;
            padding: 2px 6px; border-radius: 0;
            margin-top: 5px;
            font-family: var(--font-mono);
        }
        .trend-up { background: rgba(78,138,92,0.15); color: var(--green-light); }
        .trend-down { background: rgba(160,72,64,0.15); color: var(--red); }

        /* ===== 主内容区域 ===== */
        .main-content {
            position: relative; z-index: 1;
            padding: 16px 28px 28px;
            display: grid;
            grid-template-columns: 1fr 360px;
            gap: 14px;
        }
        .main-col { display: flex; flex-direction: column; gap: 14px; }
        .side-col { display: flex; flex-direction: column; gap: 14px; }

        /* ===== 通用面板样式 ===== */
        .panel {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 0;
            overflow: hidden;
        }
        .panel-header {
            padding: 10px 16px;
            border-bottom: 1px solid var(--border);
            display: flex; align-items: center; justify-content: space-between;
            background: var(--bg-card-alt);
        }
        .panel-header h2 {
            font-size: 12px; font-weight: 600;
            display: flex; align-items: center; gap: 8px;
            font-family: var(--font-sans);
            color: var(--text-primary);
            letter-spacing: 0.5px;
        }
        .panel-header h2 .icon {
            width: 20px; height: 20px;
            border-radius: 0;
            display: flex; align-items: center; justify-content: center;
            font-size: 9px; font-weight: 700;
            font-family: var(--font-mono);
            background: var(--border-strong);
            color: var(--text-primary);
            letter-spacing: 0;
        }
        .icon-battle { background: var(--accent) !important; color: #0c0e13 !important; }
        .icon-chart { background: var(--cyan) !important; color: #0c0e13 !important; }
        .icon-log { background: var(--border-strong) !important; }
        .icon-explain { background: var(--amber-warn) !important; color: #0c0e13 !important; }
        .icon-queue { background: var(--green) !important; color: #0c0e13 !important; }
        .icon-control { background: var(--blue-slate) !important; color: #fff !important; }
        .panel-header .badge {
            font-size: 9px; padding: 2px 8px;
            border-radius: 0;
            background: transparent;
            color: var(--text-muted);
            border: 1px solid var(--border);
            font-family: var(--font-mono);
            letter-spacing: 0.5px;
        }
        .panel-header .badge.live {
            background: transparent;
            color: var(--green-light);
            border-color: rgba(78,138,92,0.4);
            position: relative;
            padding-left: 16px;
        }
        .panel-header .badge.live::before {
            content: '';
            position: absolute;
            left: 5px; top: 50%;
            width: 6px; height: 6px;
            background: var(--green);
            border-radius: 50%;
            transform: translateY(-50%);
            box-shadow: 0 0 4px var(--green);
            animation: pulse-dot 2s infinite;
        }
        @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .panel-body { padding: 12px 16px; }

        /* ===== PPO vs FCFS 对战面板 ===== */
        .battle-panel { position: relative; }
        .battle-canvas-wrap {
            position: relative;
            width: 100%;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 0;
            overflow: hidden;
        }
        #battle-canvas { width: 100%; display: block; }
        .battle-stats {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 1px;
            margin-top: 0;
            background: var(--border);
            border: 1px solid var(--border);
            border-top: none;
        }
        .battle-stat {
            background: var(--bg-card-alt);
            border-radius: 0;
            padding: 10px 12px;
            text-align: center;
        }
        .battle-stat .label { font-size: 9px; color: var(--text-muted); margin-bottom: 4px; font-family:var(--font-mono); text-transform:uppercase; letter-spacing:0.8px; }
        .battle-stat .value { font-size: 20px; font-weight: 700; font-variant-numeric: tabular-nums; font-family:var(--font-mono); }
        .battle-stat.ppo .value { color: var(--accent-light); }
        .battle-stat.fcfs .value { color: var(--text-secondary); }
        .battle-stat.diff .value { color: var(--green-light); }
        .battle-controls {
            display: flex; gap: 0; margin-top: 8px;
            align-items: center;
            border-top: 1px solid var(--border);
            padding-top: 8px;
        }
        .battle-controls .btn-sm {
            padding: 5px 12px; font-size: 10px;
            border-radius: 0; border: 1px solid var(--border);
            background: var(--bg-input); color: var(--text-secondary);
            cursor: pointer; transition: all 0.15s;
            font-weight: 600;
            font-family: var(--font-mono);
            margin-right: -1px;
        }
        .battle-controls .btn-sm:hover { border-color: var(--accent); color: var(--accent); z-index:1; }
        .battle-controls .btn-sm.active {
            background: var(--accent);
            border-color: var(--accent); color: #0c0e13;
        }
        .battle-controls .battle-info {
            font-size: 10px; color: var(--text-muted);
            margin-left: auto;
            font-family: var(--font-mono);
        }
        .battle-legend {
            display: flex; gap: 16px; margin-top: 8px;
            font-size: 10px; color: var(--text-secondary);
            font-family: var(--font-mono);
        }
        .legend-item { display: flex; align-items: center; gap: 6px; }
        .legend-line { width: 18px; height: 2px; border-radius: 0; }
        .legend-ppo { background: var(--accent); box-shadow: 0 0 4px rgba(212,148,58,0.4); }
        .legend-fcfs { background: var(--text-muted); }
        .legend-baseline { background: var(--text-muted); border-top: 1px dashed var(--text-muted); height: 0; }

        /* ===== 资源趋势图 ===== */
        .chart-canvas-wrap {
            position: relative; width: 100%;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 0; overflow: hidden;
        }
        #resource-canvas { width: 100%; display: block; }
        .chart-legend {
            display: flex; flex-wrap: wrap; gap: 14px;
            margin-top: 8px;
            font-size: 10px; color: var(--text-secondary);
            font-family: var(--font-mono);
        }

        /* ===== 双栏小面板布局 ===== */
        .two-col-panels {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }

        /* ===== 决策日志 ===== */
        .decision-list {
            max-height: 240px;
            overflow-y: auto;
            padding-right: 4px;
        }
        .decision-list::-webkit-scrollbar { width: 3px; }
        .decision-list::-webkit-scrollbar-track { background: transparent; }
        .decision-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 0; }
        .decision-item {
            display: flex; gap: 10px;
            padding: 7px 0;
            border-bottom: 1px solid var(--border);
            font-size: 11px;
        }
        .decision-item:last-child { border-bottom: none; }
        .decision-step {
            min-width: 36px;
            font-family: var(--font-mono);
            font-size: 9px;
            color: var(--text-muted);
            padding-top: 2px;
        }
        .decision-content { flex: 1; }
        .decision-action {
            display: inline-block;
            padding: 1px 6px;
            border-radius: 0;
            font-weight: 600;
            font-size: 9px;
            margin-right: 5px;
            font-family: var(--font-mono);
            border: 1px solid;
        }
        .action-quantum { background: rgba(212,148,58,0.12); color: var(--accent-light); border-color: rgba(212,148,58,0.25); }
        .action-classical { background: rgba(74,158,158,0.12); color: var(--cyan-light); border-color: rgba(74,158,158,0.25); }
        .action-hybrid { background: rgba(78,138,92,0.12); color: var(--green-light); border-color: rgba(78,138,92,0.25); }
        .decision-reward {
            font-weight: 600;
            color: var(--green-light);
            font-variant-numeric: tabular-nums;
            font-family: var(--font-mono);
            font-size: 10px;
        }
        .decision-reward.negative { color: var(--red); }
        .decision-meta {
            color: var(--text-muted);
            font-size: 9px;
            margin-top: 2px;
            font-family: var(--font-mono);
        }

        /* ===== 特征贡献度 ===== */
        .feature-bars { padding: 2px 0; }
        .feature-bar-row {
            display: flex; align-items: center; gap: 8px;
            margin-bottom: 6px;
            font-size: 10px;
        }
        .feature-name {
            min-width: 72px;
            color: var(--text-secondary);
            font-size: 9px;
            text-align: right;
            font-family: var(--font-mono);
        }
        .feature-bar-bg {
            flex: 1; height: 12px;
            background: var(--bg-primary);
            border-radius: 0;
            overflow: hidden;
            position: relative;
            border: 1px solid var(--border);
        }
        .feature-bar-fill {
            height: 100%;
            border-radius: 0;
            transition: width 0.4s ease;
            position: relative;
        }
        .feature-bar-fill.positive {
            background: var(--green);
        }
        .feature-bar-fill.negative {
            background: var(--accent);
        }
        .feature-value {
            min-width: 40px;
            text-align: right;
            font-size: 9px;
            font-variant-numeric: tabular-nums;
            color: var(--text-muted);
            font-family: var(--font-mono);
        }

        /* ===== 任务队列表格 ===== */
        .task-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 10px;
        }
        .task-table th {
            text-align: left;
            padding: 7px 10px;
            color: var(--text-muted);
            font-weight: 600;
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            border-bottom: 1px solid var(--border-strong);
            white-space: nowrap;
            font-family: var(--font-mono);
            background: var(--bg-card-alt);
        }
        .task-table td {
            padding: 6px 10px;
            border-bottom: 1px solid var(--border);
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-secondary);
        }
        .task-table tbody tr { transition: background 0.15s; }
        .task-table tbody tr:hover { background: var(--bg-card-alt); }
        .task-table-wrap {
            max-height: 280px;
            overflow-y: auto;
        }
        .task-table-wrap::-webkit-scrollbar { width: 3px; }
        .task-table-wrap::-webkit-scrollbar-track { background: transparent; }
        .task-table-wrap::-webkit-scrollbar-thumb { background: var(--border); border-radius: 0; }
        .status-tag {
            display: inline-block;
            padding: 1px 6px;
            border-radius: 0;
            font-size: 9px;
            font-weight: 600;
            font-family: var(--font-mono);
            border: 1px solid;
        }
        .status-tag.pending { background: rgba(196,144,64,0.12); color: var(--accent); border-color: rgba(196,144,64,0.25); }
        .status-tag.running { background: rgba(212,148,58,0.12); color: var(--accent-light); border-color: rgba(212,148,58,0.3); }
        .status-tag.completed { background: rgba(78,138,92,0.12); color: var(--green-light); border-color: rgba(78,138,92,0.25); }
        .status-tag.failed { background: rgba(160,72,64,0.12); color: var(--red); border-color: rgba(160,72,64,0.25); }
        .priority-high { color: var(--red); font-weight: 700; }
        .priority-medium { color: var(--accent); font-weight: 600; }
        .priority-low { color: var(--green-light); }

        /* ===== 控制面板 ===== */
        .control-grid {
            display: flex;
            flex-direction: column;
            gap: 14px;
        }
        .control-section h3 {
            font-size: 9px;
            color: var(--text-muted);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            font-weight: 700;
            font-family: var(--font-mono);
            padding-bottom: 5px;
            border-bottom: 1px solid var(--border);
        }
        .form-group { margin-bottom: 7px; }
        .form-group label {
            display: block;
            font-size: 9px;
            color: var(--text-muted);
            margin-bottom: 3px;
            font-family: var(--font-mono);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .form-group input, .form-group select {
            width: 100%;
            padding: 6px 10px;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 0;
            color: var(--text-primary);
            font-size: 11px;
            outline: none;
            transition: border-color 0.15s;
            font-family: var(--font-mono);
        }
        .form-group input:focus, .form-group select:focus {
            border-color: var(--accent);
            box-shadow: inset 0 0 0 1px rgba(212,148,58,0.15);
        }
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 7px;
        }
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 0;
            font-size: 11px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.15s;
            font-family: var(--font-mono);
            letter-spacing: 0.5px;
        }
        .btn-primary {
            background: var(--accent);
            color: #0c0e13;
            width: 100%;
            border: 1px solid var(--accent-dark);
        }
        .btn-primary:hover { background: var(--accent-light); }
        .btn-primary:active { transform: translateY(1px); }
        .strategy-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 0;
        }
        .strategy-btn {
            padding: 5px 10px;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 0;
            color: var(--text-secondary);
            font-size: 10px;
            cursor: pointer;
            transition: all 0.15s;
            font-weight: 600;
            font-family: var(--font-mono);
            margin-right: -1px;
            margin-bottom: -1px;
        }
        .strategy-btn:hover { border-color: var(--accent); color: var(--accent); z-index:1; }
        .strategy-btn.active {
            background: var(--accent);
            border-color: var(--accent);
            color: #0c0e13;
            font-weight: 700;
        }
        .strategy-btn.recommended {
            border-color: rgba(212,148,58,0.4);
            position: relative;
        }
        .strategy-btn.recommended::after {
            content: '*';
            position: absolute;
            top: -3px; right: -1px;
            font-size: 10px;
            color: var(--accent);
            font-weight: 700;
        }

        /* ===== 策略排名条 ===== */
        .ranking-bar {
            display: flex; align-items: center; gap: 8px;
            margin-bottom: 5px;
            font-size: 10px;
        }
        .ranking-pos {
            width: 20px; text-align: center;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
        }
        .ranking-pos.gold { color: var(--accent-light); font-weight: 800; }
        .ranking-name { min-width: 60px; color: var(--text-secondary); font-family:var(--font-mono); font-size:10px; }
        .ranking-bar-bg {
            flex: 1; height: 14px;
            background: var(--bg-primary);
            border-radius: 0;
            overflow: hidden;
            border: 1px solid var(--border);
        }
        .ranking-bar-fill {
            height: 100%;
            border-radius: 0;
            background: var(--border-strong);
            transition: width 0.5s;
        }
        .ranking-bar-fill.best {
            background: var(--accent);
            box-shadow: 0 0 6px rgba(212,148,58,0.3);
        }
        .ranking-score {
            min-width: 48px; text-align: right;
            font-variant-numeric: tabular-nums;
            color: var(--text-primary);
            font-family: var(--font-mono);
            font-weight: 600;
            font-size: 10px;
        }

        /* ===== Toast ===== */
        .toast-container {
            position: fixed;
            top: 70px; right: 20px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .toast {
            padding: 10px 14px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            border-radius: 0;
            font-size: 11px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            max-width: 280px;
            font-family: var(--font-mono);
            color: var(--text-primary);
        }
        .toast.success { border-left-color: var(--green); }
        .toast.info { border-left-color: var(--cyan); }
        .toast.warn { border-left-color: var(--amber-warn); }

        /* ===== 加载动画 ===== */
        .loading-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: var(--bg-primary);
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
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading-text {
            margin-top: 16px;
            color: var(--text-primary);
            font-size: 13px;
            font-family: var(--font-mono);
            font-weight: 600;
            letter-spacing: 1px;
        }
        .loading-sub {
            margin-top: 6px;
            color: var(--text-muted);
            font-size: 10px;
            font-family: var(--font-mono);
            letter-spacing: 0.5px;
        }

        /* ===== 空状态 ===== */
        .empty-state {
            text-align: center;
            padding: 20px;
            color: var(--text-muted);
            font-size: 11px;
            font-family: var(--font-mono);
        }

        /* ===== 响应式 ===== */
        @media (max-width: 1100px) {
            .main-content { grid-template-columns: 1fr; }
            .two-col-panels { grid-template-columns: 1fr; }
        }
        @media (max-width: 768px) {
            .header { padding: 10px 14px; }
            .header-titles h1 { font-size: 15px; }
            .status-cards { padding: 12px; gap: 1px; grid-template-columns: repeat(2, 1fr); }
            .main-content { padding: 10px 14px 14px; }
            .card-value { font-size: 20px; }
        }
    </style>
</head>
<body>

    <!-- 加载遮罩 -->
    <div class="loading-overlay" id="loading">
        <div class="loader"></div>
        <div class="loading-text">量子RL调度系统启动中...</div>
        <div class="loading-sub">正在加载PPO模型并初始化仿真环境</div>
    </div>

    <!-- 顶部标题栏 -->
    <div class="header">
        <div class="header-left">
            <div class="logo">Q</div>
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
            <div class="card-label">量子比特利用率</div>
            <div class="card-value" id="val-qubit">0%</div>
            <div class="card-sub">实时资源占用</div>
        </div>
        <div class="status-card card-purple">
            <div class="card-label">任务队列长度</div>
            <div class="card-value" id="val-queue">0</div>
            <div class="card-sub">等待调度执行</div>
        </div>
        <div class="status-card card-amber">
            <div class="card-label">平均等待时间</div>
            <div class="card-value" id="val-wait">0s</div>
            <div class="card-sub">最近100个任务</div>
        </div>
        <div class="status-card card-green">
            <div class="card-label">已完成任务</div>
            <div class="card-value" id="val-completed">0</div>
            <div class="card-sub">累计吞吐量</div>
        </div>
        <div class="status-card card-cyan">
            <div class="card-label">吞吐量</div>
            <div class="card-value" id="val-throughput">0</div>
            <div class="card-sub">任务/分钟</div>
        </div>
        <div class="status-card card-pink">
            <div class="card-label">当前策略</div>
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
                    <h2><span class="icon icon-battle">01</span> PPO vs FCFS 实时策略对比</h2>
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
                    <h2><span class="icon icon-chart">02</span> 资源利用率趋势</h2>
                    <span class="badge live">● 实时</span>
                </div>
                <div class="panel-body">
                    <div class="chart-canvas-wrap">
                        <canvas id="resource-canvas" height="200"></canvas>
                    </div>
                    <div class="chart-legend">
                        <div class="legend-item"><div class="legend-line" style="background:#d4943a"></div> 量子比特利用率</div>
                        <div class="legend-item"><div class="legend-line" style="background:#4a9e9e"></div> 队列长度</div>
                        <div class="legend-item"><div class="legend-line" style="background:#4e8a5c"></div> 完成速率</div>
                    </div>
                </div>
            </div>

            <!-- 决策日志 + 特征贡献度 双栏 -->
            <div class="two-col-panels">
                <!-- 决策日志 -->
                <div class="panel">
                    <div class="panel-header">
                        <h2><span class="icon icon-log">03</span> 最近决策</h2>
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
                        <h2><span class="icon icon-explain">04</span> 决策可解释性</h2>
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

            <!-- 策略排名 -->
            <div class="panel">
                <div class="panel-header">
                    <h2><span class="icon icon-rank" style="background:#d4943a; color:#0c0e13;">05</span> 8种策略性能排名</h2>
                    <span class="badge">14维公平重训</span>
                </div>
                <div class="panel-body" id="ranking-body">
                    <div class="empty-state">加载排名数据...</div>
                </div>
            </div>

        </div>

        <!-- 右侧栏 -->
        <div class="side-col">

            <!-- 任务队列 -->
            <div class="panel">
                <div class="panel-header">
                    <h2><span class="icon icon-queue">06</span> 任务队列</h2>
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
                    <h2><span class="icon icon-control">07</span> 控制面板</h2>
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

            <!-- 系统信息 -->
            <div class="panel">
                <div class="panel-header">
                    <h2><span class="icon" style="background:#5a7080; color:#fff;">i</span> 系统信息</h2>
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

        // 网格线 - 深色控制台主题
        ctx.strokeStyle = 'rgba(37,42,53,0.8)';
        ctx.lineWidth = 1;
        for (var gi=0; gi<=4; gi++) {
            var gy = pad.top + (gi/4)*ch;
            ctx.beginPath(); ctx.moveTo(pad.left, gy); ctx.lineTo(w-pad.right, gy); ctx.stroke();
            var gv = yMax - (gi/4)*yRange;
            ctx.fillStyle = '#5c5850';
            ctx.font = '10px "JetBrains Mono", Consolas, monospace';
            ctx.textAlign = 'right';
            ctx.fillText(Math.round(gv), pad.left-6, gy+3);
        }

        // 零基线
        var zeroY = yPos(0);
        ctx.strokeStyle = 'rgba(92,88,80,0.5)';
        ctx.setLineDash([4,4]);
        ctx.beginPath(); ctx.moveTo(pad.left, zeroY); ctx.lineTo(w-pad.right, zeroY); ctx.stroke();
        ctx.setLineDash([]);

        // 绘制曲线函数
        function drawLine(data, color, fillColor) {
            if (data.length < 2) return;
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
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

        drawLine(fcfsHist, '#5c5850', 'rgba(92,88,80,0.06)');
        drawLine(ppoHist, '#d4943a', 'rgba(212,148,58,0.10)');

        // 端点
        if (ppoHist.length > 0) {
            ctx.fillStyle = '#d4943a';
            ctx.strokeStyle = '#0c0e13';
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(xPos(ppoHist.length-1), yPos(ppoHist[ppoHist.length-1]), 4, 0, Math.PI*2); ctx.fill(); ctx.stroke();
        }
        if (fcfsHist.length > 0) {
            ctx.fillStyle = '#5c5850';
            ctx.strokeStyle = '#0c0e13';
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
            ctx.fillStyle = '#5c5850';
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

        // 网格 - 深色控制台主题
        ctx.strokeStyle = 'rgba(37,42,53,0.8)';
        ctx.lineWidth = 1;
        for (var gi=0; gi<=4; gi++) {
            var gy = pad.top + (gi/4)*ch;
            ctx.beginPath(); ctx.moveTo(pad.left, gy); ctx.lineTo(w-pad.right, gy); ctx.stroke();
            ctx.fillStyle = '#5c5850'; ctx.font='10px "JetBrains Mono", Consolas, monospace'; ctx.textAlign='right';
            ctx.fillText(Math.round(yLeftMax - (gi/4)*yLeftMax)+'%', pad.left-6, gy+3);
        }
        // 右Y轴标签
        ctx.textAlign = 'left';
        for (var gi=0; gi<=4; gi++) {
            var gy = pad.top + (gi/4)*ch;
            var gv = Math.round(yRightMax - (gi/4)*yRightMax);
            ctx.fillStyle = '#5c5850';
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

        drawLine(utils, yLeft, '#d4943a');
        drawLine(queues, yRight, '#4a9e9e');
        drawLine(throughs, function(v){return yLeft(Math.min(v*5, 100));}, '#4e8a5c');

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
    function renderStatus(status) {
        if (!status) return;
        document.getElementById('val-qubit').textContent = ((status.qubit_utilization||0)*100).toFixed(1)+'%';
        document.getElementById('val-queue').textContent = status.queue_length||0;
        document.getElementById('val-wait').textContent = (status.average_wait_time||0).toFixed(1)+'s';
        document.getElementById('val-completed').textContent = status.completed_tasks||0;
        // 吞吐量：如果API没有提供，则根据已完成任务和步数估算
        var throughput = status.throughput;
        if (throughput === undefined || throughput === null) {
            var steps = status.current_step || 1;
            var completed = status.completed_tasks || 0;
            // 假设每步约3秒，换算为任务/分钟
            throughput = steps > 0 ? (completed / (steps * 3 / 60)) : 0;
        }
        document.getElementById('val-throughput').textContent = throughput.toFixed(1);
        document.getElementById('val-strategy').textContent = (status.current_strategy||'-').toUpperCase();
        document.getElementById('val-step').textContent = 'Step: '+(status.current_step||0);

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
                '<td style="font-family:monospace;color:var(--text-muted);font-size:11px;">'+(t.task_id||'-').slice(-6)+'</td>'+
                '<td><span style="font-size:11px;">'+(t.task_type||'-')+'</span></td>'+
                '<td>'+(t.qubit_count||'-')+'</td>'+
                '<td><span class="'+prioClass(t.priority)+'">'+(t.priority||'-')+'</span></td>'+
                '<td><span class="status-tag '+(t.status||'pending')+'">'+statusText(t.status||'pending')+'</span></td>'+
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
        diffEl.style.color = diff>=0 ? '#6aab7a' : '#a04840';
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
        for (var i=decisionLog.length-1; i>=Math.max(0,decisionLog.length-15); i--) {
            var d = decisionLog[i];
            var actLabel = d.action_label || actionText(d.action).text;
            var actCls = d.action === 0 ? 'action-classical' : d.action === 1 ? 'action-quantum' : 'action-hybrid';
            var rwd = d.reward||0;
            html += '<div class="decision-item">'+
                '<div class="decision-step">#'+(d.step||i)+'</div>'+
                '<div class="decision-content">'+
                    '<span class="decision-action '+actCls+'">'+actLabel+'</span>'+
                    '<span class="decision-reward '+(rwd<0?'negative':'')+'">'+(rwd>=0?'+':'')+rwd.toFixed(1)+'</span>'+
                    '<div class="decision-meta">'+(d.task_id?'任务 '+(d.task_id+'').slice(-6):'')+(d.source?' · '+d.source:'')+' · ep_r='+((d.episode_reward||0).toFixed(0))+'</div>'+
                '</div></div>';
        }
        list.innerHTML = html;
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
    // 初始化
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

        // 兜底HTTP轮询
        setTimeout(startHttpPolling, 5000);
    })();
    </script>
</body>
</html>"""
