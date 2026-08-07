"""生成答辩PPT文件（Issue #543）。

基于 答辩PPT大纲.md 的18页大纲，使用 python-pptx 生成基础 .pptx 文件。
生成的PPT包含标题、要点和关键数据表格，可作为团队进一步美化的基础。
"""

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ============================================================
# 配色方案（参考大纲附录）
# ============================================================
COLOR_PRIMARY = RGBColor(0x00, 0x66, 0xFF)  # 天衍云蓝
COLOR_SECONDARY = RGBColor(0x8B, 0x5C, 0xF6)  # 量子紫
COLOR_SUCCESS = RGBColor(0x10, 0xB9, 0x81)  # 成功绿
COLOR_WARNING = RGBColor(0xF5, 0x9E, 0x0B)  # 警告橙
COLOR_DARK = RGBColor(0x1E, 0x29, 0x3B)  # 深色文字
COLOR_GRAY = RGBColor(0x64, 0x74, 0x8B)  # 灰色文字
COLOR_LIGHT_BG = RGBColor(0xF8, 0xFA, 0xFC)  # 浅灰背景
COLOR_WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def add_title_slide(prs: Presentation, title: str, subtitle: str) -> None:
    """添加封面幻灯片。"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # 空白布局
    # 标题
    txBox = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(8), Inches(1.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.alignment = PP_ALIGN.CENTER
    # 副标题
    txBox2 = slide.shapes.add_textbox(Inches(1), Inches(3.8), Inches(8), Inches(1))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = subtitle
    p2.font.size = Pt(20)
    p2.font.color.rgb = COLOR_GRAY
    p2.alignment = PP_ALIGN.CENTER


def add_content_slide(
    prs: Presentation,
    title: str,
    bullets: list[str],
    sub_bullets: list[list[str]] | None = None,
) -> None:
    """添加内容幻灯片（标题+要点列表）。"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # 空白布局
    # 标题栏背景
    shape = slide.shapes.add_shape(
        1,
        Inches(0),
        Inches(0),
        Inches(10),
        Inches(1.1),  # MSO_SHAPE.RECTANGLE
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = COLOR_PRIMARY
    shape.line.fill.background()
    # 标题文字
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(9), Inches(0.8))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE
    p.alignment = PP_ALIGN.LEFT
    # 内容要点
    txBox2 = slide.shapes.add_textbox(Inches(0.6), Inches(1.4), Inches(8.8), Inches(5.5))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    sub_bullets = sub_bullets or [[] for _ in bullets]
    for i, (bullet, subs) in enumerate(zip(bullets, sub_bullets, strict=False)):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = bullet
        p.font.size = Pt(18)
        p.font.color.rgb = COLOR_DARK
        p.space_after = Pt(8)
        p.level = 0
        for sub in subs:
            sp = tf2.add_paragraph()
            sp.text = sub
            sp.font.size = Pt(16)
            sp.font.color.rgb = COLOR_GRAY
            sp.space_after = Pt(4)
            sp.level = 1


def add_table_slide(
    prs: Presentation,
    title: str,
    intro: str,
    headers: list[str],
    rows: list[list[str]],
    note: str = "",
) -> None:
    """添加带表格的幻灯片。"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    # 标题栏
    shape = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(10), Inches(1.1))
    shape.fill.solid()
    shape.fill.fore_color.rgb = COLOR_PRIMARY
    shape.line.fill.background()
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(9), Inches(0.8))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE
    # 介绍文字
    if intro:
        txIntro = slide.shapes.add_textbox(Inches(0.6), Inches(1.2), Inches(8.8), Inches(0.6))
        tfIntro = txIntro.text_frame
        pIntro = tfIntro.paragraphs[0]
        pIntro.text = intro
        pIntro.font.size = Pt(16)
        pIntro.font.color.rgb = COLOR_GRAY
    # 表格
    n_rows = len(rows) + 1
    n_cols = len(headers)
    tbl_top = Inches(1.9) if intro else Inches(1.4)
    tbl_height = Inches(0.4 * n_rows)
    table_shape = slide.shapes.add_table(
        n_rows, n_cols, Inches(0.8), tbl_top, Inches(8.4), tbl_height
    )
    table = table_shape.table
    # 表头
    for j, header in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = header
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        for paragraph in cell.text_frame.paragraphs:
            paragraph.font.size = Pt(14)
            paragraph.font.bold = True
            paragraph.font.color.rgb = COLOR_WHITE
            paragraph.alignment = PP_ALIGN.CENTER
    # 数据行
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = val
            if i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = COLOR_LIGHT_BG
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(13)
                paragraph.font.color.rgb = COLOR_DARK
                paragraph.alignment = PP_ALIGN.CENTER
    # 备注
    if note:
        txNote = slide.shapes.add_textbox(Inches(0.6), Inches(6.5), Inches(8.8), Inches(0.5))
        tfNote = txNote.text_frame
        tfNote.word_wrap = True
        pNote = tfNote.paragraphs[0]
        pNote.text = note
        pNote.font.size = Pt(11)
        pNote.font.color.rgb = COLOR_GRAY
        pNote.font.italic = True


def generate_ppt(output_path: str) -> None:
    """生成18页答辩PPT。"""
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    # === 第1页：封面 ===
    add_title_slide(
        prs,
        "量子RL驱动的天衍云平台智能调度系统",
        '中国电信集团有限公司 2026年度"揭榜挂帅"擂台赛\n选题编号：XA-202609\nAI赋能量子计算，量子反馈优化AI —— 三层架构的双向赋能智能调度系统',
    )

    # === 第2页：问题定义 ===
    add_content_slide(
        prs,
        "量子云计算面临的核心矛盾",
        [
            "问题：量子资源稀缺且昂贵，当前量子云平台普遍采用FCFS调度",
            "  - 导致资源利用率低、任务等待时间长、高负载时低优先级任务饥饿",
            "影响：传统FCFS不区分量子/经典任务，量子通道拥堵与经典通道空转",
            "  - 高优先级量子任务常因排队错过最佳执行窗口",
            "切入：将强化学习（RL）引入量子云任务调度",
            "  - 实现量子/经典任务智能分流与动态优化",
            "  - 利用真机噪声反馈优化RL策略鲁棒性",
        ],
    )

    # === 第3页：方案概览 ===
    add_content_slide(
        prs,
        "双向赋能三层架构：调度层 + 编译层 + 量子反馈层",
        [
            "核心思路：用AI智能调度量子资源，用量子真机噪声反馈评估AI噪声敏感性",
            "第一层：调度层 — PPO智能调度实时最优分流，+20.2% vs FCFS（N=250, p=7.56e-12）",
            "第二层：编译层 — PPO替代SABRE比特映射，深电路SWAP减少+38.5%(N=80, Wilcoxon p=2.75e-02，事后子集方向性提示)",
            "第三层：量子赋能AI — 真机噪声模型提取→仿真环境校准→PPO噪声敏感性评估（负向证据，噪声致奖励下降12.43%）",
            "双向赋能闭环：AI优化量子调度→量子真机噪声反馈反哺AI训练环境→更优调度决策",
            "关键数字：PPO +20.2%、防饥饿0.60 vs 22.00、推理延迟~1ms、315次真机调用100%成功",
        ],
    )

    # === 第4页：RL调度引擎（建模）===
    add_content_slide(
        prs,
        "RL调度引擎：16维状态空间 + 毫秒级决策",
        [
            "状态空间（16维）：量子比特可用率、队列长度、平均等待时间、保真度等",
            "动作空间（4类）：经典执行、量子执行、混合执行、量子误差缓解(QEM)",
            "奖励函数设计：多目标奖励",
            "  - 兼容性约束（错配惩罚 -2.0）",
            "  - 执行收益（量子最高 10×speedup）",
            "  - 等待惩罚（超阈值累加）",
            "  - 利用率惩罚（量子闲置 -1.0）",
            "决策延迟：单次前向推理 ~1ms（100-10000任务规模），满足实时调度需求",
        ],
    )

    # === 第5页：8策略对比 ===
    add_table_slide(
        prs,
        "PPO策略综合奖励领先：比真实FCFS提升+20.2%（N=250）",
        "8策略公平对比（N=250，50 seed × 5 episode，16维交付模型）",
        ["策略", "平均奖励", "vs FCFS", "备注"],
        [
            ["PPO", "1982.69", "+20.2%", "第一名, p=7.56e-12"],
            ["FCFS", "1648.91", "基线", "真实 FCFS（量子路由）"],
            ["SJF", "774.86", "-53.0%", "短任务优先"],
            ["DQN(Random占位)", "602.37", "-63.5%", "DQN已删除，Random替代"],
            ["Greedy", "80.71", "-95.1%", "量子优先，均衡场景崩溃"],
            ["Quantum-Only", "-826.59", "-150.1%", "单一资源基线"],
        ],
        "多目标权衡（诚实披露）：PPO 综合奖励 +20.2%（vs 真实 FCFS），量子利用率 -3.3%（R-P-01 未达标）",
    )

    # === 第6页：高负载防饥饿 ===
    add_table_slide(
        prs,
        "高负载防饥饿：PPO在极端拥塞下的智能分流",
        "场景：任务到达率 > 吞吐量（λ=1.2），极端拥塞场景",
        ["策略", "总奖励", "饥饿数", "量子分配"],
        [
            ["PPO", "2046.03", "0.60", "有"],
            ["FCFS", "1489.74", "22.00", "0"],
            ["SJF", "1354.58", "16.80", "几乎0"],
        ],
        "核心发现：PPO防饥饿能力远超FCFS（0.60 vs 22.00），是唯一能有效利用量子资源的策略",
    )

    # === 第7页：编译层优化 ===
    add_content_slide(
        prs,
        "AI赋能编译层：PPO替代SABRE比特映射（公平对比v2）",
        [
            "问题：传统量子电路编译使用启发式算法（SABRE）进行比特映射，效率较低",
            "方案：使用PPO强化学习替代传统启发式算法，直接学习最优比特映射策略",
            "核心数据：公平对比v2（4×4 2D网格拓扑，同池配对60电路，Issue #451）",
            "  - 深电路(14-16q) SWAP减少+38.5%（N=80, Wilcoxon p=2.75e-02显著；全60电路p=8.40e-01不显著，诚实披露）",
            "  - 原76.4%为不公平对比已废弃",
            "  - 整体p=8.40e-01不显著，端到端编译对比（布局+路由）",
            "模型信息：ppo_compilation_agent.zip，200k steps，4×4 2D网格全电路分布训练",
            "叙事：AI从调度到编译的全栈赋能量子计算",
        ],
    )

    # === 第8页：量子赋能AI（主方向）===
    add_content_slide(
        prs,
        "量子赋能AI：真机噪声反馈评估PPO噪声敏感性",
        [
            "核心逻辑：利用量子硬件的真实噪声特征反哺AI模型训练环境",
            "闭环流程：",
            "  - 真机H门测量 → 噪声模型提取（P(0)/P(1)分布）",
            "  - → 仿真环境奖励函数校准 → PPO噪声敏感性评估",
            "  - → 更优调度决策 → 更多真机反馈",
            "验证数据（noise_feedback_v2 权威口径，N=50）：",
            "  - 真机噪声分布注入训练：奖励 +1.5%（p=0.584 不显著，方向性证据）",
            "  - 派单率 p=0.020 显著但效应量极小（99.6%→99.8%），且评估存在训练/测试分布混杂，待交叉评估复核",
            "  - 诚实定位：不宣称奖励显著闭环，也不宣称鲁棒性统计成立",
            "核心价值：真正的量子→AI闭环——利用量子硬件真实物理特性反哺AI",
        ],
    )

    # === 第9页：QUBO退火（已降级）===
    add_content_slide(
        prs,
        "QUBO退火优化（探索性方向，已降级为可选功能）",
        [
            "原方向：将PPO决策头参数（260参数）构造为QUBO矩阵，通过D-Wave neal模拟退火求解",
            "结果：奖励提升+6.4%，p=0.9430（20seeds权威，不显著），训练时间开销+74.5%",
            "当前定位：",
            "  - 退火模块已标记为@deprecated（2026-07-27）",
            "  - 默认关闭（ANNEALING_ENABLED=false）",
            "  - 代码保留用于展示QUBO建模能力",
            "诚实声明：QUBO退火方向目前为探索性结果，性能提升未达统计显著",
            "量子赋能AI主方向已转向真机噪声反馈评估PPO噪声敏感性",
        ],
    )

    # === 第10页：推理延迟验证 ===
    add_table_slide(
        prs,
        "推理延迟验证：PPO决策延迟稳定在~1ms",
        "不同任务规模下的推理延迟对比",
        ["任务规模", "PPO延迟(ms)", "FCFS延迟(ms)", "SJF延迟(ms)"],
        [
            ["100", "0.882", "0.431", "0.372"],
            ["500", "0.904", "0.386", "0.381"],
            ["1000", "1.018", "0.359", "0.368"],
            ["5000", "0.914", "0.373", "0.419"],
            ["10000", "0.897", "0.370", "0.387"],
        ],
        "核心发现：PPO单步决策延迟稳定在~1ms，远低于100ms实时调度要求；O(1)复杂度，不随任务规模增长",
    )

    # === 第11页：真机验证 ===
    add_table_slide(
        prs,
        "真机闭环验证：315次SDK调用100%成功",
        "天衍-287（105数据比特+182耦合比特超导量子计算机）",
        ["策略", "平均奖励", "备注"],
        [
            ["PPO", "1736.32", "Cohen's d=5.33, p<0.001"],
            ["SJF", "575.33", "—"],
            ["FCFS", "383.00", "基线"],
        ],
        "诚实声明：真机验证主要证明SDK可用性（315次100%成功），性能数字以仿真N=250结果为准",
    )

    # === 第12页：多场景压力测试 ===
    add_content_slide(
        prs,
        "多场景压力测试：PPO跨场景适应性最强",
        [
            "5种压力场景：均衡负载、高负载、量子资源波动、混合潮汐、突发到达",
            "PPO综合稳定性最强：5场景中2次第一、2次第二、1次第三，平均排名1.8",
            "量子资源波动场景优势最大：PPO 2997.68 vs FCFS 1566.39，+91.4%（历史探索数据，诚实化前旧 FCFS 基线，权威 stress 数据待重跑核定，仅作场景适配参考）",
            "Greedy两极分化：高负载场景第一，但量子波动场景暴跌至倒数第一（-95.3%）",
            "场景-算法适配决策树：",
            "  - 量子资源波动 → PPO（最优）",
            "  - 均衡负载 → PPO（最优）",
            "  - 高负载 → Greedy（略优）或PPO（稳定次优）",
            "  - 突发洪峰 → FCFS（最稳定）",
        ],
    )

    # === 第13页：系统演示 ===
    add_content_slide(
        prs,
        "Web监控面板：实时调度可视化",
        [
            "技术栈：FastAPI后端 + Vue3前端 + Echarts图表，单页应用无刷新",
            "核心功能：",
            "  - 实时任务队列看板（任务ID、类型、优先级、等待时间）",
            "  - RL Agent决策过程可视化（当前状态→策略输出→选中动作）",
            "  - 资源利用率仪表盘（量子/经典实时跳动）",
            "  - 高负载拥塞分流可视化（λ=1.2场景）",
            "  - 多机器状态监控（3台真机在线状态、负载、最近提交）",
            "演示亮点：任务到达后~1ms内完成决策，面板实时刷新无延迟",
        ],
    )

    # === 第14页：创新点总结 ===
    add_content_slide(
        prs,
        "三大创新点",
        [
            "创新1：AI赋能量子计算调度（核心claim）",
            "  - PPO强化学习实现毫秒级动态决策，+20.2% vs FCFS（p=7.56e-12）",
            "  - 三层架构：调度层+编译层+高负载防饥饿（0.60 vs 22.00）",
            "创新2：量子赋能AI（主方向：真机噪声反馈）",
            "  - 真机噪声→模型校准→PPO噪声敏感性评估（负向证据，噪声致奖励下降12.43%），形成量子→AI感知闭环",
            "  - 奖励 +1.5%（p=0.584 不显著）、派单率 p=0.020 显著但效应量极小——诚实定位为方向性证据，不宣称统计成立",
            "创新3：天衍云平台全链路对接",
            "  - 315次真机调用100%成功，三态熔断器+三级降级策略",
        ],
    )

    # === 第15页：工程实现 ===
    add_content_slide(
        prs,
        "工程化实现：稳定性与可扩展性",
        [
            "三级降级策略：PPO模型推理→规则引擎兜底→默认动作分配+系统告警",
            "熔断器机制：三态转换（CLOSED/OPEN/HALF_OPEN），连续失败3次自动降级",
            "监控体系：Prometheus集成7项核心监控指标",
            "CI/CD：GitHub Actions自动测试、代码质量检查、文档验证",
            "测试覆盖率：≥80%（CI 门禁 --cov-fail-under=80），主套件 3717 测试用例，0 ruff/mypy错误",
        ],
    )

    # === 第16页：价值量化 ===
    add_table_slide(
        prs,
        "落地价值与经济估算",
        "经济价值估算（基于假设条件）",
        ["收益项", "估算金额/年"],
        [
            ["机时成本节省", "¥10,950"],
            ["科研时间价值", "¥360,000"],
            ["故障恢复节省", "¥5,000"],
            ["合计", "¥375,950"],
        ],
        "诚实声明：经济价值估算基于假设（100用户规模、¥100/天机时单价等），实际效果需规模化部署验证",
    )

    # === 第17页：团队介绍 ===
    add_content_slide(
        prs,
        "团队分工",
        [
            "团队规模：8人",
            "算法组（3人）：RL环境设计、PPO/DQN训练、噪声校准模块",
            "工程组（3人）：天衍云API对接、Mock环境、Web监控面板、CI/CD",
            "产品组（2人）：实验报告、PPT制作、演示视频、文档整理",
            "[待填充]：成员姓名、学校/单位、个人照片",
            "[待填充]：指导老师姓名、单位",
        ],
    )

    # === 第18页：致谢 + Q&A ===
    add_content_slide(
        prs,
        "致谢 & 预设问答",
        [
            "致谢：",
            "  - 感谢中国电信集团有限公司提供揭榜挂帅擂台赛平台",
            "  - 感谢天衍云平台的真机支持（天衍-287超导量子计算机）",
            "  - 感谢指导老师的悉心指导",
            "预设Q&A：",
            "  - Q: 利用率为什么比FCFS低？ A: 多目标权衡（诚实披露）：+20.2% 奖励 vs -3.3% 利用率，R-P-01 目标未达成",
            "  - Q: 量子赋能AI为什么是探索性的？ A: QUBO退火p=0.9430不显著，已转向真机噪声反馈；N=50噪声注入奖励+1.5%（p=0.584不显著）仅方向性证据，派单率p=0.020显著但效应量极小，诚实定位不宣称统计成立；N=25配对实验实为噪声敏感性负向结果（噪声致奖励下降12.43%），非赋能证据",
            "  - Q: 真机为什么只测了单比特门？ A: 免费机时多比特门成功率~30%，优先验证SDK可用性",
            "  - Q: PPO不是在所有场景都最优？ A: PPO核心价值是跨场景适应性（平均排名1.8）",
        ],
    )

    prs.save(output_path)
    print(f"PPT已生成: {output_path}（共{len(prs.slides)}页）")


if __name__ == "__main__":
    generate_ppt("deliverable_models/答辩PPT.pptx")
