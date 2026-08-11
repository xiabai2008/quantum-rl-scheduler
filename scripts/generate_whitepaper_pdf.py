#!/usr/bin/env python3
"""
技术白皮书PDF生成脚本
将 technical_whitepaper.md 转换为专业排版的PDF文档
"""

import os
import platform
import re
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ============================================================
# 配色方案 - 量子深空主题
# ============================================================
PRIMARY = HexColor("#0d1b2a")  # 深空蓝
SECONDARY = HexColor("#1b263b")  # 深灰蓝
ACCENT = HexColor("#3bc9db")  # Cherenkov量子青
ACCENT_DARK = HexColor("#22a8b8")  # 深色青
PURPLE = HexColor("#a78bfa")  # 量子紫
TEXT_DARK = HexColor("#1a1a2e")  # 正文深色
TEXT_BODY = HexColor("#2d3748")  # 正文灰色
TEXT_MUTED = HexColor("#718096")  # 辅助文字
BG_LIGHT = HexColor("#f7fafc")  # 表格浅底
BG_ALT = HexColor("#edf2f7")  # 交替行
BORDER = HexColor("#e2e8f0")  # 边框
SUCCESS = HexColor("#34d399")  # 成功绿
WARNING = HexColor("#fbbf24")  # 警告黄


# ============================================================
# 字体注册
# ============================================================
def register_fonts():
    """注册中文字体"""
    system = platform.system()
    font_candidates = []

    if system == "Windows":
        font_candidates = [
            ("C:/Windows/Fonts/msyh.ttc", 0),  # Microsoft YaHei
            ("C:/Windows/Fonts/msyhbd.ttc", 0),  # Microsoft YaHei Bold
            ("C:/Windows/Fonts/simsun.ttc", 0),  # SimSun
        ]
    elif system == "Darwin":
        font_candidates = [
            ("/System/Library/Fonts/PingFang.ttc", 0),
            ("/Library/Fonts/Arial Unicode.ttf", 0),
        ]
    else:
        font_candidates = [
            ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
            ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0),
        ]

    regular_font = None
    bold_font = None

    for path, idx in font_candidates:
        if os.path.exists(path):
            try:
                if regular_font is None:
                    pdfmetrics.registerFont(TTFont("CJK", path, subfontIndex=idx))
                    regular_font = "CJK"
                if bold_font is None and "bd" in path.lower():
                    pdfmetrics.registerFont(TTFont("CJK-Bold", path, subfontIndex=idx))
                    bold_font = "CJK-Bold"
            except Exception:
                continue

    if regular_font is None:
        raise RuntimeError("未找到中文字体，请安装Microsoft YaHei或Noto Sans CJK")

    if bold_font is None:
        bold_font = regular_font

    return regular_font, bold_font


# ============================================================
# 自定义Flowable
# ============================================================
class ColoredDivider(Flowable):
    """彩色分隔线"""

    def __init__(self, width, height=2, color=ACCENT, space_before=6, space_after=12):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.color = color
        self.spaceBefore = space_before
        self.spaceAfter = space_after

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)


class CodeBlock(Flowable):
    """代码块"""

    def __init__(self, text, width, font_name="CJK", font_size=9, padding=10):
        Flowable.__init__(self)
        self.text = text
        self._width = width
        self.font_name = font_name
        self.font_size = font_size
        self.padding = padding
        self.line_height = font_size * 1.4
        lines = text.split("\n")
        self.lines = lines
        self._height = len(lines) * self.line_height + padding * 2

    def wrap(self, availWidth, availHeight):  # noqa: N803
        self._width = availWidth
        return (availWidth, self._height)

    def draw(self):
        c = self.canv
        # 背景
        c.setFillColor(HexColor("#0d1117"))
        c.roundRect(0, 0, self._width, self._height, 4, fill=1, stroke=0)
        # 文字
        c.setFillColor(HexColor("#c9d1d9"))
        c.setFont(self.font_name, self.font_size)
        y = self._height - self.padding - self.font_size
        for line in self.lines:
            c.drawString(self.padding, y, line.rstrip())
            y -= self.line_height


# ============================================================
# 样式定义
# ============================================================
def create_styles(font_name, bold_font):
    """创建专业排版样式"""
    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "CoverTitle",
        fontName=bold_font,
        fontSize=28,
        leading=38,
        textColor=PRIMARY,
        spaceAfter=12,
        alignment=TA_CENTER,
        wordWrap="CJK",
    )
    styles["cover_subtitle"] = ParagraphStyle(
        "CoverSubtitle",
        fontName=font_name,
        fontSize=14,
        leading=22,
        textColor=TEXT_MUTED,
        spaceAfter=30,
        alignment=TA_CENTER,
        wordWrap="CJK",
    )
    styles["cover_meta"] = ParagraphStyle(
        "CoverMeta",
        fontName=font_name,
        fontSize=11,
        leading=18,
        textColor=TEXT_BODY,
        alignment=TA_CENTER,
        wordWrap="CJK",
    )
    styles["h1"] = ParagraphStyle(
        "H1",
        fontName=bold_font,
        fontSize=20,
        leading=28,
        textColor=PRIMARY,
        spaceBefore=24,
        spaceAfter=10,
        wordWrap="CJK",
    )
    styles["h2"] = ParagraphStyle(
        "H2",
        fontName=bold_font,
        fontSize=15,
        leading=22,
        textColor=SECONDARY,
        spaceBefore=18,
        spaceAfter=8,
        wordWrap="CJK",
    )
    styles["h3"] = ParagraphStyle(
        "H3",
        fontName=bold_font,
        fontSize=12,
        leading=18,
        textColor=ACCENT_DARK,
        spaceBefore=12,
        spaceAfter=6,
        wordWrap="CJK",
    )
    styles["body"] = ParagraphStyle(
        "Body",
        fontName=font_name,
        fontSize=10.5,
        leading=18,
        textColor=TEXT_BODY,
        spaceBefore=0,
        spaceAfter=8,
        firstLineIndent=21,
        alignment=TA_JUSTIFY,
        wordWrap="CJK",
    )
    styles["body_no_indent"] = ParagraphStyle(
        "BodyNoIndent",
        fontName=font_name,
        fontSize=10.5,
        leading=18,
        textColor=TEXT_BODY,
        spaceBefore=0,
        spaceAfter=8,
        firstLineIndent=0,
        alignment=TA_JUSTIFY,
        wordWrap="CJK",
    )
    styles["list_item"] = ParagraphStyle(
        "ListItem",
        fontName=font_name,
        fontSize=10.5,
        leading=17,
        textColor=TEXT_BODY,
        spaceBefore=2,
        spaceAfter=2,
        leftIndent=20,
        firstLineIndent=0,
        wordWrap="CJK",
    )
    styles["abstract"] = ParagraphStyle(
        "Abstract",
        fontName=font_name,
        fontSize=10.5,
        leading=18,
        textColor=TEXT_BODY,
        spaceBefore=6,
        spaceAfter=10,
        leftIndent=15,
        rightIndent=15,
        firstLineIndent=21,
        alignment=TA_JUSTIFY,
        wordWrap="CJK",
        backColor=HexColor("#f0f9ff"),
        borderPadding=12,
    )
    styles["keywords"] = ParagraphStyle(
        "Keywords",
        fontName=bold_font,
        fontSize=10,
        leading=16,
        textColor=TEXT_MUTED,
        spaceBefore=4,
        spaceAfter=12,
        wordWrap="CJK",
    )
    styles["table_header"] = ParagraphStyle(
        "TableHeader",
        fontName=bold_font,
        fontSize=9.5,
        leading=14,
        textColor=white,
        alignment=TA_CENTER,
        wordWrap="CJK",
    )
    styles["table_cell"] = ParagraphStyle(
        "TableCell",
        fontName=font_name,
        fontSize=9,
        leading=14,
        textColor=TEXT_BODY,
        alignment=TA_CENTER,
        wordWrap="CJK",
    )
    styles["table_cell_left"] = ParagraphStyle(
        "TableCellLeft",
        fontName=font_name,
        fontSize=9,
        leading=14,
        textColor=TEXT_BODY,
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    styles["caption"] = ParagraphStyle(
        "Caption",
        fontName=font_name,
        fontSize=9,
        leading=13,
        textColor=TEXT_MUTED,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=10,
        wordWrap="CJK",
    )
    styles["note"] = ParagraphStyle(
        "Note",
        fontName=font_name,
        fontSize=9.5,
        leading=16,
        textColor=TEXT_MUTED,
        spaceBefore=6,
        spaceAfter=8,
        leftIndent=10,
        rightIndent=10,
        firstLineIndent=0,
        backColor=HexColor("#fffbeb"),
        borderPadding=8,
        wordWrap="CJK",
    )
    styles["ref"] = ParagraphStyle(
        "Ref",
        fontName=font_name,
        fontSize=9.5,
        leading=16,
        textColor=TEXT_BODY,
        spaceBefore=3,
        spaceAfter=3,
        leftIndent=18,
        firstLineIndent=-18,
        wordWrap="CJK",
    )
    styles["toc_h1"] = ParagraphStyle(
        "TOCH1",
        fontName=bold_font,
        fontSize=11,
        leading=20,
        textColor=PRIMARY,
        spaceBefore=8,
        spaceAfter=2,
        wordWrap="CJK",
    )
    styles["toc_h2"] = ParagraphStyle(
        "TOCH2",
        fontName=font_name,
        fontSize=10,
        leading=17,
        textColor=TEXT_BODY,
        spaceBefore=2,
        spaceAfter=1,
        leftIndent=15,
        wordWrap="CJK",
    )
    return styles


# ============================================================
# Markdown解析
# ============================================================
def inline_format(text):
    """处理行内Markdown格式：粗体、斜体、代码"""
    # 转义HTML特殊字符
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # 粗体 **text**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # 斜体 *text*
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    # 行内代码 `code`（用 CJK 字体避免中文注释渲染为乱码方块）
    text = re.sub(r"`(.+?)`", r'<font face="CJK" size="9" backColor="#edf2f7">\1</font>', text)
    # 上标
    text = text.replace("⁻", "<super>-</super>")
    return text


def parse_markdown(md_text, styles, content_width):
    """解析Markdown为Platypus Flowable列表"""
    story = []
    lines = md_text.split("\n")
    i = 0
    in_code_block = False
    code_lines = []
    table_rows = []
    in_table = False

    while i < len(lines):
        line = lines[i].rstrip()

        # 代码块
        if line.startswith("```"):
            if in_code_block:
                # 结束代码块
                code_text = "\n".join(code_lines)
                story.append(CodeBlock(code_text, content_width - 40))
                story.append(Spacer(1, 8))
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # 表格检测
        if "|" in line and line.strip().startswith("|"):
            if not in_table:
                in_table = True
                table_rows = []
            # 跳过分隔行
            if re.match(r"^[\s|:\-]+$", line):
                i += 1
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            table_rows.append(cells)
            i += 1
            continue
        elif in_table:
            # 表格结束，生成表格
            story.append(build_table(table_rows, styles))
            story.append(Spacer(1, 10))
            in_table = False
            table_rows = []

        # 空行
        if not line.strip():
            i += 1
            continue

        # 分隔线
        if line.strip() == "---":
            story.append(Spacer(1, 6))
            story.append(
                ColoredDivider(
                    content_width - 80, height=1, color=BORDER, space_before=4, space_after=8
                )
            )
            i += 1
            continue

        # 标题
        if line.startswith("# "):
            # 一级标题（封面标题单独处理）
            i += 1
            continue
        if line.startswith("## "):
            text = inline_format(line[3:].strip())
            story.append(Paragraph(text, styles["h1"]))
            story.append(ColoredDivider(120, height=2, color=ACCENT, space_before=2, space_after=8))
            i += 1
            continue
        if line.startswith("### "):
            text = inline_format(line[4:].strip())
            story.append(Paragraph(text, styles["h2"]))
            i += 1
            continue
        if line.startswith("#### "):
            text = inline_format(line[5:].strip())
            story.append(Paragraph(text, styles["h3"]))
            i += 1
            continue

        # 列表项
        list_match = re.match(r"^(\s*)(\d+)\.\s+(.+)", line)
        bullet_match = re.match(r"^(\s*)[-*]\s+(.+)", line)
        if list_match:
            indent = len(list_match.group(1))
            num = list_match.group(2)
            text = inline_format(list_match.group(3))
            style = styles["list_item"]
            style.leftIndent = 20 + indent * 10
            story.append(Paragraph(f"{num}. {text}", style))
            i += 1
            continue
        if bullet_match:
            indent = len(bullet_match.group(1))
            text = inline_format(bullet_match.group(2))
            style = styles["list_item"]
            style.leftIndent = 20 + indent * 10
            story.append(Paragraph(f"• {text}", style))
            i += 1
            continue

        # 引用块（> 开头）
        if line.startswith(">"):
            text = inline_format(line[1:].strip())
            story.append(Paragraph(text, styles["note"]))
            i += 1
            continue

        # 普通段落
        text = inline_format(line.strip())
        # 摘要特殊处理
        if text == "摘要" or text == "<b>摘要</b>":
            i += 1
            continue
        if text.startswith("**关键词**") or text.startswith("<b>关键词</b>"):
            story.append(Paragraph(text, styles["keywords"]))
            i += 1
            continue

        story.append(Paragraph(text, styles["body"]))
        i += 1

    # 处理末尾的表格
    if in_table and table_rows:
        story.append(build_table(table_rows, styles))

    return story


def build_table(rows, styles):
    """构建专业样式的表格"""
    if not rows:
        return Spacer(1, 0)

    # 转换单元格为Paragraph
    table_data = []
    for r_idx, row in enumerate(rows):
        para_row = []
        for cell in row:
            cell_text = inline_format(cell.strip())
            if r_idx == 0:
                para_row.append(Paragraph(cell_text, styles["table_header"]))
            else:
                # 第一列左对齐，其他居中
                is_first_col = len(para_row) == 0
                style = styles["table_cell_left"] if is_first_col else styles["table_cell"]
                para_row.append(Paragraph(cell_text, style))
        table_data.append(para_row)

    # 计算列宽
    col_count = len(table_data[0])
    available_width = 470  # A4可用宽度约
    if col_count <= 3:
        col_widths = [available_width / col_count] * col_count
    elif col_count == 4:
        col_widths = [available_width * 0.25] * 4
    elif col_count == 5:
        col_widths = [
            available_width * 0.15,
            available_width * 0.35,
            available_width * 0.15,
            available_width * 0.2,
            available_width * 0.15,
        ]
    elif col_count == 6:
        col_widths = [
            available_width * 0.1,
            available_width * 0.2,
            available_width * 0.15,
            available_width * 0.15,
            available_width * 0.2,
            available_width * 0.2,
        ]
    else:
        col_widths = [available_width / col_count] * col_count

    table = Table(table_data, colWidths=col_widths, repeatRows=1)

    # 表格样式
    table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), SECONDARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, -1), "CJK"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 1), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
    ]

    # 交替行背景
    for r in range(1, len(table_data)):
        if r % 2 == 0:
            table_style.append(("BACKGROUND", (0, r), (-1, r), BG_LIGHT))
        else:
            table_style.append(("BACKGROUND", (0, r), (-1, r), white))

    table.setStyle(TableStyle(table_style))
    return table


# ============================================================
# 页面装饰
# ============================================================
def add_page_decoration(canvas_obj, doc):
    """添加页眉页脚"""
    canvas_obj.saveState()

    # 页眉线
    canvas_obj.setStrokeColor(ACCENT)
    canvas_obj.setLineWidth(0.5)
    canvas_obj.line(doc.leftMargin, A4[1] - 40, A4[0] - doc.rightMargin, A4[1] - 40)

    # 页眉文字
    canvas_obj.setFont("CJK", 8)
    canvas_obj.setFillColor(TEXT_MUTED)
    canvas_obj.drawString(doc.leftMargin, A4[1] - 32, "AI赋能量子计算任务调度系统 — 技术白皮书")
    canvas_obj.drawRightString(A4[0] - doc.rightMargin, A4[1] - 32, "v9.1")

    # 页脚
    canvas_obj.setStrokeColor(BORDER)
    canvas_obj.line(doc.leftMargin, 40, A4[0] - doc.rightMargin, 40)
    canvas_obj.setFont("CJK", 8)
    canvas_obj.setFillColor(TEXT_MUTED)
    canvas_obj.drawCentredString(A4[0] / 2, 28, f"- {doc.page} -")

    # 左侧装饰条
    if doc.page > 1:
        canvas_obj.setFillColor(ACCENT)
        canvas_obj.setFillAlpha(0.3)
        canvas_obj.rect(0, 0, 3, A4[1], fill=1, stroke=0)

    canvas_obj.restoreState()


# ============================================================
# 封面
# ============================================================
def build_cover(styles, content_width):
    """构建封面页"""
    story = []
    story.append(Spacer(1, 1.8 * inch))

    # Logo/装饰
    story.append(
        ColoredDivider(content_width, height=3, color=ACCENT, space_before=0, space_after=30)
    )

    story.append(Paragraph("AI赋能量子计算任务调度系统", styles["cover_title"]))
    story.append(
        Paragraph(
            "技术白皮书",
            ParagraphStyle(
                "CoverTitle2",
                fontName="CJK-Bold",
                fontSize=22,
                leading=30,
                textColor=ACCENT_DARK,
                spaceAfter=16,
                alignment=TA_CENTER,
                wordWrap="CJK",
            ),
        )
    )
    story.append(
        Paragraph("Quantum-RL Co-optimized Intelligent Scheduling System", styles["cover_subtitle"])
    )

    story.append(Spacer(1, 0.8 * inch))

    # 项目信息表格
    meta_data = [
        [
            Paragraph("<b>项 目 信 息</b>", styles["table_header"]),
            Paragraph("", styles["table_header"]),
        ],
        [Paragraph("版本", styles["table_cell"]), Paragraph("v9.1", styles["table_cell"])],
        [Paragraph("日期", styles["table_cell"]), Paragraph("2026年8月9日", styles["table_cell"])],
        [
            Paragraph("团队", styles["table_cell"]),
            Paragraph("量子RL调度团队", styles["table_cell"]),
        ],
        [
            Paragraph("代码仓库", styles["table_cell"]),
            Paragraph("github.com/xiabai2008/quantum-rl-scheduler", styles["table_cell"]),
        ],
        [
            Paragraph("开源协议", styles["table_cell"]),
            Paragraph("MIT License", styles["table_cell"]),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[120, 330])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), SECONDARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("SPAN", (0, 0), (-1, 0)),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, -1), "CJK"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("BACKGROUND", (0, 1), (-1, 1), BG_LIGHT),
                ("BACKGROUND", (0, 3), (-1, 3), BG_LIGHT),
                ("BACKGROUND", (0, 5), (-1, 5), BG_LIGHT),
            ]
        )
    )
    story.append(meta_table)

    story.append(Spacer(1, 1 * inch))
    story.append(
        ColoredDivider(content_width, height=3, color=ACCENT, space_before=0, space_after=20)
    )
    story.append(
        Paragraph(
            "本技术白皮书所有数字均来自可复现的实验结果<br/>"
            "审计脚本 scripts/ci/audit_authoritative_metrics.py 验证通过",
            ParagraphStyle(
                "CoverFooter",
                fontName="CJK",
                fontSize=9,
                leading=15,
                textColor=TEXT_MUTED,
                alignment=TA_CENTER,
                wordWrap="CJK",
            ),
        )
    )

    story.append(PageBreak())
    return story


# ============================================================
# 主函数
# ============================================================
def main():
    # 路径设置：基于本脚本位置推算项目根目录（<root>/scripts/），避免硬编码
    project_root = Path(__file__).resolve().parent.parent
    md_path = project_root / "docs" / "technical_whitepaper.md"
    pdf_path = project_root / "docs" / "technical_whitepaper.pdf"

    print(f"读取Markdown: {md_path}")
    with open(md_path, encoding="utf-8") as f:
        md_content = f.read()

    # 注册字体
    print("注册中文字体...")
    font_regular, font_bold = register_fonts()
    print(f"使用字体: {font_regular} / {font_bold}")

    # 创建样式
    styles = create_styles(font_regular, font_bold)

    # 文档设置
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
        title="AI赋能量子计算任务调度系统 - 技术白皮书",
        author="量子RL调度团队",
        subject="技术白皮书",
    )

    content_width = A4[0] - doc.leftMargin - doc.rightMargin

    # 构建文档
    story = []

    # 封面
    story.extend(build_cover(styles, content_width))

    # 解析正文（跳过第一行标题）
    lines = md_content.split("\n")
    # 找到第一个##之前的内容是摘要区域
    body_start = 0
    for idx, line in enumerate(lines):
        if line.startswith("## "):
            body_start = idx
            break

    body_md = "\n".join(lines[body_start:])
    story.extend(parse_markdown(body_md, styles, content_width))

    # 生成PDF
    print(f"生成PDF: {pdf_path}")
    doc.build(story, onFirstPage=add_page_decoration, onLaterPages=add_page_decoration)
    print("PDF生成成功！")

    # 检查文件大小
    size_kb = pdf_path.stat().st_size / 1024
    print(f"文件大小: {size_kb:.1f} KB")
    return pdf_path


if __name__ == "__main__":
    main()
