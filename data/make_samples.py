"""data/src/*.txt 원본을 data/raw/ 의 PDF·docx 로 렌더링한다 (가상 샘플 문서 생성용).

원본 형식: '@key value' 메타, '## ' 장, '### ' 조/문항, '|a|b' 표(첫 줄이 머리글), 나머지는 문단.
사용: python data/make_samples.py
"""
import re
from pathlib import Path
from xml.sax.saxutils import escape

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).parent
SRC, OUT = ROOT / "src", ROOT / "raw"
FONT_REG, FONT_BOLD = "C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf"
NAVY = colors.HexColor("#1F3A5F")


def parse(path):
    """-> (meta dict, blocks list). block = (kind, payload), kind in h1/h2/p/table."""
    meta, blocks, rows = {}, [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            rows.append([c.strip() for c in line.strip("|").split("|")])
            continue
        if rows:
            blocks.append(("table", rows))
            rows = []
        if not line.strip():
            continue
        if line.startswith("@"):
            k, _, v = line[1:].partition(" ")
            meta[k] = v.strip()
        elif line.startswith("### "):
            blocks.append(("h2", line[4:].strip()))
        elif line.startswith("## "):
            blocks.append(("h1", line[3:].strip()))
        else:
            blocks.append(("p", line.strip()))
    if rows:
        blocks.append(("table", rows))
    return meta, blocks


# ---------- PDF ----------
def build_pdf(meta, blocks, out):
    pdfmetrics.registerFont(TTFont("Malgun", FONT_REG))
    pdfmetrics.registerFont(TTFont("MalgunB", FONT_BOLD))
    st = {
        "cover_co": ParagraphStyle("cco", fontName="MalgunB", fontSize=16, leading=24, alignment=1, textColor=NAVY),
        "cover_t": ParagraphStyle("ct", fontName="MalgunB", fontSize=34, leading=46, alignment=1, textColor=NAVY),
        "cover_m": ParagraphStyle("cm", fontName="Malgun", fontSize=12, leading=20, alignment=1),
        "cover_n": ParagraphStyle("cn", fontName="Malgun", fontSize=9, leading=14, alignment=1, textColor=colors.grey),
        "h1": ParagraphStyle("h1", fontName="MalgunB", fontSize=15, leading=22, spaceBefore=14, spaceAfter=8, textColor=NAVY),
        "h2": ParagraphStyle("h2", fontName="MalgunB", fontSize=11.5, leading=17, spaceBefore=10, spaceAfter=3),
        "p": ParagraphStyle("p", fontName="Malgun", fontSize=10, leading=16, spaceAfter=3),
        "sub": ParagraphStyle("sub", fontName="Malgun", fontSize=10, leading=16, leftIndent=12, spaceAfter=2),
        "toc1": ParagraphStyle("toc1", fontName="MalgunB", fontSize=10.5, leading=17, spaceBefore=5),
        "toc2": ParagraphStyle("toc2", fontName="Malgun", fontSize=9.5, leading=14, leftIndent=14),
        "cell": ParagraphStyle("cell", fontName="Malgun", fontSize=9.5, leading=14),
        "cellb": ParagraphStyle("cellb", fontName="MalgunB", fontSize=9.5, leading=14),
    }

    def footer(c, d):
        c.saveState()
        c.setFont("Malgun", 8.5)
        c.setFillColor(colors.grey)
        if d.page > 1:
            c.drawCentredString(A4[0] / 2, 12 * mm, f"{meta['company']} · {meta['title']}  -  {d.page}  -")
        c.restoreState()

    def table(rows):
        data = [[Paragraph(escape(c), st["cellb" if i == 0 else "cell"]) for c in r] for i, r in enumerate(rows)]
        ncol = len(rows[0])
        width = A4[0] - 40 * mm
        cw = [width / ncol] * ncol if ncol != 3 else [width * 0.18, width * 0.16, width * 0.66]
        t = Table(data, colWidths=cw, repeatRows=1)
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return [Spacer(1, 3), t, Spacer(1, 6)]

    f = [Spacer(1, 70 * mm), Paragraph(escape(meta["company"]), st["cover_co"]), Spacer(1, 10 * mm),
         Paragraph(escape(meta["title"]), st["cover_t"]), Spacer(1, 16 * mm),
         Paragraph(f"{escape(meta['version'])} · 시행 {meta['effective']}", st["cover_m"]), Spacer(1, 50 * mm),
         Paragraph(escape(meta["note"]), st["cover_n"]), PageBreak()]
    f.append(Paragraph("목차", st["h1"]))
    for kind, val in blocks:
        if kind in ("h1", "h2"):
            f.append(Paragraph(escape(val), st["toc1" if kind == "h1" else "toc2"]))
    f.append(PageBreak())
    for kind, val in blocks:
        if kind == "h1":
            f.append(Paragraph(escape(val), st["h1"]))
        elif kind == "h2":
            f.append(Paragraph(escape(val), st["h2"]))
        elif kind == "p":
            f.append(Paragraph(escape(val), st["sub"] if re.match(r"^\d+\.\s", val) else st["p"]))
        else:
            f += table(val)
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=20 * mm,
                      bottomMargin=20 * mm, title=meta["title"], author=meta["company"]).build(f, onFirstPage=footer, onLaterPages=footer)


# ---------- docx ----------
def _font(style, size, bold=None, color=None):
    style.font.name = "맑은 고딕"
    style.font.size = Pt(size)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕")
    if bold is not None:
        style.font.bold = bold
    if color:
        style.font.color.rgb = color


def build_docx(meta, blocks, out):
    d = Document()
    for name, size, bold, color in [("Normal", 10.5, None, None), ("Heading 1", 16, True, RGBColor(0x1F, 0x3A, 0x5F)),
                                    ("Heading 2", 12, True, RGBColor(0, 0, 0)), ("Title", 30, True, RGBColor(0x1F, 0x3A, 0x5F))]:
        _font(d.styles[name], size, bold, color)
    for text, style in [(meta["company"], "Heading 1"), (meta["title"], "Title")]:
        d.add_paragraph(text, style=style).alignment = WD_ALIGN_PARAGRAPH.CENTER
    d.add_paragraph(f"{meta['version']} · 시행 {meta['effective']}").alignment = WD_ALIGN_PARAGRAPH.CENTER
    d.add_paragraph(meta["note"])
    d.add_page_break()
    first_h2 = True
    for kind, val in blocks:
        if kind == "h1":
            d.add_heading(val, level=1)
        elif kind == "h2":
            if meta.get("pagebreak") == "article" and not first_h2:
                d.add_page_break()
            first_h2 = False
            d.add_heading(val, level=2)
        elif kind == "p":
            d.add_paragraph(val)
        else:
            t = d.add_table(rows=len(val), cols=len(val[0]), style="Table Grid")
            for i, row in enumerate(val):
                for j, cell in enumerate(row):
                    t.cell(i, j).text = cell
                    if i == 0:
                        for r in t.cell(i, j).paragraphs[0].runs:
                            r.font.bold = True
            d.add_paragraph()
    d.save(out)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for src in sorted(SRC.glob("*.txt")):
        meta, blocks = parse(src)
        name = re.sub(r"\s+", "", meta["title"])
        out = OUT / f"{name}.{meta['format']}"
        (build_pdf if meta["format"] == "pdf" else build_docx)(meta, blocks, out)
        print("생성:", out.name)
