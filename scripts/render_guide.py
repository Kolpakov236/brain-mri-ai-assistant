"""Regenerate the short guide from its Markdown source using installed project dependencies."""
from __future__ import annotations

import html
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    font_dir = Path(matplotlib.get_data_path()) / 'fonts/ttf'
    for name, filename in [('GuideSans', 'DejaVuSans.ttf'), ('GuideBold', 'DejaVuSans-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    body = ParagraphStyle('body', fontName='GuideSans', fontSize=9, leading=13, spaceAfter=7, textColor=colors.HexColor('#20332b'))
    heading = ParagraphStyle('heading', parent=body, fontName='GuideBold', fontSize=12, leading=17, spaceBefore=12, spaceAfter=7, keepWithNext=True)
    title = ParagraphStyle('title', parent=body, fontSize=22, leading=28, spaceAfter=16)
    code = ParagraphStyle('code', parent=body, fontSize=8.5, leading=12, backColor=colors.HexColor('#edf0ea'), borderPadding=5, spaceAfter=6)
    story = []
    for paragraph in (ROOT / 'docs/guide_for_friend.md').read_text(encoding='utf-8').split('\n\n'):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if paragraph == '<!-- pagebreak -->':
            story.append(PageBreak()); continue
        if paragraph.startswith('# '):
            story.append(Paragraph(html.escape(paragraph[2:]), title))
        elif paragraph.startswith('## '):
            story.append(Paragraph(html.escape(paragraph[3:]), heading))
        elif paragraph.startswith('`'):
            for line in paragraph.splitlines():
                story.append(Paragraph(html.escape(line.strip('`')), code))
        elif paragraph[0].isdigit() and '. ' in paragraph:
            for line in paragraph.splitlines():
                story.append(Paragraph(html.escape(line), body))
        else:
            story.append(Paragraph(html.escape(paragraph.replace('`', '')).replace('\n', ' '), body))
    def footer(canvas, document):
        canvas.setStrokeColor(colors.HexColor('#ccd3ca')); canvas.setLineWidth(.5); canvas.line(42, 38, 553, 38)
        canvas.setFillColor(colors.HexColor('#53665b')); canvas.setFont('GuideSans', 8)
        canvas.drawString(42, 24, 'НейроАссистент / Инструкция к прототипу')
        canvas.drawRightString(553, 24, str(document.page))
    path = ROOT / 'docs/guide_for_friend.pdf'
    SimpleDocTemplate(str(path), title='НейроАссистент: краткая инструкция', author='NervuSSwagus / СПбГПМУ',
                      leftMargin=42, rightMargin=42, topMargin=36, bottomMargin=52).build(story, onFirstPage=footer, onLaterPages=footer)
    print(path)


if __name__ == '__main__':
    main()
