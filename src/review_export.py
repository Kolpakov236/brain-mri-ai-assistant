"""Portable physician review bundle: PDF, HTML, text, JSON and lossless masks."""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import os
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PdfImage

from .review import draft_report, editor_mask, mask_stats, overlay


def png_bytes(a: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(a).save(buffer, format="PNG")
    return buffer.getvalue()


def export_review(case: dict, editor: dict, text: str, physician: str, approved: bool,
                  root: Path | None = None) -> dict:
    """Save a consistent snapshot. Unapplied editor changes block export."""
    if not case:
        raise ValueError("Сначала выполните анализ.")
    mask = editor_mask(editor, case["mask"].shape)
    if not np.array_equal(mask, case["mask"]):
        raise ValueError("Область изменена. Сначала нажмите «Обновить текст по области F1», затем проверьте текст.")
    if not text.strip():
        raise ValueError("Заполните текст заключения.")
    if approved and not physician.strip():
        raise ValueError("Для отметки проверки укажите имя или инициалы врача.")
    text = text.replace(chr(0x2014), '-').replace(chr(0x2013), '-')
    root = root or Path(os.environ.get("MRI_REVIEW_DIR", Path(__file__).resolve().parents[1] / ".review_exports"))
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"{case['id']}_{uuid.uuid4().hex[:12]}"
    directory.mkdir()
    status = "Отмечено как проверенное. Электронная подпись отсутствует" if approved else "Черновик. Проверка врача не отмечена"
    stamp = datetime.now(timezone.utc).isoformat()
    record = {k: case[k] for k in ("id", "image_sha256", "name", "probs", "pred", "confidence", "agreement", "warnings", "method", "revision", "history")}
    record.update({"saved_at_utc": stamp, "physician": physician.strip(), "approved": bool(approved), "status": status,
                   "auto_region": mask_stats(case["auto_mask"]), "final_region": mask_stats(mask),
                   "region_changed": bool(not np.array_equal(mask, case["auto_mask"])),
                   "automatic_draft": case.get("initial_draft", draft_report(case)),
                   "draft_for_current_region": draft_report(case), "final_text": text,
                   "coordinate_system": "native image pixels, origin top-left, bbox end exclusive",
                   "model_checkpoint_sha256": case.get("model_sha256"),
                   "mask_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()})
    final_overlay = overlay(case["rgb"], mask, np.zeros_like(mask))
    auto_overlay = overlay(case["rgb"], case["auto_mask"], case["uncertain"])
    caption = ("F1 выделено зелёным. Измерения указаны в пикселях изображения."
               if mask.any() else "Выделенная область отсутствует. Проверьте снимок и текст заключения.")
    for name, arr in (("original.png", case["rgb"]), ("region_final.png", mask.astype(np.uint8) * 255),
                      ("region_auto.png", case["auto_mask"].astype(np.uint8) * 255),
                      ("uncertainty_auto.png", case["uncertain"].astype(np.uint8) * 255),
                      ("overlay_final.png", final_overlay), ("overlay_auto.png", auto_overlay)):
        (directory / name).write_bytes(png_bytes(arr))
    (directory / "review.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "conclusion.txt").write_text(f"{status}\nПроверяющий: {physician.strip() or 'не указан'}\n{stamp}\n\n{text}", encoding="utf-8")
    encoded = base64.b64encode(png_bytes(final_overlay)).decode("ascii")
    headings = {"Описание", "Измерения F1", "Надёжность", "Ограничения"}
    paragraphs = [p for p in text.split("\n\n") if p]
    body_html = ''.join(f'<h2>{html.escape(p)}</h2>' if p in headings else f'<p>{html.escape(p).replace(chr(10), "<br>")}</p>' for p in paragraphs)
    reviewer = physician.strip() or 'не указан'
    date = datetime.fromisoformat(stamp).strftime('%d.%m.%Y, %H:%M UTC')
    status_short = 'Проверка отмечена' if approved else 'Черновик'
    document = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>МРТ головного мозга</title>
    <style>*{{box-sizing:border-box}}body{{font:15px/1.6 Arial,sans-serif;color:#202b24;background:#f1f2ef;margin:0}}
    article{{max-width:800px;margin:32px auto;background:#fff;padding:40px 48px}}header{{border-bottom:1px solid #ccd4ce;padding-bottom:16px;margin-bottom:24px;color:#4d6154;font-size:12px}}
    h1{{font-size:27px;font-weight:400;margin:12px 0}}h2{{font-size:16px;font-weight:600;margin:24px 0 8px}}p{{margin:8px 0}}figure{{margin:24px 0;text-align:center}}img{{height:220px;max-width:100%;object-fit:contain}}figcaption,footer,.meta{{font-size:12px;color:#526158}}footer{{border-top:1px solid #ccd4ce;padding-top:16px;margin-top:24px}}
    @media(max-width:600px){{article{{margin:0;padding:24px}}}}@media print{{body{{background:#fff}}article{{margin:0;padding:0}}}}</style>
    <article><header>НейроАссистент / Протокол {case['id']}</header><h1>МРТ головного мозга</h1>
    <div class="meta">{status_short} · {date}<br>Проверяющий: {html.escape(reviewer)}</div>
    <figure><img src="data:image/png;base64,{encoded}" alt="Выделение на исходном снимке"><figcaption>{caption}</figcaption></figure>
    {body_html}<footer>Учебный прототип. Не является медицинским изделием. Отметка проверки не заменяет электронную подпись.</footer></article></html>'''
    (directory / "conclusion.html").write_text(document, encoding="utf-8")
    font_dir = Path(matplotlib.get_data_path()) / "fonts/ttf"
    if "ReviewSans" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("ReviewSans", str(font_dir / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("ReviewBold", str(font_dir / "DejaVuSans-Bold.ttf")))
    style = ParagraphStyle("review", fontName="ReviewSans", fontSize=9, leading=13, spaceAfter=5, textColor=colors.HexColor("#202b24"))
    section = ParagraphStyle("section", parent=style, fontName="ReviewBold", fontSize=10, leading=14, spaceBefore=8, spaceAfter=4, keepWithNext=True)
    title = ParagraphStyle("title", parent=style, fontSize=21, leading=28, spaceAfter=6)
    meta = ParagraphStyle("meta", parent=style, fontSize=8, leading=12, textColor=colors.HexColor("#56675c"))
    story = [Paragraph("НейроАссистент / Протокол анализа", meta), Spacer(1, 10), Paragraph("МРТ головного мозга", title),
             Paragraph(html.escape(f"{status_short} · {date}  /  Проверяющий: {reviewer}"), meta), Spacer(1, 12)]
    ih, iw = final_overlay.shape[:2]
    display_scale = min(460 / iw, 180 / ih)
    story += [PdfImage(io.BytesIO(png_bytes(final_overlay)), width=iw * display_scale, height=ih * display_scale),
              Spacer(1, 4), Paragraph(caption, meta), Spacer(1, 6)]
    for p in paragraphs:
        story.append(Paragraph(html.escape(p).replace("\n", "<br/>"), section if p in headings else style))
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#cdd5ce")); canvas.setLineWidth(.5)
        canvas.line(42, 42, 553, 42)
        canvas.setFont("ReviewSans", 7)
        canvas.setFillColor(colors.HexColor("#59675e"))
        canvas.drawString(42, 29, "Учебный прототип. Не является медицинским изделием.")
        canvas.drawRightString(553, 29, f"{case['id']} / {doc.page}")
        if approved:
            canvas.drawString(42, 18, "Отметка проверки не заменяет электронную подпись.")
    pdf = directory / "conclusion.pdf"
    SimpleDocTemplate(str(pdf), leftMargin=42, rightMargin=42, topMargin=36, bottomMargin=54,
                      title="МРТ головного мозга", author="НейроАссистент").build(story, onFirstPage=footer, onLaterPages=footer)
    archive = directory / "review_bundle.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in directory.iterdir():
            if f != archive:
                z.write(f, f.name)
    return {"pdf": str(pdf), "zip": str(archive), "html": str(directory / "conclusion.html"), "directory": str(directory)}
