"""Local MRI review workspace. Run: python app.py."""
from __future__ import annotations

import html
import threading
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from src.config import CFG, CLASSES, CLASSES_RU
from src.data import preprocess
from src.explain import NO_TUMOR_IDX, Explainer
from src.model import load_checkpoint

ROOT = Path(__file__).resolve().parent
CSS = (ROOT / "assets/workspace.css").read_text(encoding="utf-8")
DARK_JS = "() => { document.body.classList.add('dark'); document.documentElement.lang = 'ru'; }"
LOCK = threading.Lock()
_explainer: Explainer | None = None
_img_size = CFG.img_size
DISCLAIMER = "Проверьте результат по полному исследованию."
ACCUSATIVE = {"glioma": "глиому", "meningioma": "менингиому", "pituitary": "опухоль гипофиза"}
EMPTY_HTML = '<p class="empty-line">После анализа здесь появятся класс и вероятности</p>'


def get_explainer() -> Explainer:
    global _explainer, _img_size
    if _explainer is None:
        model, ckpt = load_checkpoint(CFG.checkpoint, "cpu")
        _img_size = ckpt["img_size"]
        _explainer = Explainer(model, ckpt["arch"], "cpu", CFG)
    return _explainer


def top_html() -> str:
    return ('<header class="workspace-header"><span class="wordmark">НейроАссистент</span>'
            '<span class="workspace-subtitle">МРТ головного мозга</span>'
            '<span class="local-note">Локальный учебный прототип</span></header>')


def explanation_text(name: str, probs: np.ndarray, pred: int, has_box: bool) -> str:
    conf = float(probs[pred])
    lines = [f"**{html.escape(name)}**"]
    if pred == NO_TUMOR_IDX:
        lines.append("На этом срезе модель не выявила признаков опухоли. Другие срезы не оценивались.")
    else:
        lines.append(f"Модель предполагает {ACCUSATIVE[CLASSES[pred]]}. Вероятность класса: {conf:.1%}.")
        if has_box:
            lines.append("Контур показывает область внимания Grad-CAM. Точность границы опухоли не проверена.")
    if conf < CFG.low_conf_threshold:
        lines.append("Низкая уверенность. Нужна дополнительная проверка.")
    second = int(np.argsort(-probs)[1])
    lines.append(f"Следующий вариант: {CLASSES_RU[CLASSES[second]]}, {probs[second]:.1%}.")
    return "\n\n".join(lines)


def result_card(name: str, probs: np.ndarray, pred: int) -> str:
    bars = ''.join(f'<div class="prob-row"><span>{CLASSES_RU[c].capitalize()}</span>'
                   f'<div class="prob-track"><i style="width:{probs[i]*100:.1f}%"></i></div>'
                   f'<b>{probs[i]:.1%}</b></div>' for i,c in enumerate(CLASSES))
    attention = "Низкая уверенность" if probs[pred] < CFG.low_conf_threshold else "Проверьте результат"
    return (f'<section class="batch-result"><div class="batch-head"><h3>{html.escape(name)}</h3>'
            f'<span>{attention}</span></div><div class="batch-prediction">'
            f'{html.escape(CLASSES_RU[CLASSES[pred]].capitalize())}</div>{bars}</section>')


def analyze(files: list[str] | None):
    if not files:
        return [], EMPTY_HTML, "Выберите снимки"
    explainer = get_explainer()
    gallery, cards, texts = [], [], []
    for f in files:
        name = Path(f).name
        try:
            with Image.open(f) as im:
                gray = preprocess(im, _img_size)
        except Exception:
            texts.append(f"**{html.escape(name)}**: файл не удалось прочитать")
            continue
        with LOCK:
            ex = explainer.explain(gray)
        gallery.extend([(ex.overlay_box, f"{name}: область внимания"),
                        (ex.overlay_heat, f"{name}: Grad-CAM")])
        cards.append(result_card(name, ex.probs, ex.pred))
        texts.append(explanation_text(name, ex.probs, ex.pred, ex.bbox is not None))
    return gallery, ''.join(cards) or EMPTY_HTML, "\n\n---\n\n".join(texts)


def build_ui() -> gr.Blocks:
    from src.review_ui import build_review_ui
    examples = sorted(str(p) for p in CFG.examples_dir.glob('*.jpg'))
    with gr.Blocks(title="НейроАссистент | Проверка МРТ", analytics_enabled=False) as demo:
        gr.HTML(top_html())
        with gr.Tabs():
            with gr.Tab("Снимок и заключение"):
                build_review_ui(get_explainer, LOCK)
            with gr.Tab("Пакетный анализ"):
                gr.HTML('<div class="section-heading"><h2>Несколько снимков</h2>'
                        '<p>Классификация и Grad-CAM для каждого файла</p></div>')
                with gr.Row():
                    inp = gr.File(label="Снимки JPG / PNG", file_count="multiple", file_types=["image"], type="filepath")
                    btn = gr.Button("Анализировать снимки", variant="primary")
                if examples:
                    gr.Examples([[[p]] for p in examples], inputs=[inp], label="Примеры")
                with gr.Row():
                    with gr.Column(scale=7):
                        gallery = gr.Gallery(label="Выделение и Grad-CAM", columns=2, height=520, object_fit="contain", elem_classes="viewer-surface")
                    with gr.Column(scale=5):
                        summary = gr.HTML(EMPTY_HTML)
                with gr.Accordion("Пояснения к снимкам", open=False):
                    text = gr.Markdown()
                gr.HTML('<p class="method-note">Этот режим сохраняет анализ первого этапа. '
                        'Для правки области и подготовки документа откройте «Снимок и заключение». '
                        'Модель оценивает отдельные 2D-срезы. Клиническая валидация не проводилась.</p>')
                btn.click(analyze, inp, [gallery, summary, text])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, css=CSS, js=DARK_JS,
                      theme=gr.themes.Base(primary_hue="emerald", neutral_hue="zinc", radius_size="sm"))
