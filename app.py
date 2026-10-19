"""Прототип врачебного ассистента: загрузка МРТ-снимков -> класс, вероятности, выделенная область, пояснение.

Запуск: python app.py  (по умолчанию http://127.0.0.1:7860)
Инструмент вспомогательный и не заменяет заключение врача.
"""
from __future__ import annotations

import html
import json
import math
import threading
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from src.config import CFG, CLASSES, CLASSES_RU
from src.data import preprocess
from src.explain import NO_TUMOR_IDX, Explainer
from src.model import load_checkpoint

ACCUSATIVE = {
    "glioma": "глиому",
    "meningioma": "менингиому",
    "pituitary": "опухоль гипофиза",
}
DISCLAIMER = "Результат носит вспомогательный характер и требует подтверждения врачом."
LOCK = threading.Lock()  # Grad-CAM использует хуки модели - обрабатываем снимки по одному
CLASS_COLORS = {"glioma": "#ef7564", "meningioma": "#6a5cff", "notumor": "#43d9ad", "pituitary": "#4aa8ff"}

_explainer: Explainer | None = None
_img_size = CFG.img_size


def _icon(body: str) -> str:
    """Линейная SVG-иконка 18x18 (белый контур) для плиток."""
    return ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


ICONS = {
    "target": _icon('<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5"/>'),
    "pulse": _icon('<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'),
    "shield": _icon('<path d="M12 2l8 3v6c0 5-3.5 9-8 11-4.5-2-8-6-8-11V5z"/><polyline points="9 12 11 14 15 10"/>'),
    "trend": _icon('<polyline points="3 17 9 11 13 15 21 7"/><polyline points="15 7 21 7 21 13"/>'),
}

CSS = """
:root, .dark {
  --na-bg: #262c58; --na-card: #323968; --na-card-2: #2b3260; --na-line: #414a85;
  --na-text: #eef0ff; --na-muted: #a9afd6; --na-coral: #ef7564; --na-blue: #4aa8ff; --na-violet: #6a5cff;
  --na-grad: linear-gradient(135deg, #c2338f 0%, #7a3cf0 100%);
  --body-background-fill: var(--na-bg); --background-fill-primary: var(--na-card);
  --background-fill-secondary: var(--na-card-2); --block-background-fill: transparent;
  --block-border-color: transparent; --border-color-primary: var(--na-line);
  --body-text-color: var(--na-text); --body-text-color-subdued: var(--na-muted);
  --block-label-text-color: var(--na-muted); --block-title-text-color: var(--na-text);
  --input-background-fill: var(--na-card-2); --input-border-color: var(--na-line);
  --button-primary-background-fill: var(--na-grad); --button-primary-background-fill-hover: var(--na-grad);
  --button-primary-text-color: #fff; --button-secondary-background-fill: var(--na-card-2);
  --button-secondary-text-color: var(--na-text); --button-secondary-border-color: var(--na-line);
  --color-accent: #7a3cf0; --link-text-color: #8fa0ff; --panel-background-fill: var(--na-card-2);
  --table-even-background-fill: transparent; --table-odd-background-fill: transparent;
  --block-radius: 18px; --button-large-radius: 14px;
}
body, .gradio-container {
  background: radial-gradient(1200px 700px at 10% -10%, #343c7a 0%, transparent 60%),
              radial-gradient(900px 600px at 100% 110%, #3a2f7a 0%, transparent 55%), var(--na-bg) !important;
  color: var(--na-text);
}
.gradio-container { max-width: 1480px !important; padding: 24px 28px !important; }
footer { display: none !important; }
.na-top { display: flex; align-items: center; gap: 28px; padding: 6px 4px 18px; }
.na-brand { display: flex; align-items: center; gap: 10px; font-size: 20px; font-weight: 700; }
.na-logo { width: 38px; height: 38px; border-radius: 12px; background: var(--na-grad); display: grid; place-items: center; font-size: 20px; }
.na-nav { display: flex; gap: 26px; margin-left: 12px; color: var(--na-muted); font-size: 16px; }
.na-nav .active { color: #f06aa8; border-bottom: 2px solid #c2338f; padding-bottom: 12px; }
.na-badge { margin-left: auto; padding: 8px 16px; border: 1px solid #5662b8; border-radius: 12px; color: #cfd5ff; font-size: 13px; background: rgba(90,100,200,.12); }
.na-card, .na-profile { background: var(--na-card) !important; border-radius: 24px !important; padding: 22px !important;
  box-shadow: 0 10px 30px rgba(10,12,40,.35); border: 1px solid rgba(255,255,255,.04) !important; position: relative; overflow: hidden; }
.na-card::after, .na-profile::after { content: ""; position: absolute; left: -40px; bottom: -60px; width: 220px; height: 160px;
  background: radial-gradient(closest-side, rgba(70,80,255,.35), transparent); pointer-events: none; }
.na-title { font-size: 19px; font-weight: 650; margin-bottom: 12px; }
.na-sub { color: var(--na-muted); font-size: 13px; }
.na-profile { text-align: center; margin-bottom: 16px; }
.na-avatar { width: 92px; height: 92px; margin: 6px auto 12px; border-radius: 50%; background: #262c58; display: grid; place-items: center; font-size: 46px; box-shadow: inset 0 0 0 3px #3b4380; }
.na-name { font-size: 20px; font-weight: 700; }
.na-role { color: var(--na-muted); margin-top: 4px; }
.na-tiles { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 18px; text-align: left; }
.na-tile { background: var(--na-card-2); border-radius: 16px; padding: 14px; }
.na-ico { width: 32px; height: 32px; border-radius: 10px; display: grid; place-items: center; font-size: 16px; margin-bottom: 10px; }
.na-val { font-size: 24px; font-weight: 700; }
.na-lab { color: var(--na-muted); font-size: 12px; margin-top: 2px; }
.na-donut { display: flex; flex-direction: column; align-items: center; }
.na-legend { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 14px; margin-top: 14px; font-size: 12.5px; color: var(--na-muted); width: 100%; }
.na-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
.na-res { background: var(--na-card); border-radius: 24px; padding: 20px 22px; margin-bottom: 14px; position: relative; overflow: hidden; box-shadow: 0 10px 30px rgba(10,12,40,.35); }
.na-res-head { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
.na-res-name { font-size: 17px; font-weight: 650; }
.na-pill { padding: 5px 12px; border-radius: 999px; font-size: 12.5px; font-weight: 600; }
.na-pill.warn { background: rgba(255,184,76,.16); color: #ffc76b; }
.na-pill.ok { background: rgba(90,100,200,.22); color: #c9d0ff; }
.na-grid4 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 16px; }
.na-bars { display: grid; gap: 10px; }
.na-bar-row { display: grid; grid-template-columns: 170px 1fr 56px; align-items: center; gap: 12px; font-size: 14px; }
.na-bar { height: 10px; border-radius: 99px; background: #262c58; overflow: hidden; }
.na-bar > span { display: block; height: 100%; border-radius: 99px; }
.na-pct { text-align: right; color: var(--na-muted); font-variant-numeric: tabular-nums; }
.na-empty { color: var(--na-muted); text-align: center; padding: 26px 8px; }
.na-warn { color: var(--na-muted); font-size: 13px; line-height: 1.5; padding: 4px 6px; }
button.primary { font-weight: 650 !important; box-shadow: 0 8px 20px rgba(122,60,240,.35); }
.na-card .prose { color: var(--na-text); }
.na-card .prose p { margin: .45em 0; }
.gallery-item, .thumbnail-item { border-radius: 14px !important; }
@media (max-width: 900px) { .na-grid4 { grid-template-columns: 1fr; } .na-bar-row { grid-template-columns: 120px 1fr 48px; } }
"""

DARK_JS = "() => { document.body.classList.add('dark'); }"


def get_explainer() -> Explainer:
    """Ленивая загрузка модели (CPU - надёжно и достаточно быстро для одного снимка)."""
    global _explainer, _img_size
    if _explainer is None:
        model, ckpt = load_checkpoint(CFG.checkpoint, "cpu")
        _img_size = ckpt["img_size"]
        _explainer = Explainer(model, ckpt["arch"], "cpu", CFG)
    return _explainer


def load_metrics() -> dict | None:
    """Метрики test из reports/metrics.json (если файл есть) - для плиток сайдбара."""
    path = CFG.reports_dir / "metrics.json"
    try:
        return json.loads(path.read_text())["test"]
    except (OSError, KeyError, ValueError):
        return None


def donut_svg(counts: dict[str, int]) -> str:
    """Кольцевая диаграмма состава тестовой выборки (inline SVG)."""
    r, total = 62.0, max(sum(counts.values()), 1)
    circ = 2 * math.pi * r
    offset, parts = 0.0, []
    for c in CLASSES:
        dash = counts[c] / total * circ
        parts.append(f'<circle r="{r}" cx="90" cy="90" fill="none" stroke="{CLASS_COLORS[c]}" stroke-width="26" '
                     f'stroke-dasharray="{dash:.1f} {circ - dash:.1f}" stroke-dashoffset="{-offset:.1f}" '
                     f'transform="rotate(-90 90 90)"/>')
        offset += dash
    center = (f'<text x="90" y="88" text-anchor="middle" fill="#eef0ff" font-size="26" font-weight="700">{total}</text>'
              f'<text x="90" y="108" text-anchor="middle" fill="#a9afd6" font-size="12">снимков</text>')
    return f'<svg width="180" height="180" viewBox="0 0 180 180">{"".join(parts)}{center}</svg>'


def top_html() -> str:
    """Верхняя панель: бренд, раздел, плашка прототипа."""
    return ('<div class="na-top"><div class="na-brand"><span class="na-logo">🧠</span>НейроАссистент</div>'
            '<div class="na-nav"><span class="active">Анализ снимка</span></div>'
            '<div class="na-badge">Учебный прототип · вспомогательный инструмент</div></div>')


def sidebar_html() -> str:
    """Левая колонка: карточка ассистента, ключевые метрики и состав test (числа из metrics.json)."""
    m = load_metrics()
    if m:
        acc, sens = f"{m['accuracy']:.1%}", f"{m['tumor_vs_normal']['sensitivity']:.1%}"
        spec, auc = f"{m['tumor_vs_normal']['specificity']:.1%}", f"{m['roc_auc_ovr_macro']:.4f}"
        counts = {c: m["per_class"][c]["support"] for c in CLASSES}
    else:
        acc = sens = spec = auc = "-"
        counts = {c: 1 for c in CLASSES}

    def tile(ico: str, bg: str, val: str, lab: str, color: str) -> str:
        return (f'<div class="na-tile"><div class="na-ico" style="background:{bg}">{ico}</div>'
                f'<div class="na-val" style="color:{color}">{val}</div><div class="na-lab">{lab}</div></div>')

    legend = "".join(f'<div><span class="na-dot" style="background:{CLASS_COLORS[c]}"></span>'
                     f'{CLASSES_RU[c].capitalize()} · {counts[c]}</div>' for c in CLASSES)
    return (
        '<div class="na-profile"><div class="na-avatar">🧠</div>'
        '<div class="na-name">ИИ-ассистент врача</div><div class="na-role">Анализ МРТ головного мозга</div>'
        '<div class="na-tiles">'
        + tile(ICONS["target"], "linear-gradient(135deg,#e4574f,#c2338f)", acc, "Accuracy (test)", "#ef7564")
        + tile(ICONS["pulse"], "linear-gradient(135deg,#5b6bff,#4aa8ff)", sens, "Чувствительность", "#6fb7ff")
        + tile(ICONS["shield"], "linear-gradient(135deg,#7a3cf0,#5b6bff)", spec, "Специфичность", "#a99bff")
        + tile(ICONS["trend"], "linear-gradient(135deg,#2db38a,#4aa8ff)", auc, "ROC-AUC (OvR)", "#43d9ad")
        + '</div></div>'
        '<div class="na-card na-donut"><div class="na-title" style="align-self:flex-start">Тестовая выборка</div>'
        + donut_svg(counts) + f'<div class="na-legend">{legend}</div></div>')


def explanation_text(name: str, probs: np.ndarray, pred: int, has_box: bool) -> str:
    """Текст-пояснение на русском для одного снимка."""
    conf = float(probs[pred])
    cls = CLASSES[pred]
    lines = [f"**{name}**"]
    if conf < CFG.low_conf_threshold:
        lines.append(f"**Внимание: низкая уверенность ({conf:.0%}), требуется внимание специалиста.**")
    if pred == NO_TUMOR_IDX:
        lines.append(f"Модель не обнаружила признаков опухоли (уверенность {conf:.0%}). Область не выделяется.")
    else:
        lines.append(f"Модель предполагает {ACCUSATIVE[cls]} (уверенность {conf:.0%}).")
        if has_box:
            lines.append("Красной рамкой и жёлтым контуром выделена область, на которую модель опиралась при решении. "
                         "Это приблизительная локализация (Grad-CAM), а не точная граница опухоли.")
    second = int(np.argsort(-probs)[1])
    lines.append(f"Следующий по вероятности вариант: {CLASSES_RU[CLASSES[second]]} ({probs[second]:.0%}).")
    lines.append(DISCLAIMER)
    return "\n\n".join(lines)


def result_card(name: str, probs: np.ndarray, pred: int) -> str:
    """HTML-карточка результата: плитки (класс, уверенность, P(опухоль)) и полосы вероятностей."""
    conf = float(probs[pred])
    low = conf < CFG.low_conf_threshold
    pill = ('<span class="na-pill warn">Низкая уверенность - нужна проверка специалиста</span>' if low
            else '<span class="na-pill ok">Требуется подтверждение врачом</span>')

    def tile(val: str, lab: str, color: str) -> str:
        return f'<div class="na-tile"><div class="na-val" style="color:{color}">{val}</div><div class="na-lab">{lab}</div></div>'

    tiles = (tile(html.escape(CLASSES_RU[CLASSES[pred]].capitalize()), "Предсказание", CLASS_COLORS[CLASSES[pred]])
             + tile(f"{conf:.0%}", "Уверенность", "#ffc76b" if low else "#eef0ff")
             + tile(f"{1 - probs[NO_TUMOR_IDX]:.0%}", "Вероятность опухоли (любого типа)", "#6fb7ff"))
    bars = "".join(
        f'<div class="na-bar-row"><span>{CLASSES_RU[c].capitalize()}</span>'
        f'<div class="na-bar"><span style="width:{probs[i] * 100:.1f}%;background:{CLASS_COLORS[c]}"></span></div>'
        f'<span class="na-pct">{probs[i]:.1%}</span></div>' for i, c in enumerate(CLASSES))
    return (f'<div class="na-res"><div class="na-res-head"><span class="na-res-name">{html.escape(name)}</span>{pill}</div>'
            f'<div class="na-grid4">{tiles}</div><div class="na-bars">{bars}</div></div>')


EMPTY_HTML = ('<div class="na-res"><div class="na-empty">Загрузите снимок и нажмите «Анализировать»: '
              'здесь появятся класс, уверенность и вероятности.</div></div>')


def analyze(files: list[str] | None):
    """Обработчик интерфейса: список путей к файлам -> (галерея, карточки результатов, пояснения)."""
    if not files:
        return [], EMPTY_HTML, "Загрузите один или несколько снимков."
    explainer = get_explainer()
    gallery, cards, texts = [], [], []
    for f in files:
        name = Path(f).name
        try:
            gray = preprocess(Image.open(f), _img_size)
        except Exception:
            texts.append(f"**{name}**: не удалось прочитать файл как изображение.")
            continue
        with LOCK:
            ex = explainer.explain(gray)
        label = f"{name}: {CLASSES_RU[CLASSES[ex.pred]]} ({ex.confidence:.0%})"
        gallery += [(ex.overlay_box, f"{label} - выделенная область"), (ex.overlay_heat, f"{name} - карта Grad-CAM")]
        cards.append(result_card(name, ex.probs, ex.pred))
        texts.append(explanation_text(name, ex.probs, ex.pred, ex.bbox is not None))
    return gallery, "".join(cards) or EMPTY_HTML, "\n\n---\n\n".join(texts)


def build_ui() -> gr.Blocks:
    """Собирает интерфейс Gradio (русский язык, тёмная тема)."""
    example_files = sorted(p for p in CFG.examples_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    with gr.Blocks(title="ИИ-ассистент: МРТ головного мозга") as demo:
        gr.HTML(top_html())
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320):
                gr.HTML(sidebar_html())
                with gr.Column(elem_classes="na-card"):
                    gr.HTML('<div class="na-title">Загрузка снимков</div>')
                    inp = gr.File(label="МРТ-снимки (jpg, png)", file_count="multiple", file_types=["image"],
                                  type="filepath")
                    btn = gr.Button("Анализировать", variant="primary", size="lg")
                    if example_files:
                        gr.Examples(examples=[[[str(p)]] for p in example_files], inputs=[inp],
                                    label="Примеры для демонстрации")
            with gr.Column(scale=3, min_width=520):
                with gr.Column(elem_classes="na-card"):
                    gr.HTML('<div class="na-title">Выделенная область и карта Grad-CAM</div>'
                            '<div class="na-sub">Рамка и контур показывают область, на которую опиралась модель '
                            '(приблизительная локализация)</div>')
                    gallery = gr.Gallery(show_label=False, columns=2, height=450, object_fit="contain")
                summary = gr.HTML(EMPTY_HTML)
                with gr.Column(elem_classes="na-card"):
                    gr.HTML('<div class="na-title">Пояснение</div>')
                    text = gr.Markdown()
                gr.HTML(f'<div class="na-warn">Это учебный прототип: он не является медицинским изделием и не ставит '
                        f'диагноз, результат требует подтверждения врачом. Порог низкой уверенности: '
                        f'{CFG.low_conf_threshold:.0%}. Модель обучена на одном составном наборе 2D-срезов, внешняя '
                        f'валидация не проводилась, локализация приблизительная (Grad-CAM).</div>')
        btn.click(analyze, inputs=inp, outputs=[gallery, summary, text])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, css=CSS, js=DARK_JS,
                      theme=gr.themes.Base(primary_hue="violet", neutral_hue="indigo"))
