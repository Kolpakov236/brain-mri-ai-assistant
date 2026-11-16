"""Physician review UI; model callbacks share the original Grad-CAM lock."""
from __future__ import annotations

import hashlib
import html
import threading

import gradio as gr

from .config import CFG, CLASSES, CLASSES_RU
from .review import analyze_review, apply_region, draft_report, editor_value, overlay
from .review_export import export_review


def build_review_ui(get_explainer, model_lock: threading.Lock):
    case_state = gr.State(None)
    gr.HTML('<div class="section-heading"><h2>Проверка снимка</h2>'
            '<p>Выделение F1 и проект заключения</p></div>')
    with gr.Row(elem_id="upload-row"):
        file = gr.File(label="Снимок JPG / PNG", type="filepath", file_types=[".jpg", ".jpeg", ".png"], height=84)
        run = gr.Button("Анализировать", variant="primary")
    paths = sorted(str(p) for p in CFG.examples_dir.glob("*.jpg"))
    if paths:
        gr.Examples([[p] for p in paths], inputs=[file], label="Примеры", elem_id="example-rail")
    summary = gr.HTML('<p class="empty-line">Выберите снимок или откройте пример</p>', elem_id="case-summary")
    with gr.Row(equal_height=False, elem_id="review-columns"):
        with gr.Column(scale=7, elem_id="image-column"):
            with gr.Tabs():
                with gr.Tab("Разметка модели"):
                    preview = gr.Image(label="Область F1", type="numpy", interactive=False, height=520,
                                       buttons=["fullscreen", "download"], elem_classes="viewer-surface")
                with gr.Tab("Правка F1"):
                    editor = gr.ImageEditor(label="Кисть добавляет, ластик удаляет", type="numpy", image_mode="RGBA",
                                            sources=[], transforms=(), brush=gr.Brush(default_size=16, colors=["#28dc96"], color_mode="fixed"),
                                            layers=False, height=520, format="png", interactive=True, elem_classes="viewer-surface")
            gr.HTML('<div class="region-legend"><span><i class="legend-key stable"></i>Согласованная область</span>'
                    '<span><i class="legend-key uncertain"></i>Нестабильная граница</span></div>')
            gr.HTML('<p class="helper">После правки F1 обновите текст. Текущая версия текста останется в истории.</p>')
            apply = gr.Button("Обновить текст по области F1")
        with gr.Column(scale=5, elem_id="report-column"):
            text = gr.Textbox(label="Проект заключения", lines=20, interactive=True, elem_id="report-text",
                              placeholder="Текст появится после анализа. Его можно исправить перед сохранением.")
            with gr.Row(elem_id="reviewer-row"):
                physician = gr.Textbox(label="Врач", placeholder="Имя или инициалы")
            approved = gr.Checkbox(label="Снимок, область и текст проверены", value=False, elem_id="approval")
            save = gr.Button("Сохранить заключение", variant="primary")
            status = gr.Markdown(elem_id="save-result")
            with gr.Row(elem_id="download-row"):
                pdf = gr.DownloadButton("Заключение PDF", visible=False)
                archive = gr.DownloadButton("Материалы ZIP", visible=False)
    with gr.Accordion("Метод и ограничения", open=False):
        gr.Markdown("Grad-CAM показывает область внимания классификатора. Точность границ опухоли не проверена. "
                    "Анализируется один 2D-срез без масштаба в мм и подтверждённой анатомической ориентации. "
                    "Ядро и отёк не разделяются. Внешняя клиническая валидация не проводилась. "
                    "Отметка проверки не заменяет электронную подпись.")

    def do_analyze(path):
        if not path:
            raise gr.Error("Выберите изображение.")
        try:
            with model_lock:
                case = analyze_review(path, get_explainer())
            case["model_sha256"] = hashlib.sha256(CFG.checkpoint.read_bytes()).hexdigest()
        except Exception as e:
            raise gr.Error(f"Не удалось выполнить анализ: {e}") from e
        cls = CLASSES_RU[CLASSES[case["pred"]]]
        confidence = f"{case['confidence']:.1%}".replace('.', ',')
        message = (f'<div class="case-line"><span class="case-diagnosis">{html.escape(cls.capitalize())}</span>'
                   f'<span class="case-meta">Вероятность класса {confidence}</span>'
                   f'<span class="case-meta">Устойчивость {round(case["agreement"] * 5)}/5</span>'
                   f'<span class="case-meta">{str(round(case["elapsed_s"], 1)).replace(".", ",")} с</span></div>')
        if case["confidence"] < CFG.low_conf_threshold or case["agreement"] < 1:
            message += '<p class="review-alert">Нужна дополнительная проверка: низкая уверенность или нестабильный класс</p>'
        return case, overlay(case["rgb"], case["mask"], case["uncertain"]), editor_value(case), draft_report(case), message, False, "", gr.update(visible=False), gr.update(visible=False)

    def do_apply(case, value, previous):
        if not case:
            raise gr.Error("Сначала выполните анализ.")
        try:
            updated = apply_region(case, value, previous)
        except ValueError as e:
            raise gr.Error(str(e)) from e
        return updated, draft_report(updated), False, f"Текст обновлён по области F1. Версия {updated['revision']}.", gr.update(visible=False), gr.update(visible=False)

    def do_save(case, value, report, reviewer, checked):
        try:
            files = export_review(case, value, report, reviewer or "", checked)
        except ValueError as e:
            raise gr.Error(str(e)) from e
        return "Сохранено. PDF и материалы доступны ниже.", gr.update(value=files["pdf"], visible=True), gr.update(value=files["zip"], visible=True)

    run.click(do_analyze, [file], [case_state, preview, editor, text, summary, approved, status, pdf, archive], concurrency_limit=1)
    apply.click(do_apply, [case_state, editor, text], [case_state, text, approved, status, pdf, archive])
    save.click(do_save, [case_state, editor, text, physician, approved], [status, pdf, archive])
    # A changed text/region invalidates approval and previously offered downloads.
    def invalidate():
        return False, "Есть несохранённые изменения", gr.update(visible=False), gr.update(visible=False)
    text.input(invalidate, [], [approved, status, pdf, archive])
    editor.input(invalidate, [], [approved, status, pdf, archive])
    physician.input(invalidate, [], [approved, status, pdf, archive])
    approved.input(lambda: ("Статус изменён. Сохраните новую версию.", gr.update(visible=False), gr.update(visible=False)),
                   [], [status, pdf, archive])
    def clear_case():
        return None, None, None, "", "Снимок изменён. Выполните анализ.", False, "", gr.update(visible=False), gr.update(visible=False)
    file.change(clear_case, [], [case_state, preview, editor, text, summary, approved, status, pdf, archive])
