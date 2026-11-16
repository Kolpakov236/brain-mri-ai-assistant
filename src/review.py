"""Stage 2: perturbation-consensus ROI, native-image coordinates and grounded drafts.

The ROI remains an explanation of a classifier, not a validated tumor segmentation.
No anatomy, patient identity, physical size or glioblastoma subtype is inferred.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import CLASSES, CLASSES_RU, CFG
from .data import preprocess
from .explain import NO_TUMOR_IDX, largest_component

METHOD = "gradcam-consensus-v1"


def native_map(a: np.ndarray, width: int, height: int, discrete: bool = False) -> np.ndarray:
    """Undo the exact rounded letterbox used by data.preprocess, then resize."""
    size = a.shape[0]
    scale = size / max(width, height)
    nw, nh = max(1, round(width * scale)), max(1, round(height * scale))
    x, y = (size - nw) // 2, (size - nh) // 2
    return cv2.resize(a[y:y + nh, x:x + nw].astype(np.float32), (width, height),
                      interpolation=cv2.INTER_NEAREST if discrete else cv2.INTER_LINEAR)


def mask_stats(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"area_px": 0, "bbox": None, "position": "область не выделена"}
    h, w = mask.shape
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    cx, cy = float(xs.mean()) / w, float(ys.mean()) / h
    horizontal = "слева" if cx < 1 / 3 else "справа" if cx > 2 / 3 else "по центру"
    vertical = "в верхней трети" if cy < 1 / 3 else "в нижней трети" if cy > 2 / 3 else "в средней трети"
    return {"area_px": int(len(xs)), "bbox": bbox,
            "position": f"{horizontal}, {vertical} изображения"}


def overlay(rgb: np.ndarray, mask: np.ndarray, uncertain: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    for selected, color in ((uncertain, (255, 160, 32)), (mask, (50, 220, 160))):
        out[selected] = (out[selected].astype(float) * .52 + np.array(color) * .48).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (40, 255, 170), max(1, round(max(rgb.shape[:2]) / 400)))
    return out


def analyze_review(path: str, explainer) -> dict:
    start = time.perf_counter()
    with Image.open(path) as im:
        im.load()
        # Bound memory for the interactive editor without silently resizing the input.
        if im.width * im.height > 16_000_000:
            raise ValueError("Изображение слишком большое: максимум 16 млн пикселей.")
        rgb = np.asarray(im.convert("RGB")).copy()
    h, w = rgb.shape[:2]
    size = explainer.cfg.img_size
    gray = preprocess(Image.fromarray(rgb), size)
    base = explainer.explain(gray)
    variants = [(0, 0, 1.), (3, 0, 1.), (-3, 0, 1.), (0, 3, .92), (0, -3, 1.08)]
    maps, probabilities = [], []
    for dx, dy, gain in variants:
        if (dx, dy, gain) == variants[0]:
            ex = base
        else:
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            variant = cv2.warpAffine(np.clip(gray.astype(float) * gain, 0, 255).astype(np.uint8), m,
                                     (size, size), borderMode=cv2.BORDER_CONSTANT)
            ex = explainer.explain(variant, target=base.pred)
        aligned = cv2.warpAffine(ex.cam, np.float32([[1, 0, -dx], [0, 1, -dy]]), (size, size))
        maps.append(aligned)
        probabilities.append(ex.probs)
    maps = np.stack(maps)
    # Votes across both transforms and nearby thresholds expose threshold sensitivity.
    votes = np.stack([m >= t * m.max() if m.max() > 1e-8 else np.zeros_like(m, bool)
                      for m in maps for t in (.4, .5, .6)]).mean(0)
    mean_cam = maps.mean(0)
    if base.pred == NO_TUMOR_IDX or mean_cam.max() <= 1e-8:
        mask, uncertain = np.zeros_like(gray, bool), np.zeros_like(gray, bool)
    else:
        mask, _ = largest_component(votes.astype(np.float32), .6)
        mask &= votes >= .6
        uncertain = (votes > 0) & (votes < .8)
    native_mask = native_map(mask, w, h, True).astype(bool)
    native_uncertain = native_map(uncertain, w, h, True).astype(bool)
    prob = np.stack(probabilities)
    agreement = float((prob.argmax(1) == base.pred).mean())
    warnings = []
    if base.confidence < CFG.low_conf_threshold:
        warnings.append("Низкая уверенность классификатора: требуется дополнительная проверка.")
    if agreement < 1:
        warnings.append("Предсказанный класс меняется при небольших преобразованиях снимка.")
    if min(w, h) < 128 or gray.std() < 8:
        warnings.append("Ограниченное качество изображения: малый размер или низкая вариативность яркости.")
    if base.pred != NO_TUMOR_IDX and not native_mask.any():
        warnings.append("Не удалось получить устойчивую область: границы не определены.")
    if native_uncertain.any():
        warnings.append("Оранжевые участки чувствительны к преобразованиям и порогу; границу должен проверить врач.")
    warnings.extend([
        "Доступен отдельный 2D-снимок без DICOM-метаданных и полного исследования. Ориентация и масштаб в мм неизвестны.",
        "Выделение основано на Grad-CAM. Это область внимания модели, не подтверждённая граница опухоли. Точность границ не валидирована.",
        "Разделение опухолевого ядра и отёка не выполняется. Глиобластома как отдельный класс не определяется.",
        "Вероятности не гарантируют правильность. Модель не проходила внешнюю клиническую валидацию.",
    ])
    case = {
        "id": hashlib.sha256(rgb.tobytes()).hexdigest()[:16], "image_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "name": Path(path).name, "rgb": rgb, "mask": native_mask, "auto_mask": native_mask.copy(),
        "baseline_mask": native_map(base.mask, w, h, True).astype(bool),
        "uncertain": native_uncertain, "probs": base.probs.tolist(), "pred": base.pred,
        "confidence": base.confidence, "agreement": agreement, "warnings": warnings,
        "method": METHOD, "revision": 0, "history": [], "elapsed_s": time.perf_counter() - start,
        "cam": native_map(mean_cam, w, h),
    }
    case["initial_draft"] = draft_report(case)
    return case


def draft_report(case: dict) -> str:
    s = mask_stats(case["mask"])
    h, w = case["rgb"].shape[:2]
    probability = f"{case['confidence']:.1%}".replace('.', ',')
    lines = [f"Снимок: {case['name']}, {w} × {h} пикселей.", "Описание",
             f"Предположение модели: {CLASSES_RU[CLASSES[case['pred']]]}."]
    if s["bbox"]:
        x0, y0, x1, y1 = s["bbox"]
        lines += [f"Область F1: {s['position']}.", "Измерения F1",
                  f"F1: площадь выделения {s['area_px']} пикселей. Рамка {x1-x0} × {y1-y0} пикселей.",
                  f"Координаты x=[{x0}, {x1}), y=[{y0}, {y1}). Начало отсчёта в верхнем левом углу."]
    elif case["pred"] == NO_TUMOR_IDX:
        lines.append("На представленном срезе классификатор не выявил признаков опухоли. Это не исключает патологию на других срезах.")
    else:
        lines.append("Область F1 не выделена. Локализация недоступна. Нужна ручная проверка.")
    if case["revision"]:
        lines.append(f"Выделение исправлено вручную, версия {case['revision']}. Карта нестабильности показывает исходный результат модели.")
    lines += ["Надёжность", f"Вероятность класса: {probability}. Класс сохранился в {round(case['agreement'] * 5)} из 5 вариантов снимка."]
    signals = []
    if case["confidence"] < CFG.low_conf_threshold:
        signals.append("Вероятность ниже порога 70%. Нужна дополнительная проверка.")
    if case["agreement"] < 1:
        signals.append("При небольших изменениях снимка модель меняет класс.")
    if any("качество" in s for s in case["warnings"]):
        signals.append("Размер или контраст снимка ограничивает анализ.")
    if case["uncertain"].any():
        signals.append("Оранжевые участки нестабильны при изменении яркости, сдвиге и выборе порога.")
    lines += signals
    lines += ["Ограничения",
              "Оценён один 2D-срез. Масштаб в мм и анатомическая ориентация неизвестны. Измерения относятся к выделению, а не к подтверждённому размеру опухоли.",
              "Grad-CAM показывает внимание модели. Точность границ не проверена. Ядро и отёк не разделяются, глиобластома как подтип не определяется. Внешняя клиническая валидация не проводилась."]
    return "\n\n".join(lines).replace(chr(0x2014), '-').replace(chr(0x2013), '-')


def editor_value(case: dict) -> dict:
    layer = np.zeros((*case["mask"].shape, 4), np.uint8)
    layer[case["mask"]] = (40, 220, 150, 160)
    return {"background": case["rgb"].copy(), "layers": [layer],
            "composite": overlay(case["rgb"], case["mask"], np.zeros_like(case["mask"]))}


def editor_mask(value: dict, expected_shape: tuple) -> np.ndarray:
    if not value or value.get("background") is None:
        raise ValueError("Сначала загрузите снимок и выполните анализ.")
    if np.asarray(value["background"]).shape[:2] != expected_shape:
        raise ValueError("Размер холста изменился. Повторите анализ исходного снимка.")
    result = np.zeros(expected_shape, bool)
    for layer in value.get("layers") or []:
        arr = np.asarray(layer)
        if arr.ndim != 3 or arr.shape[:2] != expected_shape or arr.shape[2] != 4:
            raise ValueError("Некорректный слой: ожидается RGBA-маска исходного размера.")
        result |= arr[..., 3] > 0
    return result


def apply_region(case: dict, value: dict, old_text: str) -> dict:
    import copy
    updated = copy.deepcopy(case)
    mask = editor_mask(value, case["mask"].shape)
    if not np.array_equal(mask, case["mask"]):
        updated["revision"] += 1
        updated["history"].append({"event": "region_changed", "revision": updated["revision"],
                                    "previous_text": old_text, "before": mask_stats(case["mask"]), "after": mask_stats(mask)})
        updated["mask"] = mask
    else:
        updated["history"].append({"event": "draft_regenerated", "revision": updated["revision"],
                                    "previous_text": old_text})
    return updated
