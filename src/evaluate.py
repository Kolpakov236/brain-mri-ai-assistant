"""Оценка на test: метрики, калибровка, подгруппы, проверка локализации, анализ ошибок, report.md.

Запуск: python -m src.evaluate [--checkpoint models/best.pt]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

from .config import CFG, CLASSES, CLASSES_RU, Config
from .data import load_dataset, make_loader, to_tensor
from .explain import NO_TUMOR_IDX, Explainer
from .model import load_checkpoint
from .train import predict_loader
from .utils import get_device, seed_everything

SHORT_RU = {"glioma": "глиома", "meningioma": "менингиома", "notumor": "норма", "pituitary": "гипофиз"}


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #
def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """ECE по максимальной вероятности и таблица для reliability-графика."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = float((pred[m] == y[m]).mean()), float(conf[m].mean())
            ece += m.mean() * abs(acc - c)
            table.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "acc": acc, "conf": c})
    return float(ece), table


def classification_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    """Основные метрики многоклассовой классификации и бинарной задачи 'опухоль vs норма'."""
    pred = probs.argmax(1)
    p, r, f1, sup = precision_recall_fscore_support(y, pred, labels=range(len(CLASSES)), zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    auc = roc_auc_score(label_binarize(y, classes=range(len(CLASSES))), probs, average=None)
    tumor_true = y != NO_TUMOR_IDX
    tumor_pred = pred != NO_TUMOR_IDX
    tumor_score = 1.0 - probs[:, NO_TUMOR_IDX]
    tp, fn = int((tumor_true & tumor_pred).sum()), int((tumor_true & ~tumor_pred).sum())
    tn, fp = int((~tumor_true & ~tumor_pred).sum()), int((~tumor_true & tumor_pred).sum())
    ece, table = expected_calibration_error(probs, y)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1.mean()),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]),
                          "support": int(sup[i]), "roc_auc_ovr": float(auc[i])} for i, c in enumerate(CLASSES)},
        "roc_auc_ovr_macro": float(auc.mean()),
        "confusion_matrix": cm.tolist(),
        "tumor_vs_normal": {
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "roc_auc": float(roc_auc_score(tumor_true, tumor_score)),
        },
        "ece": ece,
        "reliability": table,
    }


def subgroup_metrics(df: pd.DataFrame, probs: np.ndarray, y: np.ndarray) -> dict:
    """Точность и recall опухолей по подгруппам формата файла и размера (проверка шортката по источнику)."""
    pred = probs.argmax(1)
    size = np.where((df.width == 224) & (df.height == 224), "224x224",
                    np.where((df.width == 512) & (df.height == 512), "512x512", "другой размер"))
    mode = np.where(df["mode"] == "L", "1 канал (L)", "3-4 канала (RGB/др.)")
    out: dict = {}
    for name, keys in (("формат файла", mode), ("размер", size)):
        for k in sorted(set(keys)):
            m = keys == k
            tumor = m & (y != NO_TUMOR_IDX)
            out[f"{name}: {k}"] = {
                "n": int(m.sum()), "accuracy": float((pred[m] == y[m]).mean()),
                "tumor_recall": float((pred[tumor] != NO_TUMOR_IDX).mean()) if tumor.any() else None,
                "share_tumor": float((y[m] != NO_TUMOR_IDX).mean()),
            }
    return out


def low_confidence_stats(probs: np.ndarray, y: np.ndarray, thr: float) -> dict:
    """Что происходит с точностью выше и ниже порога уверенности (используется в приложении)."""
    conf, pred = probs.max(1), probs.argmax(1)
    low = conf < thr
    return {
        "threshold": thr,
        "share_low": float(low.mean()),
        "acc_low": float((pred[low] == y[low]).mean()) if low.any() else None,
        "acc_high": float((pred[~low] == y[~low]).mean()),
        "errors_total": int((pred != y).sum()),
        "errors_flagged_low": int(((pred != y) & low).sum()),
    }


# --------------------------------------------------------------------------- #
# Графики
# --------------------------------------------------------------------------- #
def plot_confusion(cm: list, path: Path) -> None:
    """Матрица ошибок (абсолютные числа и доли по строке)."""
    cm = np.array(cm)
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    labels = [SHORT_RU[c] for c in CLASSES]
    ax.set(xticks=range(4), yticks=range(4), xticklabels=labels, yticklabels=labels,
           xlabel="предсказано", ylabel="истинный класс", title="Матрица ошибок (test)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_roc(probs: np.ndarray, y: np.ndarray, path: Path) -> None:
    """ROC-кривые one-vs-rest по классам и бинарная 'опухоль vs норма'."""
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    onehot = label_binarize(y, classes=range(4))
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(onehot[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{SHORT_RU[c]} (AUC {roc_auc_score(onehot[:, i], probs[:, i]):.3f})")
    fpr, tpr, _ = roc_curve(y != NO_TUMOR_IDX, 1 - probs[:, NO_TUMOR_IDX])
    ax.plot(fpr, tpr, "k--", label="опухоль vs норма")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8)
    ax.set(xlabel="доля ложных тревог (FPR)", ylabel="чувствительность (TPR)", title="ROC-кривые (test)")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_reliability(table: list[dict], ece: float, path: Path) -> None:
    """Reliability-график: уверенность модели против фактической точности."""
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="идеальная калибровка")
    ax.plot([t["conf"] for t in table], [t["acc"] for t in table], "o-", label=f"модель (ECE {ece:.3f})")
    ax.set(xlabel="средняя уверенность", ylabel="доля верных", title="Калибровка (test)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def grid(images: list[np.ndarray], titles: list[str], cols: int, path: Path, cell: float = 2.6) -> None:
    """Сетка изображений с подписями."""
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cell * cols, (cell + 0.45) * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, im, t in zip(np.atleast_1d(axes).ravel(), images, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# Локализация и ошибки
# --------------------------------------------------------------------------- #
def occlusion_check(model: torch.nn.Module, explainer: Explainer, arr: np.ndarray, y: np.ndarray,
                    probs: np.ndarray, n: int, seed: int) -> dict:
    """Косвенная проверка карты: падение уверенности при закрытии рамки Grad-CAM против случайной рамки того же размера.

    Это не IoU (разметки нет), а проверка того, что выделенная область действительно важна для решения модели.
    """
    rng = np.random.default_rng(seed)
    ok = np.where((probs.argmax(1) == y) & (y != NO_TUMOR_IDX))[0]
    ok = rng.choice(ok, min(n, len(ok)), replace=False)
    drops_box, drops_rand, areas = [], [], []
    for i in ok:
        gray = arr[i]
        ex = explainer.explain(gray)
        if ex.bbox is None:
            continue
        x0, y0, x1, y1 = ex.bbox
        w, h = x1 - x0, y1 - y0
        rx, ry = int(rng.integers(0, 224 - w + 1)), int(rng.integers(0, 224 - h + 1))
        occ_box, occ_rand = gray.copy(), gray.copy()
        occ_box[y0:y1, x0:x1] = 0
        occ_rand[ry:ry + h, rx:rx + w] = 0
        with torch.no_grad():
            p = torch.softmax(model(torch.stack([to_tensor(occ_box), to_tensor(occ_rand)])), 1).numpy()
        drops_box.append(ex.probs[ex.pred] - p[0, ex.pred])
        drops_rand.append(ex.probs[ex.pred] - p[1, ex.pred])
        areas.append(w * h / (224 * 224))
    db, dr = np.array(drops_box), np.array(drops_rand)
    return {
        "n": int(len(db)),
        "mean_bbox_area_share": float(np.mean(areas)),
        "mean_drop_gradcam_box": float(db.mean()),
        "mean_drop_random_box": float(dr.mean()),
        "share_box_drop_greater": float((db > dr).mean()),
    }


def error_and_example_grids(explainer: Explainer, arr: np.ndarray, y: np.ndarray, probs: np.ndarray,
                            out: Path, seed: int) -> list[dict]:
    """Сетки: до 16 ошибок (самые уверенные) и по 2 верных примера на класс."""
    pred, conf = probs.argmax(1), probs.max(1)
    wrong = np.where(pred != y)[0]
    wrong = wrong[np.argsort(-conf[wrong])][:16]
    imgs, titles, info = [], [], []
    for i in wrong:
        ex = explainer.explain(arr[i])
        imgs.append(ex.overlay_box)
        titles.append(f"истина: {SHORT_RU[CLASSES[y[i]]]}\nпредсказано: {SHORT_RU[CLASSES[pred[i]]]} ({conf[i]:.0%})")
        info.append({"index": int(i), "true": CLASSES[y[i]], "pred": CLASSES[pred[i]], "confidence": float(conf[i])})
    if imgs:
        grid(imgs, titles, 4, out / "errors_grid.png")
    rng = np.random.default_rng(seed)
    imgs, titles = [], []
    for c in range(len(CLASSES)):
        good = np.where((pred == y) & (y == c))[0]
        for i in rng.choice(good, 2, replace=False):
            ex = explainer.explain(arr[i])
            imgs += [ex.overlay_box, ex.overlay_heat]
            titles += [f"{SHORT_RU[CLASSES[c]]} ({conf[i]:.0%})", "Grad-CAM"]
    grid(imgs, titles, 4, out / "gradcam_examples.png")
    return info


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #
def pct(x: float) -> str:
    """Доля -> проценты с одним знаком."""
    return f"{100 * x:.1f}%"


def build_report(m: dict, summary: dict, ckpt: dict, thr: float) -> str:
    """Текст report.md: все числа берутся из словаря метрик."""
    t, tv = m["test"], m["test"]["tumor_vs_normal"]
    cm = np.array(t["confusion_matrix"])
    off = [(cm[i, j], i, j) for i in range(4) for j in range(4) if i != j]
    top = sorted(off, reverse=True)[:3]
    confusions = "\n".join(
        f"- {CLASSES_RU[CLASSES[i]]} -> предсказана как {CLASSES_RU[CLASSES[j]]}: {n} из {cm[i].sum()} ({pct(n / cm[i].sum())})"
        for n, i, j in top)
    rows = "\n".join(
        f"| {CLASSES_RU[c]} | {pct(v['precision'])} | {pct(v['recall'])} | {pct(v['f1'])} | {v['support']} | {v['roc_auc_ovr']:.3f} |"
        for c, v in t["per_class"].items())
    sub = "\n".join(
        f"| {k} | {v['n']} | {pct(v['accuracy'])} | "
        f"{pct(v['tumor_recall']) if v['tumor_recall'] is not None else '-'} | {pct(v['share_tumor'])} |"
        for k, v in m["subgroups"].items())
    lc, loc = m["low_confidence"], m["localization_occlusion"]
    rm = summary["removed_by_reason"]
    return f"""# Отчёт об оценке модели

Модель: {ckpt['arch']} (transfer learning), лучшая эпоха {ckpt['epoch']}, val macro-F1 {ckpt['val_macro_f1']:.4f}.
Все числа ниже посчитаны скриптом `python -m src.evaluate` на отложенной тестовой выборке ({t['n']} снимков).

## Данные и разбиение

- Исходно {summary['raw_total']} файлов. Удалено точных дубликатов: {rm.get('duplicate', 0)}, синтетических аугментированных копий (суффиксы BR/DA/HF/RO/SP/VF): {rm.get('augmented_copy', 0)}, битых: {rm.get('corrupt', 0)}. Осталось {summary['kept_total']}.
- Штатное разбиение Train/Test не использовано: между ними найдены одинаковые снимки (утечка). Пул пересобран и разбит 70/15/15 стратифицированно по классам и по группам близких копий ({summary['groups']} групп). Размеры: {summary['split_sizes']}. Пересечений групп между выборками: {summary['group_overlap_between_splits']}.
- **Ограничение:** идентификаторов пациентов в данных нет, поэтому разбиение по пациентам невозможно. Срезы одного пациента с разных ракурсов могут оказаться в разных выборках, метрики могут быть завышены.

## Метрики (test)

- Accuracy: **{pct(t['accuracy'])}**, macro-F1: **{pct(t['macro_f1'])}**, ROC-AUC (OvR, macro): **{t['roc_auc_ovr_macro']:.3f}**.

| Класс | Precision | Recall | F1 | N | ROC-AUC |
|---|---|---|---|---|---|
{rows}

Опухоль vs норма (бинарная задача): чувствительность **{pct(tv['sensitivity'])}** (пропущено опухолей: {tv['fn']} из {tv['tp'] + tv['fn']}), специфичность **{pct(tv['specificity'])}** (ложных тревог: {tv['fp']} из {tv['tn'] + tv['fp']}), ROC-AUC {tv['roc_auc']:.3f}.

Рисунки: `confusion_matrix.png`, `roc_curves.png`.

## Калибровка уверенности

ECE = {t['ece']:.3f} (`reliability.png`). Порог "низкая уверенность" в приложении: {thr:.2f}. Ниже порога оказывается {pct(lc['share_low'])} снимков; точность ниже порога {pct(lc['acc_low']) if lc['acc_low'] is not None else '-'}, выше порога {pct(lc['acc_high'])}. Ошибок всего {lc['errors_total']}, из них помечено как низкая уверенность {lc['errors_flagged_low']}.

## Разбор по подгруппам (проверка зависимости от источника снимка)

| Подгруппа | N | Accuracy | Recall опухолей | Доля опухолей |
|---|---|---|---|---|
{sub}

Если качество заметно отличается между форматами/размерами, модель частично использует особенности источника, а не патологию.

## Локализация

Разметки области опухоли (маски, рамки) в данных нет, поэтому IoU/Dice посчитать нельзя. Локализация - слабо контролируемая (Grad-CAM, порог {CFG.cam_threshold:.0%} от максимума, наибольшая связная компонента) и **приблизительная**. Примеры: `gradcam_examples.png`.

Косвенная проверка (закрытие области): на {loc['n']} верно классифицированных опухолевых снимках закрытие рамки Grad-CAM снижает уверенность в среднем на {loc['mean_drop_gradcam_box']:.3f}, закрытие случайной рамки того же размера - на {loc['mean_drop_random_box']:.3f}; рамка Grad-CAM оказалась важнее случайной в {pct(loc['share_box_drop_greater'])} случаев. Средняя площадь рамки: {pct(loc['mean_bbox_area_share'])} кадра. Это показывает важность области для решения модели, но не точность совпадения с границами опухоли.

## Анализ ошибок

Наиболее частые путаницы:

{confusions}

Сетка самых уверенных ошибок: `errors_grid.png`. Возможные причины: схожий вид менингиом и глиом на отдельных срезах, малые опухоли гипофиза, разный ракурс (аксиальный, коронарный, сагиттальный) и последовательность, разное качество исходных снимков.

## Ограничения

- Один составной источник данных, нет внешней валидации на снимках другой клиники и сканера.
- 2D-срезы вместо 3D-объёмов, нет клинических данных пациента.
- Нет ID пациентов: возможна остаточная утечка между выборками.
- Класс "норма" содержит иные последовательности (T2, FLAIR, разные ракурсы), чем часть опухолевых снимков: модель может опираться на тип последовательности.
- Локализация приблизительная (Grad-CAM), без разметки для количественной оценки.
- Результат носит вспомогательный характер и требует подтверждения врачом.
"""


def evaluate(cfg: Config, ckpt_path: Path, n_occ: int) -> dict:
    """Полная оценка: метрики, графики, отчёт."""
    seed_everything(cfg.seed)
    cfg.ensure_dirs()
    device = get_device()
    df, arr = load_dataset(cfg)
    model, ckpt = load_checkpoint(ckpt_path, device)
    probs, y, _ = predict_loader(model, make_loader(df, arr, "test", cfg), device)
    test_mask = (df["split"] == "test").to_numpy()
    df_t, arr_t = df[test_mask].reset_index(drop=True), arr[test_mask]

    metrics = {"test": classification_metrics(probs, y)}
    metrics["subgroups"] = subgroup_metrics(df_t, probs, y)
    metrics["low_confidence"] = low_confidence_stats(probs, y, cfg.low_conf_threshold)

    cpu_model, _ = load_checkpoint(ckpt_path, "cpu")  # Grad-CAM на CPU: надёжнее, чем на MPS
    explainer = Explainer(cpu_model, ckpt["arch"], "cpu", cfg)
    metrics["localization_occlusion"] = occlusion_check(cpu_model, explainer, arr_t, y, probs, n_occ, cfg.seed)
    metrics["errors"] = error_and_example_grids(explainer, arr_t, y, probs, cfg.reports_dir, cfg.seed)

    plot_confusion(metrics["test"]["confusion_matrix"], cfg.reports_dir / "confusion_matrix.png")
    plot_roc(probs, y, cfg.reports_dir / "roc_curves.png")
    plot_reliability(metrics["test"]["reliability"], metrics["test"]["ece"], cfg.reports_dir / "reliability.png")

    (cfg.reports_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    summary = json.loads((cfg.reports_dir / "data_summary.json").read_text())
    (cfg.reports_dir / "report.md").write_text(build_report(metrics, summary, ckpt, cfg.low_conf_threshold))
    np.save(cfg.reports_dir / "test_probs.npy", probs)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=CFG.checkpoint)
    ap.add_argument("--n-occlusion", type=int, default=150)
    a = ap.parse_args()
    res = evaluate(CFG, a.checkpoint, a.n_occlusion)
    t = res["test"]
    print(f"accuracy {t['accuracy']:.4f}  macro-F1 {t['macro_f1']:.4f}  AUC {t['roc_auc_ovr_macro']:.4f}")
    print("tumor vs normal:", {k: round(v, 4) for k, v in t["tumor_vs_normal"].items() if isinstance(v, float)})
