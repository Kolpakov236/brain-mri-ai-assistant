"""Разведка данных (Шаг 0): баланс классов, сетка примеров, пример аугментированных копий.

Запуск: python -m src.recon
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .config import CFG, CLASSES, CLASSES_RU
from .data import scan_files

N_PER_CLASS = 6


def class_balance(df) -> None:
    """Гистограмма числа снимков по классам и штатным сплитам."""
    counts = df.groupby(["label", "orig_split"]).size().unstack(fill_value=0).reindex(CLASSES)
    ax = counts.plot(kind="bar", stacked=True, figsize=(7, 4), color=["#4C78A8", "#F58518"])
    ax.set_xticklabels([CLASSES_RU[c] for c in counts.index], rotation=15)
    ax.set_ylabel("число снимков")
    ax.set_title("Баланс классов (исходные Train + Test, до чистки)")
    for i, total in enumerate(counts.sum(axis=1)):
        ax.text(i, total + 30, str(total), ha="center")
    plt.tight_layout()
    plt.savefig(CFG.reports_dir / "class_balance.png", dpi=150)
    plt.close()


def samples_grid(df, seed: int = CFG.seed) -> None:
    """Сетка случайных примеров: строки - классы."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(CLASSES), N_PER_CLASS, figsize=(2 * N_PER_CLASS, 2.1 * len(CLASSES)))
    for r, cls in enumerate(CLASSES):
        paths = df[(df.label == cls) & ~df.is_aug]["path"].to_numpy()
        for c, p in enumerate(rng.choice(paths, N_PER_CLASS, replace=False)):
            ax = axes[r, c]
            ax.imshow(Image.open(p).convert("L"), cmap="gray")
            ax.axis("off")
            if c == 0:
                ax.set_title(CLASSES_RU[cls], loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(CFG.reports_dir / "samples_grid.png", dpi=130)
    plt.close()


def aug_suffixes_grid(df) -> None:
    """Один исходный снимок и его предаугментированные копии (для проверки смысла суффиксов)."""
    aug = df[df.is_aug & (df.label == "glioma")]
    fam = aug["family"].iloc[0]
    members = df[(df.family == fam)].sort_values("path")
    fig, axes = plt.subplots(1, len(members), figsize=(2.2 * len(members), 2.6))
    for ax, (_, row) in zip(np.atleast_1d(axes), members.iterrows()):
        ax.imshow(Image.open(row["path"]).convert("L"), cmap="gray")
        ax.set_title(row["path"].split("/")[-1], fontsize=7)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(CFG.reports_dir / "aug_suffixes_grid.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    CFG.ensure_dirs()
    index = scan_files(CFG)
    class_balance(index)
    samples_grid(index)
    aug_suffixes_grid(index)
    print(index.groupby(["label", "orig_split"]).size().unstack())
    print("предаугментированных копий по имени:", int(index.is_aug.sum()))
