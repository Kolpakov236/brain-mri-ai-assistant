"""Копирует по одному верно классифицированному снимку каждого класса из test в examples/ для демо.

Запуск: python -m src.make_examples  (после python -m src.evaluate)
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG, CLASSES

MIN_CONF = 0.90


def main() -> None:
    """Выбирает случайный (seed из конфига) верный и уверенный пример на класс."""
    df = pd.read_csv(CFG.reports_dir / "splits.csv")
    test = df[df.split == "test"].reset_index(drop=True)
    probs = np.load(CFG.reports_dir / "test_probs.npy")
    rng = np.random.default_rng(CFG.seed)
    CFG.examples_dir.mkdir(exist_ok=True)
    for old in CFG.examples_dir.glob("example_*"):
        old.unlink()
    for c, cls in enumerate(CLASSES):
        ok = np.where((test.y == c) & (probs.argmax(1) == c) & (probs.max(1) >= MIN_CONF))[0]
        i = int(rng.choice(ok))
        src = Path(test.loc[i, "path"])
        dst = CFG.examples_dir / f"example_{cls}{src.suffix.lower()}"
        shutil.copy(src, dst)
        print(f"{cls}: {src.name} -> {dst.name} (уверенность {probs[i].max():.0%})")


if __name__ == "__main__":
    main()
