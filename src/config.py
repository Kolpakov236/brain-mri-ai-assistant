"""Единая конфигурация проекта: пути, гиперпараметры, пороги.

Пути можно переопределить переменными окружения (MRI_DATA_DIR и т.д.).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CLASSES: tuple[str, ...] = ("glioma", "meningioma", "notumor", "pituitary")
CLASSES_RU: dict[str, str] = {
    "glioma": "глиома",
    "meningioma": "менингиома",
    "notumor": "опухоль не обнаружена",
    "pituitary": "опухоль гипофиза",
}
TUMOR_CLASSES: tuple[str, ...] = ("glioma", "meningioma", "pituitary")

# Суффиксы заранее аугментированных копий в именах файлов (G_675_BR_.jpg и т.п.)
AUG_SUFFIXES: tuple[str, ...] = ("BR", "DA", "HF", "RO", "SP", "VF")


@dataclass(frozen=True)
class Config:
    """Параметры данных, обучения и интерпретации."""

    data_dir: Path = Path(os.environ.get("MRI_DATA_DIR", ROOT / "data"))
    reports_dir: Path = Path(os.environ.get("MRI_REPORTS_DIR", ROOT / "reports"))
    models_dir: Path = Path(os.environ.get("MRI_MODELS_DIR", ROOT / "models"))
    cache_dir: Path = Path(os.environ.get("MRI_CACHE_DIR", ROOT / ".cache"))
    examples_dir: Path = ROOT / "examples"

    seed: int = 42
    img_size: int = 224
    split: tuple[float, float, float] = (0.70, 0.15, 0.15)  # train / val / test
    dhash_size: int = 12  # размер dHash для поиска близких копий
    dhash_max_dist: int = 4  # порог Хэмминга для объединения в одну группу

    model_name: str = "efficientnet_b0"
    epochs: int = 12
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    patience: int = 4  # early stopping по val macro-F1
    num_workers: int = 0

    cam_threshold: float = 0.5  # доля от максимума тепловой карты
    low_conf_threshold: float = 0.70  # ниже - предупреждение в приложении

    checkpoint: Path = field(default=ROOT / "models" / "best.pt")

    def ensure_dirs(self) -> None:
        """Создаёт выходные каталоги."""
        for d in (self.reports_dir, self.models_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)


CFG = Config()
