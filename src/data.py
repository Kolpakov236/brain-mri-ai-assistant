"""Подготовка данных: индексация, чистка, дедупликация, группы, сплит, Dataset.

Запуск: python -m src.data  (строит reports/splits.csv и кэш изображений).
"""
from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from .config import AUG_SUFFIXES, CFG, CLASSES, Config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_IMG_EXT = {".jpg", ".jpeg", ".png"}
_AUG_RE = re.compile(rf"^([GMNP])_(\d+)_({'|'.join(AUG_SUFFIXES)})_\.\w+$")
_FAMILY_RE = re.compile(r"^([GMNP])_(\d+)(?:_\w*)?\.\w+$")


# --------------------------------------------------------------------------- #
# Предобработка одного снимка (общая для обучения и приложения)
# --------------------------------------------------------------------------- #
def preprocess(img: Image.Image, size: int) -> np.ndarray:
    """Серое изображение -> квадрат size x size (uint8), пропорции сохранены, фон чёрный."""
    gray = img.convert("L")
    w, h = gray.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    arr = np.asarray(gray.resize((nw, nh), Image.LANCZOS))
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = arr
    return canvas


def to_tensor(gray: np.ndarray) -> torch.Tensor:
    """uint8 (H, W) -> нормализованный float-тензор (3, H, W)."""
    x = torch.from_numpy(gray).float().div_(255.0).unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


def _dhash(gray: np.ndarray, size: int) -> np.ndarray:
    """Разностный хэш (size*size бит) для поиска близких копий."""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).ravel().astype(np.uint8)


# --------------------------------------------------------------------------- #
# Индексация и чистка
# --------------------------------------------------------------------------- #
def _process_file(args: tuple[Path, int, int]) -> dict:
    """Читает файл: md5, метаданные, предобработанный кадр, хэши (обычный и зеркальный)."""
    path, size, hsize = args
    rec: dict = {"ok": False, "error": ""}
    try:
        rec["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
        with Image.open(path) as im:
            im.load()
            rec["width"], rec["height"] = im.size
            rec["mode"] = im.mode
            rgb = np.asarray(im.convert("RGB"), dtype=np.int16)
            rec["gray_content"] = bool(
                im.mode in ("L", "LA", "I;16")
                or (np.abs(rgb[..., 0] - rgb[..., 1]).max() <= 2 and np.abs(rgb[..., 1] - rgb[..., 2]).max() <= 2)
            )
            gray = preprocess(im, size)
        rec["gray"] = gray
        rec["h"] = _dhash(gray, hsize)
        rec["hf"] = _dhash(gray[:, ::-1].copy(), hsize)
        rec["ok"] = True
    except Exception as exc:  # битый файл
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def scan_files(cfg: Config) -> pd.DataFrame:
    """Список файлов Train/Test с классом и признаками по имени (аугментированная копия, семейство)."""
    rows = []
    for split_dir in ("Train", "Test"):
        for cls in CLASSES:
            for p in sorted((cfg.data_dir / split_dir / cls).iterdir()):
                if p.suffix.lower() not in _IMG_EXT:
                    continue
                aug = _AUG_RE.match(p.name)
                fam = _FAMILY_RE.match(p.name) or aug
                rows.append({
                    "path": str(p),
                    "label": cls,
                    "y": CLASSES.index(cls),
                    "orig_split": split_dir.lower(),
                    "is_aug": bool(aug),
                    "family": f"{fam.group(1)}_{fam.group(2)}" if fam else "",
                })
    return pd.DataFrame(rows)


class _UnionFind:
    """Система непересекающихся множеств для объединения копий в группы."""

    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_groups(df: pd.DataFrame, hashes: np.ndarray, hashes_flip: np.ndarray, max_dist: int) -> np.ndarray:
    """Группы близких копий: по семейству в имени и по dHash (в т.ч. зеркальных). Возвращает id группы."""
    n = len(df)
    uf = _UnionFind(n)
    fam_first: dict[str, int] = {}
    for i, fam in enumerate(df["family"].tolist()):
        if fam:
            if fam in fam_first:
                uf.union(fam_first[fam], i)
            else:
                fam_first[fam] = i
    b = hashes.astype(np.float32)
    bf = hashes_flip.astype(np.float32)
    s = b.sum(1)
    for other in (b, bf):
        so = other.sum(1)
        for start in range(0, n, 2000):
            blk = b[start:start + 2000]
            dist = s[start:start + 2000, None] + so[None, :] - 2.0 * blk @ other.T
            ii, jj = np.where(dist <= max_dist)
            for i, j in zip(ii + start, jj):
                if i < j:
                    uf.union(int(i), int(j))
    roots = np.array([uf.find(i) for i in range(n)])
    return pd.factorize(roots)[0]


def assign_split(df: pd.DataFrame, cfg: Config) -> list[str]:
    """Стратифицированное разбиение по группам: 20 фолдов -> 14 train / 3 val / 3 test (70/15/15)."""
    n_splits = 20
    n_val, n_test = round(cfg.split[1] * n_splits), round(cfg.split[2] * n_splits)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    split = np.array(["train"] * len(df), dtype=object)
    for fold, (_, te) in enumerate(sgkf.split(df, df["y"], df["group"])):
        if fold < n_test:
            split[te] = "test"
        elif fold < n_test + n_val:
            split[te] = "val"
    return split.tolist()


def summarize(raw: pd.DataFrame, kept: pd.DataFrame, removed: pd.DataFrame) -> dict:
    """Сводка для отчёта: что удалено, итоговые размеры, проверка утечки по группам."""
    leak = {
        f"{a}_{b}": int(len(set(kept.loc[kept.split == a, "group"]) & set(kept.loc[kept.split == b, "group"])))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    return {
        "raw_total": int(len(raw)),
        "removed_by_reason": {k: int(v) for k, v in removed["reason"].value_counts().items()},
        "kept_total": int(len(kept)),
        "groups": int(kept["group"].nunique()),
        "largest_group": int(kept["group"].value_counts().max()),
        "group_overlap_between_splits": leak,
        "split_sizes": {k: int(v) for k, v in kept["split"].value_counts().items()},
        "split_by_class": {
            s: {k: int(v) for k, v in kept[kept.split == s]["label"].value_counts().items()}
            for s in ("train", "val", "test")
        },
        "gray_content_by_class": kept.groupby("label")["gray_content"].mean().round(3).to_dict(),
        "gray_content_by_split": kept.groupby("split")["gray_content"].mean().round(3).to_dict(),
    }


def build_dataset(cfg: Config = CFG) -> pd.DataFrame:
    """Полный конвейер: чистка, дедуп, группы, сплит. Пишет splits.csv, removed.csv, кэш и сводку."""
    cfg.ensure_dirs()
    df = scan_files(cfg)
    jobs = [(Path(p), cfg.img_size, cfg.dhash_size) for p in df["path"]]
    with ThreadPoolExecutor(max_workers=8) as ex:
        recs = list(tqdm(ex.map(_process_file, jobs), total=len(jobs), desc="Чтение снимков"))
    meta = pd.DataFrame([{k: v for k, v in r.items() if k not in ("gray", "h", "hf")} for r in recs])
    df = pd.concat([df, meta], axis=1)
    cols = ["path", "label", "reason", "error"]
    removed: list[pd.DataFrame] = []

    bad = ~df["ok"]
    removed.append(df[bad].assign(reason="corrupt")[cols])
    keep = df[~bad].copy()

    # Точные дубликаты: оставляем один файл (приоритет оригиналу, затем по пути)
    keep = keep.sort_values(["is_aug", "path"], kind="stable")
    dup = keep.duplicated("md5", keep="first")
    removed.append(keep[dup].assign(reason="duplicate")[cols])
    keep = keep[~dup]

    # Синтетические аугментированные копии
    aug = keep["is_aug"]
    removed.append(keep[aug].assign(reason="augmented_copy")[cols])
    keep = keep[~aug].sort_index()

    idx = keep.index.to_numpy()
    grays = np.stack([recs[i]["gray"] for i in idx])
    hashes = np.stack([recs[i]["h"] for i in idx])
    hashes_f = np.stack([recs[i]["hf"] for i in idx])
    keep = keep.reset_index(drop=True)

    keep["group"] = build_groups(keep, hashes, hashes_f, cfg.dhash_max_dist)
    keep["split"] = assign_split(keep, cfg)

    keep = keep.drop(columns=["ok", "error", "is_aug"])
    keep.to_csv(cfg.reports_dir / "splits.csv", index=False)
    rem = pd.concat(removed, ignore_index=True)
    rem.to_csv(cfg.reports_dir / "removed.csv", index=False)
    np.save(cfg.cache_dir / f"images_{cfg.img_size}.npy", grays)

    summary = summarize(df, keep, rem)
    (cfg.reports_dir / "data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return keep


def load_dataset(cfg: Config = CFG) -> tuple[pd.DataFrame, np.ndarray]:
    """Загружает splits.csv и кэш изображений (строка i таблицы = кадр i массива)."""
    df = pd.read_csv(cfg.reports_dir / "splits.csv")
    arr = np.load(cfg.cache_dir / f"images_{cfg.img_size}.npy")
    assert len(df) == len(arr), "splits.csv и кэш рассинхронизированы - перезапустите python -m src.data"
    return df, arr


# --------------------------------------------------------------------------- #
# Dataset и загрузчики
# --------------------------------------------------------------------------- #
class _AddNoise:
    """Гауссов шум для тензора [0, 1] (имитация шумных снимков)."""

    def __init__(self, std: float = 0.03, p: float = 0.25) -> None:
        self.std, self.p = std, p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            x = (x + torch.randn_like(x) * self.std).clamp_(0, 1)
        return x


def train_transform() -> v2.Compose:
    """Аугментации train: малые повороты/сдвиги/масштаб, яркость/контраст, blur, шум. Флипов нет."""
    return v2.Compose([
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        v2.ColorJitter(brightness=0.25, contrast=0.25),
        v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.25),
        _AddNoise(),
    ])


class MRIDataset(Dataset):
    """Датасет из кэша uint8-кадров. train=True включает аугментации."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, train: bool) -> None:
        self.images, self.labels = images, labels
        self.tf = train_transform() if train else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        gray = self.images[i]
        if self.tf is not None:
            x = torch.from_numpy(gray).float().div(255.0).unsqueeze(0)
            x = self.tf(x).squeeze(0)
            gray = (x.clamp(0, 1) * 255.0).round().byte().numpy()
        return to_tensor(gray), int(self.labels[i])


def make_loader(df: pd.DataFrame, arr: np.ndarray, split: str, cfg: Config = CFG) -> DataLoader:
    """DataLoader для одного сплита."""
    mask = (df["split"] == split).to_numpy()
    ds = MRIDataset(arr[mask], df.loc[mask, "y"].to_numpy(), train=(split == "train"))
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=(split == "train"),
                      num_workers=cfg.num_workers, drop_last=False)


def class_weights(y: np.ndarray) -> torch.Tensor:
    """Веса классов для взвешенного CrossEntropy: n / (K * count_k)."""
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)
    return torch.tensor(len(y) / (len(CLASSES) * counts), dtype=torch.float32)


if __name__ == "__main__":
    build_dataset()
    print((CFG.reports_dir / "data_summary.json").read_text())
