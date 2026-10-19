"""Локализация через Grad-CAM: тепловая карта -> маска -> наибольшая компонента -> рамка.

Локализация слабо контролируемая (масок разметки в данных нет), поэтому область приблизительная.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn

from .config import CFG, CLASSES, Config
from .data import to_tensor
from .model import cam_target_layer

NO_TUMOR_IDX = CLASSES.index("notumor")


@dataclass
class Explanation:
    """Результат анализа одного снимка (координаты рамки в пикселях кадра img_size)."""

    pred: int
    probs: np.ndarray
    cam: np.ndarray  # (H, W) в [0, 1]
    mask: np.ndarray  # (H, W) bool, наибольшая связная компонента
    bbox: tuple[int, int, int, int] | None  # (x0, y0, x1, y1)
    overlay_heat: np.ndarray  # RGB: снимок + тепловая карта + рамка
    overlay_box: np.ndarray  # RGB: снимок + контур + рамка

    @property
    def confidence(self) -> float:
        """Вероятность предсказанного класса."""
        return float(self.probs[self.pred])


def largest_component(cam: np.ndarray, threshold: float) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Бинарная маска по порогу (доля от максимума) и наибольшая связная компонента с её рамкой."""
    binary = (cam >= threshold * cam.max()).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary, dtype=bool), None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(stats[k, c]) for c in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    return labels == k, (x, y, x + w, y + h)


def render_overlays(gray: np.ndarray, cam: np.ndarray, mask: np.ndarray,
                    bbox: tuple[int, int, int, int] | None, scale: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Два изображения (RGB): с тепловой картой и чистое с контуром + рамкой. bbox None -> без выделения."""
    size = gray.shape[0] * scale
    base = cv2.cvtColor(cv2.resize(gray, (size, size), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
    heat = cv2.applyColorMap((cv2.resize(cam, (size, size)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay_heat = base.copy()
    if bbox is not None:
        overlay_heat = cv2.addWeighted(base, 0.6, heat, 0.4, 0)
    overlay_box = base.copy()
    if bbox is not None:
        contours, _ = cv2.findContours(cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x0, y0, x1, y1 = (v * scale for v in bbox)
        for img in (overlay_heat, overlay_box):
            cv2.drawContours(img, contours, -1, (255, 220, 0), 2)
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 40, 40), 2)
    return overlay_heat, overlay_box


class Explainer:
    """Предсказание класса + Grad-CAM + рамка для одного кадра (uint8, size x size)."""

    def __init__(self, model: nn.Module, arch: str, device: torch.device | str = "cpu", cfg: Config = CFG) -> None:
        self.model = model.to(device).eval()
        self.device, self.cfg = device, cfg
        self.cam = GradCAM(model=self.model, target_layers=[cam_target_layer(self.model, arch)])

    def explain(self, gray: np.ndarray, target: int | None = None) -> Explanation:
        """Анализ кадра. target=None - объясняется предсказанный класс. Для notumor область не выделяется."""
        x = to_tensor(gray).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x).float(), 1)[0].cpu().numpy()
        pred = int(probs.argmax())
        cls = pred if target is None else target
        cam = self.cam(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]
        if cls == NO_TUMOR_IDX:
            mask, bbox = np.zeros(cam.shape, dtype=bool), None
        else:
            mask, bbox = largest_component(cam, self.cfg.cam_threshold)
        heat, box = render_overlays(gray, cam, mask, bbox)
        return Explanation(pred, probs, cam, mask, bbox, heat, box)
