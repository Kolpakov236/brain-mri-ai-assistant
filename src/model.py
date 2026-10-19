"""Модель классификации: создание, сохранение/загрузка чекпоинта, слой для Grad-CAM."""
from __future__ import annotations

from pathlib import Path

import timm
import torch
from torch import nn

from .config import CLASSES


def build_model(arch: str = "efficientnet_b0", pretrained: bool = True) -> nn.Module:
    """timm-модель с головой на len(CLASSES) классов."""
    return timm.create_model(arch, pretrained=pretrained, num_classes=len(CLASSES))


def save_checkpoint(path: Path, model: nn.Module, arch: str, img_size: int, epoch: int, val_f1: float) -> None:
    """Сохраняет веса и метаданные, достаточные для инференса."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "arch": arch, "classes": list(CLASSES),
        "img_size": img_size, "epoch": epoch, "val_macro_f1": val_f1,
    }, path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    """Загружает модель в eval-режиме и словарь метаданных чекпоинта."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def cam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Последний свёрточный блок признаков для Grad-CAM."""
    if arch.startswith("efficientnet"):
        return model.bn2
    if arch.startswith("resnet"):
        return model.layer4[-1]
    raise ValueError(f"Нет целевого слоя Grad-CAM для архитектуры {arch}")
