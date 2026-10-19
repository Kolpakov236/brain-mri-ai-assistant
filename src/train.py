"""Обучение классификатора (transfer learning).

Запуск: python -m src.train [--epochs N] [--arch efficientnet_b0|resnet50] [--smoke]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from .config import CFG, Config
from .data import class_weights, load_dataset, make_loader
from .model import build_model, save_checkpoint
from .utils import get_device, seed_everything


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    """Возвращает (вероятности, истинные метки, средний CE-loss) по загрузчику."""
    model.eval()
    probs, ys, loss_sum = [], [], 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        logits = model(x.to(device))
        loss_sum += ce(logits, y.to(device)).item()
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        ys.append(y.numpy())
    y_all = np.concatenate(ys)
    return np.concatenate(probs), y_all, loss_sum / len(y_all)


def plot_curves(log: pd.DataFrame, path: Path) -> None:
    """Кривые loss и val-метрик по эпохам."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(log.epoch, log.train_loss, label="train")
    ax[0].plot(log.epoch, log.val_loss, label="val")
    ax[0].set(xlabel="эпоха", ylabel="loss", title="Loss")
    ax[0].legend()
    ax[1].plot(log.epoch, log.val_macro_f1, label="val macro-F1")
    ax[1].plot(log.epoch, log.val_acc, label="val accuracy")
    ax[1].set(xlabel="эпоха", title="Метрики на валидации")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def train(cfg: Config, arch: str, epochs: int, amp: bool, smoke: bool, tag: str) -> float:
    """Полный цикл обучения. Возвращает лучший val macro-F1."""
    seed_everything(cfg.seed)
    device = get_device()
    print(f"Устройство: {device}, архитектура: {arch}, эпох: {epochs}")
    df, arr = load_dataset(cfg)
    if smoke:  # маленькая подвыборка для проверки конвейера
        parts = [df[df.split == s].sample(min((df.split == s).sum(), 160), random_state=cfg.seed)
                 for s in ("train", "val", "test")]
        df = pd.concat(parts)
        arr = arr[df.index.to_numpy()]
        df = df.reset_index(drop=True)
    train_loader = make_loader(df, arr, "train", cfg)
    val_loader = make_loader(df, arr, "val", cfg)

    model = build_model(arch, pretrained=True).to(device)
    weights = class_weights(df.loc[df.split == "train", "y"].to_numpy()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, epochs=epochs, steps_per_epoch=len(train_loader), pct_start=0.2)
    use_amp = amp and device.type in ("cuda", "mps")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    ckpt_path = cfg.checkpoint if tag == "" else cfg.models_dir / f"best_{tag}.pt"
    rows, best_f1, bad_epochs = [], -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item() * len(y)
            n += len(y)
        probs, ys, val_loss = predict_loader(model, val_loader, device)
        pred = probs.argmax(1)
        f1 = f1_score(ys, pred, average="macro")
        rows.append({"epoch": epoch, "train_loss": loss_sum / n, "val_loss": val_loss,
                     "val_acc": float((pred == ys).mean()), "val_macro_f1": f1,
                     "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0})
        print({k: round(v, 4) for k, v in rows[-1].items()})
        if f1 > best_f1:
            best_f1, bad_epochs = f1, 0
            save_checkpoint(ckpt_path, model, arch, cfg.img_size, epoch, f1)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stopping на эпохе {epoch}")
                break

    log = pd.DataFrame(rows)
    suffix = "_smoke" if smoke else (f"_{tag}" if tag else "")
    log.to_csv(cfg.reports_dir / f"train_log{suffix}.csv", index=False)
    plot_curves(log, cfg.reports_dir / f"curves{suffix}.png")
    print(f"Лучший val macro-F1: {best_f1:.4f} -> {ckpt_path}")
    return best_f1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    ap.add_argument("--arch", default=CFG.model_name)
    ap.add_argument("--amp", action="store_true", help="смешанная точность (на CUDA включается всегда)")
    ap.add_argument("--smoke", action="store_true", help="быстрая проверка на маленькой подвыборке")
    ap.add_argument("--tag", default="", help="суффикс чекпоинта (для бейзлайна, напр. resnet50)")
    a = ap.parse_args()
    CFG.ensure_dirs()
    if a.smoke:
        a.tag = "smoke"
    train(CFG, a.arch, a.epochs, a.amp or torch.cuda.is_available(), a.smoke, a.tag)
