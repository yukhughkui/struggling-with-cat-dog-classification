"""训练/评估引擎：所有实验共用同一套逻辑，保证对比公平。"""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from .data import normalize_batch


def set_seed(seed: int) -> None:
    """固定所有随机源，保证同一配置可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # 输入尺寸固定，开启基准加速


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp: bool, aug) -> float:
    """训练一个 epoch，返回样本平均 loss。

    aug 是运行在 GPU 上的 torchvision v2 增强模块（输入 uint8 张量）。
    """
    model.train()
    total_loss, total_n = 0.0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        images = normalize_batch(aug(images))

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_n += bs
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool = True, eval_tf=None):
    """评估，返回 (样本平均 loss, 准确率, y_true, y_pred, y_prob)。"""
    model.eval()
    total_loss, total_n = 0.0, 0
    y_true, y_pred, y_prob = [], [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        images = normalize_batch(eval_tf(images) if eval_tf is not None else images)
        with autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        probs = torch.softmax(logits.float(), dim=1)
        preds = probs.argmax(dim=1)
        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        y_true.append(targets.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
        y_prob.append(probs.cpu().numpy())

    return (
        total_loss / max(total_n, 1),
        float((np.concatenate(y_true) == np.concatenate(y_pred)).mean()),
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(y_prob),
    )


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    out_dir: str | Path,
    patience: int = 3,
    use_amp: bool = True,
    aug=None,
    eval_tf=None,
    log_fn=print,
):
    """训练主循环：按验证准确率保存最优权重，按验证 loss 早停。

    返回 (best_val_acc, best_val_loss, history, best_path)。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    history: list[dict] = []
    best_acc, best_acc_loss = -1.0, float("inf")
    min_val_loss = float("inf")
    bad_epochs = 0
    scaler = GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                     scaler, use_amp, aug)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device,
                                              use_amp, eval_tf)
        epoch_sec = time.perf_counter() - t0

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "epoch_sec": round(epoch_sec, 3),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        log_fn(
            f"  epoch {epoch:02d}/{epochs} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | {epoch_sec:.1f}s"
        )

        # 以验证准确率为准保存最优权重
        if val_acc > best_acc + 1e-6:
            best_acc, best_acc_loss = val_acc, val_loss
            torch.save(model.state_dict(), best_path)
        # 以验证 loss 为准做早停（单独记录历史最低 loss）
        if val_loss < min_val_loss - 1e-4:
            min_val_loss = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(history)

        if bad_epochs >= patience:
            log_fn(f"  早停：验证 loss 连续 {patience} 轮未改善（epoch {epoch}）")
            break

    # 恢复验证集上最优的权重
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return best_acc, best_acc_loss, history, best_path


def save_predictions(out_path: str | Path, y_true, y_pred, y_prob) -> None:
    np.savez_compressed(out_path, y_true=y_true, y_pred=y_pred, y_prob=y_prob)


def gpu_memory_report(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    alloc = torch.cuda.memory_allocated(device) / 1024**2
    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    return f"显存占用 {alloc:.0f} MB / 峰值 {peak:.0f} MB"
