"""评估出图：损失/准确率曲线、混淆矩阵、错误样本网格。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 服务器/无界面环境也能出图
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .data import normalize_batch

# 中文字体，避免中文标题出现方块
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_curves(history: list[dict], out_path: str | Path, title: str = "") -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], marker="o", ms=3, label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], marker="s", ms=3, label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[0].set_title("Loss")

    axes[1].plot(epochs, [h["val_acc"] for h in history], marker="o", ms=3, color="tab:green")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val accuracy"); axes[1].grid(alpha=.3)
    axes[1].set_title("Validation accuracy")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, class_names, out_path: str | Path, title: str = "") -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


@torch.no_grad()
def plot_error_grid(model, loader, device, class_names, out_path: str | Path, k: int = 15,
                    eval_tf=None) -> int:
    """按“预测置信度从高到低”挑出错误样本并可视化。"""
    from torch.amp import autocast

    model.eval()
    errors: list[tuple[torch.Tensor, int, int, float]] = []
    for images, targets in loader:
        x = images.to(device, non_blocking=True)
        x = eval_tf(x) if eval_tf is not None else x
        with autocast("cuda", enabled=device.type == "cuda"):
            logits = model(normalize_batch(x))
        probs = torch.softmax(logits.float(), dim=1).cpu()
        preds = probs.argmax(dim=1)
        confs = probs.max(dim=1).values
        for i in range(images.size(0)):
            if preds[i].item() != targets[i].item():
                errors.append((x[i].cpu(), int(targets[i]), int(preds[i]), float(confs[i])))
    errors.sort(key=lambda t: t[3], reverse=True)
    errors = errors[:k]

    if not errors:
        return 0

    ncols = 5
    nrows = int(np.ceil(len(errors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (img, true_i, pred_i, conf) in zip(axes, errors):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(f"P={class_names[pred_i]} T={class_names[true_i]} {conf:.2f}",
                     color="red", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Top-{len(errors)} highest-confidence errors", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return len(errors)
