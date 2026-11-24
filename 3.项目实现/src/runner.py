"""单次实验运行器：构建数据/模型/优化器，训练并落盘。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import build_eval_transform, build_train_aug, make_loaders
from .engine import evaluate, fit, gpu_memory_report, save_predictions, set_seed
from .models import build_custom_cnn, build_resnet18, count_params, unfreeze_layer4


class _LimitedLoader:
    """可重复迭代的“只取前 n 个 batch”包装（冒烟测试用）。

    注意不能用一次性生成器：fit 每个 epoch 都会重新迭代验证集。
    """

    def __init__(self, loader, n: int | None):
        self.loader = loader
        self.n = n

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if self.n is not None and i >= self.n:
                break
            yield batch


def build_model(spec: dict) -> nn.Module:
    name = spec["model"]
    if name == "cnn":
        return build_custom_cnn(use_gap=True, width_mult=spec.get("width_mult", 1.0),
                                use_bn=spec.get("use_bn", True))
    if name == "cnn_heavy":
        return build_custom_cnn(use_gap=False, width_mult=spec.get("width_mult", 1.0),
                                use_bn=spec.get("use_bn", True))
    if name == "resnet18":
        return build_resnet18(freeze_backbone=(spec.get("stage") == "A"))
    raise ValueError(f"未知模型：{name}")


def run_one(spec: dict, data_root: str, out_root: str, num_workers: int = 0,
            limit_batches: int | None = None, verbose: bool = True) -> dict:
    run_id = spec["run_id"]
    out_dir = Path(out_root) / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(spec["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = print if verbose else (lambda *a, **k: None)
    log(f"\n===== {run_id} =====")
    log(f"config: {json.dumps(spec, ensure_ascii=False)}")

    train_loader, val_loader, test_loader, class_to_idx = make_loaders(
        data_root=data_root,
        batch_size=spec["batch_size"],
        num_workers=num_workers,
        seed=spec.get("split_seed", 42),
        valid_size=spec.get("valid_size", 0.1),
    )
    aug = build_train_aug(224, augment=spec["augment"])
    eval_tf = build_eval_transform(224)

    model = build_model(spec)

    # 阶段 B：先载入阶段 A 的最优权重，再解冻 layer4
    if spec.get("stage") == "B":
        init_from = Path(spec["init_from"])
        if not init_from.exists():
            raise FileNotFoundError(f"阶段 A 权重不存在：{init_from}")
        state = torch.load(init_from, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        unfreeze_layer4(model)

    model.to(device)
    (out_dir / "model_structure.txt").write_text(str(model), encoding="utf-8")

    criterion = nn.CrossEntropyLoss()

    if spec.get("stage") == "B":
        # 分层学习率：backbone 小学习率，分类头正常学习率
        lr = spec["lr"]
        optimizer = Adam([
            {"params": model.layer4.parameters(), "lr": lr * 0.1},
            {"params": model.fc.parameters(), "lr": lr},
        ])
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = Adam(params, lr=spec["lr"])

    scheduler = CosineAnnealingLR(optimizer, T_max=max(spec["epochs"], 1), eta_min=spec["lr"] * 0.01)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    best_acc, best_loss, history, best_path = fit(
        model,
        _LimitedLoader(train_loader, limit_batches),
        _LimitedLoader(val_loader, limit_batches),
        criterion,
        optimizer,
        scheduler,
        device,
        epochs=spec["epochs"],
        out_dir=out_dir,
        patience=spec.get("patience", 3),
        use_amp=(device.type == "cuda"),
        aug=aug,
        eval_tf=eval_tf,
        log_fn=log,
    )
    train_sec = time.perf_counter() - t0

    # 验证集最优结果（测试集留到 final_eval 统一评估）
    val_loss, val_acc, y_true, y_pred, y_prob = evaluate(
        model, _LimitedLoader(val_loader, limit_batches), criterion, device,
        use_amp=(device.type == "cuda"), eval_tf=eval_tf,
    )
    save_predictions(out_dir / "val_preds.npz", y_true, y_pred, y_prob)

    metrics = {
        "run_id": run_id,
        "best_val_acc": best_acc,
        "best_val_loss": best_loss,
        "final_val_acc": val_acc,
        "final_val_loss": val_loss,
        "train_sec": round(train_sec, 1),
        "epochs_run": len(history),
        "params_total": count_params(model),
        "params_trainable": count_params(model, trainable_only=True),
        "gpu": gpu_memory_report(device),
        "class_to_idx": class_to_idx,
        "config": spec,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"完成：best_val_acc={best_acc:.4f}，用时 {train_sec/60:.1f} 分钟，"
        f"参数量 {metrics['params_total']/1e6:.2f}M")
    return metrics


def evaluate_on_test(spec: dict, data_root: str, out_root: str, num_workers: int = 0) -> dict:
    """最终测试集评估：只对已训练好的 best.pt 做一次前向，不参与任何调参。"""
    from .plots import plot_confusion, plot_curves, plot_error_grid

    run_id = spec["run_id"]
    out_dir = Path(out_root) / "results" / run_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(spec)
    state = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    _, _, test_loader, class_to_idx = make_loaders(
        data_root=data_root,
        batch_size=spec["batch_size"],
        num_workers=num_workers,
        seed=spec.get("split_seed", 42),
        valid_size=spec.get("valid_size", 0.1),
    )
    eval_tf = build_eval_transform(224)
    criterion = nn.CrossEntropyLoss()
    loss, acc, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion, device,
                                                 use_amp=(device.type == "cuda"), eval_tf=eval_tf)
    save_predictions(out_dir / "test_preds.npz", y_true, y_pred, y_prob)

    class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    plot_confusion(y_true, y_pred, class_names, out_dir / "test_cm.png", title=run_id)
    history = json.loads((out_dir / "history.json").read_text(encoding="utf-8"))
    plot_curves(history, out_dir / "curves.png", title=run_id)
    n_err = plot_error_grid(model, test_loader, device, class_names,
                            out_dir / "test_errors.png", k=15, eval_tf=eval_tf)

    metrics = {
        "run_id": run_id,
        "test_loss": loss,
        "test_acc": acc,
        "n_test": int(len(y_true)),
        "n_errors_shown": n_err,
        "params_total": count_params(model),
        "train_sec": json.loads((out_dir / "metrics.json").read_text(encoding="utf-8")).get("train_sec"),
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
