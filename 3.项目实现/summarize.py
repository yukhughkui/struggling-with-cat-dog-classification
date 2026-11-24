"""汇总所有 run：生成 runs.csv、experiment_summary.csv/md 与对比图。"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体，避免图表标题出现方块
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

EXPERIMENTS = {
    "E1": {
        "title": "E1 数据增强：开 vs 关",
        "groups": {
            "无增强": lambda rid: rid.startswith("cnn_gap_aug_off"),
            "有增强": lambda rid: rid.startswith("cnn_gap_aug_on"),
        },
    },
    "E2": {
        "title": "E2 自搭 CNN vs ResNet18 迁移学习",
        "groups": {
            "自搭 CNN": lambda rid: rid.startswith("cnn_gap_aug_on"),
            "ResNet18 迁移": lambda rid: rid.startswith("resnet_stageB"),
        },
    },
    "E6": {
        "title": "E6 冻结分类头 vs 解冻 layer4 微调",
        "groups": {
            "仅训分类头": lambda rid: rid.startswith("resnet_stageA"),
            "解冻微调": lambda rid: rid.startswith("resnet_stageB"),
        },
    },
    "E7": {
        "title": "E7 轻量化：GAP vs 大 FC",
        "groups": {
            "GAP 轻量版": lambda rid: rid.startswith("cnn_gap_aug_on"),
            "Flatten+大FC": lambda rid: rid.startswith("cnn_heavy"),
        },
    },
}


def load_runs(out_root: Path) -> list[dict]:
    rows = []
    for run_dir in sorted((out_root / "results").iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_p, met_p = run_dir / "config.json", run_dir / "metrics.json"
        if not cfg_p.exists() or not met_p.exists():
            continue
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        met = json.loads(met_p.read_text(encoding="utf-8"))
        test_p = run_dir / "test_metrics.json"
        test = json.loads(test_p.read_text(encoding="utf-8")) if test_p.exists() else {}
        rows.append({
            "run_id": cfg["run_id"],
            "model": cfg["model"],
            "augment": cfg["augment"],
            "stage": cfg.get("stage") or "-",
            "seed": cfg["seed"],
            "epochs": met.get("epochs_run"),
            "params_M": round(met.get("params_total", 0) / 1e6, 3),
            "train_min": round((met.get("train_sec") or 0) / 60, 2),
            "best_val_acc": round(met.get("best_val_acc", 0), 4),
            "test_acc": round(test.get("test_acc", float("nan")), 4),
            "test_loss": round(test.get("test_loss", float("nan")), 4),
        })
    return rows


def write_runs_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    return (float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)


def summarize(rows: list[dict], out_dir: Path) -> None:
    summary_rows = []
    for exp_id, exp in EXPERIMENTS.items():
        for label, match in exp["groups"].items():
            subset = [r for r in rows if match(r["run_id"]) and not np.isnan(r["test_acc"])]
            if not subset:
                continue
            test_m, test_s = mean_std([r["test_acc"] for r in subset])
            val_m, val_s = mean_std([r["best_val_acc"] for r in subset])
            sec_m, sec_s = mean_std([r["train_min"] for r in subset])
            summary_rows.append({
                "experiment": exp_id,
                "variant": label,
                "n_seeds": len(subset),
                "params_M": subset[0]["params_M"],
                "test_acc_mean": round(test_m, 4),
                "test_acc_std": round(test_s, 4),
                "val_acc_mean": round(val_m, 4),
                "val_acc_std": round(val_s, 4),
                "train_min_mean": round(sec_m, 2),
                "train_min_std": round(sec_s, 2),
                "epochs_mean": round(float(np.mean([r["epochs"] for r in subset])), 1),
            })

    csv_path = out_dir / "experiment_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Markdown 表
    lines = ["| 实验 | 配置 | 种子数 | 参数量(M) | 测试准确率 | 验证准确率 | 训练时长(min) | 平均 epoch |",
             "|---|---|---|---|---|---|---|---|"]
    for r in summary_rows:
        lines.append(
            f"| {r['experiment']} | {r['variant']} | {r['n_seeds']} | {r['params_M']} | "
            f"{r['test_acc_mean']*100:.2f}% ± {r['test_acc_std']*100:.2f}% | "
            f"{r['val_acc_mean']*100:.2f}% ± {r['val_acc_std']*100:.2f}% | "
            f"{r['train_min_mean']:.1f} ± {r['train_min_std']:.1f} | {r['epochs_mean']} |"
        )
    (out_dir / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 每个实验一张对比图：测试准确率（含误差棒）+ 训练时长
    for exp_id, exp in EXPERIMENTS.items():
        subset = [r for r in summary_rows if r["experiment"] == exp_id]
        if len(subset) < 2:
            continue
        labels = [r["variant"] for r in subset]
        accs = [r["test_acc_mean"] * 100 for r in subset]
        errs = [r["test_acc_std"] * 100 for r in subset]
        secs = [r["train_min_mean"] for r in subset]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(labels, accs, yerr=errs, capsize=6, color=["#7aa6c2", "#e0a06a"])
        axes[0].set_ylabel("Test accuracy (%)")
        axes[0].set_ylim(max(0, min(accs) - 8), 100)
        for i, v in enumerate(accs):
            axes[0].text(i, v + 1, f"{v:.2f}", ha="center", fontsize=9)
        axes[0].grid(axis="y", alpha=.3)
        axes[0].set_title("Accuracy (mean ± std)")

        axes[1].bar(labels, secs, color=["#7aa6c2", "#e0a06a"])
        axes[1].set_ylabel("Training time (min)")
        for i, v in enumerate(secs):
            axes[1].text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)
        axes[1].grid(axis="y", alpha=.3)
        axes[1].set_title("Training cost")
        fig.suptitle(exp["title"])
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_{exp_id}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(Path(__file__).resolve().parent.parent / "outputs"))
    args = parser.parse_args()
    out_root = Path(args.out_root)
    out_dir = out_root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_runs(out_root)
    if not rows:
        raise SystemExit("没有找到任何 run 结果")
    write_runs_csv(rows, out_dir / "runs.csv")
    summarize(rows, out_dir)
    print(f"已生成：{out_dir / 'runs.csv'}、experiment_summary.csv/.md、fig_E*.png")


if __name__ == "__main__":
    main()
