"""四组对比实验的统一入口。

用法示例：
  # 冒烟测试（1 epoch、每轮 3 个 batch）
  python run_experiments.py --smoke

  # 正式实验（默认 3 个种子）
  python run_experiments.py

  # 只跑迁移学习两组阶段
  python run_experiments.py --only transfer
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 允许以 `python run_experiments.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.runner import evaluate_on_test, run_one  # noqa: E402

DEFAULT_DATA = r"d:\DeepStudy\Project1\Cat_Dog_data"
DEFAULT_OUT = str(Path(__file__).resolve().parent.parent / "outputs")


def build_specs(args) -> list[dict]:
    seeds = [args.seeds[0]] if args.smoke else args.seeds
    epochs_cnn = 1 if args.smoke else args.epochs_cnn
    epochs_heavy = 1 if args.smoke else args.epochs_heavy
    epochs_a = 1 if args.smoke else args.epochs_a
    epochs_b = 1 if args.smoke else args.epochs_b
    bs = args.batch_size
    patience = 0 if args.smoke else None

    groups = {"transfer", "e1", "e7"} if not args.only else set(args.only)
    base = dict(batch_size=bs, valid_size=0.1, split_seed=42)

    specs: list[dict] = []
    # 迁移学习（先跑，风险前置）
    if "transfer" in groups:
        for seed in seeds:
            specs.append(dict(base, run_id=f"resnet_stageA_s{seed}", model="resnet18",
                              augment=True, seed=seed, epochs=epochs_a, lr=1e-3,
                              stage="A", patience=patience or 2))
            specs.append(dict(base, run_id=f"resnet_stageB_s{seed}", model="resnet18",
                              augment=True, seed=seed, epochs=epochs_b, lr=1e-3,
                              stage="B", patience=patience or 2,
                              init_from=str(Path(args.out_root) / "results" /
                                            f"resnet_stageA_s{seed}" / "best.pt")))
    # E1 / E2 / E7 的自搭 CNN
    if "e1" in groups:
        for seed in seeds:
            specs.append(dict(base, run_id=f"cnn_gap_aug_on_s{seed}", model="cnn",
                              augment=True, seed=seed, epochs=epochs_cnn, lr=1e-3,
                              patience=patience or 3))
            specs.append(dict(base, run_id=f"cnn_gap_aug_off_s{seed}", model="cnn",
                              augment=False, seed=seed, epochs=epochs_cnn, lr=1e-3,
                              patience=patience or 3))
    if "e7" in groups:
        for seed in seeds:
            specs.append(dict(base, run_id=f"cnn_heavy_aug_on_s{seed}", model="cnn_heavy",
                              augment=True, seed=seed, epochs=epochs_heavy, lr=1e-3,
                              patience=patience or 3))
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    parser.add_argument("--out-root", default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs-cnn", type=int, default=20)
    parser.add_argument("--epochs-heavy", type=int, default=12)
    parser.add_argument("--epochs-a", type=int, default=6)
    parser.add_argument("--epochs-b", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Windows 上 spawn 每个 worker 都要重新 import torch，默认 0")
    parser.add_argument("--only", nargs="+", default=None,
                        help="可选：transfer / e1 / e7")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true", help="已存在结果也重跑")
    parser.add_argument("--skip-final-eval", action="store_true")
    args = parser.parse_args()

    args.out_root = str(Path(args.out_root))
    Path(args.out_root, "results").mkdir(parents=True, exist_ok=True)
    if not (Path(__file__).resolve().parent / "cache" / "train_256.npy").exists():
        print("提示：未找到 256px 缓存，将逐张解码 JPEG（很慢）。建议先运行：python prepare_cache.py")

    specs = build_specs(args)
    limit = 3 if args.smoke else None
    print(f"共 {len(specs)} 个 run，输出目录：{args.out_root}")

    t_all = time.perf_counter()
    done: list[dict] = []
    for spec in specs:
        metrics_path = Path(args.out_root) / "results" / spec["run_id"] / "metrics.json"
        if metrics_path.exists() and not args.force and not args.smoke:
            print(f"跳过已完成：{spec['run_id']}")
            done.append(json.loads(metrics_path.read_text(encoding="utf-8")))
            continue
        done.append(run_one(spec, args.data_root, args.out_root,
                            num_workers=args.num_workers, limit_batches=limit))

    print(f"\n训练阶段总耗时：{(time.perf_counter() - t_all) / 60:.1f} 分钟")

    if not args.skip_final_eval:
        print("\n===== 最终测试集评估 =====")
        for spec in specs:
            metrics_path = Path(args.out_root) / "results" / spec["run_id"] / "test_metrics.json"
            if metrics_path.exists() and not args.force and not args.smoke:
                print(f"跳过已完成：{spec['run_id']}")
                continue
            m = evaluate_on_test(spec, args.data_root, args.out_root, num_workers=args.num_workers)
            print(f"  {spec['run_id']}: test_acc={m['test_acc']:.4f}")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
