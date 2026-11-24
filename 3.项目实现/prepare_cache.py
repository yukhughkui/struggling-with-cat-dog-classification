"""把图片一次性解码并缩放到 256×256，存成 uint8 内存映射缓存。

背景：本机冷启动 import torch 很慢；DataLoader 多进程在 Windows 上又会为每个
worker 重新 import torch，启动成本极高。因此先把 25000 张图解码一次落盘，
训练时直接读数组，并把数据增强搬到 GPU 上做。

加速手段：
- JPEG 用 draft() 让解码器直接按 1/2、1/4、1/8 降采样，避免解码超大原图；
- 多进程分片写入同一个 memmap（各写各的区间，互不冲突）。

用法： python prepare_cache.py
产物： project/cache/{train,test}_256.npy、*_labels.npy、*_meta.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

DATA_ROOT = Path(r"d:\DeepStudy\Project1\Cat_Dog_data")
CACHE_DIR = Path(__file__).resolve().parent / "cache"
SIZE = 256
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
N_PROC = 8


def resize_center_crop(im: Image.Image, size: int) -> Image.Image:
    w, h = im.size
    scale = size / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    im = im.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - size) // 2, (nh - size) // 2
    return im.crop((left, top, left + size, top + size))


def scan_split(split: str) -> tuple[list[tuple[str, int]], dict[str, int]]:
    """按 ImageFolder 的规则扫描：类目录按字母序，文件按文件名排序。"""
    root = DATA_ROOT / split
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples: list[tuple[str, int]] = []
    for c in classes:
        files = sorted([p for p in (root / c).iterdir() if p.suffix.lower() in EXTS])
        samples.extend((str(p), class_to_idx[c]) for p in files)
    return samples, class_to_idx


def fill_chunk(job) -> int:
    arr_path, samples, start, end, size = job
    arr = np.load(arr_path, mmap_mode="r+")
    done = 0
    for i in range(start, end):
        path, _ = samples[i]
        with Image.open(path) as im:
            im.draft("RGB", (size, size))  # JPEG 直接按比例降采样，显著加速
            im = im.convert("RGB")
            im = resize_center_crop(im, size)
            arr[i] = np.asarray(im, dtype=np.uint8)
        done += 1
    arr.flush()
    return done


def build_split(split: str) -> None:
    arr_path = CACHE_DIR / f"{split}_{SIZE}.npy"
    if arr_path.exists():
        print(f"已存在，跳过：{arr_path}")
        return

    samples, class_to_idx = scan_split(split)
    n = len(samples)
    print(f"{split}: {n} 张，类别映射 {class_to_idx}", flush=True)

    arr = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.uint8, shape=(n, SIZE, SIZE, 3))
    arr.flush()
    del arr

    t0 = time.time()
    bounds = np.linspace(0, n, N_PROC + 1).astype(int)
    jobs = [(str(arr_path), samples, int(bounds[i]), int(bounds[i + 1]), SIZE) for i in range(N_PROC)]
    with Pool(N_PROC) as pool:
        for k, done in enumerate(pool.imap_unordered(fill_chunk, jobs), 1):
            print(f"  {split}: 第 {k}/{N_PROC} 片完成（本片 {done} 张，累计用时 {time.time()-t0:.0f}s）", flush=True)

    labels = np.array([lab for _, lab in samples], dtype=np.int64)
    np.save(CACHE_DIR / f"{split}_labels.npy", labels)
    (CACHE_DIR / f"{split}_meta.json").write_text(
        json.dumps({"class_to_idx": class_to_idx, "n": n, "size": SIZE,
                    "order": "类目录字母序 + 文件名排序（与 ImageFolder 一致）"},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"完成 {split}：{n} 张，用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        build_split(split)
    print("缓存就绪：", CACHE_DIR)


if __name__ == "__main__":
    # Windows spawn 下必须放在 __main__ 保护里
    sys.exit(main())
