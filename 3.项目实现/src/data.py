"""数据管道：ImageFolder / 内存缓存 + 90:10 划分。

设计说明：
- 数据增强放在 GPU 上做（torchvision v2 支持 CUDA 张量），因为本机 CPU 单进程
  逐图做 RandomRotation/ColorJitter 非常慢；
- 优先读取 prepare_cache.py 生成的 256×256 uint8 内存映射缓存，避免每个 epoch
  重复解码 JPEG；
- 训练集与验证/测试集使用不同的处理：训练随机增强，验证/测试只做中心裁剪。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import v2

# ImageNet 统计量：与预训练权重对齐；自搭 CNN 也用同一套，保证 E2 对比只有一个变量
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "cache"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_train_aug(size: int = 224, augment: bool = True):
    """返回在 GPU 上执行的训练增强模块（输入 uint8 CHW 张量）。"""
    if augment:
        return v2.Compose([
            v2.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=12),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        ])
    # 对照组：确定性中心裁剪
    return v2.Compose([v2.CenterCrop(size)])


def build_eval_transform(size: int = 224):
    """验证/测试：只做中心裁剪，不带任何随机性。"""
    return v2.Compose([v2.CenterCrop(size)])


def normalize_batch(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [B,3,H,W] -> float32 标准化张量。"""
    x = x_uint8.float().div_(255.0)
    mean = torch.tensor(MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


class CachedImageDataset(Dataset):
    """读取 uint8 内存映射缓存，返回 [3,256,256] uint8 张量。"""

    def __init__(self, arr_path: str | Path, labels_path: str | Path):
        self.arr = np.load(arr_path, mmap_mode="r")
        self.labels = np.load(labels_path)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        arr = np.array(self.arr[i])                        # 复制，避免只读 memmap 警告
        img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return img, int(self.labels[i])


class FolderImageDataset(Dataset):
    """缓存不存在时的兜底：逐张解码并统一到 256×256。"""

    def __init__(self, root: str | Path, size: int = 256):
        root = Path(root)
        classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples: list[tuple[Path, int]] = []
        for c in classes:
            files = sorted([p for p in (root / c).iterdir() if p.suffix.lower() in EXTS])
            self.samples.extend((p, self.class_to_idx[c]) for p in files)
        self.size = size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = self.size / min(w, h)
            im = im.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
            nw, nh = im.size
            left, top = (nw - self.size) // 2, (nh - self.size) // 2
            im = im.crop((left, top, left + self.size, top + self.size))
            arr = np.asarray(im, dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), int(label)


def split_indices(n: int, valid_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """先整体打乱再切分，避免按路径顺序切出“纯 cat / 纯 dog”的验证集。"""
    idx = np.random.RandomState(seed).permutation(n)
    n_val = int(round(n * valid_size))
    return idx[n_val:], idx[:n_val]


def make_loaders(
    data_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    valid_size: float = 0.1,
    cache_dir: str | Path | None = None,
):
    """返回 train/val/test 三个 DataLoader 与类别映射。

    每个 batch 的形状为 images [B,3,256,256] uint8、labels [B]。
    具体的增强/裁剪在训练循环里按设备执行。
    """
    root = Path(data_root)
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    train_arr = cache / "train_256.npy"
    test_arr = cache / "test_256.npy"

    if train_arr.exists() and test_arr.exists() and (cache / "train_meta.json").exists():
        class_to_idx = json.loads((cache / "train_meta.json").read_text(encoding="utf-8"))["class_to_idx"]
        train_ds = CachedImageDataset(train_arr, cache / "train_labels.npy")
        val_ds = CachedImageDataset(train_arr, cache / "train_labels.npy")
        test_ds = CachedImageDataset(test_arr, cache / "test_labels.npy")
    else:
        train_dir, test_dir = root / "train", root / "test"
        if not train_dir.is_dir() or not test_dir.is_dir():
            raise FileNotFoundError(f"数据目录不符合 ImageFolder 结构：{root}")
        train_ds = FolderImageDataset(train_dir)
        val_ds = FolderImageDataset(train_dir)
        test_ds = FolderImageDataset(test_dir)
        class_to_idx = train_ds.class_to_idx

    train_idx, val_idx = split_indices(len(train_ds), valid_size, seed)
    train_set = Subset(train_ds, train_idx.tolist())
    val_set = Subset(val_ds, val_idx.tolist())

    common = dict(num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0,
                  prefetch_factor=6 if num_workers > 0 else None)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader, class_to_idx


def dump_split(n: int, seed: int, valid_size: float, out_path: str | Path) -> None:
    """把划分规模与抽样索引落盘，便于复现与审计。"""
    train_idx, val_idx = split_indices(n, valid_size, seed)
    payload = {
        "seed": seed,
        "valid_size": valid_size,
        "n_total": n,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "train_idx_head": train_idx[:20].tolist(),
        "val_idx_head": val_idx[:20].tolist(),
    }
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
