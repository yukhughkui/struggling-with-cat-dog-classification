r"""数据就绪自检：数量 / 类别 / 尺寸分布 / 损坏文件。

用法： python check_data.py "d:\DeepStudy\Project1\Cat_Dog_data"
"""
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

EXPECT = {"train": {"cat": 11250, "dog": 11250},
          "test": {"cat": 1250, "dog": 1250}}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan(root: Path):
    problems, stats = [], {}
    for split in ("train", "test"):
        for cls in ("cat", "dog"):
            d = root / split / cls
            if not d.is_dir():
                problems.append(f"[缺失目录] {d}")
                continue
            files = [p for p in d.iterdir() if p.suffix.lower() in EXTS]
            others = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() not in EXTS]
            if others:
                problems.append(f"[非图片文件] {d} 下有 {len(others)} 个，例如 {others[0].name}")
            sizes, bad = [], []
            for p in files:
                try:
                    with Image.open(p) as im:
                        im.verify()
                    with Image.open(p) as im:
                        sizes.append(im.size)  # (宽, 高)
                except Exception as e:  # 损坏文件只记录，不中断
                    bad.append((p.name, str(e)))
            if bad:
                problems.append(f"[损坏/无法读取] {d} 有 {len(bad)} 张，例如 {bad[:3]}")
            expect = EXPECT[split][cls]
            if len(files) != expect:
                problems.append(f"[数量不符] {d}: 实际 {len(files)}，期望 {expect}")
            if sizes:
                ws = sorted(s[0] for s in sizes)
                hs = sorted(s[1] for s in sizes)
                med = lambda a: a[len(a) // 2]  # noqa: E731
                ratios = Counter(round(w / h, 2) for w, h in sizes)
                stats[f"{split}/{cls}"] = {
                    "count": len(files),
                    "w_min/med/max": (ws[0], med(ws), ws[-1]),
                    "h_min/med/max": (hs[0], med(hs), hs[-1]),
                    "aspect_top3": ratios.most_common(3),
                }
    return stats, problems


def main(root_str: str):
    root = Path(root_str)
    if not root.is_dir():
        print(f"根目录不存在：{root}")
        return
    stats, problems = scan(root)
    print("=" * 68)
    for k, v in stats.items():
        print(f"{k:12s} 数量={v['count']:6d}  宽 min/中位/max={v['w_min/med/max']}  "
              f"高 min/中位/max={v['h_min/med/max']}")
        print(f"{'':12s} 长宽比 top3={v['aspect_top3']}")
    print("=" * 68)
    total = sum(v["count"] for v in stats.values())
    n_train = sum(v["count"] for k, v in stats.items() if k.startswith("train"))
    n_test = sum(v["count"] for k, v in stats.items() if k.startswith("test"))
    print(f"图片总数：{total}；训练:测试 = {n_train}:{n_test}")
    if problems:
        print("\n发现问题：")
        for p in problems:
            print(" -", p)
    else:
        print("\n检查通过：目录结构、数量、文件完整性均正常。")
    print("\n注意：尺寸不统一是正常的，由 transforms.Resize + CenterCrop 统一到 224×224。")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"d:\DeepStudy\Project1\Cat_Dog_data")
