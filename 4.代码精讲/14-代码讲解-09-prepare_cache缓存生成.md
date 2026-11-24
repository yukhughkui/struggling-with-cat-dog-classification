# 代码讲解 09 · `prepare_cache.py`：为什么先把 jpg 解码成 .npy

> 位置：链路最前面，一次性预处理。训练前把 25000 张 jpg 解码、缩放、存成大数组。
> 一句话职责：**用 8 个进程把图片解码成 `256×256` 的 uint8 内存映射文件，让训练时不再反复解 JPEG。**

---

## 一、这个文件在链路中的位置

```
Cat_Dog_data/{train,test}/{cat,dog}/*.jpg
        │
        ▼
prepare_cache.py  ← 你在这一篇
        │  只跑一次，约 5.5 分钟
        ▼
project/cache/
    train_256.npy   [22500,256,256,3] uint8  ≈ 4.42 GB
    train_labels.npy
    train_meta.json
    test_256.npy    [ 2500,256,256,3] uint8  ≈ 0.49 GB
    test_labels.npy
    test_meta.json
        │
        ▼
src/data.py  CachedImageDataset 直接按下标读数组
```

---

## 二、为什么要做这一步（这是本项目的性能关键）

如果不做缓存，每个 epoch 都要：

1. 打开 22500 个 jpg 文件；
2. JPEG 解码（一张大图解码可能要几十毫秒）；
3. 缩放、裁剪；
4. 再送进模型。

12 个 epoch 就是 27 万次解码，CPU 根本忙不过来，GPU 大部分时间在等数据。

缓存之后：解码只做一次，训练时直接 `arr[i]` 读一块内存，快几个数量级。

另外两个背景：

- 本机 Windows 下 DataLoader 每开一个子进程都要重新 `import torch`（实测冷启动约 247 秒），所以 `num_workers=0`，只能靠“预解码 + GPU 增强”提速；
- 缓存用 `memmap`（内存映射），4.4GB 的数组不会一次性全读进 16GB 内存，而是用到哪页读哪页。

---

## 三、常量与函数

| 名字 | 干什么 |
|---|---|
| `DATA_ROOT` / `CACHE_DIR` / `SIZE` / `N_PROC` | 数据路径、缓存路径、目标边长 256、进程数 8 |
| `resize_center_crop()` | 等比缩放，让短边到 256，再中心裁 256×256 |
| `scan_split()` | 扫描一个 split，得到 `(路径, 标签)` 列表和类别映射 |
| `fill_chunk()` | 子进程任务：解码一段下标，写进 memmap 的对应区间 |
| `build_split()` | 建一个 split 的缓存文件（已存在则跳过） |
| `main()` | 依次建 train、test |

---

## 四、逐段讲关键代码

### 4.1 `resize_center_crop()`：先缩后裁

```python
scale = size / min(w, h)
nw, nh = int(round(w * scale)), int(round(h * scale))
im = im.resize((nw, nh), Image.BILINEAR)
left, top = (nw - size) // 2, (nh - size) // 2
return im.crop((left, top, left + size, top + size))
```

- 以**短边**为准缩放到 256：比如 4000×3000 的图，短边 3000→256，长边变成约 341，然后中心裁掉两边，得到 256×256。
- 等价于 torchvision 的 `Resize(256) + CenterCrop(256)`。
- **为什么不直接 `resize((256,256))`**：那会把猫压扁/拉长，改变形状。先等比缩放再裁，内容不变形。
- **为什么是 256 而不是 224**：留一点余量。训练时还要随机裁剪到 224，如果缓存就是 224，随机裁剪就没空间了。

### 4.2 `scan_split()`：顺序必须和 ImageFolder 一致

```python
classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
class_to_idx = {c: i for i, c in enumerate(classes)}
files = sorted([p for p in (root / c).iterdir() if p.suffix.lower() in EXTS])
```

- 类别目录按字母序：`cat` 在前、`dog` 在后 → cat=0、dog=1。
- 文件按文件名排序。
- **为什么要和 ImageFolder 一致**：`data.py` 的兜底路径 `FolderImageDataset` 也按这个规则扫。两条路径的标签顺序一致，缓存和原图才能对得上。
- **如果顺序不一致**：缓存里第 100 张的标签是 0，但实际是狗——模型在学“猫=0、狗=1”时会得到矛盾的监督信号，准确率上不去。
- `class_to_idx` 会写进 `train_meta.json`，`data.py` 读它来确认类别映射。

### 4.3 `fill_chunk()`：一个子进程干的活

```python
arr = np.load(arr_path, mmap_mode="r+")
for i in range(start, end):
    with Image.open(path) as im:
        im.draft("RGB", (size, size))
        im = im.convert("RGB")
        im = resize_center_crop(im, size)
        arr[i] = np.asarray(im, dtype=np.uint8)
```

- `mmap_mode="r+"`：以“可写”方式映射文件，但不会把整个文件读进内存。
- `im.draft("RGB", (size, size))`：**JPEG 专用加速**。告诉解码器“我最终只要 256 左右”，JPEG 可以直接按 1/2、1/4、1/8 降采样解码，跳过大部分像素。这是把 4 小时压到 4 分钟的关键之一。
  - **删掉 `draft`**：每张图都完整解码再缩小，速度会慢好几倍。
  - `draft` 对 PNG 等格式无效，只是一个“建议”。
- `convert("RGB")`：统一成三通道。灰度图、带透明通道的图都能处理。
- `arr[i] = ...`：直接写进映射文件的第 i 个位置。

### 4.4 `build_split()`：分片并行写同一个文件

```python
arr = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.uint8,
                                shape=(n, SIZE, SIZE, 3))
bounds = np.linspace(0, n, N_PROC + 1).astype(int)
jobs = [(str(arr_path), samples, int(bounds[i]), int(bounds[i + 1]), SIZE)
        for i in range(N_PROC)]
with Pool(N_PROC) as pool:
    for k, done in enumerate(pool.imap_unordered(fill_chunk, jobs), 1):
        print(...)
```

- 先创建一个“空的大数组文件”，形状 `[n,256,256,3]`，类型 uint8。
- 用 `linspace` 把 22500 切成 8 段（每段约 2812 张），每段一个子进程。
- **每个子进程各自写自己的区间**，互不重叠，所以不需要加锁。`imap_unordered` 表示“谁先完成先汇报”，所以日志里的分段顺序可能是乱的，不影响结果。
- **为什么是 8 个进程**：本机 28 个 CPU 核，但解码是 I/O + CPU 混合，8 个已经能打满磁盘；再多收益递减，还更吃内存。
- **删掉多进程、改成单进程**：实测约 8 张/秒，22500 张要 45 分钟以上。

```python
labels = np.array([lab for _, lab in samples], dtype=np.int64)
np.save(CACHE_DIR / f"{split}_labels.npy", labels)
(cache / f"{split}_meta.json").write_text(json.dumps({...}))
```

- 标签单独存一个小文件，`meta` 存类别映射、数量、尺寸、排序规则。
- `data.py` 读这三个文件来构造 `CachedImageDataset`。

```python
if arr_path.exists():
    print(f"已存在，跳过：{arr_path}")
    return
```

- 已经建过就跳过，所以重复运行 `一键复现.py` 不会重复解码。
- **注意**：只检查 `.npy` 是否存在，不检查 labels/meta 是否齐全。如果上次在“写完数组、还没写 labels”时被杀掉，下次会被误判为“已完成”。真遇到这种情况，手动删除 `project/cache/` 整个目录再跑。

### 4.5 `main()` 与 Windows 保护

```python
if __name__ == "__main__":
    sys.exit(main())
```

- Windows 的 `multiprocessing` 是 `spawn` 模式：子进程会重新 import 主模块。**没有这层保护**，子进程一 import 又启动新的 Pool，无限递归，最后报 `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase`。
- Linux 下默认 `fork`，没这层也能跑；但写上跨平台更安全。

---

## 五、如果你想改，改哪里

| 想做的事 | 改哪里 | 说明 |
|---|---|---|
| 改缓存分辨率 | `SIZE = 256` | 改完要同步改训练输入尺寸和模型展平尺寸，否则形状错 |
| 加快/减慢预处理 | `N_PROC = 8` | 8~12 一般够用 |
| 换数据目录 | `DATA_ROOT` | 目录必须符合 `train/cat` 结构 |
| 换缓存位置 | `CACHE_DIR` | `data.py` 的 `DEFAULT_CACHE` 要一起改 |
| 重新生成缓存 | 删除 `project/cache/` 后运行 | 否则会被“已存在”跳过 |
| 减小磁盘占用 | 把 `SIZE` 降到 224 或改存 float16 | float16 会损失精度，一般 uint8 就够 |
| 把缓存写到别的盘 | 改 `CACHE_DIR` 为绝对路径 | 4.9GB，注意磁盘空间 |

---

## 六、新手易错点

1. **在 Windows 上忘了 `if __name__ == "__main__":`。**
   多进程会无限递归启动子进程，报 `bootstrapping phase` 错误。这是 Windows 多进程的经典坑。

2. **`draft()` 写在 `convert()` 之后。**
   `draft` 必须在真正解码像素之前调用才有效。写在 `convert` 后面等于没加速。

3. **缓存只建了一半就中断，下次却直接跳过。**
   `build_split` 只看 `.npy` 在不在。中断后最好检查 `labels.npy` 和 `meta.json` 是否也在；不确定就删掉整个 cache 目录重建。

4. **磁盘空间不够。**
   训练缓存 4.42GB + 测试缓存 0.49GB，约占 5GB。放缓存前先确认磁盘。

5. **训练正在跑的时候删缓存。**
   内存映射文件被删后，进程可能读到空页或直接崩溃。要重建缓存，先停掉训练。
