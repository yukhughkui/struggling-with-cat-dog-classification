# 代码讲解 08 · `check_data.py`：训练前的体检

> 位置：链路最前面，一次性工具。训练前确认“数据没缺、没坏、目录结构对”。
> 一句话职责：**逐张打开 25000 张图，统计数量/尺寸，报告缺目录、数量不符、损坏文件。**

---

## 一、这个文件在链路中的位置

```
Cat_Dog_data/  (jpg)
        │
        ▼
check_data.py  ← 你在这一篇
        │  只读、不改数据、不生成文件
        ▼
prepare_cache.py → 之后才是训练
```

它不产出任何文件，只往屏幕上打印一份“体检报告”。`一键复现.py` 的第 1 步就是它。

---

## 二、常量与函数

| 名字 | 干什么 |
|---|---|
| `EXPECT` | 期望的数量：train 每类 11250，test 每类 1250 |
| `EXTS` | 认哪些后缀是图片 |
| `scan(root)` | 扫描四个目录，返回统计信息和问题列表 |
| `main(root_str)` | 打印统计、总数、问题；全好则打印“检查通过” |

---

## 三、逐段讲关键代码

### 3.1 期望值写死

```python
EXPECT = {"train": {"cat": 11250, "dog": 11250},
          "test":  {"cat": 1250,  "dog": 1250}}
```

- Kaggle 的 Dogs vs Cats 就是这个规模，训练集猫狗各 11250（**完全平衡**），测试集各 1250。
- 写死的好处：少一张图都能发现。坏处：换数据集要改这里，否则满屏“数量不符”。
- **删掉数量检查**：少几百张图也能正常训练，但你不会知道，报告里的“22500 张”就是假的。

### 3.2 扫描：四个目录、两类问题

```python
for split in ("train", "test"):
    for cls in ("cat", "dog"):
        d = root / split / cls
        if not d.is_dir():
            problems.append(f"[缺失目录] {d}")
            continue
```

- 目录结构必须是 `train/cat`、`train/dog`、`test/cat`、`test/dog`。这是 `ImageFolder` 的约定。
- 缺一个目录就记一条问题并跳过——**不中断**，这样一次能看到所有问题，而不是改一个跑一次。

```python
files  = [p for p in d.iterdir() if p.suffix.lower() in EXTS]
others = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() not in EXTS]
```

- `.suffix.lower()` 让 `.JPG` 也能识别。
- 目录里混进 `.txt`、`.DS_Store` 之类会单独报出来。**不检查**：训练时它们会被当成图片去解码，直接报错。

```python
with Image.open(p) as im:
    im.verify()
with Image.open(p) as im:
    sizes.append(im.size)
```

这是本文件最值得学的一个细节：

- `Image.open()` 只是“打开文件头”，并不会把像素读进来（懒加载）。
- `im.verify()` 会检查文件完整性，但**调用后这个对象就作废了**，不能再读尺寸。
- 所以必须**第二次 `Image.open`** 才能拿到 `(宽, 高)`。
- **删掉 `verify()`**：损坏的图检查不出来，要等到训练时 `__getitem__` 才炸，而且报错位置离真正原因很远。
- **只 open 一次然后 verify + size**：会抛异常或拿到空尺寸。

```python
except Exception as e:
    bad.append((p.name, str(e)))
```

- 坏图只记录、不中断。一张坏图不值得让整个体检失败，但你必须知道它存在。

### 3.3 统计尺寸与长宽比

```python
ws = sorted(s[0] for s in sizes)
med = lambda a: a[len(a) // 2]
ratios = Counter(round(w / h, 2) for w, h in sizes)
stats[...] = {"count": ..., "w_min/med/max": (ws[0], med(ws), ws[-1]), ...}
```

- 数据集的图尺寸**不统一**是正常的，这里只是让你心里有数。
- 中位数比平均值更抗极端值：一张 10000×10000 的图会把平均值拉飞，但不影响中位数。
- 长宽比取两位小数后统计 top3，能看出大部分图是横图还是方图。
- 这些统计**不影响训练**。训练时统一由 `Resize + CenterCrop`（缓存阶段是缩放到 256 再中心裁）处理。
- **删掉 `sorted`**：中位数就取错了，统计没意义。

### 3.4 输出结论

```python
print(f"图片总数：{total}；训练:测试 = {n_train}:{n_test}")
if problems:
    print("\n发现问题：")
    ...
else:
    print("\n检查通过：目录结构、数量、文件完整性均正常。")
```

- 最后一行“检查通过”是 `一键复现.py` 判断成功的依据（它在输出里找这四个字）。
- 注意：**根目录不存在时，`main` 只打印一行然后 `return`，退出码仍是 0**。所以不能只看退出码，要看输出内容。
- **删掉“检查通过”这句**：一键脚本会误判为失败（或需要改判断逻辑）。

---

## 四、如果你想改，改哪里

| 想做的事 | 改哪里 | 说明 |
|---|---|---|
| 换数据集 | `EXPECT` 的数量、`EXTS` | 目录结构不变的话只改 EXPECT |
| 换数据路径 | 命令行参数 `python check_data.py "路径"` | 默认是 `d:\DeepStudy\Project1\Cat_Dog_data` |
| 加一项检查（如重复文件） | `scan()` 里加逻辑，往 `problems` 里 append | 保持“只记录不中断”的风格 |
| 检查 EXIF 旋转 | `Image.open` 后读 `im.getexif()` | 手机照片常见，本项目数据没有 |
| 把报告写到文件 | `main()` 里加 `open(..., "w")` | 当前只打印 |
| 只检查不校验完整性（更快） | 删掉 `im.verify()` 那次 open | 2.5 万张图能快不少 |

---

## 五、新手易错点

1. **`verify()` 之后继续用同一个对象读尺寸。**
   会报错或拿到空值。必须重新 `Image.open`。代码里两次 open 不是冗余，是必需的。

2. **只看退出码判断体检是否通过。**
   根目录不存在时函数只是 `return`，退出码仍是 0。要看输出里有没有“检查通过”。

3. **以为尺寸不统一是数据有问题。**
   Kaggle 猫狗数据本来就什么尺寸都有，训练管道会统一处理。真正要担心的是损坏文件、数量不符、目录缺失。

4. **每次训练前都跑一遍。**
   它要逐张打开 2.5 万张图两次，比较慢。数据没动过就不用重复跑；`一键复现.py` 每次都会跑，是为了让流程完整、可自查。
