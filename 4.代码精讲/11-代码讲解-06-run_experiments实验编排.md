# 代码讲解 06 · `run_experiments.py`：15 个 run 的总调度

> 位置：整条链路的最外层。它不做任何模型计算，只负责“生成 15 份配置，逐个交给 runner”。
> 一句话职责：**定义所有实验（spec）、解析命令行参数、循环跑 run、最后统一评估测试集。**

---

## 一、这个文件在链路中的位置

```
命令行 / 一键复现.py
        │
        ▼
run_experiments.py  ← 你在这一篇
    build_specs()   生成 15 个 spec 字典
    main()
      └─ for spec in specs:  ← 外层循环
             runner.run_one(spec, ...)
      └─ for spec in specs:  ← 第二轮：最终测试
             runner.evaluate_on_test(spec, ...)
        │
        ▼
outputs/results/<run_id>/...
```

`engine.py` 是“一个 epoch 怎么跑”，这个文件是“**哪些实验、按什么顺序跑**”。

---

## 二、函数清单

| 函数 | 干什么 |
|---|---|
| `build_specs(args)` | 根据参数生成 15 个 spec（迁移 6 + E1 6 + E7 3） |
| `main()` | 解析参数、循环训练、循环最终评估 |

---

## 三、逐段讲关键代码

### 3.1 两个默认路径

```python
DEFAULT_DATA = r"d:\DeepStudy\Project1\Cat_Dog_data"
DEFAULT_OUT  = str(Path(__file__).resolve().parent.parent / "outputs")
```

- `DEFAULT_OUT` 用 `__file__` 往上找两级，所以无论你在哪个目录运行，输出都落在 `.../cnn-notebook-walkthrough/outputs`。
- 用 `Path` 而不是字符串拼接，跨平台更稳。

### 3.2 `build_specs()`：15 个 spec 是怎么拼出来的

```python
base = dict(batch_size=bs, valid_size=0.1, split_seed=42)
```

`base` 是所有实验共享的部分：90:10 划分、划分种子固定 42。**这就是“控制变量”**：15 个 run 用同一套训练/验证划分，比出来的差异才可信。

```python
if "transfer" in groups:
    for seed in seeds:
        specs.append(dict(base, run_id=f"resnet_stageA_s{seed}", model="resnet18",
                          augment=True, seed=seed, epochs=epochs_a, lr=1e-3,
                          stage="A", patience=patience or 2))
        specs.append(dict(base, run_id=f"resnet_stageB_s{seed}", model="resnet18",
                          ...
                          stage="B", patience=patience or 2,
                          init_from=str(Path(args.out_root) / "results" /
                                        f"resnet_stageA_s{seed}" / "best.pt")))
```

- 每个种子生成两个 spec：阶段 A（冻结 backbone）和阶段 B（解冻 layer4 微调）。
- `init_from` 指向**同一输出目录下**阶段 A 的 `best.pt`。所以 `--out-root` 一改，读档路径也跟着改。
- `dict(base, ...)` 是“复制 base 再覆盖字段”，不会改到 base 本身。

```python
if "e1" in groups:
    specs.append(dict(base, run_id=f"cnn_gap_aug_on_s{seed}",  model="cnn", augment=True, ...))
    specs.append(dict(base, run_id=f"cnn_gap_aug_off_s{seed}", model="cnn", augment=False, ...))

if "e7" in groups:
    specs.append(dict(base, run_id=f"cnn_heavy_aug_on_s{seed}", model="cnn_heavy", augment=True, ...))
```

- E1：同一个 GAP 模型，只差 `augment=True/False`。
- E7：`cnn`（GAP 头）vs `cnn_heavy`（Flatten+大 FC 头），都开增强。
- `model` 字符串就是 `runner.build_model()` 里的分支名。

**spec 到底长什么样**：可以在任意 `outputs/results/<run_id>/config.json` 里看到完整内容，例如：

```json
{"batch_size": 48, "valid_size": 0.1, "split_seed": 42,
 "run_id": "cnn_gap_aug_on_s42", "model": "cnn", "augment": true,
 "seed": 42, "epochs": 12, "lr": 0.001, "patience": 3}
```

**删掉 `base` 里的 `split_seed`**：每个 run 会用自己的训练种子去切分数据，15 个 run 的训练/验证集都不一样，E1/E2 的对比就混入了“数据划分不同”这个变量，结论不可信。

### 3.3 参数解析：默认值和 README 的差异要留意

```python
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
parser.add_argument("--epochs-cnn", type=int, default=20)
parser.add_argument("--epochs-heavy", type=int, default=12)
parser.add_argument("--epochs-a", type=int, default=6)
parser.add_argument("--epochs-b", type=int, default=8)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--only", nargs="+", default=None)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--force", action="store_true")
```

- `--only` 接一个列表，比如 `--only transfer` 或 `--only e1 e7`；不传就是全跑。
- **`--epochs-cnn` 默认是 20**，而本项目报告用的是 12（README 的命令里显式传了 `--epochs-cnn 12`）。如果你只敲 `python run_experiments.py`，自搭 CNN 会跑 20 轮，结果和报告里的数字不可比。`一键复现.py` 已经帮你传了 12。
- `--num-workers` 默认 0 是 Windows 的无奈之举（见 `data.py` 篇）。

### 3.4 `main()`：外层循环

```python
args.out_root = str(Path(args.out_root))
Path(args.out_root, "results").mkdir(parents=True, exist_ok=True)
if not (Path(__file__).resolve().parent / "cache" / "train_256.npy").exists():
    print("提示：未找到 256px 缓存，将逐张解码 JPEG（很慢）。建议先运行：python prepare_cache.py")
```

- 先建输出目录；再检查缓存，没有只给提示、不强制退出（因为 `data.py` 有 `FolderImageDataset` 兜底）。

```python
for spec in specs:
    metrics_path = Path(args.out_root) / "results" / spec["run_id"] / "metrics.json"
    if metrics_path.exists() and not args.force and not args.smoke:
        print(f"跳过已完成：{spec['run_id']}")
        continue
    done.append(run_one(spec, args.data_root, args.out_root, ...))
```

这就是**外层 15 次循环**，也是“断点续跑”的实现：

- 判断标准很朴素：`metrics.json` 存在 = 这个 run 训练完成了。
- 不想重跑就**不要加 `--force`**；想覆盖已有结果就加 `--force`。
- `--smoke` 时永远不跳过，保证能测到。

```python
if not args.skip_final_eval:
    print("\n===== 最终测试集评估 =====")
    for spec in specs:
        if metrics_path.exists() and not args.force and not args.smoke:
            continue
        m = evaluate_on_test(spec, ...)
```

**为什么训练和测试分成两轮循环**：

1. 先确保 15 个 run 全部训练完，再统一评估；
2. 测试集只在前向时用一次，不参与任何训练决策；
3. 阶段 B 依赖阶段 A 的权重，分两轮也避免顺序问题。

**把测试评估塞进训练循环会怎样**：每个 run 刚训完就立刻看测试集，容易忍不住“根据测试成绩再调一调”，测试集就失去“独立考试”的意义了。

### 3.5 `--smoke` 和 `--skip-final-eval`

- `--smoke`：只跑第 1 个种子、1 个 epoch、每个 loader 只取 3 个 batch，几十秒验证代码没写坏。
- `--skip-final-eval`：只训练、不做测试评估，适合“先看看曲线”。
- **注意**：smoke 用的 run_id 和正式实验相同（例如 `resnet_stageA_s42`），会**覆盖**该目录下的正式结果。已经有正式结果后不要再跑 smoke。

---

## 四、如果你想改，改哪里

| 想做的事 | 改哪里 | 说明 |
|---|---|---|
| 改跑的种子 | 命令行 `--seeds 42 43` 或这里 `default=[...]` | 种子越多，均值越可靠，但更慢 |
| 改自搭 CNN 轮数 | 命令行 `--epochs-cnn 12` | 默认 20，别忘传 |
| 改迁移两阶段轮数 | `--epochs-a` / `--epochs-b` | 默认 6 / 8 |
| 改 batch | `--batch-size`（全局） | 迁移用 32、CNN 用 48 是分两次命令传的 |
| 改学习率 | `build_specs()` 里的 `lr=1e-3` | 所有 run 一起变 |
| 改早停耐心 | `build_specs()` 里的 `patience or 2/3` | 迁移 2，CNN 3 |
| 加一组新实验（如 E3 学习率） | `build_specs()` 加分支 + `summarize.py` 的 `EXPERIMENTS` 加一组 | 两处配套，否则汇总里看不到 |
| 换数据/输出目录 | `--data-root` / `--out-root` | 输出会整体搬到新目录 |
| 只跑某一组 | `--only transfer` / `--only e1 e7` | 可以多个值 |
| 强制重跑 | `--force` | 覆盖已有结果 |

---

## 五、新手易错点

1. **不传 `--epochs-cnn 12` 直接跑。**
   默认是 20 轮，自搭 CNN 会跑得更久，结果和报告里的 12 轮不可比。复现时严格按 README 的命令，或用 `一键复现.py`。

2. **在已有正式结果后跑 `--smoke`。**
   smoke 的 run_id 和正式实验相同（如 `resnet_stageA_s42`），会把那个目录的 `best.pt`、`metrics.json` 覆盖成“只跑 3 个 batch”的垃圾结果。smoke 只在代码刚改完、还没有正式结果时用。

3. **以为不加 `--force` 就会重跑。**
   不会。`metrics.json` 存在就跳过。改了代码想验证效果，必须加 `--force`，或者删掉对应 run 目录。

4. **`--only` 拼错名字。**
   可选值是 `transfer`、`e1`、`e7`，写错不会报错，只会一个 run 都不生成（`groups` 里匹配不到）。跑完看到“共 0 个 run”就是拼错了。

5. **只训练不测试，然后去 `summarize.py` 找测试成绩。**
   如果用了 `--skip-final-eval`，`test_metrics.json` 不存在，汇总表里的 `test_acc` 会是 `nan`。要么补跑最终评估，要么别跳过。
