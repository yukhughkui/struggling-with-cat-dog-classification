# 代码讲解 04 · `src/runner.py`：把零件组装成一个 run

> 位置：中间层。它上面是 `run_experiments.py`（调度 15 个 run），下面调用 `data.py / models.py / engine.py / plots.py`。
> 一句话职责：**拿到一个 spec 字典，走完“建数据 → 建模型 → 训练 → 验证 → 最终测试 → 存结果”的全流程。**

---

## 一、这个文件在链路中的位置

```
run_experiments.py（外层：15 个 run）
        │  传进来一个 spec，例如
        │  {"run_id": "resnet_stageB_s42", "model": "resnet18", "stage": "B", ...}
        ▼
src/runner.py  run_one()                ← 你在这一篇
        ├─ data.make_loaders()          拿 train/val/test
        ├─ models.build_*()             造模型
        ├─ engine.fit()                 训练 + 验证 + 早停
        └─ 写 metrics.json / config.json
        │
        ▼（训练全部结束后）
src/runner.py  evaluate_on_test()
        ├─ 载入 best.pt
        ├─ engine.evaluate()            测试集前向
        ├─ plots.plot_*()               混淆矩阵 / 曲线 / 错误样本
        └─ 写 test_metrics.json
```

如果把整个项目比作一家餐厅：`data.py` 是采购，`models.py` 是菜谱，`engine.py` 是厨师，`runner.py` 是**店长**——负责按订单把采购、菜谱、厨师串起来，最后把菜装盘（存结果）。

---

## 二、函数/类清单

| 名字 | 类型 | 干什么 |
|---|---|---|
| `_LimitedLoader` | 类 | 只取前 n 个 batch 的包装器（冒烟测试用） |
| `build_model()` | 函数 | 按 `spec["model"]` 造对应模型 |
| `run_one()` | 函数 | 跑完一个 run 的训练 + 验证，落盘 |
| `evaluate_on_test()` | 函数 | 用 best.pt 在测试集上做最终评估并出图 |

---

## 三、逐段讲关键代码

### 3.1 `_LimitedLoader`：一个踩过坑的小工具

```python
class _LimitedLoader:
    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if self.n is not None and i >= self.n:
                break
            yield batch
```

- 用途：`--smoke` 冒烟测试时只跑 3 个 batch，几十秒验证整条链路通不通。
- **为什么写成类而不是生成器函数**：如果写成 `(batch for i, batch in enumerate(loader) if i < n)` 这样的一次性生成器，第一个 epoch 用完后它就空了；第二个 epoch 验证集拿到 0 个样本，`np.concatenate` 直接报错。这个是真实踩过的 bug，注释里也写了。
- 正常训练时 `limit_batches=None`，它只是原样转发所有 batch，几乎没有额外开销。

### 3.2 `build_model()`：spec 里的字符串怎么变成模型

```python
if name == "cnn":
    return build_custom_cnn(use_gap=True, ...)
if name == "cnn_heavy":
    return build_custom_cnn(use_gap=False, ...)
if name == "resnet18":
    return build_resnet18(freeze_backbone=(spec.get("stage") == "A"))
```

- `spec["model"]` 只是一个字符串，这里负责翻译成真正的网络。
- **阶段 A 冻结主干**（`freeze_backbone=True`），阶段 B 不冻结（后面再手动解冻 layer4）。
- `spec.get("width_mult", 1.0)`、`spec.get("use_bn", True)`：如果 spec 里没写就用默认值。当前 `build_specs` 没传这两个，所以全是默认。
- **想加新模型，必须在这里加分支**，否则 spec 写了也没用。

### 3.3 `run_one()`：一个 run 的完整生命周期

#### 第一步：准备输出目录和随机种子

```python
out_dir = Path(out_root) / "results" / run_id
out_dir.mkdir(parents=True, exist_ok=True)
set_seed(spec["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- 每个 run 一个独立目录，互不覆盖。
- 种子在**建模型之前**设好，保证参数初始化也可复现。

#### 第二步：数据

```python
train_loader, val_loader, test_loader, class_to_idx = make_loaders(
    data_root=data_root,
    batch_size=spec["batch_size"],
    num_workers=num_workers,
    seed=spec.get("split_seed", 42),
    valid_size=spec.get("valid_size", 0.1),
)
```

- 注意 `seed` 用的是 `split_seed`（固定 42），不是训练种子。**这样 15 个 run 用的是同一套训练/验证划分**，对比才公平。
- 训练种子只影响参数初始化、数据打乱、增强的随机。

```python
aug = build_train_aug(224, augment=spec["augment"])
eval_tf = build_eval_transform(224)
```

- 训练用 `aug`（可能带随机增强），验证/测试用 `eval_tf`（只中心裁剪）。
- **这里写死了 224**。如果改输入尺寸，这里和 `models.py` 的 `14*14` 要一起改。

#### 第三步：模型，以及阶段 B 的“读档”

```python
model = build_model(spec)

if spec.get("stage") == "B":
    init_from = Path(spec["init_from"])
    if not init_from.exists():
        raise FileNotFoundError(f"阶段 A 权重不存在：{init_from}")
    state = torch.load(init_from, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    unfreeze_layer4(model)
```

这是迁移学习两阶段的关键：

1. `build_resnet18(freeze_backbone=False)` 造出一个“主干可训练、头已换成二分类”的模型；
2. 把阶段 A 训好的权重**整份读进来**（包括分类头）；
3. 再解冻 layer4。

**删掉第 2 步会怎样**：阶段 B 从随机初始化的分类头开始，等于重新学，E6（冻结 vs 微调）就失去意义。
**删掉第 3 步会怎样**：阶段 B 和阶段 A 完全一样，准确率不会提升。
`weights_only=True` 是安全加载，只允许权重张量，防止加载恶意 pickle。

#### 第四步：优化器和学习率调度

```python
if spec.get("stage") == "B":
    optimizer = Adam([
        {"params": model.layer4.parameters(), "lr": lr * 0.1},
        {"params": model.fc.parameters(),     "lr": lr},
    ])
else:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=spec["lr"])
```

- 阶段 B 用**分层学习率**：layer4 是预训练好的，只轻轻调（0.1 倍学习率）；分类头是新初始化的，正常学。
- **如果统一用 1e-3 学 layer4**：预训练特征会被快速“洗掉”（灾难性遗忘），效果反而比只训头差。
- 阶段 A 的 `if p.requires_grad` 很重要：只把可训练参数交给优化器。**不筛会怎样**：虽然冻结参数的梯度是 None，Adam 不会更新它们，但优化器状态会多占内存，日志里的参数组也不干净。

```python
scheduler = CosineAnnealingLR(optimizer, T_max=max(spec["epochs"], 1),
                              eta_min=spec["lr"] * 0.01)
```

- 余弦退火：学习率从 `lr` 沿余弦曲线降到 `eta_min`。前中期大步走，后期小步收尾。
- `T_max` 必须等于总 epoch 数，否则曲线对不上。

#### 第五步：训练

```python
best_acc, best_loss, history, best_path = fit(
    model, _LimitedLoader(train_loader, limit_batches),
    _LimitedLoader(val_loader, limit_batches),
    ...
    patience=spec.get("patience", 3),
    use_amp=(device.type == "cuda"),
    aug=aug, eval_tf=eval_tf,
)
```

- 训练细节全在 `engine.fit()`，这里只负责把零件递进去。
- `use_amp` 只在有 GPU 时开。

#### 第六步：用最优权重再验证一次，写 metrics

```python
val_loss, val_acc, y_true, y_pred, y_prob = evaluate(model, val_loader, ...)
save_predictions(out_dir / "val_preds.npz", ...)
```

- `fit()` 结束时已经把 best.pt 装回模型，所以这次评估拿到的是“最优那一轮”的验证结果。
- 存 `val_preds.npz` 方便以后分析，不需要重新跑模型。

```python
metrics = {"run_id": ..., "best_val_acc": ..., "train_sec": ..., "params_total": ..., ...}
(out_dir / "metrics.json").write_text(...)
(out_dir / "config.json").write_text(...)
```

- `metrics.json` 是“这次跑得怎么样”，`config.json` 是“这次是怎么跑的”。
- `summarize.py` 就是读这两个文件加 `test_metrics.json` 来汇总的。**字段名别随便改**，否则汇总脚本读不到。

### 3.4 `evaluate_on_test()`：测试集只跑这一次

```python
model = build_model(spec)
state = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
```

- 重新造一个结构相同的模型，载入最优权重。
- 注意这里 `build_model(spec)` 对阶段 B 会造“不冻结”的版本，但结构完全一样，载入权重没问题。
- **为什么不直接复用训练时的模型对象**：因为最终评估是训练全部结束后单独跑的（`run_experiments.py` 的第二段循环），进程里没有那个对象了。从磁盘读 `best.pt` 更干净，也保证“测试用的就是保存下来的最优权重”。

```python
_, _, test_loader, class_to_idx = make_loaders(...)
loss, acc, y_true, y_pred, y_prob = evaluate(model, test_loader, ...)
```

- 测试集 2500 张，batch 大小和训练时一致（迁移 32、自搭 CNN 48）。
- 只做前向，不更新任何参数。

```python
plot_confusion(y_true, y_pred, class_names, out_dir / "test_cm.png", title=run_id)
history = json.loads((out_dir / "history.json").read_text(...))
plot_curves(history, out_dir / "curves.png", title=run_id)
n_err = plot_error_grid(model, test_loader, device, class_names,
                        out_dir / "test_errors.png", k=15, eval_tf=eval_tf)
```

- 三张图：混淆矩阵、训练曲线、置信度最高的 15 张错图。
- `curves.png` 其实只依赖 `history.json`，不需要模型；放在这里是为了“一个 run 的产物集中生成”。
- **删掉 `evaluate_on_test`**：你只有验证集成绩，没有最终测试成绩；`summarize.py` 里 `test_acc` 全是 nan，汇总表是空的。

---

## 四、如果你想改，改哪里

| 想做的事 | 改哪里 | 说明 |
|---|---|---|
| 加一个新模型 | `build_model()` 加分支 + `models.py` 加构造函数 + `run_experiments.py` 加 spec | 三处配套 |
| 改学习率 | 不改这里，改 `run_experiments.py` 的 `lr=1e-3` | spec 会传进来 |
| 改阶段 B 的分层倍率 | 这里 `lr * 0.1` | 0.1 表示 layer4 只轻轻调 |
| 换优化器 / 调度器 | 这里 `Adam(...)`、`CosineAnnealingLR(...)` | engine 不关心具体是哪个 |
| 改早停耐心 | `run_experiments.py` 的 `patience` | 这里只是读取 |
| 改输入尺寸 | 这里两处 `224` + `models.py` 的 `14*14` | 要一起改 |
| 改最终测试的 batch | `make_loaders(batch_size=spec["batch_size"])` | 一般不用动 |
| 多存一个指标（如 F1） | `metrics` 字典里加字段 | 记得同步改 `summarize.py` |
| 换错误样本张数 | `evaluate_on_test` 里 `k=15` | 图会变高 |
| 关掉 AMP | `use_amp=(device.type == "cuda")` 改 False | 慢一些，更稳 |

---

## 五、新手易错点

1. **单独跑阶段 B。**
   阶段 B 依赖阶段 A 的 `best.pt`，文件不存在会直接 `FileNotFoundError`。正确顺序一定是先 A 后 B；`run_experiments.py` 的 spec 列表也是按这个顺序生成的。

2. **在 `run_one` 里对测试集调参。**
   测试集只在 `evaluate_on_test` 里用一次，而且是在所有训练决策（早停、选最优）都做完之后。如果你在训练循环里偷看测试集，最终成绩就不可信了，评分也会被扣。

3. **改了 `metrics.json` 的字段名却忘了改 `summarize.py`。**
   汇总脚本按 `best_val_acc`、`train_sec`、`params_total` 这些键读取。改名后汇总会报 KeyError 或显示 0。

4. **以为 `evaluate_on_test` 会重新训练。**
   它只做前向。如果 `best.pt` 是旧的，测试成绩就是旧的；改了训练代码后必须重跑训练，测试才会更新。

5. **阶段 B 忘了解冻就怪“迁移学习没用”。**
   `unfreeze_layer4` 没执行时，阶段 B 只是在阶段 A 的基础上又训了几轮冻结模型，准确率几乎不变。看 `metrics.json` 里的 `params_trainable` 能确认：阶段 A 约 1026，阶段 B 约 9.4M。
