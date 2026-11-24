# 代码讲解 03 · `src/engine.py`：一轮训练到底发生了什么

> 位置：链路的核心发动机。`runner.py` 把数据、模型、优化器准备好，交给这里的 `fit()`，它负责“一个 epoch 一个 epoch 地训练”。
> 一句话职责：**训练一个 epoch、评估一次、保存最优权重、按验证 loss 早停。**

---

## 一、这个文件在链路中的位置

```
src/runner.py  run_one()
     │  准备 model / train_loader / val_loader / optimizer / scheduler
     ▼
src/engine.py  fit()                    ← 你在这一篇
     └─ for epoch in 1..N:
           ├─ train_one_epoch()  ← 内层 batch 循环：前向→loss→反向→更新
           └─ evaluate()         ← 验证集：只前向，不算梯度
     │
     ▼
   返回 best.pt + history
```

这一篇是整条链里最值得反复读的一块。看懂 `fit()`，你就看懂了所有深度学习训练代码的骨架。

---

## 二、函数清单

| 函数 | 干什么 |
|---|---|
| `set_seed(seed)` | 固定 Python / NumPy / PyTorch / CUDA 的随机源 |
| `accuracy_from_logits()` | 从 logits 算准确率的小工具（当前没被调用，保留备用） |
| `train_one_epoch()` | 训练一个 epoch，返回平均 loss |
| `evaluate()` | 在验证/测试集上评估，返回 loss、准确率、全部预测 |
| `fit()` | 训练主循环：逐 epoch 训练+验证、存最优、早停 |
| `save_predictions()` | 把预测结果存成 `.npz` |
| `gpu_memory_report()` | 打印显存占用，写进 metrics.json |

---

## 三、逐段讲关键代码

### 3.1 `set_seed()`：为什么每个 run 开头都要设种子

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
```

- 深度学习里有很多随机：参数初始化、数据打乱、Dropout、数据增强。种子固定，这些随机就变成“固定剧本”，别人能复现你的结果。
- `cudnn.benchmark = True`：让 cuDNN 自动挑最快的卷积算法。因为输入尺寸固定（224×224），挑一次就能一直用。
- **删掉会怎样**：同一个配置跑两次结果不一样，没法判断“提升”是改代码带来的还是运气好。

### 3.2 `train_one_epoch()`：训练一个 epoch 的完整流水

```python
model.train()
for images, targets in loader:
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    images = normalize_batch(aug(images))
```

- `model.train()`：把模型切到“训练模式”。Dropout 开始随机丢神经元，BatchNorm 用当前 batch 的均值方差。**删掉**：Dropout 不生效，BN 用旧统计量，训练会异常。
- `for images, targets in loader`：这就是内层 batch 循环，一次 32/48 张。
- `.to(device)`：把 uint8 图从内存搬到显存。`non_blocking=True` 配合 `pin_memory` 做异步拷贝。
- `aug(images)`：在 **GPU** 上做随机裁剪/翻转/旋转/调色，输出 224×224。
- `normalize_batch(...)`：除以 255、减均值、除标准差，变成模型需要的 float。

```python
optimizer.zero_grad(set_to_none=True)
with autocast("cuda", enabled=use_amp):
    logits = model(images)
    loss = criterion(logits, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

这五行是整个项目的心脏，逐行说：

1. **`optimizer.zero_grad()`**：清空上一批留下的梯度。PyTorch 的梯度默认是**累加**的。**删掉**：这一批的梯度会加上前面所有批的，相当于学习率越跑越大，很快就会 loss 爆炸。
   `set_to_none=True` 是更省内存的写法（直接置 None，而不是填 0）。
2. **`autocast`**：AMP（混合精度）——把一部分计算换成 float16，速度快、省显存，同时保留 float32 的关键部分避免精度崩。`enabled=use_amp` 让 CPU 上自动关闭。
3. **`logits = model(images)`**：前向传播，得到 `[B,2]` 的分数。
4. **`loss = criterion(logits, targets)`**：`criterion` 是 `CrossEntropyLoss`，内部先 softmax 再算“正确答案的概率有多低”，输出一个标量。
5. **`scaler.scale(loss).backward()`**：反向传播。AMP 下 loss 会先乘一个缩放系数，防止 float16 梯度过小变成 0。`scaler.step()` 会先还原梯度再更新参数，`scaler.update()` 调整下一轮的缩放系数。**如果不用 AMP**，这三行就等价于 `loss.backward(); optimizer.step()`。

```python
total_loss += loss.item() * bs
total_n += bs
return total_loss / max(total_n, 1)
```

按样本数加权平均。**为什么不是 `losses.append(loss)` 再平均**：最后一个 batch 可能只有 4 张（22500 不能被 32 整除），直接平均 batch 会把小 batch 的权重放大。

### 3.3 `evaluate()`：只前向，不学习

```python
model.eval()
...
with torch.no_grad():
    ...
```

- `model.eval()`：切到评估模式。Dropout 关闭，BatchNorm 用训练时累计的统计量。**删掉**：验证时 Dropout 还在随机丢神经元，同一个模型两次评估结果都不一样，val_acc 会抖。
- `@torch.no_grad()`（装饰在函数上）：不记录计算图。**删掉**：每批的中间结果都留在显存里等反向传播，几批之后就 OOM。
- 返回五个东西：平均 loss、准确率、真实标签、预测标签、预测概率。后三个后面画混淆矩阵和错误样本要用。

```python
probs = torch.softmax(logits.float(), dim=1)
preds = probs.argmax(dim=1)
```

`softmax` 把两个分数变成“概率”，`argmax` 取概率大的那个作为预测类别。

### 3.4 `fit()`：一个 run 的 epoch 循环

```python
for epoch in range(1, epochs + 1):
    train_loss = train_one_epoch(...)
    val_loss, val_acc, _, _, _ = evaluate(...)
    ...
    if scheduler is not None:
        scheduler.step()
```

- 这就是“中层循环”：每个 epoch 先训练一遍，再验证一遍。
- **`scheduler.step()` 放在 epoch 末尾，不是 batch 里**。本项目用余弦退火，学习率从 1e-3 缓慢降到 1e-5。**如果放到 batch 循环里**：一个 epoch 有几百个 batch，学习率几步就降到接近 0，模型学不动。
- **删掉验证**：你就不知道模型什么时候最好，也没法早停。

```python
if val_acc > best_acc + 1e-6:
    best_acc, best_acc_loss = val_acc, val_loss
    torch.save(model.state_dict(), best_path)
```

- 按**验证准确率**保存最优权重。1e-6 是浮点容差，避免“看起来相等”也触发保存。
- 最终测试集评估用的就是这份 `best.pt`，不是最后一个 epoch 的权重。

```python
if val_loss < min_val_loss - 1e-4:
    min_val_loss = val_loss
    bad_epochs = 0
else:
    bad_epochs += 1
if bad_epochs >= patience:
    break
```

- 早停看的是**验证 loss**，不是准确率。原因：准确率是离散的（2250 张里错 1 张差 0.044%），经常原地抖；loss 是连续的，更能反映“是否还在进步”。
- `patience=3` 表示连续 3 轮 loss 没改善就停。
- 注意：**保存最优看 acc、早停看 loss**，两个标准是分开的。所以可能出现“best.pt 来自第 4 轮，但第 9 轮 loss 更低、acc 略低”的情况。这是有意的，代码里注释也写了。

```python
if best_path.exists():
    model.load_state_dict(torch.load(best_path, ...))
```

- 训练结束后把“最优那一轮”的权重装回模型。**删掉**：`run_one` 后面用当前模型再评估验证集，拿到的就是最后一轮（可能已经过拟合）的结果，和 `best_acc` 对不上。

### 3.5 为什么每个 epoch 都重写一遍 `history.csv`

```python
with (out_dir / "history.csv").open("w", ...) as f:
    writer.writerows(history)
```

不是追加，而是把整个 `history` 列表重写一遍。好处是文件永远是完整的、合法的 CSV；即使训练中途被打断，你也保留着前面所有 epoch 的记录。代价是每轮多写几 KB，可以忽略。

### 3.6 `save_predictions()` 与 `gpu_memory_report()`

- `save_predictions()`：把 `y_true / y_pred / y_prob` 存成压缩的 `.npz`，以后不用重新跑模型就能画混淆矩阵、算别的指标。
- `gpu_memory_report()`：返回“当前占用 / 峰值占用”，写进 `metrics.json`，方便判断显存够不够。

---

## 四、如果你想改，改哪里

| 想做的事 | 改哪里 | 说明 |
|---|---|---|
| 改学习率 | `run_experiments.py` 的 `build_specs()` 里 `lr=1e-3` | 也可在 `runner.py` 里按阶段改 |
| 改优化器（SGD、AdamW） | `src/runner.py` 里 `from torch.optim import Adam` 和 `Adam(...)` | 不在本文件；engine 只调用传进来的 optimizer |
| 改学习率调度 | `runner.py` 的 `CosineAnnealingLR` | 想换 StepLR 就改这一行 |
| 改早停耐心 | `run_experiments.py` 里 `patience=...` | 迁移 2，自搭 CNN 3 |
| 换评估指标（F1、AUC） | `evaluate()` 里用 `y_true/y_pred/y_prob` 现算 | 返回值里已经有原始预测 |
| 关掉 AMP | `runner.py` 里 `use_amp=(device.type == "cuda")` 改成 False | 会慢一些，但数值更稳 |
| 加梯度裁剪 | `train_one_epoch()` 里 `scaler.step` 之前加 `torch.nn.utils.clip_grad_norm_` | 训练不稳时用 |
| 改 loss（如加权交叉熵） | `runner.py` 里 `criterion = nn.CrossEntropyLoss()` | 本数据类别平衡，不需要 |
| 每 N 轮才验证一次 | `fit()` 里加 `if epoch % N == 0` | 验证很耗时的时候用 |

---

## 五、新手易错点

1. **忘了 `optimizer.zero_grad()`。**
   梯度会跨 batch 累加，效果等于学习率越来越大，通常几十个 batch 后 loss 变 nan。这是新手第一大坑。

2. **评估时忘了 `model.eval()` 或 `torch.no_grad()`。**
   前者让 Dropout/BN 行为错误，val_acc 抖动；后者让显存持续增长，最后 OOM。两个都要有。

3. **把 `scheduler.step()` 写进 batch 循环。**
   余弦退火是按 epoch 设计的，放进 batch 里学习率会瞬间降到最低，模型几乎不学。

4. **以为最后保存的是“最后一轮”的模型。**
   本项目保存的是验证集最优（`best.pt`），训练结束还会把最优权重装回去。测试集评估用的也是 `best.pt`。

5. **用训练集 loss 判断早停。**
   训练 loss 几乎只会降，用它早停永远不会触发。早停必须看验证集。
