# 代码讲解 05 · `src/plots.py`：成绩单是怎么画出来的

> 位置：链路的出口之一。训练和评估算完数字后，这个文件负责把数字变成人看得懂的图。
> 一句话职责：**画三张图——loss/准确率曲线、混淆矩阵、置信度最高的错误样本。**

---

## 一、这个文件在链路中的位置

```
engine.evaluate()  →  一堆数字：loss、acc、y_true、y_pred、y_prob
        │
        ▼
src/plots.py  ← 你在这一篇
    plot_curves()       用 history.csv 画训练曲线
    plot_confusion()    用 y_true / y_pred 画 2×2 混淆矩阵
    plot_error_grid()   把分错的图挑出来拼成一张大图
        │
        ▼
outputs/results/<run_id>/curves.png、test_cm.png、test_errors.png
```

它**不参与**训练和评分，只负责展示。删掉它，实验照样能跑完，只是没有图可看。但报告和 PPT 都依赖这些图，所以它很重要。

---

## 二、函数清单

| 函数 | 输入 | 输出 |
|---|---|---|
| `plot_curves()` | 每个 epoch 的 history 列表 | `curves.png`（左 loss，右 val_acc） |
| `plot_confusion()` | 真实标签、预测标签、类别名 | `test_cm.png`，同时返回 2×2 矩阵 |
| `plot_error_grid()` | 模型、测试 loader、设备 | `test_errors.png`，返回画了几张 |

---

## 三、逐段讲关键代码

### 3.1 开头三行：为什么必须先设 `Agg`

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

- `Agg` 是“只画到文件、不弹窗口”的后端。
- **不设会怎样**：在没有图形界面的环境（服务器、某些 IDE 终端、后台任务）里，matplotlib 可能报 `TclError` 或卡住。
- 顺序不能反：必须在 `import pyplot` **之前**调用 `matplotlib.use`，否则设置无效。

```python
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
```

- 第一行让中文标题能正常显示。Windows 自带“微软雅黑”，所以第一优先。
- 第二行让负号正常显示。不设的话，坐标轴上的负号会变成方块。
- **如果图里中文全是方块**：说明当前系统找不到前两个字体，需要装中文字体，或在列表里加一个已安装的字体名。

### 3.2 `plot_curves()`：两条曲线，一眼看训练状态

```python
axes[0].plot(epochs, [h["train_loss"] for h in history], marker="o", ms=3, label="train")
axes[0].plot(epochs, [h["val_loss"] for h in history], marker="s", ms=3, label="val")
...
axes[1].plot(epochs, [h["val_acc"] for h in history], ...)
```

- 左图：训练 loss 和验证 loss。两条都下降是健康的；训练降、验证升是过拟合。
- 右图：验证准确率。看它有没有早停、最优在第几轮。
- `history` 就是 `engine.fit()` 里每轮追加的那个列表，字段有 `epoch / train_loss / val_loss / val_acc / epoch_sec / lr`。
- **删掉 `fig.savefig`**：图画在内存里，什么都没留下。
- **删掉 `plt.close(fig)`**：每画一张图就多占一份内存和文件句柄。15 个 run 连续跑，matplotlib 会警告“More than 20 figures have been opened”。

### 3.3 `plot_confusion()`：错在哪，一目了然

```python
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
```

- `confusion_matrix` 来自 sklearn：行是真实类别，列是预测类别。
- 本项目结果是 `[[1237, 13], [18, 1232]]`：
  - 左上 1237：真猫判成猫；
  - 右上 13：真猫判成狗（漏判）；
  - 左下 18：真狗判成猫（误报）；
  - 右下 1232：真狗判成狗。
- `labels=range(len(class_names))` 保证矩阵的行列顺序和 `class_names` 一致。**不传 labels**：如果某一类一张都没预测到，矩阵会少一行/列，和类别名对不上。
- `values_format="d"`：格子里显示整数，不显示科学计数法。
- 返回值 `cm` 虽然当前没被调用方使用，但留着方便你在 notebook 里直接算精确率、召回率。

### 3.4 `plot_error_grid()`：模型最“自信地错”的图

```python
@torch.no_grad()
def plot_error_grid(model, loader, device, class_names, out_path, k=15, eval_tf=None):
```

- `@torch.no_grad()`：只做推理，不建计算图，省显存。
- 遍历整个测试集，把所有分错的样本收集起来：

```python
for i in range(images.size(0)):
    if preds[i].item() != targets[i].item():
        errors.append((x[i].cpu(), int(targets[i]), int(preds[i]), float(confs[i])))
```

- 存的是 `x[i].cpu()`，即**已经在 GPU 上裁到 224 的图**，不是原图。所以图上看到的是“模型实际看到的画面”。
- `.cpu()` 很重要：如果把 GPU 张量攒在列表里，2500 张图的显存会一直涨，可能 OOM。
- `confs` 是 softmax 后的最大概率，代表“模型有多确定”。

```python
errors.sort(key=lambda t: t[3], reverse=True)
errors = errors[:k]
```

- 按置信度**从高到低**排序，只取前 k 张。
- **为什么按置信度排**：模型 99% 确定是狗、结果是猫，这种错误最能说明问题（可能是标签错、也可能是模型抓住了错误特征）。随机挑 15 张反而看不出规律。
- **不排序**：图里就是随机错误，分析价值低很多。

```python
ncols = 5
nrows = int(np.ceil(len(errors) / ncols))
...
ax.imshow(img.permute(1, 2, 0).numpy())
ax.set_title(f"P={class_names[pred_i]} T={class_names[true_i]} {conf:.2f}", color="red")
```

- 每行 5 张，自动算行数。
- `permute(1,2,0)`：把 CHW 转回 HWC，`imshow` 才能显示。**不 permute**：会报“Invalid shape for image data”。
- 标题里 `P=` 是预测（Predicted），`T=` 是真实（True），后面的数字是置信度。

```python
if not errors:
    return 0
```

- 如果一张都没错（理论上可能），直接返回，避免画一张空图时报错。

---

## 四、如果你想改，改哪里

| 想做的事 | 改哪里 | 说明 |
|---|---|---|
| 改错误样本张数 | `runner.evaluate_on_test()` 里的 `k=15` | 也可以改函数默认值 |
| 改每行几张 | `plot_error_grid()` 里的 `ncols = 5` | 图会自动变高 |
| 改图的清晰度 | 各函数 `fig.savefig(..., dpi=150)` | 报告用可以调到 200 |
| 改图的大小 | `figsize=(11,4)` / `(4.6,4.2)` | 单位是英寸 |
| 加一张新图（如每类准确率） | 在 `plots.py` 加函数，再到 `runner.evaluate_on_test()` 调用 | 两处配套 |
| 改混淆矩阵配色 | `disp.plot(..., cmap="Blues")` | 可换 `Greens`、`Oranges` |
| 改中文标题 | 各函数的 `title` 参数 | 调用处传 `run_id`，想改成中文就在这里传 |
| 让图显示准确率数值 | `plot_curves` 里加 `ax.text` | 参考 `summarize.py` 的柱状图写法 |
| 换中文字体 | 文件开头 `rcParams["font.sans-serif"]` | 换成你系统里有的字体 |

---

## 五、新手易错点

1. **`matplotlib.use("Agg")` 写在 `import pyplot` 之后。**
   等于没设，仍然可能弹窗或报错。必须放在最前面，且只调用一次。

2. **`imshow` 前忘了 `permute`。**
   PyTorch 是 `[C,H,W]`，matplotlib 要 `[H,W,C]`。不改就会报形状错误，或者画出一张颜色错乱的图。

3. **把 GPU 张量存进列表。**
   `errors.append((x[i], ...))` 不 `.cpu()`，测试集跑完显存会被这些中间图占满。本项目显式写了 `.cpu()`。

4. **混淆矩阵的类别顺序和标题对不上。**
   `class_names` 必须按标签 0、1 的顺序排（本项目是 `sorted(class_to_idx.items(), key=lambda kv: kv[1])`）。顺序反了，“猫狗”会标反。

5. **每张图不 `close`。**
   在循环里画 15 个 run 的图，不关闭会累积 figure，最后内存告警、图也可能画崩。
