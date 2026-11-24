# 代码讲解 02 · `src/models.py`：模型长什么样

> 位置：链路中游。`data.py` 把图送进来，模型负责把 `[B,3,224,224]` 变成 `[B,2]` 的分数。
> 一句话职责：**定义两个模型——自搭 CNN（两种分类头）和 ResNet18 迁移学习——以及冻结/解冻、数参数的小工具。**

---

## 一、这个文件在链路中的位置

```
src/data.py  →  [B,3,224,224] 标准化后的 float 张量
        │
        ▼
src/models.py  ← 你在这一篇
    CustomCNN.forward / ResNet18.forward
        │
        ▼
        [B,2] logits（猫的分数、狗的分数）
        │
        ▼
src/engine.py  →  CrossEntropyLoss(logits, 标签) → 反向传播
```

这个文件**只描述网络结构**，不负责训练、不负责读数据、不负责保存权重。它像一个“图纸”，`runner.py` 按图纸造出模型，`engine.py` 负责训练它。

---

## 二、函数/类清单

| 名字 | 类型 | 干什么 |
|---|---|---|
| `ConvBlock` | 类 | 一个卷积块：若干 3×3 卷积 + BN + ReLU，最后 2×2 池化 |
| `CustomCNN` | 类 | 自搭 CNN：4 个卷积块 + 3 层全连接，GAP/大 FC 两种头 |
| `build_custom_cnn()` | 函数 | 按 `use_gap`、`width_mult` 造自搭 CNN |
| `build_resnet18()` | 函数 | 造 ImageNet 预训练 ResNet18，换二分类头，可冻结主干 |
| `unfreeze_layer4()` | 函数 | 阶段 B：解冻 layer4 和分类头 |
| `count_params()` | 函数 | 数参数量（可只数可训练的） |

---

## 三、逐段讲关键代码

### 3.1 `ConvBlock`：为什么卷积后面跟 BN，且不要 bias

```python
layers.append(nn.Conv2d(c_in, out_ch, kernel_size=3, padding=1, bias=not use_bn))
if use_bn:
    layers.append(nn.BatchNorm2d(out_ch))
layers.append(nn.ReLU(inplace=True))
```

- **`kernel_size=3, padding=1`**：3×3 卷积配 padding=1，输出的高宽和输入一样（224→224）。**删掉 padding**：每过一层卷积，边长少 2，4 个块下来图会越来越小，形状对不上。
- **`bias=not use_bn`**：卷积后面紧跟 BatchNorm 时，bias 是多余的——BN 会先减均值再加一个可学习的 β，bias 的作用被完全吸收。关掉 bias 省一点参数，也不影响效果。
- **BatchNorm2d**：把每个通道的数值拉回均值 0、方差 1，再学一个缩放和偏移。大白话：让每层输入别忽大忽小，训练更稳、可以用更大的学习率。**删掉 BN**：收敛变慢，学习率 1e-3 可能直接训崩；这时 conv 的 `bias` 也必须打开。
- **`ReLU(inplace=True)`**：负数变 0，提供非线性。`inplace=True` 表示直接改原张量、省内存。
- **`MaxPool2d(2,2)`**：每 2×2 取最大值，高宽各减半。这是“池化”，作用是缩小特征图、扩大感受野、减少计算量。

一个 `ConvBlock` 里 `n_conv=2` 就是“两层卷积 + 一次池化”。整个自搭 CNN 的 7 个卷积是这样凑出来的：2+2+2+1。

### 3.2 `CustomCNN` 的主干：4 个块，尺寸一路减半

```python
self.b1 = ConvBlock(3, c1, n_conv=2)
self.b2 = ConvBlock(c1, c2, n_conv=2)
self.b3 = ConvBlock(c2, c3, n_conv=2)
self.b4 = ConvBlock(c3, c4, n_conv=1)
```

`channels=(32,64,128,256)`，通道数翻倍、高宽减半：

| 经过 | 输出形状 |
|---|---|
| 输入 | `[B,3,224,224]` |
| b1（2 卷积 + 池化） | `[B,32,112,112]` |
| b2 | `[B,64,56,56]` |
| b3 | `[B,128,28,28]` |
| b4 | `[B,256,14,14]` |

**为什么越深通道越多、尺寸越小**：空间信息浓缩成“更抽象、更多种”的特征。这是 CNN 的通用设计套路。

### 3.3 分类头：GAP 还是展平，这是 E7 的实验变量

```python
self.gap = nn.AdaptiveAvgPool2d(1)
in_features = c4 if use_gap else c4 * 14 * 14
...
self.fc1 = nn.Linear(in_features, head_hidden)
self.fc2 = nn.Linear(head_hidden, head_hidden // 2)
self.fc3 = nn.Linear(head_hidden // 2, num_classes)
```

- **GAP（全局平均池化）**：`AdaptiveAvgPool2d(1)` 把 `[B,256,14,14]` 每个通道求平均，变成 `[B,256,1,1]`，再拉平成 `[B,256]`。意思是“这张图里每种特征平均有多强”，位置信息被丢掉了，但参数量极小。
- **大 FC 对照组**：不池化，直接 `flatten` 成 `[B, 256×14×14] = [B, 50176]`，再接到 512 维全连接。
  - 第一层就是 `50176 × 512 ≈ 2570 万` 参数，整个模型 **26.407M**；
  - GAP 版只有 **0.848M**，相差约 31 倍。
  - E7 的结论就是：参数少 31 倍的 GAP 版，测试准确率反而略高、方差更小。

**`c4 * 14 * 14` 里的 14 是怎么来的**：224 经过 4 次池化 → 224/2/2/2/2 = 14。这个数字是**写死**的，和输入 224 强绑定。

**删掉 GAP 会怎样**：就变成大 FC 版，参数暴涨、更容易过拟合、显存占用更大。E7 的对照组就是这么来的，所以不能简单说“删掉”，而是“这是一组对照实验”。

### 3.4 全连接头的细节

```python
self.dropout1 = nn.Dropout(dropout)
self.fc1 = nn.Linear(in_features, head_hidden)
self.bn_fc1 = nn.BatchNorm1d(head_hidden) if use_bn else nn.Identity()
```

- **Dropout(0.5)**：训练时随机把一半神经元的输出置 0，逼模型不要依赖某几个神经元。**只在训练时生效**，`model.eval()` 后自动关闭。
- **BatchNorm1d**：对全连接层输出做同样的“拉回标准分布”。`use_bn=False` 时换成 `nn.Identity()`——一个“什么都不做”的层，这样 `forward` 里不用写 if。
- **三层 FC 的尺寸**：`in_features → 512 → 256 → 2`。最后一层输出 2 个数，对应猫、狗。

**删掉 Dropout**：训练集准确率会明显高于验证集（过拟合），尤其大 FC 版。

### 3.5 `forward`：一条数据在模型里怎么走

```python
x = self.b1(x)   # [B,3,224,224]  -> [B,32,112,112]
...
x = self.gap(x)
x = torch.flatten(x, 1)
x = self.dropout1(x)
x = torch.relu(self.bn_fc1(self.fc1(x)))
...
return self.fc3(x)          # [B,2]
```

- `torch.flatten(x, 1)`：从第 1 维开始展平。`[B,256,1,1] → [B,256]`；大 FC 版是 `[B,256,14,14] → [B,50176]`。**参数 1 不能丢**：`flatten(x)` 会把 batch 维也拍平，变成一维，后面全连接就全乱了。
- 注意最后一层**没有 softmax**。PyTorch 的 `CrossEntropyLoss` 内部自带 softmax，模型输出原始分数（logits）就行。**多加一个 softmax**：loss 会算错，训练效果变差。

### 3.6 `build_custom_cnn()`：方便改宽度

```python
base = (32, 64, 128, 256)
channels = tuple(max(8, int(round(c * width_mult / 8)) * 8) for c in base)
```

把通道数按 `width_mult` 缩放，并取整到 8 的倍数（GPU 更喜欢 8 的倍数）。`width_mult=1.0` 就是原始宽度；0.5 就是半宽，参数约降到 1/4。当前实验都用 1.0。

### 3.7 `build_resnet18()`：迁移学习的关键三行

```python
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
if freeze_backbone:
    for p in model.parameters():
        p.requires_grad = False
model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
```

1. **`weights=...IMAGENET1K_V1`**：加载在 ImageNet 1000 类上训好的权重。这就是“迁移学习”的来源——卷积层已经会认边缘、纹理、耳朵、眼睛。
2. **先冻结，再换头**（注释里特意写了 Part 8 的坑）：
   - 冻结 = 把所有参数的 `requires_grad` 设成 False，反向传播不更新它们。
   - **顺序不能反**：如果先换头再冻结全部参数，新分类头也会被冻住，整个模型没有可训练参数，loss 不降、准确率 50%。
   - 反过来，先把旧的全冻住，再换头，新头默认 `requires_grad=True`，正好只训新头。
3. **换头**：ResNet18 原来的 `fc` 输出 1000 类，这里换成 `Dropout + Linear(512, 2)`。
   - `in_features = model.fc.in_features` 在替换**之前**取，等于 512。**如果先替换再取**，拿到的是新层的 512，虽然这次也碰巧对，但换别的模型就会出错。

**阶段 A 只有 1026 个可训练参数**（512×2 权重 + 2 个 bias），因为主干全冻住了。**阶段 B 解冻 layer4** 后，可训练参数约 9.4M。

### 3.8 `unfreeze_layer4()`

```python
for p in model.layer4.parameters():
    p.requires_grad = True
for p in model.fc.parameters():
    p.requires_grad = True
```

只解冻最后一个 stage（layer4）和分类头。**为什么不解冻全部**：layer4 已经是最抽象的特征，微调它性价比最高；前面的层学的是边缘纹理，冻着既省时间又防止把预训练特征“洗掉”。

**删掉这步**：阶段 B 和阶段 A 一模一样，E6 实验就没意义了，而且评分标准里“只用预训练不微调”要扣 15 分。

### 3.9 `count_params()`

```python
ps = (p for p in ps if p.requires_grad) if trainable_only else ps
return sum(p.numel() for p in ps)
```

`numel()` 是一个参数张量里的元素个数。`trainable_only=True` 只数“可训练的”，用来验证冻结是否生效（阶段 A 应该只有 1026）。

---

## 四、如果你想改，改哪里

| 想做的事 | 改哪里 | 怎么改 |
|---|---|---|
| 换一个模型（如 VGG、EfficientNet） | ① `models.py` 加一个 `build_xxx()`；② `runner.py` 的 `build_model()` 加一个分支；③ `run_experiments.py` 的 `build_specs()` 里加 `model="xxx"` 的 spec | 三步缺一不可 |
| 改自搭 CNN 宽度 | `build_custom_cnn(width_mult=0.5)` 或改 `base=(32,64,128,256)` | 宽度减半，参数约 1/4 |
| 加/减卷积层 | `CustomCNN.__init__` 里的 `b1..b4` 和 `ConvBlock(n_conv=...)` | **加池化层后必须同步改 `c4*14*14`**，否则展平尺寸错 |
| 改 Dropout 强度 | `CustomCNN` 的 `dropout` 参数、`build_resnet18(dropout=0.3)` | 过拟合就调大，欠拟合就调小 |
| 换 ResNet 深度 | `build_resnet18` 里改成 `resnet34/resnet50` + 对应 `Weights` | resnet50 的 `fc.in_features` 是 2048，代码里已用变量，不用手改 |
| 解冻更多层 | `unfreeze_layer4()` 里再加 `model.layer3.parameters()` | 解冻越多越慢，也越容易过拟合小数据集 |
| 改类别数 | `num_classes` 参数 | 猫狗是 2；换多分类数据集时改这里 |
| 不用预训练 | `build_resnet18(pretrained=False)` | 准确率会明显下降，只能当消融对照 |

---

## 五、新手易错点

1. **冻结和换头的顺序写反。**
   这是 Part 8 就踩过的坑：先换头再 `for p in model.parameters(): p.requires_grad = False`，新头也被冻住，训练时 loss 纹丝不动。正确顺序是“先冻结 → 再换头”。

2. **把 `14*14` 当成永远成立。**
   它只对 224 输入、4 次池化成立。你把输入改成 256，池化后是 16×16，应该写 `c4*16*16`；否则 `Linear` 报形状不匹配。用 `print(model)` 或 `model_structure.txt` 能看到每层实际形状。

3. **在模型最后又加了一个 softmax。**
   `CrossEntropyLoss` 期望原始 logits。多套一层 softmax 等于做了两次，梯度变小、收敛变差。

4. **以为 `count_params()` 数的是“可训练参数”。**
   默认数全部参数（包括冻结的）。验证冻结是否成功要用 `count_params(model, trainable_only=True)`，阶段 A 应该是 1026。

5. **用 batch=1 调试自搭 CNN。**
   `BatchNorm1d` 在训练模式下要求每个通道至少有 2 个样本，batch=1 会报 `Expected more than 1 value per channel`。调试时至少用 batch=2。
