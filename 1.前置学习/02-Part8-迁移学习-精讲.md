# Part 8 · Transfer Learning（迁移学习）逐 Cell 精讲

> 对应文件：`Part 8 - Transfer Learning (Exercises).ipynb`
> 前置：必须先看懂 Part 7（`ImageFolder` / `transforms` / `DataLoader`）。本讲义会直接引用那里的结论，不再重复展开。
> 读者定位：研一零基础，刚学完 CNN，知道"卷积层提特征、全连接层做分类"，但没接触过"预训练模型"和"微调"。

---

## 0. 先看全局：迁移学习到底在"迁"什么

Part 7 结尾作者给了一个悲观判断：**从零训 CNN 做猫狗分类，很难超过 80%**。原因是猫狗数据集只有 2 万张量级，而卷积网络有上千万参数，数据不够 → 严重过拟合。

迁移学习的核心洞察只有一句：

> **卷积网络的"前面几层"学的是通用视觉特征**（边缘、角点、纹理、色块），这些东西对猫、狗、汽车、X 光片都通用；**只有最后几层才是"针对具体任务"的**。

所以做法是：

```
别人的网络（在 ImageNet 120 万张图、1000 类上训练好）
┌──────────────────────────────┬──────────────┐
│  features：卷积层堆叠          │ classifier   │
│  = 万能特征提取器              │ = 针对 1000 类 │
│  【保留，且冻结】              │ 【扔掉，换新的】│
└──────────────────────────────┴──────────────┘
                     ↓ 改造后
┌──────────────────────────────┬──────────────────────┐
│  features（参数不动）          │ 新分类头 Linear(1024→500) │
│                              │ → ReLU → Linear(500→2)   │
│                              │ → LogSoftmax             │
└──────────────────────────────┴──────────────────────┘
```

本 notebook 要学的三个操作，全都围绕这张图：

1. **`models.densenet121(pretrained=True)`** —— 把别人训好的网络连同权重下载下来；
2. **`param.requires_grad = False`** —— 把所有卷积层"冻结"，训练时不动它们；
3. **`model.classifier = 新的 nn.Sequential(...)`** —— 把最后那个"1000 类"的分类头换成"2 类"的。

另外还有一个工程问题：**跑在 GPU 还是 CPU**，以及为什么 GPU 能快 100 倍。

> 注意文件名叫 `(Exercises)`，说明这是**练习版**：Cell 3、Cell 12 是留给你填的（Cell 3 甚至因为没填完而处于"语法不合法"状态），Cell 5、7、10 是**示范代码**（可以直接跑）。

### 全文记号约定

- 形状写成 `[N, C, H, W]`。
- "冻结"= `requires_grad=False`，即反向传播时不计算该参数的梯度、优化器也不会更新它。
- "分类头"= 网络最后那几层专门输出类别分数的全连接层。

---

## Cell 0（markdown）

**英文原文**

> # Transfer Learning
>
> In this notebook, you'll learn how to use pre-trained networks to solved challenging problems in computer vision. Specifically, you'll use networks trained on [ImageNet](http://www.image-net.org/) [available from torchvision](http://pytorch.org/docs/0.3.0/torchvision/models.html).
>
> ImageNet is a massive dataset with over 1 million labeled images in 1000 categories. It's used to train deep neural networks using an architecture called convolutional layers. I'm not going to get into the details of convolutional networks here, but if you want to learn more about them, please [watch this](https://www.youtube.com/watch?v=2-Ol7ZB0MmU).
>
> Once trained, these models work astonishingly well as feature detectors for images they weren't trained on. Using a pre-trained network on images not in the training set is called transfer learning. Here we'll use transfer learning to train a network that can classify our cat and dog photos with near perfect accuracy.
>
> With `torchvision.models` you can download these pre-trained networks and use them in your applications. We'll include `models` in our imports now.

**中文翻译**

> # 迁移学习
>
> 在这个 notebook 里，你将学习如何使用预训练网络来解决计算机视觉中的难题。具体来说，你将使用在 [ImageNet](http://www.image-net.org/) 上训练过的、[可从 torchvision 获取](http://pytorch.org/docs/0.3.0/torchvision/models.html)的网络。
>
> ImageNet 是一个庞大的数据集，包含 1000 个类别、超过 100 万张带标签的图像。它被用来训练深度神经网络，所用的架构叫做卷积层。这里我不打算深入卷积网络的细节，如果你想了解更多，请[观看这个视频](https://www.youtube.com/watch?v=2-Ol7ZB0MmU)。
>
> 训练完成之后，这些模型作为"特征检测器"的效果好得惊人——即使面对它们从未训练过的图像。把预训练网络用于训练集之外的图像，这种做法就叫做迁移学习。这里我们将用迁移学习训练一个网络，让它能以近乎完美的准确率对猫狗照片进行分类。
>
> 借助 `torchvision.models`，你可以下载这些预训练网络并在自己的应用中使用。我们现在就把 `models` 加进导入列表。

**这段的重点**：

- **预训练模型 = 在 ImageNet（1000 类、120 万张）上学过的网络**，它的卷积层已经会"看"图像了。
- **"在没见过的数据上依然好用"** 是迁移学习成立的前提——这也是它能把你项目的准确率从 70% 拉到 95%+ 的原因。
- 关键库是 `torchvision.models`，它提供 `densenet121`、`resnet18`、`vgg16` 等现成架构 + 权重。

---

## Cell 1（code）· 导入

```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
```

### ①【这个 cell 在干什么】

配置绘图，并导入本轮需要的全部模块：`nn`（搭网络）、`optim`（优化器）、`F`（函数式接口）、以及 torchvision 的 `models`（预训练模型仓库）。

### ②【逐行拆解】

**第 1–2 行** `%matplotlib inline` / `%config InlineBackend.figure_format = 'retina'`

- 同 Part 7：让图内联显示、高分屏清晰。**纯显示配置**，删掉不影响训练。

**第 4 行** `import matplotlib.pyplot as plt`

- 绘图接口。本 notebook 里其实几乎没画图（练习部分才有），属于"顺手带上"。

**第 6 行** `import torch`

- PyTorch 主包。这里主要用 `torch.cuda.is_available()`、`torch.device`、`torch.save/load` 相关能力。

**第 7 行** `from torch import nn`

- `nn` = `torch.nn`，**网络层的仓库**：`nn.Linear`、`nn.ReLU`、`nn.Sequential`、`nn.LogSoftmax`、`nn.NLLLoss` 都在这里。
- 为什么单独 import：写 `nn.Linear` 比 `torch.nn.Linear` 短，是 PyTorch 社区的统一习惯。

**第 8 行** `from torch import optim`

- `optim` = `torch.optim`，**优化器仓库**：`optim.Adam`、`optim.SGD`。
- 注意它和 `nn` 是分开的模块：`nn` 管"层"，`optim` 管"怎么更新参数"。

**第 9 行** `import torch.nn.functional as F`

- `F` 是 `nn` 的**函数式版本**：同样是 ReLU、softmax、max_pool，但**没有可学习参数**，所以不需要实例化成层对象，直接 `F.relu(x)` 调用。
- 区别记忆法：**有参数的用 `nn.`（要 `self.xxx = nn.Conv2d(...)`），没参数的用 `F.`（直接 `F.relu`）**。
- 本 notebook 的这个 cell 其实没用到 `F`（是给后续练习准备的），删掉不报错。

**第 10 行** `from torchvision import datasets, transforms, models`

- 相比 Part 7，**多了 `models`**。这是本 notebook 的主角。
- `models` 里每个函数（`densenet121`、`resnet18`…）都能"构建架构 + 加载预训练权重"一步到位。
- 删掉 `models`：Cell 5 的 `models.densenet121(...)` 报 `NameError`。

### ③【张量形状变化】

无数据流动，**形状不变**。

### ④【新手易错点】

1. **`nn` 和 `F` 混用搞不清**。例如写 `self.relu = nn.ReLU()` 后又在 `forward` 里写 `F.relu(self.relu(x))`——重复激活，虽然不报错但破坏语义。
2. **`from torch import nn` 写成 `import torch.nn`**。后者也能用，但必须写 `torch.nn.Linear`，多敲字且容易漏。
3. **`torchvision` 版本太老没有 `models`**（0.2 以前）。现在 pip 装的都自带，一般不会遇到。

---

## Cell 2（markdown）· 224 与三个神奇数字

**英文原文**

> Most of the pretrained models require the input to be 224x224 images. Also, we'll need to match the normalization used when the models were trained. Each color channel was normalized separately, the means are `[0.485, 0.456, 0.406]` and the standard deviations are `[0.229, 0.224, 0.225]`.

**中文翻译**

> 大多数预训练模型要求输入是 224×224 的图像。此外，我们还需要与这些模型训练时所使用的标准化方式保持一致。每个颜色通道是分别标准化的，均值是 `[0.485, 0.456, 0.406]`，标准差是 `[0.229, 0.224, 0.225]`。

**这段的重点**：

- **224×224 是硬约束**。DenseNet/ResNet/VGG 的架构里，全连接层的输入维度是按 224 输入算死的；喂 32×32 或 256×256 进去，要么维度对不上报错，要么（对全卷积结构）虽然能跑但语义已经偏了。
- **这三个均值/标准差是 ImageNet 训练集的统计量**（顺序是 RGB），所有 ImageNet 预训练模型都按这套数做过标准化。
- **为什么必须"对齐"**：模型的第一层卷积是在"已经减过均值、除过标准差"的分布上学到的权重。如果你喂 0~1 的原始像素，等于输入尺度整体偏了一个量级，特征全乱，准确率会从 95% 掉到 60% 甚至更低。**这是迁移学习第一号杀手。**
- 记法：`mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]`，两个都是 RGB 顺序、三个数都接近 0.5 和 0.22。

---

## Cell 3（code）· 真正的"数据管道 + 迁移学习专用 transform"（练习，未填完）

```python
data_dir = 'Cat_Dog_data'

# TODO: Define transforms for the training data and testing data
train_transforms =

test_transforms =

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
```

### ①【这个 cell 在干什么】

这是练习骨架：要求你按 Part 7 的套路写出训练/测试两套 transform，然后建 `ImageFolder` 和 `DataLoader`。**当前状态下不能运行**——两行 `=` 后面是空的。

### ②【逐行拆解】

**第 1 行 `data_dir = 'Cat_Dog_data'`**

- 数据父目录，下面拼 `/train`、`/test`。和 Part 7 Cell 7 完全一致。

**第 3 行（注释）** `# TODO: Define transforms for the training data and testing data`

- 作者的提示：这里要你自己写两套 transform。TODO 是"待办"标记，很多 IDE 会高亮。

**第 4 行 `train_transforms =`**

- **这一行是语法错误**。Python 里赋值号右边不能为空，运行本 cell 会直接抛 `SyntaxError: invalid syntax`（Jupyter 会在执行前就拒绝）。
- 它本来要填的东西（练习意图）是一条 `transforms.Compose([...])`，需要包含 Part 7 里那四个训练用变换：
  1. 随机旋转（`RandomRotation(30)`，参数是**度**）；
  2. 随机裁剪缩放（`RandomResizedCrop(224)`，输出固定 224）；
  3. 随机水平翻转（`RandomHorizontalFlip()`，默认 50% 概率）；
  4. `ToTensor()`（HWC→CHW、/255）；
  5. **比 Part 7 多一步**：`Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` —— 必须是 Cell 2 那三个数，不能再用 `[0.5]*3`。
- 顺序铁律：**几何变换 → `ToTensor` → `Normalize`**。`Normalize` 必须在 `ToTensor` 之后，因为它操作的是张量而不是 PIL 图。
- 不填会怎样：整个 cell `SyntaxError`，后面所有 cell 都无法运行（`train_data` 未定义）。

**第 6 行 `test_transforms =`**

- 同样是**语法错误**的占位。
- 该填的内容：**确定性的** `Resize(255)` → `CenterCrop(224)` → `ToTensor()` → `Normalize(...)`（同样的三个 mean/std）。
- 为什么不能带随机：评估要可复现；而且测试时"增强"只会引入噪声，拉低准确率。
- 为什么 `Normalize` 必须有：训练集 normalize 了，测试集也必须，否则输入分布不一致 —— 这是准确率"莫名只有 50%"的头号原因。

**第 9–10 行 `train_data = datasets.ImageFolder(...)` / `test_data = ...`**

- 和 Part 7 一样：训练目录挂 `train_transforms`，测试目录挂 `test_transforms`。
- 注意 `transform=train_transforms` 是**关键字参数**，不能省（第一个位置参数已经被路径占了）。
- 若 `train_transforms` 是 `None`（比如你只写了 `train_transforms = None` 想绕过语法错误）：`ImageFolder` 不报错，但 `dataset[0]` 返回的是 PIL 图像，`DataLoader` 拼 batch 时会崩。**"能建对象"不等于"能用"**。

**第 12 行 `trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)`**

- `batch_size=64`：比 Part 7 的 32 大了一倍。原因：**迁移学习只训练分类头**，反向传播的计算量和显存占用都小得多，可以把 batch 开大、吞吐更高。
- `shuffle=True`：这次写对了（Part 7 Cell 7 漏了）。
- 没写 `num_workers`，默认 0（主进程读数据），在 Windows 上最省事。

**第 13 行 `testloader = torch.utils.data.DataLoader(test_data, batch_size=64)`**

- 不写 `shuffle`，默认 `False`，评估顺序固定、结果可复现。

### ③【张量形状变化】

（按"填完之后"的正确状态推算）

```
训练分支：
原图 [H₀,W₀,3] → RandomRotation(30) → RandomResizedCrop(224)
            → RandomHorizontalFlip → ToTensor → [3,224,224]
            → Normalize → [3,224,224]（值域约 [-2.1, 2.6]）
            → DataLoader(64) → [64,3,224,224]，labels [64]

测试分支：
原图 → Resize(255) → CenterCrop(224) → ToTensor → Normalize
     → [3,224,224] → DataLoader(64) → [64,3,224,224]，labels [64]
```

注意 `Normalize` **不改变形状**，只改变数值分布。

### ④【新手易错点】

1. **直接运行本 cell 看到 `SyntaxError`，以为是环境坏了**。这是练习骨架，必须先填。
2. **`Normalize` 用了 `[0.5,0.5,0.5]`**（照抄 Part 7）。形状没错、代码能跑，但预训练模型会"看不懂"输入，准确率腰斩。**这是本 notebook 最贵的错误。**
3. **`Normalize` 写在 `ToTensor` 前面**。会报 `TypeError`（PIL 图没有 `.sub_` 之类张量方法）。
4. **只给训练集 normalize，忘了测试集**。训练 loss 正常下降，但验证准确率死活上不去。
5. **`shuffle=True` 加到了 testloader 上**。不影响正确性，但让结果不可复现，答辩时说不清。
6. **`batch_size` 与显存**：DenseNet121 前向要缓存大量中间激活，显存小的卡（4GB）上 64 可能 OOM，改 16/32 即可。

---

## Cell 4（markdown）

**英文原文**

> We can load in a model such as [DenseNet](http://pytorch.org/docs/0.3.0/torchvision/models.html#id5). Let's print out the model architecture so we can see what's going on.

**中文翻译**

> 我们可以加载一个类似 [DenseNet](http://pytorch.org/docs/0.3.0/torchvision/models.html#id5) 这样的模型。我们把模型架构打印出来，看看里面到底是什么样子。

**这段的重点**：选 **DenseNet121** 作为示范模型（121 指带权重的层数）。打印架构是理解"features / classifier 两段式"最直观的方式——下一格一定要认真看输出。

---

## Cell 5（code）· 加载预训练 DenseNet121

```python
model = models.densenet121(pretrained=True)
model
```

### ①【这个 cell 在干什么】

从 torchvision 下载 DenseNet121 的 ImageNet 预训练权重，构建模型对象，并把整个网络结构打印出来（notebook 里最后一行表达式会自动显示 `repr`）。

### ②【逐行拆解】

**第 1 行 `model = models.densenet121(pretrained=True)`**

- `models.densenet121`：DenseNet 家族中"121 层"这一档的构造函数。
- 参数 `pretrained=True`：
  - `True` → **下载并加载**在 ImageNet 上训练好的权重（约 30MB），存到 `~/.cache/torch/hub/checkpoints/`。
  - `False`（默认）→ 只搭空架构，权重随机初始化，等于从零训练，本 notebook 就白学了。
  - **版本提示**：torchvision ≥ 0.13 推荐新写法 `weights=DenseNet121_Weights.IMAGENET1K_V1`，`pretrained=True` 会给出 `UserWarning` 但仍可用；更新版本可能彻底移除，报 `TypeError: __init__() got an unexpected keyword argument 'pretrained'` 时改新写法即可。
- 第一次运行需要联网下载，**会卡住十几秒到几分钟**，不是死机。第二次起走本地缓存。
- 删掉 `pretrained=True`：模型仍是 DenseNet121，但全是随机权重，特征层毫无意义，训练会退化成"从零训一个 700 万参数的深网络"，准确率崩。
- 为什么选 DenseNet 而不是 ResNet：DenseNet 参数量小（约 800 万）、特征复用强，CPU 上也能勉强跑；作者用它做演示比较"轻"。

**第 2 行 `model`**

- 单独一行写变量名，Jupyter 会把它当表达式求值并显示 `repr(model)`。
- 输出是一大段嵌套结构，你要能从中找到三个关键信息：
  1. `(features): Sequential(...)` —— 一长串 `Conv2d / BatchNorm2d / ReLU / MaxPool2d / _DenseBlock / _Transition`；
  2. `(classifier): Linear(in_features=1024, out_features=1000)` —— **只有一层**全连接；
  3. 中间有个 `(norm5): BatchNorm2d(1024, ...)` 和 `AdaptiveAvgPool2d`（有些版本不显式打印）。
- 删掉会怎样：不影响后续代码，只是你失去了"看清结构"的机会。

### ③【张量形状变化】

本 cell 不处理数据，但要知道这个网络的**输入输出契约**：

```
输入 [N, 3, 224, 224]
  → features（conv/pool/dense block/transition 堆叠）
  → 特征图 [N, 1024, 7, 7]      ← 224 经过 5 次下采样 = 7
  → ReLU + AdaptiveAvgPool2d((1,1))   （在 forward 内部完成）
  → [N, 1024]                   ← 每个样本压成一个 1024 维向量
  → classifier: Linear(1024→1000)
  → [N, 1000]                   ← ImageNet 1000 类的打分
```

**`[N, 1024]` 这个"1024"就是后面新分类头的输入维度**，必须记牢，写错就报维度不匹配。

### ④【新手易错点】

1. **断网/公司网络受限导致下载失败**，报 `URLError` / `ReadTimeoutError`。解决：手动下载权重文件放进 `~/.cache/torch/hub/checkpoints/`，或用国内镜像。
2. **以为 `pretrained=True` 会"直接能分猫狗"**。它输出的是 1000 类，其中确实有猫和狗两类，但你没法保证编号，也不能微调；必须换分类头。
3. **在无 GPU 的机器上跑，觉得"怎么这么慢"**。DenseNet121 前向一次 64 张图，CPU 上要好几秒。
4. **看输出时找不到 classifier**，误以为模型坏了。它在打印结果的**最末尾**。
5. **`model` 和 `print(model)` 的区别**：前者用 `repr`（带缩进的结构化显示），后者用 `str`（内容基本一样）。都行。

---

## Cell 6（markdown）· 两段式结构：features 与 classifier

**英文原文**

> This model is built out of two main parts, the features and the classifier. The features part is a stack of convolutional layers and overall works as a feature detector that can be fed into a classifier. The classifier part is a single fully-connected layer `(classifier): Linear(in_features=1024, out_features=1000)`. This layer was trained on the ImageNet dataset, so it won't work for our specific problem. That means we need to replace the classifier, but the features will work perfectly on their own. In general, I think about pre-trained networks as amazingly good feature detectors that can be used as the input for simple feed-forward classifiers.

**中文翻译**

> 这个模型由两个主要部分构成：`features`（特征部分）和 `classifier`（分类器部分）。`features` 部分是一堆卷积层的堆叠，整体上充当"特征检测器"，它的输出可以喂给分类器。`classifier` 部分是一个单独的全连接层 `(classifier): Linear(in_features=1024, out_features=1000)`。这一层是在 ImageNet 数据集上训练的，所以它不适用于我们这个具体问题。这意味着我们需要替换掉分类器，而 `features` 部分本身可以完美地工作。总的来说，我把预训练网络理解为"极其优秀的特征检测器"，可以作为简单的**前馈分类器**的输入。

**这段的重点**：

- **`features` 是通用的，`classifier` 是任务专属的**。这是整个迁移学习的方法论基础，也是你项目报告里"为什么冻结卷积层"的标准答案。
- `features` 输出 **1024 维向量**，这就是新分类头的 `in_features`。
- 换头 = 把 `Linear(1024→1000)` 换成 `Linear(1024→2)`（中间可以再加隐藏层）。
- 注意原文说的"simple feed-forward classifiers"：**后面的分类头就是普通 MLP**，不需要卷积。

---

## Cell 7（code）· 冻结 + 换头（本 notebook 的核心 8 行）

```python
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
```

### ①【这个 cell 在干什么】

把预训练模型的**所有参数冻住**（不参与反向传播），然后用一个"1024→500→2→LogSoftmax"的小 MLP 替换掉原来的 1000 类分类头。

### ②【逐行拆解】

**第 1 行（注释）** `# Freeze parameters so we don't backprop through them`

- 说明动机：冻结是为了**不做反向传播**——省显存、省时间，同时防止小数据集把好不容易学到的通用特征"冲毁"（灾难性遗忘）。

**第 2–3 行 `for param in model.parameters(): param.requires_grad = False`**

- `model.parameters()`：**递归**返回模型里所有 `nn.Parameter`（卷积核权重、偏置、BatchNorm 的 γ/β 全都在内），是一个生成器。
- `param.requires_grad`：每个张量的布尔属性，默认 `True`。
  - `True`：前向时记录计算图，`loss.backward()` 时计算它的梯度，`optimizer.step()` 时更新它。
  - `False`：**不记录、不计算、不更新**。梯度为 `None`，`optimizer` 会直接跳过。
- 为什么要写成循环而不是 `model.features.requires_grad_(False)`：`requires_grad_()` 这种写法在部分版本/部分模块上不能递归到子层，逐参数遍历是最保险的写法。
- **位置极其重要**：必须在**创建新 classifier 之前**执行。如果放在 `model.classifier = classifier` 之后，新分类头的参数也会被冻住 → 整个网络没有一个参数可训练 → loss 永远不降、准确率恒为 50%，而且**不报任何错**。这是本 notebook 最隐蔽的 bug。
- 删掉这两行会怎样：所有层都参与训练。在 1.7 万张图上微调 700 万参数，几分钟内就开始过拟合，而且训练速度慢 5~10 倍。

**第 5 行 `from collections import OrderedDict`**

- Python 标准库的**有序字典**。3.7+ 的普通 `dict` 也已保证插入顺序，但 `nn.Sequential` 的官方示例一直用 `OrderedDict`，因为它能"给每一层起名字"。
- 为什么需要名字：`print(model)` 时你会看到
  ```
  (classifier): Sequential(
    (fc1): Linear(in_features=1024, out_features=500, bias=True)
    (relu): ReLU()
    (fc2): Linear(in_features=500, out_features=2, bias=True)
    (output): LogSoftmax(dim=1)
  )
  ```
  名字让调试、保存/加载（`state_dict` 的 key 就是名字）都可读。不写名字的话会自动编号 `(0)(1)(2)(3)`。
- 删掉会怎样：只要把 `OrderedDict([...])` 换成直接传 `[...]` 列表（每项写成 `(名字, 模块)` 元组形式也可），仍然能工作。但只删 `import` 不删用法会 `NameError`。

**第 6–10 行 `classifier = nn.Sequential(...)`**

- `nn.Sequential`：**顺序容器**，`forward` 时按定义顺序依次调用各层，`x = layer(x)` 一路传下去。适合"线性堆叠"的结构。
- `('fc1', nn.Linear(1024, 500))`：
  - `nn.Linear(in_features, out_features, bias=True)`：全连接层，内部就是 `y = xWᵀ + b`。
  - `1024`：**必须等于 features 输出的维度**（Cell 5 里算出的 `[N,1024]`）。写错（比如写 512）会报 `mat1 and mat2 shapes cannot be multiplied (64x1024 and 512x500)`。
  - `500`：隐藏层宽度，**人为选的超参数**，没有硬性规定。500 是"够表达又不至于过拟合"的经验值；改成 256/1024 都能跑，效果会变。
  - 这一层的参数量：`1024*500 + 500 = 512,500`。
- `('relu', nn.ReLU())`：
  - 无参数激活函数，`f(x)=max(0,x)`，把负值截成 0，给网络引入非线性。
  - 为什么加在两层 Linear 之间：两个 Linear 直接相连等价于**一个** Linear（矩阵乘法可结合），不加激活就等于白加一层。
  - `nn.ReLU()` 默认 `inplace=False`；设 `inplace=True` 可省一点显存，但在 Sequential 里可能有副作用，教学代码不用。
- `('fc2', nn.Linear(500, 2))`：
  - `500` 必须等于上一层输出；`2` 是**类别数**——猫/狗两类。
  - 这是"迁移"的关键动作：把 1000 改成你任务的类别数。若做 10 类（比如 CIFAR-10）就写 10。
  - 输出 `[N, 2]`，每个样本两个"分数"。
- `('output', nn.LogSoftmax(dim=1))`：
  - `LogSoftmax(dim=1)`：沿 **dim=1（类别维）** 做 softmax 再取 log。输出每个样本两类各自的对数概率，两数之和的 `exp` 为 1，且都为负。
  - 为什么用 `LogSoftmax` 而不是 `Softmax`：配合 `nn.NLLLoss` 时，`log_softmax + NLLLoss` **在数值上比 `softmax + CrossEntropyLoss` 手写版更稳定**（避免 log(0) 和下溢）。而 `nn.CrossEntropyLoss` 本身就等于 `LogSoftmax + NLLLoss`，二者不要重复叠加。
  - `dim=1` 不能省也不能改成 0：输入是 `[N, 2]`，`dim=0` 会在**样本之间**做归一化，等于把不同图片混在一起算概率，完全错误。
  - 删掉这一层：模型输出变成未归一化的 logits（可正可负、不求和为 1）。此时若仍用 `NLLLoss`，loss 会是错的（NLLLoss 期望输入是对数概率），训练效果明显变差但不一定报错。

**第 12 行 `model.classifier = classifier`**

- **Python 属性赋值**。`model` 是 `nn.Module`，`classifier` 是它的一个子模块属性；赋值会把新模块注册进 `model._modules`，于是：
  - `model.parameters()` 会包含新分类头的参数；
  - `model.to(device)` 会把新分类头一起搬走；
  - `state_dict()` 的 key 变成 `classifier.fc1.weight` 等。
- 被替换掉的旧 `Linear(1024,1000)` 会被 Python 垃圾回收，显存释放。
- 为什么能直接赋值：`nn.Module.__setattr__` 做了特殊处理，检测到赋的是 `nn.Module` 就自动注册为子模块。
- 删掉会怎样：`model` 仍然是 1000 类输出，`output` 形状 `[N,1000]`，而 `labels` 只有 0/1 → `NLLLoss` 报 `Target 0 is out of bounds`（或维度不匹配），训练无法进行。

### ③【张量形状变化】

```
输入 [64, 3, 224, 224]
  → features（冻结）→ [64, 1024, 7, 7]
  → 全局平均池化 → [64, 1024]
  → fc1: Linear(1024,500) → [64, 500]
  → ReLU → [64, 500]
  → fc2: Linear(500,2) → [64, 2]
  → LogSoftmax(dim=1) → [64, 2]（对数概率）
标签 labels: [64]（int64，取值 0/1）
```

参数量对比：features 约 700 万（**全部冻结**）；新分类头 `512,500 + 1,002 ≈ 51.4 万`（**可训练**）。也就是说你只训练了原网络的 **7%** 左右。

### ④【新手易错点】

1. **冻结写在换头之后 → 全体冻结，模型学不动**。症状：loss 几乎不变、准确率恒 50%、不报错。**排查方法**：`sum(p.requires_grad for p in model.parameters())`，若为 0 就是踩坑了。
2. **`in_features` 写错**（不是 1024）。不同模型不一样：`resnet18` 是 512，`vgg16` 是 4096，`densenet121` 才是 1024。换模型必须查。
3. **`out_features` 用了 1000 或 1**。1000 是 ImageNet 的类数；写 1 会让输出 `[N,1]`，与 `NLLLoss` 的二分类用法不匹配。二分类也是**输出 2 维**，不是 1 维。
4. **`LogSoftmax` 和 `CrossEntropyLoss` 搭配**。`CrossEntropyLoss` 内部已含 softmax，若再接 `LogSoftmax` 等于做了两次归一化，loss 会异常偏小、梯度被压扁。
5. **`dim` 写错**（`dim=0` 或忘写）。忘写会报 `TypeError`（必填参数）；写 0 则静默错误。
6. **以为换了头就不用管旧权重**。旧头是被丢弃了，但 features 的权重还在，这才是迁移学习省下的部分。

---

## Cell 8（markdown）· 为什么必须上 GPU

**英文原文**

> With our model built, we need to train the classifier. However, now we're using a **really deep** neural network. If you try to train this on a CPU like normal, it will take a long, long time. Instead, we're going to use the GPU to do the calculations. The linear algebra computations are done in parallel on the GPU leading to 100x increased training speeds. It's also possible to train on multiple GPUs, further decreasing training time.
>
> PyTorch, along with pretty much every other deep learning framework, uses [CUDA](https://developer.nvidia.com/cuda-zone) to efficiently compute the forward and backwards passes on the GPU. In PyTorch, you move your model parameters and other tensors to the GPU memory using `model.to('cuda')`. You can move them back from the GPU with `model.to('cpu')` which you'll commonly do when you need to operate on the network output outside of PyTorch. As a demonstration of the increased speed, I'll compare how long it takes to perform a forward and backward pass with and without a GPU.

**中文翻译**

> 模型搭好了，接下来需要训练分类器。但是，我们现在用的是**一个非常深的**神经网络。如果你像平常那样在 CPU 上训练，会花非常非常长的时间。所以我们改用 GPU 来做计算。线性代数运算在 GPU 上可以并行执行，从而带来 100 倍的训练提速。同时也可以在多个 GPU 上训练，进一步缩短训练时间。
>
> PyTorch 和几乎所有其他深度学习框架一样，使用 [CUDA](https://developer.nvidia.com/cuda-zone) 来在 GPU 上高效地计算前向和反向传播。在 PyTorch 中，你用 `model.to('cuda')` 把模型参数和其他张量搬到 GPU 显存里；也可以用 `model.to('cpu')` 把它们搬回来——当你需要在 PyTorch 之外处理网络输出时，经常会这么做。为了演示提速效果，我会比较一下有 GPU 和没 GPU 时，一次前向 + 反向传播的耗时。

**这段的重点**：

- **`.to(device)` 是"搬运"**：模型和输入数据**必须在同一个设备上**，否则报 `Expected all tensors to be on the same device`。
- 设备字符串：`'cpu'`、`'cuda'`、`'cuda:0'`（第 0 块卡）。
- **`model.to(device)` 是原地修改**，不需要重新赋值（写成 `model = model.to(device)` 也不会错，只是多余）。
- **输出要回到 CPU 才能用 numpy / matplotlib**：`output.cpu().numpy()`。GPU 张量不能直接 `.numpy()`。
- 你项目答辩时如果只有 CPU，这段"GPU 快 100 倍"的演示正好可以说明"为什么我们的训练用了 XX 分钟/为什么我们选择冻结特征只训分类头"。

---

## Cell 9（code）

```python
import time
```

### ①【这个 cell 在干什么】

导入标准库 `time`，供下一格计时用。

### ②【逐行拆解】

**第 1 行 `import time`**

- Python 标准库，无需安装。这里只用 `time.time()`：返回当前时间戳（秒，float）。
- 用法：`start = time.time()` … `elapsed = time.time() - start`。
- 为什么单独一个 cell：Jupyter 里作者习惯"一个 import 一格"，方便你只重跑需要的部分。
- 删掉会怎样：下一格 `NameError: name 'time' is not defined`。
- 补充：更精确的计时应该用 `time.perf_counter()`；`time.time()` 精度足够做这种数量级对比。

### ③【张量形状变化】

无，**形状不变**。

### ④【新手易错点】

1. **`import time` 和 `from datetime import time` 搞混**，后者是"时间点"类型，没有 `.time()` 函数。
2. **在 GPU 上计时忘了 `torch.cuda.synchronize()`**。CUDA 是**异步**的：`loss.backward()` 发出指令就返回，实际计算还在排队。不 synchronize 的话计时会严重偏小（这也是本 notebook 计时结果只能"看个感觉"的原因之一）。

---

## Cell 10（code）· CPU vs GPU 计时对比

```python
for device in ['cpu', 'cuda']:

    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.to(device)

    for ii, (inputs, labels) in enumerate(trainloader):

        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        start = time.time()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if ii==3:
            break
        
    print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")
```

### ①【这个 cell 在干什么】

在 CPU 和 GPU 上各跑 4 个 batch 的"前向 + loss + 反向 + 更新"，打印单批耗时，直观对比加速比。

### ②【逐行拆解】

**第 1 行 `for device in ['cpu', 'cuda']:`**

- 依次取 `'cpu'` 和 `'cuda'` 两个设备名。
- 为什么先 CPU 后 GPU：先跑慢的，跑的时候你有时间看日志；也方便对比输出顺序。
- **无 GPU 的机器会在第二轮 `'cuda'` 时抛 `RuntimeError: Found no NVIDIA driver` / `Torch not compiled with CUDA enabled`**。这是本 cell 在你电脑上最可能的"失败点"，属于预期行为（练习环境假设有卡）。
- 更稳妥的写法（作者在下一个 markdown 里给了）：`device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`。

**第 3 行 `criterion = nn.NLLLoss()`**

- `NLLLoss` = Negative Log Likelihood Loss，**负对数似然损失**，用于多分类。
- 它期望的输入是**对数概率**（`log_softmax` 的输出，也就是我们分类头的最后一层），标签是**类别索引**（int64，不是 one-hot）。
- 公式：`loss = -mean( log_prob[i, label[i]] )`，即"取正确类别的对数概率，加负号，求平均"。
- 为什么不是 `CrossEntropyLoss`：因为 Cell 7 的分类头已经带 `LogSoftmax` 了。两者**必须配对**，不能随意换。
- 删掉会怎样：下面 `criterion(...)` 报 `NameError`。
- 注意它在 `for device` 循环**内部**重建，其实没必要（无状态），放在循环外也一样。

**第 4 行（注释）** `# Only train the classifier parameters, feature parameters are frozen`

- 提醒：优化器只接收 `model.classifier.parameters()`。

**第 5 行 `optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)`**

- `optim.Adam(params, lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0)`：
  - `params`：**只传分类头的参数**。传 `model.parameters()` 也不会错（冻结参数的 `grad` 为 `None`，Adam 会跳过），但显式只传 classifier 更清晰、也省一点内存。
  - `lr=0.001`：学习率，Adam 的默认值。Adam 对学习率不敏感（自带自适应步长），1e-3 是标准起点。
  - `betas`：一阶/二阶矩的指数衰减率，一般不动。
  - `eps`：防止除零的小量。
- 为什么用 Adam 而不是 SGD：**只训练一个浅分类头**时，Adam 收敛快、几乎不用调参；SGD 往往需要配 momentum 和更细的学习率。
- 关键细节：`optimizer` 在每次 `device` 循环里**重新创建**。这很重要——Adam 内部有 `state`（每个参数的动量缓存），如果把 CPU 轮次的 optimizer 拿到 GPU 轮次继续用，state 里的张量还在 CPU 上，会报设备不一致。
- 删掉会怎样：`optimizer.step()` 报 `NameError`；即使删掉 `step()`，参数也永远不更新。

**第 7 行 `model.to(device)`**

- 把**所有参数和 buffer**（包括 BatchNorm 的 running_mean/var）搬到指定设备，**原地修改**。
- 每次循环都要调：因为上一轮可能已经在另一个设备上了。
- 为什么必须做：模型在 CPU、输入在 GPU → 前向第一层就报 `Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`。
- 删掉会怎样：`'cpu'` 轮还能跑（都在 CPU），`'cuda'` 轮会在 `model.forward(inputs)` 处报设备不一致。

**第 9 行 `for ii, (inputs, labels) in enumerate(trainloader):`**

- `enumerate` 同时给出**序号 `ii`** 和元素 `(inputs, labels)`。
- 为什么需要 `ii`：下面用 `if ii==3: break` 控制"只跑 4 个 batch"。
- `trainloader` 每轮取 64 张（Cell 3 的 `batch_size=64`）。
- 注意：GPU 轮会**重新遍历** trainloader（因为 `shuffle=True`，拿到的样本和 CPU 轮不同），这对计时无影响。

**第 12 行 `inputs, labels = inputs.to(device), labels.to(device)`**

- 把**数据**搬到同一设备。`.to()` 返回**新张量**（不是原地），所以要重新赋值。
- 对比 `model.to(device)` 是原地、`tensor.to(device)` 是返回新对象——这个不对称经常把人绕晕。
- 若张量已经在目标设备上，`.to()` 直接返回自身，不复制（所以写 `data.to(device)` 总是安全的）。
- 删掉会怎样：GPU 轮报设备不一致。
- **计时细节**：这一行在 `start` **之前**，所以"数据搬运耗时"没有被计入——这是作者有意为之（只比计算）。

**第 14 行 `start = time.time()`**

- 记录本 batch 开始时刻。
- **注意它在循环体内**，所以每跑一个 batch 就被重置一次。循环结束后 `start` 保存的是**最后一个 batch（ii==3）**的开始时间。

**第 16 行 `outputs = model.forward(inputs)`**

- 显式调用 `forward`。等价于 `outputs = model(inputs)`，但**不推荐**显式写 `.forward()`——它会绕过 `nn.Module.__call__` 里的 hook 机制（如 `register_forward_hook`）。教学代码这么写是为了强调"这是在跑前向"。
- 张量：`[64,3,224,224]` → `[64,2]`（对数概率）。
- 删掉会怎样：`outputs` 未定义，下一行报错。

**第 17 行 `loss = criterion(outputs, labels)`**

- 计算**标量** loss（0 维张量）。
- `outputs [64,2]` 与 `labels [64]` 按 batch 维对齐，每个样本取自己正确类别的对数概率。
- 删掉会怎样：`loss` 未定义，`loss.backward()` 报错。

**第 18 行 `loss.backward()`**

- **反向传播**：从 `loss` 出发，沿计算图自动求每个 `requires_grad=True` 的张量的梯度，**累加**到 `.grad` 属性上。
- 因为 features 被冻结，只有分类头的 `.grad` 会被填充，features 的 `.grad` 保持 `None`。这正是"冻结省算力"的体现。
- 删掉会怎样：所有 `.grad` 为 `None`，`optimizer.step()` 什么也不做（不报错！），loss 永远不降——**静默失败**，很难查。
- **关键坑**：`backward()` 是**累加**而不是覆盖。本 cell 没有 `optimizer.zero_grad()`，所以第 2、3、4 个 batch 的梯度是在前面积累的基础上继续加的。对"计时"没影响，但对训练是错的。

**第 19 行 `optimizer.step()`**

- 按各参数的 `.grad` 更新一次参数（Adam 用自己的规则）。
- 本 cell 缺 `optimizer.zero_grad()`，所以效果是"用累积梯度更新"，参数会被推得过大。**这行本身没错，错在少了清零**。
- 删掉会怎样：不报错，但参数永远不变。

**第 21–22 行 `if ii==3: break`**

- `ii` 从 0 开始，`ii==3` 时是**第 4 个** batch。执行完这个 batch 的 `step()` 后跳出。
- 为什么用 4 个 batch：只看数量级，不需要跑完整个 epoch（否则 CPU 轮要几十分钟）。
- 删掉会怎样：会把整个 epoch（约 268 个 batch）跑完，CPU 轮耗时极长，你可能以为程序卡死。

**第 24 行 `print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")`**

- f-string 格式化：`{device}` 插设备名，`{(...)/3:.3f}` 保留 3 位小数。
- **`/3` 是这段代码的瑕疵**：`start` 在循环内每次重置，最终窗口只有**最后 1 个 batch**，却除以 3，等于把单批耗时**低估了 3 倍**。正确做法是把 `start` 放到 `for ii` 循环之前，或者统计每批时间再累加。
  - 你要理解作者的**意图**：4 个 batch 里第 1 个包含 CUDA 初始化（很慢），所以想"丢掉第一个、用后面 3 个平均"。但实现时 `start` 放错了位置，没达到目的。
  - **这个 bug 不影响"GPU 比 CPU 快"的定性结论**（两边同样低估），但绝对数值不可信。答辩时不要引用这里的秒数。
- 删掉会怎样：看不到结果，白跑。

### ③【张量形状变化】

```
trainloader（batch_size=64）
  → 一个 batch：inputs [64,3,224,224]，labels [64]
  → inputs.to(device)  → 形状不变，设备变为 cuda:0（或仍是 cpu）
  → model.forward     → outputs [64,2]（对数概率）
  → criterion         → loss [] （0 维标量）
  → backward          → 各可训练参数 .grad 形状同参数本身
                        （fc1.weight [500,1024]、fc2.weight [2,500]）
  → step              → 参数原地更新，形状不变
```

### ④【新手易错点】

1. **CPU-only 机器直接 `RuntimeError`**。这是最常见的问题：把 `['cpu','cuda']` 改成 `['cpu']`，或改用 `torch.cuda.is_available()` 判断。
2. **漏了 `optimizer.zero_grad()`**。本例就是。梯度跨 batch 累加 → 更新步长越来越大 → loss 爆炸或直接 `nan`。
3. **`model.to(device)` 在 `to` 之后又重建了 model**。比如先 `model.to('cuda')` 然后 `model = models.resnet18(...)`，新模型会在 CPU 上。
4. **计时忘了 `torch.cuda.synchronize()`**。CUDA 异步执行，`time.time()` 测到的是"指令下发耗时"，不是"算完耗时"，GPU 会显得快得不真实。
5. **显式 `.forward()`**。虽然能跑，但会跳过 hook；生产代码写 `model(inputs)`。
6. **`criterion` 和输出层不配对**。这里 `NLLLoss` 配 `LogSoftmax`；如果换成 `CrossEntropyLoss`，就要把分类头里的 `LogSoftmax` 删掉。
7. **以为跑完这段模型就训好了**。这段只是"计时演示"，只更新了 4 次参数，模型基本还是原样。

---

## Cell 11（markdown）· 设备无关写法 + 练习

**英文原文**

> You can write device agnostic code which will automatically use CUDA if it's enabled like so:
> ```python
> # at beginning of the script
> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
>
> ...
>
> # then whenever you get a new Tensor or Module
> # this won't copy if they are already on the desired device
> input = data.to(device)
> model = MyModule(...).to(device)
> ```
>
> From here, I'll let you finish training the model. The process is the same as before except now your model is much more powerful. You should get better than 95% accuracy easily.
>
> >**Exercise:** Train a pretrained models to classify the cat and dog images. Continue with the DenseNet model, or try ResNet, it's also a good model to try out first. Make sure you are only training the classifier and the parameters for the features part are frozen.

**中文翻译**

> 你可以写"设备无关"的代码，让它在 CUDA 可用时自动使用 GPU，像这样：
> ```python
> # 脚本开头
> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
>
> ...
>
> # 之后每当你得到新的张量或模型时
> # 如果它们已经在目标设备上，这一步不会发生拷贝
> input = data.to(device)
> model = MyModule(...).to(device)
> ```
>
> 到这里，模型的训练就交给你来完成了。流程和之前完全一样，只不过现在的模型强大得多。你应该能轻松拿到 95% 以上的准确率。
>
> >**练习：** 训练一个预训练模型来对猫狗图像分类。可以继续用 DenseNet，也可以试试 ResNet——它也是个很好的上手选择。务必确保你**只训练分类器**，并且 features 部分的参数是冻结的。

**这段的重点**：

- **`torch.device("cuda:0" if ... else "cpu")` 是写训练脚本的标准开头**，能让同一份代码在有人/没人 GPU 的机器上都跑。
- 训练循环的结构和 Part 7 之前学的**一模一样**：`zero_grad → forward → loss → backward → step`，只是多了 `.to(device)`。**迁移学习不改变训练流程，只改变模型来源。**
- 作者承诺"轻松 95%+"——这就是迁移学习的威力：1.7 万张图训练一个 51 万参数的分类头，几个 epoch 就能到 95%。
- 练习的硬性要求："只训练分类器 + features 冻结"，这正是你项目评分里"冻结部分层并微调分类头"的对应动作。

---

## Cell 12（code）· 练习占位

```python
## TODO: Use a pretrained model to classify the cat and dog images
```

### ①【这个 cell 在干什么】

空的练习占位，要求你自己补全"完整训练 + 验证"循环。

### ②【逐行拆解】

- 只有一行注释，**无可执行代码**，运行无输出。
- 这行注释描述的任务，按前面的知识拆成 6 步：
  1. 确认 Cell 3 的两套 transform 已填好（含正确的 `Normalize`）；
  2. 建 `trainloader` / `testloader`（Cell 3 已给框架）；
  3. 加载预训练模型 → 冻结 features → 换分类头（Cell 5 + 7 已完成）;
  4. 定义 `criterion = nn.NLLLoss()` 与 `optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)`；
  5. 写 epoch 循环：训练段 `model.train()` + 验证段 `model.eval()`，每段都 `zero_grad → forward → loss → backward(仅训练) → step(仅训练)`；
  6. 记录/打印每个 epoch 的训练与验证 loss（以及准确率），最后在 `testloader` 上算一次测试准确率。

### ③【张量形状变化】

无可执行代码；若补全，形状主线与 Cell 10 一致：`[64,3,224,224] → [64,2]`，loss 为标量。

### ④【新手易错点】

1. **验证段忘了 `model.eval()`**。BatchNorm 会继续用当前 batch 的统计量、Dropout 会继续随机丢弃，验证 loss 抖得厉害；而且**结果不可复现**。
2. **验证段忘了 `torch.no_grad()`**。能跑，但会构建计算图、浪费大量显存，容易 OOM。
3. **验证段误写了 `optimizer.step()`**。等于拿验证集训练，数据泄漏，准确率虚高。
4. **算平均 loss 时除以 batch 数而不是样本数**。正确做法（CIFAR 那份 notebook 里有示范）：`loss.item() * data.size(0)` 累加，最后除以样本总数。
5. **忘了"保存最优模型"**。一个 epoch 训完就结束，拿到的可能是过拟合的最后一版；应该按验证 loss 最低来存 `state_dict`。
6. **以为"冻结"等于"完全不参与前向"**。冻结只是不更新，**前向仍然要算**（否则没有特征输出），所以 GPU 时间并不会降到 0。

---

## 本 notebook 小结（复习用）

| 关键词 | 一句话记忆 |
|---|---|
| 迁移学习 | 借别人在 ImageNet 上学到的"通用特征提取器" |
| `densenet121(pretrained=True)` | 下载架构 + ImageNet 权重；需联网，首次慢 |
| features / classifier | 前者通用且冻结，后者任务专属且重训 |
| `requires_grad=False` | 冻结：不算梯度、不更新；**必须在换头之前** |
| 新分类头 | `Linear(1024,500) → ReLU → Linear(500,2) → LogSoftmax(dim=1)` |
| `NLLLoss` | 配 `LogSoftmax`；标签是类别索引，不是 one-hot |
| `Adam(classifier.parameters(), lr=1e-3)` | 只喂可训练参数；Adam 自适应、免调参 |
| `model.to(device)` | 原地搬模型；`tensor.to(device)` 返回新张量 |
| ImageNet 标准化 | `mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]` |
| 形状主线 | `[64,3,224,224]` → `[64,1024]` → `[64,2]` |

**相对 Part 7 的三处升级**：① transform 多了一步 `Normalize`（且必须用 ImageNet 统计量）；② 模型从"自己搭"变成"下载 + 改造"；③ batch_size 从 32 提到 64（因为只训浅层分类头）。

**给你的项目的一句话建议**：Part 8 直接对应你项目"迁移学习实现"这 30 分里的核心动作——**冻结 + 换头 + 微调**。评分标准里有一条"仅提交预训练模型且未进行任何微调：−15 分"，指的就是"只调 `predict` 不改分类头"；而你按 Cell 7 的做法换头并训练，就是合规的"微调（fine-tune）"。
