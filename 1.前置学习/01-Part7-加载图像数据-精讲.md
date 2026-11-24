# Part 7 · Loading Image Data（加载图像数据）逐 Cell 精讲

> 对应文件：`Part 7 - Loading Image Data (Solution).ipynb`
> 读者定位：研一零基础、刚学完 CNN。本讲义假设你只会"手写 MNIST 的 `TensorDataset`"，对"从硬盘读图片"完全陌生。

---

## 0. 先看全局：这个 notebook 到底解决了什么问题

前面的课程（MNIST / Fashion-MNIST）里，数据是**已经打包成一个张量**交给你的：`train_data` 直接就是 `[60000, 1, 28, 28]` 的浮点数。所以那时代码里根本没有"读文件"这一步。

真实项目不是这样。你手上只有：

```
Cat_Dog_data/train/cat/cat.0.jpg
Cat_Dog_data/train/cat/cat.1.jpg
...
Cat_Dog_data/train/dog/dog.0.jpg
...
```

也就是**一堆 JPG 文件 + 目录名当标签**。于是有三件事必须自己干：

1. **把文件读进内存，并自动知道标签**（目录名 → 类别编号）→ `datasets.ImageFolder`
2. **把每张图变成统一尺寸、统一数值范围的张量**（不同照片尺寸不一样、像素是 0~255 的整数）→ `transforms`
3. **把一堆样本切成 mini-batch 送进网络**（还要打乱）→ `DataLoader`

这三步就是工业界所谓 **数据管道（data pipeline）**。Part 7 讲的就是它。它的输出，就是后面 Part 8、以及你项目里自定义 CNN 的输入。

本机数据实际情况（我数过文件）：

| 目录 | 文件数 |
|---|---|
| `Cat_Dog_data/train/cat` | 11250 |
| `Cat_Dog_data/train/dog` | 11250 |
| `Cat_Dog_data/test/cat` | 1250 |
| `Cat_Dog_data/test/dog` | 1250 |

训练集共 22500 张，**猫狗各 11250，类别完全平衡**；测试集共 2500 张，同样各 1250。类别平衡意味着不需要额外做重采样/类别权重，准确率也可以直接当评价指标用。

### 全文记号约定

- 形状统一写成 `[N, C, H, W]` = 批大小、通道数、高、宽。
- `[32, 3, 224, 224]` 读作"32 张图、每张 3 通道（RGB）、224×224 像素"。
- 讲到"删掉会怎样"时，指的是**单独删掉这一行/这个参数**的后果，不是让你真删。

---

## Cell 0（markdown）

**英文原文**

> # Loading Image Data
>
> So far we've been working with fairly artificial datasets that you wouldn't typically be using in real projects. Instead, you'll likely be dealing with full-sized images like you'd get from smart phone cameras. In this notebook, we'll look at how to load images and use them to train neural networks.
>
> We'll be using a [dataset of cat and dog photos](https://www.kaggle.com/c/dogs-vs-cats) available from Kaggle. Here are a couple example images:
>
> `<img src='assets/dog_cat.png'>`
>
> We'll use this dataset to train a neural network that can differentiate between cats and dogs. These days it doesn't seem like a big accomplishment, but five years ago it was a serious challenge for computer vision systems.

**中文翻译**

> # 加载图像数据
>
> 到目前为止，我们一直在处理相当"人造"的数据集——这类数据你在真实项目里通常不会用到。真实情况下，你面对的往往是智能手机相机拍出来的那种全尺寸照片。在这个 notebook 里，我们来看看如何加载图像，并用它们来训练神经网络。
>
> 我们将使用 Kaggle 上提供的[猫狗照片数据集](https://www.kaggle.com/c/dogs-vs-cats)。下面是几张示例图片：
>
> `<img src='assets/dog_cat.png'>`
>
> 我们会用这个数据集训练一个能区分猫和狗的神经网络。在今天看来这似乎算不上什么了不起的成就，但在五年前，这对计算机视觉系统来说是一个严峻的挑战。

**这段的重点**：从"玩具数据集（已经打包好的张量）"切换到"真实图像（一堆文件）"，这是本 notebook 的核心动机；数据来源是 Kaggle Dogs vs Cats。

---

## Cell 1（code）· 导入与环境设置

```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms

import helper
```

### ①【这个 cell 在干什么】

配置绘图后端，并导入后面要用的四个库：绘图、PyTorch、torchvision 的数据集与变换模块、课程自带的显示工具。

### ②【逐行拆解】

**第 1 行 `%matplotlib inline`**

- 这是 **Jupyter 魔法命令（magic command）**，以 `%` 开头，不是 Python 语法，只在 notebook / IPython 里有效。
- 作用：让 `plt.show()` 或任何绘图结果**直接显示在 cell 下方的输出区**，而不是弹出一个独立窗口。
- 为什么这么写：教学 notebook 必须"代码 + 图"放在同一页才好读。
- 删掉会怎样：在老的 Jupyter Notebook 里，图会弹到外部窗口（甚至卡住执行）；在新版 JupyterLab 里默认就是 inline，删掉一般不影响显示。所以它是"保险"性质的一行。

**第 2 行 `%config InlineBackend.figure_format = 'retina'`**

- 又一个魔法命令，改的是 matplotlib 的**内联后端配置**。
- `InlineBackend`：负责把图画成 PNG 再塞进输出区的那个后端。
- `figure_format = 'retina'`：输出 2 倍分辨率的图（高 DPI），在高分屏上不糊。
- 删掉会怎样：图还是能显示，只是会模糊一些，属于**纯观感**设置。

**第 4 行 `import matplotlib.pyplot as plt`**

- 导入 matplotlib 的绘图接口，惯例别名为 `plt`。
- 为什么是 `pyplot` 而不是 `matplotlib`：`pyplot` 是"像 MATLAB 一样"的命令式接口，画图只需要 `plt.subplots`、`ax.imshow` 这类短调用。
- 删掉会怎样：后面 Cell 8 用到 `plt.subplots` 会直接 `NameError: name 'plt' is not defined`。

**第 6 行 `import torch`**

- 导入 PyTorch 主包。注意这里**没有 `as`**，因为 `torch` 本身就够短。
- 本 notebook 里它主要用来调 `torch.utils.data.DataLoader`。
- 删掉会怎样：Cell 3 的 `torch.utils.data.DataLoader(...)` 会报 `NameError`。

**第 7 行 `from torchvision import datasets, transforms`**

- `torchvision` 是 PyTorch 官方的**计算机视觉工具包**，可以理解为"图像版的补充库"。它不随 `torch` 一起导入，要单独 import。
- `datasets`：里面装着各种现成数据集类，本课用的是 `datasets.ImageFolder`、`datasets.CIFAR10`。
- `transforms`：里面装着各种图像变换函数/类，如 `Resize`、`ToTensor`。
- 为什么用 `from ... import a, b` 而不是 `import torchvision`：后面就能直接写 `transforms.Compose(...)`，少敲一层前缀。
- 删掉会怎样：Cell 3 里 `transforms.Compose` 和 `datasets.ImageFolder` 全部 `NameError`。

**第 9 行 `import helper`**

- 导入课程作者写的本地脚本 `helper.py`（和你这个 notebook 放在同一目录）。
- 这个文件里封装了 `imshow()` 等小工具，专门用来在 notebook 里显示**CHW 格式的 torch 张量**（matplotlib 原生只认 HWC 的 numpy 数组，直接塞张量会报错或显示错乱）。
- 为什么这么写：把"张量 → 可显示图片"的脏活藏起来，让你专注数据管道。
- 删掉会怎样：Cell 4 的 `helper.imshow(...)` 报 `NameError`。如果 `helper.py` 文件不在同目录，会报 `ModuleNotFoundError: No module named 'helper'`——这是新手最常见的第一道坎。

### ③【张量形状变化】

本 cell 没有任何数据流动，**形状不变**。

### ④【新手易错点】

1. **把魔法命令当 Python 抄进 `.py` 文件**：`%matplotlib inline` 在普通脚本里是语法错误。脚本里应该用 `matplotlib.use('Agg')` 之类的方式。
2. **`import helper` 报错**：不是 pip 包，是课程附带的单个 `.py` 文件，必须和 notebook 在同一目录（或手动加进 `sys.path`）。网上搜"pip install helper"是没用的。
3. **`torchvision` 装了但 import 失败**：通常是 torch 和 torchvision 版本不匹配（二者必须成对安装）。报错信息里通常带 `operator torchvision::nms does not exist` 之类，就是版本错配。

---

## Cell 2（markdown）· 全文核心：ImageFolder / Transforms / DataLoader

**英文原文**

> The easiest way to load image data is with `datasets.ImageFolder` from `torchvision` ([documentation](http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder)). In general you'll use `ImageFolder` like so:
>
> ```python
> dataset = datasets.ImageFolder('path/to/data', transform=transform)
> ```
>
> where `'path/to/data'` is the file path to the data directory and `transform` is a sequence of processing steps built with the [`transforms`](http://pytorch.org/docs/master/torchvision/transforms.html) module from `torchvision`. ImageFolder expects the files and directories to be constructed like so:
> ```
> root/dog/xxx.png
> root/dog/xxy.png
> root/dog/xxz.png
>
> root/cat/123.png
> root/cat/nsdf3.png
> root/cat/asd932_.png
> ```
>
> where each class has it's own directory (`cat` and `dog`) for the images. The images are then labeled with the class taken from the directory name. So here, the image `123.png` would be loaded with the class label `cat`. You can download the dataset already structured like this [from here](https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip). I've also split it into a training set and test set.
>
> ### Transforms
>
> When you load in the data with `ImageFolder`, you'll need to define some transforms. For example, the images are different sizes but we'll need them to all be the same size for training. You can either resize them with `transforms.Resize()` or crop with `transforms.CenterCrop()`, `transforms.RandomResizedCrop()`, etc. We'll also need to convert the images to PyTorch tensors with `transforms.ToTensor()`. Typically you'll combine these transforms into a pipeline with `transforms.Compose()`, which accepts a list of transforms and runs them in sequence. It looks something like this to scale, then crop, then convert to a tensor:
>
> ```python
> transform = transforms.Compose([transforms.Resize(255),
>                                 transforms.CenterCrop(224),
>                                 transforms.ToTensor()])
> ```
>
> There are plenty of transforms available, I'll cover more in a bit and you can read through the [documentation](http://pytorch.org/docs/master/torchvision/transforms.html).
>
> ### Data Loaders
>
> With the `ImageFolder` loaded, you have to pass it to a [`DataLoader`](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader). The `DataLoader` takes a dataset (such as you would get from `ImageFolder`) and returns batches of images and the corresponding labels. You can set various parameters like the batch size and if the data is shuffled after each epoch.
>
> ```python
> dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
> ```
>
> Here `dataloader` is a [generator](https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/). To get data out of it, you need to loop through it or convert it to an iterator and call `next()`.
>
> ```python
> # Looping through it, get a batch on each loop 
> for images, labels in dataloader:
>     pass
>
> # Get one batch
> images, labels = next(iter(dataloader))
> ```
>
> >**Exercise:** Load images from the `Cat_Dog_data/train` folder, define a few transforms, then build the dataloader.

**中文翻译**

> 加载图像数据最简单的方式是用 `torchvision` 里的 `datasets.ImageFolder`（[文档](http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder)）。一般来说你会这样使用 `ImageFolder`：
>
> ```python
> dataset = datasets.ImageFolder('path/to/data', transform=transform)
> ```
>
> 其中 `'path/to/data'` 是数据目录的文件路径，而 `transform` 是一系列处理步骤——由 `torchvision` 的 `transforms` 模块构建。`ImageFolder` 期望文件和目录按下面的方式组织：
> ```
> root/dog/xxx.png
> root/dog/xxy.png
> root/dog/xxz.png
>
> root/cat/123.png
> root/cat/nsdf3.png
> root/cat/asd932_.png
> ```
>
> 也就是每个类别有自己的目录（`cat` 和 `dog`）。图像随后会以它所在目录名作为类别标签被加载。所以在这个例子里，图片 `123.png` 会被加载并打上类别标签 `cat`。你可以从[这里](https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip)下载已经按这种结构组织好的数据集。我也已经把它划分成了训练集和测试集。
>
> ### 变换（Transforms）
>
> 当你用 `ImageFolder` 加载数据时，需要定义一些变换。例如，这些图像尺寸各不相同，但训练时我们需要它们全都一样大。你可以用 `transforms.Resize()` 缩放它们，也可以用 `transforms.CenterCrop()`、`transforms.RandomResizedCrop()` 等来裁剪。我们还需要用 `transforms.ToTensor()` 把图像转换成 PyTorch 张量。通常你会用 `transforms.Compose()` 把这些变换组合成一条流水线，它接受一个变换列表，并按顺序依次执行。大致就是这样：先缩放、再裁剪、最后转成张量：
>
> ```python
> transform = transforms.Compose([transforms.Resize(255),
>                                 transforms.CenterCrop(224),
>                                 transforms.ToTensor()])
> ```
>
> 可用的变换非常多，我稍后会再介绍一些，你也可以翻阅[文档](http://pytorch.org/docs/master/torchvision/transforms.html)。
>
> ### 数据加载器（Data Loaders）
>
> `ImageFolder` 加载好之后，你必须把它传给一个 [`DataLoader`](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader)。`DataLoader` 接受一个数据集（比如 `ImageFolder` 的产物），并返回一批一批的图像以及对应的标签。你可以设置各种参数，比如批大小（batch size），以及每个 epoch 之后是否打乱数据。
>
> ```python
> dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
> ```
>
> 这里的 `dataloader` 是一个[生成器](https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/)。要从里面取数据，你需要循环遍历它，或者把它转成迭代器再调用 `next()`。
>
> ```python
> # 循环遍历，每次循环拿到一个 batch
> for images, labels in dataloader:
>     pass
>
> # 取一个 batch
> images, labels = next(iter(dataloader))
> ```
>
> >**练习：** 从 `Cat_Dog_data/train` 文件夹加载图像，定义几个变换，然后构建 dataloader。

**这段的重点**：

- **`ImageFolder` 不需要标签文件**，它靠"子目录名 = 类别名"自动打标签；这是本 notebook 唯一需要记住的约定。
- **`Compose` 是一条流水线**，列表顺序 = 执行顺序，顺序写错（比如把 `ToTensor` 放最前面）会导致后面的变换吃不到 PIL 图像而报错。
- **`DataLoader` 是"按批取数的生成器"**，不是列表，所以不能 `dataloader[0]`。
- 这三者构成 `dataset（一条样本）→ transform（单样本加工）→ dataloader（批量 + 打乱）` 的分层结构。

---

## Cell 3（code）· 构建训练集 DataLoader

```python
data_dir = 'Cat_Dog_data/train'

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
```

### ①【这个 cell 在干什么】

定义一条"缩放 → 中心裁剪 → 转张量"的预处理流水线，用 `ImageFolder` 扫描训练目录生成数据集，再包成每批 32 张、随机打乱的 `DataLoader`。

### ②【逐行拆解】

**第 1 行 `data_dir = 'Cat_Dog_data/train'`**

- 只是一个字符串变量，存目录路径。
- 为什么单独拎出来：后面（Cell 7）还要用 `data_dir + '/test'`，避免路径硬编码到处复制。
- 相对路径的参照系是**notebook 的工作目录**（通常就是 `.ipynb` 所在目录），不是 `.py` 文件的位置。这是新手最容易踩的坑之一。
- 删掉会怎样：下面 `ImageFolder(data_dir, ...)` 报 `NameError`。

**第 3–5 行 `transform = transforms.Compose([...])`**

逐层拆开看：

- `transforms.Compose(list)`：把多个变换串成一条管道。它本身是**可调用对象**，`transform(pil_image)` 会依次执行列表里的每一步。注意它不会立刻执行——只是"登记"了一个处理方案，真正执行是在 `ImageFolder` 取样本（`dataset[i]`）时。
- `transforms.Resize(255)`：
  - 参数是**一个整数**时，含义是"把**较短的那条边**缩放到 255，长边按原比例跟着变"。
  - 例：一张 400×300（宽×高）的图 → 变成 340×255（短边 255）。
  - 所以这一步输出的**尺寸仍然不固定**（取决于原图长宽比），形状记为 `[3, 255, W']`，其中 `W' ≥ 255`。
  - 为什么是 255 而不是 224：先留一点余量，下一步才有东西可裁。
- `transforms.CenterCrop(224)`：
  - 参数是一个整数时，表示裁成 **224×224 的正方形**，从图像**正中间**裁。
  - 这一步同时完成两件事：**统一尺寸**（不管上一步是 255×340 还是 255×255，出来都是 224×224）和**去除边缘干扰**。
  - 至此 PIL 图像形状固定为 `[3, 224, 224]`（HWC 语义上是 224×224×3）。
- `transforms.ToTensor()`：
  - 无参数。做三件事：
    1. 把 PIL Image / numpy 数组转成 `torch.FloatTensor`；
    2. **维度顺序从 HWC 换成 CHW**（PyTorch 卷积层要求通道在前）；
    3. **数值从 0~255 的 uint8 归一化到 0.0~1.0 的 float32**（除以 255）。
  - 这一步是**必须的**，没有它后面网络没法吃数据（卷积层不接受 PIL 对象）。
- 为什么用 `Resize` + `CenterCrop` 而不是直接 `Resize((224, 224))`：直接 resize 到正方形会**拉伸变形**（把 400×300 压成 224×224，猫会被压扁），影响特征；先等比缩放再中心裁剪能保持物体几何形状不变。
- 删掉 `Compose` 里的 `ToTensor` 会怎样：`dataset[0]` 返回的还是 PIL 图像，`DataLoader` 拼 batch 时会报错（无法把 PIL 对象 stack 成张量），训练直接跑不起来。

**第 6 行 `dataset = datasets.ImageFolder(data_dir, transform=transform)`**

- `ImageFolder(root, transform=None)`：
  - `root`：数据根目录，**必须**满足"每个子目录是一个类别"的结构。
  - `transform`：对**每张图**在读出后立刻施加的变换（就是我们上面那条管道）。
- 内部干了什么（了解即可，但很关键）：
  1. 遍历 `root` 下所有子目录，把子目录名排序后建立 `class_to_idx` 映射。**排序是字母序**，所以本项目得到 `{'cat': 0, 'dog': 1}`。
  2. 递归收集所有图片文件，生成 `(文件路径, 类别索引)` 的列表，存在 `dataset.samples` 里。
  3. 提供 `dataset[i]` → 返回 `(tensor, label)`；`len(dataset)` → 样本总数。
- 为什么用它：省掉了"自己写 for 循环读文件夹 + 自己维护 label 字典"的几十行代码。
- 删掉会怎样：`dataloader` 那行没有数据集可用，直接 `NameError`。
- **重要副作用**：`ImageFolder` 在构造时**不读图片内容，只扫路径**。所以这一步很快，即使有 1 万多张图。

**第 7 行 `dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)`**

- `torch.utils.data.DataLoader` 的参数逐个说：
  - `dataset`：任何支持 `len()` 和 `dataset[i] -> (x, y)` 的对象（叫"map-style dataset"）。`ImageFolder` 正好符合。
  - `batch_size=32`：每批取 32 条样本，`collate_fn` 会把 32 个 `[3,224,224]` 沿新维度堆成 `[32,3,224,224]`。
    - 为什么是 32：显存/内存与梯度稳定性的折中。太小（如 1）梯度噪声大、训练慢；太大（如 512）显存爆、且可能降低泛化。32/64 是最常见的默认值。
  - `shuffle=True`：每个 epoch 开始时**重新随机打乱样本顺序**。
    - 为什么必须开：`ImageFolder` 是按路径排序的，而本项目训练集是 `cat/` 在前 `dog/` 在后（两类各 11250 张，数量平衡但顺序是聚在一起的）。不打乱的话，一个 batch 里的 32 张会全部来自同一类，梯度方向严重偏斜，模型会在两类之间"来回震荡"，收敛很差。
    - 对**测试集**则通常设 `shuffle=False`（本 notebook Cell 7 就没写，默认 `False`），因为评估只关心整体指标，顺序无所谓，保持可复现更好。
  - 还有两个常用参数本行没写：
    - `num_workers`：用几个子进程读数据，默认 `0`（主进程读）。Windows 上设大于 0 需要 `if __name__ == '__main__'` 保护，所以教学代码通常留 0。
    - `drop_last`：是否丢掉最后不满一批的样本，默认 `False`。
- 删掉会怎样：后面 `next(iter(dataloader))` 报 `NameError`；而且你就得自己手写 batch 循环。

### ③【张量形状变化】

```
磁盘 JPG（尺寸不定，例如 400×300×3, uint8）
   │  ImageFolder 读取 → PIL Image
   ▼
PIL [H, W, C] = [300, 400, 3]（示例）
   │  Resize(255)：短边→255
   ▼
PIL [255, 340, 3]（长边随之变化，尺寸不固定）
   │  CenterCrop(224)
   ▼
PIL [224, 224, 3]
   │  ToTensor()：HWC→CHW，/255
   ▼
Tensor [3, 224, 224]，float32，值域 [0,1]
   │  DataLoader(batch_size=32) 堆叠
   ▼
一个 batch：images [32, 3, 224, 224]，labels [32]（int64，取值 0/1）
```

整个训练集规模：`len(dataset) = 22500`（猫 11250 + 狗 11250），因此一个 epoch 有 `22500 // 32 = 703` 个 batch（最后一个 batch 只有 4 张）。

### ④【新手易错点】

1. **目录结构不对**。如果写成 `data_dir = 'Cat_Dog_data/train/cat'`，`ImageFolder` 会把每张图当成一个"类别"（因为找不到子目录），或者直接报 `FileNotFoundError: Found no valid file for the classes ...`。正确层级是"根目录/类别目录/图片"。
2. **类别编号不是你以为的顺序**。`class_to_idx` 是**字母序**：`cat=0, dog=1`。如果你自己按"猫=1, 狗=0"去解读预测结果，准确率会看起来只有 50%（其实是全对）。用 `dataset.class_to_idx` 打印确认。
3. **把 `ToTensor` 放在 `Compose` 的第一位**。`Resize`/`CenterCrop` 期望输入是 PIL 图像，喂张量会报 `TypeError`。顺序永远是"几何变换在前，`ToTensor` 在后，`Normalize` 最后"。
4. **`shuffle=True` 忘了加**。代码不报错，但 loss 曲线会剧烈震荡、准确率上不去，而且极难定位原因。
5. **`batch_size` 与显存**。224×224×3 的图比 MNIST 大 64 倍，32 张一批在 CPU 上训练就已经很慢；迁移学习时（Part 8）作者把 batch 调到 64，是因为只训练分类头、反向传播很轻。
6. **混淆"dataset 长度"和"batch 数量"**。`len(dataset)=22500`，`len(dataloader)=703`。算平均 loss 时除以哪个，结果差 32 倍。

---

## Cell 4（code）· 取一个 batch 并显示第一张图

```python
# Run this to test your data loader
images, labels = next(iter(dataloader))
helper.imshow(images[0], normalize=False)
```

### ①【这个 cell 在干什么】

从 dataloader 里取出第一个 batch，把其中第 0 张图显示出来，用来肉眼验证"数据管道没写错"。

### ②【逐行拆解】

**第 1 行（注释）** `# Run this to test your data loader`

- 纯注释，作者提示"跑这一格来检查你的 dataloader"。

**第 2 行 `images, labels = next(iter(dataloader))`**

- `iter(dataloader)`：调用 `DataLoader.__iter__()`，创建一个**迭代器**。每次 `iter()` 都会新建一个迭代器（并且因为 `shuffle=True`，每次拿到的 batch 内容不同）。
- `next(迭代器)`：取出**下一个**元素。因为迭代器是新建的，这里拿到的是**第 1 个 batch**。
- 解包 `images, labels = ...`：`DataLoader` 每个元素是 `(图像batch, 标签batch)` 二元组，直接解包成两个变量。
- 为什么不用 `for` 循环：这里只想"偷看"一批，不需要遍历全部。
- 删掉会怎样：`images` 未定义，下一行 `NameError`。
- **版本注意**：老教程常写 `dataloader.next()`（本例的 CIFAR notebook 里就是这么写的），PyTorch 1.5 之后 `DataLoader` 已经没有 `.next()` 方法，必须写 `next(iter(dataloader))`。

**第 3 行 `helper.imshow(images[0], normalize=False)`**

- `images[0]`：取 batch 里第 0 张，形状 `[3, 224, 224]`（CHW）。注意**取下标是去掉 batch 维**，不是取通道。
- `helper.imshow(img, ax=None, normalize=True)`：课程自定义函数，内部大致做三件事：
  1. 如果 `normalize=True`，按 ImageNet 的 mean/std 把已经标准化过的张量"反标准化"回 0~1；
  2. `img.numpy().transpose((1,2,0))` 把 CHW 转成 matplotlib 需要的 HWC；
  3. `ax.imshow(...)` 显示。
- `normalize=False` 的原因：本 cell 的 transform **没有做 `Normalize`**，像素值本来就在 0~1。如果这里写 `True`，helper 会去减均值除标准差，图会变成一片诡异的灰绿色/过曝。
- 为什么测试数据管道要"看图"：形状对不对（比如把 H 和 W 搞反、通道错位成 BGR）从代码上看不出来，但一眼看图就能发现"猫变成蓝精灵"。
- 删掉会怎样：不报错，但你失去了唯一一次"肉眼验收"。

### ③【张量形状变化】

```
dataloader（batch_size=32）
   │ next(iter(...))
   ▼
images [32, 3, 224, 224] float32, 值域 [0,1]
labels [32] int64
   │ images[0]（去掉 batch 维）
   ▼
[3, 224, 224]
   │ helper 内部 transpose(1,2,0)
   ▼
[224, 224, 3] → 交给 matplotlib 显示
```

### ④【新手易错点】

1. **`images[0]` 和 `images[:, 0]` 搞混**。前者是"第 0 张图"（`[3,224,224]`），后者是"所有图的第 0 个通道（红通道）"（`[32,224,224]`）。显示时用错会得到一张 32 张图叠在一起的怪东西。
2. **`normalize` 参数和 transform 里的 `Normalize` 是两回事**。前者是"显示时反标准化"，后者是"训练前标准化"。只要 transform 里加了 `Normalize`，显示时就该设 `normalize=True`；反之设 `False`。这个对应关系弄反，是最常见的"图颜色不对"来源。
3. **`iter()` 每次调用都会重置**。如果写成 `next(iter(dataloader))` 两次，你拿到的是两个**独立的新迭代器各自的第一批**，不是前两批。想连续取多批，必须先把迭代器存下来：`it = iter(dataloader); a = next(it); b = next(it)`。
4. **在 Windows 上 `num_workers>0` 且没加 `if __name__ == '__main__'` 会卡死/报错**——本 notebook 没设，所以你暂时不会遇到，但以后自己写脚本要记住。

---

## Cell 5（markdown）

**英文原文**

> If you loaded the data correctly, you should see something like this (your image will be different):
>
> `<img src='assets/cat_cropped.png' width=244>`

**中文翻译**

> 如果你正确地加载了数据，你应该会看到类似这样的画面（你的图像会不一样）：
>
> `<img src='assets/cat_cropped.png' width=244>`

**这段的重点**：这是一张"验收标准图"——看到一张正常的、正方形的、颜色正常的猫图，就说明 `Resize + CenterCrop + ToTensor` 三步都对了。

---

## Cell 6（markdown）· 数据增强（Data Augmentation）

**英文原文**

> ## Data Augmentation
>
> A common strategy for training neural networks is to introduce randomness in the input data itself. For example, you can randomly rotate, mirror, scale, and/or crop your images during training. This will help your network generalize as it's seeing the same images but in different locations, with different sizes, in different orientations, etc.
>
> To randomly rotate, scale and crop, then flip your images you would define your transforms like this:
>
> ```python
> train_transforms = transforms.Compose([transforms.RandomRotation(30),
>                                        transforms.RandomResizedCrop(224),
>                                        transforms.RandomHorizontalFlip(),
>                                        transforms.ToTensor(),
>                                        transforms.Normalize([0.5, 0.5, 0.5], 
>                                                             [0.5, 0.5, 0.5])])
> ```
>
> You'll also typically want to normalize images with `transforms.Normalize`. You pass in a list of means and list of standard deviations, then the color channels are normalized like so
>
> ```input[channel] = (input[channel] - mean[channel]) / std[channel]```
>
> Subtracting `mean` centers the data around zero and dividing by `std` squishes the values to be between -1 and 1. Normalizing helps keep the network weights near zero which in turn makes backpropagation more stable. Without normalization, networks will tend to fail to learn.
>
> You can find a list of all [the available transforms here](http://pytorch.org/docs/0.3.0/torchvision/transforms.html). When you're testing however, you'll want to use images that aren't altered other than normalizing. So, for validation/test images, you'll typically just resize and crop.
>
> >**Exercise:** Define transforms for training data and testing data below. Leave off normalization for now.

**中文翻译**

> ## 数据增强
>
> 训练神经网络的一个常见策略是：在输入数据本身引入随机性。例如，你可以在训练时随机旋转、镜像、缩放和/或裁剪图像。这会帮助你的网络更好地泛化，因为网络看到的是同一批图像，但它们出现在不同的位置、不同的尺寸、不同的朝向上。
>
> 要想随机旋转、缩放、裁剪，然后再翻转图像，你可以这样定义变换：
>
> ```python
> train_transforms = transforms.Compose([transforms.RandomRotation(30),
>                                        transforms.RandomResizedCrop(224),
>                                        transforms.RandomHorizontalFlip(),
>                                        transforms.ToTensor(),
>                                        transforms.Normalize([0.5, 0.5, 0.5], 
>                                                             [0.5, 0.5, 0.5])])
> ```
>
> 你通常还会用 `transforms.Normalize` 对图像做标准化。你传入一个均值列表和一个标准差列表，然后各个颜色通道会按下面的方式被标准化：
>
> ```input[channel] = (input[channel] - mean[channel]) / std[channel]```
>
> 减去 `mean` 会让数据以 0 为中心，除以 `std` 会把数值压缩到 -1 到 1 之间。标准化有助于让网络权重保持在 0 附近，从而让反向传播更稳定。如果不做标准化，网络往往会学不起来。
>
> 你可以从[这里](http://pytorch.org/docs/0.3.0/torchvision/transforms.html)找到所有可用变换的列表。不过在做测试时，你希望使用的图像除了标准化之外不要有任何其他改动。所以对于验证/测试图像，通常只需要缩放和裁剪。
>
> >**练习：** 在下面为训练数据和测试数据分别定义变换。现在先不要加标准化。

**这段的重点**：

- **数据增强 = 训练时故意"折腾"输入**（旋转/翻转/随机裁剪），让模型见过更多变化，从而**抑制过拟合、提升泛化**。它只在训练时用，测试时必须关掉。
- **训练集用"随机"变换，测试集用"确定"变换**，这是全局最重要的原则。
- **`Normalize` 是另一件事**，不属于增强：它是把像素分布拉到 0 均值附近，让训练稳定。它的参数不是随便填的，要和预训练模型训练时用的一致（Part 8 会看到 ImageNet 的三个固定数值）。
- 原文说"没有标准化网络会学不起来"有点夸张，但在**深网络 + 大学习率**时确实很容易发散，对新手建议直接照做。
- 注意原文的 `Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])` 是针对 0~1 输入的一个"偷懒版"（把任意图都拉成均值 0.5、标准差 0.5），它**不等于** ImageNet 的真实统计量。项目里用预训练模型时必须换成 Part 8 的数值。

---

## Cell 7（code）· 训练 / 测试两套 transforms

```python
data_dir = 'Cat_Dog_data'

# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()]) 

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])


# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
```

### ①【这个 cell 在干什么】

把数据管道升级成"训练集带随机增强、测试集只做确定性缩放裁剪"的双管道，并分别建好两个 DataLoader。

### ②【逐行拆解】

**第 1 行 `data_dir = 'Cat_Dog_data'`**

- 这次存的是**父目录**，因为下面要拼 `/train` 和 `/test`。
- 为什么改写：一行变量供两处复用，改数据集位置时只改这一处。

**第 4–7 行 `train_transforms = transforms.Compose([...])`**

- `transforms.RandomRotation(30)`：
  - 参数 `30` 单位是**度**，表示在 `[-30°, +30°]` 之间**均匀随机**取一个角度旋转。
  - 角度越大增强越强，但也越可能把目标转出画面（猫被转出边界后角落填黑）。30° 是保守的常用值。
  - 每取一次样本就重新随机一次 —— 也就是说同一张图在第 1 个 epoch 和第 2 个 epoch 被转的角度不同，这是增强生效的关键。
  - 删掉：模型对"倾斜的猫"鲁棒性下降，但代码不会报错。
- `transforms.RandomResizedCrop(224)`：
  - 三个默认参数：`scale=(0.08, 1.0)`、`ratio=(3/4, 4/3)`、`interpolation`。
  - `scale`：先随机裁出原图面积的 **8%~100%** 的一块（模拟"物体远近/大小不同"）。
  - `ratio`：随机取长宽比在 0.75~1.33 之间（模拟"物体被压扁或拉长"）。
  - 最后把这块裁剪**缩放**到 224×224。
  - 为什么不用 `Resize+CenterCrop`：那是确定性的、每张图永远裁中心；`RandomResizedCrop` 每次裁的位置/大小都不同，等价于"凭空造出新样本"。
  - 副作用：`scale` 下限 0.08 有时会把猫裁得只剩一只耳朵，属于"过度增强"，会拖慢收敛。
- `transforms.RandomHorizontalFlip()`：
  - 默认 `p=0.5`，以 50% 概率**左右镜像**。
  - 为什么猫狗可以翻转：猫朝左朝右都是猫，标签不变。但**文字识别、方向盘角度回归**这类任务翻转会改变语义，绝不能用。
  - 这是性价比最高的增强，几乎零成本。
- `transforms.ToTensor()`：同上，HWC→CHW、/255。**必须在所有几何变换之后**。
- 注意：作者这个 cell **故意没写 `Normalize`**（练习要求"先不加"），所以这里像素仍是 0~1。

**第 9–11 行 `test_transforms = transforms.Compose([...])`**

- `Resize(255)` + `CenterCrop(224)`：和 Cell 3 完全一样的确定性流程。
- 为什么不加随机旋转/裁剪：评估必须**可复现、可比较**。如果测试图每次都随机变，两次评估的准确率会抖动，你无法判断模型是不是真的变好了。
- 为什么仍然要 Resize+Crop：模型只接受 224×224 输入，尺寸必须统一。
- 为什么这里也没有 Normalize：为了和训练集保持**同一套数值范围**（都还在 0~1）。如果训练集 normalize 了而测试集没有，输入分布不一致，准确率会崩——这是新手最致命的错误之一。

**第 14–15 行 `train_data = datasets.ImageFolder(...)` / `test_data = ...`**

- `data_dir + '/train'` 字符串拼接 → `'Cat_Dog_data/train'`。
- 训练集挂 `train_transforms`，测试集挂 `test_transforms`。**一个数据集配一套 transform**，不要共用。
- 注意两个 `ImageFolder` 目录不同，所以是两个**独立**的数据集对象（不是同一份数据切分）。本 notebook 的 train/test 是作者预先切好的文件夹。

**第 17–18 行 `trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)` / `testloader = ...`**

- **这里没有写 `shuffle=True`**！训练集用的是默认 `shuffle=False`。
- 后果：`ImageFolder` 按路径排序，`train/cat/*` 全部排在 `train/dog/*` 前面，于是前 `11250/32 ≈ 352` 个 batch 全是猫，后面 352 个 batch 全是狗（两类数量相等，所以刚好对半分）。模型会先"只学猫"再"忘掉猫去学狗"，训练极不稳定。
- 这是原 notebook 的一个**疏漏**（对比 Cell 3 是有 `shuffle=True` 的）。你在自己项目里必须加上。
- `testloader` 不写 `shuffle` 是对的（评估不需要打乱）。

### ③【张量形状变化】

```
训练分支（每次取样本都不同）：
原图 [H₀, W₀, 3]（不定）
  → RandomRotation(30)        → [H₁, W₁, 3]（尺寸不变，角落填补）
  → RandomResizedCrop(224)    → [224, 224, 3]（裁+缩，比例随机）
  → RandomHorizontalFlip()    → [224, 224, 3]
  → ToTensor()                → [3, 224, 224]
  → DataLoader(32)            → [32, 3, 224, 224]

测试分支（每次取样本都一样）：
原图 [H₀, W₀, 3]
  → Resize(255)               → [255, W', 3]
  → CenterCrop(224)           → [224, 224, 3]
  → ToTensor()                → [3, 224, 224]
  → DataLoader(32)            → [32, 3, 224, 224]
```

规模：`len(train_data)=22500` → 703 个 batch；`len(test_data)=2500` → 79 个 batch（最后一个 4 张）。

### ④【新手易错点】

1. **训练集忘了 `shuffle=True`**（本例就是）。表现是 loss 曲线呈"阶梯状"忽高忽低，或者准确率长时间卡在 50%。**这是本 cell 最值得记的一条。**
2. **测试集也加了随机增强**。准确率会在 90%~95% 之间乱跳，且无法复现。
3. **训练/测试的 Normalize 不一致**。要么都加、要么都不加，参数还必须完全相同。
4. **`RandomResizedCrop` 与 `Resize` 混着用**。两者都做"缩放"，串在一起会二次失真；正常是"训练用 RandomResizedCrop，测试用 Resize+CenterCrop"，二选一。
5. **`RandomRotation` 的角度单位**。是度，不是弧度，也不是"比例"。
6. **以为增强能"增加数据量"**。数据集样本数不变（还是 22500），增强只是让每个 epoch 看到的"版本"不同。

---

## Cell 8（code）· 四宫格预览测试集图像

```python
# change this to the trainloader or testloader 
data_iter = iter(testloader)

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,4), ncols=4)
for ii in range(4):
    ax = axes[ii]
    helper.imshow(images[ii], ax=ax, normalize=False)
```

### ①【这个 cell 在干什么】

取测试集的一个 batch，用 1 行 4 列的子图把前 4 张并排画出来，用来对比"训练增强后的样子"和"测试时的样子"。

### ②【逐行拆解】

**第 1 行（注释）** `# change this to the trainloader or testloader`

- 提示：把下面的 `testloader` 换成 `trainloader`，就能看到加了随机增强的训练图长什么样。这是作者留的"开关"。

**第 2 行 `data_iter = iter(testloader)`**

- 显式把迭代器**存进变量**。
- 为什么这次要存：下面只取一次，存不存都行；但养成"要连续取多批就存下来"的习惯很重要（Cell 4 的坑）。
- 删掉会怎样：下一行 `next(data_iter)` 中 `data_iter` 未定义。

**第 4 行 `images, labels = next(data_iter)`**

- 取第一个 batch：`images [32,3,224,224]`，`labels [32]`。
- 注意这里 `labels` 取了却**没用**（本 cell 不显示标签文字）。不影响运行，属于冗余变量。

**第 5 行 `fig, axes = plt.subplots(figsize=(10,4), ncols=4)`**

- `plt.subplots(nrows=1, ncols=4, figsize=(10,4))`：
  - `nrows` 默认 1，所以是 1×4 的网格。
  - `figsize=(10,4)`：整张画布 10 英寸宽、4 英寸高（英寸，不是像素；retina 下会 ×2）。
  - 返回值：`fig` 是整张画布对象，`axes` 是**长度为 4 的 ndarray**（因为有 4 列；如果只有 1 个子图，返回的是单个 Axes 而不是数组）。
- 删掉会怎样：后面 `axes[ii]` 报 `NameError`。

**第 6–8 行 `for ii in range(4): ax = axes[ii]; helper.imshow(images[ii], ax=ax, normalize=False)`**

- `range(4)`：只想画前 4 张，不是全部 32 张（32 张挤在 1×4 网格里会重叠）。
- `axes[ii]`：第 `ii` 个子图对象（0 基）。
- `helper.imshow(images[ii], ax=ax, ...)`：
  - 第二个参数 `ax=ax` 让它**画到指定的子图上**；不传的话 helper 会自己新建一张图，结果就是 4 张独立的图，而不是并排。
  - `normalize=False`：同样因为本 pipeline 没有 `Normalize`。
- 删掉 `ax=ax` 会怎样：不报错，但 4 张图各占一个 figure，看起来像"只画了最后一张"（因为 subplot 是空的）。

### ③【张量形状变化】

```
testloader
  → iter / next → images [32, 3, 224, 224], labels [32]
  → images[ii]  → [3, 224, 224]
  → helper 内部 transpose(1,2,0) → [224, 224, 3]
  → 绘制到 axes[ii]（1×4 网格中的第 ii 格）
```

### ④【新手易错点】

1. **`axes` 的维度随网格形状变化**。`ncols=4` 时 `axes` 是 1 维数组；如果写成 `nrows=2, ncols=2`，`axes` 会变成 2×2 的二维数组，此时 `axes[ii]` 拿到的是"一行"而不是"一个子图"，必须写 `axes.flat[ii]` 或 `axes[i//2][i%2]`。
2. **忘了 `ax=ax`**。4 张图不并排，且子图区留白。
3. **`normalize` 与 transform 不对应**（同 Cell 4）。
4. **把 `trainloader` 和 `testloader` 搞混**：看训练图时，同一张图每次运行角度都不同（因为随机增强）；看测试图时，同一张图永远一样。这正好是判断"增强有没有生效"的小技巧。

---

## Cell 9（markdown）

**英文原文**

> Your transformed images should look something like this.
>
> `<center>Training examples:</center>`
> `<img src='assets/train_examples.png' width=500px>`
>
> `<center>Testing examples:</center>`
> `<img src='assets/test_examples.png' width=500px>`

**中文翻译**

> 你变换后的图像应该看起来像这样。
>
> `<center>训练样本示例：</center>`
> `<img src='assets/train_examples.png' width=500px>`
>
> `<center>测试样本示例：</center>`
> `<img src='assets/test_examples.png' width=500px>`

**这段的重点**：训练图会"歪、缺角、大小不一"（增强生效），测试图则"端正、居中、统一"（确定性流程）。两者视觉差异就是你判断 transform 有没有配对的地方。

---

## Cell 10（markdown）· 收尾：为什么必须用预训练模型

**英文原文**

> At this point you should be able to load data for training and testing. Now, you should try building a network that can classify cats vs dogs. This is quite a bit more complicated than before with the MNIST and Fashion-MNIST datasets. To be honest, you probably won't get it to work with a fully-connected network, no matter how deep. These images have three color channels and at a higher resolution (so far you've seen 28x28 images which are tiny).
>
> In the next part, I'll show you how to use a pre-trained network to build a model that can actually solve this problem.

**中文翻译**

> 到这里，你应该已经能为训练和测试加载数据了。接下来，你应该尝试构建一个能对猫和狗进行分类的网络。这比之前用 MNIST 和 Fashion-MNIST 数据集要复杂不少。说实话，无论你的全连接网络有多深，你大概都没法让它真正work。这些图像有三个颜色通道，分辨率也更高（到目前为止你见到的都是 28×28 这种极小的图）。
>
> 在下一部分，我会展示如何使用预训练网络来构建一个真正能解决这个问题的模型。

**这段的重点**：

- **全连接网络（MLP）在真实图像上会失效**，原因有两个：参数量爆炸（224×224×3 的输入，第一层就要几百万参数）和平移不变性缺失（猫往右挪 10 像素，MLP 就把它当全新样本）。
- 这直接引出 Part 8 的**迁移学习**：用别人在大数据上训练好的卷积层当"特征提取器"。
- 顺带记住这个数字关系：**输入分辨率 224 是 ImageNet 预训练模型的"标准口径"**，这就是前面为什么死磕 224。

---

## Cell 11（code）· 可选作业

```python
# Optional TODO: Attempt to build a network to classify cats vs dogs from this dataset
```

### ①【这个 cell 在干什么】

一个空的"可选练习"占位符，鼓励你从零写一个 CNN 来跑猫狗分类。

### ②【逐行拆解】

- 整个 cell 只有一行注释，**没有任何可执行代码**，运行它什么也不会发生（不报错、无输出）。
- `Optional TODO`：作者标注"可做可不做"。他的判断是"你大概率做不好"，因为从零训练的 CNN 在 224×224 的猫狗上，如果不做大量调参，准确率通常只有 70%~80%，很难达到项目要求的 80%+。

### ③【张量形状变化】

无可执行代码，**无形状变化**。

### ④【新手易错点】

1. **以为这是必做作业**。它不影响后面 Part 8 的进行。
2. **真去写了却直接照抄 28×28 的 MNIST 网络**。输入变成 224，第一个全连接层的输入维度必须重算（`3*224*224` 而不是 `1*28*28`），否则报 `mat1 and mat2 shapes cannot be multiplied`。
3. **在 CPU 上硬训**。224×224 的图 + 卷积，CPU 上一个 epoch 可能要几十分钟，容易让人误以为"代码卡死了"。

---

## 本 notebook 小结（复习用）

| 关键词 | 一句话记忆 |
|---|---|
| `ImageFolder` | 目录名即标签，按字母序编号（cat=0, dog=1） |
| `Compose` | 变换流水线，顺序敏感：几何 → `ToTensor` → `Normalize` |
| `Resize(255)+CenterCrop(224)` | 先等比缩放保形状，再裁中心统一尺寸（测试用） |
| `RandomResizedCrop/Rotation/Flip` | 训练专用的随机增强，抑制过拟合（测试禁用） |
| `ToTensor()` | HWC→CHW，0~255→0~1，必须有 |
| `Normalize(mean,std)` | 逐通道 `(x-mean)/std`，训练与测试参数必须一致 |
| `DataLoader` | 生成器，`batch_size`、`shuffle`（训练必开） |
| `next(iter(dl))` | 取第一批；老代码里的 `dl.next()` 已废弃 |
| 形状主线 | `[3,224,224]` → 堆叠 → `[32,3,224,224]`，标签 `[32]` |

**给你的项目的一句话建议**：Part 7 的产物不是模型，而是"可再生出 batch 的管道"。项目里"数据预处理 + 数据增强"这 10~20 分（含"未使用数据增强 −10"这条扣分项），成败完全取决于这个 cell 写得对不对、以及训练集有没有 `shuffle=True`。
