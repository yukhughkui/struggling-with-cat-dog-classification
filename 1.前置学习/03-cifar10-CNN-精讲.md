# CIFAR-10 CNN Solution（从零搭 CNN）逐 Cell 精讲

> 对应文件：`cifar10_cnn_solution.ipynb`（原文件约 900KB，其中绝大部分是内嵌的输出图片；本讲义只讲代码与文字）
> 前置：Part 7（数据管道）、Part 8（迁移学习）。本讲义反过来是"不借助预训练、从零搭 CNN"的完整示范。
> 读者定位：研一零基础，刚学完 `nn.Conv2d` / `MaxPool2d` / 全连接层，还没完整写过一次训练 + 评估。

---

## 0. 先看全局：这份 notebook 的地位

Part 7 教"怎么把图片喂进来"，Part 8 教"怎么借别人的网络"。这份 notebook 补上第三块：**自己从零定义并训练一个 CNN，然后评估、可视化、分析错误**。它正好是你项目里"自定义 CNN 模型（≥3 个卷积层 + 池化 + 全连接，报告训练/验证曲线与混淆矩阵）"的模板。

它跑的是 **CIFAR-10**（10 类、32×32 彩色小图、5 万训练 + 1 万测试），不是猫狗。这不是跑题——CIFAR-10 是"从零训 CNN"的经典基准，32×32 让 CPU 也能在可接受时间内跑完，而网络结构（3 个 conv + 3 个 pool + 3 个 fc）**可以直接迁移到猫狗任务上，只需改几处维度**（最后一节会给出具体改法）。

全篇是一条完整流水线：

```
检查 GPU → 加载并切分数据 → 看数据（整批 / 单图逐像素）
      → 定义网络 → 定义损失与优化器 → 训练（含验证与保存最优）
      → 载入最优权重 → 测试（总体 + 分类别准确率）→ 可视化预测结果
```

### 全文记号约定

- 形状统一 `[N, C, H, W]`。
- `N=20` 是本 notebook 的 batch size，`N=64` 只有个别地方出现。
- 讲到"删掉会怎样"，指单独删掉该行/该参数的后果，不是让你真删。

---

## Cell 0（markdown）

**英文原文**

> # Convolutional Neural Networks
> ---
> In this notebook, we train a **CNN** to classify images from the CIFAR-10 database.
>
> The images in this database are small color images that fall into one of ten classes; some example images are pictured below.
>
> `<img src='notebook_ims/cifar_data.png' width=70% height=70% />`

**中文翻译**

> # 卷积神经网络
> ---
> 在这个 notebook 中，我们训练一个 **CNN**，对来自 CIFAR-10 数据库的图像进行分类。
>
> 这个数据库中的图像是小型彩色图像，它们属于十个类别之一；下面展示了一些示例图像。

**这段的重点**：任务定义 = **10 分类**、**32×32 彩色图**。记住这两个数字，后面 `nn.Conv2d(3, ...)` 的 3 和 `nn.Linear(64*4*4, 500)` 里的 4 都由它们推出来。

---

## Cell 1（markdown）

**英文原文**

> ### Test for [CUDA](http://pytorch.org/docs/stable/cuda.html)
>
> Since these are larger (32x32x3) images, it may prove useful to speed up your training time by using a GPU. CUDA is a parallel computing platform and CUDA Tensors are the same as typical Tensors, only they utilize GPU's for computation.

**中文翻译**

> ### 检测 [CUDA](http://pytorch.org/docs/stable/cuda.html)
>
> 由于这些图像更大（32×32×3），使用 GPU 来加速训练时间可能会很有用。CUDA 是一个并行计算平台，CUDA 张量与普通张量完全一样，只是它们利用 GPU 来进行计算。

**这段的重点**：

- 原文说"larger (32x32x3)"是**相对于 MNIST 的 28×28×1** 而言的，不是绝对意义上的大图。
- **CUDA 张量和普通张量的 API 完全一致**，区别只在"算在哪块硬件上"。所以你不需要学两套 API，只需要在合适的位置 `.cuda()` / `.to(device)`。

---

## Cell 2（code）· 检测 GPU

```python
import torch
import numpy as np

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
```

### ①【这个 cell 在干什么】

导入 `torch` 与 `numpy`，检测本机 GPU 是否可用，并存成一个全局布尔变量 `train_on_gpu`，供后面所有 cell 判断"要不要搬数据"。

### ②【逐行拆解】

**第 1 行 `import torch`**

- PyTorch 主包。本 notebook 后面用的 `torch.cuda`、`torch.utils.data`、`torch.max`、`torch.save/load` 全部来自它。

**第 2 行 `import numpy as np`**

- NumPy，数值计算库，惯例别名 `np`。
- 本 notebook 里有三类用途：
  1. `np.random.shuffle(indices)` 打乱索引；
  2. `np.floor / np.Inf / np.sum / np.squeeze` 等数学操作；
  3. `images.numpy()` 把张量转成数组交给 matplotlib 显示。
- 删掉会怎样：Cell 4 的 `np.random.shuffle`、Cell 15 的 `np.Inf` 全部 `NameError`。

**第 5 行 `train_on_gpu = torch.cuda.is_available()`**

- `torch.cuda.is_available()`：返回 `True/False`，判断"当前 PyTorch 能否使用 CUDA"。
- 它同时检查三件事：**有没有 NVIDIA 显卡**、**驱动是否正常**、**PyTorch 是否编译了 CUDA 支持**（CPU-only 版本永远返回 `False`）。
- 为什么存成变量：后面会用到十几次（每个训练/测试循环都要判断），每次都调函数没必要；也方便你手动改成 `train_on_gpu = False` 来强制走 CPU 调试。
- 删掉会怎样：后面所有 `if train_on_gpu:` 报 `NameError`。
- **注意**：`torch.cuda.is_available()` 返回 True 不代表显存够用，也不代表你装了 `cudatoolkit` 对应版本；真正跑起来才可能暴露问题。

**第 7–10 行 `if not train_on_gpu: ... else: ...`**

- 标准条件分支，只负责**打印提示信息**，不影响任何计算。
- 为什么先判断 `not`：让"没有 GPU"这个更常见/更需要提醒的情况写在前面。
- 删掉会怎样：不影响后续代码，只是你失去了"当前跑在哪"的信息。**排查"为什么这么慢"时，这行输出是第一手证据。**

### ③【张量形状变化】

无数据流动，**形状不变**。

### ④【新手易错点】

1. **看到 `CUDA is not available` 就以为代码错了**。CPU 上照样能跑完，只是慢（本 notebook 在普通笔记本 CPU 上大约 1~2 小时/30 epoch）。
2. **装了 GPU 版 PyTorch 但仍返回 False**。
   - 常见原因：驱动太旧、`torch` 与 `torchvision` 版本不匹配、装成了 `cpu` 后缀的 wheel。
   - 自检：`torch.__version__` 里应带 `+cu`（如 `2.1.0+cu118`），`torch.version.cuda` 应不为 `None`。
3. **在 `if` 里用 `=` 而不是 `==`**（复制代码时手滑），会变成赋值、永远为真。
4. **以为 `train_on_gpu` 是"正在使用 GPU"**。它只是"可以用 GPU"，真正搬数据发生在后面每处 `if train_on_gpu: data.cuda()`。

---

## Cell 3（markdown）

**英文原文**

> ---
> ## Load the [Data](http://pytorch.org/docs/stable/torchvision/datasets.html)
>
> Downloading may take a minute. We load in the training and test data, split the training data into a training and validation set, then create DataLoaders for each of these sets of data.

**中文翻译**

> ---
> ## 加载[数据](http://pytorch.org/docs/stable/torchvision/datasets.html)
>
> 下载可能需要一分钟。我们加载训练数据和测试数据，把训练数据切分成训练集和验证集，然后为这些数据分别创建 DataLoader。

**这段的重点**：**三份数据**的概念——训练集（学参数）、验证集（选超参/早停/存最优模型）、测试集（只评一次）。CIFAR-10 官方只给 train/test 两份，验证集必须**自己从 train 里切**，这就是下一个 cell 在做的事。

---

## Cell 4（code）· 加载 CIFAR-10 + 切验证集 + 三个 DataLoader

```python
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
```

### ①【这个 cell 在干什么】

下载 CIFAR-10，定义"转张量 + 标准化"流水线，然后**把 5 万张训练图按 8:2 随机切成训练/验证**，用 `SubsetRandomSampler` 让两个 DataLoader 共享同一份数据对象但取不同的子集，最后建好三个 DataLoader 并写好 10 个类名。

### ②【逐行拆解】

**第 1–3 行 导入**

- `from torchvision import datasets`：拿到 `datasets.CIFAR10`。
- `import torchvision.transforms as transforms`：别名写法，之后写 `transforms.Compose`。
- `from torch.utils.data.sampler import SubsetRandomSampler`：**采样器**。它的作用是"我只从数据集的指定下标集合里取样本"。注意新版路径是 `torch.utils.data.SubsetRandomSampler`（顶层），旧路径 `...data.sampler...` 仍然兼容。
  - 删掉会怎样：第 33、34 行 `NameError`。

**第 6 行 `num_workers = 0`**

- DataLoader 用几个**子进程**读数据。`0` = 在主进程里读（单线程）。
- 为什么设 0：Windows 上多进程需要 `if __name__ == '__main__'` 保护，Jupyter 里设 >0 容易卡死或报 `BrokenPipeError`。教学代码一律 0。
- 性能代价：CPU 上会有"GPU 等数据"的空转；设 2~4 能提速，但要处理平台问题。

**第 8 行 `batch_size = 20`**

- 每批 20 张。20 是个偏小的值：
  - 好处：显存占用低，CPU 上也能跑；一个 epoch 有 2000 个 batch（4 万 / 20），参数更新频繁，前期 loss 降得快。
  - 坏处：梯度噪声大，loss 曲线抖；batch 太小还会让 BatchNorm（本例没用）统计不准。
- 为什么不是 32/64：作者想让 CPU 用户也能跑完，且 20 能被 40000/10000 整除，最后一批不会残缺（40000/20=2000，10000/20=500，正好整除）——**这点很重要**，见 Cell 19 的坑。

**第 10 行 `valid_size = 0.2`**

- 验证集占训练集的 20%。5 万 → 验证 1 万，训练 4 万。
- 为什么是 0.2：常见经验值（10%~20%）。验证集太小则指标噪声大，太大则训练数据不够。

**第 13–16 行 `transform = transforms.Compose([...])`**

- `transforms.ToTensor()`：PIL（32×32×3，uint8 0~255）→ 张量 `[3,32,32]`，值域 0~1。
- `transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))`：
  - 两个参数都是**三元组**，分别对应 R/G/B 三个通道的均值和标准差。
  - 逐通道计算 `(x - 0.5) / 0.5`，把 0~1 映射到 **[-1, 1]**。
  - 为什么用 0.5/0.5 而不是 ImageNet 的真实统计量：CIFAR-10 是从零训练，没有"必须对齐的预训练分布"；用 0.5 是个简单对称的选择，效果够用。
  - **删掉会怎样**：输入变成 0~1，网络仍能训练，但收敛会慢一些、对学习率更敏感（这也是 Part 7/8 反复强调 Normalize 的原因）。
- **注意**：这里**没有任何数据增强**（没有 RandomCrop/Flip）。这是你项目里的一个现成改进点——项目评分明确把"未使用数据增强"列为 −10 分。

**第 19–22 行 `train_data` / `test_data`**

- `datasets.CIFAR10(root, train, download, transform)`：
  - `root='data'`：数据存放目录（相对当前工作目录），首次会自动创建。
  - `train=True/False`：**同一份数据集的两个 split**——`True` 是 5 万张训练图，`False` 是 1 万张测试图。
  - `download=True`：本地没有就下载（约 170MB）。已存在则跳过，不会重复下。
  - `transform=transform`：每取一张就施加这条流水线。
- 这里**故意把同一个 `transform` 给 train 和 test**，因为没有增强，不需要区分。
- 删掉 `download=True`：首次运行报 `RuntimeError: Dataset not found`。
- 数据规模：`len(train_data)=50000`，`len(test_data)=10000`。

**第 25–28 行 切分索引**

- `num_train = len(train_data)` → 50000。
- `indices = list(range(num_train))` → `[0, 1, ..., 49999]`。
- `np.random.shuffle(indices)` → **原地**打乱这个列表（返回 None）。为什么不打乱就直接切：`CIFAR10` 的文件是按类别顺序排的（前 5000 张全是 airplane，接着 5000 张 automobile…），直接 `[:10000]` 会得到"只有两类"的验证集，指标完全失真。**这是本 cell 最关键的一步。**
- `split = int(np.floor(valid_size * num_train))` → `int(floor(0.2*50000)) = 10000`。
  - `np.floor` 向下取整，`int` 再转成整数（`np.floor` 返回 float）。不取整的话切片下标必须是 int，会 `TypeError`。
- `train_idx, valid_idx = indices[split:], indices[:split]`：
  - `indices[10000:]` → 后 4 万个下标 = 训练；
  - `indices[:10000]` → 前 1 万个 = 验证。
  - 注意是**切片**不是"随机抽取"，但因为上一步已经整体打乱，效果等价于随机划分。

**第 31–32 行 `SubsetRandomSampler`**

- `SubsetRandomSampler(indices)`：每次迭代时，**从给定下标列表中随机（无放回）抽一个**，抽完一轮就是一个 epoch。
- 为什么用它而不是 `random_split`：`random_split` 会创建两个新的 Dataset 对象（各自包装原 dataset），也完全可行；而 `SubsetRandomSampler` 的好处是**两个 loader 共享同一个 dataset 实例**（CIFAR-10 只需加载一次到内存），更省内存。
- "Random" 体现在：每个 epoch 的取样顺序都重新打乱，等于自带 `shuffle=True`，所以下面不用再写 `shuffle`。
- 删掉会怎样：`sampler=` 参数报 `NameError`。

**第 35–40 行 三个 DataLoader**

- `train_loader = DataLoader(train_data, batch_size=20, sampler=train_sampler, num_workers=0)`：
  - `dataset` 是**完整的** `train_data`（5 万张），但 `sampler` 只给出 4 万个下标，所以一个 epoch 实际只遍历 4 万张（2000 个 batch）。
  - **`sampler` 与 `shuffle` 互斥**：传了 `sampler` 就不能再传 `shuffle=True`，否则报 `ValueError: sampler option is mutually exclusive with shuffle`。这也是这里不写 `shuffle` 的原因。
- `valid_loader`：同一个 `train_data`，换 `valid_sampler`（那 1 万个下标）。**注意它用的仍是带 Normalize 的 transform**，这是对的（训练/验证必须同分布）。
- `test_loader = DataLoader(test_data, batch_size=20, num_workers=0)`：
  - 不传 `sampler`，默认顺序遍历；`shuffle` 默认 `False`。
  - `test_data` 是另一个对象（1 万张）。
- 删掉 `num_workers=num_workers`：默认就是 0，不报错（这行是显式化）。
- 规模：`len(train_loader)=2000`、`len(valid_loader)=500`、`len(test_loader)=500`。

**第 43–44 行 `classes = [...]`**

- 10 个类名，**顺序必须和 CIFAR-10 的标签编号一致**（0=airplane … 9=truck）。这个顺序是数据集官方固定的，不能自己调换。
- 用途：Cell 7 显示标题、Cell 19 打印分类别准确率、Cell 23 显示预测结果。
- 删掉会怎样：上述三处 `NameError`；而且你没法把"标签 3"翻译成人能读的 "cat"。

### ③【张量形状变化】

```
CIFAR-10 原始：PIL 图 [32,32,3] uint8
  → ToTensor     → [3,32,32] float32，值域 [0,1]
  → Normalize    → [3,32,32] float32，值域 [-1,1]
  → DataLoader(batch_size=20) 堆叠
  → images [20,3,32,32]，labels [20]（int64，0~9）

三个 loader 的规模：
  train_loader：2000 个 batch × 20 = 40000 张
  valid_loader： 500 个 batch × 20 = 10000 张
  test_loader ： 500 个 batch × 20 = 10000 张
```

### ④【新手易错点】

1. **不 `shuffle` 就切分**，得到"只有两个类别"的验证集。表现：验证准确率异常高或异常低，且和测试准确率差距巨大。
2. **`np.random.shuffle(indices)` 写成 `indices = np.random.shuffle(indices)`**。`shuffle` 是原地操作、返回 `None`，这样写会把 `indices` 变成 `None`，下一行 `indices[split:]` 报 `TypeError: 'NoneType' object is not subscriptable`。
3. **`sampler` 和 `shuffle=True` 同时传**。直接 `ValueError`。
4. **以为验证集是从 `test_data` 里切的**。这会造成**数据泄漏**（用测试集调参、再用测试集报成绩），答辩时是硬伤。正确做法就是本例：只从 `train_data` 里切。
5. **`normalize` 的 0.5 与 Part 8 的 ImageNet 数值混用**。CIFAR 从零训练用 0.5 没问题；一旦换成预训练模型，必须改回 `[0.485,0.456,0.406]/[0.229,0.224,0.225]`。
6. **`num_workers>0` 在 Windows/Jupyter 下卡死**。改回 0。
7. **`classes` 顺序凭记忆写**。写错不影响训练，但所有"分类别准确率"和预测标题都会张冠李戴。

---

## Cell 5（markdown）

**英文原文**

> ### Visualize a Batch of Training Data

**中文翻译**

> ### 可视化一批训练数据

**这段的重点**：训练前先"看数据"。看什么？——图有没有正常显示（说明 transform 没写错）、标签和图像是否对得上（说明 `classes` 顺序没写错）。

---

## Cell 6（code）· 定义反标准化显示函数

```python
import matplotlib.pyplot as plt
%matplotlib inline

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
```

### ①【这个 cell 在干什么】

导入绘图库，并定义一个 3 行的小函数：把"标准化过的 CHW 张量"还原成"可显示的 HWC 数组"并画出来。

### ②【逐行拆解】

**第 1 行 `import matplotlib.pyplot as plt`**

- 绘图接口。后面 Cell 7/9/23 都用它。

**第 2 行 `%matplotlib inline`**

- 魔法命令，让图内联显示。注意它**出现在 import 之后**也能生效（魔法命令在 cell 执行时按顺序处理）。规范写法通常放在最前面。

**第 5–7 行 `def imshow(img): ...`**

- `def` 定义函数，参数 `img` 预期是 `[3,32,32]` 的 torch 张量。
- `img = img / 2 + 0.5`：**反标准化**。因为前面做了 `(x-0.5)/0.5`，逆运算就是 `x*0.5+0.5`，即 `x/2+0.5`，把 `[-1,1]` 拉回 `[0,1]`。
  - 这一步不做的话，`imshow` 会收到负值，matplotlib 会把它 clamp 到 0，图像整体偏暗、颜色失真（但**不会报错**）。
  - 注意这里直接对张量做了原地外的算术（返回新张量），不需要 `.numpy()`，因为 torch 张量支持 `/` 和 `+`。
- `plt.imshow(np.transpose(img, (1,2,0)))`：
  - `np.transpose(img, (1,2,0))`：把维度顺序 `(C,H,W)` 重排为 `(H,W,C)`。参数 `(1,2,0)` 的意思是"新数组的第 0 维取原第 1 维，第 1 维取原第 2 维，第 2 维取原第 0 维"。
  - 为什么必须转：matplotlib 只认 `[H,W]`（灰度）或 `[H,W,3]`（RGB）。直接喂 `[3,32,32]`，它会把"3"当成高度，画出 3 行像素的怪图，或报 `Invalid shape`。
  - 更规范的写法是先 `.numpy()` 再 transpose（`np.transpose` 对张量也能工作，内部会转换）。
- 函数**没有返回值**（返回 `None`），副作用是"在当前 axes 上画图"。
- 删掉会怎样：Cell 7/9/23 全部 `NameError`。

### ③【张量形状变化】

```
输入 img [3, 32, 32]，值域 [-1,1]
  → img/2+0.5       → [3, 32, 32]，值域 [0,1]
  → transpose(1,2,0)→ [32, 32, 3]
  → plt.imshow      → 画出 32×32 的彩色图
```

### ④【新手易错点】

1. **`transpose` 的参数写成 `(2,0,1)`**。得到 `[32,3,32]`，画出来是 3 像素宽的竖条纹。
2. **忘了反标准化**。图能出来，但偏暗、对比度差，你会误以为"数据坏了"。
3. **`img / 2 + 0.5` 的顺序写反**（`img + 0.5 / 2`）。由于运算符优先级，后者等于 `img + 0.25`，图会过曝。
4. **以为这个函数能处理整个 batch**。它只接受单张 `[3,32,32]`；传 `[20,3,32,32]` 会画出乱七八糟的拼图。
5. **在 GPU 张量上直接 transpose**。`np.transpose` 会尝试转 numpy，GPU 张量必须先 `.cpu()`。本例全程 `images` 是 CPU 的 numpy 数组，所以没暴露这个问题。

---

## Cell 7（code）· 画出 20 张训练图及标签

```python
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])
```

### ①【这个 cell 在干什么】

取一个 batch（20 张），把它们排成 2 行 10 列，每张图上标注它的类别名，用来确认"数据 + 标签"是否配对正确。

### ②【逐行拆解】

**第 2 行 `dataiter = iter(train_loader)`**

- 创建迭代器并存变量。因为下面只用一次，其实可以省；存的写法更清楚。

**第 3 行 `images, labels = dataiter.next()`**

- **`dataloader.next()` 是老 API**（PyTorch < 1.5）。新版会报 `AttributeError: 'DataLoader' object has no attribute 'next'`，要改成 `next(dataiter)`。
- 习惯上老教程里 `DataLoader` 曾有 `.next()` 方法，现在只有迭代器才有 `next()` 内置函数。

**第 4 行 `images = images.numpy()`**

- 把 `[20,3,32,32]` 的 `torch.FloatTensor` 转成同形状的 `np.ndarray`。
- 为什么转：`imshow` 里用了 `np.transpose`，虽然 torch 张量也能被 numpy 函数处理，但显式转换更稳妥，也避免 matplotlib 与 torch 的兼容性小坑。
- **限制**：只有 **CPU 且不需要梯度** 的张量能 `.numpy()`。GPU 张量要先 `.cpu()`，带梯度的要先 `.detach()`。
- 删掉会怎样：本例仍能画（因为 `np.transpose` 能吃张量），但属于"侥幸"；换个写法就崩。

**第 7 行 `fig = plt.figure(figsize=(25, 4))`**

- 新建一张画布，25 英寸宽、4 英寸高。宽高比 25:4 ≈ 6:1，正好容纳"2 行 × 10 列"的扁平布局。
- `figsize` 单位是**英寸**（受 DPI 影响），不是像素。

**第 9 行 `for idx in np.arange(20):`**

- `np.arange(20)` → `[0,1,...,19]`，和 `range(20)` 等价。用 numpy 版本纯属作者习惯。
- 为什么是 20：等于 `batch_size`，刚好画完一整批。

**第 10 行 `ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])`**

- `add_subplot(nrows, ncols, index)`：老式（MATLAB 风格）子图定位。
  - `nrows=2`、`ncols=20/2=10.0`（**浮点数**）。matplotlib 内部会 `int()` 转成 10，所以 `10.0` 能用；但写 `10` 更规范。
  - `index=idx+1`：**1 基**编号，从左到右、从上到下。
  - `xticks=[], yticks=[]`：把坐标轴刻度清空，图更干净（32×32 的图配刻度纯属干扰）。
- 等价的新写法：`fig, axes = plt.subplots(2, 10, figsize=(25,4))` 然后 `axes[i][j]`。
- 删掉会怎样：没有 axes，`imshow` 会画到"当前 axes"（可能是上一次的），结果所有图画在同一格里。

**第 11 行 `imshow(images[idx])`**

- 取第 `idx` 张 `[3,32,32]`，走 Cell 6 的反标准化 + 转置，画到**当前 axes**（就是上一行返回的 `ax`）。

**第 12 行 `ax.set_title(classes[labels[idx]])`**

- `labels[idx]` 是 0 维 int64 张量（例如 `tensor(3)`）。
- `classes[...]` 用张量当下标：Python 列表索引要求整数，而 0 维整型张量实现了 `__index__`，所以能直接用（等价于 `classes[labels[idx].item()]`）。
- 输出标题如 `cat`。
- 删掉会怎样：不报错，但没有标签，你无法验证"图与标签是否配对"。

### ③【张量形状变化】

```
train_loader → 一个 batch
  images [20,3,32,32] → .numpy() → [20,3,32,32] ndarray
  labels [20]
      │ images[idx]
      ▼
  [3,32,32] → /2+0.5 → [3,32,32] → transpose → [32,32,3]
      ▼
  画到 2×10 网格的第 idx+1 格
```

### ④【新手易错点】

1. **`.next()` 报 AttributeError**（新版 PyTorch）。改 `next(dataiter)`。
2. **`20/2` 在 Python 3 里是 `10.0`（float）**。多数情况 matplotlib 能吃，但某些版本对非整数行列会报错；写 `10` 最保险。
3. **`add_subplot` 的 index 从 1 开始**，写 `idx` 会 `ValueError: num must be 1 <= num <= 20, not 0`。
4. **`images.numpy()` 用在 GPU 张量上**。报 `TypeError: can't convert cuda:0 device type tensor to numpy`。
5. **标题和图错位**。若 `classes` 顺序写错，你会看到"一张飞机图标注成 truck"，这是数据没错、映射错了。

---

## Cell 8（markdown）

**英文原文**

> ### View an Image in More Detail
>
> Here, we look at the normalized red, green, and blue (RGB) color channels as three separate, grayscale intensity images.

**中文翻译**

> ### 更细致地查看一张图像
>
> 这里，我们把标准化后的红、绿、蓝（RGB）三个颜色通道分别看成三张独立的灰度强度图来观察。

**这段的重点**：一张 32×32 的彩色图，本质是**三个 32×32 的矩阵叠在一起**。把通道拆开看，是理解"卷积层输入为什么有 3 个通道、卷积核为什么是 `[out, in, k, k]`"最直观的方式。

---

## Cell 9（code）· 拆通道 + 逐像素标数值

```python
rgb_img = np.squeeze(images[3])
channels = ['red channel', 'green channel', 'blue channel']

fig = plt.figure(figsize = (36, 36)) 
for idx in np.arange(rgb_img.shape[0]):
    ax = fig.add_subplot(1, 3, idx + 1)
    img = rgb_img[idx]
    ax.imshow(img, cmap='gray')
    ax.set_title(channels[idx])
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y],2) if img[x][y] !=0 else 0
            ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center', size=8,
                    color='white' if img[x][y]<thresh else 'black')
```

### ①【这个 cell 在干什么】

取 batch 里第 4 张图（`images[3]`），把 R/G/B 三个通道各画成一张灰度图，并在每个像素上标注它的数值——让你亲眼看到"标准化后像素可以是负数"。

### ②【逐行拆解】

**第 1 行 `rgb_img = np.squeeze(images[3])`**

- `images[3]` → `[3,32,32]`。
- `np.squeeze`：**去掉长度为 1 的维度**。这里 `[3,32,32]` 没有长度为 1 的维度，所以**什么也不做**，`rgb_img` 仍是 `[3,32,32]`。
- 作者大概是想"确保没有多余的 batch 维"，属于防御性写法。如果写成 `images[3:4]`（`[1,3,32,32]`），squeeze 才会真正去掉第 0 维。
- 删掉会怎样：完全不影响（`[3,32,32]` 直接可用）。

**第 2 行 `channels = ['red channel', 'green channel', 'blue channel']`**

- 三个通道名，顺序对应张量的第 0 维索引 0/1/2（PyTorch 的 RGB 顺序）。
- 删掉会怎样：第 7 行 `NameError`。

**第 3 行 `fig = plt.figure(figsize = (36, 36))`**

- 36×36 英寸的大画布——因为要在每个像素上写字，图必须够大（32×32 个数字塞进小图会糊成一团）。

**第 4 行 `for idx in np.arange(rgb_img.shape[0]):`**

- `rgb_img.shape[0]` = 3（通道数），所以循环 3 次。
- 用 `shape[0]` 而不是写死 3：更通用（万一以后处理 4 通道 RGBA）。

**第 5 行 `ax = fig.add_subplot(1, 3, idx + 1)`**

- 1 行 3 列，第 `idx+1` 格。三个通道并排。

**第 6 行 `img = rgb_img[idx]`**

- 取出**单个通道** → `[32,32]`。这是理解"通道是二维矩阵"的关键一步。

**第 7 行 `ax.imshow(img, cmap='gray')`**

- `cmap='gray'`：**灰度色图**。二维数组默认会按"最小值→黑、最大值→白"映射。
- 不加 `cmap` 会用什么？matplotlib 默认 `viridis`（紫绿黄），也能看，但不直观。
- 注意：因为值域是 `[-1,1]`，`imshow` 会自动按数据的最小/最大值归一化，**负数不影响显示**（但影响你对"亮度"的直觉）。

**第 8 行 `ax.set_title(channels[idx])`**

- 标注 "red channel" 等。删掉后你就分不清哪张是哪个通道。

**第 9 行 `width, height = img.shape`**

- `img` 是 `[32,32]`，所以 `width=32, height=32`。
- 命名有点误导：`img.shape` 对二维数组是 `(行数, 列数)`，所以严格说 `width` 应该是 `height`（行 = y 方向）。因为 32×32 对称，这里看不出问题；换成非方图就会坐标错位。

**第 10 行 `thresh = img.max()/2.5`**

- 取该通道最大值除以 2.5，作为"深色/浅色"的分界阈值。
- 用途：下面标数字时，**深的背景写白字、浅的背景写黑字**，保证可读。

**第 11–12 行 双重循环 `for x in range(width): for y in range(height):`**

- 遍历 32×32 = 1024 个像素。`x` 是行下标（对应图像的 y 坐标），`y` 是列下标（对应 x 坐标）——**命名与几何方向相反**，这是本 cell 最容易绕晕的地方。

**第 13 行 `val = round(img[x][y],2) if img[x][y] !=0 else 0`**

- `img[x][y]`：单像素标量。
- `round(..., 2)`：保留 2 位小数（值域 [-1,1]，两位足够）。
- 三元表达式：如果值**恰好等于 0** 就写 `0`，否则写四舍五入值。目的只是让 "0.0" 显示成 "0"，纯美观。
- 删掉 `round`：会显示一长串浮点数（如 `-0.37254901960784315`），图会糊成一片。

**第 14–17 行 `ax.annotate(str(val), xy=(y,x), horizontalalignment='center', verticalalignment='center', size=8, color='white' if img[x][y]<thresh else 'black')`**

- `annotate(text, xy=...)`：在指定坐标写字。
- `str(val)`：把数字转成字符串。
- `xy=(y, x)`：**注意是 (列, 行)**，因为 matplotlib 坐标是 `(x, y)` = `(列, 行)`。这里把 `y` 放前面是对的，**如果写成 `(x, y)` 整张图的数字会沿对角线翻转**。
- `horizontalalignment='center'` / `verticalalignment='center'`：让文字以该像素为中心，而不是从左上角开始。
- `size=8`：字号 8pt，32 个数挤在 36 英寸的画布上刚好。
- `color=...`：像素值小于阈值（偏暗）→ 白字；否则 → 黑字。
- 删掉整个 annotate：只剩三张灰度图，你就看不到"标准化后确实有负数"这件事了。

### ③【张量形状变化】

```
images[3]              [3, 32, 32]
  → np.squeeze         [3, 32, 32]（无变化，无长度 1 的维度）
  → rgb_img[idx]       [32, 32]     ← 单通道矩阵
  → imshow(cmap='gray') 画成灰度图
  → 双重循环 32×32 = 1024 次 annotate
```

### ④【新手易错点】

1. **`xy=(y,x)` 写成 `xy=(x,y)`**。数字位置整体转置，看起来"图歪了"但代码不报错。
2. **`width, height = img.shape` 的命名误导**。真正含义是 `(rows, cols)`。
3. **`np.squeeze` 以为能去掉通道维**。它只去长度为 1 的维度，`[3,32,32]` 不会变。
4. **看到负数和大于 1 的数就以为数据错了**。这是 `Normalize` 的正常结果（值域 [-1,1]），不是 bug。
5. **`annotate` 的 1024 次调用很慢**。在大图上做这件事会卡很久；这只是教学演示，不要用在生产可视化里。
6. **`img[x][y] != 0` 的浮点比较**。标准化后几乎不可能恰好等于 0，所以这个分支基本不会触发；它只是美观处理。

---

## Cell 10（markdown）· 网络设计说明（含卷积输出尺寸公式）

**英文原文**

> ---
> ## Define the Network [Architecture](http://pytorch.org/docs/stable/nn.html)
>
> This time, you'll define a CNN architecture. Instead of an MLP, which used linear, fully-connected layers, you'll use the following:
> * [Convolutional layers](https://pytorch.org/docs/stable/nn.html#conv2d), which can be thought of as stack of filtered images.
> * [Maxpooling layers](https://pytorch.org/docs/stable/nn.html#maxpool2d), which reduce the x-y size of an input, keeping only the most _active_ pixels from the previous layer.
> * The usual Linear + Dropout layers to avoid overfitting and produce a 10-dim output.
>
> A network with 2 convolutional layers is shown in the image below and in the code, and you've been given starter code with one convolutional and one maxpooling layer.
>
> `<img src='notebook_ims/2_layer_conv.png' height=50% width=50% />`
>
> #### TODO: Define a model with multiple convolutional layers, and define the feedforward metwork behavior.
>
> The more convolutional layers you include, the more complex patterns in color and shape a model can detect. It's suggested that your final model include 2 or 3 convolutional layers as well as linear layers + dropout in between to avoid overfitting.
>
> It's good practice to look at existing research and implementations of related models as a starting point for defining your own models. You may find it useful to look at [this PyTorch classification example](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py) or [this, more complex Keras example](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py) to help decide on a final structure.
>
> #### Output volume for a convolutional layer
>
> To compute the output size of a given convolutional layer we can perform the following calculation (taken from [Stanford's cs231n course](http://cs231n.github.io/convolutional-networks/#layers)):
> > We can compute the spatial size of the output volume as a function of the input volume size (W), the kernel/filter size (F), the stride with which they are applied (S), and the amount of zero padding used (P) on the border. The correct formula for calculating how many neurons define the output_W is given by `(W−F+2P)/S+1`.
>
> For example for a 7x7 input and a 3x3 filter with stride 1 and pad 0 we would get a 5x5 output. With stride 2 we would get a 3x3 output.

**中文翻译**

> ---
> ## 定义网络[结构](http://pytorch.org/docs/stable/nn.html)
>
> 这一次，你要定义一个 CNN 架构。不再是使用线性全连接层的 MLP，而是使用下面这些：
> * [卷积层](https://pytorch.org/docs/stable/nn.html#conv2d)，可以把它理解为"一堆经过滤波的图像"的堆叠。
> * [最大池化层](https://pytorch.org/docs/stable/nn.html#maxpool2d)，它缩小输入的 x-y 尺寸，只保留上一层中最"活跃"的像素。
> * 常见的 Linear + Dropout 层，用来避免过拟合并产生 10 维输出。
>
> 下图展示了一个带 2 个卷积层的网络，代码里也是这个结构；同时你已经拿到了一个"1 个卷积层 + 1 个池化层"的起始代码。
>
> `<img src='notebook_ims/2_layer_conv.png' height=50% width=50% />`
>
> #### TODO：定义一个含多个卷积层的模型，并定义前馈（forward）行为。
>
> 你加入的卷积层越多，模型能检测到的颜色和形状模式就越复杂。建议你的最终模型包含 2 到 3 个卷积层，以及其间的线性层 + dropout，以避免过拟合。
>
> 以已有的相关模型研究和实现作为出发点来定义你自己的模型，是个好习惯。你可以参考[这个 PyTorch 分类示例](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py)或[这个更复杂的 Keras 示例](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py)，来帮助你确定最终结构。
>
> #### 卷积层的输出体积
>
> 要计算某个卷积层的输出尺寸，我们可以做下面的计算（摘自[斯坦福 cs231n 课程](http://cs231n.github.io/convolutional-networks/#layers)）：
> > 我们可以把输出体积的空间尺寸表示为输入体积尺寸（W）、卷积核/滤波器尺寸（F）、施加时的步长（S）、以及边界上使用的零填充量（P）的函数。计算输出宽度 output_W 有多少个神经元的正确公式是：`(W−F+2P)/S+1`。
>
> 例如，对 7×7 的输入和 3×3 的滤波器，步长为 1、填充为 0，我们会得到 5×5 的输出；步长为 2 时，我们会得到 3×3 的输出。

**这段的重点**：

- **卷积公式 `out = (W − F + 2P) / S + 1`** 是全篇最该背下来的一条。验证原文例子：`(7−3+0)/1+1 = 5` ✓；`(7−3+0)/2+1 = 3` ✓。
- 代码里大量使用 `padding=1` + `kernel=3` + `stride=1`，代入得 `(W−3+2)/1+1 = W`，即**尺寸不变**。这就是"为什么 conv 不改变 H×W"的答案。
- **池化 `MaxPool2d(2,2)`** 把尺寸**减半**（`32→16→8→4`），这是 `64*4*4` 这个魔法数字的来源。
- 结构建议："2~3 个 conv + 线性层 + dropout"，正好是下面 Cell 11 的 3 个 conv 版本。

---

## Cell 11（code）· 定义 CNN（全篇最重要的 cell）

```python
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # convolutional layer (sees 16x16x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees 8x8x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x

# create a complete CNN
model = Net()
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()
```

### ①【这个 cell 在干什么】

用 `nn.Module` 子类的标准写法，手工搭一个"3 组 [卷积+ReLU+池化] + 展平 + 全连接 ×2 + Dropout"的 CNN，输出 10 类分数；然后实例化、打印结构、按需搬到 GPU。

### ②【逐行拆解】

**第 1–2 行 导入**

- `import torch.nn as nn`：网络层仓库（`Conv2d`、`Linear`、`MaxPool2d`、`Dropout`）。
- `import torch.nn.functional as F`：函数式接口（`F.relu`）。
- 为什么两个都要：**带参数/状态的层**用 `nn.` 在 `__init__` 里注册（这样参数能被自动管理）；**纯函数**（ReLU）直接用 `F.` 调用，省得在 `__init__` 里创建对象。

**第 5 行 `class Net(nn.Module):`**

- **必须继承 `nn.Module`**，否则 `model.parameters()`、`.cuda()`、`.train()`、`state_dict()` 全部不可用。
- 类名随意（`Net` / `MyCNN` 都行），只要实例化时用对。

**第 6 行 `def __init__(self):`**

- 构造函数，负责**声明所有层**（只创建对象，不处理数据）。
- `self.` 前缀：把层挂到实例上，`nn.Module` 的 `__setattr__` 会自动把它注册为子模块。

**第 7 行 `super(Net, self).__init__()`**

- **必须调用父类构造函数**。`nn.Module.__init__` 会初始化 `_parameters`、`_modules`、`_buffers` 等内部字典。
- 不写会报 `AttributeError: cannot assign module before Module.__init__() call`——**这是新手写自定义网络时最经典的第一个报错**。
- 现代写法 `super().__init__()` 等价，更简洁。

**第 9 行 `self.conv1 = nn.Conv2d(3, 16, 3, padding=1)`**

逐参数说明（这是你问的那种"每个参数都要讲"的重点）：

- `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)`。
- 位置参数 `3` = `in_channels`：**输入通道数**。输入是 RGB 三通道图，所以是 3。若输入灰度图就写 1。
- 位置参数 `16` = `out_channels`（也叫 filters / feature maps）：**输出通道数**，即"用 16 个不同的卷积核去扫描同一张图，得到 16 张特征图"。
  - 每个卷积核是一个 `[3,3,3]`（out 的单个核是 in×k×k）的小张量。
  - 为什么是 16：通道数逐层翻倍（16→32→64）是 CNN 的常见节奏——浅层通道少（只认边缘）、深层通道多（要认组合模式），同时控制参数量。
- 位置参数 `3` = `kernel_size`：**卷积核边长 3**，即 3×3 的窗口。等价写法 `kernel_size=3` 或 `(3,3)`。
  - 为什么用 3×3：两个 3×3 堆叠的感受野等于一个 5×5，但参数更少、非线性更多（VGG 的核心发现）。
- 关键字参数 `padding=1`：**在输入的四周各补 1 圈 0**。
  - 作用：让 3×3 卷积后的 H、W **保持不变**（`(32−3+2)/1+1 = 32`）。
  - 不加 `padding=1`（默认 0）：每层尺寸缩小 2（32→30→28），而且边缘像素被卷积次数少、信息利用不均，深层会缩得太快。
- 省略的默认值：`stride=1`（每次滑 1 格）、`dilation=1`（不空洞）、`groups=1`（普通卷积）、`bias=True`（带偏置）。
- 这一层的参数量：`16 * 3 * 3 * 3 + 16 = 448`（权重 432 + 偏置 16）。
- 删掉 `padding=1` 会怎样：尺寸链条变成 `32→30→14→12→6→5`，最后 `64*5*5` 与 `fc1` 的 `64*4*4` 不匹配，`x.view(-1, 1024)` 会在反向/前向时报 `shape '[20, 1024]' is invalid for input of size 20*1600`。

**第 11 行 `self.conv2 = nn.Conv2d(16, 32, 3, padding=1)`**

- `in_channels=16`：**必须等于上一层 `out_channels`**，否则 `RuntimeError: Given groups=1, weight of size [32,16,3,3], expected input[20,16,16,16] to have 16 channels, but got ...`。
- `out_channels=32`：通道翻倍。
- 注释里的 "sees 16x16x16"：因为上一层后面接了池化，输入到这里是 `[N,16,16,16]`（16 通道、16×16）。
- 参数量：`32*16*3*3 + 32 = 4640`。

**第 13 行 `self.conv3 = nn.Conv2d(32, 64, 3, padding=1)`**

- 32 进、64 出，通道再翻倍。输入 `[N,32,8,8]`。
- 参数量：`64*32*3*3 + 64 = 18496`。
- 三个 conv 的通道设计 `3→16→32→64` 是"从颜色到语义"的典型递进。

**第 15 行 `self.pool = nn.MaxPool2d(2, 2)`**

- `nn.MaxPool2d(kernel_size, stride=None, padding=0)`。
- `kernel_size=2`：2×2 的窗口，取窗口内**最大值**。
- `stride=2`：窗口每次滑 2 格（不重叠）。若省略 `stride`，默认等于 `kernel_size`，所以写 `nn.MaxPool2d(2)` 效果相同。
- 效果：H、W 各减半（`32→16→8→4`），通道数不变。参数量为 **0**（池化没有可学习参数）。
- 为什么用 max 而不是 average：max 保留"最强响应"，等价于问"这个区域里有没有某种特征"，对纹理/边缘更敏感。
- **只创建一个 `self.pool` 却在 forward 里用三次**：`nn.MaxPool2d` **无状态、无可学习参数**，可以被安全复用（不像 `nn.BatchNorm2d` 那样有 running stats）。这是新手常疑惑的点。
- 删掉会怎样：尺寸不变（一直 32×32），展平后是 `64*32*32 = 65536`，`fc1` 的 `64*4*4` 完全不匹配。

**第 17 行 `self.fc1 = nn.Linear(64 * 4 * 4, 500)`**

- `nn.Linear(in_features, out_features, bias=True)`。
- `64 * 4 * 4 = 1024`：**必须精确等于展平后的元素个数**。推导：3 次池化把 32 变成 `32/2/2/2 = 4`，通道是 64，所以 `64×4×4 = 1024`。
  - **改输入尺寸（比如换成 224×224 的猫狗图）时，这个数必须重算**：`224/8 = 28` → `64*28*28 = 50176`。这是把本网络迁移到猫狗任务时**最容易漏改**的一处。
- `500`：隐藏层宽度，超参数，无硬性规定。
- 参数量：`1024*500 + 500 = 512,500`——**这一层就占了整个网络参数的绝大部分**（约 51 万 / 53 万）。
- 删掉会怎样：`forward` 里 `self.fc1(x)` 报 `AttributeError`。

**第 19 行 `self.fc2 = nn.Linear(500, 10)`**

- `in_features=500` 必须等于上一层输出；`out_features=10` = 类别数。
- 输出 `[N,10]` 是 **logits**（未归一化的分数），不是概率。
- 为什么这里**不接 softmax**：因为损失函数用 `nn.CrossEntropyLoss`（Cell 13），它内部已经包含 `log_softmax`。再接一次就重复了。
- 参数量：`500*10+10 = 5010`。
- 删掉会怎样：`forward` 报 `AttributeError`；或输出维度不对。

**第 21 行 `self.dropout = nn.Dropout(0.25)`**

- `nn.Dropout(p=0.25)`：训练时**以 25% 的概率把每个神经元输出置 0**（并且把保留的神经元按 `1/(1-p)` 放大，保持期望不变）。
- 作用：防止过拟合，相当于每次迭代训练一个不同的"子网络"，最后集成。
- 为什么是 0.25：全连接层常用 0.5，但这里网络较小、数据较多，0.25 更温和；值越大正则越强，太大会欠拟合。
- **只在 `model.train()` 模式生效**；`model.eval()` 时自动关闭（这就是 Cell 15/19 必须切模式的原因之一）。
- 同一个 `self.dropout` 在 forward 里用了两次：同样因为 Dropout 无可学习参数，可复用。
- 删掉会怎样：不报错，但训练集准确率会明显高于测试集（过拟合）。

**第 24 行 `x = self.pool(F.relu(self.conv1(x)))`**

- 从内往外读：`self.conv1(x)` → `F.relu(...)` → `self.pool(...)`。
  - 先卷积：提取特征，输出 `[N,16,32,32]`；
  - 再 ReLU：负值归零（非线性），形状不变；
  - 再池化：`[N,16,16,16]`。
- 为什么把 `relu` 放在 `pool` 前面而不是后面：先激活再池化是标准顺序（池化的是"激活强度"；反过来先池化再 ReLU 也可以，但会丢失"负值里的信息"，且与主流实现不一致）。
- 这行把三个操作压成一行，嵌套顺序**必须从内往外读**，新手常误以为是"先 pool 再 conv"。

**第 25–26 行 同样模式**

- `conv2`：`[N,16,16,16] → [N,32,16,16] → ReLU → [N,32,8,8]`；
- `conv3`：`[N,32,8,8] → [N,64,8,8] → ReLU → [N,64,4,4]`。
- 注意注释里写的 "sees 16x16x16"、"sees 8x8x32" 是对应 conv 的**输入**形状，和这里的推导一致。

**第 28 行 `x = x.view(-1, 64 * 4 * 4)`**

- `view`：**重塑形状，不复制数据**（要求内存连续）。
- `-1` 表示"这一维让 PyTorch 自己算"：`N` 张图，每张展平成 1024 个数。
- 输入 `[20,64,4,4]` → 输出 `[20,1024]`。
- 为什么必须展平：`nn.Linear` 只接受二维 `[batch, features]`。
- **为什么用 `-1` 而不是写死 `20`**：最后一个 batch 可能不满 20（本例恰好整除，但换数据集就不一定），写死会在最后一批崩掉。
- 删掉会怎样：`fc1` 收到四维张量，报 `mat1 and mat2 shapes cannot be multiplied (1280x4 and 1024x500)` 之类。
- 备选：`x = torch.flatten(x, 1)` 或 `x.reshape(...)`（`reshape` 在内存不连续时会自动复制，更安全）。

**第 30 行 `x = self.dropout(x)`**

- 展平后立刻 dropout，随机丢弃 25% 的 1024 个特征，防止全连接层过度依赖某几个位置的特征。
- 形状不变 `[20,1024]`。

**第 32 行 `x = F.relu(self.fc1(x))`**

- 全连接 + 激活：`[20,1024] → [20,500] → ReLU → [20,500]`。
- 如果只写 `self.fc1(x)` 不加 `F.relu`：网络退化成"两个线性层串联"，数学上等价于**一个**线性层，表达能力大幅下降（准确率会掉几个点）。

**第 34 行 `x = self.dropout(x)`**

- 第二次 dropout（同一模块），形状不变。
- 注意两个 dropout 的 `p` 相同（0.25），因为共用一个模块。如果想让两层不同强度，需要创建两个模块。

**第 36 行 `x = self.fc2(x)`**

- 输出 `[20,10]` 的 logits。
- **这里不加 ReLU**：最后一层要保留负值（负分代表"越不可能是这个类"），加了 ReLU 会把所有负类压成 0，破坏 softmax 的区分度。
- **也不加 softmax**：交给 `CrossEntropyLoss`。

**第 37 行 `return x`**

- 返回 `[N,10]`。没有返回值就是 `None`，训练循环立刻报错。

**第 40 行 `model = Net()`**

- 实例化。此时所有层按默认方式初始化（`Conv2d`/`Linear` 用 Kaiming 均匀分布），参数量约 53 万。
- 删掉会怎样：后面 `print(model)`、`model.cuda()` 报 `NameError`。

**第 41 行 `print(model)`**

- 打印结构，输出形如：
  ```
  Net(
    (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (fc1): Linear(in_features=1024, out_features=500, bias=True)
    (fc2): Linear(in_features=500, out_features=10, bias=True)
    (dropout): Dropout(p=0.25, inplace=False)
  )
  ```
- 这是**验收工具**：检查 `padding` 有没有生效、`fc1` 的 in_features 对不对。

**第 44–45 行 `if train_on_gpu: model.cuda()`**

- `model.cuda()` 把**所有参数和 buffer** 搬到 GPU（等价于 `model.to('cuda')`），**原地修改**。
- 为什么用 `if`：CPU 机器上 `model.cuda()` 会直接报 `AssertionError: Torch not compiled with CUDA enabled`。
- 删掉会怎样：模型留在 CPU，而 Cell 15 里 `data.cuda()` 把数据搬到 GPU → 前向报"tensor 在不同设备上"。
- 注意位置：**必须在实例化之后**。如果先 `model.cuda()` 再 `model = Net()`，新模型又回 CPU 了。

### ③【张量形状变化】

完整前向（以 batch=20 为例）：

```
x                          [20, 3, 32, 32]
 conv1(3→16, k=3, p=1)     [20, 16, 32, 32]   参数量 448
 ReLU                      [20, 16, 32, 32]
 pool(2,2)                 [20, 16, 16, 16]
 conv2(16→32, k=3, p=1)    [20, 32, 16, 16]   参数量 4,640
 ReLU                      [20, 32, 16, 16]
 pool(2,2)                 [20, 32,  8,  8]
 conv3(32→64, k=3, p=1)    [20, 64,  8,  8]   参数量 18,496
 ReLU                      [20, 64,  8,  8]
 pool(2,2)                 [20, 64,  4,  4]
 view(-1, 1024)            [20, 1024]
 dropout(0.25)             [20, 1024]
 fc1(1024→500)             [20, 500]          参数量 512,500
 ReLU                      [20, 500]
 dropout(0.25)             [20, 500]
 fc2(500→10)               [20, 10]           参数量 5,010
返回 logits                [20, 10]
```

尺寸推导只用了两个公式：

```
Conv2d:  out = (in − k + 2p)/s + 1 = (32 − 3 + 2)/1 + 1 = 32   （padding=1 保持尺寸）
MaxPool: out = in / 2                                          （32→16→8→4）
```

总参数量 ≈ `448 + 4640 + 18496 + 512500 + 5010 = 541,094`。

### ④【新手易错点】

1. **忘了 `super(Net, self).__init__()`**。报 `AttributeError: cannot assign module before Module.__init__() call`。
2. **`Conv2d` 的 `in_channels` 与上一层 `out_channels` 对不上**。报 `expected input to have N channels`。
3. **`fc1` 的 `64*4*4` 是硬编码的**。换输入尺寸（224）必须改成 `64*28*28`；算错就报 `view size is not compatible`。
4. **`view` 的 `-1` 写死成 `20`**。最后一批不满时会崩；永远用 `-1`。
5. **最后一层加了 ReLU 或 softmax**。配 `CrossEntropyLoss` 时会重复归一化，loss 异常。
6. **`pool` 写成 `nn.MaxPool2d(2, 1)`**。滑动步长 1 会导致尺寸只减 1（32→31），链条全乱。
7. **把 `forward` 里的嵌套顺序读反**。`self.pool(F.relu(self.conv1(x)))` 是 conv→relu→pool，不是反过来。
8. **`model.cuda()` 放在 `model = Net()` 之前**，或在 `if train_on_gpu` 外面无条件调用（CPU 上直接崩）。
9. **以为 dropout 也作用在测试时**。`model.eval()` 会自动关闭它；忘了 `eval()` 会让测试准确率随机波动、偏低。

---

## Cell 12（markdown）

**英文原文**

> ### Specify [Loss Function](http://pytorch.org/docs/stable/nn.html#loss-functions) and [Optimizer](http://pytorch.org/docs/stable/optim.html)
>
> Decide on a loss and optimization function that is best suited for this classification task. The linked code examples from above, may be a good starting point; [this PyTorch classification example](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py) or [this, more complex Keras example](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py). Pay close attention to the value for **learning rate** as this value determines how your model converges to a small error.
>
> #### TODO: Define the loss and optimizer and see how these choices change the loss over time.

**中文翻译**

> ### 指定[损失函数](http://pytorch.org/docs/stable/nn.html#loss-functions)与[优化器](http://pytorch.org/docs/stable/optim.html)
>
> 为这个分类任务选定最合适的损失函数和优化函数。上面链接的代码示例可以作为不错的起点；[这个 PyTorch 分类示例](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py)或[这个更复杂的 Keras 示例](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py)。要特别关注**学习率**的取值，因为它决定了你的模型收敛到小误差的速度。
>
> #### TODO：定义损失函数和优化器，并观察这些选择如何随时间改变损失。

**这段的重点**：

- 分类任务默认组合：**多分类 + logits 输出 → `CrossEntropyLoss`**；**优化器 → SGD 或 Adam**。
- **学习率是最重要的超参数**：太大会震荡/发散（loss 变 `nan`），太小会收敛极慢。`0.01`（SGD）和 `1e-3`（Adam）是两个最常用的起点。

---

## Cell 13（code）· 损失函数与优化器

```python
import torch.optim as optim

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

### ①【这个 cell 在干什么】

导入优化器模块，定义"交叉熵损失"和"随机梯度下降优化器（学习率 0.01）"。

### ②【逐行拆解】

**第 1 行 `import torch.optim as optim`**

- 优化器仓库，别名 `optim`（社区惯例）。

**第 4 行 `criterion = nn.CrossEntropyLoss()`**

- `nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduction='mean')`，最常用的是 `reduction='mean'`（对 batch 求平均）。
- 它**等价于 `LogSoftmax` + `NLLLoss` 两步合一**，所以：
  - 输入必须是**原始 logits** `[N,10]`（不能先 softmax，否则重复归一化）；
  - 标签必须是**类别索引** `[N]`（int64），不能是 one-hot。
- 输出是**标量**（0 维张量），值越小越好。
- 为什么用它而不是 MSE：分类任务用交叉熵，梯度性质更好（预测错时梯度大、预测对时梯度小），收敛快得多。MSE 配 softmax 会出现"梯度消失"。
- 删掉会怎样：Cell 15 的 `criterion(output, target)` 报 `NameError`。

**第 7 行 `optimizer = optim.SGD(model.parameters(), lr=0.01)`**

- `optim.SGD(params, lr, momentum=0, weight_decay=0, nesterov=False)`。
- `params=model.parameters()`：把**全部参数**（约 53 万）交给优化器管理。这里没有冻结任何层，所以全都要训。
- `lr=0.01`：学习率，SGD 的常用起点（0.1 太大容易发散，0.001 太慢）。**这是你项目里最值得做对比实验的超参**。
- 没写 `momentum`：默认 0，即"纯 SGD"。加上 `momentum=0.9` 通常收敛更快更稳——这是本 notebook 可改进的点之一。
- 为什么用 SGD 而不是 Adam：作者想展示"经典配置"；SGD 泛化往往略好但要调参。Adam（`lr=1e-3`）上手更容易。
- 删掉会怎样：Cell 15 的 `optimizer.zero_grad()/step()` 报 `NameError`。
- **注意**：`optimizer` 必须在 `model` 创建之后定义，因为它持有的是**参数对象的引用**；如果之后重建了 `model`，旧 `optimizer` 还在更新旧参数（静默失效）。

### ③【张量形状变化】

无数据流动。约定：`outputs [20,10]` + `targets [20]` → `loss` 标量 `[]`。

### ④【新手易错点】

1. **在模型最后加了 `softmax` 又用 `CrossEntropyLoss`**。loss 会偏小、梯度被压扁，准确率上不去。
2. **标签是 one-hot**（`[20,10]`）。`CrossEntropyLoss` 报 `Dimension out of range` 或结果错误。
3. **学习率写成 `1`**。loss 直接 `nan`（梯度爆炸）。
4. **`model.parameters()` 写成 `model.parameters`（漏括号）**。报 `TypeError: 'method' object is not iterable`。
5. **先建 optimizer 再建 model**。`model` 未定义直接报错；更隐蔽的是"重建 model 后没重建 optimizer"。

---

## Cell 14（markdown）

**英文原文**

> ---
> ## Train the Network
>
> Remember to look at how the training and validation loss decreases over time; if the validation loss ever increases it indicates possible overfitting. (In fact, in the below example, we could have stopped around epoch 33 or so!)

**中文翻译**

> ---
> ## 训练网络
>
> 记住要观察训练损失和验证损失随时间如何下降；如果验证损失出现上升，就说明可能过拟合了。（事实上，在下面的示例中，我们本可以在第 33 个 epoch 左右就停下来！）

**这段的重点**：

- **判断过拟合的标准动作：训练 loss 继续降、验证 loss 开始升**。这就是"早停（early stopping）"的信号。
- 原文说"33 epoch 就该停"，但代码里写的是 `n_epochs=30`——**注释和代码不一致**（作者大概改过数字没改注释）。理解意图即可：**最优模型往往不在最后一个 epoch**，所以要"按验证 loss 保存最优"。
- 你项目报告里的"损失曲线 + 早停"分析，就是这一段的知识点。

---

## Cell 15（code）· 训练循环（含验证与保存最优模型）

```python
# number of epochs to train the model
n_epochs = 30

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss
```

### ①【这个 cell 在干什么】

跑 30 个 epoch：每个 epoch 先在 4 万张训练图上更新参数，再在 1 万张验证图上只算 loss；打印两者的**样本平均 loss**；只要验证 loss 创新低就保存一次权重。

### ②【逐行拆解】

**第 2 行 `n_epochs = 30`**

- 训练轮数。"一个 epoch = 完整遍历一次训练子集（4 万张 / 2000 个 batch）"。
- 30 轮是 CPU 可承受与"够收敛"的折中。GPU 上一般 10~20 轮就够。

**第 4 行 `valid_loss_min = np.Inf`**

- `np.Inf` 是浮点正无穷。作为"历史最低验证 loss"的初始值，保证第一个 epoch 一定满足"更低"。
- 新版 numpy 推荐 `np.inf`（`np.Inf` 已弃用但仍可用）。
- 删掉会怎样：下面 `valid_loss <= valid_loss_min` 报 `NameError`。

**第 6 行 `for epoch in range(1, n_epochs+1):`**

- 产生 `1..30`。**从 1 开始**是因为打印时要给人看"Epoch 1"，更符合直觉。
- 注意 `range(1, 31)` 不含 31，所以正好 30 轮。

**第 9–10 行 `train_loss = 0.0` / `valid_loss = 0.0`**

- **每个 epoch 开头清零**，用来**累加整个 epoch 的 loss**。
- 注意它们是"加权和"（见 `loss.item()*data.size(0)`），不是"平均"。

**第 15 行 `model.train()`**

- 切到**训练模式**。影响两类层：
  - `Dropout`：开始随机丢弃；
  - `BatchNorm`：用当前 batch 的均值/方差，并更新 running stats（本例没用 BN）。
- 为什么必须写：默认就是 train 模式，但上一个 epoch 末尾切过 `eval()`，不切回来 dropout 就永远关着 → 正则失效、容易过拟合。
- 删掉会怎样：**第二轮起 dropout 失效**（因为第一轮末尾调了 `eval()`），loss 曲线看起来"更好"但泛化变差。

**第 16 行 `for data, target in train_loader:`**

- 遍历 2000 个 batch，每个 `data [20,3,32,32]`、`target [20]`。
- `train_loader` 用 `SubsetRandomSampler`，每个 epoch 的取样顺序都不同（自带 shuffle）。

**第 18–19 行 `if train_on_gpu: data, target = data.cuda(), target.cuda()`**

- 把数据和标签搬到 GPU。
- **模型已在 Cell 11 搬到 GPU**，所以这里必须同步搬数据，否则设备不一致报错。
- `target` 也要搬：`CrossEntropyLoss` 内部要比较 `output`（GPU）和 `target`，两者必须在同一设备。
- 删掉会怎样：GPU 环境下前向报 `Expected all tensors to be on the same device`。

**第 21 行 `optimizer.zero_grad()`**

- **清空所有参数的 `.grad`**（设为 0 或 None）。
- 为什么必须：PyTorch 的 `backward()` 是**梯度累加**，不清零就会把上一个 batch 的梯度也加上，等效学习率不断变大，最终爆炸或 `nan`。
- 为什么放在 forward 之前也可以：位置在 `backward()` 之前都行，社区习惯放最前面。
- 删掉会怎样：训练 loss 先降后剧烈震荡/`nan`，是最经典的"看起来能跑但结果烂"的 bug。
- 性能提示：`optimizer.zero_grad(set_to_none=True)` 更快（新版默认已是这样）。

**第 23 行 `output = model(data)`**

- 前向：`[20,3,32,32] → [20,10]` 的 logits。
- 用 `model(data)` 而不是 `model.forward(data)`：`__call__` 会触发 hooks，是官方推荐写法。

**第 25 行 `loss = criterion(output, target)`**

- 交叉熵：对 20 个样本各自算 `-log(softmax(logits)[正确类])`，再取平均 → 标量。

**第 27 行 `loss.backward()`**

- 反向传播，把梯度累加到每个参数的 `.grad`。
- 计算图在 `backward()` 后即被释放（除非 `retain_graph=True`）。

**第 29 行 `optimizer.step()`**

- 按 `.grad` 更新参数。SGD 的更新式：`w ← w − lr * grad`。
- 顺序铁律：`zero_grad → forward → loss → backward → step`。把 `step()` 放到 `backward()` 之前，梯度还是上一批的（第一批是 None），参数永远不更新。

**第 31 行 `train_loss += loss.item()*data.size(0)`**

- `loss.item()`：把 0 维张量转成 Python float（**必须 `.item()`**，否则会保留计算图、显存持续增长）。
- `* data.size(0)`：乘 batch 内样本数（20），把"这个 batch 的平均 loss"还原成"这个 batch 的总 loss"。
- 为什么要这样：只有加权求和，最后除以总样本数，才能得到**真正的样本平均 loss**。如果直接累加每个 batch 的平均值，当最后一个 batch 不满时会略微偏高。
- 删掉 `*data.size(0)`：结果仍近似正确（本例每批都是 20），但换数据集就不准。
- 删掉 `.item()`：`train_loss` 会变成张量且携带计算图 → 显存爆掉（OOM）。

**第 34 行 `model.eval()`**

- 切到**评估模式**：关闭 Dropout、BatchNorm 改用 running stats。
- 为什么必须有：不写的话验证 loss 会被 dropout 的随机性污染，且每轮结果不同；配合 `train()` 的切换，才构成标准的"训练/评估"双模式。

**第 35–42 行 验证循环**

- 结构比训练循环少了三样：`zero_grad`、`backward`、`step`。
- **为什么不写 `torch.no_grad()`**：作者漏了。虽然结果正确（因为没有 `backward`，梯度不会被更新），但**会构建完整的计算图**，显存占用明显增加。正确写法是 `with torch.no_grad():` 包住整个循环。这是本 cell 一个值得记住的改进点。
- `valid_loss += loss.item()*data.size(0)`：同样的加权累加。

**第 45–46 行 `train_loss = train_loss/len(train_loader.sampler)` / `valid_loss = .../len(valid_loader.sampler)`**

- **除以样本数，不是 batch 数**。
  - `len(train_loader.sampler) = 40000`（`SubsetRandomSampler` 的长度）；
  - `len(valid_loader.sampler) = 10000`。
- 为什么不用 `len(train_loader)`：那是 2000（batch 数），除出来会大 20 倍，虽然单调性不变，但数值没法和其他实验对比。
- 删掉这两行：打印的是"30 轮累加的 loss"，会看到一条巨大的、毫无意义的递增曲线。

**第 49–51 行 `print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(...))`**

- `{}` 顺序填充 `epoch`；`{:.6f}` 保留 6 位小数。
- `\t` 是制表符，用于对齐。
- 观察要点：两个 loss 都应下降；若 `Validation Loss` 在某轮后开始上升 → 过拟合信号。

**第 54–60 行 `if valid_loss <= valid_loss_min: ... torch.save(model.state_dict(), 'model_cifar.pt'); valid_loss_min = valid_loss`**

- `valid_loss <= valid_loss_min`：**只保存"历史最优"**。用 `<=` 而不是 `<`，是允许"持平也覆盖"（影响很小）。
- `torch.save(model.state_dict(), 'model_cifar.pt')`：
  - `state_dict()` 是**有序字典**，只存"参数名 → 张量"（权重、偏置、BN 的 running stats），**不存网络结构**。
  - 相比 `torch.save(model)`（整个对象 pickle），`state_dict` **更推荐**：文件小、跨版本兼容好、加载时必须显式重建结构（更透明）。
  - 文件保存在**当前工作目录**，名字 `model_cifar.pt`。
- `valid_loss_min = valid_loss`：更新历史最优。**这一行不能漏**，否则每个 epoch 都满足条件、每轮都覆盖保存，最终留下的是"最后一轮"而不是"最优轮"。
- 删掉整块：训练照常跑，但没有任何权重落盘，Cell 17 的 `torch.load` 直接 `FileNotFoundError`。

### ③【张量形状变化】

```
每个 epoch：
  train_loader → 2000 个 batch
     data [20,3,32,32] → model → output [20,10] → loss []（标量）
  valid_loader → 500 个 batch
     data [20,3,32,32] → model → output [20,10] → loss []（标量）

累计量：train_loss / valid_loss 都是 Python float（标量）
保存内容：state_dict 中每个参数的形状，例如
  conv1.weight [16,3,3,3]、conv1.bias [16]
  fc1.weight  [500,1024]、fc2.weight [10,500]
```

### ④【新手易错点】

1. **漏 `optimizer.zero_grad()`** → 梯度累加、loss 爆炸。
2. **验证段忘了 `model.eval()`** → dropout 仍在随机丢弃，验证 loss 抖动、偏高。
3. **验证段忘了 `torch.no_grad()`**（本例就是）→ 显存占用翻倍，大模型直接 OOM。
4. **验证段误加了 `optimizer.step()`** → 用验证集训练，数据泄漏，测试准确率虚高。
5. **`train_loss` 除以 `len(train_loader)`** → 数值放大 20 倍，曲线没有可比性。
6. **忘记 `valid_loss_min = valid_loss`** → 每轮都覆盖保存，最后留下的是过拟合的末轮模型。
7. **`loss.item()` 写成 `loss`** → 累加的是带计算图的张量，显存线性增长直到 OOM。
8. **`model.train()` / `model.eval()` 放反或漏写** → 训练时 dropout 关闭（过拟合）、评估时 dropout 开启（指标偏低）。
9. **CPU 上跑 30 epoch 以为卡死**。一个 epoch 在普通 CPU 上要 1~3 分钟，30 轮接近一小时；建议先用 `n_epochs=1` 验证流程能跑通。

---

## Cell 16（markdown）

**英文原文**

> ### Load the Model with the Lowest Validation Loss

**中文翻译**

> ### 加载验证损失最低的那个模型

**这段的重点**：训练时保存的是"历史最优"权重，所以**训练完不能直接用内存里的 `model`**（那是最后一轮的、可能已过拟合的权重），必须把它换成磁盘上那份最优的。

---

## Cell 17（code）· 载入最优权重

```python
model.load_state_dict(torch.load('model_cifar.pt'))
```

### ①【这个 cell 在干什么】

从磁盘读回 `model_cifar.pt` 里的参数字典，覆盖当前 `model` 的参数，让模型回到"验证 loss 最低"的状态。

### ②【逐行拆解】

**唯一一行，拆成两半看：**

- 内层 `torch.load('model_cifar.pt')`：
  - 反序列化文件，得到一个 **OrderedDict**（键是参数名，值是张量）。
  - 默认加载到 **CPU**。若原来是 GPU 保存的，在新版 PyTorch 里直接 load 到 CPU 的模型也能用；但如果目标是 GPU，建议 `torch.load(path, map_location=device)`，避免"先 CPU 再搬"的额外开销，也能规避某些设备不匹配报错。
  - `weights_only` 参数：新版（2.6+）默认 `True`，只允许加载张量，更安全；本例只存了 state_dict，不受影响。
- 外层 `model.load_state_dict(...)`：
  - 按**参数名**逐一匹配并**原地覆盖**模型里的参数张量。
  - 键名必须完全一致。若你在保存后改了网络结构（比如 `fc1` 改名成 `fc`），会报 `RuntimeError: Error(s) in loading state_dict for Net: Missing key(s) ... / Unexpected key(s) ...`。
  - 严格模式默认 `strict=True`（键必须完全对应）。调试时可用 `strict=False` 跳过不匹配的键，但生产环境不要这么干。
- **它不返回新模型**（返回的是 `_IncompatibleKeys` 报告对象），模型是**原地**被修改的。
- 顺序很重要：必须在 `model = Net()` 和（若用 GPU）`model.cuda()` **之后**执行；在 `.cuda()` 之前 load 也行（load 后参数在 CPU，再 `.cuda()` 会一起搬）。
- 删掉会怎样：`model` 保持最后一轮的权重，测试准确率可能比最优低 1~3 个点；如果训练根本没跑过（无 `model_cifar.pt`），这行直接 `FileNotFoundError`。

### ③【张量形状变化】

无形状变化（形状由模型结构决定），只替换数值：

```
model_cifar.pt（OrderedDict）
  'conv1.weight' [16,3,3,3]  → 写入 model.conv1.weight
  'conv1.bias'   [16]        → ...
  'fc1.weight'   [500,1024]  → ...
  'fc2.weight'   [10,500]    → ...
```

### ④【新手易错点】

1. **路径错**：文件在 notebook 的工作目录下，用相对路径 `'model_cifar.pt'`；如果你在别的目录启动 Jupyter，会 `FileNotFoundError`。
2. **先 load 再改结构**：改了层名/层数后 load 会报 key 不匹配。
3. **以为 `load_state_dict` 会返回模型**：写成 `model = model.load_state_dict(...)`，结果 `model` 变成一个报告对象，后面全部崩。
4. **在 GPU 上训练、在 CPU 上加载时报设备错误**：加 `map_location='cpu'`（或 `map_location=device`）。
5. **忘记这一步就直接测试**：不报错，但你测的不是"最优模型"。

---

## Cell 18（markdown）

**英文原文**

> ---
> ## Test the Trained Network
>
> Test your trained model on previously unseen data! A "good" result will be a CNN that gets around 70% (or more, try your best!) accuracy on these test images.

**中文翻译**

> ---
> ## 测试训练好的网络
>
> 在你从未见过的数据上测试训练好的模型！一个"不错"的结果是：CNN 在这些测试图像上能达到 70% 左右的准确率（或者更高，尽力而为！）。

**这段的重点**：

- **CIFAR-10 的"及格线"是 70%**（从零训的小 CNN）。低于 60% 说明流程有 bug（多半是漏了 `zero_grad`、数据没 normalize、或标签错位）。
- **测试集只能用于最终评估**，不能拿来调参。这也是为什么前面要单独切验证集。

---

## Cell 19（code）· 测试：总体准确率 + 分类别准确率

```python
# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
```

### ①【这个 cell 在干什么】

在 1 万张测试图上算：总体 loss、总体准确率，以及**每个类别各自的准确率**（哪类容易错），为后面的"弱点分析"提供数据。

### ②【逐行拆解】

**第 2 行 `test_loss = 0.0`**

- 累加 `loss × 样本数` 的容器。

**第 3–4 行 `class_correct = list(0. for i in range(10))` / `class_total = ...`**

- 两个长度为 10 的 Python 列表，分别记录"每类答对多少"和"每类共多少"。
- `list(生成器)` 的写法等价于 `[0.]*10`，只是显得"显式"。
- 为什么不放在循环里：它们要跨 batch 累积，必须在外层初始化。

**第 6 行 `model.eval()`**

- 关闭 dropout。**测试时漏这行，准确率会偏低且每次运行结果不同**。

**第 8 行 `for data, target in test_loader:`**

- 500 个 batch，每批 20 张。

**第 10–11 行 搬到 GPU**

- 同训练循环。

**第 13–14 行 `output = model(data)` / `loss = criterion(output, target)`**

- `output [20,10]`，`loss` 标量。
- **注意这里也没有 `torch.no_grad()`**（作者漏了），能跑但浪费显存。

**第 16 行 `test_loss += loss.item()*data.size(0)`**

- 加权累加，同训练循环。

**第 18 行 `_, pred = torch.max(output, 1)`**

- `torch.max(tensor, dim)` 返回**命名元组 `(values, indices)`**：
  - `values`：该维度上的最大值（这里 = 最大 logit）；
  - `indices`：最大值所在的**下标** = 预测类别。
- 用 `_` 接收"不需要的 values"，这是 Python 惯例（下划线表示"丢弃"）。
- `dim=1`：沿**类别维**取最大。`output [20,10]`，`dim=1` → `pred [20]`，每个元素是 0~9。
- `dim=0` 会得到 `[10]`（每类在 20 个样本中的最大分），完全错误。
- 删掉会怎样：`pred` 未定义，下一行报错。

**第 20 行 `correct_tensor = pred.eq(target.data.view_as(pred))`**

- `pred.eq(...)`：**逐元素比较是否相等**，返回同形状的 bool 张量 `[20]`。
- `target.data.view_as(pred)`：
  - `.data` 是绕过 autograd 拿原始张量（老写法，现在更推荐 `.detach()`）；
  - `.view_as(pred)` 把 `target` 塑形成和 `pred` 一样的 `[20]`。这里形状本来就一致，属于保险写法。
- 结果 `correct_tensor`：`[True, False, True, ...]`，表示每个样本是否预测正确。
- 删掉会怎样：`correct` 未定义。

**第 21 行 `correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())`**

- 三元表达式：**GPU 上要先 `.cpu()`** 才能 `.numpy()`。
- `np.squeeze`：去掉长度 1 的维度。对 `[20]` 无影响；作者大概想处理"最后一批只有 1 个样本"的边界情况（那时是 `[1]`，squeeze 后变 0 维，索引行为会变）。
- 得到 numpy 的 bool 数组 `[20]`，便于下面用 `correct[i]` 索引。

**第 23–26 行 `for i in range(batch_size): label = target.data[i]; class_correct[label] += correct[i].item(); class_total[label] += 1`**

- 逐个样本统计：
  - `label = target.data[i]`：0 维张量，取值 0~9；
  - `class_correct[label]`：**用张量当列表下标**（0 维整型张量有 `__index__`，合法）；
  - `correct[i].item()`：bool → `True/False` → `1.0/0.0`，加到累计值上；
  - `class_total[label] += 1`：该类样本总数 +1。
- **坑**：这里写死了 `range(batch_size)`（=20）。如果最后一个 batch 不满 20 张（`test_data` 数量不能被 20 整除时），`target.data[i]` 会 `IndexError`。本例 10000/20 正好整除，所以没暴露。
  - 更稳的写法：`for i in range(len(target))` 或 `for i, label in enumerate(target)`。

**第 29 行 `test_loss = test_loss/len(test_loader.dataset)`**

- `test_loader.dataset` 是 `test_data`，`len` = 10000 → 得到样本平均 loss。
- 注意这里用的是 `len(test_loader.dataset)`（样本数），和前一个 cell 的 `len(loader.sampler)` 是**同一个含义**（总样本数），两种写法都对。

**第 30 行 `print('Test Loss: {:.6f}\n'.format(test_loss))`**

- 打印测试 loss，`\n` 空一行。

**第 32–38 行 分类别准确率**

- `for i in range(10)`：每个类别。
- `if class_total[i] > 0:`：防御性判断（若某类一个样本都没有，避免除零）。
- `'Test Accuracy of %5s: %2d%% (%2d/%2d)' % (...)`：老式 `%` 格式化：
  - `%5s` 右对齐宽度 5 的字符串（类名）；
  - `%2d%%` 整数宽度 2，后面 `%%` 输出一个字面 `%`；
  - `100 * class_correct[i] / class_total[i]`：该类准确率（百分比）；
  - `np.sum(class_correct[i])`：单个 float 求和（冗余，直接用值也一样）。
- 输出形如 `Test Accuracy of   cat: 71% (712/1000)`。
- `else` 分支：`N/A`，实际不会触发（每类 1000 张）。

**第 41–43 行 总体准确率**

- `100. * np.sum(class_correct) / np.sum(class_total)`：**乘 `100.`（带小数点）**是为了强制浮点除法（Python 2 习惯；Python 3 里 `/` 本来就是浮点，但没有坏处）。
- `np.sum(class_correct)`：把长度 10 的列表加起来 = 总答对数。
- 输出形如 `Test Accuracy (Overall): 78% (7812/10000)`。

### ③【张量形状变化】

```
test_loader
  → data [20,3,32,32]，target [20]
  → model(data)            → output [20,10]
  → criterion              → loss []（标量）
  → torch.max(output, 1)   → values [20], pred [20]
  → pred.eq(target)        → correct_tensor [20]（bool）
  → .cpu().numpy().squeeze() → correct [20]（numpy bool）
  → 分类别累计              → class_correct / class_total 各 10 个 float
```

### ④【新手易错点】

1. **忘了 `model.eval()`**。准确率偏低且每次不同。
2. **`torch.max` 的 `dim` 写成 0**。`pred` 形状变成 `[10]`，`pred.eq(target)` 广播出错或结果荒谬。
3. **忘了 `.cpu()`**（GPU 环境）。报 `can't convert cuda:0 device type tensor to numpy`。
4. **`for i in range(batch_size)` 与"最后一批不满"冲突**。本例侥幸整除；换数据集就会 `IndexError`。
5. **`class_correct[label]` 里 `label` 是张量**。能用但容易让人困惑；写 `.item()` 更清晰。
6. **分母用了 `len(test_loader)`**（batch 数）。loss 放大 20 倍。
7. **把测试准确率当成"最终成绩"却没检查分类别结果**。总体 78% 可能是"汽车 95% + 猫 55%"，而报告里"错误分析"恰恰要看这个。

---

## Cell 20（markdown）

**英文原文**

> ### Question: What are your model's weaknesses and how might they be improved?

**中文翻译**

> ### 问题：你的模型有哪些弱点？可以如何改进？

**这段的重点**：这是项目报告里"结果分析/错误案例"的标准提问方式——**不要只报一个准确率数字，要指出"哪类差、为什么差、怎么改"**。

---

## Cell 21（markdown）

**英文原文**

> **Answer**: This model seems to do best on vehicles rather than animals. For example, it does best on the automobile class and worst on the cat class. I suspect it's because animals vary in color and size and so it would improve this model if I could increase the number of animal images in the first place or perhaps if I added another convolutional layer to detect finer patterns in these images. I could also experiment with a smaller learning rate so that the model takes small steps in the right direction as it is training.

**中文翻译**

> **回答**：这个模型似乎在"交通工具"上表现最好，而不是"动物"上。例如，它在 automobile（汽车）类上表现最好，在 cat（猫）类上表现最差。我猜测原因是动物的颜色和尺寸变化更大；所以，如果能首先增加动物图像的数量，或许能给这个模型带来改进；或者，如果我再加上一个卷积层来检测这些图像中更精细的模式，也会有帮助。我也可以尝试更小的学习率，让模型在训练时朝着正确方向迈出更小的步子。

**这段的重点**：

- **分析范式**：先给现象（哪类最好/最差）→ 再给假设（动物类内差异大）→ 再给可执行的改进（加数据、加卷积层、调学习率）。
- 这三条改进正好对应你项目的加分项：**数据增强（等效于加数据多样性）、加深网络、学习率调度**。
- 注意"cat 最差"在 CIFAR-10 里是公认难点（猫的姿态/毛色/背景差异极大，且与 dog、deer 视觉上易混）。答辩时能说出这一点，说明你真的看了混淆矩阵。

---

## Cell 22（markdown）

**英文原文**

> ### Visualize Sample Test Results

**中文翻译**

> ### 可视化测试结果样例

**这段的重点**：把"预测 vs 真值"画出来，用**颜色（绿/红）**一眼看出对错，是"错误案例可视化"最省力的实现方式，也是项目 PPT 里最讨喜的一页。

---

## Cell 23（code）· 可视化 20 个测试样本的预测结果

```python
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images.numpy()

# move model inputs to cuda, if GPU available
if train_on_gpu:
    images = images.cuda()

# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images.cpu()[idx])
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))
```

### ①【这个 cell 在干什么】

取一个测试 batch，前向一次，把 20 张图的"预测类（真值类）"写在标题上，**猜对写绿字、猜错写红字**。

### ②【逐行拆解】

**第 2–3 行 `dataiter = iter(test_loader)` / `images, labels = dataiter.next()`**

- 老 API `.next()`，新版要写 `next(dataiter)`（同 Cell 7）。
- 因为 `test_loader` 没有 shuffle，每次运行拿到的都是**同一批**（前 20 张），结果可复现。

**第 4 行 `images.numpy()`**

- **这一行没有任何作用**：`images` 在 CPU 上，`.numpy()` 返回**新数组**，但没有赋值给任何变量，结果被丢弃。
- 作者大概原本想写成 `images = images.numpy()`，后来改成在下面用 `images.cpu()[idx]`，就忘了删。
- 删掉完全不影响（属于"死代码"）。

**第 6–8 行 `if train_on_gpu: images = images.cuda()`**

- **原地**把 `images` 变量指向 GPU 上的新张量（注意这里重新赋值了，所以后面要用 `images.cpu()` 才能显示）。
- 这里只搬输入，不搬 `labels`——因为下面只用来比较、不需要在 GPU 上算。

**第 10–11 行 `output = model(images)` / `_, preds_tensor = torch.max(output, 1)`**

- 同 Cell 19：`output [20,10]` → `preds_tensor [20]`（预测类别索引）。

**第 12 行 `preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())`**

- 转 numpy，GPU 时先 `.cpu()`。`squeeze` 对 `[20]` 无影响。

**第 15 行 `fig = plt.figure(figsize=(25, 4))`**

- 同 Cell 7 的画布。

**第 16–20 行 循环绘图**

- `fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])`：2 行 10 列（`20/2=10.0`，同 Cell 7 的提醒）。
- `imshow(images.cpu()[idx])`：
  - `images` 可能是 GPU 张量，先 `.cpu()` 才能交给 matplotlib（Cell 6 的 `imshow` 内部用 `np.transpose`）。
  - `[idx]` 取第 idx 张 `[3,32,32]`。
- `ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]), color=...)`：
  - 格式是 `"预测 (真值)"`，例如 `cat (dog)` 表示"把狗错认成猫"。
  - `color=("green" if preds[idx]==labels[idx].item() else "red")`：
    - `preds[idx]` 是 numpy 整数；`labels[idx].item()` 是 Python 整数。比较时**必须 `.item()`**，否则是"numpy 整数 vs torch 张量"的比较，虽然多数情况能广播出正确结果，但写成 `.item()` 最稳。
    - 猜对→绿，猜错→红。

### ③【张量形状变化】

```
test_loader → images [20,3,32,32]（CPU）
  → (可选) .cuda() → [20,3,32,32]（GPU）
  → model          → output [20,10]
  → torch.max(...,1)→ preds_tensor [20]（GPU 或 CPU）
  → .cpu().numpy().squeeze() → preds [20]
  → images.cpu()[idx] → [3,32,32] → imshow → [32,32,3]
```

### ④【新手易错点】

1. **`.next()` 报错**（新版）。
2. **`images.numpy()` 这行以为有用**。它是死代码，不赋值就什么都没发生。
3. **漏了 `.cpu()` 就去画**。GPU 环境报 `can't convert cuda tensor to numpy`。
4. **`preds[idx]==labels[idx]` 忘记 `.item()`**。多数时候能跑，但可能得到 `tensor(True)` 而不是 `True`，在 `if` 里仍可用；规范做法是统一转 Python 标量。
5. **标题顺序看反**：是 `"预测 (真值)"`，不是 `"真值 (预测)"`。报告里写错会被认为"没看懂自己的输出"。
6. **只展示猜对的样本**。那是自欺欺人；错误案例才是有价值的分析素材。

---

## Cell 24（code）· 空 cell

```python
（空）
```

### ①【这个 cell 在干什么】

一个空的 code cell，没有代码，运行不产生任何输出。通常是作者预留的"自己做实验"的位置（比如画混淆矩阵、试数据增强）。

### ②【逐行拆解】

- 没有行可拆。执行它只会让 notebook 的执行计数 +1。

### ③【张量形状变化】

无。

### ④【新手易错点】

- 误以为"这里漏了代码"。它就是空的，不影响运行。

---

## 本 notebook 小结（复习用）

| 关键词 | 一句话记忆 |
|---|---|
| `datasets.CIFAR10` | 5 万训练 + 1 万测试，32×32，10 类 |
| `valid_size=0.2` + `np.random.shuffle` | 先打乱索引再切，否则验证集只有两类 |
| `SubsetRandomSampler` | 两个 loader 共享同一 dataset，自带 shuffle，与 `shuffle=` 互斥 |
| `nn.Conv2d(3,16,3,padding=1)` | in=3、out=16、k=3、补 1 圈 0 → 尺寸不变 |
| 尺寸公式 | `out = (W−F+2P)/S+1`；池化 `out = W/2` |
| `64*4*4` | 3 次池化把 32→4，通道 64，展平得 1024 |
| `x.view(-1, 1024)` | `-1` 自动算 batch，别写死 20 |
| `CrossEntropyLoss` | 吃 logits + 类别索引，内部含 softmax |
| 训练五步 | `zero_grad → forward → loss → backward → step` |
| `train()` / `eval()` | 训练开 dropout，评估关 dropout，必须切换 |
| 保存最优 | `if valid_loss <= min: torch.save(state_dict())` 且更新 `min` |
| 测试 | `torch.max(output,1)` 取类别；分类别统计要防"最后一批不满" |

**相对猫狗项目的三处改法**（只讲不改）：

1. `datasets.CIFAR10` → `datasets.ImageFolder`（Part 7 的管道），`transform` 改成 224 尺寸 + 必要的数据增强；
2. `nn.Linear(64*4*4, 500)` → `nn.Linear(64*28*28, 500)`（224/8=28），其余结构不动；
3. `fc2` 的 `10` → `2`，`classes` 换成 `['cat','dog']`。

---

# 三个 notebook 与你项目的任务对应表

| Notebook | 对应项目任务项（来自《深度学习基础课程猫狗识别项目说明与评分标准》） | 具体对应点 |
|---|---|---|
| **Part 7 - Loading Image Data** | **二、数据集与预处理要求**（Kaggle Dogs vs Cats；统一尺寸建议 224×224；像素归一化；**必须应用数据增强**）；也是"（一）必做内容 · 自定义 CNN"的**数据管道前置步骤** | `ImageFolder` 加载 `Cat_Dog_data`；`Resize(255)+CenterCrop(224)` 实现"统一 224×224"；`Normalize` 实现"像素归一化"；`RandomRotation/RandomResizedCrop/RandomHorizontalFlip` 对应"必须应用数据增强"（直接对冲评分里"未使用数据增强 −10 分"）；`DataLoader` 提供 batch 输入 |
| **Part 8 - Transfer Learning** | **三（一）必做内容 · 迁移学习实现**（选用 VGG16/ResNet18 等经典模型，**冻结部分层并微调分类头**）；同时对应评分"模型实现 30 分"与扣分项"仅提交预训练模型且未进行任何微调 −15 分"的保护 | `densenet121(pretrained=True)` 等价于选经典预训练模型（可换成 `resnet18`/`vgg16`）；`requires_grad=False` = 冻结卷积层；替换 `classifier` 并训练 = 微调分类头；ImageNet 三个 mean/std = 与预训练模型对齐的标准化。注意：项目要求"经典模型如 VGG16/ResNet18"，DenseNet 属同类，答辩时说明选型理由即可 |
| **cifar10_cnn_solution** | **三（一）必做内容 · 自定义 CNN 模型**（从零构建 ≥3 个卷积层 + 池化 + 全连接，报告训练/验证准确率、损失曲线与混淆矩阵）；以及"结果分析 20 分"中的错误样本分析 | `Net` 的 3×[Conv+ReLU+Pool] + 2×FC 正好满足"≥3 卷积层"；训练循环里的 train/valid loss 就是"损失曲线"；Cell 19 的分类别准确率 + Cell 23 的绿/红可视化对应"错误案例可视化"（混淆矩阵需自己再加）；数据是 CIFAR-10，**需按本讲义末尾的三处改法换成 Cat_Dog_data 才能直接用于项目** |

**一句话总结三者的关系**：Part 7 解决"**数据怎么进来**"，cifar10 解决"**网络怎么从零搭、怎么训、怎么评**"，Part 8 解决"**怎么借预训练模型把准确率拉过 80%**"。项目"必做内容"的两条主线（自定义 CNN + 迁移学习），正好由后两份 notebook 一一对应。


