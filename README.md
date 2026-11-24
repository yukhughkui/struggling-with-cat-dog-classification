# 猫狗图像分类 · CNN 与迁移学习

本项目对应《深度学习基础课程项目说明与评分标准》，包含：

- 从零搭建的 CNN（7 卷积层 / 4 池化层 / 3 全连接层，含 BN 与 Dropout）
- ResNet18 迁移学习（冻结 backbone 训分类头 → 解冻 layer4 微调）
- 四组对比实验：E1 数据增强、E2 自搭 vs 迁移、E6 冻结 vs 微调、E7 轻量化（GAP vs 大 FC）
- 多种子复现（默认 42/43/44，结果报均值 ± 标准差）
- 评估产出：准确率、损失/准确率曲线、混淆矩阵、高置信度错误样本

## 目录结构

```
project/
├─ check_data.py          数据就绪检查（数量/类别/尺寸/损坏文件）
├─ run_experiments.py     四组实验统一入口
├─ summarize.py           汇总 runs → CSV/Markdown/对比图
├─ requirements.txt
├─ src/
│  ├─ data.py             数据管道（90:10 划分、两套 transforms）
│  ├─ models.py           自搭 CNN / ResNet18
│  ├─ engine.py           训练与评估循环
│  ├─ runner.py           单 run 运行与最终测试评估
│  └─ plots.py            曲线 / 混淆矩阵 / 错误样本
└─ README.md
```

## 环境

- Python 3.10，PyTorch 2.7.1+cu118，torchvision 0.22.1
- 显卡：NVIDIA RTX 4070 Laptop（8GB）
- 依赖安装：`pip install -r requirements.txt`

本机解释器：`D:\DeepStudy\Anaconda\envs\torch_env\python.exe`

## 数据

数据目录需满足 `ImageFolder` 约定：

```
Cat_Dog_data/
├─ train/cat/*.jpg   (11250)
├─ train/dog/*.jpg   (11250)
└─ test/cat/*.jpg    (1250)
   test/dog/*.jpg    (1250)
```

检查数据：

```bash
python check_data.py "d:\DeepStudy\Project1\Cat_Dog_data"
```

## 运行

```bash
# 0) 一次性预处理：把 25000 张图解码成 256×256 uint8 内存映射缓存（约 5GB）
python prepare_cache.py

# 冒烟测试：1 epoch、每轮 3 个 batch
python run_experiments.py --smoke

# 正式实验（复现本项目结果的两条命令）
python run_experiments.py --only transfer --seeds 42 43 44 --force
python run_experiments.py --only e1 e7 --seeds 42 43 44 --force --epochs-cnn 12 --epochs-heavy 12 --batch-size 48

# 汇总结果
python summarize.py
```

说明：
- 迁移学习用 batch 32；E1/E7 的自搭 CNN 用 batch 48、12 epoch（为控制总时长）。
- 本机冷启动 `import torch` 很慢，Windows 下 DataLoader 子进程又会各自重新 import，因此默认 `num_workers=0`，并用 `prepare_cache.py` 的缓存避免每个 epoch 重复解码 JPEG；数据增强在 GPU 上执行（torchvision v2）。
- 结果：ResNet18 阶段 B 测试准确率 98.72% ± 0.18%（3 seeds），详见 `../outputs/PROJECT_REPORT.md`。

所有产物写入 `../outputs/`：

- `results/<run_id>/`：`best.pt`、`history.csv`、`curves.png`、`test_preds.npz`、`test_cm.png`、`test_errors.png`、`metrics.json`、`test_metrics.json`
- `summary/`：`runs.csv`、`experiment_summary.csv/.md`、`fig_E1/E2/E6/E7.png`

## 实验定义

| 编号 | 变量 | 对照组 | 实验组 |
|---|---|---|---|
| E1 | 是否使用数据增强 | `cnn_gap_aug_off` | `cnn_gap_aug_on` |
| E2 | 模型来源 | `cnn_gap_aug_on` | `resnet_stageB` |
| E6 | 是否解冻微调 | `resnet_stageA` | `resnet_stageB` |
| E7 | 分类头聚合方式 | `cnn_heavy`（Flatten+大 FC） | `cnn_gap_aug_on`（GAP） |

变量控制：同一 90:10 划分（`split_seed=42`）、同一测试集、同一评估代码；每个配置 3 个训练种子。

## 复现性

- 所有随机源由 `set_seed(seed)` 固定（Python / NumPy / PyTorch / CUDA）
- 划分索引由 `split_seed=42` 固定，不随训练种子变化
- 每个 run 保存 `config.json` 与逐 epoch `history.csv`
- 早停与最优权重选择只用验证集；测试集仅用于最终评估

## 引用与来源标注

- 数据集：Kaggle *Dogs vs. Cats*（课程提供）
- 预训练权重：`torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`，来源 PyTorch torchvision 官方模型库
- 数据管道写法参考课程 Notebook Part 7；迁移学习两阶段写法参考 Part 8；自搭 CNN 结构参考 `cifar10_cnn_solution.ipynb` 并针对 224×224 输入重新设计
- 除上述参考外，模型、训练、评估、实验编排代码均为本项目自行实现
