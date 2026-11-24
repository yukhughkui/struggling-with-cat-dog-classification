# 猫狗图像分类项目 · 提交材料说明

## 目录

```
猫狗分类_项目提交材料/
├─ code/                      完整可运行代码（13 个文件）
│  ├─ 一键复现.py             零参数入口：检查 → 缓存 → 训练 → 汇总
│  ├─ check_data.py           数据清点（数量/类别/尺寸/损坏）
│  ├─ prepare_cache.py        预处理：短边缩放 256 + 中心裁 256 + uint8 缓存
│  ├─ run_experiments.py      四组实验统一入口
│  ├─ summarize.py            汇总 runs → CSV / Markdown / 对比图
│  ├─ requirements.txt        依赖与安装说明
│  ├─ README.md               代码运行说明（更详细的逐步命令）
│  └─ src/
│     ├─ data.py              90:10 划分、训练/评估 transforms、DataLoader
│     ├─ models.py            自搭 CNN（GAP / 大 FC）、ResNet18 迁移
│     ├─ engine.py            训练/评估循环、AMP、早停、保存最优
│     ├─ runner.py            单 run 组装与最终测试评估
│     └─ plots.py             曲线 / 混淆矩阵 / 错误样本
├─ report/
│  └─ PROJECT_REPORT.md       项目报告（方法、四组实验、结果、错误分析、结论）
├─ summary/                   三张汇总表（不含图片）
│  ├─ runs.csv                15 个 run 的逐条结果
│  ├─ experiment_summary.csv  四组实验汇总（均值 ± 标准差）
│  └─ experiment_summary.md   同上，Markdown 版
├─ figures/                   全量图表（15 个 run × 3 张 + 4 张对比图 = 49 张）
│  ├─ <run_id>/               每个 run 一个子目录，共 15 个
│  │  ├─ curves.png           训练/验证 loss 与准确率随 epoch 的曲线
│  │  ├─ test_cm.png          测试集 2500 张的混淆矩阵（行=真实，列=预测）
│  │  └─ test_errors.png      测试集错误样本网格（最多 15 张，标真实/预测/置信度）
│  └─ summary/
│     └─ fig_E1 / E2 / E6 / E7.png   四组实验的均值±标准差对比图
└─ logs/                      三次训练的完整日志（transfer / cnn_experiments / e7_rerun）
```

### 关于图表：这里是全量，不是精选

`figures/` 收录的是**全部 15 个 run 的图表，共 45 张**（每个 run 都有 `curves.png`、`test_cm.png`、`test_errors.png`），加上 `figures/summary/` 下 **4 张**四组对比图，合计 **49 张**。

- 早期版本中放在 `figures/` 根目录的 5 张"精选图"，经 MD5 核对确认是 `resnet_stageB_s42`（3 张）与 `cnn_gap_aug_on_s42`（2 张）的字节级副本，已删除，只保留按 run 归档的这一份，包内不存在重复图片。
- 报告正文里写的路径（`outputs/results/<run_id>/*.png`、`outputs/summary/fig_E*.png`）指源码工程中的位置；在本提交包内分别对应 `figures/<run_id>/*.png` 与 `figures/summary/fig_E*.png`，文件名不变。

## 主要结果

| 实验 | 配置 | 测试准确率（3 seeds） |
|---|---|---|
| E1 | 有增强 vs 无增强 | 91.67% ± 0.53% vs 88.16% ± 8.04% |
| E2 | 自搭 CNN vs ResNet18 迁移 | 91.67% ± 0.53% vs **98.72% ± 0.18%** |
| E6 | 仅训分类头 vs 解冻微调 | 97.81% ± 0.38% vs **98.72% ± 0.18%** |
| E7 | GAP（0.848M） vs 大 FC（26.407M） | **91.67% ± 0.53%** vs 90.76% ± 4.68% |

主结果：ResNet18 两阶段迁移测试准确率 **98.72% ± 0.18%**（2500 张测试集，3 个随机种子）。各 run 的曲线与混淆矩阵见 `figures/<run_id>/`，四组对比图见 `figures/summary/`。

## 代码入口

| 目的 | 入口 |
|---|---|
| 一键完整复现 | `python 一键复现.py`（在 `code/` 目录下执行；自动跳过已完成步骤，可断点续跑） |
| 只跑迁移两组 | `python run_experiments.py --only transfer` |
| 只跑 E1 / E7 | `python run_experiments.py --only e1 e7 --epochs-cnn 12 --epochs-heavy 12 --batch-size 48` |
| 重新汇总图表 | `python summarize.py` |

## 运行方式

数据目录需为：

```
Cat_Dog_data/train/{cat,dog}/*.jpg
Cat_Dog_data/test/{cat,dog}/*.jpg
```

```bash
pip install -r requirements.txt
python check_data.py "d:\DeepStudy\Project1\Cat_Dog_data"
python prepare_cache.py
python run_experiments.py --only transfer --seeds 42 43 44 --force
python run_experiments.py --only e1 e7 --seeds 42 43 44 --force --epochs-cnn 12 --epochs-heavy 12 --batch-size 48
python summarize.py
```

环境：Python 3.10、PyTorch 2.7.1+cu118、torchvision 0.22.1、RTX 4070 Laptop 8GB。

## 说明

- 代码注释为中文；随机种子、数据划分、每个 run 的配置（`config.json`）、逐 epoch 记录（`history.json`）与测试预测（`test_preds.npz`）均已落盘，可复现。
- 本包只放代码、报告、汇总表、图表与日志，**不含** `best.pt` 权重、`results/` 目录、`cache/` 缓存与虚拟环境；重新运行会从头训练。
- 预训练权重来自 `torchvision` 官方 `ResNet18_Weights.IMAGENET1K_V1`，已在报告中标注来源。
- 数据增强、模型结构、训练与评估代码为自行实现；数据管道与两阶段迁移写法参考课程 Notebook Part 7 / Part 8。
- 本次提交不含 PPT。
