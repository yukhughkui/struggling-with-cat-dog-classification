"""模型定义：自搭 CNN（GAP / 大 FC 两种分类头）与 ResNet18 迁移学习。

设计对应方案 §3、§4：
- 自搭 CNN：4 个卷积块（7 个 Conv2d）、4 个 MaxPool2d、3 个 Linear、BN、Dropout
- 轻量化对照：同一主干，GAP 头 vs Flatten+大 FC 头
- 迁移学习：torchvision ResNet18（ImageNet 预训练），替换 fc 为二分类头
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ConvBlock(nn.Module):
    """两个 3x3 卷积 + BN + ReLU，最后接一次 2x2 最大池化。"""

    def __init__(self, in_ch: int, out_ch: int, n_conv: int = 2, use_bn: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_ch
        for _ in range(n_conv):
            layers.append(nn.Conv2d(c_in, out_ch, kernel_size=3, padding=1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            c_in = out_ch
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """自搭 CNN。

    use_gap=True  : 全局平均池化 + 3 层全连接（轻量版，0.85M 参数）
    use_gap=False : 展平 + 大全连接（E7 对照组，约 26M 参数）
    """

    def __init__(
        self,
        num_classes: int = 2,
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        use_bn: bool = True,
        use_gap: bool = True,
        head_hidden: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.use_gap = use_gap
        c1, c2, c3, c4 = channels
        self.b1 = ConvBlock(3, c1, n_conv=2, use_bn=use_bn)
        self.b2 = ConvBlock(c1, c2, n_conv=2, use_bn=use_bn)
        self.b3 = ConvBlock(c2, c3, n_conv=2, use_bn=use_bn)
        self.b4 = ConvBlock(c3, c4, n_conv=1, use_bn=use_bn)

        self.gap = nn.AdaptiveAvgPool2d(1)
        in_features = c4 if use_gap else c4 * 14 * 14  # 224 输入经 4 次池化 → 14x14
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_features, head_hidden)
        self.bn_fc1 = nn.BatchNorm1d(head_hidden) if use_bn else nn.Identity()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(head_hidden, head_hidden // 2)
        self.bn_fc2 = nn.BatchNorm1d(head_hidden // 2) if use_bn else nn.Identity()
        self.dropout3 = nn.Dropout(dropout * 0.6)
        self.fc3 = nn.Linear(head_hidden // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.b1(x)   # [B,3,224,224]   -> [B,32,112,112]
        x = self.b2(x)   # [B,32,112,112]  -> [B,64,56,56]
        x = self.b3(x)   # [B,64,56,56]    -> [B,128,28,28]
        x = self.b4(x)   # [B,128,28,28]   -> [B,256,14,14]
        if self.use_gap:
            x = self.gap(x)                 # [B,256,1,1]
            x = torch.flatten(x, 1)         # [B,256]
        else:
            x = torch.flatten(x, 1)         # [B,50176]
        x = self.dropout1(x)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout3(x)
        return self.fc3(x)


def build_custom_cnn(
    use_gap: bool = True,
    width_mult: float = 1.0,
    use_bn: bool = True,
    num_classes: int = 2,
) -> CustomCNN:
    base = (32, 64, 128, 256)
    channels = tuple(max(8, int(round(c * width_mult / 8)) * 8) for c in base)
    return CustomCNN(
        num_classes=num_classes,
        channels=channels,
        use_bn=use_bn,
        use_gap=use_gap,
    )


def build_resnet18(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """ResNet18 + 新分类头。

    注意：冻结必须在换头之前完成，否则新头也会被冻结（Part 8 的坑）。
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def unfreeze_layer4(model: nn.Module) -> None:
    """阶段 B：解冻最后一个 stage（layer4）以及分类头。"""
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    ps = model.parameters()
    if trainable_only:
        ps = (p for p in ps if p.requires_grad)
    return sum(p.numel() for p in ps)
