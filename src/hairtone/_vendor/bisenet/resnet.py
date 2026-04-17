# Vendored and modified from
# https://github.com/zllrunning/face-parsing.PyTorch/blob/master/resnet.py.
# Original copyright (c) 2019 zll. Released under the MIT License — see the
# full text in ./LICENSE (ships inside every wheel and sdist).
#
# Modifications made for hairtone (2026-04-18):
#   - removed the ImageNet auto-download (`modelzoo.load_url`) at construction
#     because the caller always overwrites with the CelebAMask-HQ state dict
#   - removed training-only helpers (`get_params`, `init_weight`)
#   - removed the `__main__` demo block
#   - added type annotations and module-level helper prefixes
#
# Vendoring rationale: pulling the whole upstream repo via pip drags in
# dataset loaders, trainers, and a stale pretrained-ResNet URL. hairtone
# only needs the ContextPath forward pass, so we ship the minimal subset.

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = _conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        shortcut = x if self.downsample is None else self.downsample(x)
        return self.relu(shortcut + residual)


def _create_layer_basic(
    in_chan: int, out_chan: int, bnum: int, stride: int = 1
) -> nn.Sequential:
    layers: list[nn.Module] = [BasicBlock(in_chan, out_chan, stride=stride)]
    for _ in range(bnum - 1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    """ResNet-18 backbone without the fully-connected head.

    Unlike the upstream implementation, construction does **not** download
    the ImageNet pretrained weights — hairtone always loads a full BiSeNet
    state dict after construction, so doing so would be wasteful and would
    add an unnecessary network dependency at library import time.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = _create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = _create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = _create_layer_basic(256, 512, bnum=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)
        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32
