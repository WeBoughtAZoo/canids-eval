# -*- coding: utf-8 -*-
"""
training.models.rgbnet —— FD-RGBNet（下界基线 / 受控对照）
================================================================================
对应 §4.6：深度可分离卷积 CNN（DWConv + PWConv）×3 → GAP → 双头。

★ 同构性声明（论文必须写）：本模型的 CAN → "类 RGB 图像" → CNN 流水线与
  CANDIDS 同构，作为 **isomorphic baseline** 使用，用于隔离"统一帧级编码 +
  填充掩码"带来的增量。它**不是**推荐方案。

输入为 V2 字节图 [B, C, T, 8]：C=8（4 数据通道 + 4 逐字节填充掩码通道），
消融 Ab-1b 去掩码后 C=4。去掉掩码 ⇒ 模型无法区分"真实 0x00"与"填充 0x00"，
这正是 C1 需要隔离量化的填充伪影。

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DualHead, ModelOutput


class DepthwiseSeparableBlock(nn.Module):
    """DWConv(k×k, groups=C) → BN → ReLU → PWConv(1×1) → BN → ReLU。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, padding=pad, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.dw(x)), inplace=True)
        x = F.relu(self.bn2(self.pw(x)), inplace=True)
        return self.drop(x)


class FDRGBNet(nn.Module):
    """字节图 → 轻量 CNN（双头）。"""

    needs = ("v2",)

    def __init__(self, *, in_channels: int, stem_channels: int,
                 block_channels: Sequence[int], kernel_size: int,
                 dropout: float, head_hidden: int, n_classes: int):
        super().__init__()
        pad = kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        prev = stem_channels
        for out_ch in block_channels:
            blocks.append(DepthwiseSeparableBlock(prev, out_ch, kernel_size, dropout))
            prev = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.head = DualHead(prev, head_hidden, n_classes, dropout)

    def forward(self, x2: torch.Tensor, **_: torch.Tensor) -> ModelOutput:
        """x2: [B, C, T, 8] → ModelOutput。"""
        h = self.stem(x2)
        h = self.blocks(h)
        z = h.mean(dim=(2, 3))          # GAP over 时间 × 字节
        z = self.drop(z)
        logit, type_logits = self.head(z)
        return ModelOutput(logit=logit, type_logits=type_logits, embedding=z)
