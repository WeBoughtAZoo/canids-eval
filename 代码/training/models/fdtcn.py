# -*- coding: utf-8 -*-
"""
training.models.fdtcn —— FD-TCN（主线模型）
================================================================================
对应《最终研究方案与SCI论文大纲-v3定稿.html》§4.5：

    Input (B, 64, 16) float32
    └─ Proj: Conv1d(16 → 32, k=1)
    └─ TCN Block ×4 (Causal Dilated, k=3), dilation d=[1,2,4,8]
       每个 Block: CausalConv1d → BatchNorm → ReLU → Dropout(0.2) → [SE(r=16)] → 残差
    └─ 时序聚合: concat(last_step, mean_pool) → (B, 2C)
    └─ Head A 二分类 / Head B 攻击类型多分类

感受野：RF = 1 + 2·Σ(k−1)·d = 1 + 2·2·(1+2+4+8) = 61 帧（< W=64，覆盖整窗）
残差策略：in≠out 时用 1×1 Conv 投影，in=out 时用恒等（Identity），
          避免为等维块引入无意义的 1×1 卷积（否则参数量会偏离 §4.5 的 55–60 K）。

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .common import (CausalConv1d, DualHead, ModelOutput, SEBlock, pool_dim_factor,
                     temporal_aggregate)


class TCNBlock(nn.Module):
    """因果膨胀残差块。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int,
                 dropout: float, se_reduction: int, residual: str):
        super().__init__()
        self.conv = CausalConv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.se = SEBlock(out_ch, se_reduction) if se_reduction and se_reduction > 0 else nn.Identity()

        if residual == "always" or (residual == "projection_if_needed" and in_ch != out_ch):
            self.shortcut: nn.Module = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        elif residual in ("projection_if_needed", "none"):
            self.shortcut = nn.Identity()
        else:
            raise ValueError(f"未知 residual 策略：{residual}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.bn(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.se(h)
        return h + self.shortcut(x)


class FDTCN(nn.Module):
    """统一帧级编码下的因果膨胀 TCN（双头；不做跨协议训练）。"""

    needs = ("v1",)

    def __init__(self, *, in_channels: int, proj_channels: int,
                 block_channels: Sequence[int], kernel_size: int,
                 dilations: Sequence[int], dropout: float, se_reduction: int,
                 residual: str, head_hidden: int, n_classes: int,
                 dual_head: bool = True, pooling: str = "last_mean",
                 aux_frame: bool = False):
        super().__init__()
        if len(block_channels) != len(dilations):
            raise ValueError("block_channels 与 dilations 长度必须一致")

        self.proj = nn.Conv1d(in_channels, proj_channels, kernel_size=1)
        blocks: list[nn.Module] = []
        prev = proj_channels
        for out_ch, d in zip(block_channels, dilations):
            blocks.append(TCNBlock(prev, out_ch, kernel_size, d, dropout, se_reduction, residual))
            prev = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        # Ab-14：dual_head=False 时只保留二分类头（用于"双头 vs 单头"消融）
        self.dual_head = bool(dual_head)
        self.pooling = str(pooling)
        factor = pool_dim_factor(self.pooling)
        self.head = (DualHead(prev * factor, head_hidden, n_classes, dropout)
                     if self.dual_head else None)
        self.bin_head = (None if self.dual_head else
                         nn.Sequential(nn.Linear(prev * factor, head_hidden), nn.ReLU(inplace=True),
                                       nn.Dropout(dropout), nn.Linear(head_hidden, 1)))
        # ★ 2026-10-28：逐帧二分类头（1×1 卷积），仅 aux_frame=True 时创建。
        #   与主头共享主干，参数量 +C 个（66 K 上约 +65），不改变推理契约。
        self.frame_head = (nn.Conv1d(prev, 1, kernel_size=1) if bool(aux_frame) else None)

    @property
    def receptive_field(self) -> int:
        rf = 1
        for blk in self.blocks:
            rf += 2 * (blk.conv.kernel_size[0] - 1) * blk.conv.dilation[0]
        return rf

    def forward(self, x1: torch.Tensor, **_: torch.Tensor) -> ModelOutput:
        """x1: [B, T, C] → ModelOutput。"""
        h = x1.transpose(1, 2)          # [B,C,T]
        h = self.proj(h)
        h = self.blocks(h)
        h = self.drop(h)
        z = temporal_aggregate(h, self.pooling)
        if self.head is not None:
            logit, type_logits = self.head(z)
        else:
            logit = self.bin_head(z).squeeze(-1)
            # 单头模式：类型 logits 置零占位，loss 的 lambda_b 必须为 0
            type_logits = logit.new_zeros((logit.size(0), 1))
        frame_logits = (self.frame_head(h).squeeze(1) if self.frame_head is not None else None)
        return ModelOutput(logit=logit, type_logits=type_logits, embedding=z,
                           frame_logits=frame_logits)
