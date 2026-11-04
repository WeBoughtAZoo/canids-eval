# -*- coding: utf-8 -*-
"""
training.models.sequence —— 主干族对照（LSTM / 因果 Transformer）
================================================================================
用途：§6.4 对比方法中的"时序深度"一族。两者都**严格因果**：
  - LSTM 只取最后一个时间步的隐状态；
  - Transformer 用上三角 −inf 掩码，第 t 个位置只能看到 ≤t。
否则与 FD-TCN 的"在线可部署"前提不对等，对比失去意义。

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .common import DualHead, ModelOutput, temporal_aggregate


class LSTMNet(nn.Module):
    needs = ("v1",)

    def __init__(self, *, in_channels: int, hidden: int, num_layers: int,
                 dropout: float, head_hidden: int, n_classes: int,
                 aux_frame: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.drop = nn.Dropout(dropout)
        self.head = DualHead(hidden, head_hidden, n_classes, dropout)
        # 逐帧二分类头（aux_frame 消融用）；默认关闭，保持与既有 run 一致
        self.frame_head = nn.Linear(hidden, 1) if bool(aux_frame) else None

    def forward(self, x1: torch.Tensor, **_: torch.Tensor) -> ModelOutput:
        h, _ = self.lstm(x1)                 # [B,T,H]
        z = self.drop(h[:, -1])
        logit, type_logits = self.head(z)
        frame_logits = self.frame_head(h).squeeze(-1) if self.frame_head is not None else None
        return ModelOutput(logit=logit, type_logits=type_logits, embedding=z,
                           frame_logits=frame_logits)


class PositionalEncoding(nn.Module):
    """正弦位置编码（非学习参数，避免小模型被位置参数占满容量）。"""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerNet(nn.Module):
    """因果 Transformer 编码器（小模型，用于主干族对照）。"""

    needs = ("v1",)

    def __init__(self, *, in_channels: int, d_model: int, nhead: int,
                 num_layers: int, dim_feedforward: int, dropout: float,
                 head_hidden: int, n_classes: int, max_len: int,
                 aux_frame: bool = False):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.head = DualHead(d_model * 2, head_hidden, n_classes, dropout)
        self.frame_head = nn.Linear(d_model, 1) if bool(aux_frame) else None

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def forward(self, x1: torch.Tensor, **_: torch.Tensor) -> ModelOutput:
        h = self.pos(self.proj(x1))                     # [B,T,D]
        h = self.encoder(h, mask=self._causal_mask(h.size(1), h.device))
        frame_logits = self.frame_head(h).squeeze(-1) if self.frame_head is not None else None
        h = h.transpose(1, 2)                           # [B,D,T] → 复用 temporal_aggregate
        z = self.drop(temporal_aggregate(h))
        logit, type_logits = self.head(z)
        return ModelOutput(logit=logit, type_logits=type_logits, embedding=z,
                           frame_logits=frame_logits)
