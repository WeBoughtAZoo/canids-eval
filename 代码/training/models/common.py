# -*- coding: utf-8 -*-
"""
training.models.common —— 三个主干共用的小组件
================================================================================
把"因果卷积 / SE / 双头 / 输出契约"集中在一处，保证三主干的结构差异只在
特征提取部分，头结构与输出语义完全一致（这是 T9 对比表公平性的前提）。

输出契约（全项目统一）：
    ModelOutput.logit        [B]      二分类 logit（未经 sigmoid）
    ModelOutput.type_logits  [B, K]   攻击类型 logit（未经 softmax）
    ModelOutput.embedding    [B, D]   时序聚合表征（蒸馏 / 可视化用）

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelOutput:
    logit: torch.Tensor
    type_logits: torch.Tensor
    embedding: torch.Tensor
    # ★ 2026-10-28：逐帧二分类 logits [B, T]，仅当模型启用 aux_frame 时非空。
    #   窗口标签是 any_frame，帧级标签本来就存在；用它对稀疏正例做密集监督，
    #   是审计中"90% 误差来自 ≤8 帧窗口"的直接对策。
    frame_logits: torch.Tensor | None = None

    def prob(self) -> torch.Tensor:
        return torch.sigmoid(self.logit)


class CausalConv1d(nn.Conv1d):
    """严格因果的一维卷积：只在左侧补 (k−1)·d 个零，输出第 t 步只看 ≤t 的输入。

    `padding="same"` 会左右各补一半，导致未来信息泄漏（对在线检测是致命缺陷），
    因此这里显式左补。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size,
                         dilation=dilation, padding=0, **kwargs)
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        return super().forward(x)


class SEBlock(nn.Module):
    """通道注意力（Squeeze-and-Excitation），r = reduction。"""

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # [B,C,T]
        s = x.mean(dim=2)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s.unsqueeze(-1)


class DualHead(nn.Module):
    """二分类头 + 攻击类型多分类头（结构对称，仅输出维度不同）。"""

    def __init__(self, in_dim: int, hidden: int, n_classes: int, dropout: float):
        super().__init__()
        # ★ 子模块名不得用 `type`：会遮蔽 nn.Module.type()，导致 forward 时触发
        #   `Module.type(dst_type)` 而报 "dtype must be a type..."。
        self.bin_head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
        self.type_head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.bin_head(z).squeeze(-1), self.type_head(z)


def temporal_aggregate(h: torch.Tensor, mode: str = "last_mean") -> torch.Tensor:
    """时序聚合：末步 + 均值池化（+ 可选最大池化）。

    为什么加 max
    ------------
    均值池化对"整窗 64 帧里只有 1–2 帧异常"的窗口会把证据稀释 32–64 倍
    （实测 CTT 正窗口中位仅 4 个攻击帧）。max 对稀疏脉冲不衰减，是与均值互补的
    统计量；两者拼接后由线性头自行加权，不引入额外超参。

    mode
    ----
    last_mean       → [last, mean]          维度 2C（默认，保持与既有 113 个 run 一致）
    last_mean_max   → [last, mean, max]     维度 3C（稀疏正例消融用）
    """
    if mode == "last_mean":
        return torch.cat([h[:, :, -1], h.mean(dim=2)], dim=1)
    if mode == "last_mean_max":
        return torch.cat([h[:, :, -1], h.mean(dim=2), h.amax(dim=2)], dim=1)
    raise ValueError(f"未知 pooling 模式：{mode}（可选 last_mean / last_mean_max）")


def pool_dim_factor(mode: str) -> int:
    """pooling 模式对应的特征倍数（用于计算 head 输入维度）。"""
    return 3 if mode == "last_mean_max" else 2


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


@torch.no_grad()
def count_macs(model: nn.Module, *, needs: Sequence[str], window_T: int,
               v1_channels: int, v2_channels: int) -> int:
    """一次前向的乘加次数（MACs），用于 v4 §6.2 的静态效率表。

    统计 Conv1d / Conv2d / Linear / LSTM / MultiheadAttention 的乘加；
    BN、激活、池化按惯例不计。用 forward hook 统计真实执行路径，
    避免手写公式与实际结构脱节。

    ★ 2026-10-27 修正（跨架构不可比）
    ------------------------------
    原实现只挂 Conv/Linear，导致 **LSTM 的循环运算与 attention 的 QK^T/AV
    完全未被统计**（LSTM 实测仅 4,480，而解析值约 3.9e5，差两个数量级），
    使 MACs 列在跨模型比较时系统性偏向循环/注意力基线。现补齐：
      · nn.LSTM            → 逐层 4·H·(in_l+H)·T·B·方向数
      · nn.MultiheadAttention → in_proj(3·E²) + QK^T + AV（out_proj 由 Linear 分支计入）
    """
    macs = 0

    def _hook(module: nn.Module, inputs, output) -> None:
        nonlocal macs
        out = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(module, nn.Conv1d):
            k = module.kernel_size[0]
            macs += int(out.numel() * k * module.in_channels / module.groups)
        elif isinstance(module, nn.Conv2d):
            kh, kw = module.kernel_size
            macs += int(out.numel() * kh * kw * module.in_channels / module.groups)
        elif isinstance(module, nn.Linear):
            macs += int(out.numel() * module.in_features)
        elif isinstance(module, nn.LSTM):
            bf = bool(getattr(module, "batch_first", False))
            b = int(out.shape[0] if bf else out.shape[1])
            t = int(out.shape[1] if bf else out.shape[0])
            h = int(module.hidden_size)
            # nn.LSTM 不暴露 num_directions 属性，只能由 bidirectional 推断
            d = 2 if bool(getattr(module, "bidirectional", False)) else 1
            layers = int(module.num_layers)
            macs += b * t * 4 * h * (int(module.input_size) + h) * d
            if layers > 1:
                # 后续层输入维是 h·D
                macs += b * t * 4 * h * (2 * h) * d * (layers - 1)
        elif isinstance(module, nn.MultiheadAttention):
            bf = bool(getattr(module, "batch_first", False))
            b = int(out.shape[0] if bf else out.shape[1])
            n = int(out.shape[1] if bf else out.shape[0])
            e = int(module.embed_dim)
            # in_proj 的权重是裸 Parameter（非 nn.Linear），Linear 分支抓不到
            macs += b * n * 3 * e * e
            # QK^T 与 attn@V
            macs += b * n * n * e * 2

    handles = [m.register_forward_hook(_hook)
               for m in model.modules()
               if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear,
                                 nn.LSTM, nn.MultiheadAttention))]
    was_training = model.training
    model.eval()
    try:
        batch = 1
        # ★ 2026-10-30 修复（GPU 批次）：哑输入必须与模型同设备/同 dtype。
        #   CPU 训练时模型与哑输入都在 cpu，问题被掩盖；一旦 model.to("cuda")，
        #   这里会在第一个 Conv1d 上报 "weight on cuda, other tensors on cpu"。
        try:
            ref = next(model.parameters())
            dev, dt = ref.device, ref.dtype
        except StopIteration:                      # 无参数模型（不应出现）
            dev, dt = torch.device("cpu"), torch.float32
        kwargs: dict[str, torch.Tensor] = {}
        if "v1" in needs:
            kwargs["x1"] = torch.zeros(batch, window_T, v1_channels, device=dev, dtype=dt)
        if "v2" in needs:
            kwargs["x2"] = torch.zeros(batch, v2_channels, window_T, 8, device=dev, dtype=dt)
        model(**kwargs)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return macs
