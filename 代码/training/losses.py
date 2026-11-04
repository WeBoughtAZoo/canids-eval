# -*- coding: utf-8 -*-
"""
training.losses —— 多任务损失
================================================================================
§4.8：L = λ_A·(加权 BCE 或 Focal) + λ_B·CE + λ_reg·||W||₂

- 二分类头承担"检测"（主指标来源），多分类头只承担"攻击类型归因"；
- `pos_weight` 由训练集窗口级 neg/pos 计算（`auto`），HCRL 会得到 <1（正类占多数），
  这是数据事实，不应人为改成 >1；
- `lambda_reg` 默认 0：L2 由 `optimizer.weight_decay` 承担，避免同一正则被计两次；
  若审稿要求显式 L2，把 lambda_reg 调正、weight_decay 调 0 即可（两者都在 YAML 里）。

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Sigmoid Focal Loss（Lin et al., ICCV 2017），用于消融 Ab-6。"""

    def __init__(self, gamma: float, alpha: float):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logit)
        ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - p_t).pow(self.gamma) * ce).mean()


class MultiTaskLoss(nn.Module):
    """二分类 + 攻击类型多分类 + 可选显式 L2。"""

    def __init__(self, cfg: Mapping[str, Any], *, pos_weight: float,
                 class_weight: torch.Tensor | None, n_classes: int,
                 aux_frame_weight: float = 0.0, frame_pos_weight: float = 1.0):
        super().__init__()
        bcfg = cfg["loss"]["binary"]
        self.bin_type = str(bcfg["type"])
        self.lambda_a = float(cfg["loss"]["lambda_a"])
        self.lambda_b = float(cfg["loss"]["lambda_b"])
        self.lambda_reg = float(cfg["loss"]["lambda_reg"])
        self.n_classes = int(n_classes)
        # ★ 2026-10-28：逐帧辅助监督。窗口标签 any_frame 把"1 帧异常"和"64 帧异常"
        #   混为一类；帧级标签本来就在张量里，直接拿来做密集监督，是最对症的正则。
        self.aux_frame_weight = float(aux_frame_weight)
        self.frame_pos_weight = float(frame_pos_weight)

        if self.bin_type == "weighted_bce":
            self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
            self.binary = None
        elif self.bin_type == "focal":
            self.binary = BinaryFocalLoss(float(bcfg["focal_gamma"]), float(bcfg["focal_alpha"]))
            self.register_buffer("pos_weight", torch.tensor([1.0], dtype=torch.float32))
        else:
            raise ValueError(f"未知 loss.binary.type={self.bin_type}")

        mcfg = cfg["loss"]["multiclass"]
        self.label_smoothing = float(mcfg["label_smoothing"])
        cw = class_weight if str(mcfg["class_weight"]) == "balanced" else None
        self.register_buffer("class_weight", cw if cw is not None
                             else torch.zeros(0), persistent=False)

    def forward(self, out, y: torch.Tensor, y_type: torch.Tensor,
                y_frame: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        yf = y.float()
        if self.bin_type == "weighted_bce":
            loss_a = F.binary_cross_entropy_with_logits(
                out.logit, yf, pos_weight=self.pos_weight.to(out.logit.device))
        else:
            loss_a = self.binary(out.logit, yf)

        if self.n_classes > 1:
            cw = self.class_weight.to(out.type_logits.device) if self.class_weight.numel() else None
            loss_b = F.cross_entropy(out.type_logits, y_type.long(),
                                     weight=cw, label_smoothing=self.label_smoothing)
        else:
            loss_b = out.type_logits.sum() * 0.0

        loss = self.lambda_a * loss_a + self.lambda_b * loss_b

        # ---- 逐帧辅助 BCE（仅当模型有 frame_logits 且传入 y_frame 时生效） ----
        loss_f = out.logit.sum() * 0.0
        fl = getattr(out, "frame_logits", None)
        if self.aux_frame_weight > 0 and fl is not None and y_frame is not None:
            loss_f = F.binary_cross_entropy_with_logits(
                fl, y_frame.float(),
                pos_weight=torch.as_tensor(self.frame_pos_weight, device=fl.device),
            )
            loss = loss + self.aux_frame_weight * loss_f

        if self.lambda_reg > 0:
            l2 = sum(p.pow(2).sum() for p in self.parameters() if p.requires_grad and p.ndim > 1)
            loss = loss + self.lambda_reg * l2

        return {"loss": loss, "loss_bin": loss_a.detach(),
                "loss_type": loss_b.detach(), "loss_frame": loss_f.detach()}


def compute_class_weight(counts: Mapping[int, int], n_classes: int) -> torch.Tensor:
    """按窗口数反频率加权（sklearn 'balanced' 口径：n / (K · n_c)）。"""
    total = float(sum(counts.values()))
    w = torch.ones(n_classes, dtype=torch.float32)
    for c in range(n_classes):
        n_c = float(counts.get(c, 0))
        w[c] = (total / (n_classes * n_c)) if n_c > 0 else 0.0
    return w
