# -*- coding: utf-8 -*-
"""
training.models —— 模型注册表
================================================================================
新增一个主干只需两步：
  1. 在 `training/models/` 下实现 nn.Module，声明类属性 `needs = ("v1",)` 或 `("v2",)`；
  2. 在本文件 `_REGISTRY` 注册，并在 train.yaml 的 `model.variants` 里加一节参数。

`needs` 决定数据层加载哪一路张量（V1 帧向量 / V2 字节图），避免无谓 I/O。
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping

import torch.nn as nn

from .common import ModelOutput, count_parameters
from .fdtcn import FDTCN
from .rgbnet import FDRGBNet
from .sequence import LSTMNet, TransformerNet

__all__ = [
    "ModelOutput", "count_parameters", "build_model", "model_inputs", "MODEL_REGISTRY",
]

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "fdtcn": FDTCN,
    "rgbnet": FDRGBNet,
    "lstm": LSTMNet,
    "transformer": TransformerNet,
}


def model_inputs(name: str) -> tuple[str, ...]:
    """该模型需要的数据输入（v1 = 帧向量，v2 = 字节图）。"""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"未知模型 {name}；可选 {sorted(MODEL_REGISTRY)}")
    return tuple(getattr(MODEL_REGISTRY[name], "needs", ("v1",)))


def build_model(name: str, variant_cfg: Mapping[str, Any], *,
                n_classes: int, window_T: int) -> nn.Module:
    """按配置构造模型。

    只把 `__init__` 真实接受的键传进去（其余如 n_classes=auto 已在配置层解析），
    这样新增超参不需要改本函数。
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"未知模型 {name}；可选 {sorted(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    params = inspect.signature(cls.__init__).parameters
    kwargs: dict[str, Any] = {}
    for key, value in variant_cfg.items():
        if key in params:
            kwargs[key] = value
    kwargs["n_classes"] = int(n_classes)
    if "max_len" in params:
        kwargs["max_len"] = int(window_T)
    missing = [k for k in params
               if k not in ("self", "kwargs", "args")
               and params[k].default is inspect.Parameter.empty
               and k not in kwargs]
    if missing:
        raise TypeError(f"模型 {name} 缺少必需参数：{missing}")
    return cls(**kwargs)
