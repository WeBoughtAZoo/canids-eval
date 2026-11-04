# -*- coding: utf-8 -*-
"""
preprocess —— 车载 CAN / CAN FD 三数据集统一预处理包
================================================================================
对应方案：《三数据集形态解析与预处理方案.html》（2026-10-16）

模块职责
--------
config     全局常量与口径（单点定义）
utils      通用工具：hex 向量化解析、日志、目录、md5
cleaning   缺失值处理 / 异常值处理 / 断言校验
parsers    三个独立解析器 → 统一中间表示（中间 schema，变长载荷不填充）
splitting  capture 内时间序 60/20/20 连续切分
features   编码层：feat_v1（16 维帧向量）与 feat_v2（8×8 字节图 + 掩码）
pipeline   编排：解析 → 清洗 → 切分 → 统计 → 编码 → 落盘 → 清单
dataset    惰性窗口化 Dataset（训练只读 L2），含窗内任一帧为攻击的标注规则
"""

__version__ = "1.0.0"
__all__ = [
    "config",
    "utils",
    "cleaning",
    "parsers",
    "splitting",
    "features",
    "pipeline",
    "dataset",
]
