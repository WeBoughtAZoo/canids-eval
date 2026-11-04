# -*- coding: utf-8 -*-
"""
training —— 车载 CAN / CAN FD 入侵检测：训练与测试模块
================================================================================
与 `preprocess/` **完全隔离**：本包含不 import 任何 `preprocess.*`，
唯一接口是磁盘上的 02_tensors/*.npy 布局与 03_manifest/* 清单。

模块职责（自上而下依赖）
------------------------
    config      配置加载 / 校验 / 消融派生 / 路径解析（唯一参数入口）
    manifest    03_manifest 只读索引（切分区间、攻击类型、窗口数校验）
    data        惰性窗口 Dataset + 采集级采样 + DataLoader
    models/     主干网络（fdtcn / rgbnet / lstm / transformer）
    losses      多任务损失（加权 BCE / Focal + CE）
    calibration τ 搜索（排序 + 前缀和，O(N log N + G log N)）
    metrics     指标 / bootstrap / 分层表 / 可评估性分级
    stats       McNemar / Wilcoxon / Holm
    trainer     训练循环（早停 val MCC、余弦退火、最优回滚）
    evaluator   推理 + 评估落盘
    kdlogic     KD-Logic（DL → 约束 CART 蒸馏）
    figures     SCI 图表（14 张，PDF + PNG）
    tables      SCI 表格（booktabs LaTeX + CSV：T7/T8/T10/T12/T13）
    compare     配对显著性检验（McNemar + Holm）
    artifacts   运行目录 / checkpoint / manifest
    pipeline    单 run 编排（数据 → 训练 → 评估 → 出图）

版本：1.0.0（2026-10-17）
"""

__version__ = "1.0.0"
__all__ = ["config", "manifest", "data", "models", "losses", "calibration",
           "metrics", "stats", "trainer", "evaluator", "kdlogic", "figures",
           "tables", "compare", "artifacts", "pipeline", "utils", "style"]
