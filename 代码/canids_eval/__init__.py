# -*- coding: utf-8 -*-
"""
canids_eval —— 最小评测包（面向"给了我错误率，但没给口径"的场景）
================================================================================
本包不是新实现，而是把本项目已落地的三个口径**打包成可直接调用的命令**：

  1. `report`  —— 给定 `predictions_test.npz`，报 MCC / AUC-PR / 归一化 AUC-PR / prevalence /
                 该 τ 的实际 FPR（**不重标定**，避免 test 自定 τ 的泄漏）；
  2. `compare` —— 两 run 的配对检验：capture 层符号翻转；若给分组，再报**组层** p 与 p_min；
  3. `density` —— 给定窗内攻击帧数，按密度分箱报召回（解释"差距来自哪里"）。

依赖同仓库的 `代码/training/`（`stats.py` / `metrics.py`），以保证**数字与论文逐位一致**；
不复制公式，避免两套实现漂移。

作者：小巴（WorkBuddy）    日期：2026-10-30
"""

from __future__ import annotations

__all__ = ["cli"]
__version__ = "1.0.0"
