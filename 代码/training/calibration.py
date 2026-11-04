# -*- coding: utf-8 -*-
"""
training.calibration —— 阈值标定（val 选 τ，test 原样使用）
================================================================================
为什么必须单独标定
------------------
块化交错切分后三段先验已趋同，但 HCRL 仍有 train 57% vs test 34% 的差异，
CTT 亦然。固定 0.5 会让 MCC/F1/FPR 同时失真；论文要求"每张结果表披露 prevalence"，
因此 τ 只能在 val 上按目标指标搜索，再原样用于 test。

实现要点
--------
对 N 达 300 万的 test 集，逐阈值重算一遍混淆矩阵是 O(N·G) ≈ 3×10⁹，不可接受。
改为**排序 + 前缀和**：一次 argsort，之后每个 τ 用 searchsorted 取 tp/fp，O(N log N + G log N)。

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _grid(cfg: Mapping[str, Any]) -> np.ndarray:
    g = cfg["calibration"]["grid"]
    lo, hi, step = float(g["min"]), float(g["max"]), float(g["step"])
    n = int(round((hi - lo) / step)) + 1
    return np.linspace(lo, hi, n)


def sweep_curve(y_true: np.ndarray, y_prob: np.ndarray, cfg: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """在阈值网格上一次性算出 MCC / F1 / FPR / Precision / Recall。"""
    y = np.asarray(y_true).astype(np.int64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    if y.size != p.size:
        raise ValueError("y_true 与 y_prob 长度不一致")

    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order]
    cum_pos = np.concatenate([[0], np.cumsum(ys, dtype=np.int64)])
    n_pos = int(ys.sum())
    n_neg = int(ys.size - n_pos)
    taus = _grid(cfg)
    return _curve_at(ps, cum_pos, n_pos, n_neg, taus)


def _curve_at(ps: np.ndarray, cum_pos: np.ndarray, n_pos: int, n_neg: int,
              taus: np.ndarray) -> dict[str, np.ndarray]:
    """给定**已排序**分数与前缀和，在任意阈值数组上算全套指标。"""
    # searchsorted(..., 'left') 返回 < τ 的个数 → 这些是"预测为负"的样本
    n_below = np.searchsorted(ps, taus, side="left")
    fn = cum_pos[n_below]                    # 正样本中预测为负
    tn = n_below - cum_pos[n_below]          # 负样本中预测为负
    tp = n_pos - fn
    fp = n_neg - tn

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(n_pos > 0, tp / max(n_pos, 1), 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), 0.0)
        fpr = np.where(n_neg > 0, fp / max(n_neg, 1), 0.0)
        # 先转 float64 再相乘：tp/fp/tn/fn 为 int64 数组，测试集量级下乘积会 int64 溢出；
        # 转为 float 后 np.sqrt 对任意量级稳定，也不会出现错误负值。
        mcc_den = np.sqrt(
            (tp + fp).astype(np.float64)
            * (tp + fn).astype(np.float64)
            * (tn + fp).astype(np.float64)
            * (tn + fn).astype(np.float64)
        )
        mcc = np.where(mcc_den > 0, (tp * tn - fp * fn) / np.maximum(mcc_den, 1e-12), 0.0)

    return {
        "threshold": taus, "mcc": mcc, "f1": f1, "fpr": fpr,
        "precision": precision, "recall": recall,
        "tp": tp.astype(np.int64), "fp": fp.astype(np.int64),
        "n_pos": np.full_like(taus, n_pos), "n_neg": np.full_like(taus, n_neg),
    }


def select_threshold(curve: Mapping[str, np.ndarray], cfg: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
    """按 `calibration.metric` 在曲线上选 τ。

    - mcc : argmax MCC
    - f1  : argmax F1
    - fpr : 在 FPR ≤ target 的约束下取 Recall 最大（若无满足点则取 FPR 最小点）
    """
    metric = str(cfg["calibration"]["metric"])
    taus = np.asarray(curve["threshold"])
    if metric == "mcc":
        score = np.asarray(curve["mcc"])
        idx = _plateau_index(score, taus)
        key = "mcc"
    elif metric == "f1":
        score = np.asarray(curve["f1"])
        idx = _plateau_index(score, taus)
        key = "f1"
    elif metric == "fpr":
        target = cfg["calibration"]["fpr_target"]
        if target is None:
            raise ValueError("calibration.metric=fpr 时必须给出 calibration.fpr_target")
        fpr = np.asarray(curve["fpr"])
        recall = np.asarray(curve["recall"])
        ok = np.flatnonzero(fpr <= float(target))
        if ok.size:
            idx = int(ok[np.argmax(recall[ok])])
        else:
            idx = int(np.argmin(fpr))
        key = "fpr"
    else:
        raise ValueError(f"未知 calibration.metric={metric}")

    info = {
        "metric": metric,
        "value": float(np.asarray(curve[key])[idx]),
        "mcc": float(np.asarray(curve["mcc"])[idx]),
        "f1": float(np.asarray(curve["f1"])[idx]),
        "fpr": float(np.asarray(curve["fpr"])[idx]),
        "recall": float(np.asarray(curve["recall"])[idx]),
        "precision": float(np.asarray(curve["precision"])[idx]),
        "n_pos": int(np.asarray(curve["n_pos"])[idx]),
        "n_neg": int(np.asarray(curve["n_neg"])[idx]),
        # ★ 2026-10-28：选中点是否落在网格（或哨兵点）的最上端。为 True 时
        #   说明 MCC 在阈值上单调、最优点未被网格覆盖，该 τ 只能当"下界最优"读。
        "at_grid_max": bool(idx == taus.size - 1),
    }
    return float(taus[idx]), info


def _plateau_index(score: np.ndarray, taus: np.ndarray, tol: float = 1e-9) -> int:
    """指标在多个 τ 上并列最优时，取**并列区间的中位 τ**，而不是最小的那个。

    为什么重要：argmax 在常量段上返回首个元素，可能把 τ 拉到 0.001 这类极端值，
    使 FPR 在分布漂移时急剧放大。取中位是"最优集合的中心"，更稳健也更可辩护。
    """
    best = float(np.nanmax(score))
    idxs = np.flatnonzero(np.abs(score - best) <= tol)
    if idxs.size == 0:
        return int(np.nanargmax(score))
    return int(idxs[idxs.size // 2])


def calibrate(y_val: np.ndarray, p_val: np.ndarray, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """标定入口：返回 τ、曲线与选中点信息。

    ★ 2026-10-28：网格上界修正。ROAD 上 5/5 seeds 的 τ 恰好落在网格最大值 0.999，
    说明最优点可能在网格之外（例如"把 0.999 附近的少数误报排除掉"），
    而线性网格无法表达 0.9995 这样的阈值。打开 `grid.include_sentinel` 后，
    一旦命中上界，就在 **(grid_max, max_score]** 的全部去重取值上补搜一次
    （最多几千个点，O(N)），并如实标注 `refined / n_candidates`。
    """
    curve = sweep_curve(y_val, p_val, cfg)
    tau, info = select_threshold(curve, cfg)
    info["refined"] = False

    if bool(cfg["calibration"]["grid"].get("include_sentinel", False)) and info["at_grid_max"]:
        p = np.asarray(p_val, dtype=np.float64).ravel()
        y = np.asarray(y_val).astype(np.int64).ravel()
        grid_max = float(curve["threshold"][-1])
        if np.any(p > grid_max):
            order = np.argsort(p, kind="mergesort")
            ps, ys = p[order], y[order]
            cum = np.concatenate([[0], np.cumsum(ys, dtype=np.int64)])
            n_pos, n_neg = int(ys.sum()), int(ys.size - ys.sum())
            # 决策只取决于"哪些不同取值被判为正"。由于判定是 p ≥ τ，
            # 用**相邻不同取值的中点**作候选，才能表达"保留 0.999、排除 0.9992"
            # 这种落在两个观测值之间的切分（直接用取值本身会漏掉它）。
            uniq = np.unique(ps[ps > grid_max])
            edges = np.concatenate([[grid_max], uniq])
            cand = (edges[:-1] + edges[1:]) / 2.0
            cand = np.append(cand, np.nextafter(float(ps[-1]), np.inf))
            cand = cand[cand > grid_max]
            c2 = _curve_at(ps, cum, n_pos, n_neg, cand)
            tau2, info2 = select_threshold(c2, cfg)
            if float(info2["mcc"]) > float(info["mcc"]) + 1e-12:
                info2["refined"] = True
                # 保留"网格阶段曾命中上界"这一事实：τ 已不在网格内，
                # 但该标记说明原始网格不足以表达最优点。
                info2["grid_hit_upper_bound"] = True
                info2["n_candidates"] = int(cand.size)
                info2["grid_max"] = grid_max
                tau, info = tau2, info2
    return {"threshold": tau, "selected": info, "curve": curve}
