# -*- coding: utf-8 -*-
"""
training.density —— 攻击帧密度分层评测
================================================================================
为什么需要它
------------
窗口标签规则是 `any_frame`：窗内**任一帧**为攻击即标正。于是同一"正窗口"
可能含 1 个攻击帧，也可能 64 个全是攻击帧。二者的检测难度完全不同，
但 MCC 把它们混成一个数，掩盖了真正的失败模式。

2026-10-28 审计（CTT test）：正窗口中位仅 **4** 个攻击帧（W=64），
72.6% 的窗口 ≤8 帧；把这批窗口按密度分箱后，FD-TCN 与 LSTM 的召回差
从 0（≥25 帧）扩大到 20–33 pt（≤8 帧），**90% 的总召回差距来自 ≤8 帧的窗口**。
因此密度分层不是"附加分析"，而是解释架构差异的主证据。

口径
----
· 密度 = 窗内攻击帧数（整数 1..W），由 `dataset.sources[c].label` 的前缀和精确得到；
· 分箱边界来自 `eval.density_bins`（默认 1–2 / 3–8 / 9–24 / 25–48 / 49–64）；
· 每箱同时报"自身 τ 下召回"与"匹配 FPR 操作点下召回"，后者用 val 标定的 τ；
· 只统计**正窗口**；负窗口不参与（它们没有密度可言）。

作者：小巴（WorkBuddy）    日期：2026-10-28
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .data import WindowDataset


def window_attack_counts(dataset: WindowDataset, cap: np.ndarray,
                         pos: np.ndarray) -> np.ndarray:
    """逐窗口的攻击帧数（int16）。

    参数
    ----
    dataset : WindowDataset（或 limit_dataset 产生的浅拷贝，`sources` 仍有效）
    cap     : [N] capture 下标（`predict()` 的 `cap`）
    pos     : [N] 窗口在 capture 内的起始帧号（`predict()` 的 `pos`）

    实现：对每个 capture 读一次 `label.npy`，做前缀和，再按窗口端点相减。
    单 capture 只加载一次，CTT 最大 capture 约 9.2×10⁵ 帧（<1 MB int8）。
    """
    cap = np.asarray(cap).ravel()
    pos = np.asarray(pos).ravel()
    out = np.zeros(cap.size, dtype=np.int16)
    if cap.size == 0:
        return out
    sources = getattr(dataset, "sources", None)
    if sources is None:
        raise RuntimeError("dataset 缺少 sources，无法读取帧级标签")

    win = int(dataset.window)
    order = np.argsort(cap, kind="stable")
    bounds = np.searchsorted(cap[order], np.arange(int(cap.max()) + 2))
    for c in range(int(cap.max()) + 1):
        idx = order[bounds[c]:bounds[c + 1]]
        if idx.size == 0:
            continue
        lab = np.asarray(sources[c].label, dtype=np.int32)
        cum = np.concatenate([[0], np.cumsum(lab, dtype=np.int64)])
        s = pos[idx]
        if int(s.max()) + win > lab.size:
            raise ValueError(f"capture {c}: 窗口越界（pos {int(s.max())}+{win} > {lab.size}）")
        out[idx] = (cum[s + win] - cum[s]).astype(np.int16)
    return out


def density_table(y_true: np.ndarray, y_prob: np.ndarray, counts: np.ndarray,
                  threshold: float, *,
                  bins: Sequence[Sequence[int]],
                  tau_at_fpr: float | None = None) -> list[dict[str, Any]]:
    """按攻击帧数分箱的召回表（只含正窗口）。"""
    y = np.asarray(y_true).astype(np.int64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    d = np.asarray(counts).ravel()
    if not (y.size == p.size == d.size):
        raise ValueError("y / p / counts 长度不一致")

    pos_mask = y == 1
    n_pos = int(pos_mask.sum())
    rows: list[dict[str, Any]] = []
    if n_pos == 0:
        return rows

    in_bins = _mask_from_bins(d, bins)
    for lo, hi in bins:
        m = pos_mask & (d >= int(lo)) & (d <= int(hi))
        n = int(m.sum())
        if n == 0:
            continue
        rows.append({
            "density_bin": f"{int(lo)}-{int(hi)}",
            "density_lo": int(lo), "density_hi": int(hi),
            "n_positive": n,
            "frac_of_positives": n / n_pos,
            "recall": float((p[m] >= float(threshold)).mean()),
            "recall_at_fpr": (float((p[m] >= float(tau_at_fpr)).mean())
                              if tau_at_fpr is not None and np.isfinite(tau_at_fpr) else float("nan")),
            "mean_score": float(p[m].mean()),
            "median_score": float(np.median(p[m])),
        })

    # 超出最后一档上界的窗口（例如 W>64 时）单列，避免静默丢弃
    rest = pos_mask & ~in_bins
    if rest.any():
        rows.append({
            "density_bin": "out_of_range", "density_lo": int(bins[-1][1]) + 1,
            "density_hi": int(d.max()), "n_positive": int(rest.sum()),
            "frac_of_positives": float(rest.sum()) / n_pos,
            "recall": float((p[rest] >= float(threshold)).mean()),
            "recall_at_fpr": (float((p[rest] >= float(tau_at_fpr)).mean())
                              if tau_at_fpr is not None and np.isfinite(tau_at_fpr) else float("nan")),
            "mean_score": float(p[rest].mean()),
            "median_score": float(np.median(p[rest])),
        })

    rows.append({
        "density_bin": "overall", "density_lo": 1, "density_hi": int(d.max()),
        "n_positive": n_pos, "frac_of_positives": 1.0,
        "recall": float((p[pos_mask] >= float(threshold)).mean()),
        "recall_at_fpr": (float((p[pos_mask] >= float(tau_at_fpr)).mean())
                          if tau_at_fpr is not None and np.isfinite(tau_at_fpr) else float("nan")),
        "mean_score": float(p[pos_mask].mean()),
        "median_score": float(np.median(p[pos_mask])),
    })
    return rows


def _mask_from_bins(d: np.ndarray, bins: Sequence[Sequence[int]]) -> np.ndarray:
    m = np.zeros(d.shape, dtype=bool)
    for lo, hi in bins:
        m |= (d >= int(lo)) & (d <= int(hi))
    return m


def summary(y_true: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    """密度分布的浓缩描述（进 metrics_*.json，便于跨 run 比较）。"""
    y = np.asarray(y_true).astype(np.int64).ravel()
    d = np.asarray(counts).ravel()
    pos = d[y == 1]
    if pos.size == 0:
        return {"n_positive": 0}
    return {
        "n_positive": int(pos.size),
        "median": float(np.median(pos)),
        "mean": float(pos.mean()),
        "p10": float(np.percentile(pos, 10)),
        "p90": float(np.percentile(pos, 90)),
        "frac_le_2": float((pos <= 2).mean()),
        "frac_le_8": float((pos <= 8).mean()),
        "frac_ge_25": float((pos >= 25).mean()),
    }
