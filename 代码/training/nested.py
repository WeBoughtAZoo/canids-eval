# -*- coding: utf-8 -*-
"""
training.nested —— 嵌套（录制组 → capture → 窗口）推断
================================================================================
要解决的问题（审计 D4-1 / R-1）
--------------------------------
簇级 McNemar 已把"窗口当独立样本"这一层修掉，但**停得太早**：CTT v1.5 的
224 个 test capture 来自 `set_01`–`set_04` 四次独立录制，同一 SET 内的文件共享
车辆状态、日志、注入脚本与采集环境。把 224 个文件当可交换单元，仍是
pseudoreplication，只是升了一级 —— 而这正是本文用来批评他人的同一种错误。

因此本模块提供两层口径：

  1. **file 层**（= 现有 `mcnemar_cluster`）：每 capture 一个净优势 d_k，
     在 K 个 capture 上做符号翻转；
  2. **group 层**（本模块新增）：先把 d_k 汇成 D_g = Σ_{k∈g} d_k，
     在 G 个录制组上做符号翻转，报最粗粒度下的 p。

并给出**结构性下限**：G 个组全部同向时最小可达 p = 2/2^G。CTT 的 G=4 ⇒
p_min = 0.125，ROAD/HCRL 的 G = capture 数（无上层嵌套，退化为 file 层）。
这必须与 p 一起报，否则"p 不显著"会被误读为"效应不存在"。

同时给 ΔMCC 的**两阶段自助**置信区间：外层重采样组、内层重采样组内 capture。
它回答的不是"两个模型谁更强"，而是"换一批录制会话，这个差值会漂多少"。

口径约定
--------
- 两模型各用自己 val 标定的 τ 生成判决（与 `compare.compare_pair` 一致）；
- 差值定义为 A − B（调用方决定 A/B 的角色）；
- 只用计数聚合，不重算逐窗口分数 —— 因此可在既有 `predictions_*.npz` 上离线复算。

作者：小巴（WorkBuddy）    日期：2026-10-30（审计整改 D4-1）
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .stats import mcnemar_cluster


def _mcc(tp: float, fp: float, fn: float, tn: float) -> float:
    """由 2×2 计数算 MCC（NumPy 2.x 下必须转 float，防 int64 溢出）。"""
    tp, fp, fn, tn = (float(tp), float(fp), float(fn), float(tn))
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den > 0 else float("nan")


def _per_cluster_counts(y: np.ndarray, pred: np.ndarray, inv: np.ndarray,
                        k: int) -> dict[str, np.ndarray]:
    pos = y == 1
    neg = ~pos
    hit = pred == 1
    return {
        "tp": np.bincount(inv, weights=(pos & hit), minlength=k),
        "fp": np.bincount(inv, weights=(neg & hit), minlength=k),
        "fn": np.bincount(inv, weights=(pos & ~hit), minlength=k),
        "tn": np.bincount(inv, weights=(neg & ~hit), minlength=k),
    }


def _sum_counts(c: Mapping[str, np.ndarray], idx: np.ndarray) -> dict[str, float]:
    return {key: float(np.asarray(c[key])[idx].sum()) for key in ("tp", "fp", "fn", "tn")}


def mcnemar_nested(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                   clusters: np.ndarray, groups: np.ndarray, *,
                   n_boot: int = 1000, n_perm: int = 20000,
                   exact_max_groups: int = 16, seed: int = 0) -> dict:
    """两层符号翻转 + 两阶段自助。参数与返回见模块文档。

    `clusters` 是逐窗口的 capture 下标（npz 的 `cap`）；`groups` 是**逐 capture** 的
    组标签（长度 = 簇数，下标即 cluster id），由 `groups.capture_groups()` 给出。
    """
    y = np.asarray(y_true).astype(np.int64).ravel()
    a = np.asarray(pred_a).astype(np.int64).ravel()
    b = np.asarray(pred_b).astype(np.int64).ravel()
    cl = np.asarray(clusters).ravel()
    if not (y.size == a.size == b.size == cl.size):
        raise ValueError("y_true / pred_a / pred_b / clusters 长度不一致")

    # ---- file 层：沿用既有簇级检验（保证与 comparison.csv 同源可核对）
    file_level = mcnemar_cluster(y, a, b, cl, n_perm=n_perm, seed=seed)
    _, inv = np.unique(cl, return_inverse=True)
    k = int(inv.max()) + 1

    g = np.asarray(groups)
    if g.size != k:
        raise ValueError(f"groups 长度 {g.size} 与簇数 {k} 不一致（映射顺序错位？）")
    gid, ginv = np.unique(g, return_inverse=True)
    n_groups = int(gid.size)

    # ---- 统计量：每捕获净优势 d_k = (A 错 B 对) − (A 对 B 错)
    a_ok = (a == y)
    b_ok = (b == y)
    d = (np.bincount(inv, weights=(~a_ok & b_ok), minlength=k).astype(np.int64)
         - np.bincount(inv, weights=(a_ok & ~b_ok), minlength=k).astype(np.int64))
    D = np.bincount(ginv, weights=d, minlength=n_groups)

    out = {
        "n_windows": int(y.size), "n_clusters": k, "n_groups": n_groups,
        "groups": [str(x) for x in gid.tolist()],
        "p_value_file": float(file_level["p_value"]),
        "method_file": str(file_level["method"]),
        "p_min_file": float(file_level["p_min"]),
        "b": int(file_level["b"]), "c": int(file_level["c"]),
        "cluster_net_advantage": [int(x) for x in D.tolist()],
        "sign_consistency": float(np.mean(np.sign(D[D != 0]) > 0)) if np.any(D != 0) else float("nan"),
    }

    # ---- group 层符号翻转
    if not np.any(D != 0):
        out.update({"p_value_group": 1.0, "method_group": "group_degenerate",
                    "p_min_group": 1.0, "p_at_floor_group": True, "stat_group": 0.0})
    else:
        t_obs = float(abs(D.sum()))
        if n_groups <= exact_max_groups:
            signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * n_groups)).T.reshape(-1, n_groups)
            null = np.abs(signs @ D.astype(np.float64))
            p = float((null >= t_obs - 1e-12).mean())
            method, p_min = "group_exact_signflip", 2.0 / float(signs.shape[0])
        else:
            rng = np.random.default_rng(seed)
            d_f = D.astype(np.float64)
            hit = 0
            done = 0
            while done < int(n_perm):
                m = min(4096, int(n_perm) - done)
                s = rng.integers(0, 2, size=(m, n_groups)).astype(np.float64) * 2 - 1
                hit += int((np.abs(s @ d_f) >= t_obs - 1e-12).sum())
                done += m
            p = float((hit + 1) / (int(n_perm) + 1))
            method, p_min = "group_monte_carlo_signflip", 1.0 / (int(n_perm) + 1)
        out.update({"p_value_group": p, "method_group": method, "stat_group": t_obs,
                    "p_min_group": p_min, "p_at_floor_group": bool(p <= p_min + 1e-15)})

    # ---- 两阶段自助：外层组、内层组内 capture
    ca = _per_cluster_counts(y, a, inv, k)
    cb = _per_cluster_counts(y, b, inv, k)
    members = [np.flatnonzero(ginv == j) for j in range(n_groups)]
    rng = np.random.default_rng(seed + 1)
    deltas = np.empty(int(n_boot), dtype=np.float64)
    for r in range(int(n_boot)):
        picked = rng.integers(0, n_groups, size=n_groups)
        idx_parts = [members[j][rng.integers(0, members[j].size, size=members[j].size)]
                     for j in picked]
        idx = np.concatenate(idx_parts) if idx_parts else np.zeros(0, np.int64)
        sa = _sum_counts(ca, idx)
        sb = _sum_counts(cb, idx)
        deltas[r] = (_mcc(sa["tp"], sa["fp"], sa["fn"], sa["tn"])
                     - _mcc(sb["tp"], sb["fp"], sb["fn"], sb["tn"]))

    valid = deltas[np.isfinite(deltas)]
    total_a = _sum_counts(ca, np.arange(k))
    total_b = _sum_counts(cb, np.arange(k))
    out.update({
        "delta_mcc": float(_mcc(total_a["tp"], total_a["fp"], total_a["fn"], total_a["tn"])
                           - _mcc(total_b["tp"], total_b["fp"], total_b["fn"], total_b["tn"])),
        "delta_ci_lo": float(np.percentile(valid, 2.5)) if valid.size else float("nan"),
        "delta_ci_hi": float(np.percentile(valid, 97.5)) if valid.size else float("nan"),
        "n_boot": int(valid.size),
        "bootstrap": "two_stage_group_then_cluster",
    })
    return out


def group_table(y_true: np.ndarray, pred: np.ndarray, clusters: np.ndarray,
                groups: np.ndarray, *, threshold: float | None = None) -> list[dict]:
    """逐组精度表（F3 的"按 SET 分组报 ΔMCC"直接用它）。

    调用方对 full 与 PF 各调一次，再按组拼 Δ —— 见 `run_audit.py setwise`。
    """
    y = np.asarray(y_true).astype(np.int64).ravel()
    p = np.asarray(pred).astype(np.int64).ravel()
    cl = np.asarray(clusters).ravel()
    _, inv = np.unique(cl, return_inverse=True)
    k = int(inv.max()) + 1
    cnt = _per_cluster_counts(y, p, inv, k)
    g = np.asarray(groups)
    gid, ginv = np.unique(g, return_inverse=True)

    rows: list[dict] = []
    for j, name in enumerate(gid.tolist()):
        idx = np.flatnonzero(ginv == j)
        s = _sum_counts(cnt, idx)
        n_pos = s["tp"] + s["fn"]
        rows.append({
            "group": str(name), "n_clusters": int(idx.size),
            "n_windows": int(s["tp"] + s["fp"] + s["fn"] + s["tn"]),
            "n_pos": int(n_pos), "n_neg": int(s["fp"] + s["tn"]),
            "prevalence": float(n_pos / max(1.0, s["tp"] + s["fp"] + s["fn"] + s["tn"])),
            "mcc": float(_mcc(s["tp"], s["fp"], s["fn"], s["tn"])),
            "recall": float(s["tp"] / n_pos) if n_pos else float("nan"),
            "fpr": float(s["fp"] / (s["fp"] + s["tn"])) if (s["fp"] + s["tn"]) else float("nan"),
            "threshold": threshold,
        })
    return rows


def merge_group_delta(rows_a: Sequence[Mapping], rows_b: Sequence[Mapping]) -> list[dict]:
    """把同一组的两个模型行拼成 Δ 表（A − B）。缺失组按 NaN 处理并标注。"""
    by_b = {r["group"]: r for r in rows_b}
    out: list[dict] = []
    for ra in rows_a:
        rb = by_b.get(ra["group"])
        if rb is None:
            continue
        out.append({
            "group": ra["group"], "n_clusters": ra["n_clusters"], "n_windows": ra["n_windows"],
            "n_pos": ra["n_pos"], "prevalence": ra["prevalence"],
            "mcc_a": ra["mcc"], "mcc_b": rb["mcc"],
            "delta_mcc": (ra["mcc"] - rb["mcc"])
            if (math.isfinite(ra["mcc"]) and math.isfinite(rb["mcc"])) else float("nan"),
            "recall_a": ra["recall"], "recall_b": rb["recall"],
            "delta_recall": ra["recall"] - rb["recall"],
            "fpr_a": ra["fpr"], "fpr_b": rb["fpr"],
        })
    return out
