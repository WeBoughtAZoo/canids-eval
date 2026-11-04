# -*- coding: utf-8 -*-
"""
training.openset —— 未知攻击的标准度量（替代"full−known 差值"）
================================================================================
为什么必须重算（审计 F4 / D6）
------------------------------
v6 把 `full` 与 `known-attack` 两个口径的 MCC 差称作 "open-set degradation"。
该差值是"从测试集剔除零样本家族后重算 MCC"，因此**数学上必然为正**，且随模型
变强而变小（弱模型 full 更差 → 差值更大）。它度量的是<b>最难子集的占比</b>，
不是识别未知攻击的能力；任何分类器都会得到同量级的值，不可跨模型比较开集能力。

本模块改用开集/新颖性检测的标准量，全部可在既有 `predictions_test.npz` 上离线复算：

  · **AUROC(unknown vs normal)**：未知攻击窗口 vs **正常**窗口 —— 操作上对应
    "能不能把未知攻击当成异常从正常流量里挑出来"，不受"已知攻击本来就得高分"的干扰；
  · **AUROC(unknown vs known-all)**：标准 OOD 定义（负类含已知攻击）。本文的二元
    检测器会把已知攻击判为攻击（这是正确的），因此在 OOD 口径下它们全被算作
    "高分负类"，AUROC 被结构性压低。两个数并列报，并写明口径差异；
  · **TPR@FPR_normal**：在正常窗口误报率固定为 1% / 0.1% 时，未知攻击的召回；
  · **TPR@τ**：部署操作点（val 标定 τ）上的未知召回 —— 与既有 masquerade 数字对齐。

为什么**不报 OSCR / AUOSCR**
----------------------------
OSCR（Dhamija et al., 2018）要求分类器具备**拒识类**：已知样本被"拒绝"才算横轴。
本文五个主干都是二元检测器（正常 / 攻击），没有拒识头，已知样本永远被强制判类，
"拒绝率"恒为 0，OSCR 不可识别。把"未知被收为正常的比例"当横轴、把"已知闭集正确率"
当纵轴画出的曲线不是 OSCR，若冠以 AUOSCR 之名即为指标误用（与 F4 同类错误）。
故本模块只输出该曲线供作图参考（`curve_kind` 显式标注），**不输出 AUOSCR 标量**。

口径声明
--------
- "未知" = `label==1 & y_type==unknown_idx`；本文只有 CTT masquerade 一个干净家族，
  因此全部数字标注 **single-family case study**，不得表述为开集泛化能力。
- 概率分数越高越"像攻击"，unknown 视为正类。

作者：小巴（WorkBuddy）    日期：2026-10-30（审计整改 F4 / D6）
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def _auroc(scores: np.ndarray, pos: np.ndarray) -> float:
    """秩和法 AUROC（并列取平均秩），O(n log n)。"""
    n_pos = int(pos.sum())
    n_neg = int(pos.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    _, inv, counts = np.unique(s_sorted, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg_rank = cum - (counts - 1) / 2.0
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = avg_rank[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(scores: np.ndarray, pos: np.ndarray) -> float:
    """非插值 AP（与 sklearn average_precision 同定义）。"""
    n_pos = int(pos.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hit = pos[order]
    tp = np.cumsum(hit)
    precision = tp / np.arange(1, hit.size + 1)
    return float((precision * hit).sum() / n_pos)


def _tpr_at_fpr(scores_u: np.ndarray, scores_n: np.ndarray, target_fpr: float) -> dict:
    """正常窗口误报率目标 ≤ target 时，未知攻击的召回。

    取经验分位点 τ = s_n[n−k]（k = ⌈target·n⌉），实际 FPR 由 `fpr_actual` 如实回报：
    分数大量并列时（如 ROAD 负样本 99% 恰为 0）实际 FPR 会超过目标，这必须暴露而
    不是用插值掩盖。
    """
    if scores_n.size == 0 or scores_u.size == 0:
        return {"tau": float("nan"), "tpr": float("nan"), "fpr_actual": float("nan")}
    s_n = np.sort(scores_n)
    k = int(np.ceil(target_fpr * s_n.size))
    idx = max(0, s_n.size - k)
    tau = float(s_n[idx])
    return {"tau": tau,
            "tpr": float((scores_u >= tau).mean()),
            "fpr_actual": float((s_n >= tau).mean())}


def openset_metrics(y_true: np.ndarray, y_type: np.ndarray, prob: np.ndarray,
                    threshold: float, unknown_idx: int, *,
                    n_grid: int = 256) -> dict:
    """返回 unknown 检测的全套度量 + 供作图的检测曲线。"""
    y = np.asarray(y_true).astype(np.int64).ravel()
    yt = np.asarray(y_type).astype(np.int64).ravel()
    p = np.asarray(prob, dtype=np.float64).ravel()
    if not (y.size == yt.size == p.size):
        raise ValueError("y_true / y_type / prob 长度不一致")

    unknown = (y == 1) & (yt == int(unknown_idx))
    normal = (y == 0)
    if not unknown.any():
        return {"n_unknown": 0, "note": "该语料无零样本家族，开集度量不适用"}

    s_u, s_n = p[unknown], p[normal]

    out: dict = {
        "n_unknown": int(unknown.sum()), "n_normal": int(normal.sum()),
        "n_known_all": int((~unknown).sum()),
        "n_families": 1, "scope": "single-family case study",
        "prevalence_unknown": float(unknown.mean()),
        # 主口径：未知 vs 正常（新颖性检测）
        "auroc_unknown_vs_normal": _auroc(np.concatenate([s_u, s_n]),
                                          np.concatenate([np.ones(s_u.size, bool),
                                                          np.zeros(s_n.size, bool)])),
        "aupr_unknown_vs_normal": _average_precision(
            np.concatenate([s_u, s_n]),
            np.concatenate([np.ones(s_u.size, bool), np.zeros(s_n.size, bool)])),
        # 副口径：标准 OOD（负类含已知攻击）—— 会因"已知攻击本就高分"而被结构性压低
        "auroc_unknown_vs_known": _auroc(p, unknown),
        "threshold": float(threshold),
        "tpr_unknown_at_tau": float((s_u >= float(threshold)).mean()),
        "fpr_normal_at_tau": float((s_n >= float(threshold)).mean()),
    }

    for target in (0.001, 0.01):
        r = _tpr_at_fpr(s_u, s_n, target)
        key = f"{target * 100:g}".replace(".", "p")
        out[f"tpr_at_fpr_normal_{key}"] = r["tpr"]
        out[f"tau_at_fpr_normal_{key}"] = r["tau"]
        out[f"fpr_normal_actual_{key}"] = r["fpr_actual"]

    # 作图用曲线：x = 正常误报率，y = 未知召回（"检测曲线"，非 OSCR）
    taus = np.quantile(np.concatenate([s_u, s_n]), np.linspace(1.0, 0.0, int(n_grid)))
    taus = np.unique(taus)
    s_n_sorted, s_u_sorted = np.sort(s_n), np.sort(s_u)
    fpr = (s_n_sorted.size - np.searchsorted(s_n_sorted, taus, side="left")) / s_n_sorted.size
    tpr = (s_u_sorted.size - np.searchsorted(s_u_sorted, taus, side="left")) / s_u_sorted.size
    order = np.argsort(fpr)
    out["curve"] = {"fpr_normal": fpr[order].tolist(), "tpr_unknown": tpr[order].tolist(),
                    "curve_kind": "detection_curve (not OSCR; binary detector has no reject head)"}
    return out


def regime_sensitivity(mcc_full: float, mcc_known: float) -> dict:
    """把旧的"开集退化"改名并钉死解释边界（审计 F4 的最小修复）。

    保留原数字以便与历史产物对齐，但显式声明它是**子集难度**而非开集能力。
    """
    return {
        "metric": "regime_shift_sensitivity",
        "delta_mcc_full_minus_known": float(mcc_full - mcc_known),
        "interpretation": ("known 口径剔除了零样本家族，差值 = 最难子集占比的算术后果；"
                           "随模型变强而减小，不得解释为开集能力，不得跨模型比较"),
        "legacy_name": "open-set degradation（已废弃，仅作历史对齐）",
    }


def summarize_table(rows: Mapping[str, dict]) -> list[dict]:
    """把多模型的结果摊平成可写 CSV 的行。"""
    out: list[dict] = []
    for model, r in rows.items():
        if not r or r.get("n_unknown", 0) == 0:
            out.append({"model": model, "note": r.get("note", "")})
            continue
        out.append({
            "model": model,
            "n_unknown": r["n_unknown"], "n_normal": r["n_normal"],
            "auroc_unknown_vs_normal": r["auroc_unknown_vs_normal"],
            "aupr_unknown_vs_normal": r["aupr_unknown_vs_normal"],
            "auroc_unknown_vs_known": r["auroc_unknown_vs_known"],
            "tpr_at_fpr_normal_0p1": r["tpr_at_fpr_normal_0p1"],
            "tpr_at_fpr_normal_1": r["tpr_at_fpr_normal_1"],
            "tpr_unknown_at_tau": r["tpr_unknown_at_tau"],
            "fpr_normal_at_tau": r["fpr_normal_at_tau"],
            "threshold": r["threshold"],
        })
    return out
