# -*- coding: utf-8 -*-
"""
training.metrics —— 指标计算（含分层评测与可评估性分级）
================================================================================
口径（§6.3，r2 硬性要求）：
  1. 主指标 MCC / AUC-PR，且**一律窗口级**（模型输入就是窗口）；
  2. 每张表必须披露 prevalence，并给出归一化 AUC-PR：
         AUC-PR_norm = (AUC-PR − prevalence) / (1 − prevalence)
     其中 prevalence 即随机分类器的期望 AUC-PR；
  3. 依赖阈值的指标（MCC/F1/FPR/Recall）在 val 上定 τ、test 原样使用；
  4. 按攻击类型分层，并按 test 正窗口数分五级（充分/可用/不足/不可评估/zero-shot）。

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import math
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


# --------------------------------------------------------------------------- #
# 基础二分类指标
# --------------------------------------------------------------------------- #
def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def recall_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float,
                  *, tau: float | None = None) -> tuple[float, float]:
    """在给定 FPR 上界处取阈值并返回 (τ, Recall)。

    与 MCC/F1 不同，这是**操作点**指标（v4 §6.1 要求报 Recall@FPR∈{0.1%,1%}）。

    ★ 2026-10-27 修正（操作点泄漏）
    ------------------------------
    原实现**总是**用传入数组自身的负样本分位数反解 τ，因此当本函数被用于
    test 时，等于"用 test 的标签与分数分布挑操作点"——这是不可部署的 oracle 操作点。
    现增加 `tau` 参数：
      · `tau` 给定 → 直接用它在该 split 上评估（**部署口径**，τ 来自 val）；
      · `tau=None` → 退回原行为，**仅允许在标定 split（val）上使用**。
    """
    y = np.asarray(y_true).astype(np.int64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    pos = p[y == 1]
    if pos.size == 0:
        return float("nan"), float("nan")
    if tau is None:
        neg = p[y == 0]
        if neg.size == 0:
            return float("nan"), float("nan")
        tau = _tau_for_fpr(neg, float(target_fpr))
    tau = float(tau)
    return tau, float((pos >= tau).mean())


def _tau_for_fpr(neg: np.ndarray, target_fpr: float) -> float:
    """求满足 `mean(neg >= τ) ≤ target_fpr` 的**最宽松** τ。

    ★ 2026-10-27：原实现用 `np.quantile(neg, 1−α, method="higher")`。
    这在分数**连续**时等价，但对**离散/并列值多**的模型（CART/RF 只有几十个不同
    输出值）会退化：分位数正好落在某个高频取值上，导致实际 FPR 远超目标
    （实测 CART 在 HCRL 上名义 1% 时实际 99.9%）。
    改为在去重后的取值上直接搜索，天然处理并列；连续分数下结果与分位数法逐位一致。
    """
    neg = np.asarray(neg, dtype=np.float64).ravel()
    uniq = np.unique(neg)                      # 升序
    cnt_lt = np.searchsorted(np.sort(neg), uniq, side="left")
    frac_ge = 1.0 - cnt_lt / neg.size          # 取该值为 τ 时被误判的负样本比例
    ok = np.flatnonzero(frac_ge <= target_fpr)
    if ok.size:
        return float(uniq[ok[0]])              # 满足约束的最小 τ（召回最大）
    return float(np.nextafter(neg.max(), np.inf))


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
                   *, op_taus: Mapping[float, float] | None = None) -> dict[str, Any]:
    """给定阈值，返回全套二分类指标（含 prevalence 与归一化 AUC-PR）。

    `op_taus` : 目标 FPR → τ 的映射（**必须在 val 上标定**）。给定时按部署口径评估，
    并同时返回该 τ 在**本 split** 上的**实际 FPR**；未给定时退回"本数组自定 τ"
    （仅用于 val 标定）。
    """
    y = np.asarray(y_true).astype(np.int64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    if y.size != p.size:
        raise ValueError(f"y_true 与 y_prob 长度不一致：{y.size} vs {p.size}")

    pred = (p >= float(threshold)).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    n = tp + fp + tn + fn
    n_pos = tp + fn
    n_neg = tn + fp
    prevalence = _safe_div(n_pos, n)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, n_pos)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    specificity = _safe_div(tn, n_neg)
    fpr = _safe_div(fp, n_neg)
    # MCC 分母含四个计数项的乘积；测试集窗口数达百万级时乘积为超大 Python int，
    # 直接喂给 np.sqrt 会在 NumPy 2.x 触发 ufunc 派发异常（val 窗口少则侥幸通过）。
    # 先转 float 再用 math.sqrt，对任意量级都稳定且不受 int64 溢出影响。
    _mcc_term = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc_den = math.sqrt(float(_mcc_term)) if _mcc_term > 0 else 0.0
    mcc = _safe_div(tp * tn - fp * fn, mcc_den)

    out: dict[str, Any] = {
        "threshold": float(threshold),
        "n": n, "n_pos": n_pos, "n_neg": n_neg, "prevalence": prevalence,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "mcc": mcc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "accuracy": _safe_div(tp + tn, n),
        "auc_pr": float("nan"),
        "auc_pr_norm": float("nan"),
        "auc_roc": float("nan"),
        "recall_at_fpr_0p1": float("nan"),   # Recall @ FPR ≤ 0.1%
        "recall_at_fpr_1": float("nan"),     # Recall @ FPR ≤ 1%
        "tau_at_fpr_0p1": float("nan"),
        "tau_at_fpr_1": float("nan"),
        # ★ 实际 FPR：给定 τ（来自 val）在本 split 上真正达到的误报率。
        #   名义目标为 0.1% / 1%，实际值可高于目标（先验漂移），必须与本列并报。
        "fpr_at_fpr_0p1": float("nan"),
        "fpr_at_fpr_1": float("nan"),
    }

    if n_pos > 0:
        given = {float(k): float(v) for k, v in (op_taus or {}).items()
                 if v is not None and np.isfinite(float(v))}
        for tag, target in (("0p1", 0.001), ("1", 0.01)):
            tau = given.get(target)
            if tau is None:
                if n_neg == 0:
                    continue
                # 仅在"本 split 即标定 split"时才允许自定 τ
                tau, rec = recall_at_fpr(y, p, target)
            else:
                rec = float((p[y == 1] >= tau).mean())
            out[f"tau_at_fpr_{tag}"] = float(tau)
            out[f"recall_at_fpr_{tag}"] = float(rec)
            out[f"fpr_at_fpr_{tag}"] = (float((p[y == 0] >= tau).mean())
                                        if n_neg > 0 else float("nan"))

    # AUC 需要两个类都存在
    if n_pos > 0 and n_neg > 0:
        ap = float(average_precision_score(y, p))
        out["auc_pr"] = ap
        out["auc_pr_norm"] = _safe_div(ap - prevalence, 1.0 - prevalence) if prevalence < 1 else float("nan")
        out["auc_roc"] = float(roc_auc_score(y, p))
    return out


# --------------------------------------------------------------------------- #
# Bootstrap 置信区间
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    *,
    metric_names: Sequence[str],
    n_resamples: int,
    ci: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    """对给定指标做**分层 i.i.d.** bootstrap（保持正负样本数不变）。

    ★ 2026-10-28 审计结论：窗口级 i.i.d. 自助**低估不确定性约 10×**，
      因为同一 capture 内的窗口高度相关（CTT：224 簇，i.i.d. 给 ±0.6 pt，
      按 capture 聚类给 ±7 pt）。**论文与正文只能引用 `cluster_bootstrap_ci`**，
      本函数保留仅为兼容旧产物与单簇数据集（K=1）的退化场景。

    返回 {metric: {"lo":..,"hi":..,"mean":..,"std":..}}；退化样本跳过。
    """
    y = np.asarray(y_true).astype(np.int64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if pos.size == 0 or neg.size == 0:
        return {}

    rng = np.random.default_rng(int(seed))
    samples: dict[str, list[float]] = {m: [] for m in metric_names}
    for _ in range(int(n_resamples)):
        ip = rng.choice(pos, size=pos.size, replace=True)
        ineg = rng.choice(neg, size=neg.size, replace=True)
        idx = np.concatenate([ip, ineg])
        m = binary_metrics(y[idx], p[idx], threshold)
        for name in metric_names:
            val = m.get(name, float("nan"))
            if isinstance(val, (int, float)) and np.isfinite(val):
                samples[name].append(float(val))

    alpha = (1.0 - float(ci)) / 2.0
    out: dict[str, dict[str, float]] = {}
    for name, vals in samples.items():
        if not vals:
            continue
        arr = np.asarray(vals)
        out[name] = {
            "lo": float(np.quantile(arr, alpha)),
            "hi": float(np.quantile(arr, 1 - alpha)),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "n_resamples": int(arr.size),
        }
    return out


#: 可由 2×2 计数直接复算的指标（聚类自助走快路径，O(K) 每轮）
_COUNT_METRICS = ("mcc", "f1", "precision", "recall", "fpr",
                  "specificity", "balanced_accuracy", "accuracy")


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    *,
    clusters: np.ndarray,
    metric_names: Sequence[str],
    n_resamples: int,
    ci: float,
    seed: int,
    auc_max_windows: int = 200_000,
) -> dict[str, Any]:
    """**按簇（capture）重采样**的自助置信区间。

    为什么必须按簇
    --------------
    同一 capture 内的窗口共享车辆、路况、负载与攻击实例，远非独立样本。
    以窗口为单位重采样等价于把有效样本量放大到 10⁶ 量级，会把 CI 压窄约 10×
    （实测 CTT：i.i.d. ±0.6 pt vs 聚类 ±7 pt），使"显著"变成廉价结论。

    实现
    ----
    · 预聚合每簇的 tp/fp/tn/fn → 每轮只对 K 个簇求和，O(K)，与窗口数无关；
    · 目标指标由计数直接复算（mcc/f1/precision/recall/fpr/specificity/
      balanced_accuracy/accuracy）；
    · AUC-PR / AUC-ROC 需要逐样本排序，仅在窗口数 ≤ `auc_max_windows` 时计算，
      否则跳过并在返回值的 `_auc_skipped` 中标注（不静默给错数）。

    返回结构与 `bootstrap_ci` 一致，另加 `_n_clusters` / `_method` / `_auc_skipped`。
    """
    y = np.asarray(y_true).astype(np.int64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    c = np.asarray(clusters).ravel()
    if not (y.size == p.size == c.size):
        raise ValueError(f"y/p/clusters 长度不一致：{y.size}/{p.size}/{c.size}")

    uniq, inv = np.unique(c, return_inverse=True)
    K = int(uniq.size)
    if K < 2:
        return {"_n_clusters": K, "_method": "cluster", "_auc_skipped": True,
                "_note": "簇数 < 2，无法做聚类自助"}

    pred = p >= float(threshold)
    tp = np.bincount(inv[pred & (y == 1)], minlength=K).astype(np.float64)
    fp = np.bincount(inv[pred & (y == 0)], minlength=K).astype(np.float64)
    fn = np.bincount(inv[~pred & (y == 1)], minlength=K).astype(np.float64)
    tn = np.bincount(inv[~pred & (y == 0)], minlength=K).astype(np.float64)

    wanted = [m for m in metric_names if m in _COUNT_METRICS]
    want_auc = any(m in ("auc_pr", "auc_roc") for m in metric_names)
    auc_ok = bool(want_auc and y.size <= int(auc_max_windows))
    # 簇 → 窗口下标的边界（仅 AUC 路径需要）
    if auc_ok:
        order = np.argsort(inv, kind="stable")
        bounds = np.searchsorted(inv[order], np.arange(K + 1))

    rng = np.random.default_rng(int(seed))
    buckets: dict[str, list[float]] = {m: [] for m in wanted}
    if auc_ok:
        buckets["auc_pr"] = []
        buckets["auc_roc"] = []

    for _ in range(int(n_resamples)):
        draw = rng.integers(0, K, size=K)          # 有放回抽簇
        TP, FP, FN, TN = tp[draw].sum(), fp[draw].sum(), fn[draw].sum(), tn[draw].sum()
        vals = _count_metrics(TP, FP, FN, TN)
        for m in wanted:
            buckets[m].append(vals[m])
        if auc_ok:
            idx = np.concatenate([order[bounds[j]:bounds[j + 1]] for j in draw])
            buckets["auc_pr"].append(float(average_precision_score(y[idx], p[idx])))
            buckets["auc_roc"].append(float(roc_auc_score(y[idx], p[idx])))

    alpha = (1.0 - float(ci)) / 2.0
    out: dict[str, Any] = {}
    for name, vals in buckets.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out[name] = {
            "lo": float(np.quantile(arr, alpha)),
            "hi": float(np.quantile(arr, 1 - alpha)),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "n_resamples": int(arr.size),
        }
    out["_n_clusters"] = K
    out["_method"] = "cluster"
    out["_auc_skipped"] = bool(want_auc and not auc_ok)
    return out


def _count_metrics(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    """由 2×2 计数直接算指标（聚类自助的快路径）。"""
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    n = tp + fp + fn + tn
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if n else 0.0
    return {
        "mcc": ((tp * tn - fp * fn) / den) if den else 0.0,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "precision": precision,
        "recall": recall,
        "fpr": _safe_div(fp, fp + tn),
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "accuracy": _safe_div(tp + tn, n),
    }


# --------------------------------------------------------------------------- #
# 分层评测
# --------------------------------------------------------------------------- #
def grade_evaluability(n_pos: int, *, sufficient: int, usable: int,
                       insufficient: int, zero_shot: bool) -> str:
    """§6.3 r2 的可评估性分级。"""
    if zero_shot:
        return "zero-shot"
    if n_pos >= sufficient:
        return "sufficient"
    if n_pos >= usable:
        return "usable"
    if n_pos >= insufficient:
        return "insufficient"
    return "not_evaluable"


def per_type_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    attack_type: np.ndarray,
    *,
    is_masquerade: np.ndarray | None,
    zero_shot_types: Iterable[str],
    grade_cfg: Mapping[str, int],
    level: str = "type",
    group: np.ndarray | None = None,
    tau_at_fpr: float | None = None,
) -> list[dict[str, Any]]:
    """按攻击类型分层输出（每行一个类型 + 一行 overall），口径为 **one-vs-rest**。

    为什么必须是 OvR
    ----------------
    若把负集限定为"正常流量"（one-vs-normal），precision 会把其他攻击类型的
    漏判排除在分母之外，得到偏高的数值——与整体 precision 不可比，审稿人会直接质疑。
    因此这里对每个类型构造 y_bin = (type == t)，负集 = 其余全部窗口（正常 + 其他攻击），
    即标准多分类 OvR；`recall` 与 OvR 下等价，不受影响。

    列名带 `_ovr` 后缀，避免与整体指标混淆。
    """
    y = np.asarray(y_true).astype(np.int64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    at = np.asarray(group if group is not None else attack_type).astype(object).ravel()
    masq = np.asarray(is_masquerade).astype(bool).ravel() if is_masquerade is not None \
        else np.zeros_like(y, dtype=bool)
    zs = set(zero_shot_types)

    # ★ 2026-10-27：τ 必须由 val 传入（部署口径）；未传入时才退回本 split 自定，
    #   否则家族级的 recall_at_fpr_1 会与整体指标一起构成操作点泄漏。
    if tau_at_fpr is None:
        tau_op, _ = recall_at_fpr(y, p, 0.01)   # 全局 1% FPR 操作点，循环外只算一次
    else:
        tau_op = float(tau_at_fpr)
    rows: list[dict[str, Any]] = []
    types = sorted({str(t) for t in at[y == 1]})
    for t in types:
        # ★ 必须同时满足"属于该组"且"确实是攻击窗口"：
        #   HCRL/ROAD 的攻击采集里混有正常帧，只用 (at == t) 会把同采集的正常窗口
        #   也算成正样本（曾使 HCRL 三家族正样本数之和 = 全部窗口数 > 真实正样本数）。
        pos = (at == t) & (y == 1)
        n_pos = int(pos.sum())
        if n_pos == 0:
            continue
        m = binary_metrics(pos.astype(np.int8), p, threshold)
        grade = grade_evaluability(n_pos, zero_shot=(t in zs),
                                   sufficient=int(grade_cfg["sufficient"]),
                                   usable=int(grade_cfg["usable"]),
                                   insufficient=int(grade_cfg["insufficient"]))
        rows.append({
            "level": level,
            "attack_type": t,
            "is_masquerade": bool(masq[pos].any()),
            "n_positive_windows": n_pos,
            "n_negative_reference": int((~pos).sum()),
            "recall": m["recall"],
            # 不报 per-family precision / F1 / MCC：检测器是二分类的，OvR 负集里
            # 含其他家族的**真实攻击**，它们被正确报警却会记成 FP，使 precision 失真。
            # 家族级只报"检出率 + 排序质量（AUC-PR/OvR）"，与部署口径一致。
            "auc_pr_ovr": m["auc_pr"],
            "auc_pr_norm_ovr": m["auc_pr_norm"],
            "mean_score": float(p[pos].mean()),
            "median_score": float(np.median(p[pos])),
            "recall_at_fpr_1": float((p[pos] >= tau_op).mean()) if np.isfinite(tau_op) else float("nan"),
            "evaluability": grade,
            "zero_shot": bool(t in zs),
        })

    overall = binary_metrics(y, p, threshold)
    rows.append({
        "level": level,
        "attack_type": "overall",
        "is_masquerade": bool(masq[y == 1].any()) if (y == 1).any() else False,
        "n_positive_windows": overall["n_pos"],
        "n_negative_reference": overall["n_neg"],
        "recall": overall["recall"],
        "auc_pr_ovr": overall["auc_pr"],
        "auc_pr_norm_ovr": overall["auc_pr_norm"],
        "mean_score": float(p[y == 1].mean()) if (y == 1).any() else float("nan"),
        "median_score": float(np.median(p[y == 1])) if (y == 1).any() else float("nan"),
        "recall_at_fpr_1": overall["recall_at_fpr_1"],
        "evaluability": "overall",
        "zero_shot": False,
    })
    return rows


# --------------------------------------------------------------------------- #
# 多分类（攻击类型归因）指标
# --------------------------------------------------------------------------- #
def multiclass_metrics(y_type: np.ndarray, pred_cls: np.ndarray,
                       label_names: Sequence[str]) -> dict[str, Any]:
    """攻击类型头的整体指标（仅在有攻击窗口时计算）。"""
    y = np.asarray(y_type).astype(np.int64).ravel()
    pred = np.asarray(pred_cls).astype(np.int64).ravel()
    if y.size == 0 or pred.size == 0:
        return {}
    mask = y > 0
    if mask.sum() == 0:
        return {"n": 0}
    yt, yp = y[mask], pred[mask]
    classes = sorted(set(yt.tolist()) | set(yp.tolist()))
    per_class = []
    for c in classes:
        tp = int(((yp == c) & (yt == c)).sum())
        fp = int(((yp == c) & (yt != c)).sum())
        fn = int(((yp != c) & (yt == c)).sum())
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        per_class.append({
            "class_index": int(c),
            "class_name": label_names[c] if c < len(label_names) else str(c),
            "n": int((yt == c).sum()),
            "precision": prec, "recall": rec,
            "f1": _safe_div(2 * prec * rec, prec + rec),
        })
    return {
        "n": int(mask.sum()),
        "accuracy": _safe_div((yt == yp).sum(), yt.size),
        "macro_f1": float(np.mean([r["f1"] for r in per_class])) if per_class else 0.0,
        "per_class": per_class,
    }


# --------------------------------------------------------------------------- #
# 结果汇总
# --------------------------------------------------------------------------- #
#: 无论 `eval.metrics` 怎么配都**必须落盘**的字段。
#: 分三类：① 溯源（阈値/样本量/先验）；② 混淆计数（fig04 直接索引）；
#: ③ 操作点（T7c/T7f 的核心列，且 τ 是"部署口径"的唯一凭据）。
#: 少了它们，下游表格会出现空白列、fig04 会 KeyError、T7d 会整张表消失。
MANDATORY_KEYS: tuple[str, ...] = (
    "threshold", "n", "n_pos", "n_neg", "prevalence", "tp", "fp", "tn", "fn", "mcc",
    "recall_at_fpr_0p1", "recall_at_fpr_1", "tau_at_fpr_0p1", "tau_at_fpr_1",
    "fpr_at_fpr_0p1", "fpr_at_fpr_1",
)

#: 只影响"写不写"的可选指标（`eval.metrics` 未给出时的默认全集，与旧行为一致）
DEFAULT_OPTIONAL_KEYS: tuple[str, ...] = (
    "auc_pr", "auc_pr_norm", "auc_roc", "f1", "precision", "recall", "fpr",
    "specificity", "balanced_accuracy", "accuracy",
)


def summarize_split(metrics: Mapping[str, Any], *, split: str,
                    n_windows: int, bootstrap: Mapping[str, Any] | None,
                    report: Sequence[str] | None = None) -> dict[str, Any]:
    """把单个 split 的指标整理成可写 JSON 的结构。

    `report` 即 `eval.metrics` 白名单：**只裁剪可选指标**，
    `MANDATORY_KEYS` 恒写入。这样白名单可以真正生效，又不会打断下游表格与图
    （2026-10-27 修正：原实现用硬编码元组，配置对该 JSON 完全无效）。
    """
    keep: list[str] = list(MANDATORY_KEYS)
    for k in (report if report is not None else DEFAULT_OPTIONAL_KEYS):
        if k not in keep:
            keep.append(str(k))
    out = {"split": split, "n_windows_indexed": int(n_windows)}
    out.update({k: metrics.get(k) for k in keep})
    if bootstrap:
        # 只保留 {metric: {lo,hi,...}} 结构；`_n_used` 等标量另放
        out["bootstrap"] = {k: dict(v) for k, v in bootstrap.items()
                            if isinstance(v, Mapping)}
        for key in ("_n_used", "_n_clusters"):
            if key in bootstrap:
                out[f"bootstrap{key}"] = int(bootstrap[key])
        # ★ 2026-10-28：自助法口径必须随数字落盘，否则下游无法区分
        #   i.i.d.（±0.6 pt）与按 capture 聚类（±7 pt）两种误差棒。
        if "_method" in bootstrap:
            out["bootstrap_method"] = str(bootstrap["_method"])
        if "_auc_skipped" in bootstrap:
            out["bootstrap_auc_skipped"] = bool(bootstrap["_auc_skipped"])
    return out
