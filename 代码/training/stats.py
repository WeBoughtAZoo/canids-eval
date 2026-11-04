# -*- coding: utf-8 -*-
"""
training.stats —— 统计检验（McNemar / Wilcoxon / Holm）
================================================================================
§5 统计口径：
  - **McNemar**：同一测试集上两模型二分类判决的配对差异（用同一 τ 各自标定后的判决）；
  - **McNemar-cluster**（2026-10-28 新增，论文应以此为准）：窗口共享重叠帧、且同一
    capture 内的窗口高度相关，窗口级 McNemar 把 n 当成独立样本数，p 值系统性偏小
    （实测同一对模型窗口级 p=1e-40 量级，簇级 p=0.03）。本函数把不一致对按 capture
    聚合成"每簇净优势"，做簇级符号翻转置换检验；
  - **Wilcoxon signed-rank**：跨 seed 的连续指标配对比较（5 seeds）；
  - **Holm**：多重比较校正，控制 family-wise error。

作者：小巴（WorkBuddy）    日期：2026-10-17（2026-10-28 增簇级 McNemar）
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
            *, correction: bool = True) -> dict:
    """McNemar 检验（H0：两模型错误率相同）。

    返回 b = A 错 B 对、c = A 对 B 错、统计量与 p 值。
    b+c < 25 时用精确二项检验，否则用带/不带连续性校正的 χ²(df=1)。
    """
    y = np.asarray(y_true).astype(np.int64).ravel()
    a = np.asarray(pred_a).astype(np.int64).ravel()
    bpred = np.asarray(pred_b).astype(np.int64).ravel()
    if not (y.size == a.size == bpred.size):
        raise ValueError("y_true / pred_a / pred_b 长度不一致")

    a_ok = a == y
    b_ok = bpred == y
    n01 = int((~a_ok & b_ok).sum())     # b
    n10 = int((a_ok & ~b_ok).sum())     # c
    n = n01 + n10

    if n == 0:
        return {"b": 0, "c": 0, "statistic": 0.0, "p_value": 1.0,
                "method": "degenerate", "n_discordant": 0}

    if n < 25:
        p = float(stats.binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue)
        return {"b": n01, "c": n10, "statistic": float(min(n01, n10)),
                "p_value": p, "method": "exact_binomial", "n_discordant": n}

    delta = abs(n01 - n10) - (1.0 if correction else 0.0)
    stat = float(delta ** 2 / n)
    p = float(stats.chi2.sf(stat, df=1))
    return {"b": n01, "c": n10, "statistic": stat, "p_value": p,
            "method": "chi2_corrected" if correction else "chi2",
            "n_discordant": n}


def mcnemar_cluster(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                    clusters: np.ndarray, *, n_perm: int = 20000,
                    exact_max_clusters: int = 16, seed: int = 0) -> dict:
    """簇级 McNemar：把窗口级不一致对按 capture 聚合，再做簇级符号翻转检验。

    为什么必须做
    ------------
    窗口级 McNemar 的 n 是**窗口数**，但窗口彼此不独立：
      · 相邻窗口共享 W−stride 帧（train stride 32 / test stride 64，重叠 0–32 帧）；
      · 同一 capture 内的窗口来自同一次录音、同一攻击脚本、几乎同一信道状态。
    把 n 当独立样本 → 不一致对计数被"复制"放大 → p 值可小几个数量级。
    论文若报窗口级 p，审稿人一问"窗口独立吗"就崩。

    做法
    ----
    1. 逐簇累计 n01_k（A 错 B 对）与 n10_k（A 对 B 错），d_k = n01_k − n10_k；
    2. H0：两模型无差异 ⇒ 每簇 d_k 关于 0 对称 ⇒ 符号 s_k ∈ {−1,+1} 可随机翻转；
    3. 统计量 T = |Σ_k s_k·d_k|，块内成对置换 ⇒ 精确枚举（K ≤ exact_max_clusters）
       或 Monte Carlo（固定 seed 可复现）；
    4. p = P(T_perm ≥ T_obs)。

    只按"簇内净优势"计数，等价于把每簇当作一个观测——这是保守但可辩护的口径。

    返回
    ----
    {"p_value", "method", "n_clusters", "n_clusters_discordant",
     "b", "c", "statistic", "p_value_window", "method_window",
     "n_discordant_window", "p_resolution", "p_min", "p_at_floor"}

    注意 `p_min`：符号翻转是双侧检验，K 个簇全部同向时也只能取到 2/2^K。
    因此 **K=3 时最小可达 p=0.25，K=5 时 0.0625**——HCRL 只有 3 个 capture，
    在簇级口径下无论模型多好都不可能报出 p<0.05。这不是模型不足，是语料设计的上限，
    论文必须写明（否则审稿人会要求"补显著性"，而这是补不出来的）。
    """
    y = np.asarray(y_true).astype(np.int64).ravel()
    a = np.asarray(pred_a).astype(np.int64).ravel()
    b = np.asarray(pred_b).astype(np.int64).ravel()
    cl = np.asarray(clusters).ravel()
    if not (y.size == a.size == b.size == cl.size):
        raise ValueError("y_true / pred_a / pred_b / clusters 长度不一致")

    win = mcnemar(y, a, b, correction=True)          # 保留作对照，标注 anti-conservative

    _, inv = np.unique(cl, return_inverse=True)
    k = int(inv.max()) + 1
    a_ok = (a == y).astype(np.int64)
    b_ok = (b == y).astype(np.int64)
    n01 = np.bincount(inv, weights=(1 - a_ok) * b_ok, minlength=k).astype(np.int64)
    n10 = np.bincount(inv, weights=a_ok * (1 - b_ok), minlength=k).astype(np.int64)
    d = n01 - n10

    base = {"b": int(n01.sum()), "c": int(n10.sum()),
            "n_clusters": k,
            "n_clusters_discordant": int((d != 0).sum()),
            "n_discordant_window": win["n_discordant"],
            "p_value_window": float(win["p_value"]),
            "method_window": win["method"]}

    if not np.any(d != 0):
        base.update(signflip_p(d))
        return base

    res = signflip_p(d, n_perm=n_perm, exact_max_clusters=exact_max_clusters, seed=seed)
    base.update(res)
    return base


def signflip_p(d: np.ndarray, *, n_perm: int = 20000,
               exact_max_clusters: int = 16, seed: int = 0) -> dict:
    """簇级（或组级）符号翻转检验的核心。

    输入 `d` 为每簇的净优势（b_k − c_k），返回 p 值、方法与结构性下限。
    ★ 2026-10-30 抽出为独立函数：`mcnemar_cluster` 与 `verify_artifacts.py`
      共用同一实现——验证脚本必须独立重算，但不能有第二套公式。
    """
    d = np.asarray(d, dtype=np.int64).ravel()
    k = int(d.size)
    if not np.any(d != 0):
        return {"p_value": 1.0, "method": "cluster_degenerate", "statistic": 0.0,
                "p_resolution": 1.0, "p_min": 1.0, "p_at_floor": True}

    t_obs = float(abs(d.sum()))
    if k <= exact_max_clusters:
        signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * k)).T.reshape(-1, k)
        stats_null = np.abs(signs @ d.astype(np.float64))
        p = float((stats_null >= t_obs - 1e-12).mean())
        method = "cluster_exact_signflip"
        resolution = 1.0 / float(signs.shape[0])
        p_min = 2.0 / float(signs.shape[0])
    else:
        # 分块 Monte Carlo：K 可达数百，一次性生成 n_perm×K 会吃掉数百 MB
        rng = np.random.default_rng(seed)
        d_f = d.astype(np.float64)
        hit = 0
        done = 0
        chunk = 4096
        while done < int(n_perm):
            m = min(chunk, int(n_perm) - done)
            signs = rng.integers(0, 2, size=(m, k)).astype(np.float64) * 2 - 1
            hit += int((np.abs(signs @ d_f) >= t_obs - 1e-12).sum())
            done += m
        p = float((hit + 1) / (int(n_perm) + 1))
        method = "cluster_monte_carlo_signflip"
        resolution = 1.0 / (int(n_perm) + 1)
        p_min = resolution

    return {"p_value": p, "method": method, "statistic": t_obs,
            "p_resolution": resolution, "p_min": p_min,
            "p_at_floor": bool(p <= p_min + 1e-15)}


def wilcoxon_paired(values_a: Sequence[float], values_b: Sequence[float]) -> dict:
    """Wilcoxon signed-rank（配对、双侧）。样本 < 6 时不做检验（功效不足）。"""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.size != b.size:
        raise ValueError("两组样本数必须一致")
    if a.size < 6 or np.allclose(a, b):
        return {"statistic": float("nan"), "p_value": float("nan"),
                "n": int(a.size), "note": "样本不足或零差异，未检验"}
    res = stats.wilcoxon(a, b, alternative="two-sided")
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue),
            "n": int(a.size), "note": ""}


def holm(pvalues: Iterable[float]) -> list[float]:
    """Holm–Bonferroni 校正（保持输入顺序）。"""
    p = np.asarray(list(pvalues), dtype=np.float64)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def aggregate_seeds(rows: Iterable[Mapping[str, float]],
                    metric_names: Sequence[str]) -> dict[str, dict[str, float]]:
    """把多个 seed 的指标聚合成 mean / std / median / min / max。"""
    rows = list(rows)
    out: dict[str, dict[str, float]] = {}
    for name in metric_names:
        vals = np.asarray([r.get(name, np.nan) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out[name] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "median": float(np.median(vals)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n_seeds": int(vals.size),
        }
    return out
