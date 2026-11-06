#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calib_diag.py —— 标定诊断：可靠性、ECE、温度标定，以及"不可达操作点"的归因
================================================================================
输入：`calib_logits.py` 落盘的 logit/prob/label（同一 checkpoint 的前向结果）。
输出：`outputs/audit/calibration/calib_summary.csv` + `calib_bins.csv`

三个问题
--------
1. 分数是否标定？—— 15 箱可靠性 + ECE（test）。
2. 顶部并列是真并列还是 float32 sigmoid 饱和？—— 统计 p==1.0 的负样本在 **logit 空间**
   有多少个互异取值。
3. 温度标定能否让 1% 误报率的操作点重新可达？—— 在 val 上拟合 T，重算 test 的
   ECE 与 1% FPR 处的（oracle）MCC。

注意：温度是单调变换，**不会改变排序**；若并列来自 sigmoid 的浮点饱和，换到 logit 域
即可分辨；若 logit 本身相同，则任何单调变换都救不回来。本脚本正是用来区分这两种情形。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/audit/calibration"


def ece(prob: np.ndarray, y: np.ndarray, bins: int = 15) -> tuple[float, list[tuple[float, float, int]]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    total, pts = 0.0, []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        conf = float(prob[m].mean())
        acc = float(y[m].mean())
        total += n / len(prob) * abs(acc - conf)
        pts.append((conf, acc, n))
    return total, pts


def nll(logit: np.ndarray, y: np.ndarray, T: float = 1.0) -> float:
    z = logit / T
    # log(1+e^z) 的数值稳定写法
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def fit_temperature(logit: np.ndarray, y: np.ndarray) -> float:
    grid = np.geomspace(0.2, 60.0, 400)
    losses = [nll(logit, y, T) for T in grid]
    return float(grid[int(np.argmin(losses))])


def tau_for_fpr(neg: np.ndarray, target: float) -> float | None:
    """满足 mean(neg >= τ) <= target 的最小 τ（与训练侧 metrics._tau_for_fpr 同规则）。"""
    neg = np.sort(np.asarray(neg, dtype=np.float64))
    uniq = np.unique(neg)
    frac_ge = 1.0 - np.searchsorted(neg, uniq, side="left") / neg.size
    ok = np.flatnonzero(frac_ge <= target)
    return float(uniq[ok[0]]) if ok.size else None


def mcc_at(tp: int, fp: int, fn: int, tn: int) -> float:
    d = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return 0.0 if d == 0 else (tp * tn - fp * fn) / d


def analyse(path: Path, label: str, rows: list[dict], bins_rows: list[dict]) -> None:
    z = np.load(path)
    logit = z["logit"].astype(np.float64)
    prob = z["prob"].astype(np.float64)
    y = z["label"].astype(np.int64)
    neg_p, pos_p = prob[y == 0], prob[y == 1]

    e_raw, pts = ece(prob, y)
    t = 1.0
    zt_path = path.with_name(path.name.replace("_test_logits.npz", "_val_logits.npz"))
    if label.endswith("test") and zt_path.exists():
        zv = np.load(zt_path)
        t = fit_temperature(zv["logit"].astype(np.float64), zv["label"].astype(np.int64))
    p_t = 1.0 / (1.0 + np.exp(-logit / t))
    e_t, _ = ece(p_t, y)

    tie_mass = float(np.mean(neg_p >= 1.0))
    tied = logit[(y == 0) & (prob >= 1.0)]
    n_distinct = int(np.unique(tied).size) if tied.size else 0

    # 1% FPR 操作点：float32 prob 域 vs logit 域
    tau_p = tau_for_fpr(neg_p, 0.01)
    tau_z = tau_for_fpr(logit[y == 0], 0.01)
    def at(tau, use_logit):
        pred = (logit >= tau).astype(np.int64) if use_logit else (prob >= tau).astype(np.int64)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        return mcc_at(tp, fp, fn, tn), fp / max(fp + tn, 1)
    mcc_p, fpr_p = at(tau_p, False) if tau_p is not None else (float("nan"), float("nan"))
    mcc_z, fpr_z = at(tau_z, True) if tau_z is not None else (float("nan"), float("nan"))

    rows.append({
        "split": label, "n": int(len(y)), "n_pos": int(y.sum()),
        "ece": round(e_raw, 4), "ece_temperature": round(e_t, 4), "temperature": round(t, 3),
        "neg_tie_at_1": round(tie_mass, 5), "tied_neg_distinct_logits": n_distinct,
        "neg_max_prob": round(float(neg_p.max()), 6), "neg_max_logit": round(float(logit[y == 0].max()), 3),
        "tau_prob": "infeasible" if tau_p is None else round(tau_p, 6),
        "mcc_at_1pct_prob": round(mcc_p, 4) if tau_p is not None else "infeasible",
        "fpr_at_1pct_prob": round(fpr_p, 5) if tau_p is not None else "infeasible",
        "tau_logit": "infeasible" if tau_z is None else round(tau_z, 4),
        "mcc_at_1pct_logit": round(mcc_z, 4) if tau_z is not None else "infeasible",
        "fpr_at_1pct_logit": round(fpr_z, 5) if tau_z is not None else "infeasible",
    })
    for conf, acc, n in pts:
        bins_rows.append({"split": label, "bin_conf": round(conf, 4), "bin_acc": round(acc, 4), "n": n})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*_logits.npz")
    a = ap.parse_args()
    rows: list[dict] = []
    bins_rows: list[dict] = []
    for p in sorted(OUT.glob(a.glob)):
        label = p.name.replace("_logits.npz", "")
        analyse(p, label, rows, bins_rows)
        print(f"  {label}: ECE={rows[-1]['ece']:.4f} tie={rows[-1]['neg_tie_at_1']:.3f} "
              f"distinct={rows[-1]['tied_neg_distinct_logits']}")
    if not rows:
        print("没有输入文件；先跑 calib_logits.py"); return
    with open(OUT / "calib_summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(OUT / "calib_bins.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bins_rows[0])); w.writeheader(); w.writerows(bins_rows)
    print("wrote", OUT / "calib_summary.csv")


if __name__ == "__main__":
    main()
