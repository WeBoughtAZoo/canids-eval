#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_baselines.py —— 表格特征外部基线（T9），不训练神经网络
================================================================================
目的
----
论文需要一个"非深度"的强基线，回答审稿人必问的一句：**深度时序模型是否必要？**
本脚本在**与 FD-TCN 完全相同的窗口划分**上，用 80 维窗口统计特征
（16 通道 × mean/std/last/min/max，与 KD-Logic 的 `kd.feature_stats` 同源）训练三种
经典分类器，并复用同一套阈值标定与指标口径。

模型
----
  gbdt  GradientBoosting（sklearn HistGradientBoosting，工业界常用的 GBDT 实现）
  rf    RandomForest（400 树）
  cart  决策树（depth=8, leaves≤32，与 KD-Logic 的容量上限对齐，作为"可审计规则"参照）

★ 说明：环境中**未安装 xgboost**，故以 sklearn 的 HistGradientBoosting 作为 GBDT 代表。
论文中须写作 "gradient-boosted decision trees (HistGBDT)"，**不得**写成 XGBoost。

产物（不写入 run 目录，避免污染 run 发现）
------------------------------------------
  <batch>/_baselines/<dataset>/<model>/seed<seed>/
      metrics_val.json, metrics_test.json, predictions_test.npz, meta.json

用法
----
  python 代码/run_baselines.py --datasets hcrl_canfd road --set project.run_name=main
  python 代码/run_baselines.py --datasets ctt15 --models gbdt --set project.run_name=main
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training import calibration as calib     # noqa: E402
from training import config as C              # noqa: E402
from training import figures as F             # noqa: E402
from training import metrics as M             # noqa: E402
from training import pipeline as P            # noqa: E402
from training.utils import get_logger, setup_logging, write_json  # noqa: E402

log = get_logger("baselines")

STATS = {
    "mean": lambda w: w.mean(axis=2),
    "std": lambda w: w.std(axis=2),
    "last": lambda w: w[:, :, -1],
    "min": lambda w: w.min(axis=2),
    "max": lambda w: w.max(axis=2),
}
CHUNK = 40_000


# --------------------------------------------------------------------------- #
# 快速特征抽取（与 kdlogic.extract_features 同定义，但按 capture 向量化）
# --------------------------------------------------------------------------- #
def extract(dataset, cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (X [N, C·S], y, y_type, cap)。

    复用 `dataset.index`（权威窗口表），用 sliding_window_view 一次性取整窗，
    避免 Python 逐窗循环（CTT 上有 4.7M 窗口，逐窗约需 20+ 分钟）。
    """
    stats = list(cfg["kd"]["feature_stats"])
    bad = [s for s in stats if s not in STATS]
    if bad:
        raise ValueError(f"kd.feature_stats 含未知统计量：{bad}")

    idx = dataset.index
    n = len(idx)
    fid = idx.file_idx
    starts = idx.start
    f = next((i for i, s in enumerate(dataset.sources) if s.v1 is not None), None)
    if f is None:
        raise ValueError("基线需要 V1 特征，但该数据集未加载 v1")
    c = int(dataset.sources[f].v1.shape[1])
    X = np.empty((n, c * len(stats)), dtype=np.float32)

    order = np.argsort(fid, kind="mergesort")
    bounds = np.searchsorted(fid[order], np.arange(len(dataset.sources) + 1))
    for i, src in enumerate(dataset.sources):
        if src.v1 is None:
            raise ValueError(f"capture {src.capture} 未加载 v1")
        rows = order[bounds[i]:bounds[i + 1]]
        if rows.size == 0:
            continue
        v = np.asarray(src.v1)
        win = np.lib.stride_tricks.sliding_window_view(v, dataset.window, axis=0)  # (N-W+1, C, W)
        pos = starts[rows]
        for lo in range(0, rows.size, CHUNK):
            sl = slice(lo, min(lo + CHUNK, rows.size))
            # ★ 必须先转 float32：v1 是 float16，64 个元素求和会在 float16 下溢出成 inf
            #   （kdlogic 走 DataLoader 时同样是先转 float32 再统计）
            blk = win[pos[sl]].astype(np.float32)                  # (b, C, W)
            parts = [STATS[s](blk).astype(np.float32) for s in stats]
            X[rows[sl]] = np.concatenate(parts, axis=1)
    return X, idx.labels.astype(np.int64), idx.type_idx.astype(np.int64), fid.astype(np.int32)


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
def make_model(name: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    if name == "gbdt":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
            early_stopping=False, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, n_jobs=-1,
            class_weight="balanced_subsample", random_state=seed)
    if name == "cart":
        return DecisionTreeClassifier(
            max_depth=8, max_leaf_nodes=32, class_weight="balanced", random_state=seed)
    raise KeyError(f"未知基线 {name}")


def _op_points(y: np.ndarray, p: np.ndarray) -> dict[float, float]:
    out: dict[float, float] = {}
    for t in (0.001, 0.01):
        tau, _ = M.recall_at_fpr(y, p, t)
        if np.isfinite(tau):
            out[t] = float(tau)
    return out


def run_one(cfg: Mapping[str, Any], dataset: str, model_name: str, seed: int) -> dict[str, Any]:
    t0 = time.time()
    prep = P.prepare_data(cfg, dataset, seed=seed)
    ds = prep["datasets"]
    label_map = prep["label_map"]
    unknown = next((v for k, v in label_map.items() if k.endswith("unknown_attack")), None)

    log.info("[%s/%s] 抽取特征…", dataset, model_name)
    Xtr, ytr, _, _ = extract(ds["train"], cfg)
    Xva, yva, tva, cva = extract(ds["val"], cfg)
    Xte, yte, tte, cte = extract(ds["test"], cfg)
    log.info("[%s/%s] 特征就绪：train %s / val %s / test %s，%d 维（%.0fs）",
             dataset, model_name, Xtr.shape, Xva.shape, Xte.shape, Xtr.shape[1], time.time() - t0)

    clf = make_model(model_name, seed)
    clf.fit(Xtr, ytr)
    del Xtr
    pva = clf.predict_proba(Xva)[:, 1].astype(np.float64)
    tau = float(calib.calibrate(yva, pva, cfg)["threshold"])
    op = _op_points(yva, pva)
    pte = clf.predict_proba(Xte)[:, 1].astype(np.float64)

    out_dir = F.batch_root(cfg) / "_baselines" / dataset / model_name / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, y, p, ty, cp, src in (("val", yva, pva, tva, cva, "calibrated_on_val"),
                                     ("test", yte, pte, tte, cte, "inherited_from_val")):
        main = M.binary_metrics(y, p, tau, op_taus=op)
        summary = M.summarize_split(main, split=split, n_windows=int(y.size),
                                    bootstrap=None, report=cfg["eval"]["metrics"])
        summary["op_source"] = src
        summary["operating_points"] = {
            "fpr_0p1": op.get(0.001), "fpr_1": op.get(0.01), "source": src}
        summary["baseline"] = model_name
        write_json(out_dir / f"metrics_{split}.json", summary)
        if split == "test":
            np.savez(out_dir / "predictions_test.npz",
                     prob=p.astype(np.float32), label=y.astype(np.int8),
                     y_type=ty.astype(np.int16), cap=cp.astype(np.int32),
                     threshold=np.float64(tau))
    meta = {"dataset": dataset, "model": model_name, "seed": seed,
            "n_features": int(Xva.shape[1]), "threshold": tau,
            "unknown_index": unknown, "fit_seconds": round(time.time() - t0, 1),
            "features": "16 channels x [mean,std,last,min,max] (same as KD-Logic)"}
    write_json(out_dir / "meta.json", meta)

    mt = json.loads((out_dir / "metrics_test.json").read_text(encoding="utf-8"))
    log.info("[%s/%s] MCC=%.4f AUC-PRn=%.4f R@1%%FPR=%.4f（实际FPR %.3f%%）τ=%.4f 用时 %.0fs",
             dataset, model_name, mt["mcc"], mt["auc_pr_norm"],
             mt["recall_at_fpr_1"], 100 * mt["fpr_at_fpr_1"], tau, time.time() - t0)
    return {"dataset": dataset, "model": model_name, "mcc": mt["mcc"],
            "auc_pr_norm": mt["auc_pr_norm"], "seconds": meta["fit_seconds"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="表格特征外部基线")
    ap.add_argument("--config", default=str(Path(__file__).parent / "training" / "configs" / "train.yaml"))
    ap.add_argument("--datasets", nargs="+", default=["hcrl_canfd", "road", "ctt15"])
    ap.add_argument("--models", nargs="+", default=["gbdt", "rf", "cart"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--set", dest="overrides", action="append", default=None)
    args = ap.parse_args(argv)

    cfg = C.load(args.config, C.parse_overrides(args.overrides))
    # 基线用 V1 特征，故显式指定只依赖 v1 的模型，避免 prepare_data 按 rgbnet 只加载 v2
    cfg["model"]["name"] = "fdtcn"
    setup_logging(str(cfg["runtime"]["log_level"]), None)
    fails = 0
    for ds in args.datasets:
        for mdl in args.models:
            for sd in args.seeds:
                try:
                    run_one(cfg, ds, mdl, sd)
                except Exception as e:  # noqa: BLE001
                    fails += 1
                    log.exception("[%s/%s] 失败：%s", ds, mdl, e)
    log.info("完成：失败 %d", fails)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
