#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_backfill_density.py —— 为既有 run 回填 per_density_test.csv（不回训、不重推）
================================================================================
为什么需要
----------
攻击帧密度分层（`training/density.py`）是 2026-10-28 新加的产物，只在**新评测**里
随 `evaluate()` 一起落盘。2026-10-27 及更早的 113 个 run 的快照里没有
`per_density_test.csv`，因此 `run_tables.py` 的 t11_density 一直是空表。

重跑 `run_eval.py` 能补，但要重做全部推理（数小时）。其实密度与模型无关：
同一数据集的 test 切分只有一份 `(cap, pos)`，所以**每个数据集算一次**，
再和每个 run 快照里的 `prob / label / threshold` 组合即可。

用法
----
  python 代码/run_backfill_density.py --batch-root outputs/training/main
  python 代码/run_backfill_density.py --batch-root outputs/training/main --datasets ctt15,road
  python 代码/run_backfill_density.py --batch-root outputs/training/main --force

产物
----
  <run>/per_density_test.csv        与 evaluate() 落盘格式完全一致
  <batch>/density_backfill.csv      本次回填清单（run / 窗口数 / 是否跳过 / 原因）

注意
----
· 只写 CSV，不改 `predictions_test.npz`（避免破坏既有快照的可复现性）；
· 窗口数与本数据集当前切分不一致的 run 会被**跳过并记录原因**（例如用过 data.limits）；
· `recall_at_fpr` 列与 `recall` 取同一个 τ（快照里只存了 val 标定的那一个），
  这一点与 evaluate() 完全一致，因为 evaluate() 的 tau_at_fpr 也是同一个 τ。

作者：小巴（WorkBuddy）    日期：2026-10-28
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from training import config as C          # noqa: E402
from training import density as D         # noqa: E402
from training import pipeline as P        # noqa: E402
from training.utils import get_logger, setup_logging, write_csv  # noqa: E402

log = get_logger("backfill")


def _counts_for_dataset(cfg: dict, dataset: str, seed: int) -> tuple[np.ndarray, dict]:
    """构建该数据集的 test 切分，返回逐窗口攻击帧数与 (cap, pos) 索引表。"""
    model = str(cfg["model"]["name"])
    data = P.prepare_data(cfg, dataset, seed=seed)
    ds = data["datasets"]["test"]
    # WindowDataset 把扁平索引放在 ds.index：(file_idx, start) = (capture, 窗口起点)
    idx = ds.index
    counts = D.window_attack_counts(ds, idx.file_idx, idx.start)
    key = (idx.file_idx.astype(np.int64) * 1_000_000 + idx.start.astype(np.int64))
    return counts, {"key": key, "n": counts.size, "model": model}


def _align(npz_cap: np.ndarray, npz_pos: np.ndarray, ref: dict) -> np.ndarray | None:
    """把 run 快照的 (cap, pos) 对齐到参考切分；顺序或规模不同则返回 None。"""
    if npz_cap.size == ref["n"]:
        same = (npz_cap.astype(np.int64) * 1_000_000 + npz_pos.astype(np.int64)) == ref["key"]
        if bool(np.all(same)):
            return np.arange(ref["n"])
    lut = {int(k): i for i, k in enumerate(ref["key"])}
    try:
        return np.array([lut[int(c) * 1_000_000 + int(p)]
                         for c, p in zip(npz_cap, npz_pos)], dtype=np.int64)
    except KeyError:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="回填密度分层表（不回训）")
    ap.add_argument("--config", default=str(Path(__file__).parent / "training" / "configs" / "train.yaml"))
    ap.add_argument("--set", dest="overrides", action="append", default=None)
    ap.add_argument("--batch-root", default=None, help="批根目录（默认取配置）")
    ap.add_argument("--datasets", default=None, help="逗号分隔；默认处理批内全部数据集")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 per_density_test.csv")
    args = ap.parse_args(argv)

    cfg = C.load(args.config, C.parse_overrides(args.overrides))
    setup_logging(str(cfg["runtime"]["log_level"]), None)
    root = Path(args.batch_root) if args.batch_root else Path(C.get_opt(cfg, "_resolved.output_root",
                                                                      "outputs/training")) / str(cfg["project"]["run_name"])
    if not root.exists():
        log.error("批根目录不存在：%s", root)
        return 1

    want = {s.strip() for s in args.datasets.split(",")} if args.datasets else None
    bins = cfg["eval"]["density_bins"]
    seed = int(cfg["data"].get("split_seed", 42)) if isinstance(cfg["data"], dict) else 42

    rows: list[dict] = []
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        dataset = ds_dir.name
        if dataset.startswith("_") or (want and dataset not in want):
            continue
        runs = sorted({p.parent for p in ds_dir.rglob("predictions_test.npz")})
        if not runs:
            continue

        cfg_ds = P.prepare_config(cfg, dataset=dataset, model="fdtcn", variant="full", seed=seed)
        try:
            counts, ref = _counts_for_dataset(cfg_ds, dataset, seed)
        except Exception as e:                                    # noqa: BLE001
            log.warning("[%s] 构建 test 切分失败，整批跳过：%s", dataset, e)
            for r in runs:
                rows.append({"dataset": dataset, "run": str(r), "n_windows": 0,
                             "status": "skip_dataset_error", "reason": str(e)[:120]})
            continue
        log.info("[%s] test 切分 %d 窗口，密度中位 %.0f", dataset, counts.size,
                 float(np.median(counts[counts > 0])) if (counts > 0).any() else 0.0)

        for run in runs:
            out_csv = run / str(cfg["artifacts"]["per_density_table"]).format(split="test")
            if out_csv.exists() and not args.force:
                rows.append({"dataset": dataset, "run": str(run), "n_windows": 0,
                             "status": "skip_exists", "reason": ""})
                continue
            with np.load(run / "predictions_test.npz", allow_pickle=False) as z:
                y, p = z["label"], z["prob"].astype(np.float64)
                cap, pos = z["cap"], z["pos"]
                tau = float(z["threshold"]) if "threshold" in z.files else 0.5

            idx = _align(cap, pos, ref)
            if idx is None:
                rows.append({"dataset": dataset, "run": str(run), "n_windows": int(y.size),
                             "status": "skip_misaligned",
                             "reason": f"与当前切分不匹配（{y.size} vs {ref['n']}）"})
                log.warning("跳过（切分不匹配）：%s", run)
                continue

            dens = D.density_table(y, p, counts[idx], tau, bins=bins, tau_at_fpr=tau)
            write_csv(out_csv, dens)
            rows.append({"dataset": dataset, "run": str(run), "n_windows": int(y.size),
                         "status": "written", "reason": ""})

    summary = root / "density_backfill.csv"
    if rows:
        write_csv(summary, rows)
        ok = sum(1 for r in rows if r["status"] == "written")
        log.info("回填完成：写入 %d，跳过 %d；清单 %s", ok, len(rows) - ok, summary)
    else:
        log.warning("没有找到任何 predictions_test.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
