#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_adaptive.py —— 自适应阈值回放（在线分位数 vs 固定 τ）
================================================================================
用途：回答审稿人"你的固定 τ 在跨采集时 FPR 超标 1.4–3×，怎么部署？"

  python 代码/run_adaptive.py --run-dir outputs/training/main/ctt15/fdtcn/full/seed42
  python 代码/run_adaptive.py --batch-root outputs/training/main --target-rate 0.01

产物
----
· 单 run：`<run>/adaptive_test.csv`（逐块 τ / FPR / 召回）
· 批处理：`<batch-root>/adaptive/adaptive_summary.csv`（每 run 一行汇总）

口径（2026-10-28 修订）
--------------------
· 默认 `--mode capped`：缓冲只接收 `p <= cap` 的样本，`cap = val τ × --cap-factor`，
  目标是**误报率**（FPR）而不是告警率。
· `--mode all` 是告警率口径，`--mode negative` 有截断自反馈缺陷；
  两者都只作对照，**不得写入论文**（详见 training/adaptive.py 模块 docstring）。
· 若得不出合理 cap（例如快照里没有 threshold 字段），会退化为 all 并在日志里警告。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from training.adaptive import simulate, temporal_order  # noqa: E402
from training.utils import get_logger, setup_logging, write_csv  # noqa: E402

log = get_logger("adaptive")


def _one(run_dir: Path, *, target_rate: float, window: int, step: int,
         mode: str, cap_factor: float, cap: float | None,
         label_delay: int | None) -> dict | None:
    npz = run_dir / "predictions_test.npz"
    if not npz.exists():
        log.warning("跳过（缺少 predictions_test.npz）：%s", run_dir)
        return None
    z = np.load(npz)
    y, p = z["label"], z["prob"]
    # 基线快照（sklearn 系列）只存 cap、不存 pos，无法还原时间序：
    # 退回原数组顺序并在日志里标注，避免静默假设"原顺序就是时间序"。
    if "pos" in z.files and "cap" in z.files:
        order = temporal_order(z["cap"], z["pos"])
    else:
        order = None
        log.warning("快照缺少 cap/pos，按原数组顺序回放（非时间序）：%s", run_dir)
    fixed = float(z["threshold"]) if "threshold" in z.files else None
    if mode == "capped" and fixed is None and cap is None:
        log.warning("缺少 threshold 锚点，cap 退化为 1.0（等价 all）：%s", run_dir)
    res = simulate(y, p, target_rate=target_rate, window=window, step=step,
                   order=order, fixed_threshold=fixed, mode=mode,
                   cap=cap, cap_factor=cap_factor, label_delay=label_delay)
    rows = []
    for i, b in enumerate(res["blocks"]):
        rows.append({"block": i, **b})
    write_csv(run_dir / "adaptive_test.csv", rows)

    o, f = res["overall"], res.get("fixed", {})
    row = {
        "run_dir": str(run_dir),
        "n_windows": int(o.get("n", 0)),
        "target_rate": target_rate,
        "adaptive_fpr": o.get("fpr"),
        "adaptive_recall": o.get("recall"),
        "fixed_fpr": f.get("fpr"),
        "fixed_recall": f.get("recall"),
        "fixed_tau": fixed,
        "window": window, "step": step, "mode": mode,
        "ordered": bool(order is not None),
        "cap": res["params"]["cap"], "cap_factor": cap_factor,
    }
    log.info("[%s] 自适应 FPR=%.4f 召回=%.4f | 固定 τ FPR=%.4f 召回=%.4f",
             run_dir.name, row["adaptive_fpr"] or float("nan"),
             row["adaptive_recall"] or float("nan"),
             row["fixed_fpr"] or float("nan"), row["fixed_recall"] or float("nan"))
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="自适应阈值回放")
    ap.add_argument("--run-dir", action="append", default=None,
                    help="可重复；单个 run 目录（含 predictions_test.npz）")
    ap.add_argument("--batch-root", default=None, help="批量根目录（递归找 predictions_test.npz）")
    ap.add_argument("--target-rate", type=float, default=0.01, help="期望 FPR（默认 1%%）")
    ap.add_argument("--window", type=int, default=16384,
                    help="环形缓冲长度（窗口数）；labeled 模式实测 4k→16k→64k FPR 单调改善")
    ap.add_argument("--step", type=int, default=512, help="每多少窗口更新一次 τ")
    ap.add_argument("--mode", choices=["labeled", "capped", "all", "negative"],
                    default="labeled",
                    help="labeled=延迟标签定期重标定（默认，推荐）；capped=按 val τ×cap-factor "
                         "设接纳上限（可能退化）；all=告警率口径；negative=有偏，仅对照")
    ap.add_argument("--label-delay", type=int, default=None,
                    help="labeled 模式的标签延迟（窗口数），默认 = step")
    ap.add_argument("--cap-factor", type=float, default=3.0,
                    help="cap = val τ × 该系数（仅 capped 模式）")
    ap.add_argument("--cap", type=float, default=None, help="直接指定接纳上限，覆盖 --cap-factor")
    ap.add_argument("--out", default=None, help="汇总 CSV 路径")
    args = ap.parse_args(argv)

    setup_logging("INFO", Path(args.out).parent / "adaptive.log" if args.out else None)
    dirs: list[Path] = []
    if args.run_dir:
        dirs += [Path(d) for d in args.run_dir]
    if args.batch_root:
        dirs += sorted({p.parent for p in Path(args.batch_root).rglob("predictions_test.npz")})
    if not dirs:
        ap.error("至少给一个 --run-dir 或 --batch-root")

    rows = []
    for d in dirs:
        r = _one(d, target_rate=args.target_rate, window=args.window,
                 step=args.step, mode=args.mode,
                 cap_factor=args.cap_factor, cap=args.cap,
                 label_delay=args.label_delay)
        if r:
            rows.append(r)
    if not rows:
        log.error("没有任何可处理的 run")
        return 1

    if len(rows) > 1:
        out = Path(args.out) if args.out else Path(args.batch_root) / "adaptive" / "adaptive_summary.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_csv(out, rows)
        log.info("汇总写入 %s（%d 行）", out, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
