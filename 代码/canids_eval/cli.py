#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
canids_eval.cli —— 三个子命令的入口
================================================================================
用法（在仓库根目录执行）
------------------------
  # 1) 单 run 口径报告（不重标定 τ）
  python -m canids_eval report --pred outputs/training/main/ctt15/fdtcn/full/seed42/predictions_test.npz

  # 2) 两 run 配对检验（capture 层；给 --groups 再加组层）
  python -m canids_eval compare \
      --a outputs/training/main/ctt15/fdtcn/full/seed42/predictions_test.npz \
      --b outputs/training/main/ctt15/lstm/full/seed42/predictions_test.npz \
      --groups groups_ctt.json

  # 3) 密度分层召回
  python -m canids_eval density --pred ... --counts outputs/audit/density_counts_ctt_seed42.npy

`groups` 文件格式
------------------
  ① JSON：{"0": "set_01", "1": "set_01", ...}（键为 cap 下标，值为组名）；或
  ② .npy：长度 = 簇数的一维字符串/整数数组（下标即 cap）。

输出：默认 JSON 到 stdout；`--out path.json` 落盘。

作者：小巴（WorkBuddy）    日期：2026-10-30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 同仓库的 training 包（与本包同级目录）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from training import metrics as M          # noqa: E402
    from training import nested as N           # noqa: E402
    from training.stats import holm, mcnemar_cluster   # noqa: E402
except ImportError as e:                        # pragma: no cover
    raise SystemExit(
        "无法导入 training 包：请把 canids_eval 与 代码/training 放在同一目录下再运行。\n"
        f"原始错误：{e}"
    )


# --------------------------------------------------------------------------- #
def _load_pred(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"找不到预测文件：{p}")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _load_groups(path: str | Path) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return np.asarray(np.load(p, allow_pickle=False))
    obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        keys = sorted(obj, key=lambda k: int(k))
        return np.asarray([str(obj[k]) for k in keys])
    return np.asarray([str(x) for x in obj])


def _pred(z: dict) -> np.ndarray:
    return (np.asarray(z["prob"]) >= float(z["threshold"])).astype(np.int64)


# --------------------------------------------------------------------------- #
def cmd_report(args) -> dict:
    z = _load_pred(args.pred)
    y = np.asarray(z["label"]).astype(np.int64)
    p = np.asarray(z["prob"], dtype=np.float64)
    tau = float(z["threshold"])
    m = M.binary_metrics(y, p, tau)
    out = {k: m[k] for k in ("n", "n_pos", "n_neg", "prevalence", "threshold",
                             "mcc", "auc_pr", "auc_pr_norm", "auc_roc",
                             "recall", "fpr", "precision")}
    out["note"] = ("τ 取自该 run 的 val 标定（不改动）；本命令不在 test 上重定 τ，"
                   "因此表格里的 FPR 是**该 τ 的实际 FPR**，不是名义目标。")
    return out


def cmd_compare(args) -> dict:
    a, b = _load_pred(args.a), _load_pred(args.b)
    if not np.array_equal(a["label"], b["label"]) or not np.array_equal(a["cap"], b["cap"]):
        raise SystemExit("两 run 的 label/cap 序列不一致，禁止配对（数据版本或切分不同）")
    y = np.asarray(a["label"]).astype(np.int64)
    pa, pb = _pred(a), _pred(b)

    out: dict = {"n_windows": int(y.size)}
    cl = mcnemar_cluster(y, pa, pb, np.asarray(a["cap"]))
    out["capture_level"] = {
        "n_clusters": cl["n_clusters"], "p_value": cl["p_value"],
        "method": cl["method"], "p_min": cl["p_min"], "p_at_floor": cl["p_at_floor"],
        "b": cl["b"], "c": cl["c"],
    }
    if args.groups:
        g = _load_groups(args.groups)
        res = N.mcnemar_nested(y, pa, pb, np.asarray(a["cap"]), g,
                               n_boot=int(args.n_boot), n_perm=int(args.n_perm),
                               seed=int(args.seed))
        out["group_level"] = {
            "n_groups": res["n_groups"], "groups": res["groups"],
            "p_value": res["p_value_group"], "method": res["method_group"],
            "p_min": res["p_min_group"], "p_at_floor": res["p_at_floor_group"],
            "delta_mcc": res["delta_mcc"],
            "delta_ci": [res["delta_ci_lo"], res["delta_ci_hi"]],
            "cluster_net_advantage": res["cluster_net_advantage"],
        }
        out["note"] = ("组层 p 是最保守口径：G 个组决定下限 2/2^G。若 p_at_floor=true，"
                       "只能写「该粒度下不可判定」，并以各组净优势的符号一致性作证据。")
    else:
        out["note"] = ("未给 --groups，仅报 capture 层。若同一语料含多次录制（如 can-train-and-test "
                       "的 set_01–04），capture 仍不独立，须补组层。")
    return out


def cmd_density(args) -> dict:
    z = _load_pred(args.pred)
    y = np.asarray(z["label"]).astype(np.int64)
    p = np.asarray(z["prob"], dtype=np.float64)
    tau = float(z["threshold"])
    counts = np.asarray(np.load(args.counts, allow_pickle=False)).astype(np.int64)
    if counts.size != y.size:
        raise SystemExit(f"counts 长度 {counts.size} 与窗口数 {y.size} 不一致")

    bins = []
    for spec in args.bins.split(","):
        a, b = spec.split("-")
        bins.append((int(a), int(b)))
    pred = (p >= tau).astype(np.int64)
    rows = []
    for lo, hi in bins:
        m = (y == 1) & (counts >= lo) & (counts <= hi)
        n = int(m.sum())
        rows.append({"bin": f"{lo}-{hi}", "n_pos": n,
                     "share": float(n / max(1, int((y == 1).sum()))),
                     "recall": float(pred[m].mean()) if n else float("nan")})
    return {"threshold": tau, "n_pos": int((y == 1).sum()), "bins": rows,
            "note": "召回按窗口 any_frame 标签、该 run 自己的 τ 计算；密度 = 窗内攻击帧数。"}


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="canids_eval", description="CAN IDS 最小评测包")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("report", help="单 run 口径报告")
    p1.add_argument("--pred", required=True)
    p1.add_argument("--out", default=None)

    p2 = sub.add_parser("compare", help="两 run 配对检验")
    p2.add_argument("--a", required=True)
    p2.add_argument("--b", required=True)
    p2.add_argument("--groups", default=None, help="cap→组的映射（JSON 或 .npy）")
    p2.add_argument("--n-boot", default=1000, type=int)
    p2.add_argument("--n-perm", default=20000, type=int)
    p2.add_argument("--seed", default=0, type=int)
    p2.add_argument("--out", default=None)

    p3 = sub.add_parser("density", help="密度分层召回")
    p3.add_argument("--pred", required=True)
    p3.add_argument("--counts", required=True)
    p3.add_argument("--bins", default="1-2,3-8,9-24,25-48,49-64")
    p3.add_argument("--out", default=None)

    args = ap.parse_args(argv)
    res = {"report": cmd_report, "compare": cmd_compare, "density": cmd_density}[args.cmd](args)
    text = json.dumps(res, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"written: {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
