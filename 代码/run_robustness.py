#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_robustness.py —— E1 输入扰动敏感性（eval-only，不重训）
================================================================================
目的
----
回答审稿人必问的一句："这些检测器只在干净输入上有效吗？"
在不重训、不做对抗优化的前提下，对**测试期的输入**施加四类具有物理含义的扰动，
观察 MCC / recall / FPR 的退化幅度。

扰动定义（作用于 16 维帧级编码 v1，逐帧施加后再按原口径重新开窗）
--------------------------------------------------------------------
  drop     随机丢帧（总线丢包 / 过载）：保留率 1−p
  dup      随机重传帧（重传风暴）：按概率 p 复制一帧
  jitter   时间抖动：dt_norm *= (1+N(0,σ))，dt_dev 同步加噪；σ = 5% / 20%
  idswap   ID 伪造：按概率 p 把该帧的 id_norm 换成同一 capture 内另一帧的 ID
  payload  载荷聚合量扰动：对 byte_mean/std/entropy/nonzero_ratio/payload_delta/u32
           乘性加噪 (1+N(0,σ))，比例型通道裁剪到 [0,1]

口径纪律（与论文一致）
----------------------
· **不重定阈值**：一律用该 run 在 val 上标定的 τ（`metrics_test.json` 的 threshold）；
· **不重训、不选模型**：只做前向；
· **同一样本集**：所有条件使用同一批 capture（等距抽样），因此条件之间可比；
· **只报窗口级指标**，并注明受影响窗口占比；
· **只评估仅用 v1 的模型**（FD-TCN / LSTM / Transformer）。FD-RGBNet 还吃字节图，
  v1 扰动与 v2 不一致会产生假象，故排除并在报告中说明。

用法
----
  python 代码/run_robustness.py --run-dir outputs/training/main/ctt15/fdtcn/full/seed42 \
      --max-captures 24 --out outputs/audit/robustness_ctt15_fdtcn.csv
  # 批量：--run-dir 可给多个，或直接用 --models 扫描 main 批次（见 --scan）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training import config as C                     # noqa: E402
from training import metrics as M                    # noqa: E402
from training.evaluator import load_run_for_eval     # noqa: E402
from training.utils import get_logger, setup_logging, write_csv  # noqa: E402

log = get_logger("robust")

CONDITIONS: dict[str, dict[str, Any]] = {
    "clean":    {"kind": "none"},
    "drop05":   {"kind": "drop", "p": 0.05},
    "drop10":   {"kind": "drop", "p": 0.10},
    "jitter05": {"kind": "jitter", "sigma": 0.05},
    "jitter20": {"kind": "jitter", "sigma": 0.20},
    "idswap05": {"kind": "idswap", "p": 0.05},
    "payload10": {"kind": "payload", "sigma": 0.10},
}


# --------------------------------------------------------------------------- #
def channel_index(cfg: dict, names: list[str]) -> dict[str, int]:
    ch = list(cfg["data"]["v1"]["channels"])
    return {n: ch.index(n) for n in names if n in ch}


def windows_in_range(a: int, b: int, W: int, stride: int) -> np.ndarray:
    """与 manifest.count_windows 一致：窗口只在 range 内滑动。"""
    if b - a < W:
        return np.zeros(0, dtype=np.int64)
    n = 1 + (b - a - W) // stride
    return a + np.arange(n, dtype=np.int64) * stride


def perturb(v1: np.ndarray, label: np.ndarray, cond: dict, idx: dict[str, int],
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """按条件扰动一段连续帧；返回（可能变长的）帧数组与标签。"""
    kind = cond["kind"]
    x = v1.astype(np.float32, copy=True)
    y = label.copy()
    n = x.shape[0]
    if kind == "none" or n == 0:
        return x, y

    if kind == "drop":
        keep = rng.random(n) >= float(cond["p"])
        return x[keep], y[keep]

    if kind == "dup":
        m = rng.random(n) < float(cond["p"])
        if not m.any():
            return x, y
        # 在原位置后插入一份副本
        out_x = np.repeat(x, np.where(m, 2, 1), axis=0)
        out_y = np.repeat(y, np.where(m, 2, 1))
        return out_x, out_y

    if kind == "jitter":
        s = float(cond["sigma"])
        i_dt, i_dev = idx["dt_norm"], idx["dt_dev"]
        # 以该段的**标准差**为抖动尺度（dt_norm 可能已居中，不能拿中位数当基准；
        # 2026-10-31 实测中位数可为负 → rng.normal 报 scale < 0）
        base = float(np.std(x[:, i_dt]))
        if not np.isfinite(base) or base <= 0:
            base = float(np.std(x[:, i_dev])) or 1e-3
        base = abs(base)
        x[:, i_dt] = x[:, i_dt] + rng.normal(0, abs(s * base), n)
        x[:, i_dev] = x[:, i_dev] + rng.normal(0, abs(s * base), n)
        return x, y

    if kind == "idswap":
        p = float(cond["p"])
        m = rng.random(n) < p
        if not m.any():
            return x, y
        donor = rng.integers(0, n, int(m.sum()))
        x[m, idx["id_norm"]] = x[donor, idx["id_norm"]]
        return x, y

    if kind == "payload":
        s = float(cond["sigma"])
        for name in ("byte_mean", "byte_std", "byte_entropy", "nonzero_ratio",
                     "payload_delta", "payload_u32"):
            j = idx.get(name)
            if j is None:
                continue
            x[:, j] = x[:, j] * (1.0 + rng.normal(0, s, n))
        for name in ("nonzero_ratio",):                     # 比例型通道裁到 [0,1]
            j = idx.get(name)
            if j is not None:
                x[:, j] = np.clip(x[:, j], 0.0, 1.0)
        return x, y

    raise ValueError(f"未知扰动：{kind}")


def eval_condition(model, needs, frames: dict, cond: dict, idx: dict, *,
                   device: str, W: int, stride: int, batch: int,
                   rng: np.random.Generator) -> dict[str, np.ndarray]:
    """对选定 captures 的 test ranges 逐段扰动、开窗、前向。

    ★ split 张量只存该 split 的帧（按 ranges 顺序拼接），manifest 的 range 是
      整条 capture 的**绝对帧号** —— 必须用前缀和换算成本地偏移，
      否则会读到别的区段（这是本项目记录在案的坑 #1）。
    """
    import torch

    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    caps: list[np.ndarray] = []
    with torch.no_grad():
        for cap_id, (v1, lab, ranges) in frames.items():
            lens = [b - a for a, b in ranges]
            offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
            for k, _ in enumerate(ranges):
                seg_x = v1[offs[k]:offs[k + 1]]
                seg_y = lab[offs[k]:offs[k + 1]]
                xs, ys = perturb(seg_x, seg_y, cond, idx, rng)
                n = xs.shape[0]
                if n < W:
                    continue
                starts = windows_in_range(0, n, W, stride)
                for i in range(0, starts.size, batch):
                    ss = starts[i:i + batch]
                    xb = np.stack([xs[s:s + W] for s in ss]).astype(np.float32)
                    yb = np.asarray([bool(ys[s:s + W].any()) for s in ss], dtype=np.int64)
                    t = torch.as_tensor(xb, device=device)
                    out = model(x1=t)
                    probs.append(torch.sigmoid(out.logit).cpu().numpy())
                    labels.append(yb)
                    caps.append(np.full(ss.size, cap_id, dtype=np.int32))
    if not probs:
        return {"prob": np.zeros(0), "label": np.zeros(0, dtype=np.int64),
                "cap": np.zeros(0, dtype=np.int32)}
    return {"prob": np.concatenate(probs), "label": np.concatenate(labels),
            "cap": np.concatenate(caps)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E1 输入扰动敏感性（eval-only）")
    ap.add_argument("--config", default=str(Path(__file__).parent / "training" / "configs" / "train.yaml"))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--max-captures", type=int, default=24, help="等距抽取多少个 test capture")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=None)
    args = ap.parse_args(argv)

    cfg = C.load(args.config, C.parse_overrides(args.overrides))
    setup_logging(str(cfg["runtime"]["log_level"]), None)

    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "metrics_test.json").read_text(encoding="utf-8"))
    tau = float(meta["threshold"])
    dataset = run_dir.parts[-4] if run_dir.parts[-1].startswith("seed") else None
    # run_dir = <batch>/<dataset>/<model>/<variant>/seed<N>
    parts = run_dir.resolve().parts
    dataset = parts[-4]
    model_name = parts[-3]

    model, ckpt = load_run_for_eval(run_dir / "checkpoints" / "best.pt", device=args.device)
    needs = tuple(type(model).needs)
    if needs != ("v1",):
        log.warning("模型 %s 的输入为 %s；E1 的扰动只作用于 v1，跳过该模型以免 v1/v2 不一致",
                    model_name, needs)
        return 2

    W = int(cfg["data"]["window"]["T"])
    stride = int(cfg["data"]["window"]["stride"]["test"])
    idx = channel_index(cfg, ["dt_norm", "dt_dev", "id_norm", "byte_mean", "byte_std",
                              "byte_entropy", "nonzero_ratio", "payload_delta", "payload_u32"])

    # 读取该语料 test split 的 ranges 与张量路径
    base = (Path(cfg["_resolved"]["tensors_root"]) / cfg["paths"]["dataset_dirs"][dataset]
            / cfg["paths"]["tensor_subdir"] / "test")
    splits = json.loads((Path(cfg["_resolved"]["tensors_root"]) / cfg["paths"]["dataset_dirs"][dataset]
                         / cfg["paths"]["manifest_subdir"] / "splits.json").read_text(encoding="utf-8"))
    caps = sorted(splits["captures"])
    caps = [c for c in caps if (splits["captures"][c].get("ranges") or {}).get("test")]
    if len(caps) > args.max_captures:
        pick = np.linspace(0, len(caps) - 1, args.max_captures).round().astype(int)
        caps = [caps[i] for i in sorted(set(pick))]
    log.info("语料 %s：抽取 %d/%d 个 test capture", dataset, len(caps), len(splits["captures"]))

    frames: dict[int, tuple] = {}
    for i, name in enumerate(caps):
        v1 = np.load(base / name / "v1.npy")
        lab = np.load(base / name / "label.npy")
        ranges = [(int(a), int(b)) for a, b in splits["captures"][name]["ranges"]["test"]]
        frames[i] = (v1, lab.astype(np.uint8), ranges)

    rows: list[dict] = []
    clean_mcc = None
    for cname in args.conditions:
        cond = CONDITIONS[cname]
        rng = np.random.default_rng(args.seed)
        res = eval_condition(model, needs, frames, cond, idx, device=args.device,
                             W=W, stride=stride, batch=args.batch, rng=rng)
        m = M.binary_metrics(res["label"], res["prob"], tau)
        row = {"dataset": dataset, "model": model_name, "condition": cname,
               "n": int(m["n"]), "n_pos": int(m["n_pos"]), "prevalence": float(m["prevalence"]),
               "mcc": float(m["mcc"]), "recall": float(m["recall"]), "fpr": float(m["fpr"]),
               "precision": float(m["precision"]), "threshold": tau}
        if cname == "clean":
            clean_mcc = row["mcc"]
        row["d_mcc"] = float(row["mcc"] - clean_mcc) if clean_mcc is not None else 0.0
        rows.append(row)
        log.info("%-10s MCC=%.4f recall=%.4f FPR=%.5f  n=%d", cname, row["mcc"],
                 row["recall"], row["fpr"], row["n"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows, list(rows[0].keys()))
    log.info("写出：%s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
