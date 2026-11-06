#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calib_logits.py —— 导出指定 run 的 logits（仅推理，不训练、不改任何已有产物）
================================================================================
用途
----
标定诊断需要**未经过 sigmoid** 的分数：已发布的 `predictions_*.npz` 只存 `prob`，
而 float32 sigmoid 在 logit ≳ 17 时全部饱和到 1.0，饱和点的先后次序无法从 prob 恢复。
本脚本用同一 checkpoint 重跑前向，落 `logit / prob / label`，供 `calib_diag.py` 使用。

用法
----
  python 代码/scripts/calib_logits.py --run-dir outputs/training/loso_set04/ctt15/fdtcn/full/seed42
  python 代码/scripts/calib_logits.py --run-dir <dir> --splits val,test --out-dir outputs/audit/calibration

产物
----
  <out-dir>/<batch>_<split>_logits.npz   含 logit(float32) / prob(float32) / label(uint8)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "代码"))

from training import config as C                      # noqa: E402
from training.artifacts import RunContext             # noqa: E402
from training.data import make_loader                 # noqa: E402
from training.evaluator import _forward, load_run_for_eval  # noqa: E402
from training.pipeline import prepare_data            # noqa: E402


def _patch_tensors_root(cfg: dict) -> dict:
    """GPU 租赁机上训练的 run，其 config_resolved.yaml 记的是 /localdisk-tmp/... 路径。
    若该路径本机不存在，按末两级（如 set_04/CTT_v1.5）在项目根下重新定位，
    并把 _resolved 里所有派生路径（dataset_roots / project_root）一并改写。"""
    cur = Path(str(cfg["_resolved"]["tensors_root"]))
    if cur.exists():
        return cfg
    cand = ROOT.joinpath(*cur.parts[-2:])
    if not cand.exists():
        raise FileNotFoundError(f"tensors_root 本机不存在，且无法按末两级定位：{cur}")
    cfg["_resolved"]["tensors_root"] = str(cand)
    cfg["_resolved"]["project_root"] = str(ROOT)
    cfg["paths"]["tensors_root"] = str(cand)
    cfg["_resolved"]["dataset_roots"] = {
        k: str(cand / sub) for k, sub in cfg["paths"]["dataset_dirs"].items()
    }
    print(f"[paths] tensors_root 改写：{cur} -> {cand}")
    return cfg


def dump(run_dir: Path, splits: list[str], out_dir: Path, max_batches: int | None) -> None:
    import torch

    cfg = _patch_tensors_root(C.load(run_dir / "config_resolved.yaml"))
    run = RunContext(cfg, dataset=cfg["data"]["datasets"][0], model=cfg["model"]["name"],
                     variant=str(cfg.get("_meta", {}).get("variant") or "full"),
                     seed=int(cfg["train"]["seeds"][0]))
    run.run_dir = run_dir
    run.ckpt_dir = run_dir / "checkpoints"

    ckpt_path = run.best_ckpt if run.best_ckpt.exists() else run.final_ckpt
    model, ckpt = load_run_for_eval(ckpt_path, device="cpu")
    prepared = prepare_data(cfg, ckpt["dataset"], seed=int(ckpt["seed"]))
    needs = tuple(type(model).needs)
    bs = int(cfg["train"]["eval_batch_size"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{run_dir.parts[-5]}_{ckpt['dataset']}_{run_dir.parts[-3]}_{run_dir.parts[-2]}"

    for split in splits:
        ds = prepared["datasets"][split]
        loader = make_loader(ds, cfg, split=split, batch_size=bs, shuffle=False)
        logits, probs, labels = [], [], []
        t0 = time.time()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                out = _forward(model, batch, "cpu", needs)
                logits.append(out.logit.detach().float().cpu().numpy())
                probs.append(torch.sigmoid(out.logit).detach().float().cpu().numpy())
                labels.append(batch["y"].numpy().astype(np.uint8))
                if max_batches is not None and i + 1 >= max_batches:
                    break
        lg = np.concatenate(logits)
        pb = np.concatenate(probs)
        lb = np.concatenate(labels)
        path = out_dir / f"{tag}_{split}_logits.npz"
        np.savez_compressed(path, logit=lg, prob=pb, label=lb)
        dt = time.time() - t0
        print(f"[{split}] {len(lb):,} 窗口  {dt:.1f}s  "
              f"logit max={lg.max():.3f}  p==1.0 {(pb[lb == 0] == 1.0).mean() * 100:.2f}% (neg)  -> {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/audit/calibration"))
    ap.add_argument("--max-batches", type=int, default=None, help="仅用于计时试跑")
    a = ap.parse_args()
    dump(Path(a.run_dir), [s for s in a.splits.split(",") if s],
         Path(a.out_dir), a.max_batches)


if __name__ == "__main__":
    main()
