#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_kd.py —— KD-Logic 蒸馏入口（教师 = 已训练好的 FD-TCN run）
================================================================================
用法
----
  python 代码/run_kd.py --run-dir "outputs/training/baseline/hcrl_canfd/fdtcn/full/seed42"

产物写入教师 run 目录下的 `kd/`：
  kd_results.json     保真度 / 规则复杂度 / 蒸馏 vs 直接训练 / zero-shot
  rules_preview.txt   前 N 条规则（论文 F11 可直接引用）
  tree.dot            Graphviz 源文件（如需渲染 PDF 自行 `dot -Tpdf`）

前置条件：`kd.enabled=true` 或本脚本显式调用（脚本本身即视为启用）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training import config as C                  # noqa: E402
from training import pipeline as P                # noqa: E402
from training.artifacts import RunContext         # noqa: E402
from training.evaluator import zero_shot_types    # noqa: E402
from training.kdlogic import run_kd               # noqa: E402
from training.utils import (configure_torch, get_logger, set_seed, setup_logging)  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="KD-Logic：DL 教师 → 约束 CART 蒸馏")
    ap.add_argument("--run-dir", required=True, help="教师 run 目录")
    ap.add_argument("--ckpt", default=None, help="默认用该 run 的 checkpoints/best.pt")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"缺少配置快照：{cfg_path}")
    cfg = C.load(cfg_path)
    setup_logging(str(cfg["runtime"]["log_level"]), None)
    log = get_logger("cli")

    torch = configure_torch(cfg)
    device = torch.device(str(cfg["runtime"]["device"]))
    seed = int(cfg["train"]["seeds"][0])
    set_seed(seed, deterministic=bool(cfg["runtime"]["deterministic"]))

    dataset = str(cfg["data"]["datasets"][0])
    prepared = P.prepare_data(cfg, dataset, seed=seed)
    cfg["_zero_shot_types"] = zero_shot_types(
        prepared["label_map"], str(cfg["label_space"]["unknown_label"]),
        prepared["datasets"]["test"])

    run = RunContext(cfg, dataset=dataset, model=str(cfg["model"]["name"]),
                     variant=cfg.get("_variant", "full"), seed=seed)
    run.run_dir = run_dir
    run.ckpt_dir = run_dir / "checkpoints"
    run.fig_dir = run_dir / str(cfg["paths"]["figures_subdir"])

    ckpt = Path(args.ckpt) if args.ckpt else run.best_ckpt
    log.info("教师 checkpoint：%s", ckpt)
    res = run_kd(cfg, teacher_ckpt=ckpt, run=run, datasets=prepared["datasets"], device=device)
    log.info("蒸馏结果：%s", {k: res["distilled"][k] for k in ("fidelity", "mcc", "n_rules", "depth")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
