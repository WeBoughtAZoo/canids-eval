#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_eval.py —— 仅评估入口（不训练，可复现实验指标并重绘图表）
================================================================================
用途
----
1. 审稿人/合作者拿到 `outputs/.../seed42/` 目录后，一条命令复算全部指标；
2. 训练中断后只想重算指标；
3. 改了出图样式后只重绘。

用法
----
  python 代码/run_eval.py --run-dir "outputs/training/baseline/ctt15/fdtcn/full/seed42"
  python 代码/run_eval.py --all
  python 代码/run_eval.py --run-dir <dir> --no-figures

说明
----
- 阈值取自 checkpoint（训练时在 val 上标定），**不会**在 test 上重选；
- 标签表取自 checkpoint，不重新推断，避免口径漂移；
- 产物覆盖同名文件（metrics_*.json / per_type_*.csv / predictions_*.npz）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training import compare as CMP               # noqa: E402
from training import config as C                  # noqa: E402
from training import figures as F                 # noqa: E402
from training import tables as T                  # noqa: E402
from training import pipeline as P                # noqa: E402
from training.utils import get_logger, setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="车载 CAN / CAN FD 入侵检测 —— 仅评估入口")
    ap.add_argument("--config", default=str(Path(__file__).parent / "training" / "configs" / "train.yaml"),
                    help="仅用于 --all 时确定 output_root 与出图样式")
    ap.add_argument("--run-dir", default=None, help="单个 run 目录（含 config_resolved.yaml 与 checkpoints/）")
    ap.add_argument("--all", action="store_true", help="重算 output_root 下全部 run")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--set", dest="overrides", action="append", default=None)
    args = ap.parse_args(argv)

    if not args.run_dir and not args.all:
        ap.error("需要 --run-dir 或 --all")

    base = C.load(args.config, C.parse_overrides(args.overrides))
    setup_logging(str(base["runtime"]["log_level"]), None)
    log = get_logger("cli")

    if args.run_dir:
        run_dirs = [Path(args.run_dir).resolve()]
    else:
        run_dirs = [r.dir for r in F.discover_runs(base["_resolved"]["output_root"], base)]
        log.info("发现 %d 个 run", len(run_dirs))

    failures = 0
    for rd in run_dirs:
        try:
            s = P.reevaluate(rd, figures=not args.no_figures, base_cfg=base)
            log.info("[%s] test MCC=%.4f AUC-PR=%.4f F1=%.4f τ=%.4f",
                     rd.name, s["test"]["mcc"], s["test"]["auc_pr"],
                     s["test"]["f1"], s["threshold"])
        except Exception as e:
            failures += 1
            log.exception("评估失败 %s：%s", rd, e)
    if not args.no_figures and run_dirs:
        try:
            F.generate_aggregate(base)     # 全部 run 评估完后统一出汇总图
        except Exception as e:
            log.warning("汇总图生成失败：%s", e)
    try:
        CMP.run_comparison(base)
        T.generate_all(base)
    except Exception as e:
        log.warning("表格 / 配对检验生成失败：%s", e)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
