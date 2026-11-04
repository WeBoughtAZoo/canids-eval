#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_figures.py —— 图表生成入口（只读产物，不加载模型）
================================================================================
用法
----
  # 重绘单个 run 的全部图
  python 代码/run_figures.py --run-dir "outputs/training/baseline/ctt15/fdtcn/full/seed42"

  # 重绘 output_root 下全部 run + 跨 run 汇总图（fig08/09/12）
  python 代码/run_figures.py --all

  # 只出某几张
  python 代码/run_figures.py --run-dir <dir> --ids fig02 fig07

  # 打印图表目录（主题、命名规则、格式），不生成
  python 代码/run_figures.py --catalog
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training import config as C                  # noqa: E402
from training import figures as F                 # noqa: E402
from training.utils import get_logger, setup_logging, write_csv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SCI 论文图表生成入口")
    ap.add_argument("--config", default=str(Path(__file__).parent / "training" / "configs" / "train.yaml"))
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ids", nargs="+", default=None, help="只生成这些图，如 fig02 fig07")
    ap.add_argument("--catalog", action="store_true", help="只打印图表目录")
    ap.add_argument("--set", dest="overrides", action="append", default=None,
                    help="覆盖配置项（如 --set project.run_name=smoke）")
    args = ap.parse_args(argv)

    cfg = C.load(args.config, C.parse_overrides(args.overrides))
    setup_logging(str(cfg["runtime"]["log_level"]), None)
    log = get_logger("cli")

    if args.catalog:
        rows = F.catalog_rows(cfg)
        out = Path(cfg["_resolved"]["output_root"]) / "figure_catalog.csv"
        write_csv(out, rows)
        for r in rows:
            print(f"{r['figure_id']}  {r['filename_rule']:<42} {r['title_zh']}")
        log.info("图表目录已写入：%s", out)
        return 0

    if not args.run_dir and not args.all:
        ap.error("需要 --run-dir / --all / --catalog")

    targets = [Path(args.run_dir).resolve()] if args.run_dir else \
        [r.dir for r in F.discover_runs(cfg["_resolved"]["output_root"], cfg)]
    log.info("目标 run 数：%d", len(targets))
    total = 0
    for rd in targets:
        rows = F.generate_for_run(rd, cfg, ids=args.ids)
        total += sum(1 for r in rows if r["generated"])
    agg = F.generate_aggregate(cfg, ids=args.ids)
    total += sum(1 for r in agg if r["generated"])
    log.info("完成：成功生成 %d 张图（每张按配置输出多份格式）", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
