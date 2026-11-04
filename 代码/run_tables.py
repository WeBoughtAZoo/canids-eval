#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_tables.py —— 论文表格生成入口（LaTeX booktabs + CSV）
================================================================================
用法
----
  # 生成全部表（读批次内所有 run 的 metrics / per_type）
  python 代码/run_tables.py

  # 只生成某几张
  python 代码/run_tables.py --only t7_main t10_ablation

  # 指定批次
  python 代码/run_tables.py --set project.run_name=main

产物
----
  <batch>/tables/table_main.{tex,csv}
  <batch>/tables/table_masquerade.{tex,csv}
  <batch>/tables/table_ablation.{tex,csv}
  <batch>/tables/table_complexity.{tex,csv}
  <batch>/tables/table_stats.{tex,csv}        ← 需先跑 run_compare.py
  <batch>/tables/tables_index.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training import config as C          # noqa: E402
from training import tables as T          # noqa: E402
from training.utils import get_logger, setup_logging  # noqa: E402

KEYS = list(T.TABLE_FUNCS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="论文表格生成（booktabs LaTeX + CSV）")
    ap.add_argument("--config", default=str(Path(__file__).parent / "training" / "configs" / "train.yaml"))
    ap.add_argument("--only", nargs="+", default=None, choices=KEYS)
    ap.add_argument("--set", dest="overrides", action="append", default=None)
    args = ap.parse_args(argv)

    cfg = C.load(args.config, C.parse_overrides(args.overrides))
    setup_logging(str(cfg["runtime"]["log_level"]), None)
    log = get_logger("cli")

    rows = T.generate_all(cfg, only=args.only)
    if not rows:
        log.warning("没有生成任何表（检查 tables.enabled 与批次内是否存在 run）")
        return 0
    for r in rows:
        log.info("%-14s %s", r["table"], "OK  " + r["files"] if r["generated"] else "跳过（缺产物）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
