#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_compare.py —— 配对显著性检验入口（McNemar + Holm，产出 T13）
================================================================================
用法
----
  # 按配置自动配对（同数据集的主干两两 + 消融 vs full）
  python 代码/run_compare.py

  # 只比指定的两个 run
  python 代码/run_compare.py --a "outputs/.../fdtcn/full/seed42" --b "outputs/.../rgbnet/full/seed42"

产物
----
  <batch>/comparison/comparison.csv   （b, c, statistic, p, p_holm, significant）
  若随后运行 run_tables.py，会额外得到 tables/table_stats.{tex,csv}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from training import compare as CMP       # noqa: E402
from training import config as C          # noqa: E402
from training import tables as T          # noqa: E402
from training.utils import get_logger, setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="McNemar 配对检验（Holm 校正）")
    ap.add_argument("--config", default=str(Path(__file__).parent / "training" / "configs" / "train.yaml"))
    ap.add_argument("--a", default=None, help="run A 目录")
    ap.add_argument("--b", default=None, help="run B 目录")
    ap.add_argument("--set", dest="overrides", action="append", default=None)
    ap.add_argument("--regime", choices=["full", "known"], default="full",
                    help="评测口径：full=全部正样本（默认）；known=剔除 zero-shot 正样本（§4.5）")
    ap.add_argument("--out-name", default=None,
                    help="输出文件名（默认 comparison.csv / comparison_known.csv）")
    args = ap.parse_args(argv)

    cfg = C.load(args.config, C.parse_overrides(args.overrides))
    setup_logging(str(cfg["runtime"]["log_level"]), None)
    log = get_logger("cli")

    if bool(args.a) != bool(args.b):
        ap.error("--a 与 --b 必须同时给出")

    if args.a and args.b:
        res = CMP.compare_pair(args.a, args.b,
                               correction=bool(cfg["eval"]["mcnemar"]["correction"]),
                               cluster=bool(C.get_opt(cfg, "eval.mcnemar.cluster", True)),
                               regime=args.regime)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    fname = args.out_name or ("comparison.csv" if args.regime == "full"
                              else f"comparison_{args.regime}.csv")
    rows = CMP.run_comparison(cfg, regime=args.regime, filename=fname)
    if not rows:
        log.warning("没有可配对的 run；先训练至少两个模型或一个消融变体")
        return 0
    for r in rows:
        log.info("%-28s b=%d c=%d p=%.3e p_holm=%.3e %s",
                 r["comparison"], r["b"], r["c"], r["p_value"], r["p_holm"],
                 "significant" if r["significant"] else "")
    # 顺手刷新 T13
    try:
        T.table_stats(cfg)
    except Exception as e:
        log.warning("T13 表格生成失败：%s", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
