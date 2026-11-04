#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_train.py —— 训练入口（单模型 / 多数据集 / 多消融 / 多 seed）
================================================================================
只负责参数解析与调度，全部实现位于 `training/` 包；所有数值参数来自 YAML。

快速开始
--------
  # 1) 按配置默认值跑一次（数据集/模型/变体/种子全部取自 YAML）
  python 代码/run_train.py

  # 2) 主线：CTT v1.5 上的 FD-TCN，5 个种子
  python 代码/run_train.py --datasets ctt15 --models fdtcn --seeds 42 43 44 45 46

  # 3) 消融：Ab-1（去 pad_ratio）与 Ab-1b（去字节图掩码）
  python 代码/run_train.py --variants Ab-1 Ab-1b --models fdtcn

  # 4) 只做数据侧体检，不训练（校验 manifest / 标签表 / 窗口数）
  python 代码/run_train.py --dry-run

  # 5) 临时覆盖任意配置项（写进本次 run 的 config_resolved.yaml，可追溯）
  python 代码/run_train.py --set train.epochs=10 --set data.limits.max_train_windows=20000

参数
----
  --config    配置文件路径（默认 代码/training/configs/train.yaml）
  --datasets  一个或多个（默认 data.datasets）
  --models    一个或多个（默认 model.name）
  --variants  一个或多个消融变体（默认 ablation.default_variant）
  --seeds     一个或多个整数种子（默认 train.seeds）
  --set k=v   覆盖任意配置项，可重复
  --no-figures 训练后不出图
  --dry-run   只做数据体检
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training import compare as CMP           # noqa: E402
from training import config as C              # noqa: E402
from training import figures as F             # noqa: E402
from training import pipeline as P            # noqa: E402
from training import tables as T              # noqa: E402
from training.utils import get_logger, setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="车载 CAN / CAN FD 入侵检测 —— 训练入口（FD-TCN 主线）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default=str(Path(__file__).parent / "training" / "configs" / "train.yaml"))
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=None,
                    help="覆盖配置项，形如 --set train.epochs=10")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-summary", action="store_true",
                    help="跳过收尾的配对检验与表格生成（并行分片跑 run 时必用）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    overrides = C.parse_overrides(args.overrides)
    cfg = C.load(args.config, overrides)
    setup_logging(str(cfg["runtime"]["log_level"]),
                  Path(cfg["_resolved"]["output_root"]) / "train.log")
    log = get_logger("cli")

    datasets = args.datasets or list(cfg["data"]["datasets"])
    models = args.models or [str(cfg["model"]["name"])]
    variants = args.variants or [str(cfg["ablation"]["default_variant"])]
    seeds = args.seeds or [int(s) for s in cfg["train"]["seeds"]]

    for d in datasets:
        if d not in cfg["paths"]["dataset_dirs"]:
            log.error("未知数据集 %s（可选 %s）", d, sorted(cfg["paths"]["dataset_dirs"]))
            return 2
    for m in models:
        if m not in cfg["model"]["variants"]:
            log.error("未知模型 %s（可选 %s）", m, sorted(cfg["model"]["variants"]))
            return 2
    for v in variants:
        if v not in cfg["ablation"]["variants"]:
            log.error("未知消融变体 %s（可选 %s）", v, sorted(cfg["ablation"]["variants"]))
            return 2

    log.info("=" * 78)
    log.info("训练批次 | datasets=%s models=%s variants=%s seeds=%s", datasets, models, variants, seeds)
    log.info("配置=%s（指纹 %s）", args.config, cfg["_meta"]["config_sha256_16"])
    log.info("产物根目录=%s", cfg["_resolved"]["output_root"])
    log.info("口径：W=%d  stride(train/val/test)=%d/%d/%d  窗口标签=%s",
             cfg["data"]["window"]["T"],
             cfg["data"]["window"]["stride"]["train"],
             cfg["data"]["window"]["stride"]["val"],
             cfg["data"]["window"]["stride"]["test"],
             cfg["data"]["window"]["label_rule"])
    log.info("=" * 78)

    if args.dry_run:
        ok = True
        for d in datasets:
            for m in models:
                for v in variants:
                    eff = P.prepare_config(cfg, dataset=d, model=m, variant=v, seed=seeds[0])
                    try:
                        prep = P.prepare_data(eff, d, seed=seeds[0])
                    except Exception as e:
                        log.error("[dry-run] %s/%s/%s 失败：%s", d, m, v, e)
                        ok = False
                        continue
                    log.info("[dry-run] %s/%s/%s | windows train/val/test = %d/%d/%d | 类别=%d",
                             d, m, v, prep["stats"]["train"]["n_windows"],
                             prep["stats"]["val"]["n_windows"],
                             prep["stats"]["test"]["n_windows"], len(prep["label_map"]))
        return 0 if ok else 1

    t0 = time.perf_counter()
    failures = 0
    for d in datasets:
        for m in models:
            for v in variants:
                for s in seeds:
                    try:
                        P.run_single(cfg, dataset=d, model=m, variant=v, seed=s,
                                     figures=not args.no_figures)
                    except Exception as e:  # 单个 run 失败不阻断其余组合
                        failures += 1
                        log.exception("[%s/%s/%s/seed%d] 失败：%s", d, m, v, s, e)
                        if bool(cfg["runtime"]["fail_fast"]):
                            return 1
    if not args.no_figures:
        try:
            F.generate_aggregate(cfg)     # 批次末尾统一出跨 run 汇总图
        except Exception as e:
            log.warning("汇总图生成失败：%s", e)
    if args.no_summary:
        # ★ 2026-10-27：并行跑多个 run 时必须用它，否则每个进程都会写同一份
        #   comparison/ 与 tables/，互相覆盖出半成品表。全部 run 跑完后再单独执行
        #   `run_compare.py` + `run_tables.py` 即可。
        log.info("已跳过批次汇总（--no-summary）：请在全部分片结束后统一跑表")
    else:
        try:
            CMP.run_comparison(cfg)           # 配对检验（T13）
            T.generate_all(cfg)               # 论文表格（T7/T8/T10/T12/T13）
        except Exception as e:
            log.warning("表格 / 配对检验生成失败：%s", e)
    log.info("全部完成：%d 个 run，失败 %d 个，总耗时 %.1f s",
             len(datasets) * len(models) * len(variants) * len(seeds),
             failures, time.perf_counter() - t0)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
