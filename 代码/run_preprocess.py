#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_preprocess.py —— 三数据集预处理统一入口（CLI）
================================================================================
依据《三数据集形态解析与预处理方案.html》实现，覆盖：
  HCRL CAN-FD · ROAD · can-train-and-test v1.5
流程：解析 → 清洗（缺失值/异常值）→ 时间序切分 → 训练集统计 → 统一编码+归一化 → 落盘 → 清单

快速开始
--------
  # 1) 全量预处理（HCRL + ROAD + CTT v1.5），产物写入"训练数据集/"
  python 代码/run_preprocess.py --datasets all

  # 2) 冒烟测试：每个数据集只跑前 1 个 capture
  python 代码/run_preprocess.py --datasets all --limit 1 --force

  # 3) 只跑某个数据集 / 指定 capture
  python 代码/run_preprocess.py --datasets road
  python 代码/run_preprocess.py --datasets ctt15 --captures set_01__test_06_masquerade__masquerade-timed-1 --force

  # 4) 只列出将要处理的 capture（不写盘）
  python 代码/run_preprocess.py --datasets ctt15 --list

常用参数
--------
  --datasets    hcrl_canfd | road | ctt15 | all（可多选，空格分隔）
  --out         产物根目录，默认 <项目根>/训练数据集
  --limit N     每个数据集最多处理 N 个 capture（用于冒烟）
  --captures    只处理这些 capture 名（逗号分隔）
  --force       忽略断点记录，强制重算
  --list        只列出 capture 清单
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让 `import preprocess` 生效（脚本位于 <项目根>/代码/ 下）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import config as C            # noqa: E402
from preprocess import parsers, pipeline      # noqa: E402
from preprocess.utils import ensure_dir, log, write_json  # noqa: E402

ALL_DATASETS = ["hcrl_canfd", "road", "ctt15"]


def _tee_log(path: Path):
    """把 stdout 同时写入 run log（便于长任务后台运行时留痕）。"""
    ensure_dir(path.parent)
    fh = open(path, "a", encoding="utf-8")

    class _Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            fh.write(s)

        def flush(self):
            sys.__stdout__.flush()
            fh.flush()

    sys.stdout = _Tee()  # type: ignore[assignment]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="车载 CAN / CAN FD 三数据集统一预处理（HCRL CAN-FD / ROAD / CTT v1.5）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--datasets", nargs="+", default=["all"],
                    help="hcrl_canfd / road / ctt15 / all（默认 all）")
    ap.add_argument("--out", type=str, default=None, help="产物根目录（默认 <项目根>/训练数据集）")
    ap.add_argument("--limit", type=int, default=None, help="每个数据集最多处理 N 个 capture")
    ap.add_argument("--captures", type=str, default=None, help="只处理这些 capture 名（逗号分隔）")
    ap.add_argument("--force", action="store_true", help="忽略断点记录，强制重算")
    ap.add_argument("--list", action="store_true", help="只列出 capture 清单，不写盘")
    ap.add_argument("--no-logfile", action="store_true", help="不写 run log 文件")
    ap.add_argument("--road-capture-holdout", action="store_true",
                    help="★ 敏感性实验：ROAD 改用 capture 级留出（整文件归入一个 split）替代块化交错。"
                         "必须与 --out 指向独立目录同时使用，避免覆盖 训练数据集/ROAD")
    args = ap.parse_args(argv)

    if args.road_capture_holdout:
        C.ROAD_CAPTURE_HOLDOUT = True
        if not args.out:
            print("错误：--road-capture-holdout 必须配合 --out 指向独立目录（禁止覆盖既有产物）",
                  file=sys.stderr)
            return 2

    datasets = ALL_DATASETS if "all" in args.datasets else args.datasets
    bad = [d for d in datasets if d not in ALL_DATASETS]
    if bad:
        print(f"未知数据集：{bad}；可选 {ALL_DATASETS}", file=sys.stderr)
        return 2

    if args.out:
        C.OUT_ROOT = Path(args.out).resolve()
    ensure_dir(C.OUT_ROOT)

    only = [s.strip() for s in args.captures.split(",")] if args.captures else None

    if not args.no_logfile:
        _tee_log(C.OUT_ROOT / "preprocess_run.log")

    log("=" * 78)
    log(f"预处理启动 | 数据集={datasets} | 产物根目录={C.OUT_ROOT}")
    log(f"原始数据根目录={C.DATA_ROOT}")
    split_note = (
        f"CTT_SPLIT=语义(train_01/02→train, 其中按攻击类型分层留出{C.CTT_VAL_QUOTA:.0%}→val;"
        f"test_01-06→test)"
        if C.CTT_SEMANTIC_SPLIT else f"SPLIT={C.SPLIT_RATIOS}"
    )
    log(f"口径：WIN_T={C.WIN_T} STRIDE(train/test)={C.STRIDE_TRAIN}/{C.STRIDE_TEST} "
        f"PAD_BYTES={C.PAD_BYTES} {split_note} "
        f"ROAD_FUZZING_MODE={C.ROAD_FUZZING_MODE} PROTO_FLAG_MODE={C.PROTO_FLAG_MODE}")
    if C.CTT_SEMANTIC_SPLIT:
        log(f"  ※ CTT v1.5：语义划分（按文件名角色整文件归入；val 为按攻击类型分层留出，"
            f"配额 {C.CTT_VAL_QUOTA:.0%}，seed={C.SEED}）")
    log(f"  ※ {list(C.BLOCK_MODE_DATASETS)}：★ 块化交错切分"
        f"（K={C.BLOCK_K}；i%{C.BLOCK_PATTERN_PERIOD}=={C.BLOCK_VAL_MOD}→val、"
        f"i%{C.BLOCK_PATTERN_PERIOD}=={C.BLOCK_TEST_MOD}→test、其余 train；"
        f"块边界对齐攻击段边界={C.ALIGN_TO_SEGMENT_EDGE}；确定性、无随机数）")
    log(f"  ※ 其余情况：capture 内时间序连续切分 {C.SPLIT_RATIOS}")
    log("=" * 78)

    # ---------------- --list ---------------- #
    if args.list:
        for ds in datasets:
            specs = parsers.list_captures(ds)
            if only:
                specs = [s for s in specs if s.capture in set(only)]
            log(f"[{ds}] {len(specs)} 个 capture")
            for s in specs:
                print(f"    {s.capture:<70} atk={s.attack_type:<18} "
                      f"masq={int(s.is_masquerade)} sup={int(s.supervised_evaluable)}")
        return 0

    # ---------------- 主流程 ---------------- #
    summary = pipeline.run(datasets, limit=args.limit, force=args.force, only_captures=only)

    log("-" * 78)
    for ds, info in summary["datasets"].items():
        if "error" in info:
            log(f"  {ds:<12} 失败：{info['error']}")
        else:
            log(f"  {ds:<12} capture={info['n_captures']:<4} 体积={info['size']:<10} "
                f"耗时={info['elapsed_sec']}s")
    log(f"汇总文件：{C.OUT_ROOT / 'preprocess_summary.json'}")
    log("预处理结束")
    write_json(C.OUT_ROOT / "preprocess_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
