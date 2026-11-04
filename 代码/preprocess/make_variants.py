#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_variants.py —— 由既有张量生成 LOSO / LOCO 切分变体（不重跑预处理）
================================================================================
背景
----
E4（CTT 留一录制组）与 E6（HCRL 留一 capture）需要**换切分**，而重解析原始 CSV 代价高。
本脚本只重排**已有张量**，不碰原始数据。两种模式的语义不同，必须区分：

  · `loso`（CTT）：capture 名带 `set_0x__` 前缀。第 k 折把 `set_0x` 的全部 capture 划为 test，
    其余组按确定性规则（排序后每 5 个取 1）抽 val，剩下为 train。
    CTT 的每个 capture **只属于一个原 split**，故可整段搬移，新 ranges 归一为 [[0, n)。
    注意：原始 ranges 之间的空隙在张量里已被去掉，因此新切分把该 capture 视为**连续序列**；
    对 CTT（每 capture 单段）这是精确等价的。

  · `loco`（HCRL / ROAD）：capture 是**块化交错**的，同一 capture 在 train/val/test 都有帧。
    第 k 折：test 只保留 capture k 的 test 段；train/val 保留其余 capture 的对应段。
    即**按 (capture, split) 过滤**，ranges 沿用绝对帧号，不搬移。

为什么用硬链接
--------------
同一文件系统下 `os.link` 让 4 折 LOSO 几乎不占额外空间（CTT 每折约 7.7 GB）；
跨盘或不支持时自动退化为复制（`--copy` 可强制）。

产物结构（与预处理产物同构，可直接作为 run_train.py 的 paths.tensors_root）
--------------------------------------------------------------------------
  <out>/<fold>/<dataset_dir>/
      02_tensors/{train,val,test}/<capture>/{v1.npy,label.npy[,...]}
      03_manifest/{splits.json, env.json, captures/<capture>.json}

用法
----
  python 代码/preprocess/make_variants.py --dataset ctt15 --mode loso --out 训练数据集_LOSO --dry-run
  python 代码/preprocess/make_variants.py --dataset hcrl_canfd --mode loco --out 训练数据集_LOCO
  # 训练（示例）：
  # python 代码/run_train.py --datasets ctt15 --models fdtcn --seeds 42 --set project.run_name=loso \
  #   --set paths.tensors_root=训练数据集_LOSO/set_01
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training import config as C                      # noqa: E402
from training.manifest import count_windows           # noqa: E402
from training.utils import get_logger, setup_logging  # noqa: E402

log = get_logger("variants")

ARRAY_NAMES = ("v1.npy", "v2.npy", "label.npy", "id_valid.npy")
SPLITS = ("train", "val", "test")


def group_of(dataset: str, capture: str) -> str:
    """分折依据：CTT 按录制组（set_0x），其余按 capture 本身（留一）。"""
    if dataset == "ctt15":
        return capture.split("__", 1)[0]
    return capture


def n_windows(n_frames: int, split: str, win: int, st_train: int, st_test: int) -> int:
    """单段连续帧的窗口数。"""
    return count_windows(n_frames, win, st_train if split == "train" else st_test)


def n_windows_ranges(ranges, split: str, win: int, st_train: int, st_test: int) -> int:
    """多段 ranges 的窗口数：**必须逐段求和**。

    ★ 2026-10-31 踩坑：用"总帧数"一次算会把每多一段就多算约 (W−1)/stride 个窗口，
      manifest.verify 的 declared vs recount 立刻报错（LOCO 首跑差 14 个）。
    """
    stride = st_train if split == "train" else st_test
    return sum(count_windows(int(b) - int(a), win, stride) for a, b in ranges)


def place(src_dir: Path, dst_dir: Path, *, copy: bool) -> None:
    """把 capture 目录放到目标位置：默认硬链接（同盘零成本），失败则复制。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ARRAY_NAMES:
        s, d = src_dir / name, dst_dir / name
        if not s.exists() or d.exists():
            continue
        if not copy:
            try:
                os.link(s, d)
                continue
            except OSError:
                pass
        shutil.copy2(s, d)


def plan_entries(dataset: str, mode: str, captures: dict, fold: str,
                 groups: dict[str, list[str]]) -> list[dict[str, Any]]:
    """返回 [(capture, src_split, dst_split, ranges)] 的扁平计划。"""
    entries: list[dict[str, Any]] = []
    if mode == "loso":
        train_pool: list[str] = [n for g in sorted(groups) if g != fold for n in sorted(groups[g])]
        for name, meta in captures.items():
            present = [sp for sp in SPLITS if meta["ranges_abs"][sp]]
            if len(present) != 1:
                raise ValueError(
                    f"[{dataset}] {name} 出现在 {present} 个 split；loso 模式要求每个 capture "
                    f"只属于一个 split（CTT 满足；HCRL/ROAD 请用 --mode loco）")
            src = present[0]
            dst = "test" if group_of(dataset, name) == fold else (
                "val" if train_pool.index(name) % 5 == 4 else "train")
            n = sum(b - a for a, b in meta["ranges_abs"][src])
            # 连续化：张量里的帧本来就是按 ranges 顺序拼接的
            entries.append({"capture": name, "src": src, "dst": dst, "ranges": [[0, n]]})
        return entries

    # loco：按 (capture, split) 过滤，归属不变
    for name, meta in captures.items():
        is_held_out = group_of(dataset, name) == fold
        for sp in SPLITS:
            rs = meta["ranges_abs"][sp]
            if not rs:
                continue
            if is_held_out and sp != "test":
                continue
            if (not is_held_out) and sp == "test":
                continue
            entries.append({"capture": name, "src": sp, "dst": sp,
                            "ranges": [[a, b] for a, b in rs]})
    return entries


def build_fold(cfg: dict, dataset: str, mode: str, fold: str, entries: list[dict],
               *, out_root: Path, copy: bool, win: int, st_train: int,
               st_test: int) -> dict[str, Any]:
    ddir = str(cfg["paths"]["dataset_dirs"][dataset])
    src_root = Path(cfg["_resolved"]["tensors_root"]) / ddir
    src_manifest = src_root / str(cfg["paths"]["manifest_subdir"])
    dst_root = out_root / fold / ddir
    dst_manifest = dst_root / str(cfg["paths"]["manifest_subdir"])
    (dst_manifest / "captures").mkdir(parents=True, exist_ok=True)

    # capture → {split: ranges}
    by_capture: dict[str, dict[str, list]] = {}
    cap_count = {sp: 0 for sp in SPLITS}
    win_sum = {sp: 0 for sp in SPLITS}

    for e in entries:
        name, src, dst, ranges = e["capture"], e["src"], e["dst"], e["ranges"]
        src_dir = src_root / str(cfg["paths"]["tensor_subdir"]) / src / name
        if not src_dir.exists():
            raise FileNotFoundError(f"缺少源张量目录：{src_dir}")
        place(src_dir, dst_root / str(cfg["paths"]["tensor_subdir"]) / dst / name, copy=copy)
        by_capture.setdefault(name, {})[dst] = ranges
        cap_count[dst] += 1
        win_sum[dst] += n_windows_ranges(ranges, dst, win, st_train, st_test)

    for name, split_ranges in by_capture.items():
        cap_json = json.loads((src_manifest / "captures" / f"{name}.json").read_text(encoding="utf-8"))
        cap_json["splits"] = {}
        for sp, ranges in split_ranges.items():
            cap_json["splits"][sp] = {
                "ranges": ranges,
                "windows_stride_train": n_windows_ranges(ranges, "train", win, st_train, st_test),
                "windows_stride_test": n_windows_ranges(ranges, "test", win, st_train, st_test),
                "split_mode": mode,
            }
        cap_json.setdefault("meta", {})["variant"] = {"fold": fold, "mode": mode}
        (dst_manifest / "captures" / f"{name}.json").write_text(
            json.dumps(cap_json, ensure_ascii=False, indent=2), encoding="utf-8")

    env = json.loads((src_manifest / "env.json").read_text(encoding="utf-8"))
    env.setdefault("constants", {})["SPLIT_MODE"] = mode
    env["variant"] = {"fold": fold, "mode": mode,
                      "note": "re-sliced from existing tensors by make_variants.py"}
    (dst_manifest / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2),
                                           encoding="utf-8")

    splits_doc = {
        "_note": f"{mode} 变体 {fold}：由既有张量重排，未重跑预处理",
        "split_mode": mode, "window_T": win,
        "stride": {"train": st_train, "test": st_test},
        "captures": {name: {"n_frames": sum(b - a for rs in sr.values() for a, b in rs),
                            "split_mode": mode, "ranges": sr,
                            "summary": {"variant": fold}}
                     for name, sr in by_capture.items()},
    }
    (dst_manifest / "splits.json").write_text(
        json.dumps(splits_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"fold": fold, "captures": cap_count, "windows": win_sum, "root": str(dst_root)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="由既有张量生成 LOSO/LOCO 变体")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent
                                            / "training" / "configs" / "train.yaml"))
    ap.add_argument("--dataset", required=True, choices=["ctt15", "road", "hcrl_canfd"])
    ap.add_argument("--mode", required=True, choices=["loso", "loco"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default=None, help="逗号分隔；默认全部")
    ap.add_argument("--copy", action="store_true", help="复制而非硬链接（跨盘时用）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", dest="overrides", action="append", default=None)
    args = ap.parse_args(argv)

    cfg = C.load(args.config, C.parse_overrides(args.overrides))
    setup_logging(str(cfg["runtime"]["log_level"]), None)

    ddir = str(cfg["paths"]["dataset_dirs"][args.dataset])
    src_manifest = (Path(cfg["_resolved"]["tensors_root"]) / ddir
                    / str(cfg["paths"]["manifest_subdir"]))
    splits = json.loads((src_manifest / "splits.json").read_text(encoding="utf-8"))
    win = int(splits.get("window_T") or cfg["data"]["window"]["T"])
    st_train = int(cfg["data"]["window"]["stride"]["train"])
    st_test = int(cfg["data"]["window"]["stride"]["test"])

    captures: dict[str, dict] = {}
    for name, node in splits["captures"].items():
        ranges = {sp: [(int(a), int(b)) for a, b in (node.get("ranges") or {}).get(sp) or []]
                  for sp in SPLITS}
        captures[name] = {"ranges_abs": ranges,
                          "n_frames": int(node.get("n_frames") or
                                          sum(b - a for rs in ranges.values() for a, b in rs))}

    groups: dict[str, list[str]] = {}
    for name in captures:
        groups.setdefault(group_of(args.dataset, name), []).append(name)
    folds = sorted(groups) if not args.folds else [f.strip() for f in args.folds.split(",")]
    for f in folds:
        if f not in groups:
            raise SystemExit(f"未知折 {f}；可选：{', '.join(sorted(groups))}")

    log.info("语料 %s，模式 %s：%d 个 capture，%d 折（%s）", args.dataset, args.mode,
             len(captures), len(folds), ", ".join(folds))

    out_root = Path(args.out)
    results = []
    for fold in folds:
        entries = plan_entries(args.dataset, args.mode, captures, fold, groups)
        if args.dry_run:
            cap = {sp: sum(1 for e in entries if e["dst"] == sp) for sp in SPLITS}
            wn = {sp: sum(n_windows_ranges(e["ranges"], sp, win, st_train, st_test)
                          for e in entries if e["dst"] == sp) for sp in SPLITS}
            log.info("[dry] %s：captures=%s windows=%s", fold, cap, wn)
            results.append({"fold": fold, "captures": cap, "windows": wn})
        else:
            res = build_fold(cfg, args.dataset, args.mode, fold, entries, out_root=out_root,
                             copy=args.copy, win=win, st_train=st_train, st_test=st_test)
            log.info("[built] %s → %s（captures=%s windows=%s）", fold, res["root"],
                     res["captures"], res["windows"])
            results.append(res)

    if not args.dry_run:
        idx = out_root / f"{args.dataset}_{args.mode}_index.json"
        idx.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("索引：%s", idx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
