#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_regime.py —— 按评测口径重算指标（只读 predictions_*.npz，**不重训**）
================================================================================
背景
----
`metrics_*.json` 只记录**单一口径**（全部正样本）。v5 需要两个额外口径：

1. **known-attack**（剔除 zero-shot 家族）
   依据 v5 §4.5：CTT 的 `unknown_attack`（label_map 索引 15）承载 masquerade，
   训练集中完全缺席。把它与已知攻击混在一起算综合 MCC，会把"学会了已知攻击"
   与"能识别未知攻击"两种能力混为一谈。→ 过滤 `label==1 & y_type==unknown` 后重算。

2. **nosuppress**（剔除 suppress 采集）的敏感性检查
   依据 v5 §4.5 声明③：`test_05_suppress` 的帧**全部标为 normal**（攻击是"报文缺失"，
   无逐帧标签），但语义上确实是攻击。任何模型若正确检出反被计为误报，
   使 CTT 的表观 FPR/MCC 系统性偏悲观。

产物（写入 run 目录，不动原有文件）
----------------------------------
  metrics_{val,test}_known.json        mode=known
  metrics_test_nosuppress.json         mode=nosuppress

用法
----
  # 全批重算两个口径
  python 代码/run_regime.py --mode known      --all --set project.run_name=main
  python 代码/run_regime.py --mode nosuppress --all --set project.run_name=main

  # 单个 run
  python 代码/run_regime.py --mode known --run-dir "outputs/training/main/ctt15/fdtcn/full/seed42"

说明
----
- 阈值 τ 沿用 `metrics_*.json` 里训练期标定的值，**不在 test 上重选**；
- 操作点 τ（Recall@FPR）在 val 的**同口径子集**上标定，再用于 test —— 与主路径一致；
- 若某口径下不存在被剔除的正样本（如 HCRL/ROAD 无 zero-shot 类），**不写文件**，
  下游表格因此天然只包含有差异的语料。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training import config as C          # noqa: E402
from training import figures as F         # noqa: E402
from training import metrics as M         # noqa: E402
from training.utils import get_logger, read_json, setup_logging, write_json  # noqa: E402

log = get_logger("regime")


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _unknown_index(run_dir: Path) -> int | None:
    """从 label_map.json 取 unknown 类的下标；无则返回 None。"""
    p = run_dir / "label_map.json"
    if not p.exists():
        return None
    lm = (read_json(p) or {}).get("label_map", {})
    for name, idx in lm.items():
        if str(name).endswith("unknown_attack"):
            return int(idx)
    return None


def _threshold(run_dir: Path, split: str) -> float | None:
    p = run_dir / f"metrics_{split}.json"
    if not p.exists():
        return None
    v = (read_json(p) or {}).get("threshold")
    return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None


def _load_split(run_dir: Path, split: str) -> dict[str, np.ndarray] | None:
    p = run_dir / f"predictions_{split}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {k: d[k] for k in d.files}


def _operating_points(y: np.ndarray, p: np.ndarray) -> dict[float, float]:
    out: dict[float, float] = {}
    for target in (0.001, 0.01):
        tau, _ = M.recall_at_fpr(y, p, target)
        if np.isfinite(tau):
            out[target] = float(tau)
    return out


def _record(run_dir: Path, name: str, *, y: np.ndarray, prob: np.ndarray,
            threshold: float, op_taus: Mapping[float, float], op_source: str,
            split: str, n_windows: int, extra: Mapping[str, Any],
            report: Sequence[str] | None = None) -> dict[str, Any]:
    main = M.binary_metrics(y, prob, threshold, op_taus=op_taus)
    summary = M.summarize_split(main, split=split, n_windows=n_windows,
                                bootstrap=None, report=report)
    summary["op_source"] = op_source
    summary["operating_points"] = {
        "fpr_0p1": (float(op_taus[0.001]) if 0.001 in op_taus else None),
        "fpr_1": (float(op_taus[0.01]) if 0.01 in op_taus else None),
        "source": op_source,
    }
    summary.update(dict(extra))
    write_json(run_dir / f"metrics_{name}.json", summary)
    return summary


def _suppress_caps(cfg: Mapping[str, Any], dataset: str) -> set[int]:
    """返回 test split 中 attack_type=='suppress' 的 capture 下标。

    下标口径与 `WindowDataset.specs` 一致：**按 capture 名排序**后取位置。
    （已用窗口数逐 capture 校验：224/224 与 floor((n−64)/64)+1 严格相符。）
    """
    roots = (cfg.get("_resolved") or {}).get("dataset_roots") or {}
    root = roots.get(dataset)
    if not root:
        return set()
    man = Path(root) / str(cfg["paths"]["manifest_subdir"]) / "dataset_stats.csv"
    if not man.exists():
        return set()
    with open(man, encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r.get("n_test") or 0) > 0]
    rows.sort(key=lambda r: r["capture"])
    return {i for i, r in enumerate(rows) if (r.get("attack_type") or "").strip() == "suppress"}


# --------------------------------------------------------------------------- #
# 两种口径
# --------------------------------------------------------------------------- #
def run_known(run_dir: Path) -> str:
    uid = _unknown_index(run_dir)
    if uid is None:
        return "跳过（标签表无 unknown 类）"
    tau = _threshold(run_dir, "test")
    if tau is None:
        return "跳过（缺 metrics_test.json 的阈值）"

    op: dict[float, float] = {}
    n_drop_val = n_drop_test = 0
    for split in ("val", "test"):
        d = _load_split(run_dir, split)
        if d is None:
            continue
        y, prob = d["label"], d["prob"]
        drop = (y == 1) & (d["y_type"] == uid)
        keep = ~drop
        if split == "val":
            n_drop_val = int(drop.sum())
            op = _operating_points(y[keep], prob[keep])
            _record(run_dir, f"{split}_known", y=y[keep], prob=prob[keep],
                    threshold=tau, op_taus=op, op_source="calibrated_on_val",
                    split=split, n_windows=int(keep.sum()),
                    extra={"regime": "known_attack",
                           "excluded_positive_windows": n_drop_val})
        else:
            n_drop_test = int(drop.sum())
            _record(run_dir, f"{split}_known", y=y[keep], prob=prob[keep],
                    threshold=tau, op_taus=op, op_source="inherited_from_val",
                    split=split, n_windows=int(keep.sum()),
                    extra={"regime": "known_attack",
                           "excluded_positive_windows": n_drop_test})

    if n_drop_test == 0 and n_drop_val == 0:
        for split in ("val", "test"):
            p = run_dir / f"metrics_{split}_known.json"
            if p.exists():
                p.unlink()
        return "跳过（无 zero-shot 正样本，known ≡ full）"
    return f"known：剔除 val {n_drop_val:,} / test {n_drop_test:,} 个正窗口"


def run_nosuppress(cfg: Mapping[str, Any], run_dir: Path, dataset: str) -> str:
    sup = _suppress_caps(cfg, dataset)
    if not sup:
        return "跳过（无 suppress 采集）"
    d = _load_split(run_dir, "test")
    tau = _threshold(run_dir, "test")
    if d is None or tau is None:
        return "跳过（缺 predictions_test.npz 或阈值）"

    cap = d["cap"]
    keep = ~np.isin(cap, np.fromiter(sup, dtype=cap.dtype))
    n_drop = int((~keep).sum())
    if n_drop == 0:
        return "跳过（suppress 窗口不在 test 中）"

    # 操作点 τ 取自**完整 val**（val 不含 suppress，故口径不变）
    op: dict[float, float] = {}
    dv = _load_split(run_dir, "val")
    if dv is not None:
        op = _operating_points(dv["label"], dv["prob"])
    _record(run_dir, "test_nosuppress", y=d["label"][keep], prob=d["prob"][keep],
            threshold=tau, op_taus=op,
            op_source="inherited_from_val" if op else "unavailable_self_calibrated",
            split="test", n_windows=int(keep.sum()),
            extra={"regime": "test_without_suppress",
                   "removed_captures": len(sup),
                   "removed_windows": n_drop})
    return f"nosuppress：剔除 {len(sup)} 个 suppress 采集 / {n_drop:,} 个窗口"


# --------------------------------------------------------------------------- #
# 操作点回填（Recall@FPR 的 τ 在 val 标定 → test 原样复用）
# --------------------------------------------------------------------------- #
def run_operating(run_dir: Path) -> str:
    """只重算并**合并**操作点相关字段，不动 mcc/f1 等既有值。

    背景：`metrics_*.json` 里的 `recall_at_fpr_*` 由旧实现产出，其 τ 是**在被评测
    split 自身**的负样本分位上取的（操作点泄漏）。修复后 τ 应在 val 上标定，
    test 原样复用，并同时报该 τ 在 test 上达到的**实际 FPR**。
    本模式直接从 `predictions_{val,test}.npz` 复算，无需模型与数据集，秒级完成。
    """
    dv = _load_split(run_dir, "val")
    dt = _load_split(run_dir, "test")
    if dv is None or dt is None:
        return "跳过（缺 predictions_{val,test}.npz）"
    op = _operating_points(dv["label"], dv["prob"])
    if not op:
        return "跳过（val 缺正/负样本，无法标定操作点）"

    changed = []
    for split, data, src in (("val", dv, "calibrated_on_val"),
                             ("test", dt, "inherited_from_val")):
        path = run_dir / f"metrics_{split}.json"
        if not path.exists():
            continue
        tau = _threshold(run_dir, split)
        if tau is None:
            continue
        main = M.binary_metrics(data["label"], data["prob"], tau, op_taus=op)
        cur = read_json(path) or {}
        before = cur.get("recall_at_fpr_1")
        for k in ("recall_at_fpr_0p1", "recall_at_fpr_1", "tau_at_fpr_0p1",
                  "tau_at_fpr_1", "fpr_at_fpr_0p1", "fpr_at_fpr_1"):
            cur[k] = main.get(k)
        cur["op_source"] = src
        cur["operating_points"] = {
            "fpr_0p1": (float(op[0.001]) if 0.001 in op else None),
            "fpr_1": (float(op[0.01]) if 0.01 in op else None),
            "source": src,
        }
        write_json(path, cur)
        changed.append(f"{split}: R@1% {before:.4f}→{main['recall_at_fpr_1']:.4f} "
                       f"实际FPR {main['fpr_at_fpr_1']:.4%}")
    return "；".join(changed) if changed else "跳过（无 metrics_*.json）"


# --------------------------------------------------------------------------- #
# 静态效率指标刷新（不推理，只重算 n_params / MACs / 感受野）
# --------------------------------------------------------------------------- #
def refresh_stats(base_cfg: Mapping[str, Any], run_dir: Path) -> str:
    """只重算静态效率指标并回写 `run_manifest.model_stats`，不做任何推理。

    为什么需要：`model_stats.macs` 是**训练时**写入 manifest 的，历史上的 run 保存的是
    旧 `count_macs` 的结果（LSTM/Transformer 因漏计循环与 attention 而偏小两个数量级）。
    修复计数器后若不刷新，T12 / fig09 仍会读到旧值。
    本模式只加载 checkpoint 建模型 + 跑一次空前向，**秒级**完成，代价远低于 `run_eval --all`。
    """
    from training.artifacts import RunContext
    from training.models import build_model, count_parameters, model_inputs
    from training.models.common import count_macs

    snap = run_dir / "config_resolved.yaml"
    if not snap.exists():
        return "跳过（缺 config_resolved.yaml）"
    cfg = C.load_snapshot(snap, base_cfg)

    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "model_final.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.pt"
    if not ckpt_path.exists():
        return "跳过（缺 checkpoint）"

    ckpt = RunContext.load_checkpoint(ckpt_path, map_location="cpu")
    name = ckpt["model_name"]
    model = build_model(name, ckpt["model_cfg"], n_classes=int(ckpt["n_classes"]),
                        window_T=int(ckpt["window_T"]))
    ch = int(ckpt["model_cfg"].get("in_channels", 16))
    macs = int(count_macs(model, needs=model_inputs(name),
                          window_T=int(cfg["data"]["window"]["T"]),
                          v1_channels=ch, v2_channels=ch))

    man_p = run_dir / "run_manifest.json"
    man = read_json(man_p) or {}
    ms = dict(man.get("model_stats") or {})
    old = ms.get("macs")
    ms.update({"n_params": int(count_parameters(model)),
               "n_params_trainable": int(count_parameters(model)),
               "receptive_field": int(getattr(model, "receptive_field", 0)),
               "macs": macs})
    man["model_stats"] = ms
    write_json(man_p, man)
    return f"macs {old} → {macs}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按评测口径重算指标（不重训）")
    ap.add_argument("--config", default=str(Path(__file__).parent / "training" / "configs" / "train.yaml"))
    ap.add_argument("--mode", required=True,
                    choices=("known", "nosuppress", "operating", "stats"))
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--set", dest="overrides", action="append", default=None)
    args = ap.parse_args(argv)

    if not args.run_dir and not args.all:
        ap.error("需要 --run-dir 或 --all")

    base = C.load(args.config, C.parse_overrides(args.overrides))
    setup_logging(str(base["runtime"]["log_level"]), None)

    if args.run_dir:
        run_dirs = [Path(args.run_dir).resolve()]
    else:
        run_dirs = [r.dir for r in F.discover_runs(base["_resolved"]["output_root"], base)]
        log.info("发现 %d 个 run", len(run_dirs))

    done = skipped = failed = 0
    for rd in run_dirs:
        try:
            man = read_json(rd / "run_manifest.json") or {}
            dataset = str(man.get("dataset") or "")
            if args.mode == "known":
                msg = run_known(rd)
            elif args.mode == "nosuppress":
                msg = run_nosuppress(base, rd, dataset)
            elif args.mode == "operating":
                msg = run_operating(rd)
            else:
                msg = refresh_stats(base, rd)
            if msg.startswith("跳过"):
                skipped += 1
                log.info("[%s/%s] %s", dataset, rd.name, msg)
            else:
                done += 1
                log.info("[%s/%s] %s", dataset, rd.name, msg)
        except Exception as e:  # noqa: BLE001
            failed += 1
            log.exception("重算失败 %s：%s", rd, e)

    log.info("mode=%s 完成：写入 %d，跳过 %d，失败 %d", args.mode, done, skipped, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
