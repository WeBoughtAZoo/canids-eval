# -*- coding: utf-8 -*-
"""
run_split_sensitivity.py —— 切分方式敏感性对照（v6 待确认项 T4）
================================================================================
把「块化交错」与「capture 级留出」两批 ROAD 结果并排，输出一张决策表。

用法
----
  python 代码/run_split_sensitivity.py \
      --base-run-dir outputs/training/main/road \
      --alt-run-dir  outputs/training/road_cap_v2/road \
      --out outputs/training/road_cap_v2/sensitivity.csv

口径
----
· 只比 test 段的窗口级指标；两批的编码、统计量、阈值协议**完全相同**，
  唯一变量是切分粒度，因此差异可干净归因；
· 同时给出「test 段有多少攻击族在 train 中缺席」——这是 capture 级留出的直接代价，
  也是判断"能否承担全面切换"的关键；
· 结果**仅供决策**，n=1 时按 v5 §5.5 禁检纪律，不做显著性检验。

作者：小巴（WorkBuddy）    日期：2026-10-28
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.utils import get_logger, write_csv  # noqa: E402

log = get_logger("sensitivity")

METRICS = ("mcc", "auc_pr_norm", "recall", "fpr", "prevalence", "n_pos")


def _read(run: Path, split: str = "test") -> dict | None:
    p = run / f"metrics_{split}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _zero_shot(run: Path) -> dict:
    """test 段中 zero-shot（train 未见过）的攻击族与占比。"""
    fam = run / "per_family_test.csv"
    out = {"n_families": 0, "n_zero_shot": 0, "zero_shot_families": ""}
    if not fam.exists():
        return out
    import csv
    rows = list(csv.DictReader(fam.open(encoding="utf-8-sig")))
    zs = [r.get("family", "?") for r in rows if str(r.get("zero_shot", "")).lower() == "true"]
    out.update({"n_families": len(rows), "n_zero_shot": len(zs),
                "zero_shot_families": ";".join(zs)})
    return out


def _discover(root: Path) -> list[tuple[str, str, str, Path]]:
    """返回 (model, variant, seed, run_dir)。"""
    out = []
    for man in sorted(root.rglob("run_manifest.json")):
        try:
            m = json.loads(man.read_text(encoding="utf-8"))
        except Exception:                                    # noqa: BLE001
            continue
        out.append((str(m.get("model", "?")), str(m.get("variant", "?")),
                    str(m.get("seed", "?")), man.parent))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="切分方式敏感性对照")
    ap.add_argument("--base-run-dir", required=True, help="块化交错的批根目录（含 road/<model>/<variant>/seedN）")
    ap.add_argument("--alt-run-dir", required=True, help="capture 级留出的批根目录")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    base = {f"{m}/{v}/seed{s}": d for m, v, s, d in _discover(Path(args.base_run_dir))}
    alt = {f"{m}/{v}/seed{s}": d for m, v, s, d in _discover(Path(args.alt_run_dir))}
    if not alt:
        log.error("capture 级留出批次为空：%s", args.alt_run_dir)
        return 1

    rows = []
    for key in sorted(alt):
        a_run, b_run = alt[key], base.get(key)
        a = _read(a_run)
        b = _read(b_run) if b_run else None
        model, variant, seed = key.split("/")
        row = {"model": model, "variant": variant, "seed": seed}
        for tag, m in (("holdout", a), ("block", b)):
            for k in METRICS:
                row[f"{tag}_{k}"] = (m.get(k) if m else None)
        if a and b:
            row["d_mcc"] = (a.get("mcc") or 0.0) - (b.get("mcc") or 0.0)
            row["d_recall"] = (a.get("recall") or 0.0) - (b.get("recall") or 0.0)
        row.update({"holdout_" + k: v for k, v in _zero_shot(a_run).items()})
        rows.append(row)
        log.info("%-34s holdout MCC=%s | block MCC=%s | Δ=%s",
                 key,
                 f"{a.get('mcc'):.4f}" if a else "—",
                 f"{b.get('mcc'):.4f}" if b else "—",
                 f"{row.get('d_mcc'):+.4f}" if b else "—")

    out = Path(args.out) if args.out else Path(args.alt_run_dir).parent / "sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, rows)
    log.info("对照表写入 %s（%d 行）", out, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
