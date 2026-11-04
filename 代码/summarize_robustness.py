#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
summarize_robustness.py —— 把 E1 的逐模型结果汇总成一张表
================================================================================
输入：`outputs/audit/robustness_<dataset>_<model>.csv`（由 run_robustness.py 生成）
输出：
  · `outputs/audit/robustness_summary.csv` —— 长表（dataset, model, condition, MCC, ΔMCC, n）
  · `outputs/audit/robustness_summary.md`  —— 宽表（每种扰动一行模型，便于贴进论文/大纲）

用法：python 代码/summarize_robustness.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

AUDIT = Path("outputs/audit")


def read(p: Path) -> list[dict]:
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    files = sorted(AUDIT.glob("robustness_*.csv"))
    files = [f for f in files if f.name != "robustness_summary.csv"]
    if not files:
        raise SystemExit("没有 robustness_*.csv；先跑 run_robustness.py")

    rows: list[dict] = []
    for f in files:
        rows.extend(read(f))

    long_path = AUDIT / "robustness_summary.csv"
    with long_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 宽表：每 (dataset, condition) 一段，列为模型
    cell: dict[tuple, dict[str, float]] = defaultdict(dict)
    nn: dict[tuple, dict[str, int]] = defaultdict(dict)
    conds: list[str] = []
    for r in rows:
        key = (r["dataset"], r["condition"])
        cell[key][r["model"]] = float(r["d_mcc"])
        nn[key][r["model"]] = int(r["n"])
        if r["condition"] not in conds:
            conds.append(r["condition"])

    models = sorted({r["model"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})

    lines = ["# E1 输入扰动敏感性（ΔMCC，单位 pt，相对 clean）", "",
             "阈值不重定（用 val 标定的 τ）；同一批 capture；只含仅用 v1 的模型。", ""]
    for ds in datasets:
        lines += [f"## {ds}", "",
                  "| 扰动 | " + " | ".join(models) + " | 窗口数（clean 比例） |",
                  "|---|" + "---|" * len(models) + "---|"]
        base_n = nn.get((ds, "clean"), {})
        for c in conds:
            if c == "clean":
                continue
            vals = cell.get((ds, c), {})
            cells = [f"{vals[m]*100:+.2f}" if m in vals else "—" for m in models]
            ratio = ""
            m0 = models[0]
            if m0 in nn.get((ds, c), {}) and base_n.get(m0):
                ratio = f"{nn[(ds, c)][m0] / base_n[m0]:.2f}×"
            lines.append(f"| {c} | " + " | ".join(cells) + f" | {ratio} |")
        lines.append("")
    (AUDIT / "robustness_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {long_path} 与 {AUDIT/'robustness_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
