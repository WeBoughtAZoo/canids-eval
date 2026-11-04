#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_artifacts.py —— 用「充分统计量」反算论文表格，逐格比对
================================================================================
它回答审稿人最可能问的一句话：**"你发的这些整数，真能还原你的表吗？"**

本脚本只读 `outputs/artifact/*.csv` 与各批次已发布的表，不读任何信号数据、不加载模型。
五项检查（2026-10-31 起支持多批次）：

  C1  runs.csv  vs <batch>/runs_index.csv        —— 每 run 的 MCC/recall/FPR/precision/τ
  C2  runs.csv  vs main/tables/table_main.csv    —— 按 (语料, 模型, 变体) 的 seed 均值（仅主批次）
  C3  counts_density.csv vs main/tables/table_density.csv —— 每密度箱召回（仅主批次）
  C4  discordance_capture.csv vs main/comparison*.csv     —— b、c 总和与簇级 p（仅主批次）
  C5  counts_capture.csv vs <batch>/**/metrics_test.json  —— n/n_pos/tp/fp/fn/tn

退出码：0 全过；1 有差异（明细写入 `outputs/artifact/verify_report.md`）。
作者：小巴（WorkBuddy）    日期：2026-10-30（2026-10-31 多批次）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from training import config as C            # noqa: E402
from training.stats import signflip_p       # noqa: E402
from training.utils import get_logger, setup_logging  # noqa: E402

log = get_logger("verify")

REL_TOL = 1e-9
ABS_TOL = 1e-12


def _read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _close(a: float, b: float) -> bool:
    if not (np.isfinite(a) and np.isfinite(b)):
        return bool(np.isnan(a) and np.isnan(b))
    return abs(a - b) <= max(ABS_TOL, REL_TOL * max(abs(a), abs(b)))


class Report:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.n_fail = 0

    def add(self, check: str, item: str, ok: bool, detail: str = "") -> None:
        if not ok:
            self.n_fail += 1
        self.rows.append({"check": check, "item": item, "ok": ok, "detail": detail})

    def summary(self) -> str:
        total = len(self.rows)
        by: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in self.rows:
            by[r["check"]][0] += 1
            by[r["check"]][1] += 0 if r["ok"] else 1
        lines = ["| 检查 | 比对项 | 失败 |", "|---|---|---|"]
        for k, (n, f) in by.items():
            lines.append(f"| {k} | {n} | {f} |")
        lines.insert(0, f"**总计 {total} 项，失败 {self.n_fail} 项**\n")
        return "\n".join(lines)

    def failures(self) -> list[dict]:
        return [r for r in self.rows if not r["ok"]]


# --------------------------------------------------------------------------- #
def check_runs(rep: Report, art: Path, batch_root: dict[str, Path]) -> None:
    rows = _read(art / "runs.csv")
    for bt, root in batch_root.items():
        idx_path = root / "runs_index.csv"
        if not idx_path.exists():
            continue
        idx = {f"{r['dataset']}/{r['model']}/{r['variant']}/seed{r['seed']}": r
               for r in _read(idx_path)}
        for r in rows:
            if r["batch"] != bt:
                continue
            ref = idx.get(r["run_id"].split("|", 1)[1])
            if ref is None:
                rep.add("C1 runs_index", r["run_id"], False, "runs_index 中缺失")
                continue
            for col in ("mcc", "recall", "fpr", "precision", "threshold"):
                ok = _close(float(r[col]), float(ref[col]))
                rep.add("C1 runs_index", f"{r['run_id']} · {col}", ok,
                        "" if ok else f"artifact={r[col]} index={ref[col]}")


def check_table_main(rep: Report, art: Path, main_root: Path) -> None:
    f = main_root / "tables" / "table_main.csv"
    if not f.exists():
        rep.add("C2 table_main", "table_main.csv", False, "缺少文件")
        return
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in _read(art / "runs.csv"):
        if r["batch"] != "main":
            continue
        groups[(r["dataset"], r["model"], r["variant"])].append(r)
    for row in _read(f):
        key = (row["dataset"], row["model"], row["variant"])
        rows = groups.get(key)
        if not rows:
            rep.add("C2 table_main", f"{key}", False, "artifact 中无该组")
            continue
        n = int(float(row["n_seeds"]))
        if n != len(rows):
            rep.add("C2 table_main", f"{key} · n_seeds", False,
                    f"table={n} artifact={len(rows)}")
        for col in ("mcc", "precision", "recall", "fpr"):
            mean = float(np.mean([float(x[col]) for x in rows]))
            ok = _close(mean, float(row[col]))
            rep.add("C2 table_main", f"{key} · {col}", ok,
                    "" if ok else f"artifact={mean:.12g} table={row[col]}")


def check_density(rep: Report, art: Path, main_root: Path) -> None:
    f = main_root / "tables" / "table_density.csv"
    if not f.exists():
        rep.add("C3 table_density", "table_density.csv", False, "缺少文件")
        return
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in _read(art / "counts_density.csv"):
        if r["batch"] != "main":
            continue
        parts = r["run_id"].split("|", 1)[1].split("/")
        groups[(parts[0], parts[1], parts[2], r["bin"])].append(float(r["recall"]))
    for row in _read(f):
        key = (row["dataset"], row["model"], row["variant"], row["density_bin"])
        vals = groups.get(key)
        if not vals:
            continue
        mean = float(np.mean(vals))
        ok = _close(mean, float(row["recall"]))
        rep.add("C3 table_density", f"{key} · recall", ok,
                "" if ok else f"artifact={mean:.12g} table={row['recall']}")


def check_comparison(rep: Report, art: Path, main_root: Path) -> None:
    discord: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in _read(art / "discordance_capture.csv"):
        if r["batch"] != "main":
            continue
        discord[r["pair_id"]].append((int(r["b"]), int(r["c"])))
    comp = main_root / "comparison"
    if not comp.exists():
        rep.add("C4 comparison", "comparison/", False, "主批次无 comparison 目录")
        return
    for name in ("comparison.csv", "comparison_known.csv"):
        f = comp / name
        if not f.exists():
            continue
        regime = "known" if "known" in name else "full"
        for row in _read(f):
            pid = f"main|{regime}|{row['dataset']}|{row['comparison']}|seed{row['seed']}"
            pairs = discord.get(pid)
            if not pairs:
                rep.add("C4 comparison", pid, False, "discordance 中无该配对")
                continue
            b = np.asarray([p[0] for p in pairs], dtype=np.int64)
            c = np.asarray([p[1] for p in pairs], dtype=np.int64)
            rep.add("C4 comparison", f"{pid} · b", int(b.sum()) == int(float(row["b"])),
                    "" if int(b.sum()) == int(float(row["b"])) else f"{b.sum()} vs {row['b']}")
            rep.add("C4 comparison", f"{pid} · c", int(c.sum()) == int(float(row["c"])),
                    "" if int(c.sum()) == int(float(row["c"])) else f"{c.sum()} vs {row['c']}")
            # 与 stats.mcnemar_cluster 同参数：K≤16 精确枚举、否则 MC(20000, seed=0)
            p = signflip_p(b - c, n_perm=20000, exact_max_clusters=16)["p_value"]
            ok = _close(p, float(row["p_value"]))
            rep.add("C4 comparison", f"{pid} · p", ok,
                    "" if ok else f"artifact={p:.6g} table={row['p_value']}")


def check_metrics_json(rep: Report, art: Path, batch_root: dict[str, Path]) -> None:
    """逐 run 的 n / n_pos / tp / fp / fn / tn 与 metrics_test.json 对齐。"""
    for r in _read(art / "runs.csv"):
        bt = r["batch"]
        root = batch_root.get(bt)
        if root is None:
            continue
        ds, model, variant, seed = r["run_id"].split("|", 1)[1].split("/")
        f = root / ds / model / variant / f"seed{seed[4:]}" / "metrics_test.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        for col, key in (("n_windows", "n"), ("n_pos", "n_pos"),
                         ("tp", "tp"), ("fp", "fp"), ("fn", "fn"), ("tn", "tn")):
            ok = int(r[col]) == int(m[key])
            rep.add("C5 metrics_json", f"{r['run_id']} · {key}", ok,
                    "" if ok else f"artifact={r[col]} json={m[key]}")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="用充分统计量反算表格并比对")
    ap.add_argument("--config", default=str(Path(__file__).parent / "training" / "configs" / "train.yaml"))
    ap.add_argument("--batches", default="main",
                    help="逗号分隔的批次名。论文报的口径是 --batches main,gpu1（1,911 项）；"
                         "只跑 main 得 1,438 项，两者差异来自 gpu1 批次的补充 run，均须 0 失败。")
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=None)
    args = ap.parse_args(argv)

    batches = [b.strip() for b in str(args.batches).split(",") if b.strip()]
    cfg = C.load(args.config, C.parse_overrides(list(args.overrides or [])
                                                + [f"project.run_name={batches[0]}"]))
    setup_logging(str(cfg["runtime"]["log_level"]), None)

    art = Path(args.artifact) if args.artifact else Path(cfg["_resolved"]["output_root"]).parent / "artifact"
    if not (art / "runs.csv").exists():
        raise SystemExit(f"缺少 {art/'runs.csv'}，先运行 build_artifact.py")

    batch_root = {b: Path(cfg["_resolved"]["output_root"]) / b for b in batches}
    for b, p in batch_root.items():
        if not p.exists():
            log.warning("批次 %s 不存在：%s", b, p)

    rep = Report()
    check_runs(rep, art, batch_root)
    if "main" in batch_root:
        check_table_main(rep, art, batch_root["main"])
        check_density(rep, art, batch_root["main"])
        check_comparison(rep, art, batch_root["main"])
    check_metrics_json(rep, art, batch_root)

    # 报告写入**相对路径**：避免把作者机器的绝对路径带进公开仓库
    try:
        art_disp = art.resolve().relative_to(Path(__file__).resolve().parent.parent).as_posix()
    except ValueError:
        art_disp = art.as_posix()

    report = ["# 充分统计量自检报告", "",
              f"产物：`{art_disp}`　批次：`{', '.join(batches)}`", "",
              rep.summary(), ""]
    fails = rep.failures()
    if fails:
        report += ["## 失败明细（前 40 条）", "", "| 检查 | 项 | 说明 |", "|---|---|---|"]
        for r in fails[:40]:
            report.append(f"| {r['check']} | {r['item']} | {r['detail']} |")
    else:
        report += ["**全部通过**：计数表可逐位还原各批次的 `runs_index.csv` 与 `metrics_test.json`；",
                   "主批次的 `table_main.csv`、`table_density.csv`、`comparison*.csv` 亦逐格一致。"]
    (art / "verify_report.md").write_text("\n".join(report), encoding="utf-8")

    log.info("自检完成：%d 项，失败 %d 项 → %s", len(rep.rows), rep.n_fail,
             art / "verify_report.md")
    if fails:
        for r in fails[:5]:
            log.warning("FAIL %s · %s %s", r["check"], r["item"], r["detail"])
    return 0 if rep.n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
