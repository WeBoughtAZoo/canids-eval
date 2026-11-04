#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_audit.py —— 2026-10-30 审计整改的计算入口（P0/P1 中"可算项"）
================================================================================
本脚本只做**不重训**的复算：输入是既有 run 的 `predictions_*.npz` 与
`outputs/pi5_deploy/summary*.csv`，因此在 PC 上几分钟内跑完，且可逐位复核。

四个子命令（参数全部来自 `train.yaml` 的 `audit:` 段，脚本内无业务常量）
--------------------------------------------------------------------------
  setwise  F3：把 full 与 Ab-12（载荷通道全删）的 ΔMCC **按录制组（SET）拆开**，
           回答"+15.2 pt 是载荷无用，还是载荷编码了 SET 身份（捷径）"。
  nested   D4-1：组→capture→窗口两层符号翻转 + 两阶段自助，报最粗粒度下的 p 与
           ΔMCC 置信区间，并披露结构性下限 p_min。
  openset  F4：用 unknown-vs-normal AUROC / TPR@FPR_normal 替代"full−known 差值"，
           并把后者改名为 regime-shift sensitivity。
  deploy   F2：钉死单一操作点（batch=1 / th=1 / FP32 / T4⁺），从 summary_v4.csv
           重算 R、N_bus、mJ，并列 R1 差异原因。

用法
----
  python 代码/run_audit.py                 # 全部
  python 代码/run_audit.py setwise nested
  python 代码/run_audit.py openset --dataset ctt15

产物：outputs/audit/{summary.md, setwise_*.csv, nested_inference.csv,
                     openset_*.csv, deploy_canonical.csv, deploy_ratios.csv}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from training import config as C           # noqa: E402
from training import groups as G           # noqa: E402
from training import nested as N           # noqa: E402
from training import openset as O          # noqa: E402
from training.utils import get_logger, setup_logging, write_csv  # noqa: E402

log = get_logger("audit")


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _batch_cfg(config_path: str, batch: str, extra: Iterable[str] | None = None) -> dict:
    overrides = list(extra or []) + [f"project.run_name={batch}"]
    return C.load(config_path, C.parse_overrides(overrides))


def _out_dir(cfg: dict) -> Path:
    """outputs/<audit.out_subdir>（与 outputs/training 同级，不混进 run 产物）。"""
    p = Path(cfg["_resolved"]["output_root"]).parent / str(cfg["audit"]["out_subdir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_run(run_dir: Path, split: str = "test") -> dict[str, np.ndarray]:
    p = Path(run_dir) / f"predictions_{split}.npz"
    if not p.exists():
        raise FileNotFoundError(f"缺少预测文件：{p}")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _pred(z: dict) -> np.ndarray:
    return (np.asarray(z["prob"]) >= float(z["threshold"])).astype(np.int64)


def _seed_list(cfg: dict, dataset: str, model: str, variant: str) -> list[int]:
    """该 (dataset, model, variant) 实际存在的 seed（目录扫描，不看配置）。"""
    base = C.resolve_run_dir(cfg, dataset, model, variant, 0).parent
    if not base.exists():
        return []
    out: list[int] = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name.startswith("seed"):
            try:
                out.append(int(d.name[4:]))
            except ValueError:
                continue
    return out


def _resolve_pair(cfg: dict, dataset: str, ma: str, va: str, mb: str, vb: str) -> tuple[Path, Path, int] | None:
    seeds = sorted(set(_seed_list(cfg, dataset, ma, va)) & set(_seed_list(cfg, dataset, mb, vb)))
    if not seeds:
        return None
    s = seeds[0]
    return (C.resolve_run_dir(cfg, dataset, ma, va, s),
            C.resolve_run_dir(cfg, dataset, mb, vb, s), s)


def _check_paired(za: dict, zb: dict, tag: str) -> None:
    if not np.array_equal(za["label"], zb["label"]) or not np.array_equal(za["cap"], zb["cap"]):
        raise ValueError(f"{tag}：两 run 的 test 序列不一致，禁止配对检验")


# --------------------------------------------------------------------------- #
# F3 · 按录制组拆分 ΔMCC
# --------------------------------------------------------------------------- #
def run_setwise(cfg: dict, dataset: str | None = None) -> list[dict]:
    node = cfg["audit"]["setwise"]
    dataset = dataset or str(node["dataset"])
    model = str(node["model"])
    va, vb = [str(x) for x in node["variants"]]
    sep = str(cfg["audit"]["group"]["separator"])

    seeds = sorted(set(_seed_list(cfg, dataset, model, va))
                   & set(_seed_list(cfg, dataset, model, vb)))
    if not seeds:
        log.warning("setwise：%s/%s 的 %s 与 %s 没有共同 seed", dataset, model, va, vb)
        return []

    grp = G.capture_groups(cfg, dataset, split="test", sep=sep)
    # cap→组映射单独落盘：外部复现者需要它才能跑 group 层检验（canids_eval compare --groups）
    (_out_dir(cfg) / f"groups_{dataset}_test.json").write_text(
        json.dumps({str(i): str(g) for i, g in enumerate(grp)}, ensure_ascii=False, indent=0),
        encoding="utf-8")
    per_seed: list[list[dict]] = []
    for s in seeds:
        za = _load_run(C.resolve_run_dir(cfg, dataset, model, va, s))
        zb = _load_run(C.resolve_run_dir(cfg, dataset, model, vb, s))
        _check_paired(za, zb, f"setwise seed{s}")
        if s == seeds[0]:
            info = G.verify_alignment(cfg, dataset, za["cap"], split="test")
            log.info("setwise：cap→组 对齐校验通过（%d 簇 / %d 窗口）",
                     info["n_caps"], info["n_windows"])
        ta = N.group_table(za["label"], _pred(za), za["cap"], grp, threshold=float(za["threshold"]))
        tb = N.group_table(zb["label"], _pred(zb), zb["cap"], grp, threshold=float(zb["threshold"]))
        rows = N.merge_group_delta(ta, tb)
        for r in rows:
            r["seed"] = s
        per_seed.append(rows)

    # 逐组跨 seed 汇总（ΔMCC 均值/标准差 + 与均值同向的 seed 数）
    agg: list[dict] = []
    for name in [r["group"] for r in per_seed[0]]:
        vals = np.asarray([next(r["delta_mcc"] for r in rows if r["group"] == name)
                           for rows in per_seed], dtype=np.float64)
        ref = next(r for r in per_seed[0] if r["group"] == name)
        sgn = float(np.sign(vals.mean()))
        agg.append({
            "group": name, "n_clusters": ref["n_clusters"], "n_windows": ref["n_windows"],
            "n_pos": ref["n_pos"], "prevalence": ref["prevalence"],
            "delta_mcc_mean": float(vals.mean()),
            "delta_mcc_std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "delta_mcc_min": float(vals.min()), "delta_mcc_max": float(vals.max()),
            "same_sign_seeds": int((np.sign(vals) == sgn).sum()), "n_seeds": int(vals.size),
            "mcc_full_mean": float(np.mean([next(r["mcc_a"] for r in rows if r["group"] == name)
                                            for rows in per_seed])),
            "mcc_pf_mean": float(np.mean([next(r["mcc_b"] for r in rows if r["group"] == name)
                                          for rows in per_seed])),
            "delta_recall_mean": float(np.mean([next(r["delta_recall"] for r in rows if r["group"] == name)
                                                for rows in per_seed])),
        })

    overall = _overall_delta(cfg, dataset, model, va, vb, seeds)
    out = _out_dir(cfg)
    stem = f"setwise_{dataset}_{va}_vs_{vb}"
    write_csv(out / f"{stem}.csv", agg, list(agg[0].keys()))
    for s, rows in zip(seeds, per_seed):
        write_csv(out / f"{stem}_seed{s}.csv", rows, list(rows[0].keys()))
    write_csv(out / f"{stem}_overall.csv", overall, list(overall[0].keys()))

    # 判读（预登记规则的机械化版本，避免事后挑选叙事）
    d = np.asarray([g["delta_mcc_mean"] for g in agg])
    absv = np.abs(d)
    verdict = {
        "n_groups": int(d.size),
        "n_same_direction": int((np.sign(d) == np.sign(d.mean())).sum()),
        "direction": "negative" if d.mean() < 0 else "positive",
        "magnitude_ratio": float(absv.max() / absv.min()) if absv.min() > 0 else float("inf"),
        "overall_delta_mean": float(np.mean([r["delta_mcc"] for r in overall])),
    }
    log.info("setwise：%d 组 × %d seeds → %s（同向 %d/%d，量级比 %.2f）",
             len(agg), len(seeds), out / f"{stem}.csv",
             verdict["n_same_direction"], verdict["n_groups"], verdict["magnitude_ratio"])
    return [{"variant_a": va, "variant_b": vb, "dataset": dataset, "model": model,
             "seeds": seeds, "groups": agg, "overall": overall, "verdict": verdict}]


def _overall_delta(cfg: dict, dataset: str, model: str, va: str, vb: str,
                   seeds: list[int]) -> list[dict]:
    """全窗口口径的 MCC（两模型各自 τ），用于与主表 / T10 对齐。"""
    rows: list[dict] = []
    for s in seeds:
        za = _load_run(C.resolve_run_dir(cfg, dataset, model, va, s))
        zb = _load_run(C.resolve_run_dir(cfg, dataset, model, vb, s))
        y = np.asarray(za["label"]).astype(np.int64)

        def mcc(p: np.ndarray) -> float:
            tp = float(((y == 1) & (p == 1)).sum())
            fp = float(((y == 0) & (p == 1)).sum())
            fn = float(((y == 1) & (p == 0)).sum())
            tn = float(((y == 0) & (p == 0)).sum())
            return N._mcc(tp, fp, fn, tn)

        ma, mb = mcc(_pred(za)), mcc(_pred(zb))
        rows.append({"seed": s, "mcc_a": ma, "mcc_b": mb, "delta_mcc": ma - mb})
    return rows


# --------------------------------------------------------------------------- #
# D4-1 · 嵌套推断
# --------------------------------------------------------------------------- #
def run_nested(cfg: dict) -> list[dict]:
    node = cfg["audit"]["nested"]
    sep = str(cfg["audit"]["group"]["separator"])
    kw = {"n_boot": int(node["n_boot"]), "n_perm": int(node["n_perm"]),
          "exact_max_groups": int(node["exact_max_groups"]), "seed": int(node["seed"])}

    rows: list[dict] = []
    for pair in node["pairs"]:
        dataset = str(pair["dataset"])
        ma, va = str(pair["model_a"]), str(pair["variant_a"])
        mb, vb = str(pair["model_b"]), str(pair["variant_b"])
        res_pair = _resolve_pair(cfg, dataset, ma, va, mb, vb)
        if res_pair is None:
            log.warning("nested：%s 上 %s/%s vs %s/%s 无共同 seed", dataset, ma, va, mb, vb)
            continue
        da, db, s = res_pair
        za, zb = _load_run(da), _load_run(db)
        _check_paired(za, zb, f"nested {dataset}")
        grp = G.capture_groups(cfg, dataset, split="test", sep=sep)
        res = N.mcnemar_nested(za["label"], _pred(za), _pred(zb), za["cap"], grp, **kw)
        rows.append({
            "dataset": dataset, "model_a": ma, "variant_a": va,
            "model_b": mb, "variant_b": vb, "seed": s,
            **{k: res[k] for k in
               ("n_windows", "n_clusters", "n_groups", "delta_mcc",
                "delta_ci_lo", "delta_ci_hi", "p_value_file", "method_file",
                "p_value_group", "method_group", "p_min_group", "p_at_floor_group")},
            "cluster_net_advantage": json.dumps(res["cluster_net_advantage"]),
            "groups": "|".join(res["groups"]),
        })
        log.info("nested：%s %s vs %s  ΔMCC=%+.4f [%+.4f, %+.4f]  p_file=%.3g  p_group=%.3g (G=%d)",
                 dataset, ma, mb, res["delta_mcc"], res["delta_ci_lo"], res["delta_ci_hi"],
                 res["p_value_file"], res["p_value_group"], res["n_groups"])

    if rows:
        out = _out_dir(cfg)
        write_csv(out / "nested_inference.csv", rows, list(rows[0].keys()))
        log.info("nested：%d 对 → %s", len(rows), out / "nested_inference.csv")
    return rows


# --------------------------------------------------------------------------- #
# F4 · 未知攻击标准度量
# --------------------------------------------------------------------------- #
def run_openset(cfg: dict, dataset: str | None = None) -> list[dict]:
    node = cfg["audit"]["openset"]
    dataset = dataset or str(node["dataset"])
    fallback = int(node["unknown_index_fallback"])
    table: dict[str, dict] = {}

    for model, variant in [[str(x) for x in m] for m in node["models"]]:
        seeds = _seed_list(cfg, dataset, model, variant)
        if not seeds:
            log.warning("openset：缺 %s/%s/%s", dataset, model, variant)
            continue
        run_dir = C.resolve_run_dir(cfg, dataset, model, variant, seeds[0])
        z = _load_run(run_dir)
        lm = run_dir / "label_map.json"
        unknown_idx = fallback
        if lm.exists():
            obj = json.loads(lm.read_text(encoding="utf-8")).get("label_map", {})
            unknown_idx = int(obj.get("unknown_attack", fallback))
        r = O.openset_metrics(z["label"], z["y_type"], z["prob"],
                              float(z["threshold"]), unknown_idx)
        table[f"{model}/{variant}"] = r
        log.info("openset：%-20s unknown=%d  AUROC(u|normal)=%.4f  TPR@1%%FPR=%.4f  TPR@τ=%.4f",
                 f"{model}/{variant}", r.get("n_unknown", 0),
                 r.get("auroc_unknown_vs_normal", float("nan")),
                 r.get("tpr_at_fpr_normal_1", float("nan")),
                 r.get("tpr_unknown_at_tau", float("nan")))

    rows = O.summarize_table(table)
    if rows:
        out = _out_dir(cfg)
        write_csv(out / f"openset_{dataset}.csv", rows, list(rows[0].keys()))
        curves = {k: v["curve"] for k, v in table.items() if v.get("curve")}
        (out / f"detection_curves_{dataset}.json").write_text(
            json.dumps(curves, ensure_ascii=False), encoding="utf-8")
        log.info("openset：%d 个模型 → %s", len(rows), out / f"openset_{dataset}.csv")
    return rows


# --------------------------------------------------------------------------- #
# F2 · 部署单操作点重算
# --------------------------------------------------------------------------- #
def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def run_deploy(cfg: dict) -> list[dict]:
    node = cfg["audit"]["deploy"]
    canon = node["canonical"]
    prec, th, bs = str(canon["precision"]), int(canon["threads"]), int(canon["batch"])

    deploy_dir = Path(cfg["_resolved"]["project_root"]) / "outputs" / "pi5_deploy"
    v4 = deploy_dir / "summary_v4.csv"
    r1 = deploy_dir / "summary.csv"
    rows: list[dict] = []

    if v4.exists():
        for r in _read_csv(v4):
            if (r["precision"] == prec and int(r["threads"]) == th and int(r["batch"]) == bs):
                rows.append({"source": "R4", "model": r["model"], "precision": r["precision"],
                             "threads": int(r["threads"]), "batch": int(r["batch"]),
                             "T4_plus_ms": float(r["T4_plus"]), "T4_p50_ms": float(r["T4_p50"]),
                             "N_bus_2k": float(r["N2k"]), "mJ_per_win": float(r["mJ"]),
                             "statistic": "T4_plus"})
    if r1.exists():
        for r in _read_csv(r1):
            if int(r["threads"]) == th and int(r["batch"]) == bs:
                rows.append({"source": "R1", "model": r["model"], "precision": "FP32",
                             "threads": int(r["threads"]), "batch": int(r["batch"]),
                             "T4_plus_ms": float(r["t4_ms"]), "T4_p50_ms": float(r["t4_ms"]),
                             "N_bus_2k": float(r["n_bus_2k"]), "mJ_per_win": float(r["mj_per_win"]),
                             "statistic": "T4_tail_inclusive"})

    if not rows:
        log.warning("deploy：未找到 summary_v4.csv / summary.csv")
        return []

    out = _out_dir(cfg)
    write_csv(out / "deploy_canonical.csv", rows, list(rows[0].keys()))

    # 比值只在 canonical 轮次内算，杜绝跨轮次混用（F2 的根因）
    by = {(r["source"], r["model"]): r for r in rows}
    ratios: list[dict] = []
    for base in ("FD-TCN", "FD-TCN-PF"):
        a = by.get(("R4", base))
        b = by.get(("R4", "LSTM"))
        if not (a and b):
            continue
        ratios.append({"pair": f"{base} vs LSTM", "source": "R4", "statistic": "T4_plus",
                       "latency_ratio": a["T4_plus_ms"] / b["T4_plus_ms"],
                       "R_N_bus_2k": a["N_bus_2k"] / b["N_bus_2k"],
                       "energy_ratio": a["mJ_per_win"] / b["mJ_per_win"],
                       "T4_a": a["T4_plus_ms"], "T4_b": b["T4_plus_ms"],
                       "N2k_a": a["N_bus_2k"], "N2k_b": b["N_bus_2k"],
                       "mJ_a": a["mJ_per_win"], "mJ_b": b["mJ_per_win"]})
    if ratios:
        write_csv(out / "deploy_ratios.csv", ratios, list(ratios[0].keys()))
        for r in ratios:
            log.info("deploy：%s 慢 %.2f×  R=%.3f  mJ 比 %.2f",
                     r["pair"], r["latency_ratio"], r["R_N_bus_2k"], r["energy_ratio"])
    return ratios


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def write_summary(cfg: dict, payload: dict[str, Any]) -> Path:
    out = _out_dir(cfg)
    lines = ["# 审计整改复算摘要（2026-10-30）", "",
             "本文件由 `代码/run_audit.py` 自动生成；全部数字可在 `outputs/audit/` 与",
             "`outputs/pi5_deploy/` 下逐项复核，未引入任何未落盘的中间结论。", ""]

    if payload.get("deploy"):
        lines += ["## F2 · 部署单操作点（batch=1 / th=1 / FP32 / T4⁺ / R4 1 万轮）", "",
                  "| 对比 | T4⁺ (ms) | N_bus@2k | mJ/win | 慢 | R | mJ 比 |",
                  "|---|---|---|---|---|---|---|"]
        for r in payload["deploy"]:
            lines.append(f"| {r['pair']} | {r['T4_a']:.3f} vs {r['T4_b']:.3f} | "
                         f"{r['N2k_a']:.2f} vs {r['N2k_b']:.2f} | "
                         f"{r['mJ_a']:.2f} vs {r['mJ_b']:.2f} | "
                         f"{r['latency_ratio']:.2f}× | **{r['R_N_bus_2k']:.2f}** | "
                         f"{r['energy_ratio']:.2f}× |")
        lines += ["", "口径：R1（3 次中位、T4 含尾均值）与 R4（单遍 1 万轮、T4⁺）不可混用；",
                  "旧稿把 R1 的 FD-TCN T4 与 LSTM p50 并列、并把 FD-TCN-PF 的 mJ/N_bus 记到",
                  "FD-TCN 名下，故 3.5×/R=0.31 不成立。mJ/win 为板载 PMIC 估算（4.0 W 假设），",
                  "只进附录，正文主指标用 N_bus。", ""]

    if payload.get("nested"):
        lines += ["## D4-1 · 嵌套推断（组→capture→窗口）", "",
                  "| 语料 | 对比 | ΔMCC | 95% CI（两阶段自助） | p（capture 层） | p（组层） | G | p_min(组) |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in payload["nested"]:
            lines.append(f"| {r['dataset']} | {r['model_a']} vs {r['model_b']} | "
                         f"{r['delta_mcc']:+.4f} | [{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}] | "
                         f"{r['p_value_file']:.3g} | {r['p_value_group']:.3g} | "
                         f"{r['n_groups']} | {r['p_min_group']:.3g} |")
        lines += ["", "组层 p 是**最保守**口径，G 个组决定可达下限（2/2^G）。若 p_group 触底，",
                  "只能写「该粒度下不可判定」，不得写「无差异」；此时以各组净优势的符号一致性作证据。", ""]

    if payload.get("setwise"):
        lines += ["## F3 · 载荷通道 ΔMCC 按录制组拆分（CTT，full − Ab-12）", "",
                  "| 组 | 簇 | 正例占比 | MCC(full) | MCC(PF) | ΔMCC（seed 均值 ± SD） | 同向 seed |",
                  "|---|---|---|---|---|---|---|"]
        for blk in payload["setwise"]:
            for g in blk["groups"]:
                lines.append(f"| {g['group']} | {g['n_clusters']} | {g['prevalence']*100:.2f}% | "
                             f"{g['mcc_full_mean']:.4f} | {g['mcc_pf_mean']:.4f} | "
                             f"{g['delta_mcc_mean']:+.4f} ± {g['delta_mcc_std']:.4f} | "
                             f"{g['same_sign_seeds']}/{g['n_seeds']} |")
        v = payload["setwise"][0]["verdict"]
        lines += ["", f"**机械判读**：{v['n_same_direction']}/{v['n_groups']} 组同向"
                  f"（{'删载荷更好' if v['direction'] == 'negative' else '删载荷更差'}），"
                  f"ΔMCC 的量级比 max/min = {v['magnitude_ratio']:.2f}×。", "",
                  "预登记规则：① 若只在 1–2 个 SET 上为大正值、其余接近 0 → 支持「载荷通道编码",
                  "SET 身份」的捷径解释；② 若各组同向 → 可排除「单一 SET 伪影」，但**不等于**",
                  "排除「工具指纹」与「降维正则化」两种替代解释（后者需 Ab-12b 容量/通道阶梯）。",
                  "本例为 ②，且量级不均，正文须同时写这两句。", ""]

    if payload.get("openset"):
        lines += ["## F4 · 未知攻击标准度量（CTT，masquerade 单家族）", "",
                  "| 模型 | unknown 窗口 | AUROC(u\\|normal) | AUROC(u\\|known-all) | TPR@1%FPR | TPR@0.1%FPR | TPR@τ（部署点） |",
                  "|---|---|---|---|---|---|---|"]
        for r in payload["openset"]:
            if "auroc_unknown_vs_normal" not in r:
                lines.append(f"| {r['model']} | — | — | — | — | — | {r.get('note','')} |")
                continue
            lines.append(f"| {r['model']} | {r['n_unknown']} | {r['auroc_unknown_vs_normal']:.4f} | "
                         f"{r['auroc_unknown_vs_known']:.4f} | {r['tpr_at_fpr_normal_1']:.4f} | "
                         f"{r['tpr_at_fpr_normal_0p1']:.4f} | {r['tpr_unknown_at_tau']:.4f} |")
        lines += ["", "口径：AUROC(u|normal) 是新颖性检测口径（未知 vs 正常），不受「已知攻击本就高分」",
                  "干扰；AUROC(u|known-all) 是标准 OOD 口径，对二元检测器结构性偏低，两者不可混引。",
                  "全部数字为 **single-family case study**；二元检测器无拒识头，**不报 OSCR/AUOSCR**。", ""]

    p = out / "summary.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    log.info("摘要写入 %s", p)
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="审计整改复算（不重训）")
    ap.add_argument("--config", default=str(Path(__file__).parent / "training" / "configs" / "train.yaml"))
    ap.add_argument("--batch", default="main", help="run 批次名（默认 main）")
    ap.add_argument("--dataset", default=None, help="覆盖 audit.*.dataset")
    ap.add_argument("--set", dest="overrides", action="append", default=None)
    ap.add_argument("checks", nargs="*", default=None,
                    choices=["all", "setwise", "nested", "openset", "deploy"])
    args = ap.parse_args(argv)

    cfg = _batch_cfg(args.config, args.batch, args.overrides)
    setup_logging(str(cfg["runtime"]["log_level"]), None)

    checks = args.checks or ["all"]
    want = set(checks if checks != ["all"] else
               ["setwise", "nested", "openset", "deploy"])
    payload: dict[str, Any] = {}
    if "deploy" in want:
        payload["deploy"] = run_deploy(cfg)
    if "setwise" in want:
        payload["setwise"] = run_setwise(cfg, dataset=args.dataset)
    if "nested" in want:
        payload["nested"] = run_nested(cfg)
    if "openset" in want:
        payload["openset"] = run_openset(cfg, dataset=args.dataset)

    write_summary(cfg, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
