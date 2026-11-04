# -*- coding: utf-8 -*-
"""
training.tables —— 论文表格生成（LaTeX booktabs + CSV）
================================================================================
为什么需要单独一层
------------------
图表清单里 T7–T13 是"表"，此前只落了 CSV（per_type_*.csv / runs_index.csv），
不能直接进论文。本模块把 run 产物聚合成**可直接 \\input 的 booktabs 表**：

  T7  主结果       dataset × model（full）→ Prev / MCC / AUC-PR / AUC-PR_norm / F1 / FPR / n_pos
  T8  masquerade   单独子表（ROAD timed/untimed、CTT timed/untimed 合并统计）
  T10 消融         variant vs full 的 ΔMCC / ΔAUC-PR + 参数量
  T12 复杂度       params / ckpt MB / INT8 估算 MB / 测试 MCC
  T13 统计检验    由 compare.py 产出，本模块负责排版

排版约定（投稿硬性要求）
------------------------
- 只用 booktabs：`\\toprule / \\midrule / \\bottomrule`，**无竖线、无 \\hline**
- 数字 3 位小数；均值±标准差写成 `0.995 $\\pm$ 0.001`
- 表注放在 tabular 之后的 `\\footnotesize` 段落（不依赖 threeparttable，IEEE 模板可直接用）
- 下划线等特殊字符统一转义

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import config as C
from . import figures as F
from .utils import get_logger, write_csv

log = get_logger("tables")

MISSING = "--"


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _esc(text: Any) -> str:
    """LaTeX 特殊字符转义（表头与类别名里常见下划线）。"""
    s = str(text)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    return s


def _fmt(value: Any, decimals: int) -> str:
    if value is None:
        return MISSING
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    if not np.isfinite(v):
        return MISSING
    return f"{v:.{decimals}f}"


def _fmt_ms(mean: float | None, std: float | None, decimals: int, n: int) -> str:
    """均值±标准差；单 seed 或 std=0 时只写均值。"""
    if mean is None or not np.isfinite(mean):
        return MISSING
    if n <= 1 or std is None or not np.isfinite(std) or std == 0:
        return _fmt(mean, decimals)
    return f"{_fmt(mean, decimals)} $\\pm$ {_fmt(std, decimals)}"


def _dec(cfg: Mapping[str, Any]) -> int:
    return int(cfg["tables"]["decimals"])


def _dec_auc(cfg: Mapping[str, Any]) -> int:
    return int(cfg["tables"]["decimals_auc"])


def _ds(cfg: Mapping[str, Any], key: str) -> str:
    """数据集显示名（论文正文用），缺省回退到内部键名。"""
    return str((cfg["tables"].get("dataset_display") or {}).get(key, key))


def _fmt_p(p: float) -> str:
    """p 值用科学计数法排版（LaTeX 里 e-01 不可读）。"""
    if p is None or not np.isfinite(float(p)):
        return MISSING
    p = float(p)
    if p >= 0.01:
        return f"{p:.3f}"
    exp = int(np.floor(np.log10(max(p, 1e-300))))
    mant = p / (10 ** exp)
    return f"${mant:.2f}\\times10^{{{exp}}}$"


def table_dir(cfg: Mapping[str, Any]) -> Path:
    d = F.batch_root(cfg) / str(cfg["tables"]["dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def to_latex(*, caption: str, label: str, colspec: str,
             header: Sequence[str], rows: Sequence[Sequence[str]],
             notes: str = "", fontsize: str = "\\small") -> str:
    """生成一个完整 table 环境（booktabs，无竖线）。"""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        fontsize,
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(str(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        lines += ["\\par\\smallskip", f"{{\\footnotesize {notes}}}"]
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _write(cfg: Mapping[str, Any], name: str, rows: Sequence[Mapping[str, Any]],
           columns: Sequence[str], latex: str) -> list[Path]:
    """按 tables.formats 落盘（tex / csv）。"""
    out: list[Path] = []
    d = table_dir(cfg)
    for fmt in cfg["tables"]["formats"]:
        if fmt == "tex":
            p = d / f"{name}.tex"
            p.write_text(latex, encoding="utf-8")
            out.append(p)
        elif fmt == "csv":
            p = d / f"{name}.csv"
            write_csv(p, rows, columns=list(columns))
            out.append(p)
    log.info("表格 %s 已写出：%s", name, ", ".join(p.name for p in out))
    return out


# --------------------------------------------------------------------------- #
# 数据收集
# --------------------------------------------------------------------------- #
def _runs_by_group(cfg: Mapping[str, Any]) -> dict[tuple[str, str, str], list[F.RunArtifacts]]:
    groups: dict[tuple[str, str, str], list[F.RunArtifacts]] = {}
    for ctx in F.discover_runs(F.batch_root(cfg), cfg):
        groups.setdefault((ctx.dataset, ctx.model, ctx.variant), []).append(ctx)
    for v in groups.values():
        v.sort(key=lambda r: r.seed)
    return groups


def _agg(runs: Sequence[F.RunArtifacts], split: str = "test") -> dict[str, Any]:
    """把一个 group 内多个 seed 的 metrics_<split> 聚合成 mean/std。

    `split` 默认 "test"（full 口径）；"test_known" / "test_nosuppress" 由
    `run_regime.py` 产出，供 T7b / S1 使用。
    """
    keys = ("mcc", "auc_pr", "auc_pr_norm", "f1", "precision", "recall",
            "fpr", "prevalence", "n_pos", "threshold",
            "recall_at_fpr_0p1", "recall_at_fpr_1",
            "fpr_at_fpr_0p1", "fpr_at_fpr_1")
    vals: dict[str, list[float]] = {k: [] for k in keys}
    for r in runs:
        m = r.metrics(split)
        for k in keys:
            v = m.get(k)
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals[k].append(float(v))
    out: dict[str, Any] = {"n_seeds": len(runs)}
    for k, arr in vals.items():
        if not arr:
            out[k] = None
            out[f"{k}_std"] = None
        else:
            out[k] = float(np.mean(arr))
            out[f"{k}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ms = runs[0].manifest.get("model_stats", {}) if runs else {}
    out["n_params"] = ms.get("n_params")
    out["checkpoint_mb"] = ms.get("checkpoint_mb")
    out["int8_estimate_mb"] = ms.get("int8_estimate_mb")
    out["macs"] = ms.get("macs")
    return out


# --------------------------------------------------------------------------- #
# T7 主结果
# --------------------------------------------------------------------------- #
def _main_variants(cfg: Mapping[str, Any]) -> tuple[list[str], dict[str, str]]:
    """T7 主表收哪些变体，以及变体在表内的显示名。

    ★ 2026-10-27：原实现硬性只收 `ablation.default_variant`（= full），
    导致协议盲共报模型 FD-TCN-PF（Ab-12）永远进不了主表。
    现在由 `tables.main.variants` 配置；缺省时退回"只有 default_variant"（与旧行为一致），
    因此旧 run 快照（快照按当前配置为底合并）仍可正常出表。
    """
    default = str(cfg["ablation"]["default_variant"])
    tcfg = cfg["tables"]
    block = tcfg.get("main") if isinstance(tcfg, Mapping) else None
    block = block if isinstance(block, Mapping) else {}
    variants = [str(v) for v in (block.get("variants") or [default])]
    if default not in variants:
        variants.insert(0, default)
    labels = {str(k): str(v) for k, v in (block.get("variant_labels") or {}).items()}
    return variants, labels


def _display_name(model: str, variant: str, default_variant: str,
                  vlabels: Mapping[str, str]) -> str:
    """变体的表内显示名。

    `tables.main.variant_labels` 视为**完整显示名覆盖**（如 Ab-12 → "FD-TCN-PF"）；
    未配置时退回 "模型名 (变体名)"，且参照行（default_variant）只写模型名。
    """
    if variant in vlabels:
        return vlabels[variant]
    base = F.display_name(model)
    return base if variant == default_variant else f"{base} ({variant})"


def table_main(cfg: Mapping[str, Any]) -> list[Path]:
    groups = _runs_by_group(cfg)
    variants, vlabels = _main_variants(cfg)
    default_variant = str(cfg["ablation"]["default_variant"])
    rows: list[dict] = []
    for (dataset, model, variant), runs in sorted(groups.items()):
        if variant not in variants:
            continue
        a = _agg(runs)
        rows.append({"dataset": dataset, "model": model, "variant": variant, **a})
    if not rows:
        return []
    # 同一数据集内按 variants 的顺序排列，保证参照行（full）在前
    rows.sort(key=lambda r: (r["dataset"], r["model"], variants.index(r["variant"])))

    dec, dauc = _dec(cfg), _dec_auc(cfg)
    header = ["Dataset", "Model", "Prev.", "MCC", "AUC-PR",
              "AUC-PR$_{\\mathrm{norm}}$", "F1", "FPR", "$n_{\\mathrm{pos}}$",
              "$n_{\\mathrm{seed}}$"]
    body = []
    for r in rows:
        name = _display_name(r["model"], r["variant"], default_variant, vlabels)
        body.append([
            _esc(_ds(cfg, r["dataset"])), _esc(name),
            _fmt(r["prevalence"], dec),
            _fmt_ms(r["mcc"], r["mcc_std"], dec, r["n_seeds"]),
            _fmt_ms(r["auc_pr"], r["auc_pr_std"], dauc, r["n_seeds"]),
            _fmt_ms(r["auc_pr_norm"], r["auc_pr_norm_std"], dauc, r["n_seeds"]),
            _fmt_ms(r["f1"], r["f1_std"], dec, r["n_seeds"]),
            _fmt(r["fpr"], dec),
            f"{int(r['n_pos']) if r['n_pos'] is not None else MISSING}",
            str(r["n_seeds"]),
        ])
    cap = str(cfg["tables"]["captions"]["main"])
    notes = ("Window-level metrics on the held-out test split; threshold calibrated on the "
             "validation split and applied unchanged. \\texttt{Prev.} is the test-set prevalence; "
             "AUC-PR$_{\\mathrm{norm}}$ = (AUC-PR − Prev.)/(1 − Prev.).")
    tex = to_latex(caption=cap, label=str(cfg["tables"]["labels"]["main"]),
                   colspec="@{}llrrrrrrrr@{}", header=header, rows=body, notes=notes)
    return _write(cfg, "table_main", rows, list(rows[0].keys()), tex)


# --------------------------------------------------------------------------- #
# T7b 主结果 · known-attack 口径（v5 §4.5 双轨之一）
# --------------------------------------------------------------------------- #
def table_known(cfg: Mapping[str, Any]) -> list[Path]:
    """与 table_main 同构，但读 `metrics_test_known.json`（剔除 zero-shot 家族）。

    只有存在 zero-shot 正样本的语料才会有该文件（本批为 CTT），
    故本表天然只含 CTT —— 这正好符合 v5 §4.5 的"Δ 只能在同一语料内比较"。
    末列给出 ΔMCC = known − full，即"开集退化"的幅度。
    """
    groups = _runs_by_group(cfg)
    variants, vlabels = _main_variants(cfg)
    default_variant = str(cfg["ablation"]["default_variant"])
    rows: list[dict] = []
    for (dataset, model, variant), runs in sorted(groups.items()):
        if variant not in variants:
            continue
        if not any(bool(r.metrics("test_known")) for r in runs):
            continue
        a = _agg(runs, split="test_known")
        b = _agg(runs, split="test")
        delta = (a["mcc"] - b["mcc"]) if (a["mcc"] is not None and b["mcc"] is not None) else None
        rows.append({"dataset": dataset, "model": model, "variant": variant,
                     "display": _display_name(model, variant, default_variant, vlabels),
                     **{f"k_{k}": v for k, v in a.items()},
                     "mcc_full": b["mcc"], "delta_mcc": delta})
    if not rows:
        return []
    rows.sort(key=lambda r: (r["dataset"], r["model"], variants.index(r["variant"])))
    dec, dauc = _dec(cfg), _dec_auc(cfg)
    body = []
    for r in rows:
        body.append([
            _esc(_ds(cfg, r["dataset"])), _esc(r["display"]),
            _fmt(r["k_prevalence"], dec),
            _fmt_ms(r["k_mcc"], r["k_mcc_std"], dec, r["k_n_seeds"]),
            _fmt_ms(r["k_auc_pr_norm"], r["k_auc_pr_norm_std"], dauc, r["k_n_seeds"]),
            _fmt_ms(r["k_recall"], r["k_recall_std"], dec, r["k_n_seeds"]),
            _fmt(r["k_fpr"], dec),
            _fmt(r["mcc_full"], dec),
            _fmt(r["delta_mcc"], dec),
            f"{int(r['k_n_pos']) if r['k_n_pos'] is not None else MISSING}",
            str(r["k_n_seeds"]),
        ])
    cap = str(cfg["tables"]["captions"]["known"])
    notes = ("Same protocol as Table~\\ref{tab:main} but with the zero-shot family removed "
             "from the positive set. $\\Delta$MCC $=$ known $-$ full; "
             "a positive value quantifies the open-set penalty that a single-regime table hides.")
    tex = to_latex(caption=cap, label=str(cfg["tables"]["labels"]["known"]),
                   colspec="@{}llrrrrrrrr@{}",
                   header=["Dataset", "Model", "Prev.", "MCC", "AUC-PR$_{\\mathrm{norm}}$",
                           "Recall", "FPR", "MCC$_{\\mathrm{full}}$", "$\\Delta$MCC",
                           "$n_{\\mathrm{pos}}$", "$n_{\\mathrm{seed}}$"],
                   rows=body, notes=notes)
    return _write(cfg, "table_known", rows, list(rows[0].keys()), tex)


# --------------------------------------------------------------------------- #
# S1 敏感性：剔除 suppress 采集（v5 §4.5 声明③）
# --------------------------------------------------------------------------- #
def table_suppress(cfg: Mapping[str, Any]) -> list[Path]:
    """对比"含 suppress 负样本"与"剔除后"的 MCC / FPR，量化该污染的影响上界。"""
    groups = _runs_by_group(cfg)
    variants, vlabels = _main_variants(cfg)
    default_variant = str(cfg["ablation"]["default_variant"])
    rows: list[dict] = []
    for (dataset, model, variant), runs in sorted(groups.items()):
        if variant not in variants:
            continue
        if not any(bool(r.metrics("test_nosuppress")) for r in runs):
            continue
        a = _agg(runs, split="test")
        b = _agg(runs, split="test_nosuppress")
        rows.append({
            "dataset": dataset, "model": model, "variant": variant,
            "display": _display_name(model, variant, default_variant, vlabels),
            "n_seeds": a["n_seeds"],
            "mcc": a["mcc"], "mcc_nosup": b["mcc"],
            "delta_mcc": ((b["mcc"] - a["mcc"])
                          if (a["mcc"] is not None and b["mcc"] is not None) else None),
            "fpr": a["fpr"], "fpr_nosup": b["fpr"],
            "delta_fpr": ((b["fpr"] - a["fpr"])
                          if (a["fpr"] is not None and b["fpr"] is not None) else None),
        })
    if not rows:
        return []
    rows.sort(key=lambda r: (r["dataset"], r["model"], variants.index(r["variant"])))
    dec = _dec(cfg)
    body = []
    for r in rows:
        body.append([
            _esc(_ds(cfg, r["dataset"])), _esc(r["display"]), str(r["n_seeds"]),
            _fmt(r["mcc"], dec), _fmt(r["mcc_nosup"], dec), _fmt(r["delta_mcc"], dec),
            _fmt(r["fpr"], dec), _fmt(r["fpr_nosup"], dec), _fmt(r["delta_fpr"], dec),
        ])
    cap = str(cfg["tables"]["captions"]["suppress"])
    notes = ("Sensitivity check: the suppress captures are removed from the test split. "
             "Their windows are labelled normal (the attack removes messages, so no per-frame "
             "labels exist) while being semantically adversarial; a detector that flags them "
             "is scored as a false positive. A near-zero $\\Delta$ therefore means the "
             "contamination does not materially bias the reported numbers.")
    tex = to_latex(caption=cap, label=str(cfg["tables"]["labels"]["suppress"]),
                   colspec="@{}llrrrrrrr@{}",
                   header=["Dataset", "Model", "Seeds", "MCC", "MCC$_{\\mathrm{no\\,sup}}$",
                           "$\\Delta$MCC", "FPR", "FPR$_{\\mathrm{no\\,sup}}$", "$\\Delta$FPR"],
                   rows=body, notes=notes)
    return _write(cfg, "table_suppress", rows, list(rows[0].keys()), tex)


# --------------------------------------------------------------------------- #
# T9 外部基线（表格特征：HistGBDT / RF / CART）对照
# --------------------------------------------------------------------------- #
BASELINE_LABELS: dict[str, str] = {
    "gbdt": "HistGBDT",
    "rf": "Random Forest",
    "cart": "CART (32 leaves)",
}


def table_baselines(cfg: Mapping[str, Any]) -> list[Path]:
    """深度时序模型 vs 表格特征强基线（v5 §5.2 / T9）。

    数据源：`<batch>/_baselines/<dataset>/<model>/seed<seed>/metrics_test.json`
    （由 `代码/run_baselines.py` 产出），外加本批次 FD-TCN / FD-TCN-PF 作参照。
    回答审稿人必问的"深度模型是否必要"。
    """
    variants, vlabels = _main_variants(cfg)
    default_variant = str(cfg["ablation"]["default_variant"])
    rows: list[dict] = []

    for mp in sorted((F.batch_root(cfg) / "_baselines").glob("*/*/seed*/metrics_test.json")):
        ds, model = mp.parts[-4], mp.parts[-3]
        m = json.loads(mp.read_text(encoding="utf-8"))
        rows.append({
            "dataset": ds, "display": BASELINE_LABELS.get(model, model),
            "kind": "tabular baseline", "n_seeds": 1,
            "mcc": m.get("mcc"), "auc_pr_norm": m.get("auc_pr_norm"),
            "recall_at_fpr_1": m.get("recall_at_fpr_1"),
            "fpr_at_fpr_1": m.get("fpr_at_fpr_1"), "fpr": m.get("fpr"),
        })

    for (ds, model, variant), runs in sorted(_runs_by_group(cfg).items()):
        if model != "fdtcn" or variant not in variants:
            continue
        a = _agg(runs)
        rows.append({
            "dataset": ds,
            "display": _display_name(model, variant, default_variant, vlabels),
            "kind": "this work", "n_seeds": a["n_seeds"],
            "mcc": a["mcc"], "auc_pr_norm": a["auc_pr_norm"],
            "recall_at_fpr_1": a["recall_at_fpr_1"],
            "fpr_at_fpr_1": a["fpr_at_fpr_1"], "fpr": a["fpr"],
        })
    if not rows:
        return []

    ours = {"this work": 0}
    rows.sort(key=lambda r: (r["dataset"], ours.get(r["kind"], 1), -float(r["mcc"] or 0)))
    dec, dauc = _dec(cfg), _dec_auc(cfg)
    body = []
    for r in rows:
        body.append([
            _esc(_ds(cfg, r["dataset"])), _esc(r["display"]),
            _fmt_ms(r["mcc"], None, dec, r["n_seeds"]),
            _fmt(r["auc_pr_norm"], dauc),
            _fmt(r["recall_at_fpr_1"], dec),
            _fmt(r["fpr_at_fpr_1"], dec),
            str(r["n_seeds"]),
        ])
    cap = str(cfg["tables"]["captions"]["baselines"])
    notes = ("Tabular baselines use the same windows and the same 80-dimensional "
             "window statistics as the rule-distillation branch (16 channels $\\times$ "
             "{mean, std, last, min, max}). All rows share one threshold protocol "
             "(calibrated on validation, applied unchanged to test). "
             "HistGBDT is scikit-learn's histogram gradient boosting; \\emph{not} XGBoost. "
             "\\emph{FPR@1\\%} is the realised test FPR at the validation-derived 1\\% "
             "operating point.")
    tex = to_latex(caption=cap, label=str(cfg["tables"]["labels"]["baselines"]),
                   colspec="@{}llrrrrr@{}",
                   header=["Dataset", "Model", "MCC", "AUC-PR$_{\\mathrm{norm}}$",
                           "Recall@FPR=1%", "FPR@1%", "$n_{\\mathrm{seed}}$"],
                   rows=body, notes=notes)
    return _write(cfg, "table_baselines", rows, list(rows[0].keys()), tex)


# --------------------------------------------------------------------------- #
# T7d 阈值稳定性与 val→test 退化（v5 §6.1）
# --------------------------------------------------------------------------- #
def table_threshold(cfg: Mapping[str, Any]) -> list[Path]:
    """逐 (数据集, 模型/变体) 给出 τ 的跨 seed 稳定性，以及 val→test 的 MCC 落差。

    回答两个问题：① "按 MCC 在 val 上选 τ" 是否稳定（CTT 实测 τ 标准差与均值同量级）；
    ② 模型从 val 到 test 掉多少（衡量可迁移性）。二者共同决定 MCC 单值能否作为部署依据。
    """
    groups = _runs_by_group(cfg)
    variants, vlabels = _main_variants(cfg)
    default_variant = str(cfg["ablation"]["default_variant"])
    rows: list[dict] = []
    for (dataset, model, variant), runs in sorted(groups.items()):
        if variant not in variants:
            continue
        taus: list[float] = []
        v_mcc: list[float] = []
        t_mcc: list[float] = []
        for r in runs:
            mv, mt = r.metrics("val"), r.metrics("test")
            tau = mv.get("threshold")
            if isinstance(tau, (int, float)) and np.isfinite(tau):
                taus.append(float(tau))
            a, b = mv.get("mcc"), mt.get("mcc")
            if isinstance(a, (int, float)) and np.isfinite(a):
                v_mcc.append(float(a))
            if isinstance(b, (int, float)) and np.isfinite(b):
                t_mcc.append(float(b))
        if not taus:
            continue
        name = _display_name(model, variant, default_variant, vlabels)
        mcc_v = float(np.mean(v_mcc)) if v_mcc else None
        mcc_t = float(np.mean(t_mcc)) if t_mcc else None
        rows.append({
            "dataset": dataset, "model": model, "variant": variant, "display": name,
            "n_seeds": len(taus),
            "tau_mean": float(np.mean(taus)),
            "tau_std": float(np.std(taus, ddof=1)) if len(taus) > 1 else 0.0,
            "tau_min": float(np.min(taus)), "tau_max": float(np.max(taus)),
            "mcc_val": mcc_v, "mcc_test": mcc_t,
            "delta_val_test": ((mcc_t - mcc_v)
                               if (mcc_v is not None and mcc_t is not None) else None),
        })
    if not rows:
        return []
    dec = _dec(cfg)
    body = []
    for r in rows:
        body.append([
            _esc(_ds(cfg, r["dataset"])), _esc(r["display"]), str(r["n_seeds"]),
            _fmt_ms(r["tau_mean"], r["tau_std"], dec, r["n_seeds"]),
            f"{_fmt(r['tau_min'], dec)}--{_fmt(r['tau_max'], dec)}",
            _fmt(r["mcc_val"], dec), _fmt(r["mcc_test"], dec),
            _fmt(r["delta_val_test"], dec),
        ])
    cap = str(cfg["tables"]["captions"]["threshold"])
    notes = ("$\\tau$ is selected per seed on the validation split by maximising MCC. "
             "A $\\tau$ std comparable to its mean indicates that threshold selection is unstable; "
             "$\\Delta$ is MCC$_{\\mathrm{test}}-{}$MCC$_{\\mathrm{val}}$.")
    tex = to_latex(caption=cap, label=str(cfg["tables"]["labels"]["threshold"]),
                   colspec="@{}llrrrrrr@{}",
                   header=["Dataset", "Model", "Seeds", "$\\tau$ (mean$\\pm$std)",
                           "$\\tau$ range", "MCC$_{\\mathrm{val}}$", "MCC$_{\\mathrm{test}}$",
                           "$\\Delta$(val$\\to$test)"],
                   rows=body, notes=notes)
    return _write(cfg, "table_threshold", rows, list(rows[0].keys()), tex)


# --------------------------------------------------------------------------- #
# T8 masquerade 子表
# --------------------------------------------------------------------------- #
def table_masquerade(cfg: Mapping[str, Any]) -> list[Path]:
    groups = _runs_by_group(cfg)
    rows: list[dict] = []
    for (dataset, model, variant), runs in sorted(groups.items()):
        if variant != str(cfg["ablation"]["default_variant"]):
            continue
        pt = runs[0].per_type("test")
        masq = [r for r in pt if str(r.get("is_masquerade")).lower() == "true"
                and r.get("attack_type") != "overall"]
        others = [r for r in pt if str(r.get("is_masquerade")).lower() != "true"
                  and r.get("attack_type") != "overall"]
        if not masq:
            continue
        n_m = sum(int(float(r["n_positive_windows"])) for r in masq)
        rec_m = float(np.average([float(r["recall"]) for r in masq],
                                 weights=[int(float(r["n_positive_windows"])) for r in masq]))
        sc_m = float(np.average([float(r["mean_score"]) for r in masq],
                                weights=[int(float(r["n_positive_windows"])) for r in masq]))
        n_o = sum(int(float(r["n_positive_windows"])) for r in others) if others else 0
        rec_o = (float(np.average([float(r["recall"]) for r in others],
                                  weights=[int(float(r["n_positive_windows"])) for r in others]))
                 if others else None)
        rows.append({"dataset": dataset, "model": model,
                     "n_masq": n_m, "recall_masq": rec_m, "mean_score_masq": sc_m,
                     "n_other": n_o, "recall_other": rec_o,
                     "types": ", ".join(sorted({str(r["attack_type"]) for r in masq}))})
    if not rows:
        return []

    dec = _dec(cfg)
    header = ["Dataset", "Model", "Masq. types", "$n_{\\mathrm{masq}}$",
              "Recall$_{\\mathrm{masq}}$", "Mean score", "Recall$_{\\mathrm{other}}$"]
    body = [[_esc(_ds(cfg, r["dataset"])), _esc(F.display_name(r["model"])), _esc(r["types"]), str(r["n_masq"]),
             _fmt(r["recall_masq"], dec), _fmt(r["mean_score_masq"], dec),
             _fmt(r["recall_other"], dec) if r["recall_other"] is not None else MISSING]
            for r in rows]
    tex = to_latex(caption=str(cfg["tables"]["captions"]["masquerade"]),
                   label=str(cfg["tables"]["labels"]["masquerade"]),
                   colspec="@{}llp{3.6cm}rrrr@{}", header=header, rows=body,
                   notes=("Masquerade rows are aggregated over all masquerade types with "
                          "window counts as weights. \\texttt{Recall\\_other} covers non-masquerade "
                          "attacks in the same test split."))
    return _write(cfg, "table_masquerade", rows, list(rows[0].keys()), tex)


# --------------------------------------------------------------------------- #
# T14 数据集规格（v4 §4.1 的可复算版本）
# --------------------------------------------------------------------------- #
def table_dataset_specs(cfg: Mapping[str, Any]) -> list[Path]:
    from . import manifest as MF

    rows: list[dict] = []
    for ds in cfg["paths"]["dataset_dirs"]:
        try:
            paths = C.dataset_paths(cfg, ds)
        except Exception:
            continue
        idx = MF.ManifestIndex(ds, paths["manifest_dir"])
        stats = MF.read_dataset_stats(paths["manifest_dir"])
        frames = sum(int(float(r["n_frames"])) for r in stats)
        atk = sum(int(float(r["n_attack_frames"])) for r in stats)
        rows.append({
            "dataset": ds,
            "protocol": stats[0]["protocol"] if stats else "",
            "captures": len(idx.captures),
            "frames": frames,
            "attack_frames": atk,
            "attack_rate": (atk / frames) if frames else 0.0,
            # 逐 capture 的 split_mode 才准确（splits.json 的全局字段只反映块化模式）
            "split_mode": stats[0].get("split_mode", "") if stats else "",
            "w_train": idx.expected_windows("train"),
            "w_val": idx.expected_windows("val"),
            "w_test": idx.expected_windows("test"),
        })
    if not rows:
        return []

    header = ["Dataset", "Protocol", "Captures", "Frames", "Attack frames",
              "Rate (\\%)", "Split", "$W_{train}$", "$W_{val}$", "$W_{test}$"]
    body = [[_esc(_ds(cfg, r["dataset"])), _esc(r["protocol"]), str(r["captures"]),
             f"{r['frames']:,}", f"{r['attack_frames']:,}",
             f"{100 * r['attack_rate']:.3f}", _esc(r["split_mode"]),
             f"{r['w_train']:,}", f"{r['w_val']:,}", f"{r['w_test']:,}"] for r in rows]
    tot_f = sum(r["frames"] for r in rows)
    tot_a = sum(r["attack_frames"] for r in rows)
    body.append(["Total", "--", str(sum(r["captures"] for r in rows)), f"{tot_f:,}",
                 f"{tot_a:,}", f"{100 * tot_a / tot_f:.3f}" if tot_f else MISSING,
                 "--", f"{sum(r['w_train'] for r in rows):,}",
                 f"{sum(r['w_val'] for r in rows):,}",
                 f"{sum(r['w_test'] for r in rows):,}"])
    tex = to_latex(caption=str(cfg["tables"]["captions"]["dataset"]),
                   label=str(cfg["tables"]["labels"]["dataset"]),
                   colspec="@{}llrrrrlrrr@{}", header=header, rows=body,
                   notes=("Frame and window counts are recomputed from the published manifest "
                          "(splits.json / dataset_stats.csv); window counts follow "
                          "1 + (L − W)/stride inside each split range, with W = 64."))
    return _write(cfg, "table_dataset", rows, list(rows[0].keys()), tex)


# --------------------------------------------------------------------------- #
# T7b 操作点（v4 §6.1：Recall @ FPR ∈ {0.1%, 1%}）
# --------------------------------------------------------------------------- #
def table_operating(cfg: Mapping[str, Any]) -> list[Path]:
    """固定 FPR 操作点（v5 §6.1）。

    ★ 2026-10-27：① 支持多变体（FD-TCN-PF 入表，与 T7a 一致）；
    ② 新增「实际 FPR」列 —— 名义目标 0.1%/1% 的 τ 来自 val，在 test 上
    实际达到的 FPR 可能高于目标（先验漂移），必须与召回并报，否则读者会
    误以为部署时真能压到 1%。
    """
    groups = _runs_by_group(cfg)
    variants, vlabels = _main_variants(cfg)
    default_variant = str(cfg["ablation"]["default_variant"])
    rows: list[dict] = []
    for (dataset, model, variant), runs in sorted(groups.items()):
        if variant not in variants:
            continue
        a = _agg(runs)
        rows.append({"dataset": dataset, "model": model, "variant": variant,
                     "display": _display_name(model, variant, default_variant, vlabels),
                     "threshold": a["threshold"],
                     "recall_at_fpr_0p1": a.get("recall_at_fpr_0p1"),
                     "recall_at_fpr_1": a.get("recall_at_fpr_1"),
                     "fpr_at_fpr_1": a.get("fpr_at_fpr_1"),
                     "fpr": a["fpr"], "n_seeds": a["n_seeds"]})
    if not rows:
        return []
    rows.sort(key=lambda r: (r["dataset"], r["model"], variants.index(r["variant"])))
    dec = _dec(cfg)
    header = ["Dataset", "Model", "$\\tau$ (MCC)", "Recall@FPR=0.1%",
              "Recall@FPR=1%", "FPR@1% (actual)", "FPR@$\\tau$", "$n_{\\mathrm{seed}}$"]
    body = [[_esc(_ds(cfg, r["dataset"])), _esc(r["display"]),
             _fmt(r["threshold"], dec), _fmt(r["recall_at_fpr_0p1"], dec),
             _fmt(r["recall_at_fpr_1"], dec), _fmt(r["fpr_at_fpr_1"], dec),
             _fmt(r["fpr"], dec), str(r["n_seeds"])]
            for r in rows]
    tex = to_latex(caption=str(cfg["tables"]["captions"]["operating"]),
                   label=str(cfg["tables"]["labels"]["operating"]), colspec="@{}llrrrrrr@{}",
                   header=header, rows=body,
                   notes=("Thresholds for the fixed-FPR operating points are calibrated on the "
                          "validation split only and then applied unchanged to the test split. "
                          "\\emph{FPR@1\\% (actual)} is the false-positive rate that this "
                          "validation-derived threshold really achieves on the test split; it "
                          "exceeds the nominal target whenever the prior shifts, and is reported "
                          "so that the operating point is not mistaken for a deployable guarantee."))
    return _write(cfg, "table_operating", rows, list(rows[0].keys()), tex)


# --------------------------------------------------------------------------- #
# T7c KD-Logic（v4 §5.1 三模型之一）
# --------------------------------------------------------------------------- #
def table_kd(cfg: Mapping[str, Any]) -> list[Path]:
    rows: list[dict] = []
    for ctx in F.discover_runs(F.batch_root(cfg), cfg):
        p = ctx.dir / "kd" / "kd_results.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        teacher_mcc = ctx.metrics("test").get("mcc")
        rows.append({
            "dataset": ctx.dataset, "teacher": ctx.model, "seed": ctx.seed,
            "fidelity": d["distilled"]["fidelity"],
            "mcc": d["distilled"]["mcc"],
            "delta_mcc": (d["distilled"]["mcc"] - teacher_mcc)
            if teacher_mcc is not None else None,
            "auc_pr": d["distilled"]["auc_pr"],
            "n_rules": d["distilled"]["n_rules"],
            "depth": d["distilled"]["depth"],
            "size_kb": d["distilled"]["size_kb"],
            "direct_mcc": d.get("direct", {}).get("mcc"),
        })
    if not rows:
        return []

    # 按 (dataset, teacher) 聚合
    agg: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        agg.setdefault((r["dataset"], r["teacher"]), []).append(r)
    out: list[dict] = []
    for (ds, teacher), grp in sorted(agg.items()):
        def mean(key):
            vals = [g[key] for g in grp if isinstance(g.get(key), (int, float))
                    and np.isfinite(g[key])]
            return float(np.mean(vals)) if vals else None
        out.append({"dataset": ds, "teacher": teacher, "n_seeds": len(grp),
                    "fidelity": mean("fidelity"), "mcc": mean("mcc"),
                    "delta_mcc": mean("delta_mcc"), "direct_mcc": mean("direct_mcc"),
                    "n_rules": mean("n_rules"), "depth": mean("depth"),
                    "size_kb": mean("size_kb")})

    dec = _dec(cfg)
    header = ["Dataset", "Teacher", "Fidelity", "MCC", "$\\Delta$MCC",
              "Direct CART", "Rules", "Depth", "Size (KB)", "$n_{\\mathrm{seed}}$"]
    body = [[_esc(_ds(cfg, r["dataset"])), _esc(F.display_name(r["teacher"])),
             _fmt(r["fidelity"], dec), _fmt(r["mcc"], dec),
             (f"{r['delta_mcc']:+.{dec}f}" if r["delta_mcc"] is not None else MISSING),
             _fmt(r["direct_mcc"], dec),
             (f"{int(r['n_rules'])}" if r["n_rules"] else MISSING),
             (f"{int(r['depth'])}" if r["depth"] else MISSING),
             _fmt(r["size_kb"], 2), str(r["n_seeds"])] for r in out]
    tex = to_latex(caption=str(cfg["tables"]["captions"]["kd"]),
                   label=str(cfg["tables"]["labels"]["kd"]), colspec="@{}llrrrrrrrr@{}", header=header, rows=body,
                   notes=("Distilled CART is constrained to $\\le$32 leaves and depth $\\le$8; "
                          "`Direct CART` is trained on ground-truth labels with the same constraints "
                          "and serves as the non-distilled lower bound."))
    return _write(cfg, "table_kd", out, list(out[0].keys()), tex)


# --------------------------------------------------------------------------- #
# T10 消融
# --------------------------------------------------------------------------- #
def table_ablation(cfg: Mapping[str, Any]) -> list[Path]:
    groups = _runs_by_group(cfg)
    base = {k: _agg(v) for k, v in groups.items()
            if k[2] == str(cfg["ablation"]["default_variant"])}
    rows: list[dict] = []
    for (dataset, model, variant), runs in sorted(groups.items()):
        if variant == str(cfg["ablation"]["default_variant"]):
            continue
        ref = base.get((dataset, model, str(cfg["ablation"]["default_variant"])))
        if ref is None:
            continue
        a = _agg(runs)
        rows.append({
            "dataset": dataset, "model": model, "variant": variant,
            "mcc": a["mcc"], "delta_mcc": (a["mcc"] - ref["mcc"])
            if a["mcc"] is not None and ref["mcc"] is not None else None,
            "auc_pr": a["auc_pr"], "delta_auc_pr": (a["auc_pr"] - ref["auc_pr"])
            if a["auc_pr"] is not None and ref["auc_pr"] is not None else None,
            "n_params": a["n_params"], "n_seeds": a["n_seeds"],
        })
    if not rows:
        return []

    dec, dauc = _dec(cfg), _dec_auc(cfg)
    header = ["Dataset", "Variant", "MCC", "$\\Delta$MCC", "AUC-PR",
              "$\\Delta$AUC-PR", "Params"]
    body = []
    for r in rows:
        dm = r["delta_mcc"]
        da = r["delta_auc_pr"]
        body.append([
            _esc(_ds(cfg, r["dataset"])), _esc(r["variant"]),
            _fmt_ms(r["mcc"], None, dec, 1),
            (f"{dm:+.{dec}f}" if dm is not None else MISSING),
            _fmt(r["auc_pr"], dauc),
            (f"{da:+.{dauc}f}" if da is not None else MISSING),
            f"{int(r['n_params']):,}" if r["n_params"] else MISSING,
        ])
    tex = to_latex(caption=str(cfg["tables"]["captions"]["ablation"]),
                   label=str(cfg["tables"]["labels"]["ablation"]),
                   colspec="@{}llrrrrr@{}", header=header, rows=body,
                   notes=("$\\Delta$ is relative to the \\texttt{full} variant of the same dataset "
                          "and backbone. Positive $\\Delta$MCC means the ablated variant is worse."))
    return _write(cfg, "table_ablation", rows, list(rows[0].keys()), tex)


# --------------------------------------------------------------------------- #
# T12 复杂度（部署指标的子集）
# --------------------------------------------------------------------------- #
def _receptive_field(cfg: Mapping[str, Any], model: str) -> int | None:
    """时序主干的感受野（帧）。RF = 1 + Σ(k−1)·d（因果卷积逐层累加）。

    只对 FD-TCN 有意义（dilations 显式给出）；其余模型无有限感受野概念，返回 None。
    用途：替代被删除的 MACs 列——它是**同架构内可比、跨架构语义明确**的结构量。
    """
    try:
        mv = cfg["model"]["variants"][model]
    except Exception:                                        # noqa: BLE001
        return None
    dil = mv.get("dilations")
    if not dil:
        return None
    k = int(mv.get("kernel_size", 3))
    # ★ 每个残差块含 **两个** 因果卷积层，故外层乘 2：
    #   RF = 1 + 2·Σ(k−1)·d_i。校验：d=[1,2,4,8], k=3 → 1+2·30 = 61（full 的 RF）。
    #   若漏掉这个 2 会得到 31，正好是 RF 扫描里 [1,1,1,1] 之外的错误值。
    return int(1 + 2 * sum((k - 1) * int(d) for d in dil))


def table_complexity(cfg: Mapping[str, Any]) -> list[Path]:
    groups = _runs_by_group(cfg)
    rows: list[dict] = []
    for (dataset, model, variant), runs in sorted(groups.items()):
        if variant != str(cfg["ablation"]["default_variant"]):
            continue
        a = _agg(runs)
        rows.append({"dataset": dataset, "model": model,
                     "n_params": a["n_params"], "checkpoint_mb": a["checkpoint_mb"],
                     "int8_estimate_mb": a["int8_estimate_mb"], "macs": a.get("macs"),
                     "receptive_field": _receptive_field(cfg, model),
                     "mcc": a["mcc"],
                     "n_seeds": a["n_seeds"]})
    if not rows:
        return []

    dec = _dec(cfg)
    # ★ 2026-10-29（v6 §6.2）：**删除 MACs 列**。
    #   原因：count_macs 只给 Conv/Linear 挂 hook，LSTM 的循环运算与 attention 的
    #   matmul 完全未统计（LSTM 记 4,480，理论 ≈3.9e5，差两个数量级），
    #   该列**跨架构不可比且系统性偏向基线**。Pi5 实测亦证明方向相反
    #   （T4 3.645 ms vs LSTM 1.043 ms），保留只会被抓。
    #   MACs 仍写入 CSV 供溯源，但不进 LaTeX 表、正文不得引用。
    header = ["Dataset", "Model", "Params", "RF", "ckpt (MB)",
              "INT8 est. (MB)", "Test MCC"]
    body = [[_esc(_ds(cfg, r["dataset"])), _esc(F.display_name(r["model"])),
             f"{int(r['n_params']):,}" if r["n_params"] else MISSING,
             str(r["receptive_field"]) if r.get("receptive_field") else "—",
             _fmt(r["checkpoint_mb"], 2),
             _fmt(r["int8_estimate_mb"], 3),
             _fmt_ms(r["mcc"], None, dec, 1)] for r in rows]
    tex = to_latex(caption=str(cfg["tables"]["captions"]["complexity"]),
                   label=str(cfg["tables"]["labels"]["complexity"]),
                   colspec="@{}llrrrrr@{}", header=header, rows=body,
                   notes=("\\texttt{ckpt} includes optimizer state; \\texttt{INT8 est.} = parameters × 1 byte, "
                          "excluding activation buffers. \\textbf{No cross-architecture operation counts are "
                          "reported}: the MAC counter covers only Conv/Linear layers, so recurrent and attention "
                          "operations are not comparable across models. \\texttt{RF} is the receptive field in "
                          "frames (computed for the temporal backbone). Latency figures are reported separately "
                          "on the Raspberry Pi 5."))
    return _write(cfg, "table_complexity", rows, list(rows[0].keys()), tex)


# --------------------------------------------------------------------------- #
# T13 统计检验（读取 compare 产物）
# --------------------------------------------------------------------------- #
def table_stats(cfg: Mapping[str, Any]) -> list[Path]:
    p = F.batch_root(cfg) / "comparison" / "comparison.csv"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8-sig", newline="") as fh:
        data = list(csv.DictReader(fh))
    if not data:
        return []

    dec = _dec(cfg)
    header = ["Dataset", "Comparison", "$b$", "$c$", "$p$", "$p_{\\mathrm{Holm}}$", "Sig."]
    body = []
    for r in data:
        pv = float(r["p_value"])
        ph = float(r["p_holm"])
        sig = "yes" if ph < float(r.get("alpha", 0.05)) else "no"
        body.append([_esc(_ds(cfg, r["dataset"])), _esc(r["comparison"]),
                     r["b"], r["c"], _fmt_p(pv), _fmt_p(ph), sig])
    tex = to_latex(caption=str(cfg["tables"]["captions"]["stats"]),
                   label=str(cfg["tables"]["labels"]["stats"]),
                   colspec="@{}llrrrrr@{}", header=header, rows=body,
                   notes=("McNemar test on paired binary decisions at each model's own "
                          "validation-calibrated threshold; $b$ = first wrong / second right, "
                          "$c$ = first right / second wrong. Holm correction across all pairs "
                          "of the same comparison family. \\textbf{Caveat:} windows within a "
                          "capture are not independent; with $10^6$ windows the test is "
                          "over-powered, so $p$ should be read together with the "
                          "capture-level bootstrap CI, not on its own."))
    # ★ 2026-10-28：落盘 CSV 去掉绝对路径（原样写入会把本机目录带进投稿附件）
    safe = [dict(r, run_a=_short_run_id(r.get("run_a", "")),
                 run_b=_short_run_id(r.get("run_b", ""))) for r in data]
    return _write(cfg, "table_stats", safe, list(safe[0].keys()), tex)


def _short_run_id(path: Any) -> str:
    """`…/ctt15/fdtcn/full/seed42` → `ctt15/fdtcn/full/seed42`。"""
    parts = Path(str(path)).parts
    return "/".join(parts[-4:]) if len(parts) >= 4 else str(path)


# --------------------------------------------------------------------------- #
# T11 攻击帧密度分层（2026-10-28 新增）
# --------------------------------------------------------------------------- #
def table_density(cfg: Mapping[str, Any]) -> list[Path]:
    """按"窗内攻击帧数"分箱的召回表。

    为什么进正文：窗口标签是 any_frame，正窗口的难度由窗内攻击帧数决定。
    实测 CTT 上 90% 的模型间召回差距来自 ≤8 帧的窗口；不拆开看，
    MCC 会把"看不见单帧"误读成"整体更弱"。
    """
    groups = _runs_by_group(cfg)
    variants, vlabels = _main_variants(cfg)
    default_variant = str(cfg["ablation"]["default_variant"])
    dec = _dec(cfg)
    fname = str(cfg["artifacts"]["per_density_table"]).format(split="test")

    rows: list[dict] = []
    for (dataset, model, variant), runs in sorted(groups.items()):
        if variant not in variants:
            continue
        per_bin: dict[str, list[dict]] = {}
        for r in runs:
            f = r.dir / fname
            if not f.exists():
                continue
            with open(f, "r", encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    if row["density_bin"] == "overall":
                        continue
                    per_bin.setdefault(row["density_bin"], []).append(row)
        if not per_bin:
            continue
        name = _display_name(model, variant, default_variant, vlabels)
        for b in sorted(per_bin, key=lambda k: int(per_bin[k][0]["density_lo"])):
            rs = per_bin[b]
            n_pos = [int(float(x["n_positive"])) for x in rs]
            rows.append({
                "dataset": dataset, "model": model, "variant": variant,
                "display": name, "density_bin": b,
                "density_lo": int(rs[0]["density_lo"]),
                "n_seeds": len(rs),
                "n_positive": int(round(float(np.mean(n_pos)))),
                "frac_of_positives": float(np.mean([float(x["frac_of_positives"]) for x in rs])),
                "recall": float(np.mean([float(x["recall"]) for x in rs])),
                "recall_at_fpr": float(np.mean([float(x["recall_at_fpr"]) for x in rs])),
            })
    if not rows:
        return []

    header = ["Dataset", "Model", "Attack frames / window", "Share of positives",
              "Recall@$\\tau$", "Recall@1% FPR", "$n_{\\mathrm{seed}}$"]
    body = []
    for r in rows:
        body.append([
            _esc(_ds(cfg, r["dataset"])), _esc(r["display"]), _esc(r["density_bin"]),
            _fmt(r["frac_of_positives"], 3), _fmt(r["recall"], dec),
            _fmt(r["recall_at_fpr"], dec), str(r["n_seeds"]),
        ])
    cap = str(cfg["tables"]["captions"].get("density", "Recall stratified by the number of attack frames per window."))
    lab = str(cfg["tables"]["labels"].get("density", "tab:density"))
    notes = ("Window labels follow \\texttt{any\\_frame}: a positive window may contain "
             "1--64 attack frames. Bins are defined on the exact count of attack frames; "
             "the last column uses the validation-calibrated 1\\% FPR threshold. "
             "The distribution of positives over bins is shown so that the aggregate MCC "
             "can be decomposed.")
    tex = to_latex(caption=cap, label=lab, colspec="@{}llrrrrr@{}",
                   header=header, rows=body, notes=notes)
    return _write(cfg, "table_density", rows, list(rows[0].keys()), tex)


#: 表格目录（顺序即论文出现顺序）
TABLE_FUNCS = {
    "t7_main": table_main,
    "t7_known": table_known,
    "t7_suppress": table_suppress,
    "t7b_operating": table_operating,
    "t7c_kd": table_kd,
    "t7d_threshold": table_threshold,
    "t14_dataset": table_dataset_specs,
    "t8_masquerade": table_masquerade,
    "t10_ablation": table_ablation,
    "t11_density": table_density,
    "t12_complexity": table_complexity,
    "t13_stats": table_stats,
    "t9_baselines": table_baselines,
}


def generate_all(cfg: Mapping[str, Any], *, only: Sequence[str] | None = None) -> list[dict]:
    """按 tables.enabled 生成全部表，返回索引行。"""
    if not bool(cfg["tables"]["enabled"]):
        return []
    rows: list[dict] = []
    for key, fn in TABLE_FUNCS.items():
        if only and key not in only:
            continue
        try:
            files = fn(cfg) or []
        except Exception as e:
            log.warning("表格 %s 生成失败：%s", key, e)
            files = []
        rows.append({"table": key, "files": "; ".join(p.name for p in files),
                     "generated": bool(files)})
    if rows:
        write_csv(table_dir(cfg) / "tables_index.csv", rows)
    return rows
