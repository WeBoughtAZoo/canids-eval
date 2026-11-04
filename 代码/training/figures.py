# -*- coding: utf-8 -*-
"""
training.figures —— SCI 论文图表自动生成
================================================================================
设计原则
--------
1. **图只读产物，不读模型**：所有图从 history.csv / metrics_*.json /
   predictions_*.npz / per_type_*.csv / threshold_curve_*.csv 重建，
   因此 `run_figures.py --run-dir <dir>` 可在不重训的情况下重绘全部图。
2. **一图一文件、文件名稳定**：`figXX_<slug>.{pdf,png}`，目录携带
   dataset/model/variant/seed 上下文（`figures/<dataset>/<model>/<variant>/`）。
3. **图内文字全英文**：投稿语言 + 字体兼容（DejaVu Sans 无中文字形）。
4. 每张图的主题、所属章节、尺寸、口径都登记在 `FIGURE_CATALOG`，
   并落地为 `figures_index.csv` —— 论文写作时按表引用，不靠记忆。

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from . import style as S
from .utils import get_logger, read_json, write_csv

log = get_logger("figures")

#: 模型键 → 论文里的显示名（图内文字用显示名，文件名仍用键）
MODEL_DISPLAY = {
    "fdtcn": "FD-TCN", "rgbnet": "FD-RGBNet",
    "lstm": "LSTM", "transformer": "Transformer",
}


def display_name(key: str) -> str:
    """`fdtcn` → `FD-TCN`；带变体后缀的 `fdtcn:Ab-1` → `FD-TCN (Ab-1)`。"""
    base, _, variant = str(key).partition(":")
    name = MODEL_DISPLAY.get(base, base.upper())
    return f"{name} ({variant})" if variant else name


# --------------------------------------------------------------------------- #
# 目录结构中的一张图
# --------------------------------------------------------------------------- #
@dataclass
class FigureSpec:
    id: str
    slug: str
    title_en: str
    title_zh: str
    section: str
    scope: str          # run | aggregate | schematic
    width: str          # single | double
    height: float
    fn: Callable[["RunArtifacts", Mapping[str, str]], list[Path]] | None = None


# --------------------------------------------------------------------------- #
# 产物读取器
# --------------------------------------------------------------------------- #
class RunArtifacts:
    """一个 run 目录的只读视图（图函数的唯一数据来源）。"""

    def __init__(self, run_dir: str | Path, cfg: Mapping[str, Any]):
        self.dir = Path(run_dir)
        self.cfg = cfg
        man = self.dir / "run_manifest.json"
        self.manifest: dict = read_json(man) if man.exists() else {}
        self.dataset = self.manifest.get("dataset") or self.dir.parents[2].name
        self.model = self.manifest.get("model") or self.dir.parents[1].name
        self.variant = self.manifest.get("variant") or self.dir.parents[0].name
        self.seed = int(self.manifest.get("seed", -1))
        self.fig_dir = self.dir / str(cfg["paths"]["figures_subdir"])
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    # -- 基础产物 -- #
    def history(self) -> list[dict]:
        import csv
        p = self.dir / "history.csv"
        if not p.exists():
            return []
        with open(p, "r", encoding="utf-8-sig", newline="") as fh:
            return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(fh)]

    def metrics(self, split: str = "test") -> dict:
        p = self.dir / f"metrics_{split}.json"
        return read_json(p) if p.exists() else {}

    def per_family(self, split: str = "test") -> list[dict]:
        return self._per_group(split, "per_family")

    def per_type(self, split: str = "test") -> list[dict]:
        return self._per_group(split, "per_type")

    def _per_group(self, split: str, kind: str) -> list[dict]:
        import csv
        p = self.dir / f"{kind}_{split}.csv"
        if not p.exists():
            return []
        with open(p, "r", encoding="utf-8-sig", newline="") as fh:
            return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(fh)]

    def curve(self, split: str = "val") -> dict[str, np.ndarray]:
        import csv
        p = self.dir / f"threshold_curve_{split}.csv"
        if not p.exists():
            return {}
        cols: dict[str, list[float]] = {}
        with open(p, "r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                for k, v in row.items():
                    cols.setdefault(k, []).append(float(v))
        return {k: np.asarray(v) for k, v in cols.items()}

    def predictions(self, split: str = "test") -> dict[str, np.ndarray]:
        p = self.dir / f"predictions_{split}.npz"
        if not p.exists():
            return {}
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def label_map(self) -> dict[str, int]:
        p = self.dir / "label_map.json"
        if not p.exists():
            return {}
        return dict(read_json(p).get("label_map", {}))

    def summary(self) -> dict:
        p = self.dir / "metrics_summary.json"
        return read_json(p) if p.exists() else {}

    def base(self, spec: FigureSpec) -> Path:
        return self.fig_dir / f"{spec.id}_{spec.slug}"


def _num(v: Any) -> Any:
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return v


# --------------------------------------------------------------------------- #
# 图 1–7：单 run 实证图
# --------------------------------------------------------------------------- #
def plt_patch(color: str, label: str, hatch: str | None = None):
    """图例色块（matplotlib.patches.Patch 的薄封装，支持黑白打印用的 hatch）。"""
    import matplotlib.patches as mpatches
    return mpatches.Patch(facecolor=color, edgecolor="black", linewidth=0.4,
                          hatch=hatch, label=label)


def fig01_training_curves(ctx: RunArtifacts, pal: Mapping[str, str]) -> list[Path]:
    """训练过程：损失 + 验证集 MCC / AUC-PR。"""
    hist = ctx.history()
    if not hist:
        return []
    ep = np.asarray([h["epoch"] for h in hist], dtype=float)
    fig, axes = S.new_figure(ctx.cfg, kind="double", nrows=1, ncols=2, height=2.4)

    ax = axes[0]
    ax.plot(ep, [h["train_loss"] for h in hist], color=pal["primary"], marker="o", ms=2.4, label="Train loss")
    if any(np.isfinite([h.get("val_loss", np.nan) for h in hist])):
        ax.plot(ep, [h.get("val_loss", np.nan) for h in hist], color=pal["secondary"],
                marker="s", ms=2.4, label="Validation loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Optimisation")
    ax.legend(frameon=False)
    S.panel_label(ax, "(a)")

    ax = axes[1]
    ax.plot(ep, [h["val_mcc_calibrated"] for h in hist], color=pal["primary"],
            marker="o", ms=2.4, label="Val MCC (calibrated $\\tau$)")
    ax.plot(ep, [h["val_auc_pr"] for h in hist], color=pal["tertiary"],
            marker="^", ms=2.4, label="Val AUC-PR")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_title("Validation performance")
    ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
    ax.legend(frameon=False, loc="lower right")
    S.panel_label(ax, "(b)")

    best = ctx.summary().get("training", {}).get("best_epoch")
    if best:
        for a in axes:
            a.axvline(best, color=pal["accent"], ls="--", lw=0.8, alpha=0.8)
    fig.tight_layout()
    return S.save(fig, ctx.base(FIG01), ctx.cfg)


def _unknown_index(ctx: RunArtifacts) -> int | None:
    """run 的 label_map 里 unknown 类的下标（无则 None）。"""
    try:
        lm = ctx.label_map() or {}
    except Exception:  # pragma: no cover
        return None
    for name, idx in lm.items():
        if str(name).endswith("unknown_attack"):
            return int(idx)
    return None


def fig02_pr_curve(ctx: RunArtifacts, pal: Mapping[str, str]) -> list[Path]:
    """测试集 Precision–Recall 曲线（含 prevalence 基线）。

    ★ 2026-10-27：若该 run 存在 zero-shot 家族（CTT 的 masquerade），
    额外给出 **known-attack 口径**的第二面板。两个面板的 prevalence 不同，
    这正是 v5 §4.5 双轨口径要暴露的事实；只有 full 时保持单面板不变。
    """
    pred = ctx.predictions("test")
    if not pred:
        return []
    from sklearn.metrics import average_precision_score, precision_recall_curve

    y = pred["label"].astype(int)
    p = pred["prob"]
    m = ctx.metrics("test")

    def _draw(ax, yy, pp, ap, prev, title):
        prec, rec, _ = precision_recall_curve(yy, pp)
        ax.plot(rec, prec, color=pal["primary"],
                label=f"{display_name(ctx.model)} (AP = {ap:.3f})")
        if np.isfinite(prev):
            ax.axhline(prev, color=pal["accent"], ls="--", lw=0.9,
                       label=f"Random (prevalence = {prev:.3f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title(title)
        ax.legend(frameon=False, loc="upper right")

    uid = _unknown_index(ctx)
    drop = ((y == 1) & (pred["y_type"] == uid)) if uid is not None else None
    has_known = bool(drop is not None and drop.any())

    if not has_known:
        fig, ax = S.new_figure(ctx.cfg, kind="single", height=2.5)
        _draw(ax, y, p, float(m.get("auc_pr", np.nan)),
              float(m.get("prevalence", np.nan)), "")
        fig.tight_layout()
        return S.save(fig, ctx.base(FIG02), ctx.cfg)

    keep = ~drop
    yk, pk = y[keep], p[keep]
    mk = ctx.metrics("test_known")
    ap_k = float(mk.get("auc_pr", average_precision_score(yk, pk)))
    prev_k = float(mk.get("prevalence", yk.mean()))

    fig, axes = S.new_figure(ctx.cfg, kind="double", nrows=1, ncols=2, height=2.45)
    _draw(axes[0], y, p, float(m.get("auc_pr", np.nan)),
          float(m.get("prevalence", np.nan)), "(a) full regime")
    _draw(axes[1], yk, pk, ap_k, prev_k,
          f"(b) known-attack regime\n({int(drop.sum()):,} zero-shot windows removed)")
    S.panel_label(axes[0], ""); S.panel_label(axes[1], "")
    fig.tight_layout()
    return S.save(fig, ctx.base(FIG02), ctx.cfg)


def fig03_roc_curve(ctx: RunArtifacts, pal: Mapping[str, str]) -> list[Path]:
    """测试集 ROC 曲线。"""
    pred = ctx.predictions("test")
    if not pred:
        return []
    from sklearn.metrics import roc_curve
    y, p = pred["label"].astype(int), pred["prob"]
    fpr, tpr, _ = roc_curve(y, p)
    auc = float(ctx.metrics("test").get("auc_roc", np.nan))

    fig, ax = S.new_figure(ctx.cfg, kind="single", height=2.5)
    ax.plot(fpr, tpr, color=pal["primary"], label=f"{display_name(ctx.model)} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color=pal["accent"], ls=":", lw=0.9, label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return S.save(fig, ctx.base(FIG03), ctx.cfg)


def fig04_confusion_binary(ctx: RunArtifacts, pal: Mapping[str, str]) -> list[Path]:
    """二分类混淆矩阵（计数 + 行归一化）。"""
    m = ctx.metrics("test")
    if not m:
        return []
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=float)
    row = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = S.new_figure(ctx.cfg, kind="double", nrows=1, ncols=2, height=2.3)
    for ax, mat, title, fmt in (
        (axes[0], cm, "Counts", "{:.0f}"),
        (axes[1], row, "Row-normalised", "{:.3f}"),
    ):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() if mat.max() > 0 else 1,
                       aspect="auto")
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        color="white" if val > mat.max() * 0.55 else "black", fontsize=8)
        ax.set_xticks([0, 1], ["Normal", "Attack"])
        ax.set_yticks([0, 1], ["Normal", "Attack"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    S.panel_label(axes[0], "(a)"); S.panel_label(axes[1], "(b)")
    fig.tight_layout()
    return S.save(fig, ctx.base(FIG04), ctx.cfg)


def fig05_confusion_types(ctx: RunArtifacts, pal: Mapping[str, str]) -> list[Path]:
    """攻击类型归因混淆矩阵（行归一化，仅统计真实攻击窗口）。"""
    pred = ctx.predictions("test")
    lm = ctx.label_map()
    if not pred or not lm:
        return []
    names = [k for k, _ in sorted(lm.items(), key=lambda kv: kv[1])]
    mask = pred["label"].astype(bool) & (pred["y_type"] > 0)
    if mask.sum() == 0:
        return []
    yt = pred["y_type"][mask].astype(int)
    yp = pred["type_pred"][mask].astype(int)
    present = sorted(set(yt.tolist()))
    idx = {c: i for i, c in enumerate(present)}
    cm = np.zeros((len(present), len(present)), dtype=float)
    for t, p_ in zip(yt, yp):
        if p_ in idx:
            cm[idx[t], idx[p_]] += 1
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = S.new_figure(ctx.cfg, kind="double", height=max(2.6, 0.32 * len(present)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    labels = [names[c] if c < len(names) else str(c) for c in present]
    ax.set_xticks(range(len(present)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(present)), labels)
    ax.set_xlabel("Predicted attack type"); ax.set_ylabel("True attack type")
    if len(present) <= 14:
        for i in range(len(present)):
            for j in range(len(present)):
                v = norm[i, j]
                if v > 0.005:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.0,
                            color="white" if v > 0.55 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Row-normalised rate")
    fig.tight_layout()
    return S.save(fig, ctx.base(FIG05), ctx.cfg)


def fig06_per_type_metrics(ctx: RunArtifacts, pal: Mapping[str, str]) -> list[Path]:
    """按**攻击家族**分层的召回率（masquerade 单列；含 test 正窗口数与可评估性分级）。"""
    rows = [r for r in ctx.per_family("test") if r.get("attack_type") != "overall"]
    if not rows:
        return []
    rows.sort(key=lambda r: float(r.get("n_positive_windows", 0)), reverse=True)
    names = [str(r["attack_type"]) for r in rows]
    vals = [float(r["recall"]) for r in rows]
    ns = [int(float(r["n_positive_windows"])) for r in rows]
    grades = [str(r["evaluability"]) for r in rows]
    color_of = {
        "sufficient": pal["primary"], "usable": pal["tertiary"],
        "insufficient": pal["quaternary"], "not_evaluable": pal["accent"],
        "zero-shot": pal["secondary"],
    }
    colors = [color_of.get(g, pal["accent"]) for g in grades]
    # 黑白打印可辨：每个等级额外用 hatch 区分（期刊常同时发行印刷版）
    hatch_of = {"sufficient": None, "usable": "//", "insufficient": "..",
                "not_evaluable": "xx", "zero-shot": "\\\\"}
    hatches = [hatch_of.get(g) for g in grades]

    fig, ax = S.new_figure(ctx.cfg, kind="double", height=max(2.6, 0.34 * len(rows)))
    ypos = np.arange(len(rows))
    bars = ax.barh(ypos, vals, color=colors, edgecolor="black", linewidth=0.4, height=0.62)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    for y, n, v in zip(ypos, ns, vals):
        ax.text(1.02, y, f"n={n}", va="center", ha="left", fontsize=7.0)
    ax.set_yticks(ypos, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.20)
    ax.set_xlabel("Recall at calibrated $\\tau$ (window level, by attack family)")
    handles = [plt_patch(color_of[g], label=g.replace("_", " "), hatch=hatch_of.get(g))
               for g in color_of]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncols=len(handles), frameon=False, title=None, columnspacing=1.0,
              handlelength=1.2)
    fig.tight_layout()
    return S.save(fig, ctx.base(FIG06), ctx.cfg)


def fig07_threshold_mcc(ctx: RunArtifacts, pal: Mapping[str, str]) -> list[Path]:
    """τ–MCC 标定曲线（MCC/F1 左轴，FPR 右轴），标注所用 τ。"""
    curve = ctx.curve("val") or ctx.curve("test")
    if not curve:
        return []
    fig, ax = S.new_figure(ctx.cfg, kind="single", height=2.5)
    ax.plot(curve["threshold"], curve["mcc"], color=pal["primary"], label="MCC")
    ax.plot(curve["threshold"], curve["f1"], color=pal["tertiary"], ls="--", label="F1")
    ax.set_xlabel("Decision threshold $\\tau$"); ax.set_ylabel("MCC / F1")
    ax2 = ax.twinx()
    ax2.plot(curve["threshold"], curve["fpr"], color=pal["secondary"], ls=":", label="FPR")
    ax2.set_ylabel("False positive rate")
    ax2.set_ylim(0, min(1.0, float(np.nanmax(curve["fpr"])) * 1.2 + 1e-6))
    ax2.spines["top"].set_visible(False)

    tau = ctx.summary().get("training", {}).get("best_threshold")
    if tau is None:
        tau = ctx.metrics("test").get("threshold")
    if tau is not None:
        ax.axvline(float(tau), color=pal["accent"], ls="--", lw=0.9)
        ax.annotate(f"$\\tau$ = {float(tau):.3f}", xy=(float(tau), ax.get_ylim()[1] * 0.92),
                    xytext=(4, 0), textcoords="offset points", fontsize=7, color=pal["accent"])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="lower center", ncols=3)
    fig.tight_layout()
    return S.save(fig, ctx.base(FIG07), ctx.cfg)


# --------------------------------------------------------------------------- #
# 图 10–11：方法示意图（不依赖训练结果）
# --------------------------------------------------------------------------- #
def fig10_encoding_schematic(ctx: RunArtifacts, pal: Mapping[str, str]) -> list[Path]:
    """统一帧级编码示意：16 维 V1 通道 + V2 字节图（含填充掩码）。"""
    import matplotlib.patches as mpatches
    cfg = ctx.cfg
    ch = list(cfg["data"]["v1"]["channels"])[:16]
    groups = [
        ("Temporal", ["dt_norm", "dt_dev", "period_stability"], pal["primary"]),
        ("Identifier", ["id_norm", "id_freq"], pal["tertiary"]),
        ("Length / padding", ["payload_len_norm", "pad_ratio"], pal["quaternary"]),
        ("Protocol flags", ["proto_flag", "brs_flag", "esi_flag"], pal["accent"]),
        ("Payload statistics", ["byte_mean", "byte_std", "byte_entropy",
                                 "nonzero_ratio", "payload_delta", "payload_u32"], "#8FAADC"),
    ]
    color = {n: c for _, names, c in groups for n in names}

    fig, axes = S.new_figure(cfg, kind="double", nrows=1, ncols=2, height=3.1,
                             gridspec_kw={"width_ratios": [1.0, 1.0]})

    # ---------------- (a) 16 维帧向量 ----------------
    ax = axes[0]
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 17.6)
    ax.axis("off")
    ax.set_title("(a) V1: 16-dimensional frame vector", loc="left",
                 fontsize=8, fontweight="bold", pad=4)
    for i, name in enumerate(ch):
        y = 16.0 - i * 0.95
        ax.add_patch(mpatches.FancyBboxPatch((0.03, y - 0.33), 0.085, 0.66,
                                             boxstyle="round,pad=0.004,rounding_size=0.02",
                                             fc=color.get(name, pal["accent"]), ec="black", lw=0.4))
        ax.text(0.0725, y, f"{i}", ha="center", va="center", fontsize=6.8, color="white")
        ax.text(0.14, y, name, ha="left", va="center", fontsize=7)
    for k, (gname, _, c) in enumerate(groups):
        y = 14.6 - k * 1.25
        ax.add_patch(mpatches.Rectangle((0.60, y - 0.20), 0.055, 0.40, fc=c, ec="black", lw=0.4))
        ax.text(0.675, y, gname, fontsize=6.8, va="center")

    # ---------------- (b) 字节图 ----------------
    ax = axes[1]
    ax.set_xlim(-1.55, 9.9)
    ax.set_ylim(-1.35, 9.9)
    ax.axis("off")
    ax.set_title("(b) V2: byte map $X \\in \\{0,1\\}^{8 \\times T \\times 8}$", loc="left",
                 fontsize=8, fontweight="bold", pad=4)
    for c in range(8):
        for b in range(8):
            is_mask = c >= 4
            ax.add_patch(mpatches.Rectangle((b + 0.04, 8 - c + 0.04), 0.92, 0.92,
                                            fc="#FFFFFF" if is_mask else "#DCE6F1",
                                            ec="black", lw=0.4,
                                            hatch="///" if is_mask else None))
        label = f"mask {c - 4}" if c >= 4 else f"data {c}"
        ax.text(0.0, 8.5 - c, label, ha="right", va="center", fontsize=6.8)
    ax.text(4.0, 9.42, "byte index 0–7", ha="center", fontsize=6.8)
    # 数据/掩码通道括号
    ax.annotate("", xy=(8.35, 8.0), xytext=(8.35, 5.0),
                arrowprops=dict(arrowstyle="<->", lw=0.7))
    ax.text(8.55, 6.5, "data\nch 0–3", fontsize=6.8, va="center")
    ax.annotate("", xy=(8.35, 4.0), xytext=(8.35, 1.0),
                arrowprops=dict(arrowstyle="<->", lw=0.7))
    ax.text(8.55, 2.5, "mask\nch 4–7", fontsize=6.8, va="center")
    ax.text(4.0, -0.85, "$\\times T = 64$ frames   |   payload padded / truncated to 32 bytes",
            ha="center", fontsize=6.8)
    fig.tight_layout()
    return S.save(fig, ctx.base(FIG10), ctx.cfg)


def fig11_fdtcn_structure(ctx: RunArtifacts, pal: Mapping[str, str]) -> list[Path]:
    """FD-TCN 结构图：4 个因果膨胀块 + 双头 + 感受野标注。"""
    import matplotlib.patches as mpatches
    cfg = ctx.cfg
    mv = cfg["model"]["variants"]["fdtcn"]
    dil = list(mv["dilations"])
    chans = list(mv["block_channels"])
    proj = int(mv["proj_channels"])

    fig, ax = S.new_figure(cfg, kind="double", height=3.0)
    ax.set_xlim(0, 15.6); ax.set_ylim(0, 6.2); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=6.8, tc="black"):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                                             boxstyle="round,pad=0.04,rounding_size=0.12",
                                             fc=fc, ec="black", lw=0.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black", shrinkA=0, shrinkB=0))

    y = 3.3
    box(0.05, y, 1.4, 0.9, "Input\n$64\\times16$", "#F2F2F2")
    arrow(1.45, y + 0.45, 1.95, y + 0.45)
    box(1.95, y, 1.4, 0.9, f"Conv1d 1×1\n16→{proj}", pal["band"])
    x = 3.65
    for i, (d, c) in enumerate(zip(dil, chans)):
        arrow(x - 0.3, y + 0.45, x, y + 0.45)
        box(x, y, 1.55, 0.9, f"TCN block {i + 1}\nd={d}, {c} ch", pal["band"])
        ax.text(x + 0.78, y - 0.30, f"RF {1 + 2 * 2 * sum(dil[: i + 1])}", ha="center", fontsize=6)
        x += 1.85
    arrow(x - 0.3, y + 0.45, x, y + 0.45)
    box(x, y, 1.35, 0.9, "Aggregate\nlast‖mean", pal["band"])
    arrow(x + 1.35, y + 0.45, x + 1.75, y + 0.45)
    box(x + 1.75, y + 0.55, 1.55, 0.62, "Head A: binary\n→ $p(\\mathrm{attack})$", pal["primary"], tc="white")
    box(x + 1.75, y - 0.25, 1.55, 0.62, "Head B: type\n→ $K$ classes", pal["tertiary"], tc="white")

    # 单块内部
    ax.text(0.05, 2.55, "Inside each TCN block", fontsize=7.2, fontweight="bold")
    inner = ["Causal\nConv1d", "BN", "ReLU", "Dropout\n0.2", "SE\n$r$=16", "+ residual"]
    x0 = 0.05
    for i, t in enumerate(inner):
        w = 1.55 if i == 0 else 1.25
        box(x0, 1.55, w, 0.75, t, "#FFFFFF" if i < 5 else pal["band"], fs=6.4)
        if i < len(inner) - 1:
            arrow(x0 + w, 1.93, x0 + w + 0.22, 1.93)
        x0 += w + 0.22
    ax.text(0.05, 0.95,
            "Causal: left-pad only ($k-1$)$d$  →  no future leakage.   "
            "Total receptive field = 1 + 2$\\sum (k-1)d$ = 61 frames (window $T$ = 64).",
            fontsize=7.0)
    ax.text(0.05, 0.55,
            "Residual: 1×1 Conv when channel width changes, identity otherwise; "
            "aggregate = concat(last step, mean pool) → $2C$.",
            fontsize=7.0)
    fig.tight_layout()
    return S.save(fig, ctx.base(FIG11), ctx.cfg)


# --------------------------------------------------------------------------- #
# 图 13–14：可选（蒸馏 / 时延），产物缺失则跳过
# --------------------------------------------------------------------------- #
def fig13_rule_complexity(ctx: RunArtifacts, pal: Mapping[str, str]) -> list[Path]:
    """KD-Logic：蒸馏 vs 直接训练的保真度 / 规则复杂度对比。"""
    p = ctx.dir / "kd" / "kd_results.json"
    if not p.exists():
        return []
    d = read_json(p)
    fig, axes = S.new_figure(ctx.cfg, kind="double", nrows=1, ncols=2, height=2.3)
    ax = axes[0]
    names = ["Distilled\nCART", "Direct\nCART"]
    mcc = [d.get("distilled", {}).get("mcc", np.nan), d.get("direct", {}).get("mcc", np.nan)]
    fid = [d.get("distilled", {}).get("fidelity", np.nan), 1.0]
    x = np.arange(2)
    ax.bar(x - 0.18, mcc, 0.34, color=pal["primary"], label="MCC")
    ax.bar(x + 0.18, fid, 0.34, color=pal["tertiary"], label="Fidelity")
    ax.set_xticks(x, names); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.set_title("Agreement with teacher")
    S.panel_label(ax, "(a)")

    ax = axes[1]
    keys = ["n_rules", "depth"]
    labels = ["Rules", "Depth"]
    vals = [d.get("distilled", {}).get("n_rules", 0), d.get("distilled", {}).get("depth", 0)]
    ax.bar(labels, vals, color=pal["secondary"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=7)
    ax.set_title(f"Rule complexity ({d.get('distilled', {}).get('size_kb', 0):.1f} KB)")
    S.panel_label(ax, "(b)")
    fig.tight_layout()
    return S.save(fig, ctx.base(FIG13), ctx.cfg)


def fig14_latency_distribution(ctx: RunArtifacts, pal: Mapping[str, str]) -> list[Path]:
    """部署时延分布（P50/P95/P99 × 线程数）；来自 latency/latency.json。

    ★ 2026-10-27：与 fig15 一致，优先读批次级 latency.json，再退回 run 级。
    """
    cands = [batch_root(ctx.cfg) / "latency" / "latency.json",
             ctx.dir / "latency" / "latency.json"]
    p = next((c for c in cands if c.exists()), None)
    if p is None:
        return []
    d = read_json(p)
    rows = d.get("results", [])
    if not rows:
        return []
    threads = [str(r["threads"]) for r in rows]
    x = np.arange(len(rows))
    fig, ax = S.new_figure(ctx.cfg, kind="single", height=2.5)
    for k, (label, color, off) in enumerate([
        ("p50", pal["primary"], -0.25), ("p95", pal["tertiary"], 0.0), ("p99", pal["secondary"], 0.25)
    ]):
        ax.bar(x + off, [r[label] for r in rows], 0.24, color=color, label=label.upper())
    ax.set_xticks(x, threads)
    ax.set_xlabel("Threads"); ax.set_ylabel("Latency (ms)")
    ax.legend(frameon=False, ncols=3)
    fig.tight_layout()
    return S.save(fig, ctx.base(FIG14), ctx.cfg)


def fig15_accuracy_latency(ctx: RunArtifacts, pal: Mapping[str, str]) -> list[Path]:
    """精度–时延 Pareto（v4 Δ9 / §5.6）。

    数据来源：`latency/latency.json`（RPi5 实测），结构见 README。
    必含外部坐标（CANDIDS = Jetson NX、MR-TCN = FPGA），且**必须标注平台**，
    否则会被误读为"同平台横比"（v4 §16 终检第 17 项）。

    ★ 2026-10-27：优先读**批次级** `<batch_root>/latency/latency.json`（跨模型 Pareto
      只需一份测量文件），读不到再退回 run 级，保持与旧产物兼容。
    """
    cands = [batch_root(ctx.cfg) / "latency" / "latency.json",
             ctx.dir / "latency" / "latency.json"]
    p = next((c for c in cands if c.exists()), None)
    if p is None:
        return []
    d = read_json(p)
    ours = d.get("results", [])
    ext = d.get("external", [])
    if not ours and not ext:
        return []

    fig, ax = S.new_figure(ctx.cfg, kind="double", height=2.8)
    mcc_default = ctx.metrics("test").get("mcc")

    for r in ours:
        lat = float(r.get("p50_ms") or r.get("latency_ms") or float("nan"))
        acc = float(r.get("mcc", mcc_default if mcc_default is not None else float("nan")))
        prec = str(r.get("precision", "fp32")).lower()
        marker = "o" if prec.startswith("fp32") else "s"
        color = pal["primary"] if marker == "o" else pal["tertiary"]
        ax.scatter(lat, acc, marker=marker, s=46, color=color,
                   edgecolor="black", linewidth=0.4, zorder=3)
        ax.annotate(f"{r.get('threads', '?')} thr / {prec.upper()}",
                    (lat, acc), textcoords="offset points", xytext=(5, 4), fontsize=6.8)

    for r in ext:
        lat = float(r.get("latency_ms", float("nan")))
        acc = float(r.get("mcc", r.get("accuracy", float("nan"))))
        ax.scatter(lat, acc, marker="D", s=52, facecolor="white",
                   edgecolor=pal["secondary"], linewidth=1.0, zorder=3)
        ax.annotate(f"{r.get('name', 'ref')} ({r.get('platform', 'n/a')})",
                    (lat, acc), textcoords="offset points", xytext=(6, -10),
                    fontsize=6.8, color=pal["secondary"])

    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms, log scale)")
    ax.set_ylabel("Test MCC")
    ax.grid(True, which="both", alpha=0.25)
    ax.margins(x=0.18, y=0.15)
    import matplotlib.lines as mlines
    legend = [
        mlines.Line2D([], [], marker="o", ls="", color=pal["primary"], label="This work, FP32"),
        mlines.Line2D([], [], marker="s", ls="", color=pal["tertiary"], label="This work, INT8"),
        mlines.Line2D([], [], marker="D", ls="", mfc="white", mec=pal["secondary"],
                      label="Published (different platform)"),
    ]
    ax.legend(handles=legend, frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    return S.save(fig, ctx.base(FIG15), ctx.cfg)


# --------------------------------------------------------------------------- #
# 图 8 / 9 / 12：跨 run 汇总图
# --------------------------------------------------------------------------- #
def batch_root(cfg: Mapping[str, Any]) -> Path:
    """一个实验批次（run_name）的根目录：output_root/<run_name>。

    汇总图与 runs_index 都收敛在批次内，避免把 smoke 与正式实验混在同一张 Pareto 图上。
    """
    name = str(cfg["project"]["run_name"] or "run")
    return Path(cfg["_resolved"]["output_root"]) / name


def discover_runs(output_root: str | Path, cfg: Mapping[str, Any]) -> list[RunArtifacts]:
    """扫描 output_root 下所有含 run_manifest.json 的 run 目录。"""
    root = Path(output_root)
    out: list[RunArtifacts] = []
    for man in sorted(root.rglob("run_manifest.json")):
        try:
            out.append(RunArtifacts(man.parent, cfg))
        except Exception as e:  # pragma: no cover
            log.warning("跳过 %s：%s", man.parent, e)
    return out


def _agg_metrics(runs: Sequence[RunArtifacts]) -> dict[str, list[float]]:
    acc: dict[str, list[float]] = {}
    for r in runs:
        m = r.metrics("test")
        for k in ("mcc", "auc_pr", "f1", "recall", "fpr"):
            v = m.get(k)
            if isinstance(v, (int, float)) and np.isfinite(v):
                acc.setdefault(k, []).append(float(v))
    return acc


def fig08_seed_stability(ctx: RunArtifacts, pal: Mapping[str, str]) -> list[Path]:
    """跨 seed 稳定性：同一 (dataset, model, variant) 的 mean ± std。

    汇总入口没有具体 run 身份，此时键取自配置（主数据集 / 主模型 / 默认变体）。
    """
    if ctx.dataset == "_aggregate":
        key = (str(ctx.cfg["data"]["datasets"][0]), str(ctx.cfg["model"]["name"]),
               str(ctx.cfg["ablation"]["default_variant"]))
    else:
        key = (ctx.dataset, ctx.model, ctx.variant)
    runs = [r for r in discover_runs(batch_root(ctx.cfg), ctx.cfg)
            if (r.dataset, r.model, r.variant) == key]
    if len(runs) < 2:
        return []
    agg = _agg_metrics(runs)
    if not agg:
        return []
    metrics = ["mcc", "auc_pr", "f1", "fpr"]
    fig, ax = S.new_figure(ctx.cfg, kind="single", height=2.5)
    x = np.arange(len(metrics))
    means = [float(np.mean(agg.get(m, [np.nan]))) for m in metrics]
    stds = [float(np.std(agg.get(m, [np.nan]), ddof=1)) if len(agg.get(m, [])) > 1 else 0.0
            for m in metrics]
    ax.bar(x, means, 0.55, yerr=stds, capsize=3, color=pal["primary"], ecolor="black")
    for i, (mu, sd) in enumerate(zip(means, stds)):
        ax.text(i, mu + sd + 0.03, f"{mu:.3f}\n±{sd:.3f}", ha="center", va="bottom",
                fontsize=6.8)
    ax.set_xticks(x, [m.upper() for m in metrics])
    ax.set_ylim(0, 1.24)
    ax.set_title(f"{len(runs)} seeds (mean $\\pm$ std)")
    fig.tight_layout()
    return S.save(fig, ctx.base(FIG08), ctx.cfg)


def fig09_model_complexity(ctx: RunArtifacts, pal: Mapping[str, str]) -> list[Path]:
    """精度–体积 Pareto：参数量 vs 测试 MCC（跨模型/跨数据集）。"""
    runs = discover_runs(batch_root(ctx.cfg), ctx.cfg)
    pts = []
    for r in runs:
        m = r.metrics("test")
        ms = r.manifest.get("model_stats", {})
        n_par = ms.get("n_params")
        if n_par and isinstance(m.get("mcc"), (int, float)):
            pts.append((float(n_par) / 1e3, float(m["mcc"]), r))
    if len(pts) < 2:
        return []
    fig, ax = S.new_figure(ctx.cfg, kind="single", height=2.6)
    seen = {}
    model_order = sorted({p[2].model for p in pts})
    colors = [pal["primary"], pal["tertiary"], pal["quaternary"], pal["secondary"]]
    for n_k, mcc, r in pts:
        key = r.model
        color = colors[model_order.index(key) % len(colors)]
        ax.scatter(n_k, mcc, s=36, color=color, edgecolor="black", linewidth=0.4, zorder=3)
        ax.annotate(display_name(key), (n_k, mcc), textcoords="offset points",
                    xytext=(4, 4), fontsize=7.0)
        seen[key] = True
    ax.set_xlabel("Parameters (K)"); ax.set_ylabel("Test MCC")
    ax.margins(x=0.12, y=0.12)
    fig.tight_layout()
    return S.save(fig, ctx.base(FIG09), ctx.cfg)


def fig12_attack_type_heatmap(ctx: RunArtifacts, pal: Mapping[str, str]) -> list[Path]:
    """攻击家族 × 模型 的召回率热力图（跨 run 汇总，masquerade 单列）。"""
    runs = discover_runs(batch_root(ctx.cfg), ctx.cfg)
    table: dict[str, dict[str, float]] = {}
    models: list[str] = []
    for r in runs:
        for row in r.per_family("test"):
            at = str(row.get("attack_type"))
            if at == "overall":
                continue
            table.setdefault(at, {})
            key = r.model if r.variant == "full" else f"{r.model}:{r.variant}"
            if key not in models:
                models.append(key)
            table[at][key] = float(row.get("recall", np.nan))
    if len(table) < 2 or len(models) < 2:
        return []
    types = sorted(table)
    mat = np.full((len(types), len(models)), np.nan)
    for i, at in enumerate(types):
        for j, mo in enumerate(models):
            mat[i, j] = table[at].get(mo, np.nan)

    fig, ax = S.new_figure(ctx.cfg, kind="double", height=max(2.6, 0.30 * len(types)))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(models)), [display_name(m) for m in models],
                  rotation=30, ha="right")
    ax.set_yticks(range(len(types)), types)
    for i in range(len(types)):
        for j in range(len(models)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.8,
                        color="white" if mat[i, j] > 0.55 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Recall at $\\tau$")
    fig.tight_layout()
    return S.save(fig, ctx.base(FIG12), ctx.cfg)


# --------------------------------------------------------------------------- #
# 图目录
# --------------------------------------------------------------------------- #
FIG01 = FigureSpec("fig01", "training_curves", "Training and validation curves",
                   "训练过程曲线（损失 / MCC / AUC-PR）", "Results 5.2", "run", "double", 2.4,
                   fig01_training_curves)
FIG02 = FigureSpec("fig02", "pr_curve", "Precision-recall curve on the test split",
                   "测试集 PR 曲线（含 prevalence 基线）", "Results 5.3", "run", "single", 2.5,
                   fig02_pr_curve)
FIG03 = FigureSpec("fig03", "roc_curve", "ROC curve on the test split",
                   "测试集 ROC 曲线", "Appendix", "run", "single", 2.5, fig03_roc_curve)
FIG04 = FigureSpec("fig04", "confusion_binary", "Binary confusion matrix",
                   "二分类混淆矩阵（计数 / 行归一化）", "Results 5.3", "run", "double", 2.3,
                   fig04_confusion_binary)
FIG05 = FigureSpec("fig05", "confusion_types", "Attack-type confusion matrix",
                   "攻击类型归因混淆矩阵", "Results 5.5", "run", "double", 2.8,
                   fig05_confusion_types)
FIG06 = FigureSpec("fig06", "per_type_recall", "Per-attack-type recall with evaluability grades",
                   "分层召回率与可评估性分级", "Results 5.4", "run", "double", 3.0,
                   fig06_per_type_metrics)
FIG07 = FigureSpec("fig07", "threshold_mcc", "Threshold calibration curve (MCC / F1 / FPR)",
                   "阈值标定曲线 τ–MCC", "Results 5.3", "run", "single", 2.5,
                   fig07_threshold_mcc)
FIG08 = FigureSpec("fig08", "seed_stability", "Seed-wise variability (mean ± std)",
                   "跨 seed 稳定性", "Results 5.2", "aggregate", "single", 2.5,
                   fig08_seed_stability)
FIG09 = FigureSpec("fig09", "accuracy_size_pareto", "Accuracy vs model size Pareto front",
                   "精度–体积 Pareto 前沿", "Discussion 6.2", "aggregate", "single", 2.6,
                   fig09_model_complexity)
FIG10 = FigureSpec("fig10", "encoding_schematic", "Unified frame encoding (V1 + V2 with padding mask)",
                   "统一帧级编码示意（16 维 + 字节图掩码）", "Method 3.3", "schematic", "double",
                   2.9, fig10_encoding_schematic)
FIG11 = FigureSpec("fig11", "fdtcn_structure", "FD-TCN architecture (causal dilated blocks, dual heads)",
                   "FD-TCN 结构图（膨胀率 / 残差 / SE / 双头）", "Method 3.4", "schematic", "double",
                   3.0, fig11_fdtcn_structure)
FIG12 = FigureSpec("fig12", "attack_type_heatmap", "Per-attack-type recall across backbones",
                   "攻击类型 × 主干性能热力图", "Discussion 6.3", "aggregate", "double", 3.0,
                   fig12_attack_type_heatmap)
FIG13 = FigureSpec("fig13", "rule_complexity", "KD-Logic fidelity and rule complexity",
                   "蒸馏规则示例与复杂度", "Results 5.7", "run", "double", 2.3,
                   fig13_rule_complexity)
FIG14 = FigureSpec("fig14", "latency_distribution", "Inference latency distribution (P50/P95/P99)",
                   "推理时延分布", "Results 5.6", "run", "single", 2.5,
                   fig14_latency_distribution)
FIG15 = FigureSpec("fig15", "accuracy_latency_pareto",
                   "Accuracy-latency Pareto front with published platform coordinates",
                   "精度–时延 Pareto（含 CANDIDS / MR-TCN 平台坐标）", "Results 5.6",
                   "run", "double", 2.8, fig15_accuracy_latency)

#: 通道归因图的固定行序（先编码类，后训练类；未列出的排最后）
ATTR_VARIANT_ORDER = ("Ab-1", "Ab-1b", "Ab-2", "Ab-12", "Ab-6",
                      "Ab-13-RF13", "Ab-13-RF29", "Ab-13-RF125", "Ab-14")


def fig16_channel_attribution(ctx: RunArtifacts, pal: Mapping[str, str]) -> list[Path]:
    """通道归因森林图：各消融变体相对同模型 full 的 ΔMCC（mean ± std，跨 seed）。

    ★ 2026-10-27 新增（v5 §14 · 5.4）。不占用既有的 fig06（per-type recall），
    单独成图，避免把"类型分解"和"通道归因"挤在一张图里。

    纪律：n≥3 用实心、n=1 用空心并标注 —— 与 v5 §0.2 的"n=1 仅作探索性"一致。
    """
    runs = discover_runs(batch_root(ctx.cfg), ctx.cfg)
    groups: dict[tuple[str, str, str], list[RunArtifacts]] = {}
    for r in runs:
        groups.setdefault((r.dataset, r.model, r.variant), []).append(r)

    def _mcc(rs: Sequence[RunArtifacts]) -> tuple[float, float, int] | None:
        v = [m.get("mcc") for m in (x.metrics("test") for x in rs)]
        v = [float(x) for x in v if isinstance(x, (int, float)) and np.isfinite(x)]
        if not v:
            return None
        return (float(np.mean(v)),
                float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))

    datasets = sorted({k[0] for k in groups})
    pairs: list[tuple[str, str, str, float, float, int]] = []
    for (ds, model, variant), rs in groups.items():
        if variant == "full":
            continue
        base = groups.get((ds, model, "full"))
        if not base:
            continue
        a, b = _mcc(rs), _mcc(base)
        if a is None or b is None:
            continue
        pairs.append((ds, model, variant, a[0] - b[0], a[1], a[2]))
    if not pairs:
        return []

    order = {v: i for i, v in enumerate(ATTR_VARIANT_ORDER)}
    pairs.sort(key=lambda t: (t[0], order.get(t[2], 99), t[1]))
    ncol = max(1, min(3, len(datasets)))
    fig, axes = S.new_figure(ctx.cfg, kind="double", nrows=1, ncols=ncol, height=3.2)
    axes = np.atleast_1d(axes)
    show = {k: str(v) for k, v in (ctx.cfg["tables"].get("dataset_display") or {}).items()}
    for ax, ds in zip(axes, datasets):
        sel = [p for p in pairs if p[0] == ds]
        if not sel:
            continue
        vlabels = {str(k): str(v) for k, v in
                   ((ctx.cfg["tables"].get("main") or {}).get("variant_labels") or {}).items()}
        labels = [vlabels.get(p[2], p[2]) for p in sel]   # 变体名唯一，无需前缀模型名
        ys = np.arange(len(sel))[::-1]
        vals = np.array([100.0 * p[3] for p in sel])
        errs = np.array([100.0 * p[4] for p in sel])
        span = float(np.max(np.abs(vals) + errs)) if len(sel) else 1.0
        span = max(span, 0.5)
        for yy, p, lab in zip(ys, sel, labels):
            delta, std, n = 100.0 * p[3], 100.0 * p[4], p[5]
            filled = n >= 3
            ax.barh(yy, delta, height=0.62,
                    color=(pal["primary"] if filled else "white"),
                    edgecolor=pal["primary"], linewidth=0.9,
                    alpha=0.85 if filled else 1.0)
            if std > 0:
                ax.errorbar(delta, yy, xerr=std, fmt="none",
                            ecolor=pal["accent"], lw=0.9, capsize=2)
            ax.annotate(f"n={n}", (delta, yy), textcoords="offset points",
                        xytext=(4 if delta >= 0 else -4, 0),
                        va="center", ha="left" if delta >= 0 else "right",
                        fontsize=6.0, color="#666")
        ax.axvline(0.0, color="#888", lw=0.8)
        ax.set_yticks(ys, labels, fontsize=6.8)
        ax.set_xlim(-span * 1.45, span * 1.45)   # 给 n= 标注留出空间
        ax.set_xlabel(r"$\Delta$MCC vs. full (pt)", fontsize=8)
        ax.set_title(show.get(ds, ds), fontsize=8.5)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.3)
    for ax in axes[len(datasets):]:
        ax.axis("off")
    fig.subplots_adjust(wspace=0.55, left=0.12, right=0.98, top=0.88, bottom=0.16)
    return S.save(fig, ctx.base(FIG16), ctx.cfg)


FIG16 = FigureSpec("fig16", "channel_attribution",
                   "Channel attribution: $\\Delta$MCC of each ablation versus the full encoding",
                   "通道归因（相对满通道编码的 ΔMCC，含跨 seed 误差棒）", "Results 5.4",
                   "aggregate", "double", 7.16, fig16_channel_attribution)


# --------------------------------------------------------------------------- #
# fig17 攻击帧密度分层（v6 新增，§14 · 5.9）
# --------------------------------------------------------------------------- #
def fig17_density_recall(ctx: RunArtifacts, pal: Mapping[str, str]) -> list[Path]:
    """召回率随窗内攻击帧数变化的曲线（跨 seed 平均），每个数据集一个 panel。

    ★ 2026-10-29 新增（v6 §14 · 5.9）。数据源是 `tables/table_density.csv`
      （由 `tables.table_density` 生成，跨 seed 聚合）。

    为什么值得单独成图：MCC / 召回把"窗内 1 个攻击帧"与"64 个攻击帧"混成一个数。
    按密度拆开后，CTT 上架构间的差距几乎全部落在 ≤8 帧的箱里，
    而 ROAD 呈现相反趋势——这张图是"低密度 × 载荷局部性"这一解释的直接证据，
    也是"密度本身不是充分条件"这一反例的载体。
    """
    csv_path = batch_root(ctx.cfg) / "tables" / "table_density.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []

    def _sort_key(b: str) -> int:
        try:
            return int(str(b).split("-")[0])
        except Exception:                                     # noqa: BLE001
            return 0

    datasets = sorted({r["dataset"] for r in rows})
    ncol = max(1, min(3, len(datasets)))
    fig, axes = S.new_figure(ctx.cfg, kind="double", nrows=1, ncols=ncol, height=3.0)
    axes = np.atleast_1d(axes)
    show = {k: str(v) for k, v in (ctx.cfg["tables"].get("dataset_display") or {}).items()}
    colors = [pal["primary"], pal["secondary"], pal["tertiary"], pal["quaternary"], pal["accent"]]
    markers = ["o", "s", "^", "D", "v"]

    for ax, ds in zip(axes, datasets):
        sub = [r for r in rows if r["dataset"] == ds]
        bins = sorted({r["density_bin"] for r in sub}, key=_sort_key)
        if not bins:
            continue
        xs = np.arange(len(bins))
        # 正例占比（取自任一行，各模型相同）
        frac = {}
        for r in sub:
            frac.setdefault(r["density_bin"], float(r.get("frac_of_positives") or 0.0))
        labels = [f"{b}\n{100 * frac.get(b, 0.0):.0f}%" for b in bins]

        models = sorted({r["display"] for r in sub})
        # 让主模型排前面，图例顺序稳定
        for i, m in enumerate(models):
            ys = []
            for b in bins:
                hit = [r for r in sub if r["display"] == m and r["density_bin"] == b]
                ys.append(float(hit[0]["recall"]) if hit else np.nan)
            ax.plot(xs, ys, marker=markers[i % len(markers)], ms=3.4, lw=1.1,
                    color=colors[i % len(colors)], label=m)
        ax.set_xticks(xs, labels, fontsize=6.4)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("Attack frames per window (% of positives)", fontsize=7.2)
        if ax is axes[0]:
            ax.set_ylabel("Recall", fontsize=8)
        ax.set_title(show.get(ds, ds), fontsize=8.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.3)

    for ax in axes[len(datasets):]:
        ax.axis("off")
    handles, labs = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labs, loc="lower center", ncol=min(5, len(labs)),
                   fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(wspace=0.24, left=0.08, right=0.99, top=0.86, bottom=0.30)
    return S.save(fig, ctx.base(FIG17), ctx.cfg)


FIG17 = FigureSpec("fig17", "density_recall",
                   "Recall stratified by the number of attack frames inside a positive window",
                   "攻击帧密度分层召回（低密度窗口是架构差距的主要来源）", "Results 5.9",
                   "aggregate", "double", 7.16, fig17_density_recall)

#: 顺序即论文出现顺序；`figures.ids` 可裁剪
FIGURE_CATALOG: tuple[FigureSpec, ...] = (
    FIG01, FIG02, FIG03, FIG04, FIG05, FIG06, FIG07, FIG08,
    FIG09, FIG10, FIG11, FIG12, FIG13, FIG14, FIG15, FIG16, FIG17,
)

#: 论文直接引用的图（fig03 为附录）→ 供 README / 清单排序
MAIN_FIGURES: tuple[str, ...] = ("fig01", "fig02", "fig04", "fig05", "fig06",
                                 "fig07", "fig08", "fig09", "fig10", "fig11", "fig12")


# --------------------------------------------------------------------------- #
# 对外入口
# --------------------------------------------------------------------------- #
def generate_for_run(run_dir: str | Path, cfg: Mapping[str, Any],
                     *, ids: Sequence[str] | None = None) -> list[dict]:
    """生成该 run 可产出的全部图，返回图表索引行。"""
    if not bool(cfg["figures"]["enabled"]):
        return []
    pal = S.apply_style(cfg)
    ctx = RunArtifacts(run_dir, cfg)
    wanted = set(ids or cfg["figures"]["ids"])
    known = {s.id for s in FIGURE_CATALOG}
    unknown = sorted(wanted - known)
    if unknown:
        log.warning("figures.ids 含未知编号（已忽略）：%s；可选 %s",
                    unknown, sorted(known))
    rows: list[dict] = []
    for spec in FIGURE_CATALOG:
        # 跨 run 汇总图只在 generate_aggregate 里出，避免写到单个 run 目录
        if spec.id not in wanted or spec.fn is None or spec.scope == "aggregate":
            continue
        try:
            files = spec.fn(ctx, pal)
        except Exception as e:  # 单张图失败不影响其余图
            log.warning("图表 %s 生成失败：%s", spec.id, e)
            files = []
        rows.append({
            "figure_id": spec.id, "slug": spec.slug,
            "title_en": spec.title_en, "title_zh": spec.title_zh,
            "section": spec.section, "scope": spec.scope, "width": spec.width,
            "formats": "/".join(cfg["figures"]["formats"]),
            "files": "; ".join(str(Path(f).relative_to(ctx.fig_dir)) for f in files),
            "generated": bool(files),
        })
        if files:
            log.info("图表 %s 已保存：%s", spec.id, ", ".join(Path(f).name for f in files))
    if rows:
        write_csv(ctx.fig_dir / "figures_index.csv", rows)
    return rows


def generate_aggregate(cfg: Mapping[str, Any], *, ids: Sequence[str] | None = None) -> list[dict]:
    """跨 run 汇总图（写到 <output_root>/_aggregate/figures/）。"""
    if not bool(cfg["figures"]["enabled"]):
        return []
    pal = S.apply_style(cfg)
    out_dir = batch_root(cfg) / "_aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    # 用一个虚拟 run 承载样式与路径
    ctx = RunArtifacts.__new__(RunArtifacts)
    ctx.dir = out_dir
    ctx.cfg = cfg
    ctx.manifest = {"dataset": "_aggregate", "model": "_aggregate", "variant": "_aggregate", "seed": -1}
    ctx.dataset, ctx.model, ctx.variant, ctx.seed = "_aggregate", "_aggregate", "_aggregate", -1
    ctx.fig_dir = out_dir / str(cfg["paths"]["figures_subdir"])
    ctx.fig_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(ids or cfg["figures"]["ids"])
    rows: list[dict] = []
    for spec in FIGURE_CATALOG:
        if spec.id not in wanted or spec.scope != "aggregate" or spec.fn is None:
            continue
        try:
            files = spec.fn(ctx, pal)
        except Exception as e:
            log.warning("汇总图 %s 生成失败：%s", spec.id, e)
            files = []
        rows.append({
            "figure_id": spec.id, "slug": spec.slug,
            "title_en": spec.title_en, "title_zh": spec.title_zh,
            "section": spec.section, "scope": spec.scope, "width": spec.width,
            "formats": "/".join(cfg["figures"]["formats"]),
            "files": "; ".join(str(Path(f).relative_to(ctx.fig_dir)) for f in files),
            "generated": bool(files),
        })
        if files:
            log.info("汇总图 %s 已保存：%s", spec.id, ", ".join(Path(f).name for f in files))
    if rows:
        write_csv(ctx.fig_dir / "figures_index.csv", rows)
    return rows


def catalog_rows(cfg: Mapping[str, Any]) -> list[dict]:
    """纯静态的图表目录（不生成图），用于文档与清单。"""
    return [{
        "figure_id": s.id, "slug": s.slug, "title_en": s.title_en, "title_zh": s.title_zh,
        "section": s.section, "scope": s.scope, "width": s.width,
        "filename_rule": f"{s.id}_{s.slug}.{{{','.join(cfg['figures']['formats'])}}}",
    } for s in FIGURE_CATALOG]
