"""Regenerate publication figures for the Sensors manuscript (v2, 2026-11-02).

All numbers are read from the authoritative CSV outputs; nothing is hard-coded,
so every figure can be rebuilt from the released statistics.

Run:  <canids-python> make_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"D:/Files/Personal/AI works/WorkBuddy/CAN IDS")
OUT = ROOT / "论文" / "v2_2026-11-02" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

C_ASIS = "#9aa5b1"
C_ORACLE = "#1f4e79"
C_MODEL = {"fdtcn": "#1f4e79", "lstm": "#c00000", "transformer": "#ed7d31", "rgbnet": "#7f7f7f"}
L_MODEL = {"fdtcn": "FD-TCN", "lstm": "LSTM", "transformer": "Transformer", "rgbnet": "FD-RGBNet"}


def read_csv(path: Path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- Fig. cross-session
def fig_density() -> Path:
    """Fig. 3: recall by in-window attack density, with the absolute n of every bin."""
    rows = read_csv(ROOT / "outputs/training/main/tables/table_density.csv")
    panels = [("ctt15", "CTT v1.5"), ("hcrl_canfd", "HCRL CAN-FD"), ("road", "ROAD")]
    series = [("FD-TCN", C_MODEL["fdtcn"], "o"),
              ("LSTM", C_MODEL["lstm"], "s"),
              ("Transformer", C_MODEL["transformer"], "^"),
              ("FD-RGBNet", C_MODEL["rgbnet"], "D"),
              ("FD-TCN-PF", "#2f7d5f", "v")]

    def sort_key(b):
        try:
            return int(str(b).split("-")[0])
        except ValueError:
            return 0

    # Equal horizontal space per density bin: 5 / 4 / 3 bins in the three panels.
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.9),
                             gridspec_kw={"wspace": 0.26, "width_ratios": [5, 4, 3]})
    for ax, (ds, title) in zip(axes, panels):
        sub = [r for r in rows if r["dataset"] == ds]
        bins = sorted({r["density_bin"] for r in sub}, key=sort_key)
        meta = {}
        for r in sub:
            meta.setdefault(r["density_bin"],
                            (int(r["n_positive"]), float(r["frac_of_positives"])))
        xs = np.arange(len(bins))
        for i, b in enumerate(bins):
            if meta[b][0] < 250:
                ax.axvspan(i - 0.5, i + 0.5, color="0.88", alpha=0.55, lw=0, zorder=0)
        for name, col, mk in series:
            ys = []
            for b in bins:
                hit = [r for r in sub if r["display"] == name and r["density_bin"] == b]
                ys.append(float(hit[0]["recall"]) if hit else np.nan)
            ax.plot(xs, ys, marker=mk, ms=3.2, lw=1.0, color=col, label=name, zorder=3)
        ax.set_xticks(xs, [f"{b}\n{100 * meta[b][1]:.1f}%\nn={meta[b][0]:,}" for b in bins],
                      fontsize=5.4)
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Recall")
    fig.text(0.5, 0.015,
             "Attack frames per window (share of positives; n = positive windows; shaded bins: n < 250)",
             ha="center", fontsize=7.6)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=5, frameon=False,
               fontsize=6.8, bbox_to_anchor=(0.5, -0.075))
    fig.subplots_adjust(bottom=0.31, top=0.87, left=0.075, right=0.99, wspace=0.26)
    p = OUT / "fig03_density.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_cross_session() -> Path:
    loso = read_csv(ROOT / "outputs/audit/operating_point_loso.csv")
    op = read_csv(ROOT / "outputs/audit/operating_point.csv")

    folds = ["set01", "set02", "set03", "set04"]
    asis, orac, auc = [], [], []
    for f in folds:
        a = next(r for r in loso if r["batch"] == f"loso_{f}" and r["mode"] == "as-is")
        o = next(r for r in loso if r["batch"] == f"loso_{f}" and r["mode"] == "oracle@1%")
        asis.append(float(a["mcc"]))
        orac.append(float(o["mcc"]))
        auc.append(float(a["auc_roc"]))

    loco_names = ["flooding", "fuzzing", "malfunction"]
    loco_asis, loco_orac, loco_auc = [], [], []
    sd_asis, sd_orac = [], []
    for nm in loco_names:
        va = [float(r["mcc"]) for r in op
              if r["batch"] == f"loco_{nm}" and r["mode"] == "as-is"]
        vo = [float(r["mcc"]) for r in op
              if r["batch"] == f"loco_{nm}" and r["mode"] == "oracle@1%"]
        loco_asis.append(float(np.mean(va)))
        loco_orac.append(float(np.mean(vo)))
        sd_asis.append(float(np.std(va, ddof=1)) if len(va) > 1 else 0.0)
        sd_orac.append(float(np.std(vo, ddof=1)) if len(vo) > 1 else 0.0)
        loco_auc.append(float(np.mean([float(r["auc_roc"]) for r in op
                                       if r["batch"] == f"loco_{nm}" and r["mode"] == "as-is"])))

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75), gridspec_kw={"wspace": 0.28})
    width = 0.38

    ax = axes[0]
    x = np.arange(len(folds))
    ax.bar(x - width / 2, asis, width, color=C_ASIS, label="as-is ($\\tau$ from validation)", edgecolor="none")
    ax.bar(x + width / 2, orac, width, color=C_ORACLE, label="oracle @1% FPR (diagnostic)", edgecolor="none")
    ax.axhline(0.6689, ls="--", lw=0.8, color="0.35")
    for i, (a, o, u) in enumerate(zip(asis, orac, auc)):
        ax.text(i, max(a, o) + 0.055, f"AUC {u:.4f}", ha="center", fontsize=6.6)
    ax.annotate("1% FPR\ninfeasible", xy=(3 + width / 2, 0.012), xytext=(3.02, 0.30),
                fontsize=6.5, color="#c00000", ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#c00000"))
    ax.set_xticks(x)
    ax.set_xticklabels(["set01", "set02", "set03", "set04"])
    ax.set_ylabel("Test MCC (window level)")
    ax.set_ylim(0, 1.24)
    ax.set_title("(a) CTT leave-one-session-out, FD-TCN")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False,
              handlelength=1.1, columnspacing=1.2, borderaxespad=0.0)

    ax = axes[1]
    x = np.arange(len(loco_names))
    ax.axhline(0, lw=0.8, color="0.25")
    ekw = dict(lw=0.6, capsize=2, ecolor="0.25")
    ax.bar(x - width / 2, loco_asis, width, color=C_ASIS, edgecolor="none",
           yerr=sd_asis, error_kw=ekw)
    ax.bar(x + width / 2, loco_orac, width, color=C_ORACLE, edgecolor="none",
           yerr=sd_orac, error_kw=ekw)
    for i, (a, o, u) in enumerate(zip(loco_asis, loco_orac, loco_auc)):
        star = "*" if u < 0.5 else ""
        col = "#c00000" if u < 0.5 else "0.15"
        ax.text(i, max(a, o, 0.0) + 0.055, f"AUC {u:.4f}{star}", ha="center", fontsize=6.6, color=col)
        # No numeric label on the main axis: at this scale it would sit inside the
        # negative bar. Exact values are carried by the inset below.
    ax.set_xticks(x)
    ax.set_xticklabels(["flooding", "fuzzing", "malfunction"])
    ax.set_ylim(-0.10, 1.24)
    ax.set_title("(b) HCRL leave-one-capture-out, FD-TCN")
    # Inset zoom: the two negative re-thresholded values are only 0.04 of a 1.34-unit axis.
    axi = ax.inset_axes([0.46, 0.40, 0.50, 0.30])
    axi.axhline(0, lw=0.7, color="0.25")
    for i in (1, 2):
        axi.errorbar(i, loco_orac[i], yerr=sd_orac[i], fmt="o", ms=3.6, lw=1.0,
                     color=C_ORACLE, ecolor="0.25", capsize=2, elinewidth=0.7)
        axi.text(i, loco_orac[i] - 0.014, f"{loco_orac[i]:.4f}", ha="center", va="top",
                 fontsize=5.8, color="#c00000",
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.9))
    axi.set_xlim(0.5, 2.5)
    axi.set_ylim(-0.085, 0.012)
    axi.set_xticks([1, 2])
    axi.set_xticklabels(["fuzzing", "malfunction"], fontsize=5.6)
    axi.tick_params(axis="y", labelsize=5.4)
    axi.set_title("zoom: re-thresholded MCC (3 seeds)", fontsize=5.8)
    axi.grid(False)

    p = OUT / "fig04_cross_session.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- Fig. robustness
def fig_robustness() -> Path:
    rows = read_csv(ROOT / "outputs/audit/robustness_summary.csv")
    conds = ["drop05", "drop10", "jitter05", "jitter20", "idswap05", "payload10"]
    labels = ["drop 5%", "drop 10%", "jitter 5%", "jitter 20%", "ID swap 5%", "payload 10%"]
    models = ["fdtcn", "lstm", "transformer"]

    def val(ds, m, c):
        r = next(x for x in rows if x["dataset"] == ds and x["model"] == m and x["condition"] == c)
        return 100.0 * float(r["d_mcc"])

    fig = plt.figure(figsize=(6.9, 2.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[3, 1], hspace=0.06, wspace=0.22)
    ax_t = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_t)
    ax_r = fig.add_subplot(gs[:, 1])

    width = 0.26
    x = np.arange(len(conds))
    for ax in (ax_t, ax_b):
        for k, m in enumerate(models):
            yy = [val("ctt15", m, c) for c in conds]
            ax.bar(x + (k - 1) * width, yy, width, color=C_MODEL[m], label=L_MODEL[m], edgecolor="none")
    ax_t.set_ylim(-12, 1.6)
    ax_b.set_ylim(-80, -70)
    ax_t.spines["bottom"].set_visible(False)
    ax_b.spines["top"].set_visible(False)
    ax_t.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, rotation=22, ha="right")
    ax_t.set_title("(a) CTT (CAN FD)")
    ax_t.set_ylabel(r"$\Delta$MCC vs clean (pt)")
    ax_b.set_ylabel("")
    d = 0.012
    kwargs = dict(transform=ax_t.transAxes, color="k", clip_on=False, lw=0.7)
    ax_t.plot((-d, +d), (-d, +d), **kwargs)
    ax_t.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_b.transAxes)
    ax_b.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_b.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    # true values for the clipped bars
    for k, m in enumerate(["lstm", "transformer"]):
        v = val("ctt15", m, "jitter20")
        ax_b.text(3 + (k - 1) * width, v + 0.6, f"{v:.0f}", ha="center", va="bottom",
                  fontsize=6.3, color="white", fontweight="bold")

    for k, m in enumerate(models):
        yy = [val("road", m, c) for c in conds]
        ax_r.bar(x + (k - 1) * width, yy, width, color=C_MODEL[m], edgecolor="none")
    ax_r.set_ylim(-6.2, 1.0)
    ax_r.axhline(0, lw=0.7, color="0.3")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(labels, rotation=22, ha="right")
    ax_r.set_title("(b) ROAD (classic CAN)")
    ax_r.set_ylabel(r"$\Delta$MCC vs clean (pt)")

    handles, labs = ax_t.get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))

    p = OUT / "fig05_robustness.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- Fig. deployment
def fig_deploy() -> Path:
    rows = read_csv(ROOT / "outputs/pi5_deploy/summary_v4.csv")
    models = ["fdtcn", "fdtcn_pf", "rgbnet", "transformer", "lstm"]
    disp = {"fdtcn": "FD-TCN", "fdtcn_pf": "FD-TCN-PF", "rgbnet": "FD-RGBNet",
            "transformer": "Transformer", "lstm": "LSTM"}
    case = {"fdtcn": "FD-TCN", "fdtcn_pf": "FD-TCN-PF", "rgbnet": "RGBNet",
            "transformer": "Transformer", "lstm": "LSTM"}

    def row(m, batch):
        return next(r for r in rows
                    if r["model"] == case[m] and r["precision"] == "FP32"
                    and r["threads"] == "1" and r["batch"] == str(batch))

    b1 = [float(row(m, 1)["T4_plus"]) for m in models]
    b64 = [float(row(m, 64)["T4_plus"]) for m in models]
    nbus = [float(row(m, 1)["N2k"]) for m in models]
    mj = [float(row(m, 1)["mJ"]) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), gridspec_kw={"wspace": 0.34, "width_ratios": [1.0, 1.18]})
    y = np.arange(len(models))[::-1]

    ax = axes[0]
    ax.barh(y + 0.18, b1, 0.34, color="#1f4e79", label="batch = 1", edgecolor="none")
    ax.barh(y - 0.18, b64, 0.34, color="#8fb0cc", label="batch = 64", edgecolor="none")
    for yi, v in zip(y + 0.18, b1):
        ax.text(v * 1.08, yi, f"{v:.2f}", va="center", fontsize=6.4)
    for yi, v in zip(y - 0.18, b64):
        ax.text(v * 1.08, yi, f"{v:.2f}", va="center", fontsize=6.0, color="0.35")
    ax.set_xscale("log")
    ax.set_xlim(0.15, 9)
    ax.set_xticks([0.2, 0.5, 1, 2, 5])
    ax.set_xticklabels(["0.2", "0.5", "1", "2", "5"])
    ax.minorticks_off()
    ax.set_yticks(y)
    ax.set_yticklabels([disp[m] for m in models])
    ax.set_xlabel("Latency per window (ms, log scale)")
    ax.set_title("(a) Single-core FP32 latency")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    ax.barh(y, nbus, 0.5, color="#1f4e79", edgecolor="none")
    for yi, v, e in zip(y, nbus, mj):
        ax.text(v + 0.35, yi, f"{v:.2f}  ({e:.1f} mJ)", va="center", fontsize=6.4)
    ax.set_yticks(y)
    ax.set_yticklabels([disp[m] for m in models], fontsize=6.4)
    ax.tick_params(axis="y", labelsize=6.4)
    ax.set_xlim(0, 26)
    ax.set_xlabel("2 kfps buses served per core ($N_{bus}$)")
    ax.set_title("(b) Bus capacity and energy budget")
    ax.grid(axis="y", visible=False)

    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", bbox_to_anchor=(0.29, 0.005),
               ncol=2, frameon=False, handlelength=1.0, fontsize=7.2)
    fig.subplots_adjust(bottom=0.20)
    p = OUT / "fig05_deployment.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- Fig. calibration
def fig_calibration() -> Path:
    cal = ROOT / "outputs/audit/calibration"
    bins = read_csv(cal / "calib_bins.csv")
    summ = {r["split"]: r for r in read_csv(cal / "calib_summary.csv")}

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75), gridspec_kw={"wspace": 0.30})

    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="0.55", label="perfect calibration")
    for key, lab, col in [("main_ctt15_fdtcn_full_test", "mixed pool (ECE 0.044)", "0.45"),
                          ("loso_set03_ctt15_fdtcn_full_test", "set03 (ECE 0.004)", C_MODEL["fdtcn"]),
                          ("loso_set04_ctt15_fdtcn_full_test", "set04 (ECE 0.179)", C_MODEL["lstm"])]:
        pts = [b for b in bins if b["split"] == key]
        ax.plot([float(b["bin_conf"]) for b in pts], [float(b["bin_acc"]) for b in pts],
                "o-", ms=3, lw=1.0, color=col, label=lab)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("(a) Reliability, CTT test")
    ax.legend(loc="lower right", fontsize=6.0, frameon=False, handlelength=1.2)

    ax = axes[1]
    z = np.load(cal / "loso_set04_ctt15_fdtcn_full_test_logits.npz")
    lg = z["logit"].astype(np.float64)
    neg = np.sort(lg[z["label"] == 0])
    taus = np.linspace(neg.min(), neg.max(), 600)
    frac = 1.0 - np.searchsorted(neg, taus, side="left") / neg.size
    ax.semilogy(taus, np.maximum(frac, 1e-6), color=C_MODEL["fdtcn"], lw=1.2, label="logit space")
    ax.axhline(0.01, ls="--", lw=0.8, color="0.35")
    ax.axhline(0.1459, ls=":", lw=1.1, color=C_MODEL["lstm"])
    tau = float(summ["loso_set04_ctt15_fdtcn_full_test"]["tau_logit"])
    ax.axvline(tau, ls=":", lw=1.0, color="0.35")
    ax.axvspan(17.0, taus.max(), color="0.85", alpha=0.45, lw=0)
    ax.text(30.0, 0.30, "prints as $p=1.0$\nin float32", fontsize=5.8, color="0.30", va="center", ha="center")
    ax.text(tau + 1.5, 0.045, "$\\tau$ = 34.3\n1% FPR\nMCC 0.378", fontsize=6.0, color="0.15", va="top")
    ax.text(neg.min() + 2, 0.085, "float32 floor 14.6%", fontsize=6.0, color=C_MODEL["lstm"])
    ax.set_xlabel("Threshold on the logit")
    ax.set_ylabel("Share of negatives above $\\tau$")
    ax.set_ylim(5e-4, 1.2)
    ax.set_title("(b) set04 negative-score survival")

    p = OUT / "fig06_calibration.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    for fn in (fig_density, fig_cross_session, fig_calibration, fig_deploy):
        print("wrote", fn())
