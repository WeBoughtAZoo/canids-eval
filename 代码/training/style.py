# -*- coding: utf-8 -*-
"""
training.style —— 出图统一风格（SCI 投稿口径）
================================================================================
约定
----
- 图内文字**一律英文**：投稿语言是英文，且 DejaVu Sans 无中文字形，
  混排会出现方框（豆腐块）。中文说明留在 README 与图表清单里。
- 尺寸按 IEEE 双栏：单栏 3.5 in、双栏 7.16 in；字号 8–10 pt（缩放后仍 ≥6 pt）。
- 矢量 PDF 供投稿，PNG 600 dpi 供预览与审阅；两者同源同尺寸。
- 配色固定（同色同义）：检测=primary 深蓝，误报/负向=secondary 红，
  参考线=accent 灰，消融/次序列=tertiary 浅蓝。

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")           # 必须在 pyplot 之前设置（无显示环境）
import matplotlib.pyplot as plt  # noqa: E402


def apply_style(cfg: Mapping[str, Any]) -> dict[str, str]:
    """把 YAML 的 figures.style 灌进 rcParams，返回调色板。"""
    style = {str(k): v for k, v in cfg["figures"]["style"].items()}
    matplotlib.rcParams.update(style)
    return {str(k): str(v) for k, v in cfg["figures"]["palette"].items()}


def width(cfg: Mapping[str, Any], kind: str = "single") -> float:
    return float(cfg["figures"]["width_in"][kind])


def new_figure(cfg: Mapping[str, Any], *, kind: str = "single", nrows: int = 1,
               ncols: int = 1, height: float = 2.4, **kwargs):
    """按论文宽度建图（默认高宽比约 0.68，接近 IEEE 单栏图）。"""
    w = width(cfg, kind)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, height * (1 if nrows == 1 else nrows * 0.92)),
                             **kwargs)
    return fig, axes


def save(fig, base: str | Path, cfg: Mapping[str, Any]) -> list[Path]:
    """按配置格式批量保存（PDF 矢量 / PNG 位图）。"""
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    formats: Sequence[str] = list(cfg["figures"]["formats"])
    dpi = int(cfg["figures"]["png_dpi"])
    out: list[Path] = []
    exact = str(cfg["figures"].get("save_bbox", "tight")).lower() == "exact"
    for ext in formats:
        p = base.with_suffix(f".{ext}")
        kw: dict[str, Any] = dict(dpi=dpi if ext in ("png", "jpg", "jpeg", "tif") else None,
                                  format=ext)
        if not exact:
            kw["bbox_inches"] = "tight"
            kw["pad_inches"] = float(
                cfg["figures"]["style"].get("savefig.pad_inches", 0.02))
        fig.savefig(p, **kw)
        out.append(p)
    plt.close(fig)
    return out


def panel_label(ax, text: str, *, loc: str = "upper left") -> None:
    """(a)/(b) 子图编号：贴在坐标轴**内部**左上/右上角。

    放在内部而不是外部，是为了避免与居中的 axes title 撞行；
    加半透明白底，保证压在网格或热力图上仍可读。
    """
    x, y, ha = {
        "upper left": (0.015, 0.975, "left"),
        "upper right": (0.985, 0.975, "right"),
    }[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="top",
            fontsize=matplotlib.rcParams["axes.titlesize"], fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0), zorder=10)


def despine(ax, *, keep: Sequence[str] = ("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
