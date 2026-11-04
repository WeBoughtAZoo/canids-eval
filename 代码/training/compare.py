# -*- coding: utf-8 -*-
"""
training.compare —— 跨模型/跨消融的配对显著性检验（T13）
================================================================================
为什么不能只报均值
------------------
两个模型 MCC 差 0.004，在 300 万窗口的测试集上可能显著，也可能只是同一条轨迹的
抖动。评审要求给出**配对检验**：同一测试样本上两个模型的判决是否系统性不同。

实现要点
--------
- 使用 McNemar（配对、二分类判决），两侧阈值各取自己 val 标定的 τ（公平性协议②）；
- 多重比较用 Holm 校正（`stats.holm`）；
- 配对前提：两个 run 的**测试窗口序列必须逐位一致**（同一 dataset / 同一 limit），
  代码里用 `label` 数组校验，不一致直接报错（避免"看似配对、实则错位"）。
- 多 seed：默认取每个 group 的最小 seed 作代表，并在结果里记录 seed；如需 5 seed
  全做，把 `compare.all_seeds` 设为 true（会对每个 seed 各出一行）。

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import config as C
from . import figures as F
from .stats import holm, mcnemar, mcnemar_cluster
from .utils import get_logger, write_csv

log = get_logger("compare")


def _pred_path(run_dir: Path, split: str = "test") -> Path:
    return Path(run_dir) / f"predictions_{split}.npz"


def load_predictions(run_dir: str | Path, split: str = "test") -> dict[str, np.ndarray]:
    p = _pred_path(Path(run_dir), split)
    if not p.exists():
        raise FileNotFoundError(f"缺少预测文件：{p}（先跑 run_eval.py）")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _unknown_index(run_dir: str | Path) -> int | None:
    """从该 run 的 label_map.json 读取 unknown 类下标（不存在则 None）。

    ★ 不能硬编码 15：那是 CTT（16 类）的值；ROAD 9 类、HCRL 5 类，兜底类下标不同。
    """
    p = Path(run_dir) / "label_map.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return None
    lm = obj.get("label_map", obj)
    if not isinstance(lm, dict):
        return None
    for key in ("unknown_attack", "unknown", "Unknown"):
        if key in lm:
            return int(lm[key])
    return None


def _regime_mask(z: dict, regime: str, unknown_idx: int | None) -> np.ndarray:
    """口径掩码。known-attack = 剔除 label==1 且 y_type==unknown 的正样本。

    与 §4.5 的双轨口径严格一致：只剔除**正样本**里的 zero-shot 类，
    负样本全部保留（它们的 y_type 不是 unknown 类，本来就落在 y_type==0）。
    若该语料没有 unknown 类（快照无下标），两口径等价，返回全 True。
    """
    y = np.asarray(z["label"])
    if regime != "known" or unknown_idx is None:
        return np.ones(y.size, dtype=bool)
    if "y_type" not in z:
        raise ValueError("快照缺少 y_type，无法按 known-attack 口径过滤")
    yt = np.asarray(z["y_type"])
    return ~((y == 1) & (yt == unknown_idx))


def compare_pair(run_a: str | Path, run_b: str | Path, *, split: str = "test",
                 correction: bool = True, cluster: bool = True,
                 regime: str = "full") -> dict[str, Any]:
    """两个 run 在同一测试集上的 McNemar 检验。

    `cluster=True`（默认）走**簇级**检验：不一致对按 capture 聚合成"每簇净优势"，
    再做符号翻转置换检验。窗口级 p 值仍保留在 `p_value_window` 里，但它把重叠窗口
    当独立样本，只能当参考，不得写进论文结论。
    """
    a = load_predictions(run_a, split)
    b = load_predictions(run_b, split)
    if a["label"].size != b["label"].size:
        raise ValueError(f"两 run 的测试窗口数不同（{a['label'].size} vs {b['label'].size}），"
                         f"无法配对：检查 data.limits 与切分配置是否一致")
    if not np.array_equal(a["label"], b["label"]):
        raise ValueError("两 run 的测试标签序列不一致，禁止配对检验（可能数据版本不同）")

    pred_a = (a["prob"] >= float(a["threshold"])).astype(np.int64)
    pred_b = (b["prob"] >= float(b["threshold"])).astype(np.int64)

    # ★ 2026-10-29：known-attack 口径的配对检验（v6 §14 · 5.8 必做项）。
    #   掩码必须同时作用于 label / pred / cap，三者保持同序对齐。
    unknown_idx = _unknown_index(run_a)
    if unknown_idx is None:
        unknown_idx = _unknown_index(run_b)
    keep = _regime_mask(a, regime, unknown_idx)
    if not keep.all():
        a = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v)
             for k, v in a.items()}
        b = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v)
             for k, v in b.items()}
        pred_a, pred_b = pred_a[keep], pred_b[keep]

    if cluster and "cap" in a and "cap" in b:
        if not np.array_equal(a["cap"], b["cap"]):
            raise ValueError("两 run 的 capture 序列不一致，无法做簇级配对")
        res = mcnemar_cluster(a["label"], pred_a, pred_b, a["cap"])
    else:
        res = mcnemar(a["label"], pred_a, pred_b, correction=correction)
        res.setdefault("method_window", res["method"])
        res.setdefault("p_value_window", res["p_value"])
        res.setdefault("n_clusters", 0)
        res.setdefault("n_clusters_discordant", 0)
        res.setdefault("n_discordant_window", res["n_discordant"])

    res.update({
        "n": int(a["label"].size),
        "regime": str(regime),
        "threshold_a": float(a["threshold"]),
        "threshold_b": float(b["threshold"]),
    })
    return res


def _group_key(rd: Path, ctx: F.RunArtifacts) -> tuple[str, str, str, int]:
    return (ctx.dataset, ctx.model, ctx.variant, ctx.seed)


def build_pairs(cfg: Mapping[str, Any]) -> list[tuple[F.RunArtifacts, F.RunArtifacts, str]]:
    """按 compare.scope 生成配对标号 (run_a, run_b, 说明)。"""
    runs = F.discover_runs(F.batch_root(cfg), cfg)
    scope = set(cfg["compare"]["scope"])
    all_seeds = bool(cfg["compare"]["all_seeds"])
    default_variant = str(cfg["ablation"]["default_variant"])

    # (dataset, model, variant) → 选代表 seed
    by_group: dict[tuple[str, str, str], list[F.RunArtifacts]] = {}
    for r in runs:
        by_group.setdefault((r.dataset, r.model, r.variant), []).append(r)
    for v in by_group.values():
        v.sort(key=lambda x: x.seed)

    pairs: list[tuple[F.RunArtifacts, F.RunArtifacts, str]] = []

    if "model" in scope:
        datasets = sorted({k[0] for k in by_group})
        for ds in datasets:
            models = sorted({k[1] for k in by_group if k[0] == ds and k[2] == default_variant})
            seeds = [min(r.seed for r in by_group[(ds, m, default_variant)]) for m in models]
            seed = min(seeds) if seeds else 0
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    ga = by_group.get((ds, models[i], default_variant))
                    gb = by_group.get((ds, models[j], default_variant))
                    if not ga or not gb:
                        continue
                    ra = next((r for r in ga if r.seed == seed), ga[0])
                    rb = next((r for r in gb if r.seed == seed), gb[0])
                    pairs.append((ra, rb,
                                  f"{F.display_name(models[i])} vs {F.display_name(models[j])}"))

    if "ablation" in scope:
        for (ds, model, variant), group in sorted(by_group.items()):
            if variant == default_variant:
                continue
            base = by_group.get((ds, model, default_variant))
            if not base:
                continue
            seed = min(r.seed for r in group)
            ra = next((r for r in base if r.seed == seed), base[0])
            rb = next((r for r in group if r.seed == seed), group[0])
            pairs.append((ra, rb, f"{variant} vs {default_variant}"))

    if all_seeds:
        # 展开为每 seed 一行
        expanded: list[tuple[F.RunArtifacts, F.RunArtifacts, str]] = []
        for ra, rb, desc in pairs:
            ga = by_group[(ra.dataset, ra.model, ra.variant)]
            gb = by_group[(rb.dataset, rb.model, rb.variant)]
            common = sorted({r.seed for r in ga} & {r.seed for r in gb})
            for s in common:
                expanded.append((next(r for r in ga if r.seed == s),
                                 next(r for r in gb if r.seed == s), f"{desc} (seed {s})"))
        pairs = expanded
    return pairs


def run_comparison(cfg: Mapping[str, Any], *, regime: str = "full",
                   filename: str = "comparison.csv") -> list[dict]:
    """执行全部配对检验，写 comparison/<filename>。

    `regime="known"` 时按 known-attack 口径过滤正样本（§4.5），
    与主表 T7a/T7b 的双轨口径一一对应。
    """
    if not bool(cfg["compare"]["enabled"]):
        return []
    pairs = build_pairs(cfg)
    if not pairs:
        log.info("没有可配对的 run（需要同数据集 ≥2 个模型，或 full 与消融并存）")
        return []

    alpha = float(cfg["compare"]["alpha"])
    rows: list[dict] = []
    for ra, rb, desc in pairs:
        try:
            res = compare_pair(ra.dir, rb.dir,
                               correction=bool(cfg["eval"]["mcnemar"]["correction"]),
                               cluster=bool(C.get_opt(cfg, "eval.mcnemar.cluster", True)),
                               regime=regime)
        except Exception as e:
            log.warning("配对失败 %s vs %s：%s", ra.dir, rb.dir, e)
            continue
        rows.append({
            "dataset": ra.dataset,
            "regime": regime,
            "comparison": desc,
            "run_a": str(ra.dir), "run_b": str(rb.dir),
            "seed": ra.seed, "n": res["n"],
            "b": res["b"], "c": res["c"],
            "statistic": res["statistic"], "method": res["method"],
            "p_value": res["p_value"], "alpha": alpha,
            "n_clusters": res.get("n_clusters", 0),
            "n_discordant_window": res.get("n_discordant_window", 0),
            "p_value_window": res.get("p_value_window"),
            "p_resolution": res.get("p_resolution"),
            "p_min": res.get("p_min"),
            "p_at_floor": res.get("p_at_floor"),
        })

    if not rows:
        return []

    # Holm 校正（同一 family：本次全部配对）
    adj = holm([r["p_value"] for r in rows])
    for r, p_adj in zip(rows, adj):
        r["p_holm"] = p_adj
        r["significant"] = bool(p_adj < alpha)

    out_dir = F.batch_root(cfg) / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / filename, rows, list(rows[0].keys()))
    n_sig = sum(1 for r in rows if r["significant"])
    log.info("配对检验完成：%d 对，Holm 校正后显著 %d 对（α=%.2f）", len(rows), n_sig, alpha)
    return rows
