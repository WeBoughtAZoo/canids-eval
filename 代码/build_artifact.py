#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_artifact.py —— 生成"充分统计量"发布包（数据不公开时的复现载体）
================================================================================
背景
----
论文采用"不发任何经处理的数据集，只发代码与不敏感材料"的策略。于是传统的
"下载数据 → 跑脚本 → 得到表"路径不存在。本脚本把复现下移到**统计量级**：

  发布每个 run 在测试集上的 2×2 计数（按 capture / 攻击类型 / 密度箱拆分），
  读者**不接触任何信号数据**，即可逐位重算论文中的 MCC、recall、FPR、
  簇级符号翻转 p、密度分层召回。

多批次（2026-10-31 增补）
--------------------------
主表（T7a/T7b）来自 CPU 批次 `main`；C4 的新证据来自 GPU 批次 `gpu1`
（设备等价性检验未通过，故新对比一律在 gpu1 内同设备完成）。
为让**两类数字都可被重算**，本脚本支持 `--batches main,gpu1`：
所有产物增加 `batch` 列，`run_id` 前缀为 `<batch>|…`，避免同名 run 冲突。

产物（`outputs/artifact/`）
--------------------------
  runs.csv                  每个 run 的规模、阈值与由计数重算的总体指标
  counts_capture.csv        逐 capture 的 TP/FP/FN/TN（簇级检验的输入）
  counts_type.csv           逐攻击类型的 TP/FN 与正常集 FP（分层召回的输入）
  counts_density.csv        逐密度箱的 TP/FN（T11 的输入）
  discordance_capture.csv   配对比较的逐 capture b/c（簇级 p 的输入；仅主批次有）
  tau_seeds.csv             每个 (批次,语料,模型,变体) 的 τ 分布
  sparse_families.csv       稀疏攻击家族的召回 + Clopper–Pearson 95% 区间
  ablation_inventory.csv    消融变体清单与"已跑/未跑"状态（按批次）
  README.md                 重算公式与最小复现步骤（自动生成）

设计约束
--------
· 只读 `predictions_*.npz` 与 `label.npy`，**不重训、不读特征张量**（v1/v2 不加载）；
· 逐 capture 读 `label.npy` 仅为还原"窗内攻击帧数"，单文件 ≤ 1 MB（int8）；
· 所有计数为整数，产物总大小 < 30 MB，可随仓库发布。

作者：小巴（WorkBuddy）    日期：2026-10-30（2026-10-31 增补多批次）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from training import config as C           # noqa: E402
from training import groups as G           # noqa: E402
from training.utils import get_logger, setup_logging, write_csv  # noqa: E402

log = get_logger("artifact")

OUT_SUBDIR = "artifact"


# --------------------------------------------------------------------------- #
# 基础
# --------------------------------------------------------------------------- #
def _batch_root(cfg: dict, batch: str) -> Path:
    return Path(cfg["_resolved"]["output_root"]) / str(batch)


def _dataset_dirs(cfg: dict) -> dict[str, str]:
    return dict(cfg["paths"]["dataset_dirs"])


def discover_runs(cfg: dict, batch: str) -> list[dict[str, Any]]:
    """扫描 <batch>/<dataset>/<model>/<variant>/seed<N>/predictions_test.npz。"""
    root = _batch_root(cfg, batch)
    keys = set(_dataset_dirs(cfg))          # 输出目录用数据集 key，不是预处理子目录名
    out: list[dict[str, Any]] = []
    if not root.exists():
        return out
    for ddir in sorted(p for p in root.iterdir() if p.is_dir() and p.name in keys):
        for mdir in sorted(p for p in ddir.iterdir() if p.is_dir()):
            for vdir in sorted(p for p in mdir.iterdir() if p.is_dir()):
                for sdir in sorted(p for p in vdir.iterdir() if p.is_dir()):
                    if not sdir.name.startswith("seed"):
                        continue
                    if not (sdir / "predictions_test.npz").exists():
                        continue
                    out.append({
                        "batch": batch,
                        "run_id": f"{batch}|{ddir.name}/{mdir.name}/{vdir.name}/{sdir.name}",
                        "dataset": ddir.name, "model": mdir.name,
                        "variant": vdir.name, "seed": int(sdir.name[4:]),
                        "run_dir": sdir,
                    })
    return out


def unknown_index(run_dir: Path) -> int | None:
    """从 label_map.json 读 unknown 类下标（known 口径需要它做掩码）。"""
    p = Path(run_dir) / "label_map.json"
    if not p.exists():
        return None
    lm = json.loads(p.read_text(encoding="utf-8")).get("label_map", {})
    for k in ("unknown_attack", "unknown", "Unknown"):
        if k in lm:
            return int(lm[k])
    return None


def regime_keep(z: dict, regime: str, uk: int | None) -> np.ndarray | None:
    """known 口径：剔除 label==1 且 y_type==unknown 的正样本（与 compare.py 同定义）。"""
    if regime != "known" or uk is None:
        return None
    y = np.asarray(z["label"]).astype(np.int64)
    if "y_type" not in z:
        return None
    return ~((y == 1) & (np.asarray(z["y_type"]).astype(np.int64) == uk))


def load_pred(run_dir: Path) -> dict[str, np.ndarray]:
    with np.load(Path(run_dir) / "predictions_test.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _pred(z: dict) -> np.ndarray:
    return (np.asarray(z["prob"]) >= float(z["threshold"])).astype(np.int64)


def _mcc(c: dict[str, int]) -> float:
    import math
    tp, fp, fn, tn = (float(c[k]) for k in ("tp", "fp", "fn", "tn"))
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den > 0 else float("nan")


def attack_frame_counts(cfg: dict, dataset: str, split: str,
                        cap: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """窗内攻击帧数（只读 label.npy，不触碰 v1/v2）。"""
    base = (Path(cfg["_resolved"]["tensors_root"]) / _dataset_dirs(cfg)[dataset]
            / str(cfg["paths"]["tensor_subdir"]) / split)
    names = G.capture_names(cfg, dataset, split)
    win = int(cfg["data"]["window"]["T"])
    out = np.zeros(cap.size, dtype=np.int16)
    order = np.argsort(cap, kind="stable")
    bounds = np.searchsorted(cap[order], np.arange(int(cap.max()) + 2))
    for c in range(int(cap.max()) + 1):
        idx = order[bounds[c]:bounds[c + 1]]
        if idx.size == 0:
            continue
        lab = np.asarray(np.load(base / names[c] / "label.npy", mmap_mode="r"),
                         dtype=np.int64)
        cum = np.concatenate([[0], np.cumsum(lab)])
        s = pos[idx]
        out[idx] = (cum[s + win] - cum[s]).astype(np.int16)
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def build(cfg: dict, out_dir: Path, batches: list[str]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for b in batches:
        found = discover_runs(cfg, b)
        if not found:
            log.warning("批次 %s 没有可用 run", b)
        runs.extend(found)
    if not runs:
        raise SystemExit("未找到任何 run：%s" % ", ".join(str(_batch_root(cfg, b)) for b in batches))
    log.info("发现 %d 个 run（批次：%s）", len(runs), ", ".join(batches))

    bins = [tuple(int(x) for x in bb) for bb in cfg["eval"]["density_bins"]]
    dense_cache: dict[str, np.ndarray] = {}

    rows_runs: list[dict] = []
    rows_cap: list[dict] = []
    rows_type: list[dict] = []
    rows_den: list[dict] = []

    for r in runs:
        z = load_pred(r["run_dir"])
        y = np.asarray(z["label"]).astype(np.int64)
        p = _pred(z)
        cap = np.asarray(z["cap"]).astype(np.int64)
        pos = np.asarray(z["pos"]).astype(np.int64)
        yt = np.asarray(z["y_type"]).astype(np.int64)
        rid, ds, bt = r["run_id"], r["dataset"], r["batch"]
        names = G.capture_names(cfg, ds, "test")

        # 1) 逐 capture（向量化 bincount；禁止逐簇 np.isin —— 224×293 万会卡死）
        k = int(cap.max()) + 1
        pos_m, hit = (y == 1), (p == 1)
        tp_k = np.bincount(cap, weights=(pos_m & hit), minlength=k).astype(np.int64)
        fp_k = np.bincount(cap, weights=((~pos_m) & hit), minlength=k).astype(np.int64)
        fn_k = np.bincount(cap, weights=(pos_m & (~hit)), minlength=k).astype(np.int64)
        tn_k = np.bincount(cap, weights=((~pos_m) & (~hit)), minlength=k).astype(np.int64)
        tot = {"n_pos": int(tp_k.sum() + fn_k.sum()), "n_neg": int(fp_k.sum() + tn_k.sum()),
               "tp": int(tp_k.sum()), "fp": int(fp_k.sum()),
               "fn": int(fn_k.sum()), "tn": int(tn_k.sum())}
        for c in range(k):
            rows_cap.append({"batch": bt, "run_id": rid, "dataset": ds, "capture": names[c],
                             "n_pos": int(tp_k[c] + fn_k[c]), "n_neg": int(fp_k[c] + tn_k[c]),
                             "tp": int(tp_k[c]), "fp": int(fp_k[c]),
                             "fn": int(fn_k[c]), "tn": int(tn_k[c])})
        rows_runs.append({
            "batch": bt, "run_id": rid, "dataset": ds, "model": r["model"],
            "variant": r["variant"], "seed": r["seed"], "threshold": float(z["threshold"]),
            "n_windows": int(y.size), **tot,
            "mcc": _mcc(tot),
            "recall": tot["tp"] / max(1, tot["tp"] + tot["fn"]),
            "fpr": tot["fp"] / max(1, tot["fp"] + tot["tn"]),
            "precision": tot["tp"] / max(1, tot["tp"] + tot["fp"]),
        })

        # 2) 逐攻击类型（正常集 FP 单列，供分层 FPR 使用）
        lm_path = r["run_dir"] / "label_map.json"
        inv = {}
        if lm_path.exists():
            inv = {int(v): kk for kk, v in
                   json.loads(lm_path.read_text(encoding="utf-8")).get("label_map", {}).items()}
        norm_mask = (y == 0)
        fp_norm = int((p[norm_mask] == 1).sum())
        for t in [int(x) for x in np.unique(yt) if int(x) > 0]:
            mask = (yt == t) & (y == 1)
            if not mask.any():
                continue
            tp = int((p[mask] == 1).sum())
            n_pos = int(mask.sum())
            rows_type.append({"batch": bt, "run_id": rid, "type_idx": t,
                              "attack_type": inv.get(t, str(t)),
                              "n_pos": n_pos, "tp": tp, "fn": n_pos - tp,
                              "n_normal": int(norm_mask.sum()), "fp_normal": fp_norm})

        # 3) 逐密度箱（计数与批次无关，按语料缓存）
        if ds not in dense_cache:
            dense_cache[ds] = attack_frame_counts(cfg, ds, "test", cap, pos)
            log.info("密度计数缓存：%s（%d 窗口）", ds, dense_cache[ds].size)
        d = dense_cache[ds]
        for lo, hi in bins:
            mask = (y == 1) & (d >= lo) & (d <= hi)
            n_pos = int(mask.sum())
            tp = int((p[mask] == 1).sum())
            rows_den.append({"batch": bt, "run_id": rid, "bin": f"{lo}-{hi}",
                             "n_pos": n_pos, "tp": tp, "fn": n_pos - tp,
                             "recall": tp / n_pos if n_pos else float("nan")})

    write_csv(out_dir / "runs.csv", rows_runs, list(rows_runs[0].keys()))
    write_csv(out_dir / "counts_capture.csv", rows_cap, list(rows_cap[0].keys()))
    write_csv(out_dir / "counts_type.csv", rows_type, list(rows_type[0].keys()))
    write_csv(out_dir / "counts_density.csv", rows_den, list(rows_den[0].keys()))
    log.info("计数表：runs=%d capture=%d type=%d density=%d",
             len(rows_runs), len(rows_cap), len(rows_type), len(rows_den))

    return {"runs": runs, "out": out_dir, "batches": batches}


def _localize(cfg: dict, p: str) -> Path:
    """把 comparison.csv 里记录的路径映射到本地。

    ★ 远端批次（gpu1）的 comparison.csv 里存的是**远端绝对路径**
      （/localdisk-tmp/canids/outputs/...），本地不存在。凡是含 "outputs/" 的路径，
      一律按本地 output_root 重建；否则原样返回。
    """
    s = str(p).replace("\\", "/")
    i = s.find("outputs/")
    if i >= 0:
        # 取 "outputs/" 之后的相对段（training/<batch>/...），拼到本地 outputs 目录
        return Path(cfg["_resolved"]["output_root"]).parent / s[i + len("outputs/"):]
    return Path(p)


def build_discordance(cfg: dict, out_dir: Path, batches: list[str]) -> int:
    """从各批次 comparison.csv 的配对定义重算逐 capture b/c（簇级 p 的充分统计量）。"""
    cache: OrderedDict[str, dict] = OrderedDict()
    cache_cap = 6

    def get(path: str) -> dict:
        if path in cache:
            cache.move_to_end(path)
            return cache[path]
        z = load_pred(Path(path))
        cache[path] = z
        if len(cache) > cache_cap:
            cache.popitem(last=False)
        return z

    rows: list[dict] = []
    for bt in batches:
        comp = _batch_root(cfg, bt) / "comparison"
        if not comp.exists():
            log.info("批次 %s 无 comparison 目录，跳过配对 b/c", bt)
            continue
        for fname in ("comparison.csv", "comparison_known.csv"):
            f = comp / fname
            if not f.exists():
                continue
            regime = "known" if "known" in f.name else "full"
            with f.open("r", encoding="utf-8-sig", newline="") as fh:
                for rec in csv.DictReader(fh):
                    a, b = str(_localize(cfg, rec["run_a"])), str(_localize(cfg, rec["run_b"]))
                    if not (Path(a) / "predictions_test.npz").exists() or \
                       not (Path(b) / "predictions_test.npz").exists():
                        log.warning("配对 %s 的 run 目录本地缺失，跳过（%s vs %s）",
                                    rec.get("comparison", "?"), a, b)
                        continue
                    za, zb = get(a), get(b)
                    y = np.asarray(za["label"]).astype(np.int64)
                    cap = np.asarray(za["cap"]).astype(np.int64)
                    pa, pb = _pred(za), _pred(zb)
                    keep = regime_keep(za, regime, unknown_index(Path(a)))
                    if keep is not None:
                        y, cap, pa, pb = y[keep], cap[keep], pa[keep], pb[keep]
                    ok_a, ok_b = (pa == y), (pb == y)
                    k = int(cap.max()) + 1
                    b_k = np.bincount(cap, weights=(~ok_a & ok_b), minlength=k).astype(np.int64)
                    c_k = np.bincount(cap, weights=(ok_a & ~ok_b), minlength=k).astype(np.int64)
                    names = G.capture_names(cfg, rec["dataset"], "test")
                    pair_id = f"{bt}|{regime}|{rec['dataset']}|{rec['comparison']}|seed{rec['seed']}"
                    for i in range(k):
                        rows.append({"batch": bt, "pair_id": pair_id, "regime": regime,
                                     "dataset": rec["dataset"], "comparison": rec["comparison"],
                                     "seed": int(rec["seed"]), "capture": names[i],
                                     "b": int(b_k[i]), "c": int(c_k[i])})
                    if int(b_k.sum()) != int(float(rec["b"])) or int(c_k.sum()) != int(float(rec["c"])):
                        log.warning("b/c 与 comparison.csv 不一致：%s（重算 %d/%d vs 记录 %s/%s）",
                                    pair_id, b_k.sum(), c_k.sum(), rec["b"], rec["c"])
    if rows:
        write_csv(out_dir / "discordance_capture.csv", rows, list(rows[0].keys()))
        log.info("discordance：%d 对 × capture → %d 行",
                 len({r['pair_id'] for r in rows}), len(rows))
    return len(rows)


def build_tau_table(cfg: dict, out_dir: Path, batches: list[str]) -> None:
    out: list[dict] = []
    for bt in batches:
        src = _batch_root(cfg, bt) / "runs_index.csv"
        if not src.exists():
            log.warning("批次 %s 缺少 runs_index.csv，跳过 τ 表", bt)
            continue
        with src.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        agg: dict[tuple, list[tuple[int, float]]] = {}
        for r in rows:
            t = r.get("threshold", "")
            if t in ("", None):
                continue
            agg.setdefault((r["dataset"], r["model"], r["variant"]), []).append(
                (int(r["seed"]), float(t)))
        for (ds, model, variant), vals in sorted(agg.items()):
            taus = np.asarray([v for _, v in vals], dtype=float)
            out.append({"batch": bt, "dataset": ds, "model": model, "variant": variant,
                        "n_seeds": int(taus.size), "tau_mean": float(taus.mean()),
                        "tau_std": float(taus.std(ddof=1)) if taus.size > 1 else 0.0,
                        "tau_min": float(taus.min()), "tau_max": float(taus.max()),
                        "tau_range": float(taus.max() - taus.min()),
                        "seeds": "|".join(str(s) for s, _ in vals)})
    if out:
        write_csv(out_dir / "tau_seeds.csv", out, list(out[0].keys()))
        log.info("τ 表：%d 组", len(out))


def build_sparse_families(runs: Iterable[dict], out_dir: Path) -> None:
    """稀疏家族的召回 + Clopper–Pearson 95% 区间（从 per_family_test.csv 读）。"""
    from scipy import stats as sps

    rows: list[dict] = []
    for r in runs:
        if r["variant"] != "full":
            continue
        f = r["run_dir"] / "per_family_test.csv"
        if not f.exists():
            continue
        with f.open("r", encoding="utf-8-sig", newline="") as fh:
            for rec in csv.DictReader(fh):
                if rec.get("level") != "family":
                    continue
                n = int(float(rec["n_positive_windows"]))
                if n >= 1000:
                    continue
                k = int(round(float(rec["recall"]) * n))
                lo = float(sps.beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
                hi = float(sps.beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
                rows.append({"batch": r["batch"], "dataset": r["dataset"], "model": r["model"],
                             "seed": r["seed"], "attack_type": rec["attack_type"],
                             "n_pos": n, "tp": k, "recall": float(rec["recall"]),
                             "cp95_lo": lo, "cp95_hi": hi,
                             "evaluability": rec.get("evaluability", "")})
    if rows:
        write_csv(out_dir / "sparse_families.csv", rows, list(rows[0].keys()))
        log.info("稀疏家族：%d 行（n_pos < 1000）", len(rows))


ABLATION_PURPOSE = {
    "full": "满通道参照",
    "Ab-1": "去 dlc 派生的显式长度通道（填充比例 + 载荷长度）",
    "Ab-1b": "去字节图 4 个填充掩码通道",
    "Ab-2": "去时序通道（验证 masquerade 的必要性）",
    "Ab-6": "Focal Loss 替代加权 BCE",
    "Ab-12": "协议盲 5 维子集（= FD-TCN-PF）",
    "Ab-12b": "10 维阶梯：只删 6 个 DLC 代理通道（容量/通道混淆的对照）",
    "Ab-13-RF17": "感受野扫描 RF=17（恒 4 块）",
    "Ab-13-RF33": "感受野扫描 RF=33",
    "Ab-13-RF45": "感受野扫描 RF=45",
    "Ab-13-RF93": "感受野扫描 RF=93",
    "Ab-14": "双头 → 单头",
    "Ab-16-fdtcn-small": "容量匹配对照（≈20K）",
    "Ab-17-lstm-large": "容量匹配对照（hidden 64）",
    "Ab-18-rgbnet8b": "RGBNet 只喂前 8 字节（CANDIDS 公平性）",
    "Ab-19-maxpool": "last+mean+max 池化",
    "Ab-20-auxframe": "逐帧辅助 BCE",
}


def build_ablation_inventory(cfg: dict, runs: Iterable[dict], out_dir: Path) -> None:
    ran: dict[tuple[str, str], set] = {}
    for r in runs:
        ran.setdefault((r["variant"], r["batch"]), set()).add(r["seed"])
    out: list[dict] = []
    for v in cfg["ablation"]["variants"]:
        rows_for_v = {b: s for (vv, b), s in ran.items() if vv == v}
        if not rows_for_v:
            out.append({"variant": v, "batch": "—", "n_runs": 0, "seeds": "",
                        "status": "not_run", "purpose": ABLATION_PURPOSE.get(v, "")})
            continue
        for b, seeds in sorted(rows_for_v.items()):
            n = len(seeds)
            out.append({"variant": v, "batch": b, "n_runs": n,
                        "seeds": "|".join(str(x) for x in sorted(seeds)),
                        "status": "run" if n >= 3 else "partial",
                        "purpose": ABLATION_PURPOSE.get(v, "")})
    write_csv(out_dir / "ablation_inventory.csv", out, list(out[0].keys()))
    log.info("消融清单：%d 行（涉及 %d 个变体）", len(out), len(cfg["ablation"]["variants"]))


README = """# 充分统计量包（sufficient-statistics artifact）

本目录是论文的**可核验载体**：不含任何原始或经处理的信号数据，只含 2×2 混淆计数。
任何读者都可用这里的整数，逐位重算论文中的分类指标与显著性检验。

## 批次（batch）列

论文有两批训练：
- `main`：CPU 批次，主表 T7a/T7b 的来源；
- `gpu1`：GPU 批次（RTX 4090），用于 C4 的补充对比与容量/池化对照。

设备等价性检验未通过（CTT FD-TCN 的 |ΔMCC| = 2.05 pt），因此**跨批次不可直接比较**；
`batch` 列与 `run_id` 前缀（`<batch>|…`）用于区分。C4 的所有对比都在 `gpu1` 内完成。

## 文件

| 文件 | 粒度 | 能重算什么 |
|---|---|---|
| `counts_capture.csv` | 每 run × 每 capture | MCC / recall / FPR / precision（全语料或任意 capture 子集）；簇级符号翻转 p（配合 `discordance_capture.csv`） |
| `counts_type.csv` | 每 run × 每攻击类型 | 分家族召回、分层 FPR（分母为正常集） |
| `counts_density.csv` | 每 run × 每密度箱 | T11 的密度分层召回 |
| `discordance_capture.csv` | 每配对 × 每 capture | 簇级 McNemar 的 b/c 与 p（符号翻转）；仅 `main` 有 |
| `tau_seeds.csv` | 每 (批次,语料,模型,变体) | τ 的种子均值/标准差/极差 |
| `sparse_families.csv` | 稀疏家族 | 召回 + Clopper–Pearson 95% 区间 |
| `ablation_inventory.csv` | 变体 × 批次 | 哪些已跑 / 未跑（含用途） |
| `runs.csv` | 每 run | 数据集规模、阈值、以及**由计数重算**的总体指标（用于自检） |

## 重算公式

MCC（由 tp/fp/fn/tn）：

```
MCC = (tp*tn - fp*fn) / sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
```

簇级符号翻转 p：令 d_k = b_k − c_k（每 capture 的净优势），
统计量 T = |Σ_k s_k d_k|，s_k ∈ {−1,+1} 独立等概率；枚举 2^K（K ≤ 16）或 Monte Carlo。
**下限**：K 个簇全部同向时最小可达 p = 2/2^K。

组层（同一语料含多次录制时）：先把 d_k 按组求和得 D_g，再在 G 个组上做同样的符号翻转，
下限为 2/2^G。

密度分层召回：对每个箱取 `tp / n_pos`。

τ 的种子分布：直接读 `tau_seeds.csv`（`tau_range` 是极差）。

## 复算示例（无需数据）

```python
import csv, math                      # 仅标准库，无需额外安装

rows = [r for r in csv.DictReader(open("counts_capture.csv", encoding="utf-8-sig"))
        if r["batch"] == "main" and r["run_id"] == "main|ctt15/fdtcn/full/seed42"]
tp, fp, fn, tn = (sum(int(r[k]) for r in rows) for k in ("tp", "fp", "fn", "tn"))
mcc = (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
print(round(mcc, 4))                  # 0.6564，与 runs.csv 中该 run 的 mcc 列一致

# 论文表 2 的 0.6689 是同一组 5 个 seed 的均值，不是单 seed：
#   把上面的 run_id 换成 5 个种子求和后再按 MCC 公式算，或直接取 runs.csv 的 5 行求平均。
```

## 为什么没有数据

三个语料的许可状态不一致。为避免"对一部分语料分发、对另一部分不分发"的不一致合规姿态，
本文**统一不分发任何派生张量**。复现路径见仓库根目录 `README.md` 的三级说明
（L0 统计量 / L1 合成语料 / L2 自行申请语料）。
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成充分统计量包")
    ap.add_argument("--config", default=str(Path(__file__).parent / "training" / "configs" / "train.yaml"))
    ap.add_argument("--batches", default="main",
                    help="逗号分隔的批次名，如 main,gpu1（默认 main）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=None)
    args = ap.parse_args(argv)

    batches = [b.strip() for b in str(args.batches).split(",") if b.strip()]
    cfg = C.load(args.config, C.parse_overrides(list(args.overrides or [])
                                                + [f"project.run_name={batches[0]}"]))
    setup_logging(str(cfg["runtime"]["log_level"]), None)

    out = Path(args.out) if args.out else Path(cfg["_resolved"]["output_root"]).parent / OUT_SUBDIR
    built = build(cfg, out, batches)
    build_discordance(cfg, out, batches)
    build_tau_table(cfg, out, batches)
    build_sparse_families(built["runs"], out)
    build_ablation_inventory(cfg, built["runs"], out)
    (out / "README.md").write_text(README, encoding="utf-8")
    log.info("产物目录：%s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
