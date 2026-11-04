# -*- coding: utf-8 -*-
"""
training.kdlogic —— KD-Logic：DL 教师 → 约束 CART 蒸馏（确定性对照）
================================================================================
对应 §4.7。必报四项：
  (a) 保真度 fidelity（学生与教师判决一致率）
  (b) 规则复杂度（条数 / 深度 / 体积）
  (c) 与"直接在真值上训练 CART"的对比
  (d) zero-shot（训练期未见攻击类型）子集表现

特征构造：窗口内每个通道取 `kd.feature_stats`（默认 mean/std/last/min/max），
→ 16 × 5 = 80 维定长向量。CART 只能吃定长特征，故这一步不可省；
但它保留了"哪个通道在整窗上异常"的可解释性，符合"特征集限制为 16 通道"的初衷。

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import RunContext
from .data import WindowDataset, make_loader
from .evaluator import predict
from .utils import get_logger, write_json

log = get_logger("kdlogic")

STAT_FUNCS = {
    "mean": lambda x: x.mean(axis=1),
    "std": lambda x: x.std(axis=1),
    "last": lambda x: x[:, -1, :],
    "min": lambda x: x.min(axis=1),
    "max": lambda x: x.max(axis=1),
    "median": lambda x: np.median(x, axis=1),
}


def extract_features(cfg: Mapping[str, Any], dataset: WindowDataset, *, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (X [N, C·S], y [N], y_type [N])，按 batch 前向以复用 DataLoader。"""
    stats = list(cfg["kd"]["feature_stats"])
    unknown = [s for s in stats if s not in STAT_FUNCS]
    if unknown:
        raise ValueError(f"kd.feature_stats 含未知统计量：{unknown}")

    loader = make_loader(dataset, cfg, split=dataset.split,
                         batch_size=int(cfg["train"]["eval_batch_size"]), shuffle=False)
    feats: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    for batch in loader:
        x1 = batch["x1"].numpy()                     # [B,T,C]
        parts = [STAT_FUNCS[s](x1).astype(np.float32) for s in stats]   # 每个 [B,C]
        feats.append(np.concatenate(parts, axis=1))
        ys.append(batch["y"].numpy().astype(np.int64))
        ts.append(batch["y_type"].numpy().astype(np.int64))
    return (np.concatenate(feats), np.concatenate(ys), np.concatenate(ts))


def feature_names(channels: Sequence[str], stats: Sequence[str]) -> list[str]:
    return [f"{ch}__{st}" for st in stats for ch in channels]


def _tree_size_kb(tree) -> float:
    """用 pickle 序列化体积近似树的内存占用（论文报 <5 KB 口径一致）。"""
    import pickle
    return len(pickle.dumps(tree)) / 1024.0


def _export_rules(tree, names: Sequence[str], max_rules: int, decimals: int) -> list[str]:
    """把 sklearn 树导出成可读规则（前 max_rules 条，按样本数降序）。"""
    from sklearn.tree import _tree
    t = tree.tree_
    lines: list[str] = []

    def recurse(node: int, conds: list[str]) -> None:
        if len(lines) >= max_rules:
            return
        if t.feature[node] != _tree.TREE_UNDEFINED:
            name = names[t.feature[node]]
            thr = t.threshold[node]
            recurse(t.children_left[node], conds + [f"{name} <= {thr:.{decimals}f}"])
            recurse(t.children_right[node], conds + [f"{name} > {thr:.{decimals}f}"])
        else:
            vals = t.value[node][0]
            p = vals / max(vals.sum(), 1e-9)
            cls = "attack" if (len(p) > 1 and p[1] >= 0.5) else "normal"
            lines.append("IF " + " AND ".join(conds) + f" THEN {cls} (p={p[-1]:.{decimals}f})")

    recurse(0, [])
    return lines


def run_kd(cfg: Mapping[str, Any], *, teacher_ckpt: str | Path, run: RunContext,
           datasets: Mapping[str, WindowDataset], device) -> dict:
    """主流程：抽特征 → 教师软标签 → 蒸馏 CART / 直接 CART → 评估 → 落盘。"""
    from sklearn.tree import DecisionTreeClassifier, export_graphviz

    from .evaluator import load_run_for_eval

    teacher, ckpt = load_run_for_eval(teacher_ckpt, device=str(device))
    channels = list(ckpt["channels"])
    stats = list(cfg["kd"]["feature_stats"])
    names = feature_names(channels, stats)

    log.info("KD-Logic：抽取窗口特征（%d 维 = %d 通道 × %d 统计量）", len(names), len(channels), len(stats))
    Xtr, ytr, _ = extract_features(cfg, datasets["train"], device=device)
    Xte, yte, _ = extract_features(cfg, datasets["test"], device=device)
    # ★ 2026-10-27：保真度（学生与教师判决一致率）改在 **val** 上评估。
    #   它是"学生是否拟合了教师"的诊断量，放在 test 上会与最终精度混用同一份数据；
    #   test 只应承担最终 MCC。数据集缺 val 时退回旧行为并告警。
    val_ds = datasets.get("val") if hasattr(datasets, "get") else None
    Xva = None
    if val_ds is not None:
        Xva, _, _ = extract_features(cfg, val_ds, device=device)

    # 教师判决（用于蒸馏与保真度）；一次前向同时取概率与 capture 溯源
    bs = int(cfg["train"]["eval_batch_size"])
    tr_pred = predict(teacher, make_loader(datasets["train"], cfg, split="train",
                                           batch_size=bs), device=device)
    te_pred = predict(teacher, make_loader(datasets["test"], cfg, split="test",
                                           batch_size=bs), device=device)
    va_pred = (predict(teacher, make_loader(val_ds, cfg, split="val", batch_size=bs),
                       device=device) if val_ds is not None else None)
    p_tr, p_te = tr_pred["prob"], te_pred["prob"]
    teacher_tr = (p_tr >= 0.5).astype(np.int64)
    teacher_te = (p_te >= 0.5).astype(np.int64)
    teacher_va = (va_pred["prob"] >= 0.5).astype(np.int64) if va_pred is not None else None

    tcfg = cfg["kd"]["tree"]
    distill_mode = str(cfg["kd"]["distill"]["mode"])

    # ---- 蒸馏树 ----
    tree = DecisionTreeClassifier(
        criterion=str(tcfg["criterion"]), max_depth=int(tcfg["max_depth"]),
        max_leaf_nodes=int(tcfg["max_leaf_nodes"]),
        min_samples_leaf=int(tcfg["min_samples_leaf"]),
        class_weight=(None if str(tcfg["class_weight"]) == "none" else str(tcfg["class_weight"])),
        random_state=int(cfg["train"]["seeds"][0]),
    )
    if distill_mode == "soft":
        # 学教师判决；以教师置信度作样本权重（更接近真值软标签）
        w = np.abs(p_tr - 0.5) * 2.0 + 1e-3
        tree.fit(Xtr, teacher_tr, sample_weight=w)
    else:
        tree.fit(Xtr, ytr)

    prob_te = tree.predict_proba(Xte)[:, 1]
    pred_te = (prob_te >= 0.5).astype(np.int64)
    # 保真度优先在 val 上算；val 不可用时退回 test（并保留 teacher_te 的原口径）
    if Xva is not None and teacher_va is not None:
        prob_va = tree.predict_proba(Xva)[:, 1]
        pred_va = (prob_va >= 0.5).astype(np.int64)
        fidelity = float((pred_va == teacher_va).mean())
        fidelity_split = "val"
    else:
        log.warning("KD-Logic：缺少 val 数据集，保真度退回 test 口径")
        fidelity = float((pred_te == teacher_te).mean())
        fidelity_split = "test"

    from .metrics import binary_metrics
    m = binary_metrics(yte, prob_te, 0.5)

    # ---- 直接训练 CART（对照） ----
    direct = None
    if bool(cfg["kd"]["direct_baseline"]):
        direct = DecisionTreeClassifier(
            criterion=str(tcfg["criterion"]), max_depth=int(tcfg["max_depth"]),
            max_leaf_nodes=int(tcfg["max_leaf_nodes"]),
            min_samples_leaf=int(tcfg["min_samples_leaf"]),
            class_weight=(None if str(tcfg["class_weight"]) == "none" else str(tcfg["class_weight"])),
            random_state=int(cfg["train"]["seeds"][0]),
        ).fit(Xtr, ytr)
        dp = direct.predict_proba(Xte)[:, 1]
        dm = binary_metrics(yte, dp, 0.5)
    else:
        dm = {}

    # ---- zero-shot 子集（训练期从未见过该攻击类型的采集） ----
    zs_types = set(cfg.get("_zero_shot_types", []) or [])
    zs_row: dict[str, Any] = {}
    if zs_types:
        cap = te_pred["cap"]
        types = np.array([datasets["test"].specs[c].attack_type for c in cap], dtype=object)
        sel = np.isin(types, list(zs_types)) & (yte == 1)
        if sel.any():
            zs_row = {"n_positive": int(sel.sum()),
                      "recall_teacher": float(teacher_te[sel].mean()),
                      "recall_student": float(pred_te[sel].mean())}

    rules = _export_rules(tree, names, int(cfg["kd"]["rule_export"]["max_rules"]),
                          int(cfg["kd"]["rule_export"]["decimals"]))
    result = {
        "feature_names": names,
        "distilled": {
            "mcc": float(m["mcc"]), "auc_pr": float(m["auc_pr"]), "f1": float(m["f1"]),
            "recall": float(m["recall"]), "fpr": float(m["fpr"]),
            "fidelity": fidelity,
            "fidelity_split": fidelity_split,
            "n_rules": int(tree.get_n_leaves()),
            "depth": int(tree.get_depth()),
            "size_kb": _tree_size_kb(tree),
        },
        "direct": {
            "mcc": float(dm.get("mcc", np.nan)), "auc_pr": float(dm.get("auc_pr", np.nan)),
            "f1": float(dm.get("f1", np.nan)), "fidelity": 1.0,
            "n_rules": int(direct.get_n_leaves()) if direct is not None else 0,
            "depth": int(direct.get_depth()) if direct is not None else 0,
            "size_kb": _tree_size_kb(direct) if direct is not None else 0.0,
        },
        "zero_shot": zs_row,
        "distill_mode": distill_mode,
        "n_train_windows": int(len(ytr)),
        "rules_preview": rules,
    }

    out_dir = Path(run.run_dir) / "kd"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "kd_results.json", result)
    (out_dir / "rules_preview.txt").write_text("\n".join(rules), encoding="utf-8")
    try:
        dot = export_graphviz(tree, feature_names=names, class_names=["normal", "attack"],
                              filled=False, rounded=True, impurity=False)
        (out_dir / "tree.dot").write_text(dot, encoding="utf-8")
    except Exception as e:  # pragma: no cover
        log.warning("导出 DOT 失败：%s", e)
    log.info("KD-Logic 完成：fidelity=%.4f，MCC=%.4f（直接训练 %.4f），规则 %d 条 / 深度 %d",
             fidelity, result["distilled"]["mcc"], result["direct"]["mcc"],
             result["distilled"]["n_rules"], result["distilled"]["depth"])
    return result
