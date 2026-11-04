# -*- coding: utf-8 -*-
"""
training.evaluator —— 推理、指标、分层表与产物落盘
================================================================================
对外只暴露两个函数：
  `predict()`          —— 模型前向，返回 numpy 概率/标签/类型/溯源
  `evaluate()`         —— 在给定 split 上算指标、写 4 个产物文件

`evaluate()` 对 val 的职责是**定 τ**（写 threshold_curve_val.csv）；
对 test 的职责是**用 τ 原样复算**（τ 必须由调用方从 val 传入，缺失即报错，
从根上杜绝"在测试集上挑阈值"这类不可复现的做法）。

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import calibration as calib
from . import density as D
from . import metrics as M
from .artifacts import RunContext
from .data import WindowDataset
from .utils import get_logger, read_json, write_csv, write_json

log = get_logger("evaluator")


# --------------------------------------------------------------------------- #
# 前向
# --------------------------------------------------------------------------- #
def _forward(model, batch: Mapping[str, Any], device, needs: tuple[str, ...]):
    kwargs: dict[str, Any] = {}
    if "v1" in needs:
        kwargs["x1"] = batch["x1"].to(device, non_blocking=False)
    if "v2" in needs:
        kwargs["x2"] = batch["x2"].to(device, non_blocking=False)
    if "idv" in batch:
        kwargs["idv"] = batch["idv"].to(device, non_blocking=False)
    return model(**kwargs)


def predict(model, loader, *, device, criterion=None) -> dict[str, np.ndarray]:
    """返回 {prob, label, y_type, type_pred, type_conf, cap, pos[, loss]}。"""
    import torch
    model.eval()
    needs = tuple(type(model).needs)
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    types: list[np.ndarray] = []
    tpred: list[np.ndarray] = []
    tconf: list[np.ndarray] = []
    caps: list[np.ndarray] = []
    poss: list[np.ndarray] = []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            out = _forward(model, batch, device, needs)
            if criterion is not None:
                y = batch["y"].to(device)
                yt = batch["y_type"].to(device)
                yf = batch["y_frame"].to(device) if "y_frame" in batch else None
                total_loss += float(criterion(out, y, yt, yf)["loss"])
                n_batches += 1
            probs.append(torch.sigmoid(out.logit).detach().cpu().numpy().astype(np.float32))
            labels.append(batch["y"].numpy().astype(np.uint8))
            types.append(batch["y_type"].numpy().astype(np.int16))
            tx = torch.softmax(out.type_logits.detach(), dim=1)
            conf, pred = tx.max(dim=1)
            tpred.append(pred.cpu().numpy().astype(np.int16))
            tconf.append(conf.cpu().numpy().astype(np.float32))
            caps.append(batch["cap"].numpy().astype(np.int32))
            poss.append(batch["pos"].numpy().astype(np.int64))

    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)   # noqa: E731
    out_dict = {
        "prob": cat(probs), "label": cat(labels), "y_type": cat(types),
        "type_pred": cat(tpred), "type_conf": cat(tconf),
        "cap": cat(caps), "pos": cat(poss),
    }
    if criterion is not None:
        out_dict["loss"] = np.float64(total_loss / max(n_batches, 1))
    return out_dict


# --------------------------------------------------------------------------- #
# 分层用的辅助数组
# --------------------------------------------------------------------------- #
def capture_attributes(dataset: WindowDataset, cap_idx: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """窗口 → (攻击类型, 攻击家族, 是否 masquerade)。"""
    types = np.array([s.attack_type for s in dataset.specs], dtype=object)
    fams = np.array([s.attack_family for s in dataset.specs], dtype=object)
    masq = np.array([s.is_masquerade for s in dataset.specs], dtype=bool)
    return types[cap_idx], fams[cap_idx], masq[cap_idx]


def family_labels(families: np.ndarray, masq: np.ndarray) -> np.ndarray:
    """家族级分组标签：masquerade 一律单列（v4 §6.1 要求）。

    ROAD 的 masquerade 采集与普通采集共享同一 attack_family（如 correlated_signal），
    若直接按 family 分组会把两者混在一起；因此 masquerade 覆盖为独立组名。
    """
    out = np.asarray(families, dtype=object).copy()
    out[masq] = "masquerade"
    return out


def zero_shot_types(label_map: Mapping[str, int], unknown_label: str,
                    dataset: WindowDataset, *, level: str = "family") -> list[str]:
    """训练期从未出现过正样本、只能被 unknown 承载的分组（默认按家族，v4 §6.1）。"""
    known = {k for k in label_map if k not in ("normal", unknown_label)}
    attr = (lambda s: s.attack_family) if level == "family" else (lambda s: s.attack_type)
    seen = {attr(s) for s in dataset.specs
            if s.attack_type in known and not s.usable_as_normal}
    all_groups = {attr(s) for s in dataset.specs if not s.usable_as_normal}
    return sorted(all_groups - seen)


# --------------------------------------------------------------------------- #
# 评估主流程
# --------------------------------------------------------------------------- #
def evaluate(
    cfg: Mapping[str, Any],
    run: RunContext,
    dataset: WindowDataset,
    model,
    *,
    split: str,
    label_map: Mapping[str, int],
    threshold: float | None,
    device,
    write_curve: bool = True,
) -> dict[str, Any]:
    """在 `split` 上评估并落盘。

    参数
    ----
    threshold : None 表示"本 split 负责定 τ"（仅允许 val）；
                否则必须是 val 已标定好的 τ，原样使用。
    """
    from .data import make_loader

    if threshold is None and split != str(cfg["calibration"]["split"]):
        raise ValueError("只有 calibration.split 可以定阈值；test 必须传入 val 标定的 τ")

    bs = int(cfg["train"]["eval_batch_size"])
    loader = make_loader(dataset, cfg, split=split, batch_size=bs, shuffle=False)
    log.info("[%s/%s] 推理 %d 窗口（batch=%d）", run.dataset, split, len(dataset), bs)
    pred = predict(model, loader, device=device)

    y, p = pred["label"], pred["prob"]
    is_calib = split == str(cfg["calibration"]["split"])

    # ---- 定阈值（仅 val） ----
    curve = None
    if threshold is None:
        cal = calib.calibrate(y, p, cfg)
        threshold = float(cal["threshold"])
        curve = cal["curve"]
        selected = cal["selected"]
        log.info("[%s] 阈值标定完成：τ=%.4f（val %s=%.4f）",
                 run.dataset, threshold, selected["metric"], selected["value"])
    else:
        src = "trainer_best_threshold" if is_calib else "inherited_from_val"
        selected = {"source": src, "threshold": float(threshold)}
        if write_curve:
            curve = calib.sweep_curve(y, p, cfg)

    # ---- 操作点 τ（Recall@FPR 的分母） ----
    # ★ 2026-10-27：标定 split 上现算并随 metrics_*.json 落盘；其余 split 复用 val 的 τ。
    #   判据用 `is_calib` 而非 `threshold is None`——`reevaluate()` 会给 val 传入
    #   已有阈值，若仍按 threshold 判定，val 就永远不会生成 operating_points。
    if is_calib:
        op_taus = _operating_points(y, p)
        op_source = "calibrated_on_val"
        log.info("[%s] 操作点 τ@1%%FPR=%.4f 已随 metrics 落盘",
                 run.dataset, op_taus.get(0.01, float("nan")))
    else:
        op_taus = _load_operating_points(run, cfg)
        op_source = "inherited_from_val" if op_taus else "unavailable_self_calibrated"

    # ---- 主指标 ----
    main = M.binary_metrics(y, p, threshold, op_taus=op_taus)

    # ---- Bootstrap ----
    # ★ 2026-10-28：默认改为**按 capture 聚类**重采样。窗口级 i.i.d. 自助把
    #   同车同路况的窗口当独立样本，实测把 CI 压窄约 10×（CTT ±0.6 → ±7 pt）。
    boot: dict[str, Any] = {}
    bcfg = cfg["eval"]["bootstrap"]
    if bool(bcfg["enabled"]):
        names = [m for m in ("mcc", "auc_pr", "f1", "recall", "fpr") if m in cfg["eval"]["metrics"]]
        if "cluster" in bcfg and bool(bcfg["cluster"]):
            boot = M.cluster_bootstrap_ci(
                y, p, threshold, clusters=pred["cap"], metric_names=names,
                n_resamples=int(bcfg["n_resamples"]), ci=float(bcfg["ci"]),
                seed=int(bcfg["seed"]),
                auc_max_windows=int(bcfg["auc_max_windows"])
                if "auc_max_windows" in bcfg else 200_000,
            )
            if boot.get("_auc_skipped"):
                log.warning("[%s/%s] 窗口数 %d 超过 auc_max_windows，AUC 的聚类 CI 已跳过",
                            run.dataset, split, y.size)
        else:
            yb, pb = _bootstrap_sample(y, p, int(bcfg["max_samples"]), int(bcfg["seed"]))
            boot = M.bootstrap_ci(yb, pb, threshold, metric_names=names,
                                  n_resamples=int(bcfg["n_resamples"]),
                                  ci=float(bcfg["ci"]), seed=int(bcfg["seed"]))
            boot["_n_used"] = int(yb.size)
            boot["_method"] = "iid"

    # ---- 多分类 ----
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    mc = M.multiclass_metrics(pred["y_type"], pred["type_pred"], label_names) \
        if cfg["label_space"]["multiclass"] else {}

    # ---- 分层（family 为主，type 为辅；v4 §6.1 要求按家族报告） ----
    types, families, masq = capture_attributes(dataset, pred["cap"])
    fam = family_labels(families, masq)
    zs_family = zero_shot_types(label_map, str(cfg["label_space"]["unknown_label"]),
                                dataset, level="family")
    zs_type = zero_shot_types(label_map, str(cfg["label_space"]["unknown_label"]),
                              dataset, level="type")
    # ★ 2026-10-27：只保留"在正样本中真实出现"的 zero-shot 组。
    #   suppress 无逐帧标签（正窗口恒 0），原实现仍会把它列为 zero-shot 家族，
    #   使图注/表注容易误写成"两个零样本家族"。
    zs_family = [g for g in zs_family if bool((fam[y == 1] == g).any())]
    zs_type = [g for g in zs_type if bool((types[y == 1] == g).any())]
    tau_op = float(op_taus[0.01]) if 0.01 in op_taus else None
    per_family = M.per_type_table(
        y, p, threshold, types, is_masquerade=masq, zero_shot_types=zs_family,
        grade_cfg=cfg["eval"]["evaluability"], level="family", group=fam,
        tau_at_fpr=tau_op,
    )
    per_type = M.per_type_table(
        y, p, threshold, types, is_masquerade=masq, zero_shot_types=zs_type,
        grade_cfg=cfg["eval"]["evaluability"], level="type",
        tau_at_fpr=tau_op,
    )

    # ---- 攻击帧密度分层（2026-10-28 审计新增） ----
    # 窗口标签是 any_frame，正窗口的"含攻击帧数"从 1 到 W 不等；把密度混在一起
    # 看 MCC 会掩盖真实的失败模式（实测 90% 的架构差距来自 ≤8 帧的窗口）。
    dens = None
    dens_rows: list[dict[str, Any]] = []
    dens_bins = cfg["eval"]["density_bins"] if "density_bins" in cfg["eval"] else None
    if dens_bins:
        dens = D.window_attack_counts(dataset, pred["cap"], pred["pos"])
        dens_rows = D.density_table(y, p, dens, float(threshold), bins=dens_bins,
                                    tau_at_fpr=tau_op)

    # ---- 落盘 ----
    summary = M.summarize_split(main, split=split, n_windows=len(dataset), bootstrap=boot,
                                report=cfg["eval"]["metrics"])
    if dens is not None:
        summary["density"] = D.summary(y, dens)
        summary["density_table"] = dens_rows
    summary["multiclass"] = mc
    summary["zero_shot_types"] = zs_family
    summary["op_source"] = op_source
    # 操作点 τ 随 val 落盘，test 侧读取复用（见 _load_operating_points）
    summary["operating_points"] = {
        "fpr_0p1": (float(op_taus[0.001]) if 0.001 in op_taus else None),
        "fpr_1": (float(op_taus[0.01]) if 0.01 in op_taus else None),
        "source": op_source,
    }
    summary["calibration"] = selected
    summary["data"] = {
        "dataset": run.dataset, "split": split,
        "n_captures": len(dataset.specs),
        "window_T": int(dataset.window),
        "stride": int(cfg["data"]["window"]["stride"][split]),
        "variant": run.variant,
    }
    write_json(run.path("metrics", split=split), summary)
    write_csv(run.path("per_family_table", split=split), per_family)   # 主：按家族
    write_csv(run.path("per_type_table", split=split), per_type)       # 辅：按类型（T8 用）
    if dens_rows:
        write_csv(run.path("per_density_table", split=split), dens_rows)  # 密度分层

    np.savez(
        run.path("predictions", split=split),
        prob=pred["prob"], label=pred["label"], y_type=pred["y_type"],
        type_pred=pred["type_pred"], type_conf=pred["type_conf"],
        cap=pred["cap"], pos=pred["pos"], threshold=np.float64(threshold),
        # ★ 2026-10-28：密度随预测一起落盘，fig/table 与事后复算不必再读帧级标签
        density=(dens if dens is not None else np.zeros(0, dtype=np.int16)),
    )

    if curve is not None:
        rows = []
        for i in range(curve["threshold"].size):
            rows.append({
                "threshold": float(curve["threshold"][i]),
                "mcc": float(curve["mcc"][i]), "f1": float(curve["f1"][i]),
                "fpr": float(curve["fpr"][i]),
                "precision": float(curve["precision"][i]),
                "recall": float(curve["recall"][i]),
            })
        write_csv(run.path("threshold_curve", split=split), rows)

    log.info("[%s/%s] MCC=%.4f  AUC-PR=%.4f (prev=%.4f)  F1=%.4f  FPR=%.4f  τ=%.4f",
             run.dataset, split, main["mcc"], main["auc_pr"], main["prevalence"],
             main["f1"], main["fpr"], threshold)
    return {"metrics": summary, "per_family": per_family, "per_type": per_type,
            "predictions": pred, "threshold": threshold, "curve": curve}


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _operating_points(y: np.ndarray, p: np.ndarray) -> dict[float, float]:
    """在**标定 split（val）**上求操作点 τ：目标 FPR → τ。

    返回的 dict 以目标 FPR（0.001 / 0.01）为键，供 `binary_metrics(op_taus=...)`
    在 test 上原样复用。任何目标求不出 τ（缺类）时跳过该键。
    """
    out: dict[float, float] = {}
    for target in (0.001, 0.01):
        tau, _ = M.recall_at_fpr(y, p, target)
        if np.isfinite(tau):
            out[target] = float(tau)
    return out


def _load_operating_points(run: RunContext, cfg: Mapping[str, Any]) -> dict[float, float]:
    """从标定 split 的 metrics_*.json 读取已落盘的操作点 τ。

    读不到时返回空 dict —— 此时 `binary_metrics` 会退回"自定 τ"，但会把
    `op_source` 标为 `unavailable_self_calibrated`，便于事后甄别，不会静默冒充部署口径。
    """
    split = str(cfg["calibration"]["split"])
    if split == "test":
        log.warning("[%s] calibration.split=test，无法复用操作点 τ", run.dataset)
        return {}
    path = Path(run.path("metrics", split=split))
    if not path.exists():
        log.warning("[%s] 缺少 %s，操作点 τ 退回本 split 自定（结果表会标 op_source）",
                    run.dataset, path.name)
        return {}
    d = read_json(path) or {}
    op = d.get("operating_points") or {}
    out: dict[float, float] = {}
    for key, target in (("fpr_0p1", 0.001), ("fpr_1", 0.01)):
        v = op.get(key)
        if isinstance(v, (int, float)) and np.isfinite(float(v)):
            out[target] = float(v)
    return out


def _bootstrap_sample(y: np.ndarray, p: np.ndarray, max_samples: int,
                      seed: int) -> tuple[np.ndarray, np.ndarray]:
    """大样本下对 bootstrap 做分层子采样（保持正负比例），控制耗时在秒级。"""
    n = y.size
    if max_samples <= 0 or n <= max_samples:
        return y, p
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    keep_pos = int(round(max_samples * pos.size / n))
    keep_neg = max_samples - keep_pos
    ip = rng.choice(pos, size=min(keep_pos, pos.size), replace=False)
    ineg = rng.choice(neg, size=min(keep_neg, neg.size), replace=False)
    idx = np.concatenate([ip, ineg])
    idx.sort()
    return y[idx], p[idx]


def load_run_for_eval(checkpoint: str | Path, *, device: str = "cpu"):
    """从 checkpoint 重建模型（评估侧无需再读训练配置）。"""
    from .models import build_model

    ckpt = RunContext.load_checkpoint(checkpoint, map_location=device)
    name = ckpt["model_name"]
    model = build_model(name, ckpt["model_cfg"], n_classes=int(ckpt["n_classes"]),
                        window_T=int(ckpt["window_T"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt
