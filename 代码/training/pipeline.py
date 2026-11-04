# -*- coding: utf-8 -*-
"""
training.pipeline —— 单次 run 的编排：数据 → 训练 → 评估 → 出图
================================================================================
把 CLI 与实现解耦：`run_train.py` / `run_eval.py` / `run_figures.py` 只解析参数，
真正的顺序全部在这里，便于单测与脚本化调用。

一次 run 的固定顺序（顺序本身有语义，不可调换）：
  1. 校验 manifest 与配置一致（窗长/步长/通道/窗口数）
  2. 从 train 侧 manifest 派生标签表（**先定标签表，再建数据**）
  3. 建三个 Dataset 并施加窗口上限（冒烟）
  4. 由训练集窗口统计 pos_weight / class_weight（**不得用 val/test**）
  5. 训练（早停监控 val MCC）
  6. val 标定 τ → test 用同一 τ 复算
  7. 写 metrics_summary / runs_index
  8. 出图

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import config as C
from . import figures as F
from . import manifest as MF
from .artifacts import RunContext
from .data import (build_capture_sampler, build_datasets, compute_pos_weight,
                   limit_dataset, make_loader)
from .evaluator import evaluate, zero_shot_types
from .losses import MultiTaskLoss, compute_class_weight
from .models import build_model, count_parameters, model_inputs
from .models.common import count_macs
from .trainer import Trainer
from .utils import (configure_torch, env_snapshot, get_logger, set_seed,
                    write_csv, write_json)

log = get_logger("pipeline")


# --------------------------------------------------------------------------- #
# 配置准备
# --------------------------------------------------------------------------- #
def prepare_config(base_cfg: Mapping[str, Any], *, dataset: str, model: str,
                   variant: str, seed: int) -> dict:
    """把全局配置收窄到单次 run，并派生消融后的生效配置。"""
    import copy
    cfg = copy.deepcopy(dict(base_cfg))
    cfg["data"]["datasets"] = [dataset]
    cfg["model"]["name"] = model
    cfg["train"]["seeds"] = [int(seed)]
    eff = C.apply_ablation(cfg, variant)
    eff["_meta"] = dict(eff.get("_meta", {}), seed=int(seed))
    return eff


def prepare_data(cfg: Mapping[str, Any], dataset: str, *, seed: int) -> dict:
    """清单校验 → 标签表 → 三个 Dataset（含窗口上限）。"""
    paths = C.dataset_paths(cfg, dataset)
    index = MF.ManifestIndex(dataset, paths["manifest_dir"])
    summary = index.verify(cfg)

    label_map = MF.build_label_map(index, cfg)
    uses = model_inputs(str(cfg["model"]["name"]))
    aux_frame = bool(cfg["loss"]["aux_frame"]["enabled"]) if "aux_frame" in cfg["loss"] else False
    if aux_frame and "v1" not in uses:
        raise ValueError(f"loss.aux_frame 需要模型读取 v1（当前 {cfg['model']['name']} 的 needs={uses}）")
    datasets = build_datasets(
        cfg, index, dataset_key=dataset, label_map=label_map,
        use_v1=("v1" in uses), use_v2=("v2" in uses),
        use_id_valid=bool(cfg["data"]["aux"]["id_valid"]),
        seed=seed, use_frame_labels=aux_frame,
    )

    limits = cfg["data"]["limits"]
    for split, key in (("train", "max_train_windows"),
                       ("val", "max_val_windows"),
                       ("test", "max_test_windows")):
        datasets[split] = limit_dataset(datasets[split], limits[key], seed=seed)

    stats = {split: ds.label_stats() for split, ds in datasets.items()}

    # ★ 早失败：val 必须同时含正负样本，否则阈值标定无定义（报错比 NaN 更可查）
    for split in ("val", "test"):
        st = stats[split]
        if st["n_positive"] == 0 or st["n_negative"] == 0:
            raise ValueError(
                f"[{dataset}/{split}] 窗口级正负样本不全（pos={st['n_positive']}, "
                f"neg={st['n_negative']}）。检查 data.limits.max_{split}_windows "
                f"是否过小，或切分是否把攻击段全部划走。"
            )
    # 训练期未见过的类型（zero-shot）只提示，不阻断：它们由 unknown 类承载
    inv = {idx: name for name, idx in label_map.items()}
    seen = {inv[i] for i, c in stats["train"]["n_by_type"].items() if c > 0 and i in inv}
    zero_shot = sorted(set(label_map) - {"normal", str(cfg["label_space"]["unknown_label"])} - seen)
    if zero_shot:
        log.warning("[%s] 训练集无正样本的类型（zero-shot，不可声称有监督性能）：%s",
                    dataset, zero_shot)

    log.info("[%s] 窗口数 train/val/test = %d / %d / %d；标签表 %d 类：%s",
             dataset, stats["train"]["n_windows"], stats["val"]["n_windows"],
             stats["test"]["n_windows"], len(label_map), list(label_map))
    return {"index": index, "manifest_summary": summary, "label_map": label_map,
            "datasets": datasets, "stats": stats}


# --------------------------------------------------------------------------- #
# 训练 + 评估 + 出图
# --------------------------------------------------------------------------- #
def run_single(base_cfg: Mapping[str, Any], *, dataset: str, model: str,
               variant: str, seed: int, figures: bool = True,
               aggregate: bool = False) -> dict:
    cfg = prepare_config(base_cfg, dataset=dataset, model=model, variant=variant, seed=seed)
    set_seed(seed, deterministic=bool(cfg["runtime"]["deterministic"]))
    torch = configure_torch(cfg)
    device = torch.device(str(cfg["runtime"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device=cuda 但当前环境不可用")

    run = RunContext(cfg, dataset=dataset, model=model, variant=variant, seed=seed)
    run.save_config(cfg)

    prepared = prepare_data(cfg, dataset, seed=seed)
    run.save_label_map(prepared["label_map"])

    datasets, stats = prepared["datasets"], prepared["stats"]
    uses = model_inputs(model)
    X = {split: datasets[split] for split in ("train", "val", "test")}

    # ---- 类别权重（仅训练集） ----
    pos_weight_cfg = cfg["loss"]["binary"]["pos_weight"]
    pos_weight = compute_pos_weight(stats["train"]) if pos_weight_cfg == "auto" \
        else float(pos_weight_cfg)
    n_classes = len(prepared["label_map"])
    # ★ class 0（normal）也必须计入，否则 balanced 会给它 0 权重、normal 窗口对
    #   类型头完全没有梯度（实现中曾把 n_by_type 直接传入，缺少 class 0）
    counts = dict(stats["train"]["n_by_type"])
    counts[0] = int(stats["train"]["n_negative"])
    class_weight = compute_class_weight(counts, n_classes) \
        if str(cfg["loss"]["multiclass"]["class_weight"]) == "balanced" else None
    log.info("[%s] pos_weight=%.3f（train prev=%.4f），n_classes=%d",
             dataset, pos_weight, stats["train"]["prevalence"], n_classes)

    # ---- 模型 / 损失 / 加载器 ----
    mv = cfg["model"]["variants"][model]
    net = build_model(model, mv, n_classes=n_classes,
                      window_T=int(cfg["data"]["window"]["T"])).to(device)
    # ---- 逐帧辅助损失的超参（帧级 pos_weight 只由训练集估计） ----
    aux_cfg = cfg["loss"]["aux_frame"] if "aux_frame" in cfg["loss"] else {"enabled": False}
    aux_w = float(aux_cfg["weight"]) if bool(aux_cfg["enabled"]) else 0.0
    frame_pw = 1.0
    if aux_w > 0:
        fs = X["train"].frame_stats()
        frame_pw = (fs["n_negative"] / fs["n_positive"]) if fs["n_positive"] else 1.0
        log.info("[%s] aux_frame=on（weight=%.2f，帧级 pos_weight=%.1f，帧正率=%.4f）",
                 dataset, aux_w, frame_pw, fs["positive_ratio"])
    criterion = MultiTaskLoss(cfg, pos_weight=pos_weight, class_weight=class_weight,
                              n_classes=n_classes, aux_frame_weight=aux_w,
                              frame_pos_weight=frame_pw).to(device)

    train_ds = X["train"]
    sampler = build_capture_sampler(
        train_ds, cfg, dataset_weight=float(cfg["data"]["sampling"]["dataset_weights"][dataset]),
        seed=seed)
    train_loader = make_loader(train_ds, cfg, split="train",
                               batch_size=int(cfg["train"]["batch_size"]),
                               shuffle=True, sampler=sampler)
    val_loader = make_loader(X["val"], cfg, split="val",
                             batch_size=int(cfg["train"]["eval_batch_size"]))
    model_stats = {
        "n_params": int(count_parameters(net)),
        "n_params_trainable": int(count_parameters(net)),
        "inputs": list(uses),
        "channels": int(mv.get("in_channels", 0)),
        "v2_channels": int(cfg["data"]["v2"].get("active_channels", 0)),
        "receptive_field": int(getattr(net, "receptive_field", 0)),
        "macs": int(count_macs(
            net, needs=uses, window_T=int(cfg["data"]["window"]["T"]),
            v1_channels=int(cfg["data"]["v1"].get("n_channels", 16)),
            v2_channels=int(cfg["data"]["v2"].get("active_channels", 8)))),
    }
    if hasattr(net, "receptive_field"):
        log.info("[%s] %s 参数量=%d，感受野=%d 帧", dataset, model,
                 model_stats["n_params"], model_stats["receptive_field"])

    run.write_manifest(effective_cfg=cfg, manifest_summary=prepared["manifest_summary"],
                       dataset_stats=stats, model_stats=model_stats,
                       env=env_snapshot(cfg))

    # ---- 训练 ----
    trainer = Trainer(cfg=cfg, model=net, criterion=criterion,
                      train_loader=train_loader, val_loader=val_loader, run=run,
                      label_map=prepared["label_map"],
                      channels=list(cfg["data"]["v1"]["channels"]),
                      v2_channels=int(cfg["data"]["v2"].get("active_channels", 0)),
                      manifest_summary=prepared["manifest_summary"], device=device)
    train_summary = trainer.fit()

    # ---- 评估（val 定 τ；test 原样复用） ----
    tau = float(trainer.best_tau if trainer.best_tau is not None else float("nan"))
    if not np.isfinite(tau):
        raise RuntimeError("训练未产出有效阈值（best_tau 为空），检查 val 集是否含正样本")

    val_res = evaluate(cfg, run, X["val"], net, split="val",
                       label_map=prepared["label_map"], threshold=tau, device=device)
    test_res = evaluate(cfg, run, X["test"], net, split="test",
                        label_map=prepared["label_map"], threshold=tau, device=device)

    # ---- 汇总 ----
    size_mb = round(run.best_ckpt.stat().st_size / (1024 * 1024), 4) if run.best_ckpt.exists() else 0.0
    model_stats["checkpoint_mb"] = size_mb
    model_stats["int8_estimate_mb"] = round(model_stats["n_params"] / 1e6, 4)  # INT8 ≈ 1 B/param
    # 训练后补写 manifest（此时才有体积信息）
    run.write_manifest(effective_cfg=cfg, manifest_summary=prepared["manifest_summary"],
                       dataset_stats=stats, model_stats=model_stats,
                       env=env_snapshot(cfg))
    summary = {
        "dataset": dataset, "model": model, "variant": variant, "seed": int(seed),
        "run_dir": str(run.run_dir),
        "config_fingerprint": C.fingerprint(cfg),
        "preprocess_fingerprint": prepared["manifest_summary"].get("config_fingerprint_preprocess"),
        "training": train_summary,
        "threshold": tau,
        "dataset_stats": stats,
        "model_stats": model_stats,
        "val": val_res["metrics"],
        "test": test_res["metrics"],
        "zero_shot_types": zero_shot_types(prepared["label_map"],
                                           str(cfg["label_space"]["unknown_label"]),
                                           X["test"]),
    }
    write_json(run.path("metrics_summary"), summary)

    # ---- 出图 ----
    if figures:
        F.generate_for_run(run.run_dir, cfg)
    if aggregate:
        # 汇总图需扫描全部 run，放到批次末尾统一生成一次即可
        F.generate_aggregate(cfg)

    _update_runs_index(cfg)
    log.info("[%s/%s/%s/seed%d] 完成 → %s", dataset, model, variant, seed, run.run_dir)
    return summary


# --------------------------------------------------------------------------- #
# 全量索引（供论文表格与汇总图直接读取）
# --------------------------------------------------------------------------- #
def _update_runs_index(cfg: Mapping[str, Any]) -> Path:
    """重扫 output_root，写一份扁平的 runs_index.csv。"""
    rows: list[dict] = []
    for ctx in F.discover_runs(F.batch_root(cfg), cfg):
        m = ctx.metrics("test")
        sm = ctx.summary()
        ms = dict(ctx.manifest.get("model_stats", {}))
        ms.update(sm.get("model_stats", {}) or {})   # summary 里的体积信息更新
        tr = sm.get("training", {})
        rows.append({
            "dataset": ctx.dataset, "model": ctx.model, "variant": ctx.variant,
            "seed": ctx.seed, "run_dir": str(ctx.dir),
            "epochs_run": tr.get("epochs_run"), "best_epoch": tr.get("best_epoch"),
            "threshold": m.get("threshold"),
            "prevalence": m.get("prevalence"),
            "mcc": m.get("mcc"), "auc_pr": m.get("auc_pr"), "auc_pr_norm": m.get("auc_pr_norm"),
            "f1": m.get("f1"), "precision": m.get("precision"), "recall": m.get("recall"),
            "fpr": m.get("fpr"),
            "n_params": ms.get("n_params"), "checkpoint_mb": ms.get("checkpoint_mb"),
            "train_seconds": tr.get("train_seconds"),
        })
    p = F.batch_root(cfg) / "runs_index.csv"
    if rows:
        write_csv(p, rows)
    return p


# --------------------------------------------------------------------------- #
# 仅评估（不训练）：从已有 run 目录复现指标并重绘图表
# --------------------------------------------------------------------------- #
def reevaluate(run_dir: str | Path, *, figures: bool = True,
               aggregate: bool = False,
               base_cfg: Mapping[str, Any] | None = None) -> dict:
    """用 run 目录里的 `config_resolved.yaml` + `checkpoints/best.pt` 重跑评估。

    - 标签表与通道表**取自 checkpoint**，避免"重新推断标签表"造成口径漂移；
    - 阈值取自 checkpoint（训练时在 val 上标定），不在 test 上重选。
    """
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"缺少配置快照：{cfg_path}")
    cfg = C.load_snapshot(cfg_path, base_cfg) if base_cfg is not None else C.load(cfg_path)

    run = RunContext(cfg, dataset=cfg["data"]["datasets"][0],
                     model=cfg["model"]["name"],
                     variant=str(cfg.get("_meta", {}).get("variant") or "full"),
                     seed=int(cfg["train"]["seeds"][0]))
    run.run_dir = run_dir
    run.ckpt_dir = run_dir / "checkpoints"
    run.fig_dir = run_dir / str(cfg["paths"]["figures_subdir"])
    run.fig_dir.mkdir(parents=True, exist_ok=True)

    from .evaluator import load_run_for_eval
    torch = configure_torch(cfg)
    device = torch.device(str(cfg["runtime"]["device"]))
    model, ckpt = load_run_for_eval(run.best_ckpt if run.best_ckpt.exists() else run.final_ckpt,
                                    device=str(device))

    dataset = ckpt["dataset"]
    prepared = prepare_data(cfg, dataset, seed=int(ckpt["seed"]))

    # 重新计算静态效率指标（参数量/MACs/感受野），使 T12 对历史 run 也能出数
    mv = cfg["model"]["variants"][ckpt["model_name"]]
    uses = model_inputs(str(ckpt["model_name"]))
    model_stats = {
        "n_params": int(count_parameters(model)),
        "n_params_trainable": int(count_parameters(model)),
        "inputs": list(uses),
        "channels": int(mv.get("in_channels", 0)),
        "v2_channels": int(cfg["data"]["v2"].get("active_channels", 0)),
        "receptive_field": int(getattr(model, "receptive_field", 0)),
        "macs": int(count_macs(model, needs=uses,
                               window_T=int(cfg["data"]["window"]["T"]),
                               v1_channels=int(cfg["data"]["v1"].get("n_channels", 16)),
                               v2_channels=int(cfg["data"]["v2"].get("active_channels", 8)))),
        "checkpoint_mb": round(run.best_ckpt.stat().st_size / (1024 * 1024), 4)
        if run.best_ckpt.exists() else None,
        "int8_estimate_mb": round(int(count_parameters(model)) / 1e6, 4),
    }
    label_map = dict(ckpt["label_map"])
    tau = ckpt.get("threshold")
    if tau is None:
        raise RuntimeError("checkpoint 未记录阈值，无法复现 test 指标")

    datasets = prepared["datasets"]
    val_res = evaluate(cfg, run, datasets["val"], model, split="val",
                       label_map=label_map, threshold=float(tau), device=device)
    test_res = evaluate(cfg, run, datasets["test"], model, split="test",
                        label_map=label_map, threshold=float(tau), device=device)

    summary_path = run.path("metrics_summary")
    summary = read_json_local(summary_path) if summary_path.exists() else {}
    summary["checkpoint_metrics"] = dict(ckpt.get("metrics", {}))
    summary.update({
        "dataset": dataset, "model": ckpt["model_name"], "variant": ckpt.get("variant"),
        "seed": int(ckpt["seed"]), "run_dir": str(run_dir),
        "threshold": float(tau), "val": val_res["metrics"], "test": test_res["metrics"],
        "model_stats": model_stats,
        "re_evaluated": True,
    })
    # 同步刷新 run_manifest，使 T12 / fig09 读取到最新静态指标
    man_path = run.path("run_manifest")
    man = read_json_local(man_path) if man_path.exists() else {}
    man["model_stats"] = model_stats
    write_json(man_path, man)
    write_json(summary_path, summary)
    if figures:
        F.generate_for_run(run_dir, cfg)
    if aggregate:
        F.generate_aggregate(cfg)
    _update_runs_index(cfg)
    return summary


def read_json_local(path: str | Path) -> dict:
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))
