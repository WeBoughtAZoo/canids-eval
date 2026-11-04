# -*- coding: utf-8 -*-
"""
training.trainer —— 训练循环（早停 / 余弦退火 / 最优权重回滚）
================================================================================
关键约定
--------
1. **早停监控 val MCC**（不是 loss）：MCC 在不平衡数据上与业务目标一致，
   loss 会被多数类主导（§4.8）。
2. 每个 epoch 在 val 上重新标定 τ 并记录 `val_mcc_calibrated`；
   最终用于 test 的 τ 来自**最优 epoch 的那一次标定**，随 checkpoint 一起保存。
3. 只保存"最优"与"最后"两个断点，磁盘占用可控（含 optimizer 状态）。

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from . import calibration as calib
from . import metrics as M
from .artifacts import RunContext
from .evaluator import predict
from .utils import get_logger, write_csv

log = get_logger("trainer")


# --------------------------------------------------------------------------- #
# 学习率调度
# --------------------------------------------------------------------------- #
def _make_scheduler(optimizer, cfg: Mapping[str, Any], total_steps: int):
    import torch
    name = str(cfg["train"]["scheduler"]["name"])
    if name == "none":
        return None
    if name != "cosine":
        raise ValueError(f"未知 scheduler={name}")

    warmup = int(cfg["train"]["scheduler"]["warmup_epochs"])
    eta_min_ratio = float(cfg["train"]["scheduler"]["eta_min_ratio"])
    warmup_steps = 0
    if warmup > 0:
        warmup_steps = max(1, int(total_steps * warmup / max(1, int(cfg["train"]["epochs"]))))

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        span = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / span)
        return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _make_optimizer(model, cfg: Mapping[str, Any]):
    import torch
    o = cfg["train"]["optimizer"]
    name = str(o["name"])
    kwargs = dict(lr=float(o["lr"]), weight_decay=float(o["weight_decay"]))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), betas=tuple(o["betas"]), **kwargs)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), betas=tuple(o["betas"]), **kwargs)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=0.9, **kwargs)
    raise ValueError(f"未知 optimizer={name}")


# --------------------------------------------------------------------------- #
# 训练器
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, *, cfg: Mapping[str, Any], model, criterion,
                 train_loader, val_loader, run: RunContext,
                 label_map: Mapping[str, int], channels: list[str],
                 v2_channels: int, manifest_summary: Mapping[str, Any],
                 device):
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run = run
        self.label_map = dict(label_map)
        self.channels = list(channels)
        self.v2_channels = int(v2_channels)
        self.manifest_summary = dict(manifest_summary)
        self.device = device

        self.optimizer = _make_optimizer(model, cfg)
        total_steps = int(cfg["train"]["epochs"]) * max(1, len(train_loader))
        self.scheduler = _make_scheduler(self.optimizer, cfg, total_steps)

        self.history: list[dict[str, Any]] = []
        self.best_score: float | None = None
        self.best_epoch = 0
        self.best_tau: float | None = None
        self.best_metrics: dict[str, Any] = {}
        self._bad_epochs = 0

    # ------------------------------------------------------------------ #
    def _monitor_value(self, val_metrics: Mapping[str, Any], val_loss: float) -> float:
        es = self.cfg["train"]["early_stopping"]
        key = str(es["monitor"])
        if key == "val_loss":
            return float(val_loss)
        if key not in val_metrics:
            raise KeyError(f"早停监控指标 {key} 不在 val 指标中（可选 {sorted(val_metrics)}）")
        val = val_metrics[key]
        return float(val) if val is not None and np.isfinite(val) else -math.inf

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        import torch
        self.model.train()
        t0 = time.perf_counter()
        agg = {"loss": 0.0, "loss_bin": 0.0, "loss_type": 0.0, "loss_frame": 0.0}
        n = 0
        clip = float(self.cfg["train"]["grad_clip_norm"])
        log_every = int(self.cfg["train"]["log_interval"])

        for step, batch in enumerate(self.train_loader, start=1):
            x1 = batch["x1"].to(self.device) if "x1" in batch else None
            x2 = batch["x2"].to(self.device) if "x2" in batch else None
            idv = batch["idv"].to(self.device) if "idv" in batch else None
            y = batch["y"].to(self.device)
            yt = batch["y_type"].to(self.device)
            yf = batch["y_frame"].to(self.device) if "y_frame" in batch else None

            kwargs = {}
            if x1 is not None:
                kwargs["x1"] = x1
            if x2 is not None:
                kwargs["x2"] = x2
            if idv is not None:
                kwargs["idv"] = idv

            out = self.model(**kwargs)
            losses = self.criterion(out, y, yt, yf)

            self.optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            n += 1
            for k in agg:
                agg[k] += float(losses[k].detach())
            if log_every and step % log_every == 0:
                log.info("  epoch %d | step %d/%d | loss=%.4f (bin=%.4f, type=%.4f) | lr=%.2e",
                         epoch, step, len(self.train_loader),
                         agg["loss"] / n, agg["loss_bin"] / n, agg["loss_type"] / n,
                         self.optimizer.param_groups[0]["lr"])

        out_stats = {f"train_{k}": v / max(n, 1) for k, v in agg.items()}
        out_stats["train_seconds"] = time.perf_counter() - t0
        out_stats["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return out_stats

    # ------------------------------------------------------------------ #
    def _validate(self, epoch: int) -> dict[str, Any]:
        # 始终记录 val loss（曲线图与早停诊断都要用），代价仅一次 criterion 前向
        pred = predict(self.model, self.val_loader, device=self.device,
                       criterion=self.criterion)
        y, p = pred["label"], pred["prob"]

        cal = calib.calibrate(y, p, self.cfg)
        tau = float(cal["threshold"])
        m_cal = M.binary_metrics(y, p, tau)
        m_05 = M.binary_metrics(y, p, 0.5)

        row = {
            "epoch": int(epoch),
            "val_loss": float(pred.get("loss", float("nan"))),
            "val_auc_pr": m_05["auc_pr"],
            "val_auc_roc": m_05["auc_roc"],
            "val_prevalence": m_05["prevalence"],
            "val_mcc@0.5": m_05["mcc"],
            "val_f1@0.5": m_05["f1"],
            "val_mcc_calibrated": m_cal["mcc"],
            "val_f1_calibrated": m_cal["f1"],
            "val_fpr_calibrated": m_cal["fpr"],
            "val_recall_calibrated": m_cal["recall"],
            "val_threshold": tau,
            "val_calibration_metric": str(self.cfg["calibration"]["metric"]),
        }
        return {"row": row, "tau": tau, "curve": cal["curve"], "pred": pred}

    # ------------------------------------------------------------------ #
    def _save(self, path: Path, epoch: int, metrics: Mapping[str, Any],
              threshold: float | None) -> None:
        self.run.save_checkpoint(
            path, model=self.model, optimizer=self.optimizer, epoch=epoch,
            metrics=metrics, label_map=self.label_map, channels=self.channels,
            v2_channels=self.v2_channels, manifest_summary=self.manifest_summary,
            threshold=threshold, history=self.history,
        )

    # ------------------------------------------------------------------ #
    def fit(self) -> dict[str, Any]:
        cfg = self.cfg
        epochs = int(cfg["train"]["epochs"])
        es = cfg["train"]["early_stopping"]
        patience = int(es["patience"])
        min_delta = float(es["min_delta"])
        mode = str(es["mode"])

        t_start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            tr = self._train_one_epoch(epoch)
            va = self._validate(epoch)
            row = dict(tr, **va["row"])
            self.history.append(row)
            write_csv(self.run.path("history"), self.history,
                      columns=list(self.history[0].keys()))

            score = self._monitor_value(
                {"mcc_calibrated": row["val_mcc_calibrated"],
                 "mcc@0.5": row["val_mcc@0.5"],
                 "auc_pr": row["val_auc_pr"],
                 "val_loss": row["val_loss"]}, row["val_loss"])
            if self.best_score is None:
                improved = True
            elif mode == "max":
                improved = score > self.best_score + min_delta
            else:
                improved = score < self.best_score - min_delta

            if improved:
                self.best_score = score
                self.best_epoch = epoch
                self.best_tau = float(va["tau"])
                self.best_metrics = dict(row)
                self._save(self.run.best_ckpt, epoch, row, self.best_tau)
                self._bad_epochs = 0
                flag = " *best*"
            else:
                self._bad_epochs += 1
                flag = f" (no improve {self._bad_epochs}/{patience})"

            self._save(self.run.last_ckpt, epoch, row, float(va["tau"]))

            log.info("epoch %d/%d | loss=%.4f | val MCC=%.4f (τ=%.3f, @0.5=%.4f) | "
                     "val AUC-PR=%.4f | %.1fs%s",
                     epoch, epochs, row["train_loss"], row["val_mcc_calibrated"],
                     row["val_threshold"], row["val_mcc@0.5"], row["val_auc_pr"],
                     row["train_seconds"], flag)

            if bool(es["enabled"]) and self._bad_epochs >= patience:
                log.info("早停触发：连续 %d 个 epoch 监控指标 %s 未提升", patience, es["monitor"])
                break

        # ---- 回滚到最优权重 ----
        import torch
        if self.run.best_ckpt.exists():
            ckpt = RunContext.load_checkpoint(self.run.best_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["state_dict"])
        self._save(self.run.final_ckpt, self.best_epoch, self.best_metrics, self.best_tau)

        summary = {
            "epochs_run": len(self.history),
            "best_epoch": self.best_epoch,
            "best_score": float(self.best_score) if self.best_score is not None else float("nan"),
            "best_threshold": self.best_tau,
            "best_metrics": self.best_metrics,
            "train_seconds": time.perf_counter() - t_start,
            "monitor": es["monitor"],
            "n_train_batches": len(self.train_loader),
            "n_val_batches": len(self.val_loader),
        }
        log.info("训练完成：best_epoch=%d，best %s=%.4f，τ=%.4f，总耗时 %.1fs",
                 self.best_epoch, es["monitor"], summary["best_score"],
                 self.best_tau or float("nan"), summary["train_seconds"])
        return summary
