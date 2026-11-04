# -*- coding: utf-8 -*-
"""
training.artifacts —— 运行目录、断点与清单
================================================================================
一个 run（数据集 × 模型 × 消融变体 × 种子）的全部产物都落在同一目录：

    <output_root>/<run_name>/<dataset>/<model>/<variant>/seed<seed>/
    ├─ config_resolved.yaml        ← 本次实际生效的配置（含消融派生结果）
    ├─ label_map.json              ← 类别表（评估必须复用它，禁止重新推断）
    ├─ env.json / run_manifest.json
    ├─ history.csv
    ├─ checkpoints/{best,last,model_final}.pt
    ├─ metrics_{split}.json / per_type_{split}.csv / predictions_{split}.npz
    ├─ threshold_curve_{split}.csv
    └─ figures/*.pdf|png

断点里同时存 `state_dict` 与**口径元数据**（通道表、label_map、窗长、指纹），
因此 `run_eval.py` 可以只给一个 .pt 路径就完成复现，不需要再读训练配置。

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from . import config as C
from .utils import write_json


class RunContext:
    """单个 run 的目录与产物路径管理。"""

    def __init__(self, cfg: Mapping[str, Any], *, dataset: str, model: str,
                 variant: str, seed: int):
        self.cfg = cfg
        self.dataset = dataset
        self.model = model
        self.variant = variant
        self.seed = int(seed)
        self.run_dir = C.resolve_run_dir(cfg, dataset, model, variant, seed)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.fig_dir = self.run_dir / str(cfg["paths"]["figures_subdir"])
        for d in (self.run_dir, self.ckpt_dir, self.fig_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def path(self, key: str, **fmt: Any) -> Path:
        return C.artifact_path(self.cfg, self.run_dir, key, **fmt)

    @property
    def best_ckpt(self) -> Path:
        return self.path("checkpoint.best")

    @property
    def last_ckpt(self) -> Path:
        return self.path("checkpoint.last")

    @property
    def final_ckpt(self) -> Path:
        return self.path("checkpoint.final")

    # ------------------------------------------------------------------ #
    def save_config(self, effective_cfg: Mapping[str, Any]) -> Path:
        """把"实际生效"的配置存成 YAML（可直接 diff 复现）。"""
        p = self.path("config_snapshot")
        payload = {k: v for k, v in effective_cfg.items() if not k.startswith("_")}
        payload["_meta"] = {
            "config_path": effective_cfg.get("_meta", {}).get("config_path"),
            "config_sha256_16": effective_cfg.get("_meta", {}).get("config_sha256_16"),
            "variant": self.variant,
            "overrides": effective_cfg.get("_meta", {}).get("overrides", {}),
            "fingerprint": C.fingerprint(effective_cfg),
        }
        p.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return p

    def save_label_map(self, label_map: Mapping[str, int]) -> Path:
        return write_json(self.path("label_map"), {
            "label_map": dict(label_map),
            "n_classes": len(label_map),
            "dataset": self.dataset,
        })

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: Path, *, model, optimizer, epoch: int,
                        metrics: Mapping[str, Any], label_map: Mapping[str, int],
                        channels: list[str], v2_channels: int,
                        manifest_summary: Mapping[str, Any],
                        threshold: float | None = None,
                        history: list[dict] | None = None) -> Path:
        """保存权重 + 口径元数据（评估侧只依赖这些元数据）。"""
        import torch
        payload = {
            "format_version": 1,
            "model_name": self.model,
            "model_cfg": dict(self.cfg["model"]["variants"][self.model]),
            "n_classes": len(label_map),
            "window_T": int(self.cfg["data"]["window"]["T"]),
            "channels": list(channels),
            "v2_channels": int(v2_channels),
            "label_map": dict(label_map),
            "variant": self.variant,
            "seed": self.seed,
            "dataset": self.dataset,
            "epoch": int(epoch),
            "metrics": dict(metrics),
            "threshold": threshold,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "config_fingerprint": C.fingerprint(self.cfg),
            "preprocess_fingerprint": manifest_summary.get("config_fingerprint_preprocess"),
            "manifest": dict(manifest_summary),
            "config_path": self.cfg.get("_meta", {}).get("config_path"),
            "history": list(history or []),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        # 体积由 run_manifest / runs_index 记录（torch.save 之前无法知道文件大小）
        return path

    @staticmethod
    def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
        import torch
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"checkpoint 不存在：{p}")
        return torch.load(p, map_location=map_location, weights_only=False)

    # ------------------------------------------------------------------ #
    def write_manifest(self, *, effective_cfg: Mapping[str, Any],
                       manifest_summary: Mapping[str, Any],
                       dataset_stats: Mapping[str, Any],
                       model_stats: Mapping[str, Any],
                       env: Mapping[str, Any]) -> Path:
        return write_json(self.path("run_manifest"), {
            "run_dir": str(self.run_dir),
            "dataset": self.dataset,
            "model": self.model,
            "variant": self.variant,
            "seed": self.seed,
            "config_fingerprint": C.fingerprint(effective_cfg),
            "preprocess": dict(manifest_summary),
            "dataset_stats": dict(dataset_stats),
            "model_stats": dict(model_stats),
            "env": dict(env),
        })
