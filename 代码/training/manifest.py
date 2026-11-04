# -*- coding: utf-8 -*-
"""
training.manifest —— 预处理清单（03_manifest）的只读索引
================================================================================
本模块是训练侧与预处理侧**唯一**的数据契约：
  - 帧级张量在 `<dataset>/02_tensors/<split>/<capture>/*.npy`
  - 切分区间在 `03_manifest/splits.json`（每 capture 的 ranges）与
    `03_manifest/captures/<capture>.json`（含 attack_type / 窗口数 / train_stats）
  - 口径常量在 `03_manifest/env.json`（用于交叉校验，防止训练读了旧产物）

★ 为什么必须按 ranges 建窗：
  块化交错切分下，一个 capture 的同一 split 由多段**不连续**区间组成。
  若直接在整条 capture 上滑动，会产生"跨块窗口"——既跨越了 split 边界（数据泄漏），
  也使 McNemar 的样本独立性失效。因此窗口**只在每个 range 内部**构造。

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

SPLITS: tuple[str, ...] = ("train", "val", "test")


class ManifestError(RuntimeError):
    """清单缺失或自相矛盾。"""


@dataclass(frozen=True)
class CaptureSpec:
    """单个采集文件在某个数据集中的元信息。"""

    dataset: str
    name: str
    protocol: str
    attack_type: str
    attack_family: str
    is_masquerade: bool
    supervised_evaluable: bool
    n_frames: int
    ranges: dict[str, tuple[tuple[int, int], ...]] = field(default_factory=dict)
    expected_windows: dict[str, int] = field(default_factory=dict)

    def n_frames_in(self, split: str) -> int:
        return sum(b - a for a, b in self.ranges.get(split, ()))

    @property
    def usable_as_normal(self) -> bool:
        """是否可作为正常流量样本（attack_type == none）。"""
        return self.attack_type in ("none", "")

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset, "capture": self.name, "protocol": self.protocol,
            "attack_type": self.attack_type, "attack_family": self.attack_family,
            "is_masquerade": self.is_masquerade,
            "supervised_evaluable": self.supervised_evaluable,
            "n_frames": self.n_frames,
            "ranges": {k: [list(r) for r in v] for k, v in self.ranges.items()},
            "expected_windows": dict(self.expected_windows),
        }


def count_windows(range_len: int, window: int, stride: int) -> int:
    """单段区间内的窗口数：0 if L<W else 1 + (L−W)//stride。"""
    if range_len < window:
        return 0
    return 1 + (range_len - window) // stride


class ManifestIndex:
    """某数据集 03_manifest 的只读视图。"""

    def __init__(self, dataset: str, manifest_dir: str | Path):
        self.dataset = dataset
        self.dir = Path(manifest_dir)
        self.splits_path = self.dir / "splits.json"
        self.env_path = self.dir / "env.json"
        self.captures_dir = self.dir / "captures"
        for p in (self.splits_path, self.env_path, self.captures_dir):
            if not Path(p).exists():
                raise ManifestError(f"[{dataset}] 清单缺失：{p}")

        self.splits_doc: dict = json.loads(self.splits_path.read_text(encoding="utf-8"))
        self.env: dict = json.loads(self.env_path.read_text(encoding="utf-8"))
        self._captures: dict[str, CaptureSpec] = {}
        self._load_captures()

    # ------------------------------------------------------------------ #
    def _load_captures(self) -> None:
        for p in sorted(self.captures_dir.glob("*.json")):
            doc = json.loads(p.read_text(encoding="utf-8"))
            name = doc.get("capture") or p.stem
            ranges: dict[str, tuple[tuple[int, int], ...]] = {}
            expected: dict[str, int] = {}
            for sp in SPLITS:
                node = (doc.get("splits") or {}).get(sp) or {}
                raw = node.get("ranges") or []
                # 兼容 splits.json 的单层 [a,b] 写法
                if raw and isinstance(raw[0], int):
                    raw = [raw]
                ranges[sp] = tuple((int(a), int(b)) for a, b in raw if int(b) > int(a))
                if "windows_stride_train" in node:
                    key = "windows_stride_train" if sp == "train" else "windows_stride_test"
                    expected[sp] = int(node.get(key, 0))
            self._captures[name] = CaptureSpec(
                dataset=self.dataset,
                name=name,
                protocol=str(doc.get("protocol", "")),
                attack_type=str(doc.get("attack_type", "none")),
                attack_family=str(doc.get("attack_family", "none")),
                is_masquerade=bool(doc.get("is_masquerade", False)),
                supervised_evaluable=bool(doc.get("supervised_evaluable", True)),
                n_frames=int(doc.get("n_frames", 0)),
                ranges=ranges,
                expected_windows=expected,
            )

    # ------------------------------------------------------------------ #
    @property
    def captures(self) -> list[CaptureSpec]:
        return [self._captures[k] for k in sorted(self._captures)]

    def captures_in(self, split: str, *, include_unsupervised: bool = True) -> list[CaptureSpec]:
        out = [c for c in self.captures if c.ranges.get(split)]
        if not include_unsupervised:
            out = [c for c in out if c.supervised_evaluable]
        return out

    def expected_windows(self, split: str) -> int:
        return sum(c.expected_windows.get(split, 0) for c in self.captures)

    def attack_types(self, split: str, *, include_unsupervised: bool = True) -> dict[str, int]:
        """split 内各攻击类型的采集文件数（不含 none）。"""
        out: dict[str, int] = {}
        for c in self.captures_in(split, include_unsupervised=include_unsupervised):
            if c.usable_as_normal:
                continue
            out[c.attack_type] = out.get(c.attack_type, 0) + 1
        return dict(sorted(out.items()))

    # ------------------------------------------------------------------ #
    def verify(self, cfg: Mapping[str, Any]) -> dict:
        """与配置 / env.json 的一致性校验，返回一份可写进 run_manifest 的摘要。"""
        win = int(cfg["data"]["window"]["T"])
        strides = cfg["data"]["window"]["stride"]

        # 1) env.json 口径必须与配置一致（否则说明训练读了旧产物）
        env_const = self.env.get("constants", {})
        diffs: list[str] = []
        if int(env_const.get("WIN_T", -1)) != win:
            diffs.append(f"WIN_T: env={env_const.get('WIN_T')} cfg={win}")
        for sp in ("train", "test"):
            if int(env_const.get(f"STRIDE_{sp.upper()}", -1)) != int(strides[sp]):
                diffs.append(f"STRIDE_{sp}: env={env_const.get('STRIDE_' + sp.upper())} cfg={strides[sp]}")

        # 2) 通道名与顺序（只告警不致命：Ablation 会主动裁剪，但必须是子集）
        env_ch = list(env_const.get("V1_CHANNELS") or [])
        cfg_ch = list(cfg["data"]["v1"]["channels"])
        unknown = [c for c in cfg_ch if c not in env_ch]
        if unknown:
            diffs.append(f"配置含 env.json 未定义的通道：{unknown}")

        if diffs:
            raise ManifestError(
                f"[{self.dataset}] 配置与产物清单不一致，拒绝训练：\n  - " + "\n  - ".join(diffs)
            )

        # 3) 窗口数复算（防跨块窗口）
        recount: dict[str, int] = {}
        declared: dict[str, int] = {}
        for sp in SPLITS:
            stride = int(strides[sp])
            n = 0
            for c in self.captures:
                n += sum(count_windows(b - a, win, stride) for a, b in c.ranges.get(sp, ()))
            recount[sp] = n
            declared[sp] = self.expected_windows(sp)

        mismatch = {sp: (declared[sp], recount[sp]) for sp in SPLITS
                    if declared[sp] != recount[sp]}
        if mismatch and bool(cfg["data"]["manifest_check"]["verify_window_counts"]):
            raise ManifestError(
                f"[{self.dataset}] 窗口数复算与清单不符（declared, recount）：{mismatch}"
            )

        return {
            "dataset": self.dataset,
            "split_mode": self.splits_doc.get("split_mode", "unknown"),
            "window_T": win,
            "stride": {k: int(v) for k, v in strides.items()},
            "n_captures": len(self.captures),
            "n_frames": {sp: sum(c.n_frames_in(sp) for c in self.captures) for sp in SPLITS},
            "n_windows_declared": declared,
            "n_windows_recount": recount,
            "config_fingerprint_preprocess": self.env.get("config_fingerprint_sha256_16"),
            "attack_types": {sp: self.attack_types(sp) for sp in SPLITS},
        }


# --------------------------------------------------------------------------- #
# 标签空间
# --------------------------------------------------------------------------- #
def build_label_map(index: ManifestIndex, cfg: Mapping[str, Any]) -> dict[str, int]:
    """从 train 侧 manifest 派生多分类类别表。

    规则（全部来自配置，不硬编码类名）：
      - 只统计 train split 中 attack_type != none 的采集文件
      - 类名按字典序排序（跨机器稳定）
      - 0 号类恒为 normal
      - unknown_class=true 时追加 unknown_attack，承载 test 侧 zero-shot 类型
    """
    ls = cfg["label_space"]
    labels: list[str] = ["normal"]
    if ls["multiclass"]:
        min_w = int(ls["min_train_windows"])
        types = index.attack_types("train")
        # 以"该类型在 train 的窗口数"过筛，避免把只有个位数样本的类型当独立类
        win_by_type = window_counts_by_type(index, cfg, "train")
        keep = [t for t in types if win_by_type.get(t, 0) >= min_w]
        labels.extend(sorted(keep))
        if ls["unknown_class"]:
            labels.append(str(ls["unknown_label"]))
    return {name: i for i, name in enumerate(labels)}


def window_counts_by_type(index: ManifestIndex, cfg: Mapping[str, Any], split: str) -> dict[str, int]:
    """按 attack_type 统计 split 内可用窗口数（近似：按帧占比，用于类别过滤与报表）。"""
    win = int(cfg["data"]["window"]["T"])
    stride = int(cfg["data"]["window"]["stride"][split])
    out: dict[str, int] = {}
    for c in index.captures_in(split):
        if c.usable_as_normal:
            continue
        n = sum(count_windows(b - a, win, stride) for a, b in c.ranges.get(split, ()))
        out[c.attack_type] = out.get(c.attack_type, 0) + n
    return out


def iter_capture_jsons(manifest_dir: str | Path) -> Iterable[dict]:
    """遍历 captures/*.json（供报表/统计脚本复用）。"""
    for p in sorted((Path(manifest_dir) / "captures").glob("*.json")):
        yield json.loads(p.read_text(encoding="utf-8"))


def read_dataset_stats(manifest_dir: str | Path) -> list[dict]:
    """读取 dataset_stats.csv（帧级口径，用于论文 Table T14/T5）。"""
    p = Path(manifest_dir) / "dataset_stats.csv"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))
