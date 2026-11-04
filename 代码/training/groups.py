# -*- coding: utf-8 -*-
"""
training.groups —— capture → 录制组（SET）映射
================================================================================
为什么需要这一层
----------------
簇级 McNemar 把每个 capture 当作一个可交换单元。但 CTT v1.5 的 224 个 test
capture 并非同一次录制：文件名前缀 `set_01`–`set_04` 对应 4 次独立录制会话
（不同车辆状态 / 日志 / 注入会话 / 采集环境）。同一 SET 内的文件共享这些条件，
彼此高度相关 —— 把 224 个文件当独立单元，仍是 pseudoreplication，只是升了一级。

本模块只做一件事：给定 dataset + split，返回与 `predictions_*.npz` 的 `cap`
下标**逐位对齐**的组标签数组。对齐规则与 `data.WindowDataset` 完全一致：

    cap = 该 split 下 captures 的**出现顺序**下标（splits.json 的插入顺序），
          过滤条件 = 该 capture 在 split 下有非空 ranges。

`verify_alignment()` 用 manifest 的区间长度复算窗口数，与 npz 的 bincount
逐簇比对，不一致直接报错 —— 防止"看似对齐、实则错位"（本项目已踩过一次）。

作者：小巴（WorkBuddy）    日期：2026-10-30（审计整改 D4-1 / F3）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

# 文件名里分隔"录制组 / 场景 / 家族"的记号（CTT v1.5）
GROUP_SEP = "__"


def manifest_path(cfg: Mapping, dataset: str, split: str = "test") -> Path:
    """`<dataset_root>/<manifest_subdir>/splits.json`（与 config.resolve_paths 同源）。"""
    root = Path(cfg["_resolved"]["dataset_roots"][dataset])
    return root / str(cfg["paths"]["manifest_subdir"]) / "splits.json"


def capture_names(cfg: Mapping, dataset: str, split: str = "test") -> list[str]:
    """按 `data.WindowDataset` 的同一规则给出 cap 下标对应的 capture 名。"""
    p = manifest_path(cfg, dataset, split)
    if not p.exists():
        raise FileNotFoundError(f"缺少切分清单：{p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    caps = obj.get("captures", {})
    # ★ 顺序必须与 manifest.ManifestIndex.captures 一致（后者用 sorted 键名），
    #   否则 cap 下标与组标签会错位——本项目的 cap 就是"排序后的出现顺序"。
    out = [name for name in sorted(caps)
           if (caps[name].get("ranges") or {}).get(split)]
    if not out:
        raise ValueError(f"[{dataset}/{split}] splits.json 中没有可用 capture")
    return out


def group_of(name: str, sep: str = GROUP_SEP) -> str:
    """capture 名 → 录制组名。

    CTT：`set_01__test_01_...__DoS-3` → `set_01`；
    HCRL / ROAD 无 `__` 前缀约定 → 组即 capture 本身（等价于单层聚类）。
    """
    return name.split(sep)[0] if sep in name else name


def capture_groups(cfg: Mapping, dataset: str, split: str = "test",
                   sep: str = GROUP_SEP) -> np.ndarray:
    """返回长度 = 簇数（= capture 数）的组标签数组，下标即 `cap`。"""
    return np.asarray([group_of(n, sep) for n in capture_names(cfg, dataset, split)])


def verify_alignment(cfg: Mapping, dataset: str, cap: np.ndarray,
                     split: str = "test") -> dict:
    """校验 `cap` 与 splits.json 的 capture 顺序逐位对齐。

    只比"每簇窗口数"这一个自由度已足够强：任何错位都会让某个簇的计数与
    manifest 复算值不等（本项目 CTT 实测 224/224 全等）。
    """
    names = capture_names(cfg, dataset, split)
    p = manifest_path(cfg, dataset, split)
    obj = json.loads(p.read_text(encoding="utf-8"))
    caps = obj["captures"]
    win_T = int(obj.get("window_T", cfg["data"]["window"]["T"]))
    stride = int(obj.get("stride", {}).get(split, cfg["data"]["window"]["stride"][split]))

    expected = []
    for n in names:
        k = 0
        for a, b in caps[n]["ranges"][split]:
            seg = int(b) - int(a)
            if seg >= win_T:
                k += (seg - win_T) // stride + 1
        expected.append(k)
    expected = np.asarray(expected, dtype=np.int64)
    got = np.bincount(np.asarray(cap).astype(np.int64), minlength=expected.size)
    got = got[:expected.size]
    if expected.size != got.size or not np.array_equal(expected, got):
        bad = [(i, int(expected[i]), int(got[i]))
               for i in range(min(expected.size, got.size)) if expected[i] != got[i]][:5]
        raise ValueError(f"[{dataset}/{split}] cap 与 splits.json 未对齐（前 5 处：{bad}）")
    return {"n_caps": int(expected.size), "n_windows": int(expected.sum()),
            "aligned": True}


def summarize_groups(groups: Sequence[str], cap: np.ndarray) -> list[dict]:
    """每组的簇数与窗口数（写进产物表头，便于复核）。"""
    g = np.asarray(groups)
    c = np.asarray(cap)
    out: list[dict] = []
    for name in dict.fromkeys(g.tolist()):
        m = g == name
        out.append({"group": name, "n_clusters": int(m.sum()),
                    "n_windows": int(np.isin(c, np.flatnonzero(m)).sum())})
    return out
