# -*- coding: utf-8 -*-
"""
training.data —— 惰性窗口化 Dataset（只读 02_tensors）
================================================================================
为什么不在磁盘上预展开窗口
--------------------------
V1 float16 × 550 万窗口 × 64 × 16 ≈ 11 GB，展开后训练读放大严重；
而帧级张量仅 442 个目录、约 31 GiB，memmap 随机切窗的 I/O 更友好。
因此本模块只在 `__getitem__` 现场切窗（与预处理 `MemmapWindows` 同思路，
但**独立实现**，并按 manifest 的 ranges 建窗，避免跨块窗口）。

三条不可动摇的口径
------------------
1. 窗口**严格落在某个 range 内部**（`start` 与 `start+T` 同属一段区间）；
2. 窗口标签 = 窗内任一帧为攻击（`any_frame`，与预处理完全一致）；
3. 帧级不做任何重采样 / 归一化（编码已在预处理完成）。

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .manifest import CaptureSpec, ManifestIndex, count_windows


# --------------------------------------------------------------------------- #
# 单 capture 的帧级句柄
# --------------------------------------------------------------------------- #
class FrameSource:
    """一个 (capture, split) 目录下的 npy 句柄；只读、可被多 worker 共享。

    ★ 2026-10-27：**按需加载**。`need` 只列出模型真正会读取的数组；
    `label` 恒加载（建窗索引与窗口标签都依赖它，且体积很小）。

    动机：CTT 的 `v2.npy` 达 15.0 GiB、`id_valid.npy` 0.23 GiB，而
    FD-TCN / LSTM / Transformer 的 `needs=("v1",)` 从不读它们。原实现无条件打开，
    导致（a）部署包必须带上这些用不到的大文件，（b）文件一旦不拷就直接 FileNotFoundError。
    注意：这**不是**性能优化——`np.load(mmap_mode="r")` 建完映射即关闭 fd
    （见 numpy `_core/memmap.py` 的 `with f_ctx as fid`），未触碰的映射页也不进 RSS。
    """

    def __init__(self, tensors_dir: Path, capture: str, split: str,
                 need: Sequence[str] = ("v1", "v2", "label", "id_valid")):
        d = Path(tensors_dir) / split / capture
        if not d.exists():
            raise FileNotFoundError(f"缺少张量目录：{d}")
        self.dir = d
        self.capture = capture
        self.split = split
        use = set(need) | {"label"}          # label 恒需要（窗口标签与索引）
        self.label = np.load(d / "label.npy", mmap_mode="r", allow_pickle=False)
        self.n_frames = int(self.label.shape[0])
        self.v1 = (np.load(d / "v1.npy", mmap_mode="r", allow_pickle=False)
                   if "v1" in use else None)
        self.v2 = (np.load(d / "v2.npy", mmap_mode="r", allow_pickle=False)
                   if "v2" in use else None)
        self.id_valid = (np.load(d / "id_valid.npy", mmap_mode="r", allow_pickle=False)
                         if "id_valid" in use else None)

    def __len__(self) -> int:
        # ★ 不再依赖 v1：v2-only 模型（rgbnet）此前靠"v1 一定被加载"侥幸成立
        return self.n_frames


# --------------------------------------------------------------------------- #
# 窗口索引
# --------------------------------------------------------------------------- #
@dataclass
class WindowIndex:
    """扁平窗口索引：第 k 个样本 = (file_idx, start)。"""

    file_idx: np.ndarray     # int32 [N]
    start: np.ndarray        # int64 [N]
    labels: np.ndarray       # uint8 [N]
    type_idx: np.ndarray     # int16 [N]（多分类目标；无攻击为 0）

    def __len__(self) -> int:
        return int(self.start.size)

    def subset(self, idx: np.ndarray) -> "WindowIndex":
        return WindowIndex(self.file_idx[idx], self.start[idx], self.labels[idx], self.type_idx[idx])


class WindowDataset(Dataset):
    """按 manifest ranges 建窗的惰性 Dataset。

    返回 dict（由 DataLoader 自动 collate）：
        x1   : float32 [T, C1]      （V1 通道，已按消融裁剪）
        x2   : float32 [C2, T, 8]   （仅 use_v2=True）
        idv  : float32 [T]          （仅 use_id_valid=True）
        y    : int64 标量           （0/1）
        y_type : int64 标量          （多分类）
        cap  : int64 标量           （capture 下标，用于分层与溯源）
        pos  : int64 标量           （窗口在 capture 内的起始帧号）
    """

    def __init__(
        self,
        tensors_dir: str | Path,
        index: ManifestIndex,
        split: str,
        *,
        window: int,
        stride: int,
        channel_index: Sequence[int],
        use_v1: bool,
        use_v2: bool,
        v2_channels: int,
        use_id_valid: bool,
        label_map: Mapping[str, int] | None,
        unknown_index: int | None,
        multiclass: bool = True,
        capture_filter: Iterable[str] | None = None,
        v2_channel_index: Sequence[int] | None = None,
        use_frame_labels: bool = False,
        label_shuffle: bool = False,
        label_shuffle_seed: int = 0,
    ):
        self.tensors_dir = Path(tensors_dir)
        self.window = int(window)
        self.split = split
        self.channel_index = np.asarray(channel_index, dtype=np.int64)
        # ★ 按模型实际输入裁剪 I/O：rgbnet 不需要 V1，读它纯属浪费带宽
        self.use_v1 = bool(use_v1)
        self.use_v2 = bool(use_v2)
        self.v2_channels = int(v2_channels)
        # ★ 2026-10-28：字节图通道可按显式下标选择（RGBNet-8B 公平性消融：
        #   CAN 2.0 语料只有前 8 字节有效，取 [0, 4]（数据 0–7 + 掩码 0–7）即可，
        #   不必让 3/4 的输入恒为零）。None → 沿用"前 n 个通道"的旧行为。
        self.v2_channel_index = (None if v2_channel_index is None
                                 else np.asarray(list(v2_channel_index), dtype=np.int64))
        if self.v2_channel_index is not None:
            self.v2_channels = int(self.v2_channel_index.size)
        # 是否返回窗内逐帧标签（供 aux_frame 损失使用）
        self.use_frame_labels = bool(use_frame_labels)
        # ★ E5 负对照（v6.4）：训练期对**窗口级 (label, type) 对做全局置换**，实现见 index 构建之后。
        #   目的不是提升性能，而是检验"没有真实特征—标签关联时，pipeline 能否刷出高分"，
        #   即是否存在经由索引 / 采样器 / 窗序构建的泄漏。val/test 恒不置换。
        #   注意：全局置换会同时打断"窗口↔capture"的关联，因此它**不能**用于检验 capture（会话）身份泄漏；
        #   正文与 CITATION.cff 的结论范围须据此限定（2026-11-03 审查 B1 修正）。
        self.label_shuffle = bool(label_shuffle) and split == "train"
        self.label_shuffle_seed = int(label_shuffle_seed)
        self.use_id_valid = bool(use_id_valid)
        self.label_map = dict(label_map or {})
        self.unknown_index = unknown_index
        self.multiclass = bool(multiclass)

        allowed = set(capture_filter) if capture_filter is not None else None
        self.specs: list[CaptureSpec] = [
            c for c in index.captures_in(split)
            if c.ranges.get(split) and (allowed is None or c.name in allowed)
        ]
        if not self.specs:
            raise ValueError(f"[{index.dataset}/{split}] 没有可用的 capture")

        self.sources: list[FrameSource] = []
        self.type_of_capture: list[int] = []
        file_idx_parts: list[np.ndarray] = []
        start_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []

        for fid, spec in enumerate(self.specs):
            # ★ 按模型实际输入裁剪需要打开的数组（见 FrameSource 文档）
            need = tuple([k for k, on in (("v1", self.use_v1), ("v2", self.use_v2),
                                          ("id_valid", self.use_id_valid)) if on])
            src = FrameSource(self.tensors_dir, spec.name, split, need=need)
            self.sources.append(src)
            self.type_of_capture.append(self._resolve_type(spec))
            fi, st, lb = self._build_capture_index(src, spec, stride, fid)
            file_idx_parts.append(fi)
            start_parts.append(st)
            label_parts.append(lb)

        self.index = WindowIndex(
            file_idx=np.concatenate(file_idx_parts).astype(np.int32) if file_idx_parts else np.zeros(0, np.int32),
            start=np.concatenate(start_parts).astype(np.int64) if start_parts else np.zeros(0, np.int64),
            labels=np.concatenate(label_parts).astype(np.uint8) if label_parts else np.zeros(0, np.uint8),
            type_idx=np.zeros(0, np.int16),
        )
        # 多分类目标：攻击窗口取其 capture 的类型；正常窗口恒 0
        type_idx = np.zeros(len(self.index), dtype=np.int16)
        fid_arr = self.index.file_idx
        type_by_file = np.asarray(self.type_of_capture, dtype=np.int16)
        pos_mask = self.index.labels.astype(bool)
        type_idx[pos_mask] = type_by_file[fid_arr[pos_mask]]
        self.index.type_idx = type_idx

        # ★ E5 负对照（v6.4）：**窗口级**全局置换训练标签（(label, type) 成对置换）。
        #   为什么不用帧级置换：帧级置换会改变正例率（any_frame 把稀疏正例抹开），
        #   并制造"有攻击的 capture → 几乎全是正窗口"这种 capture 级伪信号，污染判读。
        #   窗口级置换保持正例率与类型分布严格不变，只打断"特征 ↔ 标签"的对应，
        #   因此若 test MCC 仍显著 >0，就说明存在经由索引/采样器/capture 身份的泄漏。
        if self.label_shuffle:
            rng = np.random.default_rng(self.label_shuffle_seed)
            perm = rng.permutation(len(self.index))
            self.index.labels = self.index.labels[perm]
            self.index.type_idx = self.index.type_idx[perm]

    # ------------------------------------------------------------------ #
    def _resolve_type(self, spec: CaptureSpec) -> int:
        """attack_type → 类别下标；未知类型落到 unknown（若未预留则报错）。

        单头消融（Ab-14）下 multiclass=False，类型头不参与训练，全部置 0。
        """
        if not self.multiclass:
            return 0
        if spec.usable_as_normal:
            return 0
        if not self.label_map:
            return 0
        if spec.attack_type in self.label_map:
            return int(self.label_map[spec.attack_type])
        if self.unknown_index is not None:
            return int(self.unknown_index)
        raise ValueError(
            f"[{self.split}] capture {spec.name} 的攻击类型 {spec.attack_type!r} "
            f"不在标签表中（{sorted(self.label_map)}），且未预留 unknown 类"
        )

    def _build_capture_index(self, src: FrameSource, spec: CaptureSpec,
                             stride: int, fid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """在每段 range 内构造起点，并用前缀和求窗口标签（向量化）。

        ★ 坐标系：`<split>/<capture>/*.npy` 只保存该 split 的帧，
          因此 manifest 里的**绝对帧号**必须映射为 split 内的**局部偏移**
          （第 i 段 range 的局部起点 = 前 i 段长度之和）。
          直接把绝对帧号喂给 npy 会越界或读到错误帧。
        """
        label = np.asarray(src.label)
        cum = np.concatenate([[0], np.cumsum(label, dtype=np.int64)])
        starts: list[np.ndarray] = []
        offset = 0
        for a, b in spec.ranges[self.split]:
            seg_len = b - a
            if seg_len < self.window:
                offset += seg_len
                continue
            rel = np.arange(0, seg_len - self.window + 1, stride, dtype=np.int64)
            starts.append(rel + offset)
            offset += seg_len
        if offset != len(src):
            raise ValueError(
                f"[{self.split}/{spec.name}] ranges 总帧数 {offset} 与张量行数 {len(src)} 不一致"
            )
        if not starts:
            empty = np.zeros(0, np.int64)
            return empty.astype(np.int32), empty, np.zeros(0, np.uint8)
        st = np.concatenate(starts)
        lb = (cum[st + self.window] - cum[st] > 0).astype(np.uint8)
        fi = np.full(st.shape, fid, dtype=np.int32)
        return fi, st, lb

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int) -> dict[str, Any]:
        fid = int(self.index.file_idx[k])
        s = int(self.index.start[k])
        e = s + self.window
        src = self.sources[fid]

        out: dict[str, Any] = {
            "y": int(self.index.labels[k]),
            "y_type": int(self.index.type_idx[k]),
            "cap": fid,
            "pos": s,
        }
        if self.use_v1:
            v1 = np.asarray(src.v1[s:e], dtype=np.float32)
            out["x1"] = v1[:, self.channel_index]
        if self.use_v2:
            v2 = np.asarray(src.v2[s:e], dtype=np.float32)      # [T, 8, 8]
            v2 = np.transpose(v2, (1, 0, 2))                    # [C2, T, 8]
            v2 = (v2[self.v2_channel_index] if self.v2_channel_index is not None
                  else v2[: self.v2_channels])
            out["x2"] = v2
        if self.use_id_valid:
            out["idv"] = np.asarray(src.id_valid[s:e], dtype=np.float32)
        if self.use_frame_labels:
            out["y_frame"] = np.asarray(src.label[s:e], dtype=np.float32)
        return out

    # ------------------------------------------------------------------ #
    def label_stats(self) -> dict[str, Any]:
        """窗口级类别统计（训练集用于 pos_weight / class_weight）。"""
        y = self.index.labels
        n = int(y.size)
        n_pos = int(y.sum())
        types: dict[int, int] = {}
        if self.label_map:
            for t in self.index.type_idx[y.astype(bool)]:
                types[int(t)] = types.get(int(t), 0) + 1
        return {
            "n_windows": n,
            "n_positive": n_pos,
            "n_negative": n - n_pos,
            "prevalence": (n_pos / n) if n else 0.0,
            "n_by_type": dict(sorted(types.items())),
            "n_captures": len(self.specs),
        }

    def frame_stats(self) -> dict[str, Any]:
        """帧级正负计数（只用于 aux_frame 损失的 pos_weight）。

        与 `label_stats()` 的窗口级统计不同：窗口 any_frame 会把正例率放大约 9×，
        帧级才是分类器真正面对的 1:100 量级不平衡。
        """
        n = 0
        n_pos = 0
        for src in self.sources:
            lab = np.asarray(src.label, dtype=np.int8)
            n += int(lab.size)
            n_pos += int(lab.sum())
        return {
            "n_frames": n,
            "n_positive": n_pos,
            "n_negative": n - n_pos,
            "positive_ratio": (n_pos / n) if n else 0.0,
        }

    def capture_stats(self) -> list[dict]:
        """逐 capture 的窗口数（采集偏斜分析用）。"""
        fid = self.index.file_idx
        rows = []
        for i, spec in enumerate(self.specs):
            m = fid == i
            n = int(m.sum())
            p = int(self.index.labels[m].sum())
            rows.append({
                "capture": spec.name, "attack_type": spec.attack_type,
                "n_windows": n, "n_positive": p,
                "positive_ratio": (p / n) if n else 0.0,
            })
        return rows


# --------------------------------------------------------------------------- #
# 数据加载器与采样
# --------------------------------------------------------------------------- #
def make_loader(
    dataset: WindowDataset,
    cfg: Mapping[str, Any],
    *,
    split: str,
    batch_size: int,
    shuffle: bool = False,
    sampler: torch.utils.data.Sampler | None = None,
) -> DataLoader:
    dl = cfg["data"]["dataloader"]
    nw = int(dl["num_workers"])
    kwargs: dict[str, Any] = dict(
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=nw,
        pin_memory=bool(dl["pin_memory"]),
        drop_last=bool(dl["drop_last_train"]) and split == "train",
    )
    if nw > 0:
        kwargs["persistent_workers"] = bool(dl["persistent_workers"])
        kwargs["prefetch_factor"] = int(dl["prefetch_factor"])
    return DataLoader(dataset, **kwargs)


def build_capture_sampler(
    dataset: WindowDataset,
    cfg: Mapping[str, Any],
    *,
    dataset_weight: float = 1.0,
    seed: int,
) -> WeightedRandomSampler | None:
    """采集级均衡采样：w ∝ (1 / n_windows^alpha) × dataset_weight。

    ★ 目的是抑制"单个长采集占训练窗口 32%"（实测 ROAD ambient 采集）造成的采集偏斜，
       而不是在类层面过采样 —— **不使用 SMOTE / ADASYN / GAN**（审核结论）。
    """
    sampling = cfg["data"]["sampling"]
    if sampling["train"] == "none":
        return None
    alpha = float(sampling["alpha"])
    fid = dataset.index.file_idx
    counts = np.bincount(fid, minlength=len(dataset.specs)).astype(np.float64)
    counts[counts == 0] = 1.0
    per_capture = (counts ** (-alpha)) * float(dataset_weight)
    w = per_capture[fid]
    w = w / w.sum()

    mode = str(sampling["num_samples"])
    if mode == "fixed":
        n = int(sampling["fixed_num_samples"])
        if n <= 0:
            raise ValueError("data.sampling.fixed_num_samples 必须 > 0")
    elif mode == "dataset_size":
        n = len(dataset)
    else:
        raise ValueError(f"未知 data.sampling.num_samples={mode}")

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.double),
        num_samples=n,
        replacement=True,
        generator=gen,
    )


def limit_dataset(dataset: WindowDataset, max_windows: int | None, *, seed: int) -> WindowDataset:
    """按上限截断（仅冒烟/调试用）；用等距抽样保持各 capture 与时间段的覆盖。"""
    if max_windows is None or max_windows <= 0 or len(dataset) <= max_windows:
        return dataset
    n = len(dataset)
    # 等距抽样（确定性，不用随机数）→ 覆盖全部 capture 与时间位置
    idx = np.linspace(0, n - 1, num=int(max_windows)).round().astype(np.int64)
    idx = np.unique(idx)
    sub = WindowDataset.__new__(WindowDataset)
    sub.__dict__.update(dataset.__dict__)
    sub.index = dataset.index.subset(idx)
    return sub


# --------------------------------------------------------------------------- #
# 组装入口
# --------------------------------------------------------------------------- #
def build_datasets(
    cfg: Mapping[str, Any],
    index: ManifestIndex,
    *,
    dataset_key: str,
    label_map: Mapping[str, int] | None,
    use_v1: bool,
    use_v2: bool,
    use_id_valid: bool,
    seed: int,
    use_frame_labels: bool = False,
) -> dict[str, WindowDataset]:
    """一次构建 train / val / test 三个 Dataset（同一个 channel_index）。"""
    tensors_dir = Path(cfg["_resolved"]["dataset_roots"][dataset_key]) / cfg["paths"]["tensor_subdir"]
    win = int(cfg["data"]["window"]["T"])
    strides = cfg["data"]["window"]["stride"]
    channels = list(cfg["data"]["v1"]["channels"])
    n_v2 = int(cfg["data"]["v2"].get("active_channels", cfg["data"]["v2"]["channels"]))
    v2_idx = cfg["data"]["v2"].get("channel_index")

    # 通道名 → 原始张量下标（原始顺序来自 env.json，保证与预处理写入一致）
    env_ch = list(index.env.get("constants", {}).get("V1_CHANNELS") or [])
    if not env_ch:
        raise ValueError(f"[{dataset_key}] env.json 缺少 V1_CHANNELS")
    try:
        ch_idx = [env_ch.index(c) for c in channels]
    except ValueError as e:
        raise ValueError(f"[{dataset_key}] 通道名与 env.json 不一致：{e}") from e

    unknown_index = None
    if label_map:
        ls = cfg["label_space"]
        if ls["unknown_class"]:
            unknown_index = label_map.get(str(ls["unknown_label"]))

    out: dict[str, WindowDataset] = {}
    for split in ("train", "val", "test"):
        out[split] = WindowDataset(
            tensors_dir, index, split,
            window=win, stride=int(strides[split]), channel_index=ch_idx,
            use_v1=use_v1, use_v2=use_v2, v2_channels=n_v2, use_id_valid=use_id_valid,
            label_map=label_map, unknown_index=unknown_index,
            multiclass=bool(cfg["label_space"]["multiclass"]),
            v2_channel_index=v2_idx,
            use_frame_labels=use_frame_labels,
            label_shuffle=bool(cfg["data"].get("label_shuffle", False)),
            label_shuffle_seed=int(seed),
        )
    return out


def merge_label_maps(maps: Sequence[Mapping[str, int]]) -> dict[str, int]:
    """多数据集联合训练时合并标签表（normal 恒 0，其余按字典序）。"""
    names: set[str] = set()
    unknown: set[str] = set()
    for m in maps:
        for k, v in m.items():
            if k == "normal":
                continue
            if k.endswith("unknown_attack"):
                unknown.add(k)
            else:
                names.add(k)
    ordered = ["normal"] + sorted(names) + sorted(unknown)
    return {n: i for i, n in enumerate(ordered)}


def compute_pos_weight(stats: Mapping[str, Any]) -> float:
    """窗口级 neg/pos；HCRL 因正类占多数而 <1（这是数据事实，不是 bug）。"""
    n_pos = int(stats["n_positive"])
    n_neg = int(stats["n_negative"])
    if n_pos == 0:
        raise ValueError("训练集没有正样本，无法计算 pos_weight")
    return n_neg / n_pos
