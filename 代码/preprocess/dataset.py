# -*- coding: utf-8 -*-
"""
preprocess.dataset —— 惰性窗口化 Dataset（训练只读 L2）
================================================================================
对应方案 §6（训练代码怎么读）。

为什么不预展开窗口：V1 float16 × 2.5 亿窗口 × 64 × 16 × 2 B ≈ **512 GB**，
一次性展开必然 OOM。因此 L2 只落**帧级**张量，窗口在 `__getitem__` 现场切，
磁盘占用降为约 1/32。

窗口标注规则（方案 §4 / §10 P6，必须在论文中显式声明）：
    窗内**任一**帧为攻击 → 该窗口标为攻击（保守口径）。

步长约定：
    train split  stride = 32（半重叠，扩样）
    val/test     stride = 64（= 窗长，零重叠 → 保证 McNemar 配对检验的样本独立性）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as C


class FrameMemmap:
    """一个 capture 的帧级 memmap 句柄（训练时零解析开销）。"""

    def __init__(self, tensors_dir: Path, capture: str, split: str):
        d = Path(tensors_dir) / split / capture
        if not d.exists():
            raise FileNotFoundError(f"缺少张量目录：{d}")
        self.v1 = np.load(d / "v1.npy", mmap_mode="r", allow_pickle=False)     # [N,16] f16
        self.v2 = np.load(d / "v2.npy", mmap_mode="r", allow_pickle=False)     # [N,8,8] u8
        self.label = np.load(d / "label.npy", mmap_mode="r", allow_pickle=False)
        self.aid = np.load(d / "aid.npy", mmap_mode="r", allow_pickle=False)
        self.id_valid = np.load(d / "id_valid.npy", mmap_mode="r", allow_pickle=False)
        self.ts_rel = np.load(d / "ts_rel.npy", mmap_mode="r", allow_pickle=False)
        self.capture = capture
        self.split = split

    def __len__(self) -> int:
        return int(self.v1.shape[0])


class MemmapWindows:
    """惰性窗口化数据集（numpy 版；可直接包进 torch.utils.data.DataLoader）。

    用法
    ----
    >>> ds = MemmapWindows("训练数据集/CTT_v1.5/02_tensors", [("set_01__...__DoS-3", "test")])
    >>> x1, x2, y = ds[0]     # x1: (64,16) f32, x2: (8,64,8) f32, y: int

    参数
    ----
    tensors_dir : 数据集 02_tensors 根目录
    items       : [(capture, split), ...] 参与本数据集的文件列表
    stride      : None → 按 split 自动取（train=32，val/test=64）
    window      : 窗口长度，默认 C.WIN_T
    include_id_valid : True 时额外返回 [64] 的 id_valid 通道（用于 Fuzzing 的非法 ID 特征）
    as_float32  : 把 V1/V2 转 float32（喂给深度学习框架前通常需要）
    """

    def __init__(
        self,
        tensors_dir: str | Path,
        items: list[tuple[str, str]],
        *,
        stride: int | None = None,
        window: int = C.WIN_T,
        include_id_valid: bool = False,
        as_float32: bool = True,
    ):
        self.tensors_dir = Path(tensors_dir)
        self.window = int(window)
        self.include_id_valid = include_id_valid
        self.as_float32 = as_float32
        self.frames: list[FrameMemmap] = []
        self.index: list[tuple[int, int, int]] = []   # (文件下标, 起始帧, 步长)
        self.win_labels: list[np.ndarray] = []

        for capture, split in items:
            fm = FrameMemmap(self.tensors_dir, capture, split)
            st = stride if stride is not None else (
                C.STRIDE_TRAIN if split == "train" else C.STRIDE_TEST
            )
            n = len(fm)
            if n < self.window:
                continue
            fid = len(self.frames)
            self.frames.append(fm)
            starts = np.arange(0, n - self.window + 1, st, dtype=np.int64)
            lab = self._window_labels(np.asarray(fm.label), starts)
            self.win_labels.append(lab)
            self.index.extend((fid, int(s), int(st)) for s in starts)

    # ------------------------------------------------------------------ #
    def _window_labels(self, label: np.ndarray, starts: np.ndarray) -> np.ndarray:
        """窗内任一帧为攻击 → 窗口标攻击（向量化：前缀和差分）。"""
        c = np.concatenate([[0], np.cumsum(label, dtype=np.int64)])
        return (c[starts + self.window] - c[starts] > 0).astype(np.uint8)

    def __len__(self) -> int:
        return len(self.index)

    def _ensure_flat_labels(self) -> np.ndarray:
        """把每个 capture 的窗口标签按全局下标顺序拼平（顺序与 self.index 一致）。"""
        if self._flat_labels is None:
            self._flat_labels = (
                np.concatenate(self.win_labels) if self.win_labels else np.zeros(0, np.uint8)
            )
        return self._flat_labels

    def __getitem__(self, k: int):
        fid, s, _ = self.index[k]
        fm = self.frames[fid]
        e = s + self.window

        x1 = np.asarray(fm.v1[s:e], dtype=np.float32 if self.as_float32 else None)  # [64,16]
        x2 = np.asarray(fm.v2[s:e], dtype=np.float32 if self.as_float32 else None)  # [64,8,8]
        x2 = np.transpose(x2, (1, 0, 2))                                            # → [8,64,8]
        y = int(self._ensure_flat_labels()[k])

        if self.include_id_valid:
            idv = np.asarray(fm.id_valid[s:e], dtype=np.float32)                    # [64]
            return x1, x2, idv, y
        return x1, x2, y

    #: 窗口标签的扁平缓存（首次访问时构建）
    _flat_labels: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        """返回窗口级统计（窗口总数、正类窗口数、正类占比）。"""
        if self._flat_labels is None:
            self._flat_labels = np.concatenate(self.win_labels) if self.win_labels else np.zeros(0, np.uint8)
        n = int(self._flat_labels.size)
        p = int(self._flat_labels.sum())
        return {"n_windows": n, "n_positive_windows": p,
                "positive_ratio": (p / n) if n else 0.0, "window": self.window}


# --------------------------------------------------------------------------- #
# 可选：把窗口物化落盘（仅在确需定长窗口文件时使用，注意 512 GB 预算）             #
# --------------------------------------------------------------------------- #
def materialize_windows(
    tensors_dir: Path,
    out_dir: Path,
    capture: str,
    split: str,
    *,
    stride: int | None = None,
    batch: int = 4096,
) -> dict:
    """把某个 capture 的某个 split 展开为窗口级 .npy（可选步骤）。

    ★ 使用前请先估算磁盘：V1 float16 [Nw,64,16] + V2 uint8 [Nw,8,64,8]。
      对 CTT 全量而言这是 TB 级，**仅建议对单 capture 或抽样子集使用**。
    """
    ds = MemmapWindows(tensors_dir, [(capture, split)], stride=stride, as_float32=False)
    nw = len(ds)
    out = Path(out_dir) / split / capture
    out.mkdir(parents=True, exist_ok=True)
    if nw == 0:
        return {"capture": capture, "split": split, "n_windows": 0}

    X1 = np.lib.format.open_memmap(out / "v1.npy", mode="w+",
                                   dtype=np.float16, shape=(nw, C.WIN_T, len(C.V1_CHANNELS)))
    X2 = np.lib.format.open_memmap(out / "v2.npy", mode="w+",
                                   dtype=np.uint8, shape=(nw, C.IMG_CHANNELS, C.WIN_T, C.IMG_BYTE_PER_CH))
    Y = np.lib.format.open_memmap(out / "labels.npy", mode="w+", dtype=np.uint8, shape=(nw,))
    for i in range(nw):
        x1, x2, y = ds[i]
        X1[i] = x1
        X2[i] = x2
        Y[i] = y
    X1.flush(); X2.flush(); Y.flush()
    return {"capture": capture, "split": split, "n_windows": nw, "out": str(out)}
