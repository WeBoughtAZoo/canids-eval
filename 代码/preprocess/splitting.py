# -*- coding: utf-8 -*-
"""
preprocess.splitting —— 切分策略（三种模式，统一以"区间列表"表达）
================================================================================
对应方案 §5 G4 / §6.2（2026-10-17 修订）。

三种切分模式
------------
1) `make_splits`       传统：capture 内按行号**连续** 60/20/20（单区间）
2) `make_role_splits`  语义：按 capture 的文件名角色整文件归入某个 split（CTT v1.5 专用，单区间）
3) `make_block_splits` ★ 块化交错：单条时间线切 K 个**连续块**，按固定模模式分配
                       （多区间），块边界可对齐静默区

统一返回类型
------------
    dict[str, list[tuple[int, int]]]     # split → [(start, end), ...]（左闭右开）

为什么块化交错（2026-10-17 修订的理由，须写进论文）
--------------------------------------------------
capture 内连续切分下，攻击注入在时间轴上高度局部化：HCRL 测试段（后 20%）
仅 331 个正窗口，且 Fuzzing 与 Malfunction 的测试段正样本为 **0** ——
这使"按攻击类型分层报告"这条红线在 CAN FD 侧不可执行，并直接削弱 O1 的证据链。

块化交错**不引入随机数、不引入跨切分单元泄漏、不引入窗口重叠**：
  · 每个块是一段连续时间，块内帧序保持不变（保序）；
  · 窗口严格在块内构造，绝不跨块（因此窗口零重叠，McNemar 独立性假设 A7 仍成立）；
  · 块 → split 的映射由固定取模规则给出，完全确定性（无随机种子参与）。
它改变的只有一件事：**测试块在时间上不再是训练块之后**，
即放弃"严格时间外推"语义，换取评估统计有效性。该取舍必须写进 Limitations。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from . import config as C

#: 区间类型：左闭右开 (start, end)
Range = tuple[int, int]


# --------------------------------------------------------------------------- #
# 通用工具                                                                      #
# --------------------------------------------------------------------------- #
def slice_concat(arr: np.ndarray, ranges: list[Range]) -> np.ndarray:
    """按多区间切片并拼接（单区间时避免多余拷贝）。"""
    if not ranges:
        return arr[0:0]
    if len(ranges) == 1:
        a, b = ranges[0]
        return arr[a:b]
    return np.concatenate([arr[a:b] for a, b in ranges])


def ranges_total(ranges: Iterable[Range]) -> int:
    return int(sum(b - a for a, b in ranges))


def all_blocks(splits: dict[str, list[Range]]) -> list[Range]:
    """把所有 split 的区间合并成**按时间排序**的块列表（用于逐块计算 block-local 特征）。

    ★ 用途：`features.encode_capture(..., blocks=all_blocks(splits))`。
      给出块列表后，`payload_delta` 在每块内独立计算，块内每个 ID 的首帧差值为 0，
      从而**彻底消除跨块/跨 split 引用**（P1 修复）。
    """
    rs = [r for v in splits.values() for r in v]
    rs.sort(key=lambda x: (x[0], x[1]))
    return rs


def block_start_mask(n_frames: int, splits: dict[str, list[Range]]) -> np.ndarray:
    """标记"每个块的首帧"。用于：

    · 把块首帧的 Δt / payload_delta 置 0 —— 否则它会跨越一个被分给其他 split 的块，
      既污染 train 统计尺度（跨块间隔是"块长"量级，远大于正常帧间隔），
      又让三段特征分布不一致；
    · 拟合 train 统计量时排除这些帧。
    """
    mask = np.zeros(n_frames, dtype=bool)
    for ranges in splits.values():
        for a, _b in ranges:
            if 0 <= a < n_frames:
                mask[a] = True
    return mask


def assert_partition(splits: dict[str, list[Range]], n_frames: int) -> None:
    """校验切分是 [0, n) 的一个**合法分区**：无重叠、无空洞、每段内升序。

    ★ 块化交错下三段在时间轴上交错，因此不能再断言"train 早于 val 早于 test"，
      改为断言"分区合法性" —— 这是本方案在放弃严格时间外推后仍然守住的纪律。
    """
    if not n_frames:
        return
    covered = np.zeros(n_frames, dtype=np.int32)
    for name, ranges in splits.items():
        prev_end = -1
        for a, b in ranges:
            if not (0 <= a < b <= n_frames):
                raise AssertionError(f"[{name}] 非法区间 ({a}, {b})，n_frames={n_frames}")
            if a < prev_end:
                raise AssertionError(f"[{name}] 区间重叠或未升序：({a},{b}) 落在 {prev_end} 之前")
            prev_end = b
            covered[a:b] += 1
    if covered.min() != 1:
        n_hole = int(np.count_nonzero(covered == 0))
        n_dup = int(np.count_nonzero(covered > 1))
        raise AssertionError(f"切分未构成完整分区：空洞 {n_hole} 帧、重叠 {n_dup} 帧")


def describe_splits(splits: dict[str, list[Range]], n_frames: int) -> dict:
    """把切分整理成可写入 manifest 的字典（区间列表 + 帧数 + 占比）。"""
    out = {"n_frames": int(n_frames), "ranges": {}, "summary": {}}
    for s in C.SPLIT_NAMES:
        rng = splits.get(s, [])
        n = ranges_total(rng)
        out["ranges"][s] = [[int(a), int(b)] for a, b in rng]
        out["summary"][s] = {
            "n": int(n),
            "n_segments": len(rng),
            "ratio": round(n / n_frames, 6) if n_frames else 0.0,
        }
    return out


# --------------------------------------------------------------------------- #
# 模式 1：capture 内时间序连续 60/20/20                                        #
# --------------------------------------------------------------------------- #
def make_splits(
    n_frames: int,
    *,
    ratios: tuple[float, float, float] = C.SPLIT_RATIOS,
    min_train: int = 1,
) -> dict[str, list[Range]]:
    """传统模式：单区间连续切分。返回 {split: [(start, end)]}。"""
    if n_frames <= 0:
        return {s: [] for s in C.SPLIT_NAMES}

    n_train = int(ratios[0] * n_frames)
    n_val = int(ratios[1] * n_frames)

    if n_frames < (min_train + 2):          # 极小 capture：退化处理，仍保持三段连续
        n_train = max(1, n_frames - 2)
        n_val = 1

    n_train = max(1, min(n_train, n_frames - 2))
    n_val = max(1, min(n_val, n_frames - n_train - 1))

    return {
        "train": [(0, n_train)],
        "val": [(n_train, n_train + n_val)],
        "test": [(n_train + n_val, n_frames)],
    }


# --------------------------------------------------------------------------- #
# 模式 2：按 capture 角色整文件归入（CTT v1.5 语义划分）                          #
# --------------------------------------------------------------------------- #
def make_role_splits(n_frames: int, role: str) -> dict[str, list[Range]]:
    """整文件归入某个 split；其余 split 为空列表（不落盘）。

    ★ CTT 的 test_01–04 是"已知/未知车辆 × 已知/未知攻击"四象限，
      若按 capture 内 60/20/20 切，会把测试子集的 60% 送进训练集，
      unknown-vehicle 维度直接失效。故按文件名语义整文件归入。
    """
    if n_frames <= 0:
        return {s: [] for s in C.SPLIT_NAMES}
    n = int(n_frames)
    if role == "train":
        return {"train": [(0, n)], "val": [], "test": []}
    if role == "val":
        return {"train": [], "val": [(0, n)], "test": []}
    return {"train": [], "val": [], "test": [(0, n)]}   # role == "test"（或未知，保守归 test）


# --------------------------------------------------------------------------- #
# 模式 3：★ 块化交错（块边界对齐静默区）                                        #
# --------------------------------------------------------------------------- #
def effective_block_k(n_frames: int, k: int = C.BLOCK_K) -> int:
    """在保证每块 ≥ MIN_BLOCK_FRAMES 与 k 为 PERIOD 整数倍的前提下，取可用的最大 k。

    返回 0 表示该 capture 太短，无法使用块化交错（调用方应退回 make_splits）。
    """
    period = C.BLOCK_PATTERN_PERIOD
    kk = (k // period) * period
    while kk >= period:
        if n_frames // kk >= C.MIN_BLOCK_FRAMES:
            return kk
        kk -= period
    return 0


def _align_bounds_to_segment_edges(
    label: np.ndarray,
    n_frames: int,
    k: int,
    *,
    w_scale: float = 1.0,
) -> tuple[list[int], list[bool], int, int]:
    """把 k−1 个候选等分切点逐个微调到**最近的标签跳变点**，返回 (边界, 对齐标记, 未对齐数, 跳变点数)。

    对齐规则（确定性、无随机）
    -------------------------
    记跳变点集合 T = { t ∈ [1, n) : label[t-1] != label[t] }。
    对第 i 个候选切点 p：
      1. 搜索窗口 w = max(MIN_BLOCK_FRAMES, 块长 × SEGMENT_SEARCH_FRAC × w_scale)；
         并夹紧到"前后都还容得下各自剩余块"的范围，避免块大小失衡。
      2. 在窗口内取离 p 最近的跳变点 t* 作为实际边界。
      3. 若窗口内无跳变点，按 ALIGN_WIDEN_FACTOR 逐级放宽窗口重试（最多 ALIGN_WIDEN_STEPS 级）。
      4. 若放宽后仍无跳变点（整条时间线都是同一种标签），保持候选切点，记 aligned=False。

    为什么用"跳变点"而不是"静默区中心"
    -----------------------------------
    《五套方案》要求"HCRL CAN-FD 按'攻击段—静默段'整段切分，确保测试集中包含完整的攻击段
    而非片段残留"。切在跳变点上 ⇒ 任何攻击段都完整落在边界的一侧，**机制上不可能被劈开**；
    而"静默区中心"只能保证边界本身不在攻击中（当攻击段比搜索窗口更宽时仍会被切）。
    """
    n_edge = int(np.count_nonzero(label[1:] != label[:-1])) if label.size > 1 else 0
    edges = np.linspace(0, n_frames, k + 1).astype(np.int64)
    trans = (np.flatnonzero(label[1:] != label[:-1]) + 1) if label.size > 1 else np.zeros(0, np.int64)

    bounds = [0]
    aligned_flags: list[bool] = []
    n_unaligned = 0
    block_len = n_frames / float(k)
    w0 = max(1, int(max(C.MIN_BLOCK_FRAMES, block_len * C.SEGMENT_SEARCH_FRAC) * w_scale))

    for i in range(1, k):
        p = int(edges[i])
        chosen: int | None = None
        w = w0
        for _step in range(max(1, C.ALIGN_WIDEN_STEPS)):
            lo = max(bounds[-1] + C.MIN_BLOCK_FRAMES, p - w)
            hi = min(n_frames - (k - i) * C.MIN_BLOCK_FRAMES, p + w)
            if hi <= lo:
                break
            cand = trans[(trans >= lo) & (trans < hi)]
            if cand.size:
                chosen = int(cand[np.argmin(np.abs(cand - p))])
                break
            w = int(w * C.ALIGN_WIDEN_FACTOR)

        if chosen is None:
            lo = max(bounds[-1] + C.MIN_BLOCK_FRAMES, p - w0)
            hi = min(n_frames - (k - i) * C.MIN_BLOCK_FRAMES, p + w0)
            chosen = int(np.clip(p, lo, hi)) if hi > lo else max(p, bounds[-1] + 1)
            aligned_flags.append(False)
            n_unaligned += 1
        else:
            aligned_flags.append(True)
        bounds.append(chosen)

    bounds.append(n_frames)
    for j in range(1, len(bounds)):          # 强制严格升序
        if bounds[j] <= bounds[j - 1]:
            bounds[j] = bounds[j - 1] + 1
    return bounds, aligned_flags, n_unaligned, n_edge


def _ratio_deviation(bounds: list[int], n_frames: int, k: int) -> float:
    """计算当前边界下 train 占比相对 6/(6+2+2) 的偏差（用于比例守卫）。"""
    target = 1.0 - (2.0 / C.BLOCK_PATTERN_PERIOD)   # = 0.6
    n_train = 0
    for i in range(k):
        if i % C.BLOCK_PATTERN_PERIOD in (C.BLOCK_VAL_MOD, C.BLOCK_TEST_MOD):
            continue
        n_train += bounds[i + 1] - bounds[i]
    return abs(n_train / float(n_frames) - target)


def make_block_splits(
    n_frames: int,
    label: np.ndarray | None = None,
    *,
    k: int | None = None,
    align: bool | None = None,
) -> tuple[dict[str, list[Range]], dict]:
    """★ 块化交错切分：返回 (切分字典, 元信息)。

    分配规则（确定性）
    ------------------
        block_index i：i % 5 == 2 → val；i % 5 == 4 → test；其余 → train
    由于 k 必为 5 的整数倍，比例精确为 6 : 2 : 2。

    参数
    ----
    n_frames : 帧数
    label    : 帧级标签（用于块边界对齐静默区；None 时退化为等分边界）
    k        : 块数，None → C.BLOCK_K 经 effective_block_k 自适应
    align    : 是否把块边界对齐攻击段边界，None → C.ALIGN_TO_SEGMENT_EDGE
    """
    if n_frames <= 0:
        return {s: [] for s in C.SPLIT_NAMES}, {"k": 0, "fallback": True}

    align = C.ALIGN_TO_SEGMENT_EDGE if align is None else align
    kk = effective_block_k(n_frames) if k is None else int(k)
    if kk <= 0 or kk % C.BLOCK_PATTERN_PERIOD != 0:
        # 太短 → 退化为连续切分（并如实记录，便于审计）
        return make_splits(n_frames), {"k": 0, "fallback": True}

    if align and label is not None and label.size == n_frames:
        # ★ 比例守卫：对齐段边界不得破坏 6:2:2。若偏差超限，逐步收紧搜索窗口重试；
        #   仍不满足则退回等分边界（并如实记录 repair 过程）。
        bounds, flags, n_unaligned, n_edge = _align_bounds_to_segment_edges(label, n_frames, kk)
        repair = 0
        scale = 1.0
        while (_ratio_deviation(bounds, n_frames, kk) > C.BLOCK_RATIO_TOL
               and repair < C.BLOCK_RATIO_REPAIR_STEPS):
            repair += 1
            scale *= 0.5
            bounds, flags, n_unaligned, n_edge = _align_bounds_to_segment_edges(
                label, n_frames, kk, w_scale=scale)
        ratio_ok = _ratio_deviation(bounds, n_frames, kk) <= C.BLOCK_RATIO_TOL
        if not ratio_ok:
            bounds = [int(x) for x in np.linspace(0, n_frames, kk + 1).astype(np.int64)]
            flags = [False] * (kk - 1)
            n_unaligned = kk - 1
            repair = -1                       # -1 表示"已退回等分边界"
        align_scale = scale
    else:
        bounds = [int(x) for x in np.linspace(0, n_frames, kk + 1).astype(np.int64)]
        flags = [False] * (kk - 1)
        n_unaligned = kk - 1
        n_edge = -1
        repair = 0
        align_scale = 0.0
        ratio_ok = True

    splits: dict[str, list[Range]] = {s: [] for s in C.SPLIT_NAMES}
    block_split: list[str] = []
    for i in range(kk):
        a, b = int(bounds[i]), int(bounds[i + 1])
        if b <= a:
            block_split.append("empty")
            continue
        if i % C.BLOCK_PATTERN_PERIOD == C.BLOCK_VAL_MOD:
            sp = "val"
        elif i % C.BLOCK_PATTERN_PERIOD == C.BLOCK_TEST_MOD:
            sp = "test"
        else:
            sp = "train"
        splits[sp].append((a, b))
        block_split.append(sp)

    # 校验：切在跳变点上 ⇒ 不存在被劈开的攻击段
    if align and label is not None and label.size == n_frames:
        inner = [b for b in bounds[1:-1] if 0 < b < n_frames]
        n_split_segments = int(sum(1 for b in inner if label[b - 1] == 1 and label[b] == 1))
    else:
        n_split_segments = -1

    info = {
        "k": int(kk),
        "block_bounds": [int(x) for x in bounds],
        "block_split": block_split,
        "aligned_flags": [bool(x) for x in flags],
        "n_aligned": int(sum(flags)),
        "n_unaligned": int(n_unaligned),
        "n_label_transitions": int(n_edge),
        "n_attack_segments_cut": int(n_split_segments),
        "align_w_scale": float(align_scale),
        "align_repair_steps": int(repair),
        "ratio_guard_ok": bool(ratio_ok),
        "fallback": False,
        "mean_block_frames": float(n_frames / kk),
        "pattern": f"i%{C.BLOCK_PATTERN_PERIOD}=={C.BLOCK_VAL_MOD}->val, "
                   f"i%{C.BLOCK_PATTERN_PERIOD}=={C.BLOCK_TEST_MOD}->test, else train",
    }
    return splits, info
