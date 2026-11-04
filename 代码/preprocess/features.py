# -*- coding: utf-8 -*-
"""
preprocess.features —— 编码层：变长载荷 → 定长张量 + 归一化
================================================================================
对应方案 §4（编码层）与 §5 G5/G6。

产出两套张量（均为**帧级**，窗口在 dataset.py 中惰性切出，避免 512 GB 峰值）：
  V1  帧向量  float16 [N, 16]    供 ① FD-TCN 与 ⑤ KD-Logic
  V2  字节图  uint8   [N, 8, 8]  供 ② FD-RGBNet
      └─ 通道 0–3 = padded[0:8], [8:16], [16:24], [24:32]（4 个数据通道）
      └─ 通道 4–7 = mask[0:8],   [8:16], [16:24], [24:32]（4 个逐字节填充掩码通道）
         去掉通道 4–7 即为 Ab-1 的 "w/o 掩码" 版本。

归一化红线（方案 §5 G5）：
  `median_period(aid)` / `train_freq[aid]` / z-score 的 μ、σ **只用 train split 估计**，
  再原样应用到 val/test。因此本模块分两步：`fit_train_stats()` → `encode_capture()`。

ID 溢出（方案 §4 警示框）：
  `id_norm = min(aid, 2047) / 2047`（钳位），同时把"是否超出 11 位"写入 aux.id_valid，
  不把该强信号抹掉——这是 HCRL Fuzzing 的关键判别特征之一。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C
from .cleaning import apply_dt_clip, compute_dt, fit_dt_clip
from .utils import hexstrings_to_padded

_LOG2_PAD = float(np.log2(C.PAD_BYTES))  # log2(32) = 5 —— 32 字节香农熵的理论上界


# --------------------------------------------------------------------------- #
# 训练集统计量                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class TrainStats:
    """**仅由 train split 估计**的统计量集合（跨 split 复用的唯一归一化依据）。"""
    aid_size: int                 # 稠密数组长度（max_aid + 1）
    freq: np.ndarray              # [aid_size] 训练集内各 ID 出现次数
    med_period: np.ndarray        # [aid_size] 各 ID 的 Δt 中位数（周期）
    mean_period: np.ndarray       # [aid_size] 各 ID 的 Δt 均值
    std_period: np.ndarray        # [aid_size] 各 ID 的 Δt 标准差
    dt_log_mu: float              # z-score(log1p(Δt)) 的 μ
    dt_log_sigma: float           # z-score(log1p(Δt)) 的 σ
    dt_clip_hi: float             # Δt 鲁棒上裁剪上界（train P99.9）
    global_med_period: float      # 训练集全局中位周期（未见 ID 的兜底）
    global_std_period: float
    global_mean_period: float
    period_floor: float           # 周期分母下限（防止 med≈0 时 dt_dev 爆炸 → float16 inf）

    def to_json(self) -> dict:
        """标量部分落盘（数组太大不落盘，需要时可由 splits.json + L1 复算）。"""
        return {
            "aid_size": int(self.aid_size),
            "dt_log_mu": float(self.dt_log_mu),
            "dt_log_sigma": float(self.dt_log_sigma),
            "dt_clip_hi": float(self.dt_clip_hi),
            "global_med_period": float(self.global_med_period),
            "global_mean_period": float(self.global_mean_period),
            "global_std_period": float(self.global_std_period),
            "period_floor": float(self.period_floor),
            "n_unique_aid_in_train": int(np.count_nonzero(self.freq)),
        }


def _group_reduce(aid: np.ndarray, dt: np.ndarray, aid_size: int):
    """按 aid 分组求 Δt 的中位数 / 均值 / 标准差（向量化排序 + 分组 numpy 归约）。

    实现要点：对 aid 做**稳定排序**，同一 ID 的帧在排序后连续，
    分组边界由 np.unique(..., return_index=True) 给出；组数远小于帧数，
    因此"按组调用 numpy 归约"的总开销可忽略（即使 HCRL Fuzzing 有 65,535 个 ID）。
    """
    order = np.argsort(aid, kind="stable")
    a_sorted = aid[order]
    d_sorted = dt[order]
    uniq, starts = np.unique(a_sorted, return_index=True)
    ends = np.append(starts[1:], a_sorted.size)

    med = np.zeros(aid_size, dtype=np.float64)
    mean = np.zeros(aid_size, dtype=np.float64)
    std = np.zeros(aid_size, dtype=np.float64)
    freq = np.zeros(aid_size, dtype=np.int64)

    for u, s, e in zip(uniq, starts, ends):
        seg = d_sorted[s:e]
        med[u] = np.median(seg)
        mean[u] = seg.mean()
        std[u] = seg.std()
        freq[u] = seg.size
    return freq, med, mean, std


def fit_train_stats(aid_train: np.ndarray, dt_train: np.ndarray, aid_size: int) -> TrainStats:
    """用 train split 估计全部统计量（方案 §5 G5）。"""
    freq, med, mean, std = _group_reduce(aid_train, dt_train, aid_size)

    dt_log = np.log1p(dt_train)
    mu = float(dt_log.mean()) if dt_log.size else 0.0
    sigma = float(dt_log.std()) if dt_log.size else 0.0
    if not np.isfinite(sigma) or sigma < 1e-6:
        sigma = 1.0                      # σ 退化（全部 Δt 相同）→ 退化为恒等尺度

    # 典型周期尺度：优先取"出现 ≥2 次的 ID"的中位周期（单次出现 ID 的 med_period 不可靠）；
    # 若训练集里没有任何可靠 ID（如 ROAD fuzzing：随机 ID 几乎都只出现一次），
    # 退回用训练集正 Δt 的中位数 —— 这是物理上可解释的"总线报文典型间隔"。
    pos_dt = dt_train[dt_train > 0]
    reliable = freq >= 2
    if reliable.any():
        typical = float(np.median(med[reliable]))
        g_med = typical
        g_mean = float(mean[reliable].mean())
        g_std = float(std[reliable].mean())
    elif pos_dt.size:
        typical = float(np.median(pos_dt))
        g_med = typical
        g_mean = float(pos_dt.mean())
        g_std = float(pos_dt.std())
    else:
        typical = 0.0
        g_med = g_mean = g_std = 0.0
    period_floor = max(C.PERIOD_FLOOR_ABS, C.PERIOD_FLOOR_REL * typical) if typical > 0 \
        else C.PERIOD_FLOOR_ABS

    return TrainStats(
        aid_size=int(aid_size),
        freq=freq,
        med_period=med,
        mean_period=mean,
        std_period=std,
        dt_log_mu=mu,
        dt_log_sigma=sigma,
        dt_clip_hi=fit_dt_clip(dt_train),
        global_med_period=g_med,
        global_mean_period=g_mean,
        global_std_period=g_std,
        period_floor=period_floor,
    )


# --------------------------------------------------------------------------- #
# 载荷 → padded / mask                                                          #
# --------------------------------------------------------------------------- #
def build_padded_and_mask(data_hex: list[str], *, counters: dict[str, int]):
    """变长 hex 载荷 → (padded [N,32] uint8, mask [N,32] uint8)。

    填充规则（方案 §4）：数据右侧补 0x00 到 32 字节；
    掩码 mask[i] = 1 iff i < dlc，即"该字节是真实数据"。
    ★ mask 是 Ab-1 消融的对象：去掉掩码即让模型无法区分"真实 0x00"与"填充 0x00"。
    """
    padded, dlc = hexstrings_to_padded(
        data_hex, max_bytes=C.PAD_BYTES, odd_left_pad=True, stats=counters
    )
    idx = np.arange(C.PAD_BYTES, dtype=np.int64)[None, :]
    mask = (idx < dlc[:, None]).astype(np.uint8)
    return padded, mask, dlc


# --------------------------------------------------------------------------- #
# 逐帧通道计算                                                                  #
# --------------------------------------------------------------------------- #
def _row_entropy(padded: np.ndarray, chunk: int = 100_000) -> np.ndarray:
    """逐行香农熵（归一化到 [0,1]）。

    ★ 关键实现决策：**熵按"实际数据字节"计算，但不逐行做变长切片**——
      这里对 padded 的 32 字节整体统计，填充的 0x00 会进入直方图。
      这是刻意的：填充伪影本身就是被研究对象的信号（Ab-1），
      若在此处按 mask 排除填充，掩码消融实验将失去对照。
      论文中须写明"熵在 32 字节 padded 上计算"。
    """
    n, w = padded.shape
    out = np.zeros(n, dtype=np.float32)
    ar = np.arange(chunk, dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        m = e - s
        blk = padded[s:e].astype(np.int64)
        idx = blk + (ar[:m, None] * 256)
        cnt = np.bincount(idx.ravel(), minlength=m * 256).reshape(m, 256)
        p = cnt / float(w)
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = np.where(p > 0, np.log2(np.where(p > 0, p, 1.0)), 0.0)
        out[s:e] = -(p * lp).sum(axis=1)
    return (out / _LOG2_PAD).astype(np.float32)


def _prev_same_aid(aid: np.ndarray) -> np.ndarray:
    """返回"同一 ID 的前一帧"在原数组中的下标（无前驱时指向自身 → Δ载荷为 0）。

    向量化做法：稳定排序后同 ID 组内相邻即为时间序前驱，再散列回原下标。
    """
    n = aid.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(aid, kind="stable")
    a_sorted = aid[order]
    same = np.zeros(n, dtype=bool)
    same[1:] = a_sorted[1:] == a_sorted[:-1]

    prev_orig = np.empty(n, dtype=np.int64)
    prev_orig[0] = order[0]
    prev_orig[1:] = np.where(same[1:], order[:-1], order[1:])

    out = np.empty(n, dtype=np.int64)
    out[order] = prev_orig
    return out


def _payload_delta_block_local(
    p32: np.ndarray,
    aid: np.ndarray,
    blocks: list[tuple[int, int]],
) -> np.ndarray:
    """★ 逐块计算的载荷差分 mean(|padded − 同 ID 在**同一块内**的前一帧|) / 255。

    为什么必须逐块（P1 修复，2026-10-17）
    ------------------------------------
    块化交错切分下，若在**整条 capture** 上求"同 ID 前一帧"，那么"某块内首次出现的 ID"
    的前驱会落在**上一块**（可能属于另一个 split）。实测未隔离面：
    HCRL 3.616%（其中 `CANFD_Fuzzing` 高达 10.687%）、ROAD 0.109%。
    这虽不含标签泄漏（只依赖载荷字节），但会使 train / val / test 的特征分布不一致，
    且与本方案"每个切分单元独立计算"的声明不符。

    逐块计算后：每块内每个 ID 的首帧，其 prev 指向自身 → 差值为 0，
    等价于"块内重新开始差分"，**不存在任何跨块引用**。
    """
    n = p32.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for a, b in blocks:
        if b <= a:
            continue
        seg = p32[a:b]
        prev = _prev_same_aid(aid[a:b])
        out[a:b] = (np.abs(seg - seg[prev]).mean(axis=1) / 255.0).astype(np.float32)
    return out


def encode_capture(
    cap_arrays: dict,
    stats: TrainStats,
    *,
    proto_flag_value: int,
    counters: dict[str, int],
    reset_mask: np.ndarray | None = None,
    blocks: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """把一个 capture 的帧级中间表示编码为 V1 / V2 张量。

    参数
    ----
    cap_arrays : {'ts_rel','aid','data_hex','label'}（ts_rel 已完成相对化与单调化）
    stats      : 由 train split 估计的 TrainStats
    proto_flag_value : chan[6] 取值（主实验 0；跨协议实验按协议族赋值）
    counters   : 异常计数累加器
    reset_mask : ★ 块化交错切分下"每个切分单元（块）的首帧"布尔掩码。
                 这些帧的 Δt 会跨越一个被分给其他 split 的块（量级为"块长"，
                 远大于正常帧间隔），必须归零，否则：
                   · 污染 train 的周期统计尺度；
                   · 使 train / val / test 的特征分布不一致。
                 置零后等价于"每个切分单元独立计时"，三段完全一致。
    blocks     : ★ 切分单元（块）的区间列表 [(a,b), …]。给出时 `payload_delta` **逐块计算**
                 （块内每个 ID 的首帧差值为 0），从而**彻底消除跨块/跨 split 引用**（P1 修复）。
                 None → 退化为整条 capture 全局计算（仅用于兼容旧口径或单段切分）。

    返回
    ----
    (v1 [N,16] float16, v2 [N,8,8] uint8, aux {aid, id_valid, ts_rel, dlc, label})
    """
    ts_rel = np.asarray(cap_arrays["ts_rel"], dtype=np.float64)
    aid = np.asarray(cap_arrays["aid"], dtype=np.uint32)
    data_hex = cap_arrays["data_hex"]
    n = ts_rel.size
    if n == 0:
        return (np.zeros((0, len(C.V1_CHANNELS)), np.float16),
                np.zeros((0, C.IMG_CHANNELS, C.IMG_BYTE_PER_CH), np.uint8),
                {"aid": aid, "id_valid": np.zeros(0, np.uint8), "ts_rel": ts_rel,
                 "dlc": np.zeros(0, np.uint8), "label": np.zeros(0, np.uint8)})

    # ---------- a) 载荷 → padded / mask ----------
    padded, mask, dlc = build_padded_and_mask(data_hex, counters=counters)
    p32 = padded.astype(np.float32)

    # ---------- b) 时间通道 ----------
    dt = compute_dt(ts_rel)                                 # 首帧 Δt = 0
    if reset_mask is not None and reset_mask.any():
        # ★ 块首帧：Δt 跨越其他 split 的块 → 归零（"每个切分单元独立计时"）
        dt = dt.copy()
        dt[reset_mask] = 0.0
        counters["block_start_dt_reset"] = counters.get("block_start_dt_reset", 0) + int(reset_mask.sum())
    dt = apply_dt_clip(dt, stats.dt_clip_hi, counters=counters)
    dt_log = np.log1p(dt)
    dt_norm = np.clip((dt_log - stats.dt_log_mu) / (stats.dt_log_sigma + C.EPS),
                      -C.CLIP_DT_NORM, C.CLIP_DT_NORM)

    # ---------- c) ID 通道（含溢出保护） ----------
    aid_clamped = np.minimum(aid, C.AID_NORM_DENOM).astype(np.float32)
    id_norm = aid_clamped / float(C.AID_NORM_DENOM)
    id_valid = (aid <= C.AID_STD_MAX).astype(np.uint8)       # ★ aux 通道：11 位合法性

    idx_ok = aid < stats.aid_size
    id_freq = np.zeros(n, dtype=np.float32)
    id_freq[idx_ok] = np.log1p(stats.freq[aid[idx_ok]].astype(np.float64))
    med_p = np.full(n, stats.global_med_period, dtype=np.float64)
    mean_p = np.full(n, stats.global_mean_period, dtype=np.float64)
    std_p = np.full(n, stats.global_std_period, dtype=np.float64)
    med_p[idx_ok] = stats.med_period[aid[idx_ok]]
    mean_p[idx_ok] = stats.mean_period[aid[idx_ok]]
    std_p[idx_ok] = stats.std_period[aid[idx_ok]]

    # 周期偏差 / 周期变异系数：分母取下限，防止 med_period≈0 时溢出 float16
    floor = float(stats.period_floor)
    dt_dev = np.clip((dt - med_p) / np.maximum(med_p, floor),
                     -C.CLIP_DT_DEV, C.CLIP_DT_DEV).astype(np.float32)
    period_stability = np.clip(std_p / np.maximum(mean_p, floor),
                               0.0, C.CLIP_PERIOD_STABILITY).astype(np.float32)

    # ---------- d) 载荷统计通道 ----------
    dlc_f = dlc.astype(np.float32)
    payload_len_norm = dlc_f / float(C.PAD_BYTES)
    pad_ratio = (float(C.PAD_BYTES) - dlc_f) / float(C.PAD_BYTES)

    byte_mean = p32.mean(axis=1)
    byte_std = p32.std(axis=1)
    byte_entropy = _row_entropy(padded)
    nonzero_ratio = (padded != 0).sum(axis=1).astype(np.float32) / float(C.PAD_BYTES)

    # ★ payload_delta：优先逐块计算（P1 修复），无块信息或显式指定 global 时才退化为全局计算
    if blocks and C.PAYLOAD_DELTA_SCOPE == "block_local":
        payload_delta = _payload_delta_block_local(p32, aid, blocks)
        counters["payload_delta_block_local"] = counters.get("payload_delta_block_local", 0) + 1
    else:
        prev_idx = _prev_same_aid(aid)
        payload_delta = (np.abs(p32 - p32[prev_idx]).mean(axis=1) / 255.0).astype(np.float32)
        counters["payload_delta_global"] = counters.get("payload_delta_global", 0) + 1
        if reset_mask is not None and reset_mask.any():
            payload_delta = payload_delta.copy()
            payload_delta[reset_mask] = 0.0

    u32 = (
        (padded[:, 0].astype(np.uint64) << 24)
        | (padded[:, 1].astype(np.uint64) << 16)
        | (padded[:, 2].astype(np.uint64) << 8)
        | padded[:, 3].astype(np.uint64)
    ).astype(np.float64)
    payload_u32 = (u32 / float(1 << 32)).astype(np.float32)

    # ---------- e) 组装 V1（顺序严格对应 C.V1_CHANNELS） ----------
    # 先在 float32 中组装 → 统一量程裁剪 → 再转 float16，避免中途溢出为 inf
    v1f = np.empty((n, len(C.V1_CHANNELS)), dtype=np.float32)
    v1f[:, 0] = dt_norm
    v1f[:, 1] = dt_dev
    v1f[:, 2] = id_norm
    v1f[:, 3] = id_freq
    v1f[:, 4] = payload_len_norm
    v1f[:, 5] = pad_ratio
    v1f[:, 6] = proto_flag_value
    v1f[:, 7] = 0.0          # brs_flag：HCRL 缺该字段 → 缺失而非真实 0（须声明）
    v1f[:, 8] = 0.0          # esi_flag：同上
    v1f[:, 9] = byte_mean
    v1f[:, 10] = byte_std
    v1f[:, 11] = byte_entropy
    v1f[:, 12] = nonzero_ratio
    v1f[:, 13] = payload_delta
    v1f[:, 14] = payload_u32
    v1f[:, 15] = period_stability

    n_bad = int((~np.isfinite(v1f)).sum())
    if n_bad:
        counters["v1_nonfinite_clipped"] = counters.get("v1_nonfinite_clipped", 0) + n_bad
    # 非有限值先归零（理论不应出现；出现即说明上游统计量退化，须排查）
    v1f = np.nan_to_num(v1f, nan=0.0, posinf=C.CLIP_FLOAT16, neginf=-C.CLIP_FLOAT16)
    v1 = np.clip(v1f, -C.CLIP_FLOAT16, C.CLIP_FLOAT16).astype(np.float16)

    # ---------- f) 组装 V2 字节图 ----------
    data_ch = padded.reshape(n, C.PAD_BYTES // C.IMG_BYTE_PER_CH, C.IMG_BYTE_PER_CH)
    mask_ch = mask.reshape(n, C.PAD_BYTES // C.IMG_BYTE_PER_CH, C.IMG_BYTE_PER_CH)
    v2 = np.concatenate([data_ch, mask_ch], axis=1).astype(np.uint8)  # (N,8,8)

    aux = {
        "aid": aid,
        "id_valid": id_valid,
        "ts_rel": ts_rel,
        "dlc": dlc,
        "label": np.asarray(cap_arrays["label"], dtype=np.uint8),
    }
    return v1, v2, aux
