# -*- coding: utf-8 -*-
"""
preprocess.cleaning —— 缺失值处理 / 异常值处理 / 一致性断言
================================================================================
对应方案 §5（G1–G5）与 §8（陷阱清单）。本模块**只做逐 capture 的数组级清洗**，
不涉及跨 capture 逻辑（跨 capture 隔离由 pipeline 保证）。

明确区分三类处理
----------------
(1) 缺失值处理（missing）
    · 载荷为空 / 长度非偶数 / 超长          → utils.hexstrings_to_padded 兜底并计数
    · ROAD 的 RTR 帧（`#` 后为空）          → dlc=0, 载荷全 0（不是丢弃）
    · CTT 的 data_field 空串                → dlc=0
    · HCRL 缺 FDF/BRS/ESI 字段              → 对应通道恒 0，并在 manifest 标 "missing"
    · 无法解析的行（字段数不足 / 标签非法） → **剔除**并逐类计数，行数差额必须可解释
(2) 异常值处理（outlier）
    · 时间戳回退（Δt < 0）                  → 强制非降（np.maximum.accumulate）+ 计数
    · Δt 极端大值                           → 按 **train split** 的 P99.9 鲁棒上裁剪 + 计数
    · 仲裁 ID 超出 11 位（仅 HCRL Fuzzing） → id_norm 钳位，且 aux.id_valid 置 0
    · dlc > 32                              → 截断到 32 + 计数
(3) 归一化（normalization）—— 见 features.py，且统计量**只用 train split 估计**

所有计数都会进入 03_manifest/anomalies.csv 与 dataset_stats.csv，
是论文"数据清洗"小节的直接证据链。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import config as C
from .utils import warn


# --------------------------------------------------------------------------- #
# 缺失值：帧级合法性过滤                                                        #
# --------------------------------------------------------------------------- #
def filter_invalid_rows(
    arrays: dict[str, np.ndarray],
    *,
    valid_mask: np.ndarray,
    counters: dict[str, int],
    reason: str,
) -> dict[str, np.ndarray]:
    """按 valid_mask 过滤所有并列数组，并累加被剔除数量。

    参数
    ----
    arrays     : 同一 capture 的全部帧级数组（长度一致）
    valid_mask : bool 数组，True 保留
    counters   : 可变计数字典（原地累加 reason → 剔除数）
    reason     : 剔除原因标签（用于 anomalies.csv）
    """
    n_drop = int((~valid_mask).sum())
    if n_drop:
        counters[reason] = counters.get(reason, 0) + n_drop
    if n_drop == 0:
        return arrays
    out = {}
    for k, v in arrays.items():
        out[k] = v[valid_mask]
    return out


# --------------------------------------------------------------------------- #
# 异常值：时间戳相对化 + 单调化 + Δt 鲁棒裁剪                                    #
# --------------------------------------------------------------------------- #
def relativize_and_monotonic(
    ts_raw: np.ndarray,
    *,
    counters: dict[str, int],
    ts_base: float | None = None,
) -> np.ndarray:
    """时间戳相对化（方案 §5 G2）+ 回退异常修正。

    ts_rel = ts - (ts_base if ts_base is not None else ts_first)

    ★ 三个数据集都必须做相对化，且理由各不相同：
      · HCRL CAN-FD：原始值本身就是"采集内相对秒"（1307–4295 s）；
      · ROAD      ：每个 log 的起始时间戳是**合成基准**（1.0e9–1.3e9 量级），绝对时间无意义；
      · CTT v1.5  ：每个 CSV 都从 1672531200.0 起（各文件独立）→ 严禁跨文件拼接。

    回退修正：真实采集偶发时间戳抖动/回绕，会让 Δt 变负并污染 dt_norm。
    处理方式为 np.maximum.accumulate（强制非降），被修正帧数计入计数。
    """
    ts_raw = np.asarray(ts_raw, dtype=np.float64)
    if ts_raw.size == 0:
        return ts_raw
    base = float(ts_raw[0]) if ts_base is None else float(ts_base)
    ts_rel = ts_raw - base

    mono = np.maximum.accumulate(ts_rel)
    n_back = int(np.count_nonzero(mono != ts_rel))
    if n_back:
        counters["ts_backward_fixed"] = counters.get("ts_backward_fixed", 0) + n_back
    return mono


def compute_dt(ts_rel: np.ndarray) -> np.ndarray:
    """Δt 序列（首帧 Δt = 0，表示无前驱帧）。"""
    ts_rel = np.asarray(ts_rel, dtype=np.float64)
    dt = np.zeros_like(ts_rel)
    if ts_rel.size > 1:
        dt[1:] = np.diff(ts_rel)
    np.clip(dt, 0.0, None, out=dt)  # 二次保险：负数一律 0
    return dt


def fit_dt_clip(train_dt: np.ndarray, quantile: float = C.DT_CLIP_QUANTILE) -> float:
    """用 train split 估计 Δt 的鲁棒上界（默认 P99.9）。"""
    if train_dt.size == 0:
        return float("inf")
    pos = train_dt[train_dt > 0]
    if pos.size == 0:
        return float("inf")
    return float(np.quantile(pos, quantile))


def apply_dt_clip(dt: np.ndarray, hi: float, *, counters: dict[str, int]) -> np.ndarray:
    """按 train 估计的上界裁剪 Δt 极端值（跨 split 统一用同一上界，做对 val/test 的严格离线）。"""
    if not np.isfinite(hi):
        return dt
    n_clip = int(np.count_nonzero(dt > hi))
    if n_clip:
        counters["dt_clipped"] = counters.get("dt_clipped", 0) + n_clip
    return np.minimum(dt, hi)


# --------------------------------------------------------------------------- #
# 异常值：ID / DLC 值域校验                                                     #
# --------------------------------------------------------------------------- #
def validate_aid(aid: np.ndarray, *, counters: dict[str, int]) -> np.ndarray:
    """返回 id_valid（aid ≤ 0x7FF）。超范围 ID 只统计、**不剔除**。

    ★ 方案 §4 警示框：HCRL Fuzzing 的随机 ID 最高 0xFFFE，超出 11 位。
    "出现超出 11 位的 ID"本身就是 Fuzzing 的强特征，因此保留该帧并单独承载信号；
    但必须在论文中声明选用的是"钳位 + id_valid 双通道"方案（选项 ③）。
    """
    over = int(np.count_nonzero(aid > C.AID_STD_MAX))
    if over:
        counters["aid_over_11bit"] = counters.get("aid_over_11bit", 0) + over
        warn(f"检测到 {over} 帧仲裁 ID 超出 11 位（>0x7FF），已保留并以 id_valid=0 标记")
    return (aid <= C.AID_STD_MAX).astype(np.uint8)


def validate_dlc(dlc: np.ndarray, *, counters: dict[str, int]) -> np.ndarray:
    """校验并截断 dlc > 32 的帧（本方案 PAD_BYTES=32，实测最长即 32）。"""
    bad = int(np.count_nonzero(dlc > C.PAD_BYTES))
    if bad:
        counters["dlc_over_32"] = counters.get("dlc_over_32", 0) + bad
        warn(f"检测到 {bad} 帧 dlc > {C.PAD_BYTES}，已截断到 {C.PAD_BYTES}")
    return np.minimum(dlc, C.PAD_BYTES).astype(np.uint8)


def validate_label(label: np.ndarray, *, counters: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """校验标签取值 ∈ {0,1}，返回 (label_u8, valid_mask)。"""
    valid = (label == 0) | (label == 1)
    n_bad = int((~valid).sum())
    if n_bad:
        counters["label_invalid_dropped"] = counters.get("label_invalid_dropped", 0) + n_bad
    return label.astype(np.uint8), valid


# --------------------------------------------------------------------------- #
# 断言（一致性红线，对应方案 §9 的"校验判据"一栏）                                #
# --------------------------------------------------------------------------- #
def assert_no_nan(arr: np.ndarray, name: str) -> None:
    """L2 张量不得含 NaN/Inf。"""
    if arr.dtype.kind == "f":
        if not np.isfinite(arr).all():
            raise AssertionError(f"{name}: 存在 NaN/Inf，预处理失败（x01 红线）")


def assert_sorted_splits(split_idx: dict[str, np.ndarray]) -> None:
    """【已弃用，2026-10-17】仅适用于"连续切分"模式。

    ★ 块化交错切分下，三段在时间轴上**交错**（这是刻意设计），因此"train 早于 val 早于 test"
      这一断言不再成立，改由 `splitting.assert_partition()` 断言"分区合法性"
      （无重叠、无空洞、区间升序）——这是放弃严格时间外推后仍然守住的纪律。
    本函数保留仅供 per_capture_ratio 模式的旧脚本兼容使用。
    """
    tr, va, te = split_idx["train"], split_idx["val"], split_idx["test"]
    assert tr[1] == va[0], "train 与 val 的行区间不连续"
    assert va[1] == te[0], "val 与 test 的行区间不连续"


def assert_lengths_consistent(**arrays: np.ndarray) -> int:
    """同一 capture 内所有帧级数组长度必须一致。"""
    lens = {k: len(v) for k, v in arrays.items()}
    if len(set(lens.values())) != 1:
        raise AssertionError(f"帧级数组长度不一致：{lens}")
    return next(iter(lens.values()))


def hcrl_len_assertion(dlc_declared: np.ndarray, dlc_actual: np.ndarray, capture: str) -> int:
    """P2 断言：HCRL CAN-FD 的 col2（第三字段）必须等于实际载荷字节数。

    方案 §10 P2 要求"落地时应对三个文件都跑一次全量断言"——本函数即该断言的落地。
    返回不一致行数（正常应为 0）。
    """
    bad = int(np.count_nonzero(dlc_declared != dlc_actual))
    if bad:
        warn(f"{capture}: col2 与实际载荷字节数不一致的行数 = {bad}（方案 P2 假设被打破）")
    return bad
