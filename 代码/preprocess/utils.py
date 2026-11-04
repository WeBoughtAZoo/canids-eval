# -*- coding: utf-8 -*-
"""
preprocess.utils —— 通用工具
================================================================================
包含三类：
  1) hex / 字符串 → numpy 数组的**向量化**解析（避免 2.5 亿行级别的 Python 逐行循环）
  2) 轻量日志与进度
  3) 目录、哈希、原子写等落盘工具
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# 日志                                                                         #
# --------------------------------------------------------------------------- #
_T0 = time.time()


def log(msg: str, *, level: str = "INFO") -> None:
    """带时间戳与相对耗时的单行日志（stdout，便于重定向到 run log）。"""
    print(f"[{time.strftime('%H:%M:%S')}][{level:<4}][{time.time() - _T0:8.1f}s] {msg}", flush=True)


def warn(msg: str) -> None:
    log(msg, level="WARN")


def err(msg: str) -> None:
    log(msg, level="ERR")


# --------------------------------------------------------------------------- #
# hex 向量化解析                                                                #
# --------------------------------------------------------------------------- #
#: ASCII → 十六进制数值查表（'0'-'9' → 0-9；'A'-'F'/'a'-'f' → 10-15；其余 → 0）
_HEX_LUT = np.zeros(256, dtype=np.uint8)
for _c in range(ord("0"), ord("9") + 1):
    _HEX_LUT[_c] = _c - ord("0")
for _c in range(ord("A"), ord("F") + 1):
    _HEX_LUT[_c] = _c - ord("A") + 10
for _c in range(ord("a"), ord("f") + 1):
    _HEX_LUT[_c] = _c - ord("a") + 10


def hexstrings_to_uint(hex_strs: Sequence[str], *, dtype=np.uint32) -> np.ndarray:
    """把等宽/不等宽十六进制字符串数组**向量化**转为无符号整数数组。

    实现：用 numpy 定长 Unicode 数组的底层字节（UTF-32-LE，每字符 4 字节）取 ASCII 码，
    再经查表与权重求和，全程无 Python 级循环。

    参数
    ----
    hex_strs : 十六进制字符串序列，如 ["1E5", "0F1", "0x6e0"]（大小写均可、允许 0x 前缀）
    dtype    : 输出整数类型

    注意
    ----
    · 方案 §7.2 / 陷阱 #3：**绝不能**对 CAN FD 随机 ID 做 `s[1:]` 截首字符，
      本函数按"全字段数值"解析，0xFFFE 与 0x0FFE 不会被混为一谈。
    """
    n = len(hex_strs)
    if n == 0:
        return np.zeros(0, dtype=dtype)

    # 去掉 0x / 0X 前缀
    norm = [s[2:] if (len(s) > 2 and s[0] == "0" and s[1] in "xX") else s for s in hex_strs]
    width = max(len(s) for s in norm)
    if width == 0:
        return np.zeros(n, dtype=dtype)

    # 左补 0 到统一宽度（数值语义）
    padded = [s.rjust(width, "0") for s in norm]
    unicode_arr = np.array(padded, dtype=f"U{width}")
    # 'U{w}' 的 itemsize = w*4 字节（UTF-32 小端），故每字符首字节即 ASCII 码
    codes = np.frombuffer(unicode_arr.tobytes(), dtype=np.uint8).reshape(n, width, 4)[:, :, 0]
    digits = _HEX_LUT[codes]  # (n, width) uint8
    weights = (16 ** np.arange(width - 1, -1, -1, dtype=np.int64)).astype(np.int64)
    vals = (digits.astype(np.int64) * weights).sum(axis=1)
    return vals.astype(dtype)


def hexstrings_to_padded(
    hex_strs: Sequence[str],
    *,
    max_bytes: int,
    odd_left_pad: bool = True,
    stats: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """把十六进制载荷字符串数组转为**右侧补零**的定长字节矩阵。

    这是"缺失值处理"的核心工具之一（方案 §5 G1：载荷保持变长；
    填充只在编码层做，且填充值为 0x00）。

    参数
    ----
    hex_strs   : hex 载荷字符串（允许长度 0 – 2*max_bytes）
    max_bytes  : 目标总字节数（本方案 = 32）
    odd_left_pad: 奇数字符长度时的补齐方向。
                  True  → 左补 '0'（保持大端字节解释，即 0F1 → 00F1）
                  False → 右补 '0'
    stats      : 若提供 dict，则回写异常计数（odd_len / too_long / empty）

    返回
    ----
    (buf, dlc) : buf (N, max_bytes) uint8；dlc (N,) uint8 = 实际字节数（未填充前）
    """
    n = len(hex_strs)
    if stats is not None:
        stats.setdefault("empty_payload", 0)
        stats.setdefault("odd_hex_len", 0)
        stats.setdefault("payload_too_long", 0)

    if n == 0:
        return np.zeros((0, max_bytes), dtype=np.uint8), np.zeros(0, dtype=np.uint8)

    max_chars = 2 * max_bytes
    buf = np.zeros((n, max_bytes), dtype=np.uint8)
    dlc = np.zeros(n, dtype=np.uint8)

    # 分组：正常偶数长度且不超长 → 走向量化快路径；异常 → 逐行慢路径（数量极少）
    good_idx: list[int] = []
    good_hex: list[str] = []
    bad_idx: list[int] = []
    for i, h in enumerate(hex_strs):
        if h == "":
            if stats is not None:
                stats["empty_payload"] += 1
            bad_idx.append(i)
            continue
        if len(h) > max_chars:
            if stats is not None:
                stats["payload_too_long"] += 1
            bad_idx.append(i)
            continue
        if len(h) % 2 == 1:
            if stats is not None:
                stats["odd_hex_len"] += 1
            h = ("0" + h) if odd_left_pad else (h + "0")
        good_idx.append(i)
        good_hex.append(h)

    if good_idx:
        gi = np.asarray(good_idx, dtype=np.int64)
        joined = "".join(h.ljust(max_chars, "0") for h in good_hex)
        raw = np.frombuffer(bytes.fromhex(joined), dtype=np.uint8).reshape(len(good_hex), max_bytes)
        buf[gi] = raw
        dlc[gi] = (np.asarray([len(h) for h in good_hex], dtype=np.int64) // 2).astype(np.uint8)

    # 慢路径：空载荷 / 超长载荷（截断到 max_bytes 并记录）
    for i in bad_idx:
        h = hex_strs[i]
        if h == "":
            continue
        h = h[:max_chars]
        if len(h) % 2 == 1:
            h = h[:-1]
        b = bytes.fromhex(h)
        buf[i, : len(b)] = np.frombuffer(b, dtype=np.uint8)
        dlc[i] = len(b)

    return buf, dlc


# --------------------------------------------------------------------------- #
# 其他工具                                                                      #
# --------------------------------------------------------------------------- #
def ensure_dir(p: Path) -> Path:
    """创建目录（含父目录），返回自身，便于链式调用。"""
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def dict_array(values: Sequence, n: int, value: str):
    """构造"整列同值"的 pyarrow 字典数组（避免为 1e6 行重复构造 Python 字符串对象）。

    方案 §3 的 schema 里 src / capture / attack_type 属于 capture 级常量，
    用字典数组存储开销可忽略，同时保留"列"的形态以便下游直接按列过滤。
    """
    import pyarrow as pa

    return pa.DictionaryArray.from_arrays(
        pa.array(np.zeros(n, dtype=np.int32)), pa.array([value])
    )


def file_md5_head(path: Path, nbytes: int = 0) -> str:
    """计算文件 md5；nbytes=0 表示全文件，否则只算前 nbytes 字节（大文件快速指纹）。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        if nbytes <= 0:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        else:
            h.update(f.read(nbytes))
    return h.hexdigest()


def config_fingerprint(cfg) -> str:
    """把 config 模块中与口径相关的常量序列化后取哈希，作为可复现性凭据之一。"""
    keys = [
        "WIN_T", "STRIDE_TRAIN", "STRIDE_TEST", "PAD_BYTES", "IMG_CHANNELS",
        "V1_CHANNELS", "SPLIT_RATIOS", "DT_CLIP_QUANTILE", "ROAD_FUZZING_MODE",
        "PROTO_FLAG_MODE", "CTT_TS_BASE", "SEED", "AID_NORM_DENOM", "AID_STD_MAX",
        "CTT_INCLUDE_EXTRA_SETS",  # 影响产物构成（是否纳入 hcrl-ch / hcrl-sa）
        # CTT 语义划分：直接决定 train/val/test 归属与 val 配额，必须进指纹
        "CTT_SEMANTIC_SPLIT", "CTT_VAL_QUOTA",
        # 2026-10-17 新增：块化交错切分口径
        # （否则"换了切分方式但指纹不变"会让复现凭据失效）
        "SPLIT_MODE", "BLOCK_MODE_DATASETS", "BLOCK_K", "BLOCK_PATTERN_PERIOD",
        "BLOCK_VAL_MOD", "BLOCK_TEST_MOD",
        "ALIGN_TO_SEGMENT_EDGE", "SEGMENT_SEARCH_FRAC",
        "ALIGN_WIDEN_FACTOR", "ALIGN_WIDEN_STEPS",
        "MIN_BLOCK_FRAMES", "PAYLOAD_DELTA_SCOPE",
    ]
    payload = {k: getattr(cfg, k) for k in keys}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def write_json(path: Path, obj) -> None:
    """写 JSON（UTF-8、缩进 2、保留非 ASCII），用于 manifest。"""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TiB"


def dir_size(path: Path) -> int:
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def relpath_str(p: Path, root: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(Path(root).resolve())).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")
