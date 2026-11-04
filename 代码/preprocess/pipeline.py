# -*- coding: utf-8 -*-
"""
preprocess.pipeline —— 编排：解析 → 清洗 → 切分 → 统计 → 编码 → 落盘 → 清单
================================================================================
对应方案 §2（三层产物架构）与 §9（实操顺序 E1–E3）。

产物（按数据集分目录，统一落在"训练数据集/＜数据集＞/"下）：
  01_interim/＜capture＞.parquet         L1 统一中间表示（按 capture 分片，保留文件边界）
  02_tensors/＜split＞/＜capture＞_*.npy  L2 定长张量（帧级；训练只读这一层）
  03_manifest/splits.json               每 capture 的 train/val/test 行号切分点
  03_manifest/dataset_stats.csv         逐 capture 行数 / 攻击数 / 时间范围 / 各类计数
  03_manifest/anomalies.csv             逐 capture 的缺失值 / 异常值处理计数（证据链）
  03_manifest/env.json                  版本、日期、配置指纹（可复现性凭据）
  03_manifest/captures/＜capture＞.json  单 capture 记录（支持断点续跑）

单 capture 内部顺序（与方案 §9 的五个步骤一一对应）：
  E1 解析为中间表示 → E2 时间序 60/20/20 切分 → E3 特征编码并落盘 → E4 清单与校验
"""

from __future__ import annotations

import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import config as C
from . import parsers
from .cleaning import (
    assert_no_nan,
    relativize_and_monotonic,
    validate_label,
)
from .features import TrainStats, encode_capture, fit_train_stats
from .splitting import (
    all_blocks,
    assert_partition,
    block_start_mask,
    make_block_splits,
    make_role_splits,
    make_splits,
    ranges_total,
    slice_concat,
)
from .utils import (
    config_fingerprint,
    dict_array,
    dir_size,
    ensure_dir,
    human_bytes,
    log,
    read_json,
    warn,
    write_json,
)


# --------------------------------------------------------------------------- #
# 目录布局                                                                      #
# --------------------------------------------------------------------------- #
def dataset_out_root(dataset: str) -> Path:
    return C.OUT_ROOT / C.DATASET_DIRS[dataset]


def _paths(dataset: str) -> dict[str, Path]:
    root = dataset_out_root(dataset)
    return {
        "root": root,
        "interim": root / "01_interim",
        "tensors": root / "02_tensors",
        "manifest": root / "03_manifest",
        "captures": root / "03_manifest" / "captures",
    }


# --------------------------------------------------------------------------- #
# L1 落盘：中间表示 Parquet（变长载荷，不做填充）                                 #
# --------------------------------------------------------------------------- #
def _hex_to_bytes(h: str) -> bytes:
    if h == "":
        return b""
    if len(h) % 2 == 1:
        h = "0" + h
    return bytes.fromhex(h)


def write_interim_parquet(
    path: Path,
    *,
    ts_rel: np.ndarray,
    aid: np.ndarray,
    dlc: np.ndarray,
    data_hex: list[str],
    label: np.ndarray,
    src: str,
    capture: str,
    attack_type: str,
) -> None:
    """写 L1 Parquet（schema 见方案 §3）。

    ★ 一文件对一 capture：Parquet 天然隔离 ROAD / CTT 的独立时间基，杜绝误拼接（方案 §2）。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = len(ts_rel)
    payload = [_hex_to_bytes(h) for h in data_hex]
    table = pa.table(
        {
            "ts_rel": pa.array(ts_rel, type=pa.float64()),
            "aid": pa.array(aid, type=pa.uint32()),
            "dlc": pa.array(dlc, type=pa.uint8()),
            "data": pa.array(payload, type=pa.binary()),
            "label": pa.array(label, type=pa.uint8()),
            "src": dict_array(None, n, src),
            "capture": dict_array(None, n, capture),
            "attack_type": dict_array(None, n, attack_type),
        }
    )
    ensure_dir(path.parent)
    pq.write_table(table, str(path), compression=C.PARQUET_COMPRESSION)


# --------------------------------------------------------------------------- #
# L2 落盘：帧级定长张量                                                          #
# --------------------------------------------------------------------------- #
def write_tensors(
    tensors_dir: Path,
    capture: str,
    split: str,
    v1: np.ndarray,
    v2: np.ndarray,
    aux: dict[str, np.ndarray],
) -> dict[str, str]:
    """写一个 capture 的某一 split 的帧级张量。

    V1 float16 [n, 16] / V2 uint8 [n, 8, 8] / label uint8 [n]
    另附 aid(uint32)、id_valid(uint8)、ts_rel(float64) 三个 aux 数组，
    供分层评测（按攻击类型/按 ID）与时间序复核使用。

    ★ 惰性窗口化（方案 §6 内存预算）：只落帧级张量，窗口在 __getitem__ 现场切，
      磁盘占用降为窗口级展开的约 1/32。
    """
    d = ensure_dir(tensors_dir / split / capture)
    out: dict[str, str] = {}
    files = {
        "v1": (v1, C.DTYPE_V1),
        "v2": (v2, C.DTYPE_V2),
        "label": (aux["label"], C.DTYPE_LABEL),
        "aid": (aux["aid"], C.DTYPE_AID),
        "id_valid": (aux["id_valid"], C.DTYPE_V2),
        "ts_rel": (aux["ts_rel"], C.DTYPE_TSREL),
    }
    for name, (arr, _dt) in files.items():
        fp = d / f"{name}.npy"
        np.save(fp, arr, allow_pickle=False)
        out[name] = str(fp)
    return out


# --------------------------------------------------------------------------- #
# 单 capture 处理                                                               #
# --------------------------------------------------------------------------- #
def process_capture(
    spec: parsers.CaptureSpec,
    *,
    paths: dict[str, Path],
    road_meta: dict[str, Any] | None,
    force: bool = False,
    global_stats: "TrainStats | None" = None,
) -> dict[str, Any]:
    """处理单个 capture，返回可写入 manifest 的记录（dict）。

    参数
    ----
    global_stats : 数据集级**训练集**统计量。
        非 None 时优先使用（CTT 语义划分下必须传入，否则 test capture 无 train 区间可估统计量）；
        None 时退回"用本 capture 的 train 区间估计"（HCRL / ROAD 的 capture 内 60/20/20 路径）。
    """
    rec_path = paths["captures"] / f"{spec.capture}.json"
    if rec_path.exists() and not force:
        rec = read_json(rec_path)
        rec["_resumed"] = True
        return rec

    t0 = time.time()
    anom: dict[str, int] = {}

    # ---------------- E1 解析（统一中间表示，不做填充） ---------------- #
    cap = parsers.parse_capture(spec, metadata=road_meta) if spec.src == "road" \
        else parsers.parse_capture(spec)
    for k, v in cap.counters.items():
        anom[k] = anom.get(k, 0) + int(v)

    # ---------------- 清洗：缺失值 / 异常值 ---------------- #
    ts_base = C.CTT_TS_BASE if spec.src.startswith("ctt15") else None
    ts_rel = relativize_and_monotonic(cap.ts_raw, counters=anom, ts_base=ts_base)
    label, valid = validate_label(cap.label, counters=anom)
    if not valid.all():
        ts_rel = ts_rel[valid]
        cap.aid = cap.aid[valid]
        cap.data_hex = [h for h, v in zip(cap.data_hex, valid) if v]
        label = label[valid]

    n = int(ts_rel.size)
    if n < C.WIN_T + 2:
        warn(f"{spec.capture}: 有效帧数 {n} < {C.WIN_T}+2，无法构成一个窗口 → 跳过落盘")
        rec = _record(spec, n, 0, ts_rel, label, {}, anom, time.time() - t0, skipped=True)
        write_json(rec_path, rec)
        return rec

    # ---------------- E2 切分 ---------------- #
    # 三种模式（2026-10-17 修订）：
    #   ① 语义划分（CTT）：按文件名角色整文件归入，保四象限设计；
    #   ② ★ 块化交错（HCRL / ROAD）：单时间线切 K 块、按固定模模式分配，
    #      块边界对齐静默区，块内保序、窗口严格块内构造；
    #   ③ 传统连续 60/20/20（仅当块化交错因 capture 过短而退化时使用）。
    split_info: dict[str, Any] = {"mode": None}
    role = spec.meta.get("split_role") if C.CTT_SEMANTIC_SPLIT else None
    if role in ("train", "val", "test"):
        splits = make_role_splits(n, role)
        split_info = {"mode": "semantic_role", "role": role}
    elif C.SPLIT_MODE == "block_interleaved" and spec.src in C.BLOCK_MODE_DATASETS:
        splits, info = make_block_splits(n, label)
        split_info = {"mode": "block_interleaved_fallback" if info.get("fallback")
                      else "block_interleaved", **info}
        if info.get("fallback"):
            warn(f"{spec.capture}: 帧数 {n} 不足以块化交错，已退化为连续 60/20/20")
    else:
        splits = make_splits(n)
        split_info = {"mode": "per_capture_ratio", "ratios": list(C.SPLIT_RATIOS)}

    assert_partition(splits, n)                  # 无重叠、无空洞、区间升序
    bstart = block_start_mask(n, splits)         # 每个切分单元的首帧

    # ---------------- 训练集统计量 ---------------- #
    from .cleaning import compute_dt

    aid_size = int(max(int(cap.aid.max()) + 1, C.AID_STD_MAX + 1))
    dt_all = compute_dt(ts_rel)
    if bstart.any():
        dt_all = dt_all.copy()
        dt_all[bstart] = 0.0                     # 与编码层保持同一口径

    if global_stats is not None:
        # ★ 语义划分路径：统一使用**由 train 角色 capture 聚合**的全局统计量。
        #   这样 train / val / test 三段共享完全相同的归一化尺度，
        #   既避免用测试数据估计统计量（泄漏），又保证特征分布一致。
        stats = global_stats
    else:
        tr_ranges = splits["train"]
        aid_tr = slice_concat(cap.aid, tr_ranges)
        dt_tr = slice_concat(dt_all, tr_ranges)
        bs_tr = slice_concat(bstart, tr_ranges)
        keep = ~bs_tr                            # 排除块首帧（其 Δt 已归零，不代表真实周期）
        stats = fit_train_stats(aid_tr[keep], dt_tr[keep], aid_size)

    # ---------------- E3 编码 + 落盘 ---------------- #
    proto_flag = 1 if (C.PROTO_FLAG_MODE == "protocol" and spec.protocol == "CANFD") else 0
    v1, v2, aux = encode_capture(
        {"ts_rel": ts_rel, "aid": cap.aid, "data_hex": cap.data_hex, "label": label},
        stats,
        proto_flag_value=proto_flag,
        counters=anom,
        reset_mask=bstart,
        blocks=all_blocks(splits),   # ★ P1 修复：payload_delta 逐块计算，杜绝跨 split 引用
    )
    assert_no_nan(v1.astype(np.float32), f"{spec.capture}:v1")

    # 逐 split 落盘（★ 空区间不落盘：语义划分下角色外的 split 为空，
    #   从源头保证三段无帧重叠）
    for split in C.SPLIT_NAMES:
        rng = splits[split]
        d = paths["tensors"] / split / spec.capture
        if not rng or ranges_total(rng) == 0:
            if d.exists():                       # 移走可能存在的旧目录（避免孤儿/重复数据）
                _quarantine_orphan(d, paths["tensors"])
            continue
        write_tensors(
            paths["tensors"], spec.capture, split,
            slice_concat(v1, rng), slice_concat(v2, rng),
            {k: slice_concat(v, rng) for k, v in aux.items()},
        )

    # L1 中间表示（用编码阶段得到的 dlc，保证与 L2 一致）
    write_interim_parquet(
        paths["interim"] / f"{spec.capture}.parquet",
        ts_rel=ts_rel, aid=cap.aid, dlc=aux["dlc"], data_hex=cap.data_hex,
        label=label, src=spec.src, capture=spec.capture, attack_type=spec.attack_type,
    )

    rec = _record(spec, n, aid_size, ts_rel, label, splits, anom, time.time() - t0,
                  split_info=split_info)
    rec["train_stats"] = stats.to_json()
    rec["_resumed"] = False
    write_json(rec_path, rec)
    return rec


def _record(
    spec: parsers.CaptureSpec,
    n: int,
    aid_size: int,
    ts_rel: np.ndarray,
    label: np.ndarray,
    splits: dict[str, list[tuple[int, int]]],
    anom: dict[str, int],
    elapsed: float,
    *,
    skipped: bool = False,
    split_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """组装单 capture 的 manifest 记录（切分以**区间列表**表达，支持多区间）。"""
    rec: dict[str, Any] = {
        "src": spec.src,
        "capture": spec.capture,
        "protocol": spec.protocol,
        "attack_type": spec.attack_type,
        "attack_family": spec.attack_family,
        "is_masquerade": bool(spec.is_masquerade),
        "supervised_evaluable": bool(spec.supervised_evaluable),
        "raw_path": str(spec.path),
        "n_frames": int(n),
        "aid_size": int(aid_size),
        "n_attack_frames": int(label.sum()) if n else 0,
        "attack_ratio": float(label.mean()) if n else 0.0,
        "ts_start": float(ts_rel[0]) if n else 0.0,
        "ts_end": float(ts_rel[-1]) if n else 0.0,
        "duration_sec": float(ts_rel[-1] - ts_rel[0]) if n else 0.0,
        "skipped": bool(skipped),
        "elapsed_sec": round(float(elapsed), 3),
        "anomalies": {k: int(v) for k, v in sorted(anom.items()) if k != "raw_rows_total"},
        "raw_rows_total": int(anom.get("raw_rows_total", n)),
        "meta": spec.meta,
    }
    if split_info:
        rec["split_mode"] = split_info
    if splits:
        rec["splits"] = {}
        for sp in C.SPLIT_NAMES:
            rng = splits.get(sp, [])
            seg = slice_concat(label, rng)
            nw_tr = sum(max(0, 0 if (b - a) < C.WIN_T else 1 + (b - a - C.WIN_T) // C.STRIDE_TRAIN)
                        for a, b in rng)
            nw_te = sum(max(0, 0 if (b - a) < C.WIN_T else 1 + (b - a - C.WIN_T) // C.STRIDE_TEST)
                        for a, b in rng)
            rec["splits"][sp] = {
                "ranges": [[int(a), int(b)] for a, b in rng],
                "n_segments": len(rng),
                "n": int(ranges_total(rng)),
                "n_attack": int(seg.sum()) if seg.size else 0,
                "attack_ratio": float(seg.mean()) if seg.size else 0.0,
                "windows_stride_train": int(nw_tr),
                "windows_stride_test": int(nw_te),
            }
    return rec


def _quarantine_orphan(d: Path, tensors_dir: Path) -> None:
    """把**不该存在**的 split 目录移入隔离区（而非删除）。

    ★ 语义化划分切换时会产生孤儿目录：例如某个 capture 原本在 test/ 下有数据，
      改为 train 角色后其 test 段为零长度。这些残留若不清掉会造成
      train / val / test 三段出现**重复样本**，是评测泄漏的隐蔽来源。
    采用"移动"而非删除：① 可恢复 ② 规避批量删除的安全拦截。
    """
    dest_root = tensors_dir.parent / "_orphan_old_split"
    dest = dest_root / d.parent.name / d.name
    if dest.exists():                      # 已存在同名 → 加时间戳后缀
        dest = dest_root / d.parent.name / f"{d.name}__{int(time.time())}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(d), str(dest))


def _assign_val_captures(specs: list[parsers.CaptureSpec]) -> int:
    """从 train 角色的 capture 中**按攻击类型分层**留出一部分作 val（原地改写 meta.split_role）。

    ★ 为什么是分层留出：CTT 各类攻击的采集文件数不等，若整体随机留出，
      可能出现某类攻击在 val 中缺失（该类攻击无法标定阈值）或占比失衡。
      分层保证**每种攻击类型在 val 中都有代表**。

    留出规则（确定性，SEED=42 可复现）：
      · 组内文件数 == 1 → 不留（否则 train 将失去该类型）
      · 组内文件数 2–3 → 留 1 个
      · 组内文件数 ≥ 4 → 留 round(CTT_VAL_QUOTA × size) 个
    返回被改为 val 的 capture 数。
    """
    rng = np.random.default_rng(C.SEED)
    groups: dict[str, list[parsers.CaptureSpec]] = {}
    for s in specs:
        if s.meta.get("split_role") == "train":
            groups.setdefault(s.attack_type, []).append(s)

    n_val = 0
    for atype, members in sorted(groups.items()):
        members = sorted(members, key=lambda x: x.capture)
        size = len(members)
        if size <= 1:
            k = 0
        elif size <= 3:
            k = 1
        else:
            k = int(round(C.CTT_VAL_QUOTA * size))
            k = max(1, min(k, size - 1))          # 至少留 1 个，但 train 至少保留 1 个
        if k == 0:
            continue
        pick = rng.choice(size, size=k, replace=False)
        for i in pick:
            members[i].meta["split_role"] = "val"
            members[i].meta["val_by_stratified_holdout"] = True
            n_val += 1
    return n_val


def _fit_global_train_stats(specs: list[parsers.CaptureSpec]) -> TrainStats:
    """Pass 1：由全部 train 角色 capture 的 train 区间**聚合**数据集级统计量。

    ★ 这是 CTT 语义划分的关键配套：统计量**只**来自 train_01 / train_02 的前 80%，
      与 test_* 完全隔离，从源头杜绝"用测试数据做归一化"这一常见泄漏。
    """
    from .cleaning import compute_dt, validate_label

    aids: list[np.ndarray] = []
    dts: list[np.ndarray] = []
    for spec in specs:
        cap = parsers.parse_capture(spec)
        anom: dict[str, int] = {}
        ts_base = C.CTT_TS_BASE if spec.src.startswith("ctt15") else None
        ts_rel = relativize_and_monotonic(cap.ts_raw, counters=anom, ts_base=ts_base)
        _label, valid = validate_label(cap.label, counters=anom)
        if not valid.all():
            ts_rel = ts_rel[valid]
            cap.aid = cap.aid[valid]
        n = int(ts_rel.size)
        if n < C.WIN_T + 2:
            continue
        sp = make_role_splits(n, "train")
        rng = sp["train"]
        if not rng or ranges_total(rng) == 0:
            continue
        dt = compute_dt(ts_rel)
        # 与 process_capture 口径一致：按区间列表切片并拼接
        aids.append(np.asarray(slice_concat(cap.aid, rng), dtype=np.uint32))
        dts.append(np.asarray(slice_concat(dt, rng), dtype=np.float64))
        del cap

    if not aids:
        raise RuntimeError("Pass 1 未收集到任何 train 区间样本，无法拟合全局统计量")

    aid_all = np.concatenate(aids)
    dt_all = np.concatenate(dts)
    aid_size = int(max(int(aid_all.max()) + 1, C.AID_STD_MAX + 1))
    return fit_train_stats(aid_all, dt_all, aid_size)


# --------------------------------------------------------------------------- #
# 数据集级执行                                                                  #
# --------------------------------------------------------------------------- #
def run_dataset(
    dataset: str,
    *,
    limit: int | None = None,
    force: bool = False,
    only_captures: list[str] | None = None,
) -> dict[str, Any]:
    """跑完一个数据集的全部 capture，并输出三份 manifest。"""
    paths = _paths(dataset)
    for k in ("interim", "tensors", "manifest", "captures"):
        ensure_dir(paths[k])

    specs = parsers.list_captures(dataset)
    if only_captures:
        specs = [s for s in specs if s.capture in set(only_captures)]
    if limit is not None:
        specs = specs[:limit]
    if not specs:
        warn(f"{dataset}: 未发现任何 capture（请检查原始目录 {C.DATA_ROOT}）")
        return {"dataset": dataset, "captures": 0, "records": []}

    # ROAD 需要元数据生成标签
    road_meta = None
    if dataset == "road":
        road_meta = {}
        for sub in ("ambient", "attacks"):
            mp = C.RAW_ROAD / sub / "capture_metadata.json"
            if mp.exists():
                road_meta.update(read_json(mp))
        log(f"ROAD 元数据已载入：{len(road_meta)} 条")

    # ★★ CTT 语义划分：先由 train 角色 capture 聚合**数据集级**统计量（Pass 1），
    #    再统一用它编码全部 capture（Pass 2）。
    #    必要性：语义划分下 test capture 没有 train 区间，
    #    若各 capture 各自估计统计量会①用测试数据估计（泄漏）②train/test 特征分布不一致。
    global_stats = None
    # ★ 2026-10-28 扩展：ROAD 的 capture 级留出（敏感性实验）与 CTT 语义划分同理——
    #   val / test 的 capture 没有 train 区间，若退回"逐 capture 用本 capture 的
    #   train 区间估计统计量"，会因区间为空而得到退化统计量，
    #   使 val / test 特征整体作废（实测：val MCC 0.0057、全部概率饱和为 1.0）。
    #   故两者共用同一条"Pass 1 全局统计量"路径。
    use_global = ((dataset == "ctt15" and C.CTT_SEMANTIC_SPLIT)
                  or (dataset == "road" and C.ROAD_CAPTURE_HOLDOUT))
    if use_global:
        if dataset == "ctt15":
            # 先按攻击类型分层留出 val（原地改写 meta.split_role），
            # 再做全局统计量拟合——保证 val 的 capture **不参与**统计量估计。
            n_val = _assign_val_captures(specs)
        else:
            # ROAD：val 已由 _assign_road_capture_holdout 按族分配好，直接统计
            n_val = sum(1 for s in specs if s.meta.get("split_role") == "val")
        train_specs = [s for s in specs if s.meta.get("split_role") == "train"]
        n_te = sum(1 for s in specs if s.meta.get("split_role") == "test")
        note = ("val 由 train_01/02 按攻击类型分层留出，quota=%.0f" % C.CTT_VAL_QUOTA
                if dataset == "ctt15" else "val / test 由族内贪心均衡整文件留出")
        log(f"{dataset} 语义划分：train={len(train_specs)} val={n_val} test={n_te}（{note}）")
        if train_specs:
            log(f"{dataset} 语义划分：Pass 1 —— 由 {len(train_specs)} 个 train 角色 capture 聚合全局统计量")
            global_stats = _fit_global_train_stats(train_specs)
            log(
                f"  全局统计量：dt_log_mu={global_stats.dt_log_mu:.6f} "
                f"dt_log_sigma={global_stats.dt_log_sigma:.6f} "
                f"dt_clip_hi={global_stats.dt_clip_hi:.6g} "
                f"n_aid={int(np.count_nonzero(global_stats.freq)):,}"
            )
        else:
            warn(f"{dataset} 语义划分已开启但未发现 train 角色 capture，退回逐 capture 估计统计量")

    log(f"===== 开始处理数据集 {dataset}：{len(specs)} 个 capture =====")
    records: list[dict[str, Any]] = []
    for i, spec in enumerate(specs, 1):
        try:
            rec = process_capture(
                spec, paths=paths, road_meta=road_meta, force=force,
                global_stats=global_stats,
            )
        except Exception as e:  # 单 capture 失败不阻断整体
            warn(f"[{i}/{len(specs)}] {spec.capture} 处理失败：{type(e).__name__}: {e}")
            records.append({
                "src": spec.src, "capture": spec.capture, "raw_path": str(spec.path),
                "error": f"{type(e).__name__}: {e}", "n_frames": 0, "skipped": True,
                "attack_type": spec.attack_type, "attack_family": spec.attack_family,
                "supervised_evaluable": spec.supervised_evaluable,
                "is_masquerade": spec.is_masquerade, "protocol": spec.protocol,
                "anomalies": {}, "meta": spec.meta,
            })
            continue
        records.append(rec)
        flag = "↺" if rec.get("_resumed") else ("✗" if rec.get("skipped") else "✓")
        log(
            f"[{i}/{len(specs)}] {flag} {spec.capture} "
            f"frames={rec.get('n_frames', 0):,} atk={rec.get('attack_ratio', 0):.4%} "
            f"t={rec.get('elapsed_sec', 0):.1f}s"
        )

    _write_manifests(dataset, records, paths)
    return {"dataset": dataset, "captures": len(records), "records": records}


# --------------------------------------------------------------------------- #
# manifest 汇总                                                                 #
# --------------------------------------------------------------------------- #
def _write_manifests(dataset: str, records: list[dict[str, Any]], paths: dict[str, Path]) -> None:
    """写 splits.json / dataset_stats.csv / anomalies.csv / env.json。"""
    import csv as _csv

    # ---- splits.json（可复现性凭据） ----
    splits_payload = {
        "_note": (
            "每个 capture 的 train/val/test 行号切分点（左闭右开）。"
            "切分以**区间列表**表达：语义划分下每 split 至多 1 个区间；"
            "块化交错下每 split 有多个（不连续）区间，块内保序、窗口严格块内构造。"
            "禁止随机划分；跨 capture 严禁拼接。"
        ),
        "split_mode": ("capture_holdout_by_family"
                       if (dataset == "road" and C.ROAD_CAPTURE_HOLDOUT) else C.SPLIT_MODE),
        "block_scheme": {
            "K": C.BLOCK_K,
            "pattern": f"block_index % {C.BLOCK_PATTERN_PERIOD} == {C.BLOCK_VAL_MOD} -> val; "
                       f"== {C.BLOCK_TEST_MOD} -> test; else train",
            "align_to_segment_edge": C.ALIGN_TO_SEGMENT_EDGE,
            "segment_search_frac": C.SEGMENT_SEARCH_FRAC,
            "align_widen": {"factor": C.ALIGN_WIDEN_FACTOR, "steps": C.ALIGN_WIDEN_STEPS},
            "min_block_frames": C.MIN_BLOCK_FRAMES,
            "datasets": list(C.BLOCK_MODE_DATASETS),
            "deterministic": True,
            "note": "块化交错不使用随机数；块边界对齐静默区以避免把攻击段劈成两半",
        },
        "semantic_split": {
            "enabled": C.CTT_SEMANTIC_SPLIT,
            "val_quota": C.CTT_VAL_QUOTA,
            "seed": C.SEED,
            "note": "CTT v1.5 按文件名语义划分；val 由 train_01/02 按攻击类型分层留出（唯一使用随机种子的环节）",
        },
        "split_ratios": list(C.SPLIT_RATIOS),
        "stride": {"train": C.STRIDE_TRAIN, "test": C.STRIDE_TEST},
        "window_T": C.WIN_T,
        "captures": {
            r["capture"]: {
                "n_frames": r.get("n_frames", 0),
                "split_mode": (r.get("split_mode") or {}).get("mode", ""),
                "ranges": {sp: (r.get("splits", {}).get(sp, {}) or {}).get("ranges", [])
                           for sp in C.SPLIT_NAMES},
                "summary": {sp: {k: v for k, v in (r.get("splits", {}).get(sp, {}) or {}).items()
                                 if k != "ranges"}
                            for sp in C.SPLIT_NAMES},
            }
            for r in records
        },
    }
    write_json(paths["manifest"] / "splits.json", splits_payload)

    # ---- dataset_stats.csv ----
    cols = [
        "src", "capture", "protocol", "attack_type", "attack_family", "is_masquerade",
        "supervised_evaluable", "raw_rows_total", "n_frames", "n_attack_frames", "attack_ratio",
        "ts_start", "ts_end", "duration_sec", "split_mode", "n_segments_train",
        "n_train", "n_val", "n_test", "atk_train", "atk_val", "atk_test",
        "err", "raw_path",
    ]
    with open(paths["manifest"] / "dataset_stats.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.writer(f)
        w.writerow(cols)
        for r in sorted(records, key=lambda x: (x.get("src", ""), x.get("capture", ""))):
            sp = r.get("splits", {})
            w.writerow([
                r.get("src", ""), r.get("capture", ""), r.get("protocol", ""),
                r.get("attack_type", ""), r.get("attack_family", ""),
                r.get("is_masquerade", ""), r.get("supervised_evaluable", ""),
                r.get("raw_rows_total", ""), r.get("n_frames", ""),
                r.get("n_attack_frames", ""),
                f"{r.get('attack_ratio', 0):.6f}" if "attack_ratio" in r else "",
                r.get("ts_start", ""), r.get("ts_end", ""), r.get("duration_sec", ""),
                (r.get("split_mode") or {}).get("mode", ""),
                sp.get("train", {}).get("n_segments", ""),
                sp.get("train", {}).get("n", ""), sp.get("val", {}).get("n", ""),
                sp.get("test", {}).get("n", ""),
                sp.get("train", {}).get("n_attack", ""), sp.get("val", {}).get("n_attack", ""),
                sp.get("test", {}).get("n_attack", ""),
                r.get("error", ""), r.get("raw_path", ""),
            ])

    # ---- anomalies.csv（缺失值 / 异常值处理计数，论文证据链） ----
    keys = sorted({k for r in records for k in (r.get("anomalies") or {})})
    if keys:
        with open(paths["manifest"] / "anomalies.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = _csv.writer(f)
            w.writerow(["src", "capture", "attack_type"] + keys)
            for r in sorted(records, key=lambda x: (x.get("src", ""), x.get("capture", ""))):
                a = r.get("anomalies") or {}
                w.writerow([r.get("src", ""), r.get("capture", ""), r.get("attack_type", "")]
                           + [a.get(k, 0) for k in keys])

    # ---- env.json（可复现性凭据） ----
    import numpy as _np

    try:
        import pyarrow as _pa
        pa_ver = _pa.__version__
    except Exception:
        pa_ver = "n/a"

    env = {
        "dataset": dataset,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "access_date": "2026-10-17",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _np.__version__,
        "pyarrow": pa_ver,
        "config_fingerprint_sha256_16": config_fingerprint(C),
        "constants": {
            "WIN_T": C.WIN_T, "STRIDE_TRAIN": C.STRIDE_TRAIN, "STRIDE_TEST": C.STRIDE_TEST,
            "PAD_BYTES": C.PAD_BYTES, "V1_CHANNELS": list(C.V1_CHANNELS),
            "SPLIT_MODE": C.SPLIT_MODE,
            "BLOCK_MODE_DATASETS": list(C.BLOCK_MODE_DATASETS),
            "BLOCK_K": C.BLOCK_K, "BLOCK_PATTERN_PERIOD": C.BLOCK_PATTERN_PERIOD,
            "BLOCK_VAL_MOD": C.BLOCK_VAL_MOD, "BLOCK_TEST_MOD": C.BLOCK_TEST_MOD,
            "ALIGN_TO_SEGMENT_EDGE": C.ALIGN_TO_SEGMENT_EDGE,
            "SEGMENT_SEARCH_FRAC": C.SEGMENT_SEARCH_FRAC,
            "ALIGN_WIDEN_FACTOR": C.ALIGN_WIDEN_FACTOR,
            "ALIGN_WIDEN_STEPS": C.ALIGN_WIDEN_STEPS,
            "MIN_BLOCK_FRAMES": C.MIN_BLOCK_FRAMES,
            "PAYLOAD_DELTA_SCOPE": C.PAYLOAD_DELTA_SCOPE,
            "SPLIT_RATIOS": list(C.SPLIT_RATIOS), "DT_CLIP_QUANTILE": C.DT_CLIP_QUANTILE,
            "ROAD_FUZZING_MODE": C.ROAD_FUZZING_MODE, "PROTO_FLAG_MODE": C.PROTO_FLAG_MODE,
            "CTT_TS_BASE": C.CTT_TS_BASE, "SEED": C.SEED,
            "CTT_INCLUDE_EXTRA_SETS": C.CTT_INCLUDE_EXTRA_SETS,
            "CTT_MAIN_SETS": list(C.CTT_MAIN_SETS),
            "CTT_SEMANTIC_SPLIT": C.CTT_SEMANTIC_SPLIT,
            "CTT_VAL_QUOTA": C.CTT_VAL_QUOTA,
        },
        "n_captures": len(records),
        "n_captures_ok": sum(1 for r in records if not r.get("skipped") and "error" not in r),
        "n_frames_total": int(sum(int(r.get("n_frames", 0)) for r in records)),
        "n_attack_frames_total": int(sum(int(r.get("n_attack_frames", 0)) for r in records)),
        "manifest_dir": str(paths["manifest"]),
    }
    write_json(paths["manifest"] / "env.json", env)


# --------------------------------------------------------------------------- #
# 顶层入口                                                                      #
# --------------------------------------------------------------------------- #
def run(
    datasets: list[str],
    *,
    limit: int | None = None,
    force: bool = False,
    only_captures: list[str] | None = None,
) -> dict[str, Any]:
    """跑多个数据集，最后输出整体产出体积统计。"""
    ensure_dir(C.OUT_ROOT)
    summary: dict[str, Any] = {"datasets": {}, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    for ds in datasets:
        t0 = time.time()
        try:
            res = run_dataset(ds, limit=limit, force=force, only_captures=only_captures)
        except Exception as e:
            warn(f"数据集 {ds} 整体失败：{type(e).__name__}: {e}")
            summary["datasets"][ds] = {"error": str(e)}
            continue
        root = dataset_out_root(ds)
        summary["datasets"][ds] = {
            "n_captures": res["captures"],
            "out_dir": str(root),
            "size": human_bytes(dir_size(root)),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        log(f"数据集 {ds} 完成：{res['captures']} capture，产出 {human_bytes(dir_size(root))}，"
            f"耗时 {time.time() - t0:.1f}s")
    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(C.OUT_ROOT / "preprocess_summary.json", summary)
    return summary
