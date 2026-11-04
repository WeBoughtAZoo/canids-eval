#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
smoke_test.py —— 预处理代码的单元校验 / 冒烟测试
================================================================================
对应方案 §9 的"校验判据"与 §8 陷阱清单的自查项。全部为**快速检查**，
只读取少量行，不写全量产物。

运行：
    python 代码/tests/smoke_test.py

覆盖的检查项
------------
T1  hex 向量化解析：大小写、0x 前缀、不等宽（含 0xFFFE 不得被截首字符）
T2  HCRL CAN-FD  ：len(data) == col2（方案 P2 断言）、dlc 值域 ⊂ {8,16,24,32}
T3  ROAD         ：candump 正则、RTR 空载荷容错、标签由 metadata 三分支生成
T4  CTT v1.5     ：表头校验、标签 ∈ {0,1}、data_field hex 长度容错
T5  编码层       ：掩码在 dlc 处严格切换；>0x7FF 的 ID → id_valid=0 且 id_norm ≤ 1
T6  窗口化       ：test stride=64 时窗口零重叠；窗内任一帧攻击 → 窗口标攻击
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # 代码/
sys.path.insert(0, str(ROOT))

from preprocess import cleaning, config as C, features, parsers, splitting  # noqa: E402
from preprocess.utils import hexstrings_to_padded, hexstrings_to_uint        # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


# --------------------------------------------------------------------------- #
def t1_hex_utils() -> None:
    print("\nT1 hex 向量化解析")
    a = hexstrings_to_uint(["1E5", "0f1", "0xFFE", "0xFFFE"])
    check("大小写/0x 前缀/不等宽", list(a) == [0x1E5, 0x0F1, 0xFFE, 0xFFFE], f"got={list(a)}")
    check("0xFFFE 不被截首字符（陷阱 #3）", int(a[3]) == 0xFFFE, f"0xFFFE → {int(a[3])}")

    buf, dlc = hexstrings_to_padded(["ABCD", "", "0F1", "00" * 40], max_bytes=32)
    check("偶数长度右侧补零", bytes(buf[0, :2]) == bytes.fromhex("ABCD") and dlc[0] == 2)
    check("空载荷 → dlc=0", dlc[1] == 0)
    check("奇数字符左补 0", dlc[2] == 2 and bytes(buf[2, :2]) == bytes.fromhex("00F1"),
          f"{bytes(buf[2,:2]).hex()}")
    check("超长载荷截断到 32B", dlc[3] == 32)


# --------------------------------------------------------------------------- #
def t2_hcrl_canfd() -> None:
    print("\nT2 HCRL CAN-FD 解析")
    path = C.RAW_HCRL_CANFD / "CANFD_Malfunction.csv"
    if not path.exists():
        print("  [SKIP] 缺少文件，跳过"); return
    # 只读前 20000 行构造一个临时 spec，验字段语义
    import csv as _csv

    rows = []
    with open(path, "r", newline="") as f:
        rd = _csv.reader(f)
        for i, r in enumerate(rd):
            rows.append(r)
            if i >= 20000:
                break
    ok_len = all(len(r) >= 5 and int(r[2]) == len(r[3:3 + int(r[2])]) for r in rows)
    check("len(data) == col2（方案 P2）", ok_len)
    lens = sorted({int(r[2]) for r in rows})
    check("col2 值域 ⊆ {8,16,24,32}", set(lens) <= {8, 16, 24, 32}, f"lens={lens}")
    check("标签 ∈ {R,T}", all(r[-1] in ("R", "T") for r in rows))
    ids = hexstrings_to_uint([r[1] for r in rows])
    check("ID 恒 4 位 hex 可解析", int(ids.max()) <= 0xFFFF, f"max=0x{int(ids.max()):X}")


# --------------------------------------------------------------------------- #
def t3_road() -> None:
    print("\nT3 ROAD 解析与标签生成")
    meta_path = C.RAW_ROAD / "attacks" / "capture_metadata.json"
    if not meta_path.exists():
        print("  [SKIP] 缺少文件，跳过"); return
    import json
    meta = json.load(open(meta_path, encoding="utf-8"))

    # (a) 4 个 accelerator 必须无注入 ID（会被枚举层排除）
    acc = [k for k, v in meta.items() if k.startswith("accelerator")]
    check("accelerator 共 4 个且 injection_id=None",
          len(acc) == 4 and all(meta[k]["injection_id"] is None for k in acc))
    # (b) 3 个 fuzzing injection_id == "XXX"（不可 int()）
    fz = [k for k, v in meta.items() if str(v.get("injection_id")) == "XXX"]
    check("fuzzing 共 3 个且 injection_id='XXX'", len(fz) == 3)
    # (c) 可标注项 = 26（13 attack + 13 masquerade）
    usable = [k for k, v in meta.items()
              if v.get("injection_id") not in (None, "XXX") and v.get("injection_interval")]
    check("可精确标注 capture = 26", len(usable) == 26, f"got={len(usable)}")
    check("全部 on_dyno=true（M6）", all(v.get("on_dyno") for v in meta.values()))

    # (d) 实际解析一个 masquerade 采集，验证标签确实被置 1
    name = "correlated_signal_attack_1_masquerade"
    p = C.RAW_ROAD / "attacks" / f"{name}.log"
    if p.exists():
        spec = parsers.CaptureSpec(
            src="road", capture=name, path=p, protocol="CAN2.0",
            attack_type="masquerade", attack_family="correlated_signal",
            is_masquerade=True, meta={"split_role": "attack"},
        )
        cap = parsers.parse_road(spec, metadata=meta)
        check("解析出帧且标签含正类", len(cap) > 0 and cap.label.sum() > 0,
              f"frames={len(cap)} atk={int(cap.label.sum())}")
        # 时间戳必须相对化（基准为合成值 ~1.1e9）
        check("时间戳基准为合成大值（须相对化）", float(cap.ts_raw[0]) > 1e8,
              f"t0={float(cap.ts_raw[0]):.3f}")


# --------------------------------------------------------------------------- #
def t4_ctt15() -> None:
    print("\nT4 CTT v1.5 解析")
    p = C.RAW_CTT15 / "set_01" / "test_06_masquerade" / "masquerade-timed-1.csv"
    if not p.exists():
        print("  [SKIP] 缺少文件，跳过"); return
    spec = parsers.CaptureSpec(
        src="ctt15", capture="t", path=p, protocol="CAN2.0",
        attack_type="masquerade_timed", attack_family="masquerade",
        is_masquerade=True, meta={},
    )
    cap = parsers.parse_ctt15(spec)
    check("表头解析成功且帧数 > 0", len(cap) > 0, f"frames={len(cap)}")
    check("标签 ∈ {0,1}", set(np.unique(cap.label).tolist()) <= {0, 1})
    check("时间戳自 1672531200.0 起（各文件独立）",
          abs(float(cap.ts_raw[0]) - C.CTT_TS_BASE) < 1e-3, f"t0={float(cap.ts_raw[0])}")
    check("masquerade 有正类标签（M1）", cap.label.sum() > 0, f"atk={int(cap.label.sum())}")

    # suppress 全 0 是设计使然 → 必须标 supervised_evaluable=False
    sup_dir = C.RAW_CTT15 / "set_01" / "test_05_suppress"
    if sup_dir.exists():
        specs = [s for s in parsers.list_captures_ctt15()
                 if s.meta.get("subset") == "test_05_suppress"]
        check("test_05_suppress 被标记为不可监督评测", specs and not specs[0].supervised_evaluable)


# --------------------------------------------------------------------------- #
def t5_features() -> None:
    print("\nT5 编码层：掩码 / ID 溢出 / 归一化")
    ts = np.array([0.0, 0.001, 0.002, 0.003, 0.004], dtype=np.float64)
    aid = np.array([0x100, 0x100, 0xFFFE, 0x100, 0xFFFE], dtype=np.uint32)
    data = ["AABB", "AABBCCDD", "00", "AABB", "FF"]
    label = np.array([0, 0, 1, 0, 1], dtype=np.uint8)
    dt = cleaning.compute_dt(ts)
    stats = features.fit_train_stats(aid, dt, aid_size=0x10000)
    counters: dict[str, int] = {}
    v1, v2, aux = features.encode_capture(
        {"ts_rel": ts, "aid": aid, "data_hex": data, "label": label},
        stats, proto_flag_value=0, counters=counters,
    )
    dlc = aux["dlc"]
    # 掩码在 dlc 处严格切换
    mask_sum = v2[:, 4:, :].sum(axis=(1, 2))
    check("掩码通过字节数 == dlc", np.array_equal(mask_sum, dlc.astype(mask_sum.dtype)),
          f"mask={mask_sum.tolist()} dlc={dlc.tolist()}")
    check("mask[i]=1 iff i<dlc（严格切换）",
          bool(v2[1, 4, 3] == 1 and v2[1, 4, 4] == 0))
    # ID 溢出
    check("aid>0x7FF → id_valid=0", aux["id_valid"][2] == 0 and aux["id_valid"][0] == 1)
    check("id_norm 钳位后 ≤ 1", float(v1[:, 2].astype(np.float32).max()) <= 1.0,
          f"max={float(v1[:,2].astype(np.float32).max()):.4f}")
    check("无 NaN/Inf", bool(np.isfinite(v1.astype(np.float32)).all()))
    check("pad_ratio = (32-dlc)/32", abs(float(v1[1, 5]) - (32 - 4) / 32) < 1e-3)
    check("payload_len_norm 分母为 32（陷阱 #2）", abs(float(v1[1, 4]) - 4 / 32) < 1e-3)

    # --- 退化场景：大量"只出现一次"的 ID + 极小 Δt → dt_dev 分母趋零，曾导致 float16 溢出 ---
    m = 4096
    ts2 = np.arange(m, dtype=np.float64) * 1e-6          # Δt = 1e-6 s（远小于 1 ms）
    aid2 = (np.arange(m, dtype=np.uint32) + 0x100)       # 每个 ID 仅出现一次
    data2 = ["00"] * m
    lab2 = np.zeros(m, dtype=np.uint8)
    st2 = features.fit_train_stats(aid2, cleaning.compute_dt(ts2), aid_size=0x2000)
    v12, _, _ = features.encode_capture(
        {"ts_rel": ts2, "aid": aid2, "data_hex": data2, "label": lab2},
        st2, proto_flag_value=0, counters={},
    )
    f12 = v12.astype(np.float32)
    check("退化周期场景下 V1 全通道有限（无 float16 溢出）", bool(np.isfinite(f12).all()),
          f"period_floor={st2.period_floor:.2e} dt_dev_max={float(f12[:,1].max()):.1f}")


# --------------------------------------------------------------------------- #
def t6_windowing() -> None:
    print("\nT6 窗口化：零重叠与窗口标注")
    n = 600
    splits = splitting.make_splits(n)
    (s, e), = splits["test"]                      # 区间列表：连续模式仅 1 个区间
    nw = 1 + (e - s - C.WIN_T) // C.STRIDE_TEST
    starts = np.arange(0, e - s - C.WIN_T + 1, C.STRIDE_TEST)
    check("test stride=W → 窗口零重叠", len(starts) == nw and C.STRIDE_TEST == C.WIN_T)

    # 窗内任一帧为攻击 → 窗口标攻击
    lab = np.zeros(n, dtype=np.uint8)
    lab[s + 5] = 1                                 # 落在 test 段第一个窗口内
    seg = lab[s:e]
    c = np.concatenate([[0], np.cumsum(seg, dtype=np.int64)])
    wlab = (c[starts + C.WIN_T] - c[starts] > 0).astype(np.uint8)
    check("窗内任一帧攻击 → 窗口标攻击", int(wlab.sum()) == 1 and wlab[0] == 1)
    # 连续模式下三段严格衔接
    tr, va, te = splits["train"][0], splits["val"][0], splits["test"][0]
    check("连续模式：切分点衔接且覆盖全体",
          tr[1] == va[0] and va[1] == te[0] and te[1] == n)


# --------------------------------------------------------------------------- #
def t7_block_split() -> None:
    """★ 块化交错切分（2026-10-17 新增）：比例、分区合法性、段边界对齐、窗口零重叠。"""
    print("\nT7 块化交错切分")
    n = 10_000

    # 构造一条"攻击局部化但并非全集中在首部"的时间线（贴近 HCRL 的实际形态：
    # 攻击段散布在时间线各处、单个攻击段远窄于搜索窗口）
    lab = np.zeros(n, dtype=np.uint8)
    for a, b in ((400, 700), (1500, 1800), (2600, 2900), (3700, 4000),
                 (4800, 5100), (5900, 6200), (7100, 7400), (8300, 8600)):
        lab[a:b] = 1

    splits, info = splitting.make_block_splits(n, lab)
    splitting.assert_partition(splits, n)          # 不抛异常即通过
    check("块化交错：构成合法分区（无空洞/无重叠）", True,
          f"k={info['k']} 边界对齐 {info['n_aligned']}/{max(1, info['k']-1)}"
          f" 未对齐 {info['n_unaligned']}")

    n_tr = sum(b - a for a, b in splits["train"])
    n_va = sum(b - a for a, b in splits["val"])
    n_te = sum(b - a for a, b in splits["test"])
    check("块化交错：比例精确 6:2:2",
          abs(n_tr / n - 0.6) < 0.02 and abs(n_va / n - 0.2) < 0.02 and abs(n_te / n - 0.2) < 0.02,
          f"{n_tr/n:.3f} / {n_va/n:.3f} / {n_te/n:.3f}")

    check("块化交错：三段区间数均为 6/2/2 块",
          len(splits["train"]) == 6 and len(splits["val"]) == 2 and len(splits["test"]) == 2)

    # 段边界对齐：每个已对齐的边界必须落在标签跳变点上
    bounds = info["block_bounds"][1:-1]
    ok = all((lab[b - 1] != lab[b]) for b, f in zip(bounds, info["aligned_flags"]) if f)
    check("块边界落在标签跳变点上", ok)

    # 机制保证：不存在被块边界劈开的攻击段
    split_points = [b for _a, b in splits["train"] + splits["val"] + splits["test"] if b < n]
    n_cut = sum(1 for b in split_points if lab[b - 1] == 1 and lab[b] == 1)
    check("无攻击段被块边界劈成两半", n_cut == 0,
          f"实现自检 n_attack_segments_cut={info['n_attack_segments_cut']}")

    # 每个 split 都有攻击正窗口（块化交错的核心收益）
    def pos_windows(rngs, stride):
        c = np.concatenate([[0], np.cumsum(lab.astype(np.int64))])
        tot = 0
        for a, b in rngs:
            if b - a < C.WIN_T:
                continue
            st = np.arange(a, b - C.WIN_T + 1, stride)
            tot += int(np.count_nonzero(c[st + C.WIN_T] - c[st] > 0))
        return tot

    p_tr = pos_windows(splits["train"], C.STRIDE_TRAIN)
    p_va = pos_windows(splits["val"], C.STRIDE_TEST)
    p_te = pos_windows(splits["test"], C.STRIDE_TEST)
    check("三个 split 均含攻击正窗口（正样本不再塌陷）",
          p_tr > 0 and p_va > 0 and p_te > 0, f"train/val/test 正窗口={p_tr}/{p_va}/{p_te}")

    # 块内窗口零重叠（测试段 stride = W）
    for a, b in splits["test"]:
        if b - a >= C.WIN_T:
            st = np.arange(a, b - C.WIN_T + 1, C.STRIDE_TEST)
            starts_rel = st - a
            check("块内窗口零重叠（stride=W）",
                  len(set(starts_rel.tolist())) == len(starts_rel)
                  and all(starts_rel[i + 1] - starts_rel[i] == C.WIN_T
                          for i in range(len(starts_rel) - 1)))
            break

    # 极端短 capture 必须安全退化
    sp2, info2 = splitting.make_block_splits(300, np.zeros(300, dtype=np.uint8))
    splitting.assert_partition(sp2, 300)
    check("极短 capture 安全退化（fallback 或合法分区）",
          info2.get("fallback", False) or info2["k"] > 0)

    # 攻击全部集中在时间线前部时，块化交错仍可能使 test 无正样本 —— 这是数据属性而非实现缺陷，
    # 此处显式固化该认知（对应 ROAD fuzzing 的真实情形）。
    lab3 = np.zeros(n, dtype=np.uint8)
    lab3[400:700] = 1
    sp3, info3 = splitting.make_block_splits(n, lab3)
    splitting.assert_partition(sp3, n)
    p_te3 = sum(int(np.count_nonzero(
        np.concatenate([[0], np.cumsum(lab3.astype(np.int64))])[np.arange(a, b - C.WIN_T + 1, C.STRIDE_TEST) + C.WIN_T]
        - np.concatenate([[0], np.cumsum(lab3.astype(np.int64))])[np.arange(a, b - C.WIN_T + 1, C.STRIDE_TEST)] > 0))
        for a, b in sp3["test"] if b - a >= C.WIN_T)
    check("攻击仅集中在前 10% 时 test 可为 0（记录为已知数据属性）", True,
          f"该合成样例 test 正窗口={p_te3}")


# --------------------------------------------------------------------------- #
def t8_payload_delta_block_local() -> None:
    """★ P1 修复验证（2026-10-17）：payload_delta 必须逐块计算，杜绝跨块/跨 split 引用。"""
    print("\nT8 payload_delta 逐块计算（P1）")
    # 构造一个只有跨块依赖才暴露问题的样本：
    #   块 A（帧 0–1）：aid=0x100，载荷 00 00
    #   块 B（帧 2）  ：aid=0x100，载荷 FF    ← 全局 prev 指向块 A 的最后一帧 → delta=1.0
    ts = np.array([0.0, 0.001, 0.002], dtype=np.float64)
    aid = np.array([0x100, 0x100, 0x100], dtype=np.uint32)
    data = ["00", "00", "FF"]
    lab = np.zeros(3, dtype=np.uint8)
    stats = features.fit_train_stats(aid, cleaning.compute_dt(ts), aid_size=0x800)
    blocks = [(0, 2), (2, 3)]

    v1_blk, _, _ = features.encode_capture(
        {"ts_rel": ts, "aid": aid, "data_hex": data, "label": lab},
        stats, proto_flag_value=0, counters={}, blocks=blocks,
    )
    v1_glb, _, _ = features.encode_capture(
        {"ts_rel": ts, "aid": aid, "data_hex": data, "label": lab},
        stats, proto_flag_value=0, counters={}, blocks=None,
    )
    d_blk = float(v1_blk[2, 13]); d_glb = float(v1_glb[2, 13])
    check("逐块计算：块内首个出现该 ID 的帧 payload_delta = 0", abs(d_blk) < 1e-6, f"got={d_blk:.4f}")
    # 全局口径下该帧会引用块 A 的最后一帧：仅 1 个字节不同 → 1/32 = 0.03125
    check("全局计算（对照）仍引用上一块（= 1/32 ≈ 0.03125）",
          abs(d_glb - 1.0 / 32.0) < 1e-4, f"got={d_glb:.5f}")
    check("块内非首帧的差分仍被正确保留（帧1 相对帧0 = 0）", abs(float(v1_blk[1, 13])) < 1e-6)


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 74)
    print("预处理代码 冒烟测试 / 单元校验")
    print(f"项目根：{C.PROJECT_ROOT}")
    print("=" * 74)
    t1_hex_utils()
    t2_hcrl_canfd()
    t3_road()
    t4_ctt15()
    t5_features()
    t6_windowing()
    t7_block_split()
    t8_payload_delta_block_local()
    print("\n" + "=" * 74)
    print(f"结果：PASS={PASS}  FAIL={FAIL}")
    print("=" * 74)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
