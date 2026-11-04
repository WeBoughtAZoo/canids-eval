# -*- coding: utf-8 -*-
"""
preprocess.parsers —— 三个独立解析器 → 统一中间表示
================================================================================
对应方案 §1（真实文件形态）、§3（中间 schema）、§5.2–§5.4（各数据集专属清单）。

核心约定
--------
· **三个数据集不是一种格式，而是三种格式**，因此有三个独立解析器：
    HCRL CAN-FD : 无表头、变长列的 CSV（第三列是"已解码字节数"，不是 DLC 码）
    ROAD        : candump 文本（帧）+ capture_metadata.json（标签，日志本身无标签）
    CTT v1.5    : 有表头、定长 4 列的 CSV
· 解析器**只负责读到"统一中间表示"**：时间戳原始值、仲裁 ID、变长 hex 载荷、帧级标签。
  **不做填充**（填充留到 features.py 的编码层），**不做跨 capture 拼接**。

所有解析器返回 `Capture` 对象；被剔除行数逐类记入 `counters`。
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import config as C
from .utils import hexstrings_to_uint, log, warn

# --------------------------------------------------------------------------- #
# 数据结构                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class CaptureSpec:
    """capture 的**元信息**（不含帧数据），用于枚举、清单与进度显示。"""
    src: str                       # hcrl_canfd / road / ctt15 / ctt15_hcrl_ch / ctt15_hcrl_sa
    capture: str                   # 全局唯一 capture 名（= 切分与防泄漏边界）
    path: Path                     # 原始文件路径
    protocol: str                  # CANFD / CAN2.0
    attack_type: str               # 攻击类型标签（masquerade 单列）
    attack_family: str             # 攻击族（去 masquerade 后的底层信号/攻击名）
    is_masquerade: bool = False
    supervised_evaluable: bool = True   # False → 不进 AUC-PR/MCC 分层表
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Capture:
    """一个 capture 的**帧级中间表示**（统一 schema 的列式载体）。"""
    spec: CaptureSpec
    ts_raw: np.ndarray             # float64
    aid: np.ndarray                # uint32
    data_hex: list[str]            # 变长 hex 载荷（未填充）
    label: np.ndarray              # uint8
    counters: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.ts_raw.size)


# --------------------------------------------------------------------------- #
# 通用：按路径生成稳定 capture 名                                                #
# --------------------------------------------------------------------------- #
def _cap_name(*parts: str) -> str:
    """把目录层级拼成 capture 名；片段内的空格/横线保留，层级用 '__' 分隔。"""
    return "__".join(str(p) for p in parts if p)


# =========================================================================== #
# 解析器 1：HCRL CAN-FD                                                        #
# =========================================================================== #
_HCRL_FILES: dict[str, str] = {
    "CANFD_Flooding.csv": "Flooding",
    "CANFD_Fuzzing.csv": "Fuzzing",
    "CANFD_Malfunction.csv": "Malfunction",
}


def list_captures_hcrl_canfd() -> list[CaptureSpec]:
    """枚举 HCRL CAN-FD 的三个文件。

    ★ 方案 §5.2：三份 CSV 的时间戳首尾相接（2029→2830→3581），本是同一次连续采集。
      建议**按文件独立切分**（最安全）——本实现即按文件独立成 capture。
    """
    root = C.RAW_HCRL_CANFD
    specs: list[CaptureSpec] = []
    for fname, atype in _HCRL_FILES.items():
        p = root / fname
        if not p.exists():
            warn(f"缺少 HCRL CAN-FD 文件：{p}")
            continue
        specs.append(
            CaptureSpec(
                src="hcrl_canfd",
                capture=fname[:-4],           # CANFD_Flooding 等
                path=p,
                protocol="CANFD",
                attack_type=atype,
                attack_family=atype,
                is_masquerade=False,
                supervised_evaluable=True,
                meta={
                    "note": "无表头变长 CSV；col2=已解码字节数（非 DLC 码）；最长 payload=32B",
                    "missing_fields": ["FDF", "BRS", "ESI"],  # → 通道恒 0，须声明"缺失而非真实 0"
                },
            )
        )
    return specs


def parse_hcrl_canfd(spec: CaptureSpec) -> Capture:
    """逐行解析 HCRL CAN-FD（csv.reader，**不用** pandas 固定列宽）。

    行结构：ts, id_hex(4位), n_bytes, b0, b1, …, b(n-1), label(R/T)
    缺失值处理：
      · 列数 < 5                      → 剔除计数 hcrl_row_too_short
      · len(row) < 3 + n（体被截断）   → 剔除计数 hcrl_body_truncated
      · 数据 token 不是 2 字符 hex     → 规范化后剔除计数 hcrl_token_not_2hex
      · 标签非 R/T                    → 剔除计数 hcrl_label_invalid
    P2 断言：col2 必须等于实际载荷字节数（不一致行数记入 hcrl_len_mismatch）
    """
    counters: dict[str, int] = {}
    ts_list: list[float] = []
    id_list: list[str] = []
    data_list: list[str] = []
    ndecl_list: list[int] = []
    lab_raw: list[int] = []

    with open(spec.path, "r", newline="", encoding="utf-8", errors="replace") as f:
        rd = csv.reader(f)
        for row in rd:
            if not row:
                counters["empty_line"] = counters.get("empty_line", 0) + 1
                continue
            if len(row) < 5:
                counters["hcrl_row_too_short"] = counters.get("hcrl_row_too_short", 0) + 1
                continue
            try:
                n = int(row[2])
            except ValueError:
                counters["hcrl_len_unparsable"] = counters.get("hcrl_len_unparsable", 0) + 1
                continue
            body = row[3 : 3 + n]
            if len(body) != n:
                counters["hcrl_body_truncated"] = counters.get("hcrl_body_truncated", 0) + 1
                continue
            tag = row[-1].strip().upper()
            if tag == "R":
                lab = 0
            elif tag == "T":
                lab = 1
            else:
                counters["hcrl_label_invalid"] = counters.get("hcrl_label_invalid", 0) + 1
                continue
            # token 必须是 2 字符 hex；否则规范化（数量极少）
            hexs = "".join(body)
            if len(hexs) != 2 * n:
                hexs = "".join(f"{int(t, 16):02X}" for t in body)
                counters["hcrl_token_not_2hex"] = counters.get("hcrl_token_not_2hex", 0) + 1
            try:
                ts_list.append(float(row[0]))
            except ValueError:
                counters["hcrl_ts_unparsable"] = counters.get("hcrl_ts_unparsable", 0) + 1
                continue
            id_list.append(row[1])
            data_list.append(hexs)
            ndecl_list.append(n)
            lab_raw.append(lab)

    aid = hexstrings_to_uint(id_list, dtype=np.uint32)
    label = np.asarray(lab_raw, dtype=np.uint8)
    ndecl = np.asarray(ndecl_list, dtype=np.int64)
    actual_len = np.asarray([len(h) for h in data_list], dtype=np.int64) // 2

    # P2 断言：col2 == 实际字节数
    mismatch = int(np.count_nonzero(ndecl != actual_len))
    counters["hcrl_len_mismatch"] = mismatch
    if mismatch:
        warn(f"{spec.capture}: col2 与实际载荷长度不一致 {mismatch} 行（P2 假设被打破）")

    counters["raw_rows_total"] = len(ts_list) + sum(
        v for k, v in counters.items()
        if k in ("empty_line", "hcrl_row_too_short", "hcrl_len_unparsable",
                 "hcrl_body_truncated", "hcrl_label_invalid", "hcrl_ts_unparsable")
    )

    return Capture(
        spec=spec,
        ts_raw=np.asarray(ts_list, dtype=np.float64),
        aid=aid,
        data_hex=data_list,
        label=label,
        counters=counters,
    )


# =========================================================================== #
# 解析器 2：ROAD（candump + JSON 元数据）                                       #
# =========================================================================== #
_ROAD_PAT = re.compile(
    rb"\((\d+\.\d+)\)\s+(\S+)\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)"
)

#: ROAD 攻击族 → 是否 masquerade 由文件名后缀判定；accelerator 无注入 ID 必须排除
_ROAD_EXCLUDED: dict[str, str] = {
    "accelerator_attack_drive_1": "injection_id=None（accelerator 攻击无被篡改 ID，无法逐帧标注）",
    "accelerator_attack_drive_2": "injection_id=None（同上）",
    "accelerator_attack_reverse_1": "injection_id=None（同上）",
    "accelerator_attack_reverse_2": "injection_id=None（同上）",
}


def _road_attack_family(name: str) -> str:
    """由 capture 名推出攻击族（去掉 _masquerade 后缀与末尾 _n）。"""
    base = name
    if base.endswith("_masquerade"):
        base = base[: -len("_masquerade")]
    base = re.sub(r"_\d+$", "", base)
    base = base.replace("_attack", "")
    return base


def _road_holdout_roles(names: list[str], weights: dict[str, float]) -> dict[str, str]:
    """capture 级留出的确定性角色分配（族内贪心均衡，无随机数）。

    ★ 为什么不用"按下标取模"：第一版实测把 test 压到只占 5.6% 帧、且 test 先验
      （1.69%）是 train（0.27%）的 6 倍。此时若结果有差异，无法区分是
      "跨 capture 泛化变差"还是"先验漂移"，敏感性实验就白做了。
      故改为**按帧数（用文件体积作代理）贪心逼近 60/20/20**，把先验差压到最小。

    规则（确定性）：
      ① 组内按 (-体积, 名称) 排序（大文件优先，保证尾部仍可微调）；
      ② 依次把当前 capture 分给"目标量与已分配量差距最大"的 split；
      ③ n = 1：只进 train；n = 2：大的进 train、小的进 test（该族在 test 中 zero-shot）；
      ④ 其余：三个 split 各至少 1 条（用剩余条数做保底约束）。
    """
    names = sorted(names)
    n = len(names)
    if n == 0:
        return {}
    if n == 1:
        return {names[0]: "train"}
    if n == 2:
        a, b = sorted(names, key=lambda x: (-weights.get(x, 0.0), x))
        return {a: "train", b: "test"}

    order = ("train", "val", "test")
    total = float(sum(weights.get(x, 0.0) for x in names)) or 1.0
    target = {"train": 0.60 * total, "val": 0.20 * total, "test": 0.20 * total}
    cur = {s: 0.0 for s in order}
    out: dict[str, str] = {}
    # 保底：每个 split 至少 1 条 → 最后 3 条强制补齐
    forced_from = n - 3
    for i, nm in enumerate(sorted(names, key=lambda x: (-weights.get(x, 0.0), x))):
        w = weights.get(nm, 0.0)
        if i >= forced_from:
            # 尾部三条：优先补尚未被满足的 split
            missing = [s for s in order if s not in set(out.values())]
            pick = missing[0] if missing else min(order, key=lambda s: cur[s] - target[s])
        else:
            pick = max(order, key=lambda s: (target[s] - cur[s], -order.index(s)))
        out[nm] = pick
        cur[pick] += w
    return out


def _assign_road_capture_holdout(specs: list[CaptureSpec]) -> None:
    """把 ROAD 的 capture 按族（ambient 单独一组）整文件分配到 train/val/test。

    ★ 这是 **v6 待确认项 T4 的敏感性实验**：与块化交错的唯一差异是切分粒度，
      其余编码、统计量口径、阈值协议全部不动，从而把差异干净地归因于切分方式。
      用文件体积作帧数代理做组内贪心均衡，使两批的 split 帧占比与先验尽量可比。
    """
    groups: dict[str, list[CaptureSpec]] = {}
    for s in specs:
        # ambient 全部是无攻击的正常流量，单独成组以保证每个 split 都有负样本
        key = "ambient" if s.meta.get("split_role") == "ambient" else f"fam:{s.attack_family}"
        groups.setdefault(key, []).append(s)
    for key, members in groups.items():
        weights = {}
        for m in members:
            try:
                weights[m.capture] = float(m.path.stat().st_size)
            except OSError:                                   # noqa: BLE001
                weights[m.capture] = 0.0
        roles = _road_holdout_roles([m.capture for m in members], weights)
        for m in members:
            m.meta["split_role"] = roles.get(m.capture, "train")
            m.meta["holdout_group"] = key


def list_captures_road() -> list[CaptureSpec]:
    """枚举 ROAD 的 ambient（12）+ attacks（33，其中 4 个 accelerator 被排除）。

    ★ 方案 §5.3 / M6：33 个攻击采集**全部 on_dyno=true（台架）**，须写进 Limitation。
    """
    specs: list[CaptureSpec] = []

    amb_dir = C.RAW_ROAD / "ambient"
    if amb_dir.exists():
        for p in sorted(amb_dir.glob("*.log")):
            specs.append(
                CaptureSpec(
                    src="road", capture=p.stem, path=p, protocol="CAN2.0",
                    attack_type="none", attack_family="none",
                    supervised_evaluable=True,
                    meta={"split_role": "ambient", "label_rule": "all_normal"},
                )
            )

    atk_dir = C.RAW_ROAD / "attacks"
    if atk_dir.exists():
        for p in sorted(atk_dir.glob("*.log")):
            name = p.stem
            if name in _ROAD_EXCLUDED:
                continue  # 4 个 accelerator：无法标注 → 排除（剔除量在 manifest 中报告）
            is_masq = name.endswith("_masquerade")
            fam = _road_attack_family(name)
            specs.append(
                CaptureSpec(
                    src="road", capture=name, path=p, protocol="CAN2.0",
                    attack_type=("masquerade" if is_masq else fam),
                    attack_family=fam,
                    is_masquerade=is_masq,
                    supervised_evaluable=True,
                    meta={"split_role": "attack", "label_rule": "from_capture_metadata"},
                )
            )
    if C.ROAD_CAPTURE_HOLDOUT:
        _assign_road_capture_holdout(specs)
    return specs


def _iter_candump_lines(path: Path, chunk_bytes: int = 1 << 25):
    """分块读取 candump 文本，保证不切断行。避免一次性载入 1.1 GB 大文件。"""
    with open(path, "rb") as f:
        buf = b""
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            buf += chunk
            lines = buf.split(b"\n")
            buf = lines.pop()
            for ln in lines:
                yield ln
        if buf:
            yield buf


def parse_road(spec: CaptureSpec, metadata: dict[str, Any] | None = None) -> Capture:
    """解析 ROAD 的 candump .log 并按 metadata 生成**帧级标签**。

    正则：`\\((\\d+\\.\\d+)\\)\\s+(\\S+)\\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)`
      · `#` 后可能为空（RTR 帧，长度 0）→ dlc=0，计入 missing 计数（**不丢弃**）
      · 时间戳必须相对化（方案 §5.3、陷阱 #6）

    标签生成（方案 §3 的三分支，★ 必须显式声明选了哪种 fuzzing 口径）：
      · 无 metadata 条目 / injection_id 为 None → 该 capture 已被枚举层排除
      · injection_id == "XXX"（3 个 fuzzing，随机 ID）
            mode=interval    → 区间内全部帧标 1（ROAD README 语义，默认，最保守）
            mode=invalid_id  → 仅当 aid 不在合法 ID 集合内标 1
      · 其余（13 组 attack + 13 组 _masquerade）
            label = 1 iff (aid == int(injection_id,16)) and (interval[0] ≤ elapsed ≤ interval[1])
    """
    counters: dict[str, int] = {}
    ts_list: list[bytes] = []
    id_list: list[bytes] = []
    data_list: list[bytes] = []

    for ln in _iter_candump_lines(spec.path):
        if not ln:
            counters["empty_line"] = counters.get("empty_line", 0) + 1
            continue
        m = _ROAD_PAT.search(ln)
        if m is None:
            counters["road_line_unmatched"] = counters.get("road_line_unmatched", 0) + 1
            continue
        ts_list.append(m.group(1))
        id_list.append(m.group(3))          # ★ group(2) 是通道名 canN，group(3) 才是仲裁 ID
        payload = m.group(4)
        if payload in (b"R", b"r"):          # 远程帧标记
            counters["road_rtr_frame"] = counters.get("road_rtr_frame", 0) + 1
            data_list.append(b"")
        elif payload == b"":
            counters["road_empty_payload"] = counters.get("road_empty_payload", 0) + 1
            data_list.append(b"")
        else:
            data_list.append(payload)

    # bytes → str→ numpy 向量化转换
    ts_arr = np.asarray([t.decode("ascii") for t in ts_list], dtype=np.float64)
    aid_arr = hexstrings_to_uint([i.decode("ascii") for i in id_list], dtype=np.uint32)
    data_hex = [d.decode("ascii") for d in data_list]

    # ---------------- 标签生成 ---------------- #
    label = np.zeros(len(ts_arr), dtype=np.uint8)
    md = (metadata or {}).get(spec.capture, {})
    inj_id = md.get("injection_id")
    interval = md.get("injection_interval")

    if spec.meta.get("split_role") == "ambient" or inj_id is None or interval is None:
        counters["road_label_all_normal"] = len(ts_arr)
    else:
        elapsed = ts_arr - ts_arr[0]
        lo, hi = float(interval[0]), float(interval[1])
        in_win = (elapsed >= lo) & (elapsed <= hi)
        if str(inj_id) == "XXX":
            if C.ROAD_FUZZING_MODE == "interval":
                label[in_win] = 1
                counters["road_label_fuzzing_interval_mode"] = int(in_win.sum())
            else:  # invalid_id：仅当 ID 非法时标攻击（合法 ID 集合取本 capture 内多数 ID）
                vals, cnts = np.unique(aid_arr, return_counts=True)
                legit = vals[cnts >= max(5, int(0.001 * len(aid_arr)))]
                label[in_win & ~np.isin(aid_arr, legit)] = 1
                counters["road_label_fuzzing_invalidid_mode"] = int(label.sum())
        else:
            target = int(str(inj_id), 16)
            hit = (aid_arr == target) & in_win
            label[hit] = 1
            counters["road_label_target_frames"] = int(hit.sum())

    counters["raw_rows_total"] = len(ts_arr) + counters.get("road_line_unmatched", 0) + counters.get("empty_line", 0)
    counters["road_lines_unmatched"] = counters.get("road_line_unmatched", 0)

    return Capture(
        spec=spec,
        ts_raw=ts_arr,
        aid=aid_arr,
        data_hex=data_hex,
        label=label,
        counters=counters,
    )


# =========================================================================== #
# 解析器 3：can-train-and-test v1.5                                            #
# =========================================================================== #
_CTT_HEADER = ["timestamp", "arbitration_id", "data_field", "attack"]


def _ctt_attack_type(stem: str) -> tuple[str, str]:
    """由文件名推出 (attack_type, attack_family)。"""
    s = stem
    for suf in ("-train", "-test"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    base = re.sub(r"-\d+$", "", s)
    low = base.lower()
    if low in ("attack-free", "attack_free", "attackfree", "accessory"):
        return "none", "none"
    if low.startswith("masquerade"):
        # masquerade-timed / masquerade-untimed → 必须分开报（方案 §5.4 M1）
        kind = "masquerade_untimed" if "untimed" in low else "masquerade_timed"
        return kind, "masquerade"
    if low.startswith("suppress"):
        return "suppress", "suppress"
    if low.startswith("dos"):
        return "DoS", "DoS"
    if low.startswith("fuzzing"):
        return "Fuzzing", "Fuzzing"
    if low.startswith("gear"):
        return "GearSpoofing", "GearSpoofing"
    if low.startswith("rpm"):
        return "RpmSpoofing", "RpmSpoofing"
    if low.startswith("malfunction"):
        return "Malfunction", "Malfunction"
    return base, base


def list_captures_ctt15() -> list[CaptureSpec]:
    """枚举 CTT v1.5：set_01–04（四象限）+ hcrl-ch + hcrl-sa。

    ★ 方案 §5.4：hcrl-ch / hcrl-sa **不与 set_01–04 混合**（不同车辆、不同时间、真实 Unix 时间戳），
      作为独立 src 处理。
    ★ test_05_suppress 全 0 是**设计使然**（抑制攻击不注入报文）→ supervised_evaluable=False，
      仍照常预处理并落盘，但在清单中标记，**不进入 AUC-PR/MCC 分层表**（M2）。
    """
    specs: list[CaptureSpec] = []
    root = C.RAW_CTT15

    def _walk(set_dir: Path, src: str, set_name: str) -> None:
        for sub in sorted(p for p in set_dir.iterdir() if p.is_dir()):
            sem = C.CTT_SUBSET_SEMANTICS.get(sub.name, sub.name)
            for csvp in sorted(sub.glob("*.csv")):
                atype, afam = _ctt_attack_type(csvp.stem)
                sup = sub.name not in C.CTT_NOT_SUPERVISED
                specs.append(
                    CaptureSpec(
                        src=src,
                        capture=_cap_name(set_name, sub.name, csvp.stem),
                        path=csvp,
                        protocol="CAN2.0",
                        attack_type=atype,
                        attack_family=afam,
                        is_masquerade=afam == "masquerade",
                        supervised_evaluable=sup,
                        meta={
                            "set": set_name,
                            "subset": sub.name,
                            "subset_semantics": sem,
                            # ★ 2026-10-16：按文件名语义划分的角色（train / test）
                            #   仅当 C.CTT_SEMANTIC_SPLIT 为真时生效；
                            #   未知 subset 默认回退为 "test"（保守，避免把测试数据误当训练）。
                            "split_role": (
                                C.CTT_SUBSET_SPLIT_ROLE.get(sub.name, "test")
                                if C.CTT_SEMANTIC_SPLIT else None
                            ),
                            "ts_base": C.CTT_TS_BASE,
                            "vehicles": C.CTT_SET_VEHICLES.get(set_name, {}),
                            "note": (
                                "suppress 攻击删除报文而非注入报文，无逐帧可标注攻击帧；"
                                "标签全 0 为设计使然，不进入 supervised 分层指标（M2）"
                                if not sup else ""
                            ),
                        },
                    )
                )

    for s in C.CTT_MAIN_SETS:
        d = root / s
        if d.exists():
            _walk(d, "ctt15", s)

    # ★ 2026-10-16：默认**不纳入** hcrl-ch / hcrl-sa（config.CTT_INCLUDE_EXTRA_SETS=False）
    #   理由见 config.py 该常量处的四条说明（数据集归属 / 四象限设计 / 时间基与车辆 / 协议）。
    #   置 True 可重新纳入（届时需对二者单独处理 ts_rel 基准，勿套用 CTT_TS_BASE）。
    if C.CTT_INCLUDE_EXTRA_SETS:
        if (root / "hcrl-ch").exists():
            _walk(root / "hcrl-ch", "ctt15_hcrl_ch", "hcrl-ch")
        if (root / "hcrl-sa").exists():
            _walk(root / "hcrl-sa", "ctt15_hcrl_sa", "hcrl-sa")
    return specs


def parse_ctt15(spec: CaptureSpec) -> Capture:
    """解析 CTT v1.5 单个 CSV（有表头，定长 4 列）。

    缺失值处理：
      · 表头不符                      → 抛错（数据异常需人工介入）
      · data_field 为空串             → dlc=0（计数）
      · data_field 含非 hex 字符      → 剔除计数 ctt_data_nonhex
      · attack 非 0/1                 → 剔除计数 ctt_label_invalid
      · *严格逐文件处理，严禁跨文件拼接*（每个 CSV 都从 1672531200.0 起）
    """
    counters: dict[str, int] = {}
    ts_list: list[float] = []
    id_list: list[str] = []
    data_list: list[str] = []
    lab_raw: list[int] = []

    with open(spec.path, "r", newline="", encoding="utf-8", errors="replace") as f:
        rd = csv.reader(f)
        try:
            header = [h.strip().lower() for h in next(rd)]
        except StopIteration:
            raise ValueError(f"{spec.path} 为空文件")
        if header[:4] != _CTT_HEADER:
            raise ValueError(f"{spec.path} 表头异常：{header}（期望 {_CTT_HEADER}）")

        for row in rd:
            if not row:
                counters["empty_line"] = counters.get("empty_line", 0) + 1
                continue
            if len(row) < 4:
                counters["ctt_row_too_short"] = counters.get("ctt_row_too_short", 0) + 1
                continue
            a = row[3].strip()
            if a not in ("0", "1"):
                counters["ctt_label_invalid"] = counters.get("ctt_label_invalid", 0) + 1
                continue
            d = row[2].strip()
            if d and not _HEX_ONLY.fullmatch(d):
                counters["ctt_data_nonhex"] = counters.get("ctt_data_nonhex", 0) + 1
                continue
            try:
                ts_list.append(float(row[0]))
            except ValueError:
                counters["ctt_ts_unparsable"] = counters.get("ctt_ts_unparsable", 0) + 1
                continue
            id_list.append(row[1].strip())
            data_list.append(d)
            lab_raw.append(int(a))

    aid = hexstrings_to_uint(id_list, dtype=np.uint32)
    counters["raw_rows_total"] = len(ts_list) + sum(
        v for k, v in counters.items()
        if k in ("empty_line", "ctt_row_too_short", "ctt_label_invalid",
                 "ctt_data_nonhex", "ctt_ts_unparsable")
    )

    return Capture(
        spec=spec,
        ts_raw=np.asarray(ts_list, dtype=np.float64),
        aid=aid,
        data_hex=data_list,
        label=np.asarray(lab_raw, dtype=np.uint8),
        counters=counters,
    )


_HEX_ONLY = re.compile(r"[0-9A-Fa-f]+")


# --------------------------------------------------------------------------- #
# 统一入口                                                                      #
# --------------------------------------------------------------------------- #
DATASET_LISTER = {
    "hcrl_canfd": list_captures_hcrl_canfd,
    "road": list_captures_road,
    "ctt15": list_captures_ctt15,
}

DATASET_PARSER = {
    "hcrl_canfd": parse_hcrl_canfd,
    "road": parse_road,
    "ctt15": parse_ctt15,
}


def list_captures(dataset: str) -> list[CaptureSpec]:
    if dataset not in DATASET_LISTER:
        raise KeyError(f"未知数据集：{dataset}（可选 {sorted(DATASET_LISTER)}）")
    return DATASET_LISTER[dataset]()


def parse_capture(spec: CaptureSpec, **kwargs) -> Capture:
    """按 spec.src 前缀派发到对应解析器（ctt15_hcrl_ch / ctt15_hcrl_sa 也走 CTT 解析器）。"""
    if spec.src == "hcrl_canfd":
        return parse_hcrl_canfd(spec)
    if spec.src == "road":
        return parse_road(spec, **kwargs)
    if spec.src.startswith("ctt15"):
        return parse_ctt15(spec)
    raise KeyError(f"未知 src：{spec.src}")
