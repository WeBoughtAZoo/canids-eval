# -*- coding: utf-8 -*-
"""
preprocess.config —— 全局配置与常量
================================================================================
本文件集中定义路径、张量规格、切分比例、清洗阈值等，全部口径来自：
  《三数据集形态解析与预处理方案.html》（2026-10-16 本机实测核实版）

设计原则
--------
1. 一切口径"单点定义"，下游代码只引用常量，避免同一参数在多个文件里各写一份。
2. 常量带注释标明出处（方案 §x）与"为什么是这个值"，便于论文写作时回溯。
3. 不在此文件做任何 I/O；路径存在性由 pipeline 运行时校验。

作者：小巴（WorkBuddy）    日期：2026-10-16
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# 0. 目录约定                                                                  #
# --------------------------------------------------------------------------- #
# 代码位于 <项目根>/代码/preprocess/config.py，因此 parents[2] 即项目根
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: 原始数据集根目录（只读，绝不写入）
DATA_ROOT: Path = PROJECT_ROOT / "数据集"

#: 预处理产物根目录（用户要求：按数据集分别保存至"训练数据集"文件夹）
OUT_ROOT: Path = PROJECT_ROOT / "训练数据集"

#: 三个数据集的原始子目录名
RAW_HCRL_CANFD: Path = DATA_ROOT / "CAN-FD_Intrusion_Dataset"
RAW_ROAD: Path = DATA_ROOT / "road"
RAW_CTT15: Path = DATA_ROOT / "can-train-and-test-v1.5"

# --------------------------------------------------------------------------- #
# 1. 张量与时间窗规格（方案 §4）                                                #
# --------------------------------------------------------------------------- #
#: 时间窗长度（帧）
WIN_T: int = 64
#: 训练集滑动步长（帧）—— 半重叠，扩大训练样本
STRIDE_TRAIN: int = 32
#: 测试集步长 = 窗长 —— 窗口零重叠，保证 McNemar 检验的样本独立性（方案 §4）
STRIDE_TEST: int = 64
#: 载荷填充目标字节数。★ 实测三数据集最长 payload = 32 字节（非 64），
#: 因此 payload_len_norm / pad_ratio 的分母一律为 32（方案 §5.2、陷阱 #2）
PAD_BYTES: int = 32
#: 字节图通道数 = 4 个数据通道 + 4 个逐字节填充掩码通道（方案 §4 V2）
IMG_CHANNELS: int = 8
#: 每个字节通道的字节数 = 32 / 4
IMG_BYTE_PER_CH: int = PAD_BYTES // 4  # = 8

#: V1 帧向量通道名（顺序即张量第 2 维下标；方案 §4 表格逐项对应）
#: 注意：id_valid 为"辅助通道"，不并入 V1 的 16 维（见 aux 数组），
#: 理由是 16 维口径已在《最终研究方案与SCI论文大纲-v3定稿》冻结，
#: 而 Fuzzing 的非法 ID 强信号通过 aux/idvalid.npy 单独承载，二者不冲突。
V1_CHANNELS: tuple[str, ...] = (
    "dt_norm",          # 0  z-score(log1p(Δt))
    "dt_dev",           # 1  周期偏差（相对该 ID 的中位周期）
    "id_norm",          # 2  min(aid,2047)/2047
    "id_freq",          # 3  log1p(训练集内该 ID 出现频次)
    "payload_len_norm", # 4  dlc / 32
    "pad_ratio",        # 5  (32 - dlc) / 32  ★ Ab-1 消融对象
    "proto_flag",       # 6  协议域标识（主实验置 0）
    "brs_flag",         # 7  HCRL 缺失 → 恒 0（须声明为"缺失而非真实 0"）
    "esi_flag",         # 8  HCRL 缺失 → 恒 0（同上）
    "byte_mean",        # 9  mean(padded[0:32])
    "byte_std",         # 10 std(padded[0:32])
    "byte_entropy",     # 11 香农熵 / log2(32)，归一化到 [0,1]
    "nonzero_ratio",    # 12 count(padded != 0) / 32
    "payload_delta",    # 13 mean(|padded - 同 ID 前一帧|) / 255
    "payload_u32",      # 14 大端前 4 字节 / 2^32
    "period_stability", # 15 该 ID 周期变异系数 std/mean（训练集估计）
)

# --------------------------------------------------------------------------- #
# 2. 切分规格（方案 §5 G4，2026-10-17 修订）                                    #
# --------------------------------------------------------------------------- #
#: 时间序连续切分比例 train / val / test（仅用于 "per_capture_ratio" 模式）
SPLIT_RATIOS: tuple[float, float, float] = (0.60, 0.20, 0.20)
#: 切分名（顺序即时间先后，禁止乱序）
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

# --------------------------------------------------------------------------- #
# 2b. ★ 切分模式（2026-10-17 修订：引入块化交错切分）                            #
# --------------------------------------------------------------------------- #
#: 切分模式总开关：
#:   "block_interleaved"  —— 块化交错：单条时间线切 K 个连续块，按固定模模式分配；
#:                           块内保序、窗口严格块内构造（无跨块窗口 → 零重叠、零泄漏）
#:   "per_capture_ratio"  —— 传统：capture 内按行号连续 60/20/20
#:
#: ★ 为什么改：capture 内连续切分下，攻击注入在时间轴上高度局部化，
#:   HCRL 测试段仅 331 个正窗口（Fuzzing / Malfunction 为 0），直接使
#:   "按攻击类型分层报告" 这条红线在 CAN FD 侧不可执行，并威胁 O1 的证据链。
#:   块化交错在**不引入随机、不引入跨切分单元泄漏**的前提下解决该问题。
SPLIT_MODE: str = "block_interleaved"

#: 启用块化交错的数据集（src）。CTT v1.5 因官方四象限设计必须用语义划分，不在其中。
BLOCK_MODE_DATASETS: tuple[str, ...] = ("hcrl_canfd", "road")

#: ★ 2026-10-28 敏感性实验开关：**ROAD capture 级留出**（默认 False，不影响既有产物）。
#:   开启后 ROAD 的每条 capture 整文件归入一个 split，替代块化交错。
#:   用途：回答"块化交错是否高估了 ROAD 上的检测能力"（v6 待确认项 T4）。
#:   切换方式：环境变量 CANIDS_ROAD_CAPTURE_HOLDOUT=1（或 run_preprocess.py --road-capture-holdout）。
#:   产物必须写到**独立目录**（run_preprocess.py --out），不得覆盖 训练数据集/ROAD。
import os as _os
ROAD_CAPTURE_HOLDOUT: bool = _os.environ.get("CANIDS_ROAD_CAPTURE_HOLDOUT", "0") == "1"

#: capture 级留出的分配规则（确定性，不使用随机数）：
#:   ① 先在**族内**按 capture 名排序（ambient 单独一组）；
#:   ② n ≥ 5：沿用与块化交错相同的模模式 i%5 ∈ {0,1,3}→train、2→val、4→test；
#:   ③ n = 3 或 4：train = n−2，其余依次为 val、test（保证 test 至少 1 条）；
#:   ④ n = 2：train 1、test 1（该族在 test 中为 zero-shot，须报告）；
#:   ⑤ n = 1：只进 train（该族在 test 中缺席，须报告）。
#: 该规则保证 ambient 一定进入三个 split（否则 test 无负样本）。
ROAD_HOLDOUT_TRAIN_MODS: tuple[int, ...] = (0, 1, 3)
ROAD_HOLDOUT_VAL_MOD: int = 2
ROAD_HOLDOUT_TEST_MOD: int = 4

#: 块数。必须为 BLOCK_PATTERN_PERIOD 的整数倍，以保证比例精确为 6:2:2。
#: 取 10 的依据：实测已使 HCRL 三类攻击全部进入测试集；K=25/50 收益递减且边界损耗上升。
BLOCK_K: int = 10
#: 分配模式周期与取模值：i % PERIOD == VAL_MOD → val；== TEST_MOD → test；其余 → train
BLOCK_PATTERN_PERIOD: int = 5
BLOCK_VAL_MOD: int = 2
BLOCK_TEST_MOD: int = 4

#: ★ 块边界对齐"攻击段边界"。落实《五套方案》原要求——
#: "对 HCRL CAN-FD，按'攻击段—静默段'整段切分，确保测试集中包含完整的攻击段而非片段残留"。
#: 做法：把每个候选切点微调到搜索窗口内**最近的一个标签跳变点**（label[t-1] != label[t]）。
#: 切在跳变点上 ⇒ 任何攻击段都完整落在边界的一侧，不会被劈成两半；
#: 这比"对齐到静默区中心"更强（静默区中心只能保证边界本身不在攻击中，
#: 而跳变点保证两侧的段都完整）。
ALIGN_TO_SEGMENT_EDGE: bool = True
#: 切点搜索窗口 = 块长 × 该比例（过大将破坏块大小均衡）
SEGMENT_SEARCH_FRAC: float = 0.10
#: 若首次搜索窗口内没有跳变点，按此倍数逐级放宽（最多 ALIGN_WIDEN_STEPS 级）。
#: 用于处理"攻击段比搜索窗口还宽"的情况。
ALIGN_WIDEN_FACTOR: float = 2.0
ALIGN_WIDEN_STEPS: int = 3
#: ★ 比例守卫：对齐块边界后，若 train 占比相对 0.60 的偏差超过该容差，
#: 则按 0.5 倍逐步收紧搜索窗口重试；仍不满足则退回等分边界。
#: 作用：保证"对齐段边界"永远不得破坏 6:2:2 这一硬约束。
BLOCK_RATIO_TOL: float = 0.015
#: 比例守卫的最大收紧次数
BLOCK_RATIO_REPAIR_STEPS: int = 4
#: 每块最小帧数（低于则缩减 K；若 K=5 仍不满足则退化为 per_capture_ratio 模式）
MIN_BLOCK_FRAMES: int = 128


# --------------------------------------------------------------------------- #
# 3. 清洗与鲁棒阈值（方案 §5 G1–G5、§8 陷阱清单）                               #
# --------------------------------------------------------------------------- #
#: Δt 上裁剪分位数。Δt 用 train split 的该分位数做鲁棒裁剪（避免长停顿/掉线帧
#: 把 z-score 尺度拉爆）。方案未指定具体值，取 P99.9 为"去极端但保尾部"的折中。
DT_CLIP_QUANTILE: float = 0.999
#: 数值稳定项
EPS: float = 1e-8

# --------------------------------------------------------------------------- #
# 3b. NaN / Inf 防御阈值（float16 可表示上限约 65504）                           #
#: 周期分母下限的派生系数：floor = max(PERIOD_FLOOR_ABS, PERIOD_FLOOR_REL × 典型周期)
#: 说明：周期偏差 dt_dev = (Δt − med_period)/median_period 在"该 ID 在训练集中仅出现
#: 一次（med_period≈0）"时会被放大到 1e5 量级，直接转 float16 即溢出为 inf。
#: 因此分母必须取下限，并对结果做量程裁剪。
PERIOD_FLOOR_REL: float = 1e-2
PERIOD_FLOOR_ABS: float = 1e-6
#: 各通道量程裁剪（保语义、保 float16 可表示）
CLIP_DT_NORM: float = 50.0          # z-score 尾界；再大说明 σ 退化
CLIP_DT_DEV: float = 1000.0         # 周期偏差（相对量；饱和值，仍远低于 float16 上限）
CLIP_PERIOD_STABILITY: float = 100.0  # 周期变异系数
CLIP_FLOAT16: float = 6.0e4         # 统一安全边界（< 65504）
#: 11 位标准帧 ID 上界（id_valid 判据）
AID_STD_MAX: int = 0x7FF
#: id_norm 钳位分母（方案 §4 警示：aid/2047 在 Fuzzing 的 0xFFFE 上会溢出到 32.0）
AID_NORM_DENOM: int = 2047

#: ★ `payload_delta` 的计算范围（P1 修复，2026-10-17）
#:   "block_local" —— 逐切分单元（块）独立计算：块内每个 ID 的首帧差值为 0，
#:                    **不存在任何跨块 / 跨 split 引用**（默认，物化产物采用此口径）
#:   "global"      —— 整条 capture 全局计算（旧口径，仅用于复现对照）
#: 为什么必须 block_local：在整条 capture 上求"同 ID 前一帧"，会让"某块内首次出现的 ID"
#: 引到上一块（可能属其他 split）。实测未隔离面 HCRL 3.616%（CANFD_Fuzzing 达 10.687%）、
#: ROAD 0.109% —— 不含标签泄漏，但破坏三段特征分布一致性。
PAYLOAD_DELTA_SCOPE: str = "block_local"

# --------------------------------------------------------------------------- #
# 4. ROAD 标签生成开关（方案 §3、§5.3）                                         #
# --------------------------------------------------------------------------- #
#: fuzzing 采集的标注口径：
#:   "interval"   —— 区间内全部帧标为攻击（ROAD 官方 README 语义，最保守，默认）
#:   "invalid_id" —— 仅当该帧 ID 不在合法 ID 集合内才标攻击
#: ★ 无论选哪种，都必须在论文中显式声明（方案 §3 警示框）
ROAD_FUZZING_MODE: str = "interval"

# --------------------------------------------------------------------------- #
# 5. 协议域标识（方案 §4 chan[6]、§选题范围）                                    #
#: "zero"     —— proto_flag 恒 0（主实验口径，三数据集统一）
#: "protocol" —— 按真实协议族赋值：经典 CAN 2.0 → 0，CAN FD → 1（跨协议实验用）
# --------------------------------------------------------------------------- #
PROTO_FLAG_MODE: str = "zero"

# --------------------------------------------------------------------------- #
# 6. CTT v1.5 子集清单（方案 §1.3）                                             #
# --------------------------------------------------------------------------- #
#: 四个主集；hcrl-ch / hcrl-sa 为独立 src，不与主集混合
CTT_MAIN_SETS: tuple[str, ...] = ("set_01", "set_02", "set_03", "set_04")
CTT_EXTRA_SETS: tuple[str, ...] = ("hcrl-ch", "hcrl-sa")

#: ★ 2026-10-16 决定：**不使用 hcrl-ch / hcrl-sa**（默认关闭）
#  理由（写作时须逐条写进论文 Data 段）：
#    ① 二者是**另外两个独立数据集**（hcrl-ch = HCRL Car-Hacking，Hyundai Sonata；
#       hcrl-sa = HCRL Survival Analysis，Kia/Sonata/Spark），只是被 CTT 仓库"附带打包"，
#       并非 can-train-and-test v1.5 自身的语料；纳入会让数据集归属与引用混乱。
#    ② CTT 主集的**四象限设计**（known/unknown vehicle × known/unknown attack，
#       set_01–04 × 8 子目录）是精心构造的评测框架，混入外部集会破坏其可解释性。
#    ③ 二者时间基与主集不同（真实 Unix 时间戳 vs 主集统一从 1672531200.0 起），
#       车辆、采集年份、攻击定义也不同，混入会引入不受控的分布偏移。
#    ④ 二者均为经典 CAN 2.0 短帧，对本研究 CAN FD 主线无增量信息；
#       本研究的 CAN FD 侧由 HCRL CAN-FD 独立承担。
#  → 关闭后 CTT 仅保留 set_01–04，capture 333 → 310。
CTT_INCLUDE_EXTRA_SETS: bool = False

#: set_* 的 8 个子目录 → 语义标签（用于 manifest 与分层评测）
CTT_SUBSET_SEMANTICS: dict[str, str] = {
    "train_01_attack_free": "normal_only",
    "train_02_with_attacks": "train_with_attack",
    "test_01_known_vehicle_known_attack": "known_veh_known_atk",
    "test_02_unknown_vehicle_known_attack": "unknown_veh_known_atk",
    "test_03_known_vehicle_unknown_attack": "known_veh_unknown_atk",
    "test_04_unknown_vehicle_unknown_attack": "unknown_veh_unknown_atk",
    "test_05_suppress": "suppress",          # ★ 全 0 是设计使然，不进监督评测
    "test_06_masquerade": "masquerade",      # ★ 有标签，可正常做监督评测
}

# --------------------------------------------------------------------------- #
# 6b. ★ CTT 语义化划分（2026-10-16 改，替代 capture 内 60/20/20）                #
# --------------------------------------------------------------------------- #
#: 原 capture 内 60/20/20 切分会**破坏 CTT 的四象限设计**——
#  test_01–04（本意为测试集）各有 60% 落入 train，导致 unknown-vehicle 维度失效
#  （模型训练时已见过测试车辆流量）。故改为**按文件名语义划分**。
CTT_SEMANTIC_SPLIT: bool = True

#: subset → 划分角色（train / test）
CTT_SUBSET_SPLIT_ROLE: dict[str, str] = {
    "train_01_attack_free":        "train",   # 纯正常流量
    "train_02_with_attacks":       "train",   # 含攻击
    "test_01_known_vehicle_known_attack":     "test",
    "test_02_unknown_vehicle_known_attack":   "test",
    "test_03_known_vehicle_unknown_attack":   "test",
    "test_04_unknown_vehicle_unknown_attack": "test",
    "test_05_suppress":            "test",    # 归入 test 语义，但 supervised_evaluable=False
    "test_06_masquerade":          "test",
}

#: ★ 验证集采用**文件级分层留出**（不是文件内时间序切尾部）：
#    从 train_01 / train_02 中按攻击类型分层留出约 25% 的采集文件作 val。
#: 取 0.25 的理由：① val 需覆盖全部攻击类型，25% 足以让每种攻击至少留出 1 个文件；
#:                ② 实测留出后 val 攻击占比约 1.29%（约 17.7 万攻击帧），
#:                   与 train 的 0.63% 同量级，可正常用于阈值标定与早停；
#:                ③ 留 25% 而非 50%，保证 train 仍有约 5,000 万帧。
CTT_VAL_QUOTA: float = 0.25

#: 车辆对照（写进 manifest，供跨车辆泛化实验使用）
CTT_SET_VEHICLES: dict[str, dict[str, str]] = {
    "set_01": {"known": "Chevrolet Impala",  "unknown": "Chevrolet Silverado"},
    "set_02": {"known": "Chevrolet Traverse", "unknown": "Subaru Forester"},
    "set_03": {"known": "Chevrolet Silverado", "unknown": "Subaru Forester"},
    "set_04": {"known": "Subaru Forester",   "unknown": "Chevrolet Traverse"},
}

#: 不参与"有监督逐帧评测"的子目录（正类为 0，AUC-PR/MCC 无定义）——方案 §5.4 M2
CTT_NOT_SUPERVISED: tuple[str, ...] = ("test_05_suppress",)

# --------------------------------------------------------------------------- #
# 7. 时间戳相对化基准（方案 §3、§5 G2）                                          #
# --------------------------------------------------------------------------- #
#: CTT 每个 CSV 均从该基准起（各文件独立）→ 严禁跨文件拼接
CTT_TS_BASE: float = 1672531200.0

# --------------------------------------------------------------------------- #
# 8. 复现性与输出                                                               #
# --------------------------------------------------------------------------- #
SEED: int = 42
#: Parquet 压缩（列式 + zstd 体积约为原始 CSV 的 1/3，方案 §2）
PARQUET_COMPRESSION: str = "zstd"
#: npy 落盘类型（惰性窗口化：只存帧级张量，窗口在 __getitem__ 现场切）
DTYPE_V1 = "float16"
DTYPE_V2 = "uint8"
DTYPE_LABEL = "uint8"
DTYPE_AID = "uint32"
DTYPE_TSREL = "float64"  # 保留原始精度，仅作复核与时间序切分校验用

#: 每个数据集在"训练数据集"下的子目录名
DATASET_DIRS: dict[str, str] = {
    "hcrl_canfd": "HCRL_CANFD",
    "road": "ROAD",
    "ctt15": "CTT_v1.5",
}
