# 三数据集预处理代码 —— 运行说明

> 对应方案：**《三数据集形态解析与预处理方案.html》**（2026-10-16 本机实测核实版）  
> 覆盖数据集：**HCRL CAN-FD** · **ROAD** · **can-train-and-test v1.5**  
> 代码版本：v1.0.0 ｜ 生成日期：2026-10-16

---

## 0. 一页速览

| 项     | 内容                                                                     |
| ----- | ---------------------------------------------------------------------- |
| 入口脚本  | `代码/run_preprocess.py`                                                 |
| 校验脚本  | `代码/tests/smoke_test.py`（31 项断言）                                       |
| 产物根目录 | `训练数据集/`（按数据集分子目录）                                                     |
| 产物架构  | **三层**：L1 Parquet 中间表示 → L2 帧级定长张量 → L3 清单/manifest                    |
| 关键口径  | `WIN_T=64`，`stride(train/test)=32/64`，`PAD_BYTES=32`；CTT 用**文件级语义划分**，HCRL / ROAD 用**块化交错切分**（K=10、边界对齐攻击段边界、比例 60/20/20） |
| 主指标前提 | MCC / AUC-PR（不平衡 0%–14.32%）；**禁止随机划分**；INT8 量化后须重标定阈值                  |

**三条必须记住的红线**

1. **不生成"可直接训练的大 CSV"**。三份数据合计 2.5 亿行以上；ROAD 与 CTT 每个文件时间基独立，合并成一个大 CSV 会丢失文件边界，时间序切分随之失效。
2. **填充只在编码层做**，L1 中间表示的载荷保持变长（`data: binary`）。
3. **归一化统计量只用 train split 估计**（`median_period` / `train_freq` / z-score 的 μ、σ），再原样应用到 val/test。

---

## 1. 目录结构

```
CAN IDS/
├── 数据集/                                  ← 原始数据（只读，脚本绝不写入）
│   ├── CAN-FD_Intrusion_Dataset/            HCRL CAN-FD：3 个无表头变长 CSV
│   ├── road/                                ROAD：ambient(12) + attacks(33) + metadata.json
│   └── can-train-and-test-v1.5/             CTT v1.5：set_01–04 + hcrl-ch + hcrl-sa
│
├── 代码/                                     ← 预处理代码
│   ├── run_preprocess.py                     ★ CLI 入口
│   ├── tests/smoke_test.py                   ★ 单元校验 / 冒烟测试
│   └── preprocess/
│       ├── config.py        全局常量与口径（单点定义）
│       ├── utils.py         hex 向量化解析、日志、目录、哈希
│       ├── cleaning.py      缺失值 / 异常值处理 + 断言
│       ├── parsers.py       三个独立解析器 → 统一中间表示
│       ├── splitting.py     三种切分模式：连续 60/20/20 / 语义角色 / ★块化交错
│       ├── features.py      编码层：feat_v1（16 维）/ feat_v2（8×8 字节图）
│       ├── pipeline.py      编排：解析→清洗→切分→统计→编码→落盘→清单
│       └── dataset.py       惰性窗口化 Dataset（训练只读 L2）
│
└── 训练数据集/                                ← ★ 预处理产物（按数据集分别保存）
    ├── HCRL_CANFD/
    ├── ROAD/
    ├── CTT_v1.5/
    ├── preprocess_summary.json
    └── preprocess_run.log
```

---

## 2. 环境与依赖

已在**隔离虚拟环境**中安装，未污染系统 Python：

```bash
# 解释器
<venv>/Scripts/python.exe

# 依赖（已装好）
numpy 2.5.4 / pyarrow 26.0.0 / tqdm
```

如需在新机器重建：

```bash
PY="python"
"$PY" -m venv <venv>
"<venv>/Scripts/python.exe" -m pip install numpy pyarrow tqdm
```

> 未使用 pandas：三数据集格式差异大，且 CTT 单文件最长 `data_field` 变长，用 `csv` 模块 + 向量化 hex 解析更省内存、更可控。



---

## 3. 快速开始

所有命令在**项目根目录**下执行。

```bash
PY="<venv>/Scripts/python.exe"

# ① 先跑单元校验（31 项断言，秒级）
"$PY" 代码/tests/smoke_test.py

# ② 只列出将要处理的 capture（不写盘）
"$PY" 代码/run_preprocess.py --datasets ctt15 --list

# ③ 冒烟：每个数据集只跑前 1 个 capture
"$PY" 代码/run_preprocess.py --datasets all --limit 1 --force

# ④ ★ 全量预处理（HCRL + ROAD + CTT v1.5）→ 写入 训练数据集/
"$PY" 代码/run_preprocess.py --datasets all

# ⑤ 只处理某个数据集 / 指定 capture
"$PY" 代码/run_preprocess.py --datasets road
"$PY" 代码/run_preprocess.py --datasets ctt15 \
    --captures set_01__test_06_masquerade__masquerade-timed-1 --force
```

### 参数表

| 参数             | 说明                                          |
| -------------- | ------------------------------------------- |
| `--datasets`   | `hcrl_canfd` / `road` / `ctt15` / `all`，可多选 |
| `--out`        | 产物根目录，默认 `<项目根>/训练数据集`                      |
| `--limit N`    | 每个数据集最多处理 N 个 capture（冒烟用）                  |
| `--captures`   | 只处理指定 capture（逗号分隔，名字见 `--list` 输出）         |
| `--force`      | 忽略断点记录，强制重算                                 |
| `--list`       | 只列清单，不写盘                                    |
| `--no-logfile` | 不写 `preprocess_run.log`                     |

### 断点续跑

每个 capture 处理完会写一份 `03_manifest/captures/<capture>.json`。  
中断后**直接重跑同一命令**即可跳过已完成的 capture；加 `--force` 才重算。

---

## 4. 产物说明

每个数据集目录下统一三层：

### L1 `01_interim/<capture>.parquet` —— 统一中间表示

按 capture 分片（**一文件对一 capture**，天然隔离 ROAD/CTT 的独立时间基）。列式 + zstd 压缩，体积约为原始 CSV 的 1/3。

| 字段                                | 类型         | 含义                            |
| --------------------------------- | ---------- | ----------------------------- |
| `ts_rel`                          | float64    | **文件内相对秒** = 原始 ts − 该文件首帧 ts |
| `aid`                             | uint32     | 仲裁 ID 十进制（**全字段解析，不截首字符**）    |
| `dlc`                             | uint8      | 实际数据字节数（0–32）                 |
| `data`                            | binary     | **变长**原始载荷（不填充）               |
| `label`                           | uint8      | 0 正常 / 1 攻击（帧级）               |
| `src` / `capture` / `attack_type` | dictionary | 来源标识                          |

### L2 `02_tensors/<split>/<capture>/*.npy` —— 帧级定长张量

| 文件             | 形状 / 类型           | 说明                                       |
| -------------- | ----------------- | ---------------------------------------- |
| `v1.npy`       | `[N, 16] float16` | 16 维帧向量，供 ① FD-TCN 与 ⑤ KD-Logic          |
| `v2.npy`       | `[N, 8, 8] uint8` | 字节图帧级表示：通道 0–3 = 数据，通道 4–7 = **逐字节填充掩码** |
| `label.npy`    | `[N] uint8`       | 帧级标签                                     |
| `aid.npy`      | `[N] uint32`      | 仲裁 ID（分层评测用）                             |
| `id_valid.npy` | `[N] uint8`       | `aid ≤ 0x7FF` 的二值通道（Fuzzing 非法 ID 强信号）   |
| `ts_rel.npy`   | `[N] float64`     | 时间序复核用                                   |

> **为什么是帧级而不是窗口级**：V1 全量窗口展开 ≈ **512 GB**，必然 OOM。  
> 帧级落盘后磁盘占用降为约 **1/32**，窗口在 `__getitem__` 现场切（惰性窗口化，方案 §6）。

### L3 `03_manifest/` —— 索引与可复现性凭据

| 文件                        | 内容                                                       |
| ------------------------- | -------------------------------------------------------- |
| `splits.json`             | 每个 capture 的 train/val/test **区间列表**（左闭右开）；块化交错下每 split 有多个不连续区间，另含 `block_scheme` 全部参数        |
| `dataset_stats.csv`       | 逐 capture：原始行数、有效帧数、攻击帧数/占比、时间范围、三 split 帧数与攻击数          |
| `anomalies.csv`           | 逐 capture：**缺失值 / 异常值处理计数**（论文"数据清洗"小节的证据链）              |
| `env.json`                | Python/numpy/pyarrow 版本、处理时间、访问日期、**配置指纹 sha256**、口径常量快照 |
| `captures/<capture>.json` | 单 capture 完整记录（断点续跑 + 训练集统计量标量）                          |

---

## 5. 预处理步骤与方案的逐条对照

### 5.1 通用步骤（方案 §5.1 G1–G7）

| 步骤              | 实现位置                                                 | 落地情况                                                 |
| --------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| G1 解析为中间表示      | `parsers.py`                                         | 三个独立解析器，输出同一 schema；**不做填充**                         |
| G2 时间戳相对化       | `cleaning.relativize_and_monotonic`                  | HCRL 用首帧、ROAD 用首帧（合成基准）、CTT 用 `1672531200.0`         |
| G3 按 capture 分片 | `pipeline.py`                                        | 一 capture 一 parquet + 一 tensor 目录，**严禁跨 capture 拼接** |
| G4 切分（★2026-10-17 修订） | `splitting.make_splits` / `make_role_splits` / **`make_block_splits`** | 三种模式；切分点（区间列表 + 块边界）全部写入 `splits.json`；HCRL / ROAD 用块化交错（K=10、边界对齐攻击段边界、窗口严格块内构造） |
| G5 训练集统计量       | `features.fit_train_stats`                           | 只用 train split，跨 split 复用                            |
| G6 编码 + 落盘      | `features.encode_capture` + `pipeline.write_tensors` | V1 `float16`、V2 `uint8`、labels `uint8`               |
| G7 清单与校验        | `pipeline._write_manifests` + `cleaning` 断言          | 无 NaN 断言、**分区合法性断言**（无空洞/无重叠）、`len(data)==col2` 断言                |

### 5.1b ★ 三种切分模式（2026-10-17 修订）

`config.SPLIT_MODE` 控制，实际按数据集分发（`pipeline.process_capture` 的 E2 段）：

| 模式 | 适用 | 机制 | 关键参数 |
|---|---|---|---|
| `semantic_role` | **CTT v1.5** | 按文件名角色**整文件归入** train/val/test（保四象限设计） | `CTT_SEMANTIC_SPLIT`、`CTT_VAL_QUOTA=0.25`、`SEED=42` |
| **`block_interleaved`** ★ | **HCRL CAN-FD、ROAD** | 时间线切 **K=10** 个连续块，`i%5==2→val`、`i%5==4→test`、其余 train；**块边界对齐攻击段边界**；窗口**严格块内**构造 | `BLOCK_K=10`、`ALIGN_TO_SEGMENT_EDGE=True`、`SEGMENT_SEARCH_FRAC=0.10`、`BLOCK_RATIO_TOL=0.015` |
| `per_capture_ratio` | 兜底（capture 过短，或 `SPLIT_MODE` 改回时） | capture 内行号连续 60/20/20 | `SPLIT_RATIOS` |

**为什么改为块化交错**：capture 内连续切分下，攻击注入在时间轴上高度局部化 ——
HCRL test 段仅 331 个正窗口且 **Fuzzing / Malfunction 为 0**，使"按攻击类型分层报告"在 CAN FD 侧不可执行。
块化交错**不引入随机数、不引入跨块窗口、不引入段级泄漏**（窗口严格块内构造 ⇒ 零重叠，
McNemar 独立性假设 A7 仍成立），代价是放弃"测试块晚于训练块"的严格时间外推。

**三条静默保障**（写进 `splits.json` 的 `block_scheme`，便于复核）：
1. **段边界对齐** —— 每个内部边界微调到搜索窗口内最近的**标签跳变点**（`label[t-1] != label[t]`），
   因此**任何攻击段都不可能被边界劈成两半**（实测 `n_attack_segments_cut = 0`）；
2. **比例守卫** —— 对齐后 train 占比偏差 > 1.5 pt 时按 0.5 倍收紧窗口重试（≤4 次），
   仍不满足则退回等分边界，保证 6:2:2 不被破坏；
3. **切分单元内独立计时（r3 加强）** —— 每块首帧的 `Δt` 归零；**`payload_delta` 改为逐块独立计算**
   （`PAYLOAD_DELTA_SCOPE="block_local"`，块内每个 ID 的首帧差值取 0）；训练集统计量**排除**块首帧。
   > **P1 修复背景（2026-10-17）**：仅把"块的第 0 帧"归零是不够的——"块内首次出现但不在块首"的 ID，
   > 其同 ID 前驱仍落在上一块（可能属其他 split）。实测未隔离面：**HCRL 3.616%**（`CANFD_Fuzzing` 达 **10.687%**）、
   > **ROAD 0.109%**。改为逐块计算后，**跨 split 特征引用为 0**（已验证 HCRL 446,927 + ROAD 44,088 个块内 ID 首帧全部为 0）。
   > 该问题**不含标签泄漏**（只依赖载荷字节），但会破坏三段特征分布一致性。

**对照实验开关**：把 `config.SPLIT_MODE` 改回 `"per_capture_ratio"` 并重跑，即可复现旧的时间序口径
（对应 v3 方案的 **Ab-10 切分口径敏感性**消融）；产物指纹会随之变化，可据此校验。

### 5.2 缺失值处理

| 情形                         | 处理方式                                  | 计数键（`anomalies.csv`）                    |
| -------------------------- | ------------------------------------- | --------------------------------------- |
| HCRL 行字段数 < 5              | 剔除                                    | `hcrl_row_too_short`                    |
| HCRL 数据体被截断                | 剔除                                    | `hcrl_body_truncated`                   |
| HCRL 标签非 R/T               | 剔除                                    | `hcrl_label_invalid`                    |
| HCRL 数据 token 非 2 字符 hex   | 规范化为 `XX` 后保留                         | `hcrl_token_not_2hex`                   |
| ROAD `#` 后为空 / `#R`（RTR 帧） | **保留**，`dlc=0`、载荷全 0                  | `road_rtr_frame` / `road_empty_payload` |
| ROAD 行不匹配正则                | 剔除                                    | `road_line_unmatched`                   |
| CTT `data_field` 空串        | **保留**，`dlc=0`                        | `empty_payload`                         |
| CTT 长度为奇数的 hex             | 左补 `0`，保留                             | `odd_hex_len`                           |
| CTT `attack` 非 0/1         | 剔除                                    | `ctt_label_invalid`                     |
| 载荷超长（> 32 B）               | 截断到 32 B                              | `payload_too_long`                      |
| HCRL 缺 FDF / BRS / ESI     | 对应通道恒 0，并在 manifest 标注 **"缺失而非真实 0"** | `meta.missing_fields`                   |

### 5.3 异常值处理

| 情形                                            | 处理方式                                                                                      | 计数键                    |
| --------------------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------- |
| 时间戳回退（Δt < 0）                                 | `np.maximum.accumulate` 强制非降                                                              | `ts_backward_fixed`    |
| Δt 极端大值                                       | 按 **train split 的 P99.9** 鲁棒上裁剪                                                           | `dt_clipped`           |
| 仲裁 ID > 0x7FF（仅 HCRL Fuzzing）                 | **保留**，`id_norm` 钳位到 1.0，`id_valid=0`                                                     | `aid_over_11bit`       |
| `dlc > 32`                                    | 截断                                                                                        | `payload_too_long`     |
| **周期分母趋零**（ID 在训练集中仅出现 1 次，`median_period≈0`） | 分母取下限 `floor = max(1e-6, 0.01 × 典型周期)`，并把 `dt_dev` 裁到 ±1000、`period_stability` 裁到 [0,100] | —                      |
| V1 出现非有限值                                     | 归零并计数（理论不应出现，出现即说明统计量退化）                                                                  | `v1_nonfinite_clipped` |
| V1 全通道                                        | 转 float16 前统一裁到 ±6e4（< float16 上限 65504）                                                  | —                      |

> **为什么必须有"周期分母下限"**：`dt_dev = (Δt − median_period)/median_period`。  
> 当某 ID 在训练集中只出现一次且该帧 Δt≈0（例如恰好是采集首帧）时，`median_period≈0`，  
> 分母趋零会把 `dt_dev` 放大到 `1e5` 量级，**直接转 float16 即溢出为 `inf`**。  
> 该问题在 ROAD 的 `fuzzing_attack_2/3`、`reverse_light_off_attack_1` 等采集上被实测触发，  
> 已通过"分母取下限 + 量程裁剪 + 转 float16 前统一 clip"三层防护解决  
> （见 `tests/smoke_test.py` 的"退化周期场景"用例）。

> **ID 溢出的口径选择**：采用方案 §4 警示框的**选项 ③（钳位 + 独立 `id_valid` 通道）**。  
> 理由：`0xFFFE / 2047 ≈ 32.0`，"出现超出 11 位的 ID"本身就是 Fuzzing 的最强判别特征之一，  
> 若只做钳位会抹平该信号。**论文中必须显式声明选了选项 ③。**

### 5.4 统一编码（`features.py` → `V1_CHANNELS`）

| #  | 通道                 | 公式                                               | 备注                                   |           |              |
| -- | ------------------ | ------------------------------------------------ | ------------------------------------ | --------- | ------------ |
| 0  | `dt_norm`          | `zscore(log1p(Δt))`                              | μ/σ 取自 train                         |           |              |
| 1  | `dt_dev`           | `(Δt − median_period[aid]) / median_period[aid]` | masquerade 关键                        |           |              |
| 2  | `id_norm`          | `min(aid, 2047) / 2047`                          | 溢出钳位                                 |           |              |
| 3  | `id_freq`          | `log1p(train_freq[aid])`                         | train 统计                             |           |              |
| 4  | `payload_len_norm` | `dlc / 32`                                       | ★ 分母 **32** 非 64                     |           |              |
| 5  | `pad_ratio`        | `(32 − dlc) / 32`                                | ★ Ab-1 消融对象                          |           |              |
| 6  | `proto_flag`       | 0（主实验）                                           | `PROTO_FLAG_MODE='protocol'` 时按协议族赋值 |           |              |
| 7  | `brs_flag`         | 0                                                | HCRL 缺失 → 恒 0                        |           |              |
| 8  | `esi_flag`         | 0                                                | HCRL 缺失 → 恒 0                        |           |              |
| 9  | `byte_mean`        | `mean(padded[0:32])`                             |                                      |           |              |
| 10 | `byte_std`         | `std(padded[0:32])`                              |                                      |           |              |
| 11 | `byte_entropy`     | `H / log2(32)`                                   | 归一化到 [0,1]                           |           |              |
| 12 | `nonzero_ratio`    | `count(padded≠0) / 32`                           |                                      |           |              |
| 13 | `payload_delta`    | \`mean(                                          | padded − 同 ID 前一帧                    | ) / 255\` | 向量化（稳定排序求前驱） |
| 14 | `payload_u32`      | `int.from_bytes(padded[0:4],'big') / 2^32`       |                                      |           |              |
| 15 | `period_stability` | `std_period[aid] / mean_period[aid]`             | train 统计                             |           |              |

V2 字节图：`ch0..3 = padded[0:8],[8:16],[16:24],[24:32]`；`ch4..7 = mask[0:8],…`。  
去掉 `ch4..7` 即 **Ab-1 的 "w/o 掩码" 版本**。

---

## 6. 训练时怎么读（惰性窗口化）

```python
import sys; sys.path.insert(0, "代码")
from preprocess.dataset import MemmapWindows

ds = MemmapWindows(
    "训练数据集/CTT_v1.5/02_tensors",
    [("set_01__test_01_known_vehicle_known_attack__DoS-3", "test")],
    include_id_valid=True,
)
print(ds.stats())              # {'n_windows':…, 'positive_ratio':…}
x1, x2, idv, y = ds[0]         # x1 (64,16) f32 | x2 (8,64,8) f32 | idv (64) | y int
```

- `stride` 默认按 split 自动取：**train=32**（半重叠扩样）、**val/test=64**（= 窗长，**零重叠**，保证 McNemar 配对检验样本独立）。
- 窗口标注：**窗内任一帧为攻击 → 该窗口标为攻击**（保守口径，须在论文中显式声明）。
- 包进 PyTorch：`torch.utils.data.DataLoader(ds, batch_size=..., num_workers=0)` 即可（Windows 下多进程 mmap 需注意 `num_workers`）。
- 如需定长窗口文件：`preprocess.dataset.materialize_windows(...)`，**注意 512 GB 预算，仅建议对单 capture 或抽样子集使用**。

---

## 7. 可复现性

| 凭据                  | 位置                                            |
| ------------------- | --------------------------------------------- |
| 配置指纹（sha256 前 16 位） | `env.json → config_fingerprint_sha256_16`     |
| 全部口径常量快照            | `env.json → constants`                        |
| 每个 capture 的切分点     | `splits.json`                                 |
| 依赖版本 / 平台 / 访问日期    | `env.json`                                    |
| 逐 capture 处理记录与异常计数 | `03_manifest/captures/*.json`、`anomalies.csv` |
| 完整运行日志              | `训练数据集/preprocess_run.log`                    |

复现步骤：**同一份原始数据 + 同一 `config.py` + 同一命令** → 产物二进制一致（`SEED=42`，全流程无随机性）。

---

## 8. 论文写作必须声明的 6 处口径

1. **HCRL 的 `col2` 是"已解码字节数"，不是 DLC 码**；实测值域仅 {8,16,24,32} → **本数据集未出现 48/64 字节帧**，填充基准取 32。
2. **ID 解析用 `int(row[1], 16)` 全字段解析**，未照抄官方 `helper/preprocess.py` 的 `arbitration_id[1:]`（该做法会让 CAN FD 随机 ID 产生 16:1 冲突）。
3. **ID 溢出处理为"钳位 + `id_valid` 双通道"**（方案 §4 选项 ③）。
4. **ROAD fuzzing 的标注口径为 `interval`**（区间内全部帧标攻击，ROAD README 语义）。可改为 `invalid_id`，但无论选哪种都必须显式声明，否则 masquerade/fuzzing 的分层结果无法横向比较。


5. **`test_05_suppress` 标签全 0 是设计使然**（suppress 删除报文而非注入报文），已标记 `supervised_evaluable=False`，**不进入 AUC-PR/MCC 分层表**，不得写成"无标注已剔除"。
6. **窗口标注为"窗内任一帧为攻击"**的保守口径。

---

## 9. 已知限制（与方案 §10 前提假设一致）

* **P1** 未对原始文件做全量 md5 比对，仅抽验行数与字段。
* **P2** `len(data) == col2` 断言在 HCRL **三个文件上均全量执行**，结果见 `anomalies.csv → hcrl_len_mismatch`（应为 0）。
* **P3** ROAD `injection_interval` 的时间基与 `elapsed = ts − ts_first` 相容（已由冒烟测试在 masquerade 采集上验证）。
* **RPi 5 实测为离线回放**：本预处理产物用于 T2–T5 与全部资源指标，T1 真实总线采集时延未覆盖。

---

## 10. 常见问题

**Q：为什么 `02_tensors` 里是 `[N,16]` 而不是 `[N,64,16]`？**
A：惰性窗口化。窗口级展开需要约 512 GB，帧级只需 1/32，且窗口可由全局行号直接映射。

**Q：`atk_test` 很小（如 Flooding 仅 2.17%）正常吗？**
A：**2026-10-17 起已不是问题**。HCRL / ROAD 改用块化交错切分后，test 段正窗口由 331 → 7,584（HCRL）、1,671 → 4,556（ROAD），三类攻击全部进入 test。若仍需严格时间序口径，可设 `SPLIT_MODE="per_capture_ratio"` 复现旧行为（该对照即 v3 方案的 Ab-10 消融）。

**Q：能否把 CTT 的 set_01–04 合并成一个大文件？**
A：不可以。每个 CSV 都从 `1672531200.0` 起，合并会造出虚假时序跳变。应按 `capture`（= 单文件）分别处理，需要联合训练时在**采样器层**混合。

**Q：`hcrl-ch` / `hcrl-sa` 为什么不并入 set_01–04？**
A：不同车辆（Sonata / Kia / Spark）、不同时间（2016–2017）、**真实 Unix 时间戳**，与 set_* 的 2023 基准不同。已作为独立 `src`（`ctt15_hcrl_ch` / `ctt15_hcrl_sa`）处理。
