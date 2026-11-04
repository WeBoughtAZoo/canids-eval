# 训练与测试模块说明（training/）

 > 版本 **1.2.0** ｜ 2026-10-17 ｜ 对应《最终研究方案与SCI论文大纲-v3定稿》§4.5–§4.8、§6.2–§6.3
> 本模块与 `代码/preprocess/` **完全隔离**：不 import 任何 `preprocess.*`，只通过磁盘产物对接。
 > 变更见 §9（v1.1.0 审查修复 + v1.2.0 v4 对齐）。

---

## 0. 一句话说明

`run_train.py` 训练 → 模型落在 `outputs/training/<run_name>/<dataset>/<model>/<variant>/seed<N>/checkpoints/best.pt`
→ `run_eval.py` 复算 → `run_figures.py` / `run_tables.py` 出图出表 → `run_compare.py` 配对检验。
**所有参数只在一个文件里改**：`代码/training/configs/train.yaml`（16 段）。

---

## 1. 目录结构与职责

```
代码/
├─ preprocess/                     ← 既有预处理（冻结，不改）
├─ training/                       ← 本模块（训练与测试）
│  ├─ configs/train.yaml           ★ 唯一配置文件（16 段）
│  ├─ config.py                    加载 / 校验 / 消融派生 / 路径解析 / 指纹 / 快照兼容
│  ├─ manifest.py                  03_manifest 只读索引（ranges / attack_type / 窗口数校验）
│  ├─ data.py                      惰性窗口 Dataset（块内建窗，按模型裁剪 I/O）+ 采集级采样
│  ├─ models/  common · fdtcn · rgbnet · sequence · __init__(注册表)
│  ├─ losses.py                    加权 BCE / Focal + CE + 可选 L2
│  ├─ calibration.py               τ 搜索（排序+前缀和；平台并列取中位）
│  ├─ metrics.py                   MCC / AUC-PR / bootstrap / 分层表（OvR）
│  ├─ stats.py                     McNemar / Wilcoxon / Holm
│  ├─ trainer.py                   训练循环（早停 val MCC、余弦退火、最优回滚）
│  ├─ evaluator.py                 推理 + 评估落盘（val 定 τ，test 复用）
│  ├─ kdlogic.py                   KD-Logic 蒸馏（DL → 约束 CART）
│  ├─ figures.py                   15 张 SCI 图表（fig01–fig15，PDF+PNG）
│  ├─ tables.py                    8 张 booktabs 表格（T7/T7b/T7c/T8/T10/T12/T13/T14）
│  ├─ compare.py                   配对 McNemar + Holm（T13 数据源）
│  ├─ style.py                     rcParams / 尺寸 / 配色 / 保存（Type42 字体）
│  ├─ artifacts.py                 run 目录 / checkpoint / run_manifest
│  ├─ pipeline.py                  单 run 编排 + 仅评估 + runs_index
│  ├─ utils.py                     随机性 / 日志 / JSON·CSV / 环境快照
│  └─ tests/smoke_test.py          10 组自检
├─ run_train.py                    训练入口
├─ run_eval.py                     仅评估入口
├─ run_figures.py                  图表入口
├─ run_tables.py                   表格入口
├─ run_compare.py                  配对检验入口
└─ run_kd.py                       KD-Logic 入口
```

依赖方向：`config → manifest → data → models/losses → trainer → evaluator → pipeline → CLI`；
`figures / tables` 只依赖产物文件，可离线重绘。

---

## 2. 与预处理模块的接口（唯一数据契约）

| 项 | 位置 | 训练侧用法 |
|---|---|---|
| 帧级张量 | `<dataset>/02_tensors/<split>/<capture>/{v1,v2,label,id_valid}.npy` | `FrameSource` 只读 memmap |
| 切分区间 | `03_manifest/captures/<capture>.json → splits.<split>.ranges` | **绝对帧号 → split 内局部偏移**后建窗 |
| 攻击类型 | 同上 `attack_type / is_masquerade / supervised_evaluable` | 多分类目标与分层表 |
| 口径常量 | `03_manifest/env.json → constants` | 校验 WIN_T / STRIDE / V1_CHANNELS |
| 窗口数 | 同上 `windows_stride_train/test` | 与复算值必须相等，否则拒绝训练 |

两个必须守住的坑（代码已处理并加断言）：

1. **坐标系**：`<split>/<capture>/v1.npy` 只存该 split 的帧，manifest 行号是整条 capture 的绝对帧号。
   `WindowDataset._build_capture_index()` 按"前 i 段长度之和"换算；不换算会越界/读错帧。
2. **块内建窗**：一个 split 有多段不连续 range，窗口只在每段内部滑动，否则跨 split 泄漏。

---

## 3. 配置字段 ↔ 模块

`train.yaml` 共 **16 段**：

| YAML 段 | 主要字段 | 消费者 |
|---|---|---|
| `project` | `root / output_root / run_name` | config / artifacts |
| `paths` | `tensors_root / dataset_dirs / run_dir_template / figures_subdir` | config |
| `artifacts` | `checkpoint.{best,last,final}` 等 | artifacts |
| `data` | `window{T,stride,label_rule}` `v1.channels` `v2.*` `limits` `sampling` | manifest / data |
| `label_space` | `multiclass / min_train_windows / unknown_class` | manifest |
| `model` | `name` + `variants.<name>.*` | models |
| `loss` | `binary.type / pos_weight: auto / lambda_*` | losses |
| `train` | `epochs / batch_size / optimizer / scheduler / seeds / early_stopping` | trainer |
| `calibration` | `split: val / metric / grid` | calibration |
| `eval` | `metrics / bootstrap / per_attack_type / evaluability / mcnemar` | metrics / stats |
| `figures` | `ids / formats / png_dpi / save_bbox / width_in / style / palette` | figures / style |
| `tables` | `formats / decimals / captions / labels / dataset_display` | tables |
| `compare` | `scope: [model, ablation] / alpha / all_seeds` | compare |
| `ablation` | `default_variant` + `variants.<name>` | config |
| `kd` | `feature_stats / tree / distill` | kdlogic |
| `runtime` | `device / deterministic / num_threads / fail_fast` | utils |

**无默认值**：缺键即 `ConfigError` 并列出全部路径；`apply_ablation()` 返回派生副本
（裁剪 `data.v1.channels`、解析 `in_channels: auto`、并入变体覆盖），写入该 run 的 `config_resolved.yaml`。

---

## 4. 模型保存路径

```yaml
paths:
  run_dir_template: "{output_root}/{run_name}/{dataset}/{model}/{variant}/seed{seed}"
artifacts:
  checkpoint:
    best:  checkpoints/best.pt
    last:  checkpoints/last.pt
    final: checkpoints/model_final.pt
```

实际落盘：`outputs/training/main/ctt15/fdtcn/full/seed42/checkpoints/best.pt`。
checkpoint 含 `model_cfg / channels / label_map / window_T / threshold / 两种指纹`，
因此 `run_eval.py --run-dir <dir>` 可独立复现（**不会**重新推断标签表或重选阈值）。

---

## 5. 图表规范

### 5.1 图（14 张）

- 命名：`figures/figXX_<slug>.{pdf,png}`；汇总图在 `<batch>/_aggregate/figures/`
- 格式：PDF（矢量，**TrueType 嵌入**，`pdf.fonttype=42`）+ PNG 600 dpi
- 尺寸：单栏 `3.5 in` / 双栏 `7.16 in`；`save_bbox: tight` 下实测宽度为标称的 93–97%
  （只会被期刊放大，字号不会缩小）；需要严格等宽时改 `save_bbox: exact`
- 字号：正文 8–10 pt，最小标注 6.8 pt；字体 DejaVu Sans；图内文字全英文
- 索引：`figures/figures_index.csv`

| ID | 文件 | 主题 | 章节 |
|---|---|---|---|
| fig01 | `fig01_training_curves` | 损失 + val MCC/AUC-PR 曲线 | 5.2 |
| fig02 | `fig02_pr_curve` | PR 曲线（含 prevalence 基线） | 5.3 |
| fig03 | `fig03_roc_curve` | ROC 曲线 | 附录 |
| fig04 | `fig04_confusion_binary` | 二分类混淆矩阵 | 5.3 |
| fig05 | `fig05_confusion_types` | 类型归因混淆矩阵 | 5.5 |
| fig06 | `fig06_per_type_recall` | 分层召回 + 可评估性（色+纹理） | 5.4 |
| fig07 | `fig07_threshold_mcc` | τ–MCC/F1/FPR 标定曲线 | 5.3 |
| fig08 | `fig08_seed_stability` | 跨 seed 稳定性 | 5.2 |
| fig09 | `fig09_accuracy_size_pareto` | 精度–体积 Pareto | 6.2 |
| fig10 | `fig10_encoding_schematic` | 统一编码示意（16 维 + 字节图） | 3.3 |
| fig11 | `fig11_fdtcn_structure` | FD-TCN 结构（RF=61、双头） | 3.4 |
| fig12 | `fig12_attack_type_heatmap` | 攻击类型 × 主干热力图 | 6.3 |
| fig13 | `fig13_rule_complexity` | 蒸馏保真度/规则复杂度 | 5.7 |
| fig14 | `fig14_latency_distribution` | 时延 P50/P95/P99 | 5.7 |

### 5.2 表（5 张，booktabs）

`<batch>/tables/table_*.{tex,csv}`，无竖线、无 `\hline`，3 位小数（AUC 4 位），表注在 tabular 之后。

| 文件 | 内容 |
|---|---|
| `table_main` | 数据集 × 主干：Prev/MCC/AUC-PR/AUC-PR_norm/F1/FPR/n_pos/n_seed |
| `table_masquerade` | masquerade 子表（按窗口加权聚合） |
| `table_ablation` | 变体 vs full 的 ΔMCC / ΔAUC-PR / 参数量 |
| `table_complexity` | params / ckpt MB / INT8 估算 / 测试 MCC |
| `table_stats` | McNemar b/c/p/p_Holm（需先跑 `run_compare.py`） |

> 生成的 `.tex` 需在导言区加 `\usepackage{booktabs}`。图表清单与 F/T 编号对应关系见
> `训练与测试代码说明与图表清单.html` 与 `代码审查与投稿合规性核查报告.html`。

---

## 6. 运行

环境：`<venv>/Scripts/python.exe`
（torch 2.14.1+cpu / sklearn / matplotlib / scipy / pyyaml / numpy 2.5.4 / pyarrow 26.0.0）

```bash
# 0) 自检
python 代码/training/tests/smoke_test.py
python 代码/run_train.py --dry-run

# 1) 主线（每个数据集各一条，5 seeds）
python 代码/run_train.py --datasets hcrl_canfd --models fdtcn --seeds 42 43 44 45 46 --set project.run_name=main
python 代码/run_train.py --datasets road        --models fdtcn --seeds 42 43 44 45 46 --set project.run_name=main
python 代码/run_train.py --datasets ctt15       --models fdtcn --seeds 42 43 44 45 46 --set project.run_name=main

# 2) 基线（3 seeds）
python 代码/run_train.py --datasets hcrl_canfd road ctt15 --models rgbnet --seeds 42 43 44 --set project.run_name=main

# 3) 主干族对照
python 代码/run_train.py --datasets hcrl_canfd road ctt15 --models lstm transformer --seeds 42 43 44 --set project.run_name=main

# 4) 消融（FD-TCN）
python 代码/run_train.py --datasets hcrl_canfd road ctt15 --models fdtcn --variants Ab-1 Ab-2 Ab-6 --seeds 42 43 44 --set project.run_name=main
# 4b) 字节图掩码消融只对 rgbnet 有意义
python 代码/run_train.py --datasets hcrl_canfd road ctt15 --models rgbnet --variants Ab-1b --seeds 42 43 44 --set project.run_name=main

# 5) 评估 → 图 → 检验 → 表
python 代码/run_eval.py --all --set project.run_name=main
python 代码/run_figures.py --all --set project.run_name=main
python 代码/run_compare.py --set project.run_name=main
python 代码/run_tables.py --set project.run_name=main

# 6) KD-Logic（每个数据集的 seed42 教师）
python 代码/run_kd.py --run-dir "outputs/training/main/hcrl_canfd/fdtcn/full/seed42"
python 代码/run_kd.py --run-dir "outputs/training/main/road/fdtcn/full/seed42"
python 代码/run_kd.py --run-dir "outputs/training/main/ctt15/fdtcn/full/seed42"
python 代码/run_figures.py --run-dir "outputs/training/main/hcrl_canfd/fdtcn/full/seed42" --ids fig13
```

冒烟（几分钟验证全链路）：

```bash
python 代码/run_train.py --datasets hcrl_canfd --models fdtcn \
  --set data.limits.max_train_windows=12000 \
  --set data.limits.max_val_windows=4000 \
  --set data.limits.max_test_windows=4000 \
  --set train.epochs=3 --set project.run_name=smoke
```

---

## 7. 口径红线

1. 阈值只在 `calibration.split`（val）上选；test 传 `threshold=None` 会**抛异常**。
2. 窗内任一帧为攻击 → 窗口判攻击；指标一律窗口级。
3. 类权重只由训练集统计；禁用 SMOTE/ADASYN/GAN；采集偏斜用 `capture_balanced`（w ∝ n^-0.5）。
4. 早停监控 val MCC（非 loss）；每个 epoch 重标定 τ，最优 epoch 的 τ 随 checkpoint 保存。
5. 分层表为 **one-vs-rest**（负集 = 全部非该类型窗口），列名带 `_ovr` 后缀。
6. 旧 run 的 `config_resolved.yaml` 缺少新增段时，`load_snapshot()` 以当前配置为底补齐后校验，
   保证历史 checkpoint 仍可评估。

---

## 8. 性能基线（本机 4 线程 CPU，实测）

| 主干 | 吞吐 | HCRL（13.3 万训练窗） | ROAD（51.4 万） | CTT（153 万） |
|---|---|---|---|---|
| FD-TCN | ≈3,550 win/s | ≈44 s/epoch | ≈2.8 min/epoch | ≈8.3 min/epoch |
| FD-RGBNet | ≈1,330 win/s | ≈2.0 min/epoch | ≈7.5 min/epoch | ≈22 min/epoch |

60 epoch 单 seed 粗估：FD-TCN CTT ≈8.3 h；5 seeds ≈41 h（早停后约一半）。
数字用于排期，不是性能结论。

---

## 9. 1.1.0 变更（本次审查修复）

| 级别 | 问题 | 位置 | 处置 |
|---|---|---|---|
| P0 | PDF 使用 Type 3 字体，IEEE/Elsevier 会拒 | `style.py` + YAML `figures.style` | `pdf.fonttype=42`、`ps.fonttype=42` → 实测 46 个 PDF 全为 Type0/CIDFontType2 |
| P0 | 多分类 `class_weight` 给 normal 类 0 权重 | `pipeline.py:138` | 把 `n_negative` 计入 class 0 |
| P0 | 分层表 precision 用 one-vs-normal，数值偏高不可比 | `metrics.py:152` | 改 one-vs-rest，列名加 `_ovr` |
| P1 | 新配置段导致旧快照无法评估 | `config.py:298` | `load_snapshot()` 以当前配置为底合并 |
| P1 | τ 并列最优时取最小 τ（可能贴到 0.001） | `calibration.py:117` | 取并列区间中位 τ |
| P1 | rgbnet 仍读 V1 张量（无用 I/O） | `data.py` | `use_v1` 开关，按模型裁剪 |
| P2 | 缺"表"能力（只有 CSV） | 新增 `tables.py` | 5 张 booktabs 表 + LaTeX |
| P2 | `eval.mcnemar` 是死配置 | 新增 `compare.py` | McNemar + Holm，接入 T13 |
| P2 | checkpoint 存了永远为 None 的 `size_mb` | `artifacts.py` | 删除该字段 |
| P3 | 图例下划线、黑白打印不可辨 | `figures.py` | 图例去下划线；fig06 加 hatch |
| P3 | `--dry-run` 只检查第一个模型；死参数/死条件 | CLI | 逐模型检查；清理 `limit`、`group_by` 等 |


---

## 10. v1.2.0 变更（v4 方案对齐，2026-10-17）

| 级别 | 问题 | 位置 | 处置 |
|---|---|---|---|
| P0 | 家族分层把同采集正常窗口误判为正样本 | `metrics.py::per_type_table` | 正样本改为 `(组==t) & (y==1)`；recall 0.28–0.39 → 0.99–1.00 |
| P1 | 缺 Recall@FPR 操作点（§6.1） | `metrics.py` | 新增 `recall_at_fpr()`，并入主表与分层表 |
| P1 | 分层未按攻击家族（§6.1） | `metrics.py`/`evaluator.py` | 新增 `per_family_table`（主）+ masquerade 单列，fig06/12 改读家族 |
| P1 | 缺静态 MACs（§6.2） | `models/common.py` | 新增 `count_macs()`，评估时重算，写入 T12 |
| P1 | 缺 fig15 精度–时延 Pareto（Δ9） | `figures.py` | 新增 `fig15_accuracy_latency()` |
| P2 | 缺 Ab-12/13/14 消融（§5.3） | `configs/train.yaml` | 配方补齐 + 单头一致性守卫 |
| P2 | 缺 T14 数据集规格表（§4.1） | `tables.py` | 新增 `table_dataset_specs()` |

- 主实验结果（T7 的 MCC/AUC-PR）不受影响；仅纠正了错误的家族级 recall。
- v4 的 Δ1（分别训练）与现有 `run_single` 单数据集入口一致，无需改造。
