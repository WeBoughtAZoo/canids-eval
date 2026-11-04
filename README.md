# CAN IDS · 评测方法学与充分统计量 artifact

面向窗口化 CAN / CAN FD 入侵检测的评测代码与冻结结果。配套论文：

> Cross-Session Generalisation of CAN and CAN FD Intrusion Detection:
> A Window-Level Measurement Study with Threshold Provenance and a CPU-Only Edge Deployment

仓库不含任何语料数据或派生张量，只含代码、配置、冻结表格与一份充分统计量（2×2 混淆计数）。

---

## 1. 环境与安装

Python ≥ 3.11，仅 CPU。

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` 固定了论文运行时的版本（torch 2.14.1+cpu、numpy 2.5.4 等）。
GPU 环境下 `torch` 需按平台另装轮子，见文件内注释。

---

## 2. 验证结果（无需语料）

```bash
python 代码/verify_artifacts.py --batches main,gpu1
```

预期最后一行：

```
自检完成：1911 项，失败 0 项
```

`--batches` 必须为 `main,gpu1`。只跑 `main` 得 1,438 项——差值来自 GPU 补充批次，
两者都应 0 失败。`outputs/artifact/verify_report.md` 记录的是 1,911 口径。

验证内容：用 `outputs/artifact/` 的整数重算 `runs.csv`、`table_main.csv`、
`table_density.csv`、`comparison*.csv` 与各 run 的 `metrics_test.json`，逐格比对。

若需连训练链路一并确认（自动生成合成语料，约 10 秒）：

```bash
python 代码/smoke_pipeline.py      # 预期：=== SMOKE PASS ===
```

---

## 3. 三级复现

| 级别 | 内容 | 命令 | 耗时 | 需要语料 |
|---|---|---|---|---|
| L0 | 用充分统计量重算全部表格 | `verify_artifacts.py --batches main,gpu1` | < 1 min | 否 |
| L1 | 预处理→训练→评测→出表 全链路 | `smoke_pipeline.py` | ~10 s | 否 |
| L2 | 从原始语料重训 98 个主 run | 见 §5 | ~406 CPU·h | 是 |

---

## 4. 目录

```
代码/
  training/              训练与评测核心
    configs/train.yaml   唯一配置（16 段；无代码内默认值）
    models/              FD-TCN、FD-TCN-PF、FD-RGBNet、LSTM、Transformer、KD
    stats.py             McNemar、簇级符号翻转、Holm 校正
    nested.py            组→capture→窗口 嵌套推断
    openset.py           未知攻击度量
    groups.py            capture → 录制组映射
  preprocess/            预处理与切分变体
  canids_eval/           最小评测包（report / compare / density）
  synth/                 合成语料生成器
  tests/                 单元测试
  run_preprocess.py      预处理入口
  run_train.py           训练入口
  run_eval.py            评估入口
  run_compare.py         配对检验
  run_tables.py          出表
  run_audit.py           审计复算
  run_robustness.py      输入扰动敏感性
  run_split_sensitivity.py  ROAD 切分粒度对照
  build_artifact.py      生成充分统计量
  verify_artifacts.py    统计量 ↔ 表格 逐格比对
  smoke_pipeline.py      合成语料端到端冒烟

outputs/
  artifact/              充分统计量包 + 自检报告
  audit/                 审计复算汇总（嵌套推断、SET 拆分、开集、等价性、鲁棒性、操作点）
  training/main/tables/  冻结表格：论文表 2–7 的来源
  training/gpu1/tables/  GPU 批次同口径表格（补充表 S4）
  training/road_cap_v2/  ROAD capture 级留出（§4.4 与补充表 S7）
  pi5_deploy/summary_v4.csv  树莓派 5 部署测量（表 7）

论文/v2_2026-11-02/figures/  论文图 1–5（600 dpi PNG）
MANIFEST.csv                 逐文件清单
```

**不包含**：语料原始数据与派生张量（`.npy` / `.npz` / `.parquet`）、checkpoint（`.pt`）、
run 级 `predictions_*.npz`、其他批次的开发过程产物。理由见 `RELEASE-NOTES.md` §3。

---

## 5. L2：从原始语料重训

```bash
# 1) 取得三个语料，放置路径见论文的数据集来源说明（随论文提交，不在本仓库）
# 2) 预处理
python 代码/run_preprocess.py --datasets ctt15
# 3) 训练 + 评估（示例：CTT 上的 FD-TCN，5 seeds）
python 代码/run_train.py --datasets ctt15 --models fdtcn --seeds 42 43 44 45 46
python 代码/run_eval.py  --datasets ctt15 --models fdtcn
# 4) 配对检验 / 出表 / 出图
python 代码/run_compare.py --datasets ctt15
python 代码/run_tables.py
python 代码/run_figures.py
```

实测成本：98 个 run 共 406 CPU·h（`outputs/training/main/runs_index.csv` 的 `train_seconds` 求和）。
最小子集（每语料 1 模型 1 seed）约 10 CPU·h。

### 5.1 跨设备训练约束

本仓库不含 GPU 编排脚本。若要补充 GPU 实验，新 run 与既有 98 个 CPU run 不可直接比较
（论文限制 (1)：设备等价性检验 |ΔMCC| = 2.05 pt，未通过）。须遵守：

1. **FP32，不开 AMP**；
2. **超参不动**——epochs / batch_size / lr 沿用 `train.yaml`，只改 device 与 DataLoader 并行度；
3. **批次隔离**——GPU 产出写入独立批次目录，合入前先做设备等价性检查，报告时两批不合并。

---



## 6. 关键约定

1. **窗口在每段连续帧内构造**，不跨 capture 拼接；`W = 64`，stride 训练 32 / 校验测试 64。
2. **标签用 any_frame 规则**：窗口内任一帧为攻击即为正例。
3. **阈值在验证集上标定，原样用于测试**；测试集 FPR 是实际达到值，不是目标值。
4. **两个训练批次（main / gpu1）永不合并**，同设备对照在该批内重训基线。
5. 配置唯一来源 `training/configs/train.yaml`，缺键即报错，不设代码内默认值。

---

## 7. 测试

```bash
python 代码/tests/test_stat_core.py     # 统计核心（10 项）
python 代码/tests/smoke_test.py         # 预处理（44 项，需自备语料）
```

---

## 8. 引用与许可

- 代码：MIT（见 `LICENSE`；仅覆盖代码与文档，不覆盖第三方语料）。
- 引用信息：`CITATION.cff`。
- 归档：以 tag `v1.0.0-paper` 为不可变引用。

```bibtex
@software{xiang2026canids,
  title  = {CAN IDS evaluation toolkit and sufficient-statistics artifact},
  author = {Xiang, Yanjie and Su, Yaping and He, Zhige and Zhang, Yi},
  year   = {2026},
  url    = {https://github.com/WeBoughtAZoo/canids-eval},
  note   = {Tag v1.0.0-paper}
}
```
