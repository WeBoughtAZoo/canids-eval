# 发布说明（RELEASE NOTES）· v1.0.2-paper

> 本文件说明这个仓库**装了什么、为什么这么装、以及什么被有意排除**。
> 逐文件清单见 `MANIFEST.csv`（列：`source` / `category` / `bytes` / `sha256` / `note`，文本文件按 LF 字节计算哈希）。

---

## 1. 仓库形态：单仓，不拆两仓

早期方案（`论文提交材料/投稿包与许可规则.md` §4）曾计划拆成 `canids-eval`（代码）+
`canids-paper-artifact`（配置/表格/图）两个仓库。**现改为单仓**，理由有三：

1. 全量仅约 **11 MiB**（`MANIFEST.csv` 616 条），拆仓带来的跨仓版本对齐成本远大于收益；
2. 论文 Data Availability 用的是单数 "the project repository"，双仓会与正文不符；
3. 读者一次 `clone` 即可同时拿到代码、冻结表格与充分统计量，减少"版本对不上"的可能。

`CITATION.cff` 与论文中的仓库 URL 均指向这一个仓库。

---

## 2. 装了什么

| 类别 | 内容 | 作用 |
|---|---|---|
| `代码/`（70 个文件） | 预处理、训练/评测、统计检验、出图出表、审计复算、`canids_eval`、`synth`、`tests` | 训练与核验所需的全部源码；配置单一来源 `training/configs/train.yaml`。**不含** `remote/`（租赁 GPU 编排）与 6 个一次性分析/出图脚本 |
| `outputs/artifact/` | `counts_capture.csv`、`counts_type.csv`、`counts_density.csv`、`discordance_capture.csv`、`tau_seeds.csv`、`sparse_families.csv`、`ablation_inventory.csv`、`runs.csv`、`verify_report.md`、`README.md` | **充分统计量**：2×2 计数，可逐位重算论文全部指标与显著性检验；`verify_report.md` 记录 1,911 项比对、0 失败 |
| `outputs/audit/` | 嵌套推断、SET 拆分、开集度量、设备等价性、鲁棒性、操作点、切分敏感性等汇总表（22 个 CSV + 说明 + 3 张图） | 补充表 S2/S5/S6 与正文审计结论的来源 |
| `outputs/training/main/tables/` | `table_main.csv`、`table_density.csv`、`table_stats.csv`、`table_operating.csv`、`table_known.csv`、`table_ablation.csv` 等 14 个 | **冻结表格**：论文表 2–7 由这些 CSV 直出，不经手工转录 |
| `outputs/training/gpu1/tables/` | 同口径 11 个 | GPU 批次对照（补充表 S4） |
| `outputs/training/*/runs_index.csv` + 逐 run `metrics_*.json` | 3 份索引 + 447 个聚合指标文件 | 让 `verify_artifacts.py` **开箱可跑**；只含计数与指标，路径一律为仓库相对路径，无逐窗预测、无信号 |
| `论文/v2_2026-11-02/figures/` | `fig01_encoding.png` … `fig06_deployment.png` | 600 dpi 图件，与正文图 1–6 一一对应 |
| 仓库元文件 | `README.md`、`LICENSE`（MIT）、`CITATION.cff`、`requirements.txt`、`.gitignore`、`MANIFEST.csv` | — |

---

## 3. 有意排除的内容

| 排除项 | 体量 | 原因 |
|---|---|---|
| `训练数据集/`、`训练数据集_LOSO/`、`训练数据集_LOCO/`、`训练数据集_ROAD_cap/`、`数据集/` | ≈ 96 GB | 语料原始数据与派生张量；三个语料均由提供方公开发布、可用于研究，本仓库只是不分发任何派生内容（作者政策，详见 §4） |
| `*.npy` / `*.npz` / `*.pt` / `*.pth` / `*.ckpt` | 600 个 ckpt、409 个 predictions | 派生张量与逐窗预测；checkpoint 按合理请求单独提供 |
| `outputs/training/*/*/*/*/`（run 级目录） | — | 与上一条重合；其指标已由 `outputs/artifact/` 完整覆盖 |
| `代码/remote/`（整目录 27 个文件） | 131 KB | 租赁 GPU 编排九脚本 + SOP。属基础设施，与本论文 98 个 CPU run 无关，且含主机、端口、用户名与私钥路径 |
| `代码/analyze_c4.py`、`analyze_operating_point.py`、`analyze_variants.py` | 3 个文件 | 一次性分析，产出不在论文中 |
| `代码/make_fig18.py`、`make_fig19.py`、`make_fig20.py` | 3 个文件 | 早期编号出图，论文只用到 fig01–fig06 |
| `代码/scripts/legacy/build_v6.py` | 93 KB | 一次性文档生成器，含内网 IP 与个人绝对路径 |
| 方案 HTML、审查报告、改动台账、投稿材料 | — | 版本演化史；外流会造成 HARKing 质疑 |

`.gitignore` 已同步覆盖上述路径，误 `git add -A` 也不会带进去。

---

## 4. 数据使用与分发边界

三个语料（ROAD、HCRL CAN-FD、can-train-and-test v1.5）均由提供方主动公开发布，可用于研究，
正确引用即可。本仓库不建立许可确认档案，也不列具体许可名称。

**分发边界**：不分发任何原始或派生信号数据（含张量与逐窗预测），也不分发训练权重。
读者需自行从提供方获取语料，再运行本仓库的预处理脚本。这是作者的政策选择。

充分统计量只含聚合计数（每 capture × 每 run 的 TP/FP/FN/TN），不含任何帧级或窗口级信号。

---

## 5. 三级复现路径

| 级别 | 需要什么 | 能验证什么 | 成本 |
|---|---|---|---|
| **L0** | 仅本仓库 | 用 `outputs/artifact/` 的计数重算论文每一个指标、簇级 p 值、密度分层召回；并跑通 `verify_artifacts.py` | < 1 分钟 |
| **L1** | 仅本仓库 | `synth/` 生成合成语料，跑通 预处理→训练→评测→出表 全链路 | ≈ 10 秒 |
| **L2** | 自行申请三个语料 | 完整重训 98 个主 run | ≈ 406 CPU·h |

```bash
# L0：核验
python 代码/verify_artifacts.py --batches main,gpu1   # 1,911 项比对，须 0 失败
# L1：冒烟
python 代码/smoke_pipeline.py          # 合成语料全链路
# L2：重训（示例）
python 代码/run_preprocess.py --datasets ctt15
python 代码/run_train.py --dataset ctt15 --model fdtcn --seeds 42 43 44 45 46
```

---

## 5.1 关于「1,911 项」这个数字

`verify_artifacts.py` 的 `--batches` 决定比对范围：

| 命令 | 项数 | 用途 |
|---|---|---|
| `--batches main` （默认） | **1,438** | 只覆盖 CPU 主批次 |
| `--batches main,gpu1` | **1,911** | 论文报告的口径（含 GPU 补充批次） |

两者均须 **0 失败**。`outputs/artifact/verify_report.md` 记录的是 **1,911** 口径。
论文正文写的 1,911 与仓库一致；若只跑默认值会得到 1,438，属正常差异而非失败。

> 历史文档中另出现过 1,570 与 1,823 两个中间值，均已作废，以本文与 `verify_report.md` 为准。

---

## 6. 发布步骤（首次上传）

```bash
cd 代码/github
git init
git add -A
git commit -m "Release v1.0.2-paper: code, frozen tables and sufficient statistics"
git tag -a v1.0.2-paper -m "Matches Sensors submission v2 (2026-11-07)"
git remote add origin https://github.com/WeBoughtAZoo/canids-eval.git
git push -u origin main --tags
```

**顺序不可颠倒**：先 commit → 再打 tag → 最后 push。tag 指向的 commit 必须与论文数字一致；
之后再改主分支不影响已发布的引用。

发布后把 tag 的 commit 短哈希写进 `README.md` 的引用节，便于读者核对。

---

## 7. 维护约定

- 重新打包：`python 代码/scripts/build_release.py`（幂等，会重建 `代码/`、`outputs/`、`论文/` 与 `MANIFEST.csv`）。
- `README.md` 与 `RELEASE-NOTES.md` **不参与自动同步**，属发布版人工维护（与项目根 `README.md` 受众不同）。
- 改动任何影响论文数字的代码后，须重跑 `build_artifact.py` → `verify_artifacts.py`，两者全绿才可重新打 tag。
- 版本号与 tag 一一对应；不覆盖已发布的 tag。
