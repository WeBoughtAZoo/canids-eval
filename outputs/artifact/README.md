# 充分统计量包（sufficient-statistics artifact）

本目录是论文的**可核验载体**：不含任何原始或经处理的信号数据，只含 2×2 混淆计数。
任何读者都可用这里的整数，逐位重算论文中的**阈值型**指标（MCC、F1、precision、recall、
specificity 与实际误报率）以及显著性检验。**AUC-PR 与 AUC-ROC 不能**由 2×2 计数还原——
它们依赖完整分数分布；这两个指标的取值请与随包发布的逐 run 文件
`outputs/training/<batch>/<dataset>/<model>/<variant>/seed<N>/metrics_test.json` 核对。

## 批次（batch）列

论文有两批训练：
- `main`：CPU 批次，主表 T7a/T7b 的来源；
- `gpu1`：GPU 批次（RTX 4090），用于 C4 的补充对比与容量/池化对照。

设备等价性检验未通过（CTT FD-TCN 的 |ΔMCC| = 2.05 pt），因此**跨批次不可直接比较**；
`batch` 列与 `run_id` 前缀（`<batch>|…`）用于区分。C4 的所有对比都在 `gpu1` 内完成。

## 文件

| 文件 | 粒度 | 能重算什么 |
|---|---|---|
| `counts_capture.csv` | 每 run × 每 capture | MCC / recall / FPR / precision（全语料或任意 capture 子集）；簇级符号翻转 p（配合 `discordance_capture.csv`） |
| `counts_type.csv` | 每 run × 每攻击类型 | 分家族召回、分层 FPR（分母为正常集） |
| `counts_density.csv` | 每 run × 每密度箱 | T11 的密度分层召回 |
| `discordance_capture.csv` | 每配对 × 每 capture | 簇级 McNemar 的 b/c 与 p（符号翻转）；仅 `main` 有 |
| `tau_seeds.csv` | 每 (批次,语料,模型,变体) | τ 的种子均值/标准差/极差 |
| `sparse_families.csv` | 稀疏家族 | 召回 + Clopper–Pearson 95% 区间 |
| `ablation_inventory.csv` | 变体 × 批次 | 哪些已跑 / 未跑（含用途） |
| `runs.csv` | 每 run | 数据集规模、阈值、以及**由计数重算**的阈值型指标（MCC / recall / FPR / precision，用于自检） |

## 重算公式

MCC（由 tp/fp/fn/tn）：

```
MCC = (tp*tn - fp*fn) / sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
```

簇级符号翻转 p：令 d_k = b_k − c_k（每 capture 的净优势），
统计量 T = |Σ_k s_k d_k|，s_k ∈ {−1,+1} 独立等概率；枚举 2^K（K ≤ 16）或 Monte Carlo。
**下限**：K 个簇全部同向时最小可达 p = 2/2^K。

组层（同一语料含多次录制时）：先把 d_k 按组求和得 D_g，再在 G 个组上做同样的符号翻转，
下限为 2/2^G。

密度分层召回：对每个箱取 `tp / n_pos`。

τ 的种子分布：直接读 `tau_seeds.csv`（`tau_range` 是极差）。

## 复算示例（无需数据）

```python
import csv, math                      # 仅标准库，无需额外安装

rows = [r for r in csv.DictReader(open("counts_capture.csv", encoding="utf-8-sig"))
        if r["batch"] == "main" and r["run_id"] == "main|ctt15/fdtcn/full/seed42"]
tp, fp, fn, tn = (sum(int(r[k]) for r in rows) for k in ("tp", "fp", "fn", "tn"))
mcc = (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
print(round(mcc, 4))                  # 0.6564，与 runs.csv 中该 run 的 mcc 列一致

# 论文表 2 的 0.6689 是同一组 5 个 seed 的均值，不是单 seed：
#   把上面的 run_id 换成 5 个种子求和后再按 MCC 公式算，或直接取 runs.csv 的 5 行求平均。
```

## 为什么没有数据

三个语料的许可状态不一致。为避免"对一部分语料分发、对另一部分不分发"的不一致合规姿态，
本文**统一不分发任何派生张量**。复现路径见仓库根目录 `README.md` 的三级说明
（L0 统计量 / L1 合成语料 / L2 自行申请语料）。
