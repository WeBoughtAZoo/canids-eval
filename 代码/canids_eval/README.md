# canids-eval · 最小评测包

把"我报了 MCC，但没报口径"变成可复核的三件事：**给定阈值下的真实指标**、**正确的配对检验单位**、**差距来自哪里**。

本包是《最终研究方案 v6.1》与《SCI 论文大纲 v6.1》配套的可执行载体（审计项 D3-2 / D8-1 / I-1）。
它**不重新实现**指标公式，而是调用同仓库 `代码/training/` 的 `metrics.py` / `stats.py` / `nested.py`，
以保证与论文表格逐位一致——两套实现必然漂移，这是本项目刻意避免的。

---

## 1. 它能做什么

| 子命令 | 输入 | 输出 | 回答的问题 |
|---|---|---|---|
| `report` | 一个 `predictions_test.npz` | MCC / AUC-PR / AUC-PR<sub>norm</sub> / prevalence / **该 τ 的实际 FPR** | "这个数字是在什么先验、什么阈值下得到的？" |
| `compare` | 两个 `predictions_*.npz`（+ 可选的 cap→组映射） | capture 层符号翻转 p、组层 p 与 p<sub>min</sub>、ΔMCC 的两阶段自助 CI | "两个模型的差异，在正确的推断单位下还成立吗？" |
| `density` | 一个 npz + 每窗攻击帧数 | 按密度分箱的召回与占比 | "差距是整体差距，还是只集中在低密度窗口？" |

`predictions_*.npz` 需含四个字段：`prob`（float，攻击概率）、`label`（0/1）、`cap`（capture 下标）、
`threshold`（val 上标定的 τ）。本项目每个 run 目录下均有此文件。

---

## 2. 三条不能省的口径

1. **τ 不在 test 上重定**。`report` 直接用 npz 里存的 τ，报出的是"该 τ 在 test 上的实际 FPR"，
   不是名义目标（1% 的预算常常落到 1.1%–3%，这是先验漂移，不是 bug）。
2. **窗口不是独立样本**。`compare` 默认按 capture 聚合后再做符号翻转；
   若同一语料含多次录制（如 can-train-and-test 的 `set_01`–`set_04`），**capture 也不独立**，必须给 `--groups` 再加一层。
3. **p 有结构性下限**。K 个捕获（或 G 个组）全部同向时，最小可达 p = 2/2<sup>K</sup>（组层为 2/2<sup>G</sup>）。
   触底只能写"该粒度下不可判定"，**不能**写"无差异"。

---

## 3. 用法

在仓库根目录（`代码/` 的上一级）执行：

```bash
# 单 run 报告
python -m canids_eval report \
  --pred outputs/training/main/ctt15/fdtcn/full/seed42/predictions_test.npz

# 配对检验（capture 层 + 组层）
python -m canids_eval compare \
  --a outputs/training/main/ctt15/fdtcn/full/seed42/predictions_test.npz \
  --b outputs/training/main/ctt15/lstm/full/seed42/predictions_test.npz \
  --groups outputs/audit/groups_ctt15_test.json

# 密度分层
python -m canids_eval density --pred <npz> --counts <每窗攻击帧数.npy>
```

`--groups` 接受两种格式：JSON（`{"0":"set_01", ...}`，键为 cap 下标）或 `.npy`（长度 = 簇数的一维数组）。
本项目的 CTT 映射已随审计产物提供：`outputs/audit/groups_ctt15_test.json`。

若在 `代码/` 目录内执行，把 `python -m canids_eval` 换成对应路径即可（包与 `training/` 同级是硬要求）。

---

## 4. 最小复现（30 分钟内可跑通）

```bash
# 1) 只取一个 run 的两个 npz（不需要张量、不需要 GPU）
ls outputs/training/main/ctt15/fdtcn/full/seed42/predictions_{val,test}.npz

# 2) 报告 + 配对（各 < 1 分钟，CTT 293 万窗口）
python -m canids_eval report  --pred .../predictions_test.npz
python -m canids_eval compare --a .../fdtcn/full/seed42/predictions_test.npz \
                              --b .../lstm/full/seed42/predictions_test.npz \
                              --groups outputs/audit/groups_ctt15_test.json
```

完整重训（98 个 run）在纯 CPU 上需数千小时，**不建议**作为复现入口；
论文附录给出"最小复现子集"（每语料 1 模型 1 seed）。

---

## 5. 边界与已知限制

- **不提供开集指标**。OSCR/AUOSCR 要求分类器具备拒识头；本项目的五个主干都是二元检测器，
  已知样本永远被强制判类，"拒绝率"恒为 0，OSCR 不可识别。开集量请用
  `training/openset.py`（AUROC(unknown vs normal) 与 TPR@FPR<sub>normal</sub>）。
- **不替代预处理**。本包只吃 `predictions_*.npz`；若你没有该文件，需先跑本项目的 `run_eval.py`，
  或自行导出同名字段。
- **不做多重比较校正**。`compare` 一次只比一对；成组比较请对返回的 p 值自行做 Holm
  （`training.stats.holm` 可直接调用）。
- **能耗不在此包**。mJ/win 依赖板端 PMIC 读数，属部署测量，见 `outputs/pi5_deploy/`。

---

## 6. 与"充分统计量"的关系（数据不公开时的替代路径）

本仓库不发布任何语料数据，但发布**充分统计量**（`outputs/artifact/`）：
每个 run 在测试集上按 capture / 攻击类型 / 密度箱拆分的 TP·FP·FN·TN，以及配对比较的逐 capture `b/c`。
有了这些整数，**不需要 `predictions_*.npz` 也能重算**全部指标与簇级 p：

```python
import pandas as pd, math
c = pd.read_csv("outputs/artifact/counts_capture.csv")
run = c[c.run_id == "ctt15/fdtcn/full/seed42"]
tp, fp, fn, tn = run[["tp","fp","fn","tn"]].sum()
print((tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
```

本包的 `compare` 子命令需要 npz 输入；若只有统计量，用
`python 代码/verify_artifacts.py`（1,438 项自检）或直接按上面的公式手算。
自检报告：`outputs/artifact/verify_report.md`。

---

## 7. 引用与许可

若使用本包，请引用对应论文（投稿后补 DOI）与本仓库的 artifact 记录。
代码许可随仓库主许可；**派生数据不随包分发**——HCRL CAN-FD 的许可状态尚未确认，
详见 `论文提交材料/投稿包与许可规则.md`。
