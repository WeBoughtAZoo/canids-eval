# -*- coding: utf-8 -*-
"""
verify_2026_10_28.py —— 2026-10-28 审计改动的验收脚本
================================================================================
覆盖本次新增/修改的六处代码路径（不重训，全部在秒级完成）：

  1. metrics.cluster_bootstrap_ci   按 capture 聚类的 CI（并与 i.i.d. 对比）
  2. density.window_attack_counts    帧级标签 → 窗内攻击帧数（用真实 HCRL 张量核对）
  3. density.density_table           分箱召回表的结构与单调性
  4. config.apply_ablation           Ab-12b / Ab-18 的派生结果
  5. models + losses                 max 池化、逐帧头、aux_frame 损失
  6. calibration.sweep_curve         阈值网格哨兵点
  7. adaptive.simulate               在线分位数阈值（labeled / capped / all / negative）
  8. stats.mcnemar_cluster           簇级 McNemar（符号翻转置换）

运行：
  python 代码/training/tests/verify_2026_10_28.py
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]      # <项目根>/代码
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from training import adaptive as AD  # noqa: E402
from training import calibration as CAL  # noqa: E402
from training import config as C  # noqa: E402
from training import density as D  # noqa: E402
from training import metrics as M  # noqa: E402
from training import stats as S  # noqa: E402
from training.losses import MultiTaskLoss  # noqa: E402
from training.models import build_model  # noqa: E402

PASS = 0
FAIL = 0


def check(cond: bool, msg: str, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {msg}" + (f"  ({extra})" if extra else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}" + (f"  ({extra})" if extra else ""))


CFG = C.load(ROOT / "代码" / "training" / "configs" / "train.yaml")


# --------------------------------------------------------------------------- #
print("T1 聚类自助：CI 必须比 i.i.d. 宽，且方向一致")
rng = np.random.default_rng(0)
K = 30
per = 800
cap = np.repeat(np.arange(K), per)
# 簇内相关：每个簇有自己的偏移
offset = rng.normal(0, 1.2, K)[cap]
y = (rng.random(K * per) < 0.08).astype(np.int64)
p = 1 / (1 + np.exp(-(offset + 2.0 * y + rng.normal(0, 1, K * per))))
tau = 0.5
iid = M.bootstrap_ci(y, p, tau, metric_names=["mcc"], n_resamples=300, ci=0.95, seed=1)
clu = M.cluster_bootstrap_ci(y, p, tau, clusters=cap, metric_names=["mcc", "recall", "fpr"],
                             n_resamples=300, ci=0.95, seed=1)
w_iid = iid["mcc"]["hi"] - iid["mcc"]["lo"]
w_clu = clu["mcc"]["hi"] - clu["mcc"]["lo"]
check(w_clu > w_iid, "聚类 CI 宽于 i.i.d.（簇内相关被计入）", f"{w_clu:.4f} vs {w_iid:.4f}")
check(clu["_n_clusters"] == K, "返回有效簇数", str(clu["_n_clusters"]))
check(abs(clu["mcc"]["mean"] - M.binary_metrics(y, p, tau)["mcc"]) < 0.02,
      "聚类均值贴近点估计")


# --------------------------------------------------------------------------- #
print("T2 密度复算：与逐窗手算一致（真实 HCRL 张量）")
from training import manifest as MF  # noqa: E402
from training.data import build_datasets  # noqa: E402

ds = "hcrl_canfd"
paths = C.dataset_paths(CFG, ds)
idx = MF.ManifestIndex(ds, paths["manifest_dir"])
label_map = MF.build_label_map(idx, CFG)
cfg1 = C.apply_ablation(CFG, "full")
built = build_datasets(cfg1, idx, dataset_key=ds, label_map=label_map,
                       use_v1=True, use_v2=False, use_id_valid=False, seed=42)
test_ds = built["test"]
sub = test_ds.index.subset(np.arange(0, min(3000, len(test_ds))))
from training.data import WindowDataset  # noqa: E402
shim = WindowDataset.__new__(WindowDataset)
shim.__dict__.update(test_ds.__dict__)
shim.index = sub
cap_arr = sub.file_idx.astype(np.int64)
pos_arr = sub.start.astype(np.int64)
counts = D.window_attack_counts(shim, cap_arr, pos_arr)
# 手算前 20 个
manual = []
for k in range(20):
    c, s = int(cap_arr[k]), int(pos_arr[k])
    lab = np.asarray(shim.sources[c].label, dtype=np.int32)
    manual.append(int(lab[s:s + 64].sum()))
check(list(counts[:20]) == manual, "前 20 窗的密度与逐帧手算一致",
      f"max={int(counts.max())}")
check(int(counts[np.asarray(sub.labels) == 1].min()) >= 1, "所有正窗口密度 ≥ 1")

rows = D.density_table(np.asarray(sub.labels), np.clip(counts / 64.0, 0, 1),
                       counts, 0.5, bins=CFG["eval"]["density_bins"])
check(any(r["density_bin"] == "overall" for r in rows), "含 overall 行")
check(abs(sum(r["frac_of_positives"] for r in rows if r["density_bin"] != "overall") - 1.0) < 1e-9,
      "各箱占比之和为 1")


# --------------------------------------------------------------------------- #
print("T3 消融派生：Ab-12b / Ab-18")
a12b = C.apply_ablation(CFG, "Ab-12b")
check(len(a12b["data"]["v1"]["channels"]) == 10, "Ab-12b → 10 维",
      str(a12b["data"]["v1"]["channels"]))
check("payload_delta" in a12b["data"]["v1"]["channels"]
      and "payload_u32" in a12b["data"]["v1"]["channels"], "Ab-12b 保留 payload_delta/u32")
a18 = C.apply_ablation(CFG, "Ab-18-rgbnet8b")
from training import pipeline as P  # noqa: E402
cfg18 = P.prepare_config(CFG, dataset="hcrl_canfd", model="rgbnet",
                         variant="Ab-18-rgbnet8b", seed=42)
check(a18["data"]["v2"]["channel_index"] == [0, 4], "Ab-18 → v2 通道 [0,4]")
check(int(cfg18["model"]["variants"]["rgbnet"]["in_channels"]) == 2,
      "Ab-18（rgbnet 路径）→ in_channels=2")


# --------------------------------------------------------------------------- #
print("T4 模型：max 池化与逐帧头")
m19 = C.apply_ablation(CFG, "Ab-19-maxpool")
net19 = build_model("fdtcn", m19["model"]["variants"]["fdtcn"], n_classes=16, window_T=64)
x = torch.randn(4, 64, 16)
out19 = net19(x)
check(out19.logit.shape == (4,), "max 池化前向输出形状", str(tuple(out19.logit.shape)))

m20 = C.apply_ablation(CFG, "Ab-20-auxframe")
net20 = build_model("fdtcn", m20["model"]["variants"]["fdtcn"], n_classes=16, window_T=64)
out20 = net20(x)
check(out20.frame_logits is not None and out20.frame_logits.shape == (4, 64),
      "逐帧头输出 [B,T]", str(None if out20.frame_logits is None else tuple(out20.frame_logits.shape)))

crit = MultiTaskLoss(m20, pos_weight=5.0, class_weight=None, n_classes=16,
                     aux_frame_weight=0.3, frame_pos_weight=20.0)
y = torch.tensor([1, 0, 1, 0])
yt = torch.tensor([1, 0, 2, 0])
yf = (torch.rand(4, 64) < 0.1).float()
losses = crit(out20, y, yt, yf)
check(float(losses["loss_frame"]) > 0, "aux_frame 损失非零", f"{float(losses['loss_frame']):.4f}")
check(float(losses["loss"]) > float(losses["loss_bin"]), "总损失含辅助项")


# --------------------------------------------------------------------------- #
print("T5 阈值网格上界修正（中点补搜）")
# 100 个正例在 0.9999；900 个低分负例 + 100 个负例铺在 0.990–0.998，
# 另加 1 个负例在 0.9991（> 网格上界 0.999）。网格内最优必落在 0.999，
# 但它仍会把 0.9991 这条负例算作误报；真正的最优是 0.99905，只能靠上界补搜。
y2 = np.array([1] * 100 + [0] * 1001)
p2 = np.concatenate([np.full(100, 0.9999), np.full(900, 0.1),
                     np.linspace(0.990, 0.998, 100), [0.9991]])
cfg_s = C.apply_ablation(CFG, "full")
cfg_s["calibration"]["grid"] = dict(cfg_s["calibration"]["grid"], include_sentinel=True)
res = CAL.calibrate(y2, p2, cfg_s)
tau_s, info = float(res["threshold"]), res["selected"]
check(info.get("grid_hit_upper_bound") is True, "网格最优落在上界（问题复现）")
check(info["refined"] is True, "触发上界补搜并标注 refined",
      f"n_cand={info.get('n_candidates')}")
check(info["mcc"] > 0.999, "补搜后 MCC 达到完全可分", f"{info['mcc']:.4f}")
check(tau_s >= 0.999, "τ 越过网格上界", f"{tau_s:.6f}")


# --------------------------------------------------------------------------- #
# 契约（2026-10-28 二次修订）：默认 mode="labeled"——只把**延迟到达的已知负样本**
# 放进缓冲。此前两版（negative / capped）都建立在不成立的假设上：
#   · negative：缓冲 = {p < τ_t} 被当前阈值截断 → τ 单调下漂 → FPR 崩（实测 85%）；
#   · capped  ：假设"cap 以下无正例"，但 cap = 3·τ_val 在 τ_val>1/3 时饱和成 1.0，
#               退化为 all → 高先验语料召回腰斩（实测 HCRL 0.999→0.554）。
print("T6 自适应阈值：labeled 模式下 FPR 受控且召回保留")
n = 200_000
rng3 = np.random.default_rng(3)
y3 = (rng3.random(n) < 0.30).astype(np.int8)          # 高先验，对齐 HCRL(34%)/CTT(8%)
p3 = np.clip(np.where(y3 == 1, 0.50, 0.02) + rng3.normal(0, 0.01, n), 0.0, 1.0)

res = AD.simulate(y3, p3, target_rate=0.06, window=4096, step=512, fixed_threshold=0.05)
check(res["params"]["mode"] == "labeled", "默认走 labeled 模式")
fpr3, rec3 = res["overall"]["fpr"], res["overall"]["recall"]
check(0.03 <= fpr3 <= 0.09, "FPR 收敛到目标率 6% 附近", f"{fpr3:.4f}")
check(rec3 > 0.95, "高先验下召回保留", f"{rec3:.3f}")
check(res["params"]["label_delay"] == 512, "标签延迟按 step 落盘",
      f"{res['params']['label_delay']}")
check(res["fixed"]["recall"] > 0.95, "固定 τ 对照同样可分（差异来自 τ 漂移而非分数）",
      f"{res['fixed']['recall']:.3f}")

# 延迟为 0 等价于"标签即时可得"，应当仍接近目标
res0 = AD.simulate(y3, p3, target_rate=0.06, window=4096, step=512,
                   fixed_threshold=0.05, label_delay=0)
check(0.02 <= res0["overall"]["fpr"] <= 0.10, "零延迟下 FPR 仍受控",
      f"{res0['overall']['fpr']:.4f}")

alt = AD.simulate(y3, p3, target_rate=0.06, window=4096, step=512,
                  fixed_threshold=0.05, mode="all")
check(alt["overall"]["recall"] < rec3 - 0.2,
      "mode=all 被攻击分数污染（回归对照，证明默认值必要性）",
      f"all={alt['overall']['recall']:.3f} vs labeled={rec3:.3f}")

neg = AD.simulate(y3, p3, target_rate=0.06, window=4096, step=512,
                  fixed_threshold=0.05, mode="negative")
check(neg["overall"]["fpr"] > fpr3 + 0.2,
      "mode=negative 截断自反馈导致 FPR 崩塌（回归对照，禁止用于论文）",
      f"negative FPR={neg['overall']['fpr']:.3f} vs labeled={fpr3:.3f}")


# --------------------------------------------------------------------------- #
# 簇内高度相关 + 两模型差距一致：窗口级 p 会小到离谱，簇级必须收敛回可辩护量级。
print("T7 簇级 McNemar：同录音相关被计入，p 不得小于窗口级")
rng7 = np.random.default_rng(7)
K, per = 40, 300
cl7 = np.repeat(np.arange(K), per)
n7 = K * per
y7 = rng7.integers(0, 2, n7).astype(np.int64)
# 簇级随机效应：某簇整体更难（两模型同时变差），制造簇内相关
hard = rng7.normal(0, 0.25, K)[cl7]
err_a = np.clip(0.06 + hard, 0.0, 0.95)
err_b = np.clip(0.10 + hard, 0.0, 0.95)
pa7 = np.where(rng7.random(n7) < err_a, 1 - y7, y7)
pb7 = np.where(rng7.random(n7) < err_b, 1 - y7, y7)

res7 = S.mcnemar_cluster(y7, pa7, pb7, cl7, n_perm=4000, seed=7)
check(res7["n_clusters"] == K, "簇数正确", f"{res7['n_clusters']}")
check(res7["p_value"] >= res7["p_value_window"], "簇级 p 不小于窗口级 p（保守方向）",
      f"cluster={res7['p_value']:.3e} vs window={res7['p_value_window']:.3e}")
check(res7["method"] == "cluster_monte_carlo_signflip", "大 K 走分块 Monte Carlo",
      res7["method"])
check(res7["false"] if False else res7["b"] + res7["c"] == res7["n_discordant_window"],
      "簇级 b+c 与窗口级不一致对总数自洽",
      f"{res7['b'] + res7['c']} vs {res7['n_discordant_window']}")

# 小 K：走精确枚举，分辨率 = 2/2^K
cl8 = np.repeat(np.arange(3), 100)
y8 = np.tile([0, 1], 150).astype(np.int64)
pa8 = np.where(y8 == 0, 1, 0)                    # A 全错
pb8 = y8.copy()                                  # B 全对
res8 = S.mcnemar_cluster(y8, pa8, pb8, cl8)
check(res8["method"] == "cluster_exact_signflip", "K=3 走精确枚举", res8["method"])
check(abs(res8["p_resolution"] - 0.125) < 1e-9, "精确枚举粒度 = 1/2^3 = 0.125",
      f"{res8['p_resolution']:.3f}")
check(abs(res8["p_min"] - 0.25) < 1e-9 and res8["p_value"] == 0.25
      and res8["p_at_floor"] is True,
      "3 簇同向时 p 最小只能到 0.25（簇数决定可达下限，不是模型不行）",
      f"p={res8['p_value']:.3f} p_min={res8['p_min']:.3f}")


print("=" * 78)
print(f"结果：PASS={PASS}  FAIL={FAIL}")
print("=" * 78)
raise SystemExit(0 if FAIL == 0 else 1)
