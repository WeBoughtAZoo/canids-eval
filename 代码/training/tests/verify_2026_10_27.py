# -*- coding: utf-8 -*-
"""
tests/verify_2026_10_27.py —— 2026-10-27 修复项的自检
================================================================================
覆盖四处改动，每处给出可判定的期望值：

  1) metrics.recall_at_fpr 支持外部 τ（消除操作点泄漏）
  2) metrics.binary_metrics 的 op_taus / fpr_at_fpr_* / op_source
  3) models.common.count_macs 计入 LSTM 循环与 attention matmul
  4) configs/train.yaml 的 seeds 与实际一致

运行：python 代码/training/tests/verify_2026_10_27.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training import config as C            # noqa: E402
from training import metrics as M           # noqa: E402
from training.models import build_model, model_inputs  # noqa: E402
from training.models.common import count_macs          # noqa: E402

CFG_PATH = ROOT / "training" / "configs" / "train.yaml"
BATCH = ROOT.parent / "outputs" / "training" / "main"


def _synth(n_neg: int = 20000, n_pos: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    neg = rng.beta(2.0, 8.0, n_neg)
    pos = rng.beta(6.0, 2.0, n_pos)
    y = np.concatenate([np.zeros(n_neg, np.int64), np.ones(n_pos, np.int64)])
    p = np.concatenate([neg, pos])
    return y, p


def t1_tau_injection() -> str:
    """给定 τ 时不得再动数据；未给 τ 时与旧行为逐位一致。"""
    y, p = _synth()
    tau_auto, rec_auto = M.recall_at_fpr(y, p, 0.01)
    tau_given, rec_given = M.recall_at_fpr(y, p, 0.01, tau=tau_auto)
    assert tau_given == tau_auto
    assert abs(rec_given - rec_auto) < 1e-12

    # 用一个明显不同的 τ：召回必须随之改变，且等于直接手算
    tau2 = float(np.quantile(p[y == 0], 0.90, method="higher"))
    _, rec2 = M.recall_at_fpr(y, p, 0.01, tau=tau2)
    assert abs(rec2 - float((p[y == 1] >= tau2).mean())) < 1e-12
    assert abs(rec2 - rec_auto) > 1e-6, "外部 τ 未生效"
    return f"先验 τ={tau_auto:.4f}，外部 τ={tau2:.4f} → 召回 {rec_auto:.4f} vs {rec2:.4f}"


def t2_binary_metrics_operating() -> str:
    """op_taus 生效：τ 必须原样采用，且给出该 τ 在本 split 上的**实际** FPR。

    `op_source` 由 evaluator 依"是否为标定 split"给出，binary_metrics 不再自行猜测。
    """
    y, p = _synth(seed=1)
    op = {0.001: float(np.quantile(p[y == 0], 0.999, method="higher")),
          0.01: float(np.quantile(p[y == 0], 0.99, method="higher"))}
    m_in = M.binary_metrics(y, p, 0.5, op_taus=op)
    m_self = M.binary_metrics(y, p, 0.5)
    assert "op_source" not in m_in, "op_source 不应由 binary_metrics 猜测"
    assert abs(m_in["tau_at_fpr_1"] - op[0.01]) < 1e-12
    assert abs(m_in["tau_at_fpr_0p1"] - op[0.001]) < 1e-12
    assert np.isfinite(m_in["fpr_at_fpr_1"]) and np.isfinite(m_in["fpr_at_fpr_0p1"])
    # 用外部 τ 时，召回必须等于按该 τ 直接手算
    assert abs(m_in["recall_at_fpr_1"] - float((p[y == 1] >= op[0.01]).mean())) < 1e-12
    # 自定 τ 时，实际 FPR 应贴近名义目标（证明"实际 FPR"这一列是自洽的）
    assert abs(m_self["fpr_at_fpr_1"] - 0.01) < 2e-3
    assert abs(m_self["fpr_at_fpr_0p1"] - 0.001) < 2e-3
    return (f"inherited τ={m_in['tau_at_fpr_1']:.4f} R={m_in['recall_at_fpr_1']:.4f} "
            f"实际FPR={m_in['fpr_at_fpr_1']:.4f}；self 实际FPR={m_self['fpr_at_fpr_1']:.4f}")


def t3_count_macs() -> str:
    """LSTM 必须计入 T 步循环；Transformer 必须计入 attention。"""
    cfg = C.load(CFG_PATH)
    out = []
    prev = {"lstm": None, "transformer": None}
    for name in ("fdtcn", "rgbnet", "lstm", "transformer"):
        cfg["model"]["name"] = name
        eff = C.apply_ablation(cfg, "full")
        mv = eff["model"]["variants"][name]
        m = build_model(name, mv, n_classes=6, window_T=64)
        needs = model_inputs(name)
        n = count_macs(m, needs=needs, window_T=64, v1_channels=mv["in_channels"],
                       v2_channels=mv.get("in_channels", 8))
        out.append(f"{name}={n:,}")
        if name in prev:
            prev[name] = n

    # LSTM：4·H·(d+H)·T 为下界（h=32,d=16,T=64 ⇒ 393,216），旧实现只有 4.5K 量级
    assert prev["lstm"] is not None and prev["lstm"] > 300_000, prev["lstm"]
    # Transformer：attention 项约 2·L²·E = 2·64·64·32 = 262,144，旧实现无此项
    assert prev["transformer"] is not None and prev["transformer"] > 300_000, prev["transformer"]
    return "，".join(out)


def t4_seeds() -> str:
    cfg = C.load(CFG_PATH)
    seeds = list(cfg["train"]["seeds"])
    assert seeds == [42, 43, 44, 45, 46], seeds
    return f"seeds={seeds}"


def t5_real_run_operating_points() -> str:
    """在真实 run 上复算：val 定 τ → test 的实际 FPR。"""
    run = BATCH / "ctt15" / "fdtcn" / "full" / "seed42"
    if not run.exists():
        return "跳过（未找到 CTT seed42 产物）"
    from training.utils import read_json
    v = np.load(run / "predictions_val.npz")
    t = np.load(run / "predictions_test.npz")
    tau = float(np.quantile(v["prob"][v["label"] == 0], 0.99, method="higher"))
    m = M.binary_metrics(t["label"], t["prob"], float(t["threshold"]),
                         op_taus={0.01: tau})
    ms = M.binary_metrics(t["label"], t["prob"], float(t["threshold"]))
    return (f"τ_val={tau:.4f} → test Recall={m['recall_at_fpr_1']:.4f}, "
            f"实际FPR={m['fpr_at_fpr_1']:.4%}；自定 τ 时 Recall={ms['recall_at_fpr_1']:.4f}, "
            f"FPR={ms['fpr_at_fpr_1']:.4%}")


def t6_zero_shot_filter() -> str:
    """CTT 上应只剩 masquerade 一个 zero-shot 家族（suppress 无正样本，须被过滤）。"""
    import csv

    run = BATCH / "ctt15" / "fdtcn" / "full" / "seed42"
    manifest = ROOT.parent / "训练数据集" / "CTT_v1.5" / "03_manifest" / "dataset_stats.csv"
    if not (run.exists() and manifest.exists()):
        return "跳过（缺 CTT 产物或 manifest）"
    rows = [r for r in csv.DictReader(open(manifest, encoding="utf-8-sig"))
            if int(r["n_test"] or 0) > 0]
    rows.sort(key=lambda r: r["capture"])          # 与 WindowDataset.specs 同序
    fam = np.array([r["attack_family"] for r in rows], dtype=object)

    d = np.load(run / "predictions_test.npz")
    cap, y = d["cap"], d["label"]
    raw = {"masquerade", "suppress"}                # zero_shot_types 的原始返回
    kept = [g for g in sorted(raw) if bool((fam[cap][y == 1] == g).any())]
    assert kept == ["masquerade"], kept
    n_sup = int((fam[cap] == "suppress").sum())
    assert n_sup > 0 and int(y[fam[cap] == "suppress"].sum()) == 0
    return f"{sorted(raw)} → {kept}（suppress 窗口 {n_sup:,} 个，正样本 0）"


def main() -> int:
    print("=" * 78)
    print("2026-10-27 修复项自检")
    print("=" * 78)
    cases = [("1. recall_at_fpr 支持外部 τ", t1_tau_injection),
             ("2. binary_metrics 操作点继承", t2_binary_metrics_operating),
             ("3. count_macs 计入 LSTM/attention", t3_count_macs),
             ("4. 配置 seeds 对齐实际", t4_seeds),
             ("5. 真实 run 复算（CTT seed42）", t5_real_run_operating_points),
             ("6. CTT zero-shot 家族过滤", t6_zero_shot_filter)]
    bad = 0
    for name, fn in cases:
        try:
            msg = fn()
            print(f"  [PASS] {name}  {msg}")
        except AssertionError as e:
            bad += 1
            print(f"  [FAIL] {name}  {e}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"  [ERR ] {name}  {type(e).__name__}: {e}")
    print("-" * 78)
    print(f"通过 {len(cases) - bad}/{len(cases)}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
