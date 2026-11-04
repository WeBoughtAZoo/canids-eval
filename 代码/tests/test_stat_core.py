#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_stat_core.py —— 统计核心的数值单测（无 pytest 依赖）
================================================================================
为什么单独测这几个函数：本文最强的结论（C1/C2）建立在它们之上，
而现有 `smoke_test.py` 只覆盖预处理与切分。边界情形一旦错，错的是**结论方向**。

覆盖
----
· `metrics.binary_metrics` 的 MCC 退化情形（完美 / 全错 / 单类）
· `stats.mcnemar` 与 `signflip_p` 的精确枚举与 p_min 触底
· `nested.mcnemar_nested` 的输入校验与两阶段自助区间
· `openset._auroc` 的完美分离 / 完全反向 / 全并列
· `openset._tpr_at_fpr` 在"负样本全为 0"时的断崖行为（ROAD 的真实情形）

运行：`python 代码/tests/test_stat_core.py`（或 pytest）
作者：小巴（WorkBuddy）    日期：2026-10-30
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training import metrics as M            # noqa: E402
from training import nested as N             # noqa: E402
from training import openset as O            # noqa: E402
from training.stats import mcnemar, signflip_p  # noqa: E402


def _close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------- #
def test_mcc_perfect_and_inverse() -> None:
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.6, 0.4, 0.1])
    m = M.binary_metrics(y, p, 0.5)
    assert _close(m["mcc"], 1.0), m["mcc"]
    assert m["tp"] == 2 and m["tn"] == 2 and m["fp"] == 0 and m["fn"] == 0

    m2 = M.binary_metrics(np.array([1, 0]), np.array([0.2, 0.8]), 0.5)
    assert _close(m2["mcc"], -1.0), m2["mcc"]


def test_mcc_degenerate_single_class() -> None:
    """全正样本时 MCC 的分母含 0 → 必须返回 0 而不是抛异常或 NaN 传播。"""
    m = M.binary_metrics(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]), 0.5)
    assert m["mcc"] == 0.0, m["mcc"]


def test_mcnemar_exact_small() -> None:
    y = np.array([1, 1, 1, 1, 0, 0])
    a = np.array([1, 1, 0, 0, 0, 0])          # 2 错
    b = np.array([1, 1, 1, 1, 0, 0])          # 全对
    r = mcnemar(y, a, b)
    # b = "A 错 B 对" 的不一致对数（a 在 2 个正样本上判 0，b 判对）
    assert r["b"] == 2 and r["c"] == 0, r
    assert r["method"] == "exact_binomial"
    assert _close(r["p_value"], 0.5, 1e-12), r["p_value"]


def test_signflip_floor() -> None:
    """K=4 全同向 ⇒ p 只能取到 2/2^4 = 0.125，且必须标记为触底。"""
    d = np.array([10, 20, 30, 40])
    r = signflip_p(d)
    assert _close(r["p_value"], 0.125), r
    assert _close(r["p_min"], 0.125)
    assert r["p_at_floor"] is True


def test_signflip_degenerate() -> None:
    r = signflip_p(np.zeros(5, dtype=int))
    assert r["p_value"] == 1.0 and r["method"] == "cluster_degenerate"


def test_signflip_mixed_not_significant() -> None:
    """方向不一致（+100, −100）时统计量为 0，p 必为 1。"""
    r = signflip_p(np.array([100, -100]))
    assert _close(r["p_value"], 1.0), r


def test_nested_shape_guard() -> None:
    y = np.array([1, 0, 1, 0])
    a = np.array([1, 0, 1, 0])
    b = np.array([1, 0, 0, 0])
    cap = np.array([0, 0, 1, 1])
    try:
        N.mcnemar_nested(y, a, b, cap, np.array(["g0"]), n_boot=10)
    except ValueError as e:
        assert "groups" in str(e)
    else:
        raise AssertionError("groups 长度不符时应报错")


def test_nested_ci_brackets_point() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n)
    a = np.where(rng.random(n) < 0.9, y, 1 - y)
    b = np.where(rng.random(n) < 0.7, y, 1 - y)
    cap = rng.integers(0, 8, n)
    grp = np.array([f"g{i // 2}" for i in range(8)])
    r = N.mcnemar_nested(y, a, b, cap, grp, n_boot=200, n_perm=1000, seed=1)
    assert r["delta_ci_lo"] <= r["delta_mcc"] <= r["delta_ci_hi"], r
    assert r["n_groups"] == 4


def test_auroc_three_cases() -> None:
    pos = np.array([True, True, False, False])
    assert _close(O._auroc(np.array([0.9, 0.8, 0.2, 0.1]), pos), 1.0)
    assert _close(O._auroc(np.array([0.1, 0.2, 0.8, 0.9]), pos), 0.0)
    assert _close(O._auroc(np.zeros(4), pos), 0.5)


def test_tpr_at_fpr_cliff() -> None:
    """负样本 99% 恰为 0（ROAD 的真实情形）：1% 目标必然落空，实际 FPR 必须如实回报。"""
    s_n = np.concatenate([np.zeros(995), np.linspace(0.01, 1.0, 5)])   # 99.5% 恰为 0
    s_u = np.concatenate([np.zeros(500), np.full(500, 0.5)])
    r = O._tpr_at_fpr(s_u, s_n, 0.01)
    # 1% 分位点落在 0 上 ⇒ τ=0 ⇒ 实际 FPR=100%；断崖必须被如实回报
    assert r["fpr_actual"] > 0.01, r
    assert 0.0 <= r["tpr"] <= 1.0


# --------------------------------------------------------------------------- #
def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except AssertionError as e:
            n_fail += 1
            print(f"  [FAIL] {name}: {e}")
    print(f"\n结果：PASS={len(tests) - n_fail}  FAIL={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
