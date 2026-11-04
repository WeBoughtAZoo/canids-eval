#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_variants.py —— LOSO/LOCO 切分变体的**分区不变量**测试（只读，不写盘）
================================================================================
被测对象：`代码/preprocess/make_variants.py::plan_entries`
它决定"哪个 capture 的哪一段进哪个 split"。一旦这里错，E4/E6 的全部结论作废，
因此把不变量固化成测试，而不是靠人工看 dry-run 输出。

覆盖的不变量
------------
I1 每个 (capture, split) 至多出现一次（防止同一段被写两遍、窗口数翻倍）
I2 `loso`：每个 capture 只出现在**一个** split 中（留一组必须是干净的重切分）
I3 `loso`：test 恰好等于该折的录制组，且非空
I4 `loco`：留出 capture 只在 test；其余 capture 只在 train/val（不出现在 test）
I5 三折/四折的 test 窗口数之和 ≈ 全语料 test 规模量级（防"折内无正例"这类空折）

运行：`python 代码/tests/test_variants.py`（或 pytest）
作者：小巴（WorkBuddy）    日期：2026-10-31
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "代码"))

from preprocess.make_variants import group_of, plan_entries  # noqa: E402


def load(dataset: str, dataset_dir: str) -> dict:
    p = ROOT / "训练数据集" / dataset_dir / "03_manifest" / "splits.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    caps: dict[str, dict] = {}
    for name, node in doc["captures"].items():
        ranges = {sp: [(int(a), int(b)) for a, b in (node.get("ranges") or {}).get(sp) or []]
                  for sp in ("train", "val", "test")}
        caps[name] = {"ranges_abs": ranges,
                      "n_frames": int(node.get("n_frames") or 0)}
    return caps


def groups_of(dataset: str, caps: dict) -> dict[str, list[str]]:
    g: dict[str, list[str]] = {}
    for n in caps:
        g.setdefault(group_of(dataset, n), []).append(n)
    return g


def test_loso_invariants() -> None:
    caps = load("ctt15", "CTT_v1.5")
    groups = groups_of("ctt15", caps)
    assert len(groups) == 4, f"CTT 应有 4 个录制组，实际 {sorted(groups)}"
    for fold in sorted(groups):
        entries = plan_entries("ctt15", "loso", caps, fold, groups)
        # I1
        cnt = Counter((e["capture"], e["dst"]) for e in entries)
        assert max(cnt.values()) == 1, f"{fold}: 同一 (capture,split) 重复写入"
        # I2
        per_cap = Counter(e["capture"] for e in entries)
        assert set(per_cap.values()) == {1}, f"{fold}: 有 capture 跨多个 split"
        # I3
        test_caps = {e["capture"] for e in entries if e["dst"] == "test"}
        assert test_caps == set(groups[fold]), f"{fold}: test 集合不等于该组"
        assert test_caps, f"{fold}: test 为空"


def test_loco_invariants() -> None:
    caps = load("hcrl_canfd", "HCRL_CANFD")
    groups = groups_of("hcrl_canfd", caps)
    assert len(groups) == 3, f"HCRL 应有 3 个 capture，实际 {sorted(groups)}"
    for fold in sorted(groups):
        entries = plan_entries("hcrl_canfd", "loco", caps, fold, groups)
        # I1
        cnt = Counter((e["capture"], e["dst"]) for e in entries)
        assert max(cnt.values()) == 1, f"{fold}: 同一 (capture,split) 重复写入"
        # I4
        held = {e["capture"] for e in entries if e["dst"] == "test"}
        assert held == {fold}, f"{fold}: test 应只含留出 capture，实际 {held}"
        for e in entries:
            if e["capture"] != fold:
                assert e["dst"] != "test", f"{fold}: 非留出 capture {e['capture']} 进了 test"
            else:
                assert e["dst"] == "test", f"{fold}: 留出 capture 仍出现在 {e['dst']}"
        # I5：test 窗口非空
        n = sum(b - a for e in entries if e["dst"] == "test" for a, b in e["ranges"])
        assert n > 0, f"{fold}: test 帧数为 0"


def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    fail = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except AssertionError as e:
            fail += 1
            print(f"  [FAIL] {name}: {e}")
    print(f"\n结果：PASS={len(tests) - fail}  FAIL={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
