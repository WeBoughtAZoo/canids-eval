#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
smoke_pipeline.py —— 无真实数据时的端到端冒烟（L1 证明）
================================================================================
一条命令验证：**仓库里的代码真的能跑**，而不是"只能读"。

流程
----
1. 生成合成语料（3 capture × 4,000 帧，结构与预处理产物同构）；
2. 用该语料训练 1 个 epoch（FD-TCN，seed 42）；
3. 在 val/test 上评估，写 `predictions_*.npz` 与 `metrics_*.json`；
4. 汇总出表（`tables/table_*.csv`）；
5. 断言上述产物存在且非空，然后清理临时目录（`--keep` 可保留）。

它**不证明**论文数字可复现（那是 L0，见 `verify_artifacts.py`），
也**不替代**真实语料上的 L2 复现。合成数据的指标没有任何科学含义。

用法
----
  python 代码/smoke_pipeline.py            # 跑完即清理
  python 代码/smoke_pipeline.py --keep     # 保留产物以便检查

作者：小巴（WorkBuddy）    日期：2026-10-30
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        raise SystemExit(f"命令失败（exit={res.returncode}）：{' '.join(cmd)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="合成语料端到端冒烟")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--corpus", default="训练数据集_SYNTH")
    ap.add_argument("--out", default="outputs/synth_run")
    args = ap.parse_args(argv)

    corpus = ROOT / args.corpus
    out = ROOT / args.out
    for p in (corpus, out):
        if p.exists():
            shutil.rmtree(p)

    run([PY, str(ROOT / "代码" / "synth" / "make_synthetic_corpus.py"),
         "--out", args.corpus])
    run([PY, str(ROOT / "代码" / "run_train.py"),
         "--datasets", "ctt15", "--models", "fdtcn", "--seeds", "42", "--no-figures",
         "--set", f"paths.tensors_root={args.corpus}",
         "--set", f"project.output_root={args.out}",
         "--set", "train.epochs=1",
         "--set", "train.batch_size=32",
         "--set", "eval.bootstrap.enabled=false"])

    run_dir = out / "baseline" / "ctt15" / "fdtcn" / "full" / "seed42"
    required = [run_dir / "metrics_test.json",
                run_dir / "predictions_test.npz",
                run_dir / "label_map.json",
                out / "baseline" / "tables" / "table_main.csv"]
    missing = [p for p in required if not p.exists() or p.stat().st_size == 0]
    if missing:
        for p in missing:
            print("MISSING:", p)
        raise SystemExit(1)

    print("\n=== SMOKE PASS ===")
    for p in required:
        print(f"  ok  {p.relative_to(ROOT)}  ({p.stat().st_size} bytes)")

    if not args.keep:
        shutil.rmtree(corpus, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
        print("（临时产物已清理；加 --keep 可保留）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
