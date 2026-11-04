#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_synthetic_corpus.py —— 生成合成语料，用于"无数据也能跑通全链路"
================================================================================
为什么需要它
------------
论文不发布任何经处理的数据集，读者手里没有 `02_tensors/*.npy`。
若仓库只能"读代码"，则无法证明 `train → eval → tables → figures` 真的能跑。
本脚本生成一份**结构完全一致、但内容为合成信号**的小语料（默认 3 capture ×
4,000 帧 / 16 维），让任何人在 5 分钟内把整条流水线跑一遍。

它证明的是**代码可执行**（L1），不是**结果可复现**（L2）——后者需要真实语料，
见仓库 README 的三级说明。合成数据的指标没有任何科学含义，禁止写入论文。

产物结构（与预处理产物同构）
----------------------------
  <out>/<dataset_dir>/
    02_tensors/{train,val,test}/<capture>/{v1.npy, v2.npy, label.npy}
    03_manifest/{splits.json, env.json, captures/<capture>.json}

用法
----
  python 代码/synth/make_synthetic_corpus.py --out 训练数据集_SYNTH
  # 然后在训练时把张量根指过去（不动真实产物）：
  python 代码/run_train.py --datasets ctt15 --models fdtcn --seeds 42 --no-figures \
      --set paths.tensors_root=训练数据集_SYNTH \
      --set project.output_root=outputs/synth_run \
      --set train.epochs=1

作者：小巴（WorkBuddy）    日期：2026-10-30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training import config as C          # noqa: E402

ATTACK_TYPES = ("DoS", "Fuzzing", "RpmSpoofing")


def _frames(n: int, rng: np.random.Generator, attack: np.ndarray,
            base_id: int) -> tuple[np.ndarray, np.ndarray]:
    """生成 [n,16] float16 与 [n,8,8] uint8。攻击帧的分布与正常帧不同（可学）。"""
    v1 = np.zeros((n, 16), dtype=np.float32)
    v1[:, 0] = rng.normal(0.010, 0.001, n)                 # dt_norm
    v1[:, 1] = rng.normal(0.0, 0.001, n)                   # dt_dev
    v1[:, 2] = (base_id % 8) / 8.0                         # id_norm
    v1[:, 3] = rng.normal(0.5, 0.1, n)                     # id_freq
    v1[:, 4] = 8 / 32                                      # payload_len_norm（CAN 8 字节）
    v1[:, 5] = 1 - 8 / 32                                  # pad_ratio
    v1[:, 8] = rng.normal(0.5, 0.05, n)                    # byte_mean
    v1[:, 9] = rng.normal(0.2, 0.02, n)                    # byte_std
    v1[:, 10] = rng.normal(2.0, 0.1, n)                    # byte_entropy
    v1[:, 11] = rng.normal(0.6, 0.05, n)                   # nonzero_ratio
    v1[:, 13] = rng.normal(0.0, 0.5, n)                    # payload_delta
    v1[:, 14] = rng.normal(0.0, 1.0, n)                    # payload_u32
    v1[:, 15] = rng.normal(0.9, 0.02, n)                   # period_stability

    # 攻击段：把 ID 与字节统计推离正常分布（合成"可学信号"）
    m = attack.astype(bool)
    if m.any():
        v1[m, 2] = ((base_id + 3) % 8) / 8.0
        v1[m, 9] = rng.normal(0.45, 0.05, int(m.sum()))
        v1[m, 10] = rng.normal(2.8, 0.2, int(m.sum()))
        v1[m, 11] = rng.normal(0.95, 0.02, int(m.sum()))
        v1[m, 15] = rng.normal(0.2, 0.05, int(m.sum()))

    v2 = rng.integers(0, 256, size=(n, 8, 8), dtype=np.uint8)
    v2[:, :, :8] = np.where(m[:, None, None], (v2[:, :, :8] | 0x0F), v2[:, :, :8])
    return np.nan_to_num(v1).astype(np.float16), v2


def build(out_root: Path, dataset_dir: str, n_captures: int, n_frames: int,
          seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cfg = C.load(str(Path(__file__).resolve().parent.parent / "training" / "configs" / "train.yaml"),
                 None)
    win = int(cfg["data"]["window"]["T"])
    stride_train = int(cfg["data"]["window"]["stride"]["train"])
    stride_test = int(cfg["data"]["window"]["stride"]["test"])
    channels = list(cfg["data"]["v1"]["channels"])

    base = out_root / dataset_dir
    tensors = base / "02_tensors"
    manifest = base / "03_manifest" / "captures"
    for sp in ("train", "val", "test"):
        (tensors / sp).mkdir(parents=True, exist_ok=True)
    manifest.mkdir(parents=True, exist_ok=True)

    # 三段各占 60% / 15% / 25%（与真实语料的语义划分近似）
    n_train = int(n_frames * 0.60)
    n_val = int(n_frames * 0.15)
    n_train -= n_train % stride_train
    n_val -= n_val % stride_test

    splits_doc: dict = {"_note": "synthetic corpus for pipeline smoke only; NOT for results",
                        "split_mode": "semantic", "window_T": win,
                        "stride": {"train": stride_train, "test": stride_test},
                        "captures": {}}
    for i in range(n_captures):
        name = f"synth_{i:02d}__synthetic_attack__{ATTACK_TYPES[i % len(ATTACK_TYPES)]}"
        attack = np.zeros(n_frames, dtype=np.uint8)
        # 三个攻击段，各 120 帧；位置按 60/15/25 的切分比例放置，
        # 保证 train/val/test 各自都有正窗口（否则 val 的正负样本校验会直接报错）
        for start in (int(n_frames * 0.30), int(n_frames * 0.65), int(n_frames * 0.85)):
            attack[start:start + 120] = 1
        v1, v2 = _frames(n_frames, rng, attack, base_id=0x100 + i)

        ranges = {"train": [[0, n_train]],
                  "val": [[n_train, n_train + n_val]],
                  "test": [[n_train + n_val, n_frames]]}
        for sp in ("train", "val", "test"):
            d = tensors / sp / name
            d.mkdir(parents=True, exist_ok=True)
            a, b = ranges[sp][0]
            np.save(d / "v1.npy", v1[a:b])
            np.save(d / "v2.npy", v2[a:b])
            np.save(d / "label.npy", attack[a:b])

        def n_win(length: int, stride: int) -> int:
            return 0 if length < win else 1 + (length - win) // stride

        splits_doc["captures"][name] = {
            "n_frames": n_frames, "split_mode": "semantic", "ranges": ranges,
            "summary": {"synthetic": True},
        }
        (manifest / f"{name}.json").write_text(json.dumps({
            "capture": name, "protocol": "CAN 2.0",
            "attack_type": ATTACK_TYPES[i % len(ATTACK_TYPES)],
            "attack_family": ATTACK_TYPES[i % len(ATTACK_TYPES)],
            "is_masquerade": False, "supervised_evaluable": True,
            "n_frames": n_frames,
            "splits": {
                "train": {"ranges": ranges["train"],
                          "windows_stride_train": n_win(n_train, stride_train),
                          "windows_stride_test": n_win(n_train, stride_test)},
                "val": {"ranges": ranges["val"],
                        "windows_stride_train": n_win(n_val, stride_train),
                        "windows_stride_test": n_win(n_val, stride_test)},
                "test": {"ranges": ranges["test"],
                         "windows_stride_train": n_win(n_frames - n_train - n_val, stride_train),
                         "windows_stride_test": n_win(n_frames - n_train - n_val, stride_test)},
            },
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    (base / "03_manifest" / "splits.json").write_text(
        json.dumps(splits_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (base / "03_manifest" / "env.json").write_text(json.dumps({
        "dataset": "synthetic", "generated_at": "2026-10-30",
        "config_fingerprint_sha256_16": "synthetic",
        "constants": {"WIN_T": win, "STRIDE_TRAIN": stride_train,
                      "STRIDE_TEST": stride_test, "PAD_BYTES": 32,
                      "V1_CHANNELS": channels, "SPLIT_MODE": "semantic"},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"root": str(base), "captures": n_captures, "frames": n_frames}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成合成语料（仅用于流水线 smoke）")
    ap.add_argument("--out", default="训练数据集_SYNTH")
    ap.add_argument("--dataset-dir", default="CTT_v1.5", help="与 paths.dataset_dirs 的值一致")
    ap.add_argument("--captures", type=int, default=3)
    ap.add_argument("--frames", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    info = build(Path(args.out), args.dataset_dir, args.captures, args.frames, args.seed)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
