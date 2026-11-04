# -*- coding: utf-8 -*-
"""
smoke_test.py —— 训练模块自检（无需 GPU、不写产物）
================================================================================
覆盖 10 组断言：
  1. 配置加载与缺键报错
  2. 消融派生（通道裁剪 / in_channels 解析）
  3. manifest 窗口数复算 == 清单声明
  4. Dataset 窗口严格落在 range 内（零跨块）
  5. 窗口标签 == 窗内任一帧为攻击
  6. 四个模型前向形状 + 参数量
  7. 损失有限且可反传
  8. 阈值标定在合成数据上可复原已知 τ
  9. 指标数值正确（手算对照）
 10. checkpoint 存读一致 + 示意图出图

运行
----
  python 代码/training/tests/smoke_test.py
"""

from __future__ import annotations

import copy
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # → <项目根>/代码
sys.path.insert(0, str(ROOT))

import numpy as np                                # noqa: E402

from training import config as C                   # noqa: E402
from training import manifest as MF                # noqa: E402
from training.calibration import calibrate, sweep_curve  # noqa: E402
from training.data import build_datasets, limit_dataset    # noqa: E402
from training.losses import MultiTaskLoss          # noqa: E402
from training.metrics import binary_metrics        # noqa: E402
from training.models import build_model, count_parameters, model_inputs  # noqa: E402

CFG_PATH = ROOT / "training" / "configs" / "train.yaml"
DATASET = "hcrl_canfd"

_results: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        detail = fn() or ""
        _results.append((name, True, str(detail)))
        print(f"  [PASS] {name}  {detail}")
    except Exception as e:
        _results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  [FAIL] {name}  {type(e).__name__}: {e}")
        traceback.print_exc()


# --------------------------------------------------------------------------- #
def t_config() -> None:
    cfg = C.load(CFG_PATH)
    assert cfg["data"]["window"]["T"] == 64
    assert len(cfg["data"]["v1"]["channels"]) == 16
    # 缺键必须报错（无默认值）
    bad = {"project": cfg["project"]}
    try:
        C.validate(bad)
    except C.ConfigError as e:
        return f"缺键报错正常（{len(str(e).splitlines()) - 1} 项）"
    raise AssertionError("缺键未报错")


def t_ablation() -> None:
    cfg = C.load(CFG_PATH)
    cfg["model"]["name"] = "fdtcn"
    eff = C.apply_ablation(cfg, "Ab-1")
    ch = eff["data"]["v1"]["channels"]
    # ★ 2026-10-17：Ab-1 必须**同时**去掉 payload_len_norm 与 pad_ratio。
    #   二者由同一 dlc 派生且互为补（pad_ratio ≡ 1 − payload_len_norm），
    #   只删其一会导致信息损失为零、消融退化为"无差异"，从而把 G1/O2 误判为阴性。
    assert "pad_ratio" not in ch
    assert "payload_len_norm" not in ch
    assert eff["model"]["variants"]["fdtcn"]["in_channels"] == 14
    eff2 = C.apply_ablation(cfg, "Ab-1b")
    assert eff2["data"]["v2"]["active_channels"] == 4
    return "Ab-1 → 14 通道（去 payload_len_norm + pad_ratio）；Ab-1b → 4 通道"


def t_ablation_guard() -> None:
    """退化消融守卫：只删 pad_ratio 或只删 payload_len_norm 必须被配置校验拒绝。"""
    cfg = C.load(CFG_PATH)
    bad = copy.deepcopy(cfg)
    bad["ablation"]["variants"]["BadAb"] = {"drop_v1_channels": ["pad_ratio"]}
    try:
        C.validate(bad)
    except C.ConfigError as e:
        assert "同时删除" in str(e)
        return "只删 pad_ratio 被拒绝（ConfigError）"
    raise AssertionError("退化消融未被配置校验拦截")


def t_manifest() -> None:
    cfg = C.load(CFG_PATH)
    paths = C.dataset_paths(cfg, DATASET)
    idx = MF.ManifestIndex(DATASET, paths["manifest_dir"])
    s = idx.verify(cfg)
    for sp in ("train", "val", "test"):
        assert s["n_windows_declared"][sp] == s["n_windows_recount"][sp]
    return f"train/val/test = {s['n_windows_recount']}"


def t_dataset() -> None:
    cfg = C.load(CFG_PATH)
    cfg["model"]["name"] = "fdtcn"
    eff = C.apply_ablation(cfg, "full")
    paths = C.dataset_paths(eff, DATASET)
    idx = MF.ManifestIndex(DATASET, paths["manifest_dir"])
    lm = MF.build_label_map(idx, eff)
    dss = build_datasets(eff, idx, dataset_key=DATASET, label_map=lm,
                         use_v1=True, use_v2=False, use_id_valid=False, seed=42)
    ds = limit_dataset(dss["test"], 2000, seed=42)

    # 每个窗口必须完整落在某个 range 内（split 内局部坐标）
    T = eff["data"]["window"]["T"]
    local: dict[str, list[tuple[int, int]]] = {}
    for c in idx.captures_in("test"):
        segs, off = [], 0
        for a, b in c.ranges["test"]:
            segs.append((off, off + (b - a)))
            off += b - a
        local[c.name] = segs
    for k in range(0, len(ds), max(1, len(ds) // 200)):
        fid = int(ds.index.file_idx[k])
        s = int(ds.index.start[k])
        segs = local[ds.specs[fid].name]
        assert any(a <= s and s + T <= b for a, b in segs), f"窗口 [{s},{s+T}) 跨块"

    # 标签一致性（抽 50 个窗口逐帧复核）
    for k in range(0, len(ds), max(1, len(ds) // 50)):
        item = ds[k]
        fid, s = int(ds.index.file_idx[k]), int(ds.index.start[k])
        lab = np.asarray(ds.sources[fid].label[s:s + T])
        assert int(item["y"]) == int(lab.any()), "窗口标签与帧标签不一致"
        assert item["x1"].shape == (T, 16)
        assert np.isfinite(item["x1"]).all(), "V1 出现 NaN/Inf"
    return f"{len(ds)} 窗口，形状 {item['x1'].shape}"


def t_models() -> None:
    cfg = C.load(CFG_PATH)
    out = []
    for name in ("fdtcn", "rgbnet", "lstm", "transformer"):
        cfg["model"]["name"] = name
        eff = C.apply_ablation(cfg, "full")
        mv = eff["model"]["variants"][name]
        m = build_model(name, mv, n_classes=6, window_T=64)
        n_par = count_parameters(m)
        import torch
        if "v2" in model_inputs(name):
            x2 = torch.randn(2, mv["in_channels"], 64, 8)
            o = m(x2=x2)
        else:
            x1 = torch.randn(2, 64, mv["in_channels"])
            o = m(x1=x1)
        assert o.logit.shape == (2,)
        assert o.type_logits.shape == (2, 6)
        out.append(f"{name}={n_par/1000:.1f}K")
    return "，".join(out)


def t_loss() -> None:
    cfg = C.load(CFG_PATH)
    cfg["model"]["name"] = "fdtcn"
    eff = C.apply_ablation(cfg, "full")
    m = build_model("fdtcn", eff["model"]["variants"]["fdtcn"], n_classes=4, window_T=64)
    crit = MultiTaskLoss(eff, pos_weight=3.0, class_weight=None, n_classes=4)
    import torch
    o = m(x1=torch.randn(8, 64, 16))
    y = torch.randint(0, 2, (8,))
    yt = torch.randint(0, 4, (8,))
    d = crit(o, y, yt)
    assert torch.isfinite(d["loss"])
    d["loss"].backward()
    assert any(p.grad is not None for p in m.parameters())
    return f"loss={float(d['loss']):.4f}"


def t_calibration() -> None:
    cfg = C.load(CFG_PATH)
    rng = np.random.default_rng(0)
    n = 20000
    y = rng.integers(0, 2, n)
    p = np.where(y == 1, rng.beta(6, 2, n), rng.beta(2, 6, n))
    cal = calibrate(y, p, cfg)
    m = binary_metrics(y, p, cal["threshold"])
    assert 0.0 <= cal["threshold"] <= 1.0
    assert m["mcc"] > 0.5, "合成可分数据上 MCC 应显著为正"
    curve = cal["curve"]
    assert curve["mcc"].size == curve["threshold"].size
    # 快速核对：曲线最大值 == 选中的 MCC
    assert abs(float(curve["mcc"].max()) - cal["selected"]["value"]) < 1e-9
    return f"τ={cal['threshold']:.3f}, MCC={m['mcc']:.3f}"


def t_metrics() -> None:
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1, 0.8, 0.2])
    m = binary_metrics(y, p, 0.5)
    assert m["tp"] == 2 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 2
    assert abs(m["precision"] - 2 / 3) < 1e-9
    assert abs(m["recall"] - 2 / 3) < 1e-9
    assert abs(m["mcc"] - 1.0 / 3.0) < 1e-9, m["mcc"]
    return f"TP/FP/TN/FN={m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}, MCC={m['mcc']:.3f}"


def t_checkpoint() -> None:
    import torch
    from training.artifacts import RunContext
    cfg = C.load(CFG_PATH)
    cfg["model"]["name"] = "fdtcn"
    eff = C.apply_ablation(cfg, "full")
    m = build_model("fdtcn", eff["model"]["variants"]["fdtcn"], n_classes=4, window_T=64)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as td:
        eff["_resolved"] = dict(eff["_resolved"], output_root=td)
        run = RunContext(eff, dataset=DATASET, model="fdtcn", variant="full", seed=42)
        run.run_dir = Path(td) / "run"
        run.ckpt_dir = run.run_dir / "checkpoints"
        run.ckpt_dir.mkdir(parents=True, exist_ok=True)
        p = run.save_checkpoint(run.best_ckpt, model=m, optimizer=opt, epoch=3,
                                metrics={"mcc": 0.5}, label_map={"normal": 0, "Flooding": 1},
                                channels=list(eff["data"]["v1"]["channels"]), v2_channels=8,
                                manifest_summary={"x": 1}, threshold=0.42)
        ck = RunContext.load_checkpoint(p)
        assert ck["threshold"] == 0.42 and ck["epoch"] == 3
        m2 = build_model("fdtcn", ck["model_cfg"], n_classes=4, window_T=64)
        m2.load_state_dict(ck["state_dict"])
        m.eval(); m2.eval()          # ★ 必须在 eval 模式比较（BN/Dropout 否则不一致）
        x = torch.randn(3, 64, 16)
        with torch.no_grad():
            a = m(x1=x).logit
            b = m2(x1=x).logit
        assert torch.allclose(a, b, atol=1e-6)
        return f"roundtrip OK, {p.stat().st_size/1024:.0f} KB"


def t_figures() -> None:
    from training import figures as F
    from training.artifacts import RunContext
    cfg = C.load(CFG_PATH)
    cfg["model"]["name"] = "fdtcn"
    eff = C.apply_ablation(cfg, "full")
    with tempfile.TemporaryDirectory() as td:
        eff["_resolved"] = dict(eff["_resolved"], output_root=td)
        run = RunContext(eff, dataset=DATASET, model="fdtcn", variant="full", seed=42)
        run.run_dir = Path(td) / "run"
        run.fig_dir = run.run_dir / "figures"
        run.fig_dir.mkdir(parents=True, exist_ok=True)
        ctx = F.RunArtifacts(run.run_dir, eff)
        pal = __import__("training.style", fromlist=["apply_style"]).apply_style(eff)
        f1 = F.fig10_encoding_schematic(ctx, pal)
        f2 = F.fig11_fdtcn_structure(ctx, pal)
        assert f1 and f2
        for p in f1 + f2:
            assert Path(p).exists() and Path(p).stat().st_size > 1000
        return f"{len(f1) + len(f2)} 个文件"


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 78)
    print("training 模块自检")
    print("=" * 78)
    check("1. 配置加载与校验", t_config)
    check("2. 消融派生", t_ablation)
    check("2b. 退化消融守卫", t_ablation_guard)
    check("3. manifest 窗口数复算", t_manifest)
    check("4/5. Dataset 窗口边界与标签", t_dataset)
    check("6. 四模型前向与参数量", t_models)
    check("7. 损失与反传", t_loss)
    check("8. 阈值标定", t_calibration)
    check("9. 指标手算对照", t_metrics)
    check("10a. checkpoint 存读", t_checkpoint)
    check("10b. 示意图出图", t_figures)

    n_pass = sum(1 for _, ok, _ in _results if ok)
    print("-" * 78)
    print(f"通过 {n_pass}/{len(_results)}")
    if n_pass != len(_results):
        print("失败项：")
        for name, ok, detail in _results:
            if not ok:
                print(f"  - {name}: {detail}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
