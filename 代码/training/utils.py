# -*- coding: utf-8 -*-
"""
training.utils —— 复现性、日志与文件 I/O
================================================================================
复现性三件事（缺一不可）：
  1. 固定 python / numpy / torch / cuda 的随机源，并开启确定性算法；
  2. 把**解析后**的配置、库版本、预处理指纹、git 提交号写进 run_manifest；
  3. 断点（checkpoint）里存 label_map 与通道表 —— 评估时不需要再猜训练口径。

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """配置根日志：控制台 + 可选文件（UTF-8，避免 Windows GBK 报错）。"""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    fmt = logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # 第三方库的 INFO 噪声（fontTools 每次存 PDF 会打印字体子集化细节）
    for noisy in ("fontTools", "matplotlib.font_manager", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return logging.getLogger("canids")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"canids.{name}")


# --------------------------------------------------------------------------- #
# 随机性
# --------------------------------------------------------------------------- #
def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """固定全部随机源。deterministic=True 时关闭 cuDNN 自动寻优。"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # CPU 上的确定性算法（若某算子不支持会显式报错，比静默不确定更好）
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover
        pass


def configure_torch(cfg: Mapping[str, Any]) -> Any:
    """按配置设线程数并返回 torch 模块。"""
    import torch
    n = int(cfg["runtime"]["num_threads"])
    if n > 0:
        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(max(1, n // 2))
        except RuntimeError:
            pass  # 已初始化过则忽略
    return torch


# --------------------------------------------------------------------------- #
# JSON / CSV
# --------------------------------------------------------------------------- #
class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, set):
            return sorted(o)
        return super().default(o)


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, cls=_Encoder), encoding="utf-8")
    return p


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]],
              columns: list[str] | None = None) -> Path:
    """写 CSV（自动处理缺列与逗号）。"""
    import csv
    rows = list(rows)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    cols = columns or list(rows[0].keys())
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    return p


# --------------------------------------------------------------------------- #
# 环境快照
# --------------------------------------------------------------------------- #
def env_snapshot(cfg: Mapping[str, Any]) -> dict:
    """记录可复现所需的全部环境信息。"""
    import torch
    import sklearn
    import matplotlib
    import scipy
    import yaml as _yaml

    snap = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "pyyaml": _yaml.__version__,
        "device": str(cfg["runtime"]["device"]),
        "num_threads": int(cfg["runtime"]["num_threads"]),
        "deterministic": bool(cfg["runtime"]["deterministic"]),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        snap["cpu_count"] = os.cpu_count()
        snap["cpu_model"] = platform.processor()
    except Exception:  # pragma: no cover
        pass
    snap["git_commit"] = _git_commit(Path(cfg["_resolved"]["project_root"]))
    return snap


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # pragma: no cover
        return ""


# --------------------------------------------------------------------------- #
# 杂项
# --------------------------------------------------------------------------- #
class Timer:
    """上下文计时器：`with Timer() as t: ...` → t.elapsed 秒。"""

    def __init__(self) -> None:
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._t0


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


def file_size_mb(path: str | Path) -> float:
    p = Path(path)
    return p.stat().st_size / (1024.0 * 1024.0) if p.exists() else 0.0


def flatten_dict(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """把嵌套配置拍平，便于写 CSV / TensorBoard。"""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten_dict(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out
