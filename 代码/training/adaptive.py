# -*- coding: utf-8 -*-
"""
training.adaptive —— 自适应阈值（在线分位数）回放
================================================================================
问题
----
当前论文/代码的阈值协议是"val 定一个 τ，test 原样用"。实测它不可部署：

  · CTT 名义 1% FPR，实际 1.84%（FD-TCN）/ 2.97%（FD-TCN-PF）/ 1.43%（LSTM）；
  · 原因是负样本分数分布随采集漂移（FD-TCN 的负样本均分 val 0.0005 → test 0.0019）。

一个固定 τ 在跨采集时既不能保证 FPR，也不能保证召回。工程上的标准解法是
**在线维护分数的经验分位数**，让阈值随分布缓慢移动。

做法
----
`simulate()` 按时间顺序回放 test 窗口（同 capture 内按 `pos` 升序），
维护最近 `window` 个窗口的分数环形缓冲，每 `step` 个窗口重算一次
τ = Quantile(buffer, 1 − target_rate)，并按块统计混淆计数。

注意口径
--------
默认 `mode="labeled"`：只把**延迟到达的已知负样本**放进缓冲（`label_delay` 个窗口后才
知道标签，模拟运维复核告警）。这是唯一不带"cap 以下无正例"假设的口径——它知道谁是负样本，
因此 τ 逼近真实负样本分布的 (1−α) 分位。

实测（2026-10-28，三语料 5 组 run，α=1%）：
  · CTT / FD-TCN：FPR 1.30%、召回 **0.511 → 0.635**（自适应把漂移的 τ 拉回来了）；
  · HCRL / FD-TCN：FPR 1.27%、召回 0.999（`capped` 只有 0.554，`labeled` 完全修好）；
  · ROAD / Transformer：FPR **60%**，缓冲加大到 65 536 才降到 15%，仍回不到 1%；
  · CTT / Transformer：FPR 17%，加大缓冲只到 8.8%。
后两组不是实现缺陷，而是分数分布本身退化：ROAD 负样本 **99% 恰好为 0**、1% 在 0.98–1.0，
1% FPR 的操作点正落在这个断崖上；任何有限历史的经验分位数都无法稳定分辨。
**结论：自适应阈值能救"漂移"，救不了"操作点在分布断崖上"，也不该被写成通用解。**

次选 `mode="capped"`：缓冲只接收 `p <= cap` 的样本，`cap = max(τ_val·cap_factor, τ_val)`。
它要解决的正是下面这个陷阱：

  × `mode="negative"`（"只收判为负的样本"）初版设计是**错的**。缓冲 = {p : p < τ_t}
    是一个被当前阈值截断的集合，其 (1−α) 分位必然低于真 (1−α) 分位 → τ 单调下漂 →
    更多样本被接纳 → 继续下漂，直到 `p < τ` 几乎不成立、缓冲冻结。实测高先验
    （30% 正例）下 FPR 从 6% 漂到 53%，召回却仍 95%（因为正例分数远在上方），
    即"召回看起来没坏、FPR 已经崩了"。该模式保留仅作回归对照，不得用于论文。

  × `mode="all"` 是告警率口径：高先验语料（CTT 8%、HCRL 34%）缓冲被攻击分数占据，
    τ 被顶高、召回崩塌（实测 CTT 51%→30%、HCRL 30%→19.7%）。

`capped` 模型的假设写在明处：**存在一个上限 cap，使"分数 ≤ cap 的样本里正例占比可忽略"**。
这个假设在实测语料上成立（正例分数中位远高于负例 99 分位），但它是假设，不是定理——
cap 取得过大就退化为 `all`，取得过小（≤ τ）则召回被人为压低。`params.cap` 会落盘备查。

作者：小巴（WorkBuddy）    日期：2026-10-28
"""

from __future__ import annotations

from typing import Any

import numpy as np


def temporal_order(cap: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """按 (capture, 窗口起点) 排序，得到可回放的时间顺序。"""
    cap = np.asarray(cap).ravel()
    pos = np.asarray(pos).ravel()
    return np.lexsort((pos, cap))


def simulate(y_true: np.ndarray, y_prob: np.ndarray, *,
             target_rate: float, window: int = 4096, step: int = 512,
             warmup: int | None = None, order: np.ndarray | None = None,
             fixed_threshold: float | None = None,
             mode: str = "labeled", cap: float | None = None,
             cap_factor: float = 3.0, label_delay: int | None = None,
             min_samples: int = 200) -> dict[str, Any]:
    """在线分位数阈值回放。

    参数
    ----
    target_rate : 期望误报率（对**负样本**而言），如 0.01
    window      : 环形缓冲长度（窗口数）
    step        : 每多少个窗口更新一次 τ（越小越灵敏，越大越省算力）
    warmup      : 初始缓冲长度；默认 = window
    order       : 时间顺序（`temporal_order` 的输出）；None 表示按原顺序
    fixed_threshold : val 标定的固定 τ；同时用于对照回放，并作为 `cap` / 冷启动的锚点
    mode        : "labeled"（默认）只接收**延迟到达的已知负样本**（推荐口径）；
                  "capped" 只接收 `p <= cap` 的样本 → 近似无截断的负样本分布，
                  但在 cap 饱和（τ_val > 1/3）或高先验时会退化成 all，实测会崩；
                  "all" 全部接收（告警率口径，高先验下崩）；
                  "negative" 只接收判负样本（**已证明有偏，仅作回归对照**）
    cap         : 上限；None 时由 `fixed_threshold · cap_factor` 推出
    cap_factor  : cap 相对 val τ 的倍数，默认 3.0
    label_delay : `labeled` 模式的标签延迟（窗口数），默认 = step
    min_samples : `labeled` 模式冷启动所需的最少负样本数，不足时用 `fixed_threshold`

    返回
    ----
    {"blocks": [...], "overall": {...}, "fixed": {...}, "params": {...}}
    """
    y = np.asarray(y_true).astype(np.int8).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    if y.size != p.size:
        raise ValueError("y 与 p 长度不一致")
    if not (0 < target_rate < 1):
        raise ValueError("target_rate 必须在 (0,1)")
    if mode not in ("capped", "all", "negative", "labeled"):
        raise ValueError("mode 仅支持 capped / labeled / all / negative")
    idx = np.arange(y.size) if order is None else np.asarray(order).ravel()
    y, p = y[idx], p[idx]

    cap_v = _resolve_cap(mode, cap, fixed_threshold, cap_factor)
    delay = int(max(0, label_delay if label_delay is not None else step))

    win = int(max(16, window))
    warm = int(min(win, warmup if warmup is not None else win))
    step = int(max(1, step))

    buf = np.empty(win, dtype=np.float64)
    n_buf = 0
    head = 0
    tau = float("nan")
    if mode == "labeled":
        # 冷启动：标签还没到，先用 val 标定的 τ 顶上（没有锚点就先用 0.5）
        tau = float(fixed_threshold) if (fixed_threshold is not None
                                        and np.isfinite(float(fixed_threshold))) else 0.5

    blocks: list[dict[str, Any]] = []
    cur = {"n": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for i in range(y.size):
        # 是否允许进入环形缓冲（四个模式的唯一差异就在这里）
        if mode == "all":
            ok = True
        elif mode == "capped":
            ok = bool(p[i] <= cap_v)
        elif mode == "labeled":
            ok = False                      # labeled 不走"当前样本"，见下方延迟入队
        else:  # negative：有偏，见模块 docstring
            ok = (not np.isfinite(tau)) or bool(p[i] < tau)

        if mode == "labeled":
            # ★ 延迟标签流：第 i−delay 个窗口的标签此刻才可知，只有判为负才进缓冲。
            #   这是唯一不带"cap 之内无正例"假设的口径——它知道谁是负样本。
            j = i - delay
            if j >= 0 and y[j] == 0:
                if n_buf < win:
                    buf[n_buf] = p[j]
                    n_buf += 1
                else:
                    buf[head] = p[j]
                    head = (head + 1) % win
            if n_buf >= int(max(16, min_samples)) and (i - warm) % step == 0:
                tau = float(np.quantile(buf[:n_buf], 1.0 - target_rate))
        elif n_buf < warm:
            if ok:
                buf[n_buf] = p[i]
                n_buf += 1
                if n_buf == warm:
                    tau = float(np.quantile(buf, 1.0 - target_rate))
        else:
            if ok:
                buf[head] = p[i]
                head = (head + 1) % win
            if (i - warm) % step == 0:
                tau = float(np.quantile(buf, 1.0 - target_rate))

        pred = 1 if (np.isfinite(tau) and p[i] >= tau) else 0
        cur["n"] += 1
        if y[i] == 1:
            cur["tp"] += pred
            cur["fn"] += 1 - pred
        else:
            cur["fp"] += pred
            cur["tn"] += 1 - pred
        if cur["n"] >= step:
            blocks.append(_block_row(cur, tau))
            cur = {"n": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    if cur["n"]:
        blocks.append(_block_row(cur, tau))

    overall = _sum_counts(blocks)
    out = {"blocks": blocks, "overall": overall,
           "params": {"target_rate": float(target_rate), "window": win,
                      "step": step, "warmup": warm, "mode": mode,
                      "cap": cap_v, "cap_factor": float(cap_factor),
                      "label_delay": (delay if mode == "labeled" else None),
                      "n_windows": int(y.size)}}

    if fixed_threshold is not None and np.isfinite(float(fixed_threshold)):
        t = float(fixed_threshold)
        pred = (p >= t).astype(np.int8)
        out["fixed"] = _counts(y, pred, t)
    return out


def _resolve_cap(mode: str, cap: float | None,
                 fixed_threshold: float | None, cap_factor: float) -> float:
    """确定缓冲接纳上限。`all` 模式不设限；`capped` 无锚点时退化为 1.0（等于 all）。"""
    if mode != "capped":
        return 1.0
    if cap is not None:
        return float(np.clip(cap, 0.0, 1.0))
    if fixed_threshold is not None and np.isfinite(float(fixed_threshold)):
        t = float(fixed_threshold)
        return float(np.clip(max(t * float(cap_factor), t), 0.0, 1.0))
    return 1.0


def _block_row(c: dict[str, int], tau: float) -> dict[str, Any]:
    row = dict(c)
    n_pos = c["tp"] + c["fn"]
    n_neg = c["fp"] + c["tn"]
    row["tau"] = float(tau)
    row["fpr"] = (c["fp"] / n_neg) if n_neg else float("nan")
    row["recall"] = (c["tp"] / n_pos) if n_pos else float("nan")
    return row


def _sum_counts(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("n", "tp", "fp", "tn", "fn")
    tot = {k: int(sum(b[k] for b in blocks)) for k in keys}
    tot["fpr"] = tot["fp"] / (tot["fp"] + tot["tn"]) if (tot["fp"] + tot["tn"]) else float("nan")
    tot["recall"] = tot["tp"] / (tot["tp"] + tot["fn"]) if (tot["tp"] + tot["fn"]) else float("nan")
    return tot


def _counts(y: np.ndarray, pred: np.ndarray, tau: float) -> dict[str, Any]:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"tau": float(tau), "n": int(y.size), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "fpr": (fp / (fp + tn)) if (fp + tn) else float("nan"),
            "recall": (tp / (tp + fn)) if (tp + fn) else float("nan")}
