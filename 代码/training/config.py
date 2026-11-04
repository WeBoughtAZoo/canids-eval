# -*- coding: utf-8 -*-
"""
training.config —— 配置加载、校验与派生
================================================================================
本模块是训练/评估/出图三段共用的**唯一**配置入口。

设计原则
--------
1. **无默认值**：任何键缺失立即抛 `ConfigError` 并列出缺失路径。
   代码里不存在 `cfg.get("lr", 1e-3)` 这种写法 —— 参数只允许来自 YAML。
2. **路径单点解析**：YAML 里全部写相对路径（相对 project.root），
   由 `resolve_paths()` 一次性转成绝对路径，并写回 `cfg["_resolved"]`。
3. **消融在此派生**：`apply_ablation()` 生成"实际生效"的配置副本
   （裁剪 V1 通道 / V2 掩码通道 / 替换损失），下游模型与数据只读派生结果。
4. **指纹**：对"会影响数值结果"的键做规范化 JSON 后取 sha256 前 16 位，
   写入 checkpoint 与 run_manifest，用于确认"两次跑的是不是同一套口径"。

作者：小巴（WorkBuddy）    日期：2026-10-17
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


# --------------------------------------------------------------------------- #
# 异常与基础工具
# --------------------------------------------------------------------------- #
class ConfigError(ValueError):
    """配置缺失或非法。"""


def _deep_merge(base: dict, patch: Mapping[str, Any]) -> dict:
    """把 patch 递归合并进 base 的副本；list / 标量直接覆盖。"""
    out = copy.deepcopy(base)
    for k, v in patch.items():
        # 允许新增键（消融里会新增 drop_v1_channels 等），但类型必须一致
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(dict(out[k]), v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def require(cfg: Mapping[str, Any], dotted: str) -> Any:
    """按点路径取值；缺失即报错（不做任何兜底）。"""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise ConfigError(f"配置缺少必需键：{dotted}")
        cur = cur[part]
    return cur


def get_opt(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """仅用于**非数值**的可选项（如可选开关）。数值参数禁止使用本函数。"""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _parse_scalar(text: str) -> Any:
    """把 CLI 覆盖值按 YAML 标量解析（"true"→bool、"1e-3"→float、"[a,b]"→list）。"""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def parse_overrides(pairs: Iterable[str] | None) -> dict:
    """把 `--set a.b=1 --set c=2` 解析成嵌套 dict。"""
    if not pairs:
        return {}
    root: dict = {}
    for item in pairs:
        if "=" not in item:
            raise ConfigError(f"--set 需要 key=value 形式，收到：{item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"--set 的键为空：{item}")
        node = root
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ConfigError(f"--set 键路径冲突：{key}")
        node[parts[-1]] = _parse_scalar(raw.strip())
    return root


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def validate(cfg: Mapping[str, Any], *, resolved: bool = False) -> None:
    """结构与语义校验；尽可能一次报出全部问题。

    resolved=True 用于 **run 快照**：此时 `data.v1.channels` 可能已被消融裁剪
    （如 Ab-1 为 15 个），因此只要求"1–16 个且无重复"，不再强制 16。
    """

    errs: list[str] = []

    def need(dotted: str) -> Any:
        try:
            return require(cfg, dotted)
        except ConfigError as e:
            errs.append(str(e))
            return None

    # ---- 路径 ----
    for k in ("project.root", "project.output_root", "project.run_name",
              "paths.tensors_root", "paths.tensor_subdir", "paths.manifest_subdir",
              "paths.dataset_dirs", "paths.run_dir_template"):
        need(k)

    # ---- 数据 ----
    win_T = need("data.window.T")
    stride = need("data.window.stride")
    if isinstance(stride, Mapping):
        for sp in ("train", "val", "test"):
            if sp not in stride:
                errs.append(f"配置缺少必需键：data.window.stride.{sp}")
        if stride.get("train", 0) <= 0 or stride.get("val", 0) <= 0:
            errs.append("data.window.stride 必须为正整数")
        if stride.get("test") != win_T:
            errs.append("data.window.stride.test 必须等于 window.T（保 McNemar 样本独立）")
    chans = need("data.v1.channels")
    if isinstance(chans, list):
        n_ch = len(chans)
        if resolved:
            if not (1 <= n_ch <= 16):
                errs.append(f"data.v1.channels 长度非法（{n_ch}），应在 1–16 之间")
        elif n_ch != 16:
            errs.append(f"data.v1.channels 必须为 16 个（当前 {n_ch} 个）")
        if len(set(chans)) != len(chans):
            errs.append("data.v1.channels 存在重复通道名")
    v2h = need("data.v2.height")
    if v2h is not None and win_T is not None and v2h != win_T:
        errs.append("data.v2.height 必须等于 data.window.T")
    if need("data.window.label_rule") != "any_frame":
        errs.append("data.window.label_rule 目前仅支持 any_frame（与预处理口径一致）")

    # ---- 模型 ----
    name = need("model.name")
    variants = need("model.variants")
    if isinstance(variants, Mapping) and isinstance(name, str) and name not in variants:
        errs.append(f"model.name={name} 不在 model.variants 中（可选 {sorted(variants)}）")
    if isinstance(variants, Mapping) and isinstance(name, str) and name in variants:
        mv = variants[name]
        if name == "fdtcn":
            bc, dil = mv.get("block_channels"), mv.get("dilations")
            if not isinstance(bc, list) or not isinstance(dil, list) or len(bc) != len(dil):
                errs.append("model.variants.fdtcn: block_channels 与 dilations 长度必须一致")
            if mv.get("kernel_size") is None or mv.get("kernel_size") % 2 == 0:
                errs.append("model.variants.fdtcn: kernel_size 必须为奇数")
        if need("data.v2.channels") is not None:
            n_v2 = cfg["data"]["v2"]["channels"]
            mask = cfg["data"]["v2"]["mask_channels"]
            if name == "rgbnet" and mv.get("in_channels") == "auto":
                if mask >= n_v2:
                    errs.append("data.v2.mask_channels 必须小于 data.v2.channels")

    # ---- 损失 / 优化 ----
    if need("loss.binary.type") not in ("weighted_bce", "focal"):
        errs.append("loss.binary.type 仅支持 weighted_bce / focal")
    if need("loss.multiclass.type") != "ce":
        errs.append("loss.multiclass.type 目前仅支持 ce")
    opt = need("train.optimizer.name")
    if opt not in ("adamw", "adam", "sgd"):
        errs.append("train.optimizer.name 仅支持 adamw / adam / sgd")
    if need("train.scheduler.name") not in ("cosine", "none"):
        errs.append("train.scheduler.name 仅支持 cosine / none")
    # betas 只在 adam 系生效
    if opt == "sgd" and isinstance(cfg["train"]["optimizer"], Mapping) \
            and "betas" in cfg["train"]["optimizer"]:
        errs.append("optimizer.name=sgd 时不应给出 betas")

    # ---- 训练 / 标定 ----
    if (need("train.epochs") or 0) <= 0:
        errs.append("train.epochs 必须 > 0")
    if (need("train.batch_size") or 0) <= 0:
        errs.append("train.batch_size 必须 > 0")
    if need("calibration.split") != "val":
        errs.append("calibration.split 必须是 val（禁止用测试集定阈值）")
    if need("calibration.metric") not in ("mcc", "f1", "fpr"):
        errs.append("calibration.metric 仅支持 mcc / f1 / fpr")
    grid = need("calibration.grid")
    if isinstance(grid, Mapping):
        if not (0 <= grid.get("min", 1) < grid.get("max", 0) <= 1):
            errs.append("calibration.grid 需满足 0 ≤ min < max ≤ 1")
        if grid.get("step", 0) <= 0:
            errs.append("calibration.grid.step 必须 > 0")

    # ---- 评估 ----
    for k in ("eval.metrics", "eval.splits", "eval.evaluability"):
        need(k)
    grade = cfg.get("eval", {}).get("evaluability", {})
    if isinstance(grade, Mapping) and not (
            grade.get("sufficient", 0) > grade.get("usable", 0) > grade.get("insufficient", 0) > 0):
        errs.append("eval.evaluability 需满足 sufficient > usable > insufficient > 0")
    if "test" not in (cfg.get("eval", {}).get("splits") or []):
        errs.append("eval.splits 必须包含 test")

    # ---- 图表 ----
    fmts = need("figures.formats")
    if isinstance(fmts, list):
        bad = [f for f in fmts if f not in ("pdf", "png", "svg", "eps")]
        if bad:
            errs.append(f"figures.formats 含不支持的格式：{bad}")

    if str(cfg.get("figures", {}).get("save_bbox", "tight")).lower() not in ("tight", "exact"):
        errs.append("figures.save_bbox 仅支持 tight / exact")

    # ---- 表格 / 配对检验 ----
    for k in ("tables.enabled", "tables.dir", "tables.formats", "tables.decimals",
              "tables.captions.main", "tables.labels.main"):
        need(k)
    tf = cfg.get("tables", {}).get("formats") or []
    bad_t = [f for f in tf if f not in ("tex", "csv")]
    if bad_t:
        errs.append(f"tables.formats 含不支持的格式：{bad_t}")
    need("compare.enabled")
    sc = cfg.get("compare", {}).get("scope") or []
    bad_s = [x for x in sc if x not in ("model", "ablation")]
    if bad_s:
        errs.append(f"compare.scope 含未知取值：{bad_s}（可选 model / ablation）")

    # ---- 消融 ----
    ab = need("ablation.variants")
    if isinstance(ab, Mapping):
        if need("ablation.default_variant") not in ab:
            errs.append("ablation.default_variant 不在 ablation.variants 中")
        # resolved=True 时 data.v1.channels 已被裁剪，消融配方里的通道名是与
        # **原始 16 通道**对齐的"操作记录"，不能再拿裁剪后的列表去校验。
        if not resolved:
            valid = set(chans or [])
            for vname, spec in ab.items():
                drop = (spec or {}).get("drop_v1_channels") or []
                bad = [c for c in drop if c not in valid]
                if bad:
                    errs.append(f"ablation.variants.{vname}.drop_v1_channels 含未知通道：{bad}")
                # ★ 退化消融守卫（2026-10-17）：
                #   pad_ratio = (32−dlc)/32 与 payload_len_norm = dlc/32 由同一 dlc 派生且互为补
                #   （实测 corr = −1.000000，max|(1−x)−y| = 0），只删其中一个则**信息损失为零**，
                #   该消融必然得到"无差异"，会把"填充伪影是否影响检测"误判为阴性结论。
                if ("pad_ratio" in drop) != ("payload_len_norm" in drop):
                    errs.append(
                        f"ablation.variants.{vname} 只删了 pad_ratio / payload_len_norm 中的一个："
                        "二者互为 1−x，必须同时删除（否则该消融零信息损失，结果恒为无差异）"
                    )

    if errs:
        raise ConfigError("配置校验失败（%d 项）：\n  - %s" % (len(errs), "\n  - ".join(errs)))


# --------------------------------------------------------------------------- #
# 加载与路径解析
# --------------------------------------------------------------------------- #
def load(config_path: str | Path, overrides: Mapping[str, Any] | None = None) -> dict:
    """读取 YAML → 合并 CLI 覆盖 → 校验 → 解析路径 → 附加 _meta。"""
    p = Path(config_path).resolve()
    if not p.exists():
        raise ConfigError(f"配置文件不存在：{p}")
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ConfigError(f"配置文件根节点必须是映射：{p}")

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    validate(cfg)
    cfg = resolve_paths(cfg, p)
    cfg["_meta"] = {
        "config_path": str(p),
        "config_sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
        "overrides": json.loads(json.dumps(overrides or {}, default=str)),
    }
    return cfg


def resolve_paths(cfg: dict, config_path: Path) -> dict:
    """把 project.root 与各相对路径解析为绝对路径（原地写回 + 生成 _resolved）。"""
    root = (config_path.parent / cfg["project"]["root"]).resolve()
    cfg["project"]["root"] = str(root)

    tensors_root = (root / cfg["paths"]["tensors_root"]).resolve()
    output_root = (root / cfg["project"]["output_root"]).resolve()
    cfg["paths"]["tensors_root"] = str(tensors_root)
    cfg["project"]["output_root"] = str(output_root)

    cfg["_resolved"] = {
        "project_root": str(root),
        "tensors_root": str(tensors_root),
        "output_root": str(output_root),
        "dataset_roots": {
            key: str(tensors_root / sub)
            for key, sub in cfg["paths"]["dataset_dirs"].items()
        },
    }
    return cfg


def load_snapshot(snapshot_path: str | Path, base_cfg: Mapping[str, Any]) -> dict:
    """读取某 run 的 `config_resolved.yaml`，并**补齐**后来新增的配置段。

    为什么需要兜底：配置 schema 会演进（例如新增 tables/compare 段），
    而历史 run 的快照里没有这些键。若直接 `C.load(snapshot)` 会因缺键失败，
    导致"旧 run 无法再评估"——这违背 checkpoint 可复现的初衷。
    做法：以**当前**配置为底，用快照覆盖（快照里的路径/通道/超参优先），
    再整体校验。既保留该 run 的真实口径，又能享受新增功能。
    """
    import yaml as _yaml
    snap_path = Path(snapshot_path)
    snap = _yaml.safe_load(snap_path.read_text(encoding="utf-8")) or {}
    snap.pop("_meta", None)
    merged = _deep_merge(copy.deepcopy(dict(base_cfg)), snap)
    meta = dict(base_cfg.get("_meta", {}))
    meta["snapshot"] = str(snap_path)
    meta["variant"] = str((_yaml.safe_load(snap_path.read_text(encoding="utf-8")) or {})
                          .get("_meta", {}).get("variant") or "full")
    merged["_meta"] = meta
    validate(merged, resolved=True)
    return merged


def dataset_paths(cfg: Mapping[str, Any], dataset: str) -> dict:
    """返回某数据集的张量根与清单根（含存在性检查）。"""
    dirs = cfg["paths"]["dataset_dirs"]
    if dataset not in dirs:
        raise ConfigError(f"未知数据集 {dataset}；可选 {sorted(dirs)}")
    base = Path(cfg["_resolved"]["tensors_root"]) / dirs[dataset]
    out = {
        "dataset_root": base,
        "tensors_dir": base / cfg["paths"]["tensor_subdir"],
        "manifest_dir": base / cfg["paths"]["manifest_subdir"],
    }
    for k, v in out.items():
        if not Path(v).exists():
            raise ConfigError(f"{dataset} 缺少目录 {k}：{v}")
    return out


# --------------------------------------------------------------------------- #
# 消融派生
# --------------------------------------------------------------------------- #
def active_v1_channels(cfg: Mapping[str, Any], variant: str) -> list[str]:
    """该消融变体下实际启用的 V1 通道（保持原顺序）。"""
    all_ch = list(cfg["data"]["v1"]["channels"])
    spec = cfg["ablation"]["variants"][variant] or {}
    drop = set(spec.get("drop_v1_channels") or [])
    keep = [c for c in all_ch if c not in drop]
    if not keep:
        raise ConfigError(f"消融 {variant} 把全部通道都删掉了")
    return keep


def active_v2_channel_index(cfg: Mapping[str, Any], variant: str) -> list[int]:
    """该变体下字节图实际使用的通道下标（默认前 n 个，或显式给出）。

    ★ 2026-10-28：CAN 2.0 语料的 DLC ≤ 8，V2 的后 24 字节与 3 个掩码通道恒零。
    为做"同构基线公平性"对照（RGBNet-8B），允许变体显式指定通道子集，
    例如 `v2_channel_index: [0, 4]`（数据 0–7 + 掩码 0–7）。
    """
    n = int(cfg["data"]["v2"]["channels"])
    spec = cfg["ablation"]["variants"][variant] or {}
    if spec.get("v2_channel_index") is not None:
        idx = [int(i) for i in spec["v2_channel_index"]]
    else:
        k = n - (int(cfg["data"]["v2"]["mask_channels"]) if spec.get("drop_v2_mask_channels") else 0)
        idx = list(range(k))
    if not idx:
        raise ConfigError(f"消融 {variant} 的字节图通道为空")
    if min(idx) < 0 or max(idx) >= n:
        raise ConfigError(f"消融 {variant} 的 v2_channel_index 越界（应在 0–{n - 1}）：{idx}")
    if len(set(idx)) != len(idx):
        raise ConfigError(f"消融 {variant} 的 v2_channel_index 有重复：{idx}")
    return idx


def active_v2_channels(cfg: Mapping[str, Any], variant: str) -> int:
    """该变体下字节图通道数（含/不含掩码，或显式子集）。"""
    return len(active_v2_channel_index(cfg, variant))


def apply_ablation(cfg: Mapping[str, Any], variant: str) -> dict:
    """生成"实际生效"的配置副本（含 active_channels 等派生字段）。

    - 裁剪 `data.v1.channels` 为实际启用列表 → 数据层按该列表索引
    - 记录 `data.v2.active_channels`
    - 合并变体里的覆盖项（如 loss.binary.type=focal）
    - 把 `model.variants.<name>.in_channels` 的 auto 解析为整数
    """
    if variant not in cfg["ablation"]["variants"]:
        raise ConfigError(f"未知消融变体 {variant}；可选 {sorted(cfg['ablation']['variants'])}")

    spec = copy.deepcopy(cfg["ablation"]["variants"][variant] or {})
    derived = copy.deepcopy(dict(cfg))
    derived.pop("_meta", None)

    keep = active_v1_channels(cfg, variant)
    v2_idx = active_v2_channel_index(cfg, variant)
    n_v2 = len(v2_idx)

    # 变体自身的覆盖（除通道裁剪外的任意配置项）
    overrides = {k: v for k, v in spec.items()
                 if k not in ("drop_v1_channels", "drop_v2_mask_channels",
                              "v2_channel_index")}
    if overrides:
        derived = _deep_merge(derived, overrides)

    derived["data"]["v1"]["channels"] = keep
    derived["data"]["v1"]["n_channels"] = len(keep)
    derived["data"]["v2"]["active_channels"] = n_v2
    derived["data"]["v2"]["channel_index"] = list(v2_idx)

    mv = derived["model"]["variants"][derived["model"]["name"]]
    if mv.get("in_channels") == "auto":
        mv["in_channels"] = n_v2 if derived["model"]["name"] == "rgbnet" else len(keep)
    else:
        # 显式给了 int 就必须与实际通道数一致，否则报错（防止消融忘记改）
        expected = n_v2 if derived["model"]["name"] == "rgbnet" else len(keep)
        if int(mv["in_channels"]) != expected:
            raise ConfigError(
                f"model.variants.{derived['model']['name']}.in_channels={mv['in_channels']} "
                f"与消融 {variant} 的实际通道数 {expected} 不一致（应写 auto）"
            )

    # ---- 变体级模型覆盖的一致性检查（Ab-13 改结构、Ab-14 改头） ----
    mname = str(derived["model"]["name"])
    mv2 = derived["model"]["variants"][mname]
    if mname == "fdtcn":
        bc, dl = mv2.get("block_channels"), mv2.get("dilations")
        if not (isinstance(bc, list) and isinstance(dl, list) and len(bc) == len(dl)):
            raise ConfigError(f"消融 {variant} 后 block_channels 与 dilations 长度不一致")
        if mv2.get("dual_head", True) is False:
            if bool(derived["label_space"]["multiclass"]):
                raise ConfigError(f"消融 {variant} 关闭双头时必须同时设 label_space.multiclass=false")
            if float(derived["loss"]["lambda_b"]) != 0.0:
                raise ConfigError(f"消融 {variant} 关闭双头时必须设 loss.lambda_b=0")

    derived["_variant"] = variant
    derived["_meta"] = dict(cfg.get("_meta", {}), variant=variant)
    return derived


# --------------------------------------------------------------------------- #
# 运行目录与指纹
# --------------------------------------------------------------------------- #
def resolve_run_dir(cfg: Mapping[str, Any], dataset: str, model: str,
                    variant: str, seed: int) -> Path:
    """按 `paths.run_dir_template` 生成运行目录（唯一决定模型落盘位置）。"""
    tmpl = cfg["paths"]["run_dir_template"]
    filled = tmpl.format(
        output_root=cfg["_resolved"]["output_root"],
        run_name=cfg["project"]["run_name"] or "run",
        dataset=dataset,
        model=model,
        variant=variant,
        seed=int(seed),
    )
    p = Path(filled)
    if not p.is_absolute():
        p = Path(cfg["_resolved"]["project_root"]) / p
    return p.resolve()


def artifact_path(cfg: Mapping[str, Any], run_dir: Path, key: str, **fmt: Any) -> Path:
    """按 artifacts.<key> 模板解析产物路径。"""
    node: Any = cfg["artifacts"]
    for part in key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise ConfigError(f"配置缺少必需键：artifacts.{key}")
        node = node[part]
    rel = str(node).format(**fmt) if fmt else str(node)
    return Path(run_dir) / rel


#: 影响数值结果、需进入指纹的键（顺序固定以保证可复现）
_FINGERPRINT_KEYS = (
    "data.window", "data.v1", "data.v2", "data.sampling", "data.limits",
    "data.label_shuffle",
    "label_space", "model", "loss", "train", "calibration", "eval.bootstrap",
    "eval.evaluability", "figures.png_dpi",
)


def fingerprint(cfg: Mapping[str, Any]) -> str:
    """对影响结果的键做规范化哈希（16 位），写入 checkpoint 便于核对口径。"""
    payload: dict = {}
    for key in _FINGERPRINT_KEYS:
        node: Any = cfg
        for part in key.split("."):
            node = node.get(part) if isinstance(node, Mapping) else None
            if node is None:
                break
        payload[key] = node
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
