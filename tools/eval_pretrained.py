# tools/eval_pretrained.py
# -*- coding: utf-8 -*-
"""官方预训练 RepViT 评价工具（考核第一个产出环节，规格书模块 5）。

Source  : Self-written（口径对齐 THU-MIG/RepViT main.py --eval 与 timm validate.py）
Third-party: torch / torchvision / timm / Pillow / numpy / matplotlib（均已在 requirements.txt）
        thop 与 fvcore 是**可选**依赖：只影响 MACs 字段，缺了不许崩（填 null 并打印提示）。

用法（配置走 --cfg，覆盖一律走 --set，键全部在 `pretrained_eval.` 段下）：

    python tools/eval_pretrained.py --cfg configs/pretrained_eval.yaml \
      --set pretrained_eval.model=repvit_m0_9 \
            pretrained_eval.weights=checkpoints/pretrained/repvit_m0_9_distill_300e.pth \
            pretrained_eval.data_list=datasets/lists/imagenetv2_mf_1000.txt \
            pretrained_eval.labels=labels/imagenet_classes.txt \
            pretrained_eval.out_dir=outputs/pretrained_eval/repvit_m0_9 \
            pretrained_eval.threads=4

八步流水线：读 list -> 建变换 -> 建模型 -> 载权重 -> 融合 -> 评测 -> 测延迟 -> 写盘。

产物（<out_dir> 下，默认 outputs/pretrained_eval/<model_name>/）：
    metrics.json / latency.json / top5_samples.json / predictions.csv / cases/*.png

本文件不含任何个人绝对路径：仓库根一律由 `ROOT = Path(__file__).resolve().parents[1]` 推导。
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                 # 允许 `python tools/eval_pretrained.py`
    sys.path.insert(0, str(ROOT))

from utils.config import apply_overrides, load_yaml      # noqa: E402
from tools.viz_style import setup_style, close_all       # noqa: E402

# --------------------------------------------------------------------------- #
# 常量：口径唯一真源，禁止在别处再写第二份
# --------------------------------------------------------------------------- #
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

INTERPOLATIONS = {
    "bicubic": InterpolationMode.BICUBIC,      # 官方与 timm 均为 bicubic（整数码 3）
    "bilinear": InterpolationMode.BILINEAR,
    "nearest": InterpolationMode.NEAREST,
}

# 白名单：timm 1.0.29 已注册的 RepViT 家族（m0_6 一并放开，权重不一定有）
REPVIT_WHITELIST = ("repvit_m0_6", "repvit_m0_9", "repvit_m1_0",
                    "repvit_m1_1", "repvit_m1_5", "repvit_m2_3")

# labels/imagenet_classes.txt 的四个锚点（规格书 5.2）：行号 -> 类别名
LABEL_ANCHORS = ((0, "tench"), (65, "sea snake"), (285, "Egyptian cat"),
                 (999, "toilet tissue"))

# 规格书 5.2 的类型与默认值表（YAML 里缺哪个键就用这里的默认值）
DEFAULTS: dict = {
    "model": None,          # str  | 必填 | timm 名或带 tag 的 repvit_m0_9.dist_300e_in1k
    "weights": None,        # str  | 本地 .pth；None -> timm/HF 下载
    "data_list": None,      # str  | 必填 | "<path>\t<label>" 文本
    "labels": None,         # str  | 必填 | 1000 行类别名
    "input_size": 224,      # int
    "batch_size": 64,       # int
    "out_dir": None,        # str  | None -> outputs/pretrained_eval/<model_name>
    "crop_pct": 0.875,      # float| 官方 256/224 口径（0.95 是 timm 口径）
    "interpolation": "bicubic",
    "impl": "auto",         # auto / official / timm
    "fuse": True,           # bool | 推理前做结构重参数化
    "device": "auto",       # auto / cpu / cuda / cuda:0
    "amp": False,           # bool | 仅 CUDA 生效
    "threads": None,        # int  | None -> 打印实际线程数但不修改
    "warmup": 10,           # int
    "runs": 50,             # int
    "check_data": False,    # bool | 只校验 list/labels 对齐，不加载模型
    "data_root": None,      # str  | 可选：data_list 中相对路径的基准目录
    # configs/pretrained_eval.yaml 额外给的「口径对照」键：两套 crop_pct 各跑一遍，
    # 用于量化 Resize 256 vs 235 对 Top-1 的影响；与主 crop_pct 重复的那次直接复用主结果。
    "also_eval_crop_pct": None,   # list[float] | 例 [0.875, 0.95]
}

# 数值/布尔键的强制转型表（YAML 里写成字符串或数字都必须收敛到同一类型）
_INT_KEYS = ("input_size", "batch_size", "warmup", "runs")
_FLOAT_KEYS = ("crop_pct",)
_BOOL_KEYS = ("fuse", "amp", "check_data")


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on", "y")


def _as_int(v):
    return None if v is None else int(v)


# --------------------------------------------------------------------------- #
# 1. 配置
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser("RepViT 官方预训练评价（--cfg + --set 两种通道，无独立 flag）")
    ap.add_argument("--cfg", required=True,
                    help="YAML 配置（唯一配置通道），例 configs/pretrained_eval.yaml")
    # 必须用 action="extend"：argparse 对同名选项默认是「后者覆盖前者」，
    # 而 tools/run_all_pretrained.py 会分两批传 --set（model/weights/out_dir 一批，
    # 公共口径一批）。若用默认行为，第二批会把第一批整个吃掉 -> 静默用 YAML 里的
    # model/weights 评测出一个「看起来正常但型号不对」的结果。extend 把多批拼成一个列表。
    ap.add_argument("--set", nargs="*", default=[], dest="set", action="extend",
                    help="点号路径覆盖，例 pretrained_eval.model=repvit_m0_9（可给多批）")
    return ap.parse_args(argv)


def load_eval_config(cfg_path: str, overrides: list[str] | None = None) -> tuple[dict, dict]:
    """返回 (完整 cfg, 已收敛类型的 pretrained_eval 段)。"""
    cfg = load_yaml(cfg_path)                       # 内部已处理 _base_ 与「文件不存在」
    if overrides:
        apply_overrides(cfg, list(overrides))
    raw = cfg.get("pretrained_eval") or {}
    if not isinstance(raw, dict):
        raise SystemExit("[错误] 配置里的 `pretrained_eval:` 段必须是键值映射")

    seg = dict(DEFAULTS)
    unknown = [k for k in raw if k not in DEFAULTS]
    if unknown:
        print(f"[warn] pretrained_eval 段出现未知键（已忽略）: {unknown}")
    seg.update({k: v for k, v in raw.items() if k in DEFAULTS})
    for k in _INT_KEYS:
        seg[k] = _as_int(seg[k])
    for k in _FLOAT_KEYS:
        seg[k] = float(seg[k])
    for k in _BOOL_KEYS:
        seg[k] = _as_bool(seg[k])
    if seg["threads"] is not None:
        seg["threads"] = int(seg["threads"])
    if seg["also_eval_crop_pct"] is not None:
        seg["also_eval_crop_pct"] = [float(v) for v in seg["also_eval_crop_pct"]]
    cfg["pretrained_eval"] = seg
    return cfg, seg


def repo_path(p) -> Path:
    """把配置里的相对路径收敛到仓库根（脚本从任意 cwd 调用都能命中，且不写死绝对路径）。"""
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def model_bare_name(model: str) -> str:
    """去掉权重 tag：repvit_m0_9.dist_300e_in1k -> repvit_m0_9（两种取值落到同一输出目录）。"""
    return str(model).split(".")[0]


def resolve_out_dir(seg: dict) -> Path:
    out = seg.get("out_dir") or f"outputs/pretrained_eval/{model_bare_name(seg['model'])}"
    p = Path(out)
    return p if p.is_absolute() else (ROOT / p)


def path_for_record(p) -> str:
    """写进 JSON/CSV 的路径一律用「相对仓库根的 POSIX 字符串」，避免泄漏个人绝对路径。"""
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return p.as_posix()


# --------------------------------------------------------------------------- #
# 2. 数据读取
# --------------------------------------------------------------------------- #
def parse_list_line(line: str):
    """解析一行 `<path><分隔><label>`，返回 (path, label) 或 None（空行/注释）。

    必须 `rsplit(TAB, 1)` 再回落 `rsplit(None, 1)`：
    用 `line.split()` 遇到含空格的 Windows 中文路径会切错段，导致标签整体错位、Top-1 塌到 0.1%
    （规格书 5.10 第 5 条）。逗号分隔是给 `--set` 手写小清单留的兼容口。
    """
    s = line.rstrip("\n").rstrip("\r")
    if not s.strip() or s.lstrip().startswith("#"):
        return None
    if "\t" in s:
        p, lab = s.rsplit("\t", 1)
    elif s.rstrip().rsplit(",", 1)[-1].strip().lstrip("-").isdigit():
        p, lab = s.rstrip().rsplit(",", 1)
    else:
        p, lab = s.rsplit(None, 1)
    p, lab = p.strip().strip('"'), lab.strip()
    if not p or not lab.lstrip("-").isdigit():
        return None
    return p, int(lab)


def read_data_list(path: str) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            r = parse_list_line(line)
            if r is None:
                if line.strip():
                    raise SystemExit(f"{path}:{lineno} 无法解析（需要 '<path><TAB><label>'）: {line!r}")
                continue
            rows.append(r)
    if not rows:
        raise SystemExit(f"[错误] {path} 为空")
    return rows


def read_labels(path: str, expect_lines: int = 1000) -> list[str]:
    """按行读类别名（过滤空行），行数必须 == expect_lines（ImageNet 是 1000）。"""
    with open(path, "r", encoding="utf-8") as f:
        names = [l.strip() for l in f if l.strip()]
    if len(names) != expect_lines:
        raise SystemExit(f"[错误] labels 文件 {path} 有 {len(names)} 行，断言要求 {expect_lines} 行"
                         f"（模型的 1000 类输出下标必须与标签行号一一对应）")
    bad = [(i, names[i], want) for i, want in LABEL_ANCHORS if names[i] != want]
    if bad:
        raise SystemExit("[错误] labels 锚点校验失败（标签整体错位会导致 Top-1≈0.1%）："
                         f"{[(i, got, want) for i, got, want in bad]}")
    return names


def resolve_image_path(raw: str, data_root: str | None = None) -> Path | None:
    """把 list 里的路径解析成真实文件路径；找不到返回 None（调用方记入 bad 列表）。

    依次尝试：绝对路径 -> data_root -> 仓库根 -> data/imagenet/val。
    最后一项是兜底：清单里若写相对 ``data/imagenet/val`` 的旧式路径（ImageNet 官方 val 目录结构）
    仍能解析；当前口径的 ImageNetV2 清单写的是**相对仓库根**的完整路径。
    """
    p = Path(raw)
    if p.is_absolute():
        return p if p.exists() else None
    cands = []
    if data_root:
        cands.append(Path(data_root) / p)
    cands.append(ROOT / p)
    cands.append(ROOT / "data" / "imagenet" / "val" / p)
    for c in cands:
        if c.exists():
            return c
    return None


def scan_images(rows: list[tuple[str, int]], data_root: str | None = None):
    """预扫描：过滤坏图/缺图，统计 PIL mode 分布。

    返回 (entries, bad, modes)，entries 元素为 dict(path,label,index)，
    index 是**list 的行号**（0-based）——predictions.csv 的 index 列就是它。
    保持 `ImageFile.LOAD_TRUNCATED_IMAGES = False`：宁可把坏图剔除并声明，
    也不要静默产生半张图（规格书 5.10 第 1 条）。
    """
    assert ImageFile.LOAD_TRUNCATED_IMAGES is False, "LOAD_TRUNCATED_IMAGES 必须保持 False"
    entries, bad, modes = [], [], Counter()
    for i, (raw, lab) in enumerate(rows):
        p = resolve_image_path(raw, data_root)
        if p is None:
            bad.append({"index": i, "path": raw, "error": "FileNotFoundError"})
            continue
        try:
            with open(p, "rb") as f:            # 必须先 load()，否则延迟解码的坏图测不出来
                img = Image.open(f)
                img.load()
            modes[img.mode] += 1
        except Exception as e:                  # noqa: BLE001 - 坏图类型不可预期
            bad.append({"index": i, "path": path_for_record(p), "error": f"{type(e).__name__}: {e}"})
            continue
        entries.append({"path": p, "label": int(lab), "index": i})
    return entries, bad, modes


class ListImageDataset(Dataset):
    """按 list 顺序取图；额外返回行号，供 `all_pred[idxs] = pred` 回填。

    即使将来有人误开 shuffle=True，数值结果依然正确（只是不再是顺序预测）。
    """

    def __init__(self, entries: list[dict], transform=None):
        self.entries = entries
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        e = self.entries[i]
        with open(e["path"], "rb") as f:
            img = Image.open(f)
            img.load()
        # exif_transpose 纠正手机拍摄的 Orientation；convert('RGB') 处理 RGBA/L/P
        # 注意：torchvision 的 pil_loader 默认不做 EXIF 纠正，报告里需声明此差异
        img = ImageOps.exif_transpose(img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(e["label"]), int(e["index"])


# --------------------------------------------------------------------------- #
# 3. 预处理（数值级，规格书 5.3.2）
# --------------------------------------------------------------------------- #
def build_eval_transform(input_size: int = 224, crop_pct: float = 0.875,
                         interpolation: str = "bicubic"):
    """返回 (transform, scale)。

    scale = floor(input_size / crop_pct)：
        crop_pct=0.875 -> floor(224/0.875)=256  （官方 THU-MIG 口径，默认）
        crop_pct=0.95  -> floor(224/0.95) =235  （timm pretrained_cfg 口径）
    Resize 只按短边缩放、长边等比，所以 CenterCrop 的窗口一定落在原始像素内。
    """
    assert interpolation in INTERPOLATIONS, f"interpolation 只能取 {list(INTERPOLATIONS)}"
    scale = math.floor(input_size / crop_pct)
    interp = INTERPOLATIONS[interpolation]
    tf = transforms.Compose([
        transforms.Resize(scale, interpolation=interp),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),                       # HWC uint8 -> CHW float32 且除以 255
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf, scale


# --------------------------------------------------------------------------- #
# 4. 模型：构建 / 权重判型 / 加载 / 融合
# --------------------------------------------------------------------------- #
def load_official_module():
    """exec 加载 models/repvit_official.py。

    绝不 `import models.repit_official`：该文件厂函数本体与 timm 注册表同名，
    正常 import 会覆盖 timm 的 repvit_* 注册项（本仓库已删掉 @register_model 装饰器，
    但 exec 到独立命名空间仍是最稳的隔离方式，见规格书 2.1 约束三）。
    """
    path = ROOT / "models" / "repvit_official.py"
    if not path.exists():
        raise SystemExit(f"[错误] 缺少官方实现 {path}")
    src = path.read_text(encoding="utf-8")
    ns = {"__name__": "repvit_official_vendored", "__file__": str(path)}
    exec(compile(src, str(path), "exec"), ns)                     # noqa: S102
    return ns


def sniff_state_dict(path: Path) -> dict:
    """权重判型：官方 `*_distill_300e.pth` vs 官方 `*_timm.pth`（规格书 5.2/5.10 第 8 条）。

    判据（两条同时打印，便于报告里引用）：
      * 顶层是否有 'model' 键 -> 官方发布格式 {'model': sd, ...}，且 storage 标在 cuda:0，
        所以必须 map_location='cpu' 再取 ['model']，否则 CPU 机器报
        "Attempting to deserialize object on a CUDA device"；
      * 首个键是否等于 'features.0.0.c.weight' -> 官方命名（timm 命名首键是 'stem.conv1.c.weight'）。
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    has_container = isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict)
    sd = ckpt["model"] if has_container else ckpt
    if not isinstance(sd, dict) or not sd:
        raise SystemExit(f"[错误] 无法从 {path} 解析出 state_dict")
    keys = list(sd.keys())
    first = keys[0]
    style = "official" if (first == "features.0.0.c.weight" or has_container) else "timm"
    distillation = any(("classifier_dist" in k) or ("head_dist" in k) for k in keys)
    return {"state_dict": sd, "n_keys": len(keys), "first_key": first,
            "container": bool(has_container), "style": style,
            "distillation": bool(distillation)}


def build_model(model_name: str, num_classes: int, style: str,
                distillation: bool, pretrained: bool):
    """按 style 建网：official -> vendored 官方实现；timm -> timm.create_model。"""
    bare = model_bare_name(model_name)
    if style == "official":
        ns = load_official_module()
        ctor = ns.get(bare)
        if ctor is None:
            raise SystemExit(f"[错误] 官方实现里没有工厂函数 {bare}()")
        model = ctor(num_classes=num_classes, distillation=bool(distillation))
        return model, None

    import timm
    model = timm.create_model(model_name if "." in str(model_name) else bare,
                              pretrained=bool(pretrained), num_classes=num_classes,
                              distillation=bool(distillation))
    return model, timm


def load_weights_into(model: nn.Module, sd: dict) -> dict:
    """按形状过滤后 strict=False 加载。

    `strict=False` **不能**豁免 shape mismatch：换 num_classes 时必须先按形状过滤，
    否则会抛 "size mismatch for classifier.classifier.l.weight"（规格书 5.10 第 6 条）。
    """
    own = model.state_dict()
    filtered, dropped = {}, []
    for k, v in sd.items():
        if k in own and tuple(v.shape) == tuple(own[k].shape):
            filtered[k] = v
        elif k in own:
            dropped.append(k)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    assert not unexpected, f"加载后出现 unexpected 键（形状过滤应当已排除）: {list(unexpected)[:5]}"
    return {"n_loaded": len(filtered), "n_dropped_shape": len(dropped),
            "dropped_shape": dropped[:20], "missing": list(missing)[:20],
            "n_missing": len(missing), "unexpected": list(unexpected)[:20]}


def fuse_model(model: nn.Module, impl: str, official_ns: dict | None):
    """结构重参数化：official -> utils.replace_batchnorm；timm -> model.fuse()。"""
    if impl == "official":
        assert official_ns is not None and "replace_batchnorm" in official_ns
        official_ns["replace_batchnorm"](model)           # 顶层 RepViT 没有 fuse() 方法
    else:
        assert hasattr(model, "fuse"), "timm RepVit 应有顶层 fuse() 方法"
        model.fuse()
    return model


def count_params(model: nn.Module) -> tuple[int, int]:
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return total, trainable


# --------------------------------------------------------------------------- #
# 5. 精度与延迟
# --------------------------------------------------------------------------- #
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> list:
    """与 timm.utils.metrics.accuracy 同语义：返回**百分数**，maxk 用 min 防越界。"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader, device: str, num_total: int,
                  num_classes: int) -> dict:
    """逐 batch 前向。

    下载/写盘前必须 model.eval()：双头分类头在 train 模式返回 tuple，
    且 BN 会用 batch 统计量让输出随 batch 组成漂移。
    准确率必须按样本数加权累计（把 batch 百分比还原成正确个数再累加），
    对每个 batch 的准确率求算术平均在尾批不满时会偏差约 0.5 个百分点（规格书 5.3.4）。
    """
    model.eval()
    all_pred = np.full(num_total, -1, dtype=np.int64)
    all_top5 = np.full((num_total, 5), -1, dtype=np.int64)
    all_top5p = np.zeros((num_total, 5), dtype=np.float32)
    keep_probs = num_total * num_classes <= 5 * 10 ** 7
    probs_all = np.zeros((num_total, num_classes), dtype=np.float32) if keep_probs else None
    correct1 = correct5 = 0.0
    n_seen = 0
    for x, y, idxs in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        if isinstance(logits, (tuple, list)):      # 兜底：万一蒸馏头没关掉
            logits = (logits[0] + logits[1]) / 2
        bs = int(y.size(0))
        a1, a5 = accuracy(logits, y, (1, 5))       # 百分数
        correct1 += float(a1.item()) * bs / 100.0
        correct5 += float(a5.item()) * bs / 100.0
        probs = torch.softmax(logits.float(), dim=1).cpu()
        top5p, top5i = probs.topk(5, dim=1, largest=True, sorted=True)
        rows = idxs.numpy()
        all_pred[rows] = top5i[:, 0].numpy()
        all_top5[rows] = top5i.numpy()
        all_top5p[rows] = top5p.numpy()
        if probs_all is not None:
            probs_all[rows] = probs.numpy()
        n_seen += bs
    return {"all_pred": all_pred, "all_top5": all_top5, "all_top5p": all_top5p,
            "probs_all": probs_all, "n_seen": n_seen,
            "correct1": correct1, "correct5": correct5}


def _cpu_model() -> str:
    """CPU 型号：Windows 读 PROCESSOR_IDENTIFIER，Linux 读 /proc/cpuinfo，最后回落 platform。"""
    if os.name == "nt":
        return os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()) or "unknown"
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="ignore"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def benchmark_latency(model: nn.Module, input_size=(3, 224, 224), warmup: int = 10,
                      runs: int = 50, device: str = "cpu", threads: int | None = None,
                      amp: bool = False) -> dict:
    """单张输入、batch=1、纯前向的延迟测量。返回扁平 dict（latency.json 的 schema）。

    torch.cuda.synchronize() 只在计时前后各一次：否则测到的是内核入队时间而非执行时间。
    time.perf_counter() 是单调时钟，不受系统对时跳变影响。
    """
    if threads is not None:
        torch.set_num_threads(int(threads))
    use_cuda = str(device).startswith("cuda")
    model.eval().to(device)
    x = torch.randn(1, *input_size, device=device)      # 固定输入，跨型号可比
    ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if (amp and use_cuda) \
        else (lambda: torch.autocast("cpu", enabled=False))

    with torch.no_grad():
        for _ in range(int(warmup)):                    # 预热：吃掉 JIT/内存分配/首访页代价
            with ctx():
                model(x)
        if use_cuda:
            torch.cuda.synchronize()
        ts = []
        for _ in range(int(runs)):
            if use_cuda:
                torch.cuda.synchronize()                # 计时前清空队列
            t0 = time.perf_counter()
            with ctx():
                model(x)
            if use_cuda:
                torch.cuda.synchronize()                # 必须同步后再取时间
            ts.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(ts, dtype=np.float64)
    return {
        "warmup": int(warmup), "runs": int(runs),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()), "max_ms": float(arr.max()),
        "threads": int(torch.get_num_threads()),
        "device": str(device), "cpu_model": _cpu_model(),
        "os": platform.platform(), "torch_version": torch.__version__,
        "ep": None, "ort_version": None,      # PyTorch 路线无 ONNX Runtime，键保留以便统一解析
    }


def count_macs(model: nn.Module, input_size: int = 224) -> tuple[float | None, str]:
    """MACs（不是 FLOPs）：优先 thop，其次 fvcore，都没有就返回 (None, 原因)。

    实测 repvit_m0_9 @ 1x3x224x224：thop macs = 847,050,816 (0.847 GMACs)。
    官方 README 的 0.8G 来自 fvcore 的 FlopCountAnalysis（SE 分支的 mean/sigmoid/mul
    与残差 add 未注册计数器被漏计，实测 832,165,824）——差异是**算子计数范围**，
    不是精度问题。严禁缺包就崩：MACs 只是报告的附属字段。
    """
    probe = copy.deepcopy(model).cpu().eval()
    x = torch.randn(1, 3, input_size, input_size)
    try:
        from thop import profile
        macs, _ = profile(probe, inputs=(x,), verbose=False)
        return float(macs) / 1e9, "thop"
    except Exception as e_thop:                       # noqa: BLE001
        pass
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(probe, x).total()
        return float(flops) / 1e9, "fvcore"
    except Exception as e_fv:                         # noqa: BLE001
        print(f"[warn] 未能统计 MACs（thop/fvcore 均不可用 -> {type(e_fv).__name__}），"
              f"metrics.json 的 macs_g 记为 null")
        return None, "unavailable"


# --------------------------------------------------------------------------- #
# 6. 案例挑选与可视化（规格书 5.8）
# --------------------------------------------------------------------------- #
def pick_cases(probs: np.ndarray, targets, preds, paths, names, n_each: int = 2):
    """正确案例取「Top-1 概率最高的」；错误案例取「Top-1 概率最高的错误样本」。

    为什么必须是代码挑选而不是手工挑：手工挑图会无意识偏向「看起来漂亮」的样本，
    而考核要求不得只展示成功案例。把准则写成代码，挑选过程本身就成了可复现的证据。
    """
    conf = probs.max(1)
    correct = np.asarray(preds) == np.asarray(targets)

    def pack(i: int) -> dict:
        order = probs[i].argsort()[::-1][:5]
        return {"image_path": path_for_record(paths[i]),
                "true_label_idx": int(targets[i]),
                "true_label_name": names[int(targets[i])],
                "top5": [{"idx": int(j), "name": names[int(j)],
                          "prob": float(probs[i][j])} for j in order],
                "correct": bool(correct[i])}

    hi_correct = sorted([i for i in range(len(preds)) if correct[i]], key=lambda i: -conf[i])[:n_each]
    hi_wrong = sorted([i for i in range(len(preds)) if not correct[i]], key=lambda i: -conf[i])[:n_each]
    return [pack(i) for i in hi_correct], [pack(i) for i in hi_wrong]


def plot_case_grid(sample: dict, save_path) -> Path:
    """单图：左原图 + 右 Top-5 横向条形图（绿 = 真实类，红 = 其余）。

    实现在本文件内自带：tools/visualize.py 的冻结 API 只暴露
    load_classes/load_pred_csv/row_normalized_cm/cat_dog_block/main，
    没有 plot_case_grid，而冻结文件不允许改动。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_style()
    img = ImageOps.exif_transpose(Image.open(sample["image_path"])).convert("RGB").resize((384, 384))
    names = [t["name"] for t in sample["top5"]]
    probs = [t["prob"] * 100 for t in sample["top5"]]
    colors = ["#2e7d32" if t["idx"] == sample["true_label_idx"] else "#c62828"
              for t in sample["top5"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=120)
    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title(f"true: {sample['true_label_name']}\npred: {sample['top5'][0]['name']} "
                  f"({probs[0]:.1f}%)", fontsize=11)
    y = list(range(len(names)))[::-1]
    ax2.barh(y, probs, color=colors)
    ax2.set_yticks(y)
    ax2.set_yticklabels([n[:34] for n in names], fontsize=9)
    ax2.set_xlabel("softmax prob (%)")
    ax2.set_xlim(0, 100)
    for yy, p in zip(y, probs):
        ax2.text(min(p + 1.5, 92), yy, f"{p:.2f}%", va="center", fontsize=8)
    ax2.set_title("Top-5 (green = true class)", fontsize=11)
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------- #
# 7. check_data 模式（规格书 5.9 第 1 条）
# --------------------------------------------------------------------------- #
def check_data_mode(seg: dict) -> int:
    """只校验 list 与 labels 的对齐并打印诊断，**不加载模型**，然后退出。"""
    if not seg["data_list"] or not seg["labels"]:
        raise SystemExit("[错误] check_data 需要 pretrained_eval.data_list 与 pretrained_eval.labels")
    for req in ("data_list", "labels"):
        if not repo_path(seg[req]).exists():
            raise SystemExit(f"[错误] 找不到 pretrained_eval.{req}: {repo_path(seg[req])}"
                             f"（相对路径按仓库根解析；ImageNetV2 固定子集由 "
                             f"datasets/make_imagenetv2_subset.py 生成）")
    rows = read_data_list(repo_path(seg["data_list"]))
    labs = [y for _, y in rows]
    lo, hi = min(labs), max(labs)
    # 判据不能只看 min/max：子集抽样可能触不到 0 与 999 两个端点，
    # 所以 min==0 即认定 0-based，min>=1 才提示 1-based（规格书 5.9 第 2 条）。
    if lo < 0:
        based = "未知（出现负标签，检查 list 解析）"
    elif lo == 0 and hi == len(labs) - 1:
        based = "0-based"
    elif lo == 0:
        based = "0-based（子集未覆盖到端点，max<类别数-1 属正常）"
    else:
        based = "疑似 1-based（需减 1）"
    print(f"[list] {len(rows)} lines, label range {lo}..{hi}, 判定 {based}")

    entries, bad, modes = scan_images(rows, seg.get("data_root"))
    print(f"[list] 发现 {len(bad)} 张问题图"
          + (f"（前 3 条: {bad[:3]}，已从分母剔除）" if bad else ""))
    print(f"[list] 图像 mode 分布: {dict(modes)}（非 RGB 会被 convert('RGB') 兜住）")
    print(f"[list] 可评测样本 {len(entries)} 张; 文件: {path_for_record(repo_path(seg['data_list']))}")

    names = read_labels(repo_path(seg["labels"]))
    print(f"[labels] {len(names)} 行; 锚点校验 tench/sea snake/Egyptian cat/toilet tissue -> OK")
    coverage = len(set(labs))
    print(f"[labels] 子集覆盖 {coverage} 个类别, 最大标签 {hi} < {len(names)} -> OK")
    print("[check_data] 仅数据自检，未加载模型；去掉 pretrained_eval.check_data=true 即可完整评测")
    return 0


# --------------------------------------------------------------------------- #
# 8. 主流程
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    """分块读（8 MiB/块）避免 22 MB 权重一次性进内存。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg, seg = load_eval_config(args.cfg, args.set)
    t_start = time.time()

    # ---------------- 步骤 0：配置体检 ----------------
    if seg["check_data"]:
        return check_data_mode(seg)
    for req in ("model", "data_list", "labels"):
        if not seg[req]:
            raise SystemExit(f"[错误] pretrained_eval.{req} 是必填项（用 --set pretrained_eval.{req}=... 指定）")

    import timm                                          # 白名单校验用，且要拿版本号
    bare = model_bare_name(seg["model"])
    registered = set(timm.list_models("repvit*"))
    if bare not in registered:
        raise SystemExit(f"[错误] model={seg['model']} 不在 timm 注册表内。"
                         f"当前可用: {sorted(registered)}；本工具白名单: {list(REPVIT_WHITELIST)}")
    out_dir = resolve_out_dir(seg)
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    device = seg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(seg["amp"]) and str(device).startswith("cuda")
    if seg["threads"] is None:
        print(f"[cfg]   pretrained_eval.threads=None -> 保持 torch.get_num_threads()="
              f"{torch.get_num_threads()}（多型号横向对比必须显式固定）")

    print("=" * 78)
    print(f"[cfg]   {args.cfg}  |  model={seg['model']}  impl={seg['impl']}  fuse={seg['fuse']}")
    print(f"[cfg]   device={device} amp={use_amp} input={seg['input_size']} "
          f"crop_pct={seg['crop_pct']} interp={seg['interpolation']} bs={seg['batch_size']}")
    print(f"[cfg]   out_dir={path_for_record(out_dir)}/")

    # ---------------- 步骤 1：读 list ----------------
    for req in ("data_list", "labels"):
        if not repo_path(seg[req]).exists():
            raise SystemExit(f"[错误] 找不到 pretrained_eval.{req}: {repo_path(seg[req])}")
    rows = read_data_list(repo_path(seg["data_list"]))
    names = read_labels(repo_path(seg["labels"]))
    num_classes = len(names)
    entries, bad, modes = scan_images(rows, seg.get("data_root"))
    if not entries:
        raise SystemExit("[错误] 没有任何可用图片")
    labels_arr = np.asarray([y for _, y in rows], dtype=np.int64)
    assert 0 <= labels_arr.min() and labels_arr.max() <= num_classes - 1, \
        f"标签越界: {labels_arr.min()}..{labels_arr.max()}（labels 有 {num_classes} 行）"
    print(f"[list]  {len(rows)} lines -> 可用 {len(entries)} 张, 坏图/缺图 {len(bad)} 张, "
          f"label range {int(labels_arr.min())}..{int(labels_arr.max())}, "
          f"类别数 {len(set(labels_arr.tolist()))}, mode 分布 {dict(modes)}")

    # ---------------- 步骤 2：建变换 ----------------
    tf, scale = build_eval_transform(seg["input_size"], seg["crop_pct"], seg["interpolation"])
    print(f"[tf]    Resize({scale}, {seg['interpolation']}) -> CenterCrop({seg['input_size']}) "
          f"-> ToTensor -> Normalize(mean={IMAGENET_MEAN}, std={IMAGENET_STD})")
    dataset = ListImageDataset(entries, tf)
    loader = DataLoader(dataset, batch_size=int(seg["batch_size"]), shuffle=False,
                        drop_last=False, num_workers=0, pin_memory=False)

    def _eval_with(transform):
        """用给定变换跑一遍全量推理（同一个 list、同一个 batch_size，只换预处理口径）。"""
        ld = DataLoader(ListImageDataset(entries, transform), batch_size=int(seg["batch_size"]),
                        shuffle=False, drop_last=False, num_workers=0, pin_memory=False)
        return run_inference(model, ld, device, len(rows), num_classes)

    # ---------------- 步骤 3/4：建模型 + 载权重 ----------------
    weights = repo_path(seg["weights"]) if seg["weights"] else None
    sniff = None
    if weights is not None:
        if not weights.exists():
            raise SystemExit(f"[错误] 权重文件不存在: {weights}")
        sniff = sniff_state_dict(weights)
        print(f"[w]     {path_for_record(weights)}  {weights.stat().st_size:,} B  "
              f"sha256={sha256_file(weights)[:12]}...")
        print(f"[w]     判型: 顶层容器={'是' if sniff['container'] else '否'} "
              f"(首键 {sniff['first_key']}) -> style={sniff['style']}, "
              f"蒸馏双头={sniff['distillation']}, {sniff['n_keys']} 个张量")
    else:
        print("[w]     weights=None -> 回落 timm.create_model(..., pretrained=True) 走 HF 下载")

    impl = seg["impl"]
    if impl == "auto":
        impl = sniff["style"] if sniff is not None else "timm"
    assert impl in ("official", "timm"), f"impl 只能取 auto/official/timm，收到 {impl}"

    # 权重是裸 state_dict 或带 tag 的 HF 权重时，蒸馏头开关必须与实际权重一致
    distillation = bool(sniff["distillation"]) if sniff is not None else True
    model, timm_mod = build_model(seg["model"], num_classes, impl, distillation,
                                 pretrained=(weights is None))
    official_ns = load_official_module() if impl == "official" else None
    params_unfused, _ = count_params(model)
    print(f"[model] {bare}  impl={impl}  params_total={params_unfused:,} "
          f"(distillation={distillation})")

    load_info = {"n_loaded": 0}
    if sniff is not None:
        load_info = load_weights_into(model, sniff["state_dict"])
        assert load_info["n_missing"] == 0, f"缺失键 {load_info['missing']}（结构或 num_classes 不匹配）"
        print(f"[load]  已加载 {load_info['n_loaded']} 个张量, "
              f"按形状丢弃 {load_info['n_dropped_shape']}, missing={load_info['n_missing']}, "
              f"unexpected={len(load_info['unexpected'])}")
    model.eval().to(device)

    # 融合前后的一致性探针：用固定的一批真实图片，避免把随机输入的口径混淆进来
    probe_x, _, _ = next(iter(loader))                 # 留在 CPU：见下方 _probe 的理由

    # ---------------- 步骤 5：融合 ----------------
    # 探针必须在 CPU 上跑：CUDA 端默认开启 TF32（cudnn.allow_tf32=True，10 bit 尾数），
    # 融合前后卷积形状不同 -> cudnn 选中不同 kernel -> TF32 量化差可达 1e-2 量级，
    # 会被误判成「融合不等价」。CPU fp32 无 TF32，同一输入两次前向逐位相同（噪声 0.0），
    # 所以 max|Δlogits| 反映的是真正的重参数化误差（本机实测 ~1e-5）。
    def _probe(m: nn.Module) -> torch.Tensor:
        m.to("cpu").eval()
        with torch.no_grad():
            lg = m(probe_x)
            if isinstance(lg, (tuple, list)):           # 蒸馏双头：eval 下取两头平均
                lg = (lg[0] + lg[1]) / 2
        return lg.float().clone()

    fuse_on = bool(seg["fuse"])
    max_diff = None
    lat_unfused = None
    # MACs 必须融合前统计：官方 README 的 0.8G 与规格书实测的 0.847 GMACs 都是
    # **未融合口径**（RepVGGDW 的 3x3 + 1x1 + identity 三分支各算一次）；
    # 融合后 identity 与 1x1 被折进 3x3，thop 只会数到一个 3x3，数值会变小。
    macs_g, macs_src = count_macs(model, seg["input_size"])
    macs_g_fused = None
    if fuse_on:
        logits_before = _probe(model)
        model.to(device)
        # 未融合延迟在**目标设备**上测（报告里 fused/unfused 两组数字必须同设备同线程）
        lat_unfused = benchmark_latency(model, (3, seg["input_size"], seg["input_size"]),
                                        seg["warmup"], seg["runs"], device, seg["threads"], use_amp)
        fuse_model(model, impl, official_ns)
        params_fused, params_trainable = count_params(model)
        logits_after = _probe(model)
        macs_g_fused, _ = count_macs(model, seg["input_size"])
        max_diff = float((logits_before - logits_after).abs().max())
        tag = "replace_batchnorm" if impl == "official" else "model.fuse()"
        delta = params_unfused - params_fused
        print(f"[fuse]  {tag} -> params={params_fused:,}（未融合 {params_unfused:,}，净减 {delta:,}"
              f" = 蒸馏头 385,768 + BN 折进卷积 36,504）, max|logits diff| = {max_diff:.2e}")
        print(f"[macs]  未融合 {macs_g}GMACs -> 融合后 "
              f"{None if macs_g_fused is None else round(macs_g_fused, 4)}GMACs（{macs_src}）")
        assert max_diff < 1e-3, f"融合前后数值不等价（max|Δ|={max_diff:.3e}），拒绝写盘"
        model.to(device)
    else:
        params_fused, params_trainable = count_params(model)
        model.to(device)
        print(f"[fuse]  跳过（pretrained_eval.fuse=false）params={params_fused:,}")

    # ---------------- 步骤 6：评测 ----------------
    infer = run_inference(model, loader, device, len(rows), num_classes)
    all_pred, probs_all = infer["all_pred"], infer["probs_all"]
    good_idx = np.asarray([e["index"] for e in entries], dtype=np.int64)
    assert (all_pred[good_idx] >= 0).all(), "有样本未被回填（DataLoader 丢样本或行号错乱）"
    assert infer["n_seen"] == len(entries), f"覆盖不全: {infer['n_seen']} != {len(entries)}"
    n = len(entries)
    top1 = infer["correct1"] / n * 100.0
    top5 = infer["correct5"] / n * 100.0
    print(f"[eval]  {n} images, top1={top1:.2f}%  top5={top5:.2f}%  "
          f"（分母已剔除 {len(bad)} 张坏图）")

    # 口径对照（configs/pretrained_eval.yaml 的 also_eval_crop_pct）：
    # 只做额外的纯推理，不重复写产物；与主 crop_pct 相同的那一项直接复用主结果。
    sweep = [{"crop_pct": float(seg["crop_pct"]), "scale": int(scale),
              "top1": round(float(top1), 4), "top5": round(float(top5), 4), "reused": True}]
    for cp in (seg["also_eval_crop_pct"] or []):
        if abs(float(cp) - float(seg["crop_pct"])) < 1e-9:
            continue
        tf2, scale2 = build_eval_transform(seg["input_size"], float(cp), seg["interpolation"])
        inf2 = _eval_with(tf2)
        good2 = np.asarray([e["index"] for e in entries], dtype=np.int64)
        assert (inf2["all_pred"][good2] >= 0).all(), f"crop_pct={cp} 的推理有样本未回填"
        assert inf2["n_seen"] == n
        t1, t5 = inf2["correct1"] / n * 100.0, inf2["correct5"] / n * 100.0
        sweep.append({"crop_pct": float(cp), "scale": int(scale2),
                      "top1": round(float(t1), 4), "top5": round(float(t5), 4), "reused": False})
        print(f"[sweep] crop_pct={cp} -> Resize({scale2}) top1={t1:.2f}% top5={t5:.2f}% "
              f"(主干口径 {seg['crop_pct']} 为 {top1:.2f}%；两套口径结果不可比)")

    # ---------------- 步骤 7：延迟 ----------------
    lat = benchmark_latency(model, (3, seg["input_size"], seg["input_size"]),
                            seg["warmup"], seg["runs"], device, seg["threads"], use_amp)
    if lat_unfused is not None:
        # 报告里必须同时给出 fused/unfused 两组延迟；主键仍是融合后的那组
        lat["mean_ms_unfused"] = lat_unfused["mean_ms"]
        lat["p50_ms_unfused"] = lat_unfused["p50_ms"]
        lat["p95_ms_unfused"] = lat_unfused["p95_ms"]
    print(f"[lat]   mean={lat['mean_ms']:.2f}ms p50={lat['p50_ms']:.2f}ms "
          f"p95={lat['p95_ms']:.2f}ms (threads={lat['threads']}, "
          f"warmup={lat['warmup']}, runs={lat['runs']}"
          + (f", unfused mean={lat['mean_ms_unfused']:.2f}ms" if lat_unfused else "") + ")")

    # ---------------- 步骤 8：写盘 ----------------
    targets = np.asarray([e["label"] for e in entries], dtype=np.int64)
    preds = all_pred[good_idx]
    paths = [e["path"] for e in entries]
    probs = probs_all[good_idx] if probs_all is not None else \
        np.zeros((n, num_classes), dtype=np.float32)

    correct_cases, wrong_cases = pick_cases(probs, targets, preds, paths, names, n_each=2)
    samples = list(correct_cases) + list(wrong_cases)
    if len(samples) < 6:
        # 补到 6 条：优先取「中位置信度」的样本（既不是最好也不是最差），
        # 否则依次从置信度最高/最低两端补齐，保证同一条 list 下挑选结果可复现。
        conf = probs.max(1)
        order = np.argsort(conf)
        pool = ([int(order[len(order) // 2 - 1]), int(order[len(order) // 2])]
                + [int(i) for i in order[::-1]] + [int(i) for i in order])
        chosen = {s["image_path"] for s in samples}
        for i in pool:
            if len(samples) >= 6:
                break
            s = _pack_sample(i, probs, targets, preds, paths, names)
            if s["image_path"] in chosen:
                continue
            chosen.add(s["image_path"])
            samples.append(s)
    for i, s in enumerate(correct_cases, 1):
        plot_case_grid(s, cases_dir / f"correct_{i}.png")
    for i, s in enumerate(wrong_cases, 1):
        plot_case_grid(s, cases_dir / f"wrong_{i}.png")
    close_all()

    written = write_outputs(out_dir=out_dir, names=names, entries=entries, targets=targets,
                            preds=preds, top5=infer["all_top5"][good_idx],
                            top5p=infer["all_top5p"][good_idx], top1=top1, top5_acc=top5,
                            seg=seg, lat=lat, samples=samples, bad=bad, bare=bare,
                            impl=impl, weights=weights, params_total=params_fused,
                            params_trainable=params_trainable, params_unfused=params_unfused,
                            macs_g=macs_g, macs_src=macs_src, macs_g_fused=macs_g_fused,
                            load_info=load_info,
                            max_diff=max_diff, t_start=t_start, sweep=sweep)
    print(f"[write] {path_for_record(out_dir)}/{{metrics.json,latency.json,"
          f"top5_samples.json,predictions.csv}}\n"
          f"[write] {path_for_record(cases_dir)}/correct_1..2.png, wrong_1..2.png")
    for k, v in written.items():
        print(f"        {k}: {path_for_record(v)}")
    return 0


def _pack_sample(i: int, probs, targets, preds, paths, names) -> dict:
    order = probs[i].argsort()[::-1][:5]
    return {"image_path": path_for_record(paths[i]),
            "true_label_idx": int(targets[i]), "true_label_name": names[int(targets[i])],
            "top5": [{"idx": int(j), "name": names[int(j)], "prob": float(probs[i][j])}
                     for j in order],
            "correct": bool(preds[i] == targets[i])}


def write_outputs(*, out_dir: Path, names, entries, targets, preds, top5, top5p,
                  top1, top5_acc, seg, lat, samples, bad, bare, impl, weights,
                  params_total, params_trainable, params_unfused, macs_g, macs_src,
                  macs_g_fused, load_info, max_diff, t_start, sweep) -> dict:
    """四个产物 + 版本元信息一次性落盘。所有 schema 字段名与规格书 5.4 逐字一致。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    import timm

    metrics = {
        "model_name": bare,
        # 百分数口径（78.40 表示 78.40%），与 timm results-imagenet.csv 同口径
        "top1": round(float(top1), 4),
        "top5": round(float(top5_acc), 4),
        "weights_path": path_for_record(weights) if weights is not None else None,
        "weights_sha256": sha256_file(weights) if weights is not None else None,
        "num_images": len(entries),
        "num_classes": len(names),
        "params_total": int(params_total),
        "params_trainable": int(params_trainable),
        # macs_g 用**未融合**口径（与官方 README 的 0.8G 同基准）；macs_g_fused 是实际推理成本
        "macs_g": None if macs_g is None else round(float(macs_g), 4),
        "macs_g_fused": None if macs_g_fused is None else round(float(macs_g_fused), 4),
        "macs_source": macs_src,
        "model_file_size_mb": round(weights.stat().st_size / 1024 ** 2, 2) if weights else None,
        "input_size": int(seg["input_size"]),
        "crop_pct": float(seg["crop_pct"]),
        "interpolation": seg["interpolation"],
        "crop_pct_sweep": sweep,       # 口径对照：同一条 list 上 Resize 256 vs 235 的 Top-1/Top-5
        "batch_size": int(seg["batch_size"]),
        "precision": "amp_fp16" if (seg["amp"] and str(lat["device"]).startswith("cuda")) else "fp32",
        "device": str(lat["device"]),
        "torch_version": torch.__version__,
        "timm_version": getattr(timm, "__version__", None),
        "timestamp": ts,
        # 以下为规格书 5.10 / 5.6 要求可追溯的补充字段
        "impl": impl,
        "fused": bool(seg["fuse"]),
        "params_unfused": int(params_unfused),
        "params_fused_delta": int(params_unfused - params_total) if seg["fuse"] else 0,
        "fuse_max_abs_logits_diff": None if max_diff is None else float(max_diff),
        "bad_images": bad,
        "n_bad_images": len(bad),
        "weights_dropped_shape": load_info.get("dropped_shape", []),
        "weights_missing": load_info.get("missing", []),
        "latency_mean_ms": round(float(lat["mean_ms"]), 4),
        "latency_p95_ms": round(float(lat["p95_ms"]), 4),
        "elapsed_sec": round(time.time() - t_start, 2),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=True, indent=2)
    with open(out_dir / "latency.json", "w", encoding="utf-8") as f:
        json.dump(lat, f, ensure_ascii=True, indent=2)
    assert len(samples) >= min(6, len(entries)), \
        f"top5_samples 至少 {min(6, len(entries))} 条（题目要求 ≥6），当前 {len(samples)}"
    with open(out_dir / "top5_samples.json", "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, ensure_ascii=True, indent=2)

    # predictions.csv：列顺序固定；top5_* 三列用 '|' 连接裸值（与训练流程的 JSON 列口径不同）
    pred_csv = out_dir / "predictions.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["index", "image_path", "true_idx", "true_name", "pred_idx",
                     "pred_name", "top5_idx", "top5_name", "top5_prob", "correct"])
        for r, e in enumerate(entries):
            idx = e["index"]
            wr.writerow([idx, path_for_record(e["path"]), int(targets[r]),
                         names[int(targets[r])], int(preds[r]), names[int(preds[r])],
                         "|".join(str(int(v)) for v in top5[r]),
                         "|".join(names[int(v)] for v in top5[r]),
                         "|".join(f"{float(p):.6f}" for p in top5p[r]),
                         int(preds[r] == targets[r])])
    return {"metrics": out_dir / "metrics.json", "latency": out_dir / "latency.json",
            "top5_samples": out_dir / "top5_samples.json", "predictions": pred_csv}


if __name__ == "__main__":
    raise SystemExit(main())
