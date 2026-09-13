#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/train.py —— RepViT-M0.9 在 Oxford-IIIT Pet 37 类上的 Baseline 迁移训练。

用法：
    python tools/train.py --cfg configs/baseline.yaml
    python tools/train.py --cfg configs/baseline.yaml --set optim.lr=0.0005 train.epochs=50
    python tools/train.py --cfg configs/baseline.yaml --set train.limit_batches=5   # 冒烟测试，不入库
产物：outputs/logs/<experiment_name>_*、outputs/metrics/<experiment_name>_test.json、
      outputs/predictions/<experiment_name>_test_preds.csv、
      checkpoints/<experiment_name>_{best,last}.pt
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
# 注意：torchvision 的 RandomErasing 只有 value= 没有 mode/count，配置里的 mode=pixel/count=1
# 对应的是 timm 的实现，两者不可混用（用错会在建模阶段直接 TypeError）。
from timm.data.random_erasing import RandomErasing as TimmRandomErasing

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INTERP = {"bilinear": T.InterpolationMode.BILINEAR, "bicubic": T.InterpolationMode.BICUBIC}


# ============================== 1. 配置与种子 ==============================
def load_config(path: str) -> dict:
    """加载 YAML，并解析 `_base_` 继承（与 utils/config.py 的 load_yaml 语义一致）。

    为什么必须有：所有优化实验配置（configs/opt_*.yaml）都是
        _base_: baseline.yaml
        <只写差异键>
    形式的**片段**。不做 _base_ 合并的话，cfg 里根本没有 model/data/optim 段，
    训练会以 KeyError 崩掉，而不是给出任何有意义的报错。
    控制变量实验恰恰要求「除声明的差异键外，其余与 baseline 逐字相同」，
    所以这里用**深合并**（dict 递归合并、非 dict 直接覆盖），
    这样 opt_*.yaml 才能只写差异、其余全部继承。
    """
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("_base_", None)       # 必须 pop，否则会残留进生效配置快照
    if base_ref:
        refs = [base_ref] if isinstance(base_ref, str) else list(base_ref)
        merged: dict = {}
        for b in refs:
            b_path = Path(b) if os.path.isabs(b) else (p.parent / b)
            merged = _deep_merge(merged, load_config(str(b_path)))
        cfg = _deep_merge(merged, cfg)
    return cfg


def _deep_merge(base: dict, over: dict) -> dict:
    """递归深合并：两边都是 dict 就继续下钻，否则 over 直接覆盖 base。返回新对象。"""
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_overrides(cfg: dict, pairs: list[str]) -> dict:
    """--set a.b.c=value：支持嵌套键，value 走 yaml.safe_load 保持类型（1e-4 会解析成 float）。"""
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set 参数格式应为 key=value，收到 {p!r}")
        key, raw = p.split("=", 1)
        node = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            if k not in node or not isinstance(node[k], dict):
                raise SystemExit(f"--set 路径不存在: {key}")
            node = node[k]
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """固定全部随机源。官方 main.py 只设了 torch/numpy 且 random.seed 被注释掉，这里补齐。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        # 会显著变慢，且部分算子无确定性实现；开启时必须在报告里写明
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id: int) -> None:
    """每个 DataLoader worker 的种子必须可推导，否则增强序列不可复现。"""
    s = torch.initial_seed() % (2 ** 32)
    random.seed(s + worker_id)
    np.random.seed((s + worker_id) % (2 ** 32))


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ============================== 2. 数据 ==============================
class PetListDataset(Dataset):
    """按冻结的 datasets/lists/pet_{train,val,test}.txt 读取。

    行格式： '<image_id>\\t<class_idx_0based>'（也兼容空格分隔，两者都走 rsplit）。
    image_id 不含 .jpg，实际文件是 images/<image_id>.jpg。
    关键：必须用 rsplit(None, 1) —— 它会按「任意空白串」从右边切一次，
    同时兼容空格与 tab，且在含空格的路径上不会切错。
    用 line.split() 遇到含空格路径必然切错（会导致标签整体错位、Top-1 塌到随机水平）。
    """

    def __init__(self, list_file: str, images_dir: str, transform, num_classes: int):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.items: list[tuple[Path, int]] = []
        with open(list_file, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.rsplit(None, 1)
                if len(parts) != 2:
                    raise SystemExit(f"{list_file}:{lineno} 需要 '<image_id> <label>' 两列，收到 {line!r}")
                image_id, lab = parts
                lab = int(lab)
                if not (0 <= lab < num_classes):
                    raise SystemExit(f"{list_file}:{lineno} 标签越界 {lab}（合法范围 0..{num_classes-1}）"
                                     f"，常见原因是把 1-based 的官方 CLASS-ID 直接当标签用")
                rel = image_id if image_id.lower().endswith(".jpg") else f"{image_id}.jpg"
                p = self.images_dir / rel
                if not p.exists():
                    raise SystemExit(f"{list_file}:{lineno} 图片不存在: {p}")
                self.items.append((p, lab))
        if not self.items:
            raise SystemExit(f"{list_file} 为空")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")   # convert('RGB') 必写：灰度/带 alpha 的图会变成 1 或 4 通道
        if self.transform is not None:
            img = self.transform(img)
        return img, y, p.stem                   # 返回 image_id（= stem）以便失败案例溯源与预测 CSV

    def labels(self) -> list[int]:
        return [y for _, y in self.items]


def build_transforms(cfg: dict):
    aug, norm = cfg["aug"], cfg["aug"]["norm"]
    size = int(cfg["data"]["input_size"])
    interp = INTERP[aug["train"]["interpolation"]]
    mean, std = norm["mean"], norm["std"]

    train_ops = []
    rrc = aug["train"].get("random_resized_crop")
    if rrc:
        train_ops.append(T.RandomResizedCrop(
            size, scale=tuple(rrc["scale"]), ratio=tuple(rrc["ratio"]), interpolation=interp))
    else:
        # random_resized_crop 置 null 时的消融路径：先 Resize 短边再随机裁
        train_ops.append(T.Resize(int(aug["train"]["resize_size"]), interpolation=interp))
        train_ops.append(T.RandomCrop(size))
    if float(aug["train"]["hflip"]) > 0:
        train_ops.append(T.RandomHorizontalFlip(p=float(aug["train"]["hflip"])))
    cj = aug["train"].get("color_jitter")
    if cj and max(float(cj[k]) for k in ("brightness", "contrast", "saturation", "hue")) > 0:
        train_ops.append(T.ColorJitter(brightness=float(cj["brightness"]), contrast=float(cj["contrast"]),
                                       saturation=float(cj["saturation"]), hue=float(cj["hue"])))
    ra = aug["train"].get("randaugment")
    if ra:
        # 优化实验 O1 的唯一变量；必须挂在 ToTensor 之前（torchvision 的 RandAugment 吃 PIL）
        train_ops.append(T.RandAugment(num_ops=int(ra["n"]), magnitude=int(ra["m"]),
                                       interpolation=interp, fill=0))
    train_ops += [T.ToTensor(), T.Normalize(mean=mean, std=std)]
    rep = aug["train"].get("random_erasing")
    if rep and float(rep["prob"]) > 0:
        # 必须在 Normalize 之后：放在前面用 0 填充，0 会被标准化成 ≈-2.0 的深色块，语义不符。
        # 用 timm 的实现（有 mode/count），device 显式给 'cpu' —— 它默认是 'cuda'，CPU 上会报错。
        train_ops.append(TimmRandomErasing(probability=float(rep["prob"]), mode=rep["mode"],
                                           min_count=int(rep["count"]), max_count=int(rep["count"]),
                                           device="cpu"))
    train_tf = T.Compose(train_ops)

    val_interp = INTERP[aug["val"]["interpolation"]]
    val_ops = [T.Resize(int(aug["val"]["resize_size"]), interpolation=val_interp)]
    if aug["val"]["center_crop"]:
        val_ops.append(T.CenterCrop(int(aug["val"]["crop_size"])))
    val_ops += [T.ToTensor(), T.Normalize(mean=mean, std=std)]
    val_tf = T.Compose(val_ops)     # 验证与测试共用；绝不能复用 train_tf
    return train_tf, val_tf


def build_loaders(cfg: dict, train_tf, val_tf, images_dir: Path):
    K = int(cfg["model"]["num_classes"])
    ds = {s: PetListDataset(cfg["data"][f"{s}_list"], images_dir, tf, K)
          for s, tf in (("train", train_tf), ("val", val_tf), ("test", val_tf))}

    # --- 泄漏自检：三个集合两两无交集，且 train/val/test 都覆盖到类别 ---
    names = {s: {p.stem for p, _ in d.items} for s, d in ds.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = names[a] & names[b]
        if inter:
            raise SystemExit(f"数据泄漏：{a} 与 {b} 有 {len(inter)} 张重复图，例如 {list(inter)[:5]}")
    cov = {s: len(set(d.labels())) for s, d in ds.items()}
    print(f"[data] train={len(ds['train'])} val={len(ds['val'])} test={len(ds['test'])} | "
          f"类别覆盖 train={cov['train']}/val={cov['val']}/test={cov['test']} (K={K})")

    common = dict(num_workers=int(cfg["data"]["num_workers"]),
                  pin_memory=bool(cfg["data"]["pin_memory"]),
                  persistent_workers=bool(cfg["data"]["num_workers"]) and bool(cfg["data"]["persistent_workers"]),
                  worker_init_fn=worker_init_fn)
    generator = make_generator(int(cfg["seed"]))
    train_loader = DataLoader(ds["train"], batch_size=int(cfg["data"]["batch_size"]),
                              shuffle=True, drop_last=bool(cfg["data"]["drop_last"]),
                              generator=generator, **common)
    val_loader = DataLoader(ds["val"], batch_size=int(cfg["data"]["eval_batch_size"]),
                            shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(ds["test"], batch_size=int(cfg["data"]["eval_batch_size"]),
                             shuffle=False, drop_last=False, **common)
    return ds, train_loader, val_loader, test_loader


# ============================== 3. 模型 ==============================
def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    # 预训练权重来源二选一：pretrained_ckpt 非空走本地 .pth 热启动；否则交给 timm 从 HF 拉取。
    # 注意 model.name 写裸名 repvit_m0_9 也能取到权重：timm 的 pretrained_cfg 把它绑到
    # repvit_m0_9.dist_300e_in1k；要换 450e 就把 name 写成 repvit_m0_9.dist_450e_in1k。
    model = timm.create_model(m["name"], pretrained=bool(m["pretrained"]),
                              num_classes=int(m["num_classes"]),
                              distillation=bool(m["distillation"]),
                              drop_rate=float(m["drop_rate"]),
                              legacy=bool(m["legacy"]))
    # 两件绝对不能做的事：
    # ① 不要传 drop_path_rate —— timm 的 RepVit.__init__ 没有这个形参，传 0.0 也抛 TypeError；
    # ② 不要用 model.reset_classifier(37, distillation=False) 换头 —— 它整头重建，把预训练带进来的
    #    head.head.bn 统计量重置成 running_mean=0 / running_var=1；而且 distillation=False 时
    #    head_dist 属性是被整个删除（不是被置成 nn.Identity），任何
    #    `assert isinstance(model.head.head_dist, nn.Identity)` 都必然失败。
    # 唯一正确写法就是上面把 num_classes 与 distillation=False 一起交给 create_model：
    # timm 的 load_pretrained 会把 head.head_dist.* 的 5 个张量记为 unexpected 后丢弃，
    # head.head.bn 的统计量完整保留。这也是 C=37 单头 = 4,732,805 的判定路径。
    print(f"[model] name={m['name']} tag={model.pretrained_cfg.get('tag', '-')} "
          f"hf_hub_id={model.pretrained_cfg.get('hf_hub_id', '-')} distillation={m['distillation']}")
    assert not any(k.startswith("head.head_dist") for k in model.state_dict()), \
        "蒸馏头未关闭：state_dict 里仍有 head.head_dist.*，检查 model.distillation"
    return model


def print_param_report(model: nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head = sum(p.numel() for n, p in model.named_parameters() if n.startswith("head."))
    print(f"[model] 参数量 total={total:,} trainable={trainable:,} head={head:,}")
    print("[model] 口径：未融合、单头、C=37 = 4,732,805（≈4.73M），本任务应命中此值；"
          "timm 裸 create_model 的未融合蒸馏双头 C=1000 = 5,489,328（≈5.49M）；"
          "未融合、单头、C=1000 = 5,103,560（≈5.10M）；"
          "融合后单头 = 5,067,056（C=1000，融合后单头）/ 4,696,301（C=37，融合后单头），净减 36,504，以本机实测为准")
    return total


def warm_start_from(model: nn.Module, ckpt_path: str) -> None:
    """从本地 ckpt 热启动（官方 *.pth 或本仓库 best.pt）。

    必须处理两件事：
    (1) 官方 ckpt 是 dict，权重在 ckpt['model']；timm/HF 是裸 state_dict；
    (2) 换 num_classes 后分类头维度必然不匹配，而 nn.Module.load_state_dict 对 size mismatch
        是「无条件 raise RuntimeError」，strict=False 只控制 missing/unexpected，挡不住 shape 冲突。
        因此这里按 shape 过滤，并显式打印 missing / unexpected。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    print(f"[ckpt] {ckpt_path} 前 8 个键: {list(sd.keys())[:8]}")
    own = model.state_dict()
    kept, dropped = {}, []
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            kept[k] = v
        else:
            dropped.append(k)
    missing, unexpected = model.load_state_dict(kept, strict=False)
    print(f"[ckpt] 命中 {len(kept)} | 丢弃(形状不符/不存在) {len(dropped)} | "
          f"missing {len(missing)} | unexpected {len(unexpected)}")
    print(f"[ckpt] missing 示例: {missing[:6]}")
    print(f"[ckpt] dropped 示例: {dropped[:6]}")
    if unexpected:
        raise SystemExit(f"unexpected keys 非空，权重与模型结构不匹配: {unexpected[:10]}")


# ============================== 4. 损失 / 优化器 / 调度器 ==============================
class LabelSmoothingCrossEntropy(nn.Module):
    """L = (1-eps)*(-log p_y) + eps*mean_k(-log p_k)。eps=0 时与 nn.CrossEntropyLoss 数值等价。"""

    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = float(smoothing)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.smoothing == 0.0:
            return F.cross_entropy(x, target)
        logprobs = F.log_softmax(x, dim=-1)
        nll = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth = -logprobs.mean(dim=-1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * smooth).mean()


def mix_batch(x: torch.Tensor, y: torch.Tensor, cfg: dict, device: torch.device):
    """按 config 的 mixup/cutmix 开关做 batch 级混合；Baseline 三个开关都是 0/0.0，直接原样返回。

    返回 (x, y_a, y_b, lam)：mixup 用 lam 做两路线性插值，cutmix 用 lam 做矩形区域贴换，
    两者都是对 loss 做 `lam*CE(out,y_a) + (1-lam)*CE(out,y_b)` 的加权，语义一致。
    """
    t = cfg["train"]
    if float(t["mixup_prob"]) <= 0 or max(float(t["mixup_alpha"]), float(t["cutmix_alpha"])) <= 0:
        return x, y, y, 1.0
    if torch.rand(1).item() > float(t["mixup_prob"]):
        return x, y, y, 1.0
    perm = torch.randperm(x.size(0), device=device)
    y_a, y_b = y, y[perm]
    use_cutmix = float(t["cutmix_alpha"]) > 0 and torch.rand(1).item() < float(t["mixup_switch_prob"])
    if use_cutmix:
        lam = float(np.random.beta(t["cutmix_alpha"], t["cutmix_alpha"]))
        h, w = x.size(2), x.size(3)
        rh, rw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, h)
        x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, w)
        x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)     # 按真实面积回填 lam，否则与 loss 权重不匹配
    else:
        lam = float(np.random.beta(t["mixup_alpha"], t["mixup_alpha"]))
        x = lam * x + (1.0 - lam) * x[perm]
    return x, y_a, y_b, lam


def build_param_groups(model: nn.Module, cfg: dict) -> list[dict]:
    """backbone 与 head 不同 lr；BN 仿射参数与全部 bias 不加 weight decay。

    判定式 is_excluded = (ndim <= 1) or name.endswith('.bias')：BN/LN 的 weight 与 bias 都是 1D，
    一行同时覆盖 norm 仿射参数与所有 bias，并天然排除 ≥2D 的卷积/线性权重。
    为什么 BN 不加 wd：对 γ 做衰减会压缩特征尺度，在 RepViT 的残差通路上会改变方差传递。
    """
    o = cfg["optim"]
    base_lr = float(o["lr"])
    bb_lr = base_lr * float(o["backbone_lr_scale"])
    wd = float(o["weight_decay"])
    head_prefix = "head." if any(n.startswith("head.") for n, _ in model.named_parameters()) else "classifier."
    buckets: dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = name.startswith(head_prefix)
        no_wd = bool(o["no_decay_on_bn_bias"]) and (p.ndim <= 1 or name.endswith(".bias"))
        lr = base_lr if is_head else bb_lr
        key = (is_head, no_wd)
        if key not in buckets:
            buckets[key] = {"params": [], "lr": lr, "weight_decay": 0.0 if no_wd else wd,
                            "name": f"{'head' if is_head else 'backbone'}_{'no_decay' if no_wd else 'decay'}"}
        buckets[key]["params"].append(p)
    groups = [g for g in buckets.values() if g["params"]]
    for g in groups:
        print(f"[optim] group={g['name']:<18} n_params={len(g['params']):<4} lr={g['lr']:.3e} wd={g['weight_decay']}")
    return groups


def build_optimizer(groups: list[dict], cfg: dict):
    o = cfg["optim"]
    if o["optimizer"] == "adamw":
        return torch.optim.AdamW(groups, betas=tuple(o["betas"]), eps=float(o["eps"]))
    if o["optimizer"] == "sgd":
        return torch.optim.SGD(groups, momentum=float(o["momentum"]), nesterov=bool(o["nesterov"]))
    raise SystemExit(f"未知 optimizer: {o['optimizer']}")


def build_scheduler(optimizer, cfg: dict):
    """线性 warmup + 余弦退火，按 epoch 调 step。

    PyTorch 的 LinearLR / CosineAnnealingLR 都以 group['initial_lr'] 为基准做缩放，
    因此每个 param group 的 lr 倍率（backbone 0.1x）在整个训练过程中自动保持，
    不需要为 head/backbone 各建一套 scheduler，也不会互相覆盖 lr。
    """
    o = cfg["optim"]
    epochs, warm = int(cfg["train"]["epochs"]), int(o["warmup_epochs"])
    if o["scheduler"] == "onecycle":
        assert warm == 0, "OneCycleLR 不可与 warmup 链式叠加（文档原文 'This scheduler is not chainable.'）"
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[g["lr"] for g in optimizer.param_groups],
            epochs=epochs, steps_per_epoch=cfg["_steps_per_epoch"], pct_start=0.3)
    if o["scheduler"] == "constant" or epochs == warm:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=epochs)
    schedulers, milestones = [], []
    if warm > 0:
        schedulers.append(torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=float(o["warmup_start_factor"]), end_factor=1.0, total_iters=warm))
        milestones.append(warm)
    schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warm), eta_min=float(o["min_lr"])))
    if len(schedulers) == 1:
        return schedulers[0]
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for n, p in model.named_parameters():
        if not n.startswith("head."):
            p.requires_grad_(trainable)


# ============================== 5. 评价 ==============================
def topk_correct(logits: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> list[int]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [int(correct[:k].reshape(-1).sum().item()) for k in topk]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, K: int,
             collect_extra: bool = False) -> dict:
    """model.eval() 必写：否则 BN 用 batch 统计、dropout 生效，指标随机且偏低。"""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, n = 0.0, 0
    logits_all, y_all, names = [], [], []
    for x, y, stem in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        if isinstance(logits, (tuple, list)):     # 兜底：万一蒸馏头没关掉
            logits = (logits[0] + logits[1]) / 2
        loss_sum += float(crit(logits, y).item())
        n += y.numel()
        logits_all.append(logits.float().cpu())
        y_all.append(y.cpu())
        if collect_extra:
            names.extend(list(stem))
    logits = torch.cat(logits_all)
    y = torch.cat(y_all).numpy()
    pred = logits.argmax(1).numpy()
    c1, c5 = topk_correct(logits, torch.from_numpy(y), (1, 5))
    cm = confusion_matrix(y, pred, labels=list(range(K)))
    per_cls = (np.diag(cm) / np.maximum(cm.sum(1), 1)).tolist()
    res = {
        "loss": loss_sum / max(n, 1),
        "top1": c1 / n,
        "top5": c5 / n,
        "top1_count": int(c1),
        "acc": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),  # 37 类必须显式 average
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "per_class_recall": per_cls,
        "n": int(n),
    }
    if collect_extra:
        res["per_class_f1"] = f1_score(y, pred, labels=list(range(K)),
                                       average=None, zero_division=0).tolist()   # test.json 的必填字段
        res["_logits"] = logits.numpy()
        res["_y"] = y
        res["_pred"] = pred
        res["_names"] = names
    print(f"[eval] n={n} loss={res['loss']:.4f} top1={res['top1']*100:.2f}% top5={res['top5']*100:.2f}% "
          f"macroF1={res['macro_f1']*100:.2f}% balAcc={res['balanced_acc']*100:.2f}% "
          f"(37 类随机水平 {100.0/K:.2f}%)")
    return res


# ============================== 6. 日志 ==============================
# 逐 epoch 主表表头：全仓唯一契约，列名与顺序都不得改（见 6.4.1）。
# 分组 lr 的倍率信息、balanced_acc、is_best、时间戳写进同目录的 <exp>_metrics.jsonl（CSV 的超集）。
CSV_HEADER = ["epoch", "train_loss", "train_acc1", "val_loss", "val_top1", "val_top5",
              "val_macro_f1", "lr", "epoch_time_sec"]

# metrics.jsonl 比 CSV 多出来的列（超集部分）
JSONL_EXTRA = ["lr_backbone", "val_balanced_acc", "grad_norm_mean", "is_best", "timestamp"]


def init_logs(metrics_csv: Path) -> None:
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(CSV_HEADER)


def append_csv(metrics_csv: Path, row: dict) -> None:
    with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row[k] for k in CSV_HEADER])


def append_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=True) + "\n")


# ============================== 7. 训练循环 ==============================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg, epoch,
                    steps_path: Path, global_step: int, limit_batches: int = 0):
    model.train()
    if int(cfg["train"]["freeze_backbone_epochs"]) > epoch:
        set_backbone_trainable(model, False)     # 仅短暂预热；epoch 到期后必须解冻
    else:
        set_backbone_trainable(model, True)

    amp_on = bool(cfg["train"]["amp"]) and device.type == "cuda"
    amp_dtype = torch.float16 if cfg["train"]["amp_dtype"] == "float16" else torch.bfloat16
    limit = int(cfg["train"]["limit_batches"])   # 0 = 不限制；>0 时只跑前 N 个 batch（冒烟）
    clip = float(cfg["train"]["grad_clip"])
    log_interval = int(cfg["train"]["log_interval"])

    loss_sum, correct, seen, gnorm_sum, n_steps = 0.0, 0, 0, 0.0, 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step, (x, y, _) in enumerate(loader):
        if limit and step >= limit:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x, y_a, y_b, lam = mix_batch(x, y, cfg, device)      # Baseline 下 lam=1.0、y_a=y_b=y
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_on):
            logits = model(x)
            loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
        if not math.isfinite(float(loss.detach())):
            # 官方 engine.py 在这里是 sys.exit(1) —— 一个 NaN 会毁掉整段训练。
            # 这里改为跳过该 micro-batch 并记录，保证 30~50 epoch 不会因单批异常作废。
            print(f"[warn] epoch {epoch} step {step} loss 非有限，跳过该 micro-batch")
            optimizer.zero_grad(set_to_none=True)
            continue
        if scaler is not None and amp_on:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if scaler is not None and amp_on:
            scaler.unscale_(optimizer)             # 裁剪前必须 unscale，否则裁的是放大 65536 倍的梯度
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip) if clip > 0 \
            else torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        if scaler is not None and amp_on:
            scaler.step(optimizer)                 # 溢出时 scaler 自动跳过本次并降低 scale
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        gnorm_sum += float(gnorm)
        n_steps += 1
        global_step += 1

        bs = y.numel()
        loss_sum += float(loss.detach()) * bs
        correct += int((logits.detach().argmax(1) == y).sum().item())
        seen += bs
        if log_interval and (step + 1) % log_interval == 0:
            append_jsonl(steps_path, {
                "epoch": epoch, "step": step + 1, "global_step": global_step,
                "loss": round(loss_sum / max(seen, 1), 6),
                "lr_head": optimizer.param_groups[0]["lr"],
                "grad_norm": round(float(gnorm), 4) if n_steps else None,
                "amp_scale": float(scaler.get_scale()) if (scaler is not None and amp_on) else None,
            })
    return {"train_loss": loss_sum / max(seen, 1), "train_acc1": correct / max(seen, 1),
            "grad_norm_mean": gnorm_sum / max(n_steps, 1), "epoch_time_sec": time.time() - t0,
            "global_step": global_step}


# ============================== 8. 主流程 ==============================
def main() -> None:
    ap = argparse.ArgumentParser(description="RepViT-M0.9 Pet-37 Baseline 训练")
    ap.add_argument("--cfg", default="configs/baseline.yaml", help="yaml 配置路径")
    ap.add_argument("--set", nargs="*", default=[], help="覆盖 yaml，例如 --set optim.lr=0.0005 train.epochs=50")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--eval-only-test", action="store_true", help="只加载 best 权重跑一次 test（不训练）")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.cfg), args.set)
    exp = cfg["experiment_name"]
    limit_batches = int(cfg["train"]["limit_batches"])   # 冒烟开关，唯一的传参通道也是 --set
    log_dir = Path(cfg["eval"]["log_dir"])            # outputs/logs
    metric_dir = Path(cfg["eval"]["metric_dir"])      # outputs/metrics
    ckpt_dir = Path(cfg["eval"]["checkpoint_dir"])    # checkpoints
    pred_dir = Path(cfg["output"]["dir"]) / "predictions"
    for d in (log_dir, metric_dir, ckpt_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)
    # 全仓唯一定名规则：产物一律以 experiment_name 为前缀
    metrics_csv = log_dir / f"{exp}_metrics.csv"
    metrics_jsonl = log_dir / f"{exp}_metrics.jsonl"
    steps_jsonl = log_dir / f"{exp}_steps.jsonl"
    cfg_eff = log_dir / f"{exp}_config_effective"
    test_json = metric_dir / f"{exp}_test.json"
    counter_path = log_dir / f"{exp}_test_eval_count.json"
    pred_csv = pred_dir / f"{exp}_test_preds.csv"
    best_pt, last_pt = ckpt_dir / f"{exp}_best.pt", ckpt_dir / f"{exp}_last.pt"

    # --- 一致性断言（启动即失败优于训练 3 小时后发现） ---
    assert cfg["aug"]["val"]["crop_size"] == cfg["data"]["input_size"], "val.crop_size 必须等于 data.input_size"
    assert cfg["optim"]["backbone_lr_scale"] > 0, "backbone_lr_scale 必须 > 0（全 0 等于冻结骨干）"
    assert cfg["train"]["early_stop_min_epochs"] < cfg["train"]["epochs"]
    assert cfg["eval"]["metric_for_best"] == "val_macro_f1", "metric_for_best 唯一合法值是 val_macro_f1"
    assert cfg["model"]["distillation"] is False, "Baseline 必须关掉 timm 的蒸馏双头"
    assert "drop_path_rate" not in cfg["model"], "timm 的 RepVit 没有 drop_path_rate 形参，传了必抛 TypeError"
    assert cfg["train"]["ema"] is False, "本模块不实现 EMA，置 true 会静默失效，故直接拒绝"

    seed = int(cfg["seed"])
    seed_everything(seed, deterministic=bool(cfg["train"]["deterministic"]))

    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    print(f"[env] torch={torch.__version__} timm={timm.__version__} cuda={torch.cuda.is_available()} "
          f"device={device} | python={sys.version.split()[0]}")

    # --- 类别映射检查（题目「检查图片、类别编号和类别名称映射」） ---
    # 两份文件互为冗余、必须同序：pet_classes.txt 行号 = 下标；pet_class_to_idx.json 是 name -> idx。
    with open(cfg["data"]["class_names"], "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    with open(cfg["data"]["class_map"], "r", encoding="utf-8") as f:
        name2idx = json.load(f)
    K = int(cfg["model"]["num_classes"])
    class_map = {i: names[i] for i in range(len(names))}          # idx -> name，供预测 CSV 与 checkpoint 用
    assert len(names) == K, f"class_names 有 {len(names)} 行，模型 {K} 类"
    assert sorted(name2idx.values()) == list(range(K)) and len(name2idx) == K, "class_map 必须是 0..K-1 的双射"
    assert all(name2idx[n] == i for i, n in class_map.items()), "两份类别文件的顺序不一致"
    print(f"[label] K={K} | 0={class_map[0]} | 1={class_map[1]} | {K-1}={class_map[K-1]}")

    images_dir = Path(cfg["data"]["root"]) / cfg["data"]["images_subdir"]   # data/oxford-iiit-pet/images
    train_tf, val_tf = build_transforms(cfg)
    ds, train_loader, val_loader, test_loader = build_loaders(cfg, train_tf, val_tf, images_dir)
    print(f"[data] root={cfg['data']['root']} train_list={cfg['data']['train_list']} "
          f"val_list={cfg['data']['val_list']} test_list={cfg['data']['test_list']}")

    model = build_model(cfg).to(device)
    n_params = print_param_report(model)
    if cfg["model"]["pretrained_ckpt"]:
        warm_start_from(model, cfg["model"]["pretrained_ckpt"])

    criterion = LabelSmoothingCrossEntropy(float(cfg["train"]["label_smoothing"]))
    groups = build_param_groups(model, cfg)
    optimizer = build_optimizer(groups, cfg)
    cfg["_steps_per_epoch"] = max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, cfg)
    amp_on = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on) if amp_on else None

    init_logs(metrics_csv)
    # 生效配置快照：yaml 与 json 各一份，内容都是「合并 --set 之后的完整配置」
    effective = {k: v for k, v in cfg.items() if not k.startswith("_")}
    text = yaml.safe_dump(effective, allow_unicode=True, sort_keys=False)
    Path(f"{cfg_eff}.yaml").write_text(text, encoding="utf-8")
    Path(f"{cfg_eff}.json").write_text(json.dumps(effective, ensure_ascii=True, indent=2), encoding="utf-8")

    best_score, best_epoch, no_improve = -1.0, -1, 0
    global_step, stopped_reason = 0, "reached_epochs"
    t_start = time.time()

    if not args.eval_only_test:
        for epoch in range(int(cfg["train"]["epochs"])):
            tr = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device,
                                 cfg, epoch, steps_jsonl, global_step, limit_batches)
            global_step = tr["global_step"]
            va = evaluate(model, val_loader, device, K)
            scheduler.step()

            score = float(va[cfg["eval"]["metric_for_best"].replace("val_", "")])
            is_best = score > best_score
            if is_best:
                best_score, best_epoch, no_improve = score, epoch, 0
            else:
                no_improve += 1

            lr_head = optimizer.param_groups[-1]["lr"]
            row = {"epoch": epoch, "train_loss": round(tr["train_loss"], 6),
                   "train_acc1": round(tr["train_acc1"], 6), "val_loss": round(va["loss"], 6),
                   "val_top1": round(va["top1"], 6), "val_top5": round(va["top5"], 6),
                   "val_macro_f1": round(va["macro_f1"], 6), "lr": lr_head,
                   "epoch_time_sec": round(tr["epoch_time_sec"], 1),
                   # --- 以下是 CSV 没有、只进 jsonl 的超集列 ---
                   "lr_backbone": optimizer.param_groups[0]["lr"], "lr_head": lr_head,
                   "val_balanced_acc": round(va["balanced_acc"], 6),
                   "grad_norm_mean": round(tr["grad_norm_mean"], 4), "is_best": int(is_best),
                   "timestamp": datetime.now().isoformat(timespec="seconds")}
            append_csv(metrics_csv, row)
            append_jsonl(metrics_jsonl, {**row, "per_class_recall": va["per_class_recall"],
                                         "class_map": class_map})
            print(f"[Epoch {epoch:03d}/{int(cfg['train']['epochs'])-1:03d}] "
                  f"train_loss={row['train_loss']:.4f} train_acc1={row['train_acc1']*100:.2f}% | "
                  f"val_loss={row['val_loss']:.4f} val_top1={row['val_top1']*100:.2f}% "
                  f"val_top5={row['val_top5']*100:.2f}% macroF1={row['val_macro_f1']:.4f} | "
                  f"lr_bb={row['lr_backbone']:.2e} lr_head={row['lr_head']:.2e} | "
                  f"gnorm={row['grad_norm_mean']:.2f} | {row['epoch_time_sec']:.1f}s | "
                  f"best={best_score:.4f}(ep{best_epoch}) | no_improve={no_improve}")

            state = {
                "format_version": 2, "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "best_val_macro_f1": best_score, "best_epoch": best_epoch, "val_metrics": va,
                "config": {k: v for k, v in cfg.items() if not k.startswith("_")},  # 必须带 config
                # class_names / arch / impl 是 checkpoint 的契约键名（另见 6.5），不得改名或省略
                "class_names": names, "arch": cfg["model"]["name"], "impl": cfg["model"]["impl"],
                "class_map": class_map, "model_name": cfg["model"]["name"],
                "num_classes": K, "num_params": n_params, "seed": seed,
                "param_groups": [{"name": g["name"], "lr": g["lr"], "wd": g["weight_decay"],
                                  "n": len(g["params"])} for g in groups],
                "env": {"torch": torch.__version__, "timm": timm.__version__,
                        "python": sys.version.split()[0], "device": str(device)},
                "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                              "torch": torch.get_rng_state()},
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }
            if is_best:
                torch.save(state, best_pt)     # 只在 metric_for_best 提升时覆盖
            torch.save(state, last_pt)         # 每 epoch 覆盖

            if limit_batches and epoch >= 1:
                stopped_reason = "limit_batches"
                break
            if (epoch + 1) >= int(cfg["train"]["early_stop_min_epochs"]) and \
                    no_improve >= int(cfg["train"]["early_stop_patience"]):
                stopped_reason = (f"early_stop: {cfg['eval']['metric_for_best']} 连续 {no_improve} 个 epoch 无提升，"
                                  f"最优 {best_score:.4f} @ epoch {best_epoch}")
                print(f"[early-stop] {stopped_reason}")
                break

    # ------------------ 最终 test 评价：只跑一次 ------------------
    count = json.loads(counter_path.read_text(encoding="utf-8"))["count"] if counter_path.exists() else 0
    if limit_batches:
        print("[test] limit-batches 模式，跳过 test 评价")
    else:
        count += 1
        counter_path.write_text(json.dumps({"count": count, "last_at": datetime.now().isoformat(timespec="seconds")}),
                                encoding="utf-8")
        if count > 1:
            print(f"[test][警告] 这是第 {count} 次 test 评价；请确认没有用 test 选模型/调参（第 26 页硬性限制）")
        best_ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
        model.load_state_dict(best_ckpt["model"])
        model.to(device)
        te = evaluate(model, test_loader, device, K, collect_extra=True)
        # test 指标 JSON：字段与量纲为全仓唯一契约（top1/top5/macro_f1 都是 0~1 的小数，不是百分数）
        summary = {"experiment_name": exp, "checkpoint": str(best_pt),
                   "ckpt_sha256": hashlib.sha256(best_pt.read_bytes()).hexdigest(),
                   "split": "test", "num_samples": te["n"], "num_classes": K,
                   "top1": te["top1"], "top5": te["top5"], "macro_f1": te["macro_f1"],
                   "per_class_f1": te["per_class_f1"], "top1_count": te["top1_count"],
                   "evaluated_at": datetime.now().isoformat(timespec="seconds"), "seed": seed,
                   # --- 以下为便于自查补的记录字段（best 轮次、试跑次数、预算） ---
                   "eval_count": count, "model_name": cfg["model"]["name"],
                   "num_params": n_params, "best_epoch": best_ckpt["epoch"],
                   "best_val_macro_f1": best_ckpt["best_val_macro_f1"],
                   "epochs_run": best_ckpt["epoch"] + 1, "stopped_reason": stopped_reason,
                   "loss": te["loss"], "total_seconds": round(time.time() - t_start, 1),
                   "checkpoint_best_mb": round(best_pt.stat().st_size / 1e6, 3),
                   "checkpoint_last_mb": round(last_pt.stat().st_size / 1e6, 3)}
        test_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        if cfg["output"]["save_test_predictions"]:
            # 列名与顺序为全仓唯一契约（见 6.4.5）；后三列是 JSON 字符串
            with open(pred_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["image_id", "path", "true_idx", "true_name", "pred_idx", "pred_name",
                            "prob", "correct", "top5_idx", "top5_names", "top5_probs"])
                probs = torch.softmax(torch.from_numpy(te["_logits"]), dim=-1).numpy()
                top5 = np.argsort(-te["_logits"], axis=1)[:, :5]
                for i, name in enumerate(te["_names"]):
                    w.writerow([name, str(images_dir / f"{name}.jpg"),
                                int(te["_y"][i]), class_map[int(te["_y"][i])],
                                int(te["_pred"][i]), class_map[int(te["_pred"][i])],
                                round(float(probs[i].max()), 6), int(te["_y"][i] == te["_pred"][i]),
                                json.dumps(top5[i].tolist()),
                                json.dumps([class_map[int(j)] for j in top5[i]], ensure_ascii=False),
                                json.dumps([round(float(probs[i][j]), 6) for j in top5[i]])])
        print(f"[done] test Top-1={te['top1']*100:.2f}% Top-5={te['top5']*100:.2f}% "
              f"Macro-F1={te['macro_f1']*100:.2f}% | 结果写入 {test_json}")


if __name__ == "__main__":
    main()
