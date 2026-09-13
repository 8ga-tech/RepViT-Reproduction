# -*- coding: utf-8 -*-
"""models/repvit_timm.py —— timm 版 RepViT 的薄封装。

职责：把「建网 / 换分类头 / 加载权重 / 融合重参数化 / 数参数 / 分组 lr」收敛到一处，
     上层（tools/*、deploy/*）不再直接碰 timm 的细节。

参数量口径表（写报告时必须逐项标注口径，否则会被质疑「复现多了 0.39M」）：
    训练态双头 C=1000（timm 裸 create_model，distillation=True 默认） = 5,489,328
    未融合单头 C=1000（distillation=False）                          = 5,103,560
    未融合单头 C=37  （本任务训练态，distillation=False）             = 4,732,805
    融合后单头 C=1000（fuse() 之后）                                  = 5,067,056
    融合后单头 C=37  （fuse() 之后，净减 36,504）                     = 4,696,301
官方 README 里的 5.1M 对应「未融合单头 C=1000」，不是 timm 默认的双头 5.49M；
官方 ``main.py`` 走 ``replace_batchnorm`` 原地融合，所以官方口径的 5.07M 是融合后的数字。

两条不可触碰的红线（本模块用形参与显式 raise 把这两条钉死）：
  1. **没有 drop_path_rate**：timm 的 ``RepVit.__init__`` 里根本没有这个形参，源码中也不存在
     任何 DropPath 模块（``grep -c DropPath`` = 0），传该 kwarg 必抛
     ``TypeError: RepVit.__init__() got an unexpected keyword argument 'drop_path_rate'``；
     「先建模型再遍历 modules 设 drop_prob」同样无从下手——没有对象可设。
  2. **换头只换 Linear，绝不用 ``model.reset_classifier(37)``**：``reset_classifier`` 会整体重建
     ``head``，把预训练带进来的 ``head.head.bn`` 统计量（running_mean/running_var）重置为 0/1；
     而且 ``distillation=False`` 时 ``head_dist`` 属性是被**整个删除**（不是置成 nn.Identity），
     任何 ``assert isinstance(model.head.head_dist, nn.Identity)`` 都必然失败。
     建头的唯一正确写法是 ``timm.create_model(arch, pretrained=..., num_classes=37,
     distillation=False, drop_rate=0.0, legacy=False)``。
"""
from __future__ import annotations

import os

import timm
import torch
import torch.nn as nn

# 明确禁止透传给 timm 的 kwargs：出现即报错，而不是让 timm 抛一句看不懂的 TypeError。
_FORBIDDEN_KWARGS = ("drop_path_rate",)

# 官方实现 vs timm 实现的分类头参数前缀（用于分组 lr / 冻结骨干 / 判定 head）
HEAD_PREFIX = {"timm": "head.", "official": "classifier."}


# =============================================================================
# 1. 建网
# =============================================================================
def create_repvit(name: str, num_classes: int = 1000, pretrained: bool = False,
                  distillation: bool = False, drop_rate: float = 0.0,
                  legacy: bool = False, **kwargs) -> nn.Module:
    """建一个 timm 版 RepViT。默认 ``distillation=False``（单头，本任务口径）。

    ``pretrained`` 可以是 bool，也可以是 timm 的权重 tag（例 ``'repvit_m0_9.dist_300e_in1k'``）；
    ``name`` 写裸名 ``repvit_m0_9`` 时，timm 会经 ``pretrained_cfg`` 解析到同一个 tag。

    :raises ValueError: kwargs 中出现 ``drop_path_rate``（timm 没有该形参，透传必 TypeError）。
    """
    for bad in _FORBIDDEN_KWARGS:
        if bad in kwargs:
            raise ValueError(
                f"create_repvit() 不接受 {bad}：timm 的 RepVit.__init__ 没有这个形参，"
                "源码里也不存在任何 DropPath 模块，透传会抛 "
                "TypeError: RepVit.__init__() got an unexpected keyword argument "
                f"'{bad}'。请从配置里删掉该键。")
    if kwargs:
        # 其余 kwargs 原样透传（例如 scriptable 等 timm 支持的开关）；不认识的键由 timm 自己报错。
        pass
    model = timm.create_model(name, pretrained=pretrained, num_classes=int(num_classes),
                              distillation=bool(distillation), drop_rate=float(drop_rate),
                              legacy=bool(legacy), **kwargs)
    if int(num_classes) != 1000 and bool(distillation):
        print(f"[warn] {name}: num_classes={num_classes} 且 distillation=True，"
              "蒸馏头会被随机初始化并参与 eval 的 (x1+x2)/2 集成；"
              "Pet-37 等迁移任务请务必 distillation=False。")
    return model


# =============================================================================
# 2. 换分类头（只换 Linear，保留与类别数无关的 BN 统计量）
# =============================================================================
def _new_norm_linear(ref: nn.Module, in_dim: int, out_dim: int) -> nn.Module:
    """按参照模块 ``ref``（NormLinear）的类型重建同 device/dtype 的新头。"""
    try:
        return type(ref)(in_dim, out_dim, device=ref.l.weight.device,
                         dtype=ref.l.weight.dtype)
    except TypeError:
        return type(ref)(in_dim, out_dim).to(device=ref.l.weight.device,
                                            dtype=ref.l.weight.dtype)


def _copy_head_prior(new: nn.Module, old: nn.Module) -> None:
    """把旧头里与类别数无关的部分（``.bn`` 的仿射与 running 统计量）搬到新头。

    ``bn`` 的通道数只取决于 ``in_features``（384），与类别数无关，是宝贵的迁移先验；
    只有 ``.l.weight/.l.bias`` 与类别数绑定，必须丢弃。
    """
    if not (hasattr(new, "bn") and hasattr(old, "bn")):
        return
    if hasattr(new.bn, "num_features") and new.bn.num_features != old.bn.num_features:
        return
    with torch.no_grad():
        new.bn.load_state_dict(old.bn.state_dict())


def replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    """换掉 timm RepViT 的分类头 Linear，其余参数原封不动（就地修改并返回 model）。

    只重建 ``head.head``（``NormLinear`` = bn + l），``.bn`` 的统计量从旧头继承。
    绝不用 ``model.reset_classifier``：见模块 docstring 的红线 2。
    """
    head = getattr(model, "head", None)
    inner = getattr(head, "head", None)
    if inner is None or not hasattr(inner, "l"):
        raise TypeError(f"replace_classifier 只接受 timm RepVit（缺 head.head.l）："
                        f"{type(model).__name__}")
    dim = int(inner.l.in_features)
    new_head = _new_norm_linear(inner, dim, int(num_classes))
    _copy_head_prior(new_head, inner)
    head.head = new_head
    # distillation=True 时两个头必须一起换，否则 (x1+x2)/2 会在形状上炸掉
    if getattr(head, "distillation", False):
        dist = getattr(head, "head_dist", None)
        if dist is not None and hasattr(dist, "l"):
            new_dist = _new_norm_linear(dist, dim, int(num_classes))
            _copy_head_prior(new_dist, dist)
            head.head_dist = new_dist
    if hasattr(head, "num_classes"):
        head.num_classes = int(num_classes)
    if hasattr(model, "num_classes"):
        model.num_classes = int(num_classes)
    print(f"[head] {type(model).__name__}: 分类头 -> {num_classes} 类（保留 head.head.bn 统计量）")
    return model


# =============================================================================
# 3. 权重加载
# =============================================================================
def load_pretrained(model: nn.Module, ckpt_path: str,
                    drop_prefixes: tuple[str, ...] = ("head.head.", "head.head_dist."),
                    strict: bool = False) -> dict:
    """把本地 ckpt 载入 timm 版 RepViT，消化掉所有与类别数绑定的分类头键。

    两道过滤缺一不可：
      * **按前缀剔除**：``head.head.*`` / ``head.head_dist.*`` 是分类头候选键；当目标模型
        ``distillation=False`` 时 ``head.head_dist.*`` 在模型里根本不存在，必须整头丢掉，
        否则会被记成 unexpected。
      * **按形状兜底**：``nn.Module.load_state_dict`` 对 shape mismatch **无条件抛 RuntimeError**，
        ``strict=False`` 只管 missing/unexpected，管不了 size mismatch（1000 类头的
        ``head.head.l.weight`` 是 (1000, 384)，37 类模型是 (37, 384)）。所以必须在调用前
        按 ``model.state_dict()`` 的形状逐键过滤。

    前缀只作「候选」判定：命中前缀但形状一致、且在目标模型里存在的键（典型是
    ``head.head.bn.*``——它的通道数只由 in_features 决定）会**被保留**，那是与类别数无关的
    迁移先验，丢了等于白拿预训练。

    :return: ``{'missing','unexpected','dropped_shape','dropped_prefix','ckpt_keys'}``
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"权重不存在: {os.path.abspath(ckpt_path)}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(sd, dict):
        raise TypeError(f"{ckpt_path}: 期望 state_dict 或 {{'model': state_dict}}，"
                        f"实得 {type(sd).__name__}")

    ref = model.state_dict()
    dropped_shape, dropped_prefix = [], []
    for k, v in sd.items():
        prefix_hit = k.startswith(tuple(drop_prefixes))
        if k not in ref:
            # 目标模型里没有：命中前缀 -> 分类头候选（丢弃）；否则留给 load_state_dict 报 unexpected
            if prefix_hit:
                dropped_prefix.append(k)
            continue
        if v.shape != ref[k].shape:
            (dropped_prefix if prefix_hit else dropped_shape).append(k)

    drop = set(dropped_shape) | set(dropped_prefix)
    clean = {k: v for k, v in sd.items() if k not in drop}
    missing, unexpected = model.load_state_dict(clean, strict=strict)
    missing, unexpected = list(missing), list(unexpected)

    print(f"[load] {os.path.basename(ckpt_path)}: 张量数={len(sd)} "
          f"丢弃(形状)={len(dropped_shape)} 丢弃(前缀)={len(dropped_prefix)} "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    for tag, keys in (("missing", missing), ("unexpected", unexpected)):
        for k in keys:
            print(f"   {tag:>10} : {k}")
    if dropped_shape:
        print(f"   dropped_shape : {dropped_shape[:4]}{' ...' if len(dropped_shape) > 4 else ''}")
    return {"missing": missing, "unexpected": unexpected,
            "dropped_shape": dropped_shape, "dropped_prefix": dropped_prefix,
            "ckpt_keys": len(sd)}


# =============================================================================
# 4. 结构重参数化（融合）
# =============================================================================
def fuse_model(model: nn.Module) -> nn.Module:
    """调 timm 的 ``model.fuse()``（就地融合三分支与 BN），随后 ``eval()``，返回 model。

    注意：``model.eval()`` 只切换 BN 用 running stats，**不会**自动重参数化；
    eval 态下三分支结构依然存在。要导出与官方 5.07M 同口径的图，必须显式调本函数。
    """
    if not hasattr(model, "fuse"):
        raise TypeError(f"{type(model).__name__} 没有 fuse()，timm 版 RepViT 才有")
    out = model.fuse()
    if isinstance(out, nn.Module):     # 目前 timm 的 RepVit.fuse() 是就地操作、返回 None
        model = out
    return model.eval()


# =============================================================================
# 5. 参数量与分组 lr
# =============================================================================
def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """参数量（个）。``trainable_only=True`` 只数 ``requires_grad`` 的参数。

    参考值：``create_repvit('repvit_m0_9', num_classes=37, distillation=False)`` -> 4,732,805。
    """
    return int(sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)))


def head_prefix(model: nn.Module, cfg: dict | None = None) -> str:
    """判定分类头参数前缀：timm -> ``'head.'``，官方 -> ``'classifier.'``。

    前缀来自 ``cfg['model']['impl']``（**不允许实验配置里自造 head_prefix 键**）；
    若 impl 缺失或与模型实际结构对不上，则回退到按参数名嗅探。
    """
    impl = str(((cfg or {}).get("model") or {}).get("impl", "")).strip().lower()
    want = HEAD_PREFIX.get(impl)
    names = [n for n, _ in model.named_parameters()]
    if want and any(n.startswith(want) for n in names):
        return want
    for p in HEAD_PREFIX.values():
        if any(n.startswith(p) for n in names):
            return p
    raise ValueError("无法判定分类头前缀：模型既没有 head.* 也没有 classifier.* 参数")


def _is_no_decay(name: str, p: nn.Parameter) -> bool:
    """True = 该参数不加 weight decay。

    判定式 = ``ndim <= 1`` or ``名字以 .bias 结尾`` or ``名字含 .bn.``：
      * 归一化层的 weight/bias 都是 1D（并天然排除 >=2D 的卷积/线性权重）；
      * 对 BN 的 gamma 做衰减会压缩特征尺度，在 RepViT 的残差通路上改变方差传递。
    """
    return bool(p.ndim <= 1 or name.endswith(".bias") or ".bn." in name)


def param_groups(model: nn.Module, cfg: dict) -> list[dict]:
    """按「backbone / head」两类 + 「是否加 weight decay」构造 optimizer param groups。

    * backbone：``lr * optim.backbone_lr_scale``（Pet 微调取 0.1，即 1e-4）；
    * head    ：``lr * optim.head_lr_scale``（缺省 1.0，即 lr 本身）；
    * BN 权重与 bias 一律 ``weight_decay=0.0``（见 ``_is_no_decay``）。

    组数说明：本质是 2 类（backbone / head）；当 ``optim.no_decay_on_bn_bias=true``（Baseline
    的取值）时，每类再拆成 decay / no_decay 两个子组，故最多 4 组——AdamW 无法在单组内对不同
    参数用不同 weight_decay，必须拆组。排序保证 ``groups[0]`` 属于 backbone、``groups[-1]``
    属于 head（与 tools/train.py 里 ``optimizer.param_groups[0]/[-1]`` 的取用口径一致）。
    若 ``no_decay_on_bn_bias=false``，则恒为 2 组。
    """
    o = cfg.get("optim") or {}
    base_lr = float(o.get("lr", 1e-3))
    bb_lr = base_lr * float(o.get("backbone_lr_scale", 0.1))
    head_lr = base_lr * float(o.get("head_lr_scale", 1.0))
    wd = float(o.get("weight_decay", 0.05))
    split_no_decay = bool(o.get("no_decay_on_bn_bias", True))
    prefix = head_prefix(model, cfg)

    buckets: dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = name.startswith(prefix)
        no_wd = split_no_decay and _is_no_decay(name, p)
        key = (is_head, no_wd)
        if key not in buckets:
            buckets[key] = {"params": [], "lr": head_lr if is_head else bb_lr,
                            "weight_decay": 0.0 if no_wd else wd,
                            "name": f"{'head' if is_head else 'backbone'}"
                                    f"{'_no_decay' if no_wd else ''}"}
        buckets[key]["params"].append(p)

    groups = [buckets[k] for k in sorted(buckets)]
    for g in groups:
        g["n_params"] = len(g["params"])
        print(f"[optim] group={g['name']:<18} n_tensors={g['n_params']:<4} "
              f"lr={g['lr']:.3e} wd={g['weight_decay']}")
    if not groups:
        raise RuntimeError("param_groups: 没有任何 requires_grad=True 的参数")
    return groups


# =============================================================================
# 6. 可用型号
# =============================================================================
def available_models() -> list[str]:
    """timm 里带预训练权重的 RepViT 家族型号（默认带 HF tag，如 repvit_m0_9.dist_300e_in1k）。"""
    return sorted(timm.list_models("repvit*", pretrained=True))


if __name__ == "__main__":
    # 冒烟自检：不下载任何权重，只验证建网 / 数参数 / 融合 / 分组 lr 四条链路。
    # 用法（仓库根目录）：python models/repvit_timm.py
    print("可用型号:", available_models())
    net = create_repvit("repvit_m0_9", num_classes=37, pretrained=False, distillation=False)
    print("未融合单头 C=37 参数量 =", count_params(net), "（期望 4,732,805）")
    assert count_params(net) == 4732805, "参数量与口径表不符，检查 distillation / num_classes"
    assert not any("head_dist" in k for k in net.state_dict()), "蒸馏头未关闭"
    cfg_demo = {"model": {"impl": "timm"},
                "optim": {"lr": 1e-3, "backbone_lr_scale": 0.1, "weight_decay": 0.05,
                          "no_decay_on_bn_bias": True}}
    param_groups(net, cfg_demo)
    net = replace_classifier(net, 1000)
    print("换头到 1000 类后参数量 =", count_params(net))
    net = fuse_model(net)
    fused = count_params(net)
    print("融合后单头 C=1000 参数量 =", fused, "（期望 5,067,056）")
    assert not any(isinstance(m, nn.BatchNorm2d) for m in net.modules()), "融合后不应残留 BN"
