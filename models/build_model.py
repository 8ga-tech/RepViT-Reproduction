# -*- coding: utf-8 -*-
"""models/build_model.py —— 全项目唯一的模型构建入口。

它同时负责三件事，缺一不可：
  1. **建网**：屏蔽 timm 与官方两套实现的差异（键名、工厂函数、权限差异）；
  2. **加载**：兼容官方 ckpt（``{'model': sd}``）与 timm/HF 裸 state_dict，自动消化
     1000 -> 37 类的分类头键与形状不匹配（见下方 ``load_pretrained_weights``）；
  3. **重参数化**：timm 走 ``model.fuse()``，官方走 ``replace_batchnorm``。

两套实现的 state_dict 键名零交集，**权重必须与模型同体系配对**：
    timm   : ``stem.* / stages.* / head.head.{bn,l}.*``
    官方   : ``features.* / classifier.classifier.{bn,l}.*``
跨体系互换权重必须先用「按 state_dict 遍历顺序 1:1 重命名」的转换脚本，
不能指望 ``strict=False`` 兜住——那会把骨干权重全部记成 missing/unexpected。

红线（与 utils / deploy / tools 的既有口径保持一致，改任何一条都要同步改报告）：
  * 建头唯一正确写法是 ``timm.create_model(arch, pretrained=..., num_classes=37,
    distillation=False, drop_rate=0.0, legacy=False)``；
  * **绝不用 ``model.reset_classifier(37)``**：它会整体重建 head，丢掉与类别数无关的
    ``head.head.bn`` 统计量；且 ``distillation=False`` 时 ``head_dist`` 属性被整个删除
    （不是置成 ``nn.Identity``），``assert isinstance(model.head.head_dist, nn.Identity)`` 必失败；
  * **绝不出现 drop_path_rate**：timm 的 ``RepVit.__init__`` 没有该形参，透传必抛 TypeError。
"""
from __future__ import annotations

import json
import os
import re
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
_VENDOR = Path(__file__).with_name("repvit_official.py")

# 官方与 timm 两套实现的分类头键前缀：只到 Linear 子模块（.l.），
# 分类头的 bn/running_* 是与类别数无关的迁移先验，必须保留
DROP_PREFIXES = ('classifier.classifier.l.', 'classifier.classifier_dist.l.',
                 'head.head.l.', 'head.head_dist.l.')
HEAD_KEYS = re.compile(r'^(classifier\.classifier|classifier\.classifier_dist|head\.head|head\.head_dist)')

# 换头/卸载时按体系给的默认前缀（load_weights）：整头候选，形状一致的 bn 会被保留
DEFAULT_DROP_PREFIXES = {
    'timm': ('head.head.', 'head.head_dist.'),
    'official': ('classifier.classifier.', 'classifier.classifier_dist.'),
}
# 蒸馏头专属前缀：目标模型 distillation=False 时这些键在模型里根本不存在，必须整头丢弃
DIST_PREFIXES = {
    'timm': ('head.head_dist.',),
    'official': ('classifier.classifier_dist.',),
}
_SOURCE_ALIAS = {
    'timm': 'timm', 'timm_hf': 'timm', 'hf': 'timm', 'huggingface': 'timm',
    'official': 'official', 'thu-mig': 'official', 'thumig': 'official',
    'repvit_official': 'official', 'vendor': 'official',
}


# =============================================================================
# 4.3.3 统一加载函数（两条路线共用）
# =============================================================================
def load_pretrained_weights(model, ckpt_path, num_classes,
                            route='timm', report_path='outputs/metrics/weight_load_report.json'):
    """返回 dict 统计信息，并写 JSON。任何 size mismatch 都在函数内消化，不往上抛。

    注：``report_path`` 的默认值是**相对当前工作目录**的，命令行一律在仓库根执行，
    因此它会落到 ``<仓库根>/outputs/metrics/weight_load_report.json``。
    """
    assert os.path.isfile(ckpt_path), f'权重不存在: {os.path.abspath(ckpt_path)}'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)  # 同上

    # 1) 解包：官方是 {'model': sd}，timm/HF 是裸 state_dict
    if isinstance(ckpt, dict) and 'model' in ckpt:
        print(f'[ckpt] 官方训练 checkpoint，顶层键 = {list(ckpt.keys())}')
        sd = ckpt['model']
        fmt = 'official_dict'
    else:
        sd = ckpt
        fmt = 'bare_state_dict'
    print(f'[ckpt] {fmt} | 张量数={len(sd)} | 首键={next(iter(sd))}')

    tgt = model.state_dict()
    model_distill = bool(getattr(getattr(model, 'head', None), 'distillation', False)) or \
                    bool(getattr(getattr(model, 'classifier', None), 'distillation', False))

    # 2) 先做形状对齐：形状不符的键（1000 类分类头）直接丢弃
    dropped_shape = [k for k, v in sd.items() if k in tgt and v.shape != tgt[k].shape]
    # 3) 再按前缀丢弃分类头：只丢与类别数绑定的 Linear 子模块（见 DROP_PREFIXES 的注释）；
    #    目标模型关掉蒸馏时，整个 *_dist 头都要丢，否则它们会变成 unexpected 触发断言
    drop_prefixes = list(DROP_PREFIXES)
    if not model_distill:
        drop_prefixes += ['classifier.classifier_dist.', 'head.head_dist.']
    dropped_head = [k for k in sd.keys() if k.startswith(tuple(drop_prefixes))]
    sd_clean = {k: v for k, v in sd.items()
                if k not in set(dropped_shape) and k not in set(dropped_head)}

    # 4) 加载：strict=False 只兜 missing/unexpected，形状已在上面处理干净
    missing, unexpected = model.load_state_dict(sd_clean, strict=False)
    missing = list(missing); unexpected = list(unexpected)

    info = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'ckpt_path': os.path.abspath(ckpt_path),
        'ckpt_bytes': os.path.getsize(ckpt_path),
        'ckpt_format': fmt,
        'ckpt_tensors': len(sd),
        'route': route,
        'num_classes': num_classes,
        'distillation': model_distill,
        'dropped_by_shape': dropped_shape,
        'dropped_by_prefix': dropped_head,
        'missing_keys': missing,
        'unexpected_keys': unexpected,
        'model_params': sum(p.numel() for p in model.parameters()),
        'model_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=True, indent=2)

    print(f'[load] 丢弃(形状) {len(dropped_shape)} 键, 丢弃(前缀) {len(dropped_head)} 键')
    print(f'[load] missing={len(missing)} unexpected={len(unexpected)}')
    for k in missing:
        print('   missing   :', k)
    for k in unexpected:
        print('   unexpected:', k)
    print(f'[load] 参数量={info["model_params"]} | 报告写入 {os.path.abspath(report_path)}')

    # 5) 硬断言：不允许出现非分类头的 missing/unexpected
    bad_missing = [k for k in missing if not HEAD_KEYS.match(k)]
    assert not bad_missing, f'骨干参数缺失，权重与模型不匹配: {bad_missing}'
    assert not unexpected, f'存在无法安置的键: {unexpected}'
    return info


# =============================================================================
# 官方实现的安全加载（exec 注入私有模块名，屏蔽 @register_model）
# =============================================================================
def load_official_module() -> types.ModuleType:
    """把 vendored 官方源码以私有模块名 exec 进来，并屏蔽 @register_model。

    为什么不能直接 import：官方 model/repvit.py 用 @register_model 把
    repvit_m0_9 等名字写进 timm 注册表；一旦注册表被覆盖，同进程后续的
    timm.create_model('repvit_m0_9', pretrained=True) 会抛
    TypeError: repvit_m0_9() got an unexpected keyword argument 'pretrained_cfg'。
    把 register_model 换成恒等装饰器后，官方类与工厂函数仍然可用，
    但注册表保持干净，两套实现可以在同一进程共存。
    """
    if "repvit_official_vendor" in sys.modules:
        return sys.modules["repvit_official_vendor"]
    src = _VENDOR.read_text(encoding="utf-8")
    src = src.replace("from timm.models import register_model",
                      "def register_model(fn):\n    return fn")
    mod = types.ModuleType("repvit_official_vendor")
    mod.__file__ = str(_VENDOR)
    exec(compile(src, str(_VENDOR), "exec"), mod.__dict__)
    sys.modules["repvit_official_vendor"] = mod
    return mod
    # 注：上面的 str.replace 对本仓库的 vendored 副本是**空操作**——该文件的
    # `from timm.models import register_model` 与 6 个 @register_model 装饰器已经
    # 在拷贝时删掉了（这是 §2.1 约束三的落地方式）。替换成功与否都必须安全，
    # 因此这里不做任何 assert：若将来换成未删装饰器的官方原文件，替换会立刻生效，
    # 注册表同样不会被污染。双保险的意义就在于此。


# =============================================================================
# 建网
# =============================================================================
def _as_source(impl: str) -> str:
    """归一化实现名：'timm' / 'official'，其余一律报错（不允许静默兜底）。"""
    src = _SOURCE_ALIAS.get(str(impl).strip().lower())
    if src is None:
        raise ValueError(f"未知的 model.impl={impl!r}；只允许 'timm' 或 'official'")
    return src


def _create_raw(name: str, source: str, num_classes: int, pretrained: bool,
                distillation: bool, drop_rate: float = 0.0, legacy: bool = False) -> nn.Module:
    """底层建网（不加载本地权重），两条实现各走各的工厂。"""
    if source == 'timm':
        from models.repvit_timm import create_repvit
        return create_repvit(name, num_classes=int(num_classes), pretrained=pretrained,
                             distillation=bool(distillation), drop_rate=float(drop_rate),
                             legacy=bool(legacy))
    mod = load_official_module()
    if not hasattr(mod, name):
        raise ValueError(f"官方实现里没有工厂函数 {name!r}；"
                         f"可用：{[n for n in dir(mod) if n.startswith('repvit_')]}")
    if pretrained:
        raise ValueError(
            "官方实现不支持 pretrained=True 自动下载：官方 ckpt 的键名（features.* / "
            "classifier.*）与 timm 完全不通用，且本模块不允许 models -> deploy 的反向依赖。"
            "请显式给出本地权重路径：build_from_spec(..., ckpt='checkpoints/pretrained/"
            f"{name}_distill_300e.pth')，下载地址见 deploy/model_registry.py 的 ckpt_url。")
    return getattr(mod, name)(num_classes=int(num_classes), distillation=bool(distillation))


def build_model(cfg: dict) -> nn.Module:
    """按配置建网（唯一入口）。读 ``cfg['model']``：
    name / impl / num_classes / pretrained / pretrained_ckpt / distillation / drop_rate / legacy。

    优先级：``pretrained_ckpt`` 非空 -> 用本地权重热启动（不再让 timm 去 HF 下载）；
    否则 ``pretrained=True`` -> 交给 timm 从 HF 拉取。
    """
    m = dict(cfg.get("model") or {})
    name = str(m.get("name") or "").strip()
    if not name:
        raise ValueError("cfg['model']['name'] 缺失")
    if "drop_path_rate" in m:
        raise ValueError(
            "cfg['model'] 不得出现 drop_path_rate：timm 的 RepVit.__init__ 没有该形参，"
            "透传必抛 TypeError: RepVit.__init__() got an unexpected keyword argument "
            "'drop_path_rate'；且 timm 的 repvit 源码里不存在任何 DropPath 模块，"
            "「建完再遍历 modules 设 drop_prob」也无对象可设。请从配置里删掉该键。")
    source = _as_source(m.get("impl", "timm"))
    num_classes = int(m.get("num_classes", 1000))
    distillation = bool(m.get("distillation", False))
    drop_rate = float(m.get("drop_rate", 0.0))
    legacy = bool(m.get("legacy", False))
    ckpt = str(m.get("pretrained_ckpt") or "").strip() or None
    pretrained = bool(m.get("pretrained", False)) and ckpt is None

    model = _create_raw(name, source, num_classes, pretrained, distillation, drop_rate, legacy)
    if ckpt:
        report = load_weights(model, ckpt, source)
        print(f"[model] 本地权重热启动: missing={len(report['missing'])} "
              f"unexpected={len(report['unexpected'])} 丢弃={len(report['dropped'])}")
    elif pretrained:
        print(f"[model] {name}: 预训练权重由 timm/HF 内部加载（HF 慢时设 "
              "HF_ENDPOINT=https://hf-mirror.com）")

    # 蒸馏头必须真关掉：distillation=False 时 state_dict 里不允许残留 *_dist 键
    if not distillation:
        bad = [k for k in model.state_dict() if 'head_dist' in k or 'classifier_dist' in k]
        assert not bad, f"蒸馏头未关闭，state_dict 里仍有 {bad[:3]}；检查 model.distillation"
    print(f"[model] name={name} impl={source} num_classes={num_classes} "
          f"distillation={distillation} params={sum(p.numel() for p in model.parameters())}")
    return model


def build_from_spec(name: str, impl: str = "timm", num_classes: int = 1000,
                    pretrained: bool = False, ckpt: str | None = None,
                    distillation: bool = False) -> tuple[nn.Module, dict]:
    """按「型号 + 实现 + 类数」建网并（可选）载入本地权重。

    建头唯一正确写法见模块 docstring 的红线；本函数内部就是那句 create_model。

    :param ckpt: 本地权重路径；给了就按 ``impl`` 对应体系加载（前缀 + 形状两道过滤），
                 ``pretrained`` 交给 timm/HF 的那条路自动让位。
    :return: ``(model, load_report)``；``load_report`` 键：missing / unexpected / dropped
             / dropped_shape / dropped_prefix / ckpt_keys。没有 ckpt 时全为空、``ckpt_keys=0``。
    """
    source = _as_source(impl)
    model = _create_raw(name, source, int(num_classes), bool(pretrained and not ckpt),
                        bool(distillation))
    if ckpt:
        report = load_weights(model, ckpt, source)
    else:
        report = {"missing": [], "unexpected": [], "dropped": [],
                  "dropped_shape": [], "dropped_prefix": [], "ckpt_keys": 0,
                  "loaded": False}
        print(f"[build] {name} impl={source} num_classes={num_classes} 未加载本地权重")
    return model, report


def replace_head(model: nn.Module, num_classes: int, source: str) -> nn.Module:
    """换分类头（就地修改并返回 model）。source 决定走哪套换头逻辑。

    * ``timm``    ：重建 ``head.head``（NormLinear），继承 ``.bn`` 统计量；
    * ``official``：重建 ``classifier.classifier``（BN_Linear），同理。

    两套都不碰骨干，也不用 ``reset_classifier``（见模块 docstring 红线）。
    """
    src = _as_source(source)
    if src == 'timm':
        from models.repvit_timm import replace_classifier
        return replace_classifier(model, int(num_classes))
    mod = load_official_module()
    head = getattr(model, "classifier", None)
    inner = getattr(head, "classifier", None)
    if inner is None or not hasattr(inner, "l"):
        raise TypeError(f"replace_head(official) 期望官方 RepViT（缺 classifier.classifier.l），"
                        f"实得 {type(model).__name__}")
    dim = int(inner.l.in_features)
    new_head = mod.BN_Linear(dim, int(num_classes))
    with torch.no_grad():                       # bn 统计量与类别数无关，必须继承
        new_head.bn.load_state_dict(inner.bn.state_dict())
    head.classifier = new_head
    if getattr(head, "distillation", False) and hasattr(head, "classifier_dist"):
        old_dist = head.classifier_dist
        if hasattr(old_dist, "l"):
            new_dist = mod.BN_Linear(dim, int(num_classes))
            with torch.no_grad():
                new_dist.bn.load_state_dict(old_dist.bn.state_dict())
            head.classifier_dist = new_dist
    for obj in (model, head):
        if hasattr(obj, "num_classes"):
            obj.num_classes = int(num_classes)
    print(f"[head] official: 分类头 -> {num_classes} 类（保留 classifier.classifier.bn 统计量）")
    return model


# =============================================================================
# 权重加载
# =============================================================================
def _resolve_ckpt(ckpt_path: str | Path) -> str:
    """相对路径先按 cwd 找，找不到再按仓库根找（工具常从不同 cwd 被调用）。"""
    p = Path(ckpt_path)
    if p.is_absolute() or p.exists():
        return str(p)
    alt = ROOT / p
    return str(alt if alt.exists() else p)


def _unwrap_state_dict(ckpt: Any, ckpt_path: str) -> tuple[dict, str]:
    """解包 {'model': sd} / 裸 state_dict；两种写法都要认。"""
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"], "official_dict"
    if isinstance(ckpt, dict):
        return ckpt, "bare_state_dict"
    raise TypeError(f"{ckpt_path}: 期望 state_dict 或 {{'model': state_dict}}，"
                    f"实得 {type(ckpt).__name__}")


def _default_drop_prefixes(model: nn.Module, source: str) -> tuple[str, ...]:
    """体系默认前缀 + （模型关了蒸馏时）整个蒸馏头前缀。"""
    prefixes = list(DEFAULT_DROP_PREFIXES[source])
    if not _model_distillation(model):
        prefixes += list(DIST_PREFIXES[source])
    return tuple(prefixes)


def _model_distillation(model: nn.Module) -> bool:
    """模型当前是否带蒸馏头（timm 看 model.head.distillation，官方看 model.classifier.distillation）。"""
    for attr in ("head", "classifier"):
        sub = getattr(model, attr, None)
        if sub is not None and hasattr(sub, "distillation"):
            return bool(sub.distillation)
    return False


def load_weights(model: nn.Module, ckpt_path: str, source: str,
                 drop_prefixes: tuple[str, ...] | None = None,
                 strict: bool = False) -> dict:
    """把本地权重载入模型，消化掉「1000 类头 -> 37 类头」带来的键与形状冲突。

    两道过滤：
      * **前缀**：``drop_prefixes``（缺省按 ``source`` 取；模型关了蒸馏时追加整头
        ``*_dist.`` 前缀）。前缀命中且目标模型里不存在/形状不符 -> 丢弃；
      * **形状**：``nn.Module.load_state_dict`` 对 size mismatch **无条件抛 RuntimeError**，
        ``strict=False`` 只管 missing/unexpected，管不了它，所以调用前必须按
        ``model.state_dict()`` 的形状再滤一遍。

    命中前缀但形状一致且存在的键（``*.bn.*``，只由 in_features 决定）会被**保留**：
    那是与类别数无关的迁移先验。

    :return: ``{'missing','unexpected','dropped','dropped_shape','dropped_prefix',
              'ckpt_keys','ckpt_format','ckpt_path','strict'}``
    """
    src = _as_source(source)
    if drop_prefixes is None:
        drop_prefixes = _default_drop_prefixes(model, src)
    path = _resolve_ckpt(ckpt_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"权重不存在: {os.path.abspath(path)}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:                                   # 老格式 / 纯张量文件兜底
        ckpt = torch.load(path, map_location="cpu", weights_only=False)  # 同上
    sd, fmt = _unwrap_state_dict(ckpt, path)

    ref = model.state_dict()
    dropped_shape, dropped_prefix = [], []
    for k, v in sd.items():
        prefix_hit = k.startswith(tuple(drop_prefixes))
        if k not in ref:
            if prefix_hit:                              # 分类头候选且模型没有 -> 丢弃
                dropped_prefix.append(k)
            continue                                    # 其余留给 load_state_dict 报 unexpected
        if v.shape != ref[k].shape:
            (dropped_prefix if prefix_hit else dropped_shape).append(k)

    dropped = sorted(set(dropped_shape) | set(dropped_prefix))
    clean = {k: v for k, v in sd.items() if k not in set(dropped)}
    missing, unexpected = model.load_state_dict(clean, strict=bool(strict))
    missing, unexpected = list(missing), list(unexpected)

    backbone_missing = [k for k in missing if not HEAD_KEYS.match(k)]
    if backbone_missing:
        print(f"[warn] 有 {len(backbone_missing)} 个骨干参数缺失（权重与模型体系不一致？）："
              f"{backbone_missing[:3]}")
    print(f"[load] {os.path.basename(path)} ({fmt}, {len(sd)} 张量) -> "
          f"丢弃 形状{len(dropped_shape)}/前缀{len(dropped_prefix)} | "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print(f"        unexpected 前 5 个: {unexpected[:5]}")
    return {"missing": missing, "unexpected": unexpected, "dropped": dropped,
            "dropped_shape": dropped_shape, "dropped_prefix": dropped_prefix,
            "ckpt_keys": len(sd), "ckpt_format": fmt, "ckpt_path": os.path.abspath(path),
            "strict": bool(strict)}


# =============================================================================
# 冻结 / 描述 / 重参数化
# =============================================================================
def freeze_backbone(model: nn.Module, value: bool = True) -> int:
    """冻结（``value=True``）或解冻（``False``）骨干，分类头始终保持可训练。

    :return: 本次被**冻结**的参数张量个数（``value=False`` 时恒为 0）。
    """
    from models.repvit_timm import head_prefix
    prefix = head_prefix(model)
    n_frozen = 0
    for name, p in model.named_parameters():
        if name.startswith(prefix):
            p.requires_grad_(True)                      # 头永远可训
            continue
        p.requires_grad_(not value)
        n_frozen += int(bool(value))
    print(f"[freeze] backbone requires_grad={not value} | 本次冻结 {n_frozen} 个参数张量 | "
          f"可训练参数={sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return n_frozen


def describe_model(model: nn.Module) -> dict:
    """结构画像：``{'params','trainable','buffers','bn','conv','repvggdw','size_mb'}``。

    bn/conv/repvggdw 都是**模块个数**（repvggdw 同时匹配官方的 ``RepVGGDW`` 与 timm 的
    ``RepVggDw``）；size_mb 按 fp32 权重体积估算（``params * 4 / 1024**2``）。
    """
    params = int(sum(p.numel() for p in model.parameters()))
    info = {
        "params": params,
        "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "buffers": int(sum(b.numel() for b in model.buffers())),
        "bn": int(sum(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules())),
        "conv": int(sum(isinstance(m, nn.Conv2d) for m in model.modules())),
        "repvggdw": int(sum("repvggdw" in type(m).__name__.lower() for m in model.modules())),
        "size_mb": round(params * 4 / 1024 ** 2, 3),
    }
    return info


def fused_reparameterize(model: nn.Module, source: str) -> nn.Module:
    """结构重参数化（就地）+ ``eval()``。

    ``timm`` 走 ``model.fuse()``；``official`` 走 ``replace_batchnorm(model)``（原地融合）。
    注意 ``eval()`` 本身**不会**重参数化：eval 只让 BN 用 running stats，三分支结构依旧存在。
    融合后单头参数量：C=1000 -> 5,067,056；C=37 -> 4,696,301（净减 36,504）。
    """
    src = _as_source(source)
    if src == 'timm':
        from models.repvit_timm import fuse_model
        model = fuse_model(model)
    else:
        mod = load_official_module()
        mod.replace_batchnorm(model)        # 官方实现是原地替换、无返回值，不可依赖返回值
        model = model.eval()
    left_bn = sum(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules())
    assert left_bn == 0, f"融合后仍残留 {left_bn} 个 BatchNorm，replace/fuse 未生效"
    print(f"[reparam] {src} 融合完成 | params={sum(p.numel() for p in model.parameters())}")
    return model


# =============================================================================
# 自检入口
# =============================================================================
if __name__ == "__main__":
    # 用法（仓库根目录）：python models/build_model.py
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    model, report = build_from_spec("repvit_m0_9", "timm", num_classes=37, pretrained=False)
    print("describe:", describe_model(model))
    assert describe_model(model)["params"] == 4732805, \
        f"参数量 {describe_model(model)['params']} != 4,732,805"

    # 官方实现必须能在同一进程内经 exec 加载，且不污染 timm 注册表
    import timm
    before = set(timm.list_models("repvit*"))
    mod = load_official_module()
    off = mod.repvit_m0_9(num_classes=37, distillation=False)
    after = set(timm.list_models("repvit*"))
    print("官方实现参数量 =", describe_model(off)["params"], "| 注册表未变化:", before == after)
    print("timm 仍可用 =", sum(p.numel() for p in timm.create_model(
        "repvit_m0_9", num_classes=37, distillation=False).parameters()))

    # 换头 / 冻结 / 融合三条链路
    replace_head(model, 1000, "timm")
    freeze_backbone(model, True)
    print("融合后:", describe_model(fused_reparameterize(model, "timm")))
