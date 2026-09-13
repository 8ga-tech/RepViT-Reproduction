# -*- coding: utf-8 -*-
"""tools/check_backbone_updated.py —— 证明「骨干确实被训练过」（对应 DoD #21）。

用法：
    python tools/check_backbone_updated.py --ckpt checkpoints/baseline_best.pt

做法（唯一真源，不要另发明第二套判据）：
  1. 从 ckpt 里取 state_dict（ckpt['model']）；
  2. 用 ckpt 自己记录的 arch / impl / num_classes 重建一个**同架构**模型；
  3. 把「同架构的 ImageNet 预训练权重」灌进这个参考模型（路径从 ckpt['config'] 里读，
     缺省回落 checkpoints/pretrained/repvit_m0_9_distill_300e.pth）；
  4. 逐张量用 `(a - b).abs().max() > 0` 判定「是否已变化」，统计 changed_tensors。

为什么不能只看损失下降：迁移学习最隐蔽的失败模式是「只训了分类头、骨干被冻住」——
loss 照样降、val 照样涨，但骨干权重与预训练逐位相同。本脚本就是专门戳这个：
  changed_tensors == 0  ->  exit(1)
  changed_backbone == 0 ->  exit(1)（骨干没动，等于没做迁移学习）

预训练权重取不到时（路径不存在 / 键名方案完全对不上且无法按序对齐），脚本会**显式打印说明**
并退化为「与随机初始化对照」——这种对照下 changed_tensors 必然 > 0，只能证明「权重不是
随机初值」，强度弱于与预训练对照，报告里必须写明用的是哪种对照。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def _local_dump_metrics(name: str, payload: dict, out_dir: str = "outputs/metrics") -> str:
    import datetime
    import os
    import platform

    p = Path(out_dir)
    out_dir = str(p if p.is_absolute() else (ROOT / p))
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "experiment": name,
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "cwd_rel": ".",
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**meta, **payload}, f, ensure_ascii=True, indent=2)
    print(f"[metrics] wrote {path}")
    return path


try:  # pragma: no cover
    from utils.logging import dump_metrics as _dump_metrics  # type: ignore
except Exception:  # noqa: BLE001
    _dump_metrics = _local_dump_metrics


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


# 分类头前缀：timm 用 head.*，官方实现用 classifier.*（含 classifier_dist）
HEAD_PREFIXES = ("head", "classifier")
# 骨干里代表「stage / block」的参数前缀：timm 是 stages.*，官方实现是 features.*
STAGE_PREFIXES = ("stages.", "features.")
DEFAULT_PRETRAINED = "checkpoints/pretrained/repvit_m0_9_distill_300e.pth"


def load_official_module():
    """官方实现只能 exec 加载（见 models/repvit_official.py 文件头说明）。"""
    try:
        from models.build_model import load_official_module as _loader  # type: ignore
        return _loader()
    except Exception:  # noqa: BLE001
        pass
    path = ROOT / "models" / "repvit_official.py"
    if not path.exists():
        raise FileNotFoundError(f"找不到官方实现文件：{path}")
    spec = importlib.util.spec_from_file_location("repvit_official", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["repvit_official"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_same_arch(arch: str, impl: str, num_classes: int, distillation: bool) -> nn.Module:
    if impl == "timm":
        import timm
        return timm.create_model(arch, pretrained=False, num_classes=int(num_classes),
                                 distillation=bool(distillation))
    if impl == "official":
        mod = load_official_module()
        factory = getattr(mod, arch, None)
        if factory is None:
            raise AttributeError(f"官方实现里没有 {arch!r} 这个工厂函数")
        return factory(num_classes=int(num_classes), distillation=bool(distillation))
    raise ValueError(f"未知 impl={impl!r}")


# ---------------------------------------------------------------------------
# 从 ckpt 里抠出 arch / impl / num_classes / 预训练权重路径（键名兼容多种写法）
# ---------------------------------------------------------------------------
def meta_from_ckpt(ckpt: dict) -> dict:
    cfg = ckpt.get("config") or {}
    mcfg = cfg.get("model") or {}
    arch = ckpt.get("arch") or mcfg.get("name") or ckpt.get("model_name")
    impl = ckpt.get("impl") or mcfg.get("impl") or "timm"
    ncls = ckpt.get("num_classes") or mcfg.get("num_classes")
    distill = mcfg.get("distillation", None)
    if distill is None:
        distill = bool(int(ncls) == 1000) if ncls else False
    pre = None
    for k in ("pretrained_ckpt", "pretrained_weights", "pretrained_path", "weights"):
        v = mcfg.get(k, None)
        if v:
            pre = v
            break
    if not pre:
        for k in ("pretrained_ckpt", "pretrained_weights", "pretrained_path"):
            v = ckpt.get(k, None)
            if v:
                pre = v
                break
    return {"arch": arch, "impl": str(impl), "num_classes": int(ncls) if ncls else None,
            "distillation": bool(distill), "pretrained_path": pre}


def load_state_dict_from(path: Path) -> dict:
    """官方 ckpt 形如 {'model': sd, ...}；timm/HF 是裸 state_dict。两种都要吃。"""
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    return obj


def align_pretrained(ref_keys: list, ref_sd: dict, pre_sd: dict):
    """把预训练权重对到参考模型的键上。

    返回 (aligned_sd, mode, note)：
      * 键名直接对齐：'key'      -> 官方 ckpt 与官方实现之间（713 键逐字命中）；
      * 逐位对齐：    'position' -> timm 实现与官方 ckpt 之间。timm 把官方
        features.{i} 重排成了 stem/stages.{s}.blocks.{b}，键名完全不同，但两套
        state_dict 的**枚举顺序与形状序列逐一相同**（713 vs 713 全等），
        因此按位置配对是安全且可验证的（配错会立刻被形状序列不等断言拦下）。
    """
    direct = {k: v for k, v in pre_sd.items() if k in ref_sd and tuple(v.shape) == tuple(ref_sd[k].shape)}
    if len(direct) >= 0.5 * len(ref_keys):
        return direct, "key", f"键名直接对齐，命中 {len(direct)}/{len(ref_keys)} 个张量"

    pre_keys = list(pre_sd.keys())
    n = min(len(ref_keys), len(pre_keys))
    if n == 0:
        return {}, "none", "参考模型或预训练文件没有任何张量"
    # 逐位比较形状序列：一致的位置才配对。换头（C=37 vs C=1000）只会让末尾的分类头
    # 形状对不上，被自动剔除，骨干部分照样逐位对齐。
    shape_ok = [i for i in range(n)
                if tuple(ref_sd[ref_keys[i]].shape) == tuple(pre_sd[pre_keys[i]].shape)]
    if len(shape_ok) >= 0.95 * n:
        aligned = {ref_keys[i]: pre_sd[pre_keys[i]] for i in shape_ok}
        same_len = (len(ref_keys) == len(pre_keys))
        mode = "position" if same_len else "position_prefix"
        note = (
            f"键名方案不一致（timm 的 stem/stages.* vs 官方的 features.*），"
            f"改按「枚举顺序 + 形状序列」逐位对齐：比较前 {n} 个张量，其中 "
            f"{len(shape_ok)} 个形状完全一致（{len(shape_ok) / n * 100:.1f}%），可安全配对；"
            f"形状对不上的 {n - len(shape_ok)} 个（换头后的分类头）已剔除，保持随机初值。"
            f"若两套结构有任何差异，形状比率不会达到 95%，此处会直接判定失败")
        return aligned, mode, note
    return {}, "none", (
        f"既无法按键名对齐（仅命中 {len(direct)} 个），也无法按序对齐"
        f"（参考模型 {len(ref_keys)} 键 vs 预训练 {len(pre_keys)} 键，前 {n} 个张量里"
        f"仅 {len(shape_ok)} 个形状一致，低于 95% 阈值）")


def main() -> int:
    ap = argparse.ArgumentParser("核对骨干是否真的被训练（与同架构预训练权重逐张量比较）")
    ap.add_argument("--ckpt", default="checkpoints/baseline_best.pt")
    ap.add_argument("--pretrained", default=None,
                    help="覆盖预训练权重路径；缺省从 ckpt['config'] 读，再回落 "
                         f"{DEFAULT_PRETRAINED}")
    ap.add_argument("--out-dir", default="outputs/metrics")
    ap.add_argument("--json-name", default="backbone_updated")
    args = ap.parse_args()

    ckpt_path = _resolve(args.ckpt)
    if not ckpt_path.exists():
        print(f"[ERROR] 找不到 checkpoint：{ckpt_path}")
        print("        提示：先跑 `python tools/train.py --cfg configs/baseline.yaml` 生成 "
              "checkpoints/<experiment_name>_best.pt")
        return 1

    print("=" * 84)
    print(" check_backbone_updated —— 骨干确实被训练（DoD #21）")
    print(f" ckpt = {args.ckpt}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        print("[ERROR] checkpoint 不是 dict，无法解析结构")
        return 1
    sd = ckpt.get("model", ckpt)
    if not isinstance(sd, dict):
        print("[ERROR] checkpoint 里没有可用的 state_dict（键 'model'）")
        return 1

    meta = meta_from_ckpt(ckpt)
    if not meta["arch"]:
        print("[ERROR] checkpoint 里没有 arch / config.model.name，无法重建同架构模型")
        return 1
    ncls = meta["num_classes"] or len(ckpt.get("class_names") or []) or 1000
    print(f" arch={meta['arch']}  impl={meta['impl']}  num_classes={ncls}  "
          f"distillation={meta['distillation']}  ckpt_tensors={len(sd)}")

    ref_model = build_same_arch(meta["arch"], meta["impl"], ncls, meta["distillation"])
    ref_model.eval()
    ref_sd = ref_model.state_dict()
    ref_keys = list(ref_sd.keys())

    # ---- 预训练权重路径：CLI > ckpt config > 缺省回落 ----
    pre_path = None
    src = None
    if args.pretrained:
        pre_path, src = _resolve(args.pretrained), "cli"
    elif meta["pretrained_path"]:
        cand = Path(meta["pretrained_path"])
        pre_path = cand if cand.is_absolute() else (ROOT / cand)
        src = "ckpt.config.model.pretrained_ckpt"
    else:
        pre_path, src = _resolve(DEFAULT_PRETRAINED), "default_fallback"

    compare_mode, compare_note = None, None
    if pre_path is not None and pre_path.exists():
        pre_sd = load_state_dict_from(pre_path)
        aligned, mode, note = align_pretrained(ref_keys, ref_sd, pre_sd)
        compare_note = note
        if aligned:
            miss, unexp = ref_model.load_state_dict(aligned, strict=False)
            dropped = [k for k in pre_sd if k not in aligned]
            print(f" [预训练] source={src} path={pre_path.as_posix()}")
            print(f" [预训练] 对齐方式={mode}；{note}")
            print(f" [预训练] 灌入参考模型 {len(aligned)} 个张量；丢弃 {len(dropped)} 个"
                  f"（多为换头后的 {dropped[:3]} 等分类头权重）；missing={len(miss)}")
            ref_sd = ref_model.state_dict()       # 重新取一次（load_state_dict 原地改写）
            compare_mode = f"pretrained({mode})"
        else:
            ref_sd = ref_model.state_dict()
            compare_mode = "random_init"
            print(f" [预训练] source={src} path={pre_path.as_posix()} 无法对齐：{note}")
    else:
        compare_mode = "random_init"
        print(f" [预训练] 路径不存在（source={src}，path={pre_path}）")

    if compare_mode == "random_init":
        print(" [对照] 预训练权重不可用，退化为「与随机初始化对照」。")
        print("        该对照只能证明「ckpt 不是未训练的随机初值」，强度弱于与预训练对照，")
        print("        报告里必须写明对照来源 unavailable（原因见上一行）。")

    # ---- 逐张量比较 ----
    common = [k for k in sd if k in ref_sd and tuple(sd[k].shape) == tuple(ref_sd[k].shape)]
    changed, unchanged = [], []
    for k in common:
        a = sd[k].detach().float()
        b = ref_sd[k].detach().float()
        if float((a - b).abs().max()) > 0.0:
            changed.append(k)
        else:
            unchanged.append(k)

    def is_backbone(name: str) -> bool:
        return not any(name.startswith(p) for p in HEAD_PREFIXES)

    changed_backbone = [k for k in changed if is_backbone(k)]
    stage_changed = [k for k in changed if k.startswith(STAGE_PREFIXES)]
    shape_mismatch = [k for k in sd if k in ref_sd and tuple(sd[k].shape) != tuple(ref_sd[k].shape)]

    print("-" * 84)
    print(f" [compare] 对照来源={compare_mode}  {compare_note or ''}")
    print(f" [compare] 可比较张量 common={len(common)}（ckpt {len(sd)} 键，"
          f"形状不符 {len(shape_mismatch)} 键）")
    print(f" changed_tensors={len(changed)}   unchanged_tensors={len(unchanged)}   "
          f"changed_backbone_tensors={len(changed_backbone)}")
    print(f" stages.* 下已变化张量 {len(stage_changed)} 个，前 3 个：")
    for k in stage_changed[:3]:
        print(f"   - {k}")
    if not stage_changed:
        alt = [k for k in changed if k.startswith("features.")]
        print(f"   （本例键名方案为官方实现 features.*，前 3 个：{alt[:3]}）")

    ok = (len(changed) > 0) and (len(changed_backbone) > 0)
    if len(changed) == 0:
        print(" [FAIL] changed_tensors == 0：ckpt 与参考权重逐位相同，模型根本没有被更新过。")
    elif len(changed_backbone) == 0:
        print(" [FAIL] changed_backbone_tensors == 0：只有分类头变了，骨干与预训练逐位相同，"
              "等于没做迁移学习（多半是 freeze_backbone 没解冻）。")
    else:
        print(f" [PASS] 骨干已被训练：{len(changed_backbone)} 个骨干张量相对"
              f"{'预训练权重' if compare_mode.startswith('pretrained') else '随机初值'}发生变化。")

    payload = {
        "ckpt": args.ckpt,
        "arch": meta["arch"],
        "impl": meta["impl"],
        "num_classes": ncls,
        "pretrained_source": src,
        "pretrained_path": None if pre_path is None else pre_path.as_posix(),
        "compare_mode": compare_mode,
        "compare_note": compare_note,
        "judgement": "(a-b).abs().max() > 0",
        "n_ckpt_tensors": len(sd),
        "n_comparable": len(common),
        "n_shape_mismatch": len(shape_mismatch),
        "changed_tensors": len(changed),
        "unchanged_tensors": len(unchanged),
        "changed_backbone_tensors": len(changed_backbone),
        "stage_changed_first3": stage_changed[:3],
        "stage_changed_count": len(stage_changed),
        "pass": bool(ok),
        "note": "changed_backbone 排除了 head/classifier 前缀；骨干为 0 时即便总变化数 > 0 也判 FAIL",
    }
    _dump_metrics(args.json_name, payload, out_dir=str(_resolve(args.out_dir)))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
