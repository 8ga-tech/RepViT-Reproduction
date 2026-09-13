#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/weight_load_report.py —— 权重加载报告（DoD #12）+ 两条实现路线的等价性证明。

Source: Self-written。官方权重与官方模型来自 THU-MIG/RepViT（Apache-2.0）。

为什么需要这个脚本
------------------
官方 checkpoint 与 timm 实现**键名空间完全不相交**（实测，见下），所以
「拿官方 .pth 去 load 一个 timm 模型」必然 unexpected=713 / missing=599。
DoD #12 要求的 `n_unexpected=0` 只能在**同体系配对**下达成：

    路线 B（本仓主线，官方体系）: 官方 .pth  ->  models/repvit_official.py（vendored 官方实现）
    路线 A（timm 体系）        : timm/HF 权重 ->  timm 的 RepVit

两者不是「谁替代谁」，而是同一份网络的两种封装。本脚本同时产出：

    outputs/metrics/weight_load_report.json   <- DoD #12 的验收文件
        · n_missing / n_unexpected 取**路线 B**（同体系配对，unexpected 必须为 0）
        · missing 只允许是分类头键（1000 -> 37 换头导致的必然缺失）
    outputs/metrics/route_equivalence.json    <- 两条路线的数值等价性（max|Δlogits|）

用法：
    python tools/weight_load_report.py --ckpt checkpoints/pretrained/repvit_m0_9_distill_300e.pth
    python tools/weight_load_report.py --ckpt ... --hf-weights checkpoints/pretrained/repvit_m0_9.dist_300e_in1k.safetensors
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 分类头键前缀：换头后允许 missing 的键必须落在这里面
HEAD_KEYS = re.compile(r"^(classifier\.classifier|classifier\.classifier_dist|head\.head|head\.head_dist)")
# 分类头里与类别数绑定的只有 Linear 子模块（.l.）；主头的 bn/running_* 是迁移先验，必须保留。
# 但**蒸馏头不同**：目标模型 distillation=False 时整个 classifier_dist 头都不存在，
# 只丢它的 .l. 会把它的 .bn.* 留成 5 个 unexpected（实测复现）。
DROP_PREFIXES_BASE = ("classifier.classifier.l.", "head.head.l.")
DROP_PREFIXES_DIST = ("classifier.classifier_dist.", "head.head_dist.")


def drop_prefixes_for(distillation: bool) -> tuple[str, ...]:
    """关掉蒸馏时，整个 *_dist 头都要丢；开着时只丢与类别数绑定的 Linear。"""
    if distillation:
        return DROP_PREFIXES_BASE + tuple(p + "l." for p in DROP_PREFIXES_DIST)
    return DROP_PREFIXES_BASE + DROP_PREFIXES_DIST


def load_official_module(path: Path):
    """exec 加载 vendored 官方实现（spec §2.1 约束三：必须中性化 register_model）。

    返回一个**模块状对象**（types.SimpleNamespace），而不是裸 dict——
    exec 的 globals 是 dict，dict 上没有属性访问，直接 `mod.repvit_m0_9` 会 AttributeError。
    """
    import types

    src = path.read_text(encoding="utf-8")
    neutral = ("def register_model(fn=None, **kw):\n"
               "    if fn is None:\n        return lambda f: f\n    return fn\n")
    src = re.sub(r"^from timm\.models import register_model.*$", neutral, src, flags=re.M)
    src = re.sub(r"^from timm\.models\.registry import register_model.*$", neutral, src, flags=re.M)
    ns = {"__name__": "repvit_official_vendored", "__file__": str(path)}
    exec(compile(src, str(path), "exec"), ns)          # noqa: S102
    return types.SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")})


def unwrap(ckpt):
    """官方 .pth 是 {'model': sd}；timm/HF 是裸 state_dict。"""
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"], "official_dict"
    return ckpt, "bare_state_dict"


def route_b_official(mod, ckpt_path: Path, num_classes: int = 37):
    """路线 B：官方 ckpt -> vendored 官方实现。返回 (model, info)。"""
    sd_raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd, fmt = unwrap(sd_raw)
    m = mod.repvit_m0_9(num_classes=num_classes, distillation=False)
    tgt = m.state_dict()

    dropped_shape = [k for k, v in sd.items() if k in tgt and v.shape != tgt[k].shape]
    drop_pre = drop_prefixes_for(False)   # 本路线固定 distillation=False
    dropped_head = [k for k in sd if k.startswith(drop_pre)]
    drop = set(dropped_shape) | set(dropped_head)
    clean = {k: v for k, v in sd.items() if k not in drop}

    missing, unexpected = m.load_state_dict(clean, strict=False)
    missing, unexpected = list(missing), list(unexpected)
    m.eval()
    info = {
        "route": "B_official",
        "ckpt_path": ckpt_path.relative_to(ROOT).as_posix(),
        "ckpt_bytes": ckpt_path.stat().st_size,
        "ckpt_format": fmt,
        "ckpt_tensors": len(sd),
        "model_keys": len(tgt),
        "num_classes": num_classes,
        "distillation": False,
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "dropped_by_shape": dropped_shape,
        "dropped_by_prefix": dropped_head,
        "model_params": sum(p.numel() for p in m.parameters()),
    }
    return m, info


def route_a_timm(hf_weights: Path | None, num_classes: int = 37):
    """路线 A：timm/HF 权重 -> timm RepVit。返回 (model, info) 或 (None, info)。"""
    import timm
    m = timm.create_model("repvit_m0_9", pretrained=False,
                          num_classes=num_classes, distillation=False)
    info = {"route": "A_timm", "num_classes": num_classes, "distillation": False,
            "model_params": sum(p.numel() for p in m.parameters())}
    if hf_weights is None or not Path(hf_weights).exists():
        info["status"] = "SKIPPED (未提供 --hf-weights，或文件不存在)"
        return None, info
    sd = torch.load(hf_weights, map_location="cpu", weights_only=False)
    sd, fmt = unwrap(sd)
    tgt = m.state_dict()
    dropped_shape = [k for k, v in sd.items() if k in tgt and v.shape != tgt[k].shape]
    drop_pre = drop_prefixes_for(False)
    dropped_head = [k for k in sd if k.startswith(drop_pre)]
    drop = set(dropped_shape) | set(dropped_head)
    clean = {k: v for k, v in sd.items() if k not in drop}
    missing, unexpected = m.load_state_dict(clean, strict=False)
    m.eval()
    info.update({"ckpt_path": Path(hf_weights).relative_to(ROOT).as_posix(),
                 "ckpt_format": fmt, "ckpt_tensors": len(sd),
                 "n_missing": len(list(missing)), "n_unexpected": len(list(unexpected)),
                 "dropped_by_shape": dropped_shape, "dropped_by_prefix": dropped_head,
                 "status": "OK"})
    return m, info


@torch.no_grad()
def equivalence(ma: nn.Module, mb: nn.Module, num_classes: int = 1000,
                n: int = 8, seed: int = 0) -> dict:
    """两条路线在同一批固定输入上的 logits 逐元素比较。

    必须让两侧的 distillation 状态**一致**再比：官方 ckpt 是蒸馏训练的，
    官方实现 distillation=True 时 eval 会做 (主头+蒸馏头)/2，timm 默认也是双头；
    一旦一侧单头一侧双头，比的就是两个不同的输出（03 复盘文档「技巧 3」记录的正是这个坑）。
    """
    torch.manual_seed(seed)
    x = torch.randn(n, 3, 224, 224)
    a = ma(x)
    b = mb(x)
    a = a[0] if isinstance(a, tuple) else a
    b = b[0] if isinstance(b, tuple) else b
    d = (a - b).abs()
    return {
        "n_inputs": n, "seed": seed,
        "max_abs_diff": float(d.max().item()),
        "mean_abs_diff": float(d.mean().item()),
        "top1_agreement": float((a.argmax(1) == b.argmax(1)).float().mean().item()),
        "bitwise_identical": bool(torch.equal(a, b)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="权重加载报告（DoD #12）+ 路线等价性")
    ap.add_argument("--ckpt", default="checkpoints/pretrained/repvit_m0_9_distill_300e.pth",
                    help="官方 checkpoint（相对仓库根）")
    ap.add_argument("--hf-weights", default=None,
                    help="timm/HF 权重（相对仓库根）；给了就顺带做路线 A 与等价性对照")
    ap.add_argument("--num-classes", type=int, default=37, help="换头后的类别数")
    ap.add_argument("--vendored", default="models/repvit_official.py")
    ap.add_argument("--out", default="outputs/metrics/weight_load_report.json")
    a = ap.parse_args()

    vendored = ROOT / a.vendored
    assert vendored.exists(), f"找不到 vendored 官方实现：{vendored}"
    mod = load_official_module(vendored)
    ckpt = (ROOT / a.ckpt).resolve()
    assert ckpt.exists(), f"找不到官方权重：{ckpt}"

    print("=" * 74)
    print(" 权重加载报告（DoD #12）")
    print("=" * 74)

    _, infoB = route_b_official(mod, ckpt, a.num_classes)
    print(f"[路线 B] 官方 ckpt -> vendored 官方实现 ({a.num_classes} 类)")
    print(f"  ckpt: {infoB['ckpt_format']} | 张量数={infoB['ckpt_tensors']} | 模型键数={infoB['model_keys']}")
    print(f"  missing={infoB['n_missing']}  unexpected={infoB['n_unexpected']}"
          f"  (按形状丢弃 {len(infoB['dropped_by_shape'])}，按前缀丢弃 {len(infoB['dropped_by_prefix'])})")
    for k in infoB["missing_keys"]:
        print(f"    missing   : {k}")
    for k in infoB["unexpected_keys"][:10]:
        print(f"    unexpected: {k}")
    print(f"  参数量 = {infoB['model_params']}")

    # 硬断言（DoD #12 的语义）：unexpected 必须为 0，missing 只允许是分类头
    assert infoB["n_unexpected"] == 0, \
        f"路线 B 仍有 {infoB['n_unexpected']} 个无法安置的键：{infoB['unexpected_keys'][:5]}"
    bad = [k for k in infoB["missing_keys"] if not HEAD_KEYS.match(k)]
    assert not bad, f"骨干参数缺失，权重与模型不匹配：{bad}"

    eq = None
    infoA = None
    if a.hf_weights:
        hfw = (ROOT / a.hf_weights).resolve()
        _, infoA = route_a_timm(hfw, a.num_classes)
        print(f"\n[路线 A] timm/HF 权重 -> timm RepVit ({a.num_classes} 类)")
        print(f"  missing={infoA.get('n_missing')}  unexpected={infoA.get('n_unexpected')}")

        # 等价性：两侧都用 1000 类 + 双头，比同一个输出
        print("\n[等价性] 官方 ckpt (双头) vs timm-HF 权重 (双头) —— 必须同为 1000 类/双头再比")
        mo = mod.repvit_m0_9(num_classes=1000, distillation=True)
        sdo = unwrap(torch.load(ckpt, map_location="cpu", weights_only=False))[0]
        mo.load_state_dict(sdo, strict=True)
        mo.eval()
        import timm
        mt = timm.create_model("repvit_m0_9", pretrained=False,
                               num_classes=1000, distillation=True)
        sdt = unwrap(torch.load(hfw, map_location="cpu", weights_only=False))[0]
        mt.load_state_dict(sdt, strict=True)
        mt.eval()
        eq = equivalence(mo, mt)
        print(f"  max|Δlogits| = {eq['max_abs_diff']:.6e}   mean|Δ| = {eq['mean_abs_diff']:.6e}")
        print(f"  Top-1 一致率 = {eq['top1_agreement']:.3f}   逐位相同 = {eq['bitwise_identical']}")
        if eq["bitwise_identical"]:
            print("  => 两条路线的权重**逐位相同**，不需要手写 713->389 键转换器。")

    payload = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        # ---- DoD #12 契约键（验收命令直接取这两个名字）----
        "n_missing": infoB["n_missing"],
        "n_unexpected": infoB["n_unexpected"],
        # ---- 明细 ----
        "route_canonical": "B_official",
        "route_b": infoB,
        "route_a": infoA,
        "equivalence_official_vs_timm": eq,
        "head_key_regex": HEAD_KEYS.pattern,
        "note": ("n_missing/n_unexpected 取自路线 B（官方 .pth 配 vendored 官方实现，同体系配对）。"
                 "官方 .pth 与 timm 实现的键名空间不相交（features.N.* / classifier.classifier.* "
                 "对 stages.M.* / head.head.*，实测交集为 0），跨体系加载必然 unexpected=713，"
                 "故 DoD #12 的判据只能在同体系配对下成立。"),
    }
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"\n[report] -> {out.relative_to(ROOT).as_posix()}")
    print("[PASS] DoD #12：n_unexpected=0，missing 仅含分类头键")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
