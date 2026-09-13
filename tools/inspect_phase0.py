# tools/inspect_phase0.py
# -*- coding: utf-8 -*-
"""Phase 0 关键判定：官方权重到底能不能用、走哪条实现路线。

背景（spec §2.1 约束三）：官方 model/repvit.py 带 @register_model 装饰器，
一旦被正常 import 就会覆盖 timm 注册表里的 repvit_m0_9，导致之后
timm.create_model('repvit_m0_9', pretrained=True) 抛 TypeError。
因此必须用 exec 加载 + 把 register_model 换成恒等装饰器。

本脚本回答三个问题：
  A. 官方 ckpt 的键名长什么样（前缀分布、与 timm 键名的交集）
  B. 官方 ckpt 载入 vendored 官方实现 -> missing / unexpected 是多少
  C. timm 实现里蒸馏头的真实属性名（冒烟里 drop_distill_head 失效的原因）

用法：
    python tools/inspect_phase0.py --ckpt checkpoints/pretrained/repvit_m0_9_distill_300e.pth
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_official_module(path: Path):
    """按 spec §2.1 约束三 加载官方实现：exec + register_model 中性化。"""
    src = path.read_text(encoding="utf-8")
    # 官方写法是 `from timm.models import register_model` + 裸 @register_model
    neutral = (
        "def register_model(fn=None, **kw):\n"
        "    if fn is None:\n"
        "        return lambda f: f\n"
        "    return fn\n"
    )
    src = re.sub(r"^from timm\.models import register_model.*$", neutral,
                 src, flags=re.M)
    src = re.sub(r"^from timm\.models\.registry import register_model.*$", neutral,
                 src, flags=re.M)
    ns = {"__name__": "repvit_official_vendored", "__file__": str(path)}
    exec(compile(src, str(path), "exec"), ns)          # noqa: S102
    return ns


def top_prefix(key: str, depth: int = 2) -> str:
    return ".".join(key.split(".")[:depth])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pretrained/repvit_m0_9_distill_300e.pth")
    ap.add_argument("--official", default="models/repvit_official.py")
    ap.add_argument("--out", default="outputs/metrics/inspect_phase0.json")
    a = ap.parse_args()

    ckpt = Path(a.ckpt) if Path(a.ckpt).is_absolute() else ROOT / a.ckpt
    off_py = Path(a.official) if Path(a.official).is_absolute() else ROOT / a.official
    rep = {}

    print("=" * 72)
    # ---------- A. 官方 ckpt 键名 ----------
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"] if (isinstance(ck, dict) and "model" in ck) else ck
    rep["ckpt"] = {"file": ckpt.name, "bytes": ckpt.stat().st_size,
                   "n_keys": len(sd), "wrapper_keys": sorted(ck.keys()) if isinstance(ck, dict) else None}
    print(f"[A] {ckpt.name}  {ckpt.stat().st_size:,} bytes  {len(sd)} 键")
    prefix = Counter(top_prefix(k) for k in sd)
    rep["ckpt"]["top_prefix"] = dict(prefix.most_common())
    print(f"    顶层前缀分布: {dict(prefix.most_common())}")
    print(f"    样例键: {sorted(sd.keys())[:4]}")
    print(f"    分类头键: {[k for k in sorted(sd) if 'classifier' in k or 'head' in k][:8]}")

    # ---------- C. timm 侧结构：找蒸馏头真实属性名 ----------
    import timm
    tm = timm.create_model("repvit_m0_9", num_classes=1000)
    rep["timm_head"] = {
        "type": type(tm).__name__,
        "has_head_dist": hasattr(tm, "head_dist"),
        "child_modules": [n for n, _ in tm.named_children()],
        "head_area_params": {n: p.numel() for n, p in tm.named_parameters()
                             if n.startswith("head")},
    }
    print("-" * 72)
    print(f"[C] timm 模型类 = {type(tm).__name__}")
    print(f"    子模块: {rep['timm_head']['child_modules']}")
    print(f"    hasattr(head_dist) = {rep['timm_head']['has_head_dist']}")
    print(f"    head* 参数项: {list(rep['timm_head']['head_area_params'].items())[:10]}")
    tm_keys = set(dict(tm.named_parameters()).keys())
    rep["timm_head"]["n_params_named"] = len(tm_keys)

    # ---------- 键名交集：判断转换工作量 ----------
    common = set(sd.keys()) & tm_keys
    rep["key_overlap_official_vs_timm"] = {
        "official": len(sd), "timm": len(tm_keys),
        "intersection": len(common), "only_official": len(set(sd) - tm_keys),
    }
    print(f"    官方键 ∩ timm 键 = {len(common)} / 官方 {len(sd)} / timm {len(tm_keys)}"
          f"  -> 直接载入不可行，需键映射" if len(common) < len(sd) * 0.5 else "  -> 键名基本一致")
    print(f"    官方独有键样例: {sorted(set(sd)-tm_keys)[:4]}")
    print(f"    timm 独有键样例: {sorted(tm_keys-set(sd))[:4]}")

    # ---------- B. 官方权重载入 vendored 官方实现 ----------
    print("-" * 72)
    if not off_py.exists():
        print(f"[B] 官方实现文件不存在: {off_py}")
        rep["official_load"] = "file_missing"
    else:
        ns = load_official_module(off_py)
        for distill in (True, False):
            m = ns["repvit_m0_9"](pretrained=False, num_classes=1000, distillation=distill)
            miss, unexp = m.load_state_dict(sd, strict=False)
            n_par = sum(p.numel() for p in m.parameters())
            tag = "distill=True " if distill else "distill=False"
            rep[f"official_load_distill_{distill}"] = {
                "n_params": n_par, "n_missing": len(miss), "n_unexpected": len(unexp),
                "missing": list(miss)[:10], "unexpected": list(unexp)[:10],
            }
            print(f"[B] vendored 官方实现 {tag}:")
            print(f"    参数量={n_par:,}  missing={len(miss)}  unexpected={len(unexp)}"
                  f"  -> {'PASS (unexpected=0)' if len(unexp)==0 else 'FAIL'}")
            if miss:
                print(f"    missing 样例: {list(miss)[:5]}")
            if unexp:
                print(f"    unexpected 样例: {list(unexp)[:5]}")

        # 官方实现自己的 fuse()：验证融合后口径
        m = ns["repvit_m0_9"](pretrained=False, num_classes=1000, distillation=False)
        m.load_state_dict(sd, strict=False)
        m.eval()
        pre = sum(p.numel() for p in m.parameters())
        if hasattr(m, "fuse"):
            m.fuse()
        post = sum(p.numel() for p in m.parameters())
        rep["official_fuse"] = {"before": pre, "after": post, "delta": pre - post}
        print(f"    官方 fuse(): {pre:,} -> {post:,}  (净减 {pre-post:,})")

    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=True, indent=2, default=str)
    print("=" * 72)
    print(f"[inspect] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
