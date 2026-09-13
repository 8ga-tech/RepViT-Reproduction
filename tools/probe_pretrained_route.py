# tools/probe_pretrained_route.py
# -*- coding: utf-8 -*-
"""Phase 0 侦察：timm 侧能不能拿到可用的 ImageNet 预训练权重（绕过手写键转换器）。

背景：官方 .pth 的键（features.N.* / classifier.classifier.*）与 timm 的
键（stages.M.* / head.head.*）交集为 0，且 timm 没有内置 convert_weights。
若 hf-mirror 上的 timm/repvit_m0_9.dist_300e_in1k 可用，就有第二条通路。

本脚本回答：
  1. 设置 HF_ENDPOINT=hf-mirror.com 后，timm create_model(..., pretrained=True) 能否拿到权重
  2. timm 的键 vs 官方 .pth 的键，逐段对照，量化手写转换器的工作量
  3. 若能拿到，则与官方实现做三方 logits 比对（官方 .pth / timm-HF 权重）

用法：
    set HF_ENDPOINT=https://hf-mirror.com && python tools/probe_pretrained_route.py
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch                                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-ckpt",
                    default="checkpoints/pretrained/repvit_m0_9_distill_300e.pth")
    ap.add_argument("--out", default="outputs/metrics/probe_route.json")
    a = ap.parse_args()

    rep = {"HF_ENDPOINT": os.environ.get("HF_ENDPOINT")}
    print("=" * 72)
    print(f"[env] HF_ENDPOINT = {rep['HF_ENDPOINT']}")

    # ---------- 1. timm 侧 pretrained=True 能否拉到权重 ----------
    import timm
    print("-" * 72)
    try:
        tm = timm.create_model("repvit_m0_9", pretrained=True, num_classes=1000)
        tm.eval()
        rep["timm_pretrained"] = "ok"
        print("[1] timm create_model('repvit_m0_9', pretrained=True) -> OK")
        # 记录权重来源与本地缓存路径
        from timm.models._hub import load_state_dict_from_hf
        rep["timm_cfg"] = {k: str(v) for k, v in
                           (timm.get_pretrained_cfg("repvit_m0_9").to_dict().items())}
        print(f"    hf_hub_id  = {rep['timm_cfg'].get('hf_hub_id')}")
        print(f"    tag        = {rep['timm_cfg'].get('tag')}")
        print(f"    mean/std   = {rep['timm_cfg'].get('mean')} / {rep['timm_cfg'].get('std')}")
        print(f"    crop_pct   = {rep['timm_cfg'].get('crop_pct')}  "
              f"interpolation = {rep['timm_cfg'].get('interpolation')}")
        print(f"    input_size = {rep['timm_cfg'].get('input_size')}")
    except Exception as e:                                       # noqa: BLE001
        rep["timm_pretrained"] = f"FAIL {type(e).__name__}: {e}"
        print(f"[1] timm pretrained 失败: {type(e).__name__}: {e}")
        tm = None

    # ---------- 2. 键结构逐段对照：量化手写转换器的工作量 ----------
    print("-" * 72)
    ckpt = Path(a.official_ckpt)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    official = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

    if tm is not None:
        tkeys = list(dict(tm.named_parameters()).keys())
        okeys = list(official.keys())
        rep["key_census"] = {"official": len(okeys), "timm": len(tkeys),
                             "intersection": len(set(okeys) & set(tkeys))}

        # 按「去掉数字索引」归并出结构模式，直接看两边拓扑能否一一对应
        import re as _re

        def canon(ks):
            out = {}
            for k in ks:
                c = _re.sub(r"\d+", "N", k)
                out[c] = out.get(c, 0) + 1
            return out

        co, ct = canon(okeys), canon(tkeys)
        print(f"[2] 官方键 {len(okeys)} / timm 键 {len(tkeys)} / 交集 "
              f"{rep['key_census']['intersection']}")
        print(f"    官方结构模式（去索引后）：{len(co)} 种")
        for k in sorted(co)[:12]:
            print(f"      {co[k]:>3d}x  {k}")
        print(f"    timm 结构模式（去索引后）：{len(ct)} 种")
        for k in sorted(ct)[:12]:
            print(f"      {ct[k]:>3d}x  {k}")
        rep["canon_official"] = co
        rep["canon_timm"] = ct

    # ---------- 3. 等价性比对 ----------
    #  关键：必须在**相同 distillation 状态**下比对，否则比的是两个不同输出
    #  （官方 Classfier 在 eval + distillation=True 时返回两头的平均）
    print("-" * 72)
    off_py = ROOT / "models" / "repvit_official.py"
    if tm is not None and off_py.exists():
        import re
        src = off_py.read_text(encoding="utf-8")
        neutral = ("def register_model(fn=None, **kw):\n"
                   "    if fn is None:\n        return lambda f: f\n    return fn\n")
        src = re.sub(r"^from timm\.models import register_model.*$", neutral, src, flags=re.M)
        ns = {"__name__": "repvit_official_vendored"}
        exec(compile(src, str(off_py), "exec"), ns)              # noqa: S102

        import torch.nn.functional as F

        def off_feat(m, x):
            """复刻官方 RepViT.forward 的前半段（官方没有 forward_features 方法）。"""
            for f in m.features:
                x = f(x)
            return F.adaptive_avg_pool2d(x, 1).flatten(1)

        def tm_feat(m, x):
            return m.forward_head(m.forward_features(x), pre_logits=True)

        x = torch.randn(4, 3, 224, 224)
        rep["equivalence"] = {}

        # 双方都建 distillation=True（官方 .pth 是双头权重），但比对时显式取头，
        # 不去删模块、不依赖 forward 里的 distillation 开关（删了会让 forward 抛 AttributeError）
        om = ns["repvit_m0_9"](pretrained=False, num_classes=1000, distillation=True)
        r = om.load_state_dict(official, strict=False)
        om.eval()
        tm_a = timm.create_model("repvit_m0_9", pretrained=True, num_classes=1000).eval()
        rep["equivalence"]["official_ckpt_unexpected"] = len(r.unexpected_keys)

        with torch.no_grad():
            fo, ft = off_feat(om, x), tm_feat(tm_a, x)
            df = (fo - ft).abs()
            # 主头 logits
            lo_main = om.classifier.classifier(fo)
            lt_main = tm_a.head.head(ft)
            dl = (lo_main - lt_main).abs()
            # 蒸馏平均 logits（官方 README 的 Top-1 用的就是这个口径）
            lo_avg = (om.classifier.classifier(fo) + om.classifier.classifier_dist(fo)) / 2
            lt_avg = (tm_a.head.head(ft) + tm_a.head.head_dist(ft)) / 2
            da = (lo_avg - lt_avg).abs()

        rep["equivalence"]["pooled_features"] = {
            "max_abs": float(df.max()), "mean_abs": float(df.mean())}
        rep["equivalence"]["main_head_logits"] = {
            "max_abs": float(dl.max()), "mean_abs": float(dl.mean()),
            "top1_same": float((lo_main.argmax(1) == lt_main.argmax(1)).float().mean())}
        rep["equivalence"]["distill_avg_logits"] = {
            "max_abs": float(da.max()), "mean_abs": float(da.mean()),
            "top1_same": float((lo_avg.argmax(1) == lt_avg.argmax(1)).float().mean())}

        print("[3] 官方 .pth（vendored 官方实现）  vs  timm-HF 权重（timm 实现）")
        print(f"    official ckpt unexpected = {rep['equivalence']['official_ckpt_unexpected']}")
        print(f"    (a) 池化特征   max|Δ|={float(df.max()):.3e}  mean|Δ|={float(df.mean()):.3e}")
        print(f"    (b) 主头 logits max|Δ|={float(dl.max()):.3e}  mean|Δ|={float(dl.mean()):.3e}  "
              f"Top-1 一致率={float((lo_main.argmax(1)==lt_main.argmax(1)).float().mean()):.3f}")
        print(f"    (c) 蒸馏平均    max|Δ|={float(da.max()):.3e}  mean|Δ|={float(da.mean()):.3e}  "
              f"Top-1 一致率={float((lo_avg.argmax(1)==lt_avg.argmax(1)).float().mean()):.3f}")
        verdict = float(df.max()) < 1e-3 and float(dl.max()) < 1e-3
        rep["equivalence"]["verdict"] = "EQUIVALENT" if verdict else "NOT_EQUIVALENT"
        print(f"    -> {'PASS：两路数值等价（timm-HF 权重 = 官方权重）' if verdict else 'DIFF：两路不等价，需手写键转换器'}")
    else:
        print("[3] 跳过（缺少 timm 预训练或官方实现文件）")

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=True, indent=2, default=str)
    print("=" * 72)
    print(f"[probe] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
