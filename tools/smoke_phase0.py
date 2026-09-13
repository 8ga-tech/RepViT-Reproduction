# tools/smoke_phase0.py
# -*- coding: utf-8 -*-
"""Phase 0 可行性冒烟：把「能不能做、要多久」这件事用实测数字钉死。

本脚本是 Phase 0 的临时产物（非规格书验收项），M04/M09/M10 会用
tools/count_params.py、tools/reparam_verify.py、deploy/export_onnx.py 正式取代它。
保留原因：它一条命令覆盖了下述五件事，适合换机器时快速复验。

测量项：
  1. timm 裸建 repvit_m0_9 → 参数量（train_distill 口径，期望 5489328）
  2. 加载官方权重 repvit_m0_9_distill_300e.pth → missing / unexpected 键数
  3. C=37 未融合单头参数量（期望 4732805）
  4. eval 后融合 → 参数量（C=1000 期望 5067056 / C=37 期望 4696301）
  5. ONNX 导出 → BatchNormalization 节点数（期望 0）、Conv 数
  6. 训练吞吐：GPU/CPU 上 N 步实测 → 外推 40 epoch（2940 图 / bs64 / 45 step/epoch）

用法：
    python tools/smoke_phase0.py --ckpt checkpoints/pretrained/repvit_m0_9_distill_300e.pth
    python tools/smoke_phase0.py --ckpt ... --steps 20 --device cuda
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "metrics"

# 规格书给定的基准值（timm 1.0.29 / torch 2.14.0）。实测对不上时以实测为准并记入报告。
EXPECT = {
    "params_train_distill_c1000": 5489328,
    "params_train_single_c1000": 5103560,
    "params_backbone": 4717792,
    "params_train_single_c37": 4732805,
    "params_fused_c1000": 5067056,
    "params_fused_c37": 4696301,
}


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def count_backbone(m) -> int:
    """骨干参数量 = 去掉所有分类头（head / head_dist / head.fc 等）后的总和。"""
    total = 0
    for name, p in m.named_parameters():
        if name.startswith("head"):
            continue
        total += p.numel()
    return total


def drop_distill_head(m):
    """删除蒸馏头，得到「未融合单头」口径。

    注意 timm 1.0.29 里蒸馏头的真实路径是 head.head_dist（不是在模型顶层），
    且删掉后必须把 RepVitClassifier.distillation 置 False，否则 forward 会抛
    AttributeError（见 inspect_phase0 的输出）。
    """
    for parent_path, leaf in (("head", "head_dist"), ("", "head_dist")):
        parent = m
        for p in filter(None, parent_path.split(".")):
            parent = getattr(parent, p, None)
            if parent is None:
                break
        if parent is None or not hasattr(parent, leaf):
            continue
        removed = sum(p.numel() for p in getattr(parent, leaf).parameters())
        delattr(parent, leaf)
        if hasattr(parent, "distillation"):
            parent.distillation = False
        return m, removed
    return m, 0


def load_official(model, ckpt_path):
    """加载官方 .pth。官方文件可能是裸 state_dict，也可能是 {'model': sd} 包装。"""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
        sd, wrapper_keys = ck["model"], sorted(ck.keys())
    elif isinstance(ck, dict) and "state_dict" in ck:
        sd, wrapper_keys = ck["state_dict"], sorted(ck.keys())
    else:
        sd, wrapper_keys = ck, ["<bare state_dict>"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return {
        "ckpt_keys": len(sd),
        "wrapper_keys": wrapper_keys,
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
        "missing": list(missing),
        "unexpected": list(unexpected)[:20],
    }


def fuse_model(m):
    """优先用模型自带的 fuse()（timm 的 RepVit 实现了），否则退回 replace_batchnorm。"""
    import timm
    used = None
    if hasattr(m, "fuse") and callable(getattr(m, "fuse")):
        try:
            m.fuse()
            used = "model.fuse()"
        except Exception as e:                   # noqa: BLE001
            used = f"model.fuse() raised {type(e).__name__}: {e}"
    if used is None:
        timm.utils.replace_batchnorm(m)
        used = "timm.utils.replace_batchnorm"
    return used


def count_bn_modules(m) -> int:
    n = 0
    for mod in m.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            n += 1
        elif type(mod).__name__ == "BatchNormAct2d":
            n += 1
    return n


def onnx_node_stats(model, out_path: Path, opset: int = 13):
    import onnx
    dummy = torch.zeros(1, 3, 224, 224)
    model = model.eval().cpu()
    kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=opset)
    try:
        torch.onnx.export(model, dummy, str(out_path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(out_path), **kwargs)
    g = onnx.load(str(out_path))
    c = Counter(nd.op_type for nd in g.graph.node)
    return {"total_nodes": len(g.graph.node),
            "BatchNormalization": c.get("BatchNormalization", 0),
            "Conv": c.get("Conv", 0),
            "op_types_top": dict(c.most_common(8))}


def train_throughput(model, device, steps, batch, amp_dtype, num_threads=None):
    """只测吞吐：合成张量 + AdamW + 前反向。不做任何精度结论（规格书禁止 randn 用于一致性测试）。"""
    if num_threads and device == "cpu":
        torch.set_num_threads(num_threads)
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    crit = torch.nn.CrossEntropyLoss()
    x = torch.randn(batch, 3, 224, 224)
    y = torch.randint(0, 37, (batch,))

    def one():
        xb = x.to(device, non_blocking=True)
        yb = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        if amp_dtype is not None and device == "cuda":
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = crit(model(xb), yb)
        else:
            loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        return float(loss.detach())

    for _ in range(3):                            # 预热
        one()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    losses = [one() for _ in range(steps)]
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"steps": steps, "batch": batch, "sec_per_step": dt / steps,
            "img_per_sec": steps * batch / dt,
            "first_loss": losses[0], "last_loss": losses[-1],
            "sec_per_epoch_45steps": (dt / steps) * 45,
            "hours_40ep": (dt / steps) * 45 * 40 / 3600}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pretrained/repvit_m0_9_distill_300e.pth")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=None, help="CPU 时显式设定 torch 线程数")
    ap.add_argument("--out", default="outputs/metrics/smoke_phase0.json")
    a = ap.parse_args()

    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rep = {"device": device, "torch": torch.__version__,
           "cuda": torch.version.cuda, "expect": EXPECT}
    ckpt = Path(a.ckpt)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt

    print("=" * 72)
    print(f"[0] device={device}  torch={torch.__version__}  cuda={torch.version.cuda}")
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"    GPU: {p.name}  {p.total_memory/1024**3:.1f} GB  sm_{p.major}{p.minor}")

    import timm
    print(f"    timm={timm.__version__}")

    # ---- 1. 裸建：train_distill 口径 ----
    print("-" * 72)
    m = timm.create_model("repvit_m0_9", num_classes=1000).eval()
    rep["params_train_distill_c1000"] = n_params(m)
    print(f"[1] train_distill 双头 C=1000 : {rep['params_train_distill_c1000']:,}"
          f"  (期望 {EXPECT['params_train_distill_c1000']:,})"
          f"  {'PASS' if rep['params_train_distill_c1000']==EXPECT['params_train_distill_c1000'] else 'DIFF'}")
    rep["has_head_dist"] = hasattr(m, "head_dist")

    # ---- 2. 官方权重加载 ----
    print("-" * 72)
    if ckpt.exists():
        info = load_official(m, ckpt)
        rep["weight_load"] = info
        print(f"[2] 官方权重 {ckpt.name} ({ckpt.stat().st_size:,} bytes)")
        print(f"    ckpt 键数={info['ckpt_keys']}  包装键={info['wrapper_keys']}")
        print(f"    missing={info['n_missing']}  unexpected={info['n_unexpected']}"
              f"  -> {'PASS (unexpected=0)' if info['n_unexpected']==0 else 'FAIL'}")
        print(f"    missing 前 6: {info['missing'][:6]}")
    else:
        rep["weight_load"] = "ckpt_missing"
        print(f"[2] 权重不存在，跳过：{ckpt}")

    # ---- 3. 未融合单头 C=1000 / C=37 ----
    print("-" * 72)
    m37 = timm.create_model("repvit_m0_9", num_classes=37).eval()
    if ckpt.exists():                             # 复用已加载的骨干；分类头尺寸不同必然 missing
        load_official(m37, ckpt)
    rep["params_train_single_c1000"] = n_params(drop_distill_head(m)[0])
    rep["params_train_single_c37"] = n_params(drop_distill_head(m37)[0])
    print(f"[3] 未融合单头 C=1000 : {rep['params_train_single_c1000']:,}"
          f"  (期望 {EXPECT['params_train_single_c1000']:,})"
          f"  {'PASS' if rep['params_train_single_c1000']==EXPECT['params_train_single_c1000'] else 'DIFF'}")
    print(f"    未融合单头 C=37   : {rep['params_train_single_c37']:,}"
          f"  (期望 {EXPECT['params_train_single_c37']:,})"
          f"  {'PASS' if rep['params_train_single_c37']==EXPECT['params_train_single_c37'] else 'DIFF'}")

    # ---- 4. 融合后 ----
    print("-" * 72)
    m_f = timm.create_model("repvit_m0_9", num_classes=1000).eval()
    m37_f = timm.create_model("repvit_m0_9", num_classes=37).eval()
    if ckpt.exists():
        load_official(m_f, ckpt)
        load_official(m37_f, ckpt)
    m_f, _ = drop_distill_head(m_f)
    m37_f, _ = drop_distill_head(m37_f)
    bn_before = (count_bn_modules(m_f), count_bn_modules(m37_f))
    how = fuse_model(m_f)
    fuse_model(m37_f)
    rep["fuse_method"] = how
    rep["bn_modules_before"] = bn_before
    rep["bn_modules_after"] = (count_bn_modules(m_f), count_bn_modules(m37_f))
    rep["params_fused_c1000"] = n_params(m_f)
    rep["params_fused_c37"] = n_params(m37_f)
    print(f"[4] 融合方式: {how}")
    print(f"    BN 模块数 {bn_before} -> {rep['bn_modules_after']}")
    print(f"    融合后 C=1000 : {rep['params_fused_c1000']:,}"
          f"  (期望 {EXPECT['params_fused_c1000']:,})"
          f"  {'PASS' if rep['params_fused_c1000']==EXPECT['params_fused_c1000'] else 'DIFF'}")
    print(f"    融合后 C=37   : {rep['params_fused_c37']:,}"
          f"  (期望 {EXPECT['params_fused_c37']:,})"
          f"  {'PASS' if rep['params_fused_c37']==EXPECT['params_fused_c37'] else 'DIFF'}")
    rep["fused_net_reduction_c1000"] = EXPECT["params_train_single_c1000"] - rep["params_fused_c1000"]

    # ---- 5. ONNX 导出与 BN 节点 ----
    print("-" * 72)
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tmp = OUT_DIR / "_smoke_fused.onnx"
        rep["onnx_fused"] = onnx_node_stats(m_f, tmp)
        print(f"[5] ONNX(融合后) 节点: {rep['onnx_fused']}")
        print(f"    BatchNormalization={rep['onnx_fused']['BatchNormalization']}"
              f"  Conv={rep['onnx_fused']['Conv']}"
              f"  -> {'PASS (BN=0)' if rep['onnx_fused']['BatchNormalization']==0 else 'FAIL'}")
        m_unf = timm.create_model("repvit_m0_9", num_classes=1000).eval()
        if ckpt.exists():
            load_official(m_unf, ckpt)
        tmp2 = OUT_DIR / "_smoke_unfused.onnx"
        rep["onnx_unfused"] = onnx_node_stats(m_unf, tmp2)
        print(f"    对照(未融合) BatchNormalization={rep['onnx_unfused']['BatchNormalization']}"
              f"  Conv={rep['onnx_unfused']['Conv']}")
        tmp.unlink(missing_ok=True)
        tmp2.unlink(missing_ok=True)
    except Exception as e:                       # noqa: BLE001
        rep["onnx_fused"] = f"FAIL {type(e).__name__}: {e}"
        print(f"[5] ONNX 导出失败: {type(e).__name__}: {e}")

    # ---- 6. 训练吞吐外推 ----
    print("-" * 72)
    try:
        m_t = timm.create_model("repvit_m0_9", num_classes=37)
        if ckpt.exists():
            load_official(m_t, ckpt)
        amp = torch.bfloat16 if device == "cuda" else None
        rep["train_bench"] = train_throughput(m_t, device, a.steps, a.batch, amp, a.threads)
        tb = rep["train_bench"]
        print(f"[6] 训练吞吐 ({device}, batch={tb['batch']}, {tb['steps']} 步):")
        print(f"    {tb['sec_per_step']:.4f} s/step   {tb['img_per_sec']:.1f} img/s"
              f"   loss {tb['first_loss']:.3f} -> {tb['last_loss']:.3f}")
        print(f"    外推: 45 step/epoch = {tb['sec_per_epoch_45steps']:.1f} s/epoch"
              f"  ->  40 epoch = {tb['hours_40ep']:.2f} h")
    except Exception as e:                       # noqa: BLE001
        import traceback
        rep["train_bench"] = f"FAIL {type(e).__name__}: {e}"
        print(f"[6] 训练吞吐测试失败: {type(e).__name__}: {e}")
        traceback.print_exc()

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=True, indent=2)
    print("=" * 72)
    print(f"[smoke] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
