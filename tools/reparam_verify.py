# -*- coding: utf-8 -*-
"""tools/reparam_verify.py —— RepViT 结构重参数化验证工具

产物（均落在 --out-dir 下，默认 outputs/reparam，<model> = --model）：
      <model>_reparam_report.json / <model>_structure_before.txt / <model>_structure_after.txt
      <model>_onnx_nodes.json / logits_diff.json / onnx_before.onnx / onnx_after.onnx

用法见 9.1.1。禁止写死本机绝对路径，全部路径由 CLI 传入。
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- PASS 阈值（9.5 节）：只有这三条同时满足才算通过 ----
MAX_ABS_ERR_THRESHOLD = 1e-4     # logits 最大绝对误差
TOP1_SAME_RATE_THRESHOLD = 1.0   # Top-1 一致率必须 100%
TOP5_MIN_OVERLAP = 5             # 每条样本 Top-5 交集必须满 5


# ============================ 1. CLI ============================
def parse_args():
    p = argparse.ArgumentParser(description="RepViT 结构重参数化验证")
    p.add_argument("--model", default="repvit_m0_9")
    p.add_argument("--impl", default="timm", choices=["timm", "official"])
    p.add_argument("--weights", default=None,
                   help="checkpoint 路径；若路径不存在则当作 timm 预训练 tag 透传")
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=20240912)
    p.add_argument("--opset", type=int, default=13)
    p.add_argument("--distillation", type=int, default=1)
    p.add_argument("--skip-onnx", action="store_true")
    p.add_argument("--out-dir", default="outputs/reparam")
    args = p.parse_args()
    assert args.num_samples % args.batch_size == 0, "--num-samples 必须是 --batch-size 的整数倍"
    return args


# ============================ 2. 模型构建 ============================
def build_model_timm(name, num_classes, distillation, pretrained=False):
    """timm 实现：RepVitClassifier 默认带蒸馏头，与官方 ckpt 口径一致。"""
    import timm
    return timm.create_model(name, pretrained=pretrained,
                             num_classes=num_classes, distillation=bool(distillation)), None


def build_model_official(name, num_classes, distillation):
    """官方实现：从独立文件加载，绝不走 timm.create_model（注册表会被覆盖）。"""
    import timm  # 官方代码依赖 timm.register_model，必须先 import timm
    path = REPO_ROOT / "models" / "repvit_official.py"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 不存在。请先执行："
            "curl -sL https://raw.githubusercontent.com/THU-MIG/RepViT/main/model/repvit.py "
            "-o models/repvit_official.py")
    spec = importlib.util.spec_from_file_location("repvit_official", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["repvit_official"] = mod
    spec.loader.exec_module(mod)          # 这一步会覆盖 timm 里的 repvit_* 注册项
    print("[warn] 已加载官方实现，本进程内 timm.create_model('repvit_*') 不再可用")
    model = getattr(mod, name)(num_classes=num_classes, distillation=bool(distillation))
    return model, mod


# ============================ 3. 权重加载 ============================
def load_weights(model, weights):
    """加载本地权重文件（weights 必须是已存在的路径）。
    非本地路径视作 timm 预训练 tag，由 build_model_timm 的 pretrained= 处理，不走这里。"""
    info = {"source": str(weights), "missing": [], "unexpected": [],
            "n_keys_loaded": 0, "state_dict_probe": None}
    p = Path(weights)
    if not p.exists():
        info["source"] = None
        info["note"] = f"{weights} 不是本地文件，已作为 timm 预训练 tag 透传 pretrained="
        return info
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(p))
    else:
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        # 官方 ckpt 结构是 {'model': state_dict, 'optimizer': ..., 'epoch': ...}
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    info["state_dict_probe"] = is_fused_state_dict(sd)
    if info["state_dict_probe"]["fused"]:
        raise RuntimeError(f"{weights} 已是推理态 checkpoint，无法载入训练态模型做对照（见 9.4.4）")
    info["n_keys_loaded"] = len(sd)
    # 换 num_classes 时按形状过滤，strict=False 无法绕过 size mismatch
    own = model.state_dict()
    sd = {k: v for k, v in sd.items() if k not in own or v.shape == own[k].shape}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    info["missing"] = list(missing)[:20]
    info["unexpected"] = list(unexpected)[:20]
    info["n_missing"] = len(missing)
    info["n_unexpected"] = len(unexpected)
    if missing or unexpected:
        raise RuntimeError(
            f"权重与结构不匹配：missing={len(missing)}, unexpected={len(unexpected)}。"
            "请使用匹配的 registry key/实现；禁止把随机或部分加载模型的误差作为权重验证结果。")
    return info


def is_fused_state_dict(sd):
    """判定一个 state_dict 是否已经是推理态（融合过）。
    训练态特征：存在 *.bn.running_var / *.bn.num_batches_tracked；
    推理态特征：一个 BN 键都没有，且出现融合后新生成的 bias。
    判据**不能**写成「后缀 .conv.bias」：两类实现的 RepVGGDW 都被整体替换成裸 Conv2d，
    融合后的键尾只有 .bias（official: features.1.token_mixer.0.bias；
    timm: stages.0.blocks.0.token_mixer.bias），没有 .conv 这一层。"""
    keys = list(sd.keys())
    n_bn = sum(1 for k in keys if k.endswith("running_var"))
    has_bn_keys = any(k.endswith(("running_var", "num_batches_tracked")) for k in keys)
    n_bias = sum(1 for k in keys if k.endswith(".bias"))
    return {"n_running_var": n_bn, "n_bias": n_bias,
            "fused": (not has_bn_keys) and n_bias > 0}


# ============================ 4. 融合入口 ============================
def replace_batchnorm(net):
    """官方 THU-MIG/RepViT `utils.py::replace_batchnorm` 的等价复刻（递归、就地 setattr）。

    这里把这段十几行的递归直接实现进本文件，而不是再 vendored 一份官方 `utils.py`
    （仓库目录树里没有 `models/official_utils.py`，也不允许凭空多出一个文件）；
    行为与官方一致：遇到带 `fuse()` 的子模块就替换并继续递归，遇到裸 `BatchNorm2d`
    直接换成 `Identity`，其余子模块继续往下走。"""
    for child_name, child in net.named_children():
        if hasattr(child, "fuse"):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, nn.BatchNorm2d):
            setattr(net, child_name, nn.Identity())
        else:
            replace_batchnorm(child)


def fuse_model(model, impl):
    """按实现分派到 timm 的 model.fuse() 或官方的 replace_batchnorm()。"""
    if impl == "timm":
        assert hasattr(model, "fuse"), "timm RepVit 应有顶层 fuse() 方法"
        model.fuse()
        return model
    # 官方：RepViT 顶层类没有 fuse()，必须用 utils.replace_batchnorm 的递归逻辑
    replace_batchnorm(model)
    return model


# ============================ 5. 结构证据 ============================
BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d)
MULTIBRANCH_NAMES = ("RepVGGDW", "RepVggDw", "Residual")


def module_census(model):
    """统计模块类型直方图与关键计数，用于 structure_*.txt 与 report。"""
    tcount = Counter(type(m).__name__ for m in model.modules())
    return {
        "n_bn": sum(1 for m in model.modules() if isinstance(m, BN_TYPES)),
        "n_conv2d": sum(1 for m in model.modules() if isinstance(m, nn.Conv2d)),
        "n_sequential": tcount.get("Sequential", 0),
        "n_repvggdw": sum(tcount.get(n, 0) for n in ("RepVGGDW", "RepVggDw")),
        "n_linear": sum(1 for m in model.modules() if isinstance(m, nn.Linear)),
        "type_histogram": dict(sorted(tcount.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def param_census(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    sd = model.state_dict()
    return {"params_total": total, "params_trainable": trainable,
            "buffers_numel": buffers, "state_dict_keys": len(sd)}


def dump_structure(model, out_path, note=""):
    """把 named_modules 的层级结构与参数量写成人类可读文本。"""
    lines = [f"# {note}", f"# 模块总数 = {sum(1 for _ in model.named_modules())}"]
    for name, m in model.named_modules():
        depth = 0 if name == "" else name.count(".") + 1
        own = sum(p.numel() for p in m.parameters(recurse=False))
        lines.append(f"{'  ' * depth}{name or '<root>':<58} {type(m).__name__:<22} {own:>9}")
    c = module_census(model)
    lines += ["", "## 类型直方图", json.dumps(c["type_histogram"], indent=2, ensure_ascii=True)]
    lines += ["", "## 关键计数",
              f"BatchNorm1d/2d = {c['n_bn']}",
              f"Conv2d         = {c['n_conv2d']}",
              f"RepVGGDW       = {c['n_repvggdw']}",
              f"Linear         = {c['n_linear']}"]
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


# ============================ 6. 数值证据 ============================
def make_fixed_inputs(seed, num_samples, batch_size, size):
    """用独立 generator 生成固定随机输入；每个 batch 也可单独落盘复现。"""
    g = torch.Generator().manual_seed(seed)
    chunks = []
    for _ in range(num_samples // batch_size):
        chunks.append(torch.randn(batch_size, 3, size, size, generator=g))
    return chunks


@torch.no_grad()
def forward_all(model, chunks):
    model.eval()
    return torch.cat([model(x) for x in chunks], dim=0)


def diff_stats(y_before, y_after, topk=5, eps=1e-6):
    """核心数值判据。注意 max_rel_err 只作报告，不作为门限（见 9.5）。"""
    assert y_before.shape == y_after.shape
    d = (y_before - y_after).abs()
    rel = d / (y_before.abs() + eps)
    n = y_before.shape[0]

    top1_a, top1_b = y_before.argmax(1), y_after.argmax(1)
    top1_same = int((top1_a == top1_b).sum().item())

    tk_a = y_before.topk(topk, dim=1).indices
    tk_b = y_after.topk(topk, dim=1).indices
    # 注意：zip 迭代的是 tk_*.tolist() 的**外层**列表，所以 a / b 已经是 Python list，
    # 再调一次 .tolist() 必抛 AttributeError（规格书原稿此处即为此 bug，实测复现）。
    # 直接 set(a) & set(b)。
    overlaps = [len(set(a) & set(b))
                for a, b in zip(tk_a.tolist(), tk_b.tolist())]

    return {
        "num_samples": n,
        "max_abs_err": float(d.max().item()),
        "mean_abs_err": float(d.mean().item()),
        "max_rel_err": float(rel.max().item()),
        "rel_err_definition": "|before-after| / (|before| + 1e-6)，仅报告不作门限",
        "logits_abs_max": float(y_before.abs().max().item()),
        "top1_same_count": top1_same,
        "top1_same_rate": top1_same / n,
        "top5_overlap_histogram": {str(k): overlaps.count(k) for k in range(topk + 1)},
        "top5_min_overlap": int(min(overlaps)),
        "top5_full_overlap_rate": sum(1 for v in overlaps if v == topk) / n,
        "thresholds": {"max_abs_err": MAX_ABS_ERR_THRESHOLD,
                       "top1_same_rate": TOP1_SAME_RATE_THRESHOLD,
                       "top5_min_overlap": TOP5_MIN_OVERLAP},
    }


def judge(diff):
    ok_num = diff["max_abs_err"] < MAX_ABS_ERR_THRESHOLD
    ok_top1 = diff["top1_same_rate"] >= TOP1_SAME_RATE_THRESHOLD
    ok_top5 = diff["top5_min_overlap"] >= TOP5_MIN_OVERLAP
    return {"numeric_pass": ok_num, "top1_pass": ok_top1,
            "top5_pass": ok_top5, "pass": bool(ok_num and ok_top1 and ok_top5)}


# ============================ 7. ONNX 证据 ============================
def export_onnx(model, out_path, size, opset=17):
    """导出固定 batch=1、动态 batch 轴的 FP32 ONNX。dynamo=False 避免外置 .onnx.data。"""
    model.eval()
    dummy = torch.randn(1, 3, size, size)
    kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=opset,
                  do_constant_folding=True,
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, str(out_path), dynamo=False, **kwargs)
    except TypeError:
        # torch < 2.6 没有 dynamo 关键字，退回 TorchScript 路径
        torch.onnx.export(model, dummy, str(out_path), **kwargs)
    return out_path


def onnx_node_stats(path):
    """统计 op_type 直方图与关键算子计数。"""
    import onnx
    g = onnx.load(str(path))
    ops = Counter(n.op_type for n in g.graph.node)
    n_elem = 0
    for t in g.graph.initializer:
        n = 1
        for d in t.dims:
            n *= int(d)
        n_elem += n
    return {
        "file": Path(path).name,
        "size_bytes": Path(path).stat().st_size,
        "total_nodes": len(g.graph.node),
        "BatchNormalization": ops.get("BatchNormalization", 0),
        "Conv": ops.get("Conv", 0),
        "Add": ops.get("Add", 0),
        "Mul": ops.get("Mul", 0),
        "Div": ops.get("Div", 0),
        "Gemm": ops.get("Gemm", 0),
        "initializer_count": len(g.graph.initializer),
        "initializer_numel": n_elem,
        "op_histogram": dict(sorted(ops.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


# ============================ 8. 主流程 ============================
def resolve_registry_key(args):
    """--model 允许传 deploy/model_registry.py 的登记键（如 repvit_m0_9_pet37）。

    登记键不是 timm 模型名，直接喂给 timm.create_model 会抛
    RuntimeError: Unknown model (repvit_m0_9_pet37)。
    DoD #30 的验收命令用的正是登记键，所以这里做一次解析：
    键命中登记表 -> 用登记项里的 arch/impl/num_classes/ckpt 覆盖 CLI 默认值；
    未命中 -> 原样返回（当作 timm 名或 tag 处理），不改变原有行为。
    """
    try:
        from deploy.model_registry import MODELS
    except Exception:
        return args
    if args.model not in MODELS:
        return args
    e = MODELS[args.model]
    print(f"[registry] {args.model} -> arch={e['arch']} impl={e['impl']} "
          f"num_classes={e['num_classes']} distillation={bool(e['distillation'])}")
    args.impl = e["impl"]
    args.num_classes = e["num_classes"]
    args.distillation = bool(e["distillation"])
    if not args.weights and e.get("ckpt"):
        args.weights = str(e["ckpt"])
    # 归档名改用登记键，产物才落在 <registry_key>_*，与 DoD #30 的路径一致
    args.registry_key = args.model
    args.model = e["arch"]
    return args


def main():
    args = parse_args()
    args = resolve_registry_key(args)
    model_key = getattr(args, "registry_key", args.model)   # 产物文件名统一用这个
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    report = {"cli": vars(args), "fatal": None, "env": {}}
    import timm, torch as _t
    report["env"] = {"torch": _t.__version__, "timm": timm.__version__}
    try:
        import onnx
        report["env"]["onnx"] = onnx.__version__
    except ImportError:
        report["env"]["onnx"] = None

    # --- 构建训练态模型 ---
    if args.impl == "timm":
        model, _ = build_model_timm(
            args.model, args.num_classes, args.distillation,
            pretrained=(args.weights if args.weights and not Path(args.weights).exists() else False))
    else:
        model, _ = build_model_official(args.model, args.num_classes, args.distillation)

    report["weights"] = (load_weights(model, args.weights)
                         if args.weights and Path(args.weights).exists()
                         else {"source": None, "missing": [], "unexpected": [],
                               "note": "未加载本地权重（无 --weights，或作为 timm tag 已在 pretrained= 中生效）"})

    model.eval()                                    # 关键：必须在 fuse 之前
    before_para, before_mod = param_census(model), module_census(model)
    dump_structure(model, out_dir / f"{model_key}_structure_before.txt", "训练态（多分支）结构")

    # --- 融合前 logits ---
    chunks = make_fixed_inputs(args.seed, args.num_samples, args.batch_size, args.input_size)
    torch.save(chunks[0], out_dir / "input_ref.pt")   # 落盘，供评委复现同一次输入
    y_before = forward_all(model, chunks).clone()     # 必须 clone

    # --- 融合 ---
    fused = copy.deepcopy(model)                      # fuse 是原地改写，必须深拷贝
    fuse_model(fused, args.impl)
    fused.eval()
    assert module_census(fused)["n_bn"] == 0, "融合后仍残留 BatchNorm，融合入口用错了"
    y_after = forward_all(fused, chunks)

    # --- 数值 / 结构 / 参数 ---
    diff = diff_stats(y_before, y_after)
    after_para, after_mod = param_census(fused), module_census(fused)
    dump_structure(fused, out_dir / f"{model_key}_structure_after.txt", "推理态（融合后）结构")
    (out_dir / "logits_diff.json").write_text(
        json.dumps({"diff": diff, "judge": judge(diff)}, indent=2, ensure_ascii=False), encoding="utf-8")

    report["params"] = {"before": before_para, "after": after_para, "delta": {
        k: after_para[k] - before_para[k] for k in before_para}}
    report["modules"] = {"before": before_mod, "after": after_mod}
    # selfcheck 的 c_rep 直接读 report["before"]["n_bn"] / report["after"]["n_bn"]，
    # 而本报告把模块普查放在 report["modules"]["before|after"] 下。两处指同一份数据，
    # 顶层再镜像一份，避免「检查器与生产者的路径约定不一致」被误判成产物缺失。
    report["before"] = before_mod
    report["after"] = after_mod
    report["diff"] = diff
    report["judge"] = judge(diff)
    # c_rep 读 judge["max_abs_err"] 与 judge["top1_agree"]；本仓 judge 用的是
    # numeric_pass / top1_pass / top5_pass 三个布尔。把数值一并挂到 judge 下
    # （同名键不覆盖原有布尔，纯新增）。
    report["judge"]["max_abs_err"] = diff["max_abs_err"]
    report["judge"]["mean_abs_err"] = diff["mean_abs_err"]
    report["judge"]["top1_agree"] = diff["top1_same_rate"]
    # 顶层镜像一份最常用的判定数字，验收脚本可以一行取值（见 D2-00 DoD #29）
    report.update({"max_abs_err": diff["max_abs_err"], "mean_abs_err": diff["mean_abs_err"],
                   "top1_same_rate": diff["top1_same_rate"],
                   "top1_identical": diff["top1_same_rate"] == 1.0,
                   "num_samples": diff["num_samples"]})

    # --- ONNX ---
    if not args.skip_onnx:
        p_b = export_onnx(model, out_dir / "onnx_before.onnx", args.input_size, args.opset)
        p_a = export_onnx(fused, out_dir / "onnx_after.onnx", args.input_size, args.opset)
        s_b, s_a = onnx_node_stats(p_b), onnx_node_stats(p_a)
        (out_dir / f"{model_key}_onnx_nodes.json").write_text(json.dumps({
            "before": s_b, "after": s_a,
            "delta": {k: s_a[k] - s_b[k] for k in
                      ("total_nodes", "BatchNormalization", "Conv", "Add", "Mul", "Div")},
            "reference_magnitude": {
                "note": "以下为某次实测的参考量级，不是标准答案。口径：opset 13、1x3x224x224、batch=1、"
                        "do_constant_folding=True、torch 2.14 + timm 1.0.29 + onnx 1.22（CPU）；"
                        "同一口径下换成官方仓库实现（distillation=True → replace_batchnorm 后）是 682 -> 464，"
                        "两个总数只差若干 Identity 节点。硬判据是 BatchNormalization=0 与 Conv 减少 23，"
                        "节点总数随实现/torch 版本浮动 ±几到 ±几十",
                "before": {"total_nodes": 681, "BatchNormalization": 25, "Conv": 126},
                "after": {"total_nodes": 463, "BatchNormalization": 0, "Conv": 103}},
        }, indent=2, ensure_ascii=True), encoding="utf-8")
        report["onnx"] = {"before": s_b, "after": s_a}
        # 把 DoD #30 的两条硬判据直接打印出来（验收看的就是这两个数）
        print(f"[onnx] 训练态: total={s_b['total_nodes']} BatchNormalization={s_b['BatchNormalization']} Conv={s_b['Conv']}")
        print(f"[onnx] 推理态: total={s_a['total_nodes']} BatchNormalization={s_a['BatchNormalization']} Conv={s_a['Conv']}"
              f"   (Conv {s_b['Conv']} -> {s_a['Conv']}，少 {s_b['Conv'] - s_a['Conv']})")
        assert s_a["BatchNormalization"] == 0, f"推理态图里仍有 {s_a['BatchNormalization']} 个 BatchNormalization 节点"
        assert s_a["Conv"] == 103, f"推理态 Conv 节点应为 103，实测 {s_a['Conv']}"
        # 文件大小对比（进阶任务「比较转换前后的模型文件大小」）
        report["onnx_size_mb"] = {"before": s_b["size_bytes"] / 1e6, "after": s_a["size_bytes"] / 1e6}

    (out_dir / f"{model_key}_reparam_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    # --- 控制台摘要 ---
    print(f"[logits] max_abs_err={diff['max_abs_err']:.3e} mean_abs_err={diff['mean_abs_err']:.3e} "
          f"top1_same={diff['top1_same_count']}/{diff['num_samples']} "
          f"top5_min_overlap={diff['top5_min_overlap']}")
    # 融合后的参数量按 M 打印（见 9.2.1）：报告里一律写「约 X.XXXM」，不逐位写整数
    print(f"[params] {before_para['params_trainable']} -> 约 {after_para['params_trainable'] / 1e6:.3f}M "
          f"(Δ={after_para['params_trainable'] - before_para['params_trainable']})  "
          f"buffers {before_para['buffers_numel']} -> {after_para['buffers_numel']}")
    print(f"[verdict] {'PASS' if report['judge']['pass'] else 'FAIL'}  -> {out_dir}")
    return 0 if report["judge"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
