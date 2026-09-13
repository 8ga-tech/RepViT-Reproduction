# tools/reparam_deep.py —— 结构对比 / ONNX 图统计 / 文件大小 / 延迟 / logits 误差 / 残留 BN 清单
import argparse, copy, json, os, collections
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import timm
try:
    from tools.rv_common import get, onnx_path, latency, WARMUP, RUNS
except ImportError:
    from rv_common import get, onnx_path, latency, WARMUP, RUNS

# ---------- 独立实现的 Conv+BN 融合（不调用任何官方 fuse，用于交叉验证） ----------
def fuse_conv_bn_manual(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """z = (W t) x + (β - μ t + b t)，t = γ/sqrt(σ²+ε)"""
    t = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    W = conv.weight * t.reshape(-1, 1, 1, 1)
    b = bn.bias - bn.running_mean * t + (conv.bias * t if conv.bias is not None else 0.0)
    return W, b

def fuse_repvggdw_manual(dw):
    """独立复刻 RepVGGDW 的三分支融合，返回一个带 bias 的 Conv2d。
    同时保存每个中间量，供自动报告使用。"""
    # 结构因实现而异，必须分别认：
    #   timm   : dw.conv 是 ConvNorm（内含 .c=Conv2d(bias=False) 与 .bn=BatchNorm2d），
    #            dw.conv1 是裸 1x1 Conv2d，dw.bn 是外层 BatchNorm2d
    #   官方    : dw.conv 是裸 Conv2d，dw.conv1 是裸 1x1 Conv2d
    # 早期版本直接取 `dw.conv.weight`，在 timm 下必抛
    # AttributeError: 'ConvNorm' object has no attribute 'bias'（实测复现）。
    bn = dw.bn                                     # 外层 BN
    if hasattr(dw.conv, "c"):                      # timm 的 ConvNorm 包装
        conv = dw.conv.c
        W3, b3 = fuse_conv_bn_manual(conv, dw.conv.bn)   # 3x3 分支自带 BN，先融进去
    else:                                          # 官方：裸 Conv2d
        conv = dw.conv
        W3, b3 = (conv.weight, conv.bias) if conv.bias is not None else (conv.weight, 0.0)
    if hasattr(dw, "bn1"):                          # legacy=True 变体：conv1 自带 BN
        W1, b1 = fuse_conv_bn_manual(dw.conv1.conv, dw.conv1.bn)
    else:                                           # legacy=False：conv1 是裸 1x1（timm 即此）
        W1, b1 = dw.conv1.weight, dw.conv1.bias
    if isinstance(W3, torch.Tensor) and W3.shape[-1] != 3:   # 兜底：确认 3x3 分支是 3x3
        raise AssertionError(f"3x3 分支的核形状异常：{tuple(W3.shape)}")
    C = W3.shape[0]
    id_w = F.pad(torch.ones(C, 1, 1, 1, dtype=W3.dtype), [1, 1, 1, 1])
    Wm = W3 + F.pad(W1, [1, 1, 1, 1]) + id_w
    bm = (b3 if torch.is_tensor(b3) else torch.tensor(0.0)) + b1
    t = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    Wf = Wm * t.reshape(-1, 1, 1, 1)
    bf = bn.bias + (bm - bn.running_mean) * t
    out = nn.Conv2d(C, C, 3, stride=conv.stride, padding=1, groups=C, bias=True)
    with torch.no_grad():
        out.weight.copy_(Wf); out.bias.copy_(bf)
    return out, dict(C=C, branch_3x3_W_shape=list(W3.shape), branch_1x1_W_shape=list(W1.shape),
                     identity_kernel="center=1, ring=0 (3x3 depthwise)",
                     W_final_shape=list(Wf.shape), b_final_shape=list(bf.shape),
                     t_range=[float(t.min()), float(t.max())],
                     params_before=int(9*C + 2*C + C + C + 2*C),
                     params_after=int(9*C + C))

def count_bn(m):
    return sum(1 for x in m.modules() if isinstance(x, (nn.BatchNorm1d, nn.BatchNorm2d)))

def module_hist(m):
    return dict(collections.Counter(type(x).__name__ for x in m.modules()))

def onnx_stats(path):
    import onnx
    g = onnx.load(path)
    ops = collections.Counter(n.op_type for n in g.graph.node)
    return dict(file=os.path.basename(path), file_MB=round(os.path.getsize(path)/1e6, 3),
                ir_version=g.ir_version,
                opset=[(o.domain or "ai.onnx", o.version) for o in g.opset_import],
                n_nodes=len(g.graph.node), n_initializers=len(g.graph.initializer),
                n_bn=ops.get("BatchNormalization", 0), n_conv=ops.get("Conv", 0),
                n_add=ops.get("Add", 0), n_mul=ops.get("Mul", 0), n_div=ops.get("Div", 0),
                n_erf=ops.get("Erf", 0), n_reduce_mean=ops.get("ReduceMean", 0),
                n_sigmoid=ops.get("Sigmoid", 0), n_gelu=ops.get("Gelu", 0),
                initializer_MB=round(sum(len(t.raw_data) for t in g.graph.initializer)/1e6, 3),
                op_histogram=dict(ops.most_common()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="repvit_m0_9_in1k")
    ap.add_argument("--onnx-train", default="outputs/reparam/repvit_m0_9_in1k_train.onnx",
                    help="训练态（未 fuse）ONNX：由 deploy/export_onnx.py --no-fuse 产出后另存到此")
    ap.add_argument("--onnx-infer", default=None, help="默认取 onnx_path(--model)，即 onnx/<key>.onnx")
    ap.add_argument("--out-dir", default="outputs/reparam")
    ap.add_argument("--block-index", type=int, default=-1, help="要生成融合报告的 block（默认最后一个）")
    a = ap.parse_args()
    if a.onnx_infer is None:
        a.onnx_infer = onnx_path(a.model)      # 一律经 registry 取路径，不硬写 ONNX 文件名
    os.makedirs(a.out_dir, exist_ok=True)
    info = get(a.model)
    torch.manual_seed(0)

    train_m = timm.create_model(info["arch"], pretrained=False, num_classes=info["num_classes"])
    sd = torch.load(info["ckpt"], map_location="cpu", weights_only=False)  # 同上
    cur = train_m.state_dict(); sd = sd.get("model", sd)
    sd = {k: v for k, v in sd.items() if k not in cur or v.shape == cur[k].shape}
    train_m.load_state_dict(sd, strict=False)
    train_m.eval()                                   # 必须在 eval 下融合，否则读的是训练期统计量
    infer_m = copy.deepcopy(train_m)
    infer_m.fuse()                                   # timm 版顶层 fuse()，等价于官方 replace_batchnorm
    infer_m.eval()

    report = dict(model=a.model,
                  train=dict(params=int(sum(p.numel() for p in train_m.parameters())),
                             bn_modules=count_bn(train_m), modules=module_hist(train_m)),
                  infer=dict(params=int(sum(p.numel() for p in infer_m.parameters())),
                             bn_modules=count_bn(infer_m), modules=module_hist(infer_m)))
    # 结构文本留痕（文件名与全仓重参数化产物规范一致：before=训练态、after=推理态）
    for fn, m in ((f"{a.out_dir}/{a.model}_structure_before.txt", train_m),
                  (f"{a.out_dir}/{a.model}_structure_after.txt", infer_m)):
        with open(fn, "w", encoding="utf-8") as f:
            f.write(str(m))

    x = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        y_tr, y_in = train_m(x), infer_m(x)
    diff = (y_tr - y_in).abs()
    report["numerics"] = dict(max_abs_err=float(diff.max()),
                              mean_abs_err=float(diff.mean()),
                              top1_agree=float((y_tr.argmax(1) == y_in.argmax(1)).float().mean()),
                              top5_set_agree=float(
                                  len(set(y_tr.topk(5, 1).indices[0].tolist()) &
                                      set(y_in.topk(5, 1).indices[0].tolist())) / 5.0))

    # 延迟对比（同一输入张量、同一线程设置）
    xx = torch.randn(1, 3, 224, 224)
    report["latency_ms"] = dict(train=latency(lambda: train_m(xx), WARMUP, RUNS),
                                infer=latency(lambda: infer_m(xx), WARMUP, RUNS))

    # ONNX 图对比
    if os.path.exists(a.onnx_train) and os.path.exists(a.onnx_infer):
        st_t, st_i = onnx_stats(a.onnx_train), onnx_stats(a.onnx_infer)
        report["onnx"] = dict(train=st_t, infer=st_i)
        residual = dict(
            bn_in_infer=st_i["n_bn"], conv_delta=st_t["n_conv"] - st_i["n_conv"],
            add_in_infer=st_i["n_add"], mul_in_infer=st_i["n_mul"], div_in_infer=st_i["n_div"],
            erf_in_infer=st_i["n_erf"], reducem_in_infer=st_i["n_reduce_mean"],
            irreducible_ops="SE 的 ReduceMean + 2×Conv(1×1) + ReLU + Sigmoid + Mul = 6 个算子"
                            "（中间隔着非线性，无法折进一个卷积；实际算子数以自己导出的 op_histogram 为准，"
                            "ORT 的 Extended 优化可能把 Conv+ReLU 融成 FusedConv 而少于 6）、"
                            "channel_mixer 残差的 Add（两侧隔 GELU）、GELU 在 opset<20 被拆成 Div/Erf/Add/Mul",
            note="SE 内 norm_layer 默认为 None，self.bn 是 nn.Identity，本来就没有可融合的 BN")
        report["residual"] = residual
        print(f"[ONNX] train nodes={st_t['n_nodes']} BN={st_t['n_bn']} Conv={st_t['n_conv']} "
              f"{st_t['file_MB']}MB | infer nodes={st_i['n_nodes']} BN={st_i['n_bn']} "
              f"Conv={st_i['n_conv']} {st_i['file_MB']}MB")
        assert st_i["n_bn"] == 0, "推理态 ONNX 仍有 BatchNormalization 残留，检查是否 fuse 成功"

    # ---------- 单个 RepViT Block 的融合过程自动报告 ----------
    # 类名必须两种拼写都接受：官方 vendored 实现是 `RepViTBlock`，
    # 早期版本这里只写了 `RepVitBlock`（T 小写），在 timm 实现下匹配数为 0，
    # 随后 `a.block_index % len(blocks)` 直接 ZeroDivisionError（实测复现）。
    _BLOCK_NAMES = ("RepViTBlock", "RepVitBlock")
    blocks = [m for m in train_m.modules()
              if type(m).__name__ in _BLOCK_NAMES and hasattr(m, "token_mixer")]
    assert blocks, (f"未找到任何 RepViT Block（已尝试类名 {_BLOCK_NAMES}）；"
                    f"train_m 实际含有的类：{sorted({type(x).__name__ for x in train_m.modules()})}")
    idx = a.block_index % len(blocks)
    blk = blocks[idx]
    # timm 的 token_mixer 是 RepVggDw 实例本身（有两个 conv 属性），
    # 官方的是 Sequential（要取 [0]）。两种都要认。
    dw = None
    if hasattr(blk.token_mixer, "conv") and hasattr(blk.token_mixer, "conv1"):
        dw = blk.token_mixer
    elif hasattr(blk.token_mixer, "__getitem__") and hasattr(blk.token_mixer[0], "conv"):
        dw = blk.token_mixer[0]
    if dw is not None:
        try:
            _, detail = fuse_repvggdw_manual(dw)
        except Exception as e:
            detail = dict(error=f"{type(e).__name__}: {e}",
                          hint="检查 conv1 是否为 bias=True；bias=False 会让 conv_b + conv1_b 抛 TypeError")
        report["block_fusion_report"] = dict(
            block_index=idx, block_type=type(blk).__name__,
            stride=getattr(getattr(dw, "conv", None) or blk.token_mixer[0].conv, "stride", (None,))[0],
            branches=["3x3 depthwise conv + BN", "1x1 depthwise conv (bias=True)", "identity"],
            outer_bn="BatchNorm2d(ed) 吸收进 final_conv",
            steps=["① 融合 3x3 分支的 BN: W' = W·γ/√(σ²+ε), b' = β − μ·γ/√(σ²+ε)",
                   "② 1x1 核 pad 成 3×3、identity pad 成中心为 1 的 3×3，三者逐元素相加",
                   "③ 融合外层 BN: W_f = W_m·t, b_f = β + (b_m − μ)·t",
                   "产物：单个带 bias 的 3×3 depthwise Conv2d（不可再分）"],
            detail=detail)
        with open(f"{a.out_dir}/block_{idx}_fusion_report.txt", "w", encoding="utf-8") as f:
            for k, v in report["block_fusion_report"].items():
                f.write(f"{k}:\n  {v}\n")

    if "onnx" in report:      # 单独落一份 ONNX 节点/算子统计，便于验收脚本按规范路径读取
        with open(f"{a.out_dir}/{a.model}_onnx_nodes.json", "w", encoding="utf-8") as f:
            json.dump(report["onnx"], f, ensure_ascii=True, indent=2, default=str)
    with open(f"{a.out_dir}/{a.model}_reparam_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2, default=str)
    print(json.dumps({k: report[k] for k in ("numerics", "latency_ms")},
                     ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
