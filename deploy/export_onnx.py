# -*- coding: utf-8 -*-
"""deploy/export_onnx.py —— 导出 224x224 FP32 ONNX 并完成三道体检。"""
import argparse, json, shutil, sys, tempfile, warnings
from collections import Counter
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, onnx, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.model_registry import get, build_pt, ONNX_DIR   # noqa

def fuse(model: torch.nn.Module, reg: dict, fused: bool = True) -> torch.nn.Module:
    """顺序不可交换：build_pt() 已 load_state_dict -> eval()，这里 fuse -> 再 eval()。train 态融合必错。"""
    if not fused:
        print("[fuse] 跳过重参数化，导出训练态（多分支）结构（非交付物）")
        return model.eval()
    if reg["impl"] == "official":
        from models import repvit_official as official_repvit
        official_repvit.replace_batchnorm(model)    # vendored 官方实现，递归 fuse()、裸 BN 换 Identity
    else:
        model.fuse()                                # timm 自带入口
    model.eval()
    n_bn = sum(isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
               for m in model.modules())
    print(f"[fuse] 残留 BN 模块 = {n_bn}（应为 0；未融合态 distillation=0 为 107、=1 为 108）")
    return model

def _target(out: Path):
    """本项目根目录含中文，torch.onnx.export 的 protobuf 写盘在非 ASCII 路径上可能失败：
    先写到纯 ASCII 临时文件，导出后再 move 回去。"""
    try:
        str(out).encode("ascii"); return out, None
    except UnicodeEncodeError:
        return Path(tempfile.mkdtemp(prefix="rvonnx_")) / "model.onnx", out

def inspect(path: Path, reg: dict, x: np.ndarray, ref: np.ndarray) -> dict:
    onnx.checker.check_model(str(path), full_check=True)          # full_check 连带跑 shape inference
    md = onnx.load(str(path))
    n_vi = len(md.graph.value_info)
    inf = onnx.shape_inference.infer_shapes(md, strict_mode=False, data_prop=False)
    cnt = Counter(n.op_type for n in md.graph.node)
    n_elem = 0
    for t in md.graph.initializer:
        n = 1
        for d in t.dims: n *= d
        n_elem += n
    ext = any(t.data_location == onnx.TensorProto.EXTERNAL for t in md.graph.initializer)
    import onnxruntime as ort
    sess = ort.InferenceSession(path.read_bytes(), providers=["CPUExecutionProvider"])
    y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    return dict(
        model=reg["name"], onnx=str(path), file_size_mb=round(path.stat().st_size / 1e6, 2),
        opset=[[p.domain or "ai.onnx", p.version] for p in md.opset_import],
        ir_version=md.ir_version, producer=f"{md.producer_name} {md.producer_version}",
        nodes=len(md.graph.node), node_hist=dict(cnt.most_common()),
        BatchNormalization=cnt.get("BatchNormalization", 0), Conv=cnt.get("Conv", 0),
        Add=cnt.get("Add", 0), Gemm=cnt.get("Gemm", 0), Erf=cnt.get("Erf", 0),
        Gelu=cnt.get("Gelu", 0), GlobalAveragePool=cnt.get("GlobalAveragePool", 0),
        value_info_before=n_vi, value_info_after=len(inf.graph.value_info),
        initializer_elems=n_elem, external_data=ext, num_classes=reg["num_classes"],
        input_name=sess.get_inputs()[0].name, input_shape=list(sess.get_inputs()[0].shape),
        output_name=sess.get_outputs()[0].name, output_shape=list(sess.get_outputs()[0].shape),
        ep=sess.get_providers(),
        torch_vs_onnx_max_abs=float(np.abs(ref - y).max()),
        torch_vs_onnx_top1_same=bool(ref.argmax(1)[0] == y.argmax(1)[0]))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="extend", nargs="+", required=True,
                    help="model_registry 的 key，可多个：--model a --model b 或 --model a b")
    ap.add_argument("--out-dir", default=str(ONNX_DIR), help="默认 onnx/；文件名固定为 <key>.onnx")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--dynamic-batch", type=int, default=0)
    ap.add_argument("--no-fuse", action="store_true",
                    help="导出训练态对照（非交付物，建议配 --out-dir outputs/reparam）")
    a = ap.parse_args()
    for key in a.model:
        reg = get(key)
        model = fuse(build_pt(key), reg, fused=not a.no_fuse)
        s = reg["input_size"]
        x = torch.randn(1, 3, s, s)
        with torch.no_grad():
            ref = model(x).numpy()
        out = Path(a.out_dir) / reg["onnx"]
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp, final = _target(out)
        dyn = {"input": {0: "batch"}, "logits": {0: "batch"}} if a.dynamic_batch else None
        with torch.no_grad():
            torch.onnx.export(
                model, (x,), str(tmp), input_names=["input"], output_names=["logits"],
                opset_version=a.opset,
                dynamo=False,              # torch>=2.9 默认 True，必须显式关闭走 TorchScript
                do_constant_folding=True,  # 折 Conv+BN、消 Identity；排查精度时可传 False
                export_params=True, keep_initializers_as_inputs=False, dynamic_axes=dyn)
        if final: shutil.move(str(tmp), str(final))
        rep = inspect(out, reg, x.numpy(), ref)
        dst = ROOT / "outputs/benchmarks" / f"export_{reg['name']}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(rep, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"[export] {rep['onnx']}  {rep['file_size_mb']}MB  nodes={rep['nodes']}  "
              f"BN={rep['BatchNormalization']} Conv={rep['Conv']} Gemm={rep['Gemm']}  "
              f"|Δlogits|={rep['torch_vs_onnx_max_abs']:.3e}  -> {dst}")

if __name__ == "__main__":
    main()
