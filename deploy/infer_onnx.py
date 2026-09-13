# -*- coding: utf-8 -*-
"""deploy/infer_onnx.py —— 不依赖 torchvision 的独立预处理 + ORT 推理 + Softmax/Top-K。"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.model_registry import get, labels_for, onnx_path   # noqa

def preprocess(path, size, mean, std, crop_pct):
    """插值固定 bicubic（timm _cfg 口径），不开放给各脚本自选。"""
    img = Image.open(path).convert("RGB")            # 必须是 RGB，BGR 会让精度塌到随机
    w, h = img.size
    edge = int(round(size / crop_pct))               # Resize 短边，保持长宽比
    sc = edge / min(w, h)
    img = img.resize((int(round(w * sc)), int(round(h * sc))), Image.BICUBIC)
    l, t = (img.size[0] - size) // 2, (img.size[1] - size) // 2
    img = img.crop((l, t, l + size, t + size))       # 顺序：Resize -> CenterCrop，反了差 236/255
    x = np.asarray(img, dtype=np.float32) / 255.0    # 先 /255
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))[None]   # HWC -> NCHW

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)          # 数值稳定：不减去 max 会 exp 溢出成 inf
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)       # axis 必须是类别维，写成 axis=0 全是 1

def topk(p, k=5):
    return [(int(i), float(p[i])) for i in np.argsort(-p)[:k]]

def build_session(onnx_path, threads=None):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        so.intra_op_num_threads = int(threads)       # 0=物理核数+亲和；显式设置则不绑亲和
        so.inter_op_num_threads = 1
    # 用 read_bytes 规避 Windows 中文路径导致 ORT 加载失败
    return ort.InferenceSession(Path(onnx_path).read_bytes(), sess_options=so,
                                providers=["CPUExecutionProvider"])

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--dump-probs", action="store_true",
                    help="把完整 softmax 向量（长度 = num_classes）落盘")
    ap.add_argument("--save-json", default=None, help="Top-K 结果的落盘路径")
    a = ap.parse_args()
    reg = get(reg_key := a.model)
    en = labels_for(reg_key)
    sess = build_session(onnx_path(reg_key), a.threads)
    print("EP      :", sess.get_providers())
    for i in sess.get_inputs():
        print(f"input   : name={i.name} shape={i.shape} type={i.type}")
    for o in sess.get_outputs():
        print(f"output  : name={o.name} shape={o.shape} type={o.type}")
    x = preprocess(a.image, reg["input_size"], reg["mean"], reg["std"], reg["crop_pct"])
    t0 = time.perf_counter()
    z = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    dt = (time.perf_counter() - t0) * 1000
    prob = softmax(z)
    res = topk(prob, a.topk)
    print(f"\nimage={a.image}  onnx={Path(onnx_path(reg_key)).name}  infer={dt:.2f}ms")
    for r, (i, p) in enumerate(res, 1):
        print(f"  Top{r}: {en[i]:<46} {p * 100:6.2f}%")
    if a.dump_probs:                                  # 题目「输出各类别置信度」条的落盘证据
        assert len(prob) == len(en) == reg["num_classes"], \
            f"softmax 长度 {len(prob)} 与标签行数 {len(en)} 不一致"
        dst = ROOT / "outputs/predictions" / f"{reg_key}_{Path(a.image).stem}_probs.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(dict(model=reg_key, image=str(a.image),
                                       probs=[round(float(v), 6) for v in prob]),
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        print("[probs]", dst, f"({len(prob)} 类)")
    if a.save_json:
        dst = Path(a.save_json)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(dict(
            model=reg_key, is_pet=(reg["num_classes"] == 37), image=str(a.image),
            num_classes=reg["num_classes"], ep=sess.get_providers(),
            latency_ms=round(dt, 3),
            topk=[dict(rank=r, index=i, prob=p, name=en[i])
                  for r, (i, p) in enumerate(res, 1)]), ensure_ascii=False, indent=2),
            encoding="utf-8")
        print("[json]", dst)

if __name__ == "__main__":
    main()
