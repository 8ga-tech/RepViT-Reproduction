# tools/eval_family.py —— 家族统一评测：同一子集 / 同一预处理 / 同一后端 / 同一线程
import argparse, os, gc, json, hashlib
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import timm
try:
    from tools.rv_common import (get, onnx_path, make_transform, latency, env_report,
                             read_list, build_pt, WARMUP, RUNS)
except ImportError:
    from rv_common import (get, onnx_path, make_transform, latency, env_report,
                       read_list, build_pt, WARMUP, RUNS)

class ListDataset(Dataset):
    def __init__(self, items, transform): self.items, self.transform = items, transform
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return self.transform(Image.open(p).convert("RGB")), y

@torch.inference_mode()
def topk_acc(model, loader, topk=(1, 5)):
    cor = {k: 0 for k in topk}; n = 0
    for x, y in loader:
        lg = model(x)
        for k in topk:
            cor[k] += (lg.topk(k, 1, True, True)[1] == y[:, None]).any(1).sum().item()
        n += y.numel()
    return {f"top{k}": 100.0 * cor[k] / n for k in topk}

def macs_g(m, dev):
    from fvcore.nn import FlopCountAnalysis
    m = m.to(dev)
    return FlopCountAnalysis(m, torch.randn(1, 3, 224, 224, device=dev)).total() / 1e9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="ImageNet 验证子集图片根目录")
    ap.add_argument("--list", required=True, help="考核方提供的 val 列表 path<TAB>label")
    ap.add_argument("--model", nargs="+", default=["repvit_m0_9_in1k", "repvit_m1_0_in1k",
                                                  "repvit_m1_1_in1k", "repvit_m1_5_in1k", "repvit_m2_3_in1k"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="outputs/benchmarks/family_summary.csv")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    items = read_list(a.data_root, a.list)
    subset_sha = hashlib.sha256(open(a.list, "rb").read()).hexdigest()[:16]
    rows = []
    for key in a.model:
        info = get(key)
        # **必须走 registry 的 build_pt()**，不能自己 timm.create_model + 手动 load。
        # 官方型号的登记 impl='official'，权重的键是 features.N.* / classifier.classifier.*；
        # 而 timm 实现的键是 stages.M.* / head.head.*。两者**交集为 0**，
        # 手动 load 时 `{k: v for k in sd if k in cur}` 会把 713 个键**全部丢掉**，
        # 模型保持随机初始化，评测出来 top1 ≈ 0.1%（= 1/1000 随机水平）。
        # 实测踩过：5 个型号全部 top1=0.1、top5=0.5，图上看「跑通了」其实完全没加载。
        # build_pt() 按 impl 分派（official 走 vendored 官方实现、timm 走 timm），
        # 并且自带 missing/unexpected 硬断言，能把这类静默错误挡在前面。
        model = build_pt(key)
        model.eval()

        # transform 从模型自身 cfg 生成，保证 crop_pct=0.95 + bicubic 一致
        tf = make_transform(model, is_training=False)
        loader = DataLoader(ListDataset(items, tf), batch_size=a.batch_size,
                            shuffle=False, num_workers=4)
        # 统一口径：先转推理态再统计参数与 MACs
        fuse = getattr(model, "fuse", None)
        if callable(fuse): model.fuse()
        n_param = sum(p.numel() for p in model.parameters()) / 1e6
        g = macs_g(model, "cpu")
        lat = latency(lambda: model(torch.randn(1, 3, 224, 224)), WARMUP, RUNS)
        acc = topk_acc(model, loader)
        onnx_p = onnx_path(key)
        rows.append(dict(
            model=key, timm_id=info["arch"], param_caliber="infer_fused",
            params_M=round(n_param, 4), macs_G=round(g, 4),
            onnx_MB=round(os.path.getsize(onnx_p) / 1e6, 3) if os.path.exists(onnx_p) else np.nan,
            ckpt_MB=(round(os.path.getsize(info["ckpt"]) / 1e6, 3)
                     if info.get("ckpt") and os.path.exists(info["ckpt"]) else np.nan),
            lat_mean_ms=round(lat["mean"], 3), lat_p50_ms=round(lat["p50"], 3),
            lat_p95_ms=round(lat["p95"], 3), lat_max_ms=round(lat["mx"], 3),
            top1=round(acc["top1"], 3), top5=round(acc["top5"], 3), n_images=len(items),
            runtime="pytorch", device="cpu", ep="CPUExecutionProvider",
            threads=a.threads, batch=1, precision="FP32", warmup=WARMUP, runs=RUNS,
            transform=f"crop_pct=0.95,bicubic,input=224 (timm resolve_data_config)",
            subset_sha256=subset_sha, param_caliber_note="timm.fuse() 后单头"))
        print({k: v for k, v in rows[-1].items() if k in
               ("model", "params_M", "macs_G", "lat_p50_ms", "top1", "top5")})
        del model; gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False, encoding="utf-8-sig")
    # json 版本便于程序读取（含完整环境信息）
    with open(a.out.replace(".csv", ".json"), "w", encoding="utf-8") as f:
        json.dump(dict(env=env_report(), subset_sha256=subset_sha,
                       protocol=dict(warmup=WARMUP, runs=RUNS, threads=a.threads,
                                     batch=1, precision="FP32", device="cpu"),
                       results=rows), f, ensure_ascii=False, indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    # 规格书 §4.1 要求每个脚本都有 __main__ 守卫。
    # 缺这一行的后果不是报错，而是**静默什么都不做**（python 只定义函数然后退出 0），
    # 极易被误判成「工具跑通了」。实测 eval_family / robustness_test / interp_deep
    # 三个脚本都缺这一行。
    main()
