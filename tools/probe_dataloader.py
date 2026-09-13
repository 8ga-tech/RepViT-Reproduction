# tools/probe_dataloader.py
# -*- coding: utf-8 -*-
"""Phase 0 侦察：数据管线吞吐是否会成为训练瓶颈。

动机：smoke_phase0 测到 GPU 纯计算 0.115 s/step（batch=64 → 557 img/s），
比规格书 11.2 的估算（RTX 4060 Laptop 约 16~24 min / 40 epoch = 0.53~0.8 s/step）
快 5~7 倍。差距只可能来自数据管线（JPEG 解码 + RandomResizedCrop + 增强）。

本脚本用**合成的近似分辨率 JPEG**（600x450，接近 Oxford-IIIT Pet 的实际量级）
走真实 timm 预处理管线，测不同 worker 数下的 img/s，判断谁是瓶颈。

注意：合成图只用于**测吞吐**，不得用于任何精度或一致性结论（规格书禁止 randn 用于一致性测试）。

用法：
    python tools/probe_dataloader.py --workers 0 2 4 8 --n 512 --repeats 3
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]


class SyntheticPetLike(Dataset):
    """在内存里现生成 JPEG 字节再解码，逼近真实 JPEG 解码 + resize 的代价。"""

    def __init__(self, n, size=(600, 450), transform=None, seed=0):
        self.n, self.size, self.transform = n, size, transform
        import io
        import numpy as np
        from PIL import Image
        rng = np.random.RandomState(seed)
        self.blobs = []
        for i in range(n):
            arr = rng.randint(0, 255, (size[1], size[0], 3), dtype="uint8")
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=90)
            self.blobs.append(buf.getvalue())

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(self.blobs[i])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, i % 37


def build_transform():
    """用 timm 的官方预处理（crop_pct=0.95 / bicubic / 训练态增强）。"""
    from timm.data import create_transform, resolve_data_config
    import timm
    m = timm.create_model("repvit_m0_9", num_classes=1000)
    cfg = resolve_data_config({}, model=m)
    return create_transform(**cfg, is_training=True), cfg


def end_to_end_epoch(workers, n, batch, epochs, device):
    """真·端到端：DataLoader(真增强) + GPU + bf16 AMP + AdamW，测 s/epoch。
    这是决定工期的数字——分别测的 GPU 与 dataloader 吞吐会互相抢占，不能直接取小。"""
    import timm
    tf, _ = build_transform()
    ds = SyntheticPetLike(n, transform=tf)
    dl = DataLoader(ds, batch_size=batch, num_workers=workers, shuffle=True,
                    drop_last=True, persistent_workers=(workers > 0),
                    pin_memory=(device == "cuda"),
                    prefetch_factor=(4 if workers > 0 else None))
    m = timm.create_model("repvit_m0_9", num_classes=37).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
    crit = torch.nn.CrossEntropyLoss()
    steps = len(dl)
    res = {"steps_per_epoch": steps, "workers": workers, "batch": batch,
           "device": device, "epochs": epochs, "epoch_sec": [], "loss": []}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(epochs):
        m.train()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tot, cnt = 0.0, 0
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = crit(m(xb), yb)
            else:
                loss = crit(m(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * xb.size(0)
            cnt += xb.size(0)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        res["epoch_sec"].append(dt)
        res["loss"].append(tot / cnt)
        print(f"    epoch {ep+1}: {dt:6.2f} s   {cnt/dt:7.1f} img/s   loss={tot/cnt:.4f}")
    if device == "cuda":
        res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    del dl
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end-to-end", action="store_true",
                    help="跑真实的 DataLoader+GPU 合并负载，测 s/epoch")
    ap.add_argument("--e2e-workers", type=int, default=4)
    ap.add_argument("--e2e-epochs", type=int, default=3)
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--gpu-baseline", type=float, default=None,
                    help="GPU 纯计算 img/s（来自 smoke_phase0），用于判定瓶颈")
    ap.add_argument("--out", default="outputs/metrics/probe_dataloader.json")
    a = ap.parse_args()

    tf, cfg = build_transform()
    print("=" * 72)
    print(f"[cfg] input_size={cfg['input_size']} crop_pct={cfg.get('crop_pct')} "
          f"interpolation={cfg.get('interpolation')}")
    print(f"[cfg] 合成图 600x450，n={a.n}，batch={a.batch}")

    ds = SyntheticPetLike(a.n, transform=tf)
    rep = {"cfg": {k: str(v) for k, v in cfg.items()}, "n": a.n, "batch": a.batch,
           "workers": {}}

    if a.end_to_end:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print("-" * 72)
        print(f"[E2E] 端到端合并负载 device={dev} workers={a.e2e_workers} "
              f"epochs={a.e2e_epochs}")
        e2e = end_to_end_epoch(a.e2e_workers, a.n, a.batch, a.e2e_epochs, dev)
        rep["end_to_end"] = e2e
        sec = sum(e2e["epoch_sec"][1:]) / max(1, len(e2e["epoch_sec"]) - 1)  # 丢掉首轮
        steps = e2e["steps_per_epoch"]
        # 外推到 Pet 真实规模：2940 图 / bs64 / drop_last -> 45 step/epoch
        real_sec_per_step = sec / steps
        print("-" * 72)
        print(f"[E2E] 稳态 {sec:.2f} s/epoch（{steps} step）-> {real_sec_per_step*1000:.1f} ms/step")
        print(f"      外推 Pet 2940 图 / bs64 / 45 step：")
        for ep in (40, 50):
            print(f"        {ep} epoch = {real_sec_per_step*45*ep/60:.2f} min"
                  f"  ({real_sec_per_step*45*ep/3600:.3f} h)")
        rep["end_to_end_extrapolation"] = {
            "sec_per_step": real_sec_per_step,
            "min_40ep": real_sec_per_step * 45 * 40 / 60,
            "min_50ep": real_sec_per_step * 45 * 50 / 60,
        }

    print("-" * 72)
    for w in a.workers:
        dl = DataLoader(ds, batch_size=a.batch, num_workers=w, shuffle=True,
                        drop_last=True, persistent_workers=(w > 0),
                        prefetch_factor=(4 if w > 0 else None))
        # 预热 1 轮
        for _ in dl:
            break
        best = 0.0
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            nb = 0
            for _ in dl:
                nb += 1
            dt = time.perf_counter() - t0
            ips = nb * a.batch / dt
            best = max(best, ips)
        rep["workers"][w] = {"img_per_sec": best, "batches": nb}
        print(f"  workers={w:<2d}  {best:7.1f} img/s   ({nb} batch / {dt:.2f}s)")
        del dl

    if a.gpu_baseline:
        gpu = a.gpu_baseline
        best_w = max(rep["workers"], key=lambda k: rep["workers"][k]["img_per_sec"])
        best_ips = rep["workers"][best_w]["img_per_sec"]
        verdict = "GPU-bound" if best_ips >= gpu else "DATA-bound"
        rep["verdict"] = {"gpu_img_per_sec": gpu, "best_dataloader_img_per_sec": best_ips,
                          "best_workers": best_w, "conclusion": verdict,
                          "effective_img_per_sec": min(gpu, best_ips)}
        eff = min(gpu, best_ips)
        print("-" * 72)
        print(f"[判定] GPU 纯计算 {gpu:.1f} img/s vs 数据管线最佳 {best_ips:.1f} img/s "
              f"(workers={best_w})  ->  {verdict}")
        print(f"       有效吞吐 ≈ {eff:.1f} img/s  ->  45 step/epoch = {45*64/eff:.2f} s/epoch"
              f"  ->  40 epoch = {45*64/eff*40/3600:.2f} h")

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
