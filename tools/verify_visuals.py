# tools/verify_visuals.py  —— 返回码 0 通过，1 失败
"""产物验收：文件数量、图片尺寸、混淆矩阵行列和、Grad-CAM 值域与非零占比。

用法：python tools/verify_visuals.py --root outputs --tag B0_baseline
"""
import argparse
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image

FAIL, OK = [], []


def chk(cond, msg):
    (OK if cond else FAIL).append(msg)


def main(root="outputs", tag="B0_baseline"):
    root = Path(root)
    # --- 1) 曲线 ---
    for f in [f"curves/{tag}_curves.png", "curves/B0_vs_O1_compare.png"]:
        p = root / f
        chk(p.exists(), f"曲线 {f}: {'存在' if p.exists() else '缺失'}")
        if p.exists():
            w, h = Image.open(p).size
            chk(min(w, h) >= 600, f"曲线 {f} 尺寸 {w}x{h} (期望 min>=600)")

    # --- 2) 混淆矩阵：行和必须为 1 ---
    npy = root / f"confusion_matrix/{tag}_cm.npy"
    if npy.exists():
        cm = np.load(npy)
        rs = cm.sum(axis=1)
        chk(cm.shape == (37, 37), f"混淆矩阵维度 {cm.shape} (期望 (37,37))")
        chk(np.allclose(rs[rs > 0], 1.0, atol=1e-6),
            f"有样本行的行和均为 1 (max|rowsum-1|={np.abs(rs[rs>0]-1).max():.2e})")
        chk(np.nanmin(cm) >= 0 and np.nanmax(cm) <= 1 + 1e-9,
            f"取值域 [{np.nanmin(cm):.3f}, {np.nanmax(cm):.3f}] ⊂ [0,1]")
    else:
        chk(False, f"混淆矩阵 {npy} 缺失")

    # --- 3) 每类指标 CSV ---
    pc = root / f"confusion_matrix/{tag}_per_class.csv"
    chk(pc.exists() and sum(1 for _ in open(pc, encoding="utf-8")) == 38,
        f"每类指标 CSV 存在且 37 行数据 + 表头 (期望 38 行)")

    # --- 4) 预测图：数量与尺寸 ---
    for f, min_w in [(f"predictions/test_top5_{tag}_grid8.png", 1200),
                     ("predictions/compare_B0_vs_O1_grid4.png", 900),
                     (f"predictions/external_top5_{tag}_grid5.png", 900)]:
        p = root / f
        if p.exists():
            w, h = Image.open(p).size
            chk(w >= min_w, f"{f} 尺寸 {w}x{h} (期望 width>={min_w})")
        else:
            chk(False, f"{f} 缺失")

    # --- 5) Grad-CAM：值域 [0,1] 且非零占比 >= 0.95 ---
    for f in [f"gradcam/gradcam_correct_{tag}.png", f"gradcam/gradcam_wrong_{tag}.png",
              f"gradcam/gradcam_layer_compare_{tag}.png"]:
        p = root / f
        if not p.exists():
            chk(False, f"{f} 缺失"); continue
        w, h = Image.open(p).size
        chk(min(w, h) >= 600, f"{f} 尺寸 {w}x{h}")
    camnpy = root / f"gradcam/cam_{tag}_probe.npy"     # gradcam.py 用 --dump-cam 落盘
    if camnpy.exists():
        c = np.load(camnpy)
        chk(c.min() >= -1e-6 and abs(c.max() - 1.0) < 1e-5,
            f"Grad-CAM 值域 [{c.min():.6f}, {c.max():.6f}] ⊆ [0,1] 且 max≈1")
        ratio = float((c > 1e-6).mean())
        chk(ratio >= 0.95, f"Grad-CAM 非零占比 {ratio*100:.2f}% (期望 >=95%)")
        chk(float(np.isfinite(c).all()), "Grad-CAM 无 NaN/Inf")
    else:
        chk(False, "cam 探针文件缺失（gradcam.py 需支持 --dump-cam 保存浮点热图）")

    for m in OK:  print("[OK]  ", m)
    for m in FAIL: print("[FAIL]", m)
    print(f"\n通过 {len(OK)} 项，失败 {len(FAIL)} 项")
    return 1 if FAIL else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser("可视化产物验收")
    ap.add_argument("--root", default="outputs", help="产物根目录，相对仓库根")
    ap.add_argument("--tag", default="B0_baseline", help="实验名，与产物文件名前缀一致")
    a = ap.parse_args()
    sys.exit(main(a.root, a.tag))
