# tools/external_analysis.py
"""数据集内 vs 外部实拍图的分布差异量化。
用法：python tools/external_analysis.py --dataset-root data/oxford-iiit-pet/images \
        --external-dir external --out-dir outputs/predictions
"""
import argparse, json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageOps
import pandas as pd


def image_stats(path: Path) -> dict:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    small = np.asarray(img.resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
    gray = cv2.cvtColor((small * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # 背景复杂度代理：边缘区域（外框 20%）的 Laplacian 方差 ÷ 中心区域
    k = 45
    border = np.concatenate([gray[:k, :].ravel(), gray[-k:, :].ravel(),
                             gray[:, :k].ravel(), gray[:, -k:].ravel()])
    center = gray[k:-k, k:-k]
    lap_b = float(cv2.Laplacian(border.reshape(-1, 1), cv2.CV_32F).var())
    lap_c = float(cv2.Laplacian(center, cv2.CV_32F).var())
    hsv = cv2.cvtColor((small * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    return {
        "file": str(path).replace("\\", "/"),
        "width": w, "height": h, "aspect": w / h,
        "short_side": min(w, h),
        "brightness_mean": float(gray.mean()) / 255.0,
        "brightness_std": float(gray.std()) / 255.0,
        "saturation_mean": float(hsv[..., 1].mean()) / 255.0,
        "edge_density": float((cv2.Canny(gray, 80, 160) > 0).mean()),
        "bg_complexity_ratio": lap_b / (lap_c + 1e-6),
    }


def collect(paths, tag: str) -> pd.DataFrame:
    rows = []
    for p in paths:
        try:
            r = image_stats(p); r["group"] = tag; rows.append(r)
        except Exception as e:
            print(f"[skip] {p}: {e}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["width", "height", "aspect", "short_side", "brightness_mean",
            "brightness_std", "saturation_mean", "edge_density", "bg_complexity_ratio"]
    return df.groupby("group")[cols].agg(["mean", "std", "median"]).round(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="data/oxford-iiit-pet/images")
    ap.add_argument("--external-dir", default="external")
    ap.add_argument("--samples", type=int, default=400, help="数据集侧抽样张数")
    ap.add_argument("--out-dir", default="outputs/predictions")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    ds_files = sorted([p for p in Path(a.dataset_root).rglob("*.jpg")])
    assert ds_files, f"未找到数据集图片: {a.dataset_root}"
    sel = rng.choice(len(ds_files), size=min(a.samples, len(ds_files)), replace=False)
    ex_files = sorted([p for p in Path(a.external_dir).iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
    assert ex_files, f"未找到外部图片: {a.external_dir}"

    df = pd.concat([collect([ds_files[i] for i in sel], "dataset"),
                    collect(ex_files, "external")], ignore_index=True)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "distribution_compare_raw.csv", index=False)
    s = summarize(df)
    s.to_csv(out / "distribution_compare_summary.csv")
    print(s.to_string())
    print("\n[解读提示] 外部实拍图通常 short_side 更大、edge_density 与 "
          "bg_complexity_ratio 更高、brightness_std 更大；"
          "若外部图 Top-1 明显低于 test 集，报告里应把这些差异与错误案例一一对应起来。")


if __name__ == "__main__":
    main()
