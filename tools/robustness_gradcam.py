# tools/robustness_gradcam.py
"""Source: Self-written（复用 tools/gradcam.py 的手写 Grad-CAM 与 tools/robustness_test.py 的扰动定义）

进阶任务 5「鲁棒性分析」的 Grad-CAM 侧证据
------------------------------------------
`outputs/advanced/robustness_repvit_m0_9_pet37.json` 只有数值指标（Top-1 / Macro-F1 /
mean_conf / ECE），没有「扰动下模型关注区域怎么变」的视觉证据。本脚本补这一块：

* 取 1 张 Pet-37 测试集图片（`datasets/lists/pet_test.txt` 首行），
* 对 **3 类扰动 × 2 级 severity**（+ 干净原图对照）各出一张 Grad-CAM 叠加图，
* 拼成 `outputs/advanced/robustness_gradcam.png`，逐格元信息（预测类别/置信度/是否判对/
  CAM 非零占比）落盘 `outputs/advanced/robustness_gradcam.json`。

边界
----
* **只读**：不训练、不动 `checkpoints/`、不覆盖 `robustness_repvit_m0_9_pet37.json`
  （7 类扰动 × 6 级 severity 的权威数值仍然只由 `tools/robustness_test.py` 产出）。
* 扰动函数与 severity 取值**直接 import** `tools/robustness_test.py` 的定义，不另立一套。
* 挂载层用 `tools/gradcam.py` 的 A 档（`stages[-1].blocks[-1]`，残差相加后的激活）。
* 结论只覆盖这 1 张图片 × 这几个扰动档位，不能推广成「模型对扰动鲁棒/不鲁棒」的定量结论。

用法::

    python tools/robustness_gradcam.py
    python tools/robustness_gradcam.py --limit-image 3 --dpi 130
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image                                     # noqa: E402
from tools.gradcam import (preprocess, show_cam_on_image, grad_cam,   # noqa: E402
                           pick_target_layer, _load_pet_model)
from tools import robustness_test as RT                   # noqa: E402

CKPT = "checkpoints/baseline_best.pt"

# 每类扰动取 2 级 severity：取该扰动 severity 列表的中段与较强段（列表定义见 robustness_test.py）
CASES = [
    ("gaussian_blur", [len(RT.BLUR_SIGMA) // 2, -2]),
    ("brightness", [len(RT.BRIGHT) // 2, -1]),
    ("gaussian_noise", [len(RT.NOISE_SIGMA) // 2, -1]),
]


def pet_test_images(limit: int) -> list[Path]:
    rows = [l for l in (ROOT / "datasets/lists/pet_test.txt").read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]
    out, labels = [], []
    for r in rows[:limit]:
        img_id = r.rsplit(None, 1)[0]
        p = ROOT / "data/oxford-iiit-pet/images" / f"{img_id}.jpg"
        if p.exists():
            out.append(p)
            labels.append(int(r.rsplit(None, 1)[1]))
    return out, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-index", type=int, default=0, help="取 pet_test.txt 的第 N 行（0-based）")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--out-png", default="outputs/advanced/robustness_gradcam.png")
    ap.add_argument("--out-json", default="outputs/advanced/robustness_gradcam.json")
    a = ap.parse_args()

    paths, labels = pet_test_images(max(a.image_index + 1, 1))
    src, true_idx = paths[a.image_index], labels[a.image_index]
    model = _load_pet_model(str(ROOT / CKPT), "repvit_m0_9", 37)
    layer, layer_name = pick_target_layer(model, "A")
    names = [l.split("\t")[-1].split(",")[0].strip()
             for l in (ROOT / "labels/pet_classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.viz_style import ensure_style          # 中文字体（否则标题变方框）
    ensure_style()

    base = Image.open(src).convert("RGB")
    cols = 1 + 2                                   # 干净 + 2 级 severity
    fig, axes = plt.subplots(len(CASES), cols, figsize=(3.1 * cols, 3.4 * len(CASES)), dpi=a.dpi)
    rows_out = []
    for r, (name, levels) in enumerate(CASES):
        fn, grid = RT.PERT[name]
        for c, li in enumerate([None, levels[0], levels[1]]):
            if li is None:
                img, sev, tag = base, 0, "clean"
            else:
                sev = grid[li]
                img = fn(base, sev, seed=0) if name == "gaussian_noise" else fn(base, sev)
                tag = f"{name}={sev}"
            x, arr01 = preprocess(img, 224, 0.875)
            cam, cls, logits = grad_cam(model, layer, x, class_idx=None)
            prob = float(torch.softmax(logits[0], 0)[cls])
            overlay = show_cam_on_image(arr01, cam, image_weight=0.5)
            ax = axes[r][c] if len(CASES) > 1 else axes[c]
            ax.imshow(overlay)
            ax.set_title(f"true: {names[true_idx]}\npred: {names[cls]} ({prob*100:.1f}%)"
                         + ("  判对" if cls == true_idx else "  判错"), fontsize=9)
            ax.set_xlabel(tag, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            rows_out.append(dict(perturbation=name, severity=sev, tag=tag,
                                 pred_idx=int(cls), pred_name=names[cls], pred_prob=prob,
                                 true_idx=int(true_idx), true_name=names[true_idx],
                                 correct=bool(cls == true_idx),
                                 cam_nonzero_ratio=float((cam > 0.05 * max(cam.max(), 1e-8)).mean()),
                                 cam_max=float(cam.max())))
    fig.suptitle(f"Pet-37 baseline 的 Grad-CAM 随扰动变化（1 张图 · 挂载 {layer_name} · 不训练、不改权威鲁棒性数值）",
                 fontsize=11)
    fig.tight_layout()
    dst_png = ROOT / a.out_png
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst_png, bbox_inches="tight")
    plt.close(fig)

    rep = dict(experiment="robustness_gradcam",
               purpose="进阶 5 鲁棒性分析：扰动下 Grad-CAM 关注区域的变化（视觉证据）",
               source_image=src.relative_to(ROOT).as_posix(), true_label=names[true_idx],
               ckpt=CKPT, model="repvit_m0_9 (timm, num_classes=37, distillation=False)",
               mount_layer=layer_name, crop_pct=0.875, image_size=224,
               perturbations=[c[0] for c in CASES], n_cells=len(rows_out),
               cells=rows_out,
               readable_by="tools/robustness_gradcam.py（扰动函数与 severity 直接 import tools/robustness_test.py）",
               scope_note=("只有 1 张图片 × 3 类扰动 × 2 级 severity，属定性观察；"
                           "7 类扰动 × 6 级 severity 的定量结论仍以 "
                           "outputs/advanced/robustness_repvit_m0_9_pet37.json 为准，本文件不覆盖它。"),
               png=a.out_png)
    dst_json = ROOT / a.out_json
    dst_json.write_text(json.dumps(rep, ensure_ascii=True, indent=2), encoding="utf-8")
    n_ok = sum(1 for c in rows_out if c["correct"])
    print(f"[ok] {len(rows_out)} 格（判对 {n_ok}/{len(rows_out)}）-> {a.out_png} / {a.out_json}")
    for c in rows_out:
        print(f"  {c['tag']:<24} pred={c['pred_name']:<22} p={c['pred_prob']:.3f} "
              f"correct={c['correct']} cam_nonzero={c['cam_nonzero_ratio']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
