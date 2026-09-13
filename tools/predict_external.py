#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/predict_external.py —— 对「训练集以外的实际图片」批量推理并落盘 Top-5。

Source: Self-written。推理用 deploy/infer_onnx.py 的独立实现（PIL 手写预处理 +
softmax + Top-K，不依赖 torchvision），标签文件按 deploy/model_registry.py 的
labels_for(key) 选择——保证 37 类模型绝不会误用 ImageNet 的 1000 类标签。

产物（列名与 outputs/predictions/<tag>_<split>_preds.csv 契约逐列对齐，见规格书 §8.2.1）：
    outputs/benchmarks/external_top5_<key>.csv
供 tools/plot_predictions.py --images ... --pred-csv <该文件> 出图。

用法：
    python tools/predict_external.py --model repvit_m0_9_pet37 --dir external --num 8
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="外部图片批量推理（Top-5）")
    ap.add_argument("--model", default="repvit_m0_9_pet37", help="model_registry 的 key")
    ap.add_argument("--dir", default="external")
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from deploy.model_registry import get, labels_for, onnx_path
    from deploy.infer_onnx import preprocess, softmax, build_session

    reg = get(a.model)
    labels = labels_for(a.model)
    onnx = Path(onnx_path(a.model))
    assert onnx.exists(), f"ONNX 不存在：{onnx}；先跑 deploy/export_onnx.py --model {a.model}"

    # 只用 manifest 里登记过的图片（保证来源与许可可追溯）
    ext = ROOT / a.dir
    man = ext / "images_manifest.csv"
    assert man.exists(), f"缺少 {man}；先跑 tools/fetch_external_images.py"
    with open(man, encoding="utf-8") as f:
        files = [r["filename"] for r in csv.DictReader(f)][:a.num]
    paths = [ext / n for n in files]
    for p in paths:
        assert p.exists(), f"manifest 列出的图片不存在：{p}"

    sess = build_session(str(onnx), threads=a.threads)
    iname = sess.get_inputs()[0].name
    print(f"[onnx] {onnx.name}  EP={sess.get_providers()}  n={len(paths)}  num_classes={reg['num_classes']}")
    assert len(labels) == reg["num_classes"], f"标签行数 {len(labels)} != {reg['num_classes']}"

    rows = []
    for p in paths:
        x = preprocess(str(p), reg["input_size"], reg["mean"], reg["std"], reg["crop_pct"])
        logits = sess.run(None, {iname: x})[0][0]
        probs = softmax(logits)
        top = np.argsort(-probs)[:5]
        pred = int(top[0])
        rows.append({
            "image_id": p.stem,
            "path": p.relative_to(ROOT).as_posix(),
            "true_idx": -1,                    # 外部图片无 GT（或与 Pet 标注体系不同源）
            "true_name": "",
            "pred_idx": pred,
            "pred_name": labels[pred],
            "prob": round(float(probs[pred]), 6),
            "correct": 0,
            "top5_idx": json.dumps([int(i) for i in top]),
            "top5_names": json.dumps([labels[int(i)] for i in top], ensure_ascii=True),
            "top5_probs": json.dumps([round(float(probs[int(i)]), 6) for i in top]),
        })
        print(f"  {p.name:44s} -> {labels[pred]:24s} {probs[pred]*100:6.2f}%")

    out = Path(a.out) if a.out else ROOT / "outputs/benchmarks" / f"external_top5_{a.model}.csv"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[csv] -> {out.relative_to(ROOT).as_posix()}  n={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
