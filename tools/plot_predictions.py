#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/plot_predictions.py —— 预测结果可视化薄封装（utils/plot.py 的 CLI 入口）。

Source: Self-written。绘图函数来自 utils/plot.py，数据来自
        outputs/predictions/<tag>_<split>_preds.csv（列契约见规格书 §8.2.1）。

三种模式（对应题目第 10–11 页的三条要求）：
  [1] 测试集预测：--pred-csv ... --num 8  -> 逐张 outputs/predictions/*_grid{n}_*.png + 一张拼图
      题目要「至少 8 张测试集预测，实际图片显示 Top-5 类别及置信度」。
      **逐张落盘**是必须的：验收按 `ls outputs/predictions/*.png | wc -l` 计张数，
      只写一张拼图的话文件数永远是 1。
  [2] 正/误案例：--cases-csv ... --role wrong --num 2 -> outputs/predictions/case_wrong_<tag>.png
  [3] 训练集以外的实际图片：--images a.jpg b.jpg ... --pred-csv <外部预测CSV> -> 拼图 + 逐张

用法：
    python tools/plot_predictions.py --pred-csv outputs/predictions/baseline_test_preds.csv \
        --classes labels/pet_classes.txt --num 8 --cols 4 \
        --out outputs/predictions/test_top5_baseline_grid8.png --tag baseline
    python tools/plot_predictions.py --cases-csv outputs/predictions/cases_test_baseline.csv \
        --role wrong --num 2 --out outputs/predictions/case_wrong_baseline.png --tag baseline
    python tools/plot_predictions.py --images external/*.jpg \
        --pred-csv outputs/benchmarks/external_top5_baseline.csv --num 5 \
        --out outputs/predictions/external_top5_baseline_grid5.png --tag baseline
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _abspath(p) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (ROOT / q)


def _top5_of_row(row) -> list:
    """从逐图预测 CSV 的一行还原 Top-5 (类别名, 概率) 列表。

    契约里 top5_names / top5_probs 是 **JSON 字符串**（见规格书 §8.2.1），
    所以必须 json.loads，不能按分隔符切——类别名里可能含逗号或空格。
    """
    names = row["top5_names"]
    probs = row["top5_probs"]
    names = json.loads(names) if isinstance(names, str) else list(names)
    probs = json.loads(probs) if isinstance(probs, str) else list(probs)
    return [(str(n), float(p)) for n, p in zip(names, probs)]


def _grid(records: list, out_path: Path, cols: int, title: str) -> None:
    """把若干「原图 | Top-5 条形图」记录拼成一张图。

    utils.plot.plot_top5_prediction 一次只出一张图，这里先逐张渲染到临时 PNG，
    再用 matplotlib 拼版——避免为拼版另写一套绘图代码（两套代码迟早会不一致）。
    """
    import tempfile
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from utils.plot import setup_chinese_font, plot_top5_prediction

    setup_chinese_font()
    tmpdir = Path(tempfile.mkdtemp(prefix="rvt_top5_"))
    tiles = []
    for i, rec in enumerate(records):
        tp = tmpdir / f"t{i:03d}.png"
        plot_top5_prediction(rec["image_path"], rec["top5"], str(tp), true_name=rec.get("true_name"))
        # 逐张落盘（验收按 `ls outputs/predictions/*.png | wc -l` 计张数）
        indiv = out_path.parent / f"{out_path.stem}_case{i+1:02d}.png"
        indiv.write_bytes(tp.read_bytes())
        tiles.append(Image.open(tp).convert("RGB"))

    # 各张图的高度会因标题行数不同而略有差异（长度不同的类别名会换行），
    # 直接 hstack 会 broadcast 失败。统一缩放到同一尺寸后再拼版。
    W = max(t.width for t in tiles)
    H = max(t.height for t in tiles)
    tiles = [t.resize((W, H), Image.LANCZOS) for t in tiles]

    n = len(tiles)
    rows = (n + cols - 1) // cols
    canvas = np.full((rows * H, cols * W, 3), 255, dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = np.asarray(t)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(cols * W / 100, rows * H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(canvas); ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, dpi=100); plt.close(fig)
    print(f"[plot] 已保存 {out_path.relative_to(ROOT).as_posix()}（含 {n} 张，逐张另存 {n} 个文件）")


def main() -> int:
    ap = argparse.ArgumentParser(description="预测结果可视化")
    ap.add_argument("--pred-csv", default=None, help="逐图预测 CSV（模式 1/3）")
    ap.add_argument("--cases-csv", default=None, help="案例 CSV（模式 2，含 correct 列）")
    ap.add_argument("--classes", default="labels/pet_classes.txt")
    ap.add_argument("--images", nargs="*", default=None, help="模式 3：外部图片路径")
    ap.add_argument("--role", choices=["correct", "wrong", "auto"], default="auto",
                    help="模式 2：只取预测正确/错误的样本")
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    out = _abspath(a.out)
    title = a.title or f"RepViT-M0.9 Top-5 预测（{a.tag}）"

    if a.cases_csv:
        df = pd.read_csv(_abspath(a.cases_csv))
        if a.role != "auto" and "correct" in df.columns:
            want = 1 if a.role == "correct" else 0
            df = df[df["correct"].astype(int) == want]
        # 失败案例优先取「高置信度的错」，那才是有分析价值的案例
        if a.role == "wrong" and "prob" in df.columns:
            df = df.sort_values("prob", ascending=False)
        records = [{"image_path": str(_abspath(r["path"])),
                    "true_name": r.get("true_name"),
                    "top5": _top5_of_row(r)} for _, r in df.head(a.num).iterrows()]
        if not records:
            raise SystemExit(f"{a.cases_csv} 里没有 role={a.role} 的样本")
        _grid(records, out, a.cols, title)
        return 0

    if a.images:
        paths = []
        for pat in a.images:
            paths += sorted(glob.glob(pat)) or ([pat] if Path(pat).exists() else [])
        paths = paths[:a.num]
        if not paths:
            raise SystemExit("--images 没有匹配到任何文件")
        # 外部图片的 Top-5 必须由预测 CSV 提供（同一契约列名），按文件名对齐
        t5 = {}
        if a.pred_csv:
            df = pd.read_csv(_abspath(a.pred_csv))
            for _, r in df.iterrows():
                t5[Path(str(r["path"])).name] = _top5_of_row(r)
        records = []
        for p in paths:
            name = Path(p).name
            if name not in t5:
                raise SystemExit(f"{name} 不在 {a.pred_csv} 里；外部图片必须先跑一遍推理并落盘 Top-5")
            records.append({"image_path": p, "true_name": None, "top5": t5[name]})
        _grid(records, out, a.cols, title)
        return 0

    if not a.pred_csv:
        raise SystemExit("必须给 --cases-csv 或 --images 或 --pred-csv 之一")
    df = pd.read_csv(_abspath(a.pred_csv))
    records = [{"image_path": str(_abspath(r["path"])),
                "true_name": r.get("true_name"),
                "top5": _top5_of_row(r)} for _, r in df.head(a.num).iterrows()]
    if not records:
        raise SystemExit(f"{a.pred_csv} 为空")
    _grid(records, out, a.cols, title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
