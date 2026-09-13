#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/plot_side_by_side.py —— Baseline vs 优化模型「同一批图片」并排对比。

Source: Self-written。绘图函数来自 utils/plot.py。

对应题目第 10 页：「至少 4 张 Baseline 与优化模型的同图对比」。

**唯一必须做对的事：按 image_id 做 inner join，禁止按行号对齐。**
两份预测 CSV 是两次独立评价的产物，DataLoader 的 shuffle / drop_last 设置、
坏图剔除数量都可能不同，按行号对齐会得到「张冠李戴」的对比图——图看上去正常，
结论却完全错误。两份 CSV 都带 image_id 列（契约见规格书 §8.2.1），
所以 set_index("image_id").join(...) 是可靠且唯一的正确做法。

用法：
    python tools/plot_side_by_side.py \
        --pred-csv-a outputs/predictions/baseline_test_preds.csv \
        --pred-csv-b outputs/predictions/opt_combo_test_preds.csv \
        --name-a baseline --name-b opt_combo --num 4 \
        --out outputs/predictions/compare_baseline_vs_opt_combo_grid4.png
"""
from __future__ import annotations

import argparse
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


def _top5(row) -> list:
    names = row["top5_names"]; probs = row["top5_probs"]
    names = json.loads(names) if isinstance(names, str) else list(names)
    probs = json.loads(probs) if isinstance(probs, str) else list(probs)
    return [(str(n), float(p)) for n, p in zip(names, probs)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline vs 优化模型同图对比")
    ap.add_argument("--pred-csv-a", required=True)
    ap.add_argument("--pred-csv-b", required=True)
    ap.add_argument("--name-a", default="baseline")
    ap.add_argument("--name-b", default="opt")
    ap.add_argument("--num", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    da = pd.read_csv(_abspath(a.pred_csv_a)).set_index("image_id")
    db = pd.read_csv(_abspath(a.pred_csv_b)).set_index("image_id")

    # 只比较两边都评价过的图片；inner join（见文件头说明，禁止按行号对齐）
    common = da.index.intersection(db.index)
    if len(common) == 0:
        raise SystemExit("两份预测 CSV 的 image_id 没有交集，无法做同图对比")
    print(f"[join] A={len(da)} B={len(db)} common={len(common)}（inner join on image_id）")

    # 选题策略：优先挑「两个模型结论不同」的图片——那才是对比图的价值所在。
    # 若无差异，再退回按 A 的置信度降序取前 N 张。
    diff = da.loc[common, "pred_idx"].astype(int) != db.loc[common, "pred_idx"].astype(int)
    sel = common[diff]
    note = "两模型预测不同"
    if len(sel) < a.num:
        rest = common[~diff]
        order = da.loc[rest, "prob"].astype(float).sort_values(ascending=False).index
        sel = list(sel) + list(order[:a.num - len(sel)])
        note = "两模型预测不同者优先，不足则补高置信度样本"
    sel = list(sel)[:a.num]

    records = []
    n_disagree = 0
    for iid in sel:
        ra, rb = da.loc[iid], db.loc[iid]
        if int(ra["pred_idx"]) != int(rb["pred_idx"]):
            n_disagree += 1
        records.append({
            "image_id": str(iid),
            "image_path": str(_abspath(ra["path"])),
            "true_name": ra.get("true_name"),
            "preds": {a.name_a: _top5(ra), a.name_b: _top5(rb)},
        })

    out = _abspath(a.out)
    title = a.title or (f"{a.name_a} vs {a.name_b} —— 同一批图片 Top-5 对比（{note}）")
    from utils.plot import plot_side_by_side
    plot_side_by_side(records, [a.name_a, a.name_b], str(out), title=title)
    print(f"[plot] -> {out.relative_to(ROOT).as_posix()}  "
          f"样本数={len(records)}（其中两模型预测不同 {n_disagree} 张）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
