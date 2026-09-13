#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/plot_curves.py —— 训练曲线薄封装（utils/plot.py 的 CLI 入口）。

Source: Self-written。绘图函数来自 utils/plot.py（本项目自撰），数据来自 outputs/logs/*.csv。

题目第 10 页要求五条曲线：train loss / validation loss / validation Top-1 /
validation Macro-F1 / learning rate，且 Baseline 与优化模型**同图或同坐标范围**。
utils.plot.plot_curves 已经实现了「先求所有 run 的并集 y 范围再画」，
所以只要把两条 CSV 一起传进来，坐标范围天然一致，不存在「分别画再对比」的偏差。

用法：
    # 单实验
    python tools/plot_curves.py --runs baseline=outputs/logs/baseline_metrics.csv \
        --out outputs/curves/baseline_curves.png
    # 两实验同图（题目要求的对比图）
    python tools/plot_curves.py \
        --runs baseline=outputs/logs/baseline_metrics.csv \
               opt_combo=outputs/logs/opt_combo_metrics.csv \
        --out outputs/curves/opt_compare.png --title "Baseline vs 优化（方案A 组合式）"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 题目要求的五条曲线： (列名, y 轴标签, 是否百分比)
ALL_KEYS = [
    ("train_loss",   "训练 Loss",        False),
    ("val_loss",     "验证 Loss",        False),
    ("val_top1",     "验证 Top-1 (%)",   True),
    ("val_macro_f1", "验证 Macro-F1 (%)", True),
    ("lr",           "学习率",           False),
]
PRESETS = {
    "all": ALL_KEYS,
    "loss": [k for k in ALL_KEYS if k[0].endswith("loss")],
    "acc": [k for k in ALL_KEYS if k[0] in ("val_top1", "val_macro_f1")],
    "lr": [k for k in ALL_KEYS if k[0] == "lr"],
}


def parse_runs(pairs: list[str]) -> dict:
    """--runs name=path [name=path ...] -> {name: path}。"""
    out = {}
    for s in pairs:
        if "=" not in s:
            raise SystemExit(f"--runs 每项应为 <实验名>=<csv路径>，收到 {s!r}")
        name, path = s.split("=", 1)
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise SystemExit(f"找不到指标 CSV: {p}")
        out[name] = str(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="RepViT 训练曲线（五条，可多实验同图）")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="<实验名>=<outputs/logs/*_metrics.csv>，可给多组实现同图对比")
    ap.add_argument("--keys", nargs="+", default=["all"],
                    help=f"取值 {'/'.join(PRESETS)}，或直接写列名；默认 all（题目要求的五条）")
    ap.add_argument("--out", required=True, help="输出 PNG 路径（相对仓库根或绝对）")
    ap.add_argument("--title", default="RepViT-M0.9 训练曲线")
    a = ap.parse_args()

    runs = parse_runs(a.runs)

    # keys 解析：预置名 -> 展开；否则按列名匹配
    keys = []
    for k in a.keys:
        if k in PRESETS:
            keys += [t for t in PRESETS[k] if t not in keys]
        else:
            hit = [t for t in ALL_KEYS if t[0] == k]
            if not hit:
                raise SystemExit(f"未知的 --keys 取值 {k!r}；可用 {list(PRESETS)} 或列名 "
                                 f"{[t[0] for t in ALL_KEYS]}")
            keys += [t for t in hit if t not in keys]
    if not keys:
        raise SystemExit("keys 解析后为空")

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out

    from utils.plot import plot_curves
    print(f"[curves] runs={list(runs)}")
    print(f"[curves] keys={[t[0] for t in keys]}")
    plot_curves(runs, keys, str(out), title=a.title)
    print(f"[curves] -> {out.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
