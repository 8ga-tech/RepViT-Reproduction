# utils/plot.py
"""
Source  : Self-written
Third-party: matplotlib (PSF-based), seaborn (BSD-3), numpy, pandas, Pillow
AI-assisted: 函数骨架由 AI 起草，字体探测与验收断言由本人补写并逐行核对
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # 必须在 import pyplot 之前，否则无显示设备时报错
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

_CJK_CANDIDATES = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
                   "Source Han Sans SC", "WenQuanYi Zen Hei"]
_font_ready = False


def setup_chinese_font(verbose: bool = True) -> list:
    """全局设置中文字体与负号。所有绘图脚本的第一行调用。

    返回实际命中的字体名列表，空列表表示机器上没有中文字体（此时中文是方框）。
    """
    global _font_ready
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [n for n in _CJK_CANDIDATES if n in installed]
    if not chosen:
        print("[warn] 未找到任何中文字体，中文将显示为方框。"
              "Windows 自带 'Microsoft YaHei' 与 'SimHei'；"
              "Linux 请 apt install fonts-noto-cjk。")
    # 关键：中文字体必须排在 DejaVu Sans 之前，否则 DejaVu 永远命中
    plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False        # 否则负号变方框
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 160
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    _font_ready = True
    if verbose:
        print(f"[font] 中文字体链: {plt.rcParams['font.sans-serif'][:3]}")
    return chosen


def _ensure() -> None:
    if not _font_ready:
        setup_chinese_font(verbose=False)


# --------------------------------------------------------------------- 曲线
def plot_curves(runs: dict, keys: list, out_path: str,
                title: str = "RepViT-M0.9 训练曲线", figsize_per_col: float = 5.0):
    """训练曲线。Baseline 与优化模型用同一套坐标范围叠画，避免「美化」嫌疑。

    runs : {实验名: metrics.csv 路径}，例如
           {"B0_baseline": "outputs/logs/B0_baseline_metrics.csv",
            "O1_randaug":  "outputs/logs/O1_randaug_metrics.csv"}
    keys : [(列名, y 轴标签, 是否百分比), ...]
           题目要求的五条曲线对应：
             ("train_loss",  "训练 Loss",   False)
             ("val_loss",    "验证 Loss",   False)
             ("val_top1",    "验证 Top-1 (%)", True)
             ("val_macro_f1","验证 Macro-F1 (%)", True)
             ("lr",          "学习率",      False)
    """
    _ensure()
    assert keys, "keys 不能为空"
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_col * n, 4.2))
    axes = np.atleast_1d(axes)

    # 先确定每个子图统一的 y 范围（题目要求：采用相同坐标范围便于比较）
    for ax, (col, ylabel, _is_pct) in zip(axes, keys):
        lo, hi = np.inf, -np.inf
        for name, csv_path in runs.items():
            df = pd.read_csv(csv_path)
            if col not in df.columns:
                print(f"[warn] {csv_path} 缺少列 {col}，跳过")
                continue
            v = df[col].to_numpy(dtype=float)
            lo, hi = min(lo, float(np.nanmin(v))), max(hi, float(np.nanmax(v)))
        if np.isfinite(lo) and np.isfinite(hi):
            pad = 0.05 * (hi - lo) if hi > lo else 0.1
            ax.set_ylim(max(0.0, lo - pad) if lo >= 0 else lo - pad, hi + pad)

        for name, csv_path in runs.items():
            df = pd.read_csv(csv_path)
            if col not in df.columns:
                continue
            ax.plot(df["epoch"], df[col], marker="o", ms=3, lw=1.5, label=name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)                       # 不 close 会内存泄漏
    print(f"[plot] 已保存 {out_path}")


# ------------------------------------------------------------- 混淆矩阵
def plot_confusion_matrix(cm: np.ndarray, class_names: list, normalize: str,
                          out_path: str, title: str = "归一化混淆矩阵",
                          figsize: tuple = (16, 14), annot_max_classes: int = 12):
    """cm 必须已经是归一化后的矩阵（由调用方保证）。

    normalize 仅用于图注，取值 'true'（行归一化，对角线=每类召回率）
    或 'pred'（列归一化，对角线=每类精确率）。图注必须写清，否则读者会把
    「某列整片很暗」误读成「该类很少出现」。
    """
    _ensure()
    import seaborn as sns
    assert cm.shape == (len(class_names), len(class_names)), \
        f"混淆矩阵维度 {cm.shape} 与类别数 {len(class_names)} 不匹配"
    label = "行归一化 (normalize='true', 对角线=召回率)" if normalize == "true" \
        else "列归一化 (normalize='pred', 对角线=精确率)"
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap="Blues", vmin=0.0, vmax=1.0,
                xticklabels=class_names, yticklabels=class_names,
                square=True, linewidths=0.2, linecolor="#dddddd",
                annot=(len(class_names) <= annot_max_classes), fmt=".2f",
                cbar_kws={"shrink": 0.7, "label": label}, ax=ax)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title(f"{title}\n({label})", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    print(f"[plot] 已保存 {out_path}")


# --------------------------------------------------------------- 每类 F1
def plot_per_class_f1(scores: np.ndarray, class_names: list, out_path: str,
                      title: str = "各类别 F1（升序）", macro_line: bool = True):
    """scores: 长度等于类别数的 F1 数组（由 f1_score(..., average=None) 得到）。"""
    _ensure()
    scores = np.asarray(scores, dtype=float)
    assert len(scores) == len(class_names), \
        f"F1 长度 {len(scores)} != 类别数 {len(class_names)}"
    order = np.argsort(scores)
    names = [class_names[i] for i in order]
    vals = scores[order]
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.34 * len(names))))
    ax.barh(names, vals, color=plt.get_cmap("RdYlGn")(vals))
    if macro_line:
        ax.axvline(scores.mean(), ls="--", c="k", lw=1.2,
                   label=f"Macro-F1 = {scores.mean():.4f}")
        ax.legend()
    ax.set_xlabel("F1"); ax.set_xlim(0, 1.02)
    ax.set_title(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    print(f"[plot] 已保存 {out_path}")


# ----------------------------------------------------------- 单图 Top-5
def plot_top5_prediction(image_path: str, top5: list, out_path: str,
                         true_name: str = None, title: str = None,
                         show: bool = False):
    """image_path: 原图路径；top5: [(类别名, 概率), ...] 概率降序，来自 softmax。

    左边原图、右边横向条形图，标题给出 GT 与预测是否正确。
    """
    _ensure()
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    names = [n for n, _ in top5]
    probs = np.array([p for _, p in top5], dtype=float)
    pred_name = names[0]
    ok = (true_name is not None) and (pred_name == true_name)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0),
                                   gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax1.imshow(img); ax1.axis("off"); ax1.grid(False)
    ax1.set_title(Path(image_path).name, fontsize=8)

    ypos = np.arange(len(names))[::-1]
    colors = ["#2ca02c" if (true_name is not None and n == true_name) else "#1f77b4"
              for n in names]
    ax2.barh(ypos, probs, color=colors)
    ax2.set_yticks(ypos); ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlim(0, 1.0); ax2.set_xlabel("softmax 置信度")
    for y, p in zip(ypos, probs):
        ax2.text(p + 0.01, y, f"{p*100:.2f}%", va="center", fontsize=8)
    head = title or (f"GT: {true_name}" if true_name else "Top-5 预测")
    mark = "" if true_name is None else ("  [正确]" if ok else "  [错误]")
    fig.suptitle(head + mark + f"\nTop-1: {pred_name} ({probs[0]*100:.2f}%)", fontsize=11)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)


# ------------------------------------------------------- 多模型同图对比
def plot_side_by_side(records: list, model_names: list, out_path: str,
                      title: str = "Baseline vs 优化模型 同图对比", dpi: int = 160):
    """同一批图片、两个模型并排。对应题目第 10 页「至少 4 张 Baseline 与优化模型的同图对比」。

    records: [{'image_path': str, 'true_name': str,
               'preds': {'B0_baseline': [(name, prob), ...5 项],
                         'O1_randaug':  [(name, prob), ...5 项]}}, ...]
    每行一张图：[原图 | 模型 A 的 Top-5 文本 | 模型 B 的 Top-5 文本]
    行首色条：绿色=两模型都对，红色=至少一个错。
    """
    _ensure()
    from PIL import Image, ImageOps
    n_models = len(model_names)
    fig, axes = plt.subplots(len(records), 1 + n_models,
                             figsize=(4.0 * (1 + n_models), 4.2 * len(records)),
                             squeeze=False)
    for i, rec in enumerate(records):
        img = ImageOps.exif_transpose(Image.open(rec["image_path"])).convert("RGB")
        ax = axes[i][0]
        ax.imshow(img); ax.axis("off"); ax.grid(False)
        ax.set_title(f"GT: {rec['true_name']}\n{Path(rec['image_path']).name}", fontsize=8)

        for j, mname in enumerate(model_names):
            ax = axes[i][j + 1]
            ax.axis("off"); ax.grid(False)
            t5 = rec["preds"][mname]
            top1_name, top1_p = t5[0]
            ok = (top1_name == rec["true_name"])
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       color="#2ca02c" if ok else "#d62728", alpha=0.12))
            ax.text(0.02, 0.94, mname, fontsize=11, weight="bold",
                    va="top", transform=ax.transAxes)
            ax.text(0.02, 0.84, "正确" if ok else "错误",
                    fontsize=10, color="#2ca02c" if ok else "#d62728",
                    va="top", transform=ax.transAxes)
            lines = [f"{r}. {nm[:24]:24s} {p*100:6.2f}%" for r, (nm, p) in enumerate(t5, 1)]
            ax.text(0.02, 0.72, "\n".join(lines), fontsize=8.5, va="top",
                    family="monospace", transform=ax.transAxes)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi); plt.close(fig)
    print(f"[plot] 已保存 {out_path}")
