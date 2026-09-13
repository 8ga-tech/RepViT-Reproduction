# tools/watermark.py
"""Source: Self-written
性能图表元信息水印。所有 performance 图在 savefig 前必须调用 save_fig(fig, path, meta=...)。
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import matplotlib
matplotlib.use("Agg")            # 必须在 import pyplot 之前
import matplotlib.pyplot as plt

from tools.viz_style import setup_style     # 中文字体 + axes.unicode_minus

META_ORDER = [
    ("model", "model"), ("input_size", "in"), ("batch_size", "bs"),
    ("precision", "prec"), ("backend", "backend"), ("hardware", "hw"),
    ("ort_version", "ort"), ("warmup", "warmup"), ("runs", "runs"),
    ("file_size_mb", "size"),
]
COMPARABILITY_KEYS = ("backend", "precision", "hardware", "batch_size", "input_size")


def check_comparable(rows: list[dict], allow_mixed: bool = False) -> bool:
    """rows: 若干条 benchmark 记录。返回是否同口径；不同口径且未放行则抛错。"""
    keys = [{k: r.get(k) for k in COMPARABILITY_KEYS} for r in rows]
    same = all(k == keys[0] for k in keys)
    if not same and not allow_mixed:
        diff = [row.get("model") for row, k in zip(rows, keys) if k != keys[0]]
        raise ValueError(
            "拒绝绘制：以下记录与基准行不同口径（backend/precision/hardware/batch/input_size）"
            f"，延迟不可横向比较 -> {diff}。若确需同图，请显式传 allow_mixed=True。")
    return same


def stamp_meta(fig, meta: dict, *, mixed: bool = False, fontsize: float = 7.5):
    """把 10 项元信息渲染到图右下角（figure 坐标系，tight bbox 不会裁掉）。"""
    missing = [k for k, _ in META_ORDER if meta.get(k) in (None, "")]
    if missing:
        raise ValueError(f"性能图表元信息不完整，缺失: {missing}（题目第 13/17 页硬性要求）")
    parts = [f"{label}={meta[k]}" for k, label in META_ORDER]
    if mixed:
        parts.append("** 混合口径，禁止横向比较 **")
    text = "  |  ".join(str(p) for p in parts)
    fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=fontsize, color="#222222",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#999999",
                       alpha=0.88, linewidth=0.6))
    return fig


def save_fig(fig, path, meta: dict | None = None, *, dpi: int = 200, mixed: bool = False):
    """统一出口：有 meta 就打水印；无 meta 只允许用于非性能图（结构图/曲线图）。"""
    setup_style()
    if meta is not None:
        stamp_meta(fig, meta, mixed=mixed)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)              # 循环画 37 张图不 close 会内存泄漏并触发 RuntimeWarning
    print(f"[fig] {path}")


def example_usage():
    """示意：把 outputs/metrics/bench.jsonl 里最后一批记录画成含 P50/P95 误差棒的柱状图并打水印。"""
    import json, numpy as np
    # 同一个 model 可能有多条（重跑过），取最后一条 = 最近一次运行
    latest = {}
    for l in Path("outputs/metrics/bench.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            latest[r["model"]] = r
    recs = list(latest.values())
    mixed = not check_comparable(recs, allow_mixed=True)
    m = recs[0]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(recs))
    mean = [r["mean_ms"] for r in recs]
    lo = [r["mean_ms"] - r["p50_ms"] for r in recs]
    hi = [r["p95_ms"] - r["mean_ms"] for r in recs]
    ax.bar(x, mean, yerr=[lo, hi], capsize=4, color="#4C72B0")
    ax.set_xticks(x); ax.set_xticklabels([r["model"] for r in recs], rotation=15)
    ax.set_ylabel("延迟 (ms)"); ax.set_title("三个 ONNX 模型推理延迟")
    # 性能页水印使用「同一口径」的元信息（型号列已在 x 轴，水印里仍保留公共项）
    save_fig(fig, "figures/latency_bar_three_models.png",
             meta={**m, "model": f"{len(recs)} 个型号", "runs": f"{m['warmup']}+{m['runs']}"},
             mixed=mixed)
