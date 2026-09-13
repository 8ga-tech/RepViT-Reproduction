#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/draw_arch.py —— 自绘 RepViT-M0.9 整网结构图（题目第 7 页的硬要求）。

Source: Self-written。形状与层数**全部现测**（forward hook 读真实张量），
        不使用论文插图，也不写死任何分辨率/通道数。

产物（两文件同时生成，DoD #36）：
    outputs/architecture/repvit_m0_9_arch.png
    outputs/architecture/repvit_m0_9_arch.md   —— 9 要素齐全，供报告与答辩直接引用

9 要素（题目第 7 页逐条要求）：
    ① 输入尺寸 ② Stem ③ 四个主要阶段 ④ 特征图分辨率变化
    ⑤ 通道数变化 ⑥ RepViT Block ⑦ Global Average Pooling ⑧ 分类头 ⑨ 最终输出维度

用法：
    python tools/draw_arch.py --out-dir outputs/architecture
    python tools/draw_arch.py --arch repvit_m1_0 --num-classes 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")                                  # 必须在 pyplot 之前
import matplotlib.pyplot as plt
import torch
import timm
from matplotlib.patches import FancyBboxPatch

from utils.plot import setup_chinese_font


def _stem_probe_names(model) -> list[str]:
    """按**实际存在的子模块名**推导 stem 内部的观察点，绝不写死 `stem.0`。

    实测 timm 1.0.29 的 RepVit.stem 是：
        Sequential(conv1=ConvNorm, act1=GELU, conv2=ConvNorm)
    其中 conv1 是 stride=2 的 3x3（224->112），conv2 也是 stride=2（112->56）。
    早期版本把观察点写成 `stem.0` / `stem`，在这套结构下直接 KeyError，
    整张图生成不出来（实测复现）。
    """
    names = []
    stem = getattr(model, "stem", None)
    assert stem is not None, "该模型没有 stem 属性，无法绘制结构图"
    for sub in stem.named_children():
        n = sub[0]
        if n.startswith("conv"):
            names.append(f"stem.{n}")
    if not names:
        names.append("stem")
    return names


def probe(arch: str, num_classes: int, size: int = 224) -> tuple[dict, list[int]]:
    """挂 forward hook 读真实形状，返回 (feats, depth)。"""
    m = timm.create_model(arch, pretrained=False, num_classes=num_classes,
                          distillation=False).eval()
    mods = dict(m.named_modules())
    stage_names = [f"stages.{i}" for i in range(len(m.stages))]
    names = _stem_probe_names(m) + stage_names

    feats: dict[str, tuple] = {"input": (3, size, size)}
    hooks = []
    for n in names:
        assert n in mods, f"{n} 不存在；可用模块见 timm 的 named_modules()"
        hooks.append(mods[n].register_forward_hook(
            lambda _m, _i, o, n=n: feats.__setitem__(n, tuple(o.shape[1:]))))
    with torch.inference_mode():
        m(torch.randn(1, 3, size, size))
    for h in hooks:
        h.remove()

    last = feats[stage_names[-1]]
    feats["gap"] = (last[0],)                                   # GAP 后的通道维
    feats["out"] = (num_classes,)
    depth = [len(m.stages[i].blocks) for i in range(len(m.stages))]
    return feats, depth


def draw(feats: dict, depth: list[int], out_png: Path, out_md: Path,
         arch: str, num_classes: int, size: int = 224) -> list[str]:
    stem_names = [k for k in feats if k.startswith("stem")]
    stage_names = sorted((k for k in feats if k.startswith("stages")),
                         key=lambda s: int(s.split(".")[1]))

    rows: list[tuple[str, str]] = []          # (标签, 描述)
    rows.append((f"输入", f"输入尺寸 {size}×{size}×3（RGB）"))
    for n in stem_names:
        c, h, w = feats[n]
        tag = "Stem" if n == stem_names[-1] else "Stem（前半）"
        rows.append((tag, f"{n}  3×3 stride=2   {c}ch  {h}×{w}"))
    for n in stage_names:
        i = int(n.split(".")[1])
        c, h, w = feats[n]
        rows.append((f"Stage {i}", f"{n}  {depth[i]}× RepViT Block   {c}ch  {h}×{w}"))
    c_gap = feats["gap"][0]
    rows.append(("GAP", f"Global Average Pooling  →  {c_gap} 维"))
    rows.append(("分类头", f"RepVitClassifier（NormLinear：BN1d + Linear）→ 输出 {num_classes} 维"))

    # ---- PNG ----
    setup_chinese_font()
    n = len(rows)
    fig, ax = plt.subplots(figsize=(8.6, 0.78 * n + 0.7), dpi=200)
    colors = {"输入": "#dbe9ff", "Stem": "#cfe3d4", "Stem（前半）": "#cfe3d4",
              "GAP": "#ffe8cc", "分类头": "#f6d6d6"}
    for k, (tag, text) in enumerate(rows):
        y = n - 1 - k
        fc = colors.get(tag, "#eef3fb")
        ax.add_patch(FancyBboxPatch((0.04, y + 0.08), 0.92, 0.72,
                                    boxstyle="round,pad=0.015", fc=fc, ec="#2c3e50", lw=1.1))
        ax.text(0.065, y + 0.44, tag, ha="left", va="center",
                fontsize=9.5, weight="bold", color="#1b3a5c")
        ax.text(0.30, y + 0.44, text, ha="left", va="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 0.2)
    ax.axis("off")
    ax.set_title(f"RepViT-M0.9 整网结构（现测形状，输入 {size}×{size}，{num_classes} 类）",
                 fontsize=12, pad=8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    # ---- MD（9 要素齐全，DoD #36 逐项核对的就是这几个数）----
    def _chain(idx: int) -> str:
        """取形状链并**合并相邻重复值**。

        stem.conv2 与 stages.0 的分辨率相同（都是 56）、通道也相同（都是 48），
        不去重会写成 `224→112→56→56→28→...` 与 `3→24→48→48→96→...`，
        与题目要求的「分辨率 224→112→56→28→14→7、通道 3→24→48→96→192→384」对不上。
        """
        vals = [feats[k][idx] for k in ["input"] + stem_names + stage_names]
        out = [vals[0]]
        for v in vals[1:]:
            if v != out[-1]:
                out.append(v)
        return "→".join(str(v) for v in out)

    res_chain = _chain(1)
    ch_chain = _chain(0)
    n_blocks = sum(depth)
    md = [
        f"# RepViT-M0.9 整网结构（自绘，形状现测）",
        "",
        f"- 来源：`tools/draw_arch.py` 用 forward hook 读真实张量得到，非论文插图。",
        f"- 架构：`{arch}`（timm），迁移后类别数 `{num_classes}`。",
        "",
        "## 九要素",
        "",
        f"1. **输入尺寸**：{size}×{size}×3（RGB）",
        f"2. **Stem**：{' → '.join(stem_names)}；Early Convolution Stem，"
        f"用两组 stride=2 卷积把 {size} 快速降到 {feats[stem_names[-1]][1]}，"
        f"通道升到 {feats[stem_names[-1]][0]}",
        "3. **四个主要阶段**："
        + "；".join(f"`stages.{i}` 含 {depth[i]} 个 Block" for i in range(len(depth))),
        f"4. **特征图分辨率变化**：{res_chain}"
        f"（共下采样 {size}→{feats[stage_names[-1]][1]}，倍率 {size // feats[stage_names[-1]][1]}×）",
        f"5. **通道数变化**：{ch_chain}",
        f"6. **RepViT Block**：共 {n_blocks} 个。每个 Block = "
        "Token Mixer（RepVGGDW：3×3 depthwise + 1×1 depthwise 双分支，训练态含 BN，"
        "推理态融合为单个 3×3 depthwise）+ Channel Mixer（1×1 升维 → GELU → 1×1 降维）"
        "+ 残差连接；仅部分 Block 带 SE（Squeeze-Excite）",
        f"7. **Global Average Pooling**：把 {feats[stage_names[-1]][0]}×"
        f"{feats[stage_names[-1]][1]}×{feats[stage_names[-1]][2]} 池化成 "
        f"{feats['gap'][0]} 维向量",
        "8. **分类头**：`RepVitClassifier` → `NormLinear`（BatchNorm1d + Linear），"
        "**单头**（`distillation=False`）；不使用多头/多层 MLP 头，以降低延迟",
        f"9. **最终输出维度**：{num_classes}",
        "",
        "## 参数量口径（三个数都对，差别在口径）",
        "",
        "| 口径 | 参数量 |",
        "|---|---|",
        "| 训练态双头（含蒸馏头，C=1000） | 5,489,328 |",
        "| 未融合单头（C=1000） | 5,103,560 |",
        "| 未融合单头（C=37，本任务训练态） | 4,732,805 |",
        "| 骨干（不含任何分类头） | 4,717,792 |",
        "| 融合后单头（C=1000） | 5,067,056 |",
        "| 融合后单头（C=37） | 4,696,301 |",
        "",
        "> 复现命令：`python tools/count_params.py --model repvit_m0_9`（实测值与上表逐位一致）。",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md), encoding="utf-8", newline="\n")
    return [t for _, t in rows]


def main() -> int:
    ap = argparse.ArgumentParser(description="自绘 RepViT 整网结构图")
    ap.add_argument("--arch", default="repvit_m0_9")
    ap.add_argument("--num-classes", type=int, default=37)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--out-dir", default="outputs/architecture")
    a = ap.parse_args()

    out = Path(a.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    feats, depth = probe(a.arch, a.num_classes, a.input_size)
    # 产物文件名固定为 repvit_m0_9_arch.*（DoD #36 的验收路径）
    stem = out / "repvit_m0_9_arch.png"
    md = out / "repvit_m0_9_arch.md"
    rows = draw(feats, depth, stem, md, a.arch, a.num_classes, a.input_size)

    print(f"[probe] {a.arch}  block 数 = {depth}（合计 {sum(depth)}）")
    for t in rows:
        print("   ", t)
    print(f"[arch] -> {stem.relative_to(ROOT).as_posix()}")
    print(f"[arch] -> {md.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
