# tools/manual_fusion_align.py
"""Source: Self-written
手写融合 vs timm 参考实现：同一 block、同一批固定输入上的数值对齐（拓展 4 的加分证据）。

背景（拓展任务 4「核心模块的独立实现」）
--------------------------------------
`tools/reparam_deep.py` 里已有一份**独立手写**的 RepVGGDW 三分支融合：

* `fuse_conv_bn_manual()`：Conv+BN 折叠（`t = γ/√(σ²+ε)`）
* `fuse_repvggdw_manual()`：3×3 分支 + 1×1 分支 + identity 三分支核相加，再把外层 BN 折进去

但仓库里一直没有「手写实现与参考实现在同一输入上数值一致」的产物。本脚本补上：

1. 用 registry 建 Pet-37 baseline（**已训练权重**，`checkpoints/baseline_best.pt`，不训练、不改权重）；
2. 对全部 20 个 RepViT Block 的 token mixer（RepVggDw）分别做两件事：
   - A：`tools.reparam_deep.fuse_repvggdw_manual()`（手写）
   - B：`timm` 的 `RepVggDw.fuse()`（参考实现）
3. 比较两者的 **融合后卷积核/偏置** 与 **同批固定随机输入上的输出张量**，逐 block 记录
   `max|ΔW|` / `max|Δb|` / `max|Δout|` / `mean|Δout|` / 输出形状；
4. 落盘 `outputs/advanced/manual_fusion_alignment.json`。

口径声明（**不得与 B4 / B1 并成一句结论**）
------------------------------------------
本文件的误差是「**手写融合算法 ↔ timm 融合算法**」的差，量纲是卷积权重/输出张量本身，
**不是** logits 误差，也不是结构重参数化（B4：32 个固定随机输入，7.093e-06）
或 PyTorch↔ONNX（B1：n=12 真实图片，6.199e-06）的任何一部分。

用法::

    python tools/manual_fusion_align.py
    python tools/manual_fusion_align.py --out outputs/advanced/manual_fusion_alignment.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.reparam_deep import fuse_repvggdw_manual            # noqa: E402
from deploy.model_registry import build_pt                      # noqa: E402

SEED = 20240912


def iter_mixers(model):
    """按 (stage, block) 顺序产出每个 RepViT Block 的 token mixer。"""
    for si, stage in enumerate(model.stages):
        for bi, blk in enumerate(stage.blocks):
            yield si, bi, blk.token_mixer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="repvit_m0_9_pet37", help="registry key（默认 Pet-37 baseline）")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--size", type=int, default=28, help="比较用的输入分辨率（depthwise 卷积与分辨率无关）")
    ap.add_argument("--out", default="outputs/advanced/manual_fusion_alignment.json")
    a = ap.parse_args()

    model = build_pt(a.model).eval()            # 未融合态 + 已训练权重（不训练、不改权重）
    rows, worst = [], {"max_abs_weight_diff": 0.0, "max_abs_bias_diff": 0.0, "max_abs_out_diff": 0.0}
    with torch.no_grad():
        for si, bi, mixer in iter_mixers(model):
            C = mixer.dim
            torch.manual_seed(SEED + si * 100 + bi)          # 每个 block 一批固定输入
            x = torch.randn(a.batch, C, a.size, a.size)

            manual, manual_info = fuse_repvggdw_manual(copy.deepcopy(mixer))
            ref = copy.deepcopy(mixer).fuse()                # timm 参考实现（原地改权重后返回内核）

            y_manual, y_ref = manual(x), ref(x)
            d = (y_manual - y_ref).abs()
            dw = (manual.weight - ref.weight).abs().max().item()
            db = (manual.bias - ref.bias).abs().max().item()
            rows.append(dict(
                stage=si, block=bi, channels=C, input_shape=list(x.shape),
                manual_out_shape=list(y_manual.shape), reference_out_shape=list(y_ref.shape),
                max_abs_weight_diff=dw, max_abs_bias_diff=db,
                max_abs_out_diff=float(d.max()), mean_abs_out_diff=float(d.mean()),
                manual_params_after=manual_info["params_after"],
                manual_params_before=manual_info["params_before"],
                reference_backend="timm.models.repvit.RepVggDw.fuse()",
                manual_backend="tools.reparam_deep.fuse_repvggdw_manual()"))
            worst["max_abs_weight_diff"] = max(worst["max_abs_weight_diff"], dw)
            worst["max_abs_bias_diff"] = max(worst["max_abs_bias_diff"], db)
            worst["max_abs_out_diff"] = max(worst["max_abs_out_diff"], float(d.max()))
            print(f"[block] stage{si}.blocks[{bi}] C={C:<4} dW={dw:.3e} db={db:.3e} dOut={float(d.max()):.3e}")

    import timm
    rep = dict(
        experiment="manual_fusion_alignment",
        purpose="拓展 4：手写 Conv+BN / RepVGGDW 三分支融合与 timm 参考实现在同一输入上的数值对齐",
        model=a.model, weights="checkpoints/baseline_best.pt", trained_state="eval() 未融合态",
        seed=SEED, batch=a.batch, input_hw=[a.size, a.size], n_blocks=len(rows),
        timm_version=getattr(timm, "__version__", "unknown"), torch_version=torch.__version__,
        worst=worst,
        pass_threshold=1e-5, passed=bool(worst["max_abs_out_diff"] < 1e-5 and
                                         worst["max_abs_weight_diff"] < 1e-7),
        caliber=("本文件比较的是「手写融合算法 ↔ timm 融合算法」的卷积核与输出张量差；"
                 "不是 logits 误差。结构重参数化 B4（32 个固定随机输入，max|Δlogits| = 7.093e-06）"
                 "与 PyTorch↔ONNX B1（n=12 真实图片，max|Δlogits| = 6.199e-06）是另外两批实验，"
                 "三者在任何地方都不得并成一句结论。"),
        blocks=rows)
    dst = ROOT / a.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(rep, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"\n[ok] {len(rows)} 个 block：max|ΔW|={worst['max_abs_weight_diff']:.3e} "
          f"max|Δb|={worst['max_abs_bias_diff']:.3e} max|ΔOut|={worst['max_abs_out_diff']:.3e} "
          f"passed={rep['passed']} -> {a.out}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
