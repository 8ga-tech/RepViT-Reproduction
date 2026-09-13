# -*- coding: utf-8 -*-
"""tools/count_params.py —— RepViT 参数量「五口径」全表对齐（对应 DoD #13）。

用法：
    python tools/count_params.py --model repvit_m0_9
    python tools/count_params.py --model repvit_m0_9 --impl official

为什么必须区分口径：网络有四种「形态」，参数量各不相同，报告里只写一个数字必然被质疑。
    ① 训练态双头（distillation=True，C=1000）：timm 的 RepVit 默认结构，含 head_dist，
       eval 时会把 head 与 head_dist 的 logits 取平均 —— 这是「预训练权重发布时的形态」。
    ② 未融合单头（distillation=False，C=1000）：推理只需一个分类头，但 BN 还在，
       仍未做重参数化 —— 这是「刚 load 完权重、还没 fuse」的形态。
    ③ 骨干（不含任何分类头）：只统计 patch_embed/stem + 所有 RepViTBlock 的参数，
       是「迁移学习可复用部分」的规模，也是和论文 Table 对照时要报的数字。
    ④ 未融合单头 C=37：本任务 Pet-37 迁移训练的建网形态（也是 train.py 的建网形态）。
    ⑤ 融合后单头（model.fuse() 之后）：BN 折进卷积、RepVGGDW 三分支合并、双头平均成单头，
       这是 ONNX 导出口径，也是部署时真正的参数量。

参数量口径（全脚本唯一，写死在这里，任何一项都不得单独改）：
    params = sum(p.numel() for p in model.parameters())
即「可学习张量 nn.Parameter 的元素数之和」。BN 的 running_mean / running_var /
num_batches_tracked 是 buffer 而不是 parameter，一律不计入 —— 融合后它们会全部消失，
若把 buffer 计进来，「净减」就不是一个干净的整数，跨实现也没法对齐。

实测基准（本机 timm 1.0.29 + torch 2.14.0，repvit_m0_9）：
    训练态双头 C=1000 = 5489328        未融合单头 C=1000 = 5103560
    骨干（无头）      = 4717792        未融合单头 C=37   = 4732805
    融合后 C=1000     = 5067056        融合后 C=37       = 4696301    净减 = 36504
若本机实测与上述整数有差异，以实测为准，脚本会把 reference / delta / matches_reference
一起写进 outputs/metrics/count_params.json，并在控制台逐项标注 OK / DIFF。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

# 仓库根靠 __file__ 推导，禁止写死个人绝对路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


# ---------------------------------------------------------------------------
# 统一指标落盘：优先用 utils/logging.py 的 dump_metrics；该文件若尚未落盘则用等价实现兜底。
# 两份实现的字段与语义完全一致（元信息 experiment/timestamp/command/cwd_rel/host/platform/python）。
# ---------------------------------------------------------------------------
def _local_dump_metrics(name: str, payload: dict, out_dir: str = "outputs/metrics") -> str:
    import datetime
    import os
    import platform

    out_dir = str(_resolve(out_dir)) if not os.path.isabs(str(out_dir)) else str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "experiment": name,
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "cwd_rel": ".",
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**meta, **payload}, f, ensure_ascii=True, indent=2)
    print(f"[metrics] wrote {path}")
    return path


try:  # pragma: no cover - 取决于 utils/logging.py 是否已落盘
    from utils.logging import dump_metrics as _dump_metrics  # type: ignore
except Exception:  # noqa: BLE001
    _dump_metrics = _local_dump_metrics


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


# ---------------------------------------------------------------------------
# 建网：timm / official 两条路必须都能建，且都不得让官方代码污染 timm 注册表。
# ---------------------------------------------------------------------------
def load_official_module():
    """加载 models/repvit_official.py。

    该文件是官方 model/repvit.py 的逐字节 vendored 拷贝（已删掉 register_model 装饰器）。
    优先复用 models/build_model.load_official_module()（若已落盘）；否则用 importlib 直接
    以独立模块名 exec，绝不 `import`，以免官方注册项覆盖 timm 的同名模型。
    """
    try:
        from models.build_model import load_official_module as _loader  # type: ignore
        return _loader()
    except Exception:  # noqa: BLE001 - 文件不存在或接口未落盘都回落到本地加载
        pass
    path = ROOT / "models" / "repvit_official.py"
    if not path.exists():
        raise FileNotFoundError(f"找不到官方实现文件：{path}")
    spec = importlib.util.spec_from_file_location("repvit_official", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["repvit_official"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_model(arch: str, impl: str, num_classes: int, distillation: bool) -> nn.Module:
    """按 (impl, num_classes, distillation) 建一个未加载权重的模型（随机初始化即可，只数形状）。"""
    if impl == "timm":
        import timm
        return timm.create_model(arch, pretrained=False, num_classes=int(num_classes),
                                 distillation=bool(distillation))
    if impl == "official":
        mod = load_official_module()
        factory = getattr(mod, arch, None)
        if factory is None:
            raise AttributeError(f"官方实现里没有 {arch!r} 这个工厂函数")
        # 官方顺序不可交换：先建网，再由调用方决定是否 fuse
        return factory(num_classes=int(num_classes), distillation=bool(distillation))
    raise ValueError(f"未知 impl={impl!r}，可选 timm / official")


# ---------------------------------------------------------------------------
# 三种计数口径
# ---------------------------------------------------------------------------
def count_params(model: nn.Module) -> int:
    """口径 A：全部可学习参数（含所有分类头）。"""
    return int(sum(p.numel() for p in model.parameters()))


def count_backbone_params(model: nn.Module, head_prefixes=("head", "classifier")) -> int:
    """口径 B：骨干参数 —— 去掉 head / classifier 前缀后的可学习参数。

    两种实现的命名不同：timm 用 head.head / head.head_dist，官方用
    classifier.classifier / classifier.classifier_dist，用 startswith 同时覆盖两套前缀。
    """
    total = 0
    for name, p in model.named_parameters():
        if any(name.startswith(pre) for pre in head_prefixes):
            continue
        total += int(p.numel())
    return int(total)


def _replace_batchnorm(net: nn.Module) -> None:
    """官方仓库 utils.py 的 replace_batchnorm 语义：递归融合「带 fuse() 的模块」，
    并把残留的 BatchNorm2d 换成 Identity。timm 的 RepVit 另有自带 fuse()，此处作为
    官方实现的融合入口。注意必须在 model.eval() 之后调用（要用 running 统计量）。"""
    for child_name, child in net.named_children():
        if hasattr(child, "fuse"):
            fused = child.fuse()
            setattr(net, child_name, fused)
            _replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            _replace_batchnorm(child)


def fuse_inplace(model: nn.Module, impl: str) -> nn.Module:
    """把「未融合」模型就地重参数化成推理态（融合后）结构。

    两条实现路径都是官方语义，结果参数量完全一致：
      * timm     -> model.fuse()：原地改写，含 RepVGGDW 合并 + BN 折叠 + 双头平均；
      * official -> 递归 replace_batchnorm：逐个 Conv2d_BN/RepVGGDW/BN_Linear 融合。
    """
    model.eval()                       # 必须先 eval：BN 折进卷积用的是 running_mean/var
    if impl == "timm" and hasattr(model, "fuse"):
        model.fuse()
    else:
        _replace_batchnorm(model)
    assert not any(isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)) for m in model.modules()), \
        "融合后仍残留 BatchNorm：融合入口用错了，参数量口径不可用"
    model.eval()
    return model


def _short(model_name: str) -> str:
    """repvit_m0_9 -> m0_9，用于拼 JSON 键名（与 PROGRESS.md 的关键数字表一致）。"""
    return model_name[len("repvit_"):] if model_name.startswith("repvit_") else model_name


def _ref_table(short: str) -> dict:
    return {
        f"{short}_params_train_double_head": 5489328,
        f"{short}_params_single_head_unfused": 5103560,
        f"{short}_params_backbone": 4717792,
        f"{short}_params_pet37_unfused": 4732805,
        f"{short}_params_fused_c1000": 5067056,
        f"{short}_params_fused_c37": 4696301,
        f"{short}_params_fused_delta": 36504,
    }


# 每一项的口径说明，既打印到控制台也写进 JSON，保证任何数字都能溯源到口径
ITEMS = [
    ("训练态双头（含蒸馏头，C=1000，未融合）", "params_train_double_head"),
    ("未融合单头（C=1000）", "params_single_head_unfused"),
    ("骨干（不含任何分类头 head/classifier）", "params_backbone"),
    ("未融合单头（C=37，本任务 Pet 建网形态）", "params_pet37_unfused"),
    ("融合后单头（C=1000，model.fuse() 之后）", "params_fused_c1000"),
    ("融合后单头（C=37，model.fuse() 之后）", "params_fused_c37"),
]


def main() -> int:
    ap = argparse.ArgumentParser("RepViT 参数量五口径统计")
    ap.add_argument("--model", default="repvit_m0_9", help="模型名（timm 名或官方工厂名）")
    ap.add_argument("--impl", default="timm", choices=["timm", "official"],
                    help="timm / official；official 走 models.build_model.load_official_module()")
    ap.add_argument("--out-dir", default="outputs/metrics")
    ap.add_argument("--json-name", default="count_params")
    args = ap.parse_args()

    arch = args.model
    short = _short(arch)

    print("=" * 78)
    print(" count_params —— RepViT 参数量五口径（DoD #13）")
    print(f" model={arch}  impl={args.impl}  short={short}")
    print("-" * 78)
    print(" 参数量口径（全表统一）：params = sum(p.numel() for p in model.parameters())")
    print("   * 只数 nn.Parameter；BN 的 running_mean / running_var / num_batches_tracked")
    print("     是 buffer 不是 parameter，一律不计入（融合后这些 buffer 会全部消失）。")
    print("   * 训练态 / 未融合 / 融合后 是三种不同的结构，先固定结构再数，绝不混用。")
    print("   * 骨干口径 = named_parameters() 里去掉 head / classifier 前缀后的和。")
    print("   * 融合口径 = eval() 之后调用 fuse()（timm）或 replace_batchnorm（official）。")
    print("=" * 78)

    results: dict = {}

    # ---- ① 训练态双头 C=1000 ----
    m = build_model(arch, args.impl, 1000, True)
    results["params_train_double_head"] = count_params(m)
    del m

    # ---- ② 未融合单头 C=1000（顺带拿骨干口径）----
    m = build_model(arch, args.impl, 1000, False)
    results["params_single_head_unfused"] = count_params(m)
    results["params_backbone"] = count_backbone_params(m)
    del m

    # ---- ③ 未融合单头 C=37 ----
    m = build_model(arch, args.impl, 37, False)
    results["params_pet37_unfused"] = count_params(m)
    del m

    # ---- ④ 融合后单头 C=1000 / C=37 ----
    m = build_model(arch, args.impl, 1000, False)
    results["params_fused_c1000"] = count_params(fuse_inplace(m, args.impl))
    del m
    m = build_model(arch, args.impl, 37, False)
    results["params_fused_c37"] = count_params(fuse_inplace(m, args.impl))
    del m

    results["params_fused_delta"] = results["params_single_head_unfused"] - results["params_fused_c1000"]

    ref = _ref_table(short)
    payload_items = []
    n_diff = 0
    for i, (desc, key) in enumerate(ITEMS, 1):
        val = results[key]
        r = ref.get(f"{short}_{key}")
        if r is None:
            flag, delta = "n/a", None
        elif val == r:
            flag, delta = "OK", 0
        else:
            flag, delta = "DIFF", val - r
            n_diff += 1
        payload_items.append({"item": i, "key": f"{short}_{key}", "desc": desc,
                              "value": val, "reference": r, "flag": flag, "delta_vs_reference": delta})
        print(f" [{i}] {desc}")
        print(f"       {short}_{key} = {val:,}   ref={r if r is None else format(r, ',')}  [{flag}]"
              + ("" if delta in (0, None) else f"  (实测-基准={delta:+,})"))

    d, rd = results["params_fused_delta"], ref[f"{short}_params_fused_delta"]
    if d == rd:
        flag = "OK"
    else:
        flag = "DIFF"
        n_diff += 1
    print(f" [7] 融合净减（未融合单头 C=1000 − 融合后单头 C=1000）")
    print(f"       {short}_params_fused_delta = {d:,}   ref={rd:,}  [{flag}]"
          + ("" if d == rd else f"  (实测-基准={d - rd:+,})"))
    payload_items.append({"item": 7, "key": f"{short}_params_fused_delta",
                          "desc": "融合净减（未融合单头 C=1000 − 融合后单头 C=1000）",
                          "value": d, "reference": rd, "flag": flag,
                          "delta_vs_reference": d - rd})

    print("-" * 78)
    if n_diff == 0:
        print(f" [verdict] 全部 {len(payload_items)} 项与基准整数逐位一致。")
    else:
        print(f" [verdict] 有 {n_diff} 项与基准整数不一致；规格书规定「一律以本机实测为准」，"
              f"JSON 里已标注 delta_vs_reference，报告里需说明实现/版本差异。")

    payload = {
        "model": arch,
        "impl": args.impl,
        "unit": "params（可学习参数张量元素数，不含 buffer）",
        "param_definition": "sum(p.numel() for p in model.parameters())",
        "backbone_definition": "named_parameters() 去掉 head/classifier 前缀后的元素数之和",
        "fuse_definition": "eval() 后 timm 用 model.fuse()，official 用等价 replace_batchnorm 递归融合",
        "results": {f"{short}_{k}": v for k, v in results.items()},
        "items": payload_items,
        "reference_source": "02_AI_Agent执行规格书.md 模块 9 / PROGRESS.md 关键数字表（本机实测复现）",
        "matches_reference_all": bool(n_diff == 0),
        "n_items_diff_from_reference": int(n_diff),
    }
    _dump_metrics(args.json_name, payload, out_dir=str(_resolve(args.out_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
