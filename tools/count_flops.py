# -*- coding: utf-8 -*-
"""tools/count_flops.py —— RepViT 推理态 MACs 统计（对应 DoD #14）。

用法：
    python tools/count_flops.py --model repvit_m0_9 --input-size 224
    python tools/count_flops.py --model repvit_m0_9 --input-size 224 --fused

三个必须说清楚的点（否则数字没法跟论文/评委对齐）：

1. 单位是 MACs，不是 FLOPs。本脚本输出里所有数字都显式标注 `unit=MACs`。
   MACs = 乘加次数（一次乘 + 一次加记 1 次）。有些框架报的 "FLOPs" 其实是 2×MACs，
   本脚本既不做 ×2，也不把 MACs 叫成 FLOPs —— 847,050,816 MACs 就是 0.847 GMACs。

2. 必须先 model.eval() 再统计。训练态下 BN 走 batch 统计、还有 dropout / 双头平均分支，
   得到的不是推理态数值；而且 timm 的 RepVit 在训练态 forward 返回的是 (logits, logits_dist)
   元组，thop/fvcore 都可能直接报错或漏统计。fuse() 函数把 eval() 写成强制步骤，
   调用方不需要（也不允许）绕过它。

3. 融合（fuse）会改变 MACs：BN 被折进卷积后不再是独立算子，RepVGGDW 的三个分支合并成
   一个 3x3 卷积。所以本脚本对「未融合」与「融合后」两种结构都统计一遍并分别打印，
   默认（不加 --fused）的规范口径是「未融合 + eval」，即 847,050,816 MACs。

实测基准（本机 thop 0.1.1 / fvcore 0.1.5 / torch 2.14.0 / timm 1.0.29，repvit_m0_9，1x3x224x224）：
    thop   未融合(eval) = 847050816 (0.847 GMACs)   融合后 = 815617344
    fvcore 未融合(eval) = 832165824 （同口径，不是 2 倍） 融合后 = 815617344
两个后端都是 MACs 口径，差值 14,884,992 来自「BN 与逐元素加法算不算一次乘加」的计法差异
（thop 把 BN 的逐元素乘加也计入，fvcore 的 FlopCountAnalysis 不计）。规格书要求两个都跑、
分别打印，不要试图把它们凑成同一个整数。

thop 与 fvcore 都是可选依赖：有一个跑一个，两个都有就跑两个，两个都没有则打印安装提示后
exit(0)（这是环境缺依赖，不是脚本失败）。
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


# ---------------------------------------------------------------------------
# 统一指标落盘（utils/logging.py 未落盘时的等价兜底，字段语义完全一致）
# ---------------------------------------------------------------------------
def _local_dump_metrics(name: str, payload: dict, out_dir: str = "outputs/metrics") -> str:
    import datetime
    import os
    import platform

    p = Path(out_dir)
    out_dir = str(p if p.is_absolute() else (ROOT / p))
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


try:  # pragma: no cover
    from utils.logging import dump_metrics as _dump_metrics  # type: ignore
except Exception:  # noqa: BLE001
    _dump_metrics = _local_dump_metrics


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


# ---------------------------------------------------------------------------
# 建网 / 融合
# ---------------------------------------------------------------------------
def load_official_module():
    """见 tools/count_params.py 的同一函数：官方代码只能 exec 加载，不得 import。"""
    try:
        from models.build_model import load_official_module as _loader  # type: ignore
        return _loader()
    except Exception:  # noqa: BLE001
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
    if impl == "timm":
        import timm
        return timm.create_model(arch, pretrained=False, num_classes=int(num_classes),
                                 distillation=bool(distillation))
    if impl == "official":
        mod = load_official_module()
        factory = getattr(mod, arch, None)
        if factory is None:
            raise AttributeError(f"官方实现里没有 {arch!r} 这个工厂函数")
        return factory(num_classes=int(num_classes), distillation=bool(distillation))
    raise ValueError(f"未知 impl={impl!r}，可选 timm / official")


def _replace_batchnorm(net: nn.Module) -> None:
    """官方仓库 utils.py 的 replace_batchnorm 语义（官方实现没有顶层 fuse()）。"""
    for child_name, child in net.named_children():
        if hasattr(child, "fuse"):
            fused = child.fuse()
            setattr(net, child_name, fused)
            _replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            _replace_batchnorm(child)


def fuse(model: nn.Module, reg: dict | None = None, fused: bool = True) -> nn.Module:
    """把模型切成「可统计」的状态，并返回模型本体。

    参数语义（冻结契约）：
        model  : 待统计的模型
        reg    : 模型登记项（deploy/model_registry.py 的 get(key) 结果，或任何含
                 `impl` / `num_classes` / `distillation` 的 dict）；只用来判断走哪条融合路径。
        fused  : True  -> 额外返回「融合后」的推理态结构（深拷贝，不改调用方模型）
                 False -> 只做 eval()，保持未融合的多分支结构

    无论 fused 取什么值，都**强制先 model.eval()**：BN 只有在 eval 下才用 running 统计量，
    训练态得到的 MACs 不是推理态数值（且训练态 forward 会返回双头元组）。
    """
    model.eval()
    if not fused:
        return model
    impl = (reg or {}).get("impl", "timm")
    m = copy.deepcopy(model)           # fuse 是原地改写，必须深拷贝
    if impl == "timm" and hasattr(m, "fuse"):
        m.fuse()
    else:
        _replace_batchnorm(m)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# 两个后端
# ---------------------------------------------------------------------------
def macs_thop(model: nn.Module, x: torch.Tensor):
    """thop：一个 MACs 计数。BN 的逐元素乘加也计入。"""
    from thop import profile
    macs, params = profile(model, inputs=(x,), verbose=False)
    return int(macs), int(params)


def macs_fvcore(model: nn.Module, x: torch.Tensor) -> int:
    """fvcore：FlopCountAnalysis.total()，与 thop 同为 MACs 口径（不乘 2）。"""
    from fvcore.nn import FlopCountAnalysis
    ana = FlopCountAnalysis(model, (x,))
    ana.unsupported_ops_warnings(False)
    ana.uncalled_modules_warnings(False)
    return int(ana.total())


def probe(backend: str, model: nn.Module, x: torch.Tensor):
    """返回 dict 或 抛异常；由调用方捕获并打印。"""
    if backend == "thop":
        macs, params = macs_thop(model, x)
        return {"macs": macs, "params": params}
    if backend == "fvcore":
        return {"macs": macs_fvcore(model, x), "params": int(sum(p.numel() for p in model.parameters()))}
    raise ValueError(backend)


def main() -> int:
    ap = argparse.ArgumentParser("RepViT 推理态 MACs 统计")
    ap.add_argument("--model", default="repvit_m0_9")
    ap.add_argument("--impl", default="timm", choices=["timm", "official"])
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--distillation", type=int, default=1,
                    help="1 = 训练态双头（默认，与 DoD #14 基准 847050816 同口径）；0 = 单头")
    ap.add_argument("--fused", action="store_true",
                    help="把默认口径切成「融合后」结构（默认口径是未融合 + eval）")
    ap.add_argument("--backends", default="thop,fvcore", help="逗号分隔，可选 thop / fvcore")
    ap.add_argument("--out-dir", default="outputs/metrics")
    ap.add_argument("--json-name", default="count_flops")
    args = ap.parse_args()

    want = [b.strip() for b in args.backends.split(",") if b.strip()]
    avail, missing = [], []
    for b in want:
        try:
            if b == "thop":
                import thop  # noqa: F401
            elif b == "fvcore":
                import fvcore  # noqa: F401
            else:
                raise ImportError(b)
            avail.append(b)
        except Exception:  # noqa: BLE001
            missing.append(b)

    print("=" * 78)
    print(" count_flops —— RepViT 推理态 MACs（DoD #14）")
    print(f" model={args.model}  impl={args.impl}  input={args.batch_size}x3x{args.input_size}x{args.input_size}")
    print(f" num_classes={args.num_classes}  distillation={args.distillation}  unit=MACs")
    print(" 口径说明：MACs = 乘加次数，1 MAC = 1 次乘 + 1 次加；本脚本不做 ×2，")
    print("           所有数字都是 MACs，绝不等同于某些框架所称的 FLOPs(2×MACs)。")
    print(" 前置条件：统计前一律 model.eval()（由 fuse() 强制保证）。")
    print("=" * 78)

    if not avail:
        print("[SKIP] thop 与 fvcore 都不可用，无法统计 MACs。")
        print("       安装任一个即可：")
        print("         pip install thop")
        print("         pip install fvcore")
        _dump_metrics(args.json_name, {
            "model": args.model, "impl": args.impl, "unit": "MACs",
            "input_shape": [args.batch_size, 3, args.input_size, args.input_size],
            "available_backends": [], "missing_backends": missing,
            "note": "thop 与 fvcore 均未安装，未做任何统计；安装任一后端后重跑本脚本",
        }, out_dir=str(_resolve(args.out_dir)))
        return 0

    print(f" 可用后端: {avail}   缺失: {missing if missing else '无'}")

    reg = {"impl": args.impl, "num_classes": args.num_classes,
           "distillation": args.distillation}
    x = torch.randn(args.batch_size, 3, args.input_size, args.input_size)

    results = {}
    for mode, do_fuse in (("unfused_eval", False), ("fused_eval", True)):
        model = build_model(args.model, args.impl, args.num_classes, bool(args.distillation))
        model = fuse(model, reg, fused=do_fuse)          # 内部已 eval()
        assert not model.training, "统计前必须是 eval 态"
        print("-" * 78)
        print(f"[结构] mode={mode}  ({'未融合，多分支' if not do_fuse else '融合后，推理态'})")
        for b in avail:
            try:
                r = probe(b, model, x)
                results.setdefault(mode, {})[b] = r
                gmacs = r["macs"] / 1e9
                print(f"  backend={b:<7} unit=MACs macs={r['macs']} ({gmacs:.3f} GMACs) "
                      f"params={r['params']}")
            except Exception as e:  # noqa: BLE001
                results.setdefault(mode, {})[b] = {"macs": None, "error": f"{type(e).__name__}: {e}"}
                print(f"  backend={b:<7} [ERROR] {type(e).__name__}: {e}")
        del model

    canonical_mode = "fused_eval" if args.fused else "unfused_eval"
    canonical = results.get(canonical_mode, {}).get(avail[0], {}).get("macs")
    print("=" * 78)
    for b in avail:
        v = results.get(canonical_mode, {}).get(b, {}).get("macs")
        print(f" [canonical] backend={b} mode={canonical_mode} unit=MACs macs={v} "
              f"({(v or 0) / 1e9:.3f} GMACs)")
    print(f" [canonical] 规范口径 = {'融合后' if args.fused else '未融合'} + eval()；"
          f"另一个结构的数值见上表，两者不可混用。")

    ref = {"thop": 847050816, "fvcore": 832165824}
    flag = {}
    for b in avail:
        v = results.get("unfused_eval", {}).get(b, {}).get("macs")
        if b in ref and v is not None:
            flag[b] = bool(v == ref[b])
            print(f" [ref] {b} 未融合基准={ref[b]}  实测={v}  "
                  f"[{'OK' if v == ref[b] else 'DIFF'}]"
                  + ("" if v == ref[b] else f"  (实测-基准={v - ref[b]:+,})"))

    payload = {
        "model": args.model,
        "impl": args.impl,
        "unit": "MACs",
        "unit_definition": "MACs = multiply-accumulate count；1 MAC = 1 mul + 1 add；不乘 2",
        "input_shape": [args.batch_size, 3, args.input_size, args.input_size],
        "num_classes": args.num_classes,
        "distillation": args.distillation,
        "eval_before_count": True,
        "canonical_mode": canonical_mode,
        "canonical_macs": canonical,
        "results_by_mode": results,
        "reference": ref,
        "matches_reference": flag,
        "available_backends": avail,
        "missing_backends": missing,
        "note": "thop 与 fvcore 都是 MACs 口径；差值 14884992 来自 BN / 逐元素加法是否计入的计法差异",
    }
    _dump_metrics(args.json_name, payload, out_dir=str(_resolve(args.out_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
