# tools/check_report_assets.py
"""Source: Self-written
验收脚本：断言报告每一节所需的图表/文件都存在，缺哪张列哪张。
用法: python tools/check_report_assets.py --root . --strict
"""
from __future__ import annotations
import argparse, sys
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tools.report_spec import REPORT_SECTIONS, PPT_SLIDES, FIGURE_SPEC, BENCH_META_KEYS

MIN_BYTES = {"png": 8 * 1024, "jpg": 8 * 1024, "csv": 200, "md": 200, "json": 64,
             "txt": 32, "onnx": 1024 * 1024, "yaml": 64, "pt": 1024 * 1024}


def check(root: Path):
    ok, warn, fail = [], [], []
    for no, title, assets in REPORT_SECTIONS:
        for rel in assets:
            p = root / rel
            if not p.exists():
                fail.append(f"报告第 {no} 节《{title}》缺失: {rel}")
                continue
            suffix = p.suffix.lstrip(".").lower()
            need = MIN_BYTES.get(suffix, 1)
            size = p.stat().st_size
            if size < need:
                warn.append(f"报告第 {no} 节 {rel} 仅 {size} 字节 (<{need})，疑似空图/空表")
            else:
                ok.append(f"报告第 {no} 节 {rel} 就绪 ({size/1024:.1f} KB)")

    for no, title, figs in PPT_SLIDES:
        for rel in figs:
            (ok if (root / rel).exists() else fail).append(
                f"PPT 第 {no} 页《{title}》配图 {rel} " + ("就绪" if (root / rel).exists() else "缺失"))

    # 性能元信息 10 项必须在最近一次 benchmark 记录里齐全（bench.jsonl 每行一条，取末行）
    import json
    jl = root / "outputs/metrics/bench.jsonl"
    if not jl.exists():
        fail.append("性能测试元信息: outputs/metrics/bench.jsonl 不存在")
    else:
        lines = [l for l in jl.read_text(encoding="utf-8").splitlines() if l.strip()]
        rec = json.loads(lines[-1]) if lines else {}
        for k in BENCH_META_KEYS:
            if k == "io_nodes":
                present = bool(rec.get("onnx_input")) and bool(rec.get("onnx_output"))
            elif k == "hardware":
                present = bool(rec.get("cpu_model")) and bool(rec.get("os"))
            elif k == "runs":
                present = rec.get("warmup") is not None and rec.get("runs") is not None
            else:
                present = rec.get(k) is not None
            (ok if present else fail).append(f"性能测试元信息[{k}]: {'齐全' if present else '缺失'}")

    # ONNX 产物与两份标签文件（ImageNet-1000 / Pet-37 绝不混用）。
    # ONNX 路径一律经 registry 取，正文与脚本都不硬写 *.onnx 文件名。
    from deploy.model_registry import onnx_path
    for key in ["repvit_m0_9_pet37", "repvit_m0_9_pet37_opt", "repvit_m0_9_in1k", "repvit_m1_0_in1k"]:
        p = Path(onnx_path(key))
        (ok if p.exists() else fail).append(f"ONNX[{key}]: " +
                                            ("存在" if p.exists() else f"缺失 ({p})"))

    for rel in ["labels/imagenet_classes.txt", "labels/pet_classes.txt"]:
        (ok if (root / rel).exists() else fail).append(f"标签文件 {rel}: " +
                                                       ("存在" if (root / rel).exists() else "缺失"))

    for tag, items in (("OK", ok), ("WARN", warn), ("FAIL", fail)):
        for s in items:
            print(f"[{tag}] {s}")
    print(f"\n汇总: OK={len(ok)} WARN={len(warn)} FAIL={len(fail)}")
    return 1 if fail else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--strict", action="store_true", help="存在 FAIL 时以退出码 1 结束")
    a = ap.parse_args()
    sys.exit(check(Path(a.root).resolve()) if a.strict else (check(Path(a.root).resolve()), 0)[1])
