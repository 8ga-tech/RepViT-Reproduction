# tools/check_report_assets.py
"""Source: Self-written
验收脚本：断言报告与 PPT 每一节/每一页所需的产物都存在，缺哪张列哪张。

用法::

    python tools/check_report_assets.py --root . --strict   # 有 FAIL 时退出码 1

检查项（全部与 tools/report_spec.py 的契约同源，不另立一套期望）
---------------------------------------------------------------
1. `REPORT_SECTIONS`：报告 16 节的产物逐个存在且不像空图/空表（按后缀给最小字节数）；
2. `PPT_SLIDES`：15 页每页实际读取的落盘产物逐个存在；
3. `FIGURE_SPEC`：图表清单里的每张图都存在（缺失会同时体现在 `table_figures` 的「缺失」行）；
4. 性能测试 10 项元信息：直接复用 `tools/make_report_assets.py` 的 `build_bench_meta()`，
   即「报告里看到的那张表」必须 10 项齐全（io_nodes 允许由只读的
   `outputs/metrics/onnx_io_nodes.json` 补齐——历史 bench.jsonl 里没有该字段）；
5. 随仓库交付的 ONNX：清单取自 `deploy/model_registry.py` 的 `shipped_keys()`；
6. 两份标签文件：`labels/imagenet_classes.txt`（1000 类）与 `labels/pet_classes.txt`（37 类）。

设计口径：**这个脚本不允许「跑了就红」**。任何一条 FAIL 都必须对应一个真实缺失的交付物，
先修产物或改契约，不允许把断言删掉了事。
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tools.report_spec import (REPORT_SECTIONS, PPT_SLIDES, FIGURE_SPEC,
                               BENCH_META_KEYS, CALIBER_ROWS)

MIN_BYTES = {"png": 8 * 1024, "jpg": 8 * 1024, "json": 64, "onnx": 1024 * 1024,
             "yaml": 64, "pt": 1024 * 1024}


def suspicious(p: Path) -> str | None:
    """空图/空表判定：文本类看「非空行数 ≥ 2」（表头 + 至少一行数据），
    位图与二进制看体积下限。返回原因字符串，正常返回 None。"""
    size = p.stat().st_size
    if p.suffix.lower() in (".csv", ".md", ".txt"):
        n = len([l for l in p.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()])
        return f"只有 {n} 行非空内容（表头 + 数据行应 ≥2）" if n < 2 else None
    need = MIN_BYTES.get(p.suffix.lstrip(".").lower(), 1)
    return f"仅 {size} 字节 (<{need})" if size < need else None


def check_bench_meta(root: Path, ok: list, warn: list, fail: list) -> None:
    """性能测试 10 项：检查报告素材表的实际内容（与 make_report_assets 同一函数）。"""
    from tools.make_report_assets import build_bench_meta
    df = build_bench_meta(root)
    if df.empty:
        fail.append("性能测试元信息: outputs/metrics/bench.jsonl 无有效记录（build_bench_meta 为空）")
        return
    for _, row in df.iterrows():
        model = row.get("model")
        for k in BENCH_META_KEYS:
            v = row.get(k)
            missing = v is None or (isinstance(v, float) and v != v) or str(v).strip() in ("", "—", "None", "NoneNone")
            (fail if missing else ok).append(
                f"性能测试元信息[{model}][{k}]: {'缺失' if missing else '齐全'}")


def check(root: Path):
    ok, warn, fail, info = [], [], [], []
    for no, title, assets in REPORT_SECTIONS:
        for rel in assets:
            p = root / rel
            if not p.exists():
                fail.append(f"报告第 {no} 节《{title}》缺失: {rel}")
                continue
            why = suspicious(p)
            if why:
                warn.append(f"报告第 {no} 节 {rel} {why}，疑似空图/空表")
            else:
                ok.append(f"报告第 {no} 节 {rel} 就绪 ({p.stat().st_size/1024:.1f} KB)")

    for no, title, figs in PPT_SLIDES:
        for rel in figs:
            exists = (root / rel).exists()
            (ok if exists else fail).append(
                f"PPT 第 {no} 页《{title}》产物 {rel} " + ("就绪" if exists else "缺失"))

    for rel in FIGURE_SPEC:
        exists = (root / rel).exists()
        if not exists:
            fail.append(f"图表清单缺失: {rel}（见 tools/report_spec.py 的 FIGURE_SPEC）")

    check_bench_meta(root, ok, warn, fail)

    # ONNX 产物与两份标签文件（ImageNet-1000 / Pet-37 绝不混用）。
    # 强制清单 = registry 的 shipped=True；shipped=False 的型号是「按需导出 / 因体积不入库」，
    # 不参与强制清单，但会以 INFO 打印出来，避免「登记了却查不到」的隐性缺口。
    from deploy.model_registry import onnx_path, shipped_keys, keys
    shipped = shipped_keys()
    for key in keys():
        p = Path(onnx_path(key))
        if key in shipped:
            (ok if p.exists() else fail).append(f"ONNX[{key}]: " +
                                                ("存在" if p.exists() else f"缺失 ({p})"))
        else:
            info.append(f"ONNX[{key}]: 非交付集（shipped=False，按需导出），本机"
                        + ("已导出" if p.exists() else "未导出"))

    for rel, n in (("labels/imagenet_classes.txt", 1000), ("labels/pet_classes.txt", 37)):
        p = root / rel
        if not p.exists():
            fail.append(f"标签文件 {rel}: 缺失")
            continue
        rows = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        (ok if len(rows) == n else fail).append(
            f"标签文件 {rel}: {len(rows)} 行（应为 {n}）")

    for name, src, _, _ in CALIBER_ROWS:
        rel = src.partition("#")[0]
        exists = (root / rel).exists()
        (ok if exists else fail).append(f"结果口径行《{name}》数据源 {rel}: "
                                        + ("存在" if exists else "缺失"))

    for tag, items in (("OK", ok), ("WARN", warn), ("FAIL", fail), ("INFO", info)):
        for s in items:
            print(f"[{tag}] {s}")
    print(f"\n汇总: OK={len(ok)} WARN={len(warn)} FAIL={len(fail)} INFO={len(info)}")
    return 1 if fail else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--strict", action="store_true", help="存在 FAIL 时以退出码 1 结束")
    a = ap.parse_args()
    rc = check(Path(a.root).resolve())
    sys.exit(rc if a.strict else 0)
