# tools/make_report_assets.py
"""
Source  : Self-written
用途    : 扫描 outputs/{logs,metrics,pretrained_eval}/ 下全部 json/csv/jsonl，生成报告与 PPT
          所需的全部表格。**只搬运不编造**：任何取不到值的单元格写 "—" 而非 0。
用法    :
    python tools/make_report_assets.py --root . --out outputs/report_assets
    python tools/make_report_assets.py --only results      # 只重建结果口径对照表
"""
from __future__ import annotations
import argparse, json, math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from tools.report_spec import (RESULT_COLUMNS, CALIBER_ROWS, BENCH_META_KEYS,
                               REPORT_SECTIONS, FIGURE_SPEC, PPT_SLIDES)


# ---------- 基础 IO ----------
def _json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _jsonl_last(p: Path):
    """JSONL（每行一条记录）取最后一行 = 最近一次运行；bench.jsonl 与评测 jsonl 都适用。"""
    if not p.exists():
        return None
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


def _first(d, *keys, default=None):
    """按优先级取第一个存在的键：兼容不同脚本写出的字段命名差异。"""
    if isinstance(d, dict):
        for k in keys:
            if d.get(k) is not None:
                return d[k]
    return default


def to_md(df: pd.DataFrame) -> str:
    """不依赖 tabulate 的 Markdown 表格渲染（pip 里 tabulate 常缺席）。"""
    def cell(v):
        return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v).replace("|", "\\|")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def dump(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")  # sig: Excel 直接打开不乱码
    (out_dir / f"{name}.md").write_text(to_md(df) + "\n", encoding="utf-8")
    print(f"[asset] {name}: {len(df)} 行 -> {out_dir / (name + '.csv')}")


# ---------- 表 A：结果口径对照表 ----------
def _pct(v, already_pct: bool = False):
    """already_pct=True 用于 outputs/pretrained_eval/<model_name>/metrics.json——全文唯一的百分数例外。"""
    if v is None:
        return "—"
    return f"{float(v):.2f}%" if already_pct else f"{100 * float(v):.2f}%"


def _lat(rec):
    if rec is None:
        return "—"
    m, p50, p95 = _first(rec, "mean_ms"), _first(rec, "p50_ms"), _first(rec, "p95_ms")
    if m is None:
        return "—"
    s = f"mean {m:.2f}ms"
    if p50 is not None and p95 is not None:
        s += f" / P50 {p50:.2f} / P95 {p95:.2f}"
    return s


def build_results(root: Path) -> pd.DataFrame:
    lit = yaml.safe_load((root / "report/sources/literature.yaml").read_text(encoding="utf-8"))
    rows = []
    for name, src, note, pct in CALIBER_ROWS:
        path_str, _, anchor = src.partition("#")
        if anchor:
            rec = lit.get(anchor) or _json(root / path_str)   # literature.yaml 里未收录 -> 留空
        elif path_str.endswith(".jsonl"):
            rec = _jsonl_last(root / path_str)                # bench.jsonl 每行一条，取末行
        else:
            rec = _json(root / path_str)
        rec = rec or {}
        rows.append({
            "口径": name,
            "数据集": _first(rec, "dataset", default="—"),
            "类别数": _first(rec, "num_classes", default="—"),
            # pct=True 的那一行（官方权重实测）读到的已是百分数，不能再乘 100
            "Top-1": _pct(_first(rec, "top1", "top_1"), pct),
            "Top-5": _pct(_first(rec, "top5", "top_5"), pct),
            "Macro-F1": _pct(_first(rec, "macro_f1", "f1_macro")),
            # 参数量必须带口径后缀：融合后单头 / 未融合单头 / 未融合双头
            "参数量": (lambda v, c: "—" if v is None else f"{v:.4f} M ({c})")(
                _first(rec, "params_m", "params"), _first(rec, "params_caliber", default="未标注口径")),
            "MACs": (lambda v, c: "—" if v is None else f"{v:.3f} G ({c})")(
                _first(rec, "macs_g"), _first(rec, "macs_caliber", default="MACs 口径(未乘 2)")),
            "文件大小": (lambda v: "—" if v is None else f"{v:.2f} MB")(
                _first(rec, "file_size_mb")),
            "延迟": _lat(rec) if isinstance(rec, dict) and "mean_ms" in rec else
                    (f"{rec['latency_ms']} ms ({rec.get('latency_note', '—')})"
                     if "latency_ms" in rec else "—"),
            "依据": note,
        })
    df = pd.DataFrame(rows)
    return df[["口径", *RESULT_COLUMNS, "依据"]]

# ---------- 表 B：图表清单表 ----------
def build_figures(root: Path) -> pd.DataFrame:
    rows = []
    for rel, (purpose, requirement, producer) in FIGURE_SPEC.items():
        p = root / rel
        rows.append({
            "文件名": rel,
            "用途": purpose,
            "对应题目要求": requirement,
            "生成脚本": producer,
            "状态": "已生成" if p.exists() else "缺失",
            "大小KB": round(p.stat().st_size / 1024, 1) if p.exists() else "",
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["状态", "文件名"], ascending=[True, True])


# ---------- 表 C：实验配置总表 ----------
def _flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = json.dumps(v, ensure_ascii=True) if isinstance(v, (list, dict)) else v
    return out


def build_configs(root: Path) -> pd.DataFrame:
    rows = {}
    suffix = "_config_effective.json"
    for eff in sorted((root / "outputs/logs").glob(f"*{suffix}")):
        rows[eff.name[: -len(suffix)]] = _flatten(_json(eff))   # 文件名前缀即 experiment_name
    if not rows:
        return pd.DataFrame(columns=["配置项"])
    df = pd.DataFrame(rows).T                      # 行 = 实验，列 = 配置项
    # 生效配置里**本身就有 experiment_name 这一列**（_flatten 会把 yaml 的
    # experiment_name 字段展平进来），直接 `df.index.name = "experiment_name"`
    # 再 reset_index() 会抛
    # ValueError: cannot insert experiment_name, already exists（实测复现）。
    # 先把同名列丢掉（行索引才是唯一真源），再 reset。
    if "experiment_name" in df.columns:
        df = df.drop(columns=["experiment_name"])
    df.index.name = "experiment_name"
    return df.reset_index()


# ---------- 表 D：训练日志摘要表 ----------
def build_training_summary(root: Path, metric="val_top1") -> pd.DataFrame:
    rows = []
    suffix = "_metrics.csv"
    for csv_path in sorted((root / "outputs/logs").glob(f"*{suffix}")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        best_idx = df[metric].idxmax()
        rows.append({
            "experiment_name": csv_path.name[: -len(suffix)],
            "epochs": int(df["epoch"].max()),
            "best_epoch": int(df.loc[best_idx, "epoch"]),
            f"best_{metric}": f"{df.loc[best_idx, metric]:.4f}",
            "best_val_top5": f"{df.loc[best_idx, 'val_top5']:.4f}",
            "best_val_macro_f1": f"{df.loc[best_idx, 'val_macro_f1']:.4f}",
            "final_train_loss": f"{df['train_loss'].iloc[-1]:.4f}",
            "final_val_loss": f"{df['val_loss'].iloc[-1]:.4f}",
            # 总耗时 = 各 epoch 耗时之和（含被中止后重启的段，jsonl 里不重复计数）
            "total_time_min": round(df["epoch_time_sec"].sum() / 60, 1),
            "csv_path": csv_path.as_posix(),
        })
    return pd.DataFrame(rows)


# ---------- 表 E：性能测试元信息表（题目要求 10 项）----------
def _bench_latest(root: Path) -> list[dict]:
    """benchmark 每条运行追加一行到 outputs/metrics/bench.jsonl：同一 model 取最后一条。"""
    p = root / "outputs/metrics/bench.jsonl"
    if not p.exists():
        return []
    latest = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            latest[rec.get("model")] = rec      # 后写的覆盖先写的 = 最近一次运行
    return list(latest.values())


def build_bench_meta(root: Path) -> pd.DataFrame:
    rows = []
    for rec in _bench_latest(root):
        inp, out = rec.get("onnx_input", {}), rec.get("onnx_output", {})
        rows.append({
            "model": rec.get("model"),
            "input_size": rec.get("input_size"),
            "batch_size": rec.get("batch_size"),
            "precision": rec.get("precision"),
            "backend": ",".join(rec.get("providers", [])),
            "hardware": f"{rec.get('cpu_model')} / {rec.get('os')}",
            "ort_version": rec.get("ort_version"),
            "runs": f"warmup={rec.get('warmup')}, runs={rec.get('runs')}",
            "file_size_mb": rec.get("file_size_mb"),
            "io_nodes": f"in={inp.get('name')}{inp.get('shape')} out={out.get('name')}{out.get('shape')}",
            "mean_ms": rec.get("mean_ms"), "p50_ms": rec.get("p50_ms"), "p95_ms": rec.get("p95_ms"),
            "threads_intra": rec.get("threads_intra"),
        })
    return pd.DataFrame(rows)


# ---------- 表 F：Markdown 模板骨架 ----------
HUMAN = "<!-- HUMAN: {} -->"

SECTION_TMPL = """## {no}. {title}

{HUMAN}

{assets}
"""

def build_markdown(root: Path, out_dir: Path) -> None:
    lines = [
        "# RepViT-M0.9 迁移训练、优化与 ONNX 多模型部署（项目报告）",
        "",
        "> 本文件由 `tools/make_report_assets.py` 生成骨架，表格为脚本填充，正文需人工撰写。",
        f"> 正文 12~20 页；表格数据源见 `{out_dir.as_posix()}/`。",
        "",
    ]
    for no, title, assets in REPORT_SECTIONS:
        block = []
        for a in assets:
            # 图与表统一用相对路径引用；缺失则留下显式 TODO，不放占位图
            block.append(f"![{Path(a).stem}]({a})" if a.endswith((".png", ".jpg"))
                         else f"- 数据/材料：`{a}`")
        if no == "9":
            block.append((out_dir / "table_results.md").read_text(encoding="utf-8"))
        if no == "14":
            block.append((out_dir / "table_benchmark_meta.md").read_text(encoding="utf-8"))
        lines.append(SECTION_TMPL.format(no=no, title=title,
                                         HUMAN=HUMAN.format(f"{title}正文，{len(title)}字以上"),
                                         assets="\n\n".join(block)))
    (out_dir.parent / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[asset] 报告骨架 -> {out_dir.parent / 'REPORT.md'}")


def build_ppt_outline(out_dir: Path) -> None:
    """PPT 15 页骨架：每页给出标题、要点槽位与应插入的图（性能页强制水印）。"""
    lines = ["# 答辩 PPT 大纲（10~15 页，性能页必须带元信息水印）", ""]
    for no, title, figs in PPT_SLIDES:
        lines += [f"## 第 {no} 页 · {title}", "", HUMAN.format("3~5 个要点，结论先行"), ""]
        for f in figs:
            lines.append(f"- 配图：`{f}`" + ("（**必须调用 stamp_meta 打水印**）"
                        if "latency" in f or "acc" in f else ""))
        lines.append("")
    (out_dir.parent / "PPT_OUTLINE.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[asset] PPT 大纲 -> {out_dir.parent / 'PPT_OUTLINE.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="outputs/report_assets")
    ap.add_argument("--only", default=None, help="results/figures/configs/summary/bench/md")
    a = ap.parse_args()
    root, out = Path(a.root).resolve(), Path(a.out)

    build = {
        "results": lambda: dump(build_results(root), out, "table_results"),
        "figures": lambda: dump(build_figures(root), out, "table_figures"),
        "configs": lambda: dump(build_configs(root), out, "table_configs"),
        "summary": lambda: dump(build_training_summary(root), out, "table_training_summary"),
        "bench":   lambda: dump(build_bench_meta(root), out, "table_benchmark_meta"),
        "md":      lambda: (build_markdown(root, out), build_ppt_outline(out)),
    }
    for k, fn in build.items():
        if a.only in (None, k):
            fn()


if __name__ == "__main__":
    main()
