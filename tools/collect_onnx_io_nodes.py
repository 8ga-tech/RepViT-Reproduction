# tools/collect_onnx_io_nodes.py
"""Source: Self-written
只读取每个已落盘 ONNX 的输入/输出节点信息，写成 ``outputs/metrics/onnx_io_nodes.json``。

用途（题目五-5 性能测试第 8 条「记录模型输入输出节点信息」）
------------------------------------------------------------
``deploy/benchmark.py`` 早期版本只把 ``input:`` / ``output:`` 打印到控制台，没有写进
``outputs/metrics/bench.jsonl`` 与 ``outputs/benchmarks/*_benchmark.json``，
于是报告素材表 ``table_benchmark_meta`` 的 ``io_nodes`` 列只能是 ``in=NoneNone``。

本脚本用 ``onnx.load()`` **只读**读取图定义补齐该字段：

* 不加载 ONNX Runtime 会话，不做任何推理；
* 不重跑基准、不改动任何已发布的延迟数字（``bench.jsonl`` / ``summary.csv`` 只读）；
* 仓库相对路径写盘（不写死个人电脑绝对路径）。

用法::

    python tools/collect_onnx_io_nodes.py
    python tools/collect_onnx_io_nodes.py --out outputs/metrics/onnx_io_nodes.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def repo_rel(p: Path) -> str:
    """落盘字段一律写「相对仓库根的 POSIX 路径」（试题第 14 页：不得写死个人绝对路径）。"""
    q = Path(p).resolve()
    try:
        return q.relative_to(ROOT).as_posix()
    except ValueError:
        return q.as_posix()


def describe(value_info) -> dict:
    """把 onnx.ValueInfoProto 转成可 JSON 化的 dict（name / shape / type）。"""
    tt = value_info.type.tensor_type
    dims = []
    for d in tt.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?"))
    return dict(name=value_info.name, shape=dims, type=str(tt.elem_type))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/metrics/onnx_io_nodes.json")
    a = ap.parse_args()

    import onnx
    from deploy.model_registry import keys, onnx_path, MODELS

    models: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for key in keys():
        p = Path(onnx_path(key))
        if not p.exists():
            skipped[key] = "ONNX 未导出（按需导出的型号不参与本表）"
            continue
        graph = onnx.load(str(p), load_external_data=False).graph
        models[key] = dict(
            onnx_path=repo_rel(p),
            file_size_mb=round(p.stat().st_size / 1e6, 2),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            graph_input_count=len(graph.input),
            graph_output_count=len(graph.output),
            onnx_input=describe(graph.input[0]) if len(graph.input) else {},
            onnx_output=describe(graph.output[0]) if len(graph.output) else {},
            all_graph_inputs=[describe(v) for v in graph.input],
            all_graph_outputs=[describe(v) for v in graph.output],
            num_classes=MODELS[key]["num_classes"],
        )
        print(f"[io] {key:<24} in={models[key]['onnx_input'].get('name')}"
              f"{models[key]['onnx_input'].get('shape')} "
              f"out={models[key]['onnx_output'].get('name')}{models[key]['onnx_output'].get('shape')}")

    import onnxruntime as ort
    rep = dict(generated_by="python tools/collect_onnx_io_nodes.py",
               purpose="性能测试第 8 条：记录模型输入输出节点信息（只读 onnx.load，未重跑基准）",
               onnx_version=onnx.__version__, onnxruntime=ort.__version__,
               models=models, skipped=skipped)
    dst = ROOT / a.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(rep, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[ok] {len(models)} 个 ONNX 的 IO 节点信息 -> {repo_rel(dst)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
