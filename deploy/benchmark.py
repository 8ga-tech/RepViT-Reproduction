# -*- coding: utf-8 -*-
"""deploy/benchmark.py —— ONNX Runtime CPU 延迟测试（预热10/正式50/P50/P95/端到端分解）。"""
import argparse, csv, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.model_registry import get, onnx_path          # noqa
from deploy.infer_onnx import preprocess, softmax, build_session   # noqa

def cpu_name() -> str:
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor() or platform.machine()

def mem_gb() -> float:
    try:
        import ctypes
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = MS(); s.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.ullTotalPhys / 1024 ** 3
    except Exception:
        return float("nan")

def env_report() -> dict:
    import onnx, torch
    return dict(os=platform.platform(), machine=platform.machine(), cpu=cpu_name(),
                logical_cores=os.cpu_count(), mem_gb=round(mem_gb(), 1),
                python=sys.version.split()[0], onnxruntime=ort.__version__,
                onnx=onnx.__version__, torch=torch.__version__,
                available_providers=ort.get_available_providers(),
                power_plan="由测试者手工确认（建议「最佳性能」）")

def stats_ms(lat: np.ndarray) -> dict:
    return dict(mean_ms=round(float(lat.mean()), 3),
                min_ms=round(float(lat.min()), 3), max_ms=round(float(lat.max()), 3),
                p50_ms=round(float(np.percentile(lat, 50)), 3),
                p95_ms=round(float(np.percentile(lat, 95)), 3),
                percentile_method="numpy linear")

def run_one(name, warmup, runs, threads, image=None) -> dict:
    reg = get(name)
    sess = build_session(onnx_path(name), threads)
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    print(f"\n=== {name} ===")
    print(f"EP(actual) : {sess.get_providers()}")
    print(f"input      : {inp.name} {inp.shape} {inp.type}")
    print(f"output     : {out.name} {out.shape} {out.type}")
    x = np.random.randn(1, 3, reg["input_size"], reg["input_size"]).astype(np.float32)
    for _ in range(warmup):                       # 预热：turbo 稳态 + 图优化 + arena 预分配
        sess.run(None, {inp.name: x})
    lat = np.empty(runs, np.float64)
    for i in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})             # CPU EP 的 run 同步，返回即完成
        lat[i] = (time.perf_counter() - t0) * 1000.0
    so = sess.get_session_options()
    rep = dict(model=name, onnx_path=onnx_path(name),
               file_size_mb=round(Path(onnx_path(name)).stat().st_size / 1e6, 2),
               providers=list(sess.get_providers()),
               provider_actual=list(sess.get_providers()),
               ep=list(sess.get_providers()),          # 与 provider_actual 同义，避免验收脚本二义
               ort_version=ort.__version__, cpu_model=cpu_name(), os=platform.platform(),
               threads_intra=so.intra_op_num_threads, threads_inter=so.inter_op_num_threads,
               warmup=warmup, runs=runs, precision="FP32",
               input_size=reg["input_size"], batch_size=1,
               **stats_ms(lat), raw_ms=[round(float(v), 3) for v in lat])
    print(f"pure-infer : mean={rep['mean_ms']} P50={rep['p50_ms']} P95={rep['p95_ms']} "
          f"min={rep['min_ms']} max={rep['max_ms']} (n={runs})")
    if image and Path(image).exists():            # 端到端分解：预处理常与推理同量级，不拆会误判
        pre, inf, post = [], [], []
        for i in range(warmup + runs):
            t0 = time.perf_counter()
            xi = preprocess(image, reg["input_size"], reg["mean"], reg["std"], reg["crop_pct"])
            t1 = time.perf_counter()
            z = sess.run(None, {inp.name: xi})[0][0]
            t2 = time.perf_counter()
            _ = int(np.argmax(softmax(z)))
            t3 = time.perf_counter()
            if i >= warmup:
                pre.append((t1 - t0) * 1e3); inf.append((t2 - t1) * 1e3); post.append((t3 - t2) * 1e3)
        e2e = np.array(pre) + np.array(inf) + np.array(post)
        rep.update(preprocess_ms=round(float(np.mean(pre)), 3),
                   inference_ms=round(float(np.mean(inf)), 3),
                   postprocess_ms=round(float(np.mean(post)), 3),
                   e2e_p50_ms=round(float(np.percentile(e2e, 50)), 3),
                   e2e_p95_ms=round(float(np.percentile(e2e, 95)), 3))
        print(f"e2e        : pre={rep['preprocess_ms']}ms inf={rep['inference_ms']}ms "
              f"post={rep['postprocess_ms']}ms  e2e P50={rep['e2e_p50_ms']}ms")
    return rep

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="extend", nargs="+", required=True,
                    help="registry key，可多个：--model a --model b 或 --model a b")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--image", default="outputs/samples/sample.jpg")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/benchmarks"))
    a = ap.parse_args()
    env = env_report()
    print(json.dumps(env, ensure_ascii=True, indent=2))
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    jl = ROOT / "outputs/metrics/bench.jsonl"      # 验收脚本读这一份，逐次追加、不可覆盖
    jl.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in a.model:
        rep = run_one(m, a.warmup, a.runs, a.threads, a.image)
        rep["env"] = env
        (od / f"{m}_benchmark.json").write_text(
            json.dumps(rep, ensure_ascii=True, indent=2), encoding="utf-8")
        with jl.open("a", encoding="utf-8") as f:  # 每行一条记录，字段与 JSON 顶层完全一致
            f.write(json.dumps(rep, ensure_ascii=True) + "\n")
        rows.append(rep)
    csv_path = od / "summary.csv"
    cols = ["model", "onnx_path", "file_size_mb", "provider_actual", "input_size", "precision",
            "batch_size", "warmup", "runs", "threads_intra", "mean_ms", "p50_ms", "p95_ms",
            "min_ms", "max_ms", "cpu_model", "os", "ort_version",
            "preprocess_ms", "inference_ms", "postprocess_ms", "e2e_p50_ms"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\n[ok] ->", csv_path, "|", jl)

if __name__ == "__main__":
    main()
