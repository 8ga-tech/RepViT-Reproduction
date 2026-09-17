# -*- coding: utf-8 -*-
"""deploy/benchmark.py —— ONNX Runtime CPU 延迟测试（预热10/正式50/P50/P95/端到端分解）。"""
import argparse, csv, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.model_registry import get, onnx_path, keys   # noqa
from deploy.infer_onnx import preprocess, softmax, build_session   # noqa
from deploy.compare_torch_onnx import repo_rel            # noqa

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


# --------------------------------------------------------------------------- #
# summary.csv：**聚合**写入。deploy/run_all.sh 对每个模型单独起一个进程，因此这里
# 绝不能用 open("w") 把文件重写成「本次跑过的模型」——那会把别的模型行整段抹掉
# （本会话真实发生过：6 行被最后一次调用覆盖成 1 行，PPT 的 P14 型号元信息跟着退化）。
# 规则：以 model 为键合并「文件里已有的行 + 本次的行」，本次结果覆盖同名行，其余保留。
# --------------------------------------------------------------------------- #
SUMMARY_NAME = "summary.csv"
SUMMARY_COLS = ["model", "onnx_path", "file_size_mb", "provider_actual", "input_size", "precision",
                "batch_size", "warmup", "runs", "threads_intra", "mean_ms", "p50_ms", "p95_ms",
                "min_ms", "max_ms", "cpu_model", "os", "ort_version",
                "preprocess_ms", "inference_ms", "postprocess_ms", "e2e_p50_ms"]


def read_summary(csv_path: Path) -> dict:
    """读回已有 summary.csv（model -> 行）。文件不存在 / 空 / 坏行都安全跳过。"""
    if not csv_path.exists():
        return {}
    out: dict[str, dict] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("model"):
                out[r["model"]] = dict(r)
    return out


def write_summary(csv_path: Path, rows: list[dict]) -> None:
    """原子写：先写 .tmp 再 os.replace，避免中断留下半截表（Windows 上 os.replace 可覆盖）。"""
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, csv_path)


def _canonical_order(models) -> list[str]:
    """summary.csv 的稳定行序：先按 registry 的**登记顺序**（单一真源），未登记的按名称排序在后。

    为什么不用「文件里的旧顺序」：一旦文件被截断或手工改过，旧顺序就带着历史事故的痕迹
    （实测：覆盖写事故后 m2_3 变成了第一行）。按 registry 顺序输出，任何一次重建 / 逐模型
    调用都得到同一份行序，下游（PPT 的延迟柱状图）不会在两次运行之间悄悄换序。
    """
    reg = {k: i for i, k in enumerate(keys())}
    return sorted(models, key=lambda m: (reg.get(m, len(reg)), m))


def merge_summary(prev: dict, new_rows: list[dict]) -> list[dict]:
    """合并：以 model 为键，本次结果覆盖同名行；行序取 registry 顺序（见 _canonical_order）。"""
    merged = {**prev}
    for r in new_rows:
        merged[r["model"]] = r
    return [merged[m] for m in _canonical_order(merged)]


def summary_json_models(od: Path) -> set[str]:
    """out_dir 下已有 benchmark JSON 的模型集合 = summary.csv 应当覆盖的模型集合。"""
    suffix = "_benchmark.json"
    return {p.name[: -len(suffix)] for p in od.glob(f"*{suffix}")}


def assert_summary_complete(csv_path: Path, od: Path, new_models: list[str]) -> None:
    """行数断言：summary.csv 必须覆盖 out_dir 下每一份 *_benchmark.json。

    这是「覆盖写截断」的根因修复的一部分：截断一旦发生，这里**立刻报错**，
    不再以「文件写成功了」的形式静默通过。（同一个不变量也进了 tools/selfcheck.py）
    """
    rows = read_summary(csv_path)
    expect = summary_json_models(od)
    missing = sorted(expect - set(rows))
    if missing:
        raise SystemExit(
            f"[FATAL] {csv_path.name} 只 {len(rows)} 行，缺 {len(missing)} 个已落盘模型：{missing}；"
            f"应有 {len(expect)} 行。禁止整表覆盖写——run_all.sh 是逐模型调用本脚本的。")
    lost = sorted(set(new_models) - set(rows))
    if lost:
        raise SystemExit(f"[FATAL] 本次跑的模型没有写进 {csv_path.name}：{lost}")
    print(f"[summary] {csv_path.name}: {len(rows)} 行"
          f"（覆盖 JSON {len(expect)} 个模型：{','.join(sorted(expect))}）")


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
    # 性能测试 10 项元信息里的第 5 项（推理后端）与第 10 项（输入输出节点信息）必须落盘，
    # 不能只 print：验收脚本 tools/check_report_assets.py 与报告素材表 table_benchmark_meta
    # 都从这里取值。shape/type 直接来自本会话的输入输出 ValueInfo（与实跑图一致）。
    rep = dict(model=name, onnx_path=repo_rel(Path(onnx_path(name))),
               file_size_mb=round(Path(onnx_path(name)).stat().st_size / 1e6, 2),
               providers=list(sess.get_providers()),
               provider_actual=list(sess.get_providers()),
               ep=list(sess.get_providers()),          # 与 provider_actual 同义，避免验收脚本二义
               backend="onnxruntime(" + ",".join(sess.get_providers()) + ")",
               onnx_input=dict(name=inp.name, shape=list(inp.shape), type=inp.type),
               onnx_output=dict(name=out.name, shape=list(out.shape), type=out.type),
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
    ap.add_argument("--model", action="extend", nargs="+", required=False,
                    help="registry key，可多个：--model a --model b 或 --model a b")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--image", default="outputs/samples/sample.jpg")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/benchmarks"))
    ap.add_argument("--rebuild-summary-only", action="store_true",
                    help="不跑基准：只用 out_dir 下已有的 *_benchmark.json 重建 summary.csv"
                         "（覆盖写事故后的无损恢复路径）")
    a = ap.parse_args()
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    csv_path = od / SUMMARY_NAME
    prev = read_summary(csv_path)                  # 先读旧表：合并而不是覆盖

    if a.rebuild_summary_only:
        rows = [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(od.glob("*_benchmark.json"))]
        if not rows:
            raise SystemExit(f"[FATAL] {od} 下没有 *_benchmark.json，无法重建 summary.csv")
        write_summary(csv_path, merge_summary(prev, rows))
        assert_summary_complete(csv_path, od, [r["model"] for r in rows])
        print(f"[rebuild] 未跑基准，仅由 {len(rows)} 份 JSON 重建 {csv_path}")
        return
    if not a.model:
        raise SystemExit("[FATAL] 需要 --model（或改用 --rebuild-summary-only）")

    env = env_report()
    print(json.dumps(env, ensure_ascii=True, indent=2))
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
    write_summary(csv_path, merge_summary(prev, rows))
    assert_summary_complete(csv_path, od, [r["model"] for r in rows])
    print("\n[ok] ->", csv_path, "|", jl)

if __name__ == "__main__":
    main()
