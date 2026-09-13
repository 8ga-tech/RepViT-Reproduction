# tools/bench_igpu.py —— 集显部署：ORT-OpenVINO EP / OpenVINO 原生 / DirectML 三路 +
#                        算子回退检查 + CPU vs GPU 的 mean/P50/P95 对比表
import argparse, json, os, platform, time
import numpy as np
try:
    from tools.rv_common import onnx_path, WARMUP, RUNS, env_report
except ImportError:      # `python tools/X.py` 时 sys.path[0]=tools/
    from rv_common import onnx_path, WARMUP, RUNS, env_report

def _np_infer(sess, x):
    return sess.run(None, {sess.get_inputs()[0].name: x})

# ---------------- 路径 1：onnxruntime + OpenVINOExecutionProvider ----------------
def session_ort_openvino(path, device_type="GPU", threads=4, ov_precision="f32"):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    # 官方建议：给 OpenVINO EP 传原始图并关掉 ORT 自己的图优化，让 OpenVINO 自己优化
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts = {"device_type": device_type,                    # CPU/NPU/GPU/GPU.0/GPU.1/HETERO/MULTI/AUTO
            "load_config": json.dumps({"GPU": {"PERFORMANCE_HINT": "LATENCY",
                                               "INFERENCE_PRECISION_HINT": ov_precision}})}
    return ort.InferenceSession(path, so, providers=[("OpenVINOExecutionProvider", opts)])

# ---------------- 路径 2：openvino 原生 API（Python 3.14 唯一可用路径） ----------------
def bench_openvino_native(path, device="GPU", threads=None, hint="LATENCY", warmup=WARMUP, runs=RUNS):
    import openvino as ov
    core = ov.Core()
    if threads:
        core.set_property("CPU", {"INFERENCE_NUM_THREADS": int(threads)})
    cm = core.compile_model(path, device, {"PERFORMANCE_HINT": hint})
    exec_dev = list(cm.get_property("EXECUTION_DEVICES"))   # 关键：OpenVINO 自报的真实执行设备
    inp = cm.input(0); name = inp.get_any_name()
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in list(inp.shape)]
    x = np.random.randn(*shape).astype(np.float32)
    req = cm.create_infer_request()
    ts = _timeit(lambda: req.infer({name: x}), warmup, runs)
    ts.update(model=os.path.basename(path), backend=f"openvino:{device}",
              requested_device=device, exec_devices=exec_dev,
              device_full_name=core.get_property("GPU", "FULL_DEVICE_NAME")
              if "GPU" in core.available_devices else None,
              device_type=str(core.get_property("GPU", "DEVICE_TYPE"))
              if "GPU" in core.available_devices else None,
              warmup=warmup, runs=runs, batch=1, precision="FP32")
    return ts

def _timeit(fn, warmup, runs):
    for _ in range(warmup): fn()
    a = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); a.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(a)
    return dict(mean=float(a.mean()), p50=float(np.percentile(a, 50, method="linear")),
                p95=float(np.percentile(a, 95, method="linear")), mx=float(a.max()),
                std=float(a.std()), n=runs, percentile_method="linear")

# ---------------- 路径 3：onnxruntime-directml ----------------
def session_dml(path, device_id=0, threads=4):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.enable_mem_pattern = False          # DML 官方要求：不支持 memory pattern 优化
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL   # DML 不支持并行执行
    return ort.InferenceSession(path, so, providers=[
        ("DmlExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"])

def session_cpu(path, threads=4):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

# ---------------- 算子回退检查：关掉 CPU fallback，不崩即全图在目标 EP ----------------
def check_op_fallback(path, backend, device_id=0, device_type="GPU", threads=4):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    # 注意：该配置项与『显式带上 CPUExecutionProvider』互斥，同时设置 session 创建直接失败
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    prov = ([("DmlExecutionProvider", {"device_id": device_id})] if backend == "dml"
            else [("OpenVINOExecutionProvider", {"device_type": device_type})])
    try:
        s = ort.InferenceSession(path, so, providers=prov)
        # 逐节点分配（需要 verbose 日志级别）
        ort.set_default_logger_severity(0)
        x = np.random.randn(1, 3, 224, 224).astype(np.float32)
        s.run(None, {s.get_inputs()[0].name: x})
        return dict(fallback_disabled_ok=True, providers=s.get_providers(),
                    note="未抛异常 => 整张图都在目标 EP 上")
    except Exception as e:
        return dict(fallback_disabled_ok=False, error=f"{type(e).__name__}: {str(e)[:200]}",
                    note="存在节点无法在目标 EP 执行，这就是回退证据（配合 verbose 日志定位）")
    finally:
        ort.set_default_logger_severity(2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["repvit_m0_9_in1k", "repvit_m1_0_in1k"])
    ap.add_argument("--backend", choices=["cpu", "ort_ov", "ov_native", "dml"], default="ov_native")
    ap.add_argument("--device", default="GPU", help="OpenVINO: CPU/GPU/GPU.0/GPU.1/AUTO:GPU,CPU")
    ap.add_argument("--device-id", type=int, default=0, help="DirectML device_id")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="outputs/benchmarks/igpu_bench.json")
    a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    rows = []
    for key in a.model:
        path = onnx_path(key)          # onnx/<registry_key>.onnx
        if not os.path.exists(path):
            print("skip (missing onnx):", path); continue
        try:
            if a.backend == "cpu":
                s = session_cpu(path, a.threads); r = _timeit(lambda: _np_infer(s, x), WARMUP, RUNS)
                r.update(requested_ep="CPUExecutionProvider", actual_providers=s.get_providers(),
                         provider_options=s.get_provider_options())
            elif a.backend == "dml":
                s = session_dml(path, a.device_id, a.threads)
                r = _timeit(lambda: _np_infer(s, x), WARMUP, RUNS)
                r.update(requested_ep="DmlExecutionProvider", actual_providers=s.get_providers(),
                         provider_options=s.get_provider_options())
                # 静默回退检测：请求 DML 却拿到 CPU，这条数据绝不能标成 GPU 结果
                if s.get_providers()[0] != "DmlExecutionProvider":
                    print(f"!! {key} 发生 EP 回退 -> {s.get_providers()}，该行标记为 invalid")
                    r["invalid"] = True
            elif a.backend == "ort_ov":
                s = session_ort_openvino(path, a.device, a.threads)
                r = _timeit(lambda: _np_infer(s, x), WARMUP, RUNS)
                r.update(requested_ep="OpenVINOExecutionProvider",
                         actual_providers=s.get_providers(),
                         provider_options=s.get_provider_options())
            else:
                r = bench_openvino_native(path, a.device, a.threads)
            r.update(model=key, onnx=path, backend=a.backend, threads=a.threads,
                     file_MB=round(os.path.getsize(path) / 1e6, 3), batch=1,
                     precision="FP32", warmup=WARMUP, runs=RUNS, input="1x3x224x224")
            rows.append(r)
            print(f"{key:22s} {a.backend:10s} dev={a.device:12s} "
                  f"mean={r['mean']:.3f} P50={r['p50']:.3f} P95={r['p95']:.3f} ms")
        except Exception as e:
            print(f"[FAIL] {key} @ {a.backend}: {type(e).__name__}: {str(e)[:200]}")
    payload = dict(env=env_report(), backend=a.backend, requested_device=a.device,
                   threads=a.threads, results=rows)
    if a.backend in ("dml", "ort_ov"):
        payload["fallback_check"] = check_op_fallback(
            onnx_path(a.model[0]), a.backend, a.device_id, a.device, a.threads)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    print("saved", a.out)

if __name__ == "__main__":
    main()
