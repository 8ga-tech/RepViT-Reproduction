# tools/rv_common.py —— 全项目共享：注册表转发 / 统一预处理 / 统一计时器
import os, sys, time, json, platform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根，供 deploy.* 导入
import numpy as np, torch

# 模型注册表唯一真源是 deploy/model_registry.py（键名与字段只在那里定义：
# name/arch/impl/num_classes/labels/crop_pct/input_size/mean/std/distillation/ckpt/ckpt_url/onnx/source/note）。
# 本文件只做转发，禁止再写第二份注册表、禁止新增键名。
# 取用一律走接口：get(key) / keys() / onnx_path(key) / labels_for(key) / build_pt(key)。
# 其中 onnx_path(key) == f"onnx/{key}.onnx"，任何脚本都不得硬写别的 ONNX 文件名。
from deploy.model_registry import get, keys, onnx_path, labels_for, build_pt

WARMUP, RUNS = 10, 50   # 考核方统一复测口径

def make_transform(model=None, is_training=False, crop_pct=None):
    """预处理只允许从这里生成。crop_pct=0.95 + bicubic 是 RepViT 的 cfg，写死 Resize(256) 会整体掉点。"""
    from timm.data import resolve_data_config, create_transform
    cfg = resolve_data_config({}, model=model)
    if crop_pct is not None:
        cfg["crop_pct"] = crop_pct
    return create_transform(**cfg, is_training=is_training)

def latency(fn, warmup=WARMUP, runs=RUNS):
    """统一计时：预热丢弃 + perf_counter 正式采样 + mean/P50/P95/Max。
    n=50 时 P95 的索引是 (50-1)*0.95=46.55，即『第 3 差的那次』，必须同时报 Max。"""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(ts)
    return dict(mean=float(a.mean()), p50=float(np.percentile(a, 50, method="linear")),
                p95=float(np.percentile(a, 95, method="linear")), mx=float(a.max()),
                std=float(a.std()), n=runs, percentile_method="linear")

def env_report():
    import onnxruntime as ort
    rep = dict(os=platform.platform(), machine=platform.machine(),
               python=platform.python_version(), ort=ort.__version__,
               ort_device=ort.get_device(), available_providers=ort.get_available_providers())
    try:
        import cpuinfo  # 可选
        rep["cpu"] = cpuinfo.get_cpu_info().get("brand_raw")
    except Exception:
        rep["cpu"] = platform.processor()
    return rep

def read_list(root, listfile):
    """考核方给的 list：每行 'relpath<TAB|空格>label'。含空格的 Windows 路径必须用 rsplit(None,1)。"""
    items = []
    with open(listfile, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            p, y = ln.rsplit(None, 1)
            items.append((os.path.join(root, p), int(y)))
    ys = [y for _, y in items]
    assert min(ys) >= 0, f"标签出现负值，格式异常: min={min(ys)}"
    return items
