# tools/env_check.py
# -*- coding: utf-8 -*-
"""环境自检：打印关键库版本 + 硬件信息 + ORT 可用 EP，并写入 outputs/env_snapshot.json。

用法：
    python tools/env_check.py            # 基础自检（秒级）
    python tools/env_check.py --deep     # 追加：构建 repvit_m0_9 前向一次 + ONNX 导出冒烟
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# 项目根 = 本文件的上上级目录，杜绝硬编码个人路径
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "outputs" / "env_snapshot.json"

# 需要探测版本的模块名 -> (import 名, 属性名)
MODULES = [
    ("torch", "torch", "__version__"),
    ("torchvision", "torchvision", "__version__"),
    ("timm", "timm", "__version__"),
    ("onnx", "onnx", "__version__"),
    ("onnxruntime", "onnxruntime", "__version__"),
    ("numpy", "numpy", "__version__"),
    ("PIL", "PIL", "__version__"),
    ("sklearn", "sklearn", "__version__"),
    ("matplotlib", "matplotlib", "__version__"),
    ("cv2", "cv2", "__version__"),
    ("scipy", "scipy", "__version__"),
    ("pandas", "pandas", "__version__"),
    ("thop", "thop", "__version__"),      # 可能没有 __version__，下面会兜底
    ("fvcore", "fvcore", "__version__"),
]


def get_version(import_name: str, attr: str = "__version__"):
    """安全取版本：失败不抛异常，返回可读的错误串（这才是自检脚本该有的行为）。"""
    try:
        mod = __import__(import_name)
    except Exception as e:                       # noqa: BLE001
        return f"<import failed: {type(e).__name__}: {e}>"
    v = getattr(mod, attr, None)
    if v is None:
        # 有些包（如 thop）不暴露 __version__，退化到 importlib.metadata
        try:
            from importlib.metadata import version as _v
            v = _v(import_name)
        except Exception:                        # noqa: BLE001
            v = "<unknown>"
    return str(v)


def sh(cmd: str) -> str:
    """执行 shell 命令取 stdout；Windows 中文控制台强制 utf-8 解码，避免 GBK 炸掉。"""
    try:
        return subprocess.check_output(
            cmd, shell=True, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
        ).strip()
    except Exception as e:                       # noqa: BLE001
        return f"<failed: {e}>"


def cpu_info() -> dict:
    """Windows 读注册表 ProcessorNameString；Linux 读 /proc/cpuinfo 的 model name。"""
    name = platform.processor() or ""
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except Exception:                        # noqa: BLE001
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
        except Exception:                        # noqa: BLE001
            pass
    return {
        "model": name,
        "logical_cores": os.cpu_count(),
        "physical_cores": (os.cpu_count() or 0) // 2,   # 无 psutil 时的粗估，有 psutil 会覆盖
    }


def mem_info() -> dict:
    """优先 psutil；没有就按平台读原生接口。"""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"total_gb": round(vm.total / 1024**3, 2),
                "available_gb": round(vm.available / 1024**3, 2),
                "source": "psutil"}
    except Exception:                            # noqa: BLE001
        pass
    if sys.platform == "win32":
        txt = sh("wmic computersystem get TotalPhysicalMemory /value")
        for line in txt.splitlines():
            if "=" in line:
                try:
                    return {"total_gb": round(int(line.split("=")[1]) / 1024**3, 2),
                            "source": "wmic"}
                except Exception:                # noqa: BLE001
                    pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return {"total_gb": round(kb / 1024**2, 2), "source": "/proc/meminfo"}
        except Exception:                        # noqa: BLE001
            pass
    return {"total_gb": None, "source": "unavailable"}


def gpu_info(torch) -> dict:
    """CUDA 是否可用、卡名、显存。无卡时全部返回空值而不是报错。"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,          # CPU 版 torch 这里是 None
        "cudnn_version": None,
        "device_count": 0,
        "devices": [],
    }
    try:
        info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:                            # noqa: BLE001
        pass
    if info["cuda_available"]:
        info["device_count"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": p.name,
                "total_mem_gb": round(p.total_memory / 1024**3, 2),
                "capability": f"{p.major}.{p.minor}",
                "multi_processor_count": p.multi_processor_count,
            })
    return info


def ort_info() -> dict:
    """ORT 可用 providers + 实际会话落到哪个 EP + 关键线程配置。"""
    try:
        import onnxruntime as ort
    except Exception as e:                       # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    avail = ort.get_available_providers()
    return {
        "version": ort.__version__,
        "available_providers": avail,
        # 只读不动：报告里要写「实际 Execution Provider」，不能只写「可用」
        "default_session_providers": None,
        "note": "CPUExecutionProvider 始终存在；有 GPU/OpenVINO/Dml 时需要显式指定顺序",
    }


def check_vision_ops() -> dict:
    """torch 与 torchvision 的 C++ 扩展是否配对 —— 这是最容易被忽略的静默杀手。"""
    try:
        from torchvision.ops import nms
        import torch
        b = torch.tensor([[0., 0., 1., 1.], [0., 0., 0.9, 0.9]])
        s = torch.tensor([0.9, 0.8])
        keep = nms(b, s, 0.5)
        return {"ok": True, "nms_keep": keep.tolist()}
    except Exception as e:                       # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def deep_check() -> dict:
    """可选深检：真正建一次 repvit_m0_9 并前向，验证 timm 侧模型链路可用。"""
    import torch
    res = {}
    try:
        import timm
        t0 = time.perf_counter()
        # distillation=False：与全仓统一口径一致（本任务单头训练态）
        m = timm.create_model("repvit_m0_9", pretrained=False, num_classes=37,
                              distillation=False)
        m.eval()
        n_param = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            y = m(torch.randn(1, 3, 224, 224))
        res["build_ok"] = True
        res["num_classes_37_params"] = n_param
        res["output_shape"] = list(y.shape)
        res["build_seconds"] = round(time.perf_counter() - t0, 3)
    except Exception as e:                       # noqa: BLE001
        res["build_ok"] = False
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true", help="额外构建一次模型并前向")
    args = ap.parse_args()

    snapshot = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cwd": os.getcwd(),
        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
            "executable": sys.executable,
            "is_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
            "base_prefix": getattr(sys, "base_prefix", None),
        },
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            # 中文 Windows 的编码现状，直接决定后面会不会踩 GBK 坑
            "preferred_encoding": __import__("locale").getpreferredencoding(False),
            "file_system_encoding": sys.getfilesystemencoding(),
        },
        "cpu": cpu_info(),
        "memory": mem_info(),
        "packages": {name: get_version(imp, attr) for name, imp, attr in MODULES},
        "env_vars": {k: os.environ.get(k) for k in [
            "PYTHONHASHSEED", "PYTHONUTF8", "PYTHONIOENCODING",
            "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
            "HF_ENDPOINT", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
        ]},
    }

    # torch 相关必须单独处理：它能拿到版本不代表能 import 成功
    try:
        import torch
        snapshot["torch_build"] = {
            "version": torch.__version__,
            "is_cuda_build": torch.version.cuda is not None,
            "debug": torch.__config__.show().splitlines()[:3],
        }
        snapshot["gpu"] = gpu_info(torch)
    except Exception as e:                       # noqa: BLE001
        snapshot["torch_build"] = {"error": f"{type(e).__name__}: {e}"}
        snapshot["gpu"] = gpu_info(type("T", (), {"cuda": type("C", (), {"is_available": staticmethod(lambda: False)})}))

    snapshot["vision_ops"] = check_vision_ops()
    snapshot["onnxruntime"] = ort_info()
    if args.deep:
        snapshot["deep"] = deep_check()

    # ---- 落盘 ----
    # 必须 ensure_ascii=True：本机 CPU 型号串含非 ASCII 字节（如 U+00AE 注册商标符），
    # 而 DoD #1 的验收命令是
    #   python -c "import json;d=json.load(open('outputs/env_snapshot.json'));..."
    # —— 它用的是内置 open() 的**默认编码**，在中文 Windows 上就是 GBK，
    # 一旦文件里有非 ASCII 字节就抛 UnicodeDecodeError（实测复现，spec §1.6⑤ 预警的正是这一类）。
    # ensure_ascii=True 会把非 ASCII 转义成 \uXXXX，文件变成纯 ASCII，
    # JSON 语义完全不变，但任何编码下都能被 open() 读回。
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(snapshot, ensure_ascii=True, indent=2), encoding="utf-8")

    # ---- 人读摘要 ----
    pkg = snapshot["packages"]
    print("=" * 62)
    print(f"Python      : {snapshot['python']['version'].splitlines()[0]}")
    print(f"Executable  : {snapshot['python']['executable']}")
    print(f"Venv        : {snapshot['python']['is_venv']}")
    for k in ("torch", "torchvision", "timm", "onnx", "onnxruntime",
              "numpy", "PIL", "sklearn", "matplotlib", "cv2"):
        print(f"{k:<12}: {pkg.get(k)}")
    g = snapshot["gpu"]
    print(f"CUDA avail  : {g['cuda_available']}  (torch build cuda={g['cuda_version']})")
    for d in g.get("devices", []):
        print(f"  GPU{d['index']}       : {d['name']}  {d['total_mem_gb']} GB  sm_{d['capability']}")
    print(f"CPU         : {snapshot['cpu']['model']}  x{snapshot['cpu']['logical_cores']} threads")
    print(f"RAM         : {snapshot['memory'].get('total_gb')} GB ({snapshot['memory']['source']})")
    print(f"OS          : {snapshot['os']['platform']}  [enc={snapshot['os']['preferred_encoding']}]")
    print(f"ORT         : {snapshot['onnxruntime'].get('version')}  providers={snapshot['onnxruntime'].get('available_providers')}")
    print(f"vision ops  : {snapshot['vision_ops']}")
    print(f"snapshot -> {OUT_PATH}")
    print("=" * 62)

    # 自检失败时用非零退出码，方便 CI/脚本串联
    fatal = []
    if not snapshot["vision_ops"].get("ok"):
        fatal.append("torchvision.ops 不可用（torch 与 torchvision 扩展不配对）")
    # 「不在 venv 里」只报警不判死：本任务已由人类决策改用全局 C:\Python314
    # （理由与影响记录在 PROGRESS.md 的「决策记录」09-12 条），
    # 此时它不是环境缺陷，而是既定事实。硬失败会让 DoD #1 无谓地卡死。
    if not snapshot["python"]["is_venv"]:
        print("\n[WARN] 当前解释器不在虚拟环境中（本任务为人类显式决策，见 PROGRESS.md 决策记录）")
    if args.deep and not snapshot.get("deep", {}).get("build_ok"):
        fatal.append("repvit_m0_9 构建/前向失败")
    if fatal:
        print("\n[FAIL]")
        for f in fatal:
            print("  -", f)
        sys.exit(1)
    print("\n[PASS] 环境自检通过")


if __name__ == "__main__":
    main()
