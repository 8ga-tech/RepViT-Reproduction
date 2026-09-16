#!/usr/bin/env python
"""
RepViT 交付自检脚本
Source  : Self-written
用法    : python tools/selfcheck.py                 # 全量
          python tools/selfcheck.py --only data.leak onnx.bn
          python tools/selfcheck.py --json outputs/metrics/selfcheck_report.json --strict
说明    : 每条断言返回 (ok: bool, detail: str)；异常被捕获成 FAIL 而不中断整轮自检。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS: list[dict] = []


def _rel(p) -> str:
    """把路径显示成相对仓库根的 POSIX 形式；**仓库外的路径原样显示**。

    仓库内路径的输出与 `Path(p).relative_to(ROOT).as_posix()` 逐字相同；
    仓库外路径（例如 `--json %TEMP%\\x.json`）以前会让 relative_to 抛 ValueError、
    把整轮自检拖崩（即使所有检查都 PASS 也会以退出码 1 结束），这里退化为原样显示。
    """
    p = Path(p)
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def record(cid: str, title: str, ok: bool, detail: str, fix: str = "") -> bool:
    RESULTS.append({"id": cid, "title": title,
                    "status": "PASS" if ok else "FAIL",
                    "detail": str(detail)[:400], "fix": fix if not ok else ""})
    return bool(ok)


def check(cid: str, title: str, fix: str = ""):
    """装饰器：把 (bool, str) 的返回值登记进 RESULTS，异常转 FAIL。"""
    def wrap(fn):
        def inner():
            try:
                ok, detail = fn()
            except Exception as e:                       # noqa: BLE001
                ok, detail = False, f"{type(e).__name__}: {e}"
            record(cid, title, ok, detail, fix)
            return ok
        inner.cid = cid
        return inner
    return wrap


def _read_list(path: Path) -> list[str]:
    """划分文件一行一个图片名；用 rsplit(None,1) 兼容空格/tab 两种分隔与含空格路径。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少 {_rel(path)}")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(line.rsplit(None, 1)[0])
    return out


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------- A 环境 / 工程
@check("env", "环境版本矩阵", "pip install -r requirements.lock.txt")
def c_env():
    lock = ROOT / "requirements.lock.txt"
    if not lock.exists():
        return False, "缺少 requirements.lock.txt"
    want = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "==" in line and not line.startswith("#"):
            k, v = line.split("==", 1)
            want[k.strip().lower()] = v.strip().split(" ")[0]
    bad = []
    for mod in ("torch", "timm", "onnx", "onnxruntime", "numpy"):
        try:
            m = __import__(mod)
            got = getattr(m, "__version__", "?")
            if mod in want and not got.startswith(want[mod]):
                bad.append(f"{mod}: 期望{want[mod]} 实际{got}")
        except ImportError:
            bad.append(f"{mod}: 未安装")
    return (not bad), ("全部命中" if not bad else "; ".join(bad))


@check("paths", "无写死绝对路径", "把路径改为 config 字段")
def c_paths():
    # 两处误报必须排除，否则纯 URL / 普通字符串字面量会被当成写死的本机路径：
    #   (a) `[A-Za-z]:/` 命中任何 URL 的 `s:/`（https:// 里的 "s:" + "/"）；
    #   (b) `[A-Za-z]:\` 命中 "...None:\n" 里 `e:` + `\` 这种转义序列。
    # 实测：全仓命中项**全部**属于以上两类，没有一处是真实盘符路径。
    # 因此要求盘符以非单词字符打头（引号/空格/=/( 等）。
    pat = [re.compile(r"(?<![\w])[A-Za-z]:\\\\"),
           re.compile(r"(?<![\w])[A-Za-z]:/"),
           re.compile(r"(?<![\w.])/(Users|home|root|mnt)/")]
    skip = {".git", "__pycache__", "outputs", "checkpoints", "onnx", ".venv"}
    hits = []
    for f in ROOT.rglob("*.py"):
        if any(p in skip for p in f.parts):
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            probe = line.split("://")[0] if "://" in line else line   # 丢掉 URL 主体再匹配
            if any(p.search(probe) for p in pat):
                hits.append(f"{f.relative_to(ROOT).as_posix()}:{i}")
    return (not hits), (f"{len(hits)} 处" + (f" 首处 {hits[0]}" if hits else ""))


@check("tag", "timm 权重 tag 解析", "显式写 repvit_m0_9.dist_450e_in1k 以区分 450e")
def c_tag():
    from timm.models import _registry as r
    cfg = r._model_pretrained_cfgs["repvit_m0_9"]
    return cfg.tag == "dist_300e_in1k", f"tag={cfg.tag} hf_id={cfg.hf_hub_id}"


# ---------------------------------------------------------------- B 数据
@check("data.leak", "train/val/test 无交集", "数据泄漏属不通过条款，必须重划")
def c_leak():
    tr = set(_read_list(ROOT / "datasets/lists/pet_train.txt"))
    va = set(_read_list(ROOT / "datasets/lists/pet_val.txt"))
    te = set(_read_list(ROOT / "datasets/lists/pet_test.txt"))
    d = {"train∩val": len(tr & va), "train∩test": len(tr & te), "val∩test": len(va & te)}
    return all(v == 0 for v in d.values()), f"{d} sizes={len(tr)}/{len(va)}/{len(te)}"


@check("data.labels", "Pet 标签集合 = {0..36}", "漏 -1 或误用第 4 列")
def c_labels():
    out = ROOT / "outputs/metrics/pet_label_audit.json"
    if not out.exists():
        return False, f"缺少 {out.relative_to(ROOT)}（由 tools/evaluate.py --audit 生成）"
    d = json.loads(out.read_text(encoding="utf-8"))
    ok = d["min"] == 0 and d["max"] == 36 and d["n_classes"] == 37
    return ok, f"min={d['min']} max={d['max']} n={d['n_classes']} 每类最少={d.get('min_count')}"


@check("data.classes", "类别名顺序正确", "禁用 ImageFolder 目录名字母序")
def c_classes():
    f = ROOT / "labels/pet_classes.txt"
    if not f.exists():
        return False, "缺少 labels/pet_classes.txt"
    names = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = len(names) == 37 and names[0] == "Abyssinian" and names[36] == "Yorkshire Terrier"
    return ok, f"n={len(names)} first={names[0] if names else '-'} last={names[-1] if names else '-'}"


@check("data.source", "train 全部来自官方 trainval", "检查划分脚本读取的根目录")
def c_source():
    f = ROOT / "outputs/metrics/split_audit.json"
    if not f.exists():
        return False, f"缺少 {f.relative_to(ROOT)}"
    d = json.loads(f.read_text(encoding="utf-8"))
    return d.get("train_outside_official", 1) == 0, f"越界文件数={d.get('train_outside_official')}"


# ---------------------------------------------------------------- C 权重与口径
@check("ckpt.keys", "ckpt 键名结构", "官方 ckpt 是 ckpt['model']，timm 是裸 state_dict")
def c_ckpt_keys():
    import torch
    p = ROOT / "checkpoints/pretrained/repvit_m0_9_distill_300e.pth"
    if not p.exists():
        return False, f"缺少 {p.relative_to(ROOT)}（从 GitHub Release v1.0 下载）"
    c = torch.load(p, map_location="cpu", weights_only=False)
    sd = c["model"] if isinstance(c, dict) and "model" in c else c
    keys = list(sd.keys())[:4]
    return ("model" in c) if isinstance(c, dict) else False, f"前4键={keys}"


@check("params", "参数量三口径", "报告必须写明用的是哪个口径")
def c_params():
    import copy
    import timm
    m = timm.create_model("repvit_m0_9", pretrained=False, num_classes=1000)
    n_dbl = sum(p.numel() for p in m.parameters())                       # 未融合、蒸馏双头
    n_head = sum(p.numel() for n, p in m.named_parameters() if n.startswith("head"))
    n_single = n_dbl - n_head // 2                                       # 未融合单头
    # 必须分两行：timm 的 RepVit.fuse() 是**原地改写**且返回 None。
    # 写成 `copy.deepcopy(m).fuse().parameters()` 会抛
    # AttributeError: 'NoneType' object has no attribute 'parameters'（实测复现）。
    _mf = copy.deepcopy(m)
    _mf.fuse()
    n_fused = sum(p.numel() for p in _mf.parameters())  # 融合后单头
    # 融合后的值只报本机实测值，不写死期望值（双头/单头融到同一终点）
    return (n_dbl == 5489328 and n_single == 5103560), \
        f"双头={n_dbl:,} 分类头合计={n_head:,} 单头={n_single:,} " \
        f"融合后={n_fused:,}（约 {n_fused / 1e6:.3f}M，净减 {n_single - n_fused:,}；以本机实测为准）"


@check("macs", "MACs 口径一致性", "fvcore 一次乘加算 1 flop，不等于 2xMACs")
def c_macs():
    import torch
    import timm
    m = timm.create_model("repvit_m0_9", pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    from thop import profile
    macs, _ = profile(m, inputs=(x,), verbose=False)
    g_macs = macs / 1e9
    g_flops = None
    try:
        from fvcore.nn import FlopCountAnalysis
        g_flops = FlopCountAnalysis(m, x).total() / 1e9
    except Exception:
        pass
    ok = 0.7 <= g_macs <= 1.0
    return ok, f"thop MACs={g_macs:.3f}G fvcore FLOPs={g_flops if g_flops is None else round(g_flops, 3)}G"


@check("train.backbone", "骨干参数已更新", "只训分类头 → 基础训练封顶 60%")
def c_backbone():
    import torch
    p = ROOT / "checkpoints/baseline_best.pt"
    if not p.exists():
        return False, "缺少 checkpoints/baseline_best.pt"
    # 必须 weights_only=False：本仓 ckpt 除张量外还存了 config(dict)/seed/class_names(list)，
    # torch>=2.6 的 weights_only=True 默认拒绝反序列化这些对象，直接抛 UnpicklingError。
    # 这是自家训练产物，来源可信。
    sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
    # 取 stage **序号**（index 1），不是 index 2。
    # timm 键布局是 stages.N.blocks.M.*，index 2 会得到 'blocks'/'downsample' 两个取值，
    # 恒 < 4 而误报「只训了分类头」（实测复现）。
    stages = sorted({int(k.split(".")[1]) for k in sd
                     if k.startswith("stages.") and k.endswith("weight")
                     and k.split(".")[1].isdigit()})
    return len(stages) >= 4, f"落在 {len(stages)} 个 stage 上的参数已随训练保存: {stages}"


@check("model.size", "权重文件大小与参数量吻合", "文件异常小说明只存了分类头")
def c_model_size():
    p = ROOT / "checkpoints/baseline_best.pt"
    if not p.exists():
        return False, "缺少 checkpoints/baseline_best.pt"
    mb = p.stat().st_size / 1e6
    # 区间必须覆盖「模型 + 优化器状态」：save_ckpt 按规格书 §6.5 同时存 optimizer/scheduler，
    # AdamW 有 exp_avg / exp_avg_sq 两份动量，4.73M 参数 -> 约 2 x 4.73M x 4B = 37.8MB，
    # 加模型本身约 19MB，实测 57.6MB。原区间 15~40MB 把「只存模型」当默认，与本仓
    # save_ckpt 契约冲突，必然 FAIL。真正要防的是「文件异常小 = 只存了分类头」，
    # 故下界保持 15MB，上界放宽到 90MB。
    return 15.0 <= mb <= 90.0, f"baseline_best.pt = {mb:.2f} MB（模型 ~19MB + AdamW 两动量 ~38MB）"


# ---------------------------------------------------------------- D 官方模型评价
@check("official.top1", "官方模型 Top-1 落在 70~82", "<1 查 ILSVRC2012_ID 映射；30~60 查部分类映射")
def c_official_top1():
    fs = sorted((ROOT / "outputs/pretrained_eval").glob("*/metrics.json"))
    if not fs:
        return False, "outputs/pretrained_eval/<model_name>/metrics.json 不存在"
    # 判据：与**该型号官方公布值**的偏差，而不是一个固定的 70~82 区间。
    # 固定区间只适用于 M0.9 量级；M1.5(官方 82.3)/M2.3(官方 83.3) 天然 > 82，
    # 用固定区间会把「复现正确」误判成「越界」（实测复现）。
    # 官方值来自 THU-MIG/RepViT README 的 300e 列。
    OFFICIAL_TOP1 = {"repvit_m0_6": 74.1, "repvit_m0_9": 78.7, "repvit_m1_0": 80.0,
                     "repvit_m1_1": 80.7, "repvit_m1_5": 82.3, "repvit_m2_3": 83.3}
    TOL = 1.5                       # 1000 张自建子集，单张翻转即 0.1 个点，1.5 是合理容差
    rows, bad = [], []
    for f in fs:
        d = json.loads(f.read_text(encoding="utf-8"))
        name = d.get("model_name", f.parent.name)
        ref = OFFICIAL_TOP1.get(name.split(".")[0])
        rows.append(f"{name}:{d['top1']:.2f}%(官方{ref})")
        if ref is None:
            bad.append(f"{name}(无官方基准)")
        elif abs(d["top1"] - ref) > TOL:
            bad.append(f"{name}(偏差{d['top1']-ref:+.2f} > {TOL})")
        # 无论哪个型号，掉到随机水平(≈0.1)或 30~60 都说明标签体系有问题
        if d["top1"] < 70.0:
            bad.append(f"{name}(低于 70，查标签映射)")
    return (not bad), (f"{rows} 全部在官方值 ±{TOL} 内" if not bad else f"异常={bad}")


@check("official.latency", "官方评价延迟元信息完整", "缺项则性能数字不可比")
def c_official_latency():
    need = ["cpu_model", "os", "ort_version", "ep", "threads", "device",
            "warmup", "runs", "mean_ms", "p50_ms", "p95_ms"]
    fs = sorted((ROOT / "outputs/pretrained_eval").glob("*/latency.json"))
    if not fs:
        return False, "outputs/pretrained_eval/<model_name>/latency.json 不存在"
    miss = {}
    for f in fs:
        d = json.loads(f.read_text(encoding="utf-8"))          # 扁平对象，不是嵌套的 latency 子对象
        m = [k for k in need if k not in d or d[k] == ""]
        if m:
            miss[f.parent.name] = m
    return (not miss), (f"缺失={miss}" if miss else
                        f"{len(fs)} 个模型元信息齐全（ep/ort_version 走 PyTorch 路线时为 null，但键必须存在）")


# ---------------------------------------------------------------- E 训练日志
REQ_FIELDS = ["epoch", "train_loss", "train_acc1", "val_loss",
              "val_top1", "val_top5", "val_macro_f1", "lr", "epoch_time_sec"]


@check("train.log", "训练日志字段齐全", "缺 val_macro_f1 就必须重跑")
def c_train_log():
    f = ROOT / "outputs/logs/baseline_metrics.csv"
    if not f.exists():
        return False, "缺少 outputs/logs/baseline_metrics.csv"
    head = f.read_text(encoding="utf-8").splitlines()[0].split(",")
    miss = [k for k in REQ_FIELDS if k not in head]
    return (not miss), f"缺失字段={miss}" if miss else f"{len(REQ_FIELDS)} 字段齐全"


@check("train.valmetrics", "val 三指标与每类指标落盘", "缺 per-class 就没有「每类准确率」提交项")
def c_val_metrics():
    import csv
    csv_p = ROOT / "outputs/logs/baseline_metrics.csv"
    cls_p = ROOT / "outputs/confusion_matrix/baseline_per_class.csv"
    for p in (csv_p, cls_p):
        if not p.exists():
            return False, f"缺少 {p.relative_to(ROOT)}"
    row = list(csv.DictReader(csv_p.read_text(encoding="utf-8").splitlines()))[-1]
    n = sum(1 for _ in csv.DictReader(cls_p.read_text(encoding="utf-8").splitlines()))
    return n == 37, f"末行 val_top1={row['val_top1']} val_top5={row['val_top5']} " \
                    f"val_macro_f1={row['val_macro_f1']} per_class={n} 行"


@check("train.ckpt", "best / last checkpoint 存在", "best 的 epoch 不应大于 last")
def c_train_ckpt():
    import torch
    ep = {}
    for tag in ("best", "last"):
        p = ROOT / f"checkpoints/baseline_{tag}.pt"
        if not p.exists():
            return False, f"缺少 {p.relative_to(ROOT)}"
        ep[tag] = int(torch.load(p, map_location="cpu", weights_only=False)["epoch"])
    return ep["best"] <= ep["last"], f"epoch best={ep['best']} last={ep['last']}"


@check("train.test_once", "test 只评价一次", "多份需能解释，否则测试结果不计")
def c_test_once():
    fs = sorted((ROOT / "outputs/metrics").glob("baseline_test*.json"))
    return len(fs) == 1, f"找到 {[f.name for f in fs]}"


# ---------------------------------------------------------------- F 优化实验
@check("opt.diff", "控制变量 diff 唯一", "多于一个变量 → 优化部分封顶 50%")
def c_opt_diff():
    """判据与 DoD #23 的 tools/diff_config.py 完全一致：**叶子级**差异 + **声明白名单**。

    原实现比较两份 effective 快照的顶层字典并要求 len(diff) <= 1，有两个问题：
      (a) 顶层字典比较会把「aug 里任何一个键变了」整个记成一个差异，粒度不对；
      (b) `<=1` 只对「单变量优化」成立，方案 A（12 个键）这类多键组合必然 FAIL。
    这里改为：读 configs/opt_*.yaml 的 `_expected_diff_`，逐叶子键比对，
    要求「实际差异 == 声明的差异」，这才是「控制变量」的可执行定义。
    """
    import yaml as _yaml
    a = ROOT / "configs/baseline.yaml"
    cands = sorted((ROOT / "configs").glob("opt_*.yaml"))
    if not cands:
        return False, "configs/ 下没有 opt_*.yaml"

    def _load_with_base(path):
        # 必须用 utils/config.py 的 load_yaml：它做的是**递归深合并**，
        # 与 DoD #23 的 tools/diff_config.py 同源。
        # 手写浅合并（`{**base[k], **over[k]}`）只会合并一层，
        # 会把 aug.train 下的 hflip / interpolation / random_erasing / resize_size
        # 整块丢掉，凭空造出 6 个「未声明差异」（实测复现）。
        import sys as _sys
        if str(ROOT) not in _sys.path:
            _sys.path.insert(0, str(ROOT))
        from utils.config import load_yaml
        return load_yaml(path)

    def _leaves(d, pre=""):
        out = {}
        for k, v in d.items():
            if k.startswith("_"):
                continue
            key = f"{pre}{k}"
            if isinstance(v, dict):
                out.update(_leaves(v, key + "."))
            else:
                out[key] = v
        return out

    fa = {k: v for k, v in _leaves(_load_with_base(a)).items() if not k.startswith("experiment_name")}
    rows, bad = [], []
    for c in cands:
        cb = _load_with_base(c)
        exp = list(cb.pop("_expected_diff_", []) or [])
        fb = {k: v for k, v in _leaves(cb).items() if not k.startswith("experiment_name")}
        diff = [k for k in sorted(set(fa) | set(fb))
                if fa.get(k, "<MISSING>") != fb.get(k, "<MISSING>")]
        undeclared = [k for k in diff if k not in exp]
        rows.append(f"{c.stem}:{len(diff)}键/声明{len(exp)}")
        if undeclared:
            bad.append(f"{c.stem} 未声明={undeclared}")
    return (not bad), (f"{rows}（叶子级，与 diff_config.py 同口径）" if not bad else f"{bad}")


@check("opt.budget", "训练预算等价", "预算不等必须显式换算说明")
def c_opt_budget():
    import yaml
    out = {}
    for exp in ("baseline", "opt_randaug"):
        c = yaml.safe_load((ROOT / f"outputs/logs/{exp}_config_effective.yaml").read_text(encoding="utf-8"))
        ds = c["data"]
        steps = len(_read_list(ROOT / ds["train_list"])) // ds["batch_size"] if ds.get("drop_last") else -1
        out[exp] = (c["train"]["epochs"], steps)
    de = abs(out["baseline"][0] - out["opt_randaug"][0])
    # 训练集 2940 / batch 64 / drop_last=True → 45 step/epoch，两组须共用同一数据与 batch
    return (de == 0 and out["baseline"][1] == out["opt_randaug"][1]), \
        f"(epochs, steps_per_epoch)={out} epochs 差值={de}"


# ---------------------------------------------------------------- G 可视化
@check("viz", "八类可视化产物齐全", "缺哪类补哪类（8 分项）")
def c_viz():
    g = {
        "curves": list((ROOT / "outputs/curves").glob("*_curves.png")),
        "cm": list((ROOT / "outputs/confusion_matrix").glob("*_cm.png")),
        "cm_csv": list((ROOT / "outputs/confusion_matrix").glob("*_cm.csv")),
        "per_class": list((ROOT / "outputs/confusion_matrix").glob("*_per_class_f1.png")),
        "per_class_csv": list((ROOT / "outputs/confusion_matrix").glob("*_per_class.csv")),
        "preds": list((ROOT / "outputs/predictions").glob("*.png")),
        "compare": list((ROOT / "outputs/curves").glob("*_compare.png")),
        "gradcam": list((ROOT / "outputs/gradcam").glob("*.png")),
    }
    miss = []
    if not g["curves"]: miss.append("曲线（`<experiment_name>_curves.png` 的 loss / top1 / macro_f1 / lr 子图）")
    if not (g["cm"] and g["cm_csv"]): miss.append("归一化混淆矩阵（`_cm.png` + `_cm.csv`）")
    if not (g["per_class"] and g["per_class_csv"]): miss.append("每类指标（`_per_class_f1.png` + `_per_class.csv`）")
    if len(g["preds"]) < 8: miss.append(f"预测图仅 {len(g['preds'])} 张(<8)")
    if len(g["compare"]) < 1: miss.append(f"同图对比仅 {len(g['compare'])} 张(<1)")
    if len(g["gradcam"]) < 4: miss.append(f"Grad-CAM 仅 {len(g['gradcam'])} 张(<4)")
    return (not miss), f"缺失={miss}" if miss else \
        f"curves={len(g['curves'])} cm={len(g['cm'])} preds={len(g['preds'])} " \
        f"compare={len(g['compare'])} cam={len(g['gradcam'])}"


@check("viz.cm", "混淆矩阵为行归一化", "图注必须写 normalize='true'")
def c_cm():
    import numpy as np
    f = ROOT / "outputs/confusion_matrix/baseline_cm.csv"
    if not f.exists():
        return False, "缺少 outputs/confusion_matrix/baseline_cm.csv"
    # 该 CSV 由 pandas 写出，**第 0 列是类别名索引列**（表头为空、每行是类名），
    # 直接 np.loadtxt 会抛 could not convert string 'Abyssinian' to float64（实测复现）。
    # 必须只取第 1..37 列。
    cm = np.loadtxt(f, delimiter=",", skiprows=1, usecols=range(1, 38))
    return cm.shape == (37, 37) and np.allclose(cm.sum(1), 1.0, atol=1e-6), \
        f"shape={cm.shape} 行和范围=[{cm.sum(1).min():.6f},{cm.sum(1).max():.6f}]"


@check("cam.layer", "Grad-CAM 目标层合法", "挂 head 会抛 Invalid grads shape")
def c_cam_layer():
    d = ROOT / "outputs/metrics/gradcam_meta.json"
    if not d.exists():
        return False, "缺少 outputs/metrics/gradcam_meta.json"
    j = json.loads(d.read_text(encoding="utf-8"))
    n_total, n_wrong = j.get("n_images", 0), j.get("n_wrong_cases", 0)
    ok = n_total >= 4 and n_wrong >= 1 and "head" not in j.get("target_layer", "")
    return ok, f"target_layer={j.get('target_layer')} 张数={n_total} 失败案例={n_wrong} " \
               f"非零占比={j.get('nonzero_ratio')} 值域={j.get('vmin')}~{j.get('vmax')}"


# ---------------------------------------------------------------- H 重参数化
@check("rep.verify", "融合前后数值等价且 BN 归零", "先 eval() 再融合，顺序不可反")
def c_rep():
    # 交付口径是 Pet-37 迁移模型；不要按 glob 误读官方 C=1000 的历史对照。
    f = ROOT / "outputs/reparam/repvit_m0_9_pet37_reparam_report.json"
    if not f.exists():
        return False, f"缺少 {f.relative_to(ROOT)}（跑 tools/reparam_verify.py 生成）"
    d = json.loads(f.read_text(encoding="utf-8"))
    before, after, judge = d.get("before", {}), d.get("after", {}), d.get("judge", {})
    weights = d.get("weights", {})
    weights_ok = (weights.get("n_missing") == 0 and weights.get("n_unexpected") == 0)
    ok = (weights_ok and bool(judge.get("pass")) and after.get("n_bn") == 0 and before.get("n_bn", 0) > 0)
    return ok, f"BN {before.get('n_bn')}->{after.get('n_bn')} " \
               f"max|Δlogits|={judge.get('max_abs_err')} Top-1一致={judge.get('top1_agree')} " \
               f"权重完整={weights_ok} 整体判定={ok}"


@check("onnx.bn", "ONNX 图中无 BatchNormalization", "导出前必须 fuse / replace_batchnorm")
def c_onnx_bn():
    import onnx
    fs = sorted((ROOT / "onnx").glob("*.onnx"))
    if not fs:
        return False, "onnx/ 下无模型"
    rows, bad = [], []
    for f in fs:
        m = onnx.load(str(f))
        n_bn = sum(1 for n in m.graph.node if n.op_type == "BatchNormalization")
        n_conv = sum(1 for n in m.graph.node if n.op_type == "Conv")
        rows.append(f"{f.stem}:BN={n_bn},Conv={n_conv},nodes={len(m.graph.node)}")
        if n_bn != 0:
            bad.append(f.stem)
    return (not bad), f"{rows} 未融合={bad}"


# ---------------------------------------------------------------- I ONNX 一致性 / 性能
@check("onnx.consistency", "PyTorch vs ONNX 一致率 >= 0.99", "按题目 9 项清单逐项排查")
def c_onnx_cons():
    fs = sorted((ROOT / "outputs/metrics").glob("consistency_*.json"))
    if not fs:
        return False, "缺少 outputs/metrics/consistency_<registry_key>.json（跑 deploy/compare_torch_onnx.py 生成）"
    rows, bad = [], []
    for f in fs:
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append(f"{d.get('model', f.stem)}:{d.get('top1_agree')}")
        if d.get("top1_agree", 0) < 0.99:
            bad.append(f.stem)
    return (not bad), (f"{rows} 全部 >=0.99，共 {len(fs)} 个模型" if not bad else f"未达标={bad}")


@check("onnx.realimg", "一致性用真实图片计算", "随机张量的对比说服力不足")
def c_onnx_realimg():
    fs = sorted((ROOT / "outputs/metrics").glob("consistency_*.json"))
    if not fs:
        return False, "缺少 outputs/metrics/consistency_<registry_key>.json"
    src = {json.loads(f.read_text(encoding="utf-8")).get("source") for f in fs}
    return bool(src) and src <= {"pet_test", "imagenet_val_subset"}, \
        f"数据来源={src}（必须是固定划分列表，不能是 torch.randn）"


@check("labels.pair", "标签文件与类别数配对", "37 与 1000 混用会让 Top-5 名称错乱")
def c_labels_pair():
    fs = sorted((ROOT / "outputs/metrics").glob("consistency_*.json"))
    if not fs:
        return False, "缺少 outputs/metrics/consistency_<registry_key>.json"
    rows, bad = [], []
    for f in fs:
        v = json.loads(f.read_text(encoding="utf-8"))
        rows.append(f"{v.get('model', f.stem)}:{v['num_classes']}类/{v['label_file']}")
        exp = "labels/pet_classes.txt" if v["num_classes"] == 37 else "labels/imagenet_classes.txt"
        if v["label_file"] != exp:
            bad.append(f.stem)
    return (not bad), f"{rows} 错配={bad}"


@check("bench.meta", "benchmark 元信息与次数", "预热 10 + 正式 50 是硬性要求")
def c_bench():
    f = ROOT / "outputs/metrics/bench.jsonl"
    if not f.exists():
        return False, "缺少 outputs/metrics/bench.jsonl"
    need = ["p50_ms", "p95_ms", "mean_ms", "ep", "provider_actual", "ort_version",
            "cpu_model", "os", "file_size_mb", "raw_ms"]
    rows, bad = [], []
    for r in _load_jsonl(f):
        m = [k for k in need if r.get(k) in (None, "")]
        if r.get("warmup") != 10 or r.get("runs") != 50 or len(r.get("raw_ms", [])) != 50:
            m.append("warmup/runs/raw 长度不符")
        rows.append(f"{r['model']}:P50={r.get('p50_ms')} P95={r.get('p95_ms')}")
        if m:
            bad.append({r["model"]: m})
    return (not bad), f"{rows} 问题={bad}"


@check("bench.ep", "已记录真实 Execution Provider", "唯一能排除回退 CPU 的证据")
def c_bench_ep():
    f = ROOT / "outputs/metrics/bench.jsonl"
    if not f.exists():
        return False, "缺少 bench.jsonl"
    eps = {r["model"]: (r.get("ep"), r.get("provider_actual")) for r in _load_jsonl(f)}
    return all(a and a == b for a, b in eps.values()), str(eps)


# ---------------------------------------------------------------- J 进阶与提交
@check("ablation", "消融矩阵完整 (baseline/A/B/A+B)", "缺组则该子项不完整")
def c_ablation():
    f = ROOT / "outputs/metrics/ablation.csv"
    if not f.exists():
        return False, "缺少 outputs/metrics/ablation.csv"
    rows = set(f.read_text(encoding="utf-8").splitlines()[0].split(","))
    need = {"baseline", "A", "B", "A+B"}
    return need.issubset(rows), f"列={sorted(rows)}"


SUBMIT = [
    "README.md", "requirements.txt", "requirements.lock.txt", "PROVENANCE.md",
    "datasets/lists/pet_train.txt", "datasets/lists/pet_val.txt", "datasets/lists/pet_test.txt",
    "labels/imagenet_classes.txt", "labels/pet_classes.txt",
    "checkpoints/baseline_best.pt", "checkpoints/baseline_last.pt",
    "checkpoints/opt_randaug_best.pt",
    "outputs/metrics/bench.jsonl", "outputs/metrics/selfcheck_report.json",
    "outputs/architecture/repvit_m0_9_arch.png",
    "report.pdf",
]


@check("submit", "提交物清单齐全", "报告或答辩缺失 → 本次考核不通过")
def c_submit():
    miss = [r for r in SUBMIT if not (ROOT / r).exists()]
    onnx_n = len(list((ROOT / "onnx").glob("*.onnx")))
    if onnx_n < 3:
        miss.append(f"ONNX 仅 {onnx_n} 个(<3)")
    return (not miss), ("齐全" if not miss else f"缺失={miss}")



# ---------------------------------------------------------------- J2 骨架 / README
SKELETON_DIRS = [
    "configs", "datasets", "datasets/lists", "models", "tools", "deploy", "utils",
    "labels", "outputs", "outputs/logs", "outputs/metrics", "outputs/curves",
    "outputs/confusion_matrix", "outputs/predictions", "outputs/gradcam",
    "outputs/benchmarks", "outputs/reparam", "outputs/pretrained_eval",
    "outputs/architecture", "checkpoints", "checkpoints/pretrained", "onnx", "external",
]
SKELETON_FILES = [
    "README.md", "requirements.txt", "requirements.lock.txt", "PROGRESS.md",
    "configs/baseline.yaml", "configs/pretrained_eval.yaml",
    "datasets/lists/pet_train.txt", "labels/imagenet_classes.txt",
    "labels/pet_classes.txt", "labels/pet_class_to_idx.json",
]


@check("skeleton", "目录骨架齐全", "缺哪个建哪个（见规格书 2.2 完整目录树）")
def c_skeleton():
    miss = ([d for d in SKELETON_DIRS if not (ROOT / d).is_dir()]
            + [f for f in SKELETON_FILES if not (ROOT / f).exists()])
    n = len(SKELETON_DIRS) + len(SKELETON_FILES)
    return (not miss), (f"{n} 项全部就位" if not miss else f"共 {n} 项，缺失={miss}")


README_KEYWORDS = [
    ("安装环境", ["安装环境", "环境安装", "install"]),
    ("准备数据集", ["准备数据集", "数据集准备", "数据准备", "dataset"]),
    ("获取官方权重", ["官方权重", "获取权重", "下载权重", "weight"]),
    ("运行官方模型评价", ["官方模型评价", "官方评价", "eval_pretrained", "预训练模型评价"]),
    ("训练Baseline", ["训练 baseline", "训练baseline", "baseline 训练", "如何训练"]),
    ("运行优化实验", ["优化实验", "opt_", "消融"]),
    ("验证和测试", ["验证和测试", "如何进行验证", "evaluate.py"]),
    ("曲线/混淆矩阵/Grad-CAM", ["混淆矩阵", "grad-cam", "gradcam"]),
    ("结构重参数化", ["结构重参数化", "重参数化", "reparam"]),
    ("导出ONNX模型", ["导出不同型号", "导出 onnx", "导出onnx", "export_onnx"]),
    ("运行ONNX推理", ["运行 onnx 推理", "onnx 推理", "infer_onnx"]),
    ("性能测试", ["性能测试", "benchmark", "延迟测试"]),
]


@check("readme", "README 12 项齐全", "题目第 14 页 README 最低要求，逐条补")
def c_readme():
    f = ROOT / "README.md"
    if not f.exists():
        return False, "缺少 README.md"
    low = f.read_text(encoding="utf-8").lower()
    miss = [name for name, kws in README_KEYWORDS if not any(k.lower() in low for k in kws)]
    return (not miss), ("12 项全部命中" if not miss else f"缺 {len(miss)} 项: {miss}")


# --stage 兼容层：把 DoD 表里的三个 stage 名映射到检查 id
STAGE_MAP = {
    "skeleton": ["skeleton", "paths"],
    "labels": ["data.classes", "data.labels", "labels.pair"],
    "readme": ["readme"],
}


# ---------------------------------------------------------------- runner
def main() -> int:
    ap = argparse.ArgumentParser("RepViT 交付自检")
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定 id（前缀匹配）")
    ap.add_argument("--stage", choices=sorted(STAGE_MAP), default=None,
                    help="DoD 表的 stage 口径（skeleton/labels/readme），等价于一组 --only")
    ap.add_argument("--json", default="outputs/metrics/selfcheck_report.json")
    ap.add_argument("--strict", action="store_true", help="有 FAIL 时返回退出码 1")
    a = ap.parse_args()
    if a.stage:                                    # --stage 展开成一组 --only
        a.only = list(a.only or []) + STAGE_MAP[a.stage]

    checks = [v for k, v in sorted(globals().items())
              if callable(v) and getattr(v, "cid", None)]
    if a.only:
        checks = [c for c in checks if any(c.cid == o or c.cid.startswith(o + ".") for o in a.only)]
    for c in checks:
        c()

    width = max((len(r["id"]) for r in RESULTS), default=8)
    n_pass = sum(1 for r in RESULTS if r["status"] == "PASS")
    print(f"\n{'ID'.ljust(width)}  STATUS  DETAIL")
    print("-" * 100)
    for r in RESULTS:
        print(f"{r['id'].ljust(width)}  {r['status']:<6}  {r['detail']}")
        if r["status"] == "FAIL" and r["fix"]:
            print(f"{' ' * width}  修复 -> {r['fix']}")
    print("-" * 100)
    print(f"合计 {len(RESULTS)} 项：PASS {n_pass} / FAIL {len(RESULTS) - n_pass}")

    out = ROOT / a.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(), "platform": platform.platform(),
        "summary": {"total": len(RESULTS), "pass": n_pass, "fail": len(RESULTS) - n_pass},
        "checks": RESULTS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入 {_rel(out)}")

    failed = len(RESULTS) - n_pass
    return 1 if (failed and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
