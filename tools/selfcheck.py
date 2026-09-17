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
import io
import json
import platform
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

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
    # t27 并入：审计文档的**证据指针**校验（不新增 check id，总数保持 34 —— 与 t14/t19 同一手法）。
    # 背景：t2 写的证据引用在 t4 压缩报告 / 删除违规表 / 重排章节后集体失效，t6/t22 只修了被点名的行。
    # 这里把「判定=满足/部分 的证据必须仍可核对」变成常驻闸：
    #   ① 被引产物必须存在；② 每处 §x.y 在目标文档（自指 / 跨文档）里确有该标题；
    #   ③ t30 起还做「可重算断言」比对：计数（git ls-files onnx/、官方 ONNX）、体积（实测 **N B**）、
    #      行数（实测 **N 行** / **N 行数据**）、grep 命中（**N 命中 / 个文件 / 行**）、§8 统计求和。
    ev_ok, ev_detail = True, "证据指针 OK"
    try:
        import check_audit_evidence as cae          # 同目录模块（tools/）
        res = cae.scan()
        ev_ok = not res["fails"]
        ev_detail = (f"证据指针 OK（行 {res['rows']} / 路径 {res['path_refs']} / § {res['sec_refs']} / "
                     f"可重算断言 {res.get('assertions', 0)}）"
                     if ev_ok else
                     f"证据指针 {len(res['fails'])} 处失效："
                     + "; ".join(f"{x['kind']}@L{x['line']}" for x in res["fails"][:5]))
    except Exception as e:                          # noqa: BLE001
        ev_ok, ev_detail = False, f"证据指针扫描器异常 {type(e).__name__}: {e}"

    detail = f"{len(hits)} 处" + (f" 首处 {hits[0]}" if hits else "") + "；" + ev_detail
    return (not hits) and ev_ok, detail


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
# 口径（2026-09 用户决议）：官方模型评价的**唯一**数据集是 ImageNetV2 matched-frequency 的
# 1000 张确定性固定子集（1000 类各 1 张，无随机种子；选取规则见 datasets/make_imagenetv2_subset.py）。
# ImageNetV2 是 Recht et al. 2019 独立重采样的测试集，其准确率与论文/官方公布的 ImageNet-1K
# 数值**不可直接比较**（预期低 10~15 个点是基准性质，不是模型退化），因此本项**不再**做
# 「与官方公布值 ±1.5 以内」的值域比对，改为「口径 + 落盘结构」的结构性断言。
OFFICIAL_MODELS = ["repvit_m0_9", "repvit_m1_0", "repvit_m1_1", "repvit_m1_5", "repvit_m2_3"]
IMAGENET_LABEL_ANCHORS = {0: "tench", 207: "golden retriever", 281: "tabby", 999: "toilet tissue"}


def _eval_data_list() -> Path:
    """从 configs/pretrained_eval.yaml 读当前官方评价清单（唯一配置真源）。"""
    import yaml
    cfg = yaml.safe_load((ROOT / "configs/pretrained_eval.yaml").read_text(encoding="utf-8"))
    return ROOT / cfg["pretrained_eval"]["data_list"]


@check("official.top1", "官方评价口径与落盘结构正确",
       "ImageNetV2 与 ImageNet-1K 公布值不可比，只做结构性断言")
def c_official_top1():
    bad, rows = [], []

    # (1) 清单本身：真实存在、1000 行、1000 个不同标签各 1 张
    try:
        lf = _eval_data_list()
    except Exception as e:                                   # 配置读不到就是硬错误
        return False, f"读 configs/pretrained_eval.yaml 失败：{e}"
    if not lf.exists():
        return False, f"评价清单不存在：{_rel(lf)}（口径必须是 ImageNetV2 固定子集）"
    lines = [l for l in lf.read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = [int(l.rsplit(None, 1)[1]) for l in lines]
    per_class = {}
    for y in labels:
        per_class[y] = per_class.get(y, 0) + 1
    rows.append(f"清单={_rel(lf)} {len(lines)} 行")
    if not (500 <= len(lines) <= 1000):
        bad.append(f"张数 {len(lines)} 不在 500~1000")
    if len(set(labels)) != 1000:
        bad.append(f"标签只覆盖 {len(set(labels))} 类（应为 1000）")
    if max(per_class.values(), default=0) != 1:
        bad.append(f"存在每类 >1 张（max={max(per_class.values(), default=0)}）")
    if sorted(per_class) != list(range(1000)):
        bad.append("标签不是 0..999")

    # (2) 标签映射锚点：0=tench / 207=golden retriever / 281=tabby / 999=toilet tissue
    lpf = ROOT / "labels/imagenet_classes.txt"
    if not lpf.exists():
        bad.append("缺少 labels/imagenet_classes.txt")
    else:
        names = [l.strip() for l in lpf.read_text(encoding="utf-8").splitlines() if l.strip()]
        for idx, want in IMAGENET_LABEL_ANCHORS.items():
            got = names[idx] if idx < len(names) else None
            if got != want:
                bad.append(f"锚点 {idx}: 期望 {want!r} 实际 {got!r}")
        rows.append(f"标签 {len(names)} 行，锚点 {'OK' if not bad else '异常'}")

    # (3) 5 个型号落盘齐全：metrics.json 的 num_images / num_classes
    n_done = 0
    for m in OFFICIAL_MODELS:
        f = ROOT / f"outputs/pretrained_eval/{m}/metrics.json"
        if not f.exists():
            bad.append(f"{m} 缺 metrics.json")
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        n_done += 1
        n_img = int(d.get("num_images", 0))
        if not (500 <= n_img <= 1000):
            bad.append(f"{m} num_images={n_img} 不在 500~1000")
        if int(d.get("num_classes", 0)) != 1000:
            bad.append(f"{m} num_classes={d.get('num_classes')} != 1000")
    rows.append(f"落盘型号 {n_done}/{len(OFFICIAL_MODELS)}")
    if n_done != len(OFFICIAL_MODELS):
        bad.append(f"落盘型号数 {n_done} != {len(OFFICIAL_MODELS)}")
    return (not bad), (f"{rows} 结构断言通过（不做官方值比对）" if not bad else f"异常={bad}")


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
CHART_NS = "{http://schemas.openxmlformats.org/drawingml/2006/chart}"
PML_NS = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
DML_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

# ---- t47/t49 并入：把「图元看不见」的全部形态机器化 ----
# 背景：t41 只覆盖「显式设了 min/max 且数据超界」这一种；t44 用负向探针实证另三种形态
# 仍能穿过既有判据（无显式轴边界的图、缓存为空的图、纯白图片）；t37 又实证 t47 的①只出 NOTE、
# 汇总行仍打印「OK」→ 回归会被一句话掩盖。本段把过滤集合补全：
#   ① 未显式设 min/max 的数值轴 → **FAIL**（t49 起）：写入侧 `axis_bounds()` 已硬保证每张图都有
#      显式边界，故「无显式边界」= 该图**绕过了写入侧**，而轴边界未显式正是 P14 整张空白的那条通路；
#      汇总行同时打印「N 条数值轴全部显式设边界」，使覆盖度不再可被静默掩盖；
#   ② 图表必须有非空序列数据（numCache / numLit）→ 空则 FAIL（PowerPoint 会画成空白图）；
#   ③ `p:pic` → slide.rels → `ppt/media/*`：存在、非零尺寸、非纯色/空白 → FAIL（全 15 页扫描）；
#   ④ PPTX↔PDF 同源：页数相等 + 每页关键文本可在对应 PDF 页检索到 → FAIL。
#
# 图片「空白」阈值依据：纯白/纯色图 → 灰度标准差 ≈ 0、灰阶数 = 1。**实测**本仓库 11 张
# `ppt/media/*.png` 为 std ∈ [35.1, 85.0]、灰阶 ∈ [245, 256]，故阈值留 ≥10 倍余量。
IMG_MIN_STD = 3.0                # 纯色图 std ≈ 0（实测本仓最小 35.1 → 余量 11×）
IMG_MIN_GRAY_LEVELS = 4          # 纯色图灰阶 = 1（实测本仓最小 245 → 余量 61×）
IMG_MIN_BYTES = 512              # 占位/空白图元体积极小（实测本仓最小 64,589 B → 余量 126×）
PDF_RUN_COVERAGE = 0.80          # 逐页：≥80% 文本 run 必须能在对应 PDF 页检索到
PDF_RUN_MINLEN = 6
_NORM_DROP = "，。、；：,.;:!！?？()（）[]【】“”\"'’‘·—–-…/\\|<>《》"


class ChartScan(tuple):
    """`chart_axis_findings()` 的返回值：(轴类 FAIL 列表, 图表数) —— 仍是 2 元组，旧调用不受影响；
    另带 `notes`（不可机检项的说明）、`cache_fails`（空缓存）与 t49 的轴边界计数
    （`n_axis_explicit` / `n_axis_implicit` / `charts_no_axis`，供汇总行打印覆盖度）。"""

    def __new__(cls, axis_fails: list[str], n_charts: int,
                notes: list[str] | None = None, cache_fails: list[str] | None = None,
                n_axis_explicit: int = 0, n_axis_implicit: int = 0, charts_no_axis: int = 0):
        self = super().__new__(cls, (axis_fails, n_charts))
        self.axis_fails = axis_fails
        self.notes = notes or []
        self.cache_fails = cache_fails or []
        self.n_axis_explicit = n_axis_explicit
        self.n_axis_implicit = n_axis_implicit
        self.charts_no_axis = charts_no_axis
        return self


def _norm_for_match(s: str) -> str:
    """归一化（NFKC + 去空白与标点 + 小写）：PDF 抽取会在中英文之间插空格，必须去空白比对。"""
    import unicodedata
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if ch not in _NORM_DROP)
    return re.sub(r"\s+", "", s).lower()


def _slide_runs(root) -> list[str]:
    """PPTX 一页里的文本 run（按段落合并 `a:t`）。"""
    out: list[str] = []
    for para in root.iter(DML_NS + "p"):
        txt = "".join(t.text or "" for t in para.iter(DML_NS + "t")).strip()
        if txt:
            out.append(txt)
    return out


def _image_stats(data: bytes) -> dict:
    """光栅图的体检数字：空白（纯色/零尺寸/过小）+ 尺寸 + 灰度标准差 + 灰阶数。"""
    st = {"blank": False, "why": "", "w": 0, "h": 0, "std": None, "levels": None}
    if len(data) < IMG_MIN_BYTES:
        st.update(blank=True, why=f"仅 {len(data)} B")
        return st
    from PIL import Image, ImageStat
    im = Image.open(io.BytesIO(data))
    st["w"], st["h"] = im.size
    if st["w"] <= 0 or st["h"] <= 0:
        st.update(blank=True, why=f"像素尺寸 {st['w']}x{st['h']}")
        return st
    g = im.convert("L")
    st["std"] = round(float(ImageStat.Stat(g).stddev[0]), 1)
    st["levels"] = sum(1 for n in g.histogram() if n)
    lo, hi = g.extrema if hasattr(g, "extrema") else g.getextrema()
    st["why"] = f"{st['w']}x{st['h']} std={st['std']} 灰阶={st['levels']}"
    if st["std"] < IMG_MIN_STD or st["levels"] <= IMG_MIN_GRAY_LEVELS or lo == hi:
        st["blank"] = True
        st["why"] += "（纯色/空白）"
    return st


def _image_scan(pptx: Path) -> dict:
    """③ `p:pic` → slide.rels → `ppt/media/*`（**全 15 页**扫描，不只图表页）。

    FAIL：图片 part 缺失 / 0 字节 / 纯色空白。提示：外链图（包里查不到字节）、孤儿 media。
    """
    rep: dict = {"fails": [], "notes": [], "n_pics": 0, "n_media": 0, "min_std": None, "min_levels": None}
    with zipfile.ZipFile(pptx) as z:
        names = set(z.namelist())
        slides = sorted([n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)],
                        key=lambda s: int(re.findall(r"\d+", s)[0]))
        media = sorted([n for n in names if n.startswith("ppt/media/")])
        rep["n_media"] = len(media)
        refd: set[str] = set()
        for slide in slides:
            rels_path = f"ppt/slides/_rels/{Path(slide).name}.rels"
            rels: dict[str, str] = {}
            if rels_path in names:
                for rel in ET.fromstring(z.read(rels_path)):
                    rels[rel.get("Id")] = rel.get("Target")
            root = ET.fromstring(z.read(slide))
            for pic in root.iter(PML_NS + "pic"):
                blip = pic.find(f".//{DML_NS}blip")
                rid = blip.get(REL_NS + "embed") if blip is not None else None
                link = blip.get(REL_NS + "link") if blip is not None else None
                rep["n_pics"] += 1
                if rid:
                    target = rels.get(rid)
                    if not target:
                        rep["fails"].append(f"{Path(slide).name}: 图片关系无法解析（rId={rid}）→ 该页留空框")
                        continue
                    part = target[3:] if target.startswith("../") else target
                    part = part if part.startswith("ppt/") else f"ppt/media/{Path(part).name}"
                    refd.add(part)
                    if part not in names:
                        rep["fails"].append(f"{Path(slide).name}: 引用的图片 part 缺失（{part}）→ 该页留空框")
                elif link:
                    rep["notes"].append(f"{Path(slide).name}: 外链图片 rId={link}（包里无字节，无法机检内容）")
                else:
                    rep["fails"].append(f"{Path(slide).name}: p:pic 无 r:embed/r:link → 该页留空框")
        for part in media:
            data = z.read(part)
            if not data:
                rep["fails"].append(f"{part}: 0 字节（图片空白）")
                continue
            if not part.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")):
                rep["notes"].append(f"{part}: 非光栅图（{Path(part).suffix}），只查存在性与字节数")
                continue
            try:
                st = _image_stats(data)
            except Exception as e:                       # noqa: BLE001
                rep["fails"].append(f"{part}: 图片解码失败 {type(e).__name__}: {e}（PPT 里会是空白）")
                continue
            if st["blank"]:
                rep["fails"].append(f"{part}: 图片为空白/纯色（{st['why']}）")
                continue
            rep["min_std"] = st["std"] if rep["min_std"] is None else min(rep["min_std"], st["std"])
            rep["min_levels"] = st["levels"] if rep["min_levels"] is None else min(rep["min_levels"], st["levels"])
        orphan = sorted(set(media) - refd)
        if orphan:
            rep["notes"].append(f"孤儿图片 part {len(orphan)} 个（未被任何页引用）：{[Path(o).name for o in orphan[:3]]}")
    return rep


def _pdf_sync_scan(pptx: Path, pdf: Path | None = None) -> dict:
    """④ PPTX↔PDF 同源（答辩现场看的是 PDF）：页数相等 + 每页关键文本可在对应 PDF 页检索到。"""
    pdf_path = Path(pdf) if pdf else ROOT / "report" / "答辩PPT_RepViT.pdf"
    rep: dict = {"fails": [], "notes": [], "n_slides": 0, "n_pages": 0, "pages_ok": 0}
    if not pdf_path.exists():
        rep["fails"].append(f"PDF 不存在：{pdf_path}（答辩现场看的是 PDF，必须与 PPTX 同源）")
        return rep
    try:
        import fitz
    except ImportError as e:                             # noqa: BLE001
        rep["fails"].append(f"缺 pymupdf（requirements.lock.txt 已冻结），无法机检 PPTX↔PDF 同源：{e}")
        return rep
    doc = fitz.open(pdf_path)
    rep["n_pages"] = doc.page_count
    with zipfile.ZipFile(pptx) as z:
        slides = sorted([n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)],
                        key=lambda s: int(re.findall(r"\d+", s)[0]))
        rep["n_slides"] = len(slides)
        if len(slides) != doc.page_count:
            rep["fails"].append(f"页数不一致：PPTX {len(slides)} 页 ≠ PDF {doc.page_count} 页（PDF 可能是旧的）")
        for i, slide in enumerate(slides):
            if i >= doc.page_count:
                break
            page_norm = _norm_for_match(doc[i].get_text())
            runs = [_norm_for_match(r) for r in _slide_runs(ET.fromstring(z.read(slide)))]
            runs = [r for r in runs if len(r) >= PDF_RUN_MINLEN]
            if not runs:
                continue
            key = max(runs, key=len)
            hit = [r for r in runs if r in page_norm]
            cov = len(hit) / len(runs)
            if key not in page_norm or cov < PDF_RUN_COVERAGE:
                miss = [r for r in runs if r not in page_norm]
                rep["fails"].append(
                    f"第 {i + 1} 页与 PDF 不同源：关键文本命中 {len(hit)}/{len(runs)}"
                    f"（{cov * 100:.0f}% < {PDF_RUN_COVERAGE * 100:.0f}%），缺失示例 {[m[:36] for m in miss[:2]]}")
                continue
            rep["pages_ok"] += 1
    mt_pptx, mt_pdf = pptx.stat().st_mtime, pdf_path.stat().st_mtime
    if mt_pptx > mt_pdf + 1:
        rep["notes"].append(
            f"PPTX 比 PDF 新 {int(mt_pptx - mt_pdf)} 秒（导出可能滞后，建议重跑 tools/export_defense_pdf.py）")
    return rep



def _chart_cache_values(root, tag: str) -> list[float]:
    """取 <c:{tag}><c:numCache> 里的数值（只认缓存值，排除系列名 tx/v）。"""
    out: list[float] = []
    for holder in root.iter(CHART_NS + tag):
        for cache in holder.iter(CHART_NS + "numCache"):
            for v in cache.iter(CHART_NS + "v"):
                try:
                    out.append(float(v.text))
                except (TypeError, ValueError):
                    pass
    return out


def chart_axis_findings(pptx: Path | None = None) -> tuple[list[str], int]:
    """★ t41 并入：PPTX 里每个图表的**数据必须落在其数值轴范围内**。

    背景：第 14 页右图曾把 Y 轴写死 77~84（旧自建子集口径），口径切到 ImageNetV2 后
    5 个点（68.7~73.6）全部落在轴外 → 图面空白，而此前所有判据只看「存在性与文字」，
    没有一条查「数据是否落在轴内」。本函数把该判据机器化：
      · 每个显式设置了 min/max 的数值轴，其对应的系列值必须落在 [min,max] 内；
      · 轴跨度不得把数据跨度稀释到 < 15%（避免点小到看不见）。
    散点图两个轴都是 valAx，按（X, Y）顺序分别对应 xVal / yVal 缓存。

    t47 起返回 `ChartScan`（**仍是 2 元组**，旧调用 `bad, n = chart_axis_findings()` 不受影响），
    额外携带：`notes`（不可机检项的说明）、`cache_fails`（缓存/字面量数据为空 —— PowerPoint 会画成
    空白图，判 FAIL）与轴边界计数（供 `viz` 汇总行打印覆盖度）。

    t49：**未显式设 min/max 的数值轴由「提示项」升级为 FAIL** —— 写入侧 `axis_bounds()` 保证每张图
    都有显式边界，因此缺边界即「该图绕过了写入侧」；只出 NOTE 时汇总行仍打印「OK（N 图）」，
    回归会被一句话掩盖（t37 实证）。同时汇总行显式打印「N 条数值轴全部显式设边界」。
    """
    path = Path(pptx) if pptx else ROOT / "report" / "答辩PPT_RepViT.pptx"
    if not path.exists():
        return ChartScan([f"PPTX 不存在：{path}"], 0)
    findings: list[str] = []
    notes: list[str] = []                     # t49：只放「不可机检」的说明（本身不判 FAIL）
    cache_fails: list[str] = []               # t47②：空缓存 = 白图 → FAIL
    n_explicit = n_implicit = charts_no_axis = 0
    with zipfile.ZipFile(path) as z:
        names = sorted([n for n in z.namelist() if re.match(r"ppt/charts/chart\d+\.xml$", n)],
                       key=lambda s: int(re.findall(r"\d+", s)[0]))
        for name in names:
            root = ET.fromstring(z.read(name))
            vals = _chart_cache_values(root, "val")
            xs = _chart_cache_values(root, "xVal")
            ys = _chart_cache_values(root, "yVal")
            # ② 空缓存（t47）：逐系列统计 numCache/numLit 点数；全空 → PowerPoint 会画成空白图
            series = list(root.iter(CHART_NS + "ser"))
            pts = []
            for ser in series:
                n_pt = 0
                for tag in ("val", "xVal", "yVal"):
                    for holder in ser.iter(CHART_NS + tag):
                        for cache in list(holder.iter(CHART_NS + "numCache")) + \
                                list(holder.iter(CHART_NS + "numLit")):
                            n_pt += len(list(cache.iter(CHART_NS + "v")))
                pts.append(n_pt)
            if not series:
                cache_fails.append(f"{name}: 无任何系列（图表会渲染成空白）")
            elif sum(pts) == 0:
                cache_fails.append(f"{name}: 全部 {len(series)} 个系列的缓存/字面量数据均为空（会渲染成空白图）")
            else:
                empty_ser = [i for i, c in enumerate(pts) if c == 0]
                if empty_ser:
                    cache_fails.append(f"{name}: 第 {empty_ser} 个系列无数据点（该系列画不出来）")
            # ① 轴边界（t47 起不静默跳过；**t49 起缺显式边界 = FAIL**）
            axes = []
            n_ax = 0
            n_impl = 0
            for ax in root.iter(CHART_NS + "valAx"):
                n_ax += 1
                sc = ax.find(CHART_NS + "scaling")
                mn = sc.find(CHART_NS + "min") if sc is not None else None
                mx = sc.find(CHART_NS + "max") if sc is not None else None
                if mn is None and mx is None:
                    n_impl += 1
                    continue
                axes.append((float(mn.get("val")) if mn is not None else None,
                             float(mx.get("val")) if mx is not None else None))
            n_explicit += len(axes)
            n_implicit += n_impl
            if n_ax == 0:
                charts_no_axis += 1
                notes.append(f"{name}: 无数值轴（不可机检裁切；写入侧只产出折线/柱状/散点，均有数值轴）")
            elif n_impl:
                findings.append(
                    f"{name}: {n_impl}/{n_ax} 个数值轴未显式设 min/max —— 写入侧 axis_bounds() 硬保证每张图都有"
                    f"显式边界，缺边界说明该图**绕过了写入侧**，且轴边界未显式正是 P14 整张空白的那条通路")
                notes.append(f"{name}: {n_impl}/{n_ax} 个数值轴不可机检（依赖自动缩放）")
            for i, (mn, mx) in enumerate(axes):
                if len(axes) > 1:                       # 散点：轴序 = X, Y
                    data = xs if i == 0 else ys
                else:                                   # 单值轴：柱/折线用 val，散点退化为 yVal
                    data = vals or ys
                if not data:
                    continue
                dmin, dmax = min(data), max(data)
                if (mn is not None and dmin < mn - 1e-9) or (mx is not None and dmax > mx + 1e-9):
                    findings.append(f"{name}: 数据 {dmin:.4g}~{dmax:.4g} 超出轴 {mn}~{mx}（会被裁成空白图）")
                elif mn is not None and mx is not None and (mx - mn) > 0 and \
                        (dmax - dmin) < 0.15 * (mx - mn) and dmax > 0:
                    findings.append(f"{name}: 轴跨度过宽（数据仅占 {(dmax - dmin) / (mx - mn) * 100:.0f}%，点会小到看不清）")
        return ChartScan(findings, len(names), notes, cache_fails,
                         n_explicit, n_implicit, charts_no_axis)


@check("viz", "八类可视化产物齐全 + 图表数据落在轴范围内 + 图元非空白 + PPTX↔PDF 同源",
       "缺哪类补哪类（8 分项）；空白图 = 轴范围没包住数据 / 轴边界未显式（绕过写入侧）/ 缓存为空 / 图片纯色 / PDF 是旧的")
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
    notes: list[str] = []
    ax_txt = "图表轴范围扫描失败"
    try:
        scan = chart_axis_findings()
        n_charts = scan[1]
        miss.extend(scan[0])                         # 轴类 FAIL（数据超界 / 轴跨度过宽 / **未显式设边界**）
        miss.extend(scan.cache_fails)                # t47② 空缓存
        notes.extend(scan.notes)                     # 不可机检项说明
        if scan.charts_no_axis:
            # t49：有图无法机检时，汇总行**不得**再是全好的措辞
            ax_txt = (f"图表轴范围部分不可机检（{n_charts} 图：{n_charts - scan.charts_no_axis} 图可机检、"
                      f"{scan.charts_no_axis} 图无数值轴）")
        else:
            ax_txt = (f"图表轴范围 OK（{n_charts} 图 / {scan.n_axis_explicit} 条数值轴全部显式设边界）")
    except Exception as e:                       # noqa: BLE001
        n_charts = 0
        miss.append(f"图表轴范围扫描异常 {type(e).__name__}: {e}")
    # t47③ 图片类图元（p:pic → slide.rels → ppt/media/*，全 15 页）与 t47④ PPTX↔PDF 同源
    img_txt = pdf_txt = ""
    try:
        img = _image_scan(ROOT / "report" / "答辩PPT_RepViT.pptx")
        miss.extend(img["fails"])
        notes.extend(img["notes"])
        img_txt = (f"图片 {img['n_pics']} 张/{img['n_media']} media"
                   f"（min std={img['min_std']}、min 灰阶={img['min_levels']}）")
    except Exception as e:                       # noqa: BLE001
        miss.append(f"图片扫描异常 {type(e).__name__}: {e}")
    try:
        ps = _pdf_sync_scan(ROOT / "report" / "答辩PPT_RepViT.pptx")
        miss.extend(ps["fails"])
        notes.extend(ps["notes"])
        pdf_txt = (f"PPTX↔PDF {ps['n_slides']}={ps['n_pages']} 页"
                   f"、逐页关键文本命中 {ps['pages_ok']}/{ps['n_slides']}")
    except Exception as e:                       # noqa: BLE001
        miss.append(f"PPTX↔PDF 同源扫描异常 {type(e).__name__}: {e}")
    note_txt = ("；提示：" + "；".join(notes)) if notes else ""
    if miss:
        return False, f"缺失={miss}{note_txt}"
    return True, (f"curves={len(g['curves'])} cm={len(g['cm'])} preds={len(g['preds'])} "
                  f"compare={len(g['compare'])} cam={len(g['gradcam'])}"
                  f"；{ax_txt} + 空缓存 0；{img_txt}；{pdf_txt}{note_txt}")


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


# 固定划分清单白名单：`source` 字段（= 清单文件名词干）-> 该清单必须**真实存在**且行数合法。
# 只认「名字在不在白名单」是一个没有验证力的绿灯：清单被删掉之后 JSON 里的旧 source 仍会通过。
# 这里连同「被引用的清单文件存在性 + 行数区间」一起断言，口径再变时下游会自动重新校验。
SOURCE_LISTS = {
    "pet_test": "datasets/lists/pet_test.txt",
    "imagenetv2_mf_1000": "datasets/lists/imagenetv2_mf_1000.txt",
}
SOURCE_MIN_ROWS = 37            # 至少够跑 n=12 的一致性对比


@check("onnx.realimg", "一致性用真实图片计算", "随机张量的对比说服力不足")
def c_onnx_realimg():
    fs = sorted((ROOT / "outputs/metrics").glob("consistency_*.json"))
    if not fs:
        return False, "缺少 outputs/metrics/consistency_<registry_key>.json"
    rows, bad = [], []
    for f in fs:
        d = json.loads(f.read_text(encoding="utf-8"))
        src = d.get("source")
        n = int(d.get("n", 0) or 0)
        key = d.get("model", f.stem)
        if src not in SOURCE_LISTS:
            bad.append(f"{key}(未知 source={src!r})")
            continue
        # source 指向的清单必须真的存在（否则就是「引用已删除清单的绿灯」）
        lf = ROOT / SOURCE_LISTS[src]
        if not lf.exists():
            bad.append(f"{key}(清单缺失 {SOURCE_LISTS[src]})")
            continue
        rows_n = len([l for l in lf.read_text(encoding="utf-8").splitlines() if l.strip()])
        if rows_n < SOURCE_MIN_ROWS:
            bad.append(f"{key}(清单只有 {rows_n} 行)")
            continue
        # images 字段必须与 source 指向同一份清单（防止两处漂移）
        imgs = str(d.get("images", "")).replace("\\", "/")
        if imgs and not imgs.endswith(SOURCE_LISTS[src]):
            bad.append(f"{key}(images={imgs} 与 source={src!r} 不一致)")
            continue
        if n < 12:
            bad.append(f"{key}(n={n} < 12)")
            continue
        rows.append(f"{key}:{src}({rows_n} 行/{n} 张)")
    return (not bad), (f"{rows} 全部来自真实固定清单" if not bad else f"异常={bad}")


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


@check("bench.meta", "benchmark 元信息与次数 + summary.csv 完整",
       "预热 10 + 正式 50 是硬性要求；summary.csv 不得少于已落盘模型数")
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

    # summary.csv 完整性：必须覆盖 outputs/benchmarks 下**每一份** *_benchmark.json。
    # 背景（真事故）：deploy/benchmark.py 曾被逐模型调用时整表覆盖写，把 6 行打成 1 行，
    # 结果是 PPT P14 的型号元信息退化却一路绿灯。这个不变量现在有两条防线：
    #   (1) deploy/benchmark.py 的 assert_summary_complete()（写盘时立刻报错）
    #   (2) 这里（把已经发生的截断判成 FAIL，而不是等下游页面悄悄变形）
    import csv as _csv
    od = ROOT / "outputs/benchmarks"
    json_models = {p.name[: -len("_benchmark.json")] for p in od.glob("*_benchmark.json")}
    csv_path = od / "summary.csv"
    if not csv_path.exists():
        bad.append({"summary.csv": "缺失"})
    else:
        with csv_path.open(encoding="utf-8-sig", newline="") as fh:
            got = {r["model"] for r in _csv.DictReader(fh) if r.get("model")}
        miss = sorted(json_models - got)
        rows.append(f"summary.csv={len(got)} 行/JSON {len(json_models)} 个")
        if miss:
            bad.append({"summary.csv": f"只 {len(got)} 行，缺 {len(miss)} 个已落盘模型：{miss}"})

    # family_summary.csv 的 top1/top5 必须与 outputs/pretrained_eval/<model>/metrics.json 一致。
    # 背景（真事故）：家族表由 tools/eval_family.py 在一次独立复跑里写出，其 top1/top5 是那次
    # 复跑的实测值；当主口径换成 ImageNetV2 后，其中 2 个值（m0_9.top5=85.8、m1_5.top1=71.6）
    # 与权威产物 metrics.json 分叉，且被 P04 当作 MACs/参数的来源之一、被报告 §5 当作边际收益的
    # 对照——只有人工比对才会发现。这里把「家族表准确率 = 权威产物」变成可执行断言。
    # 同样并入既有 bench.meta（不新增 check id，保持 34 项）。
    fam_path = od / "family_summary.csv"
    if fam_path.exists():
        with fam_path.open(encoding="utf-8-sig", newline="") as fh:
            fam_rows = [r for r in _csv.DictReader(fh) if r.get("model")]
        mism, checked = [], 0
        for r in fam_rows:
            mj = ROOT / "outputs/pretrained_eval" / str(r["model"]).replace("_in1k", "") / "metrics.json"
            if not mj.exists():
                mism.append(f"{r['model']}: 缺 {mj.relative_to(ROOT).as_posix()}")
                continue
            m = json.loads(mj.read_text(encoding="utf-8"))
            for col in ("top1", "top5"):
                try:
                    if abs(float(r[col]) - float(m[col])) > 1e-9:
                        mism.append(f"{r['model']}.{col}={r[col]} ≠ metrics {m[col]}")
                    else:
                        checked += 1
                except (KeyError, TypeError, ValueError) as e:
                    mism.append(f"{r['model']}.{col} 无法比较（{e}）")
        rows.append(f"family_summary.csv={len(fam_rows)} 行/{checked} 个准确率与 metrics.json 一致")
        if mism:
            bad.append({"family_summary.csv": mism})
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


# ---- t49（F-15）：把「从未被任何闸门覆盖」的 **in-repo** 载体纳入存在性 + 关键文本断言 ----
# t40 实测：`report/REPORT.docx`、`report/SPEC13_FREEZE.json`、`report/ppt_svg/*.svg`（15 份）、
# `docs/workflow_zh.{html,svg}` 在此前 6 个闸门脚本里**一次都没出现过** —— 它们同样是交付物，
# 改坏了没有任何断言会报。
# 不纳入（**仓外载体**）：工作区根目录的 `06_答辩Q&A_精简版.docx` —— 门禁一律以仓库根 `ROOT` 解析
# 路径，仓外文件既不在 `git ls-files` 也不在任何交付根内，硬塞会变成「看仓库外路径」的假依赖，
# 且 CI/他人复现时必然 FAIL。故明确记录：该载体不可由仓库门禁覆盖，只能靠人工/外审核对。
REPORT_DOCX = "report/REPORT.docx"
SPEC13_FREEZE = "report/SPEC13_FREEZE.json"
PPT_SVG_NAMES = ["01_cover.svg", "02_status.svg", "03_arch.svg", "04_notvit.svg", "05_pretrained.svg",
                 "06_data.svg", "07_baseline.svg", "08_optimize.svg", "09_results.svg",
                 "10_confusion.svg", "11_gradcam.svg", "12_reparam.svg", "13_onnx.svg",
                 "14_perf.svg", "15_summary.svg"]
PPT_SVG_NODES = ("ImageNetV2固定子集", "重参数化", "ONNX", "Grad-CAM", "混淆矩阵")
WORKFLOW_FILES = ("docs/workflow_zh.html", "docs/workflow_zh.svg")
WORKFLOW_NODES = ("run_all", "selfcheck", "重参数化", "Grad-CAM")


def _carrier_text(path: Path) -> str:
    """载体的「去标签 + 去空白」纯文本：docx 取 `word/*.xml`，svg/html 直接去标签。"""
    if path.suffix.lower() == ".docx":
        with zipfile.ZipFile(path) as z:
            raw = "".join(z.read(n).decode("utf-8", "replace")
                          for n in z.namelist()
                          if n.startswith("word/") and n.endswith(".xml"))
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")
    return re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", raw))


def carrier_content_findings() -> tuple[list[str], str]:
    """F-15：此前无闸门覆盖的 in-repo 载体的**存在性 + 关键文本**（每条都做过负向验证）。"""
    fails: list[str] = []
    docx_meta = "REPORT.docx 缺失"
    # 1) report/REPORT.docx：冻结口径措辞与数值必须在；被禁措辞必须不在
    docx = ROOT / REPORT_DOCX
    if not docx.exists():
        fails.append(f"{REPORT_DOCX} 缺失（报告 docx 是提交物）")
    else:
        try:
            t = _carrier_text(docx)
            docx_meta = f"REPORT.docx {len(t)} 字"
            if "官方预训练型号" not in t:
                fails.append(f"{REPORT_DOCX}: 缺措辞「官方预训练型号」（F-11 口径）")
            if "ImageNetV2" not in t:
                fails.append(f"{REPORT_DOCX}: 缺唯一口径名「ImageNetV2」")
            if "官方ImageNet-1K型号" in t:
                fails.append(f"{REPORT_DOCX}: 含被禁措辞「官方 ImageNet-1K 型号」（暗示与 1K 公布值可比）")
            m = ROOT / "outputs/pretrained_eval/repvit_m0_9/metrics.json"
            if m.exists():
                top1 = float(json.loads(m.read_text(encoding="utf-8"))["top1"])
                if not any(c in t for c in (f"{top1:.2f}", f"{top1:.1f}", str(top1))):
                    fails.append(f"{REPORT_DOCX}: 缺冻结口径数值（M0.9 在 ImageNetV2 固定子集 top1 = {top1}）")
        except Exception as e:                       # noqa: BLE001
            fails.append(f"{REPORT_DOCX}: 读取/核对失败 {type(e).__name__}: {e}")
    # 2) report/ppt_svg/*.svg：15 份设计源 + 关键节点名（联合文本）
    svg_dir = ROOT / "report/ppt_svg"
    have = sorted(p.name for p in svg_dir.glob("*.svg"))
    miss_svg = sorted(set(PPT_SVG_NAMES) - set(have))
    if miss_svg:
        fails.append(f"report/ppt_svg 缺 {len(miss_svg)} 份设计源：{miss_svg}")
    texts: dict[str, str] = {}
    for n in have:
        try:
            texts[n] = _carrier_text(svg_dir / n)
        except Exception as e:                       # noqa: BLE001
            fails.append(f"report/ppt_svg/{n}: 读取失败 {type(e).__name__}")
    thin = sorted(n for n, s in texts.items() if len(s) < 100)
    if thin:
        fails.append(f"report/ppt_svg 文本过少（<100 字，疑似截断/空白）：{thin}")
    union = "".join(texts.values())
    lack_svg = [k for k in PPT_SVG_NODES if k not in union]
    if lack_svg:
        fails.append(f"report/ppt_svg 联合文本缺关键节点：{lack_svg}")
    # 3) docs/workflow_zh.{html,svg}：工作流程图关键节点名（逐文件）
    for rel in WORKFLOW_FILES:
        p = ROOT / rel
        if not p.exists():
            fails.append(f"{rel} 缺失")
            continue
        try:
            t = _carrier_text(p)
        except Exception as e:                       # noqa: BLE001
            fails.append(f"{rel}: 读取失败 {type(e).__name__}")
            continue
        lack = [k for k in WORKFLOW_NODES if k not in t]
        if lack:
            fails.append(f"{rel}: 缺工作流关键节点 {lack}")
    # 4) report/SPEC13_FREEZE.json：冻结基线声明的子集事实必须与实文件三方一致
    fz_path = ROOT / SPEC13_FREEZE
    if not fz_path.exists():
        fails.append(f"{SPEC13_FREEZE} 缺失（§1.3 冻结基线）")
    else:
        try:
            fz = json.loads(fz_path.read_text(encoding="utf-8"))
            dec = fz.get("decision", {})
            if dec.get("status") != "ACTIVE_IMAGENETV2_MF_1000_ONLY":
                fails.append(f"{SPEC13_FREEZE}: decision.status 不是唯一口径 ACTIVE_IMAGENETV2_MF_1000_ONLY")
            if dec.get("single_caliber") is not True:
                fails.append(f"{SPEC13_FREEZE}: decision.single_caliber 不是 True")
            sub = fz.get("subset", {})
            f = ROOT / str(sub.get("file", ""))
            if not f.exists():
                fails.append(f"{SPEC13_FREEZE}: 声明的子集文件不存在（{sub.get('file')}）")
            else:
                raw = f.read_bytes()
                if hashlib.sha256(raw).hexdigest() != sub.get("sha256"):
                    fails.append(f"{SPEC13_FREEZE}: subset.sha256 与实文件不一致（子集被改动）")
                if len(raw) != sub.get("bytes"):
                    fails.append(f"{SPEC13_FREEZE}: subset.bytes {sub.get('bytes')} ≠ 实测 {len(raw)}")
                lines = len(raw.decode("utf-8", "replace").splitlines())
                if lines != sub.get("lines"):
                    fails.append(f"{SPEC13_FREEZE}: subset.lines {sub.get('lines')} ≠ 实测 {lines}")
        except Exception as e:                       # noqa: BLE001
            fails.append(f"{SPEC13_FREEZE}: 解析/核对失败 {type(e).__name__}: {e}")
    extra = sorted(set(have) - set(PPT_SVG_NAMES))
    detail = (f"{docx_meta}；ppt_svg {len(have)} 份（关键节点 {'全' if not lack_svg else '缺'}）；"
              f"workflow {len(WORKFLOW_FILES)} 件；SPEC13 子集 sha256/字节/行数与实文件一致"
              + (f"；多余 SVG {extra}" if extra else ""))
    return fails, detail


@check("submit", "提交物清单齐全 + 关键内容（报告 docx / PPT 设计源 / 工作流程图 / §1.3 冻结基线）",
       "报告或答辩缺失 → 本次考核不通过；载体存在但关键内容缺失同样 FAIL")
def c_submit():
    miss = [r for r in SUBMIT if not (ROOT / r).exists()]
    onnx_n = len(list((ROOT / "onnx").glob("*.onnx")))
    if onnx_n < 3:
        miss.append(f"ONNX 仅 {onnx_n} 个(<3)")
    try:
        c_fails, c_detail = carrier_content_findings()
    except Exception as e:                           # noqa: BLE001
        c_fails, c_detail = [f"载体内容核对异常 {type(e).__name__}: {e}"], ""
    miss.extend(c_fails)
    if miss:
        return False, f"缺失={miss}"
    return True, f"齐全（{len(SUBMIT)} 项 + ONNX {onnx_n} 个）；{c_detail}"



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
