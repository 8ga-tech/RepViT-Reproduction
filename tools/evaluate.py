# tools/evaluate.py
# -*- coding: utf-8 -*-
"""自训练 checkpoint 的最终评价工具（规格书 1845-1856 行 API 契约）。

职责：载入 ckpt 在 val/test 上出指标与混淆矩阵，是「最终一次 test 评价」的执行者。

用法:
    python tools/evaluate.py --cfg configs/baseline.yaml \
        --ckpt checkpoints/baseline_best.pt --split test
    python tools/evaluate.py --cfg configs/baseline.yaml \
        --ckpt checkpoints/baseline_best.pt --split val --latency

产物（路径来自配置，不写死）:
    outputs/metrics/<experiment_name>_<split>.json        指标（top1/top5/macro_f1 为 0~1 小数）
    outputs/predictions/<experiment_name>_<split>_preds.csv  逐图预测（列名与列序是唯一真源）
    outputs/metrics/pytorch_latency.json                  --latency 时额外产出

依赖（由 M02/M04 交付，本文件惰性导入，缺失时给出可执行的提示而不是 ImportError 堆栈）:
    models.build_model.build_model(cfg)          -> nn.Module
    datasets.build.build_eval_loader(cfg, split) -> DataLoader，yield (x, y, image_id[, path])
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import apply_overrides, load_yaml        # noqa: E402

# predictions.csv 的列名与列序：全仓唯一契约（与 outputs/pretrained_eval/<m>/predictions.csv
# 不是同一份 schema，两边的解析代码绝不可互换）。后三列是 JSON 字符串。
PRED_COLUMNS = ["image_id", "path", "true_idx", "true_name", "pred_idx", "pred_name",
                "prob", "correct", "top5_idx", "top5_names", "top5_probs"]

# outputs/metrics/<exp>_<split>.json 的契约键（值 top1/top5/macro_f1 一律 0~1 小数）
METRIC_KEYS = ["experiment_name", "checkpoint", "ckpt_sha256", "split", "num_samples",
               "num_classes", "top1", "top5", "macro_f1", "per_class_f1", "top1_count",
               "evaluated_at", "seed"]


# --------------------------------------------------------------------------- #
# 1. CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser("RepViT 自训练 ckpt 评价（--cfg + --ckpt + --set）")
    ap.add_argument("--cfg", required=True, help="YAML 配置，例 configs/baseline.yaml")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint 路径，例 checkpoints/baseline_best.pt；--audit 模式下不需要")
    ap.add_argument("--audit", action="store_true",
                    help="只做数据审计（标签范围/每类计数/train 是否全部来自官方 trainval），"
                         "不加载模型；产出 outputs/metrics/pet_label_audit.json 与 split_audit.json")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    # action="extend"：多批 --set 拼接而不是互相覆盖（见 tools/eval_pretrained.py 同名注释）
    ap.add_argument("--set", nargs="*", default=[], dest="set", action="extend",
                    help="点号路径覆盖，可给多批")
    ap.add_argument("--latency", action="store_true",
                    help="额外测 PyTorch 单图延迟并写 outputs/metrics/pytorch_latency.json")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--threads", type=int, default=None,
                    help="固定 CPU 线程数；与 ONNX 基准横向对比时必须显式指定")
    return ap.parse_args(argv)


def _import_callable(module: str, attr: str):
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        raise SystemExit(f"[错误] 无法导入 {module}（{e}）。该模块由仓库骨架/模型构建环节交付，"
                         f"请确认 {module.replace('.', '/')}.py 已存在且仓库根在 sys.path 中") from e
    fn = getattr(mod, attr, None)
    if fn is None:
        raise SystemExit(f"[错误] {module} 里没有 {attr}（API 契约见规格书，不得改名）")
    return fn


# --------------------------------------------------------------------------- #
# 2. 指标
# --------------------------------------------------------------------------- #
def _to_paths(record: str) -> str:
    """写进 CSV 的 path 一律相对仓库根；不在仓库内则原样写出。"""
    p = Path(record)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return p.as_posix()


def image_id_to_path(cfg: dict, split: str) -> dict:
    """从冻结的 split list 反查 image_id -> 图片路径（相对仓库根）。

    兜底用：当 loader 只 yield (x, y, image_id) 时，CSV 的 path 列仍有真值可写。
    解析一律 rsplit(None, 1)，兼容 tab 与空格，且不会切错含空格的路径。
    """
    list_file = ROOT / cfg["data"][f"{split}_list"]
    images_dir = ROOT / cfg["data"]["root"] / cfg["data"].get("images_subdir", "images")
    mapping = {}
    if not list_file.exists():
        return mapping
    for line in list_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        rel = parts[0].strip()
        if not rel.lower().endswith((".jpg", ".jpeg", ".png")):
            rel = f"{rel}.jpg"
        mapping[Path(parts[0].strip()).stem] = images_dir / rel
    return mapping


def _split_weights(batch):
    """兼容 loader 的两种产出：(x, y, image_id) 与 (x, y, image_id, path)。"""
    if len(batch) >= 4:
        x, y, ids, paths = batch[0], batch[1], batch[2], batch[3]
    else:
        x, y, ids = batch[0], batch[1], batch[2]
        paths = None
    return x, y, ids, paths


@torch.no_grad()
def run_eval(model, loader, device, class_names: list, num_classes: int) -> dict:
    """在给定 loader 上跑一遍，返回指标与逐样本明细。

    model.eval() 必写：否则 BN 用 batch 统计量、dropout 生效，指标随机偏低。
    top1/top5/macro_f1 全部是 **0~1 小数**（与 outputs/metrics/*.json 的口径一致；
    只有 outputs/pretrained_eval/<m>/metrics.json 用百分数，两者不可混用）。
    """
    model.eval()
    logits_all, y_all, ids_all, paths_all = [], [], [], []
    for batch in loader:
        x, y, ids, paths = _split_weights(batch)
        x = x.to(device, non_blocking=True)
        logits = model(x)
        if isinstance(logits, (tuple, list)):            # 兜底：蒸馏双头未关
            logits = (logits[0] + logits[1]) / 2
        logits_all.append(logits.float().cpu())
        y_all.append(y.cpu())
        ids_all.extend(list(ids) if not torch.is_tensor(ids) else ids.tolist())
        if paths is not None:
            paths_all.extend(list(paths))

    logits = torch.cat(logits_all)
    y = torch.cat(y_all).numpy().astype(np.int64)
    n, K = len(y), int(num_classes)
    probs = torch.softmax(logits, dim=1).numpy()
    pred = logits.argmax(1).numpy().astype(np.int64)

    maxk = min(5, logits.size(1))
    topk_idx = np.argsort(-logits.numpy(), axis=1)[:, :maxk]
    top1_count = int((pred == y).sum())
    top5_count = int(sum(y[i] in topk_idx[i] for i in range(n)))
    cm = confusion_matrix(y, pred, labels=list(range(K)))
    support = cm.sum(1)
    per_class_recall = (np.diag(cm) / np.maximum(support, 1)).tolist()
    macro_f1 = float(f1_score(y, pred, labels=list(range(K)), average="macro", zero_division=0))
    per_class_f1 = f1_score(y, pred, labels=list(range(K)), average=None,
                            zero_division=0).tolist()

    res = {
        "top1": top1_count / max(n, 1),
        "top5": top5_count / max(n, 1),
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "per_class_recall": per_class_recall,
        "cm": cm,
        "y_true": y,
        "y_pred": pred,
        "probs": probs,
        "image_ids": ids_all,
        "paths": paths_all,
        "top1_count": top1_count,
        "num_samples": n,
        "num_classes": K,
        "class_names": list(class_names),
        "loss": float(torch.nn.functional.cross_entropy(logits, torch.from_numpy(y)).item()),
        "topk_idx": topk_idx,
    }
    print(f"[eval] n={n} top1={res['top1']*100:.2f}% top5={res['top5']*100:.2f}% "
          f"macroF1={res['macro_f1']*100:.2f}%（K={K}，随机水平 {100.0/max(K,1):.2f}%）")
    return res


# --------------------------------------------------------------------------- #
# 3. 落盘
# --------------------------------------------------------------------------- #
def dump_metrics_json(metrics: dict, out_path) -> Path:
    """写 outputs/metrics/<experiment_name>_<split>.json。

    只落契约键（METRIC_KEYS），混淆矩阵/概率矩阵这类大数组不进 JSON——它们是
    visualize.py 的输入，不是报告指标。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing = [k for k in METRIC_KEYS if k not in metrics]
    if missing:
        raise SystemExit(f"[错误] metrics 缺少契约键 {missing}，拒绝写出半可信的 JSON")
    payload = {}
    for k in METRIC_KEYS:
        v = metrics[k]
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        payload[k] = float(v) if isinstance(v, float) else v
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[write] {_to_paths(str(out_path))}  top1={payload['top1']:.4f} "
          f"top5={payload['top5']:.4f} macro_f1={payload['macro_f1']:.4f}")
    return out_path


def dump_predictions_csv(metrics: dict, out_path) -> Path:
    """写 outputs/predictions/<experiment_name>_<split>_preds.csv。

    列名与列序由 PRED_COLUMNS 固定；top5_idx/top5_names/top5_probs 三列是 **JSON 字符串**
    （形如 `[12, 15, 13, 14, 16]`），prob 是 0~1 小数。encoding 用 utf-8（不带 BOM），
    与 tools/train.py 的 test_preds.csv 保持一致，避免下游 pandas 读出头列带 BOM。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = metrics["class_names"]
    y_true, y_pred = metrics["y_true"], metrics["y_pred"]
    probs, topk = metrics["probs"], metrics["topk_idx"]
    ids, paths = metrics["image_ids"], metrics.get("paths") or [None] * len(y_true)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(PRED_COLUMNS)
        for i in range(len(y_true)):
            ti, pi = int(y_true[i]), int(y_pred[i])
            k = min(5, probs.shape[1])
            w.writerow([
                ids[i],
                _to_paths(paths[i]) if paths[i] else "",
                ti, names[ti], pi, names[pi],
                f"{float(probs[i, pi]):.6f}",
                int(ti == pi),
                json.dumps([int(v) for v in topk[i][:k]]),
                json.dumps([names[int(v)] for v in topk[i][:k]], ensure_ascii=False),
                json.dumps([round(float(probs[i, int(v)]), 6) for v in topk[i][:k]]),
            ])
    print(f"[write] {_to_paths(str(out_path))}  ({len(y_true)} 行 × {len(PRED_COLUMNS)} 列)")
    return out_path


def dump_latency_json(model, cfg: dict, device, out_path, warmup: int = 10, runs: int = 50,
                      threads: int | None = None) -> Path:
    """PyTorch 纯前向单图延迟（与 deploy/benchmark.py 的 ONNX 侧字段同名，便于同一套代码解析）。"""
    from tools.eval_pretrained import benchmark_latency          # 同源实现，避免两份口径
    input_size = int(cfg["data"]["input_size"])
    lat = benchmark_latency(model, (3, input_size, input_size), warmup=warmup, runs=runs,
                            device=str(device), threads=threads, amp=False)
    lat.update({"backend": "pytorch", "experiment_name": cfg["experiment_name"],
                "input_size": input_size, "batch_size": 1, "precision": "fp32"})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(lat, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[lat]  mean={lat['mean_ms']:.2f}ms p50={lat['p50_ms']:.2f}ms p95={lat['p95_ms']:.2f}ms "
          f"(threads={lat['threads']}, warmup={warmup}, runs={runs}) -> {_to_paths(str(out_path))}")
    return out_path


# --------------------------------------------------------------------------- #
# 4. 主流程
# --------------------------------------------------------------------------- #
def main(cfg_path: str, ckpt: str, split: str = "val", overrides: list[str] | None = None,
         latency: bool = False, warmup: int = 10, runs: int = 50,
         threads: int | None = None) -> int:
    t0 = time.time()
    cfg = load_yaml(cfg_path)
    if overrides:
        apply_overrides(cfg, list(overrides))
    exp = cfg["experiment_name"]
    assert split in ("val", "test"), f"split 只能取 val/test，收到 {split}"

    ckpt_path = Path(ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    if not ckpt_path.exists():
        raise SystemExit(f"[错误] checkpoint 不存在: {ckpt_path}（先跑 tools/train.py）")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = int(cfg["model"]["num_classes"])

    # --- 模型：models.build_model.build_model(cfg) 是唯一入口，禁止在评价脚本里另建一份网络 ---
    build_model = _import_callable("models.build_model", "build_model")
    model = build_model(cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    missing, unexpected = model.load_state_dict(sd, strict=False)   # 契约要求 strict=False
    if missing or unexpected:
        print(f"[warn] 非严格加载：missing={len(missing)} unexpected={len(unexpected)}"
              f"（若 missing 含分类头键名，说明结构或 num_classes 不匹配）")
    model.to(device).eval()

    # --- 数据：datasets.build.build_eval_loader(cfg, split) 是唯一入口（val/test 共用 val 变换）---
    build_eval_loader = _import_callable("datasets.build", "build_eval_loader")
    loader = build_eval_loader(cfg, split)

    # --- 类别名：ckpt 里的 class_names 优先，其次配置里的 class_names 文件 ---
    class_names = state.get("class_names") if isinstance(state, dict) else None
    if not class_names:
        names_file = ROOT / cfg["data"]["class_names"]
        class_names = [l.strip() for l in names_file.read_text(encoding="utf-8").splitlines()
                       if l.strip()]
    assert len(class_names) == K, f"类别名 {len(class_names)} 个 != num_classes {K}"

    metrics = run_eval(model, loader, device, class_names, K)
    if not metrics["paths"]:
        # loader 只给了 image_id 时，用冻结的 split list 反查真实路径
        id2path = image_id_to_path(cfg, split)
        metrics["paths"] = [id2path.get(str(i), "") for i in metrics["image_ids"]]
    metrics.update({
        "experiment_name": exp,
        "checkpoint": _to_paths(str(ckpt_path)),
        "ckpt_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        "split": split,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(cfg.get("seed", 0)),
    })

    metric_dir = ROOT / cfg["eval"]["metric_dir"]
    pred_dir = ROOT / cfg["output"]["dir"] / "predictions"
    m_path = dump_metrics_json(metrics, metric_dir / f"{exp}_{split}.json")
    c_path = dump_predictions_csv(metrics, pred_dir / f"{exp}_{split}_preds.csv")

    if latency:
        # 必须在同一次运行、同一设备、同一 input_size 上测（规格书 5.5）
        dump_latency_json(model, cfg, device, metric_dir / "pytorch_latency.json",
                          warmup=warmup, runs=runs, threads=threads)

    print(f"[done] {exp} {split} 用时 {time.time() - t0:.1f}s -> "
          f"{_to_paths(str(m_path))}, {_to_paths(str(c_path))}")
    if split == "test":
        print("[warn] test 只允许被评价一次（题目第 26 页硬性限制）；"
              "本工具不做计数，请自行确认没有用 test 选模型/调参")
    return 0


# --------------------------------------------------------------------------- #
# 数据审计（--audit）：selfcheck 的 c_labels / c_source 两个检查依赖它
# --------------------------------------------------------------------------- #
def audit_data(cfg_path: str) -> int:
    """只读划分文件与官方标注，产出两份审计 JSON。不加载模型、不需要 ckpt。"""
    import json
    import collections

    cfg = load_yaml(cfg_path)          # 与 main() 用同一个加载器（容器内已 import）
    root = ROOT / cfg["data"]["root"]
    lists = {
        "train": ROOT / cfg["data"]["train_list"],
        "val": ROOT / cfg["data"]["val_list"],
        "test": ROOT / cfg["data"]["test_list"],
    }
    num_classes = int(cfg["model"]["num_classes"])

    per_split, meta = {}, {}
    for name, lf in lists.items():
        labels = []
        ids = []
        with open(lf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                iid, lab = line.rsplit(None, 1)     # 必须 rsplit：兼容空格与 tab
                ids.append(iid)
                labels.append(int(lab))
        cnt = collections.Counter(labels)
        per_split[name] = labels
        meta[name] = {"n": len(labels), "n_unique_ids": len(set(ids)),
                      "label_min": min(labels), "label_max": max(labels),
                      "n_classes_present": len(cnt), "min_count": min(cnt.values()),
                      "max_count": max(cnt.values())}

    # ---- pet_label_audit.json（契约：min / max / n_classes / min_count）----
    all_lab = per_split["train"] + per_split["val"] + per_split["test"]
    audit = {
        "min": min(all_lab), "max": max(all_lab), "n_classes": num_classes,
        "min_count": min(m["min_count"] for m in meta.values()),
        "per_split": meta,
        "note": ("标签 0-based（= 官方 CLASS-ID - 1）。漏减 1 会让标签出现 37，"
                 "CrossEntropyLoss 直接崩；误用 list.txt 第 4 列（BREED-ID）会让类别塌到 0..2。"),
    }
    (ROOT / "outputs/metrics").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/metrics/pet_label_audit.json").write_text(
        json.dumps(audit, ensure_ascii=True, indent=2), encoding="utf-8")

    # ---- split_audit.json（契约：train_outside_official）----
    official = set()
    for f in ("trainval.txt", "test.txt"):
        fp = root / "annotations" / f
        if fp.exists():
            with open(fp, encoding="utf-8") as fh:
                official |= {l.split()[0] for l in fh if l.strip()}
    outside = []
    for name, lf in lists.items():
        with open(lf, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                iid = line.rsplit(None, 1)[0]
                if iid not in official:
                    outside.append(f"{name}:{iid}")
    s_audit = {
        "official_ids": len(official),
        "train_outside_official": sum(1 for x in outside if x.startswith("train:")),
        "val_outside_official": sum(1 for x in outside if x.startswith("val:")),
        "test_outside_official": sum(1 for x in outside if x.startswith("test:")),
        "examples": outside[:10],
        "note": ("train 全部来自官方 trainval.txt 才说明划分脚本读的是正确的根目录；"
                 "若有越界项，通常是误读了 images/ 下的孤儿图（41 张无标注图）。"),
    }
    (ROOT / "outputs/metrics/split_audit.json").write_text(
        json.dumps(s_audit, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"[audit] labels: min={audit['min']} max={audit['max']} "
          f"n_classes={audit['n_classes']} min_count={audit['min_count']}")
    print(f"[audit] split : train_outside_official={s_audit['train_outside_official']}")
    print("[audit] -> outputs/metrics/pet_label_audit.json, outputs/metrics/split_audit.json")
    return 0


if __name__ == "__main__":
    a = parse_args()
    if a.audit:
        raise SystemExit(audit_data(a.cfg))
    if not a.ckpt:
        raise SystemExit("非 --audit 模式必须给 --ckpt")
    raise SystemExit(main(a.cfg, a.ckpt, a.split, a.set,
                          latency=a.latency, warmup=a.warmup, runs=a.runs, threads=a.threads))
