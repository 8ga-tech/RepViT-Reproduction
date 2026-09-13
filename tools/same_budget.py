# -*- coding: utf-8 -*-
"""tools/same_budget.py —— 两组实验的训练预算等价性核对（对应 DoD #24）。

用法：
    python tools/same_budget.py --a outputs/logs/baseline_metrics.csv \
                                --b outputs/logs/opt_metrics.csv

为什么要单独一个脚本：Baseline 与优化实验必须「同预算」，否则精度差异里混进了
「谁多训了几轮」这个混杂因子，任何消融结论都不成立。预算等价有两个层次：
  (1) epochs 行数相等（逐 epoch CSV 的数据行数）；
  (2) 每 epoch 的迭代数相等（= 前向/反向次数，才是真正的算力预算）。

迭代数从哪里来（按可信度排序，能取到就用，取不到就明确说"不可得"，绝不猜）：
  A. CSV 里若有 step / iter / global_step 列 —— 直接用它（本仓 train.py 的 CSV 头
     里没有这一列，见 tools/train.py 的 CSV_HEADER；第三方 CSV 可能有）；
  B. 同目录的 `<experiment_name>_steps.jsonl`（train.py 的逐步日志）—— 按 epoch 分组
     取每 epoch 最大 step。注意它是**按 log_interval 抽样**写的，只在 log_interval
     整除步数时才等于真实步数，所以脚本会把它标注为 sampled，仅作旁证；
  C. 同目录的 `<experiment_name>_config_effective.json` —— 由 train_list 行数、
     batch_size、drop_last 算出 steps_per_epoch = len(list) // batch_size（drop_last）
     或 ceil(len(list) / batch_size)。这是本仓最可靠的口径。

判定（规格书 §DoD #24）：`total_iters_equal=True`，或差异 <1% 且有书面解释。
  * 差异 >= 1% -> 打印 [WARN] 并 exit(0)（规格书允许"差异<1% 或有书面解释"，
    所以这不算脚本失败）；
  * 差异 >= 1% 且没给 --explain -> 额外打印「必须补一份书面解释」的明确提示。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEP_COLUMNS = ("step", "iter", "iters", "global_step", "global_steps", "total_step", "total_steps")
WARN_PCT = 1.0


def _local_dump_metrics(name: str, payload: dict, out_dir: str = "outputs/metrics") -> str:
    import datetime
    import os
    import platform

    p = Path(out_dir)
    out_dir = str(p if p.is_absolute() else (ROOT / p))
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "experiment": name,
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "cwd_rel": ".",
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**meta, **payload}, f, ensure_ascii=True, indent=2)
    print(f"[metrics] wrote {path}")
    return path


try:  # pragma: no cover
    from utils.logging import dump_metrics as _dump_metrics  # type: ignore
except Exception:  # noqa: BLE001
    _dump_metrics = _local_dump_metrics


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------
def read_metrics_csv(path: Path) -> dict:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} 没有数据行")
    fields = list(rows[0].keys())
    step_col = next((c for c in fields if c and c.strip().lower() in STEP_COLUMNS), None)
    return {"path": path, "fields": fields, "n_rows": len(rows), "step_col": step_col,
            "rows": rows}


def read_sibling_json(stem_dir: Path, exp: str, suffix: str):
    p = stem_dir / f"{exp}{suffix}"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")), p
        except Exception:  # noqa: BLE001
            return None, p
    return None, p


def count_list_lines(list_path_rel: str) -> int | None:
    if not list_path_rel:
        return None
    p = _resolve(list_path_rel)
    if not p.exists():
        return None
    n = 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def iters_from_steps_jsonl(path: Path) -> dict | None:
    """按 epoch 分组，取每 epoch 的最大 step（抽样日志的旁证口径）。"""
    if not path.exists():
        return None
    per_epoch: dict[int, int] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                ep, st = int(o.get("epoch", -1)), int(o.get("step", 0))
                if ep < 0:
                    continue
                per_epoch[ep] = max(per_epoch.get(ep, 0), st)
    except Exception:  # noqa: BLE001
        return None
    if not per_epoch:
        return None
    vals = sorted(per_epoch.values())
    return {"path": path, "per_epoch_max_step": per_epoch,
            "median_steps_per_epoch": vals[len(vals) // 2],
            "total_iters_sampled": int(sum(per_epoch.values()))}


def analyze(csv_path: Path) -> dict:
    info = read_metrics_csv(csv_path)
    log_dir = csv_path.parent
    exp = csv_path.stem
    for suf in ("_metrics",):
        if exp.endswith(suf):
            exp = exp[: -len(suf)]

    cfg, cfg_path = read_sibling_json(log_dir, exp, "_config_effective.json")
    steps = iters_from_steps_jsonl(log_dir / f"{exp}_steps.jsonl")

    n_epochs = int(info["n_rows"])
    res = {
        "experiment": exp,
        "csv": csv_path.as_posix(),
        "csv_epochs": n_epochs,
        "step_col": info["step_col"],
        "batch_size": None,
        "drop_last": None,
        "train_list": None,
        "train_list_len": None,
        "cfg_epochs": None,
        "steps_per_epoch": None,
        "steps_source": None,
        "total_iters": None,
        "total_iters_source": None,
        "config_effective": cfg_path.as_posix() if cfg_path.exists() else None,
        "steps_jsonl": (steps or {}).get("path", Path()).as_posix() if steps else None,
        "steps_jsonl_median_per_epoch": (steps or {}).get("median_steps_per_epoch"),
    }

    # --- A. CSV 里的 step 列 ---
    if info["step_col"]:
        col = info["step_col"]
        vals = [int(float(r[col])) for r in info["rows"] if r.get(col) not in (None, "")]
        if vals:
            per_epoch = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
            res["steps_per_epoch"] = (per_epoch[0] if per_epoch else vals[0])
            res["total_iters"] = int(vals[-1])
            res["total_iters_source"] = f"csv 列 {col} 的末值"
            res["steps_source"] = f"csv 列 {col} 的逐行差分（中位 {sorted(per_epoch)[len(per_epoch) // 2] if per_epoch else 'n/a'}）"

    # --- C. config_effective.json 推算 ---
    if cfg:
        d, t = cfg.get("data", {}) or {}, cfg.get("train", {}) or {}
        res["batch_size"] = d.get("batch_size")
        res["drop_last"] = d.get("drop_last")
        res["train_list"] = d.get("train_list")
        res["cfg_epochs"] = t.get("epochs")
        n = count_list_lines(d.get("train_list"))
        res["train_list_len"] = n
        if n and res["batch_size"]:
            if res["drop_last"]:
                spe = n // int(res["batch_size"])
            else:
                spe = math.ceil(n / int(res["batch_size"]))
            res["spe_from_cfg"] = spe
            if res["total_iters"] is None:
                res["steps_per_epoch"] = spe
                res["total_iters"] = spe * n_epochs
                res["total_iters_source"] = (
                    f"config_effective: {n} 行 train_list / batch_size {res['batch_size']} "
                    f"（drop_last={res['drop_last']}）* {n_epochs} epochs")
                res["steps_source"] = "config_effective.json 推算"

    if res["total_iters"] is None:
        # 兜底：预算退化为「epoch 数」本身，并明确标注不可得
        res["total_iters"] = n_epochs
        res["total_iters_source"] = "不可得（无 step 列、无 config_effective.json），退化为 epoch 数"
        res["steps_source"] = "不可得"
    return res


def main() -> int:
    ap = argparse.ArgumentParser("两组实验训练预算等价性核对")
    ap.add_argument("--a", required=True, help="A 组（如 baseline）逐 epoch metrics CSV")
    ap.add_argument("--b", required=True, help="B 组（如 opt）逐 epoch metrics CSV")
    ap.add_argument("--explain", default=None,
                    # argparse 的 help 走 printf 风格格式化，'%' 必须写成 '%%'
                    help="差异>=1%% 时的书面解释（一句话；亦可用 @文件路径 指向解释文件）")
    ap.add_argument("--out-dir", default="outputs/metrics")
    ap.add_argument("--json-name", default="same_budget")
    args = ap.parse_args()

    pa, pb = _resolve(args.a), _resolve(args.b)
    for p in (pa, pb):
        if not p.exists():
            print(f"[ERROR] 找不到 metrics CSV：{p}")
            print("        提示：先分别跑 baseline 与优化实验，产物在 "
                  "outputs/logs/<experiment_name>_metrics.csv")
            return 1

    print("=" * 84)
    print(" same_budget —— 训练预算等价性（DoD #24）")
    a, b = analyze(pa), analyze(pb)
    for tag, r in (("A", a), ("B", b)):
        print("-" * 84)
        print(f" [{tag}] exp={r['experiment']}  csv={r['csv']}")
        print(f"     epochs(CSV 数据行) = {r['csv_epochs']}"
              + (f"   config.train.epochs = {r['cfg_epochs']}"
                 if r["cfg_epochs"] is not None else ""))
        print(f"     batch_size = {r['batch_size']}   drop_last = {r['drop_last']}   "
              f"train_list = {r['train_list']} ({r['train_list_len']} 行)")
        print(f"     steps/epoch = {r['steps_per_epoch']}   来源 = {r['steps_source']}")
        print(f"     total_iters = {r['total_iters']}   来源 = {r['total_iters_source']}")
        if r["steps_jsonl_median_per_epoch"] is not None:
            print(f"     [旁证] {r['steps_jsonl']} 每 epoch 最大 step 的中位数 = "
                  f"{r['steps_jsonl_median_per_epoch']}（抽样日志，非逐 step，仅作旁证）")
        else:
            print(f"     [旁证] 无 <exp>_steps.jsonl（或不可解析）")

    print("=" * 84)
    epochs_equal = a["csv_epochs"] == b["csv_epochs"]
    spe_equal = (a["steps_per_epoch"] == b["steps_per_epoch"]) \
        if (a["steps_per_epoch"] is not None and b["steps_per_epoch"] is not None) else None
    ta, tb = int(a["total_iters"]), int(b["total_iters"])
    denom = max(abs(ta), abs(tb), 1)
    diff_pct = abs(ta - tb) / denom * 100.0
    total_iters_equal = (ta == tb)

    print(f" [epochs]  A={a['csv_epochs']}  B={b['csv_epochs']}  equal={epochs_equal}")
    print(f" [steps/epoch]  A={a['steps_per_epoch']}  B={b['steps_per_epoch']}  "
          f"equal={spe_equal if spe_equal is not None else 'unknown（两侧均不可得）'}")
    print(f" [total_iters]  A={ta}  B={tb}  diff={ta - tb:+d} ({diff_pct:.4f}%)")
    print(f" total_iters_equal={total_iters_equal}")

    if b["batch_size"] is not None and a["batch_size"] is not None and a["batch_size"] != b["batch_size"]:
        print(f" [WARN] batch_size 不一致（A={a['batch_size']} B={b['batch_size']}）："
              f"同预算通常要求同 batch_size，请确认这是有意为之。")
    if epochs_equal and spe_equal is False:
        print(f" [WARN] epoch 数相同但每 epoch 迭代数不同：预算并不等价。")

    if total_iters_equal:
        print("[PASS] 两组训练预算逐迭代一致（total_iters_equal=True）。")
        passed = True
    elif diff_pct < WARN_PCT:
        print(f"[PASS] 两组训练预算差异 {diff_pct:.4f}% < {WARN_PCT}%，规格书允许。")
        passed = True
    else:
        passed = False
        print(f"[WARN] 两组训练预算差异 {diff_pct:.4f}% >= {WARN_PCT}%："
              f"必须在报告里给出一份书面解释（规格书 DoD #24 允许「差异<1% 或有书面解释」）。")
        if args.explain:
            print(f"[WARN] 已提供的书面解释：{args.explain}")
        else:
            print("[WARN] 未提供 --explain：请补一份书面解释，"
                  "例如 `--explain \"B 组每 epoch 多 2 个 step，因为开启了 RandomErasing 后重采样\".`")
            print("       注意：本脚本按规格书仍 exit(0)，但解释缺失必须由人工补齐。")

    payload = {
        "a": {k: v for k, v in a.items()},
        "b": {k: v for k, v in b.items()},
        "epochs_equal": bool(epochs_equal),
        "steps_per_epoch_equal": spe_equal,
        "total_iters_equal": bool(total_iters_equal),
        "total_iters_diff": int(ta - tb),
        "total_iters_diff_pct": float(diff_pct),
        "warn_threshold_pct": WARN_PCT,
        "explain": args.explain,
        "passed": bool(passed),
        "note": "预算以 total_iters 为准；不可得时退化为 epoch 数并显式标注（绝不猜）",
    }
    _dump_metrics(args.json_name, payload, out_dir=str(_resolve(args.out_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
