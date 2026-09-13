# -*- coding: utf-8 -*-
"""tools/check_cm.py —— 混淆矩阵形状 / 行归一化 / 对角线自证（对应 DoD #27）。

用法：
    python tools/check_cm.py --file outputs/confusion_matrix/baseline_cm.csv

契约（形状与归一化方式都被下游报告引用，属于冻结接口）：
  * 形状必须是 (37, 37)：37 = Pet 的类别数。矩阵维度一旦缩水（sklearn 的
    confusion_matrix 不传 labels= 时，未出现的类会被丢掉），类别名就会整体错位，
    这是最隐蔽的静默 bug，必须在入口处断言掉。
  * 按行归一化：cm[i, j] = P(pred=j | true=i)，行和恒为 1（容差 1e-6）。
    support == 0 的类（该真实类一张图都没有）行和为 0，这是**允许**的，
    单独计数，不计入违约。
  * 额外打印 argmax 落在对角线上的比例（对 support>0 的行统计），它是「按行归一化后
    每类最可能被预测成什么」的直观指标，也顺带验证对角占优。

形状不是 37x37 时 exit(1)。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


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


N_CLASSES = 37
ROW_SUM_TOL = 1e-6


def load_cm(path: Path) -> np.ndarray:
    """读 CSV 成 numpy，兼容三种写法：

      (a) 纯数值矩阵（np.savetxt 写出，可能带一行 `col` 之类的注释表头）；
      (b) pandas 带行列名（tools/visualize.py 的写法：
          `pd.DataFrame(cm, index=names, columns=names).to_csv(...)`，
          首列是类别名，行数是 38 列）；
      (c) 首列为空的行名列（index=True 且 index 无名字）。

    逐行判定：能整行转 float 的按 (a) 收；否则去掉首列再试，成功按 (b)/(c) 收；
    两者都失败说明是表头/注释行，跳过。绝不能无脑 `np.loadtxt`：丢一行表头可以，
    丢一列类别名会让整个矩阵错位。
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            # pandas 的 to_csv 表头形如 ',0,1,2,...'：首列为空的行名列 —— 必须跳过，
            # 否则它会被当成一行合法数据（剩下的列号恰好都是数字）混进矩阵。
            if len(parts) > 1 and parts[0].strip() == "":
                continue
            try:
                rows.append([float(x) for x in parts])
                continue
            except ValueError:
                pass
            if len(parts) > 1:
                try:
                    rows.append([float(x) for x in parts[1:]])   # 去掉非数值的行名列
                    continue
                except ValueError:
                    pass
            # 表头 / 注释行：跳过
    if not rows:
        raise ValueError(f"{path} 里没有解析出任何数值行")
    # 兜底：按「出现次数最多的列数」取矩阵，零星宽度不一致的行（残余表头）丢掉并告警，
    # 避免一行脏数据把整个矩阵判成坏文件。
    from collections import Counter
    widths = Counter(len(r) for r in rows)
    width = widths.most_common(1)[0][0]
    if len(widths) > 1:
        dropped = sum(c for w, c in widths.items() if w != width)
        print(f"[warn] CSV 里存在 {len(widths)} 种列数 {dict(widths)}；"
              f"按主流列数 {width} 取矩阵，丢弃 {dropped} 行（多半是残余表头）")
        rows = [r for r in rows if len(r) == width]
    else:
        # 整数行号（pandas 未命名的 index）会被当成一列数值收进来，表现为「37 行 × 38 列
        # 且首列恰为 0..36」。必须剥掉，否则形状断言会以一个假的 38 列失败。
        if len(rows) == 37 and width == 38 and all(r[0] == i for i, r in enumerate(rows)):
            print("[warn] 首列是 0..36 的整数行号（pandas 未命名 index），已自动剥离该列")
            rows = [r[1:] for r in rows]
    return np.asarray(rows, dtype=np.float64)


_DEC_RE = re.compile(r"\.(\d+)")


def detect_decimals(path: Path) -> int:
    """从原始文本里探测最多的小数位数，用于把「写文件的精度」算进容差。

    visualize.py 用 `float_format="%.6f"` 落盘：37 个元素各截断到 1e-6，行和的累积
    误差可达 37 × 0.5e-6 ≈ 1.85e-5。若死守 1e-6 容差，一个完全正确的行归一化矩阵
    反而会被判违约 —— 这是把「量化误差」错当成「没归一化」。
    """
    best = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            for m in _DEC_RE.finditer(line):
                best = max(best, len(m.group(1)))
    return best


def main() -> int:
    ap = argparse.ArgumentParser("混淆矩阵形状 / 行归一化核对")
    ap.add_argument("--file", required=True, help="行归一化混淆矩阵 CSV")
    ap.add_argument("--n-classes", type=int, default=N_CLASSES)
    ap.add_argument("--tol", type=float, default=ROW_SUM_TOL)
    ap.add_argument("--out-dir", default="outputs/metrics")
    ap.add_argument("--json-name", default="check_cm")
    args = ap.parse_args()

    path = _resolve(args.file)
    if not path.exists():
        print(f"[ERROR] 找不到混淆矩阵文件：{path}")
        print("        提示：先跑 `python tools/visualize.py --pred-csv <preds.csv> "
              "--out-dir outputs --tag <experiment_name>` 生成 "
              "outputs/confusion_matrix/<experiment_name>_cm.csv")
        return 1

    print("=" * 80)
    print(" check_cm —— 混淆矩阵形状 / 行归一化 / 对角占优（DoD #27）")
    print(f" file = {args.file}")
    cm = load_cm(path)
    print(f" parsed shape={cm.shape}  dtype={cm.dtype}")

    if cm.shape != (args.n_classes, args.n_classes):
        print(f"[FAIL] 形状必须是 ({args.n_classes}, {args.n_classes})，实际 {cm.shape}。")
        print("       可能原因：confusion_matrix 没传 labels=list(range(37))，"
              "某个类从未出现导致维度缩水 -> 类别名整体错位。")
        return 1

    row_sum = cm.sum(axis=1)
    zero_rows = np.where(row_sum == 0.0)[0]
    nonzero = row_sum[row_sum != 0.0]
    # 零样本行（该真实类在验证/测试集里一张图都没有）行和恒为 0，按契约**允许**，
    # 单独计数、不计入违约；只有 support>0 却偏离 1.0 的行才算违约。
    decimals = detect_decimals(path)
    tol_round = args.n_classes * 0.5 * (10 ** -decimals) if decimals else 0.0
    tol_eff = max(float(args.tol), float(tol_round))

    def bad_rows(tol: float) -> np.ndarray:
        return np.array([i for i in range(cm.shape[0])
                         if row_sum[i] != 0.0 and abs(row_sum[i] - 1.0) > tol], dtype=int)

    bad_strict = bad_rows(float(args.tol))
    bad = bad_rows(tol_eff)

    print("-" * 80)
    print(f" row_sum: min={row_sum.min():.12f}  max={row_sum.max():.12f}  "
          f"mean={row_sum.mean():.12f}")
    print(f" 零样本行（support==0，行和=0，允许）: {len(zero_rows)} 行"
          + (f" -> idx {zero_rows.tolist()}" if len(zero_rows) else ""))
    print(f" 非零行 row_sum: min={nonzero.min() if nonzero.size else float('nan'):.12f}  "
          f"max={nonzero.max() if nonzero.size else float('nan'):.12f}  "
          f"mean={nonzero.mean() if nonzero.size else float('nan'):.12f}  (容差 {args.tol:g})")
    print(f" 契约容差 {args.tol:g}；本文件小数位 {decimals}，量化容差 "
          f"{tol_round:.3e}，判定用容差 {tol_eff:.3e}"
          f"（量化误差不是「没归一化」，见脚本注释）")
    print(f" 违约行数：按契约容差 {len(bad_strict)} 行；按量化容差 {len(bad)} 行")

    # argmax 落对角线比例：只对 support>0 的行有意义
    argmax_all = cm.argmax(axis=1)
    diag_hit_all = int(np.sum(argmax_all == np.arange(cm.shape[0])))
    n_all = cm.shape[0]
    if nonzero.size:
        rows_nz = np.where(row_sum != 0.0)[0]
        diag_hit_nz = int(np.sum(argmax_all[rows_nz] == rows_nz))
        n_nz = rows_nz.size
    else:
        diag_hit_nz, n_nz = 0, 0
    print(f" argmax 落在对角线上的比例（全部 {n_all} 行）: {diag_hit_all}/{n_all} "
          f"= {diag_hit_all / n_all * 100:.2f}%")
    print(f" argmax 落在对角线上的比例（仅 support>0 的 {n_nz} 行）: {diag_hit_nz}/{n_nz} "
          f"= {(diag_hit_nz / n_nz * 100) if n_nz else 0.0:.2f}%")
    print(f" 对角元均值（全部行）: {float(np.diag(cm).mean()):.6f}")

    ok = len(bad) == 0
    if ok:
        extra = (f"；其中有 {len(bad_strict)} 行在严格 {args.tol:g} 下超出，"
                 f"但偏差在「%.{decimals}f 落盘」的量化误差 {tol_round:.2e} 之内，属正常"
                 if len(bad_strict) else "")
        print(f"[PASS] 形状 ({args.n_classes}, {args.n_classes})；所有非零行 row_sum≈1.0"
              f"（零样本行 {len(zero_rows)} 行单独计数）{extra}。")
    else:
        print(f"[FAIL] 有 {len(bad)} 行 row_sum 偏离 1.0 超过容差 {tol_eff:.3e}："
              f"idx {bad.tolist()[:10]}")
        print(f"       对应 row_sum = {row_sum[bad][:10].tolist()}")

    payload = {
        "file": args.file,
        "shape": list(cm.shape),
        "expected_shape": [args.n_classes, args.n_classes],
        "shape_ok": bool(cm.shape == (args.n_classes, args.n_classes)),
        "row_sum": {
            "min": float(row_sum.min()), "max": float(row_sum.max()),
            "mean": float(row_sum.mean()), "tol_contract": float(args.tol),
            "file_decimals": int(decimals),
            "tol_rounding": float(tol_round),
            "tol_effective": float(tol_eff),
            "n_zero_rows": int(len(zero_rows)),
            "zero_row_idx": zero_rows.tolist(),
            "n_bad_rows_contract_tol": int(len(bad_strict)),
            "n_bad_rows": int(len(bad)),
            "bad_row_idx": bad.tolist()[:50],
        },
        "argmax_on_diagonal": {
            "all_rows": {"hit": diag_hit_all, "n": int(n_all),
                         "ratio": diag_hit_all / n_all},
            "support_gt0_rows": {"hit": diag_hit_nz, "n": int(n_nz),
                                 "ratio": (diag_hit_nz / n_nz) if n_nz else 0.0},
        },
        "diag_mean": float(np.diag(cm).mean()),
        "pass": bool(ok),
        "note": "零样本行（support==0）行和恒为 0，按契约允许并单独计数，不计入违约；"
                "判定容差 = max(契约 1e-6, 37*0.5*10^-文件小数位)，量化误差不算违约",
    }
    _dump_metrics(args.json_name, payload, out_dir=str(_resolve(args.out_dir)))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
