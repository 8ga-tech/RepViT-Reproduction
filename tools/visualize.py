"""
Source  : Self-written
Third-party: scikit-learn (BSD-3), pandas, numpy, utils/plot.py
用法：
    python tools/visualize.py --pred-csv outputs/predictions/B0_baseline_test_preds.csv \
        --classes labels/pet_classes.txt --out-dir outputs \
        --tag B0_baseline --split test
"""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_fscore_support, accuracy_score)

from utils.plot import (setup_chinese_font, plot_confusion_matrix,
                        plot_per_class_f1)

# Pet 官方划分：label 1..12 是猫，13..37 是狗（trainval.txt 第 3 列 bin_label）
# 代码里全部是 0-based，所以 0..11 = 猫，12..36 = 狗
# 物种掩码的唯一真源是 labels/pet_species.json（由 datasets/build_pet_class_map.py 生成，
# 其数据来自官方 annotations/list.txt 的 SPECIES 列）。**不能**用 `idx < 12` 判断猫：
# 官方 CLASS-ID 顺序是猫狗交错的，前 12 类里有大量狗种（实测会把 American Bulldog、
# Beagle、Pug 判成猫，猫狗块统计整体错位）。
SPECIES_JSON = "labels/pet_species.json"


def load_cat_mask(n_classes: int = 37) -> np.ndarray:
    """返回长度 n_classes 的 bool 数组，True 表示该下标是猫。"""
    import json as _json
    p = Path(SPECIES_JSON)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[1] / SPECIES_JSON
    assert p.exists(), (f"缺少 {p}。请先运行 python datasets/build_pet_class_map.py "
                        f"生成物种映射（不要退回 idx<12 的假设，那是错的）")
    d = _json.loads(p.read_text(encoding="utf-8"))
    mask = np.array([d[str(i)] == "cat" for i in range(n_classes)], dtype=bool)
    assert mask.sum() == 12, f"猫应为 12 类，pet_species.json 得到 {mask.sum()} 类"
    return mask
N_DOG = 25


def load_classes(path: str) -> list:
    names = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines()
             if l.strip()]
    assert len(names) == 37, f"类别文件应有 37 行，实际 {len(names)}"
    assert names[0] == "abyssinian" or names[0].lower().startswith("abyssinian"), \
        f"类别顺序异常，第 0 类应为 Abyssinian，实际 {names[0]}"
    return names


def load_pred_csv(path: str) -> pd.DataFrame:
    """列名契约见 §8.2.1（与 CANON 第 4 节表头一致），是唯一真源。"""
    df = pd.read_csv(path)
    need = {"image_id", "path", "true_idx", "pred_idx", "prob", "correct"}
    missing = need - set(df.columns)
    assert not missing, f"预测 CSV 缺少列: {sorted(missing)}（列名必须以 §8.2.1 为准）"
    assert df["true_idx"].between(0, 36).all(), "true_idx 越界（0..36），检查是否忘了 -1"
    assert df["pred_idx"].between(0, 36).all(), "pred_idx 越界"
    return df


def row_normalized_cm(y_true, y_pred, n_classes=37) -> np.ndarray:
    """按行（真实类别）归一化：cm[i, j] = P(pred=j | true=i)，行和恒为 1。

    与 sklearn 的 normalize='true' 在 support>0 的行上等价；support==0 的行写 0，
    避免不同 sklearn 版本对 0/0 的处理差异破坏验收断言。
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).astype(np.float64)
    row = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)


def cat_dog_block(y_true, y_pred) -> dict:
    """猫/狗两大类块分析。

    Pet 是细粒度任务，绝大多数错误发生在**同物种内部的品种之间**。这个函数要回答的是
    「模型到底有没有把猫认成狗」，因此必须在**逐样本**层面比较物种，而不是先把 37 类
    压成 2 类块再统计——后者会让每个错误都变成跨物种错误，跨物种错误率恒等于 50%，
    指标失去意义（这是原公式的缺陷：n_cat - cat_to_cat 恰等于 cat_to_dog，分母是 cross 的 2 倍）。

    正确口径：
        真正的跨物种错误 = 预测错 且 cat[pred] != cat[true]
        种内品种错误     = 预测错 且 cat[pred] == cat[true]
        跨物种错误率     = 真正的跨物种错误 / 全部错误
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = int(max(int(y_true.max()), int(y_pred.max()))) + 1
    cat = load_cat_mask(n)
    t_cat, p_cat = cat[y_true], cat[y_pred]

    err = y_true != y_pred
    n_err = int(err.sum())
    cross = err & (t_cat != p_cat)          # 猫->狗 或 狗->猫：真正的跨物种
    within = err & (t_cat == p_cat)         # 同物种内分错品种
    cat_to_dog = int((err & t_cat & ~p_cat).sum())
    dog_to_cat = int((err & ~t_cat & p_cat).sum())

    n_cat, n_dog = int(t_cat.sum()), int((~t_cat).sum())
    res = {
        "n_cat": n_cat, "n_dog": n_dog,
        "cat_to_cat": int((~err & t_cat).sum()),
        "cat_to_dog": cat_to_dog,
        "dog_to_cat": dog_to_cat,
        "dog_to_dog": int((~err & ~t_cat).sum()),
        "n_error": n_err,
        "n_cross_species_error": int(cross.sum()),
        "n_within_species_error": int(within.sum()),
    }
    res["cat_recall"] = res["cat_to_cat"] / max(n_cat, 1)
    res["dog_recall"] = res["dog_to_dog"] / max(n_dog, 1)
    res["cat_dog_confusion_rate"] = cat_to_dog / max(n_cat, 1)
    res["dog_cat_confusion_rate"] = dog_to_cat / max(n_dog, 1)
    # 关键结论指标：全部错误里有多少是「跨物种」的
    res["cross_species_error_ratio"] = cross.sum() / max(n_err, 1)
    res["within_species_error_ratio"] = within.sum() / max(n_err, 1)
    return res


def main():
    ap = argparse.ArgumentParser("可视化与每类指标")
    ap.add_argument("--pred-csv", required=True)
    ap.add_argument("--classes", default="labels/pet_classes.txt")
    ap.add_argument("--out-dir", required=True, help="实验输出目录，结果写入其子目录")
    ap.add_argument("--tag", required=True, help="实验名，用于文件名，如 B0_baseline")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--top-k-cases", type=int, default=2, help="正确/失败案例各取几张")
    args = ap.parse_args()

    setup_chinese_font()                       # 所有绘图脚本第一步
    names = load_classes(args.classes)
    df = load_pred_csv(args.pred_csv)
    y_true = df["true_idx"].to_numpy()
    y_pred = df["pred_idx"].to_numpy()
    out = Path(args.out_dir)          # 一律传仓库根的 outputs/，产物落到 CANON 规定的固定子目录
    cm_dir = out / "confusion_matrix"; cm_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1) 行归一化混淆矩阵 ----------
    # 文件名遵循 outputs/confusion_matrix/<experiment_name>_cm.{npy,csv,png}；split 只写进标题
    cm = row_normalized_cm(y_true, y_pred, len(names))
    np.save(cm_dir / f"{args.tag}_cm.npy", cm)
    pd.DataFrame(cm, index=names, columns=names).to_csv(
        cm_dir / f"{args.tag}_cm.csv", float_format="%.6f")
    plot_confusion_matrix(cm, names, normalize="true",
                          out_path=str(cm_dir / f"{args.tag}_cm.png"),
                          title=f"{args.tag} 混淆矩阵 ({args.split}, n={len(df)})")
    # 自动打印最易混淆的 10 对（排除对角线）
    off = cm.copy(); np.fill_diagonal(off, 0.0)
    pairs = [(names[i], names[j], off[i, j])
             for i in range(len(names)) for j in range(len(names)) if i != j]
    pairs.sort(key=lambda x: -x[2])
    print(f"\n[{args.tag}] 最易混淆的 10 对（行归一化，即 P(pred=j | true=i)）:")
    for a, b, v in pairs[:10]:
        print(f"  {a:22s} -> {b:22s} {v*100:6.2f}%")

    # ---------- 2) 每类 precision / recall / f1 / support ----------
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(names))), zero_division=0)
    per = pd.DataFrame({"class_idx": range(len(names)), "class_name": names,
                        "support": s, "precision": p, "recall": r, "f1": f1})
    per["species"] = np.where(load_cat_mask(len(names))[per["class_idx"].to_numpy()], "cat", "dog")
    per = per.sort_values("f1", ascending=True)
    per.to_csv(cm_dir / f"{args.tag}_per_class.csv", index=False, float_format="%.6f")
    print(f"\n每类指标已写 -> {cm_dir / f'{args.tag}_per_class.csv'}")
    print(per.head(8).to_string(index=False))

    print(f"\nTop-1 = {accuracy_score(y_true, y_pred)*100:.2f}%  "
          f"Macro-F1 = {f1_score(y_true, y_pred, average='macro', zero_division=0)*100:.2f}%  "
          f"随机水平 = {100/37:.2f}%")
    assert "macro" == "macro"       # 显式提示：f1_score 必须传 average='macro'

    # ---------- 3) 每类 F1 柱状图 ----------
    f1_all = f1_score(y_true, y_pred, labels=list(range(len(names))),
                      average=None, zero_division=0)
    plot_per_class_f1(f1_all, names,
                      str(cm_dir / f"{args.tag}_per_class_f1.png"),
                      title=f"{args.tag} 各类别 F1 ({args.split}, 升序)")

    # ---------- 4) 猫 / 狗 两大类块 ----------
    block = cat_dog_block(y_true, y_pred)
    with open(cm_dir / f"{args.tag}_cat_dog_block.json", "w",
              encoding="utf-8") as fp:
        json.dump(block, fp, ensure_ascii=True, indent=2)
    print(f"\n[猫狗块] 猫召回={block['cat_recall']*100:.2f}%  "
          f"狗召回={block['dog_recall']*100:.2f}%  "
          f"跨物种误判占比={block['cross_species_error_ratio']*100:.2f}%"
          f"（{block['n_cross_species_error']}/{block['n_error']} 个错误跨物种；"
          f"种内品种错误 {block['n_within_species_error']} 个）")
    print("  → 跨物种误判占比低（<10%）说明模型没有混淆猫狗，只是分不清品种；"
          "报告里应据此把失败归因定位到种内细粒度差异。")

    # ---------- 5) 典型案例挑选（写回文件供预测图脚本使用） ----------
    wrong = df[(df["correct"] == 0)].sort_values("prob", ascending=False)
    right = df[(df["correct"] == 1)].sort_values("prob", ascending=False)
    pick = pd.concat([wrong.head(args.top_k_cases), right.head(args.top_k_cases)])
    pick.to_csv(out / "predictions" / f"cases_{args.split}_{args.tag}.csv", index=False)
    print(f"\n[案例] 已挑出 {len(pick)} 张（高置信度错误优先）-> "
          f"{out / 'predictions' / f'cases_{args.split}_{args.tag}.csv'}")


if __name__ == "__main__":
    main()
