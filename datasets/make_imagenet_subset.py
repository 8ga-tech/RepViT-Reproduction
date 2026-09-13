# datasets/make_imagenet_subset.py
# 运行: python datasets/make_imagenet_subset.py --n 1000 --seed 20260912
import argparse, json, random
from collections import Counter
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]               # 仓库根，禁止写死绝对路径
SYN  = PROJ / "data/imagenet/meta/imagenet_synsets.txt"
OUTD = PROJ / "outputs"; (OUTD / "metrics").mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, choices=[500, 1000], default=1000)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--val-root", default="data/imagenet/val", help="ImageNet val 目录（相对仓库根）")
    # 路径口径（本仓唯一真源 = outputs/predictions/*.csv 的列契约）：
    #   repo  —— 写出「相对仓库根」的路径（默认）。与 predictions.csv 的 `path` 列口径一致，
    #            任何脚本都能直接 open()，不需要再拼一个 root，消除歧义。
    #   val   —— 写出「相对 data/imagenet/val」的路径（旧口径，仅为兼容保留）。
    ap.add_argument("--path-mode", choices=["repo", "val"], default="repo",
                    help="list 内路径的口径：repo=相对仓库根（默认），val=相对 val-root")
    a = ap.parse_args()
    val_root = PROJ / a.val_root

    synsets = [l.strip() for l in open(SYN, encoding="utf-8") if l.strip()]
    assert len(synsets) == 1000, len(synsets)
    # 关键断言：ImageFolder 的类别序 = sorted(目录名) = wnid 字母序 = 模型输出下标
    assert sorted(p.name for p in val_root.iterdir() if p.is_dir()) == synsets, \
        "val 目录结构或类别顺序不对（是否忘了 valprep.sh？）"

    rng = random.Random(a.seed)
    rows = []
    base = PROJ if a.path_mode == "repo" else val_root
    if a.n == 1000:
        for idx, wn in enumerate(synsets):                    # 严格按索引升序遍历，保证可复现
            d = val_root / wn
            files = sorted(f.name for f in d.iterdir()
                           if f.suffix.lower() in (".jpeg", ".jpg", ".png"))
            assert len(files) == 50, (wn, len(files))         # ImageNet val 每类恰好 50 张
            rows.append(((d / rng.sample(files, 1)[0]).relative_to(base), idx))
    else:
        # 500 张 = 每 2 个类别取 1 类、每类 1 张（类别均衡、确定性，适合冒烟）
        for idx in range(0, 1000, 2):
            d = val_root / synsets[idx]
            files = sorted(f.name for f in d.iterdir() if f.suffix.lower() == ".jpeg")
            rows.append(((d / rng.sample(files, 1)[0]).relative_to(base), idx))

    out = PROJ / "datasets/lists/imagenet_val_subset.txt"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for rel, y in rows:
            f.write(f"{rel.as_posix()}\t{y}\n")               # 见 --path-mode：默认相对仓库根的路径 + TAB + 0-based 标签
    cnt = Counter(y for _, y in rows)
    rep = {"n": len(rows), "num_classes": len(cnt), "seed": a.seed, "path_mode": a.path_mode,
           "per_class_min": min(cnt.values()), "per_class_max": max(cnt.values()),
           "label_min": min(cnt), "label_max": max(cnt), "out": out.relative_to(PROJ).as_posix()}
    (OUTD / "metrics" / "subset_report.json").write_text(
        json.dumps(rep, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=True))
    expected_max = 999 if a.n == 1000 else 998
    assert len(rows) == a.n and rep["label_min"] == 0 and rep["label_max"] == expected_max

if __name__ == "__main__":
    main()
