"""datasets/make_pet_split.py —— 从官方 trainval.txt 生成每类固定 20 张的 train/val 划分 + 类别文件。

用法（唯一入口，参数名固定）：
    python datasets/make_pet_split.py --root data/oxford-iiit-pet \
        --out-dir datasets/lists --val-per-class 20 --seed 42
产物：datasets/lists/pet_train.txt（2940 行）、pet_val.txt（740 行）、pet_test.txt（3669 行）、
      labels/pet_classes.txt（37 行，行号 = 下标）、labels/pet_class_to_idx.json（name -> idx）

关键点（每一条都对应一个真实踩坑）：
  1) 官方 annotations/ 下只有 list.txt / trainval.txt / test.txt，没有 train.txt / val.txt，
     train/val 必须自己从 trainval 切出来；test 用官方 test.txt，全程不参与任何选择。
  2) 官方标注是 4 列 "image_id CLASS-ID SPECIES BREED-ID"，CLASS-ID 是 1..37；
     标签必须减 1，且第 4 列是物种内品种号（猫 1-12 / 狗 1-25），不能当 37 类标签用。
  3) 类别名必须取自官方 CLASS-ID 顺序（大写开头的猫排在小写开头的狗之前），
     不能用 ImageFolder 的目录名字母序——狗的 25 类顺序会被整体打乱。
  4) images/ 里比 list.txt 多 41 张「孤儿图」，必须按 list.txt 白名单过滤掉。
  5) 每类固定取 val_per_class 张进 val（不是按百分比）：Macro-F1 每类等权，val 每类样本数
     相等才不会让样本多的类主导早停点；类内先排序再 shuffle，消除文件系统遍历顺序的不确定性。
"""
import argparse, hashlib, json, random
from collections import Counter
from pathlib import Path


def read_ann(ann_dir: Path, name: str):
    rows = []
    for line in (ann_dir / name).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        image_id, cid, species, breed = line.split()      # 4 列固定制式
        rows.append((image_id, int(cid) - 1))             # 关键：1-based -> 0-based
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Pet train/val 划分 + 37 类类别文件生成")
    ap.add_argument("--root", default="data/oxford-iiit-pet")
    ap.add_argument("--out-dir", default="datasets/lists")
    ap.add_argument("--labels-dir", default="labels")
    ap.add_argument("--val-per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    root = Path(a.root)
    ann, img = root / "annotations", root / "images"
    out, lab = Path(a.out_dir), Path(a.labels_dir)
    out.mkdir(parents=True, exist_ok=True)
    lab.mkdir(parents=True, exist_ok=True)

    # 1) 用 list.txt 建白名单并还原 CLASS-ID -> 原始前缀名，得到 0..36 的类别名
    valid, cid2prefix = set(), {}
    for image_id, cid0 in read_ann(ann, "list.txt"):
        valid.add(image_id)
        cid2prefix[cid0] = image_id.rsplit("_", 1)[0]     # 必须 rsplit：split('_')[0] 会切出假类名
    assert len(cid2prefix) == 37, f"类别数应为 37，实得 {len(cid2prefix)}"
    # 类别名口径必须与 datasets/build_pet_class_map.py 及 torchvision 完全一致：
    #   前缀 american_pit_bull_terrier -> 显示名 "American Pit Bull Terrier"
    # 早期版本此处直接写前缀名，与 build_pet_class_map.py 的 torchvision 口径冲突；
    # 两个脚本互相覆盖会让 tools/selfcheck.py 的类别名检查随执行顺序在 PASS/FAIL 间抖动
    # （实测复现）。此处统一到显示名口径，任何执行顺序下结果都相同。
    classes = [" ".join(seg.title() for seg in cid2prefix[i].split("_")) for i in range(37)]
    (lab / "pet_classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (lab / "pet_class_to_idx.json").write_text(
        json.dumps({c: i for i, c in enumerate(classes)}, ensure_ascii=False, indent=0), encoding="utf-8")
    label_of = dict(read_ann(ann, "list.txt"))            # image_id -> 0-based 标签

    # 2) trainval 切分：类内先排序再 shuffle，每类固定取 val_per_class 张进 val
    by_cls = {}
    for image_id, cid0 in read_ann(ann, "trainval.txt"):
        if image_id in valid:
            by_cls.setdefault(cid0, []).append(image_id)
    rng = random.Random(a.seed)
    train_ids, val_ids = [], []
    for c in sorted(by_cls):
        ids = sorted(by_cls[c])
        rng.shuffle(ids)
        k = min(a.val_per_class, max(1, len(ids) // 2))   # 防御：每类样本很少时不至于把 train 掏空
        val_ids += ids[:k]
        train_ids += ids[k:]
    assert set(train_ids).isdisjoint(val_ids)

    # 3) test 用官方 test.txt，同样走白名单
    test_ids = [i for i, _ in read_ann(ann, "test.txt") if i in valid]

    def dump(ids, path: Path):
        lines = []
        for image_id in sorted(ids):
            assert (img / f"{image_id}.jpg").exists(), image_id
            lines.append(f"{image_id}\t{label_of[image_id]}")   # "<image_id>\t<0-based 标签>"，不含 .jpg
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        print(f"{path} n={len(lines)} sha256[:16]={h}")

    dump(train_ids, out / "pet_train.txt")
    dump(val_ids, out / "pet_val.txt")
    dump(test_ids, out / "pet_test.txt")

    for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        cnt = Counter(label_of[i] for i in ids)
        print(f"[{name}] n={len(ids)} 每类 min={min(cnt.values())} max={max(cnt.values())} "
              f"覆盖类别数={len(cnt)}/37")


if __name__ == "__main__":
    main()
