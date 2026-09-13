# datasets/audit_leakage.py
# 运行（仓库根）: python datasets/audit_leakage.py
import hashlib, json
from collections import defaultdict
from pathlib import Path

PROJ   = Path(__file__).resolve().parents[1]             # 仓库根，禁止写死绝对路径
LISTS  = PROJ / "datasets/lists"
IMG    = PROJ / "data/oxford-iiit-pet/images"
OUT    = PROJ / "outputs/metrics/leakage_check.json"

def read_split(name):
    """读 datasets/lists/pet_<name>.txt，返回 image_id 列表（去掉 <image_id>\\t<label> 的标签列）。"""
    return [l.rsplit(None, 1)[0].strip() for l in open(LISTS / f"pet_{name}.txt", encoding="utf-8") if l.strip()]

def md5_of(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()

def main():
    files = {s: read_split(s) for s in ("train", "val", "test")}
    stems = {s: set(v) for s, v in files.items()}
    missing = {s: [i for i in v if not (IMG / f"{i}.jpg").exists()] for s, v in files.items()}

    stem_pairs = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = stems[a] & stems[b]
        stem_pairs[f"{a}_x_{b}"] = {"count": len(inter), "examples": sorted(inter)[:10]}

    md5s = {s: defaultdict(list) for s in files}
    for s, ids in files.items():
        for i in ids:
            md5s[s][md5_of(IMG / f"{i}.jpg")].append(f"{i}.jpg")
    md5_pairs = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        dup = set(md5s[a]) & set(md5s[b])
        md5_pairs[f"{a}_x_{b}"] = {
            "count": len(dup),
            "examples": [{m: md5s[a][m][:2] + md5s[b][m][:2]} for m in sorted(dup)[:5]],
        }
    # 各 split 内部的重复图（同一张图出现两次也会污染 per-class 统计）
    internal = {s: {m: v for m, v in md5s[s].items() if len(v) > 1} for s in files}
    # 41 张孤儿图是否被误扫进来
    ann = PROJ / "data/oxford-iiit-pet/annotations"
    official = {l.split()[0] for f in ("list.txt",) for l in open(ann / f, encoding="utf-8")
                if l.strip() and not l.startswith("#")}
    all_used = set().union(*stems.values())
    orphans = sorted(all_used - official)

    rep = {
        "n": {s: len(v) for s, v in files.items()},
        "missing_files": {s: v[:10] for s, v in missing.items()},
        "stem_intersection": stem_pairs,
        "md5_intersection": md5_pairs,
        "internal_duplicates": {s: {m: v for m, v in list(internal[s].items())[:10]} for s in internal},
        "orphan_images_used": orphans,
        # ---- DoD #8 契约键：规格书的验收命令直接按这几个名字取值，不得改名 ----
        # python -c "import json;d=json.load(open('outputs/metrics/leakage_check.json'));
        #            print(d['train_val'],d['train_test'],d['val_test'],d['md5_intersections'])"
        "train_val": stem_pairs["train_x_val"]["count"],
        "train_test": stem_pairs["train_x_test"]["count"],
        "val_test": stem_pairs["val_x_test"]["count"],
        "md5_intersections": {k: v["count"] for k, v in md5_pairs.items()},
        "pass": (all(v["count"] == 0 for v in stem_pairs.values())
                 and all(v["count"] == 0 for v in md5_pairs.values())
                 and not any(missing.values())
                 and not orphans
                 and not any(internal.values())),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=True, indent=2), encoding="utf-8")

    print("样本数:", rep["n"])
    for k, v in stem_pairs.items():
        print(f"文件名交集 {k}: {v['count']}")
    for k, v in md5_pairs.items():
        print(f"MD5 交集  {k}: {v['count']}")
    print("孤儿图误入:", len(orphans), orphans[:5])
    print("内部重复图:", {s: len(v) for s, v in internal.items()})
    print("PASS" if rep["pass"] else "FAIL -> 见 " + str(OUT))
    return 0 if rep["pass"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
