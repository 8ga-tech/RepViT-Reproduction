# tools/assert_data.py
# 运行（仓库根）: python tools/assert_data.py --root data/oxford-iiit-pet
import argparse, json, os, sys
from collections import Counter
from pathlib import Path

def n_files(d: Path, suffix: str) -> int:
    """统计真实文件数。

    必须排除 macOS AppleDouble 资源叉文件（`._<name>`）。官方 Pet 归档在
    annotations/trimaps/ 下带了 7390 个 `._*.png`，它们与真实 PNG 同名同后缀，
    只按后缀计数会得到 14780 = 7390 x 2（实测复现）。这些文件不是图片，
    不解压器无关，但会让「每类图片数」断言全部翻倍。
    """
    if not d.is_dir():
        return -1
    return sum(1 for f in os.listdir(d)
               if f.endswith(suffix) and not f.startswith("._"))

def n_lines(p: Path) -> int:
    with open(p, encoding="utf-8") as f:
        return sum(1 for l in f if l.strip())

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",   default="data/oxford-iiit-pet")            # 相对仓库根
    ap.add_argument("--report", default="outputs/metrics/dataset_report.json")
    a = ap.parse_args()
    B = Path(a.root)
    IMG, ANN = B / "images", B / "annotations"

    actual = {
        "images_jpg":                 n_files(IMG, ".jpg"),
        "trimaps_png":                n_files(ANN / "trimaps", ".png"),
        "xmls_xml":                   n_files(ANN / "xmls", ".xml"),
        "trainval_txt_lines":         n_lines(ANN / "trainval.txt"),
        "test_txt_lines":             n_lines(ANN / "test.txt"),
        "list_txt_data_lines":        sum(1 for l in open(ANN / "list.txt", encoding="utf-8")
                                           if l.strip() and not l.startswith("#")),
    }
    expect = {
        "images_jpg": 7390, "trimaps_png": 7390, "xmls_xml": 3686,
        "trainval_txt_lines": 3680, "test_txt_lines": 3669, "list_txt_data_lines": 7349,
    }
    # 必须存在的路径（缺任一条直接失败，不做容错）
    required = [
        IMG, ANN, ANN / "trimaps", ANN / "xmls",
        ANN / "list.txt", ANN / "trainval.txt", ANN / "test.txt", ANN / "README",
    ]

    fails = []
    for p in required:
        if not p.exists():
            fails.append(f"缺少必须路径: {p}")
    for k, v in expect.items():
        if actual[k] != v:
            fails.append(f"{k}: 实际={actual[k]} 期望={v}")

    # 标注自洽性：trainval ∪ test == list.txt 的 id 集合；trainval ∩ test == ∅
    def ids(p):
        # 必须跳过注释行：annotations/list.txt 以
        # `#Image CLASS-ID SPECIES BREED-ID` 和若干 `#BREED ...` 说明行开头，
        # 不跳过会把 '#Image'、'#All'、'#BREED' 等 6 个词当成图片 id（实测复现），
        # 从而误报「trainval∪test 与 list.txt 不一致」。
        return {l.split()[0] for l in open(p, encoding="utf-8")
                if l.strip() and not l.startswith("#")}
    tv, te, al = ids(ANN / "trainval.txt"), ids(ANN / "test.txt"), ids(ANN / "list.txt")
    if tv & te:
        fails.append(f"trainval 与 test 有交集 {len(tv & te)} 个 id")
    if (tv | te) != al:
        fails.append(f"trainval∪test 与 list.txt 不一致: 多 {len((tv|te)-al)} 少 {len(al-(tv|te))}")
    if len(tv) != 3680 or len(te) != 3669:
        fails.append(f"唯一 id 数不对: trainval={len(tv)} test={len(te)}")

    # 每类计数直方图（0-based 标签），确认 37 类且每类 93-100（trainval 侧）
    cnt = Counter(int(l.split()[1]) - 1 for l in open(ANN / "trainval.txt", encoding="utf-8") if l.strip())
    if sorted(cnt) != list(range(37)):
        fails.append(f"trainval 类别号不是 0..36: {sorted(cnt)[:5]} ... {sorted(cnt)[-5:]}")
    else:
        lo, hi = min(cnt.values()), max(cnt.values())
        if not (90 <= lo and hi <= 100):
            fails.append(f"trainval 每类样本数异常: min={lo} max={hi}（期望 93..100）")

    rep = {"base": str(B), "actual": actual, "expected": expect,
           "trainval_per_class": {str(k): v for k, v in sorted(cnt.items())},
           "pass": not fails, "fails": fails}
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(rep, ensure_ascii=True, indent=2), encoding="utf-8")

    for k in expect:
        flag = "OK " if actual[k] == expect[k] else "BAD"
        print(f"[{flag}] {k:24s} 实际={actual[k]:6d} 期望={expect[k]:6d}")
    if fails:
        print("\n断言失败:"); [print("  -", m) for m in fails]
        return 1
    print("\nALL PASS -> ", a.report)
    return 0

if __name__ == "__main__":
    sys.exit(main())
