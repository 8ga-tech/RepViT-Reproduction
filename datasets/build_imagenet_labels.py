# datasets/build_imagenet_labels.py
# -*- coding: utf-8 -*-
"""生成 labels/imagenet_classes.txt（1000 行，wnid 字典序，人类可读名）。

背景：规格书 §1.2 把「ImageNet 类别映射」列为**考核方提供**的材料，但考核方未下发
（见 data/provided/README_这里要放什么.md）。本脚本用 timm 自带的权威数据本地生成，
并用规格书 DoD #10 给出的四个锚点做自检：

    L[0]='tench'  L[207]='golden retriever'  L[281]='tabby'  L[999]='toilet tissue'

数据来源（均为 timm 包内置，不联网）：
    timm/data/_info/imagenet_synsets.txt          —— 1000 个 wnid，字典序
    timm/data/_info/imagenet_synset_to_lemma.txt  —— wnid -> "tench, Tinca tinca"
    人类可读名取逗号前的第一段（与 DoD #10 的期望写法一致）。

产物：
    labels/imagenet_classes.txt        1000 行，每行一个类名（index 即行号-1）
    labels/imagenet_wnid_to_idx.json   {"n01440764": 0, ...}
    outputs/metrics/imagenet_labels_report.json   溯源与自检结果

用法：
    python datasets/build_imagenet_labels.py
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# DoD #10 的四个锚点（index -> 期望名）
ANCHORS = {0: "tench", 207: "golden retriever", 281: "tabby", 999: "toilet tissue"}


def load_timm_info():
    """优先用 timm 的 ImageNetInfo 取 wnid 序列；取不到就退回直接读 _info 文件。

    返回的第三项是**timm 包内的相对目录名**（不是采集机的绝对路径）：
    落盘报告里写绝对路径会让产物不可移植（试题第 14 页要求不得写死个人电脑路径）。
    """
    import timm
    info_dir = Path(timm.__file__).parent / "data" / "_info"
    syn_file = info_dir / "imagenet_synsets.txt"
    lemma_file = info_dir / "imagenet_synset_to_lemma.txt"
    if not syn_file.exists() or not lemma_file.exists():
        raise FileNotFoundError(f"timm 内置词典缺失: {info_dir}")

    wnids = [ln.strip() for ln in syn_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    lemma = {}
    for line in lemma_file.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            k, v = line.split("\t", 1)
            lemma[k.strip()] = v.split(",")[0].strip()
    return wnids, lemma, "timm/data/_info", getattr(timm, "__version__", "unknown")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="labels/imagenet_classes.txt")
    ap.add_argument("--out-wnid-json", default="labels/imagenet_wnid_to_idx.json")
    ap.add_argument("--report", default="outputs/metrics/imagenet_labels_report.json")
    a = ap.parse_args()

    wnids, lemma, info_rel, timm_version = load_timm_info()
    print("=" * 70)
    print(f"[src] timm 内置词典目录: {info_rel}（timm {timm_version}）")
    print(f"[src] wnid 数 = {len(wnids)}")

    names, missing = [], []
    for w in wnids:
        n = lemma.get(w)
        if n is None:
            missing.append(w)
            n = w                                   # 兜底：用 wnid 占位，并在报告里标出
        names.append(n)

    # ---- 自检：维度 ----
    rep = {"source": info_rel, "source_note": "timm 包内置词典目录（相对 timm 包根，不写采集机绝对路径）",
           "timm_version": timm_version,
           "n_wnids": len(wnids), "n_names": len(names),
           "n_missing_lemma": len(missing), "missing_lemma": missing[:20],
           "order": "wnid 字典序（timm/data/_info/imagenet_synsets.txt 的行序）"}

    # ---- 自检：DoD #10 四锚点 ----
    anchor_res = {str(i): {"expect": e, "got": names[i] if i < len(names) else None,
                           "pass": (i < len(names) and names[i] == e)}
                  for i, e in ANCHORS.items()}
    all_pass = all(v["pass"] for v in anchor_res.values())
    rep["dod10_anchors"] = anchor_res
    rep["dod10_pass"] = all_pass

    # ---- 自检：重复名（ImageNet 里存在同 lemma 的不同 synset，需显式记录）----
    from collections import Counter
    dup = {k: v for k, v in Counter(names).items() if v > 1}
    rep["duplicate_names"] = dup
    rep["n_unique_names"] = len(set(names))

    print("-" * 70)
    for i, e in ANCHORS.items():
        got = names[i] if i < len(names) else None
        print(f"  [{i:>3}] 期望 {e!r:<20} 实得 {got!r:<20} {'PASS' if got == e else 'FAIL'}")
    print(f"  维度: {len(names)} 行  |  唯一名 {len(set(names))}  |  重名 {len(dup)} 组"
          f"  |  缺 lemma {len(missing)}")
    if dup:
        print(f"  重名示例: {list(dup.items())[:5]}")

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"[write] {out}  ({len(names)} 行)")

    outj = ROOT / a.out_wnid_json
    outj.write_text(json.dumps({w: i for i, w in enumerate(wnids)},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {outj}")

    rp = ROOT / a.report
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rep, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[write] {rp}")
    print("=" * 70)
    if not all_pass:
        print("[FAIL] DoD #10 锚点未全部通过；标签文件不可用于评价")
        return 1
    print("[PASS] DoD #10 四锚点全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
