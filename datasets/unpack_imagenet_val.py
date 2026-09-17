# -*- coding: utf-8 -*-
"""datasets/unpack_imagenet_val.py —— 把 Parquet 形式的 ImageNet-1K 验证集还原成 ImageFolder 结构。

Source : Self-written。数据来源为 HuggingFace 数据集 mrm8488/ImageNet1K-val（其 `image.path`
         字段在上传时被剥离，因此无法直接恢复原始文件名；本脚本改用 `label` 列 + timm 的
         `imagenet_synsets.txt` 反推 WNID 目录名）。

⚠ 已不在主线流程内（2026-09 口径变更）
    官方模型评价的唯一口径是 **ImageNetV2 matched-frequency 的确定性 1000 张子集**
    （`datasets/lists/imagenetv2_mf_1000.txt`，由 `datasets/make_imagenetv2_subset.py` 构建）。
    旧的自建 ImageNet-1K val 子集（本脚本还原出的 5 万张母集 + `make_imagenet_subset.py`
    抽样 1000 张）已按用户决议**整体删除**，本脚本因此不再被任何流水线步骤调用。
    保留原因：本机确实存在 `data/imagenet/val/`，本文件是该目录的获取与校验记录（可复现来源），
    并且 `data/` 已被 `.gitignore` 忽略 —— 它不影响提交物，也不构成复现路径的一部分。
    **不要**按本脚本重新构建旧的 1000 张子集（那套口径已作废）。

为什么当初需要这一步：
    规格书 §3.6.1 要求 `data/imagenet/val` 必须是 ImageFolder 结构（`val/<wnid>/<file>.JPEG`），
    因为 ImageFolder 的类别序 = `sorted(目录名)` = wnid 字母序 = 模型输出下标 0..999。
    而本机拿到的是 14 个 parquet 分片，必须先落成目录树。

产物：
    data/imagenet/val/<wnid>/ILSVRC2012_val_<label:08d>_<i:02d>.JPEG   （1000 类 × 50 张 = 50,000）
    data/imagenet/meta/imagenet_synsets.txt                            （1000 行 wnid，供子集脚本自检）
    outputs/metrics/imagenet_unpack.json                               （统计与校验）

用法：
    python datasets/unpack_imagenet_val.py                       # 全量解包（幂等，已存在则跳过）
    python datasets/unpack_imagenet_val.py --limit-per-class 2   # 冒烟：每类只出 2 张
    python datasets/unpack_imagenet_val.py --verify-only         # 只校验目录树，不写图
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]          # 仓库根，禁止写死绝对路径
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

DEFAULT_SRC = "data/_src/imagenet_val"
DEFAULT_OUT = "data/imagenet/val"
DEFAULT_SYN = "data/imagenet/meta/imagenet_synsets.txt"

JPEG_MAGIC = b"\xff\xd8\xff"


def timm_synsets() -> list[str]:
    """取 timm 内置的 1000 个 synset（wnid），顺序即 ImageNet 官方类别下标顺序。"""
    import timm

    p = Path(timm.__file__).parent / "data" / "_info" / "imagenet_synsets.txt"
    assert p.exists(), f"找不到 timm 的 synset 清单：{p}"
    syn = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(syn) == 1000, f"synset 清单应为 1000 行，实际 {len(syn)}"
    # 四个锚点：与 labels/imagenet_classes.txt 必须一一对应（DoD #10）
    assert (syn[0], syn[207], syn[281], syn[999]) == \
           ("n01440764", "n02099601", "n02123045", "n15075141"), "synset 顺序不是标准序"
    return syn


def scan_parquets(src_dir: Path) -> tuple[list[Path], Counter]:
    """列出分片并统计每个 label 的样本数，用于确认「每类 50 张」这一事实基线。"""
    fs = sorted(glob.glob(str(src_dir / "*.parquet")))
    assert fs, f"{src_dir} 下没有 parquet 分片（是否还没下载？）"
    import pyarrow.parquet as pq

    cnt: Counter = Counter()
    for p in fs:
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=8192, columns=["label"]):
            for v in b.column("label").to_pylist():
                cnt[int(v)] += 1
    return [Path(p) for p in fs], cnt


def _bad(v: object, lo: int, hi: int) -> str | None:
    """返回人类可读的错误说明；None 表示该项通过。"""
    if not isinstance(v, int):
        return f"不是整数：{v!r}"
    if not (lo <= v <= hi):
        return f"越界：{v} 不在 [{lo}, {hi}]"
    return None


def unpack(src_dir: Path, out_dir: Path, syn_file: Path,
           limit_per_class: int | None = None) -> dict:
    import pyarrow.parquet as pq

    syn = timm_synsets()
    fs, cnt = scan_parquets(src_dir)

    # 事实基线：ImageNet-1K val 恰好 1000 类 × 50 张
    assert len(cnt) == 1000, f"类别数应为 1000，实际 {len(cnt)}"
    per_class = Counter(cnt.values())
    assert set(per_class) == {50}, f"每类张数应为 50，实际分布 {per_class.most_common(5)}"

    want = limit_per_class if limit_per_class else 50
    out_dir.mkdir(parents=True, exist_ok=True)

    written: Counter = Counter()
    skipped = 0
    for p in fs:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=2048, columns=["image", "label"]):
            img_col = batch.column("image")
            bytes_l = img_col.field("bytes").to_pylist()
            lab_l = batch.column("label").to_pylist()
            for raw, lab in zip(bytes_l, lab_l):
                lab = int(lab)
                if written[lab] >= want:
                    skipped += 1
                    continue
                i = written[lab]
                d = out_dir / syn[lab]
                d.mkdir(parents=True, exist_ok=True)
                fp = d / f"ILSVRC2012_val_{lab:08d}_{i:02d}.JPEG"
                if not (fp.exists() and fp.stat().st_size > 0):
                    assert raw is not None, f"label={lab} idx={i} 的 bytes 为空"
                    assert raw[:3] == JPEG_MAGIC, f"{fp.name} 不是 JPEG（magic={raw[:3]!r}）"
                    # 先写临时文件再原子替换：中断不会留下半张图
                    tmp = fp.with_suffix(".JPEG.tmp")
                    tmp.write_bytes(raw)
                    os.replace(tmp, fp)
                written[lab] += 1

    syn_file.parent.mkdir(parents=True, exist_ok=True)
    syn_file.write_text("\n".join(syn) + "\n", encoding="utf-8", newline="\n")

    got = Counter(written.values())
    rep = {
        "src_dir": src_dir.relative_to(PROJ).as_posix(),
        "out_dir": out_dir.relative_to(PROJ).as_posix(),
        "synsets_file": syn_file.relative_to(PROJ).as_posix(),
        "shards": len(fs),
        "parquet_total_images": int(sum(cnt.values())),
        "classes_written": len(written),
        "images_written": int(sum(written.values())),
        "per_class_distribution": {str(k): v for k, v in sorted(got.items())},
        "limit_per_class": limit_per_class,
        "skipped": skipped,
    }
    return rep


def verify(out_dir: Path, limit_per_class: int | None = None) -> dict:
    """只校验目录树：类别数、每类张数、目录名排序 == synset 顺序、JPEG 魔数。"""
    syn = timm_synsets()
    dirs = sorted(p.name for p in out_dir.iterdir() if p.is_dir())
    assert dirs == syn, (
        "val 目录的类别顺序与 synset 顺序不一致——"
        f"首个不一致：{next((a, b) for a, b in zip(dirs, syn) if a != b)}")
    per = {}
    bad_magic = []
    for wn in syn:
        files = [f for f in (out_dir / wn).iterdir()
                 if f.suffix.lower() in (".jpeg", ".jpg", ".png")]
        per[wn] = len(files)
        for f in files[:2]:
            with open(f, "rb") as fh:
                if fh.read(3) != JPEG_MAGIC:
                    bad_magic.append(f.as_posix())
    counts = Counter(per.values())
    assert not bad_magic, f"发现非 JPEG 文件：{bad_magic[:5]}"
    return {"classes": len(dirs), "per_class_distribution": dict(sorted(counts.items())),
            "total": sum(per.values()), "bad_magic": len(bad_magic)}


def main() -> int:
    ap = argparse.ArgumentParser(description="把 ImageNet-1K val 的 parquet 分片还原成 ImageFolder")
    ap.add_argument("--src-dir", default=DEFAULT_SRC, help="parquet 分片目录（相对仓库根）")
    ap.add_argument("--out-dir", default=DEFAULT_OUT, help="ImageFolder 输出目录（相对仓库根）")
    ap.add_argument("--syn-file", default=DEFAULT_SYN, help="synset 清单输出路径（相对仓库根）")
    ap.add_argument("--limit-per-class", type=int, default=None,
                    help="冒烟用：每类只解出前 N 张；缺省解全量 50 张")
    ap.add_argument("--verify-only", action="store_true", help="只校验已有目录树，不写图")
    a = ap.parse_args()

    out_dir = (PROJ / a.out_dir).resolve()
    syn_file = (PROJ / a.syn_file).resolve()

    if a.verify_only:
        rep = {"mode": "verify_only", "out_dir": a.out_dir, **verify(out_dir)}
        print(json.dumps(rep, ensure_ascii=True, indent=2))
        return 0

    rep = unpack((PROJ / a.src_dir).resolve(), out_dir, syn_file, a.limit_per_class)
    rep["verify"] = verify(out_dir)
    rep["mode"] = "unpack"

    from utils.logging import dump_metrics
    dump_metrics("imagenet_unpack", rep)
    print(json.dumps(rep, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
