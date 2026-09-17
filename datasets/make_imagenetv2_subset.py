# -*- coding: utf-8 -*-
"""datasets/make_imagenetv2_subset.py —— 校验 ImageNetV2 归档 + 构建 1000 张确定性固定子集。

Source : Self-written。数据来源 = 官方 **ImageNetV2**（Recht et al., *Do ImageNet Classifiers
         Generalize to ImageNet?*, NeurIPS 2019，官方仓库 modestyachts/ImageNetV2）Release 归档
         ``imagenetv2-matched-frequency.tar.gz``，经作者方 HuggingFace 镜像
         ``vaishaal/ImageNetV2``（``gated: false``，``license: mit``）取得。原始 S3 路径
         （``s3.amazonaws.com/imagenetv2-public/...``）已 404 失效。详见
         ``report/IMAGENETV2_PROVENANCE.md``。

★★ 口径（务必如实，禁止把它写成 ImageNet-1K 验证集）★★
    ImageNetV2 **不是** ImageNet-1K 验证集，也不含任何 ImageNet-1K 图像。它是按 ImageNet
    分布**重新采样**构建的独立测试集；matched-frequency 变体为 1000 类 × 10 张 = 10,000 张。
    本脚本从中取 1000 张（每类 1 张）构成固定子集。该子集上的准确率与 ImageNet-1K val
    准确率、与论文/官方公布值**不可直接比较**（文献中同模型通常低 10~15 个百分点）。
    类别映射仍沿用 ImageNet-1K 的 1000 个 WNID（ImageNetV2 与 ImageNet-1K 类别体系相同），
    因此 ``labels/imagenet_classes.txt`` 继续有效。

上游目录结构与两个实测坑：
    * 归档内是 ``imagenetv2-matched-frequency-format-val/<label>/<sha1>.jpeg``，
      其中 ``<label>`` 是**十进制类别下标 0..999**（不是 WNID 目录名），每个目录恰好 10 张。
    * 坑 1：文件名虽叫 ``.tar.gz``，内容其实是**不带 gzip 的 POSIX pax tar**
      （首 16 字节 = ``50 61 78 48 ...`` 即 ``PaxHeader/...``，而非 gzip 魔数 ``1f 8b``）。
      用 ``tarfile.open(..., "r:gz")`` 会直接 ``BadGzipFile``；本脚本用 ``"r:*"`` 自动探测。
    * 坑 2：**不要**用 ``torchvision.datasets.ImageFolder`` 的隐式类别序号解释 V2 目录
      —— 目录名是未补零的数字串（``0/1/2/.../10/100/999``），ImageFolder 按**字符串排序**
      编号，得到的是 ``'0','1','10','100',...`` 的错误顺序（上游 issue modestyachts/ImageNetV2#10
      即此问题）。本仓库一律用显式清单 ``<相对路径>\\t<0-based 标签>``，标签取自目录名的
      **十进制数值**，与 ``labels/imagenet_classes.txt`` 的下标一一对应。

选取规则（完全确定性、任何人可 byte-for-byte 复现，**不依赖任何随机种子**）：
    1. 按类别下标升序（0..999）遍历目录 ``<label>``；
    2. 每个目录内把**文件名做 Python ``sorted()``**（文件名是小写十六进制 sha1，等价于字节序），
       取**第一个**；
    3. 逐行写出 ``<相对仓库根的路径>\\t<0-based 标签>``（Tab 分隔，路径相对仓库根），
       与评测脚本 ``tools/eval_pretrained.py`` 要求的清单格式一致。

产物（data/ 不入提交物；清单、provenance 与脚本入提交物）：
    data/imagenetv2/matched-frequency/<label>/<sha1>.jpeg   （10,000 张，每类 10 张）
    datasets/lists/imagenetv2_mf_1000.txt                   （1000 行，官方模型评价主口径清单）

用法：
    python datasets/make_imagenetv2_subset.py --verify-only     # 只校验归档与既有清单，不解压不重写
    python datasets/make_imagenetv2_subset.py                   # 校验 + 解压（幂等）+ 构建清单
    python datasets/make_imagenetv2_subset.py --force-extract    # 强制重新解压
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]          # 仓库根，禁止写死个人绝对路径
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

TARBALL = PROJ / "data/_src/imagenetv2/imagenetv2-matched-frequency.tar.gz"
EXTRACT_ROOT = PROJ / "data/imagenetv2/matched-frequency"
OUT_LIST = PROJ / "datasets/lists/imagenetv2_mf_1000.txt"
WNID_TO_IDX = PROJ / "labels/imagenet_wnid_to_idx.json"

# 官方归档的事实基线（本机实测：体积与 sha256 均与 HF 侧 X-Linked-ETag 一致）
EXPECTED_SIZE = 1264079360
EXPECTED_SHA256 = "f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c"
HF_LINKED_ETAG = "f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c"
IMAGES_PER_CLASS = 10

IMG_EXT = (".jpeg", ".jpg", ".png")
LABEL_RE = re.compile(r"^\d{1,3}$")
# 四个锚点：下标 -> WNID（与 labels/imagenet_wnid_to_idx.json 交叉校验）
ANCHORS = {0: "n01440764", 207: "n02099601", 281: "n02123045", 999: "n15075141"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_tarball(path: Path) -> dict:
    """体积 + sha256 双校验；任一不符直接 SystemExit（不得用其它数据冒充 ImageNetV2）。"""
    if not path.exists():
        raise SystemExit(f"[错误] 找不到归档：{path.relative_to(PROJ).as_posix()}（先下载，见 README §2.2）")
    size = path.stat().st_size
    if size != EXPECTED_SIZE:
        raise SystemExit(f"[错误] 归档体积不符：实测 {size:,} B，期望 {EXPECTED_SIZE:,} B")
    digest = sha256_file(path)
    if EXPECTED_SHA256 and digest != EXPECTED_SHA256:
        raise SystemExit(f"[错误] 归档 sha256 不符：实测 {digest}，期望 {EXPECTED_SHA256}")
    print(f"[tarball] {path.relative_to(PROJ).as_posix()}  {size:,} B  sha256={digest}")
    return {"bytes": size, "sha256": digest, "sha256_matches_hf_etag": digest == HF_LINKED_ETAG}


def extract(path: Path, force: bool = False) -> dict:
    """把 ``<...>/<label>/<sha1>.jpeg`` 规范化成 EXTRACT_ROOT/<label>/<sha1>.jpeg。

    幂等：目标树已完整（1000 个类别目录、每个 10 张图）时直接跳过，除非 --force-extract。
    """
    if not force and EXTRACT_ROOT.is_dir():
        dirs = [d for d in EXTRACT_ROOT.iterdir() if d.is_dir() and LABEL_RE.match(d.name)]
        if len(dirs) == 1000 and all(
                len([f for f in d.iterdir() if f.suffix.lower() in IMG_EXT]) == IMAGES_PER_CLASS
                for d in dirs):
            print(f"[extract] 已存在完整目录树（1000 类 × {IMAGES_PER_CLASS} 张），跳过解压："
                  f"{EXTRACT_ROOT.relative_to(PROJ).as_posix()}")
            return {"skipped": True, "dirs": len(dirs)}

    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    written = 0
    roots: Counter = Counter()
    with tarfile.open(path, "r:*") as tf:      # 自动探测：上游实际是不带 gzip 的 pax tar
        for m in tf.getmembers():
            if not m.isfile():
                continue
            parts = [p for p in m.name.replace("\\", "/").split("/") if p not in ("", ".")]
            if len(parts) < 2:
                continue
            label, fname = parts[-2], parts[-1]
            if not LABEL_RE.match(label) or not (0 <= int(label) <= 999):
                continue
            if not fname.lower().endswith(IMG_EXT) or ".." in fname:
                continue
            roots["/".join(parts[:-2])] += 1
            dst_dir = EXTRACT_ROOT / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / fname
            if dst.exists() and dst.stat().st_size > 0:
                written += 1
                continue
            src = tf.extractfile(m)
            assert src is not None, m.name
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            tmp.write_bytes(src.read())
            os.replace(tmp, dst)          # 先写临时文件再原子替换：中断不留半张图
            written += 1
    print(f"[extract] {EXTRACT_ROOT.relative_to(PROJ).as_posix()}  写出 {written:,} 张；"
          f"归档内顶层目录：{dict(roots.most_common(3))}")
    return {"skipped": False, "written": written, "tar_roots": dict(roots)}


def check_label_mapping() -> dict:
    """目录名（十进制下标）→ WNID → labels/imagenet_classes.txt 行号的映射自检。"""
    wnid2idx = json.loads(WNID_TO_IDX.read_text(encoding="utf-8"))
    assert len(wnid2idx) == 1000, f"WNID→下标表应为 1000 项，实际 {len(wnid2idx)}"
    idx2wnid = {v: k for k, v in wnid2idx.items()}
    assert sorted(idx2wnid) == list(range(1000)), "WNID→下标表的下标不是 0..999"
    for idx, wn in ANCHORS.items():
        assert idx2wnid[idx] == wn, f"锚点不符：下标 {idx} 应为 {wn}，实际 {idx2wnid[idx]}"

    dirs = sorted((d.name for d in EXTRACT_ROOT.iterdir() if d.is_dir()),
                  key=lambda s: int(s))
    assert dirs == [str(i) for i in range(1000)], (
        f"目录集合不是 '0'..'999'（前 5 = {dirs[:5]}，共 {len(dirs)} 个）")
    return {"idx2wnid_anchors": {str(k): idx2wnid[k] for k in ANCHORS},
            "dirs": len(dirs)}


def build_list() -> dict:
    """按「每类别目录文件名排序取第一张」构建 1000 行确定性清单。"""
    mapping = check_label_mapping()
    wnid2idx = json.loads(WNID_TO_IDX.read_text(encoding="utf-8"))

    rows: list[tuple[str, int]] = []
    files_per_class = Counter()
    for idx in range(1000):                      # 严格按类别下标升序，保证可复现
        d = EXTRACT_ROOT / str(idx)
        files = sorted(f.name for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
        assert files, f"{idx} 目录下没有图片文件"
        files_per_class[len(files)] += 1
        rel = (d / files[0]).relative_to(PROJ)   # 规则第 2 条：文件名排序取第一张
        assert rel.as_posix().split("/")[-2] == str(idx)
        rows.append((rel.as_posix(), idx))

    assert len(rows) == 1000, len(rows)
    labels = [y for _, y in rows]
    assert labels == list(range(1000)), "标签不是 0..999 升序"
    assert len(set(labels)) == 1000, "标签有重复"
    for idx, wn in ANCHORS.items():
        assert rows[idx][0].split("/")[-2] == str(idx)

    OUT_LIST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_LIST, "w", encoding="utf-8", newline="\n") as f:
        for rel, y in rows:
            f.write(f"{rel}\t{y}\n")
    digest = sha256_file(OUT_LIST)
    print(f"[list]    {OUT_LIST.relative_to(PROJ).as_posix()}  {len(rows)} 行  sha256={digest}")
    return {
        "out": OUT_LIST.relative_to(PROJ).as_posix(),
        "rows": len(rows),
        "sha256": digest,
        "bytes": OUT_LIST.stat().st_size,
        "num_classes": len(set(labels)),
        "per_class_min": 1, "per_class_max": 1,
        "label_min": min(labels), "label_max": max(labels),
        "source_images_per_class": {str(k): v for k, v in sorted(files_per_class.items())},
        "anchors": {str(k): {"dir": rows[k][0].split("/")[-2], "wnid": ANCHORS[k],
                             "file": rows[k][0].split("/")[-1]} for k in ANCHORS},
        "label_mapping": mapping,
        "rule": ("每类别目录（目录名=十进制 0-based 下标）内文件名 sorted() 取第一个；"
                 "按下标 0..999 升序写行；无随机种子"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="校验 ImageNetV2 归档并构建 1000 张确定性固定子集")
    ap.add_argument("--verify-only", action="store_true",
                    help="只校验归档 sha256 与既有清单，不解压、不重写清单")
    ap.add_argument("--force-extract", action="store_true", help="即使目录树已完整也重新解压")
    a = ap.parse_args()

    rep: dict = {"tarball": str(TARBALL.relative_to(PROJ)).replace("\\", "/")}
    rep["verify"] = verify_tarball(TARBALL)

    if a.verify_only:
        if not OUT_LIST.exists():
            raise SystemExit(f"[错误] 清单不存在：{OUT_LIST.relative_to(PROJ).as_posix()}")
        rows = [l for l in OUT_LIST.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rows) == 1000, f"清单应为 1000 行，实际 {len(rows)}"
        rep["list"] = {"out": OUT_LIST.relative_to(PROJ).as_posix(), "rows": len(rows),
                       "sha256": sha256_file(OUT_LIST)}
        rep["label_mapping"] = check_label_mapping()
        print(json.dumps(rep, ensure_ascii=True, indent=2))
        return 0

    rep["extract"] = extract(TARBALL, force=a.force_extract)
    rep["list"] = build_list()
    print(json.dumps(rep, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
