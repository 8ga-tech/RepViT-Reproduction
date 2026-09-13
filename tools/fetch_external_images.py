#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/fetch_external_images.py —— 构建「训练集以外的实际图片」外部测试集。

Source: Self-written。图片来自本机已落盘的 ImageNet-1K 验证集（ImageFolder，见
        datasets/unpack_imagenet_val.py），**不是** Oxford-IIIT Pet 数据集。

## 为什么用这个来源（以及它为什么成立）
规格书 §8.6.1 的首选是「自己或同学拍摄的宠物照」，其次是 Wikimedia Commons。
本次执行时 commons.wikimedia.org 与 upload.wikimedia.org 均**不可达**（curl 超时，
http=000），无法取得 CC 授权图片。因此改用**跨集合的真实照片**：

  - ImageNet-1K val 是从网络收集的真实照片，采集者、拍摄设备、构图习惯、后处理
    都与 Oxford-IIIT Pet（2007–2012 年的宠物摄影）**完全不同**；
  - 其中若干类（beagle / pug / boxer ...）与 Pet 的 37 个品种同名同种，
    可以构成真正意义上的「同品种、跨集合」分布差异测试；
  - 它不属于 Oxford-IIIT Pet 的训练集、验证集或测试集，因此满足题目
    「训练集以外的实际图片」这一语义（题目限制的是不能拿 Pet 自己的 test 充数）。

## 诚实声明（会写进 manifest 与报告）
这不是「实拍原创照片」，而是「跨数据集采集的真实照片」。受网络条件限制，
规格书 §8.6.1 的首选与次选来源都不可用，此处做了替代并在报告中显式标注，
不把替代品说成原方案。每张图的来源、许可与类别对应关系都记录在
external/images_manifest.csv 里。

用法：
    python tools/fetch_external_images.py --num 8
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ImageNet wnid -> (ImageNet 类名, 对应的 Pet 类名)。只用同名同种的类，
# 其余 Pet 品种在 ImageNet 里没有一一对应的类，不做牵强映射。
BREED_MAP = {
    "n02088364": ("beagle",                 "Beagle"),
    "n02110958": ("pug",                    "Pug"),
    "n02108089": ("boxer",                  "Boxer"),
    "n02085620": ("Chihuahua",              "Chihuahua"),
    "n02111889": ("Samoyed",                "Samoyed"),
    "n02111277": ("Newfoundland",           "Newfoundland"),
    "n02111500": ("Great Pyrenees",         "Great Pyrenees"),
    "n02112018": ("Pomeranian",             "Pomeranian"),
    "n02123394": ("Persian cat",            "Persian"),
    "n02124075": ("Egyptian cat",           "Egyptian Mau"),
    "n02123597": ("Siamese cat",            "Siamese"),
    "n02112137": ("chow",                   None),
    "n02098413": ("Lhasa",                  None),
}
# 12 个候选里取前 N 个有 Pet 对应的
CANDIDATES = [(w, v[0], v[1]) for w, v in BREED_MAP.items() if v[1]]


def main() -> int:
    ap = argparse.ArgumentParser(description="构建外部（跨集合）真实图片测试集")
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--imagenet-val", default="data/imagenet/val")
    ap.add_argument("--out-dir", default="external")
    a = ap.parse_args()

    val_root = ROOT / a.imagenet_val
    if not val_root.is_dir():
        raise SystemExit(f"找不到 {val_root}；请先运行 datasets/unpack_imagenet_val.py")
    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for wnid, in_name, pet_name in CANDIDATES[:a.num]:
        d = val_root / wnid
        if not d.is_dir():
            print(f"[skip] {wnid} 不存在")
            continue
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in (".jpeg", ".jpg", ".png"))
        if not files:
            continue
        src = files[0]                      # 固定取排序后的第一张，保证可复现
        dst = out / f"{pet_name.replace(' ', '_').lower()}__{src.name}"
        shutil.copyfile(src, dst)
        rows.append({
            "filename": dst.name,
            "source": f"ImageNet-1K val ({wnid})",
            "license": "ImageNet 数据集条款（非商业研究用途）；本仓库仅作课程考核的可解释性演示",
            "note": f"ImageNet 类 '{in_name}' -> Pet 品种 '{pet_name}'；跨集合真实照片",
            "imagenet_wnid": wnid,
            "pet_class": pet_name,
        })
        print(f"[get] {dst.name}  <- {src.name}")

    man = out / "images_manifest.csv"
    with open(man, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "source", "license", "note",
                                          "imagenet_wnid", "pet_class"])
        w.writeheader()
        w.writerows(rows)
    print(f"[manifest] {man.relative_to(ROOT).as_posix()}  n={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
