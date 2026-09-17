# datasets/imagenet_subset.py
# -*- coding: utf-8 -*-
"""**固定 ImageNet 评价清单**的 Dataset 读取器（规格书 2.3.2；当前唯一口径 = ImageNetV2）。

> 2026-09 口径变更：官方模型评价的清单是 ``datasets/lists/imagenetv2_mf_1000.txt``
> （ImageNetV2 matched-frequency 的 1000 张确定性固定子集，1000 类各 1 张，
> 选取规则见 ``datasets/make_imagenetv2_subset.py`` / ``report/IMAGENETV2_PROVENANCE.md``）。
> 本文件**只读清单、不构建清单**，因此变更口径时只需换清单文件。

职责边界（本文件是全仓最容易出「静默 bug」的地方）：

* **只读**清单文件，**绝不做任何划分或采样**——清单一旦冻结，任何"顺手抽一半做验证"
  都会让 Top-1 数字不可复现。
* 解析必须用 ``line.rsplit(None, 1)``（``None`` = 按任意空白串切分），这样 Tab 与空格
  两种分隔符、**含空格的 Windows 路径**都能正确解析。用 ``line.split()`` 遇到含空格路径
  必然切错：路径被砍成两段、标签变成路径尾巴，``int()`` 抛错还算幸运，若尾巴恰好是数字
  就会静默产生一条标签错位的样本，最终表现为 Top-1 掉到随机水平。

列表行格式（当前 ImageNetV2 清单的形态）::

    data/imagenetv2/matched-frequency/0/58fbc3e7….jpeg\t0

即「相对**仓库根**的路径 + TAB + 0-based 标签（0..999）」。旧式 ``n01751748/xxx.JPEG`` 这种
相对 ``data/imagenet/val`` 的写法仍能解析（:meth:`path` 有兜底），但已不在当前口径中。

两条诊断判据（写进报告用）：

* ``check_label_range`` 的 ``suspect``：``min>=1``（疑似 1-based 未减 1）或
  ``max<num_classes-1``（疑似映射错位/抽样偏差）。注意 500 张的抽样子集 ``max=998``，
  ``suspect`` 会如实为 True，**这是正常的**，所以它只警告不报错。
* ``verify_with_imagefolder``：用 ``ImageFolder(root).class_to_idx`` 对每张图**逐图反查**，
  比 min/max 硬得多。若 Top-1 掉到 0.1% 附近（1/1000 的随机水平），基本可判定标签体系整体错位。
  ⚠ ImageNetV2 的类目录是**未补零的数字目录**（``0/1/10/100/…``），ImageFolder 的隐式序号
  会按字符串排序错位（上游 issue #10），因此该自检对 ImageNetV2 目录**不适用**：请用
  ``datasets/make_imagenetv2_subset.py`` 的锚点校验代替。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from PIL import Image, ImageFile, ImageOps
from torch.utils.data import Dataset

PROJ = Path(__file__).resolve().parents[1]      # 仓库根，禁止写死绝对路径
IMAGENET_NUM_CLASSES = 1000
IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# 与 torchvision 一致：宁可报错，也不要静默产生坏样本
ImageFile.LOAD_TRUNCATED_IMAGES = False


def _abs_path(p: str | Path) -> Path:
    """相对路径一律相对**仓库根**解析（不依赖 cwd，也不写死机器）。"""
    p = Path(p)
    return p if p.is_absolute() else (PROJ / p)


def _resolve_image_path(rel: str, root: str | Path | None = None) -> Path:
    """把列表里的一条路径解析成真实文件路径，容忍两种书写约定。

    规格书 3.5.3 的口径是「相对 ``data/imagenet/val``」，而当前交付的
    ``datasets/lists/imagenetv2_mf_1000.txt`` 里写的是**相对仓库根**
    （``data/imagenetv2/matched-frequency/0/xxx.jpeg``）——两种都在真实项目里出现过，
    所以这里按「绝对路径 -> root/rel -> 仓库根/rel -> cwd/rel」依次探测，
    取第一个存在的；都不存在时返回主约定 ``root/rel`` 供报错展示。
    逐条 ``exists()`` 只是 stat，1000 条约几十毫秒，换来的是换机器不返工。
    （``verify_with_imagefolder`` 不受影响：它只按路径的目录名查 synset，与根无关。）
    """
    p = Path(rel)
    if p.is_absolute():
        return p
    rel_posix = rel.replace("\\", "/")              # 列表里是 POSIX 路径，Windows 上也照拼
    bases: list[Path] = []
    if root is not None:
        bases.append(Path(root))
    bases += [PROJ, Path(".")]
    for base in bases:
        cand = base / rel_posix
        if cand.exists():
            return cand
    return (Path(root) if root is not None else PROJ) / rel_posix


def parse_list_file(path: str | Path) -> list[tuple[str, int]]:
    """解析 ``<path>\\t<label>`` 列表，返回 ``[(rel_or_abs_path, label)]``。

    * 分隔符统一 ``rsplit(None, 1)``：Tab / 多个空格 / 含空格路径通吃；
    * 容忍空行与 ``#`` 注释行；
    * 路径两侧的空白与成对引号被剥掉（Windows 上复制粘贴常带引号）；
    * 路径**保留扩展名**（与本仓库 Pet 侧 ``parse_split.py`` 的"去扩展名"口径不同：
      ImageNet 列表里路径带目录，不能丢）；
    * 解析不出整数标签时**立刻抛错并打印行号**，绝不静默跳过——静默跳过会让
      ``n`` 悄悄缩水，评测分母就错了。
    """
    path = _abs_path(path)
    if not path.exists():
        raise SystemExit(f"[FATAL] 找不到列表文件: {path}")
    rows: list[tuple[str, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.rsplit(None, 1)               # 关键：不能写 split()
            if len(parts) != 2:
                raise SystemExit(f"[FATAL] {path}:{lineno} 需要 '<path> <label>' 两列，实收 {s!r}")
            rel, lab = parts[0].strip().strip('"'), parts[1].strip()
            try:
                y = int(lab)
            except ValueError:
                raise SystemExit(
                    f"[FATAL] {path}:{lineno} 标签不是整数: {lab!r}；整行 {s!r}。"
                    f"若报错处是路径被切断（含空格的路径用 line.split() 解析），"
                    f"说明这一行不是 '<path> <label>' 制式")
            if not rel:
                raise SystemExit(f"[FATAL] {path}:{lineno} 路径为空")
            rows.append((rel, y))
    if not rows:
        raise SystemExit(f"[FATAL] {path} 未解析出任何样本")
    return rows


def check_label_range(items: Sequence[tuple[str, int]],
                      num_classes: int = IMAGENET_NUM_CLASSES,
                      root: str | Path | None = None) -> dict:
    """标签范围体检，返回 ``{'n','min','max','missing_files','suspect'}``。

    * ``missing_files``：按 :func:`_resolve_image_path` 的规则（root -> 仓库根 -> cwd）
      都找不到文件的条数——所以「列表相对仓库根写」与「相对 data.root 写」两种约定
      都能正确判 0。找不到文件只说明"这个根目录在当前机器上不存在"，**不代表标签有问题**。
    * ``suspect=True`` 当 ``min>=1`` 或 ``max<num_classes-1``，提示可能 1-based / 映射错位。
      注意：500 张的均衡子集每类 1 张、只覆盖偶数下标，``max=998 < 999`` 会如实置 True，
      这是**抽样偏差**而非错误——所以调用方只应打印警告，不得据此抛异常。
    """
    n = len(items)
    if n == 0:
        return {"n": 0, "min": None, "max": None, "missing_files": 0, "suspect": False}
    labels = [int(y) for _p, y in items]
    lo, hi = min(labels), max(labels)
    missing = sum(1 for rel, _y in items
                  if not _resolve_image_path(rel, root).exists())
    return {
        "n": n,
        "min": lo,
        "max": hi,
        "missing_files": missing,
        "suspect": bool(lo >= 1 or hi < int(num_classes) - 1),
    }


def verify_with_imagefolder(items: Sequence[tuple[str, int]],
                            root: str | Path) -> tuple[int, int]:
    """用 ``ImageFolder(root).class_to_idx`` 逐图反查标签，返回 ``(一致数, 不一致数)``。

    这是比 min/max 更硬的判据：min/max 在抽样子集上可能因采样偏差失效
    （500 张不一定触到 0 或 999），而逐图反查是精确校验。

    反查方式：取列表路径的**倒数第二段目录名**（ImageNet val 的 synset，如 ``n01751748``），
    查 ``class_to_idx`` 得到下标，与列表里的标签比对。目录名不在 ``class_to_idx`` 里
    （目录结构不对、路径被切错、列表被换了根）一律计入"不一致"。
    """
    root = _abs_path(root)
    if not root.is_dir():
        raise SystemExit(f"[FATAL] {root} 不是目录，无法用 ImageFolder 校验")
    try:
        from torchvision.datasets import ImageFolder
        class_to_idx = ImageFolder(root=str(root)).class_to_idx
    except Exception as exc:                        # 目录不存在 / 结构不对 / 空目录
        raise SystemExit(f"[FATAL] ImageFolder({root}) 失败: {exc}；"
                         f"期望结构 root/<synset>/<img>，且需先跑 valprep.sh")
    agree = disagree = 0
    bad: list[tuple[str, int, str, int | None]] = []
    for rel, y in items:
        parts = rel.replace("\\", "/").split("/")
        folder = parts[-2] if len(parts) >= 2 else ""
        idx = class_to_idx.get(folder)
        if idx == y:
            agree += 1
        else:
            disagree += 1
            if len(bad) < 5:
                bad.append((rel, y, folder, idx))
    print(f"[label] verify_with_imagefolder: 一致 {agree} / 不一致 {disagree} (n={len(items)})")
    if bad:
        print(f"[label] 不一致样例 (path, 列表标签, 目录名, ImageFolder 下标): {bad}")
    return agree, disagree


class ImageNetSubset(Dataset):
    """考核方固定子集，返回 ``(img_tensor, label)`` 或 ``(img_tensor, label, rel_path)``。

    参数
    ----
    root
        列表里相对路径的根，通常 ``data/imagenet/val``（其下是 ``<synset>/<xxx>.JPEG``）。
        相对路径按仓库根解析。列表里若写的是**相对仓库根**的路径（当前交付的
        ``imagenetv2_mf_1000.txt`` 就是这种），:meth:`path` 会自动回退到仓库根，
        两种约定都能读，无需改列表。
    list_file
        ``datasets/lists/imagenetv2_mf_1000.txt``（ImageNetV2 固定子集；相对路径按仓库根解析）。
    transform
        PIL Image -> Tensor；``None`` 时返回原始 PIL 图。
    return_path
        ``True`` 时返回列表里写的那条相对路径——预测 CSV、Top-5 可视化、逐类错误案例
        都要靠它把一行结果对回具体图片。
    num_classes
        标签上界（默认 1000）。越界即 ``SystemExit``：这类错误必须**在建 Dataset 时**炸，
        不能等到 loss 里才崩（那时已经跑了几分钟且看不出是哪张图）。
    """

    def __init__(self, root: str | Path, list_file: str | Path,
                 transform: Callable | None = None, return_path: bool = False,
                 num_classes: int = IMAGENET_NUM_CLASSES) -> None:
        self.root = _abs_path(root)
        self.list_file = _abs_path(list_file)
        self.transform = transform
        self.return_path = bool(return_path)
        self.num_classes = int(num_classes)
        if not self.root.is_dir():
            raise SystemExit(f"[FATAL] 图片根目录不存在: {self.root}（列表里的路径以此为根）")

        self.items = parse_list_file(self.list_file)          # [(rel, label)]
        # 一次性解析出绝对路径（1000 条 stat 而已），避免每个 batch 重复探测
        self.paths = [_resolve_image_path(rel, self.root) for rel, _y in self.items]
        report = check_label_range(self.items, self.num_classes, root=self.root)

        lo, hi = report["min"], report["max"]
        if lo is None or lo < 0 or hi >= self.num_classes:
            raise SystemExit(
                f"[FATAL] {self.list_file} 标签越界: min={lo} max={hi}，"
                f"合法范围 0..{self.num_classes - 1}。常见原因：官方 CLASS-ID 是 1-based 未减 1，"
                f"或标签被写成了 WNID / devkit 的行号")
        if report["suspect"]:
            print(f"[WARN] {self.list_file} label min={lo} max={hi}（num_classes={self.num_classes}）"
                  f"判定 suspect：可能是 1-based、映射错位，或只是抽样子集没触到首末类。"
                  f"以 verify_with_imagefolder() 的逐图反查为准")
        if report["missing_files"]:
            print(f"[WARN] {self.list_file}: {report['missing_files']}/{report['n']} 条在 "
                  f"{self.root} 下找不到文件（缺图会在 __getitem__ 处报错，不静默跳过）")
        # 打印实际生效的路径约定：列表写的是「相对 data.root」还是「相对仓库根」一目了然。
        # 这类不一致一旦静默，表现就是 1000 张全部读不到，排查成本远高于打这一行。
        n_repo = sum(1 for rel, _y in self.items
                     if not (self.root / rel.replace("\\", "/")).exists())
        conv = "相对 data.root" if n_repo == 0 else f"相对 repo root/绝对(回退 {n_repo} 条)"
        print(f"[data] ImageNetSubset n={report['n']} root={self.root} "
              f"label_range=[{lo},{hi}] num_classes={self.num_classes} 路径={conv}")

    def __len__(self) -> int:
        return len(self.items)

    def path(self, idx: int) -> Path:
        """第 ``idx`` 条样本的**绝对**图片路径（部署侧对齐/可视化用）。"""
        return self.paths[idx]

    def __getitem__(self, idx: int):
        rel, label = self.items[idx]
        path = self.path(idx)
        if not path.exists():
            raise SystemExit(f"[FATAL] 图片不存在: {path}（来自 {self.list_file} 第 {idx + 1} 条）")
        # with + load()：Windows 上 Image.open 不关闭句柄，长跑会耗尽文件句柄
        with open(path, "rb") as f:
            img = Image.open(f)
            img.load()
            img = ImageOps.exif_transpose(img)      # ImageNet 无 EXIF 旋转，保险起见
            img = img.convert("RGB")                # 丢掉 CMYK/RGBA：漏掉这步 ToTensor 会出 4 通道
        if self.transform is not None:
            img = self.transform(img)
        if self.return_path:
            return img, label, rel
        return img, label

    def labels(self) -> list[int]:
        """0-based 标签列表（供 top-k / 混淆矩阵统计）。"""
        return [y for _p, y in self.items]

    def __repr__(self) -> str:                      # pragma: no cover - 仅调试可读性
        return (f"ImageNetSubset(n={len(self.items)}, root={self.root}, "
                f"return_path={self.return_path})")


# ---------------------------------------------------------------- 自检
def _selfcheck() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="ImageNet 固定子集 Dataset 自检（只读，不采样）")
    ap.add_argument("--root", default="data/imagenet/val", help="列表里相对路径的根")
    ap.add_argument("--list", dest="list_file", default="datasets/lists/imagenetv2_mf_1000.txt",
                    help="固定评价清单；当前唯一口径 = ImageNetV2 matched-frequency 的 1000 张子集")
    ap.add_argument("--num-classes", type=int, default=IMAGENET_NUM_CLASSES)
    ap.add_argument("--verify", action="store_true", help="用 ImageFolder 逐图反查（需本地有图）")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    list_path = _abs_path(a.list_file)
    if not list_path.exists():
        print(f"[SKIP] 找不到 {list_path}，跳过 ImageNet 子集自检")
        return 0

    items = parse_list_file(list_path)
    rep = check_label_range(items, a.num_classes, root=_abs_path(a.root))
    print(f"[list] {list_path.name}: n={rep['n']} label_min={rep['min']} label_max={rep['max']} "
          f"missing_files={rep['missing_files']} suspect={rep['suspect']}")

    # 最容易踩的静默 bug：含空格的路径 + line.split()。这里直接构造一条对照样本验证。
    probe = Path(list_path).with_name(list_path.stem + "_tmp_probe.txt")
    try:
        probe.write_text("some dir with spaces/ILSVRC2012_val_00000001.JPEG\t7\n", encoding="utf-8")
        got = parse_list_file(probe)
        assert got == [("some dir with spaces/ILSVRC2012_val_00000001.JPEG", 7)], got
        print("[OK] rsplit(None, 1) 正确解析含空格路径 + Tab 分隔")
    finally:
        probe.unlink(missing_ok=True)

    if not _abs_path(a.root).is_dir():
        print(f"[SKIP] {_abs_path(a.root)} 不存在，跳过读图与 ImageFolder 校验")
        return 0
    if a.verify:
        verify_with_imagefolder(items, a.root)

    ds = ImageNetSubset(a.root, a.list_file, transform=None, return_path=True,
                        num_classes=a.num_classes)
    for i in range(min(a.batch, len(ds))):
        img, y, rel = ds[i]
        assert img.mode == "RGB" and 0 <= y < a.num_classes, (img.mode, y)
        assert Path(rel).name and ds.path(i).exists()
    print(f"[OK] 前 {min(a.batch, len(ds))} 条可读，模式 RGB，标签在 0..{a.num_classes - 1}")
    print("[OK] datasets/imagenet_subset.py 自检完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
