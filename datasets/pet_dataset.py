# datasets/pet_dataset.py
# -*- coding: utf-8 -*-
"""Oxford-IIIT Pet 37 类数据集（规格书 2.3.2）。

用法::

    from datasets.pet_dataset import PetDataset, load_pet_split
    ids = load_pet_split("datasets/lists", "val")           # 740 个 image_id
    ds = PetDataset("data/oxford-iiit-pet", ids=ids, transform=val_tf, return_id=True)
    x, y, image_id = ds[0]                                  # image_id 用于失败案例溯源与 Grad-CAM

三个必须踩准的事实（每一条都对应一次真实返工）：

1) 官方 ``annotations/`` 下**只有** ``trainval.txt``(3680) 与 ``test.txt``(3669)，
   **不存在** ``train.txt`` / ``val.txt``；train/val 必须自行从 trainval 分层切分，
   每类固定 20 张进 val -> train 2940 / val 740。产物冻结在
   ``datasets/lists/pet_{train,val,test}.txt``，由 ``datasets/make_pet_split.py``
   （CLI）与本模块的 :func:`make_pet_split` （库函数）生成，两者参数与产物路径一致，
   任何其它章节不得再定义第二套划分。

2) 官方 4 列标注 ``<image_id> CLASS-ID SPECIES BREED-ID`` 里的 ``CLASS-ID`` 是
   **1-based**（1..37）。0-based 标签 = ``int(CLASS-ID) - 1``；忘了减 1 会让
   ``CrossEntropyLoss`` 在标签 37 上崩（合法下标只有 0..36）。

3) 类名必须用 ``image_id.rsplit("_", 1)[0]`` 取前缀，**不能**写 ``split("_")[0]``——
   ``american_pit_bull_terrier_191`` 用后者会切出假类 ``american``，37 类会塌成 35 类。
   取到前缀后再「下划线转空格 + ``.title()``」，即 torchvision ``OxfordIIITPet.classes``
   的口径（大写开头的 12 个猫排在小写开头的 25 个狗之前）。本模块 ``__main__`` 会与
   ``torchvision.datasets.OxfordIIITPet`` 逐项交叉校验。

另外两条与"静默 bug"有关的约定：

* 行解析统一用 ``line.rsplit(None, 1)``（按任意空白串从右切一次），Tab 与空格通吃；
  用 ``line.split()`` 遇到含空格的路径必然切错，标签会整体错位而**不报错**。
* 读图必须 ``with open(..., "rb")`` + ``Image.open(f).load()``：Windows 上
  ``Image.open(p)`` 不关闭句柄，7390 张图跑几个 epoch 就会 ``Too many open files``。
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image, ImageFile, ImageOps
from torch.utils.data import Dataset

# ---------------------------------------------------------------- 常量
PROJ = Path(__file__).resolve().parents[1]      # 仓库根，禁止写死绝对路径
IMG_SUBDIR = "images"
ANN_SUBDIR = "annotations"
LISTS_DIR = "datasets/lists"                    # 冻结划分产物的唯一目录
LABELS_DIR = "labels"

PET_NUM_CLASSES = 37
N_TRAINVAL, N_TEST = 3680, 3669                 # 官方两个 txt 的行数，写进断言以便早期发现数据集被换过
DEFAULT_VAL_PER_CLASS = 20
DEFAULT_SEED = 42
SPLITS = ("train", "val", "test")

_WARNED: set[str] = set()   # 同一类警告只打一次：pet_class_to_idx 会再调一次 pet_class_names

# 与 torchvision 一致：宁可报错，也不要静默产生坏样本（截断图会变成半张灰图）
ImageFile.LOAD_TRUNCATED_IMAGES = False


def _abs_path(p: str | Path) -> Path:
    """相对路径一律相对**仓库根**解析（不依赖 cwd，也不写死机器）。"""
    p = Path(p)
    return p if p.is_absolute() else (PROJ / p)


# ---------------------------------------------------------------- 标注解析
def parse_pet_annotation(path: str | Path) -> list[tuple[str, int, int, int]]:
    """解析官方 4 列标注，返回 ``[(image_id, class_idx0, species0, breed0)]``。

    * 官方制式：``<image_id> CLASS-ID SPECIES BREED-ID``（空格分隔，定长 4 列）；
    * 三个编号全部转 0-based：``class_idx0 = CLASS-ID - 1``（0..36），
      ``species0 = SPECIES - 1``（0=猫 / 1=狗），``breed0 = BREED-ID - 1``（物种内品种号，
      **不是** 37 类标签，只有写物种分类时才用得到）；
    * 容忍 ``#`` 开头的表头/注释行与空行（``annotations/list.txt`` 前 6 行就是注释）。
    """
    path = _abs_path(path)
    if not path.exists():
        raise SystemExit(f"[FATAL] 找不到标注文件: {path}")
    rows: list[tuple[str, int, int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) != 4:
                raise SystemExit(
                    f"[FATAL] {path}:{lineno} 需要 4 列 '<image_id> CLASS-ID SPECIES BREED-ID'，"
                    f"实收 {len(parts)} 列: {s!r}")
            image_id, cid, species, breed = parts
            cid, species, breed = int(cid), int(species), int(breed)
            if not 1 <= cid <= PET_NUM_CLASSES:
                raise SystemExit(f"[FATAL] {path}:{lineno} CLASS-ID={cid} 越界（官方为 1..37）")
            if species not in (1, 2):
                raise SystemExit(f"[FATAL] {path}:{lineno} SPECIES={species} 越界（官方 1=猫 2=狗）")
            rows.append((image_id, cid - 1, species - 1, breed - 1))
    if not rows:
        raise SystemExit(f"[FATAL] {path} 未解析出任何样本")
    return rows


def _annotation_files(root: str | Path) -> list[Path]:
    """标注文件清单：优先 ``list.txt``（官方全量索引，7349 行 = trainval 3680 + test 3669，
    已剔除 41 张不在标注里的孤儿图）；缺失时退化为 ``trainval.txt ∪ test.txt``。

    两个函数的遍历顺序全靠它固定，禁止在调用处各写一份。
    """
    ann = _abs_path(root) / ANN_SUBDIR
    list_txt = ann / "list.txt"
    if list_txt.exists():
        return [list_txt]
    return [p for p in (ann / "trainval.txt", ann / "test.txt") if p.exists()]


def _annotation_index(root: str | Path) -> dict[str, tuple[int, int, int]]:
    """``image_id -> (class_idx0, species0, breed0)``，覆盖 train/val/test 三个集合。"""
    sources = _annotation_files(root)
    if not sources:
        raise SystemExit(f"[FATAL] {_abs_path(root) / ANN_SUBDIR} 下找不到 "
                         f"list.txt / trainval.txt / test.txt 任何一个")
    index: dict[str, tuple[int, int, int]] = {}
    for src in sources:
        if not src.exists():
            continue
        for image_id, cid0, sp0, br0 in parse_pet_annotation(src):
            prev = index.get(image_id)
            if prev is not None and prev != (cid0, sp0, br0):
                raise SystemExit(f"[FATAL] 标注自相矛盾: {image_id} {prev} vs {(cid0, sp0, br0)} ({src.name})")
            index[image_id] = (cid0, sp0, br0)
    if not index:
        raise SystemExit(f"[FATAL] {ann} 下找不到 list.txt / trainval.txt / test.txt 任何一个")
    return index


def pet_class_names(root: str | Path) -> list[str]:
    """37 个类名，下标 = 0-based 标签（复刻 ``torchvision.OxfordIIITPet.classes``）。

    公式：``" ".join(part.title() for part in prefix.split("_"))``，
    其中 ``prefix = image_id.rsplit("_", 1)[0]``。
    例：``american_pit_bull_terrier_191`` -> ``American Pit Bull Terrier``（下标 2）。
    """
    sources = _annotation_files(root)
    if not sources:
        raise SystemExit(f"[FATAL] {_abs_path(root) / ANN_SUBDIR} 下找不到任何标注文件")
    idx2prefix: dict[int, str] = {}
    for src in sources:
        for image_id, cid0, _sp0, _br0 in parse_pet_annotation(src):
            prefix = image_id.rsplit("_", 1)[0]     # 必须 rsplit：split('_')[0] 会切出假类名
            name = " ".join(part.title() for part in prefix.split("_"))
            if cid0 in idx2prefix and idx2prefix[cid0] != name:
                raise SystemExit(f"[FATAL] CLASS-ID {cid0} 对应多个前缀: {idx2prefix[cid0]} / {name}")
            idx2prefix[cid0] = name
    missing = [i for i in range(PET_NUM_CLASSES) if i not in idx2prefix]
    if missing:
        raise SystemExit(f"[FATAL] 类别下标缺失 {missing}，实得 {len(idx2prefix)} 类（应为 37）")
    names = [idx2prefix[i] for i in range(PET_NUM_CLASSES)]
    if len(set(names)) != PET_NUM_CLASSES:
        raise SystemExit(f"[FATAL] 类名有重复，说明前缀切分错了: {names}")

    # 与 labels/pet_classes.txt 交叉校验（文件缺失时静默跳过）。
    # 这里刻意只警告不抛错：本函数的返回值永远以官方标注为准，而 labels/ 下的文件是
    # 另一个脚本的产物（可能是官方前缀口径或过期的旧版），它错了不该让 PetDataset 建不起来。
    # 但必须吼出来——类名整体口径不一致会让混淆矩阵的每一行都贴错名字，是最难发现的可视化 bug。
    frozen = _abs_path(LABELS_DIR) / "pet_classes.txt"
    if frozen.exists():
        got = [ln.strip() for ln in frozen.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if got != names and str(frozen) not in _WARNED:
            _WARNED.add(str(frozen))
            as_prefix = [g.lower().replace(" ", "_") for g in got]
            if as_prefix == [n.lower().replace(" ", "_") for n in names]:
                print(f"[WARN] {frozen} 用的是官方**前缀**口径（小写下划线，"
                      f"如 {got[36]!r}），不是 torchvision 的显示名（{names[36]!r}）。"
                      f"可视化/报告请以本函数为准，或跑 datasets/build_pet_class_map.py 重新生成该文件")
            else:
                bad = [(i, a, b) for i, (a, b) in enumerate(zip(got, names)) if a != b]
                print(f"[WARN] {frozen} 与官方标注不一致（{len(bad)}/{len(names)} 处），"
                      f"前几处 (i, 文件, 实算): {bad[:5]}；以本函数为准")
    return names


def pet_class_to_idx(root: str | Path) -> dict[str, int]:
    """``{类名: 0..36}``，扁平字典；与 ``labels/pet_class_to_idx.json`` 同序同值。

    与 :func:`pet_class_names` 同样的处理：返回值以官方标注为准，
    与 ``labels/pet_class_to_idx.json`` 不一致时只打印警告（见那边的原因说明）。
    """
    names = pet_class_names(root)
    out = {n: i for i, n in enumerate(names)}
    frozen = _abs_path(LABELS_DIR) / "pet_class_to_idx.json"
    if frozen.exists():
        try:
            got = json.loads(frozen.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            got = None
            print(f"[WARN] {frozen} 不是合法 JSON，已忽略")
        if isinstance(got, dict) and got != out and str(frozen) not in _WARNED:
            _WARNED.add(str(frozen))
            keys = list(got)[:2]
            print(f"[WARN] {frozen} 的键与 torchvision 显示名不一致（文件前两个键: {keys}，"
                  f"本模块: {list(out)[:2]}）；按名查表的下游代码会 KeyError，"
                  f"跑 datasets/build_pet_class_map.py 重新生成")
    return out


# ---------------------------------------------------------------- 冻结划分的读写
def read_pet_list(path: str | Path) -> list[tuple[str, int]]:
    """读 ``datasets/lists/pet_*.txt``，返回 ``[(image_id, class_idx0)]``。

    行格式固定 ``<image_id>\\t<class_idx_0based>``，image_id **不含** ``.jpg``。
    解析统一 ``rsplit(None, 1)``：Tab / 空格通吃，且含空格的路径不会被切错。
    """
    path = _abs_path(path)
    if not path.exists():
        raise SystemExit(f"[FATAL] 找不到划分文件: {path}\n"
                         f"        先生成冻结划分: python datasets/make_pet_split.py")
    rows: list[tuple[str, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.rsplit(None, 1)
            if len(parts) != 2:
                raise SystemExit(f"[FATAL] {path}:{lineno} 需要两列 '<image_id> <label>'，实收 {s!r}")
            image_id, lab = parts
            image_id = image_id.strip().strip('"')
            if image_id.lower().endswith(".jpg"):
                image_id = image_id[:-4]            # 冻结格式不含扩展名，顺手容忍多写
            lab = int(lab)
            if not 0 <= lab < PET_NUM_CLASSES:
                raise SystemExit(f"[FATAL] {path}:{lineno} 标签 {lab} 越界（合法 0..36）；"
                                 f"常见原因是把 1-based 的官方 CLASS-ID 直接当标签用了")
            rows.append((image_id, lab))
    if not rows:
        raise SystemExit(f"[FATAL] {path} 为空")
    return rows


def load_pet_split(lists_dir: str | Path, split: str) -> list[str]:
    """``split ∈ {'train','val','test'}`` -> image_id 列表（读 ``pet_{split}.txt``）。"""
    if split not in SPLITS:
        raise SystemExit(f"[FATAL] split={split!r} 非法，只支持 {SPLITS}")
    return [i for i, _ in read_pet_list(_abs_path(lists_dir) / f"pet_{split}.txt")]


def _default_ids(root: str | Path, split: str) -> list[str]:
    """``ids=None`` 时按 split 自带划分取 id。

    * ``trainval`` / ``test``：直接读官方 ``annotations/{split}.txt``（文件系统顺序之外再排序，
      保证两次运行取到同一顺序）；
    * ``train`` / ``val``：读冻结产物 ``datasets/lists/pet_{split}.txt``（官方没有这两个文件，
      必须由 make_pet_split 切出来）；
    * ``all`` / ``list``：读官方 ``annotations/list.txt`` 全量索引。
    """
    root = _abs_path(root)
    if split in ("train", "val"):
        return load_pet_split(_abs_path(LISTS_DIR), split)
    if split in ("trainval", "test"):
        return [i for i, _c, _s, _b in parse_pet_annotation(root / ANN_SUBDIR / f"{split}.txt")]
    if split in ("all", "list"):
        return [i for i, _c, _s, _b in parse_pet_annotation(root / ANN_SUBDIR / "list.txt")]
    raise SystemExit(f"[FATAL] split={split!r} 非法，只支持 trainval/test/train/val/all")


# ---------------------------------------------------------------- Dataset
class PetDataset(Dataset):
    """从 ``root/{images,annotations}`` 读 Pet 37 类，返回 ``(img_tensor, label[, image_id])``。

    参数
    ----
    root
        形如 ``data/oxford-iiit-pet``，其下必须有 ``images/`` 与 ``annotations/``。
        相对路径按仓库根解析。
    ids
        image_id 列表（不含 ``.jpg``）；``None`` 表示用 split 自带划分（见 :func:`_default_ids`）。
        也可以直接传 ``datasets/lists/pet_*.txt`` 的路径（自动按行解析，并校验标签与官方一致）。
    split
        ``trainval`` / ``test`` / ``train`` / ``val`` / ``all``；仅在 ``ids is None`` 时生效。
    transform
        PIL Image -> Tensor 的变换；``None`` 时返回原始 PIL 图（便于调试）。
    return_id
        ``True`` 时 ``__getitem__`` 返回 ``(img, label, image_id)``——失败案例分析、Grad-CAM
        溯源、预测 CSV 回填都依赖它，缺了就无法把一张错图定位回具体 jpg。

    属性
    ----
    ids / labels / species
        与 ``__getitem__`` 的下标一一对应；``labels`` 仍是列表（property 返回副本）。
    """

    def __init__(self, root: str | Path, ids: Sequence[str] | str | Path | None = None,
                 split: str = "trainval", transform: Callable | None = None,
                 return_id: bool = False) -> None:
        self.root = _abs_path(root)
        self.images_dir = self.root / IMG_SUBDIR
        self.split = split
        self.transform = transform
        self.return_id = bool(return_id)
        if not self.images_dir.is_dir():
            raise SystemExit(f"[FATAL] 找不到图片目录 {self.images_dir}（root 应为含 images/ 与 "
                             f"annotations/ 的目录，例: data/oxford-iiit-pet）")

        index = _annotation_index(self.root)        # image_id -> (class_idx0, species0, breed0)

        parsed_labels: dict[str, int] | None = None
        if ids is None:
            candidates = _default_ids(self.root, split)
        elif isinstance(ids, (str, Path)):
            pairs = read_pet_list(ids)              # 传了 list 文件路径：顺带拿到标签做一致性校验
            candidates = [i for i, _ in pairs]
            parsed_labels = dict(pairs)
        else:
            candidates = [str(i) for i in ids]

        self.ids: list[str] = []
        self._labels: list[int] = []
        self._species: list[int] = []
        self.missing: list[str] = []                # 标注里有、磁盘上没有的 id（只提示，不静默吞掉）
        for image_id in candidates:
            info = index.get(image_id)
            if info is None:
                self.missing.append(f"{image_id}(无标注)")
                continue
            if parsed_labels is not None and parsed_labels.get(image_id) != info[0]:
                raise SystemExit(
                    f"[FATAL] 划分文件与官方 CLASS-ID-1 不一致: {image_id} "
                    f"文件={parsed_labels.get(image_id)} 官方={info[0]}"
                    f"（差 1 通常是漏了 CLASS-ID - 1）")
            if not (self.images_dir / f"{image_id}.jpg").exists():
                self.missing.append(f"{image_id}.jpg(缺图)")
                continue
            self.ids.append(image_id)
            self._labels.append(info[0])
            self._species.append(info[1])
        if not self.ids:
            raise SystemExit(f"[FATAL] {self.root} 在 split={split!r} 下解析出 0 条样本")
        if self.missing:
            print(f"[WARN] PetDataset({split}): 跳过 {len(self.missing)} 条，前 3: {self.missing[:3]}")

    # ---- 基本协议 ----
    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        path = self.images_dir / f"{image_id}.jpg"
        # with + load()：Windows 上 Image.open 不关闭句柄，7390 张图很快会耗尽句柄
        with open(path, "rb") as f:
            img = Image.open(f)
            img.load()
            img = ImageOps.exif_transpose(img)      # Pet 无 EXIF 旋转，保险起见保留
            img = img.convert("RGB")                # 丢掉 RGBA/L/P：漏掉这步 ToTensor 会出 1 或 4 通道
        if self.transform is not None:
            img = self.transform(img)
        if self.return_id:
            return img, self._labels[idx], image_id
        return img, self._labels[idx]

    # ---- 溯源与统计 ----
    def image_id(self, idx: int) -> str:
        """反查第 ``idx`` 条样本的 image_id（不含 ``.jpg``）。"""
        return self.ids[idx]

    @property
    def labels(self) -> list[int]:
        """0-based 标签列表，供分层切分 / 类别统计 / 泄漏自检用（返回副本，防止外部改坏）。"""
        return list(self._labels)

    @property
    def species(self) -> list[int]:
        """0=猫 / 1=狗，与 ``labels`` 同下标（报告里的猫狗分组分析用）。"""
        return list(self._species)

    def class_histogram(self) -> dict[int, int]:
        """``{class_idx0: 样本数}``，用于确认 val 每类等量、train/val 覆盖全 37 类。"""
        return dict(sorted(Counter(self._labels).items()))

    def __repr__(self) -> str:                      # pragma: no cover - 仅调试可读性
        return (f"PetDataset(split={self.split!r}, n={len(self.ids)}, "
                f"n_classes={len(set(self._labels))}, return_id={self.return_id})")


# ---------------------------------------------------------------- 划一分（库函数版）
def make_pet_split(root: str | Path, out_dir: str | Path = LISTS_DIR,
                   val_per_class: int = DEFAULT_VAL_PER_CLASS,
                   seed: int = DEFAULT_SEED) -> dict[str, list[str]]:
    """从官方 ``trainval.txt`` 切出 train/val（每类固定 ``val_per_class`` 张），test 用官方 ``test.txt``。

    为什么是"每类固定张数"而不是按百分比：Macro-F1 每类等权，val 每类样本数相等才不会让
    样本多的类主导早停点。类内**先 sorted 再 shuffle**——消除文件系统遍历顺序的不确定性，
    同一 seed 在任何机器上都切出同一份划分。

    产物（与 ``datasets/make_pet_split.py`` 完全一致，全仓唯一一套）：

    * ``<out_dir>/pet_train.txt`` / ``pet_val.txt`` / ``pet_test.txt``：
      行格式固定 ``<image_id>\\t<class_idx_0based>``（image_id 不含 ``.jpg``），UTF-8，LF，按 id 排序；
    * ``<out_dir>/pet_split_meta.json``：键固定
      ``seed, val_per_class, n_train, n_val, n_test, per_class_train, per_class_val,
      source_md5, generated_at``。

    返回 ``{'train': [...], 'val': [...], 'test': [...]}``（每个是 image_id 列表，已排序）。
    默认参数下应为 ``train 2940 / val 740 / test 3669``。
    """
    root = _abs_path(root)
    out = _abs_path(out_dir)
    ann = root / ANN_SUBDIR
    img = root / IMG_SUBDIR
    trainval_txt = ann / "trainval.txt"
    test_txt = ann / "test.txt"

    index = _annotation_index(root)                 # 白名单：list.txt 覆盖的 7349 个 id
    rows = parse_pet_annotation(trainval_txt)       # [(image_id, class_idx0, species0, breed0)]

    # 1) 类内先 sorted 再 shuffle，每类固定取 val_per_class 张进 val
    by_cls: dict[int, list[str]] = {}
    for image_id, cid0, _sp, _br in rows:
        if image_id in index:                       # 过滤掉不在官方索引里的 id
            by_cls.setdefault(cid0, []).append(image_id)
    rng = random.Random(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []
    for c in sorted(by_cls):
        ids_c = sorted(by_cls[c])
        rng.shuffle(ids_c)
        # 防御：某类样本极少时不至于把 train 掏空（Pet 每类 93~100 张，正常恒等于 val_per_class）
        k = min(int(val_per_class), max(1, len(ids_c) // 2))
        val_ids += ids_c[:k]
        train_ids += ids_c[k:]
    if set(train_ids) & set(val_ids):
        raise SystemExit("[FATAL] train/val 出现交集，划分逻辑有 bug")

    # 2) test 用官方 test.txt，同样走白名单（test 全程不参与任何选择）
    test_ids = [i for i, _c, _s, _b in parse_pet_annotation(test_txt) if i in index]

    splits = {"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)}

    # 3) 落盘：与 make_pet_split.py 同格式（LF、按 id 排序、id 不含 .jpg）
    out.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        ids = splits[name]
        lines = []
        for image_id in ids:
            if not (img / f"{image_id}.jpg").exists():
                raise SystemExit(f"[FATAL] {name} 里的 {image_id}.jpg 不存在于 {img}")
            lines.append(f"{image_id}\t{index[image_id][0]}")
        p = out / f"pet_{name}.txt"
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")

    cnt_train = Counter(index[i][0] for i in splits["train"])
    cnt_val = Counter(index[i][0] for i in splits["val"])
    meta = {
        "seed": int(seed),
        "val_per_class": int(val_per_class),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        # 键是 str(class_idx0)，值与 labels/pet_classes.txt 的行号一一对应
        "per_class_train": {str(k): v for k, v in sorted(cnt_train.items())},
        "per_class_val": {str(k): v for k, v in sorted(cnt_val.items())},
        # 划分的来源文件 md5（trainval.txt 被切分；test 用官方 test.txt，未经切分）
        "source_md5": hashlib.md5(trainval_txt.read_bytes()).hexdigest(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "pet_split_meta.json").write_text(
        json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"[split] seed={seed} val_per_class={val_per_class} "
          f"train={meta['n_train']} val={meta['n_val']} test={meta['n_test']} -> {out}")
    return splits


# ---------------------------------------------------------------- 自检
def _selfcheck() -> int:
    """端到端自检：划分 -> Dataset（含 return_id）-> 一次真实解码 -> 与 torchvision 交叉校验。"""
    root = PROJ / "data" / "oxford-iiit-pet"
    if not (root / ANN_SUBDIR / "trainval.txt").exists():
        print(f"[SKIP] 找不到 {root}，跳过 Pet 自检")
        return 0

    names = pet_class_names(root)
    name2idx = pet_class_to_idx(root)
    print(f"[class] n={len(names)} 0={names[0]!r} 2={names[2]!r} 36={names[36]!r} "
          f"idx[American Pit Bull Terrier]={name2idx['American Pit Bull Terrier']}")
    assert len(names) == PET_NUM_CLASSES
    assert names[0] == "Abyssinian" and names[36] == "Yorkshire Terrier", (names[0], names[36])
    assert name2idx["American Pit Bull Terrier"] == 2, "rsplit('_',1) 没生效？"

    splits = make_pet_split(root)
    assert {k: len(v) for k, v in splits.items()} == {"train": 2940, "val": 740, "test": 3669}, \
        {k: len(v) for k, v in splits.items()}

    # 与 torchvision 的 classes 交叉校验（不是 37 类就是前缀切分写错了）
    try:
        from torchvision.datasets import OxfordIIITPet
        tv = OxfordIIITPet(root=str(PROJ / "data"), split="trainval", download=False).classes
        assert list(tv) == names, [(i, a, b) for i, (a, b) in enumerate(zip(tv, names)) if a != b][:5]
        print("[OK] 与 torchvision.OxfordIIITPet.classes 逐项一致")
    except ModuleNotFoundError:
        print("[SKIP] 未安装 torchvision，跳过交叉校验")

    # 用一条真实样本验证 Dataset 的三条出口路径
    from torchvision import transforms as T
    tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    for split in SPLITS:
        ids = load_pet_split(PROJ / LISTS_DIR, split)
        ds = PetDataset(root, ids=ids, split=split, transform=tf, return_id=True)
        x, y, image_id = ds[0]
        assert x.shape == (3, 224, 224) and x.dtype.is_floating_point, (x.shape, x.dtype)
        assert ds.image_id(0) == image_id and len(ds.labels) == len(ds) == len(ids)
        hist = ds.class_histogram()
        print(f"[{split:5s}] n={len(ds):4d} 覆盖类别={len(hist)}/37 每类 min/max="
              f"{min(hist.values())}/{max(hist.values())} x={tuple(x.shape)} y={y} id={image_id}")
    print("[OK] datasets/pet_dataset.py 自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
