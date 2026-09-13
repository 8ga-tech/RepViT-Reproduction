# datasets/build.py
# -*- coding: utf-8 -*-
"""把 YAML 的 ``data`` / ``aug`` 段变成实际的 Dataset / 变换 / DataLoader。

这是**训练侧唯一的装配点**：任何脚本都不得自己再写一遍 transform 或 DataLoader 参数。

用法::

    from utils.config import build_config
    from datasets.build import build_dataloaders, build_eval_loader, num_classes_of
    cfg = build_config(["--cfg", "configs/baseline.yaml"])
    loaders = build_dataloaders(cfg)            # {'train':.., 'val':.., 'test':..}
    K = num_classes_of(cfg)

三个容易出错的数值/工程口径，全部按官方与规格书 3.7 冻结：

1) **评估变换**：``Resize(int(round(input_size / crop_pct))) + CenterCrop(input_size)``，
   插值 bicubic，ImageNet ``mean/std``。``crop_pct`` 取 ``cfg['data']['crop_pct']``，
   缺省 ``0.875`` -> ``round(224/0.875)=256``，即官方 THU-MIG ``data/datasets.py`` 的
   ``Resize(256, BICUBIC)+CenterCrop(224)`` 口径；``crop_pct=0.95`` -> ``236``
   （timm ``pretrained_cfg`` 口径，规格书 2.5.3 明写 236）。**绝不能用 aug.val.resize_size
   覆盖 crop_pct**：两套口径会差零点几个百分点，只报一套会被追问。
2) **归一化顺序**：``ToTensor -> Normalize -> (RandomErasing)``。RandomErasing 必须在
   Normalize **之后**——放前面用 0 填充，0 会被标准化成 ``(0-mean)/std≈-2.1`` 的深色块，
   语义完全不符。
3) **可复现性**：``build_loader`` 必须同时传 ``worker_init_fn=utils.seed.seed_worker``
   与显式 ``generator``。缺 worker_init_fn，``num_workers>0`` 时每个 worker 的增强随机流
   不由 ``set_seed`` 决定（同一 seed 复现不出同一条样本）；缺 generator，主进程的
   ``RandomSampler`` 顺序不可复现。两者缺一，多随机种子实验的方差都会被噪声污染。

与 ``tools/build_dataloader.py``（规格书 3.7.3 的权威参考）数值对齐：
mean/std、插值、resize 公式、归一化顺序、``val/test 绝不 drop_last`` 完全一致。
差异只有两处，且都是刻意的：本模块支持 cfg 驱动的增强开关（RandAugment 用 timm 的
``rand_augment_transform``，与 ``tools/train.py`` 里 torchvision ``T.RandAugment`` 同族），
并把 ``pin_memory`` 收敛为「CUDA 时才开」（CPU-only 上开它只是白付锁页开销）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

# ---------------------------------------------------------------- 路径引导
PROJ = Path(__file__).resolve().parents[1]              # 仓库根，禁止写死绝对路径
if str(PROJ) not in sys.path:
    # 以 `python datasets/build.py` 直接运行时 sys.path[0] 是 datasets/，仓库根不在其中，
    # 于是 `import utils.seed` / `import datasets.build` 都会失败。补一次即可（幂等）。
    sys.path.insert(0, str(PROJ))

from utils.seed import make_generator, seed_worker                      # noqa: E402
from datasets.pet_dataset import PetDataset, read_pet_list              # noqa: E402
from datasets.imagenet_subset import ImageNetSubset                     # noqa: E402

# ---------------------------------------------------------------- 常量（唯一真源）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_CROP_PCT = 0.875        # 官方 data/datasets.py 口径：Resize(256)+CenterCrop(224)
DEFAULT_INPUT_SIZE = 224
DEFAULT_EVAL_BATCH = 128
PET_LISTS_DIR = "datasets/lists"

_INTERP = {
    "nearest": InterpolationMode.NEAREST,
    "bilinear": InterpolationMode.BILINEAR,
    "linear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "cubic": InterpolationMode.BICUBIC,
    # torchvision 的枚举值：0=NEAREST 1=LANCZOS 2=BILINEAR 3=BICUBIC 4=BOX 5=HAMMING
    0: InterpolationMode.NEAREST,
    1: InterpolationMode.LANCZOS,
    2: InterpolationMode.BILINEAR,
    3: InterpolationMode.BICUBIC,
}
_INTERP_NAME = {
    InterpolationMode.NEAREST: "nearest",
    InterpolationMode.BILINEAR: "bilinear",
    InterpolationMode.BICUBIC: "bicubic",
    InterpolationMode.LANCZOS: "lanczos",
    InterpolationMode.BOX: "box",
    InterpolationMode.HAMMING: "hamming",
}


def _interp(value: Any) -> InterpolationMode:
    """'bicubic' / 3 / 已是枚举 -> ``InterpolationMode.BICUBIC``。

    官方 ``main.py`` 写的是 ``interpolation=3``（正整数），YAML 里为了可读性写字符串，
    两种都要吃下；``_interp_name`` 会把枚举再喂回来，所以枚举也必须幂等通过。
    """
    if value is None:
        return InterpolationMode.BICUBIC
    if isinstance(value, InterpolationMode):
        return value
    if value in _INTERP:
        return _INTERP[value]
    key = str(value).strip().lower()
    if key in _INTERP:
        return _INTERP[key]
    raise SystemExit(f"[FATAL] 不支持的插值: {value!r}，可选 bicubic / bilinear / nearest")


def _interp_name(value: Any) -> str:
    return _INTERP_NAME.get(_interp(value), "bicubic")


def _mean_std(cfg: dict) -> tuple[tuple, tuple]:
    """归一化常量：读 ``aug.norm``，缺省用 ImageNet 常量（禁止用 Pet 自算 mean/std）。"""
    norm = (cfg.get("aug") or {}).get("norm") or {}
    mean = tuple(float(x) for x in norm.get("mean", IMAGENET_MEAN))
    std = tuple(float(x) for x in norm.get("std", IMAGENET_STD))
    if len(mean) != 3 or len(std) != 3:
        raise SystemExit(f"[FATAL] aug.norm 必须是 3 通道，收到 mean={mean} std={std}")
    return mean, std


def _input_size(cfg: dict) -> int:
    return int((cfg.get("data") or {}).get("input_size", DEFAULT_INPUT_SIZE))


def resolve_eval_resize(cfg: dict, crop_pct: float | None = None,
                        resize_size: int | None = None) -> int:
    """评估变换的 Resize 短边长度。

    优先级：显式 ``resize_size`` > 显式 ``crop_pct`` > ``cfg['data']['crop_pct']``
    > ``cfg['aug']['val']['resize_size']`` > ``round(input_size / 0.875)``。

    ``crop_pct`` 的物理含义：中心裁剪区域占缩放后图像短边的比例，等价于
    「先把短边缩放到 ``input_size / crop_pct``，再裁 ``input_size``」。所以
    ``crop_pct`` 越小，缩放后的图越大、裁掉的越多（更强的 zoom-in 测试时增强）。
    """
    size = _input_size(cfg)
    if resize_size is not None:
        return int(resize_size)
    data = cfg.get("data") or {}
    pct = crop_pct if crop_pct is not None else data.get("crop_pct")
    if pct is not None:
        pct = float(pct)
        if not 0.0 < pct <= 1.0:
            raise SystemExit(f"[FATAL] crop_pct={pct} 非法，应在 (0, 1] 内")
        return int(round(size / pct))
    val_cfg = (cfg.get("aug") or {}).get("val") or {}
    if val_cfg.get("resize_size") is not None:
        return int(val_cfg["resize_size"])
    return int(round(size / DEFAULT_CROP_PCT))


# ---------------------------------------------------------------- 变换
def build_train_transform(cfg: dict) -> Callable:
    """训练变换，读 ``cfg['aug']['train']``；值为 ``null`` 的那一段整段不加入。

    装配顺序（与官方一致，乱序会静默掉点）::

        [RandomResizedCrop | Resize+RandomCrop] -> HFlip -> ColorJitter -> RandAugment(PIL)
        -> ToTensor -> Normalize -> RandomErasing(tensor)

    * ``random_resized_crop`` 为 ``null`` 时退化为 ``Resize(resize_size)+RandomCrop``（消融路径）；
    * ``randaugment`` 用 timm 的 ``rand_augment_transform``；``n``/``m``/``inc1`` 分别映射到
      ``-n{x}``/``-m{x}``/``-inc1``（``inc1=True`` 使用随 magnitude 单调加剧的算子集）；
    * ``random_erasing`` 用 timm 的 ``RandomErasing``（有 ``mode``/``count``，torchvision 的
      同名类没有这两个参数），``device`` 必须显式给 ``'cpu'``：它的默认值是 ``'cuda'``，
      在 CPU-only 机器上会直接抛错，而且 CUDA RNG 不受 worker 种子控制，复现不出来。
    """
    train_cfg = (cfg.get("aug") or {}).get("train")
    if not train_cfg:
        raise SystemExit("[FATAL] 配置缺少 aug.train 段")
    size = _input_size(cfg)
    interp = _interp(train_cfg.get("interpolation", "bicubic"))
    mean, std = _mean_std(cfg)

    ops: list[Callable] = []
    rrc = train_cfg.get("random_resized_crop")
    if rrc:
        ops.append(T.RandomResizedCrop(
            size, scale=tuple(float(x) for x in rrc["scale"]),
            ratio=tuple(float(x) for x in rrc["ratio"]), interpolation=interp))
    else:
        # 消融路径：先按短边 Resize 再随机裁（比 RandomResizedCrop 温和）
        ops.append(T.Resize(int(train_cfg.get("resize_size", size)), interpolation=interp))
        ops.append(T.RandomCrop(size))

    hflip = float(train_cfg.get("hflip") or 0.0)
    if hflip > 0:
        ops.append(T.RandomHorizontalFlip(p=hflip))

    cj = train_cfg.get("color_jitter")
    if cj and max(float(cj.get(k) or 0.0) for k in ("brightness", "contrast", "saturation", "hue")) > 0:
        ops.append(T.ColorJitter(brightness=float(cj.get("brightness") or 0.0),
                                 contrast=float(cj.get("contrast") or 0.0),
                                 saturation=float(cj.get("saturation") or 0.0),
                                 hue=float(cj.get("hue") or 0.0)))

    ra = train_cfg.get("randaugment")
    if ra:
        ops.append(_build_randaugment(ra, interp))

    ops += [T.ToTensor(), T.Normalize(mean=list(mean), std=list(std))]

    re_cfg = train_cfg.get("random_erasing")
    if re_cfg and float(re_cfg.get("prob") or 0.0) > 0:
        from timm.data.random_erasing import RandomErasing   # 延迟导入：只在真正用时付代价
        count = int(re_cfg.get("count", 1) or 1)
        ops.append(RandomErasing(probability=float(re_cfg["prob"]),
                                 mode=str(re_cfg.get("mode", "pixel")),
                                 min_count=count, max_count=count,
                                 device="cpu"))
    return T.Compose(ops)


def _build_randaugment(ra: dict, interp: InterpolationMode) -> Callable:
    """``{n, m, inc1[, mstd]}`` -> timm 的 RandAugment。

    timm 的接口是"配置串 + hparams"，不是一个参数一个参数地传，所以这里把 YAML 的
    结构化字段翻译成 ``rand-m{m}-n{n}[-inc1][-mstd..]``。翻译失败（timm 版本不认识某个
    section）会直接 assert 崩，不会静默退化成"没有 RandAugment"——后者会让 O1 实验组
    与 baseline 的差异凭空消失，是最难查的一类事故。
    """
    from timm.data.auto_augment import rand_augment_transform

    n = int(ra.get("n", 2))
    m = int(ra.get("m", 9))
    parts = ["rand", f"m{m}", f"n{n}"]
    if bool(ra.get("inc1", ra.get("inc", False))):
        parts.append("inc1")
    mstd = ra.get("mstd")
    hparams: dict[str, Any] = {
        # 不做 magnitude 抖动：与 torchvision 的 RandAugment 口径一致，也让 O1 可复现
        "magnitude_std": float(mstd) if mstd is not None else 0.0,
        "translate_const": 250,
        "img_mean": (128, 128, 128),
        "fillcolor": (128, 128, 128),
        "interpolation": _interp_name(interp),
    }
    if mstd is not None:
        parts.append(f"mstd{float(mstd)}")
    cfg_str = "-".join(parts)
    tf = rand_augment_transform(cfg_str, hparams)
    print(f"[aug] RandAugment(timm) config={cfg_str!r} interpolation={hparams['interpolation']}")
    return tf


def build_eval_transform(cfg: dict, crop_pct: float | None = None,
                         resize_size: int | None = None) -> Callable:
    """验证/测试变换：``Resize(short) + CenterCrop(input_size) + ToTensor + Normalize``。

    ``short = int(round(input_size / crop_pct))``，``crop_pct`` 默认取
    ``cfg['data']['crop_pct']``（缺省 0.875 -> 256）。``aug.val.center_crop=false`` 时
    只 Resize 不裁剪（消融路径）。**val 与 test 共用同一套，且绝不复用训练变换**：
    给 val 加随机增强会让不同 epoch 的指标不可比，早停点直接失去意义。
    """
    val_cfg = (cfg.get("aug") or {}).get("val") or {}
    size = _input_size(cfg)
    short = resolve_eval_resize(cfg, crop_pct=crop_pct, resize_size=resize_size)
    interp = _interp(val_cfg.get("interpolation") or (cfg.get("data") or {}).get("interpolation", "bicubic"))
    mean, std = _mean_std(cfg)

    ops: list[Callable] = [T.Resize(short, interpolation=interp)]
    if bool(val_cfg.get("center_crop", True)):
        ops.append(T.CenterCrop(int(val_cfg.get("crop_size", size))))
    ops += [T.ToTensor(), T.Normalize(mean=list(mean), std=list(std))]
    return T.Compose(ops)


# ---------------------------------------------------------------- 类别数
_WARNED: set[str] = set()       # 同一路径只警告一次，避免建三个 Dataset 刷屏


def num_classes_of(cfg: dict) -> int:
    """从类别文件行数反查类别数，与 ``model.num_classes`` 不一致时报错。

    优先 ``data.labels_file``（ImageNet 的 1000 行类名表），其次 ``data.class_names``
    （Pet 的 37 行）。行数才是真源：模型建头用的是 ``model.num_classes``，两者不一致
    会导致「标签 36 >= 输出维度 5」这类运行到一半才炸的事故，必须在这里就拦住。
    """
    data = cfg.get("data") or {}
    declared = (cfg.get("model") or {}).get("num_classes", data.get("num_classes"))
    src = data.get("labels_file") or data.get("class_names")
    n: int | None = None
    if src:
        path = Path(src)
        path = path if path.is_absolute() else PROJ / path
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
        elif str(path) not in _WARNED:
            _WARNED.add(str(path))
            print(f"[WARN] 类别文件不存在: {path}，退回配置里的 num_classes")
    if n is not None and declared is not None and int(declared) != n:
        raise SystemExit(f"[FATAL] num_classes 不一致: {src} 有 {n} 行，"
                         f"配置声明 {int(declared)}；两者必须相等（行号即类别下标）")
    if n is None:
        if declared is None:
            raise SystemExit("[FATAL] 无法确定类别数：data.labels_file / data.class_names "
                             "与 model.num_classes 都缺失")
        n = int(declared)
    return int(n)


# ---------------------------------------------------------------- Dataset
def _is_imagenet(cfg: dict) -> bool:
    name = str((cfg.get("data") or {}).get("name", "pet")).strip().lower()
    return name in ("imagenet_subset", "imagenet", "imagenet_val", "imagenet-val-subset")


def _pet_list_path(cfg: dict, split: str) -> Path:
    data = cfg.get("data") or {}
    p = data.get(f"{split}_list")
    if not p:
        p = f"{PET_LISTS_DIR}/pet_{split}.txt"
    p = Path(p)
    return p if p.is_absolute() else PROJ / p


def build_dataset(cfg: dict, split: str, train: bool) -> Dataset:
    """按 ``cfg['data']`` 建 Dataset；``split ∈ {'train','val','test'}``。

    * ``data.name`` 缺省 / ``pet`` -> :class:`PetDataset`，id 来自 ``data.<split>_list``，
      标签同时与官方 ``CLASS-ID-1`` 交叉校验（差 1 会立刻报错，不会等到 loss 里崩）；
    * ``data.name: imagenet_subset`` -> :class:`ImageNetSubset`，**不做任何划分或采样**；
    * ``return_id`` / ``return_path`` 只在 ``train=False`` 时打开：val/test 的
      ``(img, label, id)`` 三元组是预测 CSV 回填、混淆矩阵错例溯源、Grad-CAM 定位 jpg 的前提；
      训练集不需要，省一次字符串装箱。
    """
    if split not in ("train", "val", "test"):
        raise SystemExit(f"[FATAL] split={split!r} 非法，只支持 train/val/test")
    data = cfg.get("data") or {}
    tf = build_train_transform(cfg) if train else build_eval_transform(cfg)

    if _is_imagenet(cfg):
        list_file = data.get("list_file")
        if not list_file:
            raise SystemExit("[FATAL] data.name=imagenet_subset 时必须给 data.list_file")
        ds = ImageNetSubset(root=data["root"], list_file=list_file, transform=tf,
                            return_path=not train, num_classes=num_classes_of(cfg))
        print(f"[data] split={split:5s} train={int(train)} n={len(ds)} "
              f"list={Path(list_file).name} 类别数={num_classes_of(cfg)}")
        return ds

    list_path = _pet_list_path(cfg, split)
    if not list_path.exists():
        raise SystemExit(f"[FATAL] 缺少冻结划分 {list_path}；先运行 "
                         f"`python datasets/make_pet_split.py --root data/oxford-iiit-pet`")
    # ids 直接传 list 文件路径：PetDataset 会顺带把文件里的标签与官方 CLASS-ID-1 逐条比对
    ds = PetDataset(root=data["root"], ids=str(list_path), split=split, transform=tf,
                    return_id=not train)
    print(f"[data] split={split:5s} train={int(train)} n={len(ds)} "
          f"list={list_path.name} 覆盖类别={len(ds.class_histogram())}/{num_classes_of(cfg)}")
    return ds


def _split_ids(ds: Dataset) -> set[str] | None:
    """取 Dataset 的样本标识集合，用于 train/val/test 泄漏自检。"""
    if isinstance(ds, PetDataset):
        return set(ds.ids)
    if isinstance(ds, ImageNetSubset):
        return {p for p, _y in ds.items}
    return None


# ---------------------------------------------------------------- DataLoader
def _resolve_generator(seed: int) -> torch.Generator:
    """DataLoader 用的显式 generator（固定主进程 ``RandomSampler`` 的抽样顺序）。

    传 ``None`` 时 DataLoader 会自造一个随机 generator，两次运行的样本顺序就不同，
    多随机种子实验的方差无法归因。统一走 ``utils.seed.make_generator``，全仓只此一处逻辑。
    """
    return make_generator(int(seed))


def _make_loader(cfg: dict, split: str, train: bool, batch_size: int | None = None,
                 dataset: Dataset | None = None) -> DataLoader:
    data = cfg.get("data") or {}
    ds = dataset if dataset is not None else build_dataset(cfg, split, train)
    if batch_size is None:
        # train 取 data.batch_size（64 -> 2940/64 丢尾批 = 45 step/epoch）；
        # val/test 取 data.eval_batch_size（128 -> 不参与 BN/反传，放大只提速不改指标）
        batch_size = int(data["batch_size"] if train else data.get("eval_batch_size", DEFAULT_EVAL_BATCH))
    num_workers = int(data.get("num_workers", 0))
    # 增强的随机性由 worker 内的 random/numpy/torch 三套 RNG 决定；worker_init_fn=seed_worker
    # 用 torch.initial_seed() 推出可复现的 worker 种子，缺它则同一 seed 复现不出同一条样本。
    kwargs: dict[str, Any] = dict(
        batch_size=int(batch_size),
        shuffle=bool(train),                       # 训练必须打乱；val/test 必须关（配合 index 回填）
        drop_last=bool(train) and bool(data.get("drop_last", True)),   # val/test 绝不丢尾批
        num_workers=num_workers,
        # CPU-only 上 pin_memory 只是白付锁页开销；表 3.7.2 的口径是「CUDA 时才开」
        pin_memory=bool(data.get("pin_memory", False)) and torch.cuda.is_available(),
        worker_init_fn=seed_worker,                # 缺它则 num_workers>0 时增强不受 set_seed 控制
        generator=_resolve_generator(int(cfg.get("seed", 0))),
        timeout=60 if num_workers > 0 else 0,      # 死锁时不要无限挂起
        persistent_workers=bool(num_workers > 0 and data.get("persistent_workers", True)),
        prefetch_factor=2 if num_workers > 0 else None,   # num_workers=0 时必须为 None
    )
    return DataLoader(ds, **kwargs)


def build_loader(cfg: dict, split: str, train: bool) -> DataLoader:
    """建单个 DataLoader（transforms/采样/drop_last/worker 种子全部按 3.7 表冻结）。

    ``shuffle`` 与 ``sampler`` 互斥：本函数只用标准 ``RandomSampler``/``SequentialSampler``，
    若要上 ``WeightedRandomSampler`` 请自行构造 DataLoader 并把 ``shuffle`` 传 False
    （Pet 类间差异只有 184:200，Baseline 不需要采样器，分层划分才是必须的）。
    """
    return _make_loader(cfg, split, train)


def build_eval_loader(cfg: dict, split: str, batch_size: int | None = None) -> DataLoader:
    """评测专用：``train=False``（无增强、返回 id/path）、``shuffle=False``、``drop_last=False``。

    ``batch_size`` 可覆盖，用于显存不够时降批，或与部署侧 batch 对齐做一致性对比。
    """
    return _make_loader(cfg, split, train=False, batch_size=batch_size)


def build_dataloaders(cfg: dict) -> dict[str, DataLoader]:
    """``{'train':.., 'val':.., 'test':..}``，并做一次 train/val/test 泄漏自检。

    泄漏是"不通过条款"：三个集合只要有一条样本重叠，val 上的 early-stop 就已经看过
    test 的图，报告里所有泛化结论作废。所以这里直接 ``SystemExit``，不做容错继续训练。
    """
    loaders = {split: _make_loader(cfg, split, train=(split == "train"))
               for split in ("train", "val", "test")}
    if _is_imagenet(cfg):
        # 固定子集三个 split 读的是同一份列表，"泄漏"是构造使然，做自检只会误报。
        # 它本来也只用于官方权重评价（tools/eval_pretrained.py 走 build_eval_loader）。
        print("[data] data.name=imagenet_subset：不做划分也不做泄漏自检（考核方固定列表，只读）")
    else:
        ids = {s: _split_ids(dl.dataset) for s, dl in loaders.items()}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if ids[a] is None or ids[b] is None:
                continue
            inter = ids[a] & ids[b]
            if inter:
                raise SystemExit(f"[FATAL] 数据泄漏：{a} 与 {b} 有 {len(inter)} 条重复，"
                                 f"例如 {sorted(inter)[:5]}；重新划分别无他法")
    print("[data] " + " | ".join(f"{s}={len(dl.dataset)}" for s, dl in loaders.items())
          + f" | step/epoch={len(loaders['train'])}（drop_last="
          + f"{loaders['train'].drop_last}）")
    return loaders


# ---------------------------------------------------------------- 自检
def _selfcheck() -> int:
    import argparse

    from utils.config import apply_overrides, load_yaml

    ap = argparse.ArgumentParser(description="datasets/build.py 自检：变换口径 + 三个 DataLoader")
    ap.add_argument("--cfg", default="configs/baseline.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="set", help="点号路径覆盖，如 data.num_workers=0")
    a = ap.parse_args()

    cfg_path = Path(a.cfg)
    cfg = load_yaml(cfg_path if cfg_path.is_absolute() else PROJ / cfg_path)
    apply_overrides(cfg, a.set)
    print(f"[cfg] {cfg_path} seed={cfg.get('seed')}")

    tr, va = build_train_transform(cfg), build_eval_transform(cfg)
    print(f"[tf] train size={len(tr.transforms)}: "
          f"{[type(o).__name__ for o in tr.transforms]}")
    print(f"[tf] eval  size={len(va.transforms)}: "
          f"{[type(o).__name__ for o in va.transforms]}")
    size = _input_size(cfg)
    for pct in (None, DEFAULT_CROP_PCT, 0.95):
        short = resolve_eval_resize(cfg, crop_pct=pct)
        print(f"[tf] crop_pct={pct if pct is not None else 'cfg/默认'} -> Resize({short}) "
              f"+ CenterCrop({size})")

    # 三个 list 齐备才真正读数据；缺任何一个就只做静态自检（不阻断 CI/冒烟）
    lists = {s: _pet_list_path(cfg, s) for s in ("train", "val", "test")}
    if not all(p.exists() for p in lists.values()):
        miss = [str(p.relative_to(PROJ)) for p in lists.values() if not p.exists()]
        print(f"[SKIP] 缺少 {miss}，跳过 DataLoader 实建（先跑 python datasets/make_pet_split.py）")
        print("[OK] datasets/build.py 静态自检通过")
        return 0

    print(f"[K] num_classes_of={num_classes_of(cfg)}")
    data_cfg = cfg.setdefault("data", {})
    saved_workers = data_cfg.get("num_workers", 0)
    data_cfg["num_workers"] = 0                     # 自检强制单进程：Windows + 4 worker 会让冒烟变慢
    try:
        loaders = build_dataloaders(cfg)
        for split in ("train", "val", "test"):
            dl = loaders[split]
            batch = next(iter(dl))
            x, y = batch[0], batch[1]
            # train 返回 (x, y)，val/test 返回 (x, y, id)：id 是预测 CSV 回填与错例溯源的锚点
            tag = f"id={batch[2][0]!r}" if len(batch) == 3 else "(no id)"
            print(f"{split:5s} len(ds)={len(dl.dataset):5d} batches={len(dl):4d} "
                  f"x={tuple(x.shape)} {x.dtype} y={tuple(y.shape)} {y.dtype} "
                  f"range=[{x.min():.3f},{x.max():.3f}] {tag}")
        # 期望：train 45 批（floor(2940/64)，drop_last=True 丢掉尾批 60 张）；
        #        val 6 批（ceil(740/128)）、test 29 批（ceil(3669/128)），val/test 不丢尾批。
        # 注意 val/test 的 range 恰为 ImageNet 归一化区间 [-2.118, 2.640]（Resize+CenterCrop+
        # Normalize 的确定口径）；train 若开了 RandomErasing(mode=pixel) 会宽于该区间——
        # 它用标准正态填充，属预期，不是预处理跑偏。
        got = {s: len(dl) for s, dl in loaders.items()}
        assert got == {"train": 45, "val": 6, "test": 29}, got
    finally:
        data_cfg["num_workers"] = saved_workers
    print("[OK] datasets/build.py 自检通过（45 / 6 / 29 批）")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
