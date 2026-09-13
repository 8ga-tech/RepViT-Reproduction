# utils/seed.py
"""随机种子与可复现性工具（Source: Self-written，对照 THU-MIG/RepViT 官方 main.py 的 seeding 段）。

用法（训练脚本里唯一入口）::

    from utils.seed import set_seed, seed_worker, make_generator
    set_seed(cfg['train']['seed'], strict=cfg['train'].get('deterministic', False))
    g = make_generator(cfg['train']['seed'])
    loader = DataLoader(ds, shuffle=True, worker_init_fn=seed_worker, generator=g, ...)

为什么 DataLoader 要同时给 worker_init_fn 和 generator：
    worker_init_fn 只保证各 worker 的数据增强/采样不同且可复现；generator 保证主进程
    的 sampler（RandomSampler）在两次运行中抽到相同的样本顺序。缺任何一个，多随机种子
    实验的方差都会被采样噪声污染。
"""
from __future__ import annotations

import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.normcase(_os.path.abspath(_sys.path[0] or ".")) == _os.path.normcase(_HERE):
    # `python utils/seed.py` 会把 utils/ 顶成 sys.path[0]，于是任何 `import logging`
    # 都会命中本包的 utils/logging.py 而不是标准库，torch 在 import 期就 AttributeError。
    # 摘掉这一条即可复原；用 `python -m utils.seed` 或正常 import 时这里是 no-op。
    _sys.path.pop(0)

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 0, strict: bool = False) -> None:
    """固定 random / numpy / torch / torch.cuda / PYTHONHASHSEED。

    strict=False（默认）：cudnn.benchmark=True，与官方 main.py 一致，卷积算法自动
        调优，速度快，但同一 seed 两次结果**可能不同**。日常单次训练用它。
    strict=True：cudnn.benchmark=False、cudnn.deterministic=True，并设
        CUBLAS_WORKSPACE_CONFIG=:4096:8（CUDA >= 10.2 下 cuBLAS 确定性 GEMM 的必要条件），
        同时尝试 torch.use_deterministic_algorithms(True)。做多随机种子对比实验时必须
        开它，否则「提升是否超过随机波动」这个结论无从谈起。
    """
    # PYTHONHASHSEED 必须在解释器启动前生效，这里设置只对后续 spawn 的子进程有意义；
    # 但仍然要设——否则 torch DataLoader 的 worker（spawn 模式）哈希种子各不相同。
    os.environ["PYTHONHASHSEED"] = str(int(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)      # 当前 device
        torch.cuda.manual_seed_all(seed)  # 全部 device，多卡时不漏

    if strict:
        # CUBLAS_WORKSPACE_CONFIG 必须在第一次 cuBLAS 调用**之前**设置，否则无效。
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        # 与官方 main.py 一致：benchmark=True 让 cudnn 自动选最快卷积算法
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # 尽力而为：部分算子（如双线性插值的 backward）在确定性模式下没有实现，
    # 某些 torch 版本会直接抛错。这里只把它当增强手段，失败不阻断训练。
    try:
        torch.use_deterministic_algorithms(bool(strict))
    except Exception as exc:  # pragma: no cover - 取决于 torch/算子组合
        print(f"[seed] 警告: use_deterministic_algorithms({strict}) 不可用: {exc}")


def seed_worker(worker_id: int) -> None:
    """DataLoader 的 worker_init_fn。

    torch 在每次 fork/spawn 时给 worker 设的 base_seed = 主进程 seed + worker_id，
    `torch.initial_seed()` 取出它；再取模 2**32 因为 numpy/random 只接受 32 位种子
    （torch 的种子可能是 64 位，直接传给 numpy 会抛 OverflowError）。
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """返回一个已 manual_seed 的 torch.Generator，用作 DataLoader(generator=...)。

    只 manual_seed 不返回 None 是刻意的：传 None 时 DataLoader 会自造一个随机
    generator，样本顺序就不可复现了。
    """
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


if __name__ == "__main__":
    # 自检：同一 seed 下 numpy / random / torch 的抽样必须逐位一致
    set_seed(1234, strict=False)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    set_seed(1234, strict=False)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert a == b, f"set_seed 不可复现: {a} != {b}"

    g1, g2 = make_generator(7), make_generator(7)
    assert torch.rand(3, generator=g1).equal(torch.rand(3, generator=g2)), \
        "make_generator 未固定样本顺序"

    set_seed(1234, strict=True)
    print(f"[seed] 自检通过 seed=1234  strict=True  cudnn.deterministic="
          f"{torch.backends.cudnn.deterministic}  benchmark={torch.backends.cudnn.benchmark}  "
          f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
    seed_worker(0)  # 不抛错即可
    print("[seed] OK")
