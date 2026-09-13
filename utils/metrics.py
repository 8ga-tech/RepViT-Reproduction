# utils/metrics.py
"""分类指标工具箱：top-k 精度、macro-F1、每类 F1/召回、混淆矩阵、ECE。

约定：
  - 输入标签一律 **0-based**（Pet 数据集的 1..37 在载入时已减 1），类别空间恒为
    ``range(num_classes)``；
  - top1/top5/macro_f1 对外返回 **0~1 小数**，只有 topk_accuracy() 返回百分数（0~100），
    因为它是训练循环里直接 print 的那一个。

sklearn 调用必须显式传 labels / zero_division / average，原因见下：
  - 不传 ``labels=list(range(num_classes))``：若某类从未被预测到，返回数组维度会缩水，
    37 个类别名与 36 个数字整体错位——这是最隐蔽的可视化 bug；
  - 不传 ``zero_division=0``：sklearn 会打印 UndefinedMetricWarning 并把该位置成 0.0
    或 nan（跨版本行为不同），会让验收断言随机失败。
"""
from __future__ import annotations

import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.normcase(_os.path.abspath(_sys.path[0] or ".")) == _os.path.normcase(_HERE):
    # `python utils/metrics.py` 会把 utils/ 顶成 sys.path[0]，于是 torch 内部的
    # `import logging` 会命中本包的 utils/logging.py 而非标准库并直接崩溃。
    # 摘掉这一条即可复原；用 `python -m utils.metrics` 或正常 import 时这里是 no-op。
    _sys.path.pop(0)

import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_fscore_support, recall_score)

ALL_LABELS_ERR = "num_classes 必须为正整数"


def _as_np(x) -> np.ndarray:
    """张量/列表 -> numpy，且始终 detach + cpu，避免把计算图带进指标统计。"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _labels(num_classes: int) -> list[int]:
    assert isinstance(num_classes, (int, np.integer)) and num_classes > 0, ALL_LABELS_ERR
    return list(range(int(num_classes)))


def topk_accuracy(logits, target, topk=(1, 5)) -> list[float]:
    """Top-k 精度，**返回百分数（0~100）**，顺序与 topk 参数一一对应。

    与官方 main.py 的 accuracy() 数值等价（官方也是乘 100 后 print）。
    """
    logits = torch.as_tensor(_as_np(logits)).float()
    target = torch.as_tensor(_as_np(target)).long().reshape(-1)
    assert logits.dim() == 2, f"logits 应为 (N, C)，实际 {tuple(logits.shape)}"
    assert logits.size(0) == target.numel(), \
        f"logits 与 target 数量不一致: {logits.size(0)} vs {target.numel()}"

    maxk = min(max(topk), logits.size(1))
    n = target.numel()
    with torch.no_grad():
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # (N, maxk)
        pred = pred.t()                                                # (maxk, N)
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))       # (maxk, N)
        res = []
        for k in topk:
            kk = min(k, maxk)
            hit = correct[:kk].reshape(-1).float().sum().item()
            res.append(100.0 * hit / max(n, 1))
    return res


def macro_f1(y_true, y_pred, num_classes: int) -> float:
    """宏平均 F1（对 37 类等权，不受类别频次影响）。返回 0~1。"""
    y_true = _as_np(y_true).reshape(-1).astype(np.int64)
    y_pred = _as_np(y_pred).reshape(-1).astype(np.int64)
    return float(f1_score(y_true, y_pred, labels=_labels(num_classes),
                          average="macro", zero_division=0))


def per_class_f1(y_true, y_pred, num_classes: int) -> np.ndarray:
    """每类 F1，长度恒为 num_classes（未被预测到的类记 0）。"""
    y_true = _as_np(y_true).reshape(-1).astype(np.int64)
    y_pred = _as_np(y_pred).reshape(-1).astype(np.int64)
    return np.asarray(f1_score(y_true, y_pred, labels=_labels(num_classes),
                              average=None, zero_division=0), dtype=np.float64)


def per_class_recall(y_true, y_pred, num_classes: int) -> np.ndarray:
    """每类召回，长度恒为 num_classes（该类无样本时记 0）。"""
    y_true = _as_np(y_true).reshape(-1).astype(np.int64)
    y_pred = _as_np(y_pred).reshape(-1).astype(np.int64)
    return np.asarray(recall_score(y_true, y_pred, labels=_labels(num_classes),
                                   average=None, zero_division=0), dtype=np.float64)


def per_class_support(y_true, num_classes: int) -> np.ndarray:
    """每类真实样本数（类分布不均时的分母，报告里必须和 F1 一起给）。"""
    y_true = _as_np(y_true).reshape(-1).astype(np.int64)
    return np.bincount(y_true, minlength=int(num_classes))[:int(num_classes)]


def confusion(y_true, y_pred, num_classes: int, normalize: str = "true") -> np.ndarray:
    """混淆矩阵。

    normalize="true"（默认）：**手工按行归一化**，cm[i, j] = P(pred=j | true=i)，
        support=0 的行整体写 0；刻意不依赖 sklearn 的 normalize='true'——不同版本对
        0/0 的处理不同（有的写 0、有的报 nan），会破坏验收断言。
    normalize="pred"：按列（预测类别）归一化；normalize=None/"none"/"all"：原始计数。
    """
    y_true = _as_np(y_true).reshape(-1).astype(np.int64)
    y_pred = _as_np(y_pred).reshape(-1).astype(np.int64)
    labels = _labels(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(np.float64)

    if normalize in (None, "none", "all"):
        return cm
    if normalize == "true":
        axis_sum = cm.sum(axis=1, keepdims=True)     # 每行 = 该类真实样本数
    elif normalize == "pred":
        axis_sum = cm.sum(axis=0, keepdims=True)     # 每列 = 该类被预测次数
    else:
        raise ValueError(f"normalize 只能是 'true'/'pred'/'none'，实际 {normalize!r}")
    return np.divide(cm, axis_sum, out=np.zeros_like(cm), where=axis_sum > 0)


def ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error（期望校准误差）。

    ECE = Σ_b (n_b / N) * |acc(b) - conf(b)|，confidence 取预测类别的 softmax 概率。
    过自信（ECE 高、acc < conf）是知识蒸馏/标签平滑实验里最常见的副作用，
    所以调参实验必须同时报 acc 与 ECE。

    probs 可传 (N, C) 概率矩阵，也可传 (N,) 的一维「预测置信度」（此时需自行保证
    y_true 里的标签是预测标签）。
    """
    probs = np.asarray(probs, dtype=np.float64)
    y_true = _as_np(y_true).reshape(-1).astype(np.int64)
    if probs.ndim == 2:
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
    elif probs.ndim == 1:
        conf = probs
        pred = y_true                                  # 一维输入视为已给预测标签
    else:
        raise ValueError(f"probs 应为 1D 或 2D，实际 {probs.ndim}D")

    correct = (pred == y_true).astype(np.float64)
    n = correct.size
    if n == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    # 用 bins[1:-1] 做 digitize：置信度 1.0 落到最后一箱而不是越界
    idx = np.digitize(conf, bins[1:-1])
    total = 0.0
    for b in range(int(n_bins)):
        m = idx == b
        if not np.any(m):
            continue
        acc_b = correct[m].mean()
        conf_b = conf[m].mean()
        total += (m.sum() / n) * abs(acc_b - conf_b)
    return float(total)


def compute_metrics(y_true, y_pred, logits=None, num_classes: int = 37) -> dict:
    """一次性算出实验报告要的全部指标。

    返回键（至少）：
        top1, top5, macro_f1        —— 0~1 小数
        per_class_f1, per_class_recall, per_class_support
        cm                          —— 行归一化混淆矩阵
        n, num_classes, cm_counts, accuracy, macro_recall,
        top1_pct, top5_pct, ece

    logits 给了才会算 top5 与 ECE（需要概率分布）；只给 y_true/y_pred 时 top5/ece 为 None，
    但 top1 仍由 y_pred 直接算出，不会缺键。
    """
    y_true = _as_np(y_true).reshape(-1).astype(np.int64)
    y_pred = _as_np(y_pred).reshape(-1).astype(np.int64)
    assert y_true.size == y_pred.size, "y_true 与 y_pred 长度不一致"

    top1 = float((y_true == y_pred).mean()) if y_true.size else 0.0
    top5, ece_val, probs = None, None, None
    if logits is not None:
        p1, p5 = topk_accuracy(logits, y_true, topk=(1, 5))   # 百分数
        top1_from_logits = p1 / 100.0
        # 只告警不断言：TTA / 多模型融合时，调用方传入的 y_pred 可能来自 softmax 前的
        # 另一路 logits（例如各模型 logits 平均前各自 argmax），此时不一致是预期的。
        # 但 logits 是真源，top1 一律以它为准——否则 top1 与 top5 会来自两个类别空间。
        if y_true.size and abs(top1_from_logits - top1) > 1e-6:
            print(f"[metrics] 提示: y_pred 的 top1={top1:.6f} 与 logits argmax 的 "
                  f"top1={top1_from_logits:.6f} 不一致，已以 logits 为准。")
        top1, top5 = top1_from_logits, p5 / 100.0
        probs = torch.softmax(torch.as_tensor(_as_np(logits)).float(), dim=1).numpy()
        ece_val = ece(probs, y_true)

    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=_labels(num_classes), zero_division=0)
    return {
        "n": int(y_true.size),
        "num_classes": int(num_classes),
        "top1": top1,
        "top5": top5,
        "top1_pct": 100.0 * top1,
        "top5_pct": None if top5 is None else 100.0 * top5,
        "accuracy": top1,                                  # 别名，报告里两种叫法都要用
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "macro_recall": float(np.mean(r)) if len(r) else 0.0,
        "macro_precision": float(np.mean(p)) if len(p) else 0.0,
        "per_class_f1": np.asarray(f1, dtype=np.float64),
        "per_class_recall": np.asarray(r, dtype=np.float64),
        "per_class_precision": np.asarray(p, dtype=np.float64),
        "per_class_support": np.asarray(s, dtype=np.float64),
        "cm": confusion(y_true, y_pred, num_classes, normalize="true"),
        "cm_counts": confusion(y_true, y_pred, num_classes, normalize="none"),
        "ece": None if ece_val is None else float(ece_val),
        "random_level": 1.0 / int(num_classes),            # 37 类随机水平 ≈ 2.70%
    }


if __name__ == "__main__":
    # 自检：用手工可验的小样例核对 top-k、宏平均与行归一化语义
    torch.manual_seed(0)
    num_classes, n = 5, 200
    logits = torch.randn(n, num_classes)
    y_true = torch.randint(0, num_classes, (n,))
    y_pred = logits.argmax(dim=1)

    p1, p5 = topk_accuracy(logits, y_true, topk=(1, 5))
    assert abs(p1 - 100.0 * (y_pred == y_true).float().mean().item()) < 1e-4
    assert 0.0 <= p1 <= 100.0 and abs(p5 - 100.0) < 1e-6, (p1, p5)   # 5 类时 top5 恒为 100

    f1s = per_class_f1(y_true, y_pred, num_classes)
    rec = per_class_recall(y_true, y_pred, num_classes)
    assert f1s.shape == (num_classes,) and rec.shape == (num_classes,)
    assert abs(float(f1s.mean()) - macro_f1(y_true, y_pred, num_classes)) < 1e-9, \
        "macro_f1 必须等于每类 F1 的算术平均"

    cm = confusion(y_true, y_pred, num_classes, normalize="true")
    row_sum = cm.sum(axis=1)
    assert np.allclose(row_sum[row_sum > 0], 1.0), "行归一化后非零行和必须为 1"
    assert np.allclose(cm[np.sum(cm, axis=1) == 0], 0.0)
    raw = confusion(y_true, y_pred, num_classes, normalize="none")
    assert int(raw.sum()) == n, "计数矩阵元素和必须等于样本数"

    m = compute_metrics(y_true, y_pred, logits=logits, num_classes=num_classes)
    for k in ("top1", "top5", "macro_f1", "per_class_f1", "per_class_recall", "cm"):
        assert k in m, f"compute_metrics 缺少键 {k}"
    assert 0.0 <= m["ece"] <= 1.0
    print(f"[metrics] 自检通过 top1={m['top1']:.4f} top5={m['top5']:.4f} "
          f"macro_f1={m['macro_f1']:.4f} ece={m['ece']:.4f} 随机水平={m['random_level']:.4f}")
