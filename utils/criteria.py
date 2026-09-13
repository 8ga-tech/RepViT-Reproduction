# utils/criteria.py —— 损失/混合的工具函数（不是可执行脚本，故不放 tools/）
import torch.nn as nn
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


def build_mixup(cfg: dict, num_classes: int):
    """按配置构造 timm Mixup；train.mixup_prob = 0.0 时返回 None（调用方据此分派）。"""
    t = cfg['train']
    if float(t['mixup_prob']) <= 0.0:          # 先看 prob：baseline 的 alpha 也是 0.0，不能靠 alpha 判空
        return None
    return Mixup(
        mixup_alpha=t['mixup_alpha'],          # 0.2，来自 mixup 原论文 ImageNet 表格
        cutmix_alpha=t['cutmix_alpha'],        # 1.0，lambda~Beta(1,1)=U(0,1)
        cutmix_minmax=None,
        prob=t['mixup_prob'],                  # 1.0：每个 batch 都做混合
        switch_prob=t['mixup_switch_prob'],    # 0.5：每个 batch 在 mixup / cutmix 间二选一
        mode='batch',                          # 整批用同一个 lambda
        label_smoothing=t['label_smoothing'],  # 平滑在这里注入，不要在外面再叠
        num_classes=num_classes,
    )


def resolve_criteria(cfg: dict, num_classes: int):
    """返回 (mixup_fn, train_criterion, val_criterion, 描述串)。

    互斥规则：
      - mixup 开启：timm 的 mixup_target() 已按 off_value=ls/K、on_value=1-ls+off 注入平滑，
        训练 loss 必须换成 SoftTargetCrossEntropy，绝不能再包 LabelSmoothingCrossEntropy。
      - mixup 关闭且 ls>0：用 LabelSmoothingCrossEntropy（与 PyTorch 原生 label_smoothing
        在 reduction='mean' 且 u(k)=1/K 时数值等价）。
      - 两者都关：普通 CE。
    验证集永远用硬标签 CE，与训练目标解耦。
    """
    ls = float(cfg['train']['label_smoothing'])
    mixup_fn = build_mixup(cfg, num_classes)
    if mixup_fn is not None:
        train_crit = SoftTargetCrossEntropy()
        desc = f'SoftTargetCrossEntropy (mixup/cutmix 软标签, label_smoothing={ls} 由 Mixup 注入)'
        assert not isinstance(train_crit, (nn.CrossEntropyLoss, LabelSmoothingCrossEntropy)), \
            '混用软标签损失与标签平滑损失会导致双重平滑'
    elif ls > 0:
        train_crit = LabelSmoothingCrossEntropy(ls)
        desc = f'LabelSmoothingCrossEntropy({ls})'
    else:
        train_crit = nn.CrossEntropyLoss()
        desc = 'CrossEntropyLoss'
    return mixup_fn, train_crit, nn.CrossEntropyLoss(), desc
