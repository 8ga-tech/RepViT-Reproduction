# RepViT-M0.9 整网结构（自绘，形状现测）

- 来源：`tools/draw_arch.py` 用 forward hook 读真实张量得到，非论文插图。
- 架构：`repvit_m0_9`（timm），迁移后类别数 `37`。

## 九要素

1. **输入尺寸**：224×224×3（RGB）
2. **Stem**：stem.conv1 → stem.conv2；Early Convolution Stem，用两组 stride=2 卷积把 224 快速降到 56，通道升到 48
3. **四个主要阶段**：`stages.0` 含 2 个 Block；`stages.1` 含 2 个 Block；`stages.2` 含 14 个 Block；`stages.3` 含 2 个 Block
4. **特征图分辨率变化**：224→112→56→28→14→7（共下采样 224→7，倍率 32×）
5. **通道数变化**：3→24→48→96→192→384
6. **RepViT Block**：共 20 个。每个 Block = Token Mixer（RepVGGDW：3×3 depthwise + 1×1 depthwise 双分支，训练态含 BN，推理态融合为单个 3×3 depthwise）+ Channel Mixer（1×1 升维 → GELU → 1×1 降维）+ 残差连接；仅部分 Block 带 SE（Squeeze-Excite）
7. **Global Average Pooling**：把 384×7×7 池化成 384 维向量
8. **分类头**：`RepVitClassifier` → `NormLinear`（BatchNorm1d + Linear），**单头**（`distillation=False`）；不使用多头/多层 MLP 头，以降低延迟
9. **最终输出维度**：37

## 参数量口径（三个数都对，差别在口径）

| 口径 | 参数量 |
|---|---|
| 训练态双头（含蒸馏头，C=1000） | 5,489,328 |
| 未融合单头（C=1000） | 5,103,560 |
| 未融合单头（C=37，本任务训练态） | 4,732,805 |
| 骨干（不含任何分类头） | 4,717,792 |
| 融合后单头（C=1000） | 5,067,056 |
| 融合后单头（C=37） | 4,696,301 |

> 复现命令：`python tools/count_params.py --model repvit_m0_9`（实测值与上表逐位一致）。
