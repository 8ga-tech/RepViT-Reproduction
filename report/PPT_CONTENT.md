# 答辩 PPT 内容简报

> 供 `ppt-master` 生成答辩 PPT 使用。**所有数字都是本机实测值**，
> 来源文件在每页的「数据来源」里给出，可逐条溯源。
> 风格：**深色科技风**（深底 + 高亮强调色）。全篇 15 页。
> 题目要求：所有性能图表必须注明 模型型号 / 输入尺寸 / batch size / 硬件 /
> 推理后端 / 精度类型 / 测试次数。

---

## 全局事实（每页可引用）

- 论文：RepViT: Revisiting Mobile CNN From ViT Perspective，CVPR 2024，arXiv:2307.09283
- 官方代码：THU-MIG/RepViT（Apache-2.0）；现代实现：timm 1.0.29
- 硬件：RTX 4060 Laptop 8GB（CUDA 12.6）+ i7-13650HX（20 逻辑核）+ 15.8 GB RAM
- 系统：Windows 11 10.0.26200；Python 3.14.5；PyTorch 2.14.0+cu126；
  ONNX Runtime 1.30.0（CPUExecutionProvider）
- 在线仓库：https://github.com/8ga-tech/RepViT-Reproduction

---

## P01 封面

- 标题：RepViT 轻量图像分类模型复现、优化与多模型部署
- 副标题：2026 秋 one 团队 AI 算法组考核 · 答辩汇报
- 底部一行结论条：
  **ImageNet 子集 Top-1 78.20% · Pet-37 test Top-1 92.26% · 重参数化 max|Δ|=6.5e-06 · ONNX CPU P50 12.5~17.2ms**

---

## P02 任务与完成情况

六个环节一页打勾：

| 环节 | 完成情况 | 关键证据 |
|---|---|---|
| ① 官方模型运行评价 | ✅ 2 基础型号 + 3 家族型号 | `outputs/pretrained_eval/` |
| ② M0.9 迁移训练 | ✅ 40 epoch 满预算 | `checkpoints/baseline_best.pt` |
| ③ 控制变量优化 | ✅ 4 方案 + 3 组组合消融 | `configs/opt_*.yaml` |
| ④ 可视化与可解释性 | ✅ 五曲线/混淆矩阵/Grad-CAM/外部图 | `outputs/{curves,confusion_matrix,gradcam,predictions}/` |
| ⑤ 结构重参数化 | ✅ max\|Δ\|=7.092953e-06（展示 7.093e-06），Top-1 32/32 一致 | `outputs/reparam/` |
| ⑥ ONNX 多模型部署 | ✅ 3 个 ONNX，一致率 100% | `onnx/*.onnx`、`outputs/benchmarks/` |

> 注：**不做**第八章拓展任务（INT8/蒸馏/剪枝/多随机种子/边缘部署/独立实现核心模块）。

---

## P03 RepViT 核心结构（自绘图，非论文插图）

用一张「竖直流水线」图：

```
输入 224×224×3
Stem   conv1 3×3 s2 → 24ch 112×112 ；conv2 3×3 s2 → 48ch 56×56
Stage0 2×RepViT Block   48ch  56×56
Stage1 2×RepViT Block   96ch  28×28
Stage2 14×RepViT Block 192ch  14×14      ← Block 数最多
Stage3 2×RepViT Block  384ch   7×7
GAP → 384 维
Head RepVitClassifier(NormLinear: BN1d+Linear) → 37 维
```

- 分辨率 224→112→56→28→14→7；通道 3→24→48→96→192→384；Block 共 20 个
- **形状由 forward hook 现测**，非抄图
- 数据来源：`outputs/architecture/repvit_m0_9_arch.png` / `.md`（`tools/draw_arch.py`）

---

## P04 为什么 RepViT 不是标准 ViT

左右对照表：

| | 标准 ViT | RepViT |
|---|---|---|
| 空间混合 | Multi-Head Self-Attention（QK^T + Softmax） | **RepVGGDW**（3×3 depthwise + 1×1 depthwise） |
| 通道混合 | MLP / FFN | **Channel Mixer**（1×1 → GELU → 1×1） |
| 下采样 | patchify（stride=16 大核） | **Early Convolution Stem**（两组 stride=2 的 3×3） |
| 归纳偏置 | 弱（需大数据） | 强（卷积自带局部性/平移等变） |
| ONNX 图里有无注意力算子 | 有 MatMul + Softmax | **没有**——全部是 Conv/Gemm/Add/Mul/Div/Clip/Relu |

**结论**：RepViT 借鉴的是轻量 ViT 的**宏观架构选择**（Token/Channel Mixer 分离、
更深的 stem、更小的扩张比 + 更宽的通道），而不是自注意力算子本身。

---

## P05 官方多型号评价

**口径**：1000 张自建分层子集（每类 1 张，种子 20260912）；
输入 224×224；`Resize(256)+CenterCrop(224)`（crop_pct=0.875）；FP32；batch 64；
RTX 4060 Laptop；vendored 官方实现 + `replace_batchnorm` 融合态。

| 型号 | 参数量 | MACs | 本机 Top-1 | 本机 Top-5 | 官方公布 Top-1 |
|---|---|---|---|---|---|
| **RepViT-M0.9** | 5,489,328 | 0.847 G | **78.20%** | 93.70% | 78.7% |
| **RepViT-M1.0** | 7,302,796 | 1.143 G | **79.90%** | 94.20% | 80.0% |

- 预处理口径对照：M0.9 在 crop_pct=0.95 下 78.60%/93.00%；M1.0 为 80.00%/93.90%
- **声明**：该子集为**自建**，非考核方指定子集；结果不代表论文完整 ImageNet-1K 验证集结果
- 数据来源：`outputs/pretrained_eval/{repvit_m0_9,repvit_m1_0}/metrics.json`

---

## P06 数据集与训练流程

- **两套数据、两套标签，绝不混用**：
  Oxford-IIIT Pet 37 类（迁移训练）/ ImageNet-1K 1000 类（官方权重评价）
- Pet 事实基线：7390 张 jpg；trainval 3680 行；test 3669 行；
  CLASS-ID **1-based**（0-based = −1）；猫 12 类 / 狗 25 类
- 划分（唯一划分脚本，seed=42）：train **2940** / val **740**（每类恰好 20）/ test **3669**
- 泄漏检查：文件名交集 0/0/0，**MD5 交集 0/0/0**，孤儿图 0，内部重复 0
- **test 只评价一次**（`eval_count = 1`），最优模型只看 `val_macro_f1`
- 训练：AdamW lr 1e-3（backbone ×0.1）、wd 0.05、cosine + 3ep warmup、
  Label Smoothing 0.1、bf16 autocast、batch 64、**40 epoch 满预算**

---

## P07 Baseline 结果

| 指标 | 验证集 best | **测试集（唯一一次）** |
|---|---|---|
| Top-1 | — | **92.26%** |
| Top-5 | — | **99.24%** |
| Macro-F1 | 0.9502 | **92.14%** |

- 骨干**确实被训练**：`changed_tensors = 706`，其中骨干 699 个（非只训分类头）
- 分类头替换：直接以 `timm.create_model(..., num_classes=37, distillation=False)` 建网，
  **不用 `reset_classifier`**（会丢 BN 统计量）
- 权重加载：`missing=2`（仅换头后的分类 Linear）、`unexpected=0`
- 数据来源：`outputs/metrics/baseline_test.json`、`outputs/logs/baseline_metrics.csv`

---

## P08 优化假设与控制变量

四个方案 + 工程化的控制变量校验：

| 方案 | 差异键 | 假设 |
|---|---|---|
| B 混合增强 | 3 | Mixup/CutMix 软标签平滑决策面，等价扩容 |
| C 加强增强 | 8 | RandAugment + 更激进 RRC，压缩纹理记忆 |
| D 差异化 lr | 1 | backbone ×0.1→0.05，锚住预训练解 |
| **A 组合式** | **12** | 三者机制互补，前段更稳 + 末段更高 |

- **控制变量不是靠自觉**：`tools/diff_config.py` 做叶子级比对，
  实际差异键必须与配置声明的 `_expected_diff_` 逐项一致 → 7 份配置全部 `DIFF PASS`
- 保持不变：数据划分 / 预训练权重 / 随机种子 / 训练轮数(40) / 评价方式 / best 选择标准
- ❗ 关键修正：早停统一关闭（`patience=999`）——早期 `opt_mix` 在 epoch 24 早停而
  baseline 跑满 40，预算差 37.5%，控制变量不成立

---

## P09 曲线与定量结果（Baseline vs 优化）

- 五条曲线同图（同坐标范围）：train loss / val loss / val Top-1 / val Macro-F1 / lr
- 定量对比表：各实验的 val Top-1、val Macro-F1、test Top-1/5、Macro-F1
- 预算等价：8 组实验的 `(epochs, steps_per_epoch)` 均为 (40, 45)
- 数据来源：`outputs/curves/opt_compare.png`、`outputs/logs/opt_compare_summary.csv`
- ⚠️ 注：Mixup/CutMix 使训练损失不可与 baseline 直接比（软标签会抬高 train loss）

（**此页数值待 M07 完成后填入**）

---

## P10 混淆矩阵与失败案例

- 归一化混淆矩阵：37×37，行和 = 1.0，argmax 在对角线 **37/37 = 100%**，对角元均值 0.9219
- **猫狗块分析（关键结论）**：
  - 猫召回 88.17% / 狗召回 94.21%
  - **跨物种误判仅 3.17%（9/284 个错误）**；**种内品种错误 275 个**
  - → 模型**几乎不混淆猫狗**，失败集中在**细粒度品种差异**
- 最难的三类：Staffordshire Bull Terrier（F1 0.691）、
  American Pit Bull Terrier（0.703）、Ragdoll（0.766）
- 数据来源：`outputs/confusion_matrix/baseline_cm.png`、`baseline_per_class.csv`、
  `baseline_cat_dog_block.json`

---

## P11 Grad-CAM 与外部图片

- **手写实现**（不依赖 pytorch-grad-cam），按 Selvaraju et al. ICCV 2017 公式 (1)(2)
- 挂载层：`stages[-1].blocks[-1]`（A）与 `.channel_mixer.conv2`（B）——
  **两者不等价**，已出对比图；**绝不挂 head**（会抛 Invalid grads shape）
- 产物：正确案例 6 张、错误案例 6 张、挂载层对比 1 张
- **训练集以外的实际图片**（8 张跨集合实拍图）：品种判断 **8/8 全对**
  - beagle 90.32% / pug 43.86% / boxer 64.28% / chihuahua 67.98%
  - samoyed 83.50% / newfoundland 77.14% / great_pyrenees 86.95% / pomeranian 36.91%
- 分布差异：外部图短边中位数 +10.5%、亮度 +10.6%、背景复杂度 **+24%**
- 来源声明：因 Wikimedia 不可达，改用 ImageNet 跨集合实拍图，逐张登记许可，**不声称原创**
- 数据来源：`outputs/gradcam/`、`outputs/benchmarks/external_top5_repvit_m0_9_pet37.csv`

---

## P12 结构重参数化

- 训练态 Block 的 Token Mixer（RepVGGDW）= **3×3 depthwise 分支 + 1×1 depthwise 分支**，
  各自带 BN，相加后接 ReLU
- 融合公式：`W' = W·γ/√(σ²+ε)`、`b' = (b−μ)·γ/√(σ²+ε) + β`；
  1×1 核零填充成 3×3 后与 3×3 核**直接相加**
- **必须在 `eval()` 之后融合**（BN 要用 running 统计量，不是 batch 统计量）

| 指标 | Pet-37 | M0.9 官方 C=1000 |
|---|---|---|
| max_abs_err | **7.092953e-06** | 旧官方 C=1000 记录未匹配权重，不作为当前结论 |
| Top-1 一致 | **True（32/32）** | N/A（权重键不匹配） |
| BN 模块数 | **107 → 0** | N/A |
| 参数量（单头口径） | 4,732,805 → 4,696,301（净减 36,504） | 净减 36,504 |
| ONNX 节点 | BN **0**、Conv **103**（未融合 126，少 23） | 同 |

- 数据来源：`outputs/reparam/repvit_m0_9_pet37_reparam_report.json`

---

## P13 ONNX 多模型部署

| registry key | 文件 | 大小 | 图内节点 | 类别数 | 标签文件 |
|---|---|---|---|---|---|
| `repvit_m0_9_in1k` | `onnx/repvit_m0_9_in1k.onnx` | 20.36 MB | BN=0, Conv=103 | 1000 | `imagenet_classes.txt` |
| `repvit_m1_0_in1k` | `onnx/repvit_m1_0_in1k.onnx` | 27.33 MB | BN=0, Conv=103 | 1000 | `imagenet_classes.txt` |
| `repvit_m0_9_pet37` | `onnx/repvit_m0_9_pet37.onnx` | 18.89 MB | BN=0, Conv=103 | 37 | `pet_classes.txt` |

- 文件名一律经 `deploy/model_registry.py` 的 `onnx_path(key)` 取得，**任何地方不硬写**
- **独立实现**：`deploy/infer_onnx.py` 用 PIL 手写 resize+crop，不 import torchvision
  （共用一份 transform 会让一致性误差恒为 0，反而掩盖 bug）
- **PyTorch vs ONNX 一致性**（固定划分列表、真实图片）：

  固定 `datasets/lists/pet_test.txt` 前 12 张图片（`n=12`）；README 早期的 5.25e-06 是另一批 n=8 对照。

| 指标 | repvit_m0_9_pet37 |
|---|---|
| max\|Δlogits\| | **6.198883e-06**（展示 6.199e-06） |
| mean\|Δlogits\| | **1.500690e-06** |
| **Top-1 一致率** | **100.00%** |
| Top-5 集合一致率 | 100.00% |

---

## P14 性能比较

**口径必须完整标注**：ONNX Runtime **1.30.0 CPUExecutionProvider** / FP32 /
batch=1 / 224×224 / 预热 **10** 次 + 正式 **50** 次 / `threads_intra=4` /
i7-13650HX / Windows 11。

| 模型 | mean (ms) | **P50 (ms)** | **P95 (ms)** | 文件大小 |
|---|---|---|---|---|
| `repvit_m0_9_in1k` | 12.74 | 13.60 | 16.66 | 20.36 MB |
| `repvit_m1_0_in1k` | 17.49 | 17.18 | 20.82 | 27.33 MB |
| `repvit_m0_9_pet37` | 12.74 | 12.48 | 16.00 | 18.89 MB |

- **mean**：易被慢样本拉高；**P50**：典型耗时，最适合横向比较；**P95**：尾延迟
- M1.0 参数量 +33%、MACs +34.9%，但 P50 只 +26.3% → **延迟不与 MACs 成正比**
  （depthwise 算术强度低，瓶颈在内存带宽；小模型受固定开销支配）
- ❗ 官方 iPhone 12 延迟（0.9/1.0 ms）**不可与本机 CPU 结果横向比较**
- 数据来源：`outputs/benchmarks/summary.csv`

---

## P15 问题、总结与后续

**遇到的问题（选 3 条最有代表性的）**
1. 官方 `.pth` 与 timm 键名**交集为 0**（713 vs 389）→ 证明两条路线**逐位等价**
   （max|Δlogits| = 0），省去手写转换器
2. `data/provided/` 在 29 页试题 PDF 中**从未定义**（全文搜 "provided" 命中 0）→
   改判为结构性偏差，改用官方归档 + 自建子集并显式声明
3. 早停导致**预算不等价**（37.5% 差距）→ 统一关闭早停，8 组均为 40 epoch

**总结**：复现链路完整、数字可溯源、控制变量有工程化校验、部署闭环一致率 100%。

**后续计划**：INT8 量化 / 知识蒸馏（M2.3→M0.9）/ 集显后端（OpenVINO·DirectML）/
置信度校准（实测存在高置信度错误）/ 七类鲁棒性扰动。
