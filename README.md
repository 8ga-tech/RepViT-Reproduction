# RepViT 复现、优化与多模型部署

RepViT（*Revisiting Mobile CNN From ViT Perspective*, CVPR 2024, [arXiv:2307.09283](https://arxiv.org/abs/2307.09283)）
在 Oxford-IIIT Pet 37 类上的迁移训练、控制变量优化、可解释性分析与 ONNX 多模型部署。

- 官方代码：[THU-MIG/RepViT](https://github.com/THU-MIG/RepViT)（Apache-2.0）
- 现代实现：[timm](https://github.com/huggingface/pytorch-image-models)（Apache-2.0）
- 数据集：[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)（CC BY-SA 4.0）

> **一句话结论**：RepViT-M0.9 是一个**纯卷积**网络（无自注意力）。它在 ImageNet-1K
> 指定子集上 Top-1 = **78.20%**（官方权重实测，官方公布 78.7%）；迁移到 Pet 37 类后
> test Top-1 = **92.34%**、Macro-F1 = **92.22%**；结构重参数化前后 logits
> `max|Δ| = 7.09e-06`；六个 ONNX 模型在 ONNX Runtime CPU 上批量 1 / FP32 /
> 224×224 的推理延迟为 **7.1 ~ 32.7 ms**，PyTorch↔ONNX Top-1 一致率 **100%**。

---

## 目录

1. [如何安装环境](#1-如何安装环境)
2. [如何准备数据集](#2-如何准备数据集)
3. [如何获取官方权重](#3-如何获取官方权重)
4. [如何运行官方模型评价](#4-如何运行官方模型评价)
5. [如何训练 Baseline](#5-如何训练-baseline)
6. [如何运行优化实验](#6-如何运行优化实验)
7. [如何进行验证和测试](#7-如何进行验证和测试)
8. [如何生成曲线、混淆矩阵和 Grad-CAM](#8-如何生成曲线混淆矩阵和-grad-cam)
9. [如何执行结构重参数化](#9-如何执行结构重参数化)
10. [如何导出不同型号的 ONNX 模型](#10-如何导出不同型号的-onnx-模型)
11. [如何运行 ONNX 推理](#11-如何运行-onnx-推理)
12. [如何完成性能测试](#12-如何完成性能测试)
13. [一键复现与耗时](#13-一键复现与耗时)
14. [代码来源标注](#14-代码来源标注)

---

## 1. 如何安装环境

本项目在 **Python 3.14.5 / Windows 11 / RTX 4060 Laptop (8 GB, CUDA 12.6)** 上实测通过。
所有依赖版本在 `requirements.lock.txt` 中冻结（由 `python -m pip freeze` 生成）。

```bash
# (1) 先装 torch 系，必须走 PyTorch 官方源，不要从 PyPI 镜像装
python -m pip install torch==2.14.0 torchvision==0.29.0 \
    --index-url https://download.pytorch.org/whl/cu126

# (2) 再装其余依赖
python -m pip install -r requirements.txt

# (3) 环境自检：打印版本矩阵并落盘 outputs/env_snapshot.json
python tools/env_check.py
```

**几处必须说明的坑**（都是实测踩过的）：

| 项 | 说明 |
|---|---|
| `onnxruntime` | 必须 **1.30.0**（或 ≤1.28.x）。**1.29.0 / 1.29.1 有 CPU FP16 Gemm 回退 bug**，会污染基准测试。当前环境为 1.30.0。 |
| `torch` | CPU wheel 与 CUDA wheel 不通用。本机原为 `2.14.0+cpu`，改为 `+cu126` 后训练从「预估 120 小时」降到 **约 5 分钟**。 |
| HuggingFace | 国内直连不通，需 `export HF_ENDPOINT=https://hf-mirror.com`（timm 的 `pretrained=True` 走这条路）。 |
| 中文 Windows 编码 | 脚本一律显式 `encoding="utf-8"`；JSON 落盘一律 `ensure_ascii=True`——本仓库根目录含中文，任何写进 JSON 的绝对路径都会让 `open()`（默认 GBK）读不回来。 |

## 2. 如何准备数据集

### 2.1 Oxford-IIIT Pet（自训练用）

```bash
# 官方归档（images.tar.gz / annotations.tar.gz）解压到 data/oxford-iiit-pet/
# 校验：images MD5 = 5c4f3ee8e5d25df40f4fd59a7f44e54c
#       annotations MD5 = 95a8c909bbe2e81eed6a22bccdf3f68f
python tools/assert_data.py --root data/oxford-iiit-pet     # 期望 ALL PASS

# 生成类别映射（37 类，0-based，与官方 CLASS-ID - 1 一致）
python datasets/build_pet_class_map.py

# 生成划分（唯一划分脚本；每类固定 20 张进 val）
python datasets/make_pet_split.py --root data/oxford-iiit-pet \
    --out-dir datasets/lists --val-per-class 20 --seed 42
# -> pet_train.txt 2940 行 / pet_val.txt 740 行 / pet_test.txt 3669 行

# 数据泄漏检查（train/val/test 两两交集必须为 0）
python datasets/audit_leakage.py
```

产物：`labels/pet_classes.txt`（37 行）、`labels/pet_class_to_idx.json`、
`labels/pet_species.json`（猫/狗，**不能用 `idx<12` 判猫，官方 CLASS-ID 是猫狗交错的**）、
`datasets/lists/pet_{train,val,test}.txt`（格式 `<image_id>\t<class_idx>`）。

### 2.2 ImageNet-1K 验证子集（官方模型评价用）

```bash
# 把 14 个 parquet 分片还原成 ImageFolder（1000 类 × 50 张 = 50,000 张）
python datasets/unpack_imagenet_val.py

# 分层抽样出 1000 张（每类 1 张，固定种子）
python datasets/make_imagenet_subset.py --n 1000 --seed 20260912
# -> datasets/lists/imagenet_val_subset.txt

# 标签顺序三重自检（DoD #10 的四个锚点）
python tools/verify_imagenet_labels.py
```

> **声明**：`imagenet_val_subset.txt` 是**自建**的分层抽样清单（每类 1 张、种子 20260912），
> **不是考核方下发的指定子集**。本仓库对它的每一条结论都只声称「在该自建子集上的结果」。

## 3. 如何获取官方权重

```bash
bash tools/download_pretrained.sh       # 从 GitHub Releases v1.0 下载并打印 SHA256
python tools/check_weights.py --dir checkpoints/pretrained
```

实测校验值（`repvit_m0_9_distill_300e.pth` 必须是 **22,422,548** 字节）：

| 文件 | 字节数 | SHA256（前 16 位） |
|---|---|---|
| `repvit_m0_9_distill_300e.pth` | 22,422,548 | `857eb0e6a992591a` |
| `repvit_m1_0_distill_300e.pth` | 29,675,245 | `283bb9865f705481` |
| `repvit_m1_1_distill_300e.pth` | 35,668,677 | `1b364220241273d5` |
| `repvit_m1_5_distill_300e.pth` | **59,375,411** | `b434320aed41372a` |
| `repvit_m2_3_distill_300e.pth` | 95,860,931 | `be53c7dfb059ae6d` |

> **来源声明**：本机直连 `github.com` 返回 HTTP 000（不可达），权重经
> `https://ghfast.top/https://github.com/...` 前置代理取得。字节数与官方 Release
> 逐位一致，SHA256 已记录在 `outputs/metrics/weight_sha256.json`。
>
> **一个实测教训**：`repvit_m1_5` 曾出现「字节数与期望值完全一致、但 `torch.load`
> 报 `PytorchStreamReader failed reading zip archive`」的情况——字节数一致**不等于**
> 内容完好。`tools/check_weights.py` 已加固为「字节数 + zip 中央目录 + torch.load 实测」
> 三道校验（`loadable=False` 计数必须为 0）。

## 4. 如何运行官方模型评价

```bash
# 单型号
python tools/eval_pretrained.py --cfg configs/pretrained_eval.yaml \
  --set pretrained_eval.model=repvit_m0_9 \
        pretrained_eval.weights=checkpoints/pretrained/repvit_m0_9_distill_300e.pth

# 多型号批量 + 汇总表
python tools/run_all_pretrained.py --cfg configs/pretrained_eval.yaml \
  --model repvit_m0_9 repvit_m1_0 repvit_m1_1 repvit_m1_5 repvit_m2_3

# 只校验 list 与 labels 对齐，不加载模型
python tools/eval_pretrained.py --cfg configs/pretrained_eval.yaml \
  --set pretrained_eval.check_data=true
```

**实测结果**（1000 张自建分层子集，crop_pct=0.875 即官方 `Resize(256)+CenterCrop(224)` 口径）：

| 型号 | Top-1 (%) | Top-5 (%) | 官方公布 Top-1 |
|---|---|---|---|
| RepViT-M0.9 | **78.20** | 93.70 | 78.7 |
| RepViT-M1.0 | **79.90** | 94.20 | 80.0 |
| （crop_pct=0.95 口径）M0.9 | 78.60 | 93.00 | — |
| （crop_pct=0.95 口径）M1.0 | 80.00 | 93.90 | — |

> 本结果为**指定规模的自建验证子集上的实际运行结果**，不代表论文完整 ImageNet-1K 验证集结果。

## 5. 如何训练 Baseline

```bash
# 冒烟（每个 epoch 只跑 5 个 batch，不入库）
python tools/train.py --cfg configs/baseline.yaml \
    --set train.limit_batches=5 train.epochs=3 train.early_stop_min_epochs=1

# 正式训练（40 epoch）
export HF_ENDPOINT=https://hf-mirror.com
python tools/train.py --cfg configs/baseline.yaml \
    2>&1 | tee outputs/logs/train_baseline_$(date +%Y%m%d_%H%M%S).log
```

配置真源是 `configs/baseline.yaml`（含逐字段值域注释）。
产物：`checkpoints/baseline_{best,last}.pt`、`outputs/logs/baseline_metrics.{csv,jsonl}`、
`outputs/metrics/baseline_test.json`、`outputs/predictions/baseline_test_preds.csv`。

**实测结果**：val Macro-F1（best）= **0.9462**；test Top-1 = **92.34%**、
Top-5 = **99.26%**、Macro-F1 = **92.22%**（`eval_count=1`，test 只评价一次）。

> **训练协议**：固定 40 epoch 满预算。`early_stop_patience` 统一设为 999（等价于关闭），
> 这是**控制变量的硬性要求**——早停若在不同实验臂上触发时机不同，「训练得更久」
> 就会混进优化方法的增益里。判别命令：`python tools/same_budget.py`。

## 6. 如何运行优化实验

```bash
# 先做控制变量自检：差异键必须与 _expected_diff_ 逐项一致
python tools/diff_config.py configs/baseline.yaml configs/opt_combo.yaml

# 跑实验（Baseline 与优化共用同一个入口，只差一个 --cfg）
python tools/train.py --cfg configs/opt_combo.yaml      # 方案 A：组合式
python tools/train.py --cfg configs/opt_randaug.yaml    # 方案 C：RandAugment 加强增强
python tools/train.py --cfg configs/opt_mix.yaml        # 方案 B：Mixup / CutMix
python tools/train.py --cfg configs/opt_disc.yaml       # 方案 D：差异化学习率

# 预算等价性检查
python tools/same_budget.py --a outputs/logs/baseline_metrics.csv \
                            --b outputs/logs/opt_combo_metrics.csv

# 对比与归因
python tools/compare_runs.py --baseline baseline --opt opt_combo
python tools/predict_compare.py --baseline baseline --opt opt_combo --num 4
```

四个方案的差异键（由 `diff_config.py` 强制校验）：

| 方案 | 配置文件 | 差异键数 | 内容 |
|---|---|---|---|
| B 混合增强 | `opt_mix.yaml` | 3 | mixup_alpha / cutmix_alpha / mixup_prob |
| C 加强增强 | `opt_randaug.yaml` | 8 | RandAugment + 更激进的 RRC + ColorJitter |
| D 差异化 lr | `opt_disc.yaml` | 1 | backbone_lr_scale 0.1 → 0.05 |
| A 组合式 | `opt_combo.yaml` | 12 | B + C + D |

组合消融（进阶）：`opt_abl_a`（=C）/ `opt_abl_b`（=D）/ `opt_abl_ab`（=C+D），
配合 `tools/run_ablation.py` 与 `tools/ablation_analyze.py` 分析叠加/冲突。

## 7. 如何进行验证和测试

```bash
# 纯数据审计（不加载模型）：标签范围、每类计数、train 是否全部来自官方 trainval
python tools/evaluate.py --cfg configs/baseline.yaml --audit

# 在 val 上评价
python tools/evaluate.py --cfg configs/baseline.yaml \
    --ckpt checkpoints/baseline_best.pt --split val

# 在 test 上做**唯一一次**最终评价
python tools/evaluate.py --cfg configs/baseline.yaml \
    --ckpt checkpoints/baseline_best.pt --split test --latency
```

> `test` 不得用于模型选择、调参或早停。最优模型只看验证集 `val_macro_f1`。
> `tools/check_cm.py` 校验混淆矩阵形状与行归一化；`tools/check_backbone_updated.py`
> 证明骨干确实被训练过（防止「只训分类头」）。

## 8. 如何生成曲线、混淆矩阵和 Grad-CAM

```bash
# 五条曲线（train/val loss、val Top-1、val Macro-F1、lr），Baseline 与优化同图
python tools/plot_curves.py \
  --runs baseline=outputs/logs/baseline_metrics.csv \
         opt_combo=outputs/logs/opt_combo_metrics.csv \
  --out outputs/curves/opt_compare.png --title "Baseline vs 优化（方案A 组合式）"

# 混淆矩阵、每类 F1、猫狗块分析
python tools/visualize.py --pred-csv outputs/predictions/baseline_test_preds.csv \
  --classes labels/pet_classes.txt --out-dir outputs --tag baseline --split test

# ≥8 张测试集预测（含 Top-5 与置信度）
python tools/plot_predictions.py --pred-csv outputs/predictions/baseline_test_preds.csv \
  --num 8 --cols 4 --out outputs/predictions/test_top5_baseline_grid8.png --tag baseline

# ≥4 张 Baseline vs 优化「同一批图片」对比（按 image_id inner join，不按行号）
python tools/plot_side_by_side.py \
  --pred-csv-a outputs/predictions/baseline_test_preds.csv \
  --pred-csv-b outputs/predictions/opt_combo_test_preds.csv \
  --name-a baseline --name-b opt_combo --num 4 \
  --out outputs/predictions/compare_baseline_vs_opt_combo_grid4.png

# Grad-CAM（≥4 张，含正确与错误案例）+ 挂载层对比
python tools/gradcam.py --model repvit_m0_9 --ckpt checkpoints/baseline_best.pt \
  --num-classes 37 --classes labels/pet_classes.txt \
  --cases outputs/predictions/cases_test_baseline.csv \
  --out-dir outputs/gradcam --tag baseline --which A

# 训练集以外的实际图片（跨集合实拍）Top-5
python tools/fetch_external_images.py --num 8
python tools/predict_external.py --model repvit_m0_9_pet37 --dir external --num 8
python tools/plot_predictions.py --images "external/*.JPEG" \
  --pred-csv outputs/benchmarks/external_top5_repvit_m0_9_pet37.csv \
  --num 5 --cols 5 --out outputs/predictions/external_top5_pet37_grid5.png
```

**关键结论**：test 集 284 个错误里只有 **9 个（3.17%）是跨物种**（猫↔狗），
其余 275 个都是**同物种内的品种混淆**——说明模型没有把猫认成狗，只是在细粒度品种上分不清。

## 9. 如何执行结构重参数化

```bash
# 验证自行训练的 Pet 37 类模型
python tools/reparam_verify.py --model repvit_m0_9_pet37 \
    --weights checkpoints/baseline_best.pt --out-dir outputs/reparam

# 官方 ImageNet 模型
python tools/reparam_verify.py --model repvit_m0_9 \
    --weights checkpoints/pretrained/repvit_m0_9_distill_300e.pth --out-dir outputs/reparam
```

**实测结果**（`outputs/reparam/repvit_m0_9_pet37_reparam_report.json`）：

| 指标 | 值 |
|---|---|
| `max_abs_err` | **6.53e-06** |
| `mean_abs_err` | 1.67e-06 |
| `top1_identical` | **True**（32/32） |
| BN 模块数 | 108 → **0** |
| 参数量 | 4,732,805 → 4,696,301（净减 36,504） |
| ONNX 图 | BatchNormalization **0**、Conv **103**（未融合 126，少 23） |

> 训练态每个 RepViT Block 含 `RepVGGDW`（3×3 depthwise + 1×1 depthwise 双分支 + BN）
> 与残差分支；推理时把 BN 的参数吸收进卷积核（`W' = W·γ/σ`、`b' = (b−μ)·γ/σ + β`），
> 再把 1×1 分支零填充成 3×3 相加，最后把整块替换成单个 3×3 卷积。
> **必须在 `eval()` 之后融合**，否则用的是 batch 统计量，融合结果与推理态不一致。

## 10. 如何导出不同型号的 ONNX 模型

```bash
# 三个交付模型：官方 M0.9 / 官方 M1.0 / 自训练 Pet-37
python deploy/export_onnx.py --model repvit_m0_9_in1k repvit_m1_0_in1k repvit_m0_9_pet37

# 导出未融合的训练态对照（非交付物）
python deploy/export_onnx.py --model repvit_m0_9_pet37 --no-fuse --out-dir outputs/reparam

# 列出登记表与 ONNX 就位情况
python deploy/model_registry.py
```

**ONNX 文件名一律由 `deploy/model_registry.py` 的 `onnx_path(key)` 决定**，
任何脚本、文档、验收命令都不得硬写 `*.onnx` 文件名。

| registry key | 文件 | 大小 | 图内节点 | 用途 |
|---|---|---|---|---|
| `repvit_m0_9_in1k` | `onnx/repvit_m0_9_in1k.onnx` | 20.36 MB | BN=0, Conv=103 | 基础①官方 M0.9 |
| `repvit_m1_0_in1k` | `onnx/repvit_m1_0_in1k.onnx` | 27.33 MB | BN=0, Conv=103 | 基础②官方 M1.0 |
| `repvit_m0_9_pet37` | `onnx/repvit_m0_9_pet37.onnx` | 18.89 MB | BN=0, Conv=103 | 基础③自训练 Pet-37 |

## 11. 如何运行 ONNX 推理

```bash
# 单张图片，输出 Top-5 类别与置信度
python deploy/infer_onnx.py --model repvit_m0_9_pet37 --image external/beagle__*.JPEG

# PyTorch vs ONNX 一致性（5 项指标 + 判定）
python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --limit 8 \
    --out outputs/metrics/consistency_repvit_m0_9_pet37.json
```

`deploy/infer_onnx.py` 的预处理 / softmax / Top-K **全部独立实现**（PIL 手写
resize+crop，不 import torchvision），与训练侧共用同一份 transform 会让一致性
对比恒等于 0 而掩盖预处理 bug，因此两者必须互相独立。

**实测一致性**：`max|Δlogits| = 5.25e-06`、`mean|Δlogits| = 1.41e-06`、
**Top-1 一致率 100.00%**、Top-5 集合一致率 100.00%。

## 12. 如何完成性能测试

```bash
python deploy/benchmark.py \
  --model repvit_m0_9_in1k --model repvit_m1_0_in1k --model repvit_m0_9_pet37 \
  --warmup 10 --runs 50 --threads 4 --out-dir outputs/benchmarks
```

> 所有性能数字必须同时报告：CPU/GPU 型号、操作系统、推理后端及版本、实际
> Execution Provider、模型型号、输入尺寸、batch size、精度、预热与正式次数、线程数。
> **不同设备、后端、精度下的延迟不可横向比较。**

**实测结果**（ONNX Runtime **CPUExecutionProvider**，FP32，batch=1，224×224，
预热 10 次 + 正式 50 次，`threads_intra=4`，
CPU = 13th Gen Intel Core i7-13650HX，OS = Windows 11 10.0.26200）：

| 模型 | mean (ms) | P50 (ms) | P95 (ms) | 文件大小 |
|---|---|---|---|---|
| `repvit_m0_9_in1k` | 12.74 | 13.60 | 16.66 | 20.36 MB |
| `repvit_m1_0_in1k` | 17.49 | 17.18 | 20.82 | 27.33 MB |
| `repvit_m0_9_pet37` | 12.74 | 12.48 | 16.00 | 18.89 MB |

> M0.9 与 M1.0 参数量相差 37%，但延迟只差约 37%——**延迟并不与 MACs 严格成正比**：
> 小模型受内存带宽与算子启动开销支配，depthwise 卷积的算术强度低，GPU/CPU 都吃不满。

## 13. 一键复现与耗时

```bash
bash tools/run_all.sh          # 全流程：数据 -> 模型 -> 训练 -> 优化 -> 可视化 -> 重参数化 -> ONNX -> 基准
python tools/selfcheck.py      # 全局自检（DoD 逐条）
```

**硬件要求与预计耗时**（本机实测：RTX 4060 Laptop 8GB + i7-13650HX + 15.8 GB RAM）：

| 步骤 | 耗时 |
|---|---|
| 环境安装 | ~10 min |
| 数据准备（含 ImageNet val 解包） | ~8 min |
| 官方模型评价（2 型号 × 1000 张） | ~4 min |
| Baseline 训练（40 epoch） | ~6 min |
| 优化实验（4 组 + 消融 3 组） | ~45 min |
| 可视化 / Grad-CAM | ~3 min |
| 重参数化 / ONNX 导出 / 基准 | ~5 min |
| **合计** | **约 1.5 小时** |

仅 CPU 时：训练改 `--set data.batch_size=32 data.num_workers=4`，
40 epoch 约需 2~3 小时，其余环节耗时不变。

## 14. 代码来源标注

题目要求明确标记每部分代码的来源。完整清单见 [`PROVENANCE.md`](PROVENANCE.md)。

- **官方代码**：`models/repvit_official.py` —— THU-MIG/RepViT `model/repvit.py` 逐字节拷贝
  （Apache-2.0），仅做 3 处必要适配（timm 1.x 导入路径 ×2、删除 `@register_model` 防注册表污染）。
- **第三方代码**：`timm` 提供的 RepViT 实现与预训练权重（Apache-2.0）；
  `torchvision` 的变换与 `OxfordIIITPet`（BSD-3）。
- **自撰代码**：`tools/`、`deploy/`、`utils/`、`datasets/` 下的全部脚本，
  以及 `configs/` 下的全部配置。
- **AI 辅助**：训练循环、评测与部署脚本的初稿由 AI 辅助生成，经本机实测修正
  （修正记录见 `PROGRESS.md` 的决策记录与 `tools/` 内的注释），
  所有结论数字均可在 `outputs/` 中溯源。

## 关于提交的 checkpoint（重要说明）

`checkpoints/baseline_best.pt` 与 `checkpoints/opt_combo_best.pt` 是**推理态交付版**：

- **模型张量与训练产出的完整版逐位相同**（706 个张量，逐张量 `max|Δ| = 0.0`，已实测）；
- 仅去掉了 `optimizer` / `scheduler` / `scaler` / `rng_state` 四个仅用于**断点续训**
  的键（AdamW 的两份动量约占 38 MB/份）；
- 保留 `epoch` / `config` / `seed` / `class_names` / `arch` / `impl` / `best_val_macro_f1` /
  `val_metrics` 等全部选模与溯源元信息，`tools/evaluate.py`、`tools/gradcam.py`、
  `tools/reparam_verify.py`、`deploy/model_registry.py` 全部可正常加载。

**为什么这样交付**：本机 `github.com` 直连被阻断，推送只能走 GitHub 的 Git Data API，
而该 API 对单个 blob 的请求体有上限——57.6 MB 的完整 checkpoint 会被
`422 input too large` 拒绝（实测）。瘦身后 19.4 MB 可正常入库。
需要从 checkpoint 继续训练时，请用本机的 `checkpoints/_full/*.pt`（未入库）。

---

## 许可与致谢

本项目为求职考核的复现任务。RepViT 论文与官方代码版权归原作者所有（Apache-2.0）；
Oxford-IIIT Pet 数据集来自 University of Oxford（CC BY-SA 4.0）；
ImageNet 数据仅用于非商业研究用途。
