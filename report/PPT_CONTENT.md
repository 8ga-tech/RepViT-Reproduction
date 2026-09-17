# 答辩 PPT 内容与证据



15 页（P03 已把试题建议环节 2「RepViT 核心结构」与环节 3「为什么不是标准 ViT」合并为一页，腾出 P11「预测结果」页）；

图表由 tools/update_defense.py 从落盘 CSV/JSON 填充；PDF 由 tools/export_defense_pdf.py 用 PowerPoint COM 导出。



## P01



REPVIT REPRODUCTION

RepViT 轻量图像分类模型
复现、优化与多模型部署

2026 秋 one 团队 AI 算法组考核 · 答辩汇报

RepViT: Revisiting Mobile CNN From ViT Perspective · CVPR 2024 · arXiv:2307.09283
github.com/8ga-tech/RepViT-Reproduction

68.70%

ImageNetV2 固定子集 Top-1

92.34%

Pet-37 test Top-1

7.093e-06

结构重参数化（32 随机输入）max|Δ|

100%

ONNX 一致（入库 5 型号各 12 张）

口径分开：结构重参数化 = 32 个固定随机输入（max|Δ| 7.093e-06，Top-1 32/32）；ONNX 一致性 = 入库的 5 个型号（M0.9-Pet37 / M0.9 / M1.0 / M1.1 / M1.5）各 12 张真实图片（第 13 页，5 份 n=12 落盘），只覆盖这些图片，不代表全部六个 ONNX 型号（M2.3 未入库）。旧的 5.25e-06 未找到配套产物。



备注与验证命令：

封面数字的口径：68.70% = ImageNetV2 matched-frequency 固定子集 1000 张（PyTorch 2.14.0+cu126 / CUDA / batch 64；与 ImageNet-1K 公布值不可比）；92.34% = Pet-37 test 3669 张（Baseline 的 test 只评价 1 次，eval_count=1）；7.093e-06 = 结构重参数化 B4（32 个固定随机输入，outputs/reparam/repvit_m0_9_pet37_reparam_report.json）；100% = PyTorch↔ONNX B1（Pet-37 前 12 张真实图片）。重参数化误差与 ONNX 一致率是两批实验，不并成一句结论。
逐条依据与桶分级见 report/LOGITS_AUDIT_FINDINGS.md 第 1 节。
口径纪律：ImageNetV2 是 Recht et al. 2019 独立重采样的测试集，其准确率与论文/官方公布的 ImageNet-1K 数值不可直接比较（低 10~15 个点是基准性质）。



## P02



六个考核环节全部交付，34 条验收项自检 0 失败

运行 → 迁移 → 优化 → 可视化 → 重参数化 → 部署，每个数字都可在仓库中溯源

01

官方模型评价

5 个型号 · ImageNetV2 1000 张

M0.9 Top-1 68.70%

outputs/pretrained_eval/

02

M0.9 迁移训练

Pet 37 类 · 40 epoch

test Top-1 92.34%

checkpoints/baseline_best.pt

03

控制变量优化

4 方案 + 3 组消融

7 份配置 DIFF PASS

configs/opt_*.yaml

04

可视化与可解释性

五条曲线 · 混淆矩阵 · Grad-CAM

跨物种错误仅 3.56%

outputs/gradcam/

05

结构重参数化

BN 107 → 0 · Conv 126 → 103

max|Δ| 7.093e-06 · 32/32 一致

outputs/reparam/

06

ONNX 多模型部署

6 个 ONNX · ORT CPU

P50 7.68 ~ 34.41 ms

onnx/*.onnx

范围声明：不做第八章拓展任务（INT8 量化 / 知识蒸馏 / 剪枝 / 多随机种子 / 边缘部署 / 核心模块独立实现）

02

口径与边界：PyTorch 与 ONNX 的 Top-1 一致率只测了入库的 5 个型号（M0.9-Pet37 / M0.9 / M1.0 / M1.1 / M1.5）各 12 张（均为 100%）；其余型号（M2.3 未入库）只有导出检查与性能结果，不做同口径一致性声明。重参数化 7.093e-06 来自 32 个固定随机输入，与上面这批 ONNX 实验不是同一批。



备注与验证命令：

六个环节卡片对应试题基础任务 1–6 的交付物；卡片里每行末尾是落盘路径。
口径边界（试题第 16–17 页要求区分：论文 / 官方仓库 / 官方权重实跑 / 自训 Baseline / 优化模型 / PyTorch / ONNX）：卡片 01 是官方权重实跑，02/03/05 是自训结果，05 是 PyTorch 侧重参数化，06 是 ONNX 部署与性能。
ONNX 一致性只有入库的 5 个型号（M0.9-Pet37 / M0.9 / M1.0 / M1.1 / M1.5）各 12 张的落盘 JSON，其余型号不做一致性声明（不含 5.25e-06 旧引用，该值无配套产物）。
test 评价次数：Baseline 的 test 只评价 1 次；4 个优化臂跨重跑累计为 2 次，模型选择始终只用验证集 val_macro_f1，两次结果均落盘可查。



## P03



M0.9 是纯卷积流水线：名字里有 ViT，结构里没有注意力算子

环节 2 + 环节 3 合并页 · 左：M0.9 结构（形状由 forward hook 现测）；右：标准 ViT 与 RepViT 逐项对应

数据来源：outputs/architecture/repvit_m0_9_arch.md/.png（python tools/draw_arch.py）· outputs/reparam/repvit_m0_9_onnx_nodes.json

03

左栏 · 环节 2：M0.9 是一条纯卷积流水线

输入　224×224×3 RGB
Stem　conv1 3×3 s2 → 24ch 112×112；conv2 3×3 s2 → 48ch 56×56
Stage 0　2 × RepViT Block　48ch 56×56
Stage 1　2 × RepViT Block　96ch 28×28
Stage 2　14 × RepViT Block　192ch 14×14（Block 最多）
Stage 3　2 × RepViT Block　384ch 7×7
GAP → 384 维　Head RepVitClassifier（NormLinear）→ 37 维

单个 RepViT Block 内部

空间混合（Token Mixer）：RepVGGDW = 3×3 dw + 1×1 dw
通道混合（Channel Mixer）：1×1 升维 → GELU → 1×1 降维
残差连接；SE 只放在部分 Block（全局池化有真实延迟）
训练态：3×3 dw 分支 + 1×1 dw 分支，各自带 BatchNorm
推理态：BN 折进卷积核后相加 → 单个 3×3 depthwise 卷积

一句话：整条链路只有卷积 / GELU / 池化，没有注意力；名字里的 ViT 说的是宏观架构选择。

右栏 · 环节 3：标准 ViT 与 RepViT 的对应关系

环节 / 标准 ViT / RepViT

空间混合 / Multi-Head Self-Attention / RepVGGDW（3×3 dw + 1×1 dw）

通道混合 / MLP / FFN / Channel Mixer（1×1→GELU）

下采样 / patchify（stride=16 大核） / Early Conv Stem（两组 stride=2）

归纳偏置 / 弱，依赖大数据 / 强（卷积自带局部性 / 平移等变）

ONNX 图内 / MatMul + Softmax / 无注意力算子

实证：onnx/*.onnx 的算子直方图里只有 Conv / Gemm / Add / Mul / Div / Clip / Relu，没有 MatMul + Softmax 这一对注意力算子组合（outputs/reparam/repvit_m0_9_onnx_nodes.json）。

四个关键设计

① 分离 Token / Channel Mixer —— 空间与通道容量可独立调节
② 扩张比降到约 2 并同时加宽 —— 省下的 MACs 换成通道数
③ SE 不做每块都放 —— 全局池化的延迟不可忽略
④ 简单分类头 —— 单个 BN + Linear，省真实延迟
M0.9 实测：20 个 Block，按 Stage 分布 [2, 2, 14, 2]



备注与验证命令：

【本页是合并页】试题第 16–17 页建议结构的「环节 2 RepViT 核心结构」与「环节 3 为什么 RepViT 不是标准 ViT」合并到同一页：左栏 = 环节 2，右栏 = 环节 3。合并原因：PPT 需控制在 15 页内，同时新增第 11 页「预测结果」承载试题第 10 页的三条可视化硬要求（≥8 张测试集预测 / ≥4 张同图对比 / ≥5 张训练集外实际图片）。
题干 15 个建议环节的落点：1 题目与任务完成情况→P01–P02；2 RepViT 核心结构→P03 左栏；3 为什么不是标准 ViT→P03 右栏；4 官方多型号评价→P04；5 数据集和训练流程→P05；6 Baseline 结果→P06；7 优化假设与控制变量→P07；8 曲线和定量结果→P08；9 混淆矩阵与失败案例→P09；10 Grad-CAM 结果→P10；11 结构重参数化→P12；12 ONNX 多模型部署→P13；13 性能比较→P14；14 遇到的问题→P15；15 总结→P15。（P11 为新增页，对应试题第 10 页「必须提供的结果」里的预测可视化三条。）
数据来源：python tools/draw_arch.py → outputs/architecture/repvit_m0_9_arch.png / .md；ONNX 算子直方图见 outputs/reparam/repvit_m0_9_onnx_nodes.json。形状由 forward hook 现测，非论文插图。



## P04



官方权重在 ImageNetV2 固定子集上的实际表现

5 个官方型号 · ImageNetV2 matched-frequency 固定子集 1000 张（1000 类各 1 张，确定性选取）· 224×224 · batch 64 · FP32 · NVIDIA GeForce RTX 4060 Laptop GPU

数据来源：outputs/pretrained_eval/<model>/metrics.json（Top-1/Top-5/参数量/MACs/文件大小）· 清单 datasets/lists/imagenetv2_mf_1000.txt（SHA-256 见 report/IMAGENETV2_PROVENANCE.md）

04

型号 / 参数量 (M) / MACs (G) / 文件大小 (MB) / Top-1 / Top-5

M0.9 / 5.067 / 0.847 / 21.38 / 68.70% / 85.70%

M1.0 / 6.810 / 1.143 / 28.30 / 69.20% / 86.40%

M1.1 / 8.244 / 1.377 / 34.02 / 70.80% / 87.20%

M1.5 / 14.050 / 2.340 / 56.62 / 71.40% / 89.00%

M2.3 / 22.927 / 4.626 / 91.42 / 73.60% / 89.90%

预处理口径必须标

crop_pct = 0.875 → Resize(256) + CenterCrop(224)
crop_pct = 0.95 → Resize(235) + CenterCrop(224)
M0.9：68.70 / 68.30
M1.0：69.20 / 69.40
两套口径差 0.2~1.9 个点

必须声明的口径边界

· ImageNetV2 是独立重采样的测试集，不代表论文完整 ImageNet-1K 验证集结果；本页 Top-1/Top-5 与论文/官方的 ImageNet-1K 公布值不可直接比较。
· 清单固定可复现：1000 类各 1 张、按类目录内文件名 sorted() 取第一个、无随机种子（datasets/lists/imagenetv2_mf_1000.txt）。
· 参数量取「融合后单头」口径；MACs 取 thop「未融合」口径（不乘 2）；官方 iPhone 12 延迟（0.9 / 1.0 ms）与本机结果不可横向比较，延迟口径见第 14 页。

元信息补全：推理后端 PyTorch 2.14.0+cu126（CUDA，逐型号评价）；测试次数：主口径 crop_pct=0.875 每型号评价次数 1 次（1000 张），另有 crop_pct=0.95 口径对照 1 次（仅额外推断，两套口径结果不可比）。



备注与验证命令：

复跑：python tools/run_all_pretrained.py --cfg configs/pretrained_eval.yaml --model repvit_m0_9 repvit_m1_0 repvit_m1_1 repvit_m1_5 repvit_m2_3
元信息：模型 5 个官方型号；输入 224×224；batch 64；硬件 NVIDIA GeForce RTX 4060 Laptop GPU；后端 PyTorch 2.14.0+cu126（CUDA）；精度 FP32；每型号评价次数 1 次（1000 张）。
口径纪律：ImageNetV2 是 Recht et al. 2019 独立重采样的测试集，其准确率与 ImageNet-1K 的公布值不可直接比较（预期低 10~15 个点是基准性质，不是模型退化）；因此本页只报本机实测，不做跨数据集并列或差值。



## P05



两套数据两套标签，三层划分零泄漏

37 类与 1000 类必须分别使用各自的标签文件；混用会让 Top-5 名称整体错乱

Oxford-IIIT Pet 划分（唯一划分脚本，seed=42）

trainval.txt

3680 行

pet_train.txt

2940 行 · 每类 73~80

pet_val.txt

740 行 · 每类恰好 20

test.txt

3669 行

pet_test.txt

3669 行 · 每类 88~100

Baseline 的 test 只评价 1 次

eval_count = 1

官方 CLASS-ID 是 1-based，0-based 标签 = CLASS-ID − 1；类名用 rsplit('_', 1) 取前缀

泄漏检查（三层 + 内容级）

文件名交集

0 / 0 / 0

MD5 交集（防改名）

0 / 0 / 0

孤儿图误入

0

划分内部重复图

0

python datasets/audit_leakage.py

官方数据事实基线（全部实测）

7390

images/ 下的 jpg

3680

trainval.txt 行数

3669

test.txt 行数

12 / 25

猫 / 狗 类别数

1-based

官方 CLASS-ID

官方不存在 train.txt / val.txt，只有 trainval.txt 与 test.txt；train/val 必须自行分层切分

数据来源：outputs/metrics/leakage_check.json · dataset_report.json · pet_label_audit.json · split_audit.json

05



## P06



40 epoch 迁移训练把 test Top-1 做到 92.34%

AdamW · cosine + 3ep warmup · Label Smoothing 0.1 · bf16 autocast · batch 64 · seed 42

测试集（3669 张，只评价一次）

92.34%

Top-1 Accuracy

99.26%

Top-5 Accuracy

92.22%

Macro-F1

验证集（740 张）best 选模依据 val_macro_f1 = 0.9462 @ epoch 34

训练检查

✓ 骨干确实被训练

706 张量

✓ 其中骨干张量

699 个

✓ Baseline 的 test 只评价 1 次

eval_count=1

✓ 权重加载无异常

unexpected=0

1000 类 → 37 类：怎么做、差异在哪

正确做法

timm.create_model(..., num_classes=37,

distillation=False)

保留分类头的 BN 统计量（与类别数无关的迁移先验）

missing = 2（仅换头后的 Linear weight / bias）

禁止做法

model.reset_classifier(37)

会丢掉 head.head.bn 的统计量；distillation=False 时

head_dist 属性被整体删除，后续断言必然失败

数据来源：outputs/metrics/baseline_test.json · backbone_updated.json · weight_load_report.json

06



## P07



四组优化各有假设，控制变量由工具强制校验

Baseline 有两个问题：样本仅 2940 张而参数 473 万（过拟合风险）；主干前期被大学习率推离预训练解

B 混合增强

3 个差异键

Mixup α=0.2 + CutMix α=1.0，每个 batch 都做

假设：混合两张图片与标签，尝试减少对训练样本的记忆

代价：需要更多 epoch 才能收敛

C 加强数据增强

8 个差异键

RandAugment(n=2, m=9) + RRC scale 下限 0.35→0.25

假设：覆盖旋转 / 剪切 / 色调变换，减少对局部纹理的依赖

风险：2940 张下容易过增强

D 差异化学习率

1 个差异键

主干倍率 0.1 → 0.05（有效 lr 1e-4 → 5e-5）

假设：新头快速收敛，让主干更新幅度更小

代价：训练曲线整体右移

A 组合式（基础任务所选项）

12 个差异键

B + C + D 全开

假设：三者机制互补，前段更稳 + 末段更高

风险：正则叠加会延长收敛

控制变量不是靠自觉，而是可执行校验

python tools/diff_config.py configs/baseline.yaml configs/opt_combo.yaml

叶子级比对，实际差异键必须与配置声明的 _expected_diff_ 逐项一致

7 份配置全部 DIFF PASS（3 / 8 / 1 / 12 / 8 / 1 / 9 键）

python tools/same_budget.py → total_iters_equal=True（8 组均为 40 epoch × 45 step）

保持不变

数据划分 · 预训练权重 · 种子

训练预算 · 评价方式 · 选模标准

数据来源：configs/opt_*.yaml 的 _expected_diff_ · outputs/metrics/{check_opt_diff,same_budget}.json

07



## P08



训练曲线：两组都收敛，组合方案未提高测试准确率

RepViT-M0.9 · Pet-37 · 输入 224×224 · batch 64 · NVIDIA GeForce RTX 4060 Laptop GPU · PyTorch 2.14.0+cu126（bf16 autocast）· 曲线共用坐标范围 · test 评价次数 Baseline 1 / 组合 2

来源：outputs/logs/{baseline,opt_combo}_metrics.csv；outputs/metrics/{baseline,opt_combo}_test.json；outputs/logs/*_test_eval_count.json

08

训练损失

验证损失

验证 Top-1 (%)

验证 Macro-F1 (%)

学习率

测试集结果（3669 张）

指标 / Baseline / 组合

Top-1 / 92.34% / 92.12%

Top-5 / 99.26% / 99.59%

Macro-F1 / 92.22% / 92.01%

前几轮提升最快，后段逐渐稳定。
软标签改变 train loss，不能直接横比。
单种子结果不足以判断稳定提升。



备注与验证命令：

python tools/plot_curves.py --runs baseline=outputs/logs/baseline_metrics.csv opt_combo=outputs/logs/opt_combo_metrics.csv --out outputs/curves/opt_compare.png
元信息：模型 RepViT-M0.9（timm 实现，num_classes=37）；输入 224×224；训练 batch 64 / 评价 batch 128；硬件 RTX 4060 Laptop GPU（CUDA 12.6）；推理后端 PyTorch 2.14.0+cu126；精度 bf16 autocast（评价 FP32）；测试集 3669 张，评价次数 Baseline=1、组合=2（落盘 eval_count）。
曲线直接来自 CSV；epoch 从 0 开始。Mixup/CutMix、Label Smoothing 影响 train loss，不直接把训练损失高低当作泛化优劣。
单种子、同一测试集上的小差异不能证明统计显著或稳定提升。



## P09



错在哪里：多数是同物种内的品种混淆

Baseline · Pet-37 test · 3669 张 · PyTorch 2.14.0+cu126 / CUDA / FP32 / 224×224 / 每张只推理一次 · 左：行归一化混淆矩阵；中：每类 F1 柱状图；右：失败案例

来源：outputs/confusion_matrix/{baseline_cm.png,baseline_per_class.csv,baseline_per_class_f1.png,baseline_cat_dog_block.json}；outputs/predictions/case_wrong_baseline_case*.png

09

281 个错误中，10 个跨猫狗物种（3.56%），其余 271 个是同物种品种判断错误。

上面两张是失败案例：相似外形、遮挡和背景都会影响判断；热力图只能用来检查线索，不能仅凭它确定因果。

每类 F1：最低 0.696（Staffordshire Bull Terrier），最高 0.995（Japanese Chin）；37 类 Macro-F1 = 92.22%（数据源 baseline_per_class.csv）。

下一步：补充容易混淆的品种（上面这类）和不同拍摄角度。



备注与验证命令：

python tools/check_cm.py --pred-csv outputs/predictions/baseline_test_preds.csv --classes labels/pet_classes.txt --out-dir outputs/confusion_matrix
左图按真实类别逐行归一化；中间柱状图是 37 类的每类 F1（数据源 outputs/confusion_matrix/baseline_per_class.csv）。
正确的案例见第 11 页（8 张测试集预测）与第 10 页 Grad-CAM；本页只保留 2 个典型失败案例，不回避失败。



## P10



Grad-CAM：同时看判对和判错的图片

Baseline · 挂载 stages[-1].blocks[-1] · 原图 / 热力图 / 叠加图 · PyTorch 2.14.0+cu126 / CUDA / FP32 / 224×224 / 每张只推理一次

来源：outputs/gradcam/gradcam_{correct,wrong}_baseline_*.png；实拍图片的完整 Top-5 见第 11 页（outputs/predictions/external_top5_pet37_grid5.png）

10

正确：Bengal

正确：Russian Blue

失败：Boxer

失败：Egyptian Mau

关注到主体 ≠ 品种一定判断正确。热力图用于检查线索，仍要结合失败案例和完整测试集（3669 张）。



备注与验证命令：

python tools/gradcam.py --help
挂载点与原始图生成配置见 outputs/gradcam/ 下的 JSON。
外部实拍样本来自 ImageNet 跨集合图片，8/8 判对只是小样本观察，不能证明跨域泛化或没有过拟合。



## P11



预测结果：8 张测试集 + 4 组同图对比 + 5 张实拍

Baseline = RepViT-M0.9 Pet-37（test Top-1 92.34%）· 逐张 Top-5 类别与置信度 · 每个面板下方标注该面板自己的推理口径

来源：outputs/predictions/{baseline_test_preds,cases_test_baseline}.csv · baseline_vs_opt_combo_predict_compare.json · outputs/benchmarks/external_top5_repvit_m0_9_pet37.csv · distribution_compare_summary.csv

11

① 同图对比 4 组
Baseline vs 组合方案

按 image_id 配对（不按行号）；全测试集里两模型预测不同的有 142 张。
落盘 JSON 另抽的 4 张两组都判对（both_correct=4、fixed_by_opt=0、broken_by_opt=0）——同图对比要看有差异的样本。
口径：PyTorch / CUDA / FP32 / 224×224 / eval batch 128 / 每张只推理一次。

② 测试集预测 8 张（Top-5 类别 + 置信度）

取测试列表顺序前 8 张（Abyssinian_2 … Abyssinian_207）：8/8 判对，Top-1 置信度 0.508 ~ 0.900；顺序取样，不是挑出来的好例子。
口径：PyTorch / CUDA / FP32 / 224×224 / eval batch 128 / 每张只推理一次。

③ 训练集以外的实拍图片 5 张（Top-5 类别 + 置信度）

Top-1 置信度（图从左到右）：Beagle 0.903、Boxer 0.643、Chihuahua 0.680、Great Pyrenees 0.869、Newfoundland 0.771 —— 5 张都判到对应品种。
最低的 Boxer 只有 0.643：置信度低不等于判错，反过来高置信度也不保证一定可靠。
外部实拍共 8 张，本页展示按文件名排序的前 5 张（试题要求 ≥5 张）；没有 GT（真值），只看 Top-5 与置信度，不报准确率。
数据集 vs 外部图片的分布差异：短边均值 348 → 381 px，亮度均值 0.455 → 0.503，背景复杂度 0.478 → 0.593；分布不同，不能直接比准确率。
口径：ONNX Runtime 1.30.0 CPUExecutionProvider / FP32 / batch 1 / 224×224 / i7-13650HX / 每张只推理一次。



备注与验证命令：

本页三个面板的出图命令：
python tools/plot_predictions.py --pred-csv outputs/predictions/baseline_test_preds.csv --classes labels/pet_classes.txt --num 8 --cols 4 --out outputs/predictions/test_top5_baseline_grid8.png --tag baseline
python tools/plot_side_by_side.py --pred-csv-a outputs/predictions/baseline_test_preds.csv --pred-csv-b outputs/predictions/opt_combo_test_preds.csv --name-a baseline --name-b opt_combo --num 4 --out outputs/predictions/compare_baseline_vs_opt_combo_grid4.png
python tools/predict_external.py --model repvit_m0_9_pet37 --dir external --num 8  → outputs/benchmarks/external_top5_repvit_m0_9_pet37.csv
python tools/plot_predictions.py --images external/*.JPEG --pred-csv outputs/benchmarks/external_top5_repvit_m0_9_pet37.csv --num 5 --out outputs/predictions/external_top5_pet37_grid5.png --tag pet37
数据源：outputs/predictions/baseline_test_preds.csv（3669 行）、baseline_vs_opt_combo_predict_compare.json、pet_compare_ids.json、outputs/benchmarks/external_top5_repvit_m0_9_pet37.csv、distribution_compare_summary.csv。
同图对比按 image_id 做 inner join（不能按行号对齐）；外部图片没有 GT，只看 Top-5 与置信度。



## P12



重参数化：合并分支后，误差仍在设定阈值内

RepViT-M0.9 Pet-37（baseline_best.pt）· 1×3×224×224 · batch 8 · n=32 · i7-13650HX CPU · PyTorch 2.14.0+cu126 / FP32 · 1 次融合前后对比 · seed=20240912

来源：outputs/reparam/repvit_m0_9_pet37_reparam_report.json；本页是 32 个固定随机输入的重参数化实验，与第 13 页 n=12 真实图片的 ONNX 实验分开

12

PyTorch 模块数量

数值验证（32 个固定随机输入） / 结果

最大 |Δlogits| / 7.093e-06

平均 |Δlogits| / 2.031e-06

Top-1 一致 / 32 / 32

Top-5 最小交集 / 5 / 5

阈值 / 判定 / max < 1e-4 / PASS

做法：把 BN 的均值方差折进卷积核，再把 3×3、1×1 与 identity 三条分支的核相加，必须在 eval() 之后做。

看到：32 个固定随机输入 max|Δlogits| = 7.093e-06（< 1e-4 阈值），Top-1 32/32 一致，BN 模块 107 → 0。
还不能说明：这是代数等价的替换 + 已测输入 + 设定阈值，不是位级完全相等，也不代表所有输入都如此。



备注与验证命令：

python tools/reparam_verify.py --model repvit_m0_9_pet37 --weights checkpoints/baseline_best.pt --num-samples 32 --batch-size 8 --seed 20240912 --skip-onnx --out-dir outputs/verification/reparam_pet37
元信息：模型 RepViT-M0.9（num_classes=37）；输入 1×3×224×224；batch 8；n=32 个 torch.randn 固定输入（seed=20240912）；硬件 i7-13650HX / Windows 11；后端 PyTorch 2.14.0+cu126（CPU 侧 FP32 对比）；1 次完整对比。
max_abs_err=7.092952728271484e-06；mean_abs_err=2.030726818702533e-06。32 个输入是 torch.randn，不是真实测试图片。
权重 missing=0 / unexpected=0。BN 模块 107→0；ONNX 图的 BN 节点 24→0 是另一种统计，不可混用。
历史官方 C=1000 报告存在权重键不匹配，不纳入结论；见 report/LOGITS_AUDIT.md。



## P13



PyTorch ↔ ONNX：固定 n=12，比较同一输入张量

原模型 vs 融合后 ONNX · ORT 1.30.0 CPUExecutionProvider · FP32 · batch 1 · 1×3×224×224 · i7-13650HX / Windows 11 · 入库的 5 个型号各 12 张真实图片 · 每张只跑一次

来源：outputs/metrics/consistency_{repvit_m0_9_pet37,repvit_m0_9_in1k,repvit_m1_0_in1k,repvit_m1_1_in1k,repvit_m1_5_in1k}.json（入库 5 个型号各 n=12）

13

模型 / 输入列表 / n / 最大 |Δlogits| / 平均 |Δlogits| / Top-1 / Top-5

M0.9 Pet-37 / pet_test / 12 / 6.199e-06 / 1.501e-06 / 100% / 100%

M0.9 / ImageNetV2 子集 / 12 / 1.383e-05 / 2.334e-06 / 100% / 100%

M1.0 / ImageNetV2 子集 / 12 / 1.335e-05 / 2.215e-06 / 100% / 100%

M1.1 / ImageNetV2 子集 / 12 / 1.860e-05 / 2.219e-06 / 100% / 100%

M1.5 / ImageNetV2 子集 / 12 / 1.144e-05 / 1.876e-06 / 100% / 100%

1  重参数化 7.093e-06（32 个随机输入）是第 12 页的另一批实验，不与本页并成一句结论。
2  部署预处理由 PIL 独立实现；本次对比把同一张量送入两个推理后端，因此这里不比预处理实现。
3  结论仅覆盖每型号这 12 张图片：不能说成完整测试集，也不能说成全部六个模型一致。
4  旧的 5.25e-06 缺少对应完整记录，当前统一引用上表落盘值（4 位有效数字）。

复跑：python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --limit 12
完整命令、固定列表与输出路径已写入本页备注和 report/LOGITS_AUDIT.md。



备注与验证命令：

python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --images datasets/lists/pet_test.txt --limit 12 --out outputs/verification/consistency_repvit_m0_9_pet37_n12.json
元信息：输入 1×3×224×224；batch 1；硬件 i7-13650HX / Windows 11；后端 ONNX Runtime 1.30.0 CPUExecutionProvider；精度 FP32；入库的 5 个型号各 12 张真实图片、每张只跑一次。
Pet-37 与 ImageNet 型号各自按固定列表顺序取前 12 张（图片列表从落盘 JSON 的 images 字段现读：imagenetv2_mf_1000.txt / pet_test.txt）；每张预处理一次，同一 numpy 张量送入两端。
max=6.198883056640625e-06；mean=1.5006899711048998e-06；Top-1/Top-5 集合均 1.0（Pet-37）。
5.25e-06 是未找到对应完整产物的旧引用；旧命令写 limit=8 不能证明该值来自 n=8。当前权重跑 n=8 也不能复现旧值。
不是全测试集一致率，不验证两套预处理独立实现等价，也不能排除所有部署错误；只有入库的 5 个型号各 12 张，不代表全部六个 ONNX 型号（M2.3 因体积未入库，只做导出检查与性能结果）。



## P14



部署怎么选：先看同一机器上的延迟和准确率

型号：M0.9 / M0.9-Pet37 / M1.0 / M1.1 / M1.5 / M2.3 · 输入 1×3×224×224 · batch 1 · i7-13650HX · ORT 1.30.0 CPUExecutionProvider · FP32 · 预热 10 + 正式 50 次 · threads 4

来源：outputs/benchmarks/summary.csv（含 cpu_model / os / ort_version / warmup / runs / threads 全字段）；outputs/pretrained_eval/<model>/metrics.json

14

六个 ONNX 模型的推理耗时 (ms)

ImageNetV2 固定子集（1000 张）：速度与准确率

右图只放 ImageNetV2 固定子集的五个官方型号，看的是「多花多少毫秒换多少点准确率」。更大模型的额外耗时，需要结合使用场景判断。
右图元信息：准确率来自 PyTorch 2.14.0+cu126（64 batch / cuda / fp32）在 ImageNetV2 固定子集（同一 1000 张）上的评价；延迟来自本机 ONNX Runtime CPU（batch 1）。两者口径不同，只做参考。



备注与验证命令：

python deploy/benchmark.py --model repvit_m0_9_in1k --model repvit_m0_9_pet37 --model repvit_m1_0_in1k --model repvit_m1_1_in1k --model repvit_m1_5_in1k --model repvit_m2_3_in1k --warmup 10 --runs 50 --threads 4 --out-dir outputs/verification/benchmarks
元信息：6 个 ONNX 模型（M0.9 / M0.9-Pet37 / M1.0 / M1.1 / M1.5 / M2.3）；输入 1×3×224×224；batch 1；硬件 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0；后端 ONNX Runtime 1.30.0 CPUExecutionProvider；精度 FP32；每个模型预热 10 次 + 正式 50 次（threads=4）。
左图含六个部署模型，只比速度；右图只放 ImageNetV2 固定子集的五个官方型号（准确率来自 PyTorch，延迟来自 ONNX Runtime CPU）。
P50 区间（同一台机器 / ORT CPU）：官方预训练型号 7.68（M0.9） ~ 34.41（M2.3）ms；把自训练 Pet-37 也算进来时最小 7.55 ms。
官方 iPhone 延迟与本机 CPU 延迟不可直接比较。性能数据是历史落盘值，现场新测会有波动。



## P15



复现链路完整、数字可溯源、控制变量有工程化校验

34 条验收项自检 0 失败；代码零写死绝对路径；关键数字与实验口径见 outputs/ 和审计说明

三个最有代表性的问题

01

官方 .pth 与 timm 键名交集为 0（713 vs 389）

实测两条路线权重逐位相同（同进程同权重对比，max|Δlogits| = 0），省去手写键转换器

02

data/provided/ 在 29 页试题 PDF 中从未定义

改判为结构性偏差：改用官方归档构建的 ImageNetV2 固定子集与 Oxford-IIIT Pet，并在 README 与报告中显式声明

03

早停导致训练预算不等价（opt_mix 25 epoch vs baseline 40 epoch，差 37.5%）

处置：统一关闭早停，8 组实验全部 40 epoch 满预算；把预算等价性固化成可执行检查 same_budget.py

结论

① RepViT 确为纯卷积；M0.9 在 ImageNetV2 固定子集上实测 68.70%（与 ImageNet-1K 公布值不可比）

② Pet-37 test Top-1 92.34%（跨物种错误仅 3.56%）；四项优化未显示稳定提升，需多种子实验

③ 重参数化（32 个固定随机输入）max|Δ| = 7.093e-06，Top-1 32/32 一致（与下面 ONNX 实验分列）

④ ONNX 一致性（Pet-37，n=12 真实图片）max|Δ| = 6.199e-06，Top-1/Top-5 集合一致率 100%

后续计划

INT8 量化 / 知识蒸馏（M2.3 → M0.9）

集显后端（本机无集显，需换设备）

置信度校准（实测温度 T = 0.615）

七类鲁棒性扰动已跑，可扩展 severity 级

github.com/8ga-tech/RepViT-Reproduction

感谢聆听 · 敬请提问

