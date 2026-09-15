# logits 误差口径审计（2026-09-16）

## 当前答辩采用的两项证据

| 实验 | 模型与输入 | max / mean 绝对误差 | 一致性与判定 | 原始文件 |
|---|---|---|---|---|
| PyTorch–ONNX | Pet-37 baseline；pet_test.txt 按顺序前 12 张真实图片；每张独立预处理一次后，把同一张量送入两端；CPU FP32，batch=1，224×224 | 6.198883056640625e-06 / 1.5006899711048998e-06；展示 6.199e-06 / 1.501e-06 | Top-1、Top-5 集合均 12/12；max<1e-3，Top-1≥0.99 | outputs/metrics/consistency_repvit_m0_9_pet37.json |
| PyTorch 重参数化 | Pet-37 baseline；timm 单头、distillation=False；32 个固定随机输入，seed=20240912，batch=8，224×224，CPU FP32；eval 后深拷贝再 fuse | 7.092952728271484e-06 / 2.030726818702533e-06；展示 7.093e-06 / 2.031e-06 | Top-1 32/32，Top-5 最小交集5；max<1e-4；BN模块107→0 | outputs/reparam/repvit_m0_9_pet37_reparam_report.json |

两项均已实际复跑，max/mean 与上述原始 JSON 完全一致；复跑副本位于 outputs/verification/。
结果覆盖已测输入与阈值，不是位级完全相等，也不代表所有输入、所有后端或全测试集都一致。

## 旧引用如何处理

- README 的 5.25e-06 / 1.41e-06，以及旧 PPT_CONTENT 的 5.245e-06 / 1.414e-06，没有找到配套完整实验产物。旧命令写了 limit=8，但仅凭这一点不能证明数值来自 n=8。当前权重跑 n=8 得到 max=6.198883056640625e-06，不能复现旧值。旧值仅作为修订记录，不再用作实验结论。
- README/PPT_CONTENT 曾出现 6.53e-06、6.527e-06 等旧 Pet-37 重参数化值。引用同一个 outputs/reparam 文件的地方统一为 7.093e-06；不能把不同日期/权重/输入下的值视为相同实验。
- “共用 transform 会使 logits 误差恒为0”不成立。当前 compare_torch_onnx.py 使用同一张量，隔离预处理变量，只测模型/后端差异；独立实现的部署预处理仍需要另外与训练侧核对。
- 原先“六个 ONNX 模型一致率100%”范围过大：仓库只有三份 consistency JSON，分别是 Pet-37 M0.9、ImageNet M0.9/M1.0，各 n=12。其余三型号的导出或性能结果不能代替同口径一致性验证。

## 保持独立的其他实验

| 证据位置 | 它实际比较什么 | 处理方式 |
|---|---|---|
| outputs/metrics/consistency_repvit_m0_9_in1k.json | ImageNet固定子集前12张，PyTorch–ONNX；max=1.71661376953125e-05，mean=2.360081756099438e-06 | 保留独立型号与标签口径 |
| outputs/metrics/consistency_repvit_m1_0_in1k.json | ImageNet固定子集前12张，PyTorch–ONNX；max=1.811981201171875e-05，mean=2.4635197632960626e-06 | 不与Pet-37合并 |
| outputs/pretrained_eval/*/metrics.json 中 fuse_max_abs_logits_diff | 官方权重评价流程里的融合前后探针；M0.9/M1.0/M1.1/M1.5/M2.3分别约3.073e-05/3.290e-05/2.217e-05/2.587e-05/5.925e-05 | 型号、输入、设备取各文件与 eval_pretrained.py；不替换Pet-37的32输入实验 |
| outputs/benchmarks/export_*.json 与 deploy/export_onnx.py | 导出检查及单个随机输入探针；导出脚本的随机输入未固定seed | 不充当n=12真实图片一致性结果 |
| outputs/advanced/repvit_m0_9_pet37_reparam_report.json | reparam_deep.py 的历史结构探针；seed=0、2个随机输入；构造时未显式关蒸馏头，strict=False允许部分加载；数值max=2.9802322387695312e-06 | 保留历史记录，不能作为当前单头baseline的独立复现；108个BN与当前107不同 |
| outputs/reparam/repvit_m0_9_reparam_report.json | 历史C=1000报告；官方checkpoint装入timm结构，n_missing=605、n_unexpected=713；max=4.991888999938965e-07 | 无效权重验证，不作为官方预训练模型证据。旧文档2.265e-06也缺配套有效产物，移出结论 |
| outputs/metrics/probe_route.json | 两条权重路线的特征/主头/蒸馏平均logits比较，max=0 | 路线等价探针，不是ONNX或融合实验 |
| outputs/metrics/smoke_phase0.json、outputs/logs/*、结构txt | 早期冒烟、训练/评价日志与结构统计 | 原始记录保留；只在注明具体输入/型号/阶段后引用 |

PyTorch BN模块数与导出图BatchNormalization节点数也不同：当前Pet-37分别为107→0与24→0；导出器已预先折叠部分BN。

## 复跑与检索命令

从仓库根目录执行；权重、ONNX和数据按README准备。复跑写入单独目录，避免覆盖原证据。

    python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --images datasets/lists/pet_test.txt --limit 12 --out outputs/verification/consistency_repvit_m0_9_pet37_n12.json
    python tools/reparam_verify.py --model repvit_m0_9_pet37 --weights checkpoints/baseline_best.pt --num-samples 32 --batch-size 8 --seed 20240912 --skip-onnx --out-dir outputs/verification/reparam_pet37
    python tools/selfcheck.py --only rep.verify onnx.consistency onnx.realimg --strict --json outputs/verification/selfcheck_logits.json
    python tools/audit_logits_references.py
    python tools/update_defense.py

若要重新导出重参数化对照图，去掉 --skip-onnx。官方ImageNet模型使用登记键 repvit_m0_9_in1k，使实现与权重匹配；不要将官方键名直接装入timm结构。

扫描清单：outputs/verification/logits_reference_inventory.csv，覆盖Git跟踪的文本、代码、JSON、日志、SVG，以及PPTX/DOCX/PDF正文与PPT备注。原始实验产物不做数字替换；旧SVG演示源保留为历史素材，不作为答辩口径。

## 试题与答辩图表对应

- 试题p9–11、p16–17：第9页展示五个可编辑原生曲线（train loss、val loss、val Top-1、val Macro-F1、lr）；直接读取Baseline/opt_combo CSV，同图同尺度，并给测试指标表。
- 第10页：归一化混淆矩阵与两个失败案例；第11页：2正确+2失败Grad-CAM。完整预测、外部图Top-5仍在outputs中，备注给定位。
- 第12–13页：随机输入重参数化与真实图片ONNX各自列输入、样本数、阈值和命令；删去“完全等价”“排除九类错误”等超出证据的断言。
- 第14页：原生P50/P95柱状图与五个ImageNet型号的延迟–准确率散点图；标注硬件、版本、实际EP、精度、输入、batch、次数、线程。Pet-37准确率不放入ImageNet家族比较。
- 保持15页；正文采用“怎么做、看到了什么、还不能说明什么”的直白表达，完整验证命令放备注。
