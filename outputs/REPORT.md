# RepViT-M0.9 迁移训练、优化与 ONNX 多模型部署（项目报告）

> 本文件由 `tools/make_report_assets.py` 生成骨架，表格为脚本填充，正文需人工撰写。
> 正文 12~20 页；表格数据源见 `outputs/report_assets/`。

## 1. 任务背景与复现范围

<!-- HUMAN: 任务背景与复现范围正文，9字以上 -->



## 2. RepViT 论文核心思路

<!-- HUMAN: RepViT 论文核心思路正文，13字以上 -->



## 3. RepViT-M0.9 结构

<!-- HUMAN: RepViT-M0.9 结构正文，14字以上 -->

![repvit_m0_9_arch](outputs/architecture/repvit_m0_9_arch.png)

- 数据/材料：`outputs/architecture/repvit_m0_9_arch.md`

- 数据/材料：`outputs/reparam/repvit_m0_9_onnx_nodes.json`

## 4. 官方预训练模型评价

<!-- HUMAN: 官方预训练模型评价正文，9字以上 -->

- 数据/材料：`outputs/pretrained_eval/repvit_m0_9/metrics.json`

- 数据/材料：`outputs/pretrained_eval/repvit_m0_9/latency.json`

- 数据/材料：`outputs/pretrained_eval/repvit_m0_9/top5_samples.json`

- 数据/材料：`outputs/pretrained_eval/summary.csv`

## 5. 多型号规模和性能比较

<!-- HUMAN: 多型号规模和性能比较正文，10字以上 -->

![params_vs_acc](outputs/advanced/params_vs_acc.png)

![latency_vs_acc](outputs/advanced/latency_vs_acc.png)

![macs_vs_acc](outputs/advanced/macs_vs_acc.png)

- 数据/材料：`outputs/benchmarks/family_summary.csv`

- 数据/材料：`outputs/advanced/pareto_summary.json`

## 6. 数据集与数据划分

<!-- HUMAN: 数据集与数据划分正文，8字以上 -->

- 数据/材料：`datasets/lists/pet_train.txt`

- 数据/材料：`datasets/lists/pet_val.txt`

- 数据/材料：`datasets/lists/pet_test.txt`

- 数据/材料：`labels/pet_classes.txt`

- 数据/材料：`labels/imagenet_classes.txt`

- 数据/材料：`outputs/metrics/leakage_check.json`

## 7. Baseline 迁移训练

<!-- HUMAN: Baseline 迁移训练正文，13字以上 -->

- 数据/材料：`outputs/logs/baseline_metrics.csv`

- 数据/材料：`outputs/logs/baseline_config_effective.yaml`

- 数据/材料：`outputs/metrics/baseline_test.json`

- 数据/材料：`outputs/metrics/weight_load_report.json`

## 8. 优化方法与实验假设

<!-- HUMAN: 优化方法与实验假设正文，9字以上 -->

- 数据/材料：`outputs/logs/opt_combo_config_effective.yaml`

- 数据/材料：`outputs/logs/opt_randaug_config_effective.yaml`

- 数据/材料：`outputs/logs/opt_compare_summary.csv`

## 9. 定量结果

<!-- HUMAN: 定量结果正文，4字以上 -->

- 数据/材料：`outputs/report_assets/table_results.csv`

- 数据/材料：`outputs/advanced/ablation_summary.csv`

- 数据/材料：`outputs/metrics/ablation.csv`

| 口径 | 数据集 | 类别数 | Top-1 | Top-5 | Macro-F1 | 参数量 | MACs | 文件大小 | 延迟 | 依据 |
|---|---|---|---|---|---|---|---|---|---|---|
| 论文公布 | ImageNet-1K | 1000 | 78.70% | — | — | 5.1000 M (融合后单头) | 0.800 G (MACs 口径(未乘 2)) | — | 0.9 ms (iPhone 12/CoreML/bs=1，不可与本地 ORT CPU 对比) | iPhone 12 + CoreML，四舍五入到 0.1ms |
| 官方仓库公布 | ImageNet-1K | 1000 | 78.70% | 79.10% | — | 5.1000 M (融合后单头（flops.py: replace_batchnorm 后取值）) | 0.800 G (MACs 口径(未乘 2)) | — | 0.9 ms (同论文口径) | THU-MIG README Model Zoo 表 |
| 官方权重实测 | — | 1000 | 68.70% | 85.70% | — | 5.0671 M (未标注口径) | 0.847 G (MACs 口径(未乘 2)) | 21.38 MB | mean 8.41ms | 本机 PyTorch 2.14.0+cu126（CUDA）实跑；**该文件的 top1/top5 已是百分数，不得再乘 100** |
| 自训练 Baseline | — | 37 | 92.34% | 99.26% | 92.22% | 4.7328 M (未标注口径) | — | — | — | Pet-37 test split（baseline） |
| 自优化模型 | — | 37 | 92.12% | 99.59% | 92.01% | 4.7328 M (未标注口径) | — | — | — | Pet-37 test split（组合方案 opt_combo） |
| PyTorch 推理 | — | — | — | — | — | — | — | — | mean 8.41ms / P50 8.26 / P95 9.93 | batch=1, fp32, 本机 CUDA（PyTorch 2.14.0+cu126）, threads=4 |
| ONNX 部署 | — | — | — | — | — | — | — | 91.90 MB | mean 34.39ms / P50 34.41 / P95 35.48 | ORT CPUExecutionProvider |


## 10. 曲线和混淆矩阵分析

<!-- HUMAN: 曲线和混淆矩阵分析正文，9字以上 -->

![opt_compare](outputs/curves/opt_compare.png)

![baseline_curves](outputs/curves/baseline_curves.png)

![baseline_cm](outputs/confusion_matrix/baseline_cm.png)

![baseline_per_class_f1](outputs/confusion_matrix/baseline_per_class_f1.png)

- 数据/材料：`outputs/confusion_matrix/baseline_cat_dog_block.json`

## 11. Grad-CAM 与失败案例

<!-- HUMAN: Grad-CAM 与失败案例正文，14字以上 -->

![gradcam_correct_baseline](outputs/gradcam/gradcam_correct_baseline.png)

![gradcam_wrong_baseline](outputs/gradcam/gradcam_wrong_baseline.png)

![gradcam_layer_compare_baseline](outputs/gradcam/gradcam_layer_compare_baseline.png)

![test_top5_baseline_grid8](outputs/predictions/test_top5_baseline_grid8.png)

- 数据/材料：`outputs/metrics/gradcam_meta.json`

## 12. 结构重参数化

<!-- HUMAN: 结构重参数化正文，6字以上 -->

- 数据/材料：`outputs/reparam/repvit_m0_9_pet37_reparam_report.json`

- 数据/材料：`outputs/reparam/repvit_m0_9_pet37_onnx_nodes.json`

- 数据/材料：`outputs/advanced/block_22_fusion_report.txt`

## 13. ONNX 多模型部署

<!-- HUMAN: ONNX 多模型部署正文，10字以上 -->

- 数据/材料：`onnx/repvit_m0_9_in1k.onnx`

- 数据/材料：`onnx/repvit_m1_0_in1k.onnx`

- 数据/材料：`onnx/repvit_m0_9_pet37.onnx`

- 数据/材料：`outputs/metrics/consistency_repvit_m0_9_pet37.json`

- 数据/材料：`outputs/metrics/consistency_repvit_m0_9_in1k.json`

- 数据/材料：`outputs/metrics/consistency_repvit_m1_0_in1k.json`

- 数据/材料：`outputs/metrics/consistency_repvit_m1_1_in1k.json`

- 数据/材料：`outputs/metrics/consistency_repvit_m1_5_in1k.json`

## 14. 性能测试

<!-- HUMAN: 性能测试正文，4字以上 -->

- 数据/材料：`outputs/report_assets/table_benchmark_meta.csv`

- 数据/材料：`outputs/benchmarks/summary.csv`

- 数据/材料：`outputs/metrics/onnx_io_nodes.json`

- 数据/材料：`outputs/metrics/bench.jsonl`

| model | input_size | batch_size | precision | backend | hardware | ort_version | runs | file_size_mb | io_nodes | mean_ms | p50_ms | p95_ms | threads_intra |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| repvit_m0_9_in1k | 224 | 1 | FP32 | onnxruntime(CPUExecutionProvider) | 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0 | 1.30.0 | warmup=10, runs=50 | 20.36 | in=input[1, 3, 224, 224] out=logits[1, 1000] | 7.698 | 7.684 | 8.488 | 4 |
| repvit_m1_0_in1k | 224 | 1 | FP32 | onnxruntime(CPUExecutionProvider) | 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0 | 1.30.0 | warmup=10, runs=50 | 27.33 | in=input[1, 3, 224, 224] out=logits[1, 1000] | 10.422 | 10.404 | 11.37 | 4 |
| repvit_m0_9_pet37 | 224 | 1 | FP32 | onnxruntime(CPUExecutionProvider) | 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0 | 1.30.0 | warmup=10, runs=50 | 18.89 | in=input[1, 3, 224, 224] out=logits[1, 37] | 7.635 | 7.555 | 8.16 | 4 |
| repvit_m1_1_in1k | 224 | 1 | FP32 | onnxruntime(CPUExecutionProvider) | 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0 | 1.30.0 | warmup=10, runs=50 | 33.06 | in=input[1, 3, 224, 224] out=logits[1, 1000] | 11.026 | 10.929 | 11.993 | 4 |
| repvit_m1_5_in1k | 224 | 1 | FP32 | onnxruntime(CPUExecutionProvider) | 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0 | 1.30.0 | warmup=10, runs=50 | 56.35 | in=input[1, 3, 224, 224] out=logits[1, 1000] | 18.559 | 18.437 | 19.756 | 4 |
| repvit_m2_3_in1k | 224 | 1 | FP32 | onnxruntime(CPUExecutionProvider) | 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0 | 1.30.0 | warmup=10, runs=50 | 91.9 | in=input[1, 3, 224, 224] out=logits[1, 1000] | 34.392 | 34.41 | 35.48 | 4 |


## 15. 遇到的问题和解决方法

<!-- HUMAN: 遇到的问题和解决方法正文，10字以上 -->

- 数据/材料：`outputs/metrics/selfcheck_report.json`

- 数据/材料：`outputs/env_snapshot.json`

## 16. 总结与后续计划

<!-- HUMAN: 总结与后续计划正文，7字以上 -->

- 数据/材料：`outputs/metrics/dataset_report.json`

- 数据/材料：`outputs/verification/consistency_repvit_m0_9_pet37_n12.json`
