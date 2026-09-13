# RepViT-M0.9 迁移训练、优化与 ONNX 多模型部署（项目报告）

> 本文件由 `tools/make_report_assets.py` 生成骨架，表格为脚本填充，正文需人工撰写。
> 正文 12~20 页；表格数据源见 `outputs/report_assets/`。

## 1. 任务背景与复现范围

<!-- HUMAN: 任务背景与复现范围正文，9字以上 -->



## 2. RepViT 论文核心思路

<!-- HUMAN: RepViT 论文核心思路正文，13字以上 -->

![roadmap_incremental](figures/roadmap_incremental.png)

## 3. RepViT-M0.9 结构

<!-- HUMAN: RepViT-M0.9 结构正文，14字以上 -->

![repvit_m0_9_arch](outputs/architecture/repvit_m0_9_arch.png)

- 数据/材料：`outputs/architecture/repvit_m0_9_arch.md`

![m0_9_block](figures/m0_9_block.png)

![m0_9_cfg_table](figures/m0_9_cfg_table.png)

## 4. 官方预训练模型评价

<!-- HUMAN: 官方预训练模型评价正文，9字以上 -->

- 数据/材料：`outputs/pretrained_eval/repvit_m0_9_in1k/metrics.json`

![official_eval_bar](figures/official_eval_bar.png)

## 5. 多型号规模和性能比较

<!-- HUMAN: 多型号规模和性能比较正文，10字以上 -->

![params_vs_acc](figures/params_vs_acc.png)

![latency_vs_acc](figures/latency_vs_acc.png)

## 6. 数据集与数据划分

<!-- HUMAN: 数据集与数据划分正文，8字以上 -->

- 数据/材料：`datasets/lists/pet_train.txt`

- 数据/材料：`datasets/lists/pet_val.txt`

- 数据/材料：`datasets/lists/pet_test.txt`

![pet37_class_hist](figures/pet37_class_hist.png)

## 7. Baseline 迁移训练

<!-- HUMAN: Baseline 迁移训练正文，13字以上 -->

- 数据/材料：`outputs/logs/B0_baseline_metrics.csv`

- 数据/材料：`outputs/logs/B0_baseline_config_effective.yaml`

## 8. 优化方法与实验假设

<!-- HUMAN: 优化方法与实验假设正文，9字以上 -->

- 数据/材料：`outputs/logs/O1_randaug_config_effective.yaml`

## 9. 定量结果

<!-- HUMAN: 定量结果正文，4字以上 -->

- 数据/材料：`outputs/report_assets/table_results.csv`

| 口径 | 数据集 | 类别数 | Top-1 | Top-5 | Macro-F1 | 参数量 | MACs | 文件大小 | 延迟 | 依据 |
|---|---|---|---|---|---|---|---|---|---|---|
| 论文公布 | ImageNet-1K | 1000 | 78.70% | — | — | 5.1000 M (融合后单头) | 0.800 G (MACs 口径(未乘 2)) | — | 0.9 ms (iPhone 12/CoreML/bs=1，不可与本地 ORT CPU 对比) | iPhone 12 + CoreML，四舍五入到 0.1ms |
| 官方仓库公布 | ImageNet-1K | 1000 | 78.70% | 79.10% | — | 5.1000 M (融合后单头（flops.py: replace_batchnorm 后取值）) | 0.800 G (MACs 口径(未乘 2)) | — | 0.9 ms (同论文口径) | THU-MIG README Model Zoo 表 |
| 官方权重实测 | — | — | — | — | — | — | — | — | — | 本机 ORT/PyTorch，须写明子集大小；**该文件的 top1/top5 已是百分数，不得再乘 100** |
| 自训练 Baseline | — | — | — | — | — | — | — | — | — | Pet-37 test split |
| 自优化模型 | — | — | — | — | — | — | — | — | — | Pet-37 test split |
| PyTorch 推理 | — | — | — | — | — | — | — | — | — | batch=1, fp32, 本机 CPU |
| ONNX 部署 | — | — | — | — | — | — | — | 18.89 MB | mean 7.20ms / P50 7.12 / P95 7.69 | ORT CPUExecutionProvider |


## 10. 曲线和混淆矩阵分析

<!-- HUMAN: 曲线和混淆矩阵分析正文，9字以上 -->

![training_curves_all](outputs/curves/training_curves_all.png)

![O1_randaug_test_norm](outputs/confusion_matrix/O1_randaug_test_norm.png)

![per_class_f1_O1](outputs/confusion_matrix/per_class_f1_O1.png)

## 11. Grad-CAM 与失败案例

<!-- HUMAN: Grad-CAM 与失败案例正文，14字以上 -->

![gradcam_compare_O1](outputs/gradcam/gradcam_compare_O1.png)

![top5_failures_O1](outputs/predictions/top5_failures_O1.png)

## 12. 结构重参数化

<!-- HUMAN: 结构重参数化正文，6字以上 -->

- 数据/材料：`outputs/reparam/repvit_m0_9_reparam_report.json`

![reparam_graph_nodes](figures/reparam_graph_nodes.png)

## 13. ONNX 多模型部署

<!-- HUMAN: ONNX 多模型部署正文，10字以上 -->

![onnx_io_nodes](figures/onnx_io_nodes.png)

## 14. 性能测试

<!-- HUMAN: 性能测试正文，4字以上 -->

- 数据/材料：`outputs/report_assets/table_benchmark_meta.csv`

![latency_bar_three_models](figures/latency_bar_three_models.png)

| model | input_size | batch_size | precision | backend | hardware | ort_version | runs | file_size_mb | io_nodes | mean_ms | p50_ms | p95_ms | threads_intra |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| repvit_m0_9_in1k | 224 | 1 | FP32 | CPUExecutionProvider | 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0 | 1.30.0 | warmup=10, runs=50 | 20.36 | in=NoneNone out=NoneNone | 7.446 | 7.373 | 8.002 | 4 |
| repvit_m1_0_in1k | 224 | 1 | FP32 | CPUExecutionProvider | 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0 | 1.30.0 | warmup=10, runs=50 | 27.33 | in=NoneNone out=NoneNone | 9.202 | 9.147 | 9.76 | 4 |
| repvit_m0_9_pet37 | 224 | 1 | FP32 | CPUExecutionProvider | 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0 | 1.30.0 | warmup=10, runs=50 | 18.89 | in=NoneNone out=NoneNone | 7.2 | 7.122 | 7.692 | 4 |
| repvit_m1_1_in1k | 224 | 1 | FP32 | CPUExecutionProvider | 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0 | 1.30.0 | warmup=10, runs=50 | 33.06 | in=NoneNone out=NoneNone | 10.407 | 10.234 | 11.241 | 4 |
| repvit_m1_5_in1k | 224 | 1 | FP32 | CPUExecutionProvider | 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0 | 1.30.0 | warmup=10, runs=50 | 56.35 | in=NoneNone out=NoneNone | 17.779 | 17.647 | 18.926 | 4 |
| repvit_m2_3_in1k | 224 | 1 | FP32 | CPUExecutionProvider | 13th Gen Intel(R) Core(TM) i7-13650HX / Windows-11-10.0.26200-SP0 | 1.30.0 | warmup=10, runs=50 | 91.9 | in=NoneNone out=NoneNone | 32.65 | 32.687 | 33.322 | 4 |


## 15. 遇到的问题和解决方法

<!-- HUMAN: 遇到的问题和解决方法正文，10字以上 -->



## 16. 总结与后续计划

<!-- HUMAN: 总结与后续计划正文，7字以上 -->


