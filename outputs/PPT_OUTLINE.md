# 答辩 PPT 大纲（10~15 页，性能页必须带元信息水印）

## 第 1 页 · 题目与任务完成情况

<!-- HUMAN: 3~5 个要点，结论先行 -->


## 第 2 页 · 完成度总览（六个考核环节）

<!-- HUMAN: 3~5 个要点，结论先行 -->

- 配图：`outputs/metrics/selfcheck_report.json`

## 第 3 页 · RepViT 核心结构（含为什么不是标准 ViT）

<!-- HUMAN: 3~5 个要点，结论先行 -->

- 配图：`outputs/architecture/repvit_m0_9_arch.png`
- 配图：`outputs/architecture/repvit_m0_9_arch.md`
- 配图：`outputs/reparam/repvit_m0_9_onnx_nodes.json`

## 第 4 页 · 官方多型号评价

<!-- HUMAN: 3~5 个要点，结论先行 -->

- 配图：`outputs/pretrained_eval/summary.csv`
- 配图：`outputs/pretrained_eval/repvit_m0_9/metrics.json`
- 配图：`outputs/benchmarks/family_summary.csv`

## 第 5 页 · 数据集和训练流程

<!-- HUMAN: 3~5 个要点，结论先行 -->

- 配图：`outputs/metrics/leakage_check.json`
- 配图：`outputs/metrics/dataset_report.json`
- 配图：`outputs/metrics/pet_label_audit.json`
- 配图：`outputs/metrics/split_audit.json`

## 第 6 页 · Baseline 结果

<!-- HUMAN: 3~5 个要点，结论先行 -->

- 配图：`outputs/metrics/baseline_test.json`
- 配图：`outputs/metrics/weight_load_report.json`
- 配图：`outputs/metrics/backbone_updated.json`
- 配图：`outputs/logs/baseline_metrics.csv`

## 第 7 页 · 优化假设与控制变量

<!-- HUMAN: 3~5 个要点，结论先行 -->

- 配图：`outputs/logs/opt_combo_config_effective.yaml`
- 配图：`outputs/logs/opt_randaug_config_effective.yaml`
- 配图：`outputs/logs/opt_compare_summary.csv`

## 第 8 页 · 曲线和定量结果

<!-- HUMAN: 3~5 个要点，结论先行 -->

- 配图：`outputs/curves/opt_compare.png`
- 配图：`outputs/logs/baseline_metrics.csv`
- 配图：`outputs/logs/opt_combo_metrics.csv`
- 配图：`outputs/metrics/opt_combo_test.json`

## 第 9 页 · 混淆矩阵与失败案例

<!-- HUMAN: 3~5 个要点，结论先行 -->

- 配图：`outputs/confusion_matrix/baseline_cm.png`
- 配图：`outputs/confusion_matrix/baseline_per_class_f1.png`
- 配图：`outputs/predictions/case_wrong_baseline_case01.png`
- 配图：`outputs/predictions/case_wrong_baseline_case02.png`

## 第 10 页 · Grad-CAM 结果

<!-- HUMAN: 3~5 个要点，结论先行 -->

- 配图：`outputs/gradcam/gradcam_correct_baseline_Bengal_30_correct.png`
- 配图：`outputs/gradcam/gradcam_correct_baseline_Russian_Blue_205_correct.png`
- 配图：`outputs/gradcam/gradcam_wrong_baseline_boxer_2_wrong.png`
- 配图：`outputs/gradcam/gradcam_wrong_baseline_Egyptian_Mau_204_wrong.png`

## 第 11 页 · 预测结果（8 张测试集 + 4 组同图对比 + 5 张实拍）

<!-- HUMAN: 3~5 个要点，结论先行 -->

- 配图：`outputs/predictions/test_top5_baseline_grid8.png`
- 配图：`outputs/predictions/compare_baseline_vs_opt_combo_grid4.png`
- 配图：`outputs/predictions/external_top5_pet37_grid5.png`
- 配图：`outputs/predictions/baseline_vs_opt_combo_predict_compare.json`

## 第 12 页 · 结构重参数化

<!-- HUMAN: 3~5 个要点，结论先行 -->

- 配图：`outputs/reparam/repvit_m0_9_pet37_reparam_report.json`
- 配图：`outputs/reparam/repvit_m0_9_onnx_nodes.json`

## 第 13 页 · ONNX 多模型部署

<!-- HUMAN: 3~5 个要点，结论先行 -->

- 配图：`outputs/metrics/consistency_repvit_m0_9_pet37.json`
- 配图：`outputs/metrics/consistency_repvit_m0_9_in1k.json`
- 配图：`outputs/metrics/consistency_repvit_m1_0_in1k.json`
- 配图：`outputs/metrics/consistency_repvit_m1_1_in1k.json`
- 配图：`outputs/metrics/consistency_repvit_m1_5_in1k.json`

## 第 14 页 · 性能比较

<!-- HUMAN: 3~5 个要点，结论先行 -->

- 配图：`outputs/benchmarks/summary.csv`
- 配图：`outputs/pretrained_eval/summary.csv`
- 配图：`outputs/metrics/onnx_io_nodes.json`

## 第 15 页 · 遇到的问题与总结

<!-- HUMAN: 3~5 个要点，结论先行 -->

- 配图：`outputs/metrics/selfcheck_report.json`
- 配图：`outputs/metrics/dataset_report.json`
