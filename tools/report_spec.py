# tools/report_spec.py
"""Source: Self-written
报告/PPT 的契约常量。make_report_assets.py 与 check_report_assets.py 共用本文件，
保证"要求什么"与"检查什么"永不漂移。

**路径口径（2026-09 修订）**：本文件的每条产物路径都必须指向**仓库里真实存在的文件**，
由 `python tools/check_report_assets.py` 逐条断言。

早期版本这里写的是一套"计划命名"（`figures/*.png` 目录 + `B0_*` / `O1_*` 实验前缀），
而实际流水线落盘在 `outputs/{architecture,curves,confusion_matrix,gradcam,predictions,advanced}/`
与 `outputs/logs/{baseline,opt_*}_*`。那套命名从未产出过任何文件，导致验收入口
`python tools/check_report_assets.py` 恒报 `FAIL=36`（现场跑一次就与「无法复现实验结果」正面冲突）。
本次把三张表全部改成现行产物路径；`figures/` 已废弃，不再引用。
"""
from pathlib import Path

# 报告 16 节（题目第 15-16 页）与其所需产物；产物路径相对仓库根，全部必须存在。
REPORT_SECTIONS = [
    ("1", "任务背景与复现范围", []),
    # 第 2 节是文字论述（论文增量现代化设计的四条思路），报告与 PPT 都不依赖位图产物。
    ("2", "RepViT 论文核心思路", []),
    ("3", "RepViT-M0.9 结构", ["outputs/architecture/repvit_m0_9_arch.png",
                          "outputs/architecture/repvit_m0_9_arch.md",
                          "outputs/reparam/repvit_m0_9_onnx_nodes.json"]),
    ("4", "官方预训练模型评价", ["outputs/pretrained_eval/repvit_m0_9/metrics.json",
                        "outputs/pretrained_eval/repvit_m0_9/latency.json",
                        "outputs/pretrained_eval/repvit_m0_9/top5_samples.json",
                        "outputs/pretrained_eval/summary.csv"]),
    ("5", "多型号规模和性能比较", ["outputs/advanced/params_vs_acc.png",
                          "outputs/advanced/latency_vs_acc.png",
                          "outputs/advanced/macs_vs_acc.png",
                          "outputs/benchmarks/family_summary.csv",
                          "outputs/advanced/pareto_summary.json"]),
    ("6", "数据集与数据划分", ["datasets/lists/pet_train.txt", "datasets/lists/pet_val.txt",
                        "datasets/lists/pet_test.txt",
                        "labels/pet_classes.txt", "labels/imagenet_classes.txt",
                        "outputs/metrics/leakage_check.json"]),
    ("7", "Baseline 迁移训练", ["outputs/logs/baseline_metrics.csv",
                          "outputs/logs/baseline_config_effective.yaml",
                          "outputs/metrics/baseline_test.json",
                          "outputs/metrics/weight_load_report.json"]),
    ("8", "优化方法与实验假设", ["outputs/logs/opt_combo_config_effective.yaml",
                         "outputs/logs/opt_randaug_config_effective.yaml",
                         "outputs/logs/opt_compare_summary.csv"]),
    ("9", "定量结果", ["outputs/report_assets/table_results.csv",
                   "outputs/advanced/ablation_summary.csv",
                   "outputs/metrics/ablation.csv"]),
    ("10", "曲线和混淆矩阵分析", ["outputs/curves/opt_compare.png",
                           "outputs/curves/baseline_curves.png",
                           "outputs/confusion_matrix/baseline_cm.png",
                           "outputs/confusion_matrix/baseline_per_class_f1.png",
                           "outputs/confusion_matrix/baseline_cat_dog_block.json"]),
    ("11", "Grad-CAM 与失败案例", ["outputs/gradcam/gradcam_correct_baseline.png",
                             "outputs/gradcam/gradcam_wrong_baseline.png",
                             "outputs/gradcam/gradcam_layer_compare_baseline.png",
                             "outputs/predictions/test_top5_baseline_grid8.png",
                             "outputs/metrics/gradcam_meta.json"]),
    ("12", "结构重参数化", ["outputs/reparam/repvit_m0_9_pet37_reparam_report.json",
                      "outputs/reparam/repvit_m0_9_pet37_onnx_nodes.json",
                      "outputs/advanced/block_22_fusion_report.txt"]),
    # 13 节不硬写 ONNX 文件名：全部经 deploy/model_registry.py 的 onnx_path(key) 取。
    # 一致性记录见 outputs/metrics/consistency_<registry_key>.json，
    # 「随仓库交付的 ONNX 型号」以 registry 的 shipped=True 为准（见 deploy/model_registry.py）。
    ("13", "ONNX 多模型部署", ["onnx/repvit_m0_9_in1k.onnx",
                         "onnx/repvit_m1_0_in1k.onnx",
                         "onnx/repvit_m0_9_pet37.onnx",
                         "outputs/metrics/consistency_repvit_m0_9_pet37.json",
                         "outputs/metrics/consistency_repvit_m0_9_in1k.json",
                         "outputs/metrics/consistency_repvit_m1_0_in1k.json",
                         "outputs/metrics/consistency_repvit_m1_1_in1k.json",
                         "outputs/metrics/consistency_repvit_m1_5_in1k.json"]),
    ("14", "性能测试", ["outputs/report_assets/table_benchmark_meta.csv",
                    "outputs/benchmarks/summary.csv",
                    "outputs/metrics/onnx_io_nodes.json",
                    "outputs/metrics/bench.jsonl"]),
    ("15", "遇到的问题和解决方法", ["outputs/metrics/selfcheck_report.json",
                            "outputs/env_snapshot.json"]),
    ("16", "总结与后续计划", ["outputs/metrics/dataset_report.json",
                        "outputs/verification/consistency_repvit_m0_9_pet37_n12.json"]),
]

# 答辩 PPT 15 页（题目第 16-17 页）与每页实际读取的落盘产物。
# 页码 = 最终页序（P03 已把「M0.9 结构」与「为什么不是标准 ViT」合并为一页，P11 为新增「预测结果」页）；
# 页面设计源另见 report/ppt_svg/*.svg（冻结记录，页脚编号与最终页序不同，不作为本表口径）。
PPT_SLIDES = [
    ("1", "题目与任务完成情况", []),
    ("2", "完成度总览（六个考核环节）", ["outputs/metrics/selfcheck_report.json"]),
    ("3", "RepViT 核心结构（含为什么不是标准 ViT）",
     ["outputs/architecture/repvit_m0_9_arch.png",
      "outputs/architecture/repvit_m0_9_arch.md",
      "outputs/reparam/repvit_m0_9_onnx_nodes.json"]),
    ("4", "官方多型号评价", ["outputs/pretrained_eval/summary.csv",
                       "outputs/pretrained_eval/repvit_m0_9/metrics.json",
                       "outputs/benchmarks/family_summary.csv"]),
    ("5", "数据集和训练流程", ["outputs/metrics/leakage_check.json",
                        "outputs/metrics/dataset_report.json",
                        "outputs/metrics/pet_label_audit.json",
                        "outputs/metrics/split_audit.json"]),
    ("6", "Baseline 结果", ["outputs/metrics/baseline_test.json",
                         "outputs/metrics/weight_load_report.json",
                         "outputs/metrics/backbone_updated.json",
                         "outputs/logs/baseline_metrics.csv"]),
    ("7", "优化假设与控制变量", ["outputs/logs/opt_combo_config_effective.yaml",
                          "outputs/logs/opt_randaug_config_effective.yaml",
                          "outputs/logs/opt_compare_summary.csv"]),
    ("8", "曲线和定量结果", ["outputs/curves/opt_compare.png",
                        "outputs/logs/baseline_metrics.csv",
                        "outputs/logs/opt_combo_metrics.csv",
                        "outputs/metrics/opt_combo_test.json"]),
    ("9", "混淆矩阵与失败案例", ["outputs/confusion_matrix/baseline_cm.png",
                         "outputs/confusion_matrix/baseline_per_class_f1.png",
                         "outputs/predictions/case_wrong_baseline_case01.png",
                         "outputs/predictions/case_wrong_baseline_case02.png"]),
    ("10", "Grad-CAM 结果", ["outputs/gradcam/gradcam_correct_baseline_Bengal_30_correct.png",
                           "outputs/gradcam/gradcam_correct_baseline_Russian_Blue_205_correct.png",
                           "outputs/gradcam/gradcam_wrong_baseline_boxer_2_wrong.png",
                           "outputs/gradcam/gradcam_wrong_baseline_Egyptian_Mau_204_wrong.png"]),
    ("11", "预测结果（8 张测试集 + 4 组同图对比 + 5 张实拍）",
     ["outputs/predictions/test_top5_baseline_grid8.png",
      "outputs/predictions/compare_baseline_vs_opt_combo_grid4.png",
      "outputs/predictions/external_top5_pet37_grid5.png",
      "outputs/predictions/baseline_vs_opt_combo_predict_compare.json"]),
    ("12", "结构重参数化", ["outputs/reparam/repvit_m0_9_pet37_reparam_report.json",
                       "outputs/reparam/repvit_m0_9_onnx_nodes.json"]),
    ("13", "ONNX 多模型部署", ["outputs/metrics/consistency_repvit_m0_9_pet37.json",
                          "outputs/metrics/consistency_repvit_m0_9_in1k.json",
                          "outputs/metrics/consistency_repvit_m1_0_in1k.json",
                          "outputs/metrics/consistency_repvit_m1_1_in1k.json",
                          "outputs/metrics/consistency_repvit_m1_5_in1k.json"]),
    ("14", "性能比较", ["outputs/benchmarks/summary.csv",
                     "outputs/pretrained_eval/summary.csv",
                     "outputs/metrics/onnx_io_nodes.json"]),
    ("15", "遇到的问题与总结", ["outputs/metrics/selfcheck_report.json",
                          "outputs/metrics/dataset_report.json"]),
]

# 结果口径对照表的列（题目第 16 页要求区分的 7 类结果的共同字段）
RESULT_COLUMNS = ["数据集", "类别数", "Top-1", "Top-5", "Macro-F1",
                  "参数量", "MACs", "文件大小", "延迟"]

# 7 行口径：(行名, 数据来源路径, 备注口径, top1/top5 是否已经是百分数)
# 路径全部指向现行落盘产物；取不到值的单元格由 make_report_assets.py 写 "—"（不编造）。
CALIBER_ROWS = [
    ("论文公布",       "report/sources/literature.yaml#paper",          "iPhone 12 + CoreML，四舍五入到 0.1ms", False),
    ("官方仓库公布",   "report/sources/literature.yaml#official_repo",  "THU-MIG README Model Zoo 表",         False),
    ("官方权重实测",   "outputs/pretrained_eval/repvit_m0_9/metrics.json",
     "本机 PyTorch 2.14.0+cu126（CUDA）实跑；**该文件的 top1/top5 已是百分数，不得再乘 100**", True),
    ("自训练 Baseline", "outputs/metrics/baseline_test.json",           "Pet-37 test split（baseline）",       False),
    ("自优化模型",     "outputs/metrics/opt_combo_test.json",           "Pet-37 test split（组合方案 opt_combo）", False),
    ("PyTorch 推理",   "outputs/pretrained_eval/repvit_m0_9/latency.json",
     "batch=1, fp32, 本机 CUDA（PyTorch 2.14.0+cu126）, threads=4", False),
    ("ONNX 部署",      "outputs/metrics/bench.jsonl",                   "ORT CPUExecutionProvider",            False),
]

# 性能测试元信息 10 项（题目第 13 页 8 条 + 第 17 页 6 条去重）
# 取值来源：bench.jsonl 末行 + outputs/metrics/onnx_io_nodes.json（io_nodes 缺失时的只读补全）。
BENCH_META_KEYS = [
    "model",          # 1 模型型号
    "input_size",     # 2 输入尺寸
    "batch_size",     # 3 batch size
    "precision",      # 4 精度类型 fp32/fp16
    "backend",        # 5 推理后端（执行提供者 EP）
    "hardware",       # 6 硬件（CPU 型号 + 内存 + OS）
    "ort_version",    # 7 ONNX Runtime 版本
    "runs",           # 8 预热 10 次 / 正式 50 次
    "file_size_mb",   # 9 模型文件大小
    "io_nodes",       # 10 输入输出节点信息
]

# 图表清单：路径 -> (用途, 对应题目条款, 生成脚本)。全部为现行产物，不再引用 figures/。
FIGURE_SPEC = {
    "outputs/architecture/repvit_m0_9_arch.png": (
        "RepViT-M0.9 整网结构图（9 要素：输入尺寸 224；Stem；四个主要阶段；"
        "分辨率 224→112→56→28→14→7；通道 3→24→48→96→192→384；RepViT Block；"
        "Global Average Pooling；分类头；最终输出维度 37）",
        "报告第 3 节 / PPT 第 3 页", "tools/draw_arch.py"),
    "outputs/architecture/repvit_m0_9_arch.md": (
        "结构图的文字版（每层形状由 forward hook 现测）", "报告第 3 节", "tools/draw_arch.py"),
    "outputs/advanced/params_vs_acc.png": ("参数量—准确率关系图",
        "进阶任务 1 / 报告第 5 节", "tools/pareto_family.py"),
    "outputs/advanced/latency_vs_acc.png": ("延迟—准确率关系图（同子集/同预处理/同后端）",
        "进阶任务 1 / 报告第 5 节", "tools/pareto_family.py"),
    "outputs/advanced/macs_vs_acc.png": ("MACs—准确率关系图",
        "进阶任务 1 / 报告第 5 节", "tools/pareto_family.py"),
    "outputs/curves/baseline_curves.png": ("Baseline 五条训练曲线", "报告第 7 / 10 节",
                                           "tools/plot_curves.py"),
    "outputs/curves/opt_compare.png": ("Baseline vs 优化模型同图对比（共用坐标范围）",
        "报告第 9 / 10 节 / PPT 第 8 页", "tools/plot_curves.py"),
    "outputs/curves/ablation_curves.png": ("Baseline 与三组消融臂的同图训练曲线",
        "进阶任务 2 / 报告第 9 节", "tools/plot_curves.py"),
    "outputs/confusion_matrix/baseline_cm.png": ("行归一化混淆矩阵（normalize='true'）",
        "报告第 10 节 / PPT 第 9 页", "tools/check_cm.py"),
    "outputs/confusion_matrix/baseline_per_class_f1.png": ("37 类 F1 升序柱状图",
        "报告第 10 节 / PPT 第 9 页", "tools/check_cm.py"),
    "outputs/gradcam/gradcam_correct_baseline.png": ("判对案例的 Grad-CAM 拼图",
        "报告第 11 节 / PPT 第 10 页", "tools/gradcam.py"),
    "outputs/gradcam/gradcam_wrong_baseline.png": ("判错案例的 Grad-CAM 拼图",
        "报告第 11 节 / PPT 第 10 页", "tools/gradcam.py"),
    "outputs/gradcam/gradcam_layer_compare_baseline.png": ("不同挂载层的 CAM 对比",
        "报告第 11 节", "tools/gradcam.py"),
    "outputs/predictions/test_top5_baseline_grid8.png": ("测试集 8 张预测（Top-5 类别 + 置信度）",
        "报告第 11 节 / PPT 第 11 页", "tools/plot_predictions.py"),
    "outputs/predictions/compare_baseline_vs_opt_combo_grid4.png": (
        "Baseline vs 优化模型同批图片 4 组对比", "报告第 9 节 / PPT 第 11 页", "tools/plot_side_by_side.py"),
    "outputs/predictions/external_top5_pet37_grid5.png": ("训练集以外的实拍图片 5 张 Top-5",
        "报告第 11 节 / PPT 第 11 页", "tools/plot_predictions.py"),
    "outputs/predictions/case_wrong_baseline_case01.png": ("失败案例 1（含 Top-5 与置信度）",
        "报告第 10 节 / PPT 第 9 页", "tools/plot_predictions.py"),
    "outputs/predictions/case_wrong_baseline_case02.png": ("失败案例 2（含 Top-5 与置信度）",
        "报告第 10 节 / PPT 第 9 页", "tools/plot_predictions.py"),
    "outputs/report_assets/table_results.csv": ("7 类结果口径对照表",
        "题目第 16 页（报告第 9 节 / PPT 第 8 页）", "tools/make_report_assets.py"),
    "outputs/report_assets/table_benchmark_meta.csv": ("性能测试 10 项元信息表",
        "题目第 13 / 17 页（报告第 14 节）", "tools/make_report_assets.py"),
    "outputs/report_assets/table_cases.csv": ("官方模型评价的正误案例表（2 正 + 2 误）",
        "题目第 5 页 1.1-7/1.1-8（报告第 4.5 节）", "tools/make_report_assets.py"),
    "outputs/advanced/robustness_repvit_m0_9_pet37.json": (
        "7 类扰动 × 6 级 severity 的鲁棒性结果", "进阶任务 5 / 报告附录 C", "tools/robustness_test.py"),
    "outputs/advanced/robustness_gradcam.png": (
        "扰动下 Grad-CAM 关注区域的变化（3 类扰动 × 2 级 severity + 干净对照，"
        "逐格元信息见 outputs/advanced/robustness_gradcam.json）",
        "进阶任务 5 / 报告附录 C", "tools/robustness_gradcam.py"),
    "outputs/benchmarks/igpu_cpu_baseline.json": ("集显/CPU 四路后端延迟对照",
        "进阶任务 4 / 报告附录 E", "tools/bench_igpu.py"),
}


# ---------- 自检：本文件的每条路径都必须存在（供 make/check 两个脚本共用） ----------
def missing_assets(root: Path | str = ".") -> list[str]:
    """返回所有不存在（或过小）的产物路径：路径写错、产物没生成都会在这里暴露。"""
    root = Path(root)
    bad: list[str] = []
    for _, _, assets in REPORT_SECTIONS:
        bad += [a for a in assets if not (root / a).exists()]
    for _, _, figs in PPT_SLIDES:
        bad += [a for a in figs if not (root / a).exists()]
    for rel in FIGURE_SPEC:
        if not (root / rel).exists():
            bad.append(rel)
    for _, src, _, _ in CALIBER_ROWS:
        p = src.partition("#")[0]
        if not (root / p).exists():
            bad.append(p)
    # 去重且保序
    seen, out = set(), []
    for p in bad:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out
