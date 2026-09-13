# tools/report_spec.py
"""Source: Self-written
报告/PPT 的契约常量。make_report_assets.py 与 check_report_assets.py 共用本文件，
保证"要求什么"与"检查什么"永不漂移。
"""

# 结果口径对照表的列（题目第 16 页要求区分的 7 类结果的共同字段）
RESULT_COLUMNS = ["数据集", "类别数", "Top-1", "Top-5", "Macro-F1",
                  "参数量", "MACs", "文件大小", "延迟"]

# 7 行口径：(行名, 数据来源路径, 备注口径, top1/top5 是否已经是百分数)
CALIBER_ROWS = [
    ("论文公布",       "report/sources/literature.yaml#paper",          "iPhone 12 + CoreML，四舍五入到 0.1ms", False),
    ("官方仓库公布",   "report/sources/literature.yaml#official_repo",  "THU-MIG README Model Zoo 表",         False),
    ("官方权重实测",   "outputs/pretrained_eval/repvit_m0_9_in1k/metrics.json",
     "本机 ORT/PyTorch，须写明子集大小；**该文件的 top1/top5 已是百分数，不得再乘 100**", True),
    ("自训练 Baseline", "outputs/metrics/B0_baseline_test.json",        "Pet-37 test split",                   False),
    ("自优化模型",     "outputs/metrics/O1_randaug_test.json",          "Pet-37 test split",                   False),
    ("PyTorch 推理",   "outputs/metrics/pytorch_latency.json",          "batch=1, fp32, 本机 CPU",             False),
    ("ONNX 部署",      "outputs/metrics/bench.jsonl",                   "ORT CPUExecutionProvider",            False),
]

# 性能测试元信息 10 项（题目第 13 页 8 条 + 第 17 页 6 条去重）
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

# 报告 16 节（题目第 15-16 页）与其所需图表；产物路径相对仓库根
REPORT_SECTIONS = [
    ("1", "任务背景与复现范围", []),
    ("2", "RepViT 论文核心思路", ["figures/roadmap_incremental.png"]),
    ("3", "RepViT-M0.9 结构", ["outputs/architecture/repvit_m0_9_arch.png",
                          "outputs/architecture/repvit_m0_9_arch.md",
                          "figures/m0_9_block.png", "figures/m0_9_cfg_table.png"]),
    ("4", "官方预训练模型评价", ["outputs/pretrained_eval/repvit_m0_9_in1k/metrics.json",
                        "figures/official_eval_bar.png"]),
    ("5", "多型号规模和性能比较", ["figures/params_vs_acc.png", "figures/latency_vs_acc.png"]),
    ("6", "数据集与数据划分", ["datasets/lists/pet_train.txt", "datasets/lists/pet_val.txt",
                        "datasets/lists/pet_test.txt", "figures/pet37_class_hist.png"]),
    ("7", "Baseline 迁移训练", ["outputs/logs/B0_baseline_metrics.csv",
                          "outputs/logs/B0_baseline_config_effective.yaml"]),
    ("8", "优化方法与实验假设", ["outputs/logs/O1_randaug_config_effective.yaml"]),
    ("9", "定量结果", ["outputs/report_assets/table_results.csv"]),
    ("10", "曲线和混淆矩阵分析", ["outputs/curves/training_curves_all.png",
                           "outputs/confusion_matrix/O1_randaug_test_norm.png",
                           "outputs/confusion_matrix/per_class_f1_O1.png"]),
    ("11", "Grad-CAM 与失败案例", ["outputs/gradcam/gradcam_compare_O1.png",
                             "outputs/predictions/top5_failures_O1.png"]),
    ("12", "结构重参数化", ["outputs/reparam/repvit_m0_9_reparam_report.json",
                      "figures/reparam_graph_nodes.png"]),
    # 13 节不硬写 ONNX 文件名：三份产物一律由 deploy/model_registry.py 的 onnx_path(key)
    # 取 onnx/<registry_key>.onnx（key = repvit_m0_9_in1k / repvit_m1_0_in1k / repvit_m0_9_pet37_opt）
    ("13", "ONNX 多模型部署", ["figures/onnx_io_nodes.png"]),
    ("14", "性能测试", ["outputs/report_assets/table_benchmark_meta.csv",
                    "figures/latency_bar_three_models.png"]),
    ("15", "遇到的问题和解决方法", []),
    ("16", "总结与后续计划", []),
]

# 答辩 PPT 15 页（题目第 16-17 页）与每页应插入的图
PPT_SLIDES = [
    ("1", "题目与任务完成情况", []),
    ("2", "RepViT 核心结构", ["outputs/architecture/repvit_m0_9_arch.png", "figures/m0_9_block.png"]),
    ("3", "为什么 RepViT 不是标准 ViT", ["figures/roadmap_incremental.png"]),
    ("4", "官方多型号评价", ["figures/params_vs_acc.png", "figures/latency_vs_acc.png"]),
    ("5", "数据集和训练流程", ["figures/pet37_class_hist.png"]),
    ("6", "Baseline 结果", ["outputs/curves/training_curves_B0.png"]),
    ("7", "优化假设与控制变量", ["figures/config_diff_b0_o1.png"]),
    ("8", "曲线和定量结果", ["outputs/curves/training_curves_all.png"]),
    ("9", "混淆矩阵与失败案例", ["outputs/confusion_matrix/O1_randaug_test_norm.png",
                          "outputs/predictions/top5_failures_O1.png"]),
    ("10", "Grad-CAM 结果", ["outputs/gradcam/gradcam_compare_O1.png"]),
    ("11", "结构重参数化", ["figures/reparam_graph_nodes.png"]),
    ("12", "ONNX 多模型部署", ["figures/onnx_io_nodes.png"]),
    ("13", "性能比较", ["figures/latency_bar_three_models.png"]),
    ("14", "遇到的问题", []),
    ("15", "总结", []),
]

# 图表清单：路径 -> (用途, 对应题目条款, 生成脚本)
FIGURE_SPEC = {
    "figures/roadmap_incremental.png": ("论文增量现代化设计路线",
        "报告第 2 节 / PPT 第 3 页", "tools/draw_arch.py --figure roadmap"),
    "figures/m0_9_block.png": ("RepViTBlock 训练态/推理态结构图",
        "报告第 3 节 / PPT 第 2 页", "tools/draw_arch.py --figure block"),
    "figures/m0_9_cfg_table.png": ("M0.9 各 stage 的 block/cfg 参数表",
        "报告第 3 节", "tools/draw_arch.py --figure cfg_table"),
    # 题目「1.2 模型理解」的基础交分项：自绘整网结构图，9 个要素缺一不可
    "outputs/architecture/repvit_m0_9_arch.png": (
        "RepViT-M0.9 整网结构图（9 要素：输入尺寸 224；Stem；四个主要阶段；"
        "分辨率 224→112→56→28→14→7；通道 3→24→48→96→192→384；RepViT Block；"
        "Global Average Pooling；分类头；最终输出维度 37）",
        "报告第 3 节 / PPT 第 2 页", "tools/draw_arch.py"),
    "figures/official_eval_bar.png": ("官方权重在 ImageNet 验证子集上的 Top-1/Top-5",
        "报告第 4 节", "tools/eval_pretrained.py"),
    "figures/params_vs_acc.png": ("参数量—准确率关系图",
        "进阶任务 1 / 报告第 5 节 / PPT 第 4 页", "tools/pareto_family.py"),
    "figures/latency_vs_acc.png": ("延迟—准确率关系图（同子集/同预处理/同后端）",
        "进阶任务 1 / 报告第 5 节 / PPT 第 4 页", "tools/pareto_family.py"),
    "figures/pet37_class_hist.png": ("Pet-37 各类样本数分布",
        "报告第 6 节 / PPT 第 5 页", "tools/visualize.py"),
    "figures/config_diff_b0_o1.png": ("Baseline 与优化实验的 YAML 差异对照（控制变量证据）",
        "PPT 第 7 页", "tools/diff_config.py"),
    "outputs/curves/training_curves_B0.png": ("Baseline 训练曲线", "报告第 7 节", "tools/visualize.py"),
    "outputs/curves/training_curves_all.png": ("Baseline vs 优化模型同图对比",
        "报告第 9/10 节 / PPT 第 8 页", "tools/visualize.py"),
    "outputs/confusion_matrix/O1_randaug_test_norm.png": ("行归一化混淆矩阵（normalize='true'）",
        "报告第 10 节 / PPT 第 9 页", "tools/visualize.py"),
    "outputs/confusion_matrix/per_class_f1_O1.png": ("37 类 F1 升序柱状图",
        "报告第 10 节", "tools/visualize.py"),
    "outputs/gradcam/gradcam_compare_O1.png": ("Baseline vs 优化模型 Grad-CAM 对比（上原图/下 CAM）",
        "报告第 11 节 / PPT 第 10 页", "tools/gradcam.py"),
    "outputs/predictions/top5_failures_O1.png": ("高置信度错误案例 Top-5 预测图",
        "报告第 11 节 / PPT 第 9 页", "tools/visualize.py"),
    "figures/reparam_graph_nodes.png": ("重参数化前后 ONNX 算子数对比（BN/Conv/Add）",
        "报告第 12 节 / PPT 第 11 页", "tools/reparam_verify.py"),
    "figures/onnx_io_nodes.png": ("三个 ONNX 模型输入输出节点信息截图排版",
        "报告第 13 节 / PPT 第 12 页", "tools/visualize.py"),
    "figures/latency_bar_three_models.png": ("三个 ONNX 模型延迟柱状图（含 P50/P95 误差棒）",
        "报告第 14 节 / PPT 第 13 页", "tools/visualize.py"),
}
