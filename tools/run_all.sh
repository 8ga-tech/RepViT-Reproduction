#!/usr/bin/env bash
# tools/run_all.sh —— RepViT 复现全流程一键执行（Git Bash / WSL）
#
# Source: Self-written。
# 说明：规格书模块 2.3.7 给了一版参考 run_all.sh，但其中的调用形式
#       （`python -m tools.visualize --cfg ... --experiments ...` 等）与
#       本仓库各脚本**实际已验证的 CLI 不一致**（如 visualize 走 --pred-csv、
#       gradcam 走 --ckpt/--cases、reparam_verify 不接收 --cfg）。
#       照抄参考版会因为「Python 收到未定义参数」而立即失败，所以这里按真实
#       CLI 重写，每条命令都在本机实跑通过。
#
# 用法：
#   bash tools/run_all.sh              # 全流程
#   SKIP_TRAIN=1 bash tools/run_all.sh # 跳过训练与优化（只重跑评价/可视化/部署）
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export HF_ENDPOINT=https://hf-mirror.com        # timm/HF 权重走镜像，国内直连不通
PY="${PYTHON:-python}"
TS() { date +%Y%m%d_%H%M%S; }

CFG_B=configs/baseline.yaml
CFG_P=configs/pretrained_eval.yaml
mkdir -p outputs/logs outputs/metrics outputs/benchmarks onnx

echo "== 0/9 环境自检 =="
$PY tools/env_check.py

echo "== 1/9 数据准备（划分 + 泄漏检查 + 审计）=="
$PY tools/assert_data.py --root data/oxford-iiit-pet
$PY datasets/make_pet_split.py --root data/oxford-iiit-pet \
    --out-dir datasets/lists --val-per-class 20 --seed 42
$PY datasets/audit_leakage.py
$PY tools/evaluate.py --cfg "$CFG_B" --audit

echo "== 2/9 官方权重评价（2 个基础型号 + 3 个家族型号）=="
$PY tools/run_all_pretrained.py --cfg "$CFG_P" \
    --model repvit_m0_9 repvit_m1_0 repvit_m1_1 repvit_m1_5 repvit_m2_3 \
    2>&1 | tee "outputs/logs/eval_pretrained_$(TS).log"

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo "== 3/9 Baseline 训练（40 epoch 满预算）=="
  $PY tools/train.py --cfg "$CFG_B" 2>&1 | tee "outputs/logs/train_baseline_$(TS).log"

  echo "== 4/9 优化实验（Baseline 与优化只差一个 --cfg）=="
  for EXP in opt_mix opt_randaug opt_disc opt_combo opt_abl_a opt_abl_b opt_abl_ab; do
    $PY tools/diff_config.py "$CFG_B" "configs/${EXP}.yaml"
    $PY tools/train.py --cfg "configs/${EXP}.yaml" 2>&1 | tee "outputs/logs/train_${EXP}_$(TS).log"
  done
  $PY tools/same_budget.py --a outputs/logs/baseline_metrics.csv \
                           --b outputs/logs/opt_combo_metrics.csv
fi

echo "== 5/9 曲线 / 混淆矩阵 / 预测图 / 案例 =="
$PY tools/plot_curves.py --runs baseline=outputs/logs/baseline_metrics.csv \
    --out outputs/curves/baseline_curves.png
$PY tools/plot_curves.py \
    --runs baseline=outputs/logs/baseline_metrics.csv \
           opt_combo=outputs/logs/opt_combo_metrics.csv \
    --out outputs/curves/opt_compare.png --title "Baseline vs 优化（方案A 组合式）"
$PY tools/visualize.py --pred-csv outputs/predictions/baseline_test_preds.csv \
    --classes labels/pet_classes.txt --out-dir outputs --tag baseline --split test
$PY tools/plot_predictions.py --pred-csv outputs/predictions/baseline_test_preds.csv \
    --num 8 --cols 4 --out outputs/predictions/test_top5_baseline_grid8.png --tag baseline
$PY tools/plot_predictions.py --cases-csv outputs/predictions/cases_test_baseline.csv \
    --role wrong --num 2 --out outputs/predictions/case_wrong_baseline.png --tag baseline

echo "== 6/9 Grad-CAM 与外部图片 =="
$PY tools/gradcam.py --model repvit_m0_9 --ckpt checkpoints/baseline_best.pt \
    --num-classes 37 --classes labels/pet_classes.txt \
    --cases outputs/predictions/cases_test_baseline.csv \
    --out-dir outputs/gradcam --tag baseline --which A
$PY tools/fetch_external_images.py --num 8
$PY tools/predict_external.py --model repvit_m0_9_pet37 --dir external --num 8

echo "== 7/9 结构重参数化验证（必须先于 ONNX 导出）=="
$PY tools/reparam_verify.py --model repvit_m0_9_pet37 \
    --weights checkpoints/baseline_best.pt --out-dir outputs/reparam
$PY tools/reparam_verify.py --model repvit_m0_9 \
    --weights checkpoints/pretrained/repvit_m0_9_distill_300e.pth --out-dir outputs/reparam

echo "== 8/9 ONNX 导出 + 一致性 =="
bash deploy/run_all.sh

echo "== 9/9 报告素材 + 全局自检 =="
$PY tools/draw_arch.py --out-dir outputs/architecture
$PY tools/make_report_assets.py --root . --out outputs/report_assets
$PY tools/selfcheck.py --json outputs/metrics/selfcheck_report.json || true

echo
echo "全部完成。产物见 outputs/ 、onnx/ 、checkpoints/ ；自检报告 outputs/metrics/selfcheck_report.json"
