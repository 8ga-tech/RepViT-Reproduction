#!/usr/bin/env bash
# deploy/run_all.sh —— 导出 -> 推理 -> 基准 -> 一致性，一键跑完
#
# Source: Self-written（基于规格书 §10.6 的参考脚本，按其真实 CLI 校正）。
# 相对参考版的四处修正：
#   [1] 参考版给 benchmark.py 传了 --image，但本仓 benchmark 只测纯推理延迟
#       （端到端分解由 preprocess/inference/postprocess 三段独立计时给出），
#       不接收 --image，照抄会因未定义参数直接失败。
#   [2] 参考版用 outputs/samples/sample.jpg 作固定测试图，本仓库不含该文件；
#       这里改用 external/ 下的外部图片（由 tools/fetch_external_images.py 生成）。
#   [3] compare_torch_onnx.py 只对「已导出」的模型跑；导出失败时不应带病继续。
#   [4]（2026-09 同步交付态）MODELS 补齐为**随仓库交付的 5 个 ONNX**，一致性统一到
#       n=12 与入库产物同口径；m2_3 只在「本机已导出」时参与基准测试。
#       单一真源：deploy/model_registry.py 的 shipped_keys()、.gitignore、
#       tools/check_report_assets.py 的强制清单三处一致。
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
mkdir -p onnx outputs/benchmarks outputs/metrics outputs/predictions

MODELS="repvit_m0_9_in1k repvit_m1_0_in1k repvit_m1_1_in1k repvit_m1_5_in1k repvit_m0_9_pet37"

echo "== [1/4] 导出 ONNX（文件名一律经 registry 的 onnx_path()）=="
for m in $MODELS; do
  python deploy/export_onnx.py --model "$m"
done

echo "== [2/4] 单图推理（独立预处理 + softmax + Top-K）=="
SAMPLE=$(ls external/*.JPEG 2>/dev/null | head -1 || true)
if [ -n "$SAMPLE" ]; then
  for m in $MODELS; do
    python deploy/infer_onnx.py --model "$m" --image "$SAMPLE" --topk 5 \
        --dump-probs --save-json "outputs/predictions/${m}_sample.json"
  done
else
  echo "[warn] external/ 下没有图片，跳过单图推理"
  echo "       先跑：python tools/fetch_external_images.py --num 8"
fi

echo "== [3/4] 性能测试（预热 10 / 正式 50 / threads=4，每个模型独立进程）=="
# 交付集 5 个 ONNX + 本机若已导出 m2_3（92 MB，不入库）则一并测：
# 入库的 outputs/benchmarks/summary.csv 是 6 行（家族帕累托图需要 M2.3 的延迟点）。
BENCH_MODELS="$MODELS"
if [ -f onnx/repvit_m2_3_in1k.onnx ]; then
  BENCH_MODELS="$BENCH_MODELS repvit_m2_3_in1k"
fi
for m in $BENCH_MODELS; do
  python deploy/benchmark.py --model "$m" --warmup 10 --runs 50 --threads 4 \
      --out-dir outputs/benchmarks
  sleep 2      # 让上一个进程的线程池彻底退出，避免争抢把 P50 抬高
done

echo "== [4/4] PyTorch vs ONNX 一致性（n=12，与入库产物同口径）=="
# 默认清单：1000 类型号 = datasets/lists/imagenetv2_mf_1000.txt（ImageNetV2 固定子集），
# pet37 = datasets/lists/pet_test.txt。--limit 12 与入库的 consistency_*.json 一致。
for m in $MODELS; do
  python deploy/compare_torch_onnx.py --model "$m" --limit 12 \
      --out "outputs/metrics/consistency_${m}.json"
done

echo
echo "完成。产物：onnx/*.onnx 、outputs/benchmarks/summary.csv 、outputs/metrics/bench.jsonl 、"
echo "      outputs/metrics/consistency_*.json"
