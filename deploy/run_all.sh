#!/usr/bin/env bash
# deploy/run_all.sh —— 导出 -> 推理 -> 基准 -> 一致性，一键跑完
#
# Source: Self-written（基于规格书 §10.6 的参考脚本，按其真实 CLI 校正）。
# 相对参考版的三处修正：
#   [1] 参考版给 benchmark.py 传了 --image，但本仓 benchmark 只测纯推理延迟
#       （端到端分解由 preprocess/inference/postprocess 三段独立计时给出），
#       不接收 --image，照抄会因未定义参数直接失败。
#   [2] 参考版用 outputs/samples/sample.jpg 作固定测试图，本仓库不含该文件；
#       这里改用 external/ 下的外部图片（由 tools/fetch_external_images.py 生成）。
#   [3] compare_torch_onnx.py 只对「已导出」的模型跑；导出失败时不应带病继续。
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
mkdir -p onnx outputs/benchmarks outputs/metrics outputs/predictions

MODELS="repvit_m0_9_in1k repvit_m1_0_in1k repvit_m0_9_pet37"

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
for m in $MODELS; do
  python deploy/benchmark.py --model "$m" --warmup 10 --runs 50 --threads 4 \
      --out-dir outputs/benchmarks
  sleep 2      # 让上一个进程的线程池彻底退出，避免争抢把 P50 抬高
done

echo "== [4/4] PyTorch vs ONNX 一致性 =="
python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --limit 500 \
    --out outputs/metrics/consistency_repvit_m0_9_pet37.json
python deploy/compare_torch_onnx.py --model repvit_m0_9_in1k --limit 500 \
    --out outputs/metrics/consistency_repvit_m0_9_in1k.json

echo
echo "完成。产物：onnx/*.onnx 、outputs/benchmarks/summary.csv 、outputs/metrics/bench.jsonl 、"
echo "      outputs/metrics/consistency_*.json"
