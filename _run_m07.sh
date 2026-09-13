#!/usr/bin/env bash
# M07：优化实验的控制变量运行（串行；不与其它重活并发）。
#
# 预算协议：固定 40 epoch 满预算（early_stop_patience=999 继承自 baseline.yaml）。
#
# 【两条硬约束，都踩过坑】
#  1) 重试前必须删掉上一轮的逐 epoch CSV。tools/train.py 的 TrainLogger 是**追加**写：
#     上一轮在 epoch 23 崩掉时，直接重试会让 CSV 变成 23 + 40 = 63 行，
#     epoch 列出现重复且非单调，same_budget.py / compare_runs.py 读到的就是脏数据。
#  2) **绝不要在脚本运行期间修改本文件**。bash 是按偏移量增量读取脚本的，
#     运行中改写会让已在跑的实例从中途读到新内容，出现「两个 runner 交错执行、
#     互相覆盖同一份 CSV」的混乱（实测发生过：opt_combo 只剩 1 行、opt_abl_ab 24 行）。
#     要改就先把 runner 杀掉。
set -u
cd "$(dirname "$0")"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONIOENCODING=utf-8

run_one () {
  local exp="$1" ts rc
  ts=$(date +%Y%m%d_%H%M%S)
  echo "=== [M07] $exp start $(date +%H:%M:%S) ==="
  python tools/train.py --cfg "configs/${exp}.yaml" > "outputs/logs/train_${exp}_${ts}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== [M07] $exp FAILED rc=$rc -> 清掉半成品后重试一次 ==="
    rm -f "outputs/logs/${exp}_metrics.csv" "outputs/logs/${exp}_metrics.jsonl" \
          "outputs/logs/${exp}_steps.jsonl" "outputs/logs/${exp}_config_effective.yaml" \
          "outputs/logs/${exp}_config_effective.json" \
          "checkpoints/${exp}_best.pt" "checkpoints/${exp}_last.pt"
    ts=$(date +%Y%m%d_%H%M%S)
    python tools/train.py --cfg "configs/${exp}.yaml" > "outputs/logs/train_${exp}_${ts}.log" 2>&1
    rc=$?
  fi
  echo "=== [M07] $exp EXIT=$rc $(date +%H:%M:%S) ==="
}

for exp in opt_combo opt_abl_a opt_abl_b opt_abl_ab; do
  run_one "$exp"
done
echo "=== [M07] ALL DONE ==="
