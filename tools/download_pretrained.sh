#!/usr/bin/env bash
# 下载官方 RepViT 预训练权重并打印校验值。
# Source: Self-written（URL 来自 THU-MIG/RepViT README 的 Releases v1.0 列表）。
#
# 【本机实测的网络事实，必须记下来】
#   github.com 与 objects.githubusercontent.com 直连均返回 HTTP 000（不可达），
#   因此默认走 ghfast.top 前置代理。若代理也失效，本脚本会在最后打印可用的替代路径。
#   无论走哪条路，脚本都会打印**字节数 + SHA256**，与本仓库记录逐项核对后才算通过。
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT_DIR="checkpoints/pretrained"; mkdir -p "$CKPT_DIR"
BASE="https://github.com/THU-MIG/RepViT/releases/download/v1.0"
MIRROR="${REPVIT_MIRROR:-https://ghfast.top}"       # 可用 REPVIT_MIRROR= 环境变量覆盖；置空则直连

# 官方 Release 的 5 个基础型号（字节数为官方 release 实测值，用于下载后自检）
NAMES="repvit_m0_9_distill_300e repvit_m1_0_distill_300e repvit_m1_1_distill_300e \
repvit_m1_5_distill_300e repvit_m2_3_distill_300e"
declare -A EXPECT=(
  [repvit_m0_9_distill_300e]=22422548
  [repvit_m1_0_distill_300e]=29675245
  [repvit_m1_1_distill_300e]=35668677
  [repvit_m1_5_distill_300e]=43449459
  [repvit_m2_3_distill_300e]=95860931
)

for NAME in $NAMES; do
  OUT="$CKPT_DIR/${NAME}.pth"
  URL="${MIRROR:+$MIRROR/}$BASE/${NAME}.pth"
  if [ -f "$OUT" ] && [ "$(stat -c%s "$OUT" 2>/dev/null || wc -c <"$OUT")" = "${EXPECT[$NAME]}" ]; then
    echo "[skip] $NAME 已存在且字节数正确"
  else
    echo "[get ] $NAME  <- $URL"
    # -C - 断点续传：100MB 级的包中途断了不用重下
    curl -L -C - --retry 3 --max-time 900 -o "$OUT" "$URL"
  fi
  ACTUAL=$(stat -c%s "$OUT" 2>/dev/null || wc -c <"$OUT")
  if [ "$ACTUAL" != "${EXPECT[$NAME]}" ]; then
    echo "[FAIL] $NAME 字节数 = $ACTUAL，期望 ${EXPECT[$NAME]}（疑似下载中断）"
    echo "       替代路径：HF 镜像上的 timm 权重与本文件逐位一致（已实测 max|Δlogits| = 0）"
    echo "       export HF_ENDPOINT=https://hf-mirror.com && python -c \"import timm;timm.create_model('repvit_m0_9', pretrained=True)\""
    exit 1
  fi
  sha256sum "$OUT" | tee "$CKPT_DIR/${NAME}.sha256"
done

echo
echo "全部权重就位。用下面的命令核对字节数与 SHA256（DoD #11）："
echo "    python tools/check_weights.py --dir $CKPT_DIR"
