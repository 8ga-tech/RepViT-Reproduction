# tools/run_all_pretrained.py
# -*- coding: utf-8 -*-
"""多型号官方预训练权重批量评价 + summary.csv 汇总（规格书 5.7）。

逐字采用规格书 4689-4746 行给出的实现，仅做两处不改变语义的最小适配：
  [1] 顶部补本文件头与 `ROOT`（供 subprocess 指定 cwd，保证相对路径在任意调用目录下一致）；
  [2] `subprocess.run(..., cwd=ROOT)`，其余命令行构造与 summary.csv 的 8 列顺序逐字保留。

用法:
    python tools/run_all_pretrained.py --cfg configs/pretrained_eval.yaml \
            --model repvit_m0_9 repvit_m1_0 repvit_m1_1 repvit_m1_5 repvit_m2_3
"""
import argparse, csv, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # 仓库根，禁止写死个人绝对路径

# 与 eval_pretrained.py 同源：配置走 --cfg，覆盖一律走 --set（键在 pretrained_eval. 段下）
# ★ data_list 必须与 configs/pretrained_eval.yaml 的 pretrained_eval.data_list 保持一致：
#   命令行 --set 优先级高于配置文件，这里写旧清单会让批量重跑静默用错数据。
#   当前口径 = ImageNetV2 matched-frequency 的 1000 张确定性固定子集（非 ImageNet-1K 验证集）。
BASE = ['--set',
        'pretrained_eval.data_list=datasets/lists/imagenetv2_mf_1000.txt',
        'pretrained_eval.labels=labels/imagenet_classes.txt',
        'pretrained_eval.input_size=224', 'pretrained_eval.batch_size=64',
        'pretrained_eval.crop_pct=0.875', 'pretrained_eval.interpolation=bicubic',
        'pretrained_eval.warmup=10', 'pretrained_eval.runs=50', 'pretrained_eval.threads=4']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='configs/pretrained_eval.yaml')
    ap.add_argument('--model', nargs='+', required=True)      # 沿用 deploy/*.py 的 --model <key> [...] 写法
    ap.add_argument('--pretrained-dir', default='checkpoints/pretrained')
    ap.add_argument('--root-out', default='outputs/pretrained_eval')
    a = ap.parse_args()

    rows = []
    for name in a.model:
        base = name.split('.')[0]             # 去掉权重 tag：repvit_m0_9.dist_300e_in1k -> repvit_m0_9
        w = os.path.join(a.pretrained_dir, f'{base}_distill_300e.pth')
        out = os.path.join(a.root_out, base)
        cmd = [sys.executable, 'tools/eval_pretrained.py', '--cfg', a.cfg,
               '--set', f'pretrained_eval.model={name}',
                        f'pretrained_eval.weights={w}',
                        f'pretrained_eval.out_dir={out}'] + BASE
        print('[run]', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=str(ROOT))   # 单型号失败立即中断，不要静默跳过

        m = json.load(open(os.path.join(ROOT, out, 'metrics.json'), encoding='utf-8'))
        l = json.load(open(os.path.join(ROOT, out, 'latency.json'), encoding='utf-8'))
        rows.append({'model': base,
                     'params_M': round(m['params_total'] / 1e6, 3),
                     'macs_G': m['macs_g'],
                     'file_mb': m['model_file_size_mb'],
                     'top1': m['top1'], 'top5': m['top5'],
                     'latency_mean_ms': round(l['mean_ms'], 2),
                     'latency_p95_ms': round(l['p95_ms'], 2)})

    root_out = os.path.join(ROOT, a.root_out)
    os.makedirs(root_out, exist_ok=True)
    csv_path = os.path.join(root_out, 'summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=['model', 'params_M', 'macs_G', 'file_mb',
                                           'top1', 'top5', 'latency_mean_ms', 'latency_p95_ms'])
        wr.writeheader(); wr.writerows(rows)
    print('汇总 ->', os.path.abspath(csv_path))
    for r in rows:
        print('{model:<18} {params_M:>7}M {macs_G:>6}G {file_mb:>7}MiB '
              'top1={top1:>6} top5={top5:>6} {latency_mean_ms:>7}ms/{latency_p95_ms:>7}ms'.format(**r))


if __name__ == '__main__':
    main()
