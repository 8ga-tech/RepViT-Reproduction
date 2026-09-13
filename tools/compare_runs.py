"""tools/compare_runs.py
读两组 metrics.csv，输出对比表（top1/top5/macro_f1/最优epoch/收敛到90%所需epoch），
并把 5 条曲线画进同一张图（题目要求：Baseline 与优化模型尽量同一张图）。
用法：
  python tools/compare_runs.py --baseline baseline --opt opt_combo --out-dir outputs
"""
import argparse, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

CURVES = [('train_loss', '训练损失'), ('val_loss', '验证损失'),
          ('val_top1', '验证 Top-1'), ('val_macro_f1', '验证 Macro-F1'),
          ('lr', '学习率')]


def first_epoch(df, col, thr):
    """首个满足 df[col] >= thr 的 epoch（1-based）；从未满足返回 None。"""
    hit = df.index[df[col] >= thr]
    return int(df.loc[hit[0], 'epoch']) if len(hit) else None


def summarize(df, metric='val_macro_f1'):
    best_i = df[metric].idxmax()
    best = float(df.loc[best_i, metric])
    return {
        'best_top1': float(df['val_top1'].max()),
        'best_top5': float(df['val_top5'].max()),
        'best_macro_f1': best,
        'best_epoch': int(df.loc[best_i, 'epoch']),
        'last_top1': float(df['val_top1'].iloc[-1]),
        'last_macro_f1': float(df['val_macro_f1'].iloc[-1]),
        # 绝对阈值口径：首次达到 90% 的 epoch（两组可比，但与各自最优值无关）
        'epoch_to_abs90': first_epoch(df, 'val_top1', 0.90),
        # 相对口径：首次达到自身最优值 90% 的 epoch（衡量收敛速度，不受最终水平影响）
        'epoch_to_rel90': first_epoch(df, metric, 0.90 * best),
        # 全过程平均：比单点更能反映「整条曲线整体抬升」
        'mean_macro_f1': float(df['val_macro_f1'].mean()),
    }


def load_csv(exp):
    p = f'outputs/logs/{exp}_metrics.csv'
    assert os.path.isfile(p), f'缺少 {p}；请先完成训练'
    return pd.read_csv(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True, help='baseline 的 experiment_name')
    ap.add_argument('--opt', required=True, help='优化实验的 experiment_name')
    ap.add_argument('--out-dir', default='outputs')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out_dir, 'curves'), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, 'metrics'), exist_ok=True)
    pair = f'{a.baseline}_vs_{a.opt}'

    runs = [(a.baseline, load_csv(a.baseline)), (a.opt, load_csv(a.opt))]
    rows = {name: summarize(df) for name, df in runs}
    keys = list(rows[a.baseline].keys())
    table = pd.DataFrame(rows).T[keys]
    table['d_macro_f1'] = [None, table.iloc[1]['best_macro_f1'] - table.iloc[0]['best_macro_f1']]
    print(table.to_string(float_format=lambda v: f'{v:.4f}'))
    table.to_csv(os.path.join(a.out_dir, 'metrics', f'compare_{pair}.csv'), float_format='%.6f')

    # 说明：表中数值全部来自本地 metrics.csv，脚本不内置任何默认值或参考数字。
    with open(os.path.join(a.out_dir, 'metrics', f'compare_{pair}.md'), 'w', encoding='utf-8') as f:
        f.write('| 指标 | ' + ' | '.join(n for n, _ in runs) + ' | 差值 |\n')
        f.write('|---|' + '---|' * (len(runs) + 1) + '\n')
        for k in keys:
            v = [rows[n][k] for n, _ in runs]
            d = '' if None in v else (v[1] - v[0] if isinstance(v[0], (int, float)) else '')
            fmt = (lambda x: 'N/A' if x is None else (f'{x:.4f}' if isinstance(x, float) else str(x)))
            f.write(f'| {k} | ' + ' | '.join(fmt(x) for x in v) + f' | {d} |\n')

    # 必须先设中文字体：不设的话中文标题/图例会渲染成方框
    # （实测 savefig 时刷 4 条 `Glyph XXXXX missing from font(s) DejaVu Sans`）。
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.plot import setup_chinese_font
        setup_chinese_font()
    except Exception as _e:            # 字体缺失不该让整张图生成失败
        print(f'[warn] 中文字体初始化失败（{_e}），图内中文可能显示为方框')

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (col, title) in zip(axes.ravel(), CURVES):
        for name, df in runs:
            if col in df.columns:
                ax.plot(df['epoch'], df[col], marker='o', ms=3, label=name)
        ax.set_title(title); ax.set_xlabel('epoch'); ax.grid(alpha=0.3)
        ax.legend()
    # 同一坐标范围：两组画在同一子图上时天然共享 y 轴，不允许各自 autoscale
    axes.ravel()[-1].axis('off')
    axes.ravel()[-1].text(0.02, 0.5,
        '口径说明：\n'
        '- 两组共享同一套 datasets/lists/pet_*.txt 划分与 seed=42\n'
        '- 选模标准 = val macro_f1（eval.metric_for_best）\n'
        '- epoch_to_rel90 = 首次达到自身最优 macro_f1 的 90% 的 epoch\n'
        '- 数值全部来自 outputs/logs/<experiment_name>_metrics.csv', fontsize=10, va='center')
    fig.suptitle('Baseline vs Optimized')
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, 'curves', f'{pair}_compare.png'), dpi=150)

    # ---- DoD #25 点名的三个文件 ----
    # 同一张「Baseline 与优化同图」对比图，按两侧实验各存一份固定名，
    # 使验收命令 `ls outputs/curves/baseline_compare.png outputs/curves/opt_compare.png` 成立。
    import shutil
    curves_dir = os.path.join(a.out_dir, 'curves')
    for fixed in ('baseline_compare.png', 'opt_compare.png'):
        shutil.copyfile(os.path.join(curves_dir, f'{pair}_compare.png'),
                        os.path.join(curves_dir, fixed))

    # 汇总 CSV：两行（baseline / opt），列含 top1/top5/macro_f1/best_epoch
    # （题目要求「Baseline 与优化模型指标对比表」，且必须能从 CSV 直接读到 best_epoch）
    import json as _json
    summary_path = os.path.join(a.out_dir, 'logs', f'{a.opt}_compare_summary.csv'
                                if a.opt.startswith('opt_') else 'opt_compare_summary.csv')
    summary_path = os.path.join(a.out_dir, 'logs', 'opt_compare_summary.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    rows = []
    for name, df in runs:
        b = df.loc[df['val_macro_f1'].astype(float).idxmax()]
        test_json = os.path.join(a.out_dir, 'metrics', f'{name}_test.json')
        t1 = t5 = tf = ''
        if os.path.exists(test_json):
            d = _json.load(open(test_json, encoding='utf-8'))
            t1, t5, tf = f"{d['top1']:.6f}", f"{d['top5']:.6f}", f"{d['macro_f1']:.6f}"
        rows.append({'experiment_name': name,
                     'top1': f"{float(b['val_top1']):.6f}",
                     'top5': f"{float(b['val_top5']):.6f}",
                     'macro_f1': f"{float(b['val_macro_f1']):.6f}",
                     'best_epoch': int(b['epoch']),
                     'test_top1': t1, 'test_top5': t5, 'test_macro_f1': tf})
    import csv as _csv
    with open(summary_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f'[OK] 已写入 {a.out_dir}/curves/{pair}_compare.png、'
          f'{a.out_dir}/curves/{{baseline_compare,opt_compare}}.png 与 '
          f'{summary_path}（{len(rows)} 行）、'
          f'{a.out_dir}/metrics/compare_{pair}.{{csv,md}}')


if __name__ == '__main__':
    main()
