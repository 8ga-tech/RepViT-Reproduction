"""tools/budget_check.py
用法：python tools/budget_check.py configs/baseline.yaml configs/opt_combo.yaml
输出：两组实验的 steps / sample_visits / compute_ratio，并断言相等。
"""
import argparse, json, math, sys
sys.path.insert(0, '.')
from utils.config import load_yaml as load_config   # utils.config 的冻结 API 是 load_yaml

EMA_WINDOW_RATIO = 0.1     # 仅当 train.ema = true 时用于反解 decay；本规程 baseline 为 false


def n_lines(path):
    """从冻结的划分清单纯计数，不加载图片。行格式 <image_id>\\t<class_idx_0based>。"""
    with open(path, encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def budget(cfg):
    d, t = cfg['data'], cfg['train']
    n = n_lines(d['train_list'])
    bs = d['batch_size']
    spe = n // bs if d['drop_last'] else math.ceil(n / bs)
    iters = spe * t['epochs']
    visits = iters * bs
    mix = 1 if float(t['mixup_prob']) > 0 else 0
    return {
        'n_train': n, 'n_val': n_lines(d['val_list']),
        'batch_size': bs, 'drop_last': d['drop_last'],
        'steps_per_epoch': spe, 'epochs': t['epochs'],
        'total_iters': iters,
        'total_sample_visits': visits,
        'dropped_per_epoch': n - spe * bs if d['drop_last'] else 0,
        'mixup_pair_factor': 1 + mix,      # 每张图每 epoch 参与 2 次混合（主 + 客），但梯度曝光次数不变
        'input_size': d['input_size'],
        'compute_ratio': (d['input_size'] / 224.0) ** 2,
        'ema_decay': (1.0 - 1.0 / (EMA_WINDOW_RATIO * iters)) if t.get('ema') else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('baseline'); ap.add_argument('optimized')
    ap.add_argument('--note', default=None, help='存在不等价项时，必须提供解释文本')
    a = ap.parse_args()
    ca, cb = load_config(a.baseline), load_config(a.optimized)
    ba, bb = budget(ca), budget(cb)

    print(f'{"字段":<22}{"A":>16}{"B":>16}{"一致":>8}')
    for k in ba:
        va, vb = ba[k], bb[k]
        if isinstance(va, float):
            va, vb = round(va, 6), round(vb, 6)
        print(f'{k:<22}{str(va):>16}{str(vb):>16}{"OK" if va == vb else "DIFF":>8}')

    hard = [k for k in ('total_iters', 'total_sample_visits', 'epochs',
                        'batch_size', 'steps_per_epoch', 'ema_decay')
            if ba[k] != bb[k]]
    soft = [k for k in ('compute_ratio',) if ba[k] != bb[k]]

    if not hard and not soft:
        print(f'[PASS] 等价训练预算：两组均为 {ba["total_iters"]} 步 / '
              f'{ba["total_sample_visits"]} 次样本见次')
        return 0
    if hard:
        print(f'[FAIL] 优化步数或样本见次次数不等价: {hard}')
        return 1
    print(f'[DIFF] 步数与样本见次次数一致，但单步计算量不同: compute_ratio='
          f'{bb["compute_ratio"]:.4f}（{bb["input_size"]}px vs {ba["input_size"]}px）')
    if not a.note:
        print('[FAIL] 未提供 --note 解释该差异')
        return 1
    print(f'[ACCEPTED with note] {a.note}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
