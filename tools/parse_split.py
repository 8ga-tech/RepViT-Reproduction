# tools/parse_split.py（核心函数，兼容两种格式并自检）
import argparse, json, os, sys

def parse_split_list(path: str):
    """解析 '路径<空白>标签' 的划分文件。兼容空格与 tab 两种分隔符，
    兼容含空格的路径；返回 (samples, stats)。"""
    samples, bad = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip('\n\r')
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            # 关键：必须用 rsplit(None, 1)。line.split() 遇到含空格路径必然切错。
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                bad.append({'lineno': lineno, 'line': line}); continue
            p, lab = parts
            try:
                samples.append((p, int(lab)))
            except ValueError:
                bad.append({'lineno': lineno, 'line': line})

    # 分隔符与编码自检。
    # 注意：不能用 `[next(f) for _ in range(min(5, sum(1 for _ in f) or 1))]`——
    # 生成器表达式里的 `sum(1 for _ in f)` 会先把文件迭代器耗尽，紧接着的 next(f)
    # 必抛 StopIteration（规格书原稿此处即为此 bug，实测复现后修正）。
    # 正确做法：整份读进来后取前 5 行。
    with open(path, 'r', encoding='utf-8') as f:
        all_lines = f.read().splitlines()
    head = all_lines[:5]
    sep = 'tab' if any('\t' in h for h in head) else 'space'
    labels = [l for _, l in samples]
    stats = {
        'file': path, 'sep_detected': sep, 'n': len(samples),
        'n_bad_lines': len(bad), 'bad_examples': bad[:5],
        'label_min': min(labels) if labels else None,
        'label_max': max(labels) if labels else None,
        'n_unique_labels': len(set(labels)),
        'label_is_zero_based_hint': (min(labels) == 0) if labels else None,
    }
    return samples, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True)
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--num-classes', type=int, default=None)
    a = ap.parse_args()
    samples, stats = parse_split_list(a.list)
    if a.stats:
        print(json.dumps(stats, ensure_ascii=True, indent=2))
    # 硬性失败条件：任何一条都无法解释时，报错退出，不许带病继续
    if stats['n_bad_lines'] > 0:
        sys.exit(f"FATAL: {stats['n_bad_lines']} 行无法解析，见 bad_examples")
    if a.num_classes is not None and stats['n_unique_labels'] > a.num_classes:
        sys.exit(f"FATAL: 唯一标签数 {stats['n_unique_labels']} > 类别数 {a.num_classes}")
    # 判定型报警（不退出，但必须写进 PROGRESS）
    if a.num_classes and stats['label_max'] is not None and stats['label_max'] == a.num_classes:
        print('WARN: label_max == num_classes，疑似 1-based，请用 ImageFolder.class_to_idx 反查确认')

if __name__ == '__main__':
    main()
