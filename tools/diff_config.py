"""tools/diff_config.py
用法：
  python tools/diff_config.py configs/baseline.yaml configs/opt_combo.yaml
  python tools/diff_config.py configs/baseline.yaml configs/opt_mix.yaml \
         --json outputs/metrics/opt_mix_config_diff.json
规则：所有以 '_' 开头的键（_base_/_expected_diff_/_base_path/_config_path/_config_sha256）
      属于元信息，不参与差异；实验身份键（experiment_name）单独忽略。
退出码 0 = PASS，1 = FAIL。
"""
import argparse, json, sys
sys.path.insert(0, '.')
from utils.config import load_yaml as load_config   # utils.config 的冻结 API 是 load_yaml

META_IGNORE = ('experiment_name',)


def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(flatten(v, key + '.'))
        else:
            out[key] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('baseline')
    ap.add_argument('optimized')
    ap.add_argument('--json', default=None, help='把差异清单落盘，供报告引用')
    ap.add_argument('--allow-missing', action='store_true',
                    help='声明了但未变化的键只告警不失败')
    args = ap.parse_args()

    ca, cb = load_config(args.baseline), load_config(args.optimized)
    fa = {k: v for k, v in flatten(ca).items()
          if not k.startswith('_') and k not in META_IGNORE}
    fb = {k: v for k, v in flatten(cb).items()
          if not k.startswith('_') and k not in META_IGNORE}
    expect = list(cb.get('_expected_diff_', []))
    if not expect:
        print('[WARN] 优化配置未声明 _expected_diff_，所有差异都将被视为未声明变量')

    diffs = {k: (fa.get(k, '<MISSING>'), fb.get(k, '<MISSING>'))
             for k in sorted(set(fa) | set(fb)) if fa.get(k, '<MISSING>') != fb.get(k, '<MISSING>')}
    unexpected = [k for k in diffs if k not in expect]
    missing = [k for k in expect if k not in diffs]

    print(f'[A] {args.baseline}')
    print(f'[B] {args.optimized}')
    print(f'[KEY SPACE] A={len(fa)} keys, B={len(fb)} keys')
    print(f'[DIFF COUNT] {len(diffs)} / declared {len(expect)}')
    for k, (va, vb) in diffs.items():
        tag = 'EXPECTED' if k in expect else 'UNEXPECTED'
        print(f'  {tag:11s} {k:28s} {va!r} -> {vb!r}')
    if missing:
        print(f'[WARN] 声明为差异但实际未变化: {missing}')
    if args.json:
        json.dump({'baseline': args.baseline, 'optimized': args.optimized,
                   'key_space': [len(fa), len(fb)],
                   'diffs': {k: [repr(v[0]), repr(v[1])] for k, v in diffs.items()},
                   'expected': expect, 'unexpected': unexpected, 'missing': missing},
                  open(args.json, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    if unexpected:
        print(f'[FAIL] 存在未声明的变量差异: {unexpected}')
        print('       两组不是单变量对照。要么删掉该改动，要么补进 _expected_diff_ 并在报告中解释。')
        return 1
    if missing and not args.allow_missing:
        print('[FAIL] 声明与实现不一致，可能配置未生效（如 _base_ 路径写错）')
        return 1
    print(f'[PASS] 控制变量自检通过：差异键 = 声明的 {len(expect)} 个')
    return 0


if __name__ == '__main__':
    sys.exit(main())
