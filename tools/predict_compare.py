"""tools/predict_compare.py
固定同一批测试图片与索引，输出 Baseline 与优化模型的 Top-1 / Top-5 并排对比图。
用法：
  python tools/predict_compare.py --baseline baseline --opt opt_combo \
         --num 16 --seed 42 --out-dir outputs
"""
import argparse, json, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, torch
from PIL import Image
from torchvision import transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
sys.path.insert(0, '.')
from utils.config import load_yaml as load_config   # utils.config 的冻结 API 是 load_yaml
from models.build_model import build_model

ID_FILE = 'outputs/predictions/pet_compare_ids.json'   # 一次性生成并提交，禁止每次运行重生成


def read_list(path):
    """冻结的划分行格式：<image_id>\\t<class_idx_0based>，image_id 不含 .jpg。"""
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                image_id, idx = line.rsplit(None, 1)     # 统一 rsplit，image_id 里可能有空格
                rows.append((image_id, int(idx)))
    return rows


def build_eval_transform(cfg):
    a = cfg['aug']['val']
    return T.Compose([
        T.Resize(a['resize_size'], interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(a['crop_size']),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


def fixed_ids(test_ids, num, seed):
    """固定同一批图片：文件存在则直接读，保证两次运行、两台机器上都是同一批。"""
    if os.path.isfile(ID_FILE):
        return json.load(open(ID_FILE, encoding='utf-8'))['indices']
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(test_ids), size=num, replace=False).tolist())
    os.makedirs(os.path.dirname(ID_FILE), exist_ok=True)
    json.dump({'seed': seed, 'num': num, 'indices': idx, 'image_ids': [test_ids[i] for i in idx]},
              open(ID_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'[OK] 已生成固定对比集 {ID_FILE}，请提交到仓库，后续运行不再改动')
    return idx


@torch.no_grad()
def predict(model, rows, idx, root, tf, device, topk=5):
    """对指定索引逐张预测（batch=1，与部署口径一致），返回 Top-K 类别与置信度。"""
    model.eval()
    out = []
    for i in idx:
        image_id, y = rows[i]
        img = Image.open(os.path.join(root, 'images', f'{image_id}.jpg')).convert('RGB')
        logits = model(tf(img).unsqueeze(0).to(device))
        p = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()
        top = np.argsort(-p)[:topk]
        out.append({'index': int(i), 'image_id': image_id, 'target': int(y),
                    'topk_idx': top.tolist(), 'topk_p': p[top].tolist()})
    return out


def load_run(exp, device):
    cfg = load_config(f'outputs/logs/{exp}_config_effective.yaml')   # 用落盘的生效快照，不用原 yaml
    model = build_model(cfg)                       # 内部已按 distillation=False 建单头
    ckpt = torch.load(f'checkpoints/{exp}_best.pt', map_location='cpu', weights_only=False)  # 同上
    model.load_state_dict(ckpt['model'])           # state 里只有一套权重，选模键是 best_val_macro_f1
    print(f'[INFO] {exp}: epoch={ckpt.get("epoch")} '
          f'best_val_macro_f1={ckpt.get("best_val_macro_f1")}')
    model.to(device)
    return model, cfg, build_eval_transform(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--opt', required=True)
    ap.add_argument('--num', type=int, default=16)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default='outputs')
    a = ap.parse_args()
    pair = f'{a.baseline}_vs_{a.opt}'
    out_dir = os.path.join(a.out_dir, 'predictions')
    os.makedirs(out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    m0, cfg, tf = load_run(a.baseline, device)
    rows = read_list(cfg['data']['test_list'])     # 只看 test 段，且只在这一步之后才用到 test
    class_names = [l.strip() for l in open(cfg['data']['class_names'], encoding='utf-8') if l.strip()]
    root = cfg['data']['root']
    idx = fixed_ids([r[0] for r in rows], a.num, a.seed)
    res0 = predict(m0, rows, idx, root, tf, device)
    del m0
    m1, _, _ = load_run(a.opt, device)
    res1 = predict(m1, rows, idx, root, tf, device)

    stats = {'both_correct': 0, 'both_wrong': 0,
             'fixed_by_opt': 0, 'broken_by_opt': 0}
    rec = []
    fig, axes = plt.subplots(len(idx), 3, figsize=(11, 2.6 * len(idx)))
    for r, (i, r0, r1) in enumerate(zip(idx, res0, res1)):
        assert r0['index'] == r1['index'] == i
        c0, c1 = r0['topk_idx'][0], r1['topk_idx'][0]
        gt = r0['target']
        d = ('both_correct' if c0 == gt and c1 == gt else
             'both_wrong' if c0 != gt and c1 != gt else
             'fixed_by_opt' if c0 != gt else 'broken_by_opt')
        stats[d] += 1
        rec.append({'index': i, 'image_id': r0['image_id'],
                    'target': int(gt), 'target_name': class_names[gt],
                    'baseline': {'top1': class_names[c0], 'p': round(r0['topk_p'][0], 4),
                                 'top5': [[class_names[j], round(p, 4)]
                                          for j, p in zip(r0['topk_idx'], r0['topk_p'])]},
                    'optimized': {'top1': class_names[c1], 'p': round(r1['topk_p'][0], 4),
                                  'top5': [[class_names[j], round(p, 4)]
                                           for j, p in zip(r1['topk_idx'], r1['topk_p'])]},
                    'delta': d})
        img = Image.open(os.path.join(root, 'images', f"{r0['image_id']}.jpg")).convert('RGB')
        axes[r, 0].imshow(img); axes[r, 0].axis('off')
        axes[r, 0].set_title(f"#{r0['image_id']} GT={class_names[gt]}", fontsize=8)
        for col, rr, lab in ((1, r0, a.baseline), (2, r1, a.opt)):
            names = [class_names[j] for j in rr['topk_idx']][::-1]
            probs = rr['topk_p'][::-1]
            color = 'tab:green' if rr['topk_idx'][0] == gt else 'tab:red'
            axes[r, col].barh(names, probs, color=color)
            axes[r, col].set_xlim(0, 1); axes[r, col].tick_params(labelsize=7)
            axes[r, col].set_title(f"{lab}  Top1={class_names[rr['topk_idx'][0]]} "
                                   f"({rr['topk_p'][0]:.3f})", fontsize=8, color=color)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{pair}_predict_compare.png'), dpi=150)
    json.dump({'num': len(idx), 'seed': a.seed, 'runs': [a.baseline, a.opt],
               'summary': stats, 'records': rec},
              open(os.path.join(out_dir, f'{pair}_predict_compare.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print(f'[OK] {out_dir}/{pair}_predict_compare.png / .json  '
          f'共{len(idx)}张：都对{stats["both_correct"]} 优化修复{stats["fixed_by_opt"]} '
          f'优化引入错误{stats["broken_by_opt"]} 都错{stats["both_wrong"]}')


if __name__ == '__main__':
    main()
