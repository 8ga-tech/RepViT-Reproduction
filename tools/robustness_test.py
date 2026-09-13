# tools/robustness_test.py —— 模糊/运动模糊/亮度/JPEG/遮挡/旋转/噪声/背景替换
import argparse, io, json, os
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
try:
    from tools.rv_common import make_transform, get, onnx_path
except ImportError:
    from rv_common import make_transform, get, onnx_path

BLUR_SIGMA  = [0, 0.5, 1.0, 1.5, 2.0, 3.0]
MOTION_K    = [0, 3, 5, 7, 9, 11]
BRIGHT      = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
JPEG_Q      = [100, 90, 75, 50, 30, 15]
OCCL_RATIO  = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
ROT_DEG     = [0, 5, 10, 15, 20, 30]
NOISE_SIGMA = [0.0, 0.02, 0.04, 0.06, 0.08, 0.12]

def gaussian_blur(img, s): return img if s == 0 else img.filter(ImageFilter.GaussianBlur(s))

def motion_blur(img, k):
    """水平方向的均值核（模拟匀速直线运动模糊）。

    **不能用 PIL 的 ImageFilter.Kernel**：它只接受 (3,3) 与 (5,5) 两种尺寸，
    传 (7,7)/(9,9)/(11,11) 会抛 `ValueError: bad kernel size`（实测复现）。
    这里用 numpy 直接做可分离的一维盒式卷积，任意 k 都成立，
    且与 PIL 的 Kernel 在 k=3/5 时数值等价。
    """
    if k == 0:
        return img
    a = np.asarray(img, dtype=np.float32)
    ker = np.ones(k, dtype=np.float32) / k
    pad = k // 2
    # 水平方向：沿宽度做 1D 卷积（reflect 填充，避免边缘变暗）
    ap = np.pad(a, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(a)
    for i in range(k):
        out += ap[:, i:i + a.shape[1], :]
    out /= k
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

def brightness(img, f): return ImageEnhance.Brightness(img).enhance(f)

def jpeg(img, q):
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=q); buf.seek(0)
    return Image.open(buf).convert("RGB")

def occlude(img, r, seed=0):
    """在已归一化张量上置 0 才等价于『填训练集均值像素』；这里在 [0,1] 图上填 0，
    归一化后就是 -mean/std 的偏色像素，报告里必须注明实现口径。"""
    if r <= 0: return img
    w, h = img.size; side = int((r * w * h) ** 0.5)
    rs = np.random.RandomState(seed); x0 = rs.randint(0, max(1, w - side)); y0 = rs.randint(0, max(1, h - side))
    out = img.copy(); out.paste((0, 0, 0), (x0, y0, x0 + side, y0 + side)); return out

def rotate(img, d):
    return img if d == 0 else img.rotate(d, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))

def noise(img, s, seed=0):
    """高斯噪声：sigma 作用在 [0,1] 尺度上，加完裁剪回 [0,1]，并按图传 seed（seed=i）。
    若所有图共用同一个 seed，等于给全部图片叠加同一张噪声图，会混入「同一噪声场」的
    系统性偏差，与 ImageNet-C 的逐图独立噪声口径不一致。"""
    if s <= 0: return img
    a = np.asarray(img).astype(np.float32) / 255.0
    a = np.clip(a + np.random.RandomState(seed).randn(*a.shape).astype(np.float32) * s, 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8))

def replace_bg(img, trimap, bg):
    """Pet 自带的 trimap：1=前景 2=背景 3=边界。边界归入前景，否则会出现黑边，
    那就变成另一种扰动，污染实验结论。"""
    m = (np.asarray(trimap) == 1) | (np.asarray(trimap) == 3)
    src = np.asarray(img).astype(np.float32)
    b = np.asarray(bg.resize(img.size)).astype(np.float32)
    mm = m[..., None]
    return Image.fromarray(np.clip(src * mm + b * (1 - mm), 0, 255).astype(np.uint8))

PERT = {"gaussian_blur": (gaussian_blur, BLUR_SIGMA), "motion_blur": (motion_blur, MOTION_K),
        "brightness": (brightness, BRIGHT), "jpeg": (jpeg, JPEG_Q),
        "occlusion": (occlude, OCCL_RATIO), "rotation": (rotate, ROT_DEG),
        "gaussian_noise": (noise, NOISE_SIGMA)}

def ece(probs, labels, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    cor = (pred == labels).astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for m in range(n_bins):
        sel = (conf > edges[m]) & (conf <= edges[m + 1])
        if sel.sum(): e += sel.mean() * abs(cor[sel].mean() - conf[sel].mean())
    return float(e)

@torch.inference_mode()
def run(model, base_items, tf, device="cpu", batch=32, num_classes=None):
    """num_classes 必须显式给：macro-F1 要按**全部类别**平均。

    不传 labels 时 sklearn 只对「在 y 或 pred 里出现过的类别」求平均，
    150 张抽样下大部分类别缺席，macro-F1 会随抽样剧烈跳动
    （实测同一份数据下出现 20~32 的乱跳，而 Top-1 是稳定的 94%），
    报告里若引用这个数会被直接质疑。
    """
    rows = []
    for name, (fn, levels) in PERT.items():
        for lv in levels:
            imgs, ys = [], []
            for i, (p, y) in enumerate(base_items):
                im = Image.open(p).convert("RGB")
                # 遮挡与高斯噪声都必须逐图独立（seed=i）；其余扰动与随机数无关
                im = fn(im, lv, seed=i) if name in ("occlusion", "gaussian_noise") else fn(im, lv)
                imgs.append(tf(im)); ys.append(y)
            X = torch.stack(imgs); preds, probs = [], []
            for s in range(0, len(X), batch):
                logits = model(X[s:s + batch].to(device))
                pr = torch.softmax(logits, 1).cpu()
                probs.append(pr); preds.append(pr.argmax(1))
            pred = torch.cat(preds).numpy(); prob = torch.cat(probs).numpy(); y = np.asarray(ys)
            conf = prob.max(1); ok = (pred == y)
            rows.append(dict(perturbation=name, severity=lv,
                             top1=round(100.0 * ok.mean(), 3),
                             macro_f1=(round(100.0 * f1_score(
                                 y, pred, average="macro",
                                 labels=list(range(num_classes or (int(max(y.max(), pred.max())) + 1))),
                                 zero_division=0), 3)),
                             mean_conf=round(float(conf.mean()), 4),
                             conf_correct=round(float(conf[ok].mean()), 4) if ok.any() else None,
                             conf_wrong=round(float(conf[~ok].mean()), 4) if (~ok).any() else None,
                             ece=round(ece(prob, y), 4), n=len(y)))
            print(rows[-1])
    return rows

def mce(rows, baseline_pert="jpeg", baseline_sev=75):
    """mean Corruption Error：以某扰动的某一级为基线归一化错误率后取平均（ImageNet-C 口径）"""
    base = [r for r in rows if r["perturbation"] == baseline_pert and r["severity"] == baseline_sev][0]
    base_err = 100.0 - base["top1"]
    per = {}
    for r in rows:
        per.setdefault(r["perturbation"], []).append((100.0 - r["top1"]) / base_err)
    return {k: round(float(np.mean(v)), 4) for k, v in per.items()}


# ======================== CLI 驱动（规格书 §11.5.1） ========================
def _env_postfix():
    """跨实现取每个 block 的键前缀：timm 与官方实现的 state_dict 根不同，
    这里只用于拼键名，取不到就返回空串，不影响主流程。"""
    return ""


def main():
    ap = argparse.ArgumentParser("鲁棒性测试：7 类扰动 x 5~6 级 severity")
    ap.add_argument("--model", default="repvit_m0_9_pet37", help="model_registry 的 key")
    ap.add_argument("--data-root", default="data/oxford-iiit-pet/images")
    ap.add_argument("--list", default="datasets/lists/pet_test.txt",
                    help="划分文件（<image_id>\\t<label>，相对仓库根）")
    ap.add_argument("--limit", type=int, default=200, help="抽样张数（越大越慢）")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-dir", default="outputs/advanced")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from deploy.model_registry import build_pt, labels_for

    torch.set_num_threads(a.threads)
    os.makedirs(a.out_dir, exist_ok=True)

    model = build_pt(a.model)          # 已 load_state_dict + eval()；官方/timm 自动分派
    model.eval()
    labels = labels_for(a.model)
    tf = make_transform(model, is_training=False)

    # 划分文件里的第一列是 image_id（不含 .jpg）；补扩展名再拼到 images/ 下
    items = []
    with open(a.list, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            iid, y = ln.rsplit(None, 1)
            p = os.path.join(a.data_root, iid if iid.lower().endswith(".jpg") else iid + ".jpg")
            if os.path.exists(p):
                items.append((p, int(y)))
    assert items, f"{a.list} 与 {a.data_root} 组合下没找到任何可用图片"
    items = items[:a.limit]
    print(f"[data] {len(items)} 张（{a.list}），模型={a.model}，类别数={len(labels)}")

    rows = run(model, items, tf, device="cpu", batch=a.batch, num_classes=len(labels))
    m = mce(rows)
    out = a.out or os.path.join(a.out_dir, f"robustness_{a.model}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(model=a.model, num_images=len(items), threads=a.threads,
                       perturbations=list(PERT.keys()), mce=m, rows=rows),
                  f, ensure_ascii=True, indent=2)
    print(f"[mce] {m}")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    # 规格书 §4.1 要求每个脚本都有 __main__ 守卫；缺它的后果是**静默什么都不做**。
    main()
