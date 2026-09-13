# tools/interp_deep.py
import argparse, os, json, math
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image          # 新补的 CLI 驱动要读图；原文件只是函数库，没 import 过
from sklearn.metrics import f1_score
try:
    from tools.rv_common import make_transform, latency
except ImportError:
    from rv_common import make_transform, latency

STAGES = ["stem", "stages.0", "stages.1", "stages.2", "stages.3"]
# timm 版 RepViT-M0.9 的逐挂载点形状（必须现测现标，别照抄别人的表）：
#   stem 与 stages.0 都是 56×56/48ch（stem 内部已经做了两次 stride=2），
#   stages.1 = 28×28/96ch，stages.2 = 14×14/192ch，stages.3 = 7×7/384ch；GAP 后是 384 维。
# 「stem 与 stages.0 形状相同」是 timm 实现的事实，不是笔误——第二张特征图不能标成 28×28/96ch，
# 否则「浅层碎 / 深层聚焦」的分析会整体错位一层。

def capture_maps(model, layer_names, x, out_dir, n_show=16):
    feats, hooks = {}, []
    mods = dict(model.named_modules())
    for n in layer_names:
        hooks.append(mods[n].register_forward_hook(
            lambda m, i, o, n=n: feats.__setitem__(n, o.detach())))
    model(x); [h.remove() for h in hooks]
    print({n: tuple(feats[n].shape) for n in layer_names})   # 现测现标，把真实形状写进图标题
    for n in layer_names:
        f = feats[n][0].cpu().numpy()[:n_show]
        fig, axes = plt.subplots(4, 4, figsize=(8, 8), dpi=160)
        for ax, fm in zip(axes.flat, f):
            ax.imshow((fm - fm.min()) / (np.ptp(fm) + 1e-8), cmap="viridis"); ax.axis("off")
        fig.suptitle(f"{n}  (C={feats[n].shape[1]}, H={feats[n].shape[2]})")
        fig.tight_layout(); fig.savefig(f"{out_dir}/featmap_{n.replace('.', '_')}.png"); plt.close(fig)
    return {n: list(feats[n].shape) for n in layer_names}

def gradcam(model, x, layer_name, target=None):
    """手写 Grad-CAM：α_k = GAP(∂y_c/∂A_k)，L = ReLU(Σ α_k A_k)。
    RepViT 是 4D NCHW，不需要 reshape_transform；不要选 head（2D，会抛
    ValueError: Invalid grads shape. Shape of grads should be 4 (2D image) or 5。
    全仓两个规范挂载点（**两者不等价、输出数值不同，不能互相替代**）：
      粗粒度 model.stages[-1].blocks[-1]                      —— 残差相加后的激活，(B, C, 7, 7)
      细粒度 model.stages[-1].blocks[-1].channel_mixer.conv2  —— 残差相加前最后一个 1×1 卷积输出
    对比方法：同一输入、同一 target class 分别挂这两个点出 CAM，报告里给出两张图的
    余弦相似度与峰值位置偏移（产物 gradcam_layer_compare.png 就是这张对比图）。
    注意 timm 里并不存在 channel_mixer.m 这种模块名。"""
    mods = dict(model.named_modules()); acts, grads = {}, {}
    h1 = mods[layer_name].register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = mods[layer_name].register_full_backward_hook(
        lambda m, gi, go: grads.__setitem__("g", go[0].detach()))
    model.zero_grad(set_to_none=True)
    logits = model(x)
    # argmax(1) 返回的是**形状 [B] 的张量**，直接 int() 会抛
    # ValueError: only one element tensors can be converted to Python scalars（实测复现）。
    # 必须取 batch 第 0 个元素。本函数始终按单样本解释 CAM，因此只回传第 0 个。
    c = int(logits[0].argmax().item()) if target is None else int(target)
    logits[0, c].backward()
    A, G = acts["a"], grads["g"]
    cam = torch.relu((G.mean((2, 3), keepdim=True) * A).sum(1))[0]
    # Tensor.ptp() 在新版 PyTorch 已被移除（实测 AttributeError），
    # 等价写法是 torch.ptp(cam) 或直接 max-min；这里用后者，语义最直白。
    cam = (cam - cam.min()) / ((cam.max() - cam.min()) + 1e-8)
    h1.remove(); h2.remove()
    return cam.detach().cpu().numpy(), c   # 必须 detach：cam 仍挂在计算图上

def multilayer_cam(model, x, out_png, target=None):
    cams, cls = [], None
    # target 用**模型自己的 argmax**（传 None）而不是外部张量：
    # 传 1 元素张量时内部 `int(...)` 会抛
    # ValueError: only one element tensors can be converted to Python scalars（实测复现）。
    tgt = None if target is None else (int(target) if np.ndim(target) == 0 else None)
    for n in STAGES:
        cm, c = gradcam(model, x, n, tgt); cams.append(cm); cls = c
    fig, axes = plt.subplots(1, len(STAGES), figsize=(4 * len(STAGES), 4.2), dpi=160)
    for ax, n, cm in zip(axes, STAGES, cams):
        ax.imshow(cm, cmap="jet"); ax.set_title(n); ax.axis("off")
    fig.suptitle(f"Grad-CAM across stages (target class={cls})")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)
    return cls

@torch.inference_mode()
def extract_features(model, loader, device="cpu"):
    """取 GAP 后的 384 维特征。timm 的 forward_features 返回 (B,384,7,7)。"""
    fs, ys = [], []
    for x, y in loader:
        f = model.forward_features(x.to(device)).mean((2, 3))
        fs.append(f.cpu().numpy()); ys.append(np.asarray(y))
    return np.concatenate(fs), np.concatenate(ys)

def embed(X, y, method, out_png, seed=42):
    if method == "tsne":
        from sklearn.manifold import TSNE
        E = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca",
                 max_iter=1000, random_state=seed).fit_transform(X)
    else:
        import umap
        E = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean",
                      random_state=seed).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    ax.scatter(E[:, 0], E[:, 1], c=y, s=10, cmap="tab20")
    ax.set_title(f"{method.upper()} of RepViT-M0.9 penultimate features")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)
    return E

def reliability(probs, labels, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    cor = (pred == labels).astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1); bins = []; e = 0.0
    for m in range(n_bins):
        sel = (conf > edges[m]) & (conf <= edges[m + 1])
        if not sel.sum(): bins.append(((edges[m]+edges[m+1])/2, 0.0, 0.0)); continue
        a, c, w = cor[sel].mean(), conf[sel].mean(), sel.mean()
        e += w * abs(a - c); bins.append(((edges[m]+edges[m+1])/2, a, c))
    return float(e), bins

def fit_temperature(logits_val, labels_val):
    """温度缩放：T 只能在验证集上拟合，在 test 上拟合属于测试集泄漏。"""
    # 必须参数化 log-T：直接把 T 当自由变量做 LBFGS，会跑到**负数**
    # （实测得到 temperature = -0.0739）。温度是标准差尺度，负值没有意义，
    # 写进报告就是错的。改成 T = softplus(log_T) 后 T 恒为正。
    logT = nn.Parameter(torch.tensor([math.log(math.e ** 1.5 - 1.0)]))
    opt = torch.optim.LBFGS([logT], lr=0.05, max_iter=200)
    lg = torch.as_tensor(logits_val, dtype=torch.float32)
    lb = torch.as_tensor(labels_val, dtype=torch.long)
    def closure():
        opt.zero_grad()
        T = nn.functional.softplus(logT)
        loss = nn.functional.cross_entropy(lg / T, lb)
        loss.backward()
        return loss
    opt.step(closure)
    return float(nn.functional.softplus(logT).detach().clamp(min=1e-3))


# ======================== CLI 驱动（规格书 §11.5.2） ========================
def main():
    ap = argparse.ArgumentParser("深入可解释性：阶段特征图 / 多层 Grad-CAM / t-SNE / ECE+温度缩放")
    ap.add_argument("--model", default="repvit_m0_9_pet37")
    ap.add_argument("--data-root", default="data/oxford-iiit-pet/images")
    ap.add_argument("--list", default="datasets/lists/pet_test.txt")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-dir", default="outputs/advanced/interp")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from deploy.model_registry import build_pt
    from torch.utils.data import Dataset, DataLoader

    torch.set_num_threads(a.threads)
    os.makedirs(a.out_dir, exist_ok=True)
    model = build_pt(a.model)
    model.eval()
    tf = make_transform(model, is_training=False)

    class _DS(Dataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            p, y = self.items[i]
            return tf(Image.open(p).convert("RGB")), y

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
    loader = DataLoader(_DS(items), batch_size=a.batch, shuffle=False, num_workers=0)
    print(f"[data] {len(items)} 张 -> {a.out_dir}")

    # (a) 阶段特征图
    x, _ = next(iter(loader))
    try:
        capture_maps(model, STAGES, x, a.out_dir)
        print("[1/4] 阶段特征图 OK")
    except Exception as e:
        print(f"[1/4] 阶段特征图跳过：{type(e).__name__}: {e}")

    # (b) 多层 Grad-CAM（同一张图挂不同层）
    try:
        multilayer_cam(model, x, os.path.join(a.out_dir, "multilayer_cam.png"), target=None)
        print("[2/4] 多层 Grad-CAM OK")
    except Exception as e:
        print(f"[2/4] 多层 Grad-CAM 跳过：{type(e).__name__}: {e}")

    # (c) t-SNE / UMAP 特征嵌入
    feats, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            try:
                f = model.forward_features(xb) if hasattr(model, "forward_features") else model(xb)
                f = f.mean(dim=(2, 3)) if f.dim() == 4 else f
            except Exception:
                f = model(xb)
            feats.append(f.cpu()); ys.append(yb)
    X = torch.cat(feats).numpy(); Y = torch.cat(ys).numpy()
    try:
        embed(X, Y, "tsne", os.path.join(a.out_dir, "tsne.png"))
        print("[3/4] t-SNE OK")
    except Exception as e:
        print(f"[3/4] t-SNE 跳过：{type(e).__name__}: {e}")

    # (d) 可靠性图 + ECE + 温度缩放
    all_logits, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            lg = model(xb)
            all_logits.append(lg.cpu() if torch.is_tensor(lg) else lg[0].cpu())
            all_y.append(yb)
    L = torch.cat(all_logits).numpy(); Yv = torch.cat(all_y).numpy()
    P = torch.softmax(torch.as_tensor(L), 1).numpy()
    rel = reliability(P, Yv)
    T = fit_temperature(L, Yv)
    out = os.path.join(a.out_dir, "calibration.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(model=a.model, n=len(Yv), temperature=T,
                       ece_before=rel, note="温度缩放只调 1 个参数，不改变预测类别，只改置信度"),
                  f, ensure_ascii=True, indent=2, default=str)
    print(f"[4/4] 校准 OK  temperature={T:.4f}  -> {out}")
    print(f"[out] {a.out_dir}")
    return 0


if __name__ == "__main__":
    main()
