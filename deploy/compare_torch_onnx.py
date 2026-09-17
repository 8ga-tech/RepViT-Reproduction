# -*- coding: utf-8 -*-
"""deploy/compare_torch_onnx.py —— 固定图片列表上比较 PyTorch 与 ONNX 的 5 项指标。"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.model_registry import get, build_pt, onnx_path   # noqa
from deploy.infer_onnx import preprocess, build_session   # noqa


def repo_rel(p) -> str:
    """把路径写成「相对仓库根的 POSIX 路径」，只用于落盘元信息字段。

    试题第 14 页要求不得写死个人电脑绝对路径。本函数只影响 JSON 里的
    ``onnx_path`` / ``images`` 两个**非数值**字段；模型加载与数据读取仍用原路径。
    仓库外的路径（例如写到 %TEMP% 的临时复跑副本）无法相对化，原样返回 POSIX 形式，
    以免丢失信息。
    """
    q = Path(p).resolve()
    try:
        return q.relative_to(ROOT).as_posix()
    except ValueError:
        return q.as_posix()

def fuse_pt(reg):
    """必须与导出 ONNX 时的状态一致：build_pt() 重建未融合态，这里再做一次融合。"""
    m = build_pt(reg["name"])
    if reg["impl"] == "official":
        from models import repvit_official as official_repvit
        official_repvit.replace_batchnorm(m)
    else:
        m.fuse()
    return m.eval()


def load_list_paths(list_file: Path, reg: dict, limit: int) -> list[str]:
    """把划分文件解析成**真实存在的图片路径**。

    两个必须做对的地方：
      [1] 必须 `line.rsplit(None, 1)` 取第一段，不能把整行当路径。
          早期写法 `[l.strip() for l in ...]` 会把 'Abyssinian_2\t0' 整行传下去，
          PIL 直接抛 OSError: [Errno 22] Invalid argument（实测复现）。
          用 rsplit 而非 split 是因为路径本身可能含空格。
      [2] Pet 的 list 存的是 **image_id**（无目录、无扩展名），必须还原成
          data/oxford-iiit-pet/images/<image_id>.jpg；ImageNet 子集存的是
          相对仓库根的完整路径，直接用。
    """
    pet_images = ROOT / "data/oxford-iiit-pet/images"
    out: list[str] = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.rsplit(None, 1)[0]                     # [1] 只取路径段
        if not Path(p).exists():
            for cand in (pet_images / f"{p}.jpg", ROOT / p, ROOT / "data/imagenet/val" / p):
                if cand.exists():
                    p = str(cand)
                    break
            else:
                raise FileNotFoundError(
                    f"list 里的路径无法解析为存在的文件：{p!r}（来自 {list_file}）")
        out.append(p)
        if len(out) >= limit:
            break
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", default=None,
                    help="每行一个图片路径的 txt；默认按 num_classes 取 datasets/lists/ 下的固定列表"
                         "（37 类 -> pet_test.txt；1000 类 -> imagenetv2_mf_1000.txt，即 ImageNetV2 "
                         "matched-frequency 的 1000 张确定性固定子集，见 report/IMAGENETV2_PROVENANCE.md）")
    ap.add_argument("--limit", type=int, default=12,
                    help="默认 12，与入库的 outputs/metrics/consistency_*.json 口径一致")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    reg = get(a.model)
    if a.images:
        list_file = Path(a.images)
    else:
        list_file = ROOT / "datasets/lists" / (
            "pet_test.txt" if reg["num_classes"] == 37 else "imagenetv2_mf_1000.txt")
    if not list_file.exists():
        raise FileNotFoundError(
            f"固定列表不存在：{list_file}。1000 类型号的默认清单是 ImageNetV2 matched-frequency "
            f"的确定性 1000 张子集，用 `python datasets/make_imagenetv2_subset.py` 构建"
            f"（见 report/IMAGENETV2_PROVENANCE.md）；也可用 --images 显式指定其它列表。")
    paths = load_list_paths(list_file, reg, a.limit)
    pt, sess = fuse_pt(reg), build_session(onnx_path(a.model))
    iname = sess.get_inputs()[0].name
    dmax = 0.0; dsum = 0.0; n1 = 0; n5 = 0; mismatch = []
    print(f"EP={sess.get_providers()}  n={len(paths)}  model={reg['name']}")
    with torch.no_grad():
        for p in paths:
            x = preprocess(p, reg["input_size"], reg["mean"], reg["std"], reg["crop_pct"])
            z_pt = pt(torch.from_numpy(x)).numpy()[0]        # 同一份 numpy，隔离预处理变量
            z_on = sess.run(None, {iname: x})[0][0]
            d = np.abs(z_pt - z_on)
            dmax = max(dmax, float(d.max())); dsum += float(d.mean())
            s1 = int(z_pt.argmax() == z_on.argmax()); n1 += s1
            a5 = set(np.argsort(-z_pt)[:5].tolist()); b5 = set(np.argsort(-z_on)[:5].tolist())
            n5 += int(a5 == b5)
            if not s1:
                mismatch.append(dict(image=p, pt=int(z_pt.argmax()), onnx=int(z_on.argmax()),
                                     max_abs=float(d.max())))
    # 字段名对齐 selfcheck 的 c_consistency / c_realimg / c_labels_pair：
    #   model        -> registry key（不是 arch 名）
    #   num_classes  -> 37 / 1000
    #   label_file   -> 实际使用的标签文件，37 类必须是 labels/pet_classes.txt
    #   source       -> 固定划分列表的**文件名词干**，不能是随机张量；与 images 一一对应，
    #                   这样 selfcheck 能反过来校验「source 指向的清单文件真的存在」
    #   top1_agree   -> selfcheck 读这个键
    #   top1_agreement -> DoD #31 的验收命令读这个键；两个都写，避免口径分歧
    _label_file = "labels/pet_classes.txt" if reg["num_classes"] == 37 else "labels/imagenet_classes.txt"
    _source = Path(list_file).stem      # pet_test / imagenetv2_mf_1000
    # 落盘元信息里的两个路径字段一律写成仓库相对路径（见 repo_rel 的注释）：
    #   onnx_path -> onnx/<registry_key>.onnx（由 deploy/model_registry.py 给出）
    #   images    -> datasets/lists/<list>.txt（由 --images 给出，缺省时取固定划分列表）
    rep = dict(model=a.model, arch=reg["name"], onnx_path=repo_rel(onnx_path(a.model)),
               num_classes=reg["num_classes"], label_file=_label_file, source=_source,
               images=repo_rel(list_file), n=len(paths),
               max_abs_logits=dmax, mean_abs_logits=dsum / max(len(paths), 1),
               top1_agree_rate=n1 / len(paths),
               top5_set_agree_rate=n5 / len(paths),
               mismatches=mismatch[:20], mismatch_count=len(mismatch),
               threshold=dict(top1_agree_rate=0.99, max_abs_logits=1e-3))
    rep["top1_agree"] = rep["top1_agree_rate"]          # selfcheck 读的名字
    rep["top1_agreement"] = rep["top1_agree_rate"]      # DoD #31 读的名字
    rep["verdict"] = "PASS" if (rep["top1_agree_rate"] >= 0.99 and dmax < 1e-3) else "FAIL"
    dst = Path(a.out) if a.out else ROOT / "outputs/metrics" / f"consistency_{reg['name']}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(rep, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"max|Δlogits|={dmax:.3e}  mean|Δlogits|={rep['mean_abs_logits']:.3e}  "
          f"Top-1 一致率={rep['top1_agree_rate']*100:.2f}%  Top-5 集合一致率="
          f"{rep['top5_set_agree_rate']*100:.2f}%  verdict={rep['verdict']}  -> {dst}")

if __name__ == "__main__":
    main()
