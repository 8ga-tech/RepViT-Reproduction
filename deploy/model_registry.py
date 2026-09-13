# -*- coding: utf-8 -*-
"""deploy/model_registry.py —— 全仓库唯一的「模型名 -> 部署元信息」登记表。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR, LABEL_DIR = ROOT / "onnx", ROOT / "labels"
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def _entry(name, arch, impl, ncls, labels, ckpt=None, ckpt_url=None,
           source="", note=""):
    """字段固定 15 个。crop_pct / distillation 由 num_classes 推导，再由 _check() 硬校验。"""
    return dict(
        name=name, arch=arch, impl=impl, num_classes=ncls, labels=labels,
        crop_pct=0.875 if ncls == 37 else 0.95,
        input_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD,
        distillation=0 if ncls == 37 else 1,
        ckpt=ckpt, ckpt_url=ckpt_url, onnx=f"{name}.onnx",
        source=source, note=note)

_RELEASE = "https://github.com/THU-MIG/RepViT/releases/download/v1.0"

MODELS = {e["name"]: e for e in [
    # ---- 基础必做 1：官方 M0.9，ImageNet-1K 1000 类（timm _cfg: crop_pct=0.95, bicubic）----
    _entry("repvit_m0_9_in1k", "repvit_m0_9", "official", 1000, "imagenet_classes.txt",
           ckpt_url=f"{_RELEASE}/repvit_m0_9_distill_300e.pth",
           source="THU-MIG/RepViT Releases v1.0",
           note="基础必做①。eval()+replace_batchnorm() 后导出，图内 BatchNormalization=0、Conv=103。"),
    # ---- 基础必做 2：另一官方型号（M1.0，家族对比用）----
    _entry("repvit_m1_0_in1k", "repvit_m1_0", "official", 1000, "imagenet_classes.txt",
           ckpt_url=f"{_RELEASE}/repvit_m1_0_distill_300e.pth",
           source="THU-MIG/RepViT Releases v1.0",
           note="基础必做②。与 M0.9 同预处理、同 EP、同线程，仅换权重。"),
    # ---- 基础必做 3：自训练 Pet-37（timm 实现，单头 384->37）----
    _entry("repvit_m0_9_pet37", "repvit_m0_9", "timm", 37, "pet_classes.txt",
           ckpt="checkpoints/baseline_best.pt", source="本任务自训练 Baseline",
           note="基础必做③。37 类单头（distillation=0），标签必须用 pet_classes.txt，严禁与 ImageNet 标签混用。"),
    # ---- 进阶：模型家族速度—精度分析（累计 ≥4 个官方型号）----
    _entry("repvit_m1_1_in1k", "repvit_m1_1", "official", 1000, "imagenet_classes.txt",
           ckpt_url=f"{_RELEASE}/repvit_m1_1_distill_300e.pth",
           source="THU-MIG/RepViT Releases v1.0", note="进阶①家族扫描。"),
    _entry("repvit_m1_5_in1k", "repvit_m1_5", "official", 1000, "imagenet_classes.txt",
           ckpt_url=f"{_RELEASE}/repvit_m1_5_distill_300e.pth",
           source="THU-MIG/RepViT Releases v1.0", note="进阶①家族扫描。"),
    _entry("repvit_m2_3_in1k", "repvit_m2_3", "official", 1000, "imagenet_classes.txt",
           ckpt_url=f"{_RELEASE}/repvit_m2_3_distill_300e.pth",
           source="THU-MIG/RepViT Releases v1.0", note="进阶①家族扫描，最大型号。"),
    # ---- 进阶：优化模型的独立部署闭环 ----
    _entry("repvit_m0_9_pet37_opt", "repvit_m0_9", "timm", 37, "pet_classes.txt",
           ckpt="checkpoints/opt_randaug_best.pt", source="本任务自训练优化实验",
           note="进阶：优化实验模型的独立 ONNX（experiment_name 按实际所选项同步）。"),
]}

def _check(e: dict) -> None:
    """硬规则：违反直接 raise，不允许「静默丢蒸馏头 / 用错 crop_pct」流到导出阶段。"""
    if e["num_classes"] == 37:
        assert e["distillation"] == 0, f"{e['name']}: 37 类条目必须 distillation=0"
        assert e["crop_pct"] == 0.875, f"{e['name']}: 37 类条目必须 crop_pct=0.875"
        assert e["ckpt"] and str(e["ckpt"]).endswith("_best.pt"), \
            f"{e['name']}: Pet 条目 ckpt 必须指向 checkpoints/<experiment_name>_best.pt"
    else:
        assert e["distillation"] == 1, f"{e['name']}: 1000 类条目必须 distillation=1"
        assert e["crop_pct"] == 0.95, f"{e['name']}: 1000 类条目必须 crop_pct=0.95"

for _e in MODELS.values():
    _check(_e)

def get(key: str) -> dict:
    """取登记项。未登记的 key 直接报错并列出可用项，其它脚本不得直接读 MODELS。"""
    if key not in MODELS:
        raise KeyError(f"未登记的模型 {key!r}；可用：{sorted(MODELS)}")
    return dict(MODELS[key])

def keys() -> list[str]:
    return sorted(MODELS)

def onnx_path(key: str) -> str:
    """ONNX 路径一律由此函数给出：正文、验收与脚本都不得硬写 *.onnx 文件名。"""
    return str(ONNX_DIR / get(key)["onnx"])

def labels_for(key: str) -> list[str]:
    """读取标签文件并做行数自检，行数 != num_classes 立即断言失败。"""
    e = get(key)
    p = LABEL_DIR / e["labels"]
    assert p.exists(), f"标签文件缺失：{p}（应为 labels/imagenet_classes.txt 或 labels/pet_classes.txt）"
    labels = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(labels) == e["num_classes"], \
        f"{p} 行数 {len(labels)} != num_classes {e['num_classes']}"
    return labels

def build_pt(key: str):
    """重建 PyTorch 参考模型（未融合态）。顺序不可交换：load_state_dict -> eval。"""
    import torch
    e = get(key)
    if e["impl"] == "official":
        from models import repvit_official as M
        m = getattr(M, e["arch"])(num_classes=e["num_classes"],
                                  distillation=bool(e["distillation"]))
    else:
        import timm
        m = timm.create_model(e["arch"], pretrained=False, num_classes=e["num_classes"],
                              distillation=bool(e["distillation"]))
    if e["ckpt"]:
        sd = torch.load(ROOT / e["ckpt"], map_location="cpu", weights_only=False)
    elif e["ckpt_url"]:
        sd = torch.hub.load_state_dict_from_url(e["ckpt_url"], map_location="cpu")
    else:
        raise ValueError(f"{key}: ckpt 与 ckpt_url 均为空")
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"[ckpt] missing={len(miss)} unexpected={len(unexp)}")
    assert not unexp, f"存在未匹配权重键：{unexp[:3]}，结构与权重来源不一致"
    assert not miss, f"存在缺失权重键：{miss[:3]}；蒸馏头/分类头缺失会静默变成随机模型"
    m.eval()
    return m

if __name__ == "__main__":
    print(f"{'name':<24}{'cls':>5}{'impl':>10}  exists  onnx")
    for k in keys():
        p = Path(onnx_path(k))
        ok = "OK " if p.exists() else "MISS"
        print(f"{k:<24}{get(k)['num_classes']:>5}{get(k)['impl']:>10}  [{ok}]  {p.name}")
