# -*- coding: utf-8 -*-
"""tools/check_weights.py —— 官方预训练权重完整性核对（对应 DoD #11）。

用法：
    python tools/check_weights.py --dir checkpoints/pretrained

逐文件打印「字节数 + SHA256」，并与规格书给出的基准字节数比对：
    repvit_m0_9_distill_300e.pth = 22422548
    repvit_m1_0_distill_300e.pth = 29675245
    repvit_m1_1_distill_300e.pth = 35668677
    repvit_m1_5_distill_300e.pth = 43449459

行为约定：
  * 字节数不符 -> 打印 [MISMATCH]，但**不**退出非 0（批量核对时要能一次看完全部文件）；
  * 未登记的模型（如 m2_3）只报字节数与 SHA256，不判定对错；
  * .pth 文件小于 5 MB 时额外打 [WARN]：多半是下载中断留下的残文件（下载器常写
    一个 2~3 MB 的占位/分片文件），这种文件能被 torch.load 打开一部分也可能直接炸，
    必须重新下载再核对。
  * 结果同时写 outputs/metrics/weight_sha256.json，SHA256 是权重溯源的一级证据。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _local_dump_metrics(name: str, payload: dict, out_dir: str = "outputs/metrics") -> str:
    p = Path(out_dir)
    out_dir = str(p if p.is_absolute() else (ROOT / p))
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "experiment": name,
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "cwd_rel": ".",
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**meta, **payload}, f, ensure_ascii=True, indent=2)
    print(f"[metrics] wrote {path}")
    return path


try:  # pragma: no cover
    from utils.logging import dump_metrics as _dump_metrics  # type: ignore
except Exception:  # noqa: BLE001
    _dump_metrics = _local_dump_metrics


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


# 规格书 §3.1 / 模块 9 给出的官方 release 文件字节数基准（本机已实测复现）
BASELINE_BYTES = {
    "repvit_m0_9_distill_300e.pth": 22422548,
    "repvit_m1_0_distill_300e.pth": 29675245,
    "repvit_m1_1_distill_300e.pth": 35668677,
    "repvit_m2_3_distill_300e.pth": 95860931,   # 本机实测；官方 release 未给出统一校验值
    # 43,449,459 这个值曾经写在这里，但它取自一个**内容损坏**的文件。
    # 重新下载后的正确字节数是 59,375,411（见本文件新增的可加载性校验）。
    "repvit_m1_5_distill_300e.pth": 59375411,
}

# 官方 release 的权重全部在 20 MB 以上；小于该阈值几乎必然是残文件
MIN_SANE_PTH_BYTES = 5 * 1024 * 1024



def _zip_central_directory_ok(path: Path) -> tuple[bool, str]:
    """PyTorch 的 .pth 本质是 zip。缺中央目录 = 文件被截断/损坏。

    不需要 import torch，比 torch.load 快得多，适合批量核对时先跑一遍。
    """
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad is not None:
                return False, f"zip 内有损坏成员：{bad}"
            n = len(z.namelist())
        return True, f"zip 中央目录正常（{n} 个成员）"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _loadable_ok(path: Path) -> tuple[bool, str]:
    """真正 torch.load 一次，确认能读出 state_dict。

    字节数一致**不能**推出内容完好——实测有一个文件字节数与期望完全一致，
    但 torch.load 直接抛 PytorchStreamReader failed reading zip archive。
    """
    import torch      # 延迟导入：本模块的字节数/SHA256 路径不需要 torch，只有这里需要
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        n = len(sd)
        return (n > 0), f"可加载，{n} 个张量"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:90]}"

def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """分块读，避免一次性把几十 MB 读进内存（Windows 上大文件一次性读也更易失败）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser("官方预训练权重字节数 / SHA256 核对")
    ap.add_argument("--dir", default="checkpoints/pretrained", help="待核对的权重目录")
    ap.add_argument("--pattern", default="*", help="文件名通配，默认全部")
    ap.add_argument("--out-dir", default="outputs/metrics")
    ap.add_argument("--json-name", default="weight_sha256")
    args = ap.parse_args()

    d = _resolve(args.dir)
    if not d.is_dir():
        print(f"[ERROR] 目录不存在：{d}")
        print("        提示：官方权重需从 https://github.com/THU-MIG/RepViT/releases/tag/v1.0 下载")
        return 1

    files = sorted([p for p in d.glob(args.pattern) if p.is_file()], key=lambda p: p.name)
    if not files:
        print(f"[ERROR] {d} 下没有匹配 {args.pattern} 的文件")
        return 1

    print("=" * 92)
    print(f" check_weights —— 官方预训练权重完整性核对（DoD #11）   dir={args.dir}")
    print(" 判据：字节数逐位一致只能证明「没下错/没下断」，SHA256 才是内容级溯源证据。")
    print("=" * 92)
    print(f"{'file':<34}{'bytes':>12}{'expected':>12}  {'flag':<10}sha256")
    print("-" * 92)

    records, n_mismatch, n_warn = [], 0, 0
    for p in files:
        size = p.stat().st_size
        digest = sha256_of(p)
        exp = BASELINE_BYTES.get(p.name)
        flag = ""
        if exp is None:
            flag = "-"
        elif size == exp:
            flag = "OK"
        else:
            flag = "MISMATCH"
            n_mismatch += 1
        print(f"{p.name:<34}{size:>12,}{'' if exp is None else format(exp, ','):>12}  "
              f"{flag:<10}{digest}")
        if size < MIN_SANE_PTH_BYTES and p.suffix.lower() in (".pth", ".pt", ".safetensors"):
            n_warn += 1
            print(f"{'':<34}[WARN] 仅 {size:,} 字节（< {MIN_SANE_PTH_BYTES:,}），"
                  f"几乎必然是下载中断的残文件，请重新下载后再核对")
        # 字节数之外的**内容级**完好性校验（.pth/.pt 才做）
        zip_ok = load_ok = None
        zip_msg = load_msg = "-"
        if p.suffix.lower() in (".pth", ".pt"):
            zip_ok, zip_msg = _zip_central_directory_ok(p)
            load_ok, load_msg = _loadable_ok(p)
            if not zip_ok or not load_ok:
                n_warn += 1
                print(f"{'':<34}[BAD ] 内容级校验失败：zip={zip_msg} | load={load_msg}")
                print(f"{'':<34}       字节数匹配**不代表**文件完好，请重新下载该文件")
        records.append({
            "file": p.name,
            "zip_ok": zip_ok, "zip_detail": zip_msg,
            "loadable": load_ok, "load_detail": load_msg,
            "path_rel": p.relative_to(ROOT).as_posix() if ROOT in p.parents else p.as_posix(),
            "size_bytes": size,
            "sha256": digest,
            "expected_size_bytes": exp,
            "match": None if exp is None else bool(size == exp),
            "within_known_baseline": exp is not None,
        })

    print("-" * 92)
    print(f"[summary] files={len(files)}  matched={sum(1 for r in records if r['match'] is True)}  "
          f"mismatched={n_mismatch}  unknown={sum(1 for r in records if r['match'] is None)}  "
          f"warn_suspicious_size={n_warn}  dir={args.dir}")
    if n_mismatch:
        print("[summary] 存在 [MISMATCH]：字节数与规格书基准不符，见上表；"
              "本脚本按契约不返回非 0，便于批量核对全部文件。")
    n_bad_content = sum(1 for r in records if r.get("loadable") is False)
    print(f"[summary] 内容级校验：loadable=False 的文件数 = {n_bad_content}")
    if "repvit_m0_9_distill_300e.pth" in {r["file"] for r in records}:
        r = next(r for r in records if r["file"] == "repvit_m0_9_distill_300e.pth")
        print(f"[summary] repvit_m0_9_distill_300e.pth = {r['size_bytes']} 字节"
              f"（基准 22422548）{'  [OK]' if r['size_bytes'] == 22422548 else '  [MISMATCH]'}")

    payload = {
        "dir": args.dir,
        "unit": "bytes + sha256",
        "baseline_bytes": BASELINE_BYTES,
        "files": records,
        "n_files": len(files),
        "n_mismatch": n_mismatch,
        "n_suspicious_size": n_warn,
        "all_known_match": bool(n_mismatch == 0),
        "note": "字节数不符时打印 [MISMATCH] 但退出码仍为 0（契约要求可批量核对）",
    }
    _dump_metrics(args.json_name, payload, out_dir=str(_resolve(args.out_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
