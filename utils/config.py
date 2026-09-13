# utils/config.py
"""Source: Self-written（参数集参照 THU-MIG/RepViT main.py 的 argparse 重新组织为 YAML）"""
from __future__ import annotations
import argparse, copy, json, os
from pathlib import Path
from typing import Any
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并：字典递归，其余类型直接替换。
    必须深合并，否则子配置里的 train: {aug: xxx} 会把底座整个 train 段覆盖掉。"""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict:
    """加载 YAML；若含 `_base_` 字段，先递归加载父配置再深合并。"""
    path = Path(path).resolve()
    if not path.exists():
        raise SystemExit(f"[错误] 找不到配置文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("_base_", None)          # 必须 pop，否则会残留进生效配置
    if base_ref:
        base_ref = [base_ref] if isinstance(base_ref, str) else base_ref
        merged: dict = {}
        for b in base_ref:
            b_path = Path(b) if os.path.isabs(b) else (path.parent / b)
            merged = _deep_merge(merged, load_yaml(b_path))
        cfg = _deep_merge(merged, cfg)
    cfg["_config_path"] = str(path)
    return cfg


def _auto_cast(s: str) -> Any:
    """'32'->int, '0.1'->float, 'true'->bool, 'none'->None。"""
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def get(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    node = cfg
    for k in dotted_key.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def apply_overrides(cfg: dict, items: list[str]) -> dict:
    for item in items:
        if "=" not in item:
            raise SystemExit(f"[错误] --set 必须形如 key=value，收到: {item}")
        k, v = item.split("=", 1)
        set_nested(cfg, k.strip(), _auto_cast(v.strip()))
    return cfg


def build_config(argv: list[str] | None = None) -> dict:
    """命令行入口：--cfg 指定 YAML，--set 做点号路径覆盖。"""
    p = argparse.ArgumentParser("RepViT reproduction")
    p.add_argument("--cfg", required=True)
    p.add_argument("--set", nargs="*", default=[], dest="set",
                   help="点号路径覆盖，例: optim.lr=0.0005 train.epochs=50")
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print(f"[warn] 未识别的参数已忽略: {unknown}")
    cfg = load_yaml(args.cfg)   # argparse 的 --cfg 会存成 args.cfg；写成 args.config 必 AttributeError
    apply_overrides(cfg, args.set)
    cfg["_cmdline"] = "python " + " ".join(argv or [])
    return cfg


def resolve_paths(cfg: dict, root: str | Path = ".") -> dict:
    """把相对路径统一成相对仓库根目录的 POSIX 字符串（不转绝对路径，避免写死机器）。"""
    root = Path(root).resolve()
    cfg["output"]["dir"] = Path(cfg["output"]["dir"]).as_posix()
    cfg["eval"]["checkpoint_dir"] = Path(cfg["eval"]["checkpoint_dir"]).as_posix()
    cfg["eval"]["log_dir"] = Path(cfg["eval"]["log_dir"]).as_posix()
    cfg["eval"]["metric_dir"] = Path(cfg["eval"]["metric_dir"]).as_posix()
    cfg["data"]["root"] = Path(cfg["data"]["root"]).as_posix()
    cfg["_repo_root"] = str(root.as_posix())
    return cfg


def dump_effective_config(cfg: dict, out_dir: str | Path | None = None,
                          experiment_name: str | None = None) -> dict[str, Path]:
    """把"实际生效配置"写进 outputs/logs/<experiment_name>_config_effective.{yaml,json}，
    使任何一条曲线都能反查到它属于哪次运行。"""
    experiment_name = experiment_name or cfg["experiment_name"]
    out_dir = Path(out_dir or cfg["eval"]["log_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ys = out_dir / f"{experiment_name}_config_effective.yaml"
    js = out_dir / f"{experiment_name}_config_effective.json"
    with open(ys, "w", encoding="utf-8") as f:
        f.write(f"# 启动命令: {cfg.get('_cmdline','')}\n# 来源配置: {cfg.get('_config_path','')}\n")
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")},
                       f, allow_unicode=True, sort_keys=False)
    with open(js, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=True, indent=2, default=str)
    return {"yaml": ys, "json": js}
