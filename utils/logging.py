# utils/logging.py
"""训练日志三件套 + 环境采集 + metrics JSON 落盘。

三份产物各司其职：
  - TensorBoard：现场浏览，`Loss/train` 与 `Loss/val` 自动归到同一面板；
  - CSV：`pandas.read_csv` 画曲线、写报告表格；
  - JSONL：追加写、崩溃后仍可读（CSV 半行损坏会整份读不进，JSONL 只丢最后一行）。

本模块名叫 logging，但**不要**在里面 `from utils.logging import ...`；Python 3 是绝对
导入，模块内的 `import logging` 拿到的是标准库 logging，不是自己，这是刻意保留的行为。
"""
from __future__ import annotations

import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _os.path.normcase(_os.path.abspath(_sys.path[0] or ".")) == _os.path.normcase(_HERE):
    # `python utils/logging.py` 时本文件会被当成顶层脚本，随后 `import logging`
    # 又会把同一份文件当标准库 logging 再加载一遍（两个模块对象），行为诡异。
    # 摘掉 sys.path[0] 后 `import logging` 命中标准库；用 `python -m utils.logging`
    # 或正常 `from utils.logging import ...` 时这里是 no-op。
    _sys.path.pop(0)

import csv
import datetime
import json
import logging
import os
import platform
import subprocess  # noqa: F401  (保留：便于后续在 meta 里追加 git rev-parse)
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 契约：全仓 CSV 表头的唯一真源。列名与顺序不得改，改了下游 pandas 脚本全部错位。
# ---------------------------------------------------------------------------
FIELDS = ["epoch", "train_loss", "train_acc1", "val_loss", "val_top1", "val_top5",
          "val_macro_f1", "lr", "epoch_time_sec"]

# 指标名 -> TensorBoard tag。同名段（Loss/、Accuracy/）会被 TB 自动合并到同一面板。
TB_TAG = {
    "train_loss": "Loss/train",
    "val_loss": "Loss/val",
    "train_acc1": "Accuracy/train_top1",
    "val_top1": "Accuracy/val_top1",
    "val_top5": "Accuracy/val_top5",
    "val_macro_f1": "Accuracy/val_macro_f1",
    "lr": "LR",
}


def _to_jsonable(v: Any) -> Any:
    """numpy / torch 标量 -> python 原生类型，保证 json.dumps 不炸。"""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if hasattr(v, "item"):          # np.float64 / np.int64 / torch scalar
        try:
            return v.item()
        except Exception:
            pass
    if hasattr(v, "tolist"):        # np.ndarray
        try:
            return v.tolist()
        except Exception:
            pass
    return str(v)


class TrainLogger:
    """逐 epoch 记录训练指标，同时落 CSV / JSONL / TensorBoard。

    产物路径固定为 ``<out_dir>/logs/<experiment_name>_metrics.{csv,jsonl}``；
    out_dir 默认 "outputs"，与 CANON 的目录约定一致。
    """

    def __init__(self, out_dir: str | Path = "outputs", experiment_name: str = "exp",
                 enable_tb: bool = True, enable_csv: bool = True,
                 enable_jsonl: bool = True) -> None:
        self.out_dir = Path(out_dir)
        self.experiment_name = str(experiment_name)
        self.enable_csv = bool(enable_csv)
        self.enable_jsonl = bool(enable_jsonl)

        self.log_dir = self.out_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"{self.experiment_name}_metrics.csv"
        self.jsonl_path = self.log_dir / f"{self.experiment_name}_metrics.jsonl"

        self._n = 0                       # 内部 epoch 计数器（调用方没给 epoch 时兜底）
        self._csv_f = None
        self._csv_w = None
        self._header_written = False
        self._tb = None
        self.enable_tb = False

        if self.enable_csv:
            # 追加模式：实验中断后重启时接着写，不覆盖已完成的历史 epoch。
            new_file = (not self.csv_path.exists()) or self.csv_path.stat().st_size == 0
            self._csv_f = open(self.csv_path, "a", encoding="utf-8", newline="")
            self._csv_w = csv.DictWriter(self._csv_f, fieldnames=FIELDS,
                                         extrasaction="ignore")
            self._header_written = not new_file   # 已有文件视为表头已在

        if enable_jsonl:
            # 追加写：崩溃后逐行 json.loads 依然可用
            self._jsonl_f = open(self.jsonl_path, "a", encoding="utf-8")
        else:
            self._jsonl_f = None

        if enable_tb:
            # TensorBoard 是可选依赖：缺包/端口冲突/任何异常都必须降级，不能让训练崩掉。
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.out_dir / "tensorboard" / self.experiment_name
                tb_dir.mkdir(parents=True, exist_ok=True)
                self._tb = SummaryWriter(log_dir=str(tb_dir))
                self.enable_tb = True
            except Exception as exc:
                self.enable_tb = False
                self._tb = None
                print(f"[TrainLogger] 警告: TensorBoard 不可用（{type(exc).__name__}: {exc}），"
                      f"已降级为 enable_tb=False，CSV/JSONL 不受影响。")

    # -- 写入 ---------------------------------------------------------------
    def log_epoch(self, **metrics) -> None:
        """写一行 CSV + 一行 JSONL + 若干 TB scalar；每个 epoch 必须 flush。

        默认 flush_secs=120，训练中途 Ctrl+C 会丢掉最后两分钟的曲线，
        所以这里每个 epoch 都显式 flush()。
        """
        self._n += 1
        row: dict[str, Any] = {}
        for k in FIELDS:                       # 先按契约 FIELDS 顺序填
            if k in metrics:
                row[k] = _to_jsonable(metrics[k])
        for k, v in metrics.items():           # 额外字段（如 loss_ce）追加在后面
            if k not in row:
                row[k] = _to_jsonable(v)
        row["epoch"] = int(metrics.get("epoch", self._n))

        if self._csv_w is not None:
            if not self._header_written:
                self._csv_w.writeheader()
                self._header_written = True
            self._csv_w.writerow(row)
            self._csv_f.flush()                # 每 epoch 刷盘

        if self._jsonl_f is not None:
            self._jsonl_f.write(json.dumps(row, ensure_ascii=True) + "\n")
            self._jsonl_f.flush()

        if self._tb is not None:
            for key, tag in TB_TAG.items():
                if key in metrics and metrics[key] is not None:
                    self._tb.add_scalar(tag, float(metrics[key]), global_step=row["epoch"])
            self._tb.flush()

    def close(self) -> None:
        if self._csv_f is not None:
            try:
                self._csv_f.flush()
                self._csv_f.close()
            finally:
                self._csv_f = None
                self._csv_w = None
        if self._jsonl_f is not None:
            try:
                self._jsonl_f.flush()
                self._jsonl_f.close()
            finally:
                self._jsonl_f = None
        if self._tb is not None:
            try:
                self._tb.flush()
                self._tb.close()
            finally:
                self._tb = None

    # 支持 with 用法：with TrainLogger("outputs", "B0") as lg: lg.log_epoch(...)
    def __enter__(self) -> "TrainLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def get_logger(name: str, logfile: str | Path | None = None) -> logging.Logger:
    """同时输出 stdout 与（给了 logfile 时）文件的 logger。

    重复调用同一个 name 不会叠加 handler（否则日志会翻倍）。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if logfile is not None:
        logfile = Path(logfile)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        target = str(logfile.resolve())
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == target
                   for h in logger.handlers):
            fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def capture_env() -> dict:
    """采集环境指纹，写进 metrics JSON 用于溯源。"""
    def _ver(mod_name: str) -> str:
        try:
            mod = __import__(mod_name)
            return str(getattr(mod, "__version__", "unknown"))
        except Exception as exc:
            return f"not-installed ({type(exc).__name__})"

    return {
        "python": sys.version.split()[0],
        "torch": _ver("torch"),
        "timm": _ver("timm"),
        "onnxruntime": _ver("onnxruntime"),
        "platform": platform.platform(),
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
    }


# ---------------------------------------------------------------------------
# 以下 dump_metrics 逐字照抄执行规格书 §4.1(c)（第 282-304 行），不要改字段名。
# ---------------------------------------------------------------------------
def dump_metrics(name: str, payload: dict, out_dir: str = 'outputs/metrics') -> str:
    """把实验数字写成 json。强制附带可溯源元信息。"""
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        'experiment': name,
        'timestamp': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'command': ' '.join(sys.argv),                  # 采集命令，报告里直接引用
        'cwd_rel': '.',                                  # 一律相对仓库根
        'host': platform.node(),
        'platform': platform.platform(),
        'python': sys.version.split()[0],
    }
    full = {**meta, **payload}
    path = os.path.join(out_dir, f'{name}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(full, f, ensure_ascii=True, indent=2)
    print(f'[metrics] wrote {path}')
    return path


if __name__ == "__main__":
    # 自检：写一个临时实验，验证 CSV 表头 == FIELDS、JSONL 可逐行解析、TB 缺失不崩
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="logselftest_"))
    with TrainLogger(tmp, "selftest", enable_tb=True) as lg:
        for ep in (1, 2):
            lg.log_epoch(epoch=ep, train_loss=1.0 / ep, train_acc1=60.0 + ep,
                         val_loss=0.9 / ep, val_top1=0.70 + 0.01 * ep,
                         val_top5=0.90, val_macro_f1=0.65, lr=1e-3,
                         epoch_time_sec=12.5)

    with open(tmp / "logs" / "selftest_metrics.csv", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == FIELDS, f"CSV 表头与 FIELDS 不一致: {rows[0]}"
    assert len(rows) == 3, f"应有 表头+2 行，实际 {len(rows)}"

    with open(tmp / "logs" / "selftest_metrics.jsonl", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    assert len(recs) == 2 and recs[-1]["val_top1"] == 0.72

    dump_metrics("_selftest_dump", {"acc": 0.5, "env": capture_env()},
                 out_dir=str(tmp / "metrics"))
    print("[logging] 自检通过 ->", tmp)
