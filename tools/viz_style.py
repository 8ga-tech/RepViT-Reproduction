# tools/viz_style.py
# -*- coding: utf-8 -*-
"""绘图全局样式：中文字体链 + 负号修复 + 强制 Agg 后端。

Source: Self-written（字体探测逻辑与 utils/plot.py 的 setup_chinese_font 同源，
        但本模块是**被 tools/watermark.py 直接依赖的底座**，必须独立存在且无重依赖：
        watermark.py 第 12 行 `from tools.viz_style import setup_style` 是硬依赖）。

为什么必须把中文字体插到 plt.rcParams['font.sans-serif'] 的**最前面**：
matplotlib 的字体回退是「按列表顺序取第一个能渲染该字形的字体」，而 DejaVu Sans
（matplotlib 自带的默认字体）并不包含 CJK 字形；只要 DejaVu 排在中文前面，
中文就一定是方框（豆腐块），并且 matplotlib 不会报错——属于典型的「静默错」。

为什么 matplotlib.use("Agg") 必须在 import matplotlib.pyplot **之前**调用：
pyplot 在 import 时就把后端实例化了；之后再切换后端，轻则

    UserWarning: FigureCanvasAgg is non-interactive ...

重则在无显示设备（CI / 远程 shell / 答辩现场投屏前的无头机）上直接抛
`ImportError: Cannot load backend 'TkAgg'`，整个绘图脚本崩掉。
本模块在 import pyplot 前无条件切到 Agg（`Agg` 是纯文件后端，不需要 DISPLAY）。
"""
from __future__ import annotations

import sys

import matplotlib

# ---------------------------------------------------------------------------
# 1) 后端：必须在 import pyplot 之前。若 pyplot 已被别处 import 进来，这里只做一次
#    「尽力而为」的切换（用 try 包住，避免污染调用方的进程状态）。
# ---------------------------------------------------------------------------
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")            # pyplot 尚未导入 -> 可直接生效
else:                                # pragma: no cover - 取决于调用顺序
    try:
        matplotlib.use("Agg")
    except Exception:                # noqa: BLE001 - 后端已实例化时忽略
        pass

import matplotlib.pyplot as plt      # noqa: E402  （必须在 use('Agg') 之后）
from matplotlib import font_manager  # noqa: E402

# 首选字体：Windows 自带 Microsoft YaHei / SimHei / SimSun；Linux 装 fonts-noto-cjk
DEFAULT_PREFER = ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC")
# 备选：命中不到首选时的补充项（黑体优先于衬线，图表标题更清晰）
EXTRA_CJK = ("Source Han Sans SC", "Noto Sans SC", "WenQuanYi Zen Hei",
             "Arial Unicode MS", "PingFang SC")

_ready = False


def installed_fonts() -> set[str]:
    """返回本机 matplotlib 能识别到的字体名集合（一次全量扫描，结果不做缓存）。"""
    return {f.name for f in font_manager.fontManager.ttflist}


def setup_style(prefer: tuple[str, ...] = DEFAULT_PREFER) -> list[str]:
    """设置全局绘图样式，返回**实际命中**的中文字体名列表（空列表 = 无中文字体）。

    调用方约定：所有绘图脚本的第一行调用本函数（见 tools/watermark.py::save_fig）。
    """
    global _ready
    installed = installed_fonts()
    chosen = [n for n in prefer if n in installed]
    extra = [n for n in EXTRA_CJK if n in installed and n not in chosen]
    if not chosen and not extra:
        print("[warn] tools/viz_style: 未找到任何中文字体，中文将显示为方框。"
              "Windows 自带 'Microsoft YaHei'/'SimHei'；"
              "Linux 请 `apt install fonts-noto-cjk`。")
    # 关键：中文族系的字体必须排在 DejaVu Sans 之前，否则 DejaVu 永远先命中 -> 方框
    plt.rcParams["font.sans-serif"] = chosen + extra + ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False     # 否则负号 "−" 变方框
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    _ready = True
    return chosen


def ensure_style() -> list[str]:
    """惰性入口：只在第一次调用时真正设置样式（供库函数内部调用，不再重复打印）。"""
    if not _ready:
        return setup_style()
    return list(plt.rcParams["font.sans-serif"])


def close_all() -> None:
    """等价 plt.close('all')：循环画几十张图（如 37 类 per-class 图）时必须调用，
    否则 figure 句柄累积会触发 RuntimeWarning 并吃掉内存。"""
    plt.close("all")


if __name__ == "__main__":
    hit = setup_style()
    print(f"[font] 实际命中中文字体: {hit}")
    print(f"[font] font.sans-serif 链: {plt.rcParams['font.sans-serif'][:5]}")
    print(f"[font] axes.unicode_minus = {plt.rcParams['axes.unicode_minus']}")
    print(f"[backend] matplotlib backend = {matplotlib.get_backend()}")
    close_all()
    print("[ok] tools/viz_style.py 可独立运行")
