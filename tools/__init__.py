# tools/__init__.py
# -*- coding: utf-8 -*-
"""工具脚本包（命令行入口集合）。

本包内的模块**全部是可独立运行的脚本**（各自带 ``if __name__ == '__main__':`` 守卫），
因此这里刻意不 import 任何子模块：
  * 避免 `import tools` 时连锁触发 torch / timm / matplotlib 的重型导入；
  * 避免子模块之间（如 tools.eval_pretrained 与 tools.run_all_pretrained）产生隐式循环依赖；
  * 允许单独执行 `python tools/xxx.py --cfg ...` 时不产生副作用。

用法一律是「先 `--cfg` 指定 YAML，再用 `--set k=v` 做点号路径覆盖」，
与 tools/train.py 保持同一套配置规则（见规格书 5.2）。
"""

__all__ = [
    "viz_style",            # 中文字体与 matplotlib 后端（被 watermark.py 依赖）
    "eval_pretrained",      # 官方预训练权重评价（Top-1/Top-5/延迟/案例）
    "run_all_pretrained",   # 多型号批量评价 + summary.csv 汇总
    "evaluate",             # 自训练 ckpt 的 val/test 最终评价
    "train",                # Baseline 迁移训练
    "visualize",            # 指标可视化
]
