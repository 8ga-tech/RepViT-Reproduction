# utils/__init__.py
"""仓库公共工具包：配置、随机种子、日志与指标。

注意：这里刻意**不** import 任何子模块——utils 内部的模块之间存在交叉引用
（例如 tools/*.py 直接 `from utils.logging import ...`），在 __init__ 里提前
导入会引入循环依赖与副作用（logging.py 会创建目录、seed.py 会改全局随机状态）。
因此保持空白包，调用方一律显式 `from utils.xxx import yyy`。
"""

__all__: list[str] = []
