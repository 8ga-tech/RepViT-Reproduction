# deploy/__init__.py
# -*- coding: utf-8 -*-
"""部署相关脚本包（ONNX 导出 / 推理 / 基准测试 / 摄像头演示）。

与 tools/ 包同样刻意不 import 任何子模块：
  * deploy/infer_onnx.py 与 deploy/demo_camera.py 里有 onnxruntime、cv2（可选软依赖），
    若在包初始化时导入，任何 `import deploy.xxx` 都会被迫吞下这些重依赖；
  * deploy/export_onnx.py 会向 stdlib 的 sys.path 注入仓库根，导入顺序敏感。

模型注册表的唯一真源是 `deploy/model_registry.py`（`get` / `keys` / `onnx_path` /
`labels_for` / `build_pt` / `MODELS` / `ONNX_DIR`），其余脚本一律经它取路径，
禁止在代码里硬写 ONNX 文件名与权重文件名。
"""

__all__ = [
    "model_registry",      # 型号注册表：MODELS / get / keys / onnx_path / build_pt
    "export_onnx",         # PyTorch -> ONNX（onnx/<registry_key>.onnx）
    "infer_onnx",          # ONNX Runtime 单图推理
    "benchmark",           # ONNX Runtime 延迟基准（含元信息水印）
    "compare_torch_onnx",  # PyTorch 与 ONNX 逐样本一致率比对
    "demo_camera",         # 摄像头实时演示（cv2 为可选软依赖）
]
