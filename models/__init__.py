# -*- coding: utf-8 -*-
"""models —— 本项目的模型构建层（唯一允许「建网络」的地方）。

本包共三个模块，职责严格分层：

* ``repvit_official.py`` —— 官方 THU-MIG/RepViT 实现的逐字节拷贝（含 ``replace_batchnorm``）。
* ``repvit_timm.py``     —— timm 版 RepViT 的薄封装：建网 / 换头 / 加载 / 融合 / 数参数 / 分组 lr。
* ``build_model.py``     —— 唯一模型构建入口：屏蔽 timm 与官方两套实现的键名差异
  （timm 是 ``stem.* / stages.* / head.head.*``，官方是 ``features.* / classifier.classifier.*``）。

⚠ 本文件**故意不 import 任何子模块**，这是全项目最忌讳的一件事：

    ``repvit_official.py`` 一旦被普通 import（``from models import repvit_official``）执行，
    它内部通过 ``timm.models.register_model`` 把官方类挂进 timm 注册表；注册表被覆盖后，
    同进程随后的 ``timm.create_model('repvit_m0_9', pretrained=True)`` 会抛

        TypeError: repvit_m0_9() got an unexpected keyword argument 'pretrained_cfg'

    于是「官方权重 ↔ timm 模型」的对照实验（例如 tools/reparam_verify.py 的两条路线）
    无法在同一进程内完成，且报错信息完全指不到真正的原因。

    官方实现只能经 ``models.build_model.load_official_module()`` 以 exec 方式注入私有模块名
    ``repvit_official_vendor`` 来加载。调用方请显式写全路径，例如：

        from models.build_model import build_model, build_from_spec, load_official_module
        from models.repvit_timm import create_repvit, replace_classifier, fuse_model

    因此本文件只保留 docstring 与 ``__all__``，不做任何副作用导入。
"""

__all__: list[str] = []
