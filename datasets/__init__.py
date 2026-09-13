# datasets/__init__.py
# -*- coding: utf-8 -*-
"""数据集包：Oxford-IIIT Pet 37 类与考核方提供的固定 ImageNet 子集。

本包只放**数据读取**相关的代码，三条边界必须守住：

* 划分的唯一真源是 ``datasets/lists/pet_{train,val,test}.txt``（由
  ``datasets/make_pet_split.py`` 生成，行格式固定 ``<image_id>\\t<class_idx_0based>``）；
  ``datasets/pet_dataset.py`` 里的 ``make_pet_split()`` 是同一套划分的库函数版本，
  两者参数与产物路径一致，**任何其它模块不得再定义第二套划分**。
* ``datasets/imagenet_subset.py`` 只读考核方给的列表，**绝不做任何划分或采样**。
* 变换与 DataLoader 的装配唯一入口是 ``datasets.build``，其它脚本不得各写一份。

注意：这里刻意**不** import 任何子模块（与 ``utils/__init__.py`` 同一约定）——
``datasets/build.py`` 会 import ``datasets.pet_dataset`` / ``datasets.imagenet_subset``，
在 ``__init__`` 里提前导入会引入循环依赖，并让 ``import datasets`` 变成隐式的重依赖
（PIL / torch），拖慢只想读个路径常量的脚本。调用方一律显式 ``from datasets.xxx import yyy``。
"""

__all__: list[str] = []
