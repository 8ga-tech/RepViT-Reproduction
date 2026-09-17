# PROVENANCE —— 代码与资产来源标注

题目第 15 页要求候选人明确标记：**官方代码 / 第三方代码 / 自己修改的代码 / 自己新增的代码 /
AI 辅助生成或修改的内容**。本文件是这份标注的唯一真源。

---

## 一、官方代码（official，逐字节拷贝）

| 文件 | 来源 | 许可 | 改动 |
|---|---|---|---|
| `models/repvit_official.py` | [THU-MIG/RepViT](https://github.com/THU-MIG/RepViT) 的 `model/repvit.py` | Apache-2.0 | 仅 3 处必要适配，见下 |

原始文件校验：`md5 = 4c861e5eaacb72c0db78588b10827e9f`，16,962 字节。

三处适配（其余逐字节保留）：

1. `from timm.models.layers import SqueezeExcite` → `from timm.layers import SqueezeExcite`
   （timm 1.x 已废弃旧路径）
2. `from timm.models.vision_transformer import trunc_normal_` → `from timm.layers import trunc_normal_`
   （同上）
3. **删除 `from timm.models import register_model` 与全部 6 个 `@register_model` 装饰器**。
   原因：该文件一旦被正常 import 就会覆盖 timm 注册表里的 `repvit_m0_9` 等条目，
   之后同进程内 `timm.create_model('repvit_m0_9', pretrained=True)` 会抛
   `TypeError: repvit_m0_9() got an unexpected keyword argument 'pretrained_cfg'`。
   工厂函数本体全部保留，只去掉注册副作用。加载一律经
   `models/build_model.py::load_official_module()`（exec + 恒等装饰器，双保险）。

文件末尾追加的 `replace_batchnorm()` 同样来自官方仓库根目录的 `utils.py`（Apache-2.0），
逐字节保留，是官方导出推理态模型的唯一入口。

---

## 二、第三方代码与资产（third-party）

| 资产 | 来源 | 许可 | 用途 |
|---|---|---|---|
| `timm` 的 RepViT 实现 | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | Apache-2.0 | 训练与部署的主线实现 |
| 官方预训练权重 ×5 | THU-MIG/RepViT Releases v1.0 | Apache-2.0 | 官方模型评价 |
| timm/HF 预训练权重 | `timm/repvit_m0_9.dist_300e_in1k` | Apache-2.0 | 迁移训练起点 |
| Oxford-IIIT Pet 数据集 | University of Oxford | CC BY-SA 4.0 | 迁移训练 |
| **ImageNetV2 matched-frequency 测试集** | `vaishaal/ImageNetV2`（HF 作者方镜像，`gated=false`） | MIT（HF `cardData.license`；图像本体按 ImageNet 数据集同等方式使用） | **官方模型评价（唯一口径）**：归档 `imagenetv2-matched-frequency.tar.gz` = 1,264,079,360 B，sha256 `f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c`（与 HF `X-Linked-ETag` 逐字符一致）；固定子集清单 `datasets/lists/imagenetv2_mf_1000.txt`。出处与选取规则见 `report/IMAGENETV2_PROVENANCE.md` |
| ImageNet-1K 验证集（原始上游数据） | `mrm8488/ImageNet1K-val`（HF 镜像，14 分片） | 非商业研究用途 | `external/` 跨集合实拍演示图的来源；**不参与官方模型评价口径**（本仓唯一评测口径是上一行的 ImageNetV2） |
| `torchvision` / `torch` / `onnxruntime` / `scikit-learn` / `matplotlib` / `pandas` / `OpenCV` | 各自官方发布 | BSD-3 / MIT / Apache-2.0 等 | 通用依赖 |

**代理声明**：本机直连 `github.com` 返回 HTTP 000，官方权重经
`https://ghfast.top/https://github.com/...` 前置代理取得；字节数与官方 Release 逐位一致，
SHA256 记录在 `outputs/metrics/weight_sha256.json`。
`www.robots.ox.ac.uk` 不可达，Pet 归档经 HF 镜像取得，**已用官方 MD5 证明为原始文件**。

---

## 三、自撰代码（self-written）

`tools/`、`deploy/`、`utils/`、`datasets/` 下的全部 `.py`，以及 `configs/` 下的全部 `.yaml`，
均为本项目自撰（不含逐字节拷贝的第三方实现，官方代码见第一节）。

**「来源类别」的唯一真源是本文件**。文件头 `Source:` 注释只覆盖一部分文件，
清点结果（2026-09，`git ls-files tools/*.py deploy/*.py utils/*.py datasets/*.py configs/*.yaml`
共 **88** 个文件）：**20** 个文件头带 `Source:` 注释（如 `tools/report_spec.py` 的
`Source: Self-written`、`tools/make_report_assets.py` 的 `Source  : Self-written`），
其余 68 个（含 `configs/*.yaml` 与各 `__init__.py`）未逐文件加注释。
未加注释的文件不影响来源判定：它们全部是自撰代码，来源按本文件第一节到第四节的分类标注。
"
关键自撰模块：

| 文件 | 职责 |
|---|---|
| `tools/train.py` | Baseline 与优化实验共用的训练入口（自包含） |
| `tools/eval_pretrained.py` | 官方权重评价（Top-1/5、参数量、MACs、延迟、案例） |
| `tools/reparam_verify.py` | 结构重参数化数值/结构验证 |
| `tools/gradcam.py` | **手写 Grad-CAM**（不依赖 pytorch-grad-cam，满足「独立实现」） |
| `tools/selfcheck.py` | 全局自检（DoD 逐条） |
| `deploy/model_registry.py` | 型号 → 权重/标签/类别数/ONNX 路径的**单一事实来源** |
| `deploy/infer_onnx.py` | **独立实现**的预处理 + softmax + Top-K |
| `datasets/unpack_imagenet_val.py` | 把 parquet 分片还原成 ImageFolder |
| `datasets/build_pet_class_map.py` | Pet 37 类映射（与 torchvision 逐项交叉校验） |

---

## 四、AI 辅助生成或修改的内容

本项目在 Claude Code 辅助下完成。按题目要求如实标注：

**AI 参与的部分**
- `tools/`、`deploy/`、`utils/`、`datasets/` 下多数脚本的**初稿**
- 配置文件的字段与注释、本 README 与报告的文字组织
- 调试过程中的报错定位与修复建议

**人类负责的部分**
- 任务范围、模型选型（RepViT-M0.9 + M1.0）、优化方案（方案 A 组合式）的**决策**
- 全部实验的**实际执行与结果确认**；所有进入报告的数字均来自本机实跑
- 对 AI 产出代码的逐处复核

**AI 初稿中被实测推翻、并由人工修正的典型缺陷**（均已在代码注释中标明原因，
详见 `PROGRESS.md`）：

| # | 位置 | 缺陷 | 后果 |
|---|---|---|---|
| 1 | `tools/parse_split.py` | `sum(1 for _ in f)` 先耗尽迭代器，再 `next(f)` | `StopIteration`，脚本完全不可用 |
| 2 | `tools/assert_data.py` | 未排除 macOS `._*` 资源叉文件；`ids()` 未跳过 `#` 注释行 | trimaps 计数翻倍；误报「划分与 list.txt 不一致」 |
| 3 | `tools/visualize.py` | 用 `class_idx < 12` 判猫 | 官方 CLASS-ID **猫狗交错**，物种判定整体错位 |
| 4 | `tools/visualize.py` | 跨物种错误率分母把 cross 重复计了一次 | 任何输入都恒等于 50%，指标失去意义 |
| 5 | `tools/reparam_verify.py` | `zip(tk.tolist(), ...)` 后再 `.tolist()` | `AttributeError` |
| 6 | `tools/train.py` | `load_config` 不解析 `_base_` | 全部 `opt_*.yaml` 无法运行 |
| 7 | `deploy/compare_torch_onnx.py` | 未 `rsplit` 解析 list 行，整行当路径 | `OSError: Invalid argument` |
| 8 | `tools/selfcheck.py` | 以 `weights_only=True` 读自训 ckpt；`fuse()` 返回值当模块用；URL 被误判为绝对路径 | 三项自检恒 FAIL |
| 9 | 多处 JSON 落盘 | `ensure_ascii=False` | 本仓库根目录含中文，写进 JSON 的路径让 `open()`（GBK）读不回来 |
| 10 | `tools/env_check.py` | venv 缺失被判为致命错误 | 与本任务「使用全局解释器」的人类决策冲突 |

---

## 五、数据与资产的校验记录

| 资产 | 校验方式 | 结果 |
|---|---|---|
| 官方权重 ×5 | 字节数 + SHA256 | 5/5 与官方 Release 一致 |
| Pet `images.tar.gz` | 官方 MD5 `5c4f3ee8e5d25df40f4fd59a7f44e54c` | 一致 |
| Pet `annotations.tar.gz` | 官方 MD5 `95a8c909bbe2e81eed6a22bccdf3f68f` | 一致 |
| ImageNet 1000 类标签 | 四锚点 tench / golden retriever / tabby / toilet tissue | 全部命中 |
| **ImageNetV2 归档 + 固定子集** | 归档体积 + SHA256（与 HF `X-Linked-ETag` 逐字符比对）；解压后 1000 类 × 10 张；清单 1000 行 | 归档 1,264,079,360 B / `f0c37fdf…c9ca7c` 一致；清单 sha256 `d5532205d8f30099aa70ed9392078adcdbd305c47055930c66e704f7976c9d05` |
| ImageNet val（原始上游数据） | 1000 类 × 50 张、JPEG 魔数、目录序 = synset 序 | 通过（**不参与评测口径**，见第二节） |
| Pet 划分 | 三层两两交集 = 0、MD5 交集 = 0 | 通过 |

---

## 六、外部图片的许可说明

`external/` 下 8 张图片来自 ImageNet-1K 验证集（跨集合真实照片），用于
「训练集以外的实际图片」这一要求。**这不是实拍原创照片**：执行时
`commons.wikimedia.org` 与 `upload.wikimedia.org` 均不可达（curl 超时，HTTP 000），
规格书 §8.6.1 的首选（自拍）与次选（Wikimedia）都无法取得，故采用跨数据集真实照片作为替代，
并在 `external/images_manifest.csv` 逐张登记来源、许可与类别对应关系。
本仓库不声称这些图片为原创。
