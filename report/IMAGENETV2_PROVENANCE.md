# ImageNetV2 固定子集 —— 出处、许可与口径（provenance）

> 适用口径：**官方模型评价的主口径**（2026-09-16 起）。本文是唯一出处说明，
> 与 `README.md` §2.2.1、`PROGRESS.md` 决策记录、`report/SPEC13_FREEZE.json` 三处必须保持一致。
> 生成者：spec13（任务 t11）｜所有哈希与体积均为本机实测。

---

## 0. 一句话结论

官方模型评价的主口径 = **ImageNetV2 `matched-frequency` 变体中的 1000 张确定性固定子集**
（1000 类各 1 张，选取规则不依赖随机种子，任何人可 byte-for-byte 复现）。
**ImageNetV2 不是 ImageNet-1K 验证集**：它是 Recht et al. (NeurIPS 2019) 按 ImageNet 分布
**重新采样**构建的独立测试集（该变体共 1000 类 × 10 张 = 10,000 张，本次取 1000 张）。
其 Top-1 与 ImageNet-1K val 的 Top-1、与论文/官方公布值**不可直接比较**。

---

## 1. 来源与许可

| 项 | 值 |
|---|---|
| 上游项目 | `https://github.com/modestyachts/ImageNetV2`（作者方发布仓库） |
| 论文 | Recht, Roelofs, Schmidt, Shankar, **"Do ImageNet Classifiers Generalize to ImageNet?"**, NeurIPS 2019（arXiv:1902.10811） |
| 实际下载渠道 | HuggingFace 数据集 **`vaishaal/ImageNetV2`**（作者方镜像；`gated: false`、`private: false`、`license: mit`） |
| 直链（本仓实测可用） | `https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz` |
| 镜像等价直链 | `https://hf-mirror.com/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz` |
| HF 仓库快照（2026-09-16 实测） | `lastModified = 2023-03-19T17:29:43Z`；`sha = d626240be2538720e83103a0e1178d24aca8b12c`；`downloads = 13568`；`likes = 9` |
| 为什么不用官方 S3 | 官方 S3 路径 `s3.amazonaws.com/imagenetv2-public/imagenetv2-matched-frequency.tar.gz` 与 `imagenetv2-public.s3.amazonaws.com/...` **现在均返回 404**（已失效），故改走作者方 HF 镜像 |
| 许可 | **MIT**（HF `cardData.license = mit`，tags 含 `license:mit`） |

> **许可边界（逐字引用上游 README）**：`The LICENSE file does not apply to the actual image data.
> The images come from Flickr which provides corresponding license information.
> They can be used the same way as the original ImageNet dataset.`
> 即：**代码 MIT，图像本体不适用该 LICENSE**，按 ImageNet 数据集同等方式使用（研究/评测用途）。

---

## 2. 三个可用变体与本次选择理由

三个归档均在同一 HF 仓库下，**体积与 sha256 均为 2026-09-16 实测**：

| 变体（`main` 分支文件名） | 字节数 | HF `X-Linked-ETag`（= sha256） | 采样方式 |
|---|---|---|---|
| `imagenetv2-matched-frequency.tar.gz` | **1,264,079,360** | `f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c` | 按原始 ImageNet val 的 MTurk 选择频率分布**匹配**采样 ← **本次选用** |
| `imagenetv2-threshold0.7.tar.gz` | 1,249,906,176 | `32142bfe998f9cfa4141be691f6a80eeb1e27b24bd532a668eb08c26d98897a2` | 从选择频率 ≥ 0.7 的候选中每类抽 10 张 |
| `imagenetv2-top-images.tar.gz` | 1,245,927,936 | `85e4ace30f3f0d660152062254b8759dda66267b27f7ec0fd8d5768f1073a04b` | 每类选择频率最高的 10 张 |

**选择 `matched-frequency` 的理由**：它是三个变体里唯一**刻意匹配原始 ImageNet 验证集频率分布**的
一个，也是文献与社区做「ImageNet-1K vs ImageNetV2」对照时最常引用的标准变体；
`top-images` 偏向「更容易被 MTurk 一致认可的样本」，会系统性抬高准确率，不适合作为主口径；
`threshold0.7` 介于两者之间。三者都可视作同一分布下的不同难度切片，但**只能用其中一个当主口径并声明**。

---

## 3. 归档事实（实测）与两个必踩的坑

### 3.1 下载（可复制命令）

```bash
curl -L --retry 15 --retry-all-errors -C - --speed-limit 50000 --speed-time 60 \
  -o data/_src/imagenetv2/imagenetv2-matched-frequency.tar.gz \
  https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz
```

实测：HF CDN 存在**间歇性传输停滞**（速度掉到 0 并保持 20s+）。上例的
`-C -`（断点续传）+ `--speed-limit/--speed-time`（卡死即换连接）+ `--retry-all-errors` 组合可稳定完成；
本仓最后一次成功会话为「断点续传 618,192,867 B，用时 164.7 s」。

### 3.2 归档校验（本机实测值）

| 项 | 实测值 |
|---|---|
| 体积 | **1,264,079,360 B**（= 1,205.5 MiB） |
| sha256 | `f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c` |
| 与 HF `X-Linked-ETag` | **逐字符一致**（即下载内容与上游 linked 内容完全一致，未损坏） |
| 落盘位置 | `data/_src/imagenetv2/imagenetv2-matched-frequency.tar.gz`（`data/` 被 `.gitignore` 忽略，**不入提交物**） |

### 3.3 坑 1：扩展名骗人 —— 它是**未压缩的 pax tar**，不是 gzip

归档首 16 字节为 `50 61 78 48 65 61 64 65 72 2f 69 6d 61 67 65 6e`（ASCII `PaxHeader/imagen`），
**不是 gzip 魔数 `1f 8b`**。因此：

* `tarfile.open(path, "r:gz")` → `gzip.BadGzipFile: Not a gzipped file (b'Pa')`；
* `python datasets/make_imagenetv2_subset.py` 内部用 `tarfile.open(path, "r:*")` **自动探测格式**，gz/tar 都能吃。
* 本机与上游前 16 字节逐字节对照通过（`curl -r 0-15` 与本地一致），确认不是下载损坏。

### 3.4 坑 2：目录树是**数字类目录**，不是 WNID 目录；**不要用 ImageFolder 的隐式序号**

归档内容（解压后本仓规范化到 `data/imagenetv2/matched-frequency/`）：

```
imagenetv2-matched-frequency-format-val/          ← 归档内顶层目录（1 个）
├── 0/       ← 十进制 0-based 类别下标（未补零！）
│   ├── 58fbc3e79ef15162b7726de04e98c90bb91a3055.jpeg
│   ├── ...                                        （每个类别目录恰好 10 张）
├── 1/
├── ...
└── 999/
```

* 目录名是 **十进制类别下标 `0`..`999`**（未补零，所以字符串序是 `'0','1','10','100',...`）。
  这正是上游 issue **modestyachts/ImageNetV2#10「Wrongly labelled when using dataset.ImageFolder」**
  的成因：`torchvision.datasets.ImageFolder` 按**目录名字符串排序**编号，会得到错位的类别序。
  **本仓库一律使用显式清单 `<相对路径>\t<0-based 标签>`**，标签取自目录名的**十进制数值**。
* 解压后规模实测：**1000 个类别目录 / 10,000 张 `.jpeg`**，每个目录恰好 10 张。
* 文件名是小写 40 位十六进制（sha1），因此 `sorted()` 取第一个 = 按字节序取第一个，**完全确定性**。

---

## 4. 固定子集：清单、选取规则与复现

### 4.1 清单

| 项 | 值 |
|---|---|
| 路径 | **`datasets/lists/imagenetv2_mf_1000.txt`**（入提交物） |
| 行数 / 体积 | **1000 行 / 87,780 B** |
| sha256 | **`d5532205d8f30099aa70ed9392078adcdbd305c47055930c66e704f7976c9d05`** |
| 格式 | `<相对仓库根的图片路径>` + **Tab** + `<0-based 标签>`，每行一条（仓库统一清单格式：解析必须用 `rsplit(None, 1)`，不能用 `split()`） |
| 标签分布 | 标签取值为 0..999 各 1 次（`per_class_min = per_class_max = 1`） |

前三行 / 末两行（逐字）：

```
data/imagenetv2/matched-frequency/0/58fbc3e79ef15162b7726de04e98c90bb91a3055.jpeg	0
data/imagenetv2/matched-frequency/1/2fb7561b81c4e57499123d88ca1b3dfb90c12b04.jpeg	1
data/imagenetv2/matched-frequency/2/259b04e3c9046a29ae2ad0ae4e2f93b170c3c660.jpeg	2
...
data/imagenetv2/matched-frequency/998/196b2426253906482260110a4180438726265cdb.jpeg	998
data/imagenetv2/matched-frequency/999/1d1df7c0a7790159d541396e421afc49f4fe67d2.jpeg	999
```

### 4.2 选取规则（写死，**不依赖任何随机种子**）

1. 按类别下标**升序 0..999** 遍历目录 `<label>`；
2. 每个目录内把**文件名做 Python `sorted()`**（小写十六进制 sha1 ⇒ 等价字节序），取**第 1 个**；
3. 逐行写出 `<相对路径>\t<label>`。

规则以文字形式同时写在：清单所在脚本的 docstring、provenance 本文件 §4.2、
`report/SPEC13_FREEZE.json` 的 `subset.rule`。

### 4.3 复现命令（任何人、任何机器，结果 byte-for-byte 相同）

```bash
# 1) 下载（见 §3.1）
# 2) 校验 sha256 + 解压 + 生成清单（幂等）
python datasets/make_imagenetv2_subset.py
#    期望输出：tarball 1,264,079,360 B，sha256 f0c37fdf…；
#              解压 10,000 张；清单 1000 行 sha256 d5532205…
# 3) 只校验（不解压、不重写）
python datasets/make_imagenetv2_subset.py --verify-only
```

脚本会在下列任一不满足时**直接 `SystemExit`**（宁可报错，不静默降级）：
归档体积 ≠ 1,264,079,360 B；sha256 ≠ 期望值；目录集合 ≠ `'0'..'999'`；
任一目录图片数 ≠ 10；清单行数 ≠ 1000；标签不是 `0..999` 升序且互不重复。

---

## 5. 标签映射与锚点复核

* V2 使用与 ImageNet-1K **相同的 1000 个类别（WNID 体系）**，因此 `labels/imagenet_classes.txt`
  （1000 行，索引即模型输出下标）**继续有效**。
* 目录名 = 0-based 类别下标 ⇒ `labels/imagenet_classes.txt` 的第 N 行即目录 `N` 的类名。

**锚点（结构侧，`labels/imagenet_wnid_to_idx.json` 交叉校验）**

| 下标 | WNID | 类名 | 本清单取到的文件 |
|---|---|---|---|
| 0 | `n01440764` | tench | `.../0/58fbc3e79ef15162b7726de04e98c90bb91a3055.jpeg` |
| 207 | `n02099601` | golden retriever | `.../207/0b6a2b5278e63cfab3d4806851606e286a2ca0d0.jpeg` |
| 281 | `n02123045` | tabby | `.../281/376424d10c4482c36de24c0e75f38fefd20378c5.jpeg` |
| 999 | `n15075141` | toilet tissue | `.../999/1d1df7c0a7790159d541396e421afc49f4fe67d2.jpeg` |

**内容侧复核（模型实测，独立于目录名假设）**：用仓库自带官方 M0.9 权重
（`checkpoints/pretrained/repvit_m0_9_distill_300e.pth`，713 键 loaded=713 / missing=0，
`crop_pct=0.875`，CUDA）对若干类目录各跑该类 10 张，看预测下标是否等于目录名：

| 目录 | 类名 | 预测命中该目录 | 典型误判（说明） |
|---|---|---|---|
| 0 | tench | **9/10** | — |
| 207 | golden retriever | **8/10** | → 209 `Chesapeake Bay retriever`（同为寻回犬） |
| 285 | Egyptian cat | **7/10** | — |
| 999 | toilet tissue | **5/10** | → 700 `paper towel`（同场景高度相似） |
| 281 | tabby | **3/10** | → 282 `tiger cat`（虎斑猫近缘类，视觉几乎同源） |

命中率整体低于该模型在 ImageNet-1K val 上的水平，**且所有误判都落在语义相邻类**，
与「ImageNetV2 上准确率系统性下降 10~15 个点」的文献结论一致；同时证明
**「目录名 = 标准 0-based 下标」成立**，清单标签可用。

---

## 6. 与 ImageNet-1K 验证集的关系（必须原样保留的口径声明）

1. **ImageNetV2 不是 ImageNet-1K 验证集**，也不含任何 ImageNet-1K 图像。它是按 ImageNet 分布
   **重新采样**、在 2013 年之后（即 ImageNet 已经被研究了十年之后）采集并标注的**独立测试集**，
   设计目的正是为了避免自适应过拟合（adaptive overfitting），使准确率不受已有模型的影响。
2. 本变体全量 = **1000 类 × 10 张 = 10,000 张**；本仓库取其中 **1000 张**（每类 1 张）作固定子集。
3. **准确率不可比**：同一模型在 ImageNetV2 上的 Top-1 通常比 ImageNet-1K val **低 10~15 个百分点**，
   这是基准性质（分布偏移 + 更高的图像难度），**不是模型退化**。
   因此：**任何文档、PPT、报告、答辩话术中，都不得把 V2 的 Top-1 与 ImageNet-1K val 的 Top-1、
   与论文/官方公布值并列、排序或暗示可比性。**
4. **考核试题 PDF §1.3 要求「必须注明」的那句原话不适用于 V2 结果**：
   `本结果为指定ImageNet-1K验证子集上的实际运行结果，不代表论文完整ImageNet-1K验证集结果。`
   照搬到 V2 结果上即构成**虚假陈述**（V2 不是 ImageNet-1K 验证子集，也不是考核方指定子集）。
   V2 结果应使用如下**如实表述**：
   > **本结果为 ImageNetV2 matched-frequency 的 1000 张确定性固定子集（1000 类各 1 张）上的实际运行结果。
   > ImageNetV2 是 Recht et al. (NeurIPS 2019) 按 ImageNet 分布重新采样构建的独立测试集，不是
   > ImageNet-1K 验证集；其准确率与 ImageNet-1K 验证集准确率、与论文公布值不可直接比较。**
5. **本仓库只有这一套官方模型评价口径**。此前的自建 ImageNet-1K val 分层子集（每类 1 张、
   `seed=20260912`）及其清单 `datasets/lists/imagenet_val_subset.txt` 已按用户决议
   （2026-09-16：「不要留存之前旧的自建子集作为对照」）**整体删除**（`git rm`），
   **不在任何文档里留作对照、附录或历史**；仓库内不得出现第二套官方模型评价口径。
   （`data/imagenet/val/` 的 5 万张原始 ImageNet-1K val 图像仍保留在本机磁盘上，它属上游原始数据，
   与「自建子集」不是一回事，但**不参与任何评测口径**。）
6. **正面价值**：相比「按 seed 自建」的划分，ImageNetV2 公开、带标签、有版本、可哈希校验，
   在试题「所有候选人使用相同数据」这一条上**更强**——任何人拿同一归档 + 同一脚本即可
   byte-for-byte 复现这 1000 张（清单 sha256 已公开，见 §4.1）。

---

## 7. 仓库内接口与遗留待办

| 位置 | 状态 |
|---|---|
| `configs/pretrained_eval.yaml` → `pretrained_eval.data_list` | ✅ 已指向 `datasets/lists/imagenetv2_mf_1000.txt` |
| `tools/run_all_pretrained.py` → `BASE` 的 `pretrained_eval.data_list` | ✅ 已同步（该键用 `--set` 覆盖，优先级高于 yaml；不同步会让批量重跑**静默**用回旧清单） |
| `datasets/make_imagenetv2_subset.py` | ✅ 新增：校验 + 解压 + 生成清单（幂等，含全部断言） |
| `datasets/lists/imagenet_val_subset.txt`（旧自建子集清单） | ✅ **已按用户决议 `git rm` 删除**（09-16「不要留存旧的自建子集作为对照」）；不做对照、不做附录、不做历史留存 |
| `report/SPEC13_FREEZE.json` | ✅ 已再生成为 ImageNetV2 口径基线（含 V2 清单/归档哈希、运行手册入口） |
| `report/SPEC13_SWITCH_RUNBOOK.md` | ✅ 已改写为「若考核方另下发清单，如何从当前 ImageNetV2 子集切过去」 |
| `README.md` §2.2 / §4、`PROGRESS.md` 阻塞项 + 决策记录 | ✅ 已改写为 ImageNetV2 口径 |
| ⚠ `deploy/compare_torch_onnx.py:82`（默认清单）与 `:109`（`source = "imagenet_val_subset"`） | **未改**（属复评任务 t12 的范围）：不改这两处，一致性证据会自述错误来源，且 `tools/selfcheck.py:525` 的 `onnx.realimg` 白名单 `{"pet_test","imagenet_val_subset"}` 会判 FAIL |
| ⚠ `tools/selfcheck.py:267` 的容差注释（"1000 张自建子集，单张翻转即 0.1 个点"）与 V2 口径的对应关系 | **未改**（t12 负责）：V2 上准确率整体下移，容差与官方值对照的判据需要 t12 按其断言口径调整 |

---

## 8. 复核命令（任何人都可照跑）

```bash
# 归档与清单
python datasets/make_imagenetv2_subset.py --verify-only

# 清单自检：行数 / 标签唯一性 / 目录-标签一致性
python -c "r=[l.rsplit('\t',1) for l in open('datasets/lists/imagenetv2_mf_1000.txt',encoding='utf-8') if l.strip()]; print(len(r), len({int(y) for _,y in r}), r[0], r[-1])"

# 清单哈希
python -c "import hashlib;print(hashlib.sha256(open('datasets/lists/imagenetv2_mf_1000.txt','rb').read()).hexdigest())"

# 归档哈希
python -c "import sys;sys.path.insert(0,'.');from datasets.make_imagenetv2_subset import verify_tarball,TARBALL;print(verify_tarball(TARBALL)['sha256'])"

# 标签映射预检（不加载模型）
python tools/eval_pretrained.py --cfg configs/pretrained_eval.yaml --set pretrained_eval.check_data=true
```
