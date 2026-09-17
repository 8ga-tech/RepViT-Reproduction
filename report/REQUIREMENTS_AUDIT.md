# 秋招考核 PDF 全量要求审计与验收契约（REQUIREMENTS_AUDIT）

- **审计对象**：`RepViT-Reproduction`（**本仓库根目录**；下文一律用 `<repo>/` 指代仓库相对路径的根，不写采集机绝对路径），分支 `master`
- **审计基准**：`dc0527bc346a909cf5cf529f0755ff5bbdb8100e`，与 `origin/main` **同一 SHA**（`git rev-parse HEAD origin/main` 两值相等）
- **要求来源**：《2026秋one团队ai算法组考核试题.pdf》29 页，全文文本取自**工作区（仓库外）**文件 `_extract/kaoti_full.txt`（本报告所有「要求原文摘录」逐字取自该文本）；第 3/25/26 页 4 张图片型表格的 OCR 内容取自**同为工作区（仓库外）**的 `_build/exam_questions.md` 第五节
- **审计日期**：2026-09-16（21:43 前后）
- **审计性质**：**只读审计**。本次唯一写入是新建本文件；未修改仓库任何既有文件（自证见 §11）
- **判定标签**：`满足` / `部分` / `缺失` / `挂起（待考核方清单）`
- **§1.3 挂起约定**（用户决议）：凡属「考核方指定子集 / 固定图片列表及标签 / 所有候选人相同数据 / 验证子集不得用于训练或调参 / 必须注明的免责句」这 5 条，**一律判为「挂起（待考核方清单）」**，缺口列统一写「按用户决议挂起，不在本次范围」，**本报告不给出任何改动建议**。

> **协作边界（不越权）**：§1.3 口径冻结、切换 runbook、README/REPORT/PPT 的同步改写由队长分派的其它任务负责；本报告只做「判定 + 证据 + 缺口清单」，不含对上述被挂起项的改动方案。

> **后记（2026-09-17 追加）**：上面第 8/9 行的 §1.3 挂起约定**已被用户后续决议取代** —— 用户已指定 **ImageNetV2 matched-frequency 1000 张固定子集**为唯一评价口径，自建子集已整体清除。**本次只追加、不删除**：原判定列与原文一字未改，5 条挂起项的当前真实状态见 **§2.1.1**；本文其它引用旧口径的行也已在本次一并指向现行口径（逐处见 §2.1.1 末的「本次指向调整清单」）。

> **计数口径声明（重要）**：任务契约给出的分项数量与 PDF 原文的 bullet 数量在 4 处存在 ±1 差异。本报告一律**按 PDF 原文的每个 bullet 建行**（宁多不少），并在每节开头给出「PDF 原文 bullet 数 ↔ 契约声明数」的对照说明。下表为差异处：

| 节 | 契约声明 | PDF 原文 bullet 数 | 差异来源（本报告处理方式） |
|---|---|---|---|
| 五-4 必须提供的结果 | 9 项 | **10** 条 | 第 10 条「实际图片显示 Top-5 类别及置信度」被排到 PDF 第 11 页顶，易与第 9 条合并；本报告**独立建行** |
| 五-5 基础要求 | 14 条 | **13** 条 + 1 条附注 | 附注「官方 ImageNet 模型与自行训练 37 类模型必须分别使用正确的标签文件」被计入；本报告把它作为**独立行 5.4.14** |
| 五-6 最终提交内容 | 16 项 | **15** 条 | 第 16 行「候选人需要明确标记 5 类」是并列的独立要求；本报告把提交内容列为 15 行、把标注单独列为 §6.4 的 5 行 |
| 九-4 必要限制 | 17 条 | **16** 条 | 本报告列 16 行，并在节末说明「如需凑到 17，可把首句『即使完成进阶或拓展任务，也进行分数限制』单列」 |

---

## 1. 一页结论

1. **仓库当前产物对「基础 70 分」的要求基本齐全**：五-1~五-6 共 6 个基础任务、约 120 条可验收子项中，绝大多数为 `满足`，证据全部落在 `outputs/`、`configs/`、`tools/`、`deploy/` 与 `report/REPORT.md` 上，且每个「满足」都能指到真实文件或可跑通命令（§10 给出证据命令清单）。
2. **当时被用户决议挂起的是一族数据口径项**：官方模型评价子集、Pet 统一 `train_list/val_list`、免责句等 5 条在 t2 审计时判为 `挂起`（**原判定与原始证据见 §2.1，逐条保留未改**；**当前真实状态见 §2.1.1**）。当时用的自建子集已按用户决议整体清除；现行唯一口径是 **ImageNetV2 matched-frequency 的 1000 张固定子集**（`datasets/lists/imagenetv2_mf_1000.txt`）。`data/provided/` 仍为空，其 README 记载「截至 2026-09-12 全盘搜索未找到考核方下发的任何材料」，PDF 第 28 页那 9 项只是「建议考核方提前准备」，从未真正发放。
3. **发现 1 条阻断性缺口**：仓库自带验收脚本 `tools/check_report_assets.py` 现在跑出 **OK=22 / WARN=0 / FAIL=36**（其期望清单 `tools/report_spec.py` 指向已被淘汰的 `figures/` 目录与 `B0_/O1_` 命名）。现场若被评委直接运行，会与「无法复现实验结果（总分不超过 50 分）」条款正面冲突。
4. **发现 8 条得分项缺口 + 8 条体验项缺口**，全部集中在「口径未写全 / 表未填满 / 图未补画 / 字段未落盘」，**没有一条是「实验没做」**。逐条见 §9。
5. **两批 logits 误差实验的口径未被混写**：本报告在 **§3.6（基础任务 5：ONNX 部署 —— 一致性 5.5.x 行 + 两处「口径核对」块）** 与 **§5.3（进阶 3：结构重参数化 7 行）** 严格照抄 `report/LOGITS_AUDIT_FINDINGS.md` 第 1 节的桶定义——
   **B4 = 结构重参数化，Pet-37 baseline，32 个固定随机输入（`torch.randn`，seed=20240912，batch=8），`max|Δ| = 7.093e-06`、`mean|Δ| = 2.031e-06`、Top-1 32/32、BN 107→0**；
   **B1 = PyTorch↔ONNX 一致性，Pet-37，`datasets/lists/pet_test.txt` 前 12 张真实图片（n=12），`max|Δlogits| = 6.199e-06`、`mean|Δlogits| = 1.501e-06`、Top-1 与 Top-5 集合一致率均 100%**。
   两者**实验对象、输入批次、代码路径都不同**，本报告任何一处都不把两者并成一句结论，也不用其中一个替代另一个。
6. **本轮新增本文件会改变仓库已登记的「logits 引用清单不动点」**：`report/SYNC_NOTES.md` 的写作约束要求清单维持 `4124 条引用 / 153 个文件`，而 `tools/audit_logits_references.py` 扫描的是 `git ls-files` 全量文件，本文件含 `7.093e-06` / `6.199e-06` 等权威值。**新增本文件后需重跑 `python tools/audit_logits_references.py` 并更新 `report/` 下审计文档与 SYNC_NOTES 的登记值**（已列入 §9 体验项 X-8）。**（后记 2026-09-17：该项即任务 t10 的 C 段；因 t19 还将改动 `family_summary.csv` / Pareto 产物 / PPT / REPORT，不动点会被再次改变，C 段已**转出给 t20** 在 t19 落定后统一刷新与登记 —— 本条不与「保留历史」冲突，只是执行归属变更。）**

---

## 2. §1.3「官方模型评价要求」逐条判定（含 5 条挂起项）

PDF 原文（第 3~4 页）§1.3 共 4 组内容：选择型号 4 条、子集口径 6 条、强制免责句 1 条、以及各条附带的计分说明。逐条建行如下。

### 2.1 挂起项（用户决议，本次不改）

> **后记（2026-09-17 追加）**：本小节的 5 条挂起判定**保留原样**（判定列、证据列、缺口列一字未改）。用户后续已指定 ImageNetV2 为验证集，5 条的实际状态**已变** —— 当前真实状态、依据与「是否已解除挂起」逐条列在下面的 **§2.1.1**。

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **1.3-S1** 官方模型评价使用**考核方指定的 ImageNet-1K 验证子集** | 现用 `datasets/lists/imagenet_val_subset.txt`（1000 行，自建分层抽样，`python datasets/make_imagenet_subset.py --n 1000 --seed 20260912` 生成）；`data/provided/README_这里要放什么.md` 记载该目录为空；`README.md:116-117`、`report/REPORT.md:125-126` 均显式声明「**自建**、**不是考核方下发的指定子集**」 | **挂起（待考核方清单）** | 按用户决议挂起，不在本次范围 |
| **1.3-S2** 考核方**提供固定图片列表及标签** | `data/provided/` 为空（仅有说明 README）；现有 `datasets/lists/imagenet_val_subset.txt` 为自建；标签 `labels/imagenet_classes.txt`（1000 行，四锚点自检 `python tools/verify_imagenet_labels.py` 通过） | **挂起（待考核方清单）** | 按用户决议挂起，不在本次范围 |
| **1.3-S3** **所有候选人使用相同数据** | 无考核方统一清单可比对；仓库只能声明使用自建清单，`outputs/pretrained_eval/summary.csv` 未记录 `subset_sha256`（`outputs/benchmarks/family_summary.csv` 的 `subset_sha256 = f9267fefe71a8345` 只覆盖家族 5 型号那次评价） | **挂起（待考核方清单）** | 按用户决议挂起，不在本次范围 |
| **1.3-S4** **验证子集不得用于训练或调参** | 训练只用 Oxford-IIIT Pet（`datasets/lists/pet_*.txt`），ImageNet 子集只进 `tools/eval_pretrained.py`；`configs/pretrained_eval.yaml` 无任何训练/调参分支；`outputs/metrics/leakage_check.json` train∩val∩test = 0/0/0 | **挂起（待考核方清单）** | 按用户决议挂起，不在本次范围 |
| **1.3-S5** 候选人**必须注明**：`本结果为指定ImageNet-1K验证子集上的实际运行结果，不代表论文完整ImageNet-1K验证集结果。` | `README.md:171`、`report/REPORT.md:125-126`、`report/ppt_svg/05_pretrained.svg:78`、`ppt_svg/04`（PPT 第 4 页 PDF 文本可读「本结果为「该自建验证子集上的实际运行结果」，不代表论文完整 ImageNet-1K 验证集结果。」）均有**等价句**，但措辞是「自建验证子集」而非 PDF 要求的「指定」子集 | **挂起（待考核方清单）** | 按用户决议挂起，不在本次范围 |

### 2.1.1 后续决议与当前状态（2026-09-17 追加；不修改上方原判定）

> 本节只**追加**「后续决议 + 当前状态」，§2.1 的历史判定过程完整保留（判定列 / 证据列 / 缺口列一字未改）——「审计文档的可信度在于它没被事后修改」。

**后续决议（用户指令）**：本仓库官方模型评价**只保留一套口径** —— **ImageNetV2 matched-frequency 的 1000 张确定性固定子集**（1000 类各 1 张）。此前的自建子集已整体清除（清单 / 生成脚本 / 子集报告均已 `git rm`），**不留对照、不留附录、不留历史**。考核方自己下发的清单**迄今仍未到位**（`data/provided/` 仍为空）。

**现行口径的权威事实**

| 项 | 现行值 | 证据 |
|---|---|---|
| 清单 | `datasets/lists/imagenetv2_mf_1000.txt`：1000 行 / 87,780 B / sha256 `d5532205…6c9d05`；格式 `路径<TAB>标签` | 该文件本身 |
| 数据目录 / 归档 | 解压于 `data/imagenetv2/matched-frequency/<类目录>/`；归档来自 `vaishaal/ImageNetV2`（MIT），sha256 `f0c37fdf…c9ca7c` | `report/IMAGENETV2_PROVENANCE.md`、`report/SPEC13_FREEZE.json` |
| 5 型号实测 Top-1 / Top-5（%，`crop_pct=0.875`，PyTorch / CUDA） | M0.9 **68.7 / 85.7**、M1.0 **69.2 / 86.4**、M1.1 **70.8 / 87.2**、M1.5 **71.4 / 89.0**、M2.3 **73.6 / 89.9** | `outputs/pretrained_eval/summary.csv`、`outputs/pretrained_eval/<model>/metrics.json` |
| ONNX n=12 一致性 `max\|Δlogits\|` | M0.9 `1.383e-05`、M1.0 `1.335e-05`、M1.1 `1.860e-05`、M1.5 `1.144e-05`（`source = imagenetv2_mf_1000`）；**Pet-37（B1）`6.199e-06` 与结构重参数化（B4）`7.093e-06` 均不变** | `outputs/metrics/consistency_repvit_m0_9_pet37.json`、`consistency_repvit_m0_9_in1k.json`、`consistency_repvit_m1_0_in1k.json`、`consistency_repvit_m1_1_in1k.json`、`consistency_repvit_m1_5_in1k.json` |
| 口径红线 | **禁止**把 ImageNetV2 称为「ImageNet-1K 验证集」或「考核方指定的 ImageNet-1K 验证子集」；**禁止**与论文 / 官方公布的 ImageNet-1K 数值并列、算差值或暗示可比 | _coord/POLICY_IMAGENETV2_ONLY.md（仓库外协调文件，不入提交物）§六 |

**5 条挂起项的当前状态（逐条）**

| 原行 | 原判定（未改动） | 当前真实状态（2026-09-17） | 是否已解除「待考核方清单」 |
|---|---|---|---|
| **1.3-S1** 考核方指定子集 | 挂起（待考核方清单） | 无考核方清单可用；用户已指定 **ImageNetV2 matched-frequency 1000 张固定子集**为本仓库唯一评价口径（清单与哈希见上表） | **未解除**（考核方清单仍未下发）；「无口径可用」这一困境已由用户指定口径解决 |
| **1.3-S2** 固定图片列表及标签 | 挂起（待考核方清单） | 列表与标签均为**本仓库自建**：列表 = `datasets/lists/imagenetv2_mf_1000.txt`；标签 = `labels/imagenet_classes.txt`（1000 行）+ `labels/imagenet_wnid_to_idx.json`，锚点自检通过（`python tools/verify_imagenet_labels.py`） | **未解除**；现行替代 = 自建 V2 清单，**且不得称其为「ImageNet-1K 验证集」** |
| **1.3-S3** 所有候选人使用相同数据 | 挂起（待考核方清单） | 无考核方统一清单可比对；仓库侧的可核对性由清单哈希自证 —— `outputs/benchmarks/family_summary.csv` 的 `subset_sha256 = d5532205d8f30099` 对 5 个型号一致 | **未解除**（该条本质上需要考核方统一清单） |
| **1.3-S4** 验证子集不得用于训练或调参 | 挂起（待考核方清单） | 该条**本不依赖考核方**：训练只用 Oxford-IIIT Pet（`datasets/lists/pet_*.txt`）；ImageNet 侧数据只进 `tools/eval_pretrained.py` 的评价流程；`outputs/metrics/leakage_check.json` 的 train∩val∩test = 0/0/0。V2 口径下同样成立 | **已解除 → 按现状判 `满足`** |
| **1.3-S5** 必须注明指定子集的免责句 | 挂起（待考核方清单） | PDF 原句（「本结果为指定ImageNet-1K验证子集上的实际运行结果…」）**只适用于考核方下发的 1K 子集**；ImageNetV2 不是 1K 验证集，故各交付物改用**如实声明**：明写「不是 ImageNet-1K 验证集」、「不得与 1K val / 论文公布值并列、排序或算差值」 | **部分解除**：V2 口径下已有等价如实声明（`report/REPORT.md` §4.1/§4.3、`report/答辩PPT_RepViT.pptx` 第 4 页、`06_答辩Q&A_精简版.docx`）；待考核方下发 1K 子集时再套用 PDF 原句 |

**本次指向调整清单（只改证据指向，判定列全部不变）**

| 行 | 改了什么 |
|---|---|
| 页首「审计对象」 | 采集机绝对路径 → `<repo>/` 占位符（任务 A） |
| `6.11` / `X-3` | 绝对路径的复核命令改为**平台中立**写法 `git grep -lE '[A-Za-z]:\\{1,2}Users'` → **32 个文件**；`git grep -cE '[A-Za-z]:\\{1,2}Users'` → **278 行**（实测值；原用机器专属字面量，已删除） |
| `1.3-2` | 证据的 `top1` 由旧口径值改为现行 **`68.7`**（并补 `top5 = 85.7`、`crop_pct = 0.875`） |
| `1.3-6` | 证据的清单文件改为 `datasets/lists/imagenetv2_mf_1000.txt`（仍 1000 行，仍落在 500~1000 建议区间） |
| `1.1-2` | 复跑命令注明脚本内清单已同步为 ImageNetV2；`判定` 仍为 `部分`，原因由「自建子集」改为「考核方清单未到，现行用用户指定的 V2 口径」 |
| `1.1-3` | 示例值改为现行 **`68.7`/`85.7`** |
| `5.5.1` / `5.5.2` | 官方型号的 `max_abs_logits` / `mean_abs_logits` 换为现行值（`1.383e-05`/`1.335e-05`/`1.860e-05`/`1.144e-05` 与 `2.334e-06`/`2.215e-06`/`2.219e-06`/`1.876e-06`），覆盖范围由 3 型号更新为 **5 型号**；Pet-37（B1）与 B4 数值不变 |
| `5.5.7` | 证据更新为「5 份同口径产物、4 个入库官方型号全覆盖」，并在缺口列标注 **2026-09-17 缺口已闭合** |
| `T-3` | ImageNet 侧划分文件改为 V2 清单 |
| `RP-19` | 引用标题去掉「（自建子集）」字样（`report/REPORT.md` §4 已改为 ImageNetV2 口径） |
| `A1-2` | `subset_sha256` 换为现行 `d5532205d8f30099` |
| §8 说明 1 / §9 标题 / §1 第 6 条 / X-8 / 文末结论 | 各加一行**日期化后记**（不删除原句）：5 条挂起项的现状指向本节；X-8（不动点刷新）转出给 t20；§9 缺口清单的状态更新见其标题下的一行 |



### 2.2 §1.3 其余条目（非挂起，按现状判定）

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **1.3-1** 基础任务至少选择两种官方预训练型号 | `outputs/pretrained_eval/` 下 5 个型号目录（`repvit_m0_9`、`m1_0`、`m1_1`、`m1_5`、`m2_3`），各含 `outputs/pretrained_eval/*/metrics.json`、`outputs/pretrained_eval/*/latency.json`、`outputs/pretrained_eval/*/top5_samples.json`、`outputs/pretrained_eval/*/predictions.csv`、`outputs/pretrained_eval/*/cases/` | 满足 | 5 ≥ 2 |
| **1.3-2** 必须包含 **RepViT-M0.9** | `outputs/pretrained_eval/repvit_m0_9/metrics.json`（`top1 = 68.7`、`top5 = 85.7`、`num_images = 1000`、`weights_sha256 = 857eb0e6…`；口径 = ImageNetV2 matched-frequency 1000 张固定子集、`crop_pct = 0.875`） | 满足 | — |
| **1.3-3** 另选 M1.0、M1.1、M1.5 或 M2.3 中的一种 | 四种**全部**评价：`outputs/pretrained_eval/{repvit_m1_0,repvit_m1_1,repvit_m1_5,repvit_m2_3}/metrics.json` | 满足 | — |
| **1.3-4** 鼓励根据设备情况评价更多型号 | 同上（5 个型号）；`outputs/benchmarks/family_summary.csv` 覆盖 5 型号同口径 | 满足 | — |
| **1.3-5** 选择更多型号本身不直接增加分数 | 计分说明，无独立交付物；仓库按此未把型号数量写入得分论述（`report/REPORT.md:21`「不做…」清单与 §5 结论都以「边际收益」而非型号数量论证） | 满足 | 计分说明项，无交付物可验收 |
| **1.3-6** 建议 500～1000 张 | `datasets/lists/imagenetv2_mf_1000.txt` = **1000** 行（`python -c "print(sum(1 for _ in open('datasets/lists/imagenetv2_mf_1000.txt',encoding='utf-8')))"` → `1000`；ImageNetV2 matched-frequency，sha256 `d5532205…6c9d05`） | 满足 | 数量落在建议区间内；清单来源本身属挂起项 1.3-S1 |
| **1.3-7** 输入尺寸建议统一为 224×224 | `configs/pretrained_eval.yaml` 的 `input_size`；`outputs/pretrained_eval/*/metrics.json` 全部 `"input_size": 224`；`deploy/model_registry.py:15` 统一 `input_size=224` | 满足 | — |
| **1.3-8** 必须使用正确的 ImageNet 类别映射 | `labels/imagenet_classes.txt`（1000 行）+ `labels/imagenet_wnid_to_idx.json`；`python tools/verify_imagenet_labels.py` 四锚点（tench / golden retriever / tabby / toilet tissue）通过；`outputs/metrics/imagenet_labels_report.json`；`deploy/model_registry.py:80-88` 的 `labels_for()` 对行数 ≠ 类别数**硬断言** | 满足 | — |

---

## 3. 五、基础必做任务（70 分）

### 3.0 第 2 节「基础关键环节」与「所有性能数据必须同时报告」（前置门槛，第 2 页）

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **0-1** 运行并评价至少两种官方预训练型号 | `outputs/pretrained_eval/*/metrics.json`（5 型号） | 满足 | — |
| **0-2** 自行训练 RepViT-M0.9 Baseline | `configs/baseline.yaml` + `tools/train.py`；`checkpoints/baseline_best.pt`（19,386,027 B）；`outputs/logs/baseline_metrics.csv`；`outputs/metrics/baseline_test.json` | 满足 | — |
| **0-3** 完成至少一项控制变量优化 | `configs/opt_{combo,mix,randaug,disc}.yaml` 四方案 + `configs/opt_abl_{a,b,ab}.yaml` 三消融臂；`outputs/logs/opt_*_metrics.csv` | 满足 | — |
| **0-4** 完成自行训练模型及官方模型的 ONNX 独立部署 | `onnx/repvit_m0_9_pet37.onnx`（自训）+ `onnx/repvit_m0_9_in1k.onnx`、`onnx/repvit_m1_0_in1k.onnx`（官方）；推理入口 `deploy/infer_onnx.py` | 满足 | — |
| **0-5** 提交完整报告、PPT 并参加现场答辩 | `report.pdf`（26 页）、`report/REPORT.md`、`report/REPORT.docx`、`report/答辩PPT_RepViT.pptx`、`report/答辩PPT_RepViT.pdf`（15 页） | 部分 | 「参加现场答辩」是现场环节，静态不可判定；已具备的静态交付物齐备 |
| **0-6** 性能数据必须报告 CPU/GPU/集显型号 | `outputs/benchmarks/summary.csv` 的 `cpu_model = 13th Gen Intel(R) Core(TM) i7-13650HX`；`outputs/metrics/bench.jsonl` 的 `env.cpu`；`report/REPORT.md:648` | 满足 | — |
| **0-7** 操作系统 | `outputs/benchmarks/summary.csv` 的 `os = Windows-11-10.0.26200-SP0`；`report/REPORT.md:648` | 满足 | — |
| **0-8** 推理后端及版本 | `outputs/benchmarks/summary.csv` 的 `ort_version = 1.30.0`；`report/REPORT.md:645` | 满足 | — |
| **0-9** 实际 Execution Provider | `outputs/benchmarks/summary.csv` 的 `provider_actual = ['CPUExecutionProvider']`；`outputs/metrics/bench.jsonl` 的 `provider_actual` / `ep`；`deploy/benchmark.py:57` 打印 `EP(actual)`；`tools/bench_igpu.py` 在检测到回退时把该行标 `invalid` | 满足 | — |
| **0-10** 模型型号 | `outputs/benchmarks/summary.csv` 的 `model` 列（6 型号） | 满足 | — |
| **0-11** 输入尺寸 | `outputs/benchmarks/summary.csv` 的 `input_size = 224` | 满足 | — |
| **0-12** batch size | `outputs/benchmarks/summary.csv` 的 `batch_size = 1` | 满足 | — |
| **0-13** FP32/FP16/INT8 精度 | `outputs/benchmarks/summary.csv` 的 `precision = FP32` | 满足 | — |
| **0-14** 预热和正式测试次数 | `outputs/benchmarks/summary.csv` 的 `warmup = 10` / `runs = 50`；`tools/selfcheck.py` 的 `bench.meta` 检查 PASS | 满足 | — |
| **0-15** 线程数或其他重要运行配置 | `outputs/benchmarks/summary.csv` 的 `threads_intra = 4`；`outputs/metrics/bench.jsonl` 的 `threads_inter = 1` | 满足 | — |
| **0-16** 不同设备、后端和精度下的延迟不能直接横向比较 | `outputs/benchmarks/family_summary.csv`（PyTorch CPU 计时，M0.9 = 36.1 ms）与 `outputs/benchmarks/summary.csv`（ORT CPU P50 7.37 ms）**分文件存放**；`report/REPORT.md:198-199` 显式警告「注意不要混用」；`report/REPORT.md:158-159` 声明 iPhone 12 延迟不可直接比较 | 满足 | README 第 12 节另注明 `outputs/metrics/bench.jsonl` 存在两批基准（早批 P50 13.60/17.18/12.48 ms），要求引用时写明批次 |

### 3.1 基础任务 1.1：官方模型评价（15 分中的一部分，第 6 页，10 条）

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **1.1-1** 正确加载官方预训练权重 | `tools/check_weights.py --dir checkpoints/pretrained`（字节数 + zip 中央目录 + torch.load 三道校验，`loadable=False` 计数 0）；`outputs/metrics/weight_sha256.json`（5/5 与官方 Release 逐位一致）；`outputs/metrics/weight_load_report.json`（官方 ckpt + vendored 官方实现：`n_missing=2`（仅换头 Linear）、`n_unexpected=0`） | 满足 | — |
| **1.1-2** 对**指定** ImageNet 验证子集完成推理 | `python tools/run_all_pretrained.py --cfg configs/pretrained_eval.yaml --model repvit_m0_9 repvit_m1_0 repvit_m1_1 repvit_m1_5 repvit_m2_3`（脚本内的清单已同步为 ImageNetV2）；落盘 `outputs/pretrained_eval/*/predictions.csv`（各 1000 行） | 部分 | 推理已在**现行唯一口径**（ImageNetV2 matched-frequency 1000 张固定子集）上完成，但该子集**不是考核方下发的清单** → 与 §1.3 挂起项 1.3-S1/S2 耦合。**2026-09-17 后记**：用户已指定 ImageNetV2 为唯一口径，故本条按现行口径记 `部分`（考核方清单仍未到），不再按「挂起」处理 —— 详见 §2.1.1 |
| **1.1-3** 计算 Top-1 和 Top-5 准确率 | `outputs/pretrained_eval/*/metrics.json` 的 `top1`/`top5`（现行 M0.9 为 `68.7`/`85.7`）；`outputs/pretrained_eval/summary.csv` | 满足 | 注意 `metrics.json` 里已是百分数（`report_spec.py` 的 `CALIBER_ROWS` 也标注「不得再乘 100」） |
| **1.1-4** 统计参数量、MACs 和模型文件大小 | `outputs/pretrained_eval/*/metrics.json` 的 `params_total`/`macs_g`/`macs_source="thop"`/`model_file_size_mb`；`outputs/metrics/count_params.json`（三口径）、`outputs/metrics/count_flops.json`（thop 847,050,816 与 fvcore 832,165,824 同口径，**不是 2 倍**） | 满足 | — |
| **1.1-5** 统计 PyTorch 平均推理延迟 | `outputs/pretrained_eval/*/latency.json`（`warmup=10`、`runs=50`、`mean_ms`、`p50_ms`、`p95_ms`，含融合/未融合两组）；`report/REPORT.md:136`（§4.3 表第 6 行「**PyTorch 推理**」） | 满足 | — |
| **1.1-6** 输出至少 6 张图片的 Top-5 类别及置信度 | `outputs/pretrained_eval/<model>/top5_samples.json` 每型号 **`n_samples = 6`**（`repvit_m0_9 / m1_0 / m1_1 / m1_5 / m2_3` 均为 6），每条含 `top5[idx,name,prob]` | 满足 | 6 ≥ 6（踩线满足） |
| **1.1-7** 分析至少 2 个正确案例 | 图片证据：每型号 `outputs/pretrained_eval/*/cases/correct_1.png` + `outputs/pretrained_eval/*/cases/correct_2.png`（2 正）；`outputs/pretrained_eval/repvit_m0_9/top5_samples.json` 中 `correct=true` 4 条。**文字分析**：`report/REPORT.md:164-165` 只具体写了 1 个正确案例（beagle 96.8%，Top-5 余项皆为猎犬类） | 部分 | 缺「第 2 个正确案例」的逐例文字分析（要求「分析至少 2 个」，现有 1 个成文 + 2 张图）；影响五-1（15 分）的案例分项。修复见 §9 得分项 G-3 |
| **1.1-8** 分析至少 2 个错误案例 | 图片证据：每型号 `outputs/pretrained_eval/*/cases/wrong_1.png` + `outputs/pretrained_eval/*/cases/wrong_2.png`（2 误）。**文字分析**：`report/REPORT.md:166` 用一句话概括两类错误现象（细长体型小型犬混判、主色相近体型不同混判），未逐例对应到 `wrong_1/wrong_2` | 部分 | 与 1.1-7 同源：图有 2 张，成文分析未逐例；修复见 §9 得分项 G-3 |
| **1.1-9** 比较所选模型的精度、规模和延迟 | `outputs/pretrained_eval/summary.csv`（params_M / macs_G / top1 / top5 / latency）；`report/REPORT.md` §5 表（5 型号，Top-1 + 参数量 + MACs + ONNX 文件 + P50 + P95）；`outputs/advanced/{params_vs_acc,latency_vs_acc,macs_vs_acc}.png` | 满足 | — |
| **1.1-10** 分析较大模型是否在当前设备上获得了合理收益 | `report/REPORT.md:179-182`（M1.0 相比 M0.9：参数量 **+34.4 %**、MACs **+34.9 %**、P50 **+35.4 %**（7.68 → 10.40 ms）、Top-1 仅 **+0.5 个点** → 结论「在当前 CPU 部署目标下，这一档『加宽』的收益最不划算」）；同段 §五 边际收益（**acc/ms 口径**的 4 个值：M0.9→M1.0 = **+0.18**、M1.0→M1.1 = **+3.05**、M1.1→M1.5 = +0.08、M1.5→M2.3 = +0.14）；`outputs/advanced/marginal_returns.csv` | 满足 | — |
| **1.1-11**（附注）不得直接复制官方表格作为个人实验结果 | `report/REPORT.md:127-133` 起的 **§4.3「七类结果来源（逐类分开，不得互相替代）」**（标题 L127，来源表表体 `report/REPORT.md:131-137`）：论文公布 / 官方仓库公布 / 官方权重实测 / 自训练 Baseline / 自优化模型 / PyTorch 推理 / ONNX 部署 **七类各占一行**，行内结果与「依据产物」列一一对应；`report/REPORT.md:167` 明写「**本表不含任何论文/官方公布值**——ImageNet-1K 公布值与本表不可比（见 §4.3）」；把「论文/官方公布」与本机实测**相减**（差值 −0.50 / −0.10）的那张违规三方表**已删除**（删除记录见 _coord/POLICY_IMAGENETV2_ONLY.md（仓库外协调文件，不入提交物）§15.2：「`report/REPORT.md` §4.3 七类拆列、并删除了『论文/官方公布 vs 本机实测差值 −0.50/−0.10』那张违规三方表」） | 满足 | — |

> **后记（2026-09-17 追加，任务 t22）**：本节 **1.1-7** 证据列引用的「beagle 96.8%」是 t2 审计时点的文字，**现行仓库已无该值的落盘依据**（现行仓库里含 beagle 的落点全是**外部实拍图 / ONNX 演示路径**这一类、都不是本行的案例成文：`report/REPORT.md:158` 的演示图 beagle **94.47%**、`report/REPORT.md:435`/`:773` 的外部实拍 8 张 beagle **90.32%**、`report/PPT_CONTENT.md:613` 的同图对比 beagle **0.903**；其余为标签 / 清单 / 命令示例类条目）。现行案例素材以 `outputs/pretrained_eval/repvit_m0_9/top5_samples.json` 为准：**6 条样本、`correct=true` 4 条**（Band Aid 1.0000 / punching bag 0.9999 / German shepherd 0.8106 / bell pepper 0.8107）、`correct=false` 2 条（sarong→umbrella 0.9990、weasel→lesser panda 0.9984）；逐例素材见 `outputs/report_assets/table_cases.{csv,md}`（5 型号 ×（2 正 + 2 误）= **20 行**，含 案例图 / 文件名 / 真值 / 预测 Top-1 / 置信度 / Top-5 / 判定）。**原判定列与缺口列保留不改**（t2 时点的文字分析确为「1 个正确案例 + 1 句泛化错误描述」）；该缺口已于 t4 处置 —— `report/REPORT.md:144` 的 §4.5 标题即「Top-5 样例与正误案例（**2 正 + 2 误**）」。

> **后记（2026-09-17 追加，任务 t25）**：本节 **1.1-10** 的增量为**按现行产物重算**：`report/REPORT.md:179-182` 与 §五 表（`report/REPORT.md:173-174`：M0.9 = 参数量 5.067 M / Top-1 68.70 / P50 7.68 ms，M1.0 = 6.810 M / 69.20 / 10.40 ms）→ 参数量 `6.810/5.067 − 1 = +34.4 %`、MACs `1.143/0.847 − 1 = +34.9 %`、P50 `10.40/7.68 − 1 = +35.4 %`、Top-1 `+0.50` 个点。原写的 `+33.0 %` / `+7.0 %` / `+1.70 点` 与 `+0.96` / `0.00` 属旧自建子集那一轮的口径，**全仓已无该组值**（现行边际收益是 **acc/ms 口径**的 4 个值 +0.18 / +3.05 / +0.08 / +0.14，取自 `outputs/advanced/marginal_returns.csv` 与 `report/REPORT.md:179-182`）。

> **判定复核（更正数字后重新确认，非默认保留）**：**1.1-10 判定仍为「满足」** —— 「较大模型在当前设备上收益是否合理」这一问题在现行数据上**结论方向不变且更明确**：M1.0 付出 +34.4 % 参数量、+34.9 % MACs、+35.4 % P50，只换到 +0.5 个点 Top-1（边际收益 +0.18 acc/ms，是四条边际里最差的一档），报告 §五 给出的「不划算」结论有落盘依据。
> **判定复核（更正证据后重新确认，非默认保留）**：**1.1-11 判定仍为「满足」** —— 现行 §4.3 已是**七类逐行分开**的表（`report/REPORT.md:127-133`），表前显式声明「本表不含任何论文/官方公布值」（`report/REPORT.md:167`），且把论文/官方公布值与本机实测**相减**的那张违规三方表**已从全仓删除**（复核（**可机检**，见 `tools/check_audit_evidence.py` 的断言校验维度）：`git grep -F -- "论文 / 官方公布" -- ':!report/REQUIREMENTS_AUDIT.md'` → **0 命中**；`git grep -F -- "−0.50" -- ':!report/REQUIREMENTS_AUDIT.md'` → **0 命中**；`git grep -F -- "−0.10" -- ':!report/REQUIREMENTS_AUDIT.md'` → **0 命中**（本文件自身引用这三个字面量，属自指，故显式排除本文件）；删除记录见 _coord/POLICY_IMAGENETV2_ONLY.md（仓库外协调文件）§15.2）。即：**没有把官方表格直接当作个人实验结果，也没有再并列/相减**。

### 3.2 基础任务 1.2：模型理解（14 项说明 + 自绘结构图 9 要素，第 7 页）

14 项说明的成文载体：`report/REPORT.md` §2.1~§2.3 与 §三；`report/PPT_CONTENT.md` P03；PPT 第 3 页左栏（PDF 文本可读）。

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **1.2-1** RepViT 为什么称为「从 ViT 视角重新审视 Mobile CNN」 | `report/REPORT.md:53-57`（增量架构改造实验：功劳在宏观架构与训练策略，不在 MHSA）；`report/ppt_svg` 路线图；`report/sources/literature.yaml#paper`（含 arXiv URL） | 满足 | — |
| **1.2-2** RepViT 为什么仍然属于纯 CNN | `report/REPORT.md:59-63`（无自注意力算子；ONNX 图节点全为 Conv/Gemm/Add/Mul/Div/Clip/Relu，无 MatMul）；对应产物 `outputs/reparam/repvit_m0_9_pet37_onnx_nodes.json` 的 `op_histogram` 可复核 | 满足 | — |
| **1.2-3** MobileNetV3 与 RepViT Block 的主要区别 | `report/REPORT.md:69`（Token/Channel Mixer 解耦 vs MobileNetV2 倒残差把空间与通道耦合；表头「与 ViT 的 MHSA + FFN 一一对应」） | 满足 | — |
| **1.2-4** Token Mixer 和 Channel Mixer 分别负责什么 | `report/REPORT.md:69-70`、`report/REPORT.md:107-108`、PPT 第 3 页左栏 | 满足 | — |
| **1.2-5** 为什么要将二者分离 | `report/REPORT.md:69`（容量可独立调节） | 满足 | — |
| **1.2-6** 深度卷积和逐点卷积分别起什么作用 | `report/REPORT.md:76`（depthwise 空间局部混合 ≈ k²C；pointwise 跨通道 ≈ C²；把 C²k² 降到 C² + k²C） | 满足 | — |
| **1.2-7** 结构重参数化的训练态与推理态有什么区别 | `report/REPORT.md:70`、§12.1（训练态 = 3×3 dw 分支 + 1×1 dw 分支 + 残差 + BN；推理态 = 单个 3×3 depthwise）；`outputs/reparam/*_structure_before/after.txt` | 满足 | — |
| **1.2-8** 为什么重参数化后可以减少推理开销 | `report/REPORT.md:70`、§12.5（节点 −93、BN 24→0、Conv 126→103）；`report/REPORT.md:136`（§4.3 表第 6 行「**PyTorch 推理**」）实测延迟（M0.9 融合后 mean 8.41 ms vs 未融合 17.15 ms，降 51.0%；见下方「后记（2026-09-17 追加，任务 t22）」） | 满足 | — |
| **1.2-9** 为什么降低 Channel Mixer 扩张比例并增加网络宽度 | `report/REPORT.md:71` | 满足 | — |
| **1.2-10** Early Convolution Stem 的设计目的 | `report/REPORT.md:72` | 满足 | — |
| **1.2-11** 为什么使用更深的下采样层 | `report/REPORT.md:73` | 满足 | — |
| **1.2-12** SE 模块放置位置的考虑 | `report/REPORT.md:74`（仅部分 Block 放；全局池化 + 两次全连接延迟不可忽略） | 满足 | — |
| **1.2-13** RepViT 为什么采用简单分类头 | `report/REPORT.md:75`（单个 BN + Linear `NormLinear`） | 满足 | — |
| **1.2-14** M0.9、M1.0、M1.1、M1.5 和 M2.3 的主要差别 | `report/REPORT.md:78-81`（同套 Block 的宽度/深度缩放；实测 M0.9 四 stage Block 数 [2,2,14,2]）；`outputs/benchmarks/family_summary.csv`（5 型号同口径实测） | 满足 | — |
| **1.2-15**（附注）可以引用论文图片作为补充，但不能只复制论文结构图代替个人理解 | 结构图由 `tools/draw_arch.py` 用 **forward hook 现测**真实张量形状生成；`report/REPORT.md:87` 显式声明「非论文插图、非手画」；产物 `outputs/architecture/repvit_m0_9_arch.png` + `outputs/architecture/repvit_m0_9_arch.md` | 满足 | — |

> **后记（2026-09-17 追加，任务 t22）**：本节 **1.2-8** 证据列的延迟数字**已按现行产物更正**为 **M0.9 融合后 mean 8.41 ms vs 未融合 17.15 ms（降 51.0%）**；原引用「7.97 / 15.51 / 48.6%」属自建子集那一轮的评价，**全仓已无该组值**。依据：`outputs/pretrained_eval/repvit_m0_9/latency.json`（warmup=10 / runs=50，融合与未融合两组同机同后端）与 `report/REPORT.md:136`（§4.3 表第 6 行「**PyTorch 推理**」）。
> **判定复核（更正数字后重新确认，非默认保留）**：判定**仍为「满足」** —— 重参数化把 PyTorch 侧延迟降低约一半（17.15 → 8.41 ms，−51.0%），与「合并分支后减少推理开销」的机制方向一致；机制侧证据链也仍在：`outputs/reparam/repvit_m0_9_pet37_onnx_nodes.json`（BN 24→0、Conv 126→103、节点 480→387）与 `outputs/reparam/*_structure_before/after.txt`。

**自绘结构图 9 要素**（原文「候选人需要自行绘制 RepViT-M0.9 结构图，至少标明：」）：

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **1.2-F1** 输入尺寸 | `report/REPORT.md:90`「输入 224 × 224 × 3 (RGB)」；`outputs/architecture/repvit_m0_9_arch.md` | 满足 | — |
| **1.2-F2** Stem | `report/REPORT.md:92-93`（`stem.conv1` 3×3 s2 → 24ch 112×112；`stem.conv2` 3×3 s2 → 48ch 56×56） | 满足 | — |
| **1.2-F3** 四个主要阶段 | `report/REPORT.md:95-98`（Stage 0~3） | 满足 | — |
| **1.2-F4** 特征图分辨率变化 | `report/REPORT.md:105`「224 → 112 → 56 → 28 → 14 → 7（总下采样 32×）」 | 满足 | — |
| **1.2-F5** 通道数变化 | `report/REPORT.md:105`「3 → 24 → 48 → 96 → 192 → 384」 | 满足 | — |
| **1.2-F6** RepViT Block | `report/REPORT.md:107-108`（Token Mixer RepVGGDW + Channel Mixer + 残差 + 部分 SE） | 满足 | — |
| **1.2-F7** Global Average Pooling | `report/REPORT.md:100`「GAP → 384 维」 | 满足 | — |
| **1.2-F8** 分类头 | `report/REPORT.md:102`（`RepVitClassifier` → `NormLinear(BatchNorm1d + Linear)`） | 满足 | — |
| **1.2-F9** 最终输出维度 | `report/REPORT.md:102`「→ 37 维」（Pet-37 任务口径）；`tools/report_spec.py` 的 `FIGURE_SPEC` 也把 9 要素写进同一登记项 | 满足 | 输出维度按任务口径写 37；如需同时体现 ImageNet 1000 类口径，可在图注补一行（非必须） |

### 3.3 基础任务 2：RepViT-M0.9 基础训练（15 分，第 7~9 页，10 条）

「必须说明」6 项与硬约束 3 条同表列出。

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **2-1**（说明）使用了哪一种实现 | `report/REPORT.md:259`（`impl = timm`、`distillation = False` 单头）；`configs/baseline.yaml`；`configs/baseline_effective` 快照 `outputs/logs/baseline_config_effective.yaml` | 满足 | — |
| **2-2**（说明）环境版本 | `outputs/env_snapshot.json`；`report/REPORT.md:27-30`（Python 3.14.5 / torch 2.14.0+cu126 / timm 1.0.29 / ORT 1.30.0）；`requirements.txt` + `requirements.lock.txt` | 满足 | — |
| **2-3**（说明）预训练权重来源 | `report/REPORT.md:259`；`README.md:126-138`（官方 Releases v1.0 + 字节数/SHA256 表）；`outputs/metrics/weight_sha256.json` | 满足 | — |
| **2-4**（说明）与官方代码的差异 | `report/REPORT.md:287-299`（Python/timm 版本、权重载入、训练循环三项对照）；`PROVENANCE.md` §一（官方文件 3 处适配） | 满足 | — |
| **2-5**（说明）分类头如何从 1000 类修改为 37 类 | `report/REPORT.md:281-285`（`timm.create_model(..., num_classes=37, distillation=False)`，不用 `reset_classifier`）；PPT 第 6 页「正确做法 / 禁止做法」对照 | 满足 | — |
| **2-6**（说明）权重加载时是否存在缺失或不匹配参数 | `outputs/metrics/weight_load_report.json`（`n_missing = 2`、`n_unexpected = 0`、`missing_keys = classifier.classifier.l.{weight,bias}`）；`report/REPORT.md:284-285` | 满足 | — |
| **2-7** 正确准备训练、验证和测试数据 | `datasets/make_pet_split.py`（唯一划分脚本）+ `datasets/lists/pet_{train,val,test}.txt`（2940 / 740 / 3669 行）+ `datasets/build_pet_class_map.py`；`python tools/assert_data.py --root data/oxford-iiit-pet` → **ALL PASS**（`outputs/metrics/dataset_report.json` 的 `pass: true`） | 满足 | — |
| **2-8** 检查图片、类别编号和类别名称映射 | `tools/train.py:635-646` 启动即断言（两份类别文件互逆且同序、行数 == K）；`outputs/metrics/pet_label_audit.json`；`tools/selfcheck.py` 的 `data.labels` / `data.classes` PASS | 满足 | — |
| **2-9** 确保训练集、验证集和测试集不泄漏 | `python datasets/audit_leakage.py` → `outputs/metrics/leakage_check.json`：`stem_intersection` 0/0/0、`md5_intersection` 0/0/0、`internal_duplicates` 空、`orphan_images_used` 空、`pass: true` | 满足 | — |
| **2-10** 完成至少一次完整训练和验证 | `outputs/logs/train_baseline_20260913_141931.log`（+ `train_baseline_rerun.log`）；`outputs/logs/baseline_metrics.{csv,jsonl}`（40 epoch）；`outputs/logs/baseline_steps.jsonl` | 满足 | — |
| **2-11** 保存 best checkpoint 和 last checkpoint | 本机 `checkpoints/baseline_best.pt`（19,386,027 B）与 `checkpoints/baseline_last.pt`（57,605,657 B）均存在；入库版本**只含 best**（`.gitignore` 明确排除 `checkpoints/*_last.pt`，理由写在 .gitignore 注释与 README「关于提交的 checkpoint」） | 满足 | 「保存」成立（本机两文件俱在）；「提交」只交 best，与 §6.3 提交清单要求的「Baseline 权重」不冲突 |
| **2-12** 提交完整训练日志 | `outputs/logs/train_baseline_*.log`、`train_opt_*/` 共 30+ 份日志入库；`outputs/logs/*_metrics.csv` / `*_steps.jsonl` | 满足 | — |
| **2-13** 在验证集计算 Top-1、Top-5 和 Macro-F1 | `outputs/metrics/baseline_val.json`；`outputs/logs/baseline_metrics.csv` 的 `val_top1` / `val_top5` / `val_macro_f1`；`tools/selfcheck.py` 的 `train.valmetrics` PASS（含 per-class） | 满足 | — |
| **2-14** 使用最佳模型在 test 上完成一次最终评价 | `outputs/metrics/baseline_test.json`（`top1 = 0.9234123739438539`、`macro_f1 = 0.9221970249315374`、`num_samples = 3669`、**`eval_count = 1`**）；选模依据固定为 `val_macro_f1`（`tools/train.py:623` 硬断言 `metric_for_best == "val_macro_f1"`） | 满足 | 仅对 baseline 成立；4 个优化臂 `eval_count = 2`，见 §9 得分项 G-1 |
| **2-15** 统计模型参数量和模型文件大小 | `outputs/metrics/count_params.json`（未融合单头 C=37 = 4,732,805）；`outputs/metrics/backbone_updated.json` / `outputs/metrics/baseline_test.json` 的 `checkpoint_best_mb`；README 表格 | 满足 | — |
| **2-16** 输出部分测试集分类结果 | `outputs/predictions/baseline_test_preds.csv`（**3669 行数据** + 11 列，含 `top5_idx` / `top5_names` / `top5_probs`）；`outputs/predictions/test_top5_baseline_grid8.png` | 满足 | — |
| **2-17**（硬约束）基础训练必须更新 RepViT 骨干网络参数 | `python tools/check_backbone_updated.py --ckpt checkpoints/baseline_best.pt` → `outputs/metrics/backbone_updated.json`：`changed_tensors = 706`，其中骨干 699；`tools/selfcheck.py` 的 `train.backbone` PASS | 满足 | — |
| **2-18**（硬约束）不得仅训练最后一个线性分类层作为完整 Baseline | 同 2-17（699 个骨干张量变化）；`configs/baseline.yaml` 无冻结项（`backbone_lr_scale = 0.1 > 0`，`tools/train.py:621` 断言）；`configs/opt_disc.yaml` 才是差异化 lr 的对照臂 | 满足 | — |
| **2-19**（硬约束）训练轮数不作绝对限制，推荐 30～50 epochs，需说明理由 | 统一 40 epoch；`report/REPORT.md:265-267` 说明「满预算、早停等价于关闭」的理由；`outputs/metrics/same_budget.json` | 满足 | — |
| **2-20**（硬约束）不得只加载已在 Oxford-IIIT Pet 上训练完成的第三方模型 | 训练起点是 ImageNet-1K 预训练 timm/HF 权重（`timm/repvit_m0_9.dist_300e_in1k`，`report/REPORT.md:259`），非 Pet 预训练；`outputs/metrics/weight_load_report.json` 记录权重来源与 SHA256 | 满足 | — |

### 3.4 基础任务 3：一项模型优化实验（10 分，第 9~10 页，10 条提交项 + 6 项控制变量）

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **3-1** Baseline 当前存在的问题 | `report/REPORT.md:305-309`（2940 张样本 vs 4.73M 参数；Label Smoothing 仅部分缓解；分类头需 1e-3 级 lr；增强缺少旋转/剪切/色调） | 满足 | — |
| **3-2** 优化方法希望解决的问题 | `report/REPORT.md:311-318`（四方案各列「改动 / 假设 / 预期方向」） | 满足 | — |
| **3-3** 明确的实验假设 | 同 3-2（每方案一列「假设」）；PPT 第 7 页四张卡片各有「假设」。`report/REPORT.md:398-401` 还记录了「A 的预期（延长收敛）与实测不一致」 | 满足 | — |
| **3-4** 修改的代码或配置 | `configs/opt_combo.yaml`（12 个差异键）+ 其余 6 份 `configs/opt_*.yaml`；`outputs/logs/opt_*_config_effective.{yaml,json}`（有效配置快照） | 满足 | — |
| **3-5** Baseline 与优化模型的完整配置 | `outputs/logs/baseline_config_effective.yaml` 与 `outputs/logs/opt_combo_config_effective.yaml`（全字段落盘）；`outputs/report_assets/table_configs.md` | 满足 | — |
| **3-6** 两组实验的训练预算 | `outputs/metrics/same_budget.json`（8 组均为 `(epochs, steps_per_epoch) = (40, 45)`，`total_iters_equal = True`）；`python tools/same_budget.py` | 满足 | — |
| **3-7** Top-1、Top-5 和 Macro-F1 结果 | `outputs/metrics/{baseline,opt_combo}_test.json`；`outputs/logs/opt_compare_summary.csv`；`outputs/report_assets/table_training_summary.md` | 满足 | — |
| **3-8** Baseline 与优化模型训练曲线 | `outputs/curves/opt_compare.png`（5 子图同图）；`outputs/curves/baseline_vs_opt_combo_compare.png` | 满足 | — |
| **3-9** 同一批图片上的预测对比 | `outputs/predictions/baseline_vs_opt_combo_predict_compare.json`（含 `outputs/predictions/pet_compare_ids.json`，按 `image_id` inner join **不按行号**）；`outputs/predictions/compare_baseline_vs_opt_combo_grid4.png` | 满足 | — |
| **3-10** 提升或下降的原因分析 | `report/REPORT.md:348-355`（0.6 个点窄带 + 二项分布 95% CI ±0.9 + 重跑波动 ±0.08/±0.40 + 无单调关系 → 结论「未超出随机波动」，负面结论也如实写）；§9.1 组合消融交互项（A 抵消 B 的大部分伤害，`+0.541` 超加性） | 满足 | — |
| **3-C1**（控制变量）数据划分一致 | 8 组共用 `datasets/lists/pet_{train,val,test}.txt`；`tools/diff_config.py` 叶子级比对不含 data 段差异 | 满足 | — |
| **3-C2**（控制变量）预训练权重一致 | 8 组 `model.pretrained_tag` / 起点权重相同（`configs/*.yaml` 的 `model` 段无差异键）；`report/REPORT.md:325` | 满足 | — |
| **3-C3**（控制变量）随机种子一致 | 全部 `seed: 42`；`configs/*.yaml` 的 `_expected_diff_` 不含 seed；`tools/selfcheck.py` 的 `opt.diff` PASS | 满足 | — |
| **3-C4**（控制变量）训练轮数或等价训练预算一致 | 40 epoch × 45 step 全组一致；`python tools/same_budget.py` | 满足 | — |
| **3-C5**（控制变量）评价方式一致 | 同一 `tools/evaluate.py` / 同一 test loader；`configs/*.yaml` 的 `aug.val` 段无差异键 | 满足 | — |
| **3-C6**（控制变量）最佳模型选择标准一致 | `tools/train.py:623` 断言 `metric_for_best == "val_macro_f1"`；`outputs/logs/*_metrics.csv` 均记录 `val_macro_f1` | 满足 | — |
| **3-N1**（附注）优化结果不一定必须提升准确率，实验真实即可 | `report/REPORT.md:348-355` 直接结论「四项优化方法都没有带来超出随机波动的真实增益」 | 满足 | — |
| **3-N2**（附注）不得将「换成更大的 RepViT 型号」作为基础任务中唯一的优化方法 | 基础任务的优化是训练方法类（增强 / Mixup / RandAugment / 差异化 lr），型号对比另放 §5（进阶任务 1） | 满足 | — |

### 3.5 基础任务 4：曲线、可解释性与效果分析（8 分，第 10~11 页）

**必须提供的曲线（5 条）**

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **4-C1** train loss | `outputs/curves/opt_compare.png` 子图 1；数据源 `outputs/logs/{baseline,opt_combo}_metrics.csv`；`python tools/plot_curves.py --runs baseline=... opt_combo=... --out outputs/curves/opt_compare.png` | 满足 | — |
| **4-C2** validation loss | 同图子图 2（`val_loss` 列） | 满足 | — |
| **4-C3** validation Top-1 Accuracy | 同图子图 3（`val_top1`） | 满足 | — |
| **4-C4** validation Macro-F1 | 同图子图 4（`val_macro_f1`） | 满足 | — |
| **4-C5** learning rate | 同图子图 5（`lr_bb` / `lr_head`） | 满足 | — |
| **4-C6**（附注）Baseline 和优化模型应尽量绘制在同一张图，或采用相同坐标范围 | `tools/plot_curves.py` 的 `--runs a=... b=...` 单图输出；PPT 第 8 页标注「曲线共用坐标范围」 | 满足 | — |

**必须提供的结果（PDF 原文 10 条 bullet；契约写 9 项，差异见文首『计数口径声明』表）**

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **4-R1** Baseline 与优化模型指标对比表 | `outputs/metrics/compare_baseline_vs_opt_combo.{csv,md}`；`outputs/logs/opt_compare_summary.csv`；`report/REPORT.md` §9 八组表 | 满足 | — |
| **4-R2** 归一化混淆矩阵 | `outputs/confusion_matrix/baseline_cm.png`（+ `outputs/confusion_matrix/opt_combo_cm.png`）；`python tools/check_cm.py --file outputs/confusion_matrix/baseline_cm.csv` → 形状 37×37、行和 ≈1（`tol_effective` 容差内 0 违规行）、`argmax_on_diagonal` 37/37 | 满足 | — |
| **4-R3** 每类准确率或每类 F1 柱状图 | `outputs/confusion_matrix/baseline_per_class_f1.png`（+ `outputs/confusion_matrix/opt_combo_per_class_f1.png`）；数据 `outputs/confusion_matrix/baseline_per_class.csv`（按 F1 升序） | 满足 | — |
| **4-R4** 至少 8 张测试集预测结果 | `outputs/predictions/test_top5_baseline_grid8.png` + 8 张单图 `outputs/predictions/test_top5_baseline_grid8_case01.png`～`outputs/predictions/test_top5_baseline_grid8_case08.png` | 满足 | — |
| **4-R5** 至少 4 张 Baseline 与优化模型的同图对比 | `outputs/predictions/compare_baseline_vs_opt_combo_grid4.png` + `outputs/predictions/baseline_vs_opt_combo_predict_compare.png`；配对逻辑见 `outputs/predictions/pet_compare_ids.json` | 满足 | — |
| **4-R6** 至少 2 个正确案例 | `outputs/predictions/case_correct_baseline{,_case01,_case02}.png`（2 例） | 满足 | — |
| **4-R7** 至少 2 个典型失败案例 | `outputs/predictions/case_wrong_baseline{,_case01,_case02}.png`（2 例） | 满足 | — |
| **4-R8** 至少 4 张 Grad-CAM 结果，其中包含正确和错误案例 | `outputs/gradcam/` 共 15 项：`outputs/gradcam/gradcam_correct_baseline.png` + 6 张 `*_correct.png`、`outputs/gradcam/gradcam_wrong_baseline.png` + 6 张 `*_wrong.png`、`outputs/gradcam/gradcam_layer_compare_baseline.png`；元信息 `outputs/metrics/gradcam_meta.json` | 满足 | 数量远超 ≥4，正误两类齐全 |
| **4-R9** 至少 5 张训练集以外的实际图片预测结果 | `external/` 8 张跨集合真实照片（`external/images_manifest.csv` 逐张登记来源/许可）+ `outputs/predictions/external_top5_pet37_grid5.png`（5 张） | 满足 | 图片来自 ImageNet val 而非自拍（原因与替代声明见 `PROVENANCE.md` §六）；8/8 判对只作为小样本观察 |
| **4-R10** 实际图片显示 Top-5 类别及置信度 | `outputs/benchmarks/external_top5_repvit_m0_9_pet37.csv`（Top-5 类别 + 置信度）；`outputs/predictions/external_top5_pet37_grid5_case01.png`～`outputs/predictions/external_top5_pet37_grid5_case05.png`（5 张） | 满足 | — |
| **4-N1**（附注）不得只展示效果最好的图片而回避失败案例 | 4-R7 的 2 个失败案例 + `gradcam_wrong_*` 6 张 + 10.3 节「最难三类」表 + PPT 第 9 页专页失败案例 | 满足 | — |

**必须分析（10 条）**

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **4-A1** 模型是否正常收敛 | `report/REPORT.md:387-388`（epoch 5 前快速上升、epoch 15 后平台期、末 8 轮 val Top-1 极差 0.95 点） | 满足 | — |
| **4-A2** 是否存在明显过拟合或欠拟合 | `report/REPORT.md:389-392`（末轮 train_loss 0.736 / val_loss 0.352；val_loss 从 3.65 → 0.35、末 8 轮 0.34~0.35 无持续上升 → 未见过拟合形态；并解释 Label Smoothing 抬高 train loss 的假象） | 满足 | — |
| **4-A3** 学习率变化与指标变化的关系 | `report/REPORT.md:393-394`（warmup 3 ep 上升最快；cosine 中段 epoch 10~30 为主要增长区间；末段 lr 1e-5 趋平台） | 满足 | — |
| **4-A4** 优化方法对收敛速度的影响 | `report/REPORT.md:395-401`（按 `val_macro_f1` 最优 epoch：baseline 34、A 31、B 32、A+B 31；首次进入「自身最优 −0.005」平台的 epoch：baseline 15、A/A+B 12、B 15；并写明与 §8.2 预期不一致 + 单种子范围限定） | 满足 | — |
| **4-A5** 哪些品种之间容易混淆 | `report/REPORT.md:410-419`（按 F1 升序最难三类：Staffordshire Bull Terrier 0.691、American Pit Bull Terrier 0.703、Ragdoll 0.766）；数据 `outputs/confusion_matrix/baseline_per_class.csv` | 满足 | — |
| **4-A6** 错误来自主体外观、姿态、遮挡还是背景 | `report/REPORT.md:432-436` 三成因①主体外观相似②姿态/视角极端③背景干扰；§11.4 给出背景依赖样本比例约 6.2% | 满足 | — |
| **4-A7** Grad-CAM 是否关注到合理区域 | `report/REPORT.md:460-465`（正确案例热力图集中头/躯干、背景接近 0；错误案例同时点亮两类共有特征区，定位对但判别力不足） | 满足 | — |
| **4-A8** 模型是否出现依赖背景的现象 | `report/REPORT.md:467-470`（约 6.2% 样本背景有明显响应；与 §11.5 分布差异 = 外部背景纹理强 24% 互证） | 满足 | — |
| **4-A9** 实际图片与数据集图片存在什么分布差异 | `report/REPORT.md:472-483`（短边中位数 339.5 → 375.0、亮度均值 0.455 → 0.503、亮度 std 0.218 → 0.256、边缘密度 0.132 → 0.148、背景复杂度比 0.478 → 0.593）；数据 `outputs/predictions/distribution_compare_summary.csv` + `outputs/predictions/distribution_compare_raw.csv` | 满足 | — |
| **4-A10** 置信度高是否一定代表预测可靠 | `report/REPORT.md:489-493`（存在置信度 > 0.9 的错误预测；ECE 偏高、温度缩放可显著降低）；`outputs/advanced/interp/calibration.json`（`temperature = 0.6149`） | 满足 | — |

### 3.6 基础任务 5：ONNX 多模型基础部署（12 分，第 11~13 页）

**（a）至少导出并部署的三个模型**

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **5-M1** RepViT-M0.9 官方 ImageNet 预训练模型 | `onnx/repvit_m0_9_in1k.onnx`（20,359,089 B，入库）；`outputs/benchmarks/export_repvit_m0_9_in1k.json`；`python deploy/export_onnx.py --model repvit_m0_9_in1k` | 满足 | — |
| **5-M2** 另一种官方预训练型号 | `onnx/repvit_m1_0_in1k.onnx`（27,332,115 B，入库）；`onnx/repvit_m1_1_in1k.onnx`（33,061,019 B，入库） | 满足 | 交付 2 种官方替代型号 |
| **5-M3** 自行训练得到的最终 Oxford-IIIT Pet 模型 | `onnx/repvit_m0_9_pet37.onnx`（18,893,149 B，入库）；来源权重 `checkpoints/baseline_best.pt`；`outputs/benchmarks/export_repvit_m0_9_pet37.json` | 满足 | — |
| **5-M4** 至少提交三个 ONNX 模型 | `git ls-files onnx/` → **5 个**（`repvit_m0_9_in1k`、`repvit_m0_9_pet37`、`repvit_m1_0_in1k`、`repvit_m1_1_in1k`、`repvit_m1_5_in1k`），其中**官方 ONNX 4 个**；本机另有 `onnx/repvit_m2_3_in1k.onnx`（**唯一**被 `.gitignore` 排除的型号，仅指标与延迟入库） | 满足 | 5 ≥ 3；「入库的官方 ONNX **4 个**」满足进阶 1「部署至少四种官方型号」，见 §9 得分项 **G-6（已闭合）** |

**（b）基础要求（PDF 原文 13 条 bullet + 1 条附注；契约写 14 条）**

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **5.4.1** 输入尺寸：224×224 | `deploy/model_registry.py:15` 统一 `input_size=224`；`outputs/benchmarks/summary.csv` 的 `input_size = 224` | 满足 | — |
| **5.4.2** batch size：1 | `deploy/benchmark.py:76` 固定 `batch_size=1`；`deploy/infer_onnx.py` 单图入口 | 满足 | — |
| **5.4.3** 精度：FP32 | `outputs/benchmarks/summary.csv` 的 `precision = FP32`；`outputs/metrics/bench.jsonl` 同字段 | 满足 | — |
| **5.4.4** 使用 ONNX Runtime CPU | `outputs/benchmarks/summary.csv` 的 `provider_actual = ['CPUExecutionProvider']`；`onnxruntime==1.30.0`（`requirements.txt`） | 满足 | — |
| **5.4.5** 通过参数或配置文件选择模型 | `deploy/model_registry.py`（7 个登记 key 的单一事实来源 `get(key)` / `onnx_path(key)` / `labels_for(key)`）；`deploy/infer_onnx.py --model`、`deploy/benchmark.py --model`（可多次传入） | 满足 | — |
| **5.4.6** 独立实现图片预处理 | `deploy/infer_onnx.py:12-23` 用 **PIL 手写** `Resize → CenterCrop → ToTensor → Normalize`，显式「不 `import torchvision`」；`report/REPORT.md:594-597` 说明为何必须独立 | 满足 | — |
| **5.4.7** 独立实现 ONNX 推理 | `deploy/infer_onnx.py:33-42` 的 `build_session()` + `sess.run()`；不使用任何高层封装 | 满足 | — |
| **5.4.8** 独立实现 Softmax 和 Top-K 后处理 | `deploy/infer_onnx.py:25-31` 的 `softmax()` 与 `topk()` 均为手写实现；`report/REPORT.md:591-597` | 满足 | — |
| **5.4.9** 正确加载对应类别名称 | `deploy/model_registry.py:80-88` `labels_for()` 读 `labels/` 并按 `num_classes` **硬断言**行数；`labels/imagenet_classes.txt`（1000）/ `labels/pet_classes.txt`（37） | 满足 | — |
| **5.4.10** 输出 Top-1 和 Top-5 结果 | `deploy/infer_onnx.py:65-86`（`--topk` 默认 5，落盘 `topk=[{rank,index,prob,name}]`）；`outputs/benchmarks/external_top5_repvit_m0_9_pet37.csv` | 满足 | — |
| **5.4.11** 输出各类别置信度 | `deploy/infer_onnx.py:50` 的 `--dump-probs`（把完整 softmax 向量落盘）+ `softmax 长度 == 标签行数` 断言 | 满足 | — |
| **5.4.12** 统计每个 ONNX 模型的文件大小 | `outputs/benchmarks/summary.csv` 的 `file_size_mb`（6 型号）；`deploy/benchmark.py` 现场 `Path(...).stat().st_size` | 满足 | — |
| **5.4.13** 统计每个模型的推理延迟 | `outputs/benchmarks/summary.csv` 的 `mean_ms`/`p50_ms`/`p95_ms`/`min_ms`/`max_ms`；逐型号 `outputs/benchmarks/*_benchmark.json` | 满足 | — |
| **5.4.14**（附注）官方 ImageNet 模型和自行训练的 37 类模型必须分别使用正确的标签文件，不得混用类别映射 | `deploy/model_registry.py:53-65` 的 `_check()`：37 类条目断言 `distillation=0` 且 `crop_pct=0.875`，1000 类条目断言 `distillation=1` 且 `crop_pct=0.95`；`tools/selfcheck.py` 的 `labels.pair` PASS | 满足 | — |

**（c）PyTorch 与 ONNX 一致性（5 项 + 9 项排查清单附注）**

**口径核对（照抄 `report/LOGITS_AUDIT_FINDINGS.md` §1）**：本批是 **B1**——Pet-37 baseline 融合态 ↔ `onnx/repvit_m0_9_pet37.onnx`，`datasets/lists/pet_test.txt` 按顺序前 **12 张真实图片（n=12）**，每张独立预处理一次后把同一张量送两端；**与 B4（重参数化，32 个固定随机输入）不是同一批实验**。

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **5.5.1** 最大 logits 绝对误差 | `outputs/metrics/consistency_repvit_m0_9_pet37.json` 的 `max_abs_logits = 6.198883056640625e-06`（展示 **6.199e-06**）；官方型号（`source = imagenetv2_mf_1000`）：M0.9 **`1.383e-05`**、M1.0 **`1.335e-05`**、M1.1 **`1.860e-05`**、M1.5 **`1.144e-05`**（`outputs/metrics/consistency_repvit_{m0_9,m1_0,m1_1,m1_5}_in1k.json`） | 满足 | **五份**同口径产物（B1/B2/B3/B12/B13），覆盖 Pet-37 与 4 个官方入库型号 |
| **5.5.2** 平均 logits 绝对误差 | 同文件 `mean_abs_logits = 1.5006899711048998e-06`（展示 **1.501e-06**）；官方型号 M0.9 `2.334e-06`、M1.0 `2.215e-06`、M1.1 `2.219e-06`、M1.5 `1.876e-06` | 满足 | — |
| **5.5.3** Top-1 类别是否一致 | 同文件 `top1_agree_rate = 1.0`、`mismatches = []`、`mismatch_count = 0` | 满足 | — |
| **5.5.4** Top-5 类别集合是否基本一致 | 同文件 `top5_set_agree_rate = 1.0` | 满足 | — |
| **5.5.5** 固定测试集上的 Top-1 一致率（推荐 ≥99%） | 同文件 `top1_agree_rate = 1.0`、`verdict = "PASS"`、`threshold = {top1_agree_rate: 0.99, max_abs_logits: 1e-3}`；`tools/selfcheck.py` 的 `onnx.consistency`、`onnx.realimg` PASS；复跑副本 `outputs/verification/consistency_repvit_m0_9_pet37_n12.json` | 满足 | 100% ≥ 99% |
| **5.5.6**（附注）如未达到 99%，需分析 Resize/Center Crop 顺序、RGB/BGR、插值、mean/std、Softmax 维度、eval 模式、BN 状态、导出方式、数值精度（9 项） | 已达到 99%，无需逐项排查；但 9 项口径在代码层都有对应实现：`deploy/infer_onnx.py:20`（顺序注释「反了差 236/255」）、`:14`（必须 RGB）、`:18`（BICUBIC）、`deploy/model_registry.py` 的 `mean/std`、`deploy/infer_onnx.py:25`（softmax axis）、`deploy/export_onnx.py:13-22`（eval→fuse→eval）、`outputs/reparam/repvit_m0_9_pet37_onnx_nodes.json` 的 `after.BatchNormalization = 0`、`deploy/export_onnx.py` 的 opset | 满足 | `report/REPORT.md:623-625` 的措辞已收敛为「未**观察到**九类典型问题的表现，不等于已排除九类问题」 |
| **5.5.7** 「对每个 ONNX 模型至少比较」——原记「第 4 个入库 ONNX（`repvit_m1_1_in1k`）无同口径一致性记录」 | `git ls-files onnx/` 有 **5 个**入库 ONNX（其中**官方 ONNX 4 个**）；`outputs/metrics/consistency_*.json` 现有 **5 份**（Pet-37 + M0.9 + M1.0 + M1.1 + M1.5），即 4 个入库官方型号**全部**有同口径 n=12 产物 | 部分 | 本行判定为 t2 审计当时的快照。**2026-09-17 状态：该缺口已闭合**（`consistency_repvit_m1_1_in1k.json` / `consistency_repvit_m1_5_in1k.json` 已补齐），对应 §9 得分项 G-4 可关闭 |

**（d）结构重参数化要求（6 项）**

**口径核对**：本批是 **B4**——Pet-37 baseline，**32 个固定随机输入**（`torch.randn`，seed=20240912，batch=8，CPU FP32，`eval()` 后深拷贝再 `fuse()`），`max|Δ| = 7.093e-06`、`mean|Δ| = 2.031e-06`、Top-1 32/32、BN 107→0。**不得与 B1 的 6.199e-06 并写**。

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **5.6.1** 说明训练态 RepViT Block 中包含哪些分支 | `report/REPORT.md` §12.1（3×3 depthwise+BN、1×1 depthwise+BN、逐元素相加、ReLU 在 RepVGGDW 内部；Channel Mixer 不含可融合并行分支；Block 另有残差）；`outputs/advanced/block_22_fusion_report.txt` 的 `branches` | 满足 | — |
| **5.6.2** 说明卷积和 BatchNorm 如何进行等价融合 | `report/REPORT.md` §12.2（`W' = W·γ/√(σ²+ε)`、`b' = (b−μ)·γ/√(σ²+ε) + β`；多分支零填充相加 `W_fused = W_3x3' + pad(W_1x1') + pad(W_identity)`）；`tools/reparam_deep.py:11-16` 手写实现 | 满足 | — |
| **5.6.3** 确认导出的模型属于训练态结构还是推理态结构 | `report/REPORT.md` §13.4（三个交付 ONNX **全部推理态**，图内 `BatchNormalization = 0`、`Conv = 103`）；`outputs/reparam/repvit_m0_9_pet37_onnx_nodes.json` 的 `after`；`outputs/benchmarks/export_*.json` | 满足 | — |
| **5.6.4** 对转换前后的模型输出进行数值比较 | `python tools/reparam_verify.py --model repvit_m0_9_pet37 --weights checkpoints/baseline_best.pt --num-samples 32 --batch-size 8 --seed 20240912 --skip-onnx --out-dir outputs/verification/reparam_pet37` → `outputs/reparam/repvit_m0_9_pet37_reparam_report.json` 与 `outputs/reparam/logits_diff.json`（`num_samples = 32`、`logits_abs_max = 1.3994`） | 满足 | 32 个**固定随机输入**，非真实图片（口径已写死在 JSON 的 `cli` 段） |
| **5.6.5** 报告最大绝对误差 | `outputs/reparam/logits_diff.json` 的 `max_abs_err = 7.092952728271484e-06`（展示 **7.093e-06**）、`mean_abs_err = 2.030726818702533e-06`（展示 **2.031e-06**）；阈值 `max_abs_err: 1e-4`，`judge.pass = true` | 满足 | — |
| **5.6.6** 确认转换后 Top-1 预测是否一致 | `outputs/reparam/logits_diff.json` 的 `top1_same_count = 32`、`top1_same_rate = 1.0`、`top5_min_overlap = 5`、`top5_full_overlap_rate = 1.0` | 满足 | — |
| **5.6.7**（附注）若加载后已是推理态结构，需通过代码/模型结构/ONNX 计算图说明依据 | `report/REPORT.md` §12.5（节点 480→387、BN 24→0、Conv 126→103）+ `outputs/advanced/repvit_m0_9_pet37_structure_before.txt`（34,065 B）与 `outputs/advanced/repvit_m0_9_pet37_structure_after.txt`（14,187 B）（完整模块树）+ `outputs/reparam/*_onnx_nodes.json`（op_histogram） | 满足 | — |

**（e）性能测试要求（8 项）**

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **5.7.1** 预热 10 次 | `deploy/benchmark.py:107`（`--warmup` 默认 10）；`outputs/benchmarks/summary.csv` 的 `warmup = 10` | 满足 | — |
| **5.7.2** 正式运行 50 次 | `deploy/benchmark.py:108`（`--runs` 默认 50）；`outputs/benchmarks/summary.csv` 的 `runs = 50`；`outputs/metrics/bench.jsonl` 的 `raw_ms` 数组长度 50 | 满足 | — |
| **5.7.3** 报告平均延迟 | `outputs/benchmarks/summary.csv` / `outputs/metrics/bench.jsonl` 的 `mean_ms`；`report/REPORT.md` §14.2 | 满足 | — |
| **5.7.4** 报告 P50 和 P95 延迟 | 同表 `p50_ms` / `p95_ms`；`percentile_method = "numpy linear"` 已注明口径 | 满足 | — |
| **5.7.5** 记录 CPU 型号、内存和操作系统 | `outputs/metrics/bench.jsonl` 的 `env = {cpu, mem_gb: 15.8, os, logical_cores: 20}`；`outputs/benchmarks/summary.csv` 的 `cpu_model` / `os` | 满足 | — |
| **5.7.6** 记录 ONNX Runtime 版本 | `outputs/benchmarks/summary.csv` / `outputs/metrics/bench.jsonl` 的 `ort_version = 1.30.0` | 满足 | — |
| **5.7.7** 打印实际 Execution Provider | `deploy/benchmark.py:57` `print(f"EP(actual) : {sess.get_providers()}")`；同时落盘 `provider_actual` / `ep` 两个同义字段（注释说明「避免验收脚本二义」） | 满足 | — |
| **5.7.8** 记录模型输入输出节点信息 | `deploy/benchmark.py:58-59` **只打印到控制台**（`input : ...` / `output : ...`）；落盘字段 `onnx_input` / `onnx_output` 未写入 → `outputs/report_assets/table_benchmark_meta.md` 的 `io_nodes` 列显示 `in=NoneNone out=NoneNone`；`tools/check_report_assets.py` 报 `[FAIL] 性能测试元信息[io_nodes]: 缺失` | 部分 | 「记录」未落到交付产物，只在控制台出现且无 console 日志留档；修复见 §9 得分项 G-2 |
| **5.7.9**（附注）基础任务不要求固定 FPS，只要求流程正确、测试规范、可复现 | `report/REPORT.md` §14.1 协议表 + §14.3 mean/P50/P95 解读；未出现任何 FPS 排名式论述 | 满足 | — |
| **5.7.10**（附注）官方 iPhone 12 延迟不能与本地 ONNX Runtime CPU 延迟直接比较 | `report/REPORT.md:158-159`、PPT 第 4 页「官方 iPhone 12 延迟（0.9 / 1.0 ms）与本机 CPU 结果不可横向比较」 | 满足 | — |

### 3.7 基础任务 6：工程、报告与答辩（10 分，第 13~15 页）

**（a）项目代码结构（附注：不强制完全相同的结构，但必须清晰、可复现）**

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **6.1** `configs/` | 10 份 YAML：`configs/baseline.yaml`、`configs/pretrained_eval.yaml`、`configs/expected_diff.yaml`、`opt_{mix,randaug,disc,combo,abl_a,abl_b,abl_ab}.yaml` | 满足 | — |
| **6.2** `datasets/` | `datasets/` 下 10 个 `.py` + `datasets/lists/*`（5 份清单/元数据） | 满足 | — |
| **6.3** `models/` | `models/{__init__,build_model,repvit_official,repvit_timm}.py` | 满足 | — |
| **6.4** `tools/` 四个指定脚本 | `tools/eval_pretrained.py`、`tools/train.py`、`tools/evaluate.py`、`tools/visualize.py` **全部存在**（另有 40+ 个辅助脚本） | 满足 | — |
| **6.5** `deploy/` 四个指定脚本 | `deploy/export_onnx.py`、`deploy/infer_onnx.py`、`deploy/benchmark.py`、`deploy/model_registry.py` **全部存在**（另有 `deploy/compare_torch_onnx.py`、`deploy/demo_camera.py`、`deploy/run_all.sh`） | 满足 | — |
| **6.6** `outputs/` 五个指定子目录 | `outputs/curves/`、`outputs/predictions/`、`outputs/gradcam/`、`outputs/confusion_matrix/`、`outputs/benchmarks/` **全部存在且非空** | 满足 | — |
| **6.7** `checkpoints/` | `checkpoints/baseline_best.pt`、`checkpoints/opt_combo_best.pt`（入库）；本机另有 `*_last.pt` 与其余 6 组 best | 满足 | 入库只保留题目要求的两个权重，取舍理由写在 `.gitignore` 注释与 README 末节 |
| **6.8** `labels/` | `labels/{imagenet_classes.txt, imagenet_wnid_to_idx.json, pet_classes.txt, pet_class_to_idx.json, pet_species.json, pet_species.txt}` | 满足 | — |
| **6.9** `README.md` / `requirements.txt` / `report.pdf` | 三者均在仓库根目录（`report.pdf` = 实测 **1,278,056 B** / 26 页） | 满足 | — |
| **6.10** 代码清晰、可复现 | `bash tools/run_all.sh`（全流程）+ `sh deploy/run_all.sh` + `_run_m07.sh`；`requirements.lock.txt`（142 行冻结）；`python tools/selfcheck.py` → **34 PASS / 0 FAIL** | 满足 | — |
| **6.11** 不得在代码中写死个人电脑绝对路径 | `tools/selfcheck.py` 的 `paths` 检查 **0 处**（PASS）；`deploy/compare_torch_onnx.py:12` 的 `repo_rel()` 把一致性 JSON 的 `onnx_path`/`images` 规范为仓库相对路径 | 满足 | 代码层满足；**落盘内容**仍有 **32 个**入库文件含采集机绝对路径（**平台中立的复核命令**：`git grep -lE '[A-Za-z]:\\{1,2}Users'` → **32 个文件**；`git grep -cE '[A-Za-z]:\\{1,2}Users'` → **278 行**），属可移植性问题，见 §9 体验项 X-3 |

**（b）README 最低要求（12 条）**

`README.md` 的 14 节目录与 12 条要求一一对应（第 13/14 节是附加内容）。

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **R-1** 如何安装环境 | `README.md` §1「如何安装环境」（Python 3.14.5 / torch 2.14.0+cu126 安装命令 / `requirements.txt` / `tools/env_check.py` / 4 条实测坑） | 满足 | — |
| **R-2** 如何准备数据集 | `README.md` §2（§2.1 Pet：归档 MD5 校验 + `tools/assert_data.py` + `datasets/build_pet_class_map.py` + `datasets/make_pet_split.py` + `datasets/audit_leakage.py`；§2.2 官方评价子集：ImageNetV2 归档获取 + `datasets/make_imagenetv2_subset.py` + `tools/verify_imagenet_labels.py`） | 满足 | — |
| **R-3** 如何获取官方权重 | `README.md` §3（`bash tools/download_pretrained.sh` + `python tools/check_weights.py --dir checkpoints/pretrained` + 5 个权重的字节数与 SHA256 前 16 位） | 满足 | — |
| **R-4** 如何运行官方模型评价 | `README.md` §4（单型号 / 多型号 `tools/run_all_pretrained.py` / 仅校验数据三条命令 + 实测结果表） | 满足 | — |
| **R-5** 如何训练 Baseline | `README.md` §5（冒烟 + 正式训练命令 + 产物清单 + 训练协议说明） | 满足 | — |
| **R-6** 如何运行优化实验 | `README.md` §6（`tools/diff_config.py` 预检 + 四个方案训练命令 + `tools/same_budget.py` + `tools/compare_runs.py` + `tools/predict_compare.py`） | 满足 | — |
| **R-7** 如何进行验证和测试 | `README.md` §7（`tools/evaluate.py --audit` / `--split val` / `--split test --latency` + 「test 不得用于模型选择、调参或早停」说明） | 满足 | — |
| **R-8** 如何生成曲线、混淆矩阵和 Grad-CAM | `README.md` §8（`tools/plot_curves.py` / `tools/visualize.py` / `tools/plot_predictions.py` / `tools/plot_side_by_side.py` / `tools/gradcam.py` / `tools/predict_external.py` 六条命令） | 满足 | — |
| **R-9** 如何执行结构重参数化 | `README.md` §9（`tools/reparam_verify.py` 两条命令 + 实测结果表 + 融合公式与「必须 eval() 之后融合」的条件） | 满足 | — |
| **R-10** 如何导出不同型号的 ONNX 模型 | `README.md` §10（`deploy/export_onnx.py --model a b c`、`--no-fuse` 对照、`deploy/model_registry.py` 列出登记表） | 满足 | — |
| **R-11** 如何运行 ONNX 推理 | `README.md` §11（`deploy/infer_onnx.py --model ... --image ...` + 一致性对比命令 + §11.1 误差口径说明） | 满足 | — |
| **R-12** 如何完成性能测试 | `README.md` §12（`deploy/benchmark.py --warmup 10 --runs 50 --threads 4` + 必须同时报告的 10 项元信息 + 实测结果表） | 满足 | — |
| **R-N1**（附注）不得在代码中写死个人电脑绝对路径 | 同 6.11（`selfcheck` 的 `paths` 检查 PASS） | 满足 | 落盘侧见 §9 体验项 X-3 |

**（c）最终提交内容（PDF 原文 15 条 bullet，契约写 16 项，差异见文首『计数口径声明』）**

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **T-1** 完整项目代码 | `git ls-files` 共 464 个路径；核心包 `tools/`(48)、`deploy/`(8)、`datasets/`(12)、`models/`(4)、`utils/`(7) | 满足 | — |
| **T-2** 环境依赖文件 | `requirements.txt`（实测 **24 行**，其中非空 19 行 = 7 行注释 + 12 行依赖，均含版本）+ `requirements.lock.txt`（实测 **142 行** pip freeze）+ `outputs/env_snapshot.json` | 满足 | — |
| **T-3** 数据划分文件 | `datasets/lists/pet_train.txt`（2940）/`datasets/lists/pet_val.txt`（740）/`datasets/lists/pet_test.txt`（3669）+ `datasets/lists/pet_split_meta.json`；ImageNet 侧 `datasets/lists/imagenetv2_mf_1000.txt`（1000，ImageNetV2 matched-frequency） | 满足 | Pet 统一 `train_list/val_list` 属 §1.3 族的挂起项 1.4（考核方清单），本行只判定「划分文件已提交」 |
| **T-4** 所有实验配置 | `configs/` 10 份 YAML + `outputs/logs/*_config_effective.{yaml,json}`（8 组实验的有效配置快照）+ `outputs/report_assets/table_configs.{csv,md}` | 满足 | — |
| **T-5** 训练日志 | `outputs/logs/train_baseline_*.log`、`train_opt_*.log`（30+ 份）+ `*_metrics.csv`/`*_metrics.jsonl`/`*_steps.jsonl` 全部入库 | 满足 | — |
| **T-6** Baseline 权重 | `checkpoints/baseline_best.pt`（19,386,027 B，入库） | 满足 | — |
| **T-7** 最终优化模型权重 | `checkpoints/opt_combo_best.pt`（19,386,803 B，方案 A = 基础任务所选优化方案，入库） | 满足 | — |
| **T-8** 至少三个 ONNX 模型 | `git ls-files onnx/` → **5 个**（其中**官方 ONNX 4 个**） | 满足 | 「官方型号入库 **4 个**」的口径见 §3.6；原「只有 3 个」的缺口 **G-6 已闭合**（见 §9） |
| **T-9** 类别名称文件 | `labels/imagenet_classes.txt`（1000 行）+ `labels/pet_classes.txt`（37 行）+ 两份映射 JSON + `labels/pet_species.{json,txt}` | 满足 | — |
| **T-10** 曲线和可视化结果 | `outputs/curves/`(4)、`outputs/predictions/`(23 PNG + CSV)、`outputs/gradcam/`(15)、`outputs/confusion_matrix/`(8)、`outputs/advanced/interp/`(8) | 满足 | — |
| **T-11** 性能测试原始记录 | `outputs/metrics/bench.jsonl`（每行一次基准，含 `raw_ms` 50 个原始值）+ `outputs/benchmarks/*_benchmark.json`（逐型号）+ `outputs/benchmarks/summary.csv` | 满足 | `outputs/metrics/bench.jsonl` 存**两批**基准（早批与晚批），README §12 已警告引用时必须写明批次 |
| **T-12** 项目报告 PDF | `report.pdf`（实测 **1,278,056 B**、26 页，根目录；`report/REPORT.md` 为源文，`report/REPORT.docx` 为导出） | 满足 | 页数超出「建议 12～20 页」，见 §9 体验项 X-1 |
| **T-13** 答辩 PPT 源文件 | `report/答辩PPT_RepViT.pptx`（**15 页**）。**体积不固化**（t41 实测：两次重生成 8,119,653 → 8,119,664 B，zip 内含 `docProps/core.xml` 时间戳，字节不可复现）—— 实测 `python -c "import os;print(os.path.getsize('report/答辩PPT_RepViT.pptx'))"`+ `report/ppt_svg/*.svg`（15 份源） | 满足 | — |
| **T-14** 答辩 PPT 的 PDF 版本 | `report/答辩PPT_RepViT.pdf`（**15 页**）。**体积不固化**（同一轮导出稳定、但随 PPTX 每次重生成而变）—— 实测 `python -c "import os;print(os.path.getsize('report/答辩PPT_RepViT.pdf'))"` | 满足 | — |
| **T-15**（可选）备用演示视频 | 未提供 | 满足 | 题目原文标注为「**可选**的备用演示视频」，缺省不构成缺口；`report/REPORT.md` 与 PPT 均未声称提供视频 |

**（d）候选人需要明确标记（5 类）**

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **L-1** 官方代码 | `PROVENANCE.md` §一：`models/repvit_official.py` = 上游仓库 THU-MIG/RepViT 的 model/repvit.py（**仓库外路径**，非本仓库文件）逐字节拷贝（Apache-2.0，md5 `4c861e5e…`，16,962 B），仅 3 处适配；`README.md` §14 同源说明 | 满足 | — |
| **L-2** 第三方代码 | `PROVENANCE.md` §二表（timm 实现、官方权重 ×5、timm/HF 权重、Oxford-IIIT Pet、ImageNet-1K val、torchvision/torch/onnxruntime/sklearn/matplotlib/pandas/OpenCV 各带许可与用途） | 满足 | — |
| **L-3** 自己修改的代码 | `PROVENANCE.md` §一「三处适配」逐条列出（timm 1.x 导入路径 ×2、删除 `@register_model` 防注册表污染）；§四另有一张「AI 初稿中被实测推翻、并由人工修正的典型缺陷」10 行表 | 满足 | 未用独立标题「自己修改的代码」分节，靠 §一 的改动列 + §四 的修正表承载；见 §9 体验项 X-2 |
| **L-4** 自己新增的代码 | `PROVENANCE.md` §三（`tools/`、`deploy/`、`utils/`、`datasets/` 下全部 `.py` 与 `configs/` 下全部 `.yaml`）+ 8 个关键自撰模块表 | 满足 | — |
| **L-5** AI 辅助生成或修改的内容 | `PROVENANCE.md` §四（AI 参与的部分 / 人类负责的部分逐项划界）；`README.md` §14 第 4 项 | 满足 | — |
| **L-6**（附注）允许使用开源代码和 AI 辅助工具，但必须注明来源，并能够在答辩中解释提交内容 | 同 L-1~L-5；`report/REPORT.md` 附录 A 把 10 类实测缺陷与处置写成表；`PROGRESS.md` 有决策记录 | 满足 | 「能否解释」是答辩现场环节，静态不可判定 |

---

## 4. 六、报告、PPT 与现场答辩（补充覆盖，第 15~20 页）

> 本节不在任务契约的必列清单里，但 PDF 第 16 页的「报告必须区分 7 类来源」是指定核对项，且提交内容包含报告与 PPT，故一并逐条建行。

### 4.1 项目报告（16 项 + 页数 + 7 类来源 + 依据要求）

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **RP-0**（建议）正文控制在 12～20 页，不含附录 | `report.pdf` = **26 页**（PyMuPDF `page_count`；`report/SYNC_NOTES.md` §7.2 记录压缩前 37 页，本轮目标「≤30 页」） | 部分 | 建议性口径，非硬性；超出建议上限 6 页。修复见 §9 体验项 X-1 |
| **RP-1** 任务背景与复现范围 | `report/REPORT.md` 第一节（1.1 背景 / 1.2 复现范围「做/不做」两栏 / 1.3 环境指纹 / 1.4 三个参数量口径） | 满足 | — |
| **RP-2** RepViT 论文核心思路 | `report/REPORT.md` 第二节（§2.1~§2.3 逐条设计 + `report/sources/literature.yaml` 带 URL 的来源登记） | 满足 | — |
| **RP-3** RepViT-M0.9 结构 | 第三节（forward hook 现测结构图 + 分辨率/通道链 + 训练态/推理态对比） | 满足 | — |
| **RP-4** 官方预训练模型评价 | 第四节（4.1 评价口径 / 4.2 实测 / 4.3 三方对照 / 4.4 PyTorch 延迟 / 4.5 Top-5 与案例 / 4.6 收益分析） | 满足 | 案例成文数量见 1.1-7/1.1-8 |
| **RP-5** 多型号规模和性能比较 | 第五节（5 型号表 + 边际收益 + 帕累托推荐 + 口径混用警告） | 满足 | — |
| **RP-6** 数据集与数据划分 | 第六节（两个数据集两套标签 / 官方数据事实基线 / 划分方案 / 三层 + 内容级泄漏检查） | 满足 | — |
| **RP-7** Baseline 迁移训练 | 第七节（训练配置 / 结果 / 与官方代码差异；含权重键名交集为 0 的关键实测） | 满足 | — |
| **RP-8** 优化方法与实验假设 | 第八节（8.1 Baseline 问题 / 8.2 四方案假设表 / 8.3 控制变量的工程化定义） | 满足 | — |
| **RP-9** 定量结果 | 第九节（8 组实验表 + 噪声判据三条 + 9.1 消融交互项 + 9.2 预算等价） | 满足 | — |
| **RP-10** 曲线和混淆矩阵分析 | 第十节（10.1 五曲线 + 收敛/过拟合/学习率/收敛速度四问 / 10.2~10.5 混淆矩阵与失败归因） | 满足 | — |
| **RP-11** Grad-CAM 与失败案例 | 第十一节（11.1 实现与挂载层 / 11.2 结果 / 11.3 合理区域 / 11.4 背景依赖 / 11.5 分布差异 / 11.6 置信度可靠性） | 满足 | — |
| **RP-12** 结构重参数化 | 第十二节（12.1 分支 / 12.2 融合推导 / 12.3 为什么一致 + 为什么必须 eval / 12.4 实测 / 12.5 ONNX 图影响 / 12.6 单 Block 融合过程） | 满足 | — |
| **RP-13** ONNX 多模型部署 | 第十三节（13.1 三个交付模型 / 13.2 独立实现 / 13.3 一致性 / 13.4 部署侧重参数化） | 满足 | — |
| **RP-14** 性能测试 | 第十四节（14.1 协议 / 14.2 结果 / 14.3 mean-P50-P95 解读 / 14.4 为什么更大不一定按 MACs 变慢 / 14.5 集显） | 满足 | — |
| **RP-15** 遇到的问题和解决方法 | 第十五节（15.1~15.6 六项，含权重键名、`data/provided/` 结构性偏差、预算不等价、并发训练、GBK 编码、网络不可达） | 满足 | — |
| **RP-16** 总结与后续计划 | 第十六节（16.1 六条主要结论 / 16.2 可复现性 / 16.3 后续计划） | 满足 | — |
| **RP-17** 报告必须区分：**论文公布结果** | `report/sources/literature.yaml#paper`（arXiv URL + iPhone 12/CoreML 口径注）；`report/REPORT.md` §4.3 表头「论文 / 官方公布」；`outputs/report_assets/table_results.md:3` 的「论文公布」行 | 部分 | `report/REPORT.md` §4.3 把「论文公布」与「官方仓库公布」合并为一列；两者只在来源登记文件与自动生成素材里分开。修复见 §9 得分项 G-5 |
| **RP-18** 报告必须区分：**官方仓库公布结果** | `report/sources/literature.yaml#official_repo`（THU-MIG README Model Zoo，top1 0.787 / top5 0.791）；`report/REPORT.md` §7.3「官方仓库 vs 本任务」对照；`outputs/report_assets/table_results.md:4`「官方仓库公布」行 | 部分 | 与 RP-17 同源（合并列问题） |
| **RP-19** 报告必须区分：**官方权重实际运行结果** | `report/REPORT.md` §4.2/§4.3「本机实测」列（口径 = ImageNetV2 matched-frequency 1000 张固定子集） + `outputs/pretrained_eval/*/metrics.json`；`outputs/report_assets/table_results.md:5` 有该行**但数值列为空** | 部分 | 自动素材表该行未填值（正文有值）；修复见 §9 得分项 G-5 |
| **RP-20** 报告必须区分：**自行训练 Baseline 结果** | `report/REPORT.md` §7.2/§9 的 Baseline 行 + `outputs/metrics/baseline_test.json`；素材表 `outputs/report_assets/table_results.md:6` 行**数值为空** | 部分 | 同 RP-19 |
| **RP-21** 报告必须区分：**自行优化模型结果** | `report/REPORT.md` §9 的 opt_* 行 + `outputs/metrics/opt_*_test.json`；素材表 `outputs/report_assets/table_results.md:7` 行**数值为空** | 部分 | 同 RP-19 |
| **RP-22** 报告必须区分：**PyTorch 结果** | `report/REPORT.md:136`（§4.3 表第 6 行「PyTorch 推理」，PyTorch 侧延迟）、§12.4（PyTorch logits 误差，`report/REPORT.md:477`）；素材表 `outputs/report_assets/table_results.md:8` 行标注口径「batch=1, fp32, 本机 CPU」**但数值列为空** | 部分 | 同 RP-19 |
| **RP-23** 报告必须区分：**ONNX 部署结果** | `report/REPORT.md` §13/§14.2；素材表 `outputs/report_assets/table_results.md:9` 行**已填**（文件大小 18.89 MB、mean 7.20 / P50 7.12 / P95 7.69、EP = ORT CPUExecutionProvider） | 满足 | — |
| **RP-24**（附注）所有主要结论都应能在日志、配置、代码或运行结果中找到依据 | 每个 JSON 带 `command` / `timestamp` / `platform` 元字段；`report/REPORT.md` §16.2 列出去向；`report/LOGITS_AUDIT_FINDINGS.md` 建立了「数值 → 权威产物 → 复跑命令」的映射 | 满足 | — |

### 4.2 答辩 PPT（15 项建议结构 + 页数 + 性能图表 7 项标注）

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **PP-0**（建议）PPT 控制在 10～15 页 | `report/答辩PPT_RepViT.pdf` = **15 页**；`report/答辩PPT_RepViT.pptx` = 15 slides（`python -c "from pptx import Presentation; print(len(Presentation('report/答辩PPT_RepViT.pptx').slides))"` → `15`） | 满足 | 15 为建议区间上限 |
| **PP-1** 题目与任务完成情况 | PPT 第 1~2 页（封面 + 六环节卡片；备注写明每个数字的桶归属） | 满足 | — |
| **PP-2** RepViT 核心结构 | 第 3 页左栏（M0.9 结构 + 单 Block 内部），备注明确标注为「合并页」 | 满足 | — |
| **PP-3** 为什么 RepViT 不是标准 ViT | 第 3 页右栏（标准 ViT 与 RepViT 逐项对应） | 满足 | — |
| **PP-4** 官方多型号评价 | 第 4 页（5 型号表 + 两套 crop_pct 口径 + 免责句；PDF 文本可读） | 满足 | 该页缺「推理后端」与「测试次数」标注，见 PP-17 |
| **PP-5** 数据集和训练流程 | 第 5 页（Pet 划分 + 泄漏检查 + 官方数据事实基线） | 满足 | 该页写「test 只评价一次 eval_count = 1」，只对 baseline 成立，见 §9 得分项 G-1 |
| **PP-6** Baseline 结果 | 第 6 页（test 三指标 + 骨干确实被训练 + 换头正误做法） | 满足 | — |
| **PP-7** 优化假设与控制变量 | 第 7 页（四方案卡片 + `diff_config`/`same_budget` 可执行校验 + 六项保持一致清单） | 满足 | — |
| **PP-8** 曲线和定量结果 | 第 8 页（五曲线同图 + test 结果 + 「两组都收敛、组合方案未提高测试准确率」） | 满足 | — |
| **PP-9** 混淆矩阵与失败案例 | 第 9 页（行归一化混淆矩阵 + 每类 F1 + 2 个失败案例 + 跨物种 3.56%） | 满足 | — |
| **PP-10** Grad-CAM 结果 | 第 10 页（2 正确 + 2 失败 + 挂载层说明 + 「关注到主体 ≠ 判对」） | 满足 | — |
| **PP-11** 结构重参数化 | 第 12 页（BN 折核 + 三分支相加 + must-eval 警告；元信息含输入/ batch / n / 硬件 / 后端 / 精度 / 次数 / seed） | 满足 | — |
| **PP-12** ONNX 多模型部署 | 第 13 页（三型号 n=12 一致性 + 四条边界声明，含「不与第 12 页并成一句结论」） | 满足 | — |
| **PP-13** 性能比较 | 第 14 页（6 型号延迟图 + 精度对照 + 口径不同警告；元信息含 型号/输入/batch/硬件/后端/精度/次数/线程） | 满足 | — |
| **PP-14** 遇到的问题 | 第 15 页（三个最有代表性的问题：键名交集 0 / `data/provided/` 从未定义 / 早停导致预算不等价） | 满足 | — |
| **PP-15** 总结 | 第 15 页（四条结论 + 后续计划） | 满足 | — |
| **PP-16**（附注）PPT 应重点展示本人完成的实验，不建议大篇幅复制论文内容 | 15 页里仅第 3 页右栏讲论文对照，其余为自跑实验与自产图；`report/PPT_CONTENT.md` 每页带「元信息 + 数据来源」行 | 满足 | — |
| **PP-17**（附注）所有性能图表必须注明：模型型号 / 输入尺寸 / batch size / 硬件 / 推理后端 / 精度类型 / 测试次数（7 项） | 第 8/9/10/12/13/14 页的元信息行**7 项齐全**（例：第 14 页「型号：M0.9/M1.0/M1.1/M1.5/M2.3/M0.9-Pet37 · 输入 1×3×224×224 · batch 1 · i7-13650HX · ORT 1.30.0 CPUExecutionProvider · FP32 · 预热 10 + 正式 50 次」）；**第 4 页**（官方多型号评价表，含参数量/MACs）只有「1000 张 · 224×224 · batch 64 · FP32 · RTX 4060 Laptop」，缺 **推理后端** 与 **测试次数** | 部分 | 第 4 页补 2 项元信息即可；修复见 §9 得分项 G-8 |

### 4.3 现场运行展示（9 项 + 评委可执行的 11 项操作）

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **LV-1** 在程序参数中切换不同 RepViT 型号 | `deploy/infer_onnx.py --model <key>`、`deploy/benchmark.py --model a b c`、`deploy/demo_camera.py`（按键 1/2/3 现场切换，session 缓存避免卡顿）；`deploy/model_registry.py` 打印 7 个可切换 key | 满足 | — |
| **LV-2** 两种官方预训练模型的推理 | `python deploy/infer_onnx.py --model repvit_m0_9_in1k --image <img>`；`onnx/repvit_m1_0_in1k.onnx` 同法 | 满足 | — |
| **LV-3** 自行训练模型的推理 | `python deploy/infer_onnx.py --model repvit_m0_9_pet37 --image external/beagle__*.JPEG` | 满足 | — |
| **LV-4** Baseline 与优化模型曲线 | `outputs/curves/opt_compare.png`（离线可打开）；现场也可 `python tools/plot_curves.py ...` 重画 | 满足 | — |
| **LV-5** 混淆矩阵或 Grad-CAM 结果 | `outputs/confusion_matrix/*.png`、`outputs/gradcam/*.png`（离线） | 满足 | — |
| **LV-6** ONNX Runtime 对现场指定图片的推理 | `deploy/infer_onnx.py --image <现场图片>`（支持任意路径参数） | 满足 | — |
| **LV-7** 实际 Execution Provider | `deploy/benchmark.py:57` 打印；`deploy/infer_onnx.py` 的 session 可打印 providers | 满足 | — |
| **LV-8** 单张图片 Top-5 结果 | `deploy/infer_onnx.py --topk 5` 默认输出 Top-5 名称与置信度 | 满足 | — |
| **LV-9** 三个 ONNX 模型的延迟和文件大小 | `outputs/benchmarks/summary.csv`（6 型号全字段）；现场可 `python deploy/benchmark.py --model ... --warmup 10 --runs 50 --threads 4` | 满足 | — |
| **LV-10~LV-20**（评委可执行）检查训练日志 / 运行评价程序 / 修改输入图片 / 切换型号 / 修改输出目录 / 查看类别映射 / 查看训练配置 / 检查 ONNX 输入输出 / 检查 EP / 查看重参数化代码 / 对比 PyTorch 与 ONNX 输出 | 分别对应：`outputs/logs/*`；`tools/evaluate.py`/`tools/eval_pretrained.py`；`--image` 参数；`--model` 参数；`deploy/benchmark.py --out-dir` 与 `deploy/export_onnx.py --out-dir`；`deploy/model_registry.py` + `labels/`；`configs/*.yaml` + `*_config_effective.yaml`；`deploy/benchmark.py:58-59` 打印 + `outputs/reparam/*_onnx_nodes.json`；`:57`；`tools/reparam_verify.py`/`tools/reparam_deep.py`；`deploy/compare_torch_onnx.py` | 满足 | 「修改输出目录」在 `benchmark.py`/`deploy/export_onnx.py` 都有 `--out-dir`；`deploy/infer_onnx.py` 的输出走控制台与可选 `--dump-probs` 落盘路径 |
| **LV-21** 候选人可以准备备用视频，但备用视频不能完全代替现场运行 | 未提供备用视频，也无「以视频代替现场」的表述 | 满足 | 备用视频本身是可选项，不提供不构成缺口 |

### 4.4 技术提问范围（59 条）

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **QA-1** 模型与论文 11 问（为什么是纯 CNN / 为什么从 ViT 视角 / Mixer 如何划分 / DW 与 PW 区别 / 为什么扩张比小 / 为什么加宽 / Stem 优势 / 第三阶段为何多 Block / SE 为何不每块都放 / 简单头如何降延迟 / 参数量-MACs-延迟为何不完全相关） | 逐条成文位置：`report/REPORT.md` §2.1（ViT 视角）、§2.2（纯 CNN）、`:69-76` 设计表（Mixer 划分、DW/PW、扩张比、宽度、Stem、更深下采样、SE、简单头）、§3（Stage 2 独占 14 个 Block）、§14.4（参数量/MACs 与延迟非线性）；`report/PPT_CONTENT.md` P03 备注；`report/LOGITS_AUDIT.md` 提供误差口径 | 满足 | — |
| **QA-2** 结构重参数化 8 问（训练态分支 / 三分支如何融合 / BN 如何吸收 / 为何结果应一致 / 为何必须 eval 验证 / BN 统计量错误的影响 / 是否减少理论参数量 / 对 ONNX 算子图的影响） | `report/REPORT.md` §12.1~§12.5 逐条；§12.4 明确的「重参数化减少的是推理态参数量：5,489,328 → 5,067,056，净减 422,272 = 蒸馏头 385,768 + BN 折进卷积 36,504」；`outputs/reparam/*_onnx_nodes.json` | 满足 | — |
| **QA-3** 数据与训练 10 问（为何用 ImageNet 预训练 / 为何选 Pet / 为何不能用 test 调参 / 分类头如何替换 / 哪些参数不能直接加载 / 冻结与全量微调区别 / 小数据集为何易过拟合 / Mixup-CutMix-LS 各解决什么 / 为何 loss 降而 Macro-F1 不升 / 如何选 best ckpt） | `report/REPORT.md` §7.1/§7.3（预训练权重、换头、参数加载）、§6.4（test 只评价一次、选模依据）、§8.1（过拟合风险）、§8.2（三种增强的假设）、`tools/train.py:623`（选模标准硬断言）、§9（为何指标差异是噪声） | 满足 | — |
| **QA-4** 曲线和可视化 9 问（曲线属哪组实验 / 横纵坐标 / 是否过拟合 / 为何震荡 / 学习率如何影响收敛 / 哪些类别易混淆 / Grad-CAM 为何关注背景 / 高置信度错误说明什么 / 两组主要差别） | `report/REPORT.md` §10.1~§10.5、§11.3~§11.6；`report/PPT_CONTENT.md` 每页的「元信息」行直接写清曲线归属与坐标含义 | 满足 | — |
| **QA-5** 代码与工程 11 问（数据如何读 / 增强在哪 / 类别编号怎么生成 / 模型如何创建 / 分类头如何替换 / loss 在哪 / optimizer 与 scheduler 在哪 / best ckpt 如何保存 / Grad-CAM 挂哪层 / 型号如何切换 / 哪些代码自写） | `datasets/pet_dataset.py`、`tools/train.py:180-215`（增强）、`datasets/build_pet_class_map.py`（类别编号）、`models/build_model.py`（建网）、`tools/train.py:619-646`（断言与 loss/optim）、`tools/gradcam.py:1-40`（挂载层）、`deploy/model_registry.py`（型号切换）、`PROVENANCE.md` §三（自撰范围） | 满足 | — |
| **QA-6** 部署与性能 10 问（PT 与 ONNX 为何不一致 / RGB 与 BGR 影响 / Resize 与 Crop 顺序 / 为何需要预热 / mean-P50-P95 含义 / 集显为何不一定更快 / 如何确认调用集显 / 为何更大模型不按 MACs 变慢 / FP16 为何可能不加速 / 多模型如何共用推理代码） | `report/REPORT.md` §13.3（九类问题口径）、§14.3（mean/P50/P95）、§14.4（非线性原因）、附录 E（集显实测与回退识别）；`deploy/infer_onnx.py:14,20`（RGB、顺序）注释；`deploy/benchmark.py:20-30`（预热注释）；`deploy/benchmark.py` 单一 `run_one` 复用于多模型 | 满足 | — |

---

## 5. 七、进阶任务（20 分）

### 5.1 进阶 1：RepViT 模型家族速度—精度分析（4 分，第 21 页）

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **A1-1** 在基础的两个官方型号之外，累计评价并部署**至少四种官方型号** | **评价**：`outputs/pretrained_eval/` 共 5 个官方型号（m0_9 / m1_0 / m1_1 / m1_5 / m2_3）。**部署**：`git ls-files onnx/` = **5 个**，其中**官方 ONNX 4 个**（`repvit_m0_9_in1k` / `repvit_m1_0_in1k` / `repvit_m1_1_in1k` / `repvit_m1_5_in1k`）+ 自训练 Pet-37 1 个（`repvit_m0_9_pet37.onnx`）；`onnx/repvit_m2_3_in1k.onnx` 仍被 `.gitignore` 排除（**唯一**被排除的型号），仅指标与延迟入库 | 满足 | **2026-09-17 判定复核后改判**：`m1_5` 已入库 → 「部署至少四种官方型号」的**部署**半句成立（4 个官方型号随仓库交付）；原记「推送后的仓库只有 3 个」已不适用（依据见下方后记） |
| **A1-2** 使用相同验证子集 | `outputs/benchmarks/family_summary.csv` 的 `subset_sha256 = d5532205d8f30099`（= ImageNetV2 matched-frequency 清单哈希）对 5 个型号一致 | 满足 | — |
| **A1-3** 使用相同预处理 | 同表 `transform = "crop_pct=0.95,bicubic,input=224 (timm resolve_data_config)"` 5 行一致 | 满足 | 与基础 2 型号评价用的 `crop_pct=0.875` 是两套口径，`report/REPORT.md:135-137` 已显式警告「不可混用」 |
| **A1-4** 使用相同设备和推理后端 | 同表 `runtime = pytorch`、`device = cpu`、`ep = CPUExecutionProvider`、`threads = 4` 5 行一致 | 满足 | — |
| **A1-5** 统计参数量、MACs、文件大小和延迟 | 同表 `params_M`（含 `param_caliber = infer_fused` 与说明列）、`macs_G`、`onnx_MB`、`lat_mean_ms`/`lat_p50_ms`/`lat_p95_ms`/`lat_max_ms` | 满足 | — |
| **A1-6** 绘制参数量—准确率关系图 | `outputs/advanced/params_vs_acc.png`（+ `outputs/advanced/macs_vs_acc.png`） | 满足 | — |
| **A1-7** 绘制延迟—准确率关系图 | `outputs/advanced/latency_vs_acc.png` | 满足 | — |
| **A1-8** 分析不同型号的帕累托关系 | `outputs/advanced/pareto_summary.json`、`outputs/advanced/marginal_returns.csv`、`outputs/advanced/model_recommendation.csv`；`report/REPORT.md:189-199`（边际收益表 + 帕累托叙述 + 数据来源与「不要混用」警告） | 满足 | — |
| **A1-9** 推荐一个适合当前部署设备的型号并说明理由 | `report/REPORT.md:192-195`（P50 ≤ 15 ms 预算下推荐 `repvit_m0_9_in1k`，得分 0.9842；放宽到 10 ms 且优先精度时 `repvit_m1_0_in1k` 为帕累托最优）；`outputs/advanced/model_recommendation.csv` | 满足 | — |
| **A1-10**（附注）不得用官方 iPhone 延迟代替本地实际测试 | 全部延迟来自 `outputs/benchmarks/*_benchmark.json`（本机 ORT CPU）；iPhone 12 延迟只作为对照列并标注不可比 | 满足 | — |

> **后记（2026-09-17 追加，任务 t30）**：本节 **A1-1** 的 ONNX 入库计数按现行 `git ls-files onnx/` 更正为 **5 个**（其中**官方 ONNX 4 个** + 自训练 Pet-37 1 个），原「官方 ONNX 只有 3 个、`m1_5` 被 `.gitignore` 排除」属 t3 之前的时点事实（当时 `m2_3` 与 `m1_5` 都被排除）。同一更正已同步到 §3.6 的 `5-M4` / `5.5.7`、§3.7 的 `T-8`、§7 的 `L-7` 与 §9 的 G-6 行。
>
> **判定复核（更正计数后重新确认，非默认保留）**：**`A1-1` 判定由「部分」改判为「满足」** —— 复核依据：① 现行 `git ls-files onnx/` = **5 个**，其中**官方 ONNX 4 个**（`repvit_m0_9_in1k` / `repvit_m1_0_in1k` / `repvit_m1_1_in1k` / `repvit_m1_5_in1k`），已满足「在基础的两个官方型号之外，累计评价并**部署**至少四种官方型号」；② 使该行此前只能判「部分」的唯一原因（推送后的仓库只有 3 个官方 ONNX）即 §9 得分项 **G-6**，该缺口已由 t3 解除 `.gitignore` 对 `repvit_m1_5_in1k.onnx` 的排除而**闭合**（唯一仍被排除的是 `m2_3`，而要求只到「四种」）；③ 四个官方 ONNX 均有同口径 n=12 一致性产物（`outputs/metrics/consistency_*.json` 共 5 份，见 §3.6 `5.5.7`）。故原「部分」不再是合适判定，**改判「满足」**，并同步 §8 统计（§5 区块：满足 35→36、部分 4→3；合计：满足 307→308、部分 23→22）与说明 3 的对应清单。

### 5.2 进阶 2：第二项优化及组合消融（5 分，第 21 页）

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **A2-1** Baseline | `outputs/logs/baseline_metrics.csv` + `outputs/metrics/baseline_test.json` | 满足 | — |
| **A2-2** 实验 A | `configs/opt_abl_a.yaml`（= RandAugment 加强增强，8 差异键）+ `outputs/logs/opt_abl_a_metrics.csv` + `outputs/metrics/opt_abl_a_test.json` | 满足 | — |
| **A2-3** 实验 B | `configs/opt_abl_b.yaml`（= 差异化 lr，1 差异键）+ 同名 metrics/test 产物 | 满足 | — |
| **A2-4** A+B 组合实验 | `configs/opt_abl_ab.yaml`（9 差异键）+ 同名产物 | 满足 | — |
| **A2-5** 单项控制变量结果 | `outputs/advanced/ablation_summary.json` 的 `table`（四臂 val Top-1 / Top-5 / Macro-F1 / best_epoch）；`outputs/metrics/ablation.csv` | 满足 | — |
| **A2-6** 组合效果 | 同文件 `interactions`（ΔA / ΔB / ΔAB / 交互项 I / verdict）；`report/REPORT.md` §9.1 表 | 满足 | — |
| **A2-7** 训练曲线 | 四臂的**逐 epoch 曲线数据**齐全（`outputs/logs/opt_abl_{a,b,ab}_metrics.csv`），但**没有把四臂画在同一张图上的产物**（`outputs/curves/` 只有 baseline/opt_combo 相关 4 张） | 部分 | 缺「Baseline/A/B/A+B 同图训练曲线」这一张图；修复见 §9 体验项 X-4 |
| **A2-8** 指标对比 | `outputs/advanced/ablation_summary.{csv,json}` + `report/REPORT.md` §9 八组表（含四臂 test 三指标） | 满足 | — |
| **A2-9** 方法之间是否存在叠加或冲突的分析 | `report/REPORT.md:359-368`（val Top-1 交互项 +0.541 判超加性 / val Macro-F1 +0.006 可加；机制解释「B 压低主干 lr、A 提高数据难度，方向相反」）；`ablation_summary.json` 的 `noise` 段给出判据「`\|delta\| > 2σ` 才算真实增益」与单种子 σ 估计 0.8 | 满足 | — |

### 5.3 进阶 3：深入结构重参数化分析（4 分，第 21~22 页）

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **A3-1** 输出转换前后的模型结构 | `outputs/reparam/repvit_m0_9_pet37_structure_before.txt`（51,595 B）与 `outputs/reparam/repvit_m0_9_pet37_structure_after.txt`（26,169 B）；`outputs/advanced/repvit_m0_9_pet37_structure_before.txt`（34,065 B）与 `outputs/advanced/repvit_m0_9_pet37_structure_after.txt`（14,187 B）同两份；`report/REPORT.md` §12.5 汇总表 | 满足 | — |
| **A3-2** 比较 ONNX 计算图节点或算子数量 | `outputs/reparam/repvit_m0_9_pet37_onnx_nodes.json`（before 480 nodes / BN 24 / Conv 126；after 387 / BN 0 / Conv 103；`delta` 段落）；`outputs/advanced/repvit_m0_9_pet37_onnx_nodes.json` | 满足 | 文件另附 `reference_magnitude`（官方实现同口径 682→464）并注明「硬判据是 BN=0 与 Conv −23」 |
| **A3-3** 比较转换前后的模型文件大小 | `outputs/reparam/repvit_m0_9_pet37_onnx_nodes.json` 的 `before.size_bytes = 19,028,154`（19.028 MB）vs `after.size_bytes = 18,893,159`（18.893 MB）；`outputs/advanced/repvit_m0_9_pet37_reparam_report.json` 的 `onnx.train.file_MB` / `onnx.infer.file_MB` | 满足 | — |
| **A3-4** 比较转换前后的推理延迟 | `outputs/advanced/repvit_m0_9_pet37_reparam_report.json` 的 `latency_ms.train`（mean 39.25 / P50 37.88 / P95 47.27，n=50）与 `latency_ms.infer`（mean 28.66 / P50 29.47 / P95 35.39，n=50）；`report/REPORT.md:136`（§4.3 表第 6 行「**PyTorch 推理**」）另有融合/未融合 PyTorch 延迟（**8.41 vs 17.15 ms，降 51.0%**；见下方「后记（2026-09-17 追加，任务 t22）」） | 满足 | 两组延迟口径不同（此处是 CPU 训练态 vs 推理态），引用时需写明 |
| **A3-5** 比较 logits 数值误差 | B4：`outputs/reparam/logits_diff.json` 的 `max_abs_err = 7.092952728271484e-06` / `mean_abs_err = 2.030726818702533e-06`（32 个固定随机输入） | 满足 | 与 B1（n=12 真实图片 6.199e-06）**分开报告**，见 §3.6 的两处「口径核对」（B1 / B4） |
| **A3-6** 分析转换后仍然存在的 BatchNorm 或冗余算子 | `outputs/reparam/repvit_m0_9_pet37_onnx_nodes.json` 的 `delta.BatchNormalization = -24`（归零）；`outputs/advanced/repvit_m0_9_pet37_reparam_report.json` 的 `residual` 段逐类给出 Add 53 / Mul 64 / Div 27 / Erf 27 / ReduceMean 11 的来源与「为什么不可再折叠」；`report/REPORT.md` §12.5 文字结论 | 满足 | — |
| **A3-7** 对至少一个 RepViT Block 给出融合过程说明 | `outputs/advanced/block_22_fusion_report.txt` + `outputs/advanced/repvit_m0_9_pet37_reparam_report.json` 的 `block_fusion_report`（block_index 22、三条分支、四步融合、`C=384`、`params 5760 → 3840`）；`report/REPORT.md` §12.6 以 `stages.0.blocks.0` 为例复述 | 满足 | — |
| **A3-8**（加分）能独立推导并验证卷积与 BatchNorm 融合公式 | `report/REPORT.md` §12.2 给出完整推导；`tools/reparam_deep.py:11-16` `fuse_conv_bn_manual()`、`:18-52` `fuse_repvggdw_manual()` 为**不调用任何官方 fuse 的独立实现**（注释明示「用于交叉验证」） | 满足 | 独立实现的**数值对齐**没有落盘产物，见 §6.4 拓展 4 |

> **后记（2026-09-17 追加，任务 t22）**：本行引用的 `report/REPORT.md:136`（§4.3 表第 6 行「**PyTorch 推理**」）那组融合/未融合延迟**已按现行产物更正为 8.41 vs 17.15 ms（降 51.0%）**（旧引用 7.97 / 15.51 / 48.6% 来自自建子集那一轮，全仓已无该组值）；`latency_ms.train`（mean 39.25 / P50 37.88 / P95 47.27）与 `latency_ms.infer`（mean 28.66 / P50 29.47 / P95 35.39）取自 `outputs/advanced/repvit_m0_9_pet37_reparam_report.json`，**复核后与现行产物一致、未变**。
> **判定复核（更正数字后重新确认，非默认保留）**：判定**仍为「满足」** —— 两套口径的延迟都仍在落盘产物里，且「转换后延迟下降」在现行数据上同样成立（推理态 28.66 < 训练态 39.25 ms；融合后 8.41 < 未融合 17.15 ms），口径差异已在缺口列写明。

### 5.4 进阶 4：集成显卡加速部署（4 分，第 22 页）

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **A4-1** 在 ONNX Runtime CPU 基础上完成任意一种集显后端（OpenVINO GPU/AUTO、DirectML、Windows ML 或其他） | `report/REPORT.md` 附录 E：`onnxruntime.get_available_providers() = ['AzureExecutionProvider','CPUExecutionProvider']`；WMI 显示适配器只有 NVIDIA RTX 4060 Laptop，无 Intel/AMD 集显；OpenVINO EP 与 DirectML provider 均不存在；`import openvino` → `ModuleNotFoundError`。产物 `outputs/benchmarks/igpu_{cpu,ort_ov,ov_native,dml}.json`，其中 `ort_ov`/`dml` 标记为 `invalid` | 缺失 | 本机不具备集显，**未完成**；仓库已如实记录并说明依据（不是漏做）。影响：进阶 4 的 4 分不可得 |
| **A4-2** 打印实际推理后端 | `tools/bench_igpu.py` 打印实际生效 EP，并在检测到回退时标 `invalid` | 满足 | 该工具本身（即使本机跑不出集显）满足「打印」要求 |
| **A4-3** 记录集显型号和驱动版本 | 无集显可记录 | 缺失 | 与 A4-1 同因 |
| **A4-4** 确认是否存在算子回退 CPU | `tools/bench_igpu.py` 的回退检测 + `igpu_*.json` 的 `invalid` 标记 | 满足 | 工具层满足；本机实测即回退 CPU |
| **A4-5** 比较 CPU 和集显平均延迟、P50 及 P95 | 只有 CPU 侧：`outputs/benchmarks/igpu_cpu_baseline.json` | 缺失 | 无集显基线可比 |
| **A4-6** 至少比较两种 RepViT 型号 | 未进行集显侧比较 | 缺失 | 与 A4-1 同因 |
| **A4-7** 分析较小模型是否能够充分利用集显 | 未进行 | 缺失 | 与 A4-1 同因 |
| **A4-8**（附注）不得只根据任务管理器 GPU 占用判断部署成功 | 仓库的做法正是反例：脚本读 `sess.get_providers()` 并把回退行标 `invalid`，`report/REPORT.md` 附录 E 明说「而不是给一个看起来更快的假数字」 | 满足 | — |

### 5.5 进阶 5：鲁棒性、可解释性或实时演示（最高 3 分，第 22~23 页）

原文是「从以下方向选择完成」，每个方向有各自的子要求。

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **A5-1**（方向一）鲁棒性测试：测试至少三种变化（模糊/亮度/压缩/遮挡/旋转/噪声/背景替换） | `outputs/advanced/robustness_repvit_m0_9_pet37.json` 的 `perturbations` = 7 类（gaussian_blur / motion_blur / brightness / jpeg / occlusion / rotation / gaussian_noise），每类 6 级 severity；`report/REPORT.md` 附录 C | 满足 | 7 类 ≥ 3 类 |
| **A5-2**（方向一）分析准确率、置信度和 Grad-CAM 变化 | 准确率：`rows[].top1`（噪声 σ 0→0.12 时 94.0%→78.0%）；置信度：`mean_conf` 0.781→0.643、`ece` 0.167→0.178；**Grad-CAM 变化**：`git grep -F -- "gradcam" -- outputs/advanced/robustness_repvit_m0_9_pet37.json` → **0 命中**（该 JSON 无 `gradcam` 字段），也无相关图 | 部分 | 缺「扰动下的 Grad-CAM 变化」分析；因方向可自选，不影响 A5 整体完成度；修复见 §9 体验项 X-5 |
| **A5-3**（方向二）深入可解释性：不同阶段特征图可视化 / Grad-CAM 层选择对比 / t-SNE 或 UMAP / 错误类别特征分布 / 置信度校准（完成一种或多种） | `outputs/advanced/interp/`：`outputs/advanced/interp/featmap_stages_0.png`～`outputs/advanced/interp/featmap_stages_3.png` + `outputs/advanced/interp/featmap_stem.png`（5 个挂载点）、`outputs/advanced/interp/multilayer_cam.png`（同图 5 层）、`outputs/advanced/interp/tsne.png`（37 类 GAP 特征空间）、`outputs/advanced/interp/calibration.json`（温度 T = 0.6149、ECE 前后）；`report/REPORT.md` 附录 D；另有 `outputs/gradcam/gradcam_layer_compare_baseline.png`（层选择对比） | 满足 | 「错误类别特征分布分析」未单独出图，但方向要求为「一种或多种」，已完成 4 类 |
| **A5-4**（方向三）摄像头或视频分类：支持摄像头/视频输入 | `deploy/demo_camera.py`（`--source 0` 或视频路径） | 满足 | 代码就绪 |
| **A5-5**（方向三）实时显示 Top-5 结果 | `deploy/demo_camera.py` 的 `--k`（默认 5）+ 画面叠字 | 满足 | — |
| **A5-6**（方向三）显示平均 FPS | `deploy/demo_camera.py` 的 `FPS(avg30)`，注释说明计时点在 `waitKey` 之后（覆盖解码+预处理+推理+绘制四段） | 满足 | — |
| **A5-7**（方向三）支持切换模型型号 | `deploy/demo_camera.py` 按键 1/2/3 + session 缓存 | 满足 | — |
| **A5-8**（方向三）分析画面抖动导致的类别和置信度变化 | 代码有 `--jitter-report`（默认输出路径 outputs/benchmarks/realtime_jitter.json）+ `pred_hist` / `conf_hist`，但**该产物不存在**（`Test-Path` = False），仓库无实跑证据 | 部分 | 代码就绪、无实跑产物；因方向可自选且已选方向二，不影响 A5 完成度；修复见 §9 体验项 X-6 |
| **A5-9**（附注）不要求达到固定实时帧率 | 未出现任何实时帧率承诺 | 满足 | — |

---

## 6. 八、拓展任务（10 分，可选）

> 拓展任务为**可选项**（「满分 10 分，可选择一项或多项」）。`report/REPORT.md:21` 的复现范围明确写「不做…拓展任务」。以下逐条判定「现状是否有该交付物」，并注明**不完成不构成必须修复的缺口**。

### 6.1 拓展 1：模型压缩（最高 4 分，第 23~24 页）

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **E1-1** 完成任意一项：INT8 量化 / 知识蒸馏 / 结构化剪枝 / 通道裁剪 / 分类头轻量化 / 其他合理压缩方法 | 全仓无量化/剪枝/蒸馏脚本：`git grep -lE "quant|prune|distill" -- ':!report/REQUIREMENTS_AUDIT.md'` → **107 个文件**（全部是注释、文档、计划说明或权重二进制的无关命中；`tools/same_budget.py` 并不在这些命中里），仓库内没有任何量化 / 剪枝 / 蒸馏**实现**；`report/REPORT.md:773-775` 把 INT8/蒸馏/剪枝列入「后续计划」 | 缺失 | 可选任务未开展，属**团队已决议的范围外**；不构成必须修复项 |
| **E1-2** 必须比较：参数量 / MACs / 模型文件大小 / 平均延迟 / P50 和 P95 / Top-1 和 Macro-F1 / 各类别性能变化 | 无压缩前后对比 | 缺失 | 同 E1-1 |
| **E1-3**（附注）只生成压缩文件但没有实际精度和速度验证，不计完整得分 | 未生成压缩文件 | 缺失 | 同 E1-1 |

### 6.2 拓展 2：多随机种子实验（最高 2 分，第 24 页）

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **E2-1** 使用至少 3 个随机种子重复 Baseline 或关键优化实验 | 全部实验 `seed = 42`；`outputs/advanced/ablation_summary.json` 的 `noise` 段用「单种子 σ 估计 = 0.8」代替多种子，`report/REPORT.md:350-353` 用重跑一次（±0.08 / ±0.40）说明波动量级 | 缺失 | 未做 3 种子重复；仓库用重跑与二项置信区间做了**替代性**波动论证（诚实标注为「替代」） |
| **E2-2** 保持其他配置一致 | 不适用（无多种子产物） | 缺失 | 同 E2-1 |
| **E2-3** 报告均值和标准差 | 只有单种子结果，无跨种子 mean/std | 缺失 | 同 E2-1 |
| **E2-4** 分析提升是否超过随机波动 / 说明结论是否稳定 | `ablation_summary.json` 的 `noise.rule`（`\|delta\| > 2σ` 判真实增益）+ `report/REPORT.md` §9 的三条判据 | 部分 | 方法学到位但缺真实多种子证据；不构成必须修复项 |

### 6.3 拓展 3：高性能、边缘或下游应用部署（最高 2 分，第 24~25 页）

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **E3-1** 完成任意一种：TensorRT / OpenVINO IR / Core ML / Windows ML / Jetson 等边缘设备 / 手机端 / Docker 化推理服务 / Web API 或桌面端 / RepViT-SAM / 语义分割或检测下游 | 全仓无上述任一实现（无 Dockerfile、无 Web API、无 TRT/OpenVINO/CoreML 产物、无下游任务脚本）；`report/REPORT.md:773-775` 列后续计划 | 缺失 | 可选任务未开展；集显后端属进阶 4（同样因设备缺位未完成） |
| **E3-2** 需要提供实际运行证据和性能报告 | 无 | 缺失 | 同 E3-1 |
| **E3-3**（附注）下游任务不能只运行官方演示，必须包含候选人完成的数据处理、部署改造或实验分析 | 不适用 | 缺失 | 同 E3-1 |

### 6.4 拓展 4：核心模块独立实现（最高 2 分，第 25 页）

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **E4-1** 不直接复制完整官方模块，独立实现以下一个或多个：RepViT Block / RepVGGDW 结构 / 卷积与 BatchNorm 融合 / 3×3、1×1 和 Identity 分支融合 / Channel Mixer / RepViT 推理态转换程序 | `tools/reparam_deep.py:11-16` **独立实现** Conv+BN 融合（不调用官方 fuse）；`:18-52` **独立实现** RepVGGDW 三分支融合（3×3 + 1×1 pad + identity pad → 单个带 bias 的 3×3 depthwise `Conv2d`），并产出 `outputs/advanced/block_22_fusion_report.txt` 的融合细节（C=384、`t_range`、params 5760→3840） | 部分 | 代码级独立实现存在（覆盖「卷积与 BN 融合」「三分支融合」两项），但未按拓展 4 口径组织交付 |
| **E4-2** 提供输入输出尺寸测试 | 无独立实现的 IO 尺寸测试产物（`block_fusion_report` 只给了权重形状，无前向输入输出尺寸） | 缺失 | 与 E4-1 同源 |
| **E4-3** 与参考实现进行数值对齐 | 无「手写融合 vs timm `fuse()`」的数值对齐产物；`tools/reparam_deep.py:100` 用的是 `infer_m.fuse()`（timm 官方入口）产生对照模型，手写实现只用于生成结构细节 | 缺失 | 与 E4-1 同源 |
| **E4-4** 报告最大绝对误差 | `outputs/advanced/repvit_m0_9_pet37_reparam_report.json` 的 `numerics.max_abs_err = 2.9802322387695312e-06` 是 **B7 桶**（seed=0、仅 2 个随机输入的历史结构探针），`report/LOGITS_AUDIT_FINDINGS.md` §1.3 明确「产物存在但前提无效，不作为当前结论」 | 缺失 | 该数值**不得**作为拓展 4 的对齐误差引用；修复方案（若决定争取这 2 分）见 §9 可选增强 O-1 |
| **E4-5** 能够解释关键代码和数学原理 | `report/REPORT.md` §12.2 的公式推导 + `tools/reparam_deep.py` 的注释（`z = (W t)x + (β − μt + bt)`、`t = γ/√(σ²+ε)`） | 满足 | — |

---

## 7. 九、评分与评级 → 4. 必要限制（第 26~27 页）

> PDF 原文 bullet 数 **16**（契约写 17，差异见文首『计数口径声明』表：若把首句「出现以下情况时，即使完成进阶或拓展任务，也进行分数限制」单列成行，则为 17）。判定列的含义是「**触发该限制的风险**」：`满足` = 已排除/不触发；`部分` = 有触发风险或静态不可判定；`缺失` = 已触发。

| 要求原文摘录 | 仓库证据（相对路径或可直接运行的命令） | 判定 | 缺口说明 |
|---|---|---|---|
| **L-1** 只运行官方权重，没有自行训练 → 总分不超过 35 分 | `checkpoints/baseline_best.pt` + 8 组训练产物 + `outputs/logs/train_*` | 满足 | 风险不存在 |
| **L-2** 没有完成至少两种官方型号评价 → 官方模型部分最高不超过 50% | `outputs/pretrained_eval/` 5 个型号各带 `outputs/pretrained_eval/*/metrics.json`（Top-1/Top-5）与 `outputs/pretrained_eval/*/top5_samples.json` | 满足 | 风险不存在 |
| **L-3** 没有自行训练 Baseline → 总分不超过 45 分 | `outputs/metrics/baseline_test.json`（`top1 = 0.9234`、`num_samples = 3669`、`eval_count = 1`） | 满足 | 风险不存在 |
| **L-4** 只训练分类头且没有更新任何骨干阶段 → 基础训练部分最高不超过 60% | `outputs/metrics/backbone_updated.json`：`changed_tensors = 706`，其中骨干 699；`tools/selfcheck.py` 的 `train.backbone` PASS | 满足 | 风险不存在 |
| **L-5** 没有控制变量优化实验 → 总分不超过 60 分 | `configs/opt_*.yaml` 7 份 + `tools/diff_config.py` 叶子级差异校验（`opt.diff` PASS）+ `tools/same_budget.py`（`total_iters_equal = True`） | 满足 | 风险不存在 |
| **L-6** 没有完成自行训练模型的 ONNX 部署 → 总分不超过 65 分 | `onnx/repvit_m0_9_pet37.onnx`（入库）+ `outputs/benchmarks/repvit_m0_9_pet37_benchmark.json` + `deploy/infer_onnx.py --model repvit_m0_9_pet37` | 满足 | 风险不存在 |
| **L-7** 没有完成多模型 ONNX 部署 → 部署部分最高不超过 60% | `git ls-files onnx/` **5 个**模型（其中**官方 ONNX 4 个**）；`outputs/benchmarks/summary.csv` 6 型号 | 满足 | 风险不存在 |
| **L-8** 没有验证结构重参数化前后结果 → 部署部分最高不超过 70% | B4：`outputs/reparam/repvit_m0_9_pet37_reparam_report.json` + `outputs/reparam/logits_diff.json`（32 个固定随机输入，`max\|Δ\| = 7.093e-06`，Top-1 32/32，BN 107→0）；`tools/selfcheck.py` 的 `rep.verify`、`onnx.bn` PASS | 满足 | 风险不存在 |
| **L-9** 没有提交报告或没有参加答辩 → 本次考核不通过 | `report.pdf`（26 页）+ `report/REPORT.md` + `report/REPORT.docx` + PPT 源/PDF 均入库 | 部分 | 「报告已提交」成立；「是否参加答辩」是现场环节，静态不可判定 |
| **L-10** 无法复现实验结果 → 总分不超过 50 分 | `requirements.lock.txt`（142 行冻结）+ `bash tools/run_all.sh` 全流程 + `python tools/selfcheck.py`（**34 PASS / 0 FAIL**）+ 每个 JSON 带 `command` 元字段 + README 第 13 节「一键复现与耗时」 | 满足 | **但** `tools/check_report_assets.py` 现在跑出 36 条 FAIL（§9 阻断项 A-1），现场运行会直接冲击本条 |
| **L-11** 无法解释本人修改的代码 → 相关部分最高得分不超过 50% | `PROVENANCE.md`（5 类标注 + 10 行「AI 初稿被实测推翻」表）；`report/REPORT.md` 附录 A（10 类缺陷与处置）；`PROGRESS.md` 决策记录 | 满足 | 「能否解释」是答辩现场环节，静态不可判定；材料齐备 |
| **L-12** 无法说明曲线来源 → 曲线分析部分最高得分不超过 50% | `report/REPORT.md` §10.1 明写曲线文件与数据源；`report/PPT_CONTENT.md` 每页「来源」行；`outputs/curves/*` 均可用 `python tools/plot_curves.py` 从 `outputs/logs/*_metrics.csv` 重画 | 满足 | — |
| **L-13** 无法说明实验变量差异 → 优化实验部分最高得分不超过 50% | `tools/diff_config.py`（叶子级 + `_expected_diff_` 声明白名单）+ `configs/expected_diff.yaml` + `outputs/logs/*_config_effective.*` | 满足 | — |
| **L-14** 现场结果与报告明显不一致且无法解释 → 相关结果不计分 | 静态不可判定；可对照的静态证据：`report/REPORT.md` 的每个数字都带落盘路径，现场命令与该路径一致（README §4~§12 的命令即复跑命令） | 部分 | 现场环节，静态不可判定 |
| **L-15** 使用 test 进行调参或模型选择 → 测试集结果不计 | 选模标准被硬断言锁定为 `val_macro_f1`（`tools/train.py:623`）；baseline `eval_count = 1`。**但** `opt_combo` / `opt_mix` / `opt_randaug` / `opt_abl_b` 四个臂的 `eval_count = 2`（`outputs/metrics/*_test.json` 与 `outputs/logs/*_test_eval_count.json`），仓库文档中**没有**对这 4 次「第二次 test 评价」的书面归因；`tools/train.py:755-756` 只在运行时打印警告 | 部分 | 存在被评委质疑的风险（尤其 `report/REPORT.md:248` 的绝对表述「`test` 集只被评价过一次」只对 baseline 成立，PPT 第 5 页同样只写「eval_count = 1」）。修复见 §9 得分项 G-1 |
| **L-16** 数据泄漏、指标造假或实验来源不明 → 本次考核不通过 | 泄漏：`outputs/metrics/leakage_check.json`（文件名交集 0/0/0、MD5 交集 0/0/0、`pass: true`）；来源：`outputs/metrics/weight_sha256.json`（5/5 官方 SHA256）、Pet 归档官方 MD5、`external/images_manifest.csv` 逐张登记 + 「不声称原创」声明；口径：`report/LOGITS_AUDIT_FINDINGS.md` 把 8 个无配套产物的孤值明确列为「不得作为结论」 | 满足 | 风险不存在；孤值处理方式反而是加分项 |
| **L-17**（首句，若单列）出现以下情况时，即使完成进阶或拓展任务，也进行分数限制 | 本节 16 条限制已逐条对照（上文）；`report/REPORT.md` 与 PPT 均未把进阶/拓展完成度当作分数主张 | 满足 | 本行是为对齐契约「17 条」而单列的标题句 |

**建议评级对照（`九-3` 图片表格，OCR 来源为**工作区（仓库外）**的 `_build/exam_questions.md`）**：90~100 = S；80~89 = A；70~79 = B+；60~69 = B；50~59 = C；50 以下不通过。本审计不给出分数预测（评分是考核方行为），只给出「基础 70 分的可验收子项已基本闭合、进阶 1/2/3/5 有实质产物、进阶 4 与拓展 1~3 未开展」的事实结论。

---

## 8. 判定汇总

下表由脚本按本文件表格行自动核数（判定取每行第 3 列），可与正文逐行对齐。

| 区块 | 行数 | 满足 | 部分 | 缺失 | 挂起 |
|---|---|---|---|---|---|
| §2 §1.3（5 条挂起 + 8 条其余） | 13 | 8 | 0 | 0 | 5 |
| §3.0 前置门槛（关键环节 5 + 10 项元信息 + 1 条不可横比） | 16 | 15 | 1 | 0 | 0 |
| §3.1 基础 1.1（10 条 + 1 条附注） | 11 | 8 | 3 | 0 | 0 |
| §3.2 基础 1.2（14 说明 + 1 附注 + 9 图要素） | 24 | 24 | 0 | 0 | 0 |
| §3.3 基础 2（10 条 + 6 说明 + 4 硬约束） | 20 | 20 | 0 | 0 | 0 |
| §3.4 基础 3（10 条 + 6 控制变量 + 2 附注） | 18 | 18 | 0 | 0 | 0 |
| §3.5 基础 4（曲线 6 + 结果 10 + 附注 1 + 分析 10） | 27 | 27 | 0 | 0 | 0 |
| §3.6 基础 5（4 模型 + 14 基础 + 一致性 7 + 重参数化 7 + 性能 10） | 42 | 40 | 2 | 0 | 0 |
| §3.7 基础 6（结构 11 + README 13 + 提交 15 + 标注 6） | 45 | 45 | 0 | 0 | 0 |
| §4 六、报告 / PPT / 现场 / 提问（25 + 18 + 11 + 6） | 60 | 52 | 8 | 0 | 0 |
| §5 进阶 1~5（10 + 9 + 8 + 8 + 9） | 44 | 36 | 3 | 5 | 0 |
| §6 拓展 1~4（3 + 4 + 3 + 5） | 15 | 1 | 2 | 12 | 0 |
| §7 必要限制（16 条 + 单列的首句） | 17 | 14 | 3 | 0 | 0 |
| **合计** | **352** | **308** | **22** | **17** | **5** |

> 说明 1：行数含「附注 / 计分说明」行（它们同样需要可验收）。`挂起` 的 5 行按用户决议不做任何改动建议。所有「满足」判定都指向仓库内真实存在的文件或可直接运行的命令（证据见每行第 2 列与 §10）。**（后记 2026-09-17：本表是 t2 审计当时的统计快照；5 条挂起项的当前状态见 §2.1.1，其中 1.3-S4 已可判 `满足`。）**
> 说明 2：**17 行 `缺失` 全部是可选/设备受限项** —— 12 行属可选拓展任务（团队已在 `report/REPORT.md:21` 声明不做），5 行属进阶 4 集显部署（本机无集显，已如实记录并给出依据）。**基础 70 分区没有任何 `缺失`**。
> 说明 3：**22 行** `部分` 与 §9 清单的对应关系（2026-09-17，任务 t30：原 23 行中的 `A1-1` 已改判 `满足`，故减 1）：现场环节静态不可判定 3 行（`0-5`/`L-9`/`L-14`）、§1.3 耦合 1 行（`1.1-2`，按挂起处理）、七类来源表 6 行（`RP-17`~`RP-22` → **G-5**）、报告页数 1 行（`RP-0` → **G-7**）、PPT 元信息 1 行（`PP-17` → **G-8**）、案例成文 2 行（`1.1-7`/`1.1-8` → **G-3**）、一致性覆盖 1 行（`5.5.7` → **G-4**）、性能 IO 节点 1 行（`5.7.8` → **G-2**）、test 评价次数 1 行（`L-15` → **G-1**）、进阶 2 曲线 1 行（`A2-7` → **X-4**）、进阶 5 缺项 2 行（`A5-2`/`A5-8` → **X-5**/**X-6**）、可选拓展 2 行（`E2-4`/`E4-1` → 可选增强 **O-2**/**O-1**）。

---

## 9. 需修复的非 §1.3 缺口清单（按 阻断性 > 得分项 > 体验项 排序）

> **状态更新（2026-09-17，仅此一行）**：本节是 t2 审计当时的**缺口快照**。此后 t11/t12/t16/t17/t4/t9/t14 等任务已修复其中多项，实测例：阻断项 **A-1 的 `tools/check_report_assets.py` 现为 `OK=177 / WARN=0 / FAIL=0`**（原 `FAIL=36`）；得分项 **G-4 已闭合**（5 个入库 ONNX 均有同口径 n=12 一致性产物）。**复核时以各任务的实测证据为准**；逐条关闭与残留清理由最终审计 / 复核任务负责（C 段不动点刷新已转 t20）。

> 每条给出：**现象 / 影响哪一条评分限制或验收项 / 最小修复方案 / 需改动的仓库相对路径**。所有 §1.3 挂起项**不在本清单内**。

### 9.1 阻断性（必须先修）

| ID | 现象与证据 | 影响 | 最小修复方案（可直接执行） | 需改动的仓库相对路径 |
|---|---|---|---|---|
| **A-1** | 仓库自带验收脚本 `tools/check_report_assets.py` 实测 `汇总: OK=22 WARN=0 FAIL=36`（复现：`python tools/check_report_assets.py`）。三类失效：① 期望清单 `tools/report_spec.py` 指向**不存在的 `figures/` 目录**（14 条报告 + 13 条 PPT 配图）；② 期望的是已淘汰的 `B0_baseline_metrics.csv` / `O1_randaug_config_effective.yaml` / `training_curves_all.png` / `O1_randaug_test_norm.png` 等旧命名，实际产物是 `baseline_*` / `opt_*`；③ `tools/report_spec.py` 的 `CALIBER_ROWS` 指向 `outputs/metrics/{B0_baseline_test,O1_randaug_test,pytorch_latency}.json` 三个不存在的文件；④ `outputs/metrics/bench.jsonl` 缺 `backend` 与 `onnx_input/onnx_output` 字段；⑤ registry 里 `repvit_m0_9_pet37_opt` 已登记但 ONNX 未导出（`python deploy/model_registry.py` → `[MISS]`） | 现场若被评委直接运行，会与必要限制 L-10「无法复现实验结果 → 总分不超过 50 分」正面冲突；也是全仓唯一「跑了就红」的自检脚本 | ① 把 `tools/report_spec.py` 的 `REPORT_SECTIONS` / `PPT_SLIDES` / `FIGURE_SPEC` / `CALIBER_ROWS` 中的路径改为现行产物（`figures/*.png` → `outputs/architecture/*.png`、`outputs/curves/*.png`、`outputs/confusion_matrix/*.png`、`outputs/gradcam/*.png`、`outputs/advanced/*.png`；`B0_*`/`O1_*` → `baseline_*`/`opt_*`）；② 把 `BENCH_META_KEYS` 的 `backend` 映射到现有字段 `provider_actual`、`io_nodes` 映射到 `onnx_input`/`onnx_output`；③ 决定 `repvit_m0_9_pet37_opt` 是补导出（`python deploy/export_onnx.py --model repvit_m0_9_pet37_opt`）还是从 registry 摘除；④ 修完复跑并记录 `OK=N / FAIL=0`。**若决定弃用该脚本**，则同步把它从 README/报告中可能的引用处删除，并在文件头写明 `DEPRECATED`，**不能留一个会红的验收入口** | `tools/report_spec.py`、`tools/check_report_assets.py`、（可选）`deploy/export_onnx.py` + `deploy/model_registry.py` |

### 9.2 得分项（影响具体评分点）

| ID | 现象与证据 | 影响 | 最小修复方案（可直接执行） | 需改动的仓库相对路径 |
|---|---|---|---|---|
| **G-1** | 4 个优化臂在 test 上各被评价了 **2 次**（比 baseline 多一次），且文档无归因：`outputs/metrics/{opt_combo,opt_mix,opt_randaug,opt_abl_b}_test.json` 的 `eval_count = 2`（baseline 与 `opt_abl_a`/`opt_disc`/`opt_abl_ab` = 1）；`outputs/logs/opt_*_test_eval_count.json` 的 `count = 2`；`tools/train.py:747-756` 的计数器语义是「跨重跑累计」，且只在运行时打印 `[test][警告]`。而 `report/REPORT.md:248` 写「**`test` 集只被评价过一次**」、PPT 第 5 页写「test 只评价一次 / eval_count = 1」，两处都**未限定为 baseline** | 必要限制 L-15「使用 test 进行调参或模型选择 → 测试集结果不计」的答辩风险；也牵动十、公平性说明第 12 条「报告、代码、日志、模型和现场结果必须能够相互对应」 | ① 在报告 §6.4 把绝对表述改为按臂列表（给出 8 臂 `eval_count` 一览）；② 补一段归因：这 4 次多出的评价来自「早停预算修正后重训」等重跑，模型选择始终只用 `val_macro_f1`（`tools/train.py:623` 硬断言），两次结果均落盘可查；③ 在 §9 八组表加一列 `eval_count`；④ 同步 README §5/§7 与 PPT 第 5/8 页的措辞；⑤ 重导 `report.pdf` 与 `report/REPORT.docx` | `report/REPORT.md`、`README.md`、`report/PPT_CONTENT.md`、`report/答辩PPT_RepViT.pptx`(=`report/ppt_svg/05_data.svg`)、`report.pdf`、`report/REPORT.docx` |
| **G-2** | 性能测试第 8 条「记录模型输入输出节点信息」只打印不落盘：`deploy/benchmark.py:58-59` 打印 `input:` / `output:`，但 `rep` 字典无 `onnx_input`/`onnx_output` 字段 → `outputs/benchmarks/*_benchmark.json`、`outputs/metrics/bench.jsonl`、`outputs/report_assets/table_benchmark_meta.{csv,md}`（`io_nodes` 列 = `in=NoneNone out=NoneNone`）均无信息；`tools/check_report_assets.py` 报 `[FAIL] 性能测试元信息[io_nodes]: 缺失` | 五-5 性能测试 8 条中的第 8 条（硬性 8 项之一） | ① `deploy/benchmark.py` 的 `rep` 增加 `onnx_input=dict(name=inp.name,shape=list(inp.shape),type=inp.type)`、`onnx_output=dict(...)`、`backend="onnxruntime"`；② 用**只读**方式为现有 6 个 ONNX 补一份节点信息产物（`onnx.load()` 读 `graph.input/output` → `outputs/metrics/onnx_io_nodes.json`），避免为补字段重跑 6 组基准、也就不会改动已发布的延迟数字；③ 把该 JSON 并入 `table_benchmark_meta` 的 `io_nodes` 列 | `deploy/benchmark.py`、`tools/make_report_assets.py`、`outputs/metrics/onnx_io_nodes.json`（新增）、`outputs/report_assets/table_benchmark_meta.{csv,md}` |
| **G-3** | 基础 1.1 第 7/8 条要求「分析至少 2 个正确案例 / 2 个错误案例」：图片证据齐备（每型号 `outputs/pretrained_eval/<model>/cases/{correct_1,correct_2,wrong_1,wrong_2}.png`），但**文字分析**只有 `report/REPORT.md:164-165` 的 1 个正确案例（beagle 96.8%）与 1 句泛化的错误现象描述（细长体型小型犬混判），未逐例展开 | 五-1（15 分）的案例分项；评委现场可要求逐例说明 | 在 `report/REPORT.md` §4.5 把案例写成 2+2 四条（每条：图片文件名 + 真值 + 预测 + Top-5 置信度 + 一句归因），数字全部取自 `outputs/pretrained_eval/repvit_m0_9/top5_samples.json`（该文件已有 6 条样本、其中 `correct=true` 4 条）；重导 PDF/DOCX | `report/REPORT.md`、`report.pdf`、`report/REPORT.docx` |
| **G-4** | **（t2 时点现象）**「对**每个** ONNX 模型至少比较 [5 项一致性]」：入库 ONNX 4 个，同口径一致性产物只有 3 个（`outputs/metrics/consistency_{repvit_m0_9_pet37,repvit_m0_9_in1k,repvit_m1_0_in1k}.json`），`repvit_m1_1_in1k` 无。**（后记 2026-09-17，任务 t30：该缺口已闭合 —— `outputs/metrics/consistency_*.json` 现为 5 份，4 个入库官方型号全部有同口径 n=12 产物；见 §3.6 `5.5.7` 与本节顶部状态更新。）** | 五-5 一致性 5 项（部署部分）；严格读法下的覆盖不全 | **（t2 时点的修复方案，保留为历史记录、不需再执行）** 当时计划跑一次 `python deploy/compare_torch_onnx.py --model repvit_m1_1_in1k --images datasets/lists/imagenetv2_mf_1000.txt --limit 12 --out outputs/verification/consistency_repvit_m1_1_in1k_n12.json`（**已按现行唯一口径更正**：`--images` 原写的是已 `git rm` 删除且不得恢复的 `datasets/lists/imagenet_val_subset.txt`；现行命令与 `report/LOGITS_AUDIT.md:77`、`tools/audit_logrefs` 同款 —— 见 `tools/audit_logits_references.py:97` 的 `cmd=`），把结果并入 `report/REPORT.md` §13.3 的表与 `README.md` §11.1 的型号清单。**该缺口已闭合（见本行后记与 §3.6 `5.5.7`），本条不再需要执行。** | `outputs/metrics/consistency_repvit_m1_1_in1k.json`（新增）、`outputs/verification/consistency_repvit_m1_1_in1k_n12.json`（新增）、`report/REPORT.md`、`README.md` |
| **G-5** | 报告必须区分 7 类来源：正文把「论文公布」与「官方仓库公布」合并为一列（`report/REPORT.md:141` 表头「论文 / 官方公布」）；自动素材表 `outputs/report_assets/table_results.md` 虽有 **7 行数据**，但「官方权重实测 / 自训练 Baseline / 自优化模型 / PyTorch 推理」4 行的数值全为 `—`（第 5~8 行） | 六-1「报告必须区分 7 类来源」（第 16 页硬性要求） | ① 把 §4.3 表头拆成两行/两列（论文公布 78.7% 来自 `report/sources/literature.yaml#paper`；官方仓库公布 78.7%/79.1% 来自 `#official_repo`）；② 把 `outputs/report_assets/table_results.md` 的 4 行按 `outputs/pretrained_eval/*/metrics.json`、`outputs/metrics/baseline_test.json`、`outputs/metrics/opt_combo_test.json`、`outputs/pretrained_eval/*/latency.json` 填满（生成脚本 `tools/make_report_assets.py` 读 `tools/report_spec.py` 的 `CALIBER_ROWS`，注意与阻断项 A-1 一并修） | `report/REPORT.md`、`tools/report_spec.py`、`tools/make_report_assets.py`、`outputs/report_assets/table_results.{csv,md}` |
| **G-6** | 进阶 1「累计评价并**部署**至少四种官方型号」：**（t2 时点现象）**推送后的仓库当时只有 3 个官方 ONNX（`git ls-files onnx/` = m0_9_in1k / m1_0_in1k / m1_1_in1k / m0_9_pet37），`onnx/repvit_m1_5_in1k.onnx`（56 MB）与 `onnx/repvit_m2_3_in1k.onnx`（92 MB）当时都被 `.gitignore` 排除。**（后记 2026-09-17，任务 t30：本缺口已闭合 —— `repvit_m1_5_in1k.onnx` 已入库，现行 `git ls-files onnx/` = **5 个**、其中**官方 ONNX 4 个**，仅 `m2_3` 仍被排除；本行保留为 t2 时点记录。）** | **（t2 时点影响）** 七-1（4 分）的「部署」半句；当时严格读法下只能算 3 个官方型号已部署 | **（t2 时点的修复方案，保留为历史记录、不需再执行）** 三选一：① 解除 `.gitignore` 对这两个 ONNX 的排除并入库（注意仓库体积 +148 MB，PDF 第 28 页第 12 条「明确的数据和模型提交大小限制」从未下发）——**后续实际执行的是只解除 `m1_5`（t3 的 G-6 处置），`m2_3` 未入库**；② 在 `report/REPORT.md` §1.2 与 §5 显式声明「本机 5 个官方型号均已导出并跑通；**当时**入库仅 3 个，另两个可一条命令复现 `python deploy/export_onnx.py --model repvit_m1_5_in1k repvit_m2_3_in1k`」，把该条从「部署」改述为「评价 + 可复现导出」；③ 在 README 的 ONNX 表里补一行「未入库型号与复现命令」。**（2026-09-17，任务 t34：本缺口已闭合 —— 现行 `git ls-files onnx/` = **5 个**、其中**官方 ONNX 4 个**，选项②的前提「入库仅 3 个」已不适用；`m2_3` 仍可用 `python deploy/export_onnx.py --model repvit_m2_3_in1k` 一键复现。本条为历史记录，不需再执行。）** | `.gitignore` 或 `report/REPORT.md` / `README.md` / `report.pdf` |
| **G-7** | 报告 26 页 > 建议 12～20 页（`report.pdf` 的 `page_count = 26`） | 六-1「建议正文控制在 12～20 页」（**建议性**，非硬性）；影响观感与评委阅读成本 | 在现有 30 页压缩的基础上再压 6 页：优先把附录 A/C/D/E 中可迁到 PPT 的内容移出正文、把 `report/REPORT.md` §五 与 §14.2 两张重叠的表合并；**不要**为压页数删掉受保护的权威数值 | `report/REPORT.md`、`report.pdf`、`report/REPORT.docx` |
| **G-8** | PPT 所有性能图表必须注明 7 项：第 8/9/10/12/13/14 页元信息 7 项齐全，**第 4 页**（官方多型号评价表，含参数量/MACs）只有「1000 张 · 224×224 · batch 64 · FP32 · RTX 4060 Laptop」，缺「推理后端」与「测试次数」 | 六-2「所有性能图表必须注明 7 项」（第 17 页硬性要求） | 在 `report/ppt_svg/04_pretrained.svg` 的元信息行补「PyTorch 2.14.0+cu126（CUDA）」与「每型号评价 1 次（1000 张）」两段，重新编译该页并重导 PPTX/PDF | `report/ppt_svg/04_pretrained.svg`、`report/PPT_CONTENT.md`、`report/答辩PPT_RepViT.pptx`、`report/答辩PPT_RepViT.pdf` |

> **后记（2026-09-17 追加，任务 t22）**：本表 **G-3** 行引用的「beagle 96.8%」与 §3.1 的 1.1-7 同源、同样是 t2 时点文字，**该值在现行仓库已无落盘依据**（现行 beagle 命中全部属于**外部实拍图 / 演示路径**这一类：`report/REPORT.md:158` = 94.47%、`:435`/`:773` = 90.32%、`report/PPT_CONTENT.md:613` = 0.903，其余是标签 / 清单 / 命令示例）。现行案例口径以 `outputs/pretrained_eval/repvit_m0_9/top5_samples.json`（ImageNetV2）为准 —— **6 条样本、4 正 + 2 误**，与逐例题材 `outputs/report_assets/table_cases.{csv,md}`（5 型号 × 2 正 + 2 误 = 20 行）一致。**本行的 现象 / 影响 / 最小修复方案保留为 t2 时点记录，不改写历史**；处置结果见 `report/REPORT.md:144` §4.5「Top-5 样例与正误案例（2 正 + 2 误）」。该行的「最小修复方案」第 2 句已经成立（`top5_samples.json` 确有 6 条样本、`correct=true` 4 条），无需再改数字。

### 9.3 体验项（不影响得分限制，影响交付质量与答辩流畅度）

| ID | 现象与证据 | 影响 | 最小修复方案 | 需改动的仓库相对路径 |
|---|---|---|---|---|
| **X-1** | 报告页数超建议区间（同 G-7 的现象，此处只作体验记录；若 §9.2 的 G-7 已修则本项自动关闭） | 观感 | 同 G-7 | 同 G-7 |
| **X-2** | `PROVENANCE.md` §三写「各文件头部均带 `Source:` 注释标明来源类别」，实测 82 个入库 `.py` 中只有 **14** 个含 `Source:` 头 | 文档与事实不符，答辩被追问「标注是否完整」时的瑕疵 | 二选一：① 把其余脚本补齐 `Source: Self-written` / `Source: Third-party` 头；② 把 §三 的措辞改为「关键自撰模块（下表 8 个）带 `Source:` 头」 | `PROVENANCE.md`（或 `tools/*.py`、`deploy/*.py`、`datasets/*.py`、`utils/*.py` 的文件头） |
| **X-3** | **32** 个入库文件含采集机绝对路径（**平台中立的复核命令**：`git grep -lE '[A-Za-z]:\\{1,2}Users'` → **32 个文件**；`git grep -cE '[A-Za-z]:\\{1,2}Users'` → **278 行**；分布：`outputs/logs/train_opt_*.log` 17 + `outputs/benchmarks/{export_*,igpu_*}.json` 9 + `outputs/env_snapshot.json` + `outputs/metrics/bench.jsonl` + `outputs/metrics/imagenet_labels_report.json` + `outputs/verification/logits_reference_inventory.csv` + `report/{LOGITS_AUDIT_FINDINGS,VERIFICATION_LOGITS_PPT}.md` 2 = 32）。**代码层无写死路径**（`selfcheck` 的 `paths` = 0 处），问题只在落盘内容 | 可移植性与观感；与六-1 附注「不得在代码中写死个人电脑绝对路径」不冲突（该条只管代码） | 把 `deploy/benchmark.py`、`tools/eval_pretrained.py` 等落盘处的路径改成 `repo_rel()`（`deploy/compare_torch_onnx.py:12` 已有现成实现），新产物自然规范；**不建议**为纯路径字段重跑基准（会改动已发布的延迟数字），旧产物可保留并在 README 注明「早期落盘的 `onnx_path` 为采集机绝对路径」 | `deploy/benchmark.py`、`tools/env_check.py`、`README.md` |
| **X-4** | 进阶 2 的「训练曲线」没有四臂同图产物：四臂逐 epoch 数据齐全（`outputs/logs/opt_abl_{a,b,ab}_metrics.csv`），但 `outputs/curves/` 只有 baseline/opt_combo 的 4 张图 | 七-2（5 分）的「训练曲线」子项 | `python tools/plot_curves.py --runs baseline=outputs/logs/baseline_metrics.csv opt_abl_a=outputs/logs/opt_abl_a_metrics.csv opt_abl_b=outputs/logs/opt_abl_b_metrics.csv opt_abl_ab=outputs/logs/opt_abl_ab_metrics.csv --out outputs/curves/ablation_curves.png`，并在报告 §9.1 引用 | `outputs/curves/ablation_curves.png`（新增）、`report/REPORT.md` |
| **X-5** | 进阶 5 鲁棒性方向缺「Grad-CAM 变化」：`outputs/advanced/robustness_repvit_m0_9_pet37.json` 只有 `top1` / `macro_f1` / `mean_conf` / `conf_correct` / `conf_wrong` / `ece`，无 `gradcam` 相关字段（`git grep -F -- "gradcam" -- outputs/advanced/robustness_repvit_m0_9_pet37.json` → **0 命中**） | 七-5（最高 3 分）鲁棒性方向的分析完整性 | 抽 2~3 类扰动 × 2 级 severity 各出 1 张 Grad-CAM 拼图（复用 `tools/gradcam.py` 的挂载点 A/B），落盘 `outputs/advanced/robustness_gradcam.png` 并在附录 C 加一段对比 | `tools/robustness_test.py`、`outputs/advanced/robustness_gradcam.png`（新增）、`report/REPORT.md` |
| **X-6** | 摄像头/视频演示无实跑产物：`deploy/demo_camera.py` 完整（Top-5 / FPS / 按键切型号 / `--jitter-report`），但 `outputs/benchmarks/realtime_jitter.json` 不存在 | 七-5 的第三方向只有代码没有证据（该方向可自选，故不影响完成度） | 有摄像头时跑一次并落盘 `realtime_jitter.json`；无摄像头时用 `--source <视频文件>` 或直接删除 `--jitter-report` 的默认落盘声明，避免「声称有产物但找不到」 | `deploy/demo_camera.py`、`outputs/benchmarks/realtime_jitter.json`（新增或取消声明） |
| **X-7** | `report/REPORT.md:248` 与 PPT 第 5 页的「test 只评价一次」未限定为 baseline（与得分项 G-1 同源，此处记录措辞层面） | 同一事实在不同产物里的表述一致性 | 随 G-1 一并改为「Baseline 的 test 只评价一次（eval_count = 1）；优化臂因重跑累计为 2，模型选择仍只看验证集」 | `report/REPORT.md`、`report/ppt_svg/05_data.svg`、`report/PPT_CONTENT.md` |
| **X-8** | 本文件新增后，仓库已登记的「logits 引用清单不动点」会变：`tools/audit_logits_references.py` 扫描 `git ls-files` 全量文件（脚本里 `git ls-files -z` 调用，**现行 L461**），而 `report/SYNC_NOTES.md` 要求维持 `4124 条引用 / 153 个文件`；本报告刻意保留了 `7.093e-06` / `6.199e-06` / `1.501e-06` 等权威值字面量 | 审计文档之间的登记值一致性（不影响得分） | 修完本轮所有缺口后统一执行：`python tools/audit_logits_references.py`（重生成 `outputs/verification/logits_reference_inventory.csv`）→ 用 `--stdout` 读新条数 → 同步更新 `report/LOGITS_AUDIT_FINDINGS.md` §2.3/§2.4、`report/LOGITS_AUDIT.md`、`report/SYNC_NOTES.md`、`report/VERIFICATION_LOGITS_PPT.md` 里登记的「4124 / 153」与 CSV SHA256。**【2026-09-17 状态：本项即任务 t10 的 C 段；因 t19 还将改动 `outputs/benchmarks/family_summary.csv` / Pareto 产物 / PPT 三件套 / `report/REPORT.md`，不动点会在 t19 之后再次变化，故 C 段已转出给 t20 在 t19 落定后统一刷新与登记 —— 本条保留为转出记录。】** | `outputs/verification/logits_reference_inventory.csv`、`report/LOGITS_AUDIT_FINDINGS.md`、`report/LOGITS_AUDIT.md`、`report/SYNC_NOTES.md`、`report/VERIFICATION_LOGITS_PPT.md`（后两者中的登记值同样需要同步；`VERIFICATION_LOGITS_PPT.md` 不在 t10 的写入范围内，由 t20 一并收口） |

### 9.4 可选增强（**不计入必须修复清单**，仅记录可争取的分数）

| ID | 现状 | 可争取 | 最小做法 |
|---|---|---|---|
| **O-1** | 拓展 4「核心模块独立实现」：`tools/reparam_deep.py:11-52` 已有独立的手写 Conv+BN 融合与 RepVGGDW 三分支融合，但没有「手写 vs 参考实现」的数值对齐产物（现有 `advanced/..._reparam_report.json` 的 `2.980e-06` 属无效前提的 B7 桶，`report/LOGITS_AUDIT_FINDINGS.md` §1.3 已禁止引用） | 拓展最高 2 分 | 写一个小脚本：对同一 block 分别用 `fuse_repvggdw_manual()` 与 timm `fuse()` 生成推理模块，在同批固定输入上比较输出（IO 尺寸 + `max\|Δ\|`），落盘 `outputs/advanced/manual_fusion_alignment.json` |
| **O-2** | 拓展 2「多随机种子」：现有 `ablation_summary.json` 的 `noise` 用单种子 σ 估计替代 | 拓展最高 2 分 | 至少 3 个 seed 重跑 Baseline 与所选优化臂（每组 40 epoch），落盘 mean/std 与「是否超过 2σ」判定 |
| **O-3** | 拓展 1/3（压缩、边缘/下游部署）完全未开展 | 拓展最高 6 分 | 需新工作量与外部设备/依赖，属团队范围决策，不在本次修复范围 |

---

## 10. 证据命令清单（可直接运行，静态只读）

以下命令全部来自仓库内脚本，**运行前请确认是否需要写产物**；标 `[只读]` 的不写任何文件。

| 目的 | 命令 |
|---|---|
| 全仓自检（34 项，会覆写报告 JSON） | `python tools/selfcheck.py`（只读替代：`python tools/selfcheck.py --json %TEMP%\selfcheck.json`） |
| 数据完整性 `[只读]` | `python tools/assert_data.py --root data/oxford-iiit-pet` |
| 数据泄漏 `[只读]` | `python datasets/audit_leakage.py` |
| 权重校验 `[只读]` | `python tools/check_weights.py --dir checkpoints/pretrained` |
| 骨干是否被训练 `[只读]` | `python tools/check_backbone_updated.py --ckpt checkpoints/baseline_best.pt` |
| 控制变量差异 `[只读]` | `python tools/diff_config.py configs/baseline.yaml configs/opt_combo.yaml` |
| 预算等价 `[只读]` | `python tools/same_budget.py --a outputs/logs/baseline_metrics.csv --b outputs/logs/opt_combo_metrics.csv` |
| 参数量/MACs 三口径 `[只读]` | `python tools/count_params.py --model repvit_m0_9`、`python tools/count_flops.py --model repvit_m0_9 --input-size 224` |
| 混淆矩阵校验 `[只读]` | `python tools/check_cm.py --file outputs/confusion_matrix/baseline_cm.csv` |
| 模型登记表 `[只读]` | `python deploy/model_registry.py` |
| 结构重参数化复跑（写 `outputs/verification/`，覆盖不到正式产物） | `python tools/reparam_verify.py --model repvit_m0_9_pet37 --weights checkpoints/baseline_best.pt --num-samples 32 --batch-size 8 --seed 20240912 --skip-onnx --out-dir outputs/verification/reparam_pet37` |
| PyTorch↔ONNX 一致性复跑（n=12） | `python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --images datasets/lists/pet_test.txt --limit 12 --out outputs/verification/consistency_repvit_m0_9_pet37_n12.json` |
| ONNX 推理演示 | `python deploy/infer_onnx.py --model repvit_m0_9_pet37 --image external/beagle__ILSVRC2012_val_00000162_00.JPEG` |
| 性能测试 | `python deploy/benchmark.py --model repvit_m0_9_in1k --model repvit_m1_0_in1k --model repvit_m0_9_pet37 --warmup 10 --runs 50 --threads 4 --out-dir outputs/verification/benchmarks` |
| **报告素材自检（当前会红：FAIL=36，见 A-1）** `[只读]` | `python tools/check_report_assets.py` |
| logits 引用清单（会写 CSV，见 X-8） | `python tools/audit_logits_references.py --stdout`（只统计不写文件） |

**既有验收闸（本次直接引用其落盘结果，未重复执行）**

- `python tools/selfcheck.py` → `outputs/metrics/selfcheck_report.json`：`summary = {total: 34, pass: 34, fail: 0}`
- `python tools/assert_data.py` → `outputs/metrics/dataset_report.json`：`pass: true`、`fails: []`

---

## 11. 只读审计自证

1. **本次唯一写入（t2 时点记录）**：`report/REQUIREMENTS_AUDIT.md`（本文件；t2 首次写入时为 128,587 B / 740 行、未跟踪）。除此之外，t2 那一轮未创建、修改、删除仓库内任何文件。**（后记 2026-09-17，任务 t30：本文件此后经 t10 / t22 / t25 / t27 / t30 多轮就地修订。**其体积与行数是自指值 —— 写死它就会改变它，故本文件不固化**；实测命令：`python -c "import os;p='report/REQUIREMENTS_AUDIT.md';print(os.path.getsize(p), sum(1 for _ in open(p, encoding='utf-8')))"`。）**
2. **审计开始时的 `git status --porcelain`**（记录在案，见下文第 4 条复现）：当时**唯一一行**是 ` M outputs/metrics/selfcheck_report.json`，即**在本次审计开始前就已存在的预存改动**（来自此前某轮 `tools/selfcheck.py`）。该文件在本审计进行期间（21:37:31）又被**并发的队友任务**重写了一次，本审计全程未运行 `tools/selfcheck.py`，因此这两次改动都不属于本次审计。
3. **审计过程中出现、但不属于本次审计的并发改动**：审计后半段工作区曾出现 `?? _spec13_scan.py`（创建时间 `2026-09-16 21:42:41`），是本团队另一任务（§1.3 口径扫描）的临时脚本；它随后被其归属者自行删除。本审计**既未创建也未删除**它。
4. **复现自证**（审计收尾时实测）：
   ```powershell
   cd RepViT-Reproduction
   git status --porcelain      # 仅两行： M outputs/metrics/selfcheck_report.json 与 ?? report/REQUIREMENTS_AUDIT.md
   git diff --stat             # outputs/metrics/selfcheck_report.json | 2 +-   （预存改动，非本次）
   ```
   即：对既有**入库**文件而言，本审计没有新增任何 diff（`git diff --stat` 只显示那条预存改动）；本文件是新增未跟踪文件。审计期间对既有文件只做 `read` / `grep` / `git grep` / `git ls-files` / `git status` / `git rev-parse`，以及两个**只打印、不落盘**的脚本观测：`python tools/check_report_assets.py`（只读检查，见 §9 A-1 的 FAIL=36）与 `python deploy/model_registry.py`（打印登记表与 ONNX 就位情况）。
5. **审计基准与远端一致性**：`git rev-parse HEAD` = `git rev-parse origin/main` = `dc0527bc346a909cf5cf529f0755ff5bbdb8100e`（分支 `master` ↔ 远端 `main` 已同步，与任务背景描述一致）。
6. **边界声明**：§1.3 的 5 条挂起项（含 1.1-2 这条耦合项）本报告只给判定与「按用户决议挂起，不在本次范围」，**不含任何改动建议**；本报告也未触碰 `report/SPEC13_FREEZE.json`、`report/SPEC13_SWITCH_RUNBOOK.md` 等由其它任务负责的文件。
7. **本文件的后续就地修订（2026-09-17，t10 任务，只改本文件）**：① 任务 A —— 删除 3 处采集机绝对路径字面量，改用 `<repo>/` 占位符与**平台中立**的复核命令 `git grep -lE '[A-Za-z]:\\{1,2}Users'`；② 任务 B —— §2.1 的 5 条挂起判定**原样保留**，新增 §2.1.1 记录后续决议与当前真实状态；③ 一并把本文中引用已被取代口径的证据指向改为现行 ImageNetV2 口径（逐处见 §2.1.1 的「本次指向调整清单」，**判定列零改动**，判定统计当时仍为 352 行 = 满足 307 / 部分 23 / 缺失 17 / 挂起 5（**t10 时点值**；此后 t30 将 `A1-1` 由「部分」改判「满足」，**现行权威值为 §8 的 352 = 308 / 22 / 17 / 5**））。除本文件外**未修改任何其它文件**——本次修订发生在工作区已有大量并发改动（约 149 行 `git status`）之中，`git diff --stat` 不含本文件（它是未跟踪文件），故本文件的改动不会污染任何既有文件的 diff。**本次未运行 `tools/audit_logits_references.py` 的登记同步（C 段已转出给 t20，见 §9 X-8 与 §1 第 6 条）。**

---

## 12. 附录：本次审计直接引用的权威产物索引

| 类别 | 路径 |
|---|---|
| 自检与数据 | `outputs/metrics/selfcheck_report.json`、`outputs/metrics/dataset_report.json`、`outputs/metrics/leakage_check.json`、`outputs/metrics/pet_label_audit.json`、`outputs/metrics/imagenet_labels_report.json`、`outputs/metrics/split_audit.json`；现行评价子集（ImageNetV2 matched-frequency）的来源 / 许可 / 哈希见 `report/IMAGENETV2_PROVENANCE.md` |
| 官方评价 | `outputs/pretrained_eval/<model>/{metrics,latency,top5_samples}.json`、`predictions.csv`、`cases/`、`outputs/pretrained_eval/summary.csv`（家族级汇总，位于型号目录上一级） |
| 训练与测试 | `outputs/logs/{baseline,opt_*}_metrics.{csv,jsonl}`、`outputs/logs/{baseline,opt_*}_config_effective.{yaml,json}`、`outputs/metrics/{baseline,opt_*}_test.json`、`outputs/logs/*_test_eval_count.json`、`outputs/metrics/backbone_updated.json`、`outputs/metrics/weight_load_report.json` |
| 曲线与可视化 | `outputs/curves/*`、`outputs/confusion_matrix/*`、`outputs/predictions/*`、`outputs/gradcam/*`、`outputs/metrics/gradcam_meta.json` |
| 重参数化 | `outputs/reparam/repvit_m0_9_pet37_reparam_report.json`、`outputs/reparam/logits_diff.json`、`outputs/reparam/*_onnx_nodes.json`、`outputs/reparam/*_structure_{before,after}.txt`、`outputs/advanced/block_22_fusion_report.txt` |
| ONNX 与性能 | `onnx/*.onnx`、`outputs/benchmarks/*_benchmark.json`、`outputs/metrics/bench.jsonl`、`outputs/benchmarks/summary.csv`、`outputs/benchmarks/family_summary.csv` |
| 一致性 | `outputs/metrics/consistency_repvit_m0_9_pet37.json`、`consistency_repvit_m0_9_in1k.json`、`consistency_repvit_m1_0_in1k.json`、`outputs/verification/consistency_*_n12.json` |
| 进阶产物 | `outputs/advanced/{ablation_summary.{csv,json},pareto_summary.json,marginal_returns.csv,model_recommendation.csv,params_vs_acc.png,latency_vs_acc.png,macs_vs_acc.png,robustness_repvit_m0_9_pet37.json,interp/*}`、`outputs/benchmarks/igpu_*.json` |
| 报告与 PPT | `report/REPORT.md`、`report.pdf`、`report/REPORT.docx`、`report/PPT_CONTENT.md`、`report/答辩PPT_RepViT.{pptx,pdf}`、`report/ppt_svg/*.svg`、`report/LOGITS_AUDIT.md`、`report/LOGITS_AUDIT_FINDINGS.md`、`report/SYNC_NOTES.md`、`report/sources/literature.yaml`、`outputs/report_assets/table_*.{csv,md}` |
| 来源与标注 | `PROVENANCE.md`、`PROGRESS.md`、`README.md`、`external/images_manifest.csv`、`requirements.txt`、`requirements.lock.txt` |
| 代码真源 | `tools/`、`deploy/`、`datasets/`、`models/`、`utils/`、`configs/` |

**本报告不含 §1.3 的任何改动建议；§1.3 的 5 条挂起项按用户决议保持现状，等待考核方清单。**

**【后记 2026-09-17】** 上面这句是 t2 审计当时的结论，**保留不改**。此后用户已指定 **ImageNetV2 matched-frequency 1000 张固定子集**为唯一评价口径、自建子集整体清除，5 条挂起项的当前真实状态与逐条依据见 **§2.1.1**；其中 1.3-S4 已可判 `满足`，1.3-S5 部分解除（V2 口径下改用等价如实声明），1.3-S1/S2/S3 仍待考核方清单。本文其余引用旧口径的证据行已在本次一并指向现行口径（清单见 §2.1.1 的「本次指向调整清单」）。
