# 官方模型评价口径 · 切换 Runbook（若考核方另下发清单时执行）

> **当前状态：唯一口径 = ImageNetV2 matched-frequency 的 1000 张确定性固定子集**（2026-09-16 起生效）。
> 本文是「**考核方另下发固定图片列表与标签**」时的执行说明书，**不是**执行记录。
> 冻结基线：`report/SPEC13_FREEZE.json`（**revision 4**，2026-09-17 由 t19 在 revision 3 基础上刷新；`frozen_fragments` 的定位已改为内容锚点）；出处与许可：`report/IMAGENETV2_PROVENANCE.md`。
> 边界：本任务（t11）只改 `README.md` / `PROGRESS.md` / `configs/pretrained_eval.yaml` /
> `tools/run_all_pretrained.py`，新建 `datasets/make_imagenetv2_subset.py` /
> `datasets/lists/imagenetv2_mf_1000.txt` / `report/IMAGENETV2_PROVENANCE.md` /
> `report/SPEC13_FREEZE.json` / 本文，并**删除** `datasets/lists/imagenet_val_subset.txt`。

---

## 1. 现存事实（切换的起点）

### 1.1 唯一口径：ImageNetV2 matched-frequency 1000 张

| 项 | 值 |
|---|---|
| 清单 | `datasets/lists/imagenetv2_mf_1000.txt`（入提交物） |
| 规模 / 哈希 | **1000 行 / 87,780 B**，sha256 `d5532205d8f30099aa70ed9392078adcdbd305c47055930c66e704f7976c9d05` |
| 选取规则 | 按类别下标 0..999 升序；每类目录（目录名=**十进制 0-based 下标**）内文件名 `sorted()` 取第一个；**无随机种子** |
| 生成脚本 | `datasets/make_imagenetv2_subset.py`（归档 sha256 校验 + 解压 + 全部断言，幂等） |
| 归档 | HF `vaishaal/ImageNetV2` → `imagenetv2-matched-frequency.tar.gz`，**1,264,079,360 B**，sha256 `f0c37fdf…c9ca7c`（= HF `X-Linked-ETag`），MIT / 非 gated |
| 数据本体 | `data/imagenetv2/matched-frequency/<label>/<sha1>.jpeg`，1000 类 × 10 张 = 10,000 张（`data/` 被 `.gitignore` 忽略，不入提交物） |
| 标签对齐 | 目录名 = 0-based 下标，与 `labels/imagenet_classes.txt` 行号一一对应（0=tench / 207=golden retriever / 281=tabby / 999=toilet tissue；另经模型实测复核，见 provenance §5） |
| 口径纪律 | **不得**称它为「ImageNet-1K 验证集」；其 Top-1 与 1K val / 论文公布值**不可并列比较**（文献中通常低 10~15 个点） |

### 1.2 已删除、且**不得**恢复的旧口径

* `datasets/lists/imagenet_val_subset.txt`（旧自建 ImageNet-1K val 分层子集，1000 行、`seed=20260912`）
  已按用户 09-16 决议 **`git rm` 删除**：**不留存对照、不留存附录、不留存历史**。
* 仓库内**不允许**存在第二套「官方模型评价」口径。
* `data/imagenet/val/`（5 万张原始 ImageNet-1K val 图像）只是本机上游原始数据，**不参与任何评测口径**
  （政策明确不在删除范围）。

### 1.3 连带状态（**t11 时点历史快照**；各归其主，均已收口）

> **限定（2026-09-17 追加，任务 t25）**：本表是**历史快照（t11 时点；现行状态见 POLICY §10.6 / §15.2）**——表内各项均已由各自归属任务完成收口，**不要按表内的「t12 / t4 / t9 待办」再开一轮修复**。现行状态：`python tools/selfcheck.py` = **34/34 PASS**、`python tools/assert_data.py` = **ALL PASS**、`python tools/check_report_assets.py --strict` = **OK=177 / WARN=0 / FAIL=0**；`report/REPORT.md` → `REPORT.docx` → `report.pdf` 与 PPTX/PDF 均已同一轮重生成，旧口径/旧数字 0 命中。

| 位置 | 现状（t11 时点） | 归属 |
|---|---|---|
| `tools/selfcheck.py`（`onnx.realimg` 白名单、`official.top1` 容差） | 仍指向旧清单口径 → 删清单后会**短暂 FAIL**，属**预期** | t12 |
| `deploy/compare_torch_onnx.py:82`（默认清单）/ `:109`（`source` 名） | 仍指向旧清单 | t12 |
| `tools/audit_logits_references.py:83/90`、`tools/eval_pretrained.py:14/246/656` 的注释与提示 | 仍引用旧清单 | t12 |
| `datasets/make_imagenet_subset.py`、`datasets/imagenet_subset.py` | 只服务旧子集，应删除或标注废弃 | t12 |
| `outputs/**`（含 `metrics/subset_report.json`、一致性证据） | 就地替换，**不另存对照目录** | t12 |
| `report/REPORT.md` / `REPORT.docx` / `report.pdf` | 仍含旧子集叙事与数字 | t4 |
| `report/答辩PPT_RepViT.pptx` / `.pdf` / `PPT_CONTENT.md` / `ppt_svg/*.svg` | 同上 | t9 |

> 逐行落点见 `report/SPEC13_FREEZE.json` 的 `mentions.pending_sync_text_files` 与
> `mentions.pending_sync_binary_artifacts`（含 PPTX 幻灯片号 / PDF 页码 / DOCX 段落）。

---

## 2. 触发与前置确认

触发：**人类**把考核方下发的材料放进 `data/provided/`（该目录 Agent 只读不改）。

开工前确认三件事，缺一不进切换：

1. 是否**同时**给了标签（试题 PDF §1.3：「考核方提供固定图片列表及标签」）。只给列表 → 停下索要。
2. 列表里的图片**本机能否解析**（落在 `data/imagenet/val/`、考核方另发图片等）。
   注意：ImageNetV2 与 ImageNet-1K val 不是同一批图，**不能**用 V2 顶替考核方清单。
3. 本次切换**是否已获用户/队长确认**（口径改动属显式决议事项）。

---

## 3. 放置路径

| 材料 | 落地路径 | 说明 |
|---|---|---|
| 考核方原始文件 | `data/provided/<考核方原文件名>`（保留原字节、原文件名） | 只读留档；`data/` 被 `.gitignore` 忽略 → **不会**进提交物 |
| 仓库内规范化清单 | **`datasets/lists/imagenet_provided_1000.txt`**（新路径，入提交物） | Tab 分隔、`<相对仓库根路径>` + `<0-based 标签>` |
| 标签文件 | 优先沿用 `labels/imagenet_classes.txt`；仅当与考核方标签**逐行不一致**时才新增 `labels/imagenet_provided_classes.txt` | 一致性校验见 K2 |
| 图片本体 | 一般不新增；若考核方另发图片，落 `data/imagenet/val_provided/` 并同步 `data_root` 约定 | 需另写解析映射 |

> **两个坑**：① 清单只放 `data/provided/` **进不了提交物**（`.gitignore` 忽略整个 `data/`），
> 必须落一份规范化副本到 `datasets/lists/`；② **不要**覆盖 `datasets/lists/imagenetv2_mf_1000.txt`
> ——它的 sha256 已写进 README / `SPEC13_FREEZE.json` / provenance，覆盖会让全部文档对不上。

---

## 4. 校验清单 K1–K8（任一 FAIL 就停）

| # | 检查项 | 判据 | 命令 |
|---|---|---|---|
| K1 | 行数 | `500 ≤ n ≤ 1000`（PDF p3「建议500～1000张」） | `python -c "print(sum(1 for l in open('data/provided/<清单>',encoding='utf-8') if l.strip()))"` |
| K2 | 标签 0-based 且与类别索引对齐 | `0 ≤ label ≤ 999`；与 `labels/imagenet_classes.txt` 1000 行索引一致 | `python tools/eval_pretrained.py --cfg configs/pretrained_eval.yaml --set pretrained_eval.data_list=datasets/lists/imagenet_provided_1000.txt pretrained_eval.check_data=true` |
| K3 | 图片可读 | 输出行 `坏图/缺图 = 0 张` | 同 K2（`check_data` 分支 `tools/eval_pretrained.py:704-705`，建模前返回） |
| K4 | 类别覆盖与每类张数 | 记录 `per_class_min/max`；并复核 `tools/selfcheck.py:267` 的容差注释是否需按新样本数调整 | K2 输出 + 手记 |
| K5 | 与当前 V2 清单差异 | 同名交集/新增/缺失；**标签冲突数必须为 0** | 见下方差异片段 |
| K6 | 路径可解析 | 路径前缀与 `data_root` 一致（V2 是 `data/imagenetv2/...`，考核方很可能是 `data/imagenet/val/...`） | `python -c "import collections;print(collections.Counter('/'.join(l.split(chr(9))[0].split('/')[:2]) for l in open('datasets/lists/imagenet_provided_1000.txt',encoding='utf-8') if l.strip()))"` |
| K7 | 训练/调参隔离 | 该清单**不得**出现在任何训练配置里（PDF p3：「验证子集不得用于训练或调参」） | `git grep -n "imagenet_provided_1000" -- configs/ tools/train.py` |
| K8 | 留痕 | 记录考核方原始文件 sha256、字节数、下发时间、执行人确认 | `Get-FileHash data/provided/<清单> -Algorithm SHA256` |

K5 差异片段（把 `<新清单>` 换成实际路径）：

```powershell
python -c "a={l.rsplit(None,1)[0]:int(l.rsplit(None,1)[1]) for l in open('datasets/lists/imagenetv2_mf_1000.txt',encoding='utf-8') if l.strip()}; b={l.rsplit(None,1)[0]:int(l.rsplit(None,1)[1]) for l in open('<新清单>',encoding='utf-8') if l.strip()}; ka,kb=set(a),set(b); c=[(k,a[k],b[k]) for k in ka&kb if a[k]!=b[k]]; print('V2',len(a),'新',len(b),'同名交集',len(ka&kb),'新增',len(kb-ka),'缺失',len(ka-kb),'标签冲突',len(c)); print(c[:5])"
```

> V2 与 ImageNet-1K val 的**文件名空间不同**，所以「同名交集」通常为 0 —— 属正常现象；
> 该步骤真正的用途是发现「同名却标签不一致」（标签冲突 > 0 必须停下找考核方）。

---

## 5. 重跑命令

**0) 先改硬编码（否则白跑）**：`tools/run_all_pretrained.py` 的 `BASE` 用 `--set` **覆盖**了
`pretrained_eval.data_list`，优先级高于 yaml。只改 yaml 的话，批量重跑仍会**静默**用旧清单：

* A（推荐）：把 `tools/run_all_pretrained.py:21` 与 `configs/pretrained_eval.yaml:25` 一起改；
* B：从 `BASE` 里删掉该键，让 yaml 成为唯一来源。

```powershell
# 1) 预检（不加载模型）：list/labels 对齐 + 坏图统计
python tools/eval_pretrained.py --cfg configs/pretrained_eval.yaml `
  --set pretrained_eval.data_list=datasets/lists/imagenet_provided_1000.txt pretrained_eval.check_data=true

# 2) 批量重跑 5 个型号（M0.9 必做 + M1.0 + 家族扫描 3 个）
python tools/run_all_pretrained.py --cfg configs/pretrained_eval.yaml `
  --model repvit_m0_9 repvit_m1_0 repvit_m1_1 repvit_m1_5 repvit_m2_3

# 3) 逐型号等价命令（把 <MODEL> 换成裸名；与既有日志同形）
python tools/eval_pretrained.py --cfg configs/pretrained_eval.yaml `
  --set pretrained_eval.model=<MODEL> `
        pretrained_eval.weights=checkpoints/pretrained/<MODEL>_distill_300e.pth `
        pretrained_eval.out_dir=outputs/pretrained_eval/<MODEL> `
  --set pretrained_eval.data_list=datasets/lists/imagenet_provided_1000.txt `
        pretrained_eval.labels=labels/imagenet_classes.txt `
        pretrained_eval.input_size=224 pretrained_eval.batch_size=64 `
        pretrained_eval.crop_pct=0.875 pretrained_eval.interpolation=bicubic `
        pretrained_eval.warmup=10 pretrained_eval.runs=50 pretrained_eval.threads=4

# 4) ONNX n=12 一致性证据（source/images 字段必须指新清单）
python deploy/compare_torch_onnx.py --model repvit_m0_9_in1k `
  --images datasets/lists/imagenet_provided_1000.txt --limit 12 `
  --out outputs/verification/consistency_repvit_m0_9_in1k_n12.json

# 5) 自检 + 报告素材
python tools/selfcheck.py
python tools/check_report_assets.py --root . --strict
```

> 第 4 步依赖 `deploy/compare_torch_onnx.py` 的 `source` 命名与 `tools/selfcheck.py` 的白名单一致
> （两处归属 t12 一并改）。若仍写旧名，产物会**自述错误来源**且 selfcheck 判 FAIL。

---

## 6. 文档同步点（切换后必须改的位置）

| 文件 | 位置 | 改什么 |
|---|---|---|
| `configs/pretrained_eval.yaml` | `pretrained_eval.data_list`（`:25`） | 指向新清单；口径纪律注释照旧保留 |
| `tools/run_all_pretrained.py` | `BASE` 的 `data_list`（`:21`） | 与新清单一致（见 §5 第 0 步） |
| `datasets/lists/imagenetv2_mf_1000.txt` | 整文件 | **保留**（公开可复现口径）；若成为非默认口径，README 需写明如何显式指回 |
| `README.md` | §2.2「唯一口径」声明段、§2.2.1 主口径块、§4 口径边界 | 改为考核方下发口径；V2 段落降级为「历史来源/可复现出处」 |
| `PROGRESS.md` | 阻塞项「官方模型评价的数据源」段、决策记录 09-16 条目、指标表 `official_*`/`consistency_*` 行 | 同上；不得留下与新口径矛盾的表述 |
| `report/IMAGENETV2_PROVENANCE.md` | §6/§7 | 说明评测口径已改；V2 provenance 仍作为来源档案保留 |
| `report/SPEC13_FREEZE.json` | 整体再生（revision +1） | `subset` 指向新清单；`removed_paths` 记录 V2 清单是否降级 |
| 派生物 | `report/REPORT.md` → `REPORT.docx` → `report.pdf`；`答辩PPT_RepViT.pptx` → `.pdf` → `PPT_CONTENT.md` | 由 t4 / t9 依 `SPEC13_FREEZE.json` 的 pending 清单同步（含页码级定位） |

---

## 7. 免责句落点（三分法，**禁止混用**）

| 场景 | 必须使用的表述 |
|---|---|
| **考核方下发的指定子集** | 试题 PDF §1.3 逐字原句：`本结果为指定ImageNet-1K验证子集上的实际运行结果，不代表论文完整ImageNet-1K验证集结果。`（该句在 PDF 第 3 页页脚给出、跨页落在第 4 页页首） |
| **ImageNetV2 子集**（当前口径） | `本结果为 ImageNetV2 matched-frequency 的 1000 张确定性固定子集（1000 类各 1 张）上的实际运行结果。ImageNetV2 是 Recht et al. (NeurIPS 2019) 按 ImageNet 分布重新采样构建的独立测试集，不是 ImageNet-1K 验证集；其准确率与 ImageNet-1K 验证集准确率、与论文公布值不可直接比较。` |
| **任何情况** | 不得把 PDF 原句照搬到 V2 结果上（V2 既不是 1K 验证子集、也不是考核方指定子集，照搬即虚假陈述）；也不得把两套数字并列或算差值 |

现行落点：`README.md` §2.2.1 声明块与 §4 口径边界段、`report/REPORT.md` §4、
`report/PPT_CONTENT.md` P04 与 PPTX 第 4 页、`PROGRESS.md` 决策记录。

---

## 8. `crop_pct` 口径对照（透明性记录，**不是调参**）的闭合说明

**声明模板（可直接引用）**

> 主口径 `crop_pct = 0.875`（`Resize(256) + CenterCrop(224)`）在评测开始前按官方 THU-MIG 验证流程
> **a priori 固定**，与任何结果无关；`also_eval_crop_pct: [0.875, 0.95]` 只是额外按 timm
> `pretrained_cfg` 口径（`Resize(235)`）做一次**纯推理**并原样落盘，用于量化预处理口径差异。
> **该对照未用于任何模型选择或超参选择**：上报值固定取先验主口径 0.875 的数值，没有任何
> 「取更高者」的分支。

**代码证据**（`tools/eval_pretrained.py`）：`86`（默认值即官方口径）、`319-330`（两个口径都是既定标准口径）、
`752`（主变换先于任何推理）、`858-859`（主指标来自主口径推理）、`863-864`（注释：只做额外纯推理、
不重复写产物）、`865-866`（`reused: True` 直接复用主结果）、`867-879`（对照项只 append，不改 `top1`）、
`963`（落盘 `top1` = 主口径值）、`977/979`（口径与对照分字段落盘）。
消费者核查：`git grep -n crop_pct_sweep` 仅命中写入点、落盘数据与文档说明，**无任何脚本读取它做选择**。

> 切到考核方清单后这套声明**结构不变**，但绝对数值会变，必须按新清单重跑后整体替换数字。

---

## 9. 失败处理与回滚

* **任一 K 项 FAIL**：回 `data/provided/` 找考核方补齐；**不得**自行猜标签、按文件名猜类别，
  或拿 ImageNetV2 / ImageNet-1K val 的图去「凑」考核方清单。
* **切换后指标异常**：先查标签映射与路径解析（K2/K3/K6），再查预处理口径（`crop_pct` / `interpolation`）；
  **不要**为了好看去调 `crop_pct`。
* **回滚**：`datasets/lists/imagenetv2_mf_1000.txt` 与 `report/IMAGENETV2_PROVENANCE.md` 全程保留且不被覆盖，
  因此回滚 = 把 `configs/pretrained_eval.yaml:25` 与 `tools/run_all_pretrained.py:21` 指回 V2 清单，
  再用 `report/SPEC13_FREEZE.json` 的 `subset.sha256` 与 `frozen_fragments` 逐条复核。
  旧自建子集**已删除，不再恢复**（政策：不留存对照、不留存附录、不留存历史）。

---

## 10. 本文未执行的验证（诚实声明）

* §3–§8 的切换步骤**未执行**：`data/provided/` 内没有考核方清单（只有说明 README）。
* 本文引用行号为 2026-09-16 快照值；同轮其它任务（t12/t4/t9/t5/t10）会改动相关文件，行号可能漂移
  ——**口径是否被改动一律以 `report/SPEC13_FREEZE.json` 的 `fragment_sha256` 与 `subset.sha256` 为准**。
* §4 中「建议固化」的差异脚本本次**未新建**（超出 t11 边界）；执行切换时建议先补
  `tools/diff_subset_lists.py` 再动手。
