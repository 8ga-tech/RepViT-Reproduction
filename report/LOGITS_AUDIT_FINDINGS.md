# logits 误差引用审计发现（LOGITS_AUDIT_FINDINGS）

- 审计日期：2026-09-16（工作区 HEAD `2d65e63`）
- 扫描清单：`outputs/verification/logits_reference_inventory.csv`（**4791 条引用 / 179 个文件**；由 `python tools/audit_logits_references.py` 可重复生成，两次运行 SHA256 相同。**2026-09-17 t42 在暂存态第七次登记**：t35 为 4790 / 178、t32 为 4788 / 178、t28 为 4788 / 178、t23 为 4778 / 177、t21 为 4776 / 177、t20 为 4731 / 176，再早的历史登记值为 4124 条 / 153 个文件）
- 本文用途：**t2（README / REPORT / LOGITS_AUDIT 文档统一）与 t3（PPT 改写）的措辞依据。**
  第 1 节可以照着抄数值与口径；第 3 节是逐条修改单（文件 + 行号/页码/幻灯片 + 建议改法）。
- 边界：本次只做扫描、分级与发现记录。**没有改动任何 `outputs/` 实验产物的数值，没有改 README/REPORT/PPT 正文，没有 commit/push。**

---

## 0. 一页结论

1. 仓库里的 logits 误差引用可以无损地归入 **13 个实验桶**，其中 **6 个**（B1/B2/B3/B12/B13/B4）是答辩正文可以引用数值的主桶；其余是独立桶（B5/B6/B9/B10）、不得作为结论的桶（B7）和无产物的孤值桶（B8）。
2. **一个数值属于哪个桶，由「它出现在哪个落盘产物里」决定，不由散文字面决定。** 本次审计把 4 个权威产物的数值建成"规范值集合"，逐 token 按写入精度匹配，因此 `6.199e-06`（B1）与 `6.53e-06`（B8 孤值）不会被混为一谈。
3. **口径矛盾（缺陷 D1）已定案：** `report/REPORT.md:796-797` 的"5.25e-06 属于旧的 n=8 对照记录"**没有证据支持**，应按 `report/LOGITS_AUDIT.md:15` 的说法收敛。三条相互独立的证据见 D1。
4. **8 个孤值**（5.25e-06 / 1.41e-06、5.245e-06 / 1.414e-06、6.53e-06、6.527e-06、2.265e-06、4.108e-07）在仓库的 **14 个提交里的 `outputs/` 树中一次都没有出现过**；初始交付提交 `81693dd` 就已经同时存在"散文字面值"和"n=12 的落盘 JSON"。任何一个都不应再作为实验结论引用。
5. **绝对路径（缺陷 D2）**：`outputs/metrics/consistency_*.json` 的 `onnx_path` / `images` 是采集机绝对路径，来源是 `deploy/compare_torch_onnx.py:95-97`。**源码本身没有写死绝对路径**（符合试题第 14 页），问题在落盘内容；同目录 `outputs/verification/` 副本只把 `images` 相对化了，`onnx_path` 仍是绝对路径 —— 任务描述里"副本用相对路径"只对一半成立。
6. **混写与外推（缺陷 D3–D5）**：封面/状态页把"重参数化 32 输入误差"和"ONNX 一致率 100%"并列（`答辩PPT_RepViT.pptx` slide 1/2、`README.md:13`）；`六个 ONNX 模型一致率 100%`（`ppt_svg/13_onnx.svg:5`、slide 2）与"没有出现九类典型问题"（`REPORT.md:807`、`report.pdf` 第 23 页、`REPORT.docx` para 913）超出证据；"结构重参数化在数值上完全等价"（`ppt_svg/12_reparam.svg:5`、`REPORT.md:963`、`report.pdf` 第 27 页、`REPORT.docx` para 1053）同样是超证据断言。（括号里的行号/页码是**审计当时**（压缩前 37 页版）的定位；t17 重导、t22 压缩后的现状见 §D4 / §D5 末尾的「现状」条。）
7. **导出物不同步（缺陷 D6）**：`report.pdf` 第 21 页（审计当时）的 12.4 表还是**初始版**（写着 `M0.9 官方 C=1000` 的 2.265e-06 / 4.108e-07、BN `108 → 0`），而 `report/REPORT.md:720-728` 已经改成"—（旧报告权重键不匹配，不作结论）"和 `107 → 0`。文档改完必须重新导出 PDF/DOCX。**该缺陷现已修复**：现行 `report.pdf`（26 页）的 12.4 表位于**第 17 页**，C=1000 列已是"—（旧报告权重键不匹配，不作结论）"、BN `107 → 0`；`2.265`、`4.108` 与 `108 → 0` 在压缩后的 REPORT.md 与现行 PDF 中均 0 命中（详见 §D6 的「现状」）。
8. 已有 3 处**可以照抄的好样板**：`report/LOGITS_AUDIT.md:7-8`（两个主桶各带产物路径与判定阈值）、`答辩PPT_RepViT.pptx` slide 12/13（同页写明输入类型、n、阈值、复跑命令，并显式声明"与第 13 页真实图片实验分开"）、slide 13 备注（"结论仅覆盖这 12 张图片，不能写成完整测试集或全部六模型一致"）。**建议把 slide 13 备注的限定语前移到封面与状态页。**

---

## 1. 实验桶 → 权威落盘产物 → 应统一引用的展示值 → 复跑命令

（本表由 `python tools/audit_logits_references.py --table` 生成，与清单 CSV 同源）

| 实验桶 | 权威落盘产物 | 应统一引用的展示值 | 复跑命令 |
|---|---|---|---|
| `B1_onnx_pet37_n12`<br>PyTorch↔ONNX｜Pet-37 baseline｜n=12 张真实图片 | `outputs/metrics/consistency_repvit_m0_9_pet37.json` | max **6.199e-06** / mean **1.501e-06**；Top-1 与 Top-5 集合一致率均 **1.000**；verdict **PASS** | `python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --images datasets/lists/pet_test.txt --limit 12 --out outputs/verification/consistency_repvit_m0_9_pet37_n12.json` |
| `B2_onnx_m0_9_in1k_n12`<br>PyTorch↔ONNX｜M0.9 官方 ImageNetV2 子集｜n=12 张真实图片 | `outputs/metrics/consistency_repvit_m0_9_in1k.json` | max **1.383e-05** / mean **2.334e-06** | `python deploy/compare_torch_onnx.py --model repvit_m0_9_in1k --images datasets/lists/imagenetv2_mf_1000.txt --limit 12 --out outputs/verification/consistency_repvit_m0_9_in1k_n12.json` |
| `B3_onnx_m1_0_in1k_n12`<br>PyTorch↔ONNX｜M1.0 官方 ImageNetV2 子集｜n=12 张真实图片 | `outputs/metrics/consistency_repvit_m1_0_in1k.json` | max **1.335e-05** / mean **2.215e-06** | `python deploy/compare_torch_onnx.py --model repvit_m1_0_in1k --images datasets/lists/imagenetv2_mf_1000.txt --limit 12 --out outputs/verification/consistency_repvit_m1_0_in1k_n12.json` |
| `B12_onnx_m1_1_in1k_n12`<br>PyTorch↔ONNX｜M1.1 官方 ImageNetV2 子集｜n=12 张真实图片 | `outputs/metrics/consistency_repvit_m1_1_in1k.json` | max **1.860e-05** / mean **2.219e-06** | `python deploy/compare_torch_onnx.py --model repvit_m1_1_in1k --images datasets/lists/imagenetv2_mf_1000.txt --limit 12 --out outputs/verification/consistency_repvit_m1_1_in1k_n12.json` |
| `B13_onnx_m1_5_in1k_n12`<br>PyTorch↔ONNX｜M1.5 官方 ImageNetV2 子集｜n=12 张真实图片 | `outputs/metrics/consistency_repvit_m1_5_in1k.json` | max **1.144e-05** / mean **1.876e-06** | `python deploy/compare_torch_onnx.py --model repvit_m1_5_in1k --images datasets/lists/imagenetv2_mf_1000.txt --limit 12 --out outputs/verification/consistency_repvit_m1_5_in1k_n12.json` |
| `B4_reparam_pet37_32rand`<br>结构重参数化｜Pet-37 baseline｜32 个固定随机输入 seed=20240912 | `outputs/reparam/repvit_m0_9_pet37_reparam_report.json` | max **7.093e-06** / mean **2.031e-06**；Top-1 **32/32**；BN **107→0** | `python tools/reparam_verify.py --model repvit_m0_9_pet37 --weights checkpoints/baseline_best.pt --num-samples 32 --batch-size 8 --seed 20240912 --skip-onnx --out-dir outputs/verification/reparam_pet37` |
| `B5_official_fuse_probe`<br>结构重参数化｜五个官方 ImageNet 型号｜评价流程内单输入融合探针 | `outputs/pretrained_eval/*/metrics.json` 的 `fuse_max_abs_logits_diff` | M0.9 **3.073e-05**；M1.0 **3.290e-05**；M1.1 **2.217e-05**；M1.5 **2.587e-05**；M2.3 **5.925e-05** | `python tools/eval_pretrained.py --cfg configs/pretrained_eval.yaml` |
| `B6_export_probe`<br>导出检查｜单个随机输入前后向探针（导出脚本未固定 seed） | `outputs/benchmarks/export_*.json` | 按文件记录值；**不充当 n=12 真实图片一致性结果** | `python deploy/export_onnx.py --model repvit_m0_9_pet37` |
| `B7_reparam_legacy_invalid`<br>结构重参数化｜历史/无效口径记录（**不作为当前结论**） | `outputs/reparam/repvit_m0_9_reparam_report.json`（C=1000 官方权重装入 timm 结构，`n_missing=605`、`n_unexpected=713`）<br>`outputs/advanced/repvit_m0_9_pet37_reparam_report.json`（seed=0、2 个随机输入的历史结构探针） | **4.991888999938965e-07** / **2.9802322387695312e-06**（仅作历史记录） | （不提供复跑命令：产物对应无效权重或已废弃探针） |
| `B8_legacy_orphan`<br>孤值｜仓库内找不到配套产物，也没有任何提交能佐证其样本数 | **（无）** | 5.25e-06 / 1.41e-06、5.245e-06 / 1.414e-06、6.53e-06、6.527e-06、2.265e-06、4.108e-07 | （无可复跑产物：当前权重 n=8 / n=12 均得不到这些数值） |
| `B9_route_probe`<br>路线等价探针｜两条权重路线的特征 / 主头 / 蒸馏平均 logits 比较 | `outputs/metrics/probe_route.json` | **max=0**（同进程同权重路线对比，不是 ONNX 或融合实验） | `python tools/probe_pretrained_route.py --official-ckpt checkpoints/pretrained/repvit_m0_9_distill_300e.pth --out outputs/metrics/probe_route.json`（两参数均有默认值） |
| `B10_smoke_early`<br>早期冒烟与结构统计 | `outputs/metrics/smoke_phase0.json` | 按文件记录值 | `python tools/smoke_phase0.py` |

### 1.1 主桶（答辩正文只引用这 6 个）

- **B1 = 6.199e-06 / 1.501e-06**（n=12 真实图片，Pet-37，PyTorch↔ONNX）
- **B2 = 1.383e-05 / 2.334e-06**（n=12 真实图片，M0.9，ImageNetV2 固定子集）
- **B3 = 1.335e-05 / 2.215e-06**（n=12 真实图片，M1.0，ImageNetV2 固定子集）
- **B12 = 1.860e-05 / 2.219e-06**（n=12 真实图片，M1.1，ImageNetV2 固定子集）
- **B13 = 1.144e-05 / 1.876e-06**（n=12 真实图片，M1.5，ImageNetV2 固定子集）
- **B4 = 7.093e-06 / 2.031e-06**（32 个固定随机输入，Pet-37，结构重参数化）

主桶的原始值（写进 JSON 的完整精度，供需要"给出全精度"的场合使用）：

| 桶 | 原始字段 | 全精度值 |
|---|---|---|
| B1 | `max_abs_logits` / `mean_abs_logits` | 6.198883056640625e-06 / 1.5006899711048998e-06 |
| B2 | 同上 | 1.3828277587890625e-05 / 2.333735949378024e-06 |
| B3 | 同上 | 1.33514404296875e-05 / 2.2154212805010807e-06 |
| B12 | 同上 | 1.8596649169921875e-05 / 2.218950858908405e-06 |
| B13 | 同上 | 1.1444091796875e-05 / 1.8764107115506097e-06 |
| B4 | `max_abs_err` / `mean_abs_err` | 7.092952728271484e-06 / 2.030726818702533e-06 |

> 展示口径统一为 **4 位有效数字**（6.199e-06 / 1.501e-06 / 1.383e-05 / 2.334e-06 / 1.335e-05 / 2.215e-06 / 1.860e-05 / 2.219e-06 / 1.144e-05 / 1.876e-06 / 7.093e-06 / 2.031e-06）。
> （2026-09-17：官方型号一致性改按 ImageNetV2 固定子集；Buckets 表新增 B12/B13。—— 本表的展示值与本文件 §1 的规范值集合同源，改口径后必须整表刷新。）
> 不要再用 `7.1e-06` 这种 2 位写法（现出现在 `README.md:13`、`report/PPT_CONTENT.md:31,95`、`答辩PPT_RepViT.pptx` slide 1 para 19 / slide 2 para 37、`paper` 封面 SVG `01_cover.svg:40`、`02_status.svg:56`）。

### 1.2 独立桶（保留，但不得与主桶合并成一句结论）

- **B5**（官方 5 型号融合探针）：型号/输入/设备各不同，只写"官方权重评价流程内的融合前后探针"。
- **B6**（导出探针）：随机输入未固定 seed，**不能**当一致性结果。
- **B9**（路线探针）：`max|Δlogits| = 0` 描述的是"官方 .pth + 官方实现"与"timm 权重 + timm 实现"两条路线在同进程内的等价，不是 ONNX 或融合实验。
- **B10**：早期冒烟，只在注明阶段时引用。

### 1.3 不得作为结论的桶

- **B7**：产物存在但前提无效（官方 C=1000 checkpoint 装进 timm 结构，键名交集为 0 / `n_missing=605`；或 seed=0、仅 2 个随机输入、BN 108 与当前 107 不一致）。可在"我们排除了什么"里提及，必须同时写出无效前提。
- **B8**：连产物都没有，只能出现在"已废弃的旧值"说明里。

---

## 2. 审计方法与覆盖面（可复核）

### 2.1 扫描范围

`python tools/audit_logits_references.py` 扫描 **全部 Git 跟踪文件**（`git ls-files`），逐个文件产出 `location`：

| 载体 | 位置粒度 | 说明 |
|---|---|---|
| `.md/.json/.jsonl/.csv/.log/.txt/.py/.sh/.svg/.yaml` | 行号 | 文本/代码/数据/日志全量 |
| `.pptx` | `slide N para M` / `notes N for slide M para K` / `chart N on slide M data cache` | 正文、**备注**、原生图表数值缓存、图表内嵌 xlsx 数据缓存都扫 |
| `.docx` | `word/document.xml para M`（含 header/footer） | 逐段 |
| `.pdf` | `page N` | PyMuPDF 逐页提取文本 |
| 不扫 | — | 二进制 `.pt/.onnx/.npy/图片`、以及清单自身 |

结论：**4791 条引用 / 179 个文件**（2026-09-17 t42 暂存态实测；t35 为 4790 / 178、t32 与 t28 均为 4788 / 178、t23 为 4778 / 177、t21 为 4776 / 177、t20 为 4731 / 176，再早为 4124 / 153）；`kind` 分布 = `noise_float` 2775、`statement` 1218、`error_value` 798；`flag` 命中 673 条（`MIXED` 443 / 纯 247、`ORPHAN` 275 / 纯 76、`OVERCLAIM` 156 / 纯 46、`SELF_TOOL` 95 / 纯 77、`AMBIGUOUS` 6 / 纯 6、`ABS_PATH` 4 / 纯 4；同一行可带多个标记，故计数相加大于 673）。

### 2.2 分级规则（为什么可以放心按桶引用）

1. 先把权威产物的数值读成"规范值集合"（B1–B7、B12、B13 + B8 的 8 个孤值字面量）。
2. 对每个浮点 token，按**它自己写出的有效位数**判断是否等于某个规范值（例：`6.199e-06` 与 `6.198883056640625e-06` 匹配；`6.527e-06` 不匹配 B4 的 7.092952728271484e-06，落到 B8）。
3. 匹配不上规范值的 token，若行内有误差上下文（`logits` / `误差` / `abs_err` / `Δ` …）才当引用，否则标 `noise_float`。
4. 纯数值日志（`outputs/logs/*`、`outputs/metrics/*.jsonl`、`*.csv`）额外要求 token ≥4 位有效数字或无误差上下文 —— 否则学习率会被误判（见 D8）。
5. `MIXED` 标记 = 同一行出现多个不同桶的**数值**；`OVERCLAIM` = 命中 `六个/6 个 ONNX|六个模型|全部六|完全等价|九类` 等模式。

`statement` 行（没有可归因数值的口径陈述）按上下文关键词就近归桶，属于辅助信息；**数值行的桶是权威的**。

### 2.3 可重复性

```bash
python tools/audit_logits_references.py          # 重新生成 outputs/verification/logits_reference_inventory.csv
python tools/audit_logits_references.py --table  # 额外打印第 1 节的 Markdown 表
python tools/audit_logits_references.py --stdout # 只统计不写文件
```

连续两次运行的 CSV SHA256 相同（`1BE7460A…D271`，见第 6 节复核记录），无时间戳、无随机性。（历史值：t35 轮为 `E1F708BF…1B3B`、t32 轮为 `3E46CD7B…189D`、t28 轮为 `6509CC6E…68B2`、t23 轮为 `9B4F9317…5678`、t21 轮为 `650A522C…60E3`、t20 轮为 `8EABE27A…D24C94`，更早为 `2074F146…E33B`。）

### 2.4 本次独立复跑（三条证据，写在 `%TEMP%`，未动仓库产物）

用当前权重 + 当前 ONNX + 当前数据，在本机复跑：

| 命令 | 结果 | 与权威产物 |
|---|---|---|
| `python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --limit 12 --out <temp>` | max **6.199e-06** / mean **1.501e-06** / Top-1、Top-5 均 100% / PASS | **完全一致（B1）** |
| `python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --limit 8 --out <temp>` | max **6.198883056640625e-06** / mean **1.4658262017519519e-06** / 100% / PASS | 不能复现 5.25e-06 |
| `python tools/reparam_verify.py --model repvit_m0_9_pet37 --weights checkpoints/baseline_best.pt --num-samples 32 --batch-size 8 --seed 20240912 --skip-onnx --out-dir <temp>` | max **7.092952728271484e-06** / mean **2.030726818702533e-06** / Top-1 32/32 / BN **107→0** | **完全一致（B4）** |

值得注意的细节：`--limit 8` 与 `--limit 12` 的 **max 相同（6.198883056640625e-06）**，只有 mean 不同 —— 说明最大误差落在前 8 张里；"n=8 / n=12" 的区别并不会改变 max 的展示值，但两者都不是 5.25e-06。

---

### 2.5 t20 终局刷新：逐文件增量台账（2026-09-17，暂存态）

> **本节是 t20 轮次的历史台账**（4124 / 153 → 4731 / 176）。此后 t13 交付流程图又带来 **+45 / +1**，见 **§2.6**（该轮登记值 4776 / 177；最新见 §2.9）。

本轮把统计基线从 **4124 条 / 153 个文件** 刷新为 **4731 条 / 176 个文件**。总增量 **+607 条 / +23 个文件**，三段对账：

| 段 | 条数 | 文件数 |
|---|---|---|
| 新增进入统计的文件 | **+520** | 26 个 |
| 退出统计的文件（条目归零） | **−14** | 3 个 |
| 既有文件净变化 | **+101** | 22 个文件发生变化 |
| **合计** | **+607** | **+23** |

**A. 新增进入统计的文件（26 个，逐项列出）**

| 条数 | 文件 |
|---|---|
| +155 | `report/REQUIREMENTS_AUDIT.md` |
| +84 | `outputs/predictions/repvit_m1_5_in1k_beagle__ILSVRC2012_val_00000162_00_probs.json` |
| +68 | `outputs/advanced/manual_fusion_alignment.json` |
| +49 | `outputs/predictions/repvit_m1_0_in1k_beagle__ILSVRC2012_val_00000162_00_probs.json` |
| +41 | `outputs/predictions/repvit_m0_9_in1k_beagle__ILSVRC2012_val_00000162_00_probs.json` |
| +29 | `outputs/predictions/repvit_m1_1_in1k_beagle__ILSVRC2012_val_00000162_00_probs.json` |
| +15 | `tools/manual_fusion_align.py` |
| +12 | `outputs/metrics/onnx_io_nodes.json` |
| +12 | `report/SPEC13_FREEZE.json` |
| +6 | `outputs/metrics/bench.jsonl`、`outputs/report_assets/table_benchmark_meta.csv`、`outputs/report_assets/table_benchmark_meta.md`（各 +6） |
| +5 | `outputs/metrics/consistency_repvit_m1_1_in1k.json`、`consistency_repvit_m1_5_in1k.json`、`outputs/verification/consistency_repvit_m1_1_in1k_n12.json`、`consistency_repvit_m1_5_in1k_n12.json`（各 +5） |
| +4 | `report/SPEC13_SWITCH_RUNBOOK.md` |
| +3 | `deploy/model_registry.py` |
| +2 | `report/IMAGENETV2_PROVENANCE.md`、`tools/robustness_gradcam.py`（各 +2） |
| +1 | 六份 `outputs/benchmarks/repvit_*_benchmark.json`（各 +1） |

**B. 退出统计的文件（3 个）**

| 条数 | 文件 | 原因 |
|---|---|---|
| −10 | `outputs/logs/eval_family_5models.log` | 该日志按现行流程重写，不再含可归桶的数值行 |
| −2 | `outputs/report_assets/table_figures.csv` | 素材表改版，原数值行移除 |
| −2 | `outputs/report_assets/table_figures.md` | 同上 |

**C. 既有文件净变化（±1 及以上，共 22 个文件，合计 +101）**

| 变化 | 文件 | 变化 | 文件 |
|---|---|---|---|
| +28 | `report/LOGITS_AUDIT_FINDINGS.md` | +7 | `outputs/pretrained_eval/repvit_m1_1/top5_samples.json` |
| +15 | `tools/update_defense.py` | +6 | `outputs/REPORT.md` |
| +13 | `report/LOGITS_AUDIT.md` | +5 | `README.md` |
| +11 | `report/VERIFICATION_LOGITS_PPT.md` | −5 | `report/REPORT.docx`、`report/REPORT.md` |
| +10 | `tools/audit_logits_references.py` | +3 | `outputs/pretrained_eval/repvit_m0_9/top5_samples.json` |
| −8 | `PROGRESS.md` | −3 | `outputs/pretrained_eval/repvit_m2_3/top5_samples.json` |
| +8 | `report/PPT_CONTENT.md`、`report/答辩PPT_RepViT.pdf`、`report/答辩PPT_RepViT.pptx`（各 +8） | −2 | `report.pdf`；+2 `report/ppt_svg/15_summary.svg`；+1 `deploy/run_all.sh`；+1 `tools/selfcheck.py`；−1 `outputs/pretrained_eval/repvit_m1_0/top5_samples.json`；−1 `tools/report_spec.py` |

**归因（谁改的）**：`report/REQUIREMENTS_AUDIT.md`（+155）来自 t2/t10 的要求审计文档，本轮首次随暂存进入统计；`*_probs.json`（+203）与 `manual_fusion_alignment.json`（+68）、`tools/manual_fusion_align.py`、`robustness_gradcam` 来自 t12/t17 的复评与加分项；`SPEC13_FREEZE/SPEC13_SWITCH_RUNBOOK/IMAGENETV2_PROVENANCE`（+18）来自 t11/t19 的口径冻结；本体 +28 与本轮就地刷新有关；`REPORT/PPT/PDF` 的增减来自 t4/t9/t18/t19 的重生成。

---

### 2.6 t21 二次刷新：新增流程图的增量（2026-09-17，新暂存态）

> 本节是 **t21 轮次** 的台账（4776 / 177）。此后 t22 又带来 **+2 / +0**，见 **§2.7**；t25/t27 又带来 **+10 / +1**，见 **§2.8**；t30 之后条数不变、校验值刷新（§2.9）；t34 又带来 **+2 / +0**（§2.10）；t41 又带来 **+1 / +1**，现行登记值见 **§2.11（4791 / 179）**。

**为什么还要刷新**：t20 的登记前提是「此后不再新增文件」。此后 **t13 交付了两份仓库内流程图**（`docs/workflow_zh.html` 686,265 B 与 `docs/workflow_zh.svg` 249,585 B，**已 `git add` 暂存、未 commit**），必须在新暂存态再登记一次。

**结果：4776 条 / 177 个文件**（t20：4731 / 176）。总增量 **+45 条 / +1 个文件**，逐来源对账：

| 来源 | 条数 | 文件数 | 明细 |
|---|---|---|---|
| 新增 `docs/workflow_zh.svg` | **+43** | +1 | 43 条 = `statement` 38 + `error_value` 4 + `noise_float` 1；按桶 = B1 **21** + B4 **19** + 口径陈述类 2 + 噪音 1。原因是该 SVG 的文本层含「结构重参数化」「一致性」「ONNX 部署」等节点标签与 aria-label |
| 新增 `docs/workflow_zh.html` | **0** | 0 | **该后缀不在脚本的文本后缀白名单内**（`.md/.json/.jsonl/.csv/.log/.txt/.py/.sh/.svg/.yaml/.yml`）—— 即使 686 KB、含大量数值字面量，也**一条都不计** |
| 本文件新增的 §2.6 台账自身 | **+2** | 0 | **自指项**：新写进本文件的两行属可扫文本，故本节的表会把自身计入；这是本轮唯一「因写文档而涨」的部分 |
| 其它文件（含本轮一并暂存的自检报告） | **0** | 0 | 与 t20 逐文件相减无差异 |
| **合计** | **+45** | **+1** | 4731 → 4776 |

> 这条对账同时纠正一个直觉误判：**增量只由「后缀在白名单内」的新文件产生**，文件体积与数值密度不决定是否计入。

**本轮核对（沿用 §2.4 的分类框架）**：悬空权威 **0**；归不到具体桶的数值引用 **5**（与 t20 同一批，全为工具误报）；歧义行 **6**（全部在 `--dump-probs` 的概率矩阵里，工具误报）；无效前提桶 16 / 无产物孤值桶 165（设计内）；`ABS_PATH` 4（引用检测器自身字面量，工具误报）。

**附录 B 的连带变化**：B4 桶的 `error_value` 行由 210 增至 **214**（新增的 4 条来自 `docs/workflow_zh.svg`，未进 Top-7 引用方）；其余桶不变。

**严格幂等**：连跑两次，条数与 CSV 逐字节相同（登记值只写在脚本扫不到的行里）。

---

### 2.7 t23 第三次登记：t22 修复后的增量（2026-09-17，暂存态）

本轮（t23）只登记 t22 修复后的增量：**+2 条 / +0 个文件**（4776 → **4778**）。逐来源对账：

| 来源 | 条数 | 文件数 | 说明 |
|---|---|---|---|
| `report/REQUIREMENTS_AUDIT.md` | **+1** | 0 | 该文件 155 → **156** 行；t22 的 F2 在「1.2-8 为什么可以减少推理开销」行补写了现行证据与判定复核后记，新增 1 处命中 |
| 本节（§2.7）新增的台账文本自身 | **+1** | 0 | **自指项**：写进本文件的两行属可扫文本 |
| `report/ppt_svg/14_perf.svg` | 0 | 0 | 内容确有两行变化（+0.11→+0.08、+0.13→+0.14），但这些值不落进脚本的取值模式，条目数不变（0 → 0） |
| `report/ppt_svg/README.md` | 0 | 0 | 同上（0 → 0） |
| `report/SPEC13_FREEZE.json` | 0 | 0 | 12 → 12（片段定位改为内容锚点，条目数不变） |
| `report/SPEC13_SWITCH_RUNBOOK.md` | 0 | 0 | 4 → 4（只把冻结基线版本号由 3 更正为 4） |
| 其余全部文件 | 0 | 0 | 逐文件相减无差异 |
| **合计** | **+2** | **0** | 4776 → 4778 |

**本轮核对**（沿用 §2.4 的分类框架）：悬空权威 **0**；归不到具体桶的数值引用 **5**（同一批工具误报）；歧义行 **6**（工具误报）；无效前提桶 16 / 无产物孤值桶 165（设计内）；`ABS_PATH` 4（引用检测器自身字面量，工具误报）。附录 B 的按桶清点**无变化**（九个桶的 `error_value` 行数与 t21 相同）。

**严格幂等**：连跑两次，条数与 CSV 逐字节相同。

---

### 2.8 t28 第四次登记：t25 + t27 修复后的增量（2026-09-17，暂存态）

> 本节是 **t28 轮次** 的台账（4778 → 4788）。t30 之后条数不变、校验值刷新，现行登记值见 **§2.9**。

本轮（t28）只登记 t25（repair r2）与 t27（类别级修复）之后的增量：**+10 条 / +1 个文件**（4778 → **4788**）。逐来源对账：

| 来源 | 条数 | 文件数 | 说明 |
|---|---|---|---|
| 新增 `tools/check_audit_evidence.py` | **+10** | **+1** | t27 新增的证据指针校验器；10 条全部为 `statement`（方法或口径说明，不指向具体数值），按桶全归 B11 |
| `report/REQUIREMENTS_AUDIT.md` | 0 | 0 | 156 → 156（净 0）；行号与匹配文本有 121 条换位：B1 39→38、B4 16→19、B11 49→46、噪音 0→1，四者相抵 |
| `report/SPEC13_SWITCH_RUNBOOK.md` | 0 | 0 | 4 → 4（4 条换位，计数不变） |
| `tools/selfcheck.py` | 0 | 0 | 11 → 11（9 条换位，计数不变） |
| `report/ppt_svg/README.md` | 0 | 0 | 0 → 0 |
| 其余全部文件 | 0 | 0 | 逐文件相减无差异 |
| **合计** | **+10** | **+1** | 4778 → 4788 |

**本轮核对**（沿用 §2.4 的分类框架）：悬空权威 **0**；归不到具体桶的数值引用 **5**（同一批工具误报）；歧义行 **6**（工具误报）；无效前提桶 16 / 无产物孤值桶 165（设计内）；`ABS_PATH` 4（引用检测器自身字面量，工具误报）。附录 B 的按桶清点**无变化**（各桶的 `error_value` 行数与 t23 相同）。

**严格幂等**：连跑两次，条数与 CSV 逐字节相同（第三次 `--stdout` 后哈希不变）。

---

### 2.9 t32 第五次登记：t30 修复后的校验值刷新（2026-09-17，暂存态）

> 本节是 **t32 轮次** 的台账（4788 / 178，校验值刷新）。此后 t34 又带来 **+2 / +0**，见 **§2.10**；t41 又带来 **+1 / +1**，现行登记值见 **§2.11（4791 / 179）**。

本轮（t32）登记 t30（repair r3）之后的暂存态：**条数 / 文件数 +0，但清单正文与校验值已变**（4,789 行；1,245,560 → **1,245,757 B**；`6509CC6E…68B2` → **`3E46CD7B…189D`**）。逐来源对账：

| 来源 | 条数 | 文件数 | 说明 |
|---|---|---|---|
| `report/REQUIREMENTS_AUDIT.md` | 0 | 0 | 156 → 156；1 条 `statement` 行的归属由 B2 改为 B3（B2 4→3、B3 0→1），另有多处文本换位（t30 的计数更正、判定复核后记、体积与行数刷新），故清单正文随之一变 |
| `tools/check_audit_evidence.py` | 0 | 0 | 10 → 10（校验器扩到「可比对」维度，只改判定逻辑与断言表） |
| `tools/selfcheck.py` | 0 | 0 | 11 → 11（并入既有检查项，未新增 check id） |
| 其余全部文件 | 0 | 0 | 逐文件相减无差异 |
| **合计** | **0** | **0** | 4788 → 4788 |

**本轮核对**（沿用 §2.4 的分类框架）：悬空权威 **0**；归不到具体桶的数值引用 **5**（同一批工具误报）；歧义行 **6**（工具误报）；无效前提桶 16 / 无产物孤值桶 165（设计内）；`ABS_PATH` 4（引用检测器自身字面量，工具误报）。附录 B 的按桶清点**无变化**。

**严格幂等**：连跑两次，条数与 CSV 逐字节相同（第三次 `--stdout` 后哈希不变）。

---

### 2.10 t35 第六次登记：t34 末轮修复后的增量（2026-09-17，暂存态）

> 本节是 **t35 轮次** 的台账（4788 → 4790）。此后 t41 又带来 **+1 / +1**，现行登记值见 **§2.11**。

本轮（t35）登记 t34（末轮修复）之后的暂存态：**+2 条 / +0 个文件**（4788 → **4790**），清单 4,791 行；校验值 1,245,757 → **1,246,586 B**、`3E46CD7B…189D` → **`E1F708BF…1B3B`**。逐来源对账：

| 来源 | 条数 | 文件数 | 说明 |
|---|---|---|---|
| `report/REQUIREMENTS_AUDIT.md` | **+2** | 0 | 156 → 158：t34 新增的一行（L722）含两处被扫词（同一关键词的大小写各一处），两处各记 1 条，按桶全归 B1 |
| `tools/check_audit_evidence.py` | 0 | 0 | 10 → 10（新增两条断言只改判定逻辑与断言表，未增条目） |
| `tools/selfcheck.py` | 0 | 0 | 11 → 11 |
| 其余全部文件 | 0 | 0 | 逐文件相减无差异 |
| **合计** | **+2** | **0** | 4788 → 4790 |

**本轮核对**（沿用 §2.4 的分类框架）：悬空权威 **0**；归不到具体桶的数值引用 **5**（同一批工具误报）；歧义行 **6**（工具误报）；无效前提桶 16 / 无产物孤值桶 165（设计内）；`ABS_PATH` 4（引用检测器自身字面量，工具误报）。附录 B 的按桶清点**无变化**。

**严格幂等**：连跑两次，条数与 CSV 逐字节相同（第三次 `--stdout` 后哈希不变）。

---

### 2.11 t42 第七次登记：t41 修复后的增量（2026-09-17，暂存态）

本轮（t42）登记 t41（P14 右图空白修复 + 全片图表审计 + 防复发断言）之后的暂存态：**+1 条 / +1 个文件**（4790 → **4791**），清单 4,792 行；校验值 1,246,586 → **1,246,850 B**、`E1F708BF…1B3B` → **`1BE7460A…D271`**。逐来源对账：

| 来源 | 条数 | 文件数 | 说明 |
|---|---|---|---|
| `report/SYNC_NOTES.md` | **+1** | **+1** | 该文件此前 0 条；首次命中来自 §14.2 自指声明自身那一行（句子里写到审计脚本的可执行路径）—— 即 §14.2 预告的「回填行」，本轮并入新值，此后不再有悬空差额 |
| `tools/update_defense.py` | 0 | 0 | 114 → 114；106 行内容变化（轴范围改现读、措辞、页脚命令） |
| `tools/selfcheck.py` | 0 | 0 | 11 → 11；11 行内容变化（并入「数据须落在轴范围内」断言） |
| `report/PPT_CONTENT.md` / `report/答辩PPT_RepViT.pptx` | 0 | 0 | 67 → 67 / 95 → 95；各 1 行内容变化 |
| 其余全部文件 | 0 | 0 | 逐文件相减无差异（含 `report/答辩PPT_RepViT.pdf` 与 `report/REQUIREMENTS_AUDIT.md`：被扫行内容不变） |
| **合计** | **+1** | **+1** | 4790 → 4791（被扫行内容变化合计 119 行 / 4 个文件） |

**本轮核对**（沿用 §2.4 的分类框架）：悬空权威 **0**；归不到具体桶的数值引用 **5**（同一批工具误报）；歧义行 **6**（工具误报）；无效前提桶 16 / 无产物孤值桶 165（设计内）；`ABS_PATH` 4（引用检测器自身字面量，工具误报）。附录 B 的按桶清点**无变化**。

**严格幂等**：连跑两次，条数与 CSV 逐字节相同（第三次 `--stdout` 后哈希不变）。

---

## 3. 缺陷清单（逐条：文件 + 位置 + 问题 + 建议改法）

### D1【口径自相矛盾 · 必须收敛】5.25e-06 到底是不是"旧的 n=8 对照"

- 位置 A：`report/REPORT.md:796-797`
  `固定 datasets/lists/pet_test.txt 的前 12 张真实图片（n=12）。README 早期出现的 5.25e-06 属于旧的 n=8 对照记录，不能与本次落盘结果合并引用。`
- 位置 B：`report/LOGITS_AUDIT.md:15`
  `README 的 5.25e-06 / 1.41e-06，以及旧 PPT_CONTENT 的 5.245e-06 / 1.414e-06，没有找到配套完整实验产物。旧命令写了 limit=8，但仅凭这一点不能证明数值来自 n=8。当前权重跑 n=8 得到 max=6.198883056640625e-06，不能复现旧值。`

**问题**：同一件事两个口径，A 断言"属于 n=8"，B 说"不能证明来自 n=8"。

**三条独立证据**（可复核）：

1. **逐提交取证**：对全部 14 个提交的 `outputs/` 树执行 `git grep -l <literal> <commit> -- outputs/`，`5.25e-06`、`1.41e-06`、`5.245e-06`、`1.414e-06`、`6.53e-06`、`6.527e-06`、`2.265e-06`、`4.108e-07` **从未出现在任何落盘产物里**；同一方法对 `6.198883056640625e-06` 能准确定位到 `outputs/metrics/consistency_repvit_m0_9_pet37.json`（证明方法有效，不是"没搜到"）。
2. **同一次提交里两个口径就已经并存**：初始交付提交 `81693dd`（2026-09-13）的 `README.md:342` 写着 `max|Δlogits| = 5.25e-06、mean|Δlogits| = 1.41e-06`，而**同一提交**的 `outputs/metrics/consistency_repvit_m0_9_pet37.json` 已经是 `"n": 12, "max_abs_logits": 6.198883056640625e-06`。也就是说仓库里**从来没有**一个能与 5.25e-06 对应的 n=8 产物；"n=8"的唯一线索是旧 README 自己的命令行 `--limit 8`，而它与自己同一次提交的落盘 JSON 都不一致。
3. **今天复跑 n=8 也复现不了**：`--limit 8` → max **6.198883056640625e-06** / mean 1.4658262017519519e-06（§2.4）。

**建议改法**（`report/REPORT.md:796-797` 用下面这段替换；`report/LOGITS_AUDIT.md:15` 已基本正确，可把最后一句补上"同一提交已落盘 n=12 的 JSON"）：

> 固定 `datasets/lists/pet_test.txt` 的前 12 张真实图片（`n=12`）。README / PPT_CONTENT 早期出现的
> `5.25e-06` / `1.41e-06`（以及 `5.245e-06` / `1.414e-06`）在**仓库的任何提交里都没有配套产物**：
> 旧 README 的复跑命令写的是 `--limit 8`，但同一提交里落盘的 JSON 已经是 `n=12` 的 `6.199e-06`，
> 因此**既不能证明它来自 n=8，也不能把它当作 n=12 的结果**。用当前权重跑 `--limit 8` 得到
> `max=6.198883056640625e-06`、`mean=1.4658262017519519e-06`，同样无法复现旧值。旧值只作为修订记录保留。

### D2【硬写绝对路径】三个一致性 JSON 的路径字段

- 位置：
  - `outputs/metrics/consistency_repvit_m0_9_pet37.json:4`（`onnx_path`）与 `:8`（`images`）
  - `outputs/metrics/consistency_repvit_m0_9_in1k.json:4`、`:8`
  - `outputs/metrics/consistency_repvit_m1_0_in1k.json:4`、`:8`
  - 内容形如 `"C:\\Users\\14675\\Desktop\\one团队秋招ai算法\\RepViT-Reproduction\\onnx\\repvit_m0_9_pet37.onnx"`
- 来源：`deploy/compare_torch_onnx.py:95-97`
  ```python
  rep = dict(model=a.model, arch=reg["name"], onnx_path=onnx_path(a.model),
             num_classes=reg["num_classes"], label_file=_label_file, source=_source,
             images=str(list_file), n=len(paths), ...)
  ```
  `deploy/model_registry.py:76-78` 的 `onnx_path()` 返回 `str(ONNX_DIR / key)`，本身就是绝对路径；`list_file` 来自命令行，本次传的是绝对路径。
- **需要修正任务描述里的一处假设**：同目录副本 `outputs/verification/consistency_repvit_m0_9_pet37_n12.json` 只有 `images`（第 8 行）是相对路径（因为复跑时传了相对 `--images datasets/lists/pet_test.txt`），**`onnx_path`（第 4 行）依然是绝对路径**。所以"副本用相对路径"只对一半成立。
- 附带核实：**源码层面没有写死个人绝对路径**（符合试题第 14 页）—— 对 `*.py/*.sh/*.md/*.yaml/*.svg` 全量 grep `14675` / `C:\Users`，唯一命中是审计脚本自己的检测正则；绝对路径全部落在 `outputs/**`（含 `outputs/report_assets/table_training_summary.md` 在内共 50 个文件，例如 `outputs/metrics/same_budget.json:11,23,24,30,42`、`outputs/metrics/backbone_updated.json:14`、`outputs/env_snapshot.json:3`、`outputs/metrics/imagenet_labels_report.json:2` 甚至指到 `C:\Users\14675\AppData\Roaming\Python\...`）。
- **建议改法**（按优先级）：
  1. `deploy/compare_torch_onnx.py:95-97` 改为写仓库相对路径，例如
     `onnx_path=Path(onnx_path(a.model)).relative_to(ROOT).as_posix()`、
     `images=Path(list_file).resolve().relative_to(ROOT).as_posix()`。这样以后复跑不再产生绝对路径。
  2. 三个 JSON 与副本的 `onnx_path` / `images` 两个**非数值字段**改成相对路径（数值字段一律不动）。这属于"格式修"而非"数值修"，但动了 `outputs/` 下已提交文件，**需要用户明确批准后再做**。
  3. 若用户不同意动产物：至少在 `report/LOGITS_AUDIT.md` 里加一行"这三个 JSON 的 `onnx_path`/`images` 记录的是采集机绝对路径，跨机复现请按 `deploy/model_registry.py` 的 `onnx_path(key)` 重新生成"。

### D3【混写一：重参数化误差与 ONNX 一致性并列成一句】

- `README.md:12-14`（一句话结论）
  `结构重参数化前后 logits max|Δ| = 7.09e-06；六个 ONNX 模型在 ONNX Runtime CPU 上批量 1 / FP32 / 224×224 的推理延迟为 7.1 ~ 32.7 ms，PyTorch↔ONNX Top-1 一致率 100%。`
- `report/答辩PPT_RepViT.pptx` slide 1 para 19-22（封面）与 `report/答辩PPT_RepViT.pdf` 第 1 页
  `7.1e-06` → `重参数化 max|Δlogits|` → `100%` → `PyTorch↔ONNX Top-1 一致`（四块数值紧挨着排）
- `report/答辩PPT_RepViT.pptx` slide 2 para 34-37 与同页 `06 ONNX 多模型部署` 卡（`report/答辩PPT_RepViT.pdf` 第 2 页）
  `05 结构重参数化 …… max|Δ| 7.1e-06 · Top-1 32/32` 与 `06 ONNX 多模型部署 6 个 ONNX · ORT CPU P50 7.1 ~ 32.7 ms` 同页
- `report/PPT_CONTENT.md:31,33,35,37` 是上面封面的文字源（`7.1e-06` / `重参数化 max|Δlogits|` / `100%` / `PyTorch↔ONNX Top-1 一致`）

**问题**：重参数化误差（B4，32 个随机输入）与 ONNX 一致性（B1，12 张真实图片）是不同实验，紧邻排布会被读成"同一批实验的两个指标"。

**建议改法**：
- `README.md:12-14` 拆成两句并各自带口径：
  `结构重参数化（32 个固定随机输入）前后 logits max|Δ| = 7.093e-06（<1e-4 阈值，Top-1 32/32 一致）；PyTorch↔ONNX 一致性（Pet-37 前 12 张真实图片，n=12）max|Δlogits| = 6.199e-06，Top-1 与 Top-5 集合均 100%。六个 ONNX 模型在 ONNX Runtime CPU 上批量 1 / FP32 / 224×224 的 P50 延迟 7.1 ~ 32.7 ms。`
- 封面/状态页：把 `7.1e-06` 改成 `7.093e-06`，并在两个格子分别加限定词 —— `结构重参数化（32 个随机输入）max|Δ| 7.093e-06 · Top-1 32/32` 与 `ONNX 部署（一致性见第 13 页，3 份 n=12 落盘）`；把 slide 13 备注里那句 **"结论仅覆盖这 12 张图片，不能写成完整测试集或全部六模型一致"** 挪一份到封面/状态页。
- `report/PPT_CONTENT.md:31,95` 同步改成 `7.093e-06`。

### D4【混写二：把单型号一致率外推成"六个模型一致"；"九类问题"被写成已排除】

- `report/ppt_svg/13_onnx.svg:5`：`6 个 ONNX 模型，PyTorch↔ONNX Top-1 一致率 100%`
- `report/ppt_svg/15_summary.svg:45`：`③ 重参数化数值等价、BN 归零；ONNX 一致率 100%`
- `report/答辩PPT_RepViT.pptx` slide 2（`6 个 ONNX`）、`report/答辩PPT_RepViT.pdf` 第 2 页
- `report/REPORT.md:807`、`report/REPORT.docx`（`word/document.xml para 913`）、`report.pdf` 第 23 页（审计当时页码）：
  `一致率为 100%，说明没有出现题目列出的九类典型问题（Resize/CenterCrop 顺序、RGB/BGR 通道、插值方式、mean/std、Softmax 维度、eval 模式、BN 状态、导出方式、数值精度）。`
- **现状（2026-09-16 复核，PyMuPDF 实测现行 26 页 PDF）**：上面那句在压缩后的 `report/REPORT.md`（849 行）与现行 PDF 中都已 **0 命中**（t22 已按本节建议改写）；现行 PDF 里「九类」只剩**第 20 页**两处（「未观察到题目列出的九类典型问题的表现」与「不等于『已排除九类问题』」），均为修正后的限定式措辞。上面写的「第 23 页」是审计当时（压缩前 37 页版）的定位；对被压缩掉的行号（原 `:807`）同样只作历史记录。

**事实**：仓库里同口径的一致性 JSON 只有 3 份（B1/B2/B3），都是 n=12；另外 3 个型号只有导出检查与性能结果，**不能**代替一致性验证（`report/LOGITS_AUDIT.md:18` 已经写对了）。"九类问题"只能说明"本次 12 张图上没出现 Top-1/Top-5 不一致"，不构成对九类问题的排除证明。

**建议改法**：
- 把"六个 ONNX 模型一致率 100%" 统一改成：
  `PyTorch↔ONNX Top-1 一致率：Pet-37 / ImageNet M0.9 / ImageNet M1.0 三个型号各 n=12 均为 100%（来源见下表）；其余型号只有导出与性能结果，不做同口径一致性声明。`
- 九类那句改成：
  `在这 12 张图上 Top-1 与 Top-5 集合均一致，未观察到题目所述九类典型问题的表现；该结论只覆盖已测输入，不等于已排除九类问题。`

### D5【超证据断言：结构重参数化"完全等价"】

- `report/ppt_svg/12_reparam.svg:5`：`重参数化在数值上完全等价，BN 节点归零`
- `report/REPORT.md:963`、`report/REPORT.docx`（`word/document.xml para 1053`）、`report.pdf` 第 27 页（审计当时页码）：
  `结构重参数化在数值上完全等价（max|Δ| = 7.09e-06，Top-1 32/32 一致）`
- 另外 `tools/update_defense.py:136-137` 已经把这两个短语列入替换表（`"重参数化在数值上完全等价" → "重参数化误差小于阈值"`），但**只有 PPTX 被替换过**，SVG / REPORT.md / DOCX / PDF 仍是旧词。
- **现状（2026-09-16 复核，PyMuPDF 实测现行 26 页 PDF）**：「完全等价」在压缩后的 `report/REPORT.md`（839 行）与现行 PDF 中都已 **0 命中**（t22 已改写为阈值口径）；旧措辞目前只剩 `report/ppt_svg/12_reparam.svg:5` 一处，属 `LOGITS_AUDIT.md:47` 已声明的历史素材。上面写的「第 27 页 / 原 `:963`」是审计当时的定位。

**建议改法**：把"完全等价"全部改成"代数上等价，实测误差小于阈值"：
`分支合并是代数等价的替换；实测 32 个固定随机输入 max|Δlogits| = 7.093e-06（< 1e-4 阈值），Top-1 32/32 一致。该结论只覆盖已测输入与设定阈值，不是位级完全相等。`
（PPTX slide 12 para 20 与备注 notes 4 已经是这个措辞，可直接复用。）

### D6【导出物不同步：`report.pdf` 仍是初始版表格】

- 位置（审计当时）：`report.pdf` 第 21 页（12.4 实测数值对照表）；**该页内容现已修正**，现状见本节末尾的「现状」条
  现值：`max_abs_err 7.093e-06 | 2.265e-06`、`mean_abs_err 2.031e-06 | 4.108e-07`、`BN 模块数 108 → 0 | 108 → 0`
- 对照：`report/REPORT.md:720-728` 现值：`| 7.092953e-06（展示 7.093e-06） | —（旧报告权重键不匹配，不作结论） |`、`| BN 模块数 | 107 → 0 | N/A |`
- 证据：`git show 81693dd:report/REPORT.md` 的 12.4 表正是 `2.265e-06 / 4.108e-07 / BN 108 → 0`。也就是说 `report.pdf` 是**初始交付版 REPORT.md 的导出**，此后 REPORT.md 已修，PDF 未重导出。
- `report/REPORT.docx` 情况好一些（`word/document.xml para 812-813` 已是 `max_abs_err / 7.093e-06`），但仍带 `para 913`（九类）与 `para 1053`（完全等价），需与 REPORT.md 一起改后重导出。

**建议改法**：t2 完成 REPORT.md 修改后，**必须重新导出 `report.pdf` 与 `report/REPORT.docx`**，并核对第 21/23/27 页与 REPORT.md 的 12.4 / 13.3 / 16 节逐字一致。否则"REPORT.md 已改、PDF 未改"会重新制造同类缺陷。

**现状（2026-09-16 复核，PyMuPDF 实测）**：该建议已落实 —— `report.pdf` 于 t17 重导、t22 随报告压缩再导一次（t26 补写 §10.1 后又重导一次），现行 **26 页**；12.4 表在**第 17 页**（C=1000 列已是「—（旧报告权重键不匹配，不作结论）」、BN `107 → 0`，与 `report/REPORT.md:531-551` 一致，其后的并列口径备注续到第 18 页），13.3 节的**标题在第 19 页、内容续到第 20 页**（`report/REPORT.md:599` 起；`13.3` 字样另在第 23 页的复跑命令注释里出现一次），十六、总结在**第 22 页**（`report/REPORT.md:725` 起）。上面「第 21/23/27 页」是审计当时（压缩前 37 页版）的页码；`word/document.xml para 913 / 1053` 同理，DOCX 重导后段号已变，按节标题检索即可。

### D7【孤值与历史值】逐个处置

见第 4 节表格。要点：
- 5.25e-06 / 1.41e-06、5.245e-06 / 1.414e-06、6.53e-06、6.527e-06、2.265e-06、4.108e-07 **无任何产物**；
- 4.991888999938965e-07、2.9802322387695312e-06 **有产物但前提无效**（B7）；
- 同一个"旧值"在仓库里有两种写法（5.25e-06 vs 5.245e-06；6.53e-06 vs 6.527e-06），说明它们本来就是手抄/口算值，不是落盘值。

### D8【易误判：仓库里存在与误差值同形的学习率】

- `outputs/logs/train_opt_disc_20260913_151521.log:70` 等 6 个训练日志：`lr_bb=1.81e-05` 与 B3 的 `max|Δlogits| 1.811981201171875e-05` **前 3 位完全相同**；`outputs/logs/*.jsonl` 里 `lr_head` 还写成全精度。
- `outputs/logs/eval_family_5models.log:12,34,56,78,100` 里 `max|logits diff| = 3.07e-05 …` 属于 B5，**不要**与 B1 的 Pet-37 结果放在一起。

**处置**：清单 CSV 用 `kind` 列区分（`noise_float` = 只是长得像误差值的数字），任何"按数字 grep 全仓"的做法都会误命中；t2/t3 不要用 grep 数字的方式核对，请按 bucket 查清单。

### D9【已复核、未发现】任务书点名的两类混写在这份仓库里不存在

- 「把 n=32 随机输入说成真实图片」：全仓（`*.md/*.py/*.svg` + PPTX/PDF/DOCX）搜索"32"与"真实/图片/图像"同现的行，命中的两处都是**正确**的反向声明 ——
  `report/PPT_CONTENT.md:724` 与 `tools/update_defense.py:220`：`32 个输入是 torch.randn，不是真实测试图片。`；
  `答辩PPT_RepViT.pptx` slide 12 para 1-2 与备注 notes 4 也写明"固定随机输入 n=32"并"与第 13 页真实图片 ONNX 实验分开"。**保留即可，不要删。**
- 「n=8 对照记录」：见 D1 —— 问题不在"混写"，而在"给了一个没有证据的来源"，已按 D1 收敛。

---

## 4. 旧值逐个处置（对应任务书 d 项）

| 旧值 | 现出现在 | 仓库是否有配套产物 | 能否复跑 | 处置 |
|---|---|---|---|---|
| **5.25e-06 / 1.41e-06** | `README.md:343`、`:349`（说明句）；`report/LOGITS_AUDIT.md:15`；`report/答辩PPT_RepViT.pptx` slide 13 para 27 / notes 5 para 4；`report/PPT_CONTENT.md:752,764`；`tools/update_defense.py:241`；历史：`81693dd:README.md:342` | **无**（14 个提交的 `outputs/` 全不存在） | 否（n=8 复跑得 6.198883056640625e-06） | 只在"已废弃旧值"说明里出现；文案统一用 D1 的建议段落；**不得作为任何结论** |
| **5.245e-06 / 1.414e-06** | `report/LOGITS_AUDIT.md:15`；历史：`81693dd:report/PPT_CONTENT.md:228-229` | **无** | 否 | 同上；注意它与 README 的 5.25/1.41 **写法就不一致**，进一步说明是手抄值 |
| **6.53e-06** | `report/LOGITS_AUDIT.md:16`；历史：`81693dd:README.md:293`、`81693dd:report/PPT_CONTENT.md:41` | **无**（当时的落盘值就是 B4 的 7.092952728271484e-06） | 否 | 统一引用 **B4 = 7.093e-06** |
| **6.527e-06** | `report/LOGITS_AUDIT.md:16`；历史：`81693dd:report/PPT_CONTENT.md:203` | **无** | 否 | 同上；与 6.53e-06 是同一个旧值的两种写法 |
| **2.265e-06** | **历史**：审计当时 `report.pdf` 第 21 页仍在印；**现已不再印**（t17 重导后 C=1000 列改为「—」）—— 现行 12.4 表在 `report.pdf` 第 17 页、`report/REPORT.md:531-545`，两个值均 0 命中；`report/LOGITS_AUDIT.md:29` | **无** | 否 | 与 4.108e-07 一起从报表删除（D6 已修复）；最接近的 B7 产物是 4.991888999938965e-07 / 5.673758352031655e-08，**数值对不上**，不能顶替 |
| **4.108e-07** | **历史**：审计当时 `report.pdf` 第 21 页仍在印；**现已不再印**（同上） | **无** | 否 | 同上 |
| **4.991888999938965e-07**（mean 5.673758352031655e-08） | `outputs/reparam/repvit_m0_9_reparam_report.json:209,237,241`；`report/LOGITS_AUDIT.md:29` | **有**：`outputs/reparam/repvit_m0_9_reparam_report.json` | 前提无效，不作为结论 | 保留为历史记录；若正文提及，必须同时写"官方 C=1000 checkpoint 装入 timm 结构，`n_missing=605`、`n_unexpected=713`，是无效权重验证" |
| **2.9802322387695312e-06** | `outputs/advanced/repvit_m0_9_pet37_reparam_report.json:52`；`report/LOGITS_AUDIT.md:28` | **有**：`outputs/advanced/repvit_m0_9_pet37_reparam_report.json` | 前提无效（seed=0、2 个随机输入、BN 108 ≠ 107） | 保留为历史记录；**不得**用来替换 B4 的 7.093e-06 |

> 说明：以上"无产物"结论的判据是"逐提交 `git grep` 该字面量于 `outputs/` 树（排除审计清单自身）"，并用 B1/B4 的规范值做了同方法对照（能命中），排除了"搜索方式无效"的可能。

---

## 5. 改完之后必须同步的导出物与副本（防回归清单）

| 源文件 | 受影响的导出/副本 | 处置 |
|---|---|---|
| `report/REPORT.md`（9/13 两节 + 16 节小结） | `report.pdf`（审计当时第 21/23/27 页是旧版；**已于 t17 重导、t22 压缩，现行 26 页**，12.4 表 → 第 17 页、13.3 → 第 19–20 页、十六 → 第 22 页）、`report/REPORT.docx`（重导后段号已变，按节标题检索） | 改完 REPORT.md 后重新导出两者，并核对三处页码/段落 |
| `report/PPT_CONTENT.md`（:31,:95,:752,:764） | `report/答辩PPT_RepViT.pptx` 与 `report/答辩PPT_RepViT.pdf` | PPT_CONTENT 是文字源；PPTX 改完必须重新导出 PDF（当前 PDF 与 PPTX 一致，已核对 15 页/15 页） |
| `report/ppt_svg/*.svg`（`12_reparam.svg:5`、`13_onnx.svg:5`、`15_summary.svg:45`、`01_cover.svg:40`、`02_status.svg:56`） | 旧一代演示源 | `LOGITS_AUDIT.md:47` 已把它们定位为"历史素材，不作为答辩口径"；建议在文件头注释一句"已废弃，请以下一代 PPTX 为准"，或把其中的超证据措辞一并改掉，避免再次被复制进 PPTX |
| `outputs/PPT_OUTLINE.md` / `outputs/REPORT.md` | 早期副本 | 只含口径陈述，不含错误数值；确认无需再引用即可 |
| `tools/update_defense.py:136-137,220-249` | PPTX 自动改写脚本 | 其中的替换表已经包含"完全等价→误差小于阈值"、5.25e-06 说明；t3 若改 PPTX 正文，建议顺手把脚本里的文案与本文第 1 节展示值对齐 |

---

## 6. 复核命令（任何人都可以照跑）

```bash
# 1) 重新生成清单（可重复；两次运行 SHA256 一致）
python tools/audit_logits_references.py

# 2) 打印实验桶表
python tools/audit_logits_references.py --table

# 3) 复核 B1 与 B4（写入临时目录，不动仓库产物）
python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --limit 12 --out %TEMP%\consistency_n12.json
python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --limit 8  --out %TEMP%\consistency_n8.json
python tools/reparam_verify.py --model repvit_m0_9_pet37 --weights checkpoints/baseline_best.pt \
    --num-samples 32 --batch-size 8 --seed 20240912 --skip-onnx --out-dir %TEMP%\reparam_pet37

# 4) 复核"孤值从未落盘"（对每个旧值跑；B1/B4 的规范值应能命中，孤值应全部命中不到）
git log --all --format=%H | ForEach-Object { git grep -l --fixed-strings 5.25e-06 $_ -- outputs/ }

# 5) 在清单里定位某条引用
python -c "import csv;rows=list(csv.DictReader(open('outputs/verification/logits_reference_inventory.csv',encoding='utf-8-sig')));print([r for r in rows if r['file']=='README.md' and r['kind']=='error_value'])"
```

本次审计的环境事实（用于复现判断）：Python 3.14.5，已装 torch / onnxruntime / timm / onnx / PyMuPDF；`onnx/` 下 6 个 ONNX、`checkpoints/baseline_best.pt`、`data/oxford-iiit-pet/images`（7393 张）、`datasets/lists/*` 均齐备，因此 B1–B4 的复跑命令在本机可直接执行。

---

## 附录 A：清单字段说明

`outputs/verification/logits_reference_inventory.csv` 列：

| 列 | 含义 |
|---|---|
| `file` | Git 跟踪路径（POSIX 分隔） |
| `location` | 行号 / `page N` / `slide N para M` / `notes N for slide M para K` / `chart N on slide M data cache` / `word/document.xml para M` |
| `reference` | 命中片段（命中点前后各取一段） |
| `match` | 本次命中的具体文本（数值行就是那个数字） |
| `kind` | `error_value`（数值引用）/ `statement`（口径陈述）/ `noise_float`（长得像误差值的数字） |
| `bucket` | 实验桶 B1–B10 / `B11_statement` / `NOT_AN_ERROR_REF` |
| `authority` | 该桶的权威落盘产物（无则 `—`） |
| `reproducible` | `yes` / `partial` / `no` / `-` |
| `flag` | `ORPHAN` / `MIXED` / `OVERCLAIM` / `ABS_PATH` / `AMBIGUOUS` / `SELF_TOOL`（多个用 `+` 连接） |

`SELF_TOOL` 标记的行来自 `tools/audit_logits_references.py` 自身的规范值登记表，不是外部引用点，统计时应剔除。
本次 `AMBIGUOUS = 6`（2026-09-17 实测，全部落在 `outputs/predictions/*_probs.json`）：这些文件是 `--dump-probs` 落的**原始概率矩阵**，其中两位有效数字、写成 `×e-05` 形式的概率值会同时落进 B12 与 B13 的舍入容差，属**工具误报**，不是误差值引用。除此之外没有任何 token 同时命中多个桶。

## 附录 B：按桶清点"谁在引用"（`error_value` 行）

| 桶 | error_value 行数 | 主要引用方（本次实测，括号内为该文件的 error_value 行数） |
|---|---|---|
| B1 | 140 | `VERIFICATION_LOGITS_PPT.md`(30)、`LOGITS_AUDIT_FINDINGS.md`(23)、`README.md`(14)、`REQUIREMENTS_AUDIT.md`(12)、`PROGRESS.md`(8)、`LOGITS_AUDIT.md`(8)、`PPT_CONTENT.md`(5) |
| B2 | 57 | `predictions/repvit_m1_5_in1k_*_probs.json`(8)、`LOGITS_AUDIT_FINDINGS.md`(8)、`REQUIREMENTS_AUDIT.md`(5)、`predictions/repvit_m1_0_in1k_*_probs.json`(4)、`LOGITS_AUDIT.md`(4)、`VERIFICATION_LOGITS_PPT.md`(4)、`predictions/repvit_m0_9_in1k_*_probs.json`(3) |
| B3 | 58 | `LOGITS_AUDIT_FINDINGS.md`(8)、`predictions/repvit_m1_5_in1k_*_probs.json`(7)、`LOGITS_AUDIT.md`(5)、`REQUIREMENTS_AUDIT.md`(5)、`predictions/repvit_m1_1_in1k_*_probs.json`(4)、`VERIFICATION_LOGITS_PPT.md`(4)、`predictions/repvit_m0_9_in1k_*_probs.json`(3) |
| B12 | 47 | `LOGITS_AUDIT_FINDINGS.md`(8)、`REQUIREMENTS_AUDIT.md`(5)、`LOGITS_AUDIT.md`(4)、`VERIFICATION_LOGITS_PPT.md`(4)、`README.md`(2)、`consistency_repvit_m1_1_in1k.json`(2)、`predictions/repvit_m0_9_in1k_*_probs.json`(2) |
| B13 | 70 | `predictions/repvit_m1_5_in1k_*_probs.json`(11)、`predictions/repvit_m1_0_in1k_*_probs.json`(8)、`LOGITS_AUDIT_FINDINGS.md`(8)、`predictions/repvit_m1_1_in1k_*_probs.json`(5)、`REQUIREMENTS_AUDIT.md`(5)、`predictions/repvit_m0_9_in1k_*_probs.json`(4)、`LOGITS_AUDIT.md`(4) |
| B4 | 214 | `VERIFICATION_LOGITS_PPT.md`(43)、`LOGITS_AUDIT_FINDINGS.md`(32)、`REQUIREMENTS_AUDIT.md`(16)、`update_defense.py`(14)、`PPT_CONTENT.md`(12)、`答辩PPT_RepViT.pptx`(12)、`README.md`(10)（t21 起由 210 增至 214：新增 4 条来自 `docs/workflow_zh.svg`，未进 Top-7） |
| B5 | 26 | `predictions/repvit_m1_5_in1k_*_probs.json`(6)、`LOGITS_AUDIT.md`(5)、`LOGITS_AUDIT_FINDINGS.md`(5)、`predictions/repvit_m0_9_in1k_*_probs.json`(3)、`predictions/repvit_m1_0_in1k_*_probs.json`(1)、`predictions/repvit_m1_1_in1k_*_probs.json`(1)、`pretrained_eval/repvit_m0_9/metrics.json`(1) |
| B7 | 16 | `repvit_m0_9_reparam_report.json`(6)、`LOGITS_AUDIT_FINDINGS.md`(6)、`advanced/repvit_m0_9_pet37_reparam_report.json`(2)、`LOGITS_AUDIT.md`(1) |
| B8 | 171 | 主要是"旧值说明"位置：`LOGITS_AUDIT_FINDINGS.md`(72)、`VERIFICATION_LOGITS_PPT.md`(33)、`audit_logits_references.py`(17)、`LOGITS_AUDIT.md`(8)、`README.md`(7)、`report.pdf`(5) |

（`report/答辩PPT_RepViT.pdf` 与 `report/答辩PPT_RepViT.pptx` 的页/片号一一对应，PDF 为 PPTX 的当前导出。B6 导出探针、B9 路线探针、B10 早期冒烟三个桶本次没有 `error_value` 行，只有口径陈述，故未列入上表。）
