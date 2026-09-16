# 独立验证报告：logits 误差口径统一 + 答辩 PPTX 成品（t9 / t19）

> 本文件含两轮独立验证：
> - **第 1 轮（t9，attempt `4426b351…`）**：2026-09-16 18:44–18:52，verdict = **needs_revision**（blocker V1 + findings V2–V6）。正文即下文的第 0–5 节与附录。
> - **第 2 轮（t19，attempt `0f04cab8…`）**：2026-09-16 19:01–19:05，verdict = **pass**（V1–V6 全部收敛，第 1 轮已通过项无回归）。见文末「第 2 轮」小节。

- 验证人：verifier（独立验证员，本任务不修改任何被验证产物）
- 第 1 轮任务：t9（attempt_id `4426b351-0757-45da-b71e-297fd4a5f0fb`）
- 仓库：`C:\Users\14675\Desktop\one团队秋招ai算法\RepViT-Reproduction`
- 验证时点：**2026-09-16 18:44 – 18:52**（t13 的 `report/ppt_svg/README.md` 18:41:11、`09_results.svg` 18:41:17，t14 的 `tools/selfcheck.py` 18:42:29 均已完成并纳入本次验证）
- 浏览器/工具环境：Windows + PowerShell，python 3.14.5，python-pptx 1.0.2，PyMuPDF 1.28.0
- 被验证修订的 SHA256（本次核对的精确版本）：
  - `report/答辩PPT_RepViT.pptx` = `20FA9F4F…42D463`（8,118,204 B，18:32:17，15 页）
  - `report/答辩PPT_RepViT.pdf` = `F4918008…4FCD943`（1,114,610 B，18:32:33，15 页）
  - `report.pdf`（仓库根目录）= `FC8F78CF…2772361`（1,371,641 B，18:39:27，37 页）
  - `report/REPORT.docx` = `E069A15C…D1679F089`（76,426 B，18:39:05）
  - `report/PPT_CONTENT.md` = `CBDF6D24…3481A245A`（26,784 B，18:32:17）
  - `outputs/metrics/consistency_repvit_m0_9_pet37.json` = `896DBE3F…1AC68643`
  - `outputs/reparam/repvit_m0_9_pet37_reparam_report.json` = `D3A28D49…0D16332E1`

## 0. 结论（verdict = needs_revision）

**9 条验收项中 8 条 passed，第 9 条 failed。**

- logits 口径分裂（重参数化 vs PyTorch↔ONNX）已在全仓所有载体中彻底分开，**0 处并列结论句**；两类误差各自只引用唯一可复跑的落盘值。
- 实跑复现：两次复跑与入库值**逐位（float64 bit pattern）完全相同**；三个 consistency JSON 与 `outputs/reparam/`、`outputs/logs/` 未被篡改。
- 答辩 PPTX 的页数、原生图表、四项可视化、性能元信息全部满足试题第 9–11、16–17 页要求；P03+P04 合并页同时覆盖两个环节且备注写明合并事实，页脚/下标无串位。
- **唯一失败点（可直接复现，1 行级修复）**：`python tools/selfcheck.py --stage skeleton` 现在报 `paths FAIL 1 处`，而 `report/REPORT.md:1011`、`report.pdf` 第 30 页、`report/REPORT.docx`、`PROGRESS.md:223` 与 PPTX P02/P15 都写着「paths 检查 0 处 / 代码零写死绝对路径 / 37 条验收项自检 0 失败」。该命中由 t6 新增的 `tools/audit_logits_references.py:430` 引入（字符串字面量 `'C:\\Users'`，实质是误报，但仓库自带检查器判 FAIL）。

---

## 1. 验收项逐条结论

### 1.1 【passed】重扫全仓后每个 logits 误差数字唯一归属；不存在把重参数化误差与 ONNX 误差并列成同一结论的句子

命令（自写脚本，扫描 README.md / PROGRESS.md / report/\*.md / PPT_CONTENT.md / ppt_svg/\*.svg / tools/\* / deploy/\* / outputs/\*\* 等全部文本载体）：

```powershell
$env:PYTHONIOENCODING="utf-8"; python $env:TEMP\t9verify\chk_juxta.py
```

关键输出：

```
JUXTAPOSITION SCAN: sentence containing BOTH a B4(reparam) literal AND ONNX-agreement language
TOTAL juxtaposition hits = 0
```

- 桶归属抽查（全仓字面量计数，排除审计清单本身）：
  - `6.198883056640625e-06` 精确值只出现在 `outputs/metrics/consistency_repvit_m0_9_pet37.json`、其复跑副本、`README/REPORT/LOGITS_AUDIT(.md/_FINDINGS.md)/PROGRESS/PPT_CONTENT` 的说明文字、`tools/*.py` 文案表。
  - `7.092952728271484e-06` 只出现在 `outputs/reparam/repvit_m0_9_pet37_reparam_report.json`（×3）、`outputs/reparam/logits_diff.json`、`outputs/verification/{reparam_pet37,selfcheck_logits}.json` 及其说明文字。
  - 两个数值没有任何一处出现在同一段落并被合并成一条结论。
- PPTX 侧的分开口径（逐字）：
  - P01 脚注：「口径分开：结构重参数化 = 32 个固定随机输入（max|Δ| 7.093e-06，Top-1 32/32）；ONNX 一致性 = Pet-37 / M0.9 / M1.0 三个型号各 12 张真实图片（第 13 页，3 份 n=12 落盘），只覆盖这 12 张，不代表全部六个 ONNX 型号。旧的 5.25e-06 未找到配套产物。」
  - P01 备注：「重参数化误差与 ONNX 一致率是两批实验，不并成一句结论。」
  - P12 来源行：「本页是 32 个固定随机输入的重参数化实验，与第 13 页 n=12 真实图片的 ONNX 实验分开」；P13 脚注 1：「重参数化 7.093e-06（32 个随机输入）是第 12 页的另一批实验，不与本页并成一句结论。」

### 1.2 【passed】PyTorch↔ONNX 只引用 B1；结构重参数化只引用 B4

落盘值直读（`Get-Content -Raw outputs\metrics\consistency_repvit_m0_9_pet37.json`）：

```
"n": 12, "max_abs_logits": 6.198883056640625e-06, "mean_abs_logits": 1.5006899711048998e-06,
"top1_agree_rate": 1.0, "top5_set_agree_rate": 1.0, "verdict": "PASS"
```

`outputs/reparam/repvit_m0_9_pet37_reparam_report.json`：`max_abs_err=7.092952728271484e-06`、`mean_abs_err=2.030726818702533e-06`、`top1_same_count=32`、`top1_same_rate=1.0`、`modules.before.n_bn=107` → `modules.after.n_bn=0`、`Conv 126→103`、`onnx before/after nodes 480→387`。

展示值统一 4 位有效数字：
- `Select-String -Path report\ppt_svg\*.svg -Pattern 'BN 108|108 → 0|7\.1e-06|完全等价'` → **0 命中**；
- `7.1e-06` 全仓仅剩审计语境：`report/LOGITS_AUDIT.md:11`（「不要再用 7.1e-06 这类 2 位写法」的规则句）、`LOGITS_AUDIT_FINDINGS.md`（缺陷引用）、`tools/export_defense_pdf.py:19,51`（禁用词表常量）、`tools/update_defense.py:311,314`（替换表）、`outputs/verification/logits_reference_inventory.csv`（历史引用清单）。**没有任何展示载体（README / REPORT.md / PPT_CONTENT.md / PPTX / 答辩 PDF / 三份 SVG）再出现 7.1e-06。**

### 1.3 【passed】5.25e-06 表述一致且不超出证据

自写脚本对三份文档逐行取哈希：

```
README.md: marker count=1 / report/LOGITS_AUDIT.md: marker count=1 / report/REPORT.md: marker count=1
line hash=7f6bcccfb52b7a3a  (×3，完全相同)
unique paragraph text count across 3 carriers = 1
UNIQUE TEXT SHA256 = 7f6bcccfb52b7a3a2bdf015cf60dd0aba8112825d2c823de0c0c5a46c04b279f
contains '不能证明它来自 n=8': True
contains '属于旧的 n=8 对照记录': False
```

PPTX 中的同口径表述（`ppt/slides/slide13.xml` + 备注，原始 OOXML 抽取）：
- P13 脚注 4：「旧的 5.25e-06 缺少对应完整记录，当前统一引用上表落盘值（4 位有效数字）。」
- P13 备注：「5.25e-06 是未找到对应完整产物的旧引用；旧命令写 limit=8 不能证明该值来自 n=8。当前权重跑 n=8 也不能复现旧值。」
- 全文（15 页正文 + 全部备注）`完全等价 / 7.1e-06 / 九类 / 108 → 0 / BN 108 / 2.265e-06 / 4.108e-07 / 5.245e-06 / 1.414e-06 / 6.53e-06 / 6.527e-06` **全部 0 命中**。

（说明：PPTX 用的是压缩后的等价表述，不是与 Markdown 逐字节相同；四种载体的**口径与证据边界一致**，均未断言它来自 n=8。）

### 1.4 【passed】实跑复现

两次独立复跑（写入 `outputs/verification/tmp_verify/`，未覆盖入库产物）：

```powershell
Set-Location <repo>
python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --images datasets/lists/pet_test.txt --limit 12 --out outputs/verification/tmp_verify/consistency_n12.json
python tools/reparam_verify.py --model repvit_m0_9_pet37 --weights checkpoints/baseline_best.pt --num-samples 32 --batch-size 8 --seed 20240912 --skip-onnx --out-dir outputs/verification/tmp_verify/reparam
```

原始输出（关键行）：

```
[ckpt] missing=0 unexpected=0
EP=['CPUExecutionProvider']  n=12  model=repvit_m0_9_pet37
max|Δlogits|=6.199e-06  mean|Δlogits|=1.501e-06  Top-1 一致率=100.00%  Top-5 集合一致率=100.00%  verdict=PASS
[exit=0]

[registry] repvit_m0_9_pet37 -> arch=repvit_m0_9 impl=timm num_classes=37 distillation=False
[logits] max_abs_err=7.093e-06 mean_abs_err=2.031e-06 top1_same=32/32 top5_min_overlap=5
[params] 4732805 -> 4.696M (Δ=-36504)  buffers 47867 -> 0
[verdict] PASS  -> outputs\verification\tmp_verify\reparam
[exit=0]
```

逐字段 + 逐位比较（`python $env:TEMP\t9verify\chk_repro.py`）：

```
rerun vs stored: 18/18 字段全部 [OK]
float64-bit-exact max_abs_logits:  True  hex 3eda000000000000 == 3eda000000000000
float64-bit-exact mean_abs_logits: True  hex 3eb92d6a12aaaaab == 3eb92d6a12aaaaab
REPARAM: max_abs_err equal=True / mean_abs_err equal=True / top1_same_rate equal=True / num_samples equal=True
         judge 与 diff 两处字典完全相同；modules.before.n_bn=107、after.n_bn=0 一致
```

→ **浮点完全相等（不是"可解释的差异"）**。

### 1.5 【passed】产物未被篡改

```powershell
git diff --numstat -- outputs/metrics/consistency_repvit_m0_9_pet37.json outputs/metrics/consistency_repvit_m0_9_in1k.json outputs/metrics/consistency_repvit_m1_0_in1k.json
```

```
2	2	outputs/metrics/consistency_repvit_m0_9_in1k.json
2	2	outputs/metrics/consistency_repvit_m0_9_pet37.json
2	2	outputs/metrics/consistency_repvit_m1_0_in1k.json
```

`git diff` 全文只有第 4 行 `onnx_path` 与第 8 行 `images` 两处 hunk；脚本进一步断言：

```
diff keys = ['images', 'onnx_path']      exactly onnx_path+images changed: True
line counts equal (23) and 2-line-revert == HEAD bytes: True   （三个文件均为 True）
key order identical to HEAD: True
```

`n / max_abs_logits / mean_abs_logits / top1_agree_rate / top5_set_agree_rate / threshold / verdict` 全部逐字段相等；**没有发生 json 往返重排**。

`outputs/reparam/` 与 `outputs/logs/`：

```powershell
git status --porcelain -- outputs/reparam outputs/logs outputs/figures outputs/benchmarks
```

→ 空输出（与 HEAD 完全一致）。`outputs/reparam/` 全部文件 mtime 仍是 2026-09-14 00:07:2x；最后一次提交触碰 `outputs/reparam/` 是 `7f244af`（2026-09-13 17:38:39），本次会话的 5 个提交（`62aeb9e`…`2d65e63`）未再触碰。

> 自曝：我在跑 `python tools/selfcheck.py --strict --only …` 时，该工具把报告默认写到 `outputs/metrics/selfcheck_report.json`，覆盖了入库文件；我随即用 `git checkout -- outputs/metrics/selfcheck_report.json` 还原（当前 `git status` 对该文件为空）。这是本次验证过程中唯一一次对入库产物的意外写入，已复原，特此说明。

### 1.6 【passed】PPTX：页数、原生图表、四项可视化

`python-pptx` 独立打开（`python $env:TEMP\t9verify\chk_pptx.py`）：

- `SLIDE COUNT: 15`（要求 10–15）。
- 第 8 页 5 个图表全部是 `XL_CHART_TYPE.LINE (4)` 原生可编辑图表，series 为 `Baseline` / `opt_combo`，各 40 个点：train loss / val loss / val Top-1(%) / val Macro-F1(%) / lr。原始 OOXML：`charts per slide {slide8.xml:5, slide12.xml:1, slide14.xml:2}`，共 8 个 chart 部件。
- 四项可视化以 **SHA1 与落盘文件逐一对应**（不是靠页面上写了什么）：

| 幻灯片 | 内嵌图 SHA1 | 落盘来源 | 满足的要求 |
|---|---|---|---|
| Slide 9 | `5ac0c80d…c50426` | `outputs/confusion_matrix/baseline_per_class_f1.png` | 每类 F1 柱状图 |
| Slide 9 | `12bdfc43…721c` | `outputs/confusion_matrix/baseline_cm.png` | 混淆矩阵 |
| Slide 11 | `b251601d…c589` | `outputs/predictions/test_top5_baseline_grid8.png` | **8 张测试集预测（Top-5 + 置信度）** |
| Slide 11 | `2ab90ea0…19ed` | `outputs/predictions/compare_baseline_vs_opt_combo_grid4.png` | **4 组 Baseline vs 优化 同图对比** |
| Slide 11 | `bd55b63e…2c137` | `outputs/predictions/external_top5_pet37_grid5.png` | **5 张训练集以外实拍（Top-5 + 置信度）** |
| Slide 10 | 4×(224×224) | `outputs/gradcam/gradcam_{correct,wrong}_baseline_*.png` | Grad-CAM 2 正确 + 2 失败 |

- 我另外用视觉方式独立确认了图内容：`grid8` 实际为 8 个面板（Abyssinian_2/29/201/202/204/205/206/207，每个面板含 GT + Top-1 + Top-5 柱状图）；`grid5` 实际为 5 个面板（beagle 0.903 / boxer 0.643 / chihuahua 0.680 / great_pyrenees 0.869 / newfoundland 0.771），每张都带 Top-5 类别与置信度。

### 1.7 【passed】P03+P04 合并页与下标未串位

- P03 单页同时包含：左栏「环节 2：M0.9 是一条纯卷积流水线」（Stem/Stage 0-3/48-96-192-384ch/[2,2,14,2]/训练态 vs 推理态）+ 右栏「环节 3：标准 ViT 与 RepViT 的对应关系」（6 行对比表含表头「环节｜标准 ViT｜RepViT」，空间混合/通道混合/下采样/归纳偏置/ONNX 图内）+ 四个关键设计 + 实证句。
- P03 备注第一行：「【本页是合并页】…合并到同一页：左栏 = 环节 2，右栏 = 环节 3。合并原因：PPT 需控制在 15 页内，同时新增第 11 页「预测结果」…」，并列出题干 15 个环节的落点。
- 逐页核对（自己重新数一遍，未沿用旧编号）：页脚编号与页面内容一一对应 `P02=02 … P14=14`，无内容串位；首页与末页无 2 位页脚（与 t8 声明一致）。
- 脚本层面：`tools/update_defense.py` 除封面 `slides[0]`、状态页 `slides[1]`（都排在合并点之前）之外，**全部按标题 marker 定位**（`find_slide(prs, CURVES_MARK/CM_MARK/CAM_MARK/REPARAM_MARK/ONNX_MARK/BENCH_MARK)`），最后统一 `renumber(prs)` 按最终顺序重排页脚 —— 结构上不会因合并而串位。

### 1.8 【passed】性能图表元信息（型号/输入尺寸/batch/硬件/后端/精度/次数）

| 页 | 页面上的元信息（逐字摘录） | 7 项 |
|---|---|---|
| P08 曲线 | 「RepViT-M0.9 · Pet-37 · 输入 224×224 · batch 64 · NVIDIA GeForce RTX 4060 Laptop GPU · PyTorch 2.14.0+cu126（bf16 autocast）· 曲线共用坐标范围 · test 评价次数 Baseline 1 / 组合 2」 | 7/7 |
| P09 混淆矩阵 | 「Baseline · Pet-37 test · 3669 张 · PyTorch 2.14.0+cu126 / CUDA / FP32 / 224×224 / 每张只推理一次」 | 7/7 |
| P12 重参数化 | 「RepViT-M0.9 Pet-37（baseline_best.pt）· 1×3×224×224 · batch 8 · n=32 · i7-13650HX CPU · PyTorch 2.14.0+cu126 / FP32 · 1 次融合前后对比 · seed=20240912」 | 7/7 |
| P13 ONNX | 「原模型 vs 融合后 ONNX · ORT 1.30.0 CPUExecutionProvider · FP32 · batch 1 · 1×3×224×224 · i7-13650HX / Windows 11 · 每个型号各 12 张真实图片 · 每张只跑一次」 | 7/7 |
| P14 延迟 | 「型号：M0.9 / M1.0 / M1.1 / M1.5 / M2.3 / M0.9-Pet37 · 输入 1×3×224×224 · batch 1 · i7-13650HX · ORT 1.30.0 CPUExecutionProvider · FP32 · 预热 10 + 正式 50 次 · threads 4」 | 7/7 |

### 1.9 【FAILED】PPTX 关键数字可溯源 / PPT_CONTENT 一致 / PDF 页数一致

三个分句的结果：

1. **实验类关键数字 → 全部与落盘一致（通过）**：

| PPTX 数字 | 落盘文件与字段 | 结论 |
|---|---|---|
| 92.34% / 99.26% / 92.22% | `outputs/metrics/baseline_test.json` `top1=0.9234124` / `top5=0.9926410` / `macro_f1=0.9221970` | 一致 |
| 92.12% / 99.59% / 92.01% | `outputs/metrics/opt_combo_test.json` | 一致 |
| 281 错 / 10 跨物种 / 3.56% | `outputs/confusion_matrix/baseline_cat_dog_block.json` `n_error=281, n_cross_species_error=10` | 一致 |
| 每类 F1 最低 0.696 / 最高 0.995 | `baseline_per_class.csv`：`34 Staffordshire Bull Terrier 0.695652` / `17 Japanese Chin 0.995025` | 一致 |
| P50 `[7.373, 9.147, 10.234, 17.647, 32.687, 7.122]`、P95 `[8.002, 9.76, 11.241, 18.926, 33.322, 7.692]` | `outputs/benchmarks/summary.csv` 六个型号 `p50_ms`/`p95_ms` | **逐位相同** |
| 散点 Top-1 78.2 / 79.9 / 79.9 / 82.8 / 83.1 | `outputs/pretrained_eval/*/metrics.json` | 一致 |
| T = 0.615 | `outputs/advanced/interp/calibration.json` `temperature=0.6148912310600281` | 一致 |
| 7.093e-06 / 2.031e-06 / 32/32 / 107→0 / 126→103 | `outputs/reparam/repvit_m0_9_pet37_reparam_report.json` | 一致 |
| 6.199e-06 / 1.501e-06 / 1.717e-05 / 2.360e-06 / 1.812e-05 / 2.464e-06 | 三份 `outputs/metrics/consistency_*.json` | 一致 |

2. **`report/PPT_CONTENT.md` 与 PPTX 一致（通过）**：该文件由 `write_content(prs)` 直接从最终 PPTX 生成；我按 P01–P15 共 15 节逐行比对（正文 + 备注，归一化空白/标点后），**TOTAL missing = 0**，15 节全部 `[OK]`。

3. **`report/答辩PPT_RepViT.pdf` 页数 = PPTX 页数（通过）**：PyMuPDF 读到 `PAGES = 15`，与 `python-pptx` 的 15 页一致；该 PDF 全文 `完全等价 / 7.1e-06 / 九类 / 108 → 0 / 2.265e-06 / 4.108e-07` 均 0 命中。

4. **失败点（下句）**：PPTX P02 副标题与 P15 副标题写着「**37 条验收项自检 0 失败**」「代码零写死绝对路径」，`report/REPORT.md:1011` 写「无任何写死的个人绝对路径（`selfcheck --stage skeleton` 的 `paths` 检查：**0 处**）」，`PROGRESS.md:223` 写「工具零硬编码绝对路径（`paths` 检查 0 处命中）」。**这些断言在 `outputs/` 下找不到对应落盘文件，且与仓库自带检查器的实际输出直接矛盾**：

```powershell
Set-Location <repo>
python tools/selfcheck.py --stage skeleton
```

```
paths     FAIL    1 处 首处 tools/audit_logits_references.py:430
          -> 把路径改为 config 字段
skeleton  PASS    33 文件全在位
合计 2 项：PASS 1 / FAIL 1          [exit=0]
```

全量运行同样带 FAIL：

```powershell
python tools/selfcheck.py --json "$env:TEMP\selfcheck_outrepo.json"
# summary {'total': 34, 'pass': 33, 'fail': 1}
# FAILED: {"id": "paths", "status": "FAIL", "detail": "1 处 首处 tools/audit_logits_references.py:430"}
```

同一句断言还被导出进了成品：`report.pdf` **第 30 页**（「无任何写死的个人绝对路径（selfcheck --stage skeleton 的 paths 检查：0 处）」）与 `report/REPORT.docx`（同一句）。最小复现如上，一行命令即可看到 FAIL。

---

## 2. 缺陷清单（本次验证发现，均给出最小复现与建议改法）

| ID | 严重度 | 问题 | 最小复现 | 建议改法 |
|---|---|---|---|---|
| V1 | **blocker** | 交付物断言「paths 检查 0 处 / 代码零写死绝对路径 / 37 条自检 0 失败」，实际 `paths` FAIL 1 处。命中来自 t6 新增的 `tools/audit_logits_references.py:430` 的检测字面量 `'C:\\Users'`（**实质是误报**：它是检测器自身的匹配串，不是任何 I/O 路径）。但仓库自带检查器判 FAIL，交付物不能继续宣称 0 处。 | `python tools/selfcheck.py --stage skeleton` → `paths FAIL 1 处 首处 tools/audit_logits_references.py:430` | 二选一：(a) 让该字面量不再命中规则（例如改成 `'C:' + chr(92) + 'Users'` 或在 `c_paths()` 的 skip 里排除该文件并写明理由）；(b) 若选择保留，则同步改正 `report/REPORT.md:1011`、`PROGRESS.md:223`、PPTX P02/P15 的措辞并重新导出 PDF/DOCX。推荐 (a)，改动 1 行且与「代码零写死绝对路径」的事实一致。 |
| V2 | medium | 「37 条验收项」与工具实际检查数不符：`tools/selfcheck.py` 在 `81693dd`、`HEAD`、当前工作区三处都恰好是 **34** 个 `@check`（`re.findall` 计数：34 / 34 / 34，无增删）。PPTX P02/P15、`report/PDF p2`、`PPT_CONTENT.md:56,799`、`REPORT.md:1010` 的「37 条」没有落盘对应物。 | `python -c "import re,io;print(len(re.findall(r'@check\(\s*\"',io.open('tools/selfcheck.py',encoding='utf-8').read())))"` → 34 | 把文案改为工具实际口径（34 项），或在 REPORT 里说明 37 条 DoD 与 34 条自检项的映射关系；至少不要写成「37 条由 selfcheck 逐条自检」。属**既有**问题（`81693dd` 起就在），非 t7/t8 引入。 |
| V3 | medium | `report/REPORT.md` §16.1 第 2 条仍用 **92.26% / 92.14% / 3.17%**，与全仓唯一权威落盘 `outputs/metrics/baseline_test.json`（92.34 / 92.22）及本报告 §1.9 表格冲突；该句已导出到 `report.pdf` 第 29 页与 `REPORT.docx`。属**既有**问题（`git grep 92.26 81693dd -- report/REPORT.md` 命中 949 行），t7 未在 D1–D9 范围内处理。 | `git grep -n 92.26 81693dd -- report/REPORT.md`；`python -c "import json;print(json.load(open('outputs/metrics/baseline_test.json'))['top1'])"` → 0.9234123739438539 | 统一为落盘值 92.34% / 92.22% / 3.56%，或在括号里注明 92.26% 是重跑前的旧值（`REPORT.md:464` 已有「重跑一次 92.26→92.34」的说明，可交叉引用）。 |
| V4 | medium | `report/REPORT.md:1001-1002` §16.1 第 5 条仍是超证据断言「8/8 全对**说明模型没有过拟合到 Pet 的拍摄风格**」，而 PPTX P10 与 `ppt_svg/11_gradcam.svg` 已改成「8/8 判对只是小样本观察，不能证明…」。同一事实在两个交付物里口径相反。属既有措辞（HEAD 就存在），但 D5 类问题应同批收口。 | 读 `report/REPORT.md:1001` 与 PPTX slide10 备注对比 | 与 PPTX 对齐：「8 张实拍上 8/8 判对；样本量太小，不能据此证明跨域泛化或没有过拟合」，并重新导出 PDF/DOCX。 |
| V5 | low | PPTX P04 的 MACs 列口径混用：M0.9/M1.0 用 `outputs/pretrained_eval/*/metrics.json` 的 `macs_g`（0.8471/1.1428），M1.1/M1.5/M2.3 却等于 `outputs/benchmarks/family_summary.csv` 的 `macs_G`（1.3581/2.3084/4.5739）；而同一张表的「数据来源」只写了 `outputs/pretrained_eval/<model>/metrics.json · summary.csv`（此处 `macs_g` 分别是 1.3769/2.3397/4.6263，**与页面数字不一致**）。属既有问题（`81693dd` 与 `HEAD` 的 slide5 已是这 5 个数字，t8 未改动）。所有数字都能在 `outputs/` 找到，故不判为编造；但来源标注不完整。 | `git show 81693dd:report/答辩PPT_RepViT.pptx` 后抽取 slide5 文本可见 1.358/2.308/4.574 | 在「数据来源」补一句 MACs 来自 `outputs/benchmarks/family_summary.csv`（infer_fused 口径），或统一改成 metrics.json 的 macs_g。 |
| V6 | low | `report/ppt_svg/13_onnx.svg`（设计源）只画了 Pet-37 一行（6.199e-06/1.501e-06/100%），最终 PPTX P13 表已扩成 3 个型号各 12 张；`report/ppt_svg/12_reparam.svg` 标题是「…是代数等价的替换，实测误差小于阈值」，而 PPTX P12 标题是「重参数化：合并分支后，误差仍在设定阈值内」。数值无矛盾，措辞不同。按船长更正 3（ppt_svg 是设计源、以 PPTX 为准）**不判为缺陷**，仅登记。 | 读两份 SVG 与 PPTX P12/P13 | 可选：在 `ppt_svg/README.md` 的注记里补一句「13_onnx.svg 是 3 型号表之前的版本」。 |

---

## 3. D1–D6 审计项的独立复核（t6 的主张源逐条自验）

| 项 | t6 主张 | 我执行的独立命令 | 我的结论 |
|---|---|---|---|
| D1（5.25e-06 在任何提交的 outputs/ 树里都不存在） | 14 个提交全无配套产物 | 对全部 14 个提交跑 `git -C <repo> grep -l -e "5\.25e-0*6" <commit> -- outputs/`，并用同一方法跑 `6.198883056640625e-06` 作为方法有效性对照 | **成立**。抽检的 14/14 个提交中 `5.25e-06` 一次都没出现在 `outputs/`（`outputs/advanced/robustness_repvit_m0_9_pet37.json` 命中的是 `"macro_f1": 5.256`，与 logits 无关）；`62aeb9e` 起才在 `outputs/verification/logits_reference_inventory.csv`（审计清单）出现。同一命令在 **14/14** 个提交都能定位到 `outputs/metrics/consistency_repvit_m0_9_pet37.json` —— 证明方法有效，不是「没搜到」。 |
| D2（三个 JSON 只有 2 行变化） | 只动 onnx_path/images，逐行替换 | `git diff --numstat`（三个 `2 2`）+ 全文 diff（只有两个 hunk）+ 逐字段比较 + 「把两行替换回去即逐字节等于 HEAD」+ 键序比较 | **成立，且比主张更强**：`2-line-revert == HEAD bytes` 三个文件全为 `True`，关键字序与 HEAD 完全相同。 |
| D3（封面/状态页把 B4 误差与 ONNX 一致率并列） | 已分开 | 抽 PPTX slide1/slide2 XML 全文 | **已修复**。封面 4 个统计格各自带限定词（「结构重参数化（32 随机输入）max|Δ|」「ONNX Top-1 一致（3 型号各 12 张）」），并新增口径脚注；状态页卡片 05 与脚注同样分开。 |
| D4（「6 个 ONNX 一致率 100%」「九类问题已排除」外推） | 已收敛 | PPTX 原始 OOXML 全文扫 `九类`/`六个模型`/`全部六`；读 `ppt_svg/13_onnx.svg`、P02 脚注、REPORT.md §13.3 | **已收敛**。PPTX 中 `九类` 0 命中；`六个 ONNX`/`全部六` 的命中全部是**否定/限定**语境（「不代表全部六个 ONNX 型号」「也不能说成全部六个模型一致」）；`13_onnx.svg` 标题已改成「6 个 ONNX 模型；一致性只测了 3 个型号各 12 张」。 |
| D5（「完全等价」超证据） | 已从 PPTX 消失 | 原始 OOXML 全文 + `Select-String report\ppt_svg\*.svg -Pattern '完全等价'` | **PPTX 与 SVG 均已 0 命中**；PPTX P12 改为「代数等价的替换 + 已测输入 + 设定阈值，不是位级完全相等」。剩余 `完全等价` 命中只在 `report/LOGITS_AUDIT.md:92`（描述「已删去」的处置说明）、`LOGITS_AUDIT_FINDINGS.md`（缺陷引用）——即船长已声明的**已知假阳性**，不判失败。另见 V4：REPORT.md §16.1 第 5 条仍有同类超证据句（不同措辞）。 |
| D6（导出物不同步） | 已重导出 | PyMuPDF 读 `report.pdf` 页数与全文，逐页取 12.4/13.3/16.1 文本；`zipfile` 读 `REPORT.docx` 的 `word/document.xml` | **已修复，但页码需按更正 1 记**。`report.pdf` 现在是 **37 页 / 1,371,641 B / mtime 2026-09-16 18:39:27**（旧版对照：`git show HEAD:report.pdf` 与 `git show 81693dd:report.pdf` 均为 **1,421,842 B / 32 页**，且 HEAD 版与初始交付版字节相同），第 **22** 页是 12.4 表（`7.092953e-06`、`2.030727e-06`、`BN 模块数 107 → 0`、`ONNX 节点 BatchNormalization 0、Conv 103`），第 **24** 页是 13.3 + D1 段落，第 **29** 页是 16.1。全文 `2.265e-06 / 4.108e-07 / 108 → 0 / 完全等价 / 7.1e-06` 均 **0 命中**；`REPORT.docx` 同样 0 命中。 |

---

## 4. 附加项：t11 / t12 / t13 / t14

### 4.1 t11 + t12：`report/ppt_svg` 与最终 PPTX 对账

```powershell
Select-String -Path report\ppt_svg\*.svg -Pattern 'BN 108|108 → 0|7\.1e-06|完全等价'   # → 0 命中
Select-String -Path report\ppt_svg\*.svg -Pattern '2307\.09283'                        # → 01_cover.svg:26（arXiv 编号完好，未被误改）
git diff --numstat -- report/ppt_svg
```

```
3 3 01_cover.svg   3 3 02_status.svg   1 1 05_pretrained.svg   1 1 07_baseline.svg
3 3 08_optimize.svg  3 3 09_results.svg  3 3 10_confusion.svg  1 1 11_gradcam.svg
4 4 12_reparam.svg   4 4 13_onnx.svg     1 1 14_perf.svg       4 4 15_summary.svg
```

改动条数与任务自述**对得上**：t8 = 11 行 / 3 文件（12_reparam 4 + 13_onnx 4 + 15_summary 3），t11 = 2 行 / 2 文件（01_cover 1 + 02_status 1），t12 = 17 行 / 10 文件，t13 = 1 行 / 1 文件（`09_results.svg` 的「为什么说是噪声」→「为什么还不能确认是提升」），合计 31 行 / 12 文件 = `git diff --numstat` 的合计值。t12 的 15 行对账表确实覆盖全部 15 份 SVG（含 03_arch / 04_notvit / 06_data / 12_reparam / 13_onnx 五份「已核对一致、无改动」）。

逐字对齐抽查（≥4 份，全部命中）：

```
01_cover.svg  /P1  : 7.093e-06 | 结构重参数化（32 随机输入）max|Δ| | ONNX Top-1 一致（3 型号各 12 张） | 78.20% | 92.34%   → 5/5
02_status.svg /P2  : BN 107 → 0 · Conv 126 → 103 | max|Δ| 7.093e-06 · 32/32 一致 | 跨物种错误仅 3.56% | test Top-1 92.34% → 4/4
05_pretrained.svg /P4 : 官方权重在自建子集上的实际表现                        → 1/1
07_baseline.svg /P6  : 40 epoch 迁移训练把 test Top-1 做到 92.34% | 训练检查      → 2/2
12_reparam.svg /P12  : 7.093e-06 | 2.031e-06 | 107 → 0                        → 3/3
15_summary.svg /P15  : ④ 四项优化未显示稳定提升；需要多种子实验进一步判断        → 1/1
```

（V6 已登记 12_reparam 标题与 13_onnx 行数两处措辞/版本差异，按船长更正 3 不判缺陷。）

### 4.2 t13：`report/ppt_svg/README.md`

存在（1,562 B，18:41:11，`mtime` 晚于 t12），共 9 行，写明 **4 条注记**且首段即声明权威性：

1. 「本目录下的 15 份 SVG 是 `report/答辩PPT_RepViT.pptx` 的**合并前设计源**…**权威交付物是 `report/答辩PPT_RepViT.pptx`**，两者不一致时**一律以 PPTX 为准**。」
2. 合并事实 + 「全部文件的页脚编号仍是合并前的旧页序…**不要为了对齐而批量改这些页脚编号**」。
3. `09_results.svg` 对应的旧 P09 已被最终 P08 取代，「仅为历史设计记录，**不作为答辩口径**」。
4. 「本目录**没有任何脚本会自动消费这些 SVG 重建 PPTX**；PPTX 与 `PPT_CONTENT.md` 由 `python tools/update_defense.py` 统一维护」。

`09_results.svg` 的标题已改为与 PPTX 一致的口径：「为什么还不能确认是提升」（原「为什么说是噪声」），页头也改成「8 组结果差距很小，尚不能确认稳定提升」。**通过。**

### 4.3 t14：`tools/selfcheck.py --json` 指向仓库外路径

```powershell
python tools/selfcheck.py --json "$env:TEMP\selfcheck_outrepo.json"
```

```
合计 34 项：PASS 33 / FAIL 1
报告已写入 C:/Users/14675/AppData/Local/Temp/selfcheck_outrepo.json     [exit=0]
```

仓库外文件被正常创建（8,622 B，JSON 合法，`summary={'total':34,'pass':33,'fail':1}`）→ **不崩溃，通过**（FAIL 1 = 上文 V1 的 `paths`，与 `--json` 路径无关）。

```powershell
python tools/selfcheck.py --strict --only rep.verify onnx.consistency onnx.realimg
```

```
onnx.consistency  PASS  ['repvit_m0_9_in1k:1.0', 'repvit_m0_9_pet37:1.0', 'repvit_m1_0_in1k:1.0'] 全部 >=0.99，共 3 个模型
onnx.realimg      PASS  数据来源={'imagenet_val_subset', 'pet_test'}
rep.verify        PASS  BN 107->0 max|Δlogits|=7.092952728271484e-06 Top-1一致=1.0 权重完整=True 整体判定=True
合计 3 项：PASS 3 / FAIL 0     [exit=0]
```

**通过。**

---

## 5. 复核后的最终判定

| # | 验收项 | 结论 |
|---|---|---|
| 1 | 全仓重扫：logits 误差数字唯一归属；无并列结论句 | **passed**（juxtaposition hits = 0） |
| 2 | B1 → `consistency_repvit_m0_9_pet37.json`（6.198883056640625e-06 / 1.5006899711048998e-06 / 100%/100%）；B4 → `repvit_m0_9_pet37_reparam_report.json`（7.092952728271484e-06 / 2.030726818702533e-06 / 32/32 / BN 107→0） | **passed** |
| 3 | 5.25e-06 在 README/REPORT/LOGITS_AUDIT/PPTX+备注中口径一致且不超证据 | **passed**（三份文档该段哈希同一：`7f6bcccfb52b7a3a…`） |
| 4 | 两条复跑命令实跑且与入库值一致 | **passed**（float64 逐位相等） |
| 5 | 产物未被篡改（三 JSON 恰好 2 行；reparam/logs 未改写） | **passed** |
| 6 | PPTX 页数 10–15；5 条原生曲线；四项可视化全部出现 | **passed**（15 页；8 个原生 chart；grid8/grid4/grid5/每类 F1 均以 SHA1 命中落盘文件） |
| 7 | P03+P04 合并页覆盖两环节 + 备注写明合并 + 下标未串位 | **passed** |
| 8 | 性能图表元信息 7 项齐全 | **passed** |
| 9 | PPTX 关键数字可溯源 + PPT_CONTENT 一致 + PDF 页数一致 | **failed**（实验数字与 PPT_CONTENT/PDF 页数均通过；但「37 条验收项自检 0 失败 / 代码零写死绝对路径 / paths 检查 0 处」与 `python tools/selfcheck.py --stage skeleton` 的实际输出 `paths FAIL 1 处` 矛盾，见 V1/V2） |

**verdict = needs_revision**：核心口径与可视化工作全部通过，但交付物里存在一条可一行命令证伪、且被导出进 `report.pdf`/`REPORT.docx`/PPTX 的自检断言。建议先修 V1（1 行级；推荐让 `paths` 规则不再命中检测器自身的字面量，或按实际输出改写文案），再顺手收口 V2/V3/V4，然后重跑 `python tools/selfcheck.py --stage skeleton` 与 `python tools/export_defense_pdf.py` 并复验即可放行。

## 附：本次验证实际执行的命令清单

```powershell
# 仓库状态 / 差异
git status --porcelain=v1
git diff --numstat -- outputs/metrics/consistency_repvit_m0_9_pet37.json outputs/metrics/consistency_repvit_m0_9_in1k.json outputs/metrics/consistency_repvit_m1_0_in1k.json
git diff -- outputs/metrics/
git status --porcelain -- outputs/reparam outputs/logs outputs/figures outputs/benchmarks
git diff --numstat -- report/ppt_svg

# D1：逐提交取证（全部 14 个提交）
git -C <repo> grep -l -e "5\.25e-0*6" <commit> -- outputs/        # ×14，全部空
git -C <repo> grep -l -e "6\.198883056640625e-06" <commit> -- outputs/   # ×14，均命中 metrics JSON

# 实跑复现（写入 tmp_verify，不覆盖入库产物）
python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --images datasets/lists/pet_test.txt --limit 12 --out outputs/verification/tmp_verify/consistency_n12.json
python tools/reparam_verify.py --model repvit_m0_9_pet37 --weights checkpoints/baseline_best.pt --num-samples 32 --batch-size 8 --seed 20240912 --skip-onnx --out-dir outputs/verification/tmp_verify/reparam

# PPTX / PDF / DOCX / SVG
python $env:TEMP\t9verify\chk_pptx.py          # python-pptx：页数、原生图表、图片 SHA1、备注
python $env:TEMP\t9verify\chk_img.py           # 内嵌图 SHA1 → 落盘文件反查
python $env:TEMP\t9verify\chk_xml.py           # 原始 OOXML：禁用词/必备字面量/图表部件数
python $env:TEMP\t9verify\chk_pdf.py           # report.pdf(37p) / 答辩PDF(15p) / REPORT.docx
python $env:TEMP\t9verify\chk_content2.py      # PPT_CONTENT.md × PPTX + SVG × PPTX 逐字对齐
python $env:TEMP\t9verify\chk_scan.py          # 全仓 logits 误差字面量普查
python $env:TEMP\t9verify\chk_juxta.py         # 并列结论句 / 禁用短语扫描
python $env:TEMP\t9verify\chk_numbers.py       # PPTX 关键数字 → outputs/ 落盘对照
Select-String -Path report\ppt_svg\*.svg -Pattern 'BN 108|108 → 0|7\.1e-06|完全等价'   # 0 命中

# 自检工具（t14 + V1）
python tools/selfcheck.py --json "$env:TEMP\selfcheck_outrepo.json"
python tools/selfcheck.py --stage skeleton
python tools/selfcheck.py --strict --only rep.verify onnx.consistency onnx.realimg
```

> 说明：验证脚本本体一律写在 `$env:TEMP\t9verify\`，只有本报告写入 `report/VERIFICATION_LOGITS_PPT.md`；复跑产物写在 `outputs/verification/tmp_verify/`。未执行任何 `git commit` / `git push`。

---

# 第 2 轮：V1–V6 修复复核 + 第 1 轮已通过项回归

- 任务：t19（attempt_id `0f04cab8-49a3-4155-9b86-3bc347f38a4b`）
- 验证时点：**2026-09-16 19:01 – 19:05**（前置 t16 / t15 / t17 / t18 / t21 均已完成）
- 被验证修订的 SHA256（本轮核对的精确版本）：
  - `report/答辩PPT_RepViT.pptx` = `…`（8,118,258 B，18:56:15，15 页）
  - `report/答辩PPT_RepViT.pdf`（1,116,231 B，18:56:29，15 页）
  - `report.pdf`（1,370,508 B，19:00:26，37 页）
  - `report/REPORT.docx`（76,628 B，18:59:59）
  - `report/PPT_CONTENT.md`（26,844 B，18:56:15）
  - `report/REPORT.md`（63,861 B，18:59:16）｜`README.md`（27,156 B，18:59:49）｜`PROGRESS.md`（25,470 B，18:59:45）
  - `tools/audit_logits_references.py`（25,334 B，18:53:52）｜`outputs/metrics/selfcheck_report.json`（8,549 B，18:56:21，SHA256 `BC332941…A23FFE8`）

## R2-0. 结论（verdict = pass）

**9 条验收项全部 passed。** 第 1 轮的 blocker V1 与 findings V2–V6 逐条收敛；第 1 轮已通过的 8 项在修复后**无回归**（并用第 2 次实跑复现再次确认 float64 逐位相等）。本轮**未发现新的阻塞问题**。

## R2-1. 【passed】V1 — selfcheck `paths` 回到 PASS / 0 处，全量自检 34/34

```powershell
Set-Location <repo>
python tools/selfcheck.py --stage skeleton --json "$env:TEMP\t19_skeleton.json"
python tools/selfcheck.py --json "$env:TEMP\t19_full.json"          # 全量
```

原始输出（`--stage skeleton`）：

```
ID        STATUS  DETAIL
paths     PASS    0 处
skeleton  PASS    33 项全部就位
合计 2 项：PASS 2 / FAIL 0
```

JSON（`%TEMP%\t19_skeleton.json`）：`summary {'total': 2, 'pass': 2, 'fail': 0}`、`paths PASS '0 处'`、`skeleton PASS '33 项全部就位'`。

全量（**读 JSON，不看 exit code**）：

```
合计 34 项：PASS 34 / FAIL 0
file: C:\Users\...\Temp\t19_full.json  bytes: 8549  generated_at: 2026-09-16T19:01:46
summary: {'total': 34, 'pass': 34, 'fail': 0}
checks count: 34
non-PASS: 0
  [paths] PASS :: 0 处
  [readme] PASS :: 12 项全部命中
  [skeleton] PASS :: 33 项全部就位
  [rep.verify] PASS :: BN 107->0 max|Δlogits|=7.092952728271484e-06 Top-1一致=1.0 权重完整=True 整体判定=True
```

修复有效性（不是「关掉检测」）：`tools/audit_logits_references.py:430` 现为 `if ABS_PATH.search(text):`，删掉的只是冗余的 `or 'C:\\Users' in text`；`ABS_PATH = re.compile(r'[A-Za-z]:[\\/]{1,2}Users', re.I)` 仍能命中 4 种真实写法并排除负例：

```
POS True 'C:\Users\14675\x' / True 'C:\\Users\\14675\\x' / True 'C:/Users/14675/x' / True 'c:\users\14675\x'
NEG False 'https://example.com/x' / False 'no path here' / False 'D:/data/x' / False 'C:\Windows\x'
```

入库报告未被本轮触碰：`outputs/metrics/selfcheck_report.json` mtime 仍为 **18:56:21 / 8,549 B / SHA256 `BC332941EBB5E6178CDDD91A2ACA2CBEA8A14C9ED9F703B4637E9FF30A23FFE8`**，`summary {'total': 34, 'pass': 34, 'fail': 0}`、`checks = 34`、无非 PASS 项 —— 与 t15 的落盘一致，也说明本轮两次自检都按要求写到了 `%TEMP%`。

## R2-2. 【passed】V2 —「37 条验收项」全仓归零，「37 类」未被误伤

对 24 个载体做**去空白/去 Markdown 标记的归一化**扫描（避免换行折断导致漏检），模式为 `37条验收项 / DoD37 / DoD#37 / 37条DoD`：

```
[OK ] README.md / PROGRESS.md / report/REPORT.md / report/PPT_CONTENT.md / report/LOGITS_AUDIT.md   BAD={}
[OK ] report/ppt_svg/*.svg（15 份，去标签后）                                                        BAD={}
[OK ] PPTX(slides+notes)   BAD={}   GOOD={'34条验收项': 2}
[OK ] report.pdf           BAD={}   GOOD={'34条DoD': 1}
[OK ] 答辩PPT_RepViT.pdf   BAD={}   GOOD={'34条验收项': 2}
[OK ] report/REPORT.docx   BAD={}   GOOD={'34条DoD': 1}
```

- `@check` 计数 = **34**（`tools/selfcheck.py`），与文案一致；`PROGRESS.md` = `DoD 34 条`、`report/REPORT.md:1012` = `34 条 DoD 验收项`。
- PPTX 实际文案：P02「六个考核环节全部交付，**34 条验收项**自检 0 失败」、P15「**34 条验收项**自检 0 失败；代码零写死绝对路径；关键数字与实验口径见 outputs/ 和审计说明」；`PPT_CONTENT.md` / 答辩 PDF 同。
- 全仓唯一仍含字面量 `37 条验收项` 的地方是 `tools/update_defense.py` 的**替换表条目** `("37 条验收项", "34 条验收项")`（生产机制，不是断言），成品与文档 0 命中。
- **「37 类」未误改的证据（逐条列出所有含「37」的行）**：README.md 39 行、report/REPORT.md 52 行、PPTX 全部 15 页正文+备注、15 份 SVG 全部逐行复核 —— 均为合法用法：`Pet 37 类`、`Pet-37`、`37 类与 1000 类`、`1000 类 → 37 类`、`num_classes=37`、`reset_classifier(37)`、`37 维`、`37/37 = 100%`、`37×37`、`37.5%`、`3669 行`、`7.37 ms` 等；无一条被改成 34。答辩 PDF 第 2 页同时含「34 条验收项」与卡片 02「Pet 37 类 · 40 epoch」。

## R2-3. 【passed】V3 — §16.1 指标对齐落盘；`:464` 的重跑说明未被误改

```powershell
python -c "import json;d=json.load(open('outputs/metrics/baseline_test.json'));print(d['top1'],d['macro_f1'],d['num_samples'],d['eval_count'])"
# 0.9234123739438539 0.9221970249315374 3669 1
```

`report/REPORT.md:989-992`（现行）：

```
2. **迁移到 Pet 37 类效果良好**：test Top-1 = **92.34%**、Macro-F1 = **92.22%**
   （唯一落盘 `outputs/metrics/baseline_test.json`：`top1 = 0.9234123739438539`、
   `macro_f1 = 0.9221970249315374`、`num_samples = 3669`、`eval_count = 1`），
   且模型几乎不混淆猫狗（跨物种错误仅占 **3.56%**，10 / 281），全部错误集中在细粒度品种之间。
```

未误改：`REPORT.md:464` 仍是「同一份配置（baseline）重跑一次的 test Top-1 从 **92.26 变到 92.34**（**±0.08 个点**）」。全仓扫描 `92.26` 仅 3 处且同源：`REPORT.md:464`、`report.pdf` 第 16 页、`REPORT.docx` 同一句；`92.14` 全仓 0 命中。

## R2-4. 【passed】V4 — 「8/8 说明没有过拟合」已消失，与 PPTX P10 口径一致

从 PPTX slide 10 备注逐字提取的定稿口径：**「8/8 判对只是小样本观察，不能证明跨域泛化或没有过拟合」**。`report/REPORT.md` 两处已改为同一口径：

- `:642-643`：「**8/8 判对只是小样本观察，不能证明跨域泛化或没有过拟合**；本节的观察只覆盖这 8 张已测跨集合图片，不构成对全部外部图片的性能声明。」
- `:1003-1004`：「5. **跨集合泛化：8/8 判对只是小样本观察，不能证明跨域泛化或没有过拟合**（口径：8 张 ImageNet 跨集合实拍图，逐张登记来源与许可）。」

全仓扫描 `8/8 全部判断正确` = 0；`没有过拟合` 仅剩上述两处**否定式**表述。`report.pdf` / `REPORT.docx` 中该句已同步出现（规范化匹配 `8/8判对只是小样本观察，不能证明跨域泛化或没有过拟合` = True）。

## R2-5. 【passed】V5 — P04 来源补全，5 个 MACs 值逐个有出处

PPTX P04 页脚（现行，逐字）：
`数据来源：outputs/pretrained_eval/<model>/metrics.json · outputs/benchmarks/family_summary.csv（M1.1+ 的 MACs）· summary.csv`

逐值溯源（我独立读两份落盘文件比对）：

| 型号 | 页面 MACs (G) | 出处 | 同表其它口径（对照） |
|---|---|---|---|
| M0.9 | 0.847 | `pretrained_eval/repvit_m0_9/metrics.json` `macs_g = 0.8471` | `macs_g_fused 0.8156`、`family macs_G 0.8322` |
| M1.0 | 1.143 | `…/repvit_m1_0/metrics.json` `macs_g = 1.1428` | `1.1061`、`1.1255` |
| M1.1 | 1.358 | `outputs/benchmarks/family_summary.csv` `macs_G = 1.3581` | `1.3371`、`1.3769` |
| M1.5 | 2.308 | `family_summary.csv` `macs_G = 2.3084` | `2.2736`、`2.3397` |
| M2.3 | 4.574 | `family_summary.csv` `macs_G = 4.5739` | `4.516`、`4.6263` |

即 5 个值**每一个都精确等于两个来源之一**（M0.9/M1.0 取 metrics.json，M1.1+ 取 family_summary.csv），且来源说明已把两个文件都列出。答辩 PDF 第 4 页同步含 `family_summary.csv`。

## R2-6. 【passed】V6 — 跨物种错误数全仓统一为 281 / 10 / 3.56% / 271

落盘（唯一）：`outputs/confusion_matrix/baseline_cat_dog_block.json` → `n_error=281`、`n_cross_species_error=10`、`n_within_species_error=271`、`cross_species_error_ratio=0.03558718861209965`（= 10/281 → 3.56%）。

全仓一致性：`report/REPORT.md:555` = `**3.56 %（10 / 281 个错误）**`、`:556` = `271 个`、`:558` = `3.56%`、`:992` = `3.56%，10 / 281`（不再自相矛盾）；`README.md:287-288` = `281 个错误里只有 10 个（3.56%）是跨物种…其余 271 个`；PPTX P02/P15 = `跨物种错误仅 3.56%`、P09 = `281 个错误中，10 个跨猫狗物种（3.56%），其余 271 个是同物种品种判断错误`。
旧三元组（`3.17%` / `284 个` / `275 个`）全仓扫描只剩 1 处，位于**本报告第 1 轮的 V6 记录**（引用语境），成品与其它文档 0 命中。

## R2-7. 【passed】第 1 轮已通过项回归（未被本轮修复放松）

| 回归项 | 本轮复核结果 |
|---|---|
| 两类误差口径分离 | `TOTAL juxtaposition hits = 0`（重跑第 1 轮扫描脚本） |
| B1/B4 落盘值 | `6.198883056640625e-06` / `1.5006899711048998e-06`；`7.092952728271484e-06` / `2.030726818702533e-06` / 32-32 / BN 107→0，逐字段与落盘一致 |
| 5.25e-06 三处一致 | README / LOGITS_AUDIT / REPORT 三处 marker 各 1，`n8-claim=False`；行哈希仍 `7f6bcccfb52b7a3a` |
| 三个 consistency JSON 恰好 2 行且数值未动 | `git diff --numstat` 三个文件仍 `2 2`；`2-line-revert == HEAD bytes` 三份均 True；键序同 HEAD；`outputs/reparam`+`outputs/logs` 的 `git status` 为空，`repvit_m0_9_pet37_reparam_report.json` mtime 仍是 2026-09-14 00:07:32 |
| PPTX 页数 10–15 | `slides = 15` |
| 五项原生曲线 | P08 `charts=5`，类型全为 `LINE (4)`；总 chart 部件 8（P08=5 / P12=1 / P14=2） |
| 四项必需可视化 | 内嵌图 SHA1 反查落盘文件全部命中：`baseline_per_class_f1.png`(P9)、`compare_baseline_vs_opt_combo_grid4.png` / `test_top5_baseline_grid8.png` / `external_top5_pet37_grid5.png`(P11) |
| P03+P04 合并页 | 左栏标记 `左栏 · 环节 2` = True、右栏 `右栏 · 环节 3` = True、备注 `【本页是合并页】` + `题干 15 个建议环节` 均在 |
| 性能图表元信息 | P08 6/6、P09 8/8、P12 8/8、P13 7/7、P14 7/7 字段命中 |
| PPT_CONTENT.md 与 PPTX 一致 | P01–P15 逐行比对（正文+备注）`TOTAL missing = 0` |
| 答辩 PDF 与 PPTX 页数一致 | PPTX 15 = PDF 15 |
| PPTX 禁用词 | `完全等价 / 7.1e-06 / 九类 / 108 → 0 / BN 108 / 2.265e-06 / 4.108e-07 / 5.245e-06 / 1.414e-06 / 6.53e-06 / 6.527e-06` 全 0 命中；`六个 ONNX`/`全部六` 命中均为否定语境 |
| SVG 与 PPTX 逐字对齐 | 01_cover 5/5、02_status 4/4、05_pretrained 1/1、07_baseline 2/2、12_reparam 3/3、15_summary 1/1；t21 改的 3 行（02_status / 15_summary 的「34 条验收项」、05_pretrained 的 family_summary.csv 页脚）**两处同时逐字命中** |

## R2-8. 【passed】两条实跑复现（第 2 次），float64 仍然逐位相等

```powershell
python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --images datasets/lists/pet_test.txt --limit 12 --out outputs/verification/tmp_verify_r2/consistency_n12.json
python tools/reparam_verify.py --model repvit_m0_9_pet37 --weights checkpoints/baseline_best.pt --num-samples 32 --batch-size 8 --seed 20240912 --skip-onnx --out-dir outputs/verification/tmp_verify_r2/reparam
```

```
max|Δlogits|=6.199e-06  mean|Δlogits|=1.501e-06  Top-1 一致率=100.00%  Top-5 集合一致率=100.00%  verdict=PASS   [exit=0]
[logits] max_abs_err=7.093e-06 mean_abs_err=2.031e-06 top1_same=32/32 top5_min_overlap=5   [verdict] PASS        [exit=0]
```

逐位比较：

```
field-wise equal: True
max_abs_logits : bit-exact=True hex=3eda000000000000
mean_abs_logits: bit-exact=True hex=3eb92d6a12aaaaab
top1_agree_rate / top5_set_agree_rate: bit-exact=True hex=3ff0000000000000
max_abs_err / mean_abs_err / top1_same_rate / num_samples: equal=True bit-exact=True
judge equal: True | diff equal: True | n_bn before/after: 107 -> 0
```

产物未被覆盖：`outputs/reparam` 与 `outputs/logs` 的 `git status` 为空；本轮复跑写在**新目录** `outputs/verification/tmp_verify_r2/`（第 1 轮的 `tmp_verify/` 保持原样，两轮证据可分别查看）。

## R2-9. 【passed】三份导出物均为最新版（mtime + 内容双重确认）

| 文件 | mtime | 大小 | 页数 | 被改段落在其中的确认 |
|---|---|---|---|---|
| 根目录 `report.pdf` | **19:00:26**（晚于 `REPORT.md` 18:59:16） | 1,370,508 B | 37 | 含 `34条DoD`、`92.34%`、`92.22%`、`3.56%`、`10/281`、V4 否定式句、`92.26变到92.34`、`family_summary.csv`；`37条验收项`＝0；`92.26` 仅第 16 页那条重跑说明 |
| `report/REPORT.docx` | **18:59:59**（晚于 `REPORT.md` 18:59:16） | 76,628 B | — | 同上 6 项全部命中；`37条验收项`＝0；`92.26` 仅同一条重跑说明 |
| `report/答辩PPT_RepViT.pdf` | **18:56:29**（晚于 `PPTX` 18:56:15） | 1,116,231 B | 15 = PPTX | P02/P15 已是「34 条验收项」且卡片 02 仍为「Pet 37 类 · 40 epoch」；P04 含 `family_summary.csv（M1.1+ 的 MACs）`；封面仍为「结构重参数化（32 随机输入）max|Δ|」+ `7.093e-06` |

（第 1 轮记录的旧版对照：`HEAD`/`81693dd` 的 `report.pdf` = 1,421,842 B / 32 页；现在 1,370,508 B / 37 页，确为新导出版本。）

## R2-10. 第 2 轮遗留（不阻塞放行）

| ID | 严重度 | 内容 | 状态 |
|---|---|---|---|
| R2-a | info | `tools/update_defense.py` 的替换表仍以 `"37 条验收项"` 作为**查找键**（机制，不是断言）。若日后有人误把该键当成文案扫描对象，可能产生假阳性。 | 无需修改；已在 R2-2 说明 |
| R2-b | info | `report/ppt_svg/13_onnx.svg` 仍只有 Pet-37 一行模型表（PPTX P13 表为 3 型号）、`12_reparam.svg` 标题措辞与 PPTX 不同；`ppt_svg` 已由 `report/ppt_svg/README.md` 声明为合并前设计源、以 PPTX 为准。 | 第 1 轮 V6 已登记，按船长更正 3 不判缺陷 |
| R2-c | info | 第 1 轮 V5 的「P04 MACs 列混用两个来源」已通过**补全来源标注**解决；源头两种口径（`metrics.macs_g` vs `family_summary.macs_G`）仍并存于仓库，属既有数据现状。 | 不阻塞 |
| R2-d | info | 本轮新增未跟踪目录 `outputs/verification/tmp_verify/`（第 1 轮）与 `outputs/verification/tmp_verify_r2/`（第 2 轮）为验证复跑留痕，推送前由收口任务决定删除或加入 `.gitignore`。 | 待收口 |

## R2-附：第 2 轮实际执行的命令

```powershell
python tools/selfcheck.py --stage skeleton --json "$env:TEMP\t19_skeleton.json"
python tools/selfcheck.py --json "$env:TEMP\t19_full.json"
python $env:TEMP\t9verify\r2_v2.py        # V2：24 载体归一化扫描 + 全部含「37」的行
python $env:TEMP\t9verify\r2_v2346.py     # V2/V3/V4/V6 行级取证 + 各 SVG 扫描
python $env:TEMP\t9verify\r2_v456.py      # V4(PPTX) / V5 逐值溯源 / V6(README+PPTX)
python $env:TEMP\t9verify\r2_exports.py   # 三份导出物 mtime + 内容 + 页数
python $env:TEMP\t9verify\chk_juxta.py    # 回归：并列结论句 0 命中
python $env:TEMP\t9verify\chk_d1_d2.py    # 回归：5.25e-06 唯一文本 + JSON 恰好 2 行
python $env:TEMP\t9verify\chk_content2.py # 回归：PPT_CONTENT × PPTX + SVG × PPTX
python $env:TEMP\t9verify\chk_img.py      # 回归：内嵌图 SHA1 → 落盘文件
python $env:TEMP\t9verify\chk_xml.py      # 回归：OOXML 禁用词 + 图表部件数
python $env:TEMP\t9verify\r2_regress.py   # 回归：逐位复跑比较 + PPTX 结构 + 合并页 + 元信息
python $env:TEMP\t9verify\r2_final.py     # 回归：t21 三行 SVG × PPTX + 页数一致

# 两次实跑复现（写入新目录，不覆盖入库产物）
python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --images datasets/lists/pet_test.txt --limit 12 --out outputs/verification/tmp_verify_r2/consistency_n12.json
python tools/reparam_verify.py --model repvit_m0_9_pet37 --weights checkpoints/baseline_best.pt --num-samples 32 --batch-size 8 --seed 20240912 --skip-onnx --out-dir outputs/verification/tmp_verify_r2/reparam
```

> 第 2 轮同样只写了 `report/VERIFICATION_LOGITS_PPT.md`（追加本节）与 `outputs/verification/tmp_verify_r2/`；两次自检都带 `--json %TEMP%`，**未再触碰** `outputs/metrics/selfcheck_report.json`（mtime 仍 18:56:21）。未执行任何 `git commit` / `git push`。
