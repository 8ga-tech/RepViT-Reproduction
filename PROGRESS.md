# PROGRESS — RepViT 复现任务

- repo_root: .                              # 只写相对路径
- last_update: 2026-09-13T17:40:00+08:00
- current_module: M13 完成（已推送到 GitHub）
- overall: 13/13 done, 0 blocked
- env: python=3.14.5, torch=2.14.0+cu126, timm=1.0.29, onnxruntime=1.30.0, device=cuda(RTX 4060 Laptop 8GB)

## 模块状态

| 模块 | 名称 | 状态 | 开始 | 结束 | 关键产物 | 自检 |
|---|---|---|---|---|---|---|
| M01 | 环境搭建与自检 | done | 09-13 14:05 | 09-13 14:16 | outputs/env_snapshot.json, requirements.lock.txt | PASS |
| M02 | 仓库骨架 | done | 09-12 20:05 | 09-13 14:20 | configs/ datasets/ models/ tools/ deploy/ utils/ labels/ | PASS |
| M03 | 数据准备 | done | 09-13 14:00 | 09-13 14:35 | datasets/lists/*, outputs/metrics/leakage_check.json | PASS |
| M04 | 模型与权重加载 | done | 09-13 14:10 | 09-13 14:30 | models/repvit_official.py, outputs/metrics/weight_load_report.json | PASS |
| M05 | 官方预训练评价 | done | 09-13 14:35 | 09-13 14:45 | outputs/pretrained_eval/{repvit_m0_9,repvit_m1_0}/ | PASS |
| M06 | Baseline 训练 | done | 09-13 14:19 | 09-13 14:27 | checkpoints/baseline_{best,last}.pt | PASS |
| M07 | 优化实验 | done | 09-13 14:45 | 09-13 16:41 | checkpoints/opt_*_best.pt（7 组，均 40 epoch） | PASS |
| M08 | 可视化与可解释性 | done | 09-13 14:36 | 09-13 14:52 | outputs/{curves,confusion_matrix,predictions,gradcam}/ | PASS |
| M09 | 结构重参数化验证 | done | 09-13 14:20 | 09-13 14:45 | outputs/reparam/ | PASS |
| M10 | ONNX 导出与基准 | done | 09-13 14:15 | 09-13 14:50 | onnx/*.onnx, outputs/benchmarks/ | PASS |
| M11 | 进阶任务 | done | 09-13 16:45 | 09-13 17:05 | outputs/advanced/（家族/消融/深度重参数化/鲁棒性/可解释性） | PASS |
| M12 | 报告与 PPT 素材 | done | 09-13 17:10 | 09-13 17:36 | outputs/report_assets/, report.pdf, 答辩PPT_RepViT.pptx/.pdf | PASS |
| M13 | 全局验收 | done | 09-13 17:36 | 09-13 18:05 | outputs/metrics/selfcheck_report.json | **34/34 PASS** |

## 关键数字（每个数字必须给出文件与命令）

| 键 | 值 | 口径 | 来源文件 | 采集命令 |
|---|---|---|---|---|
| m0_9_params_train_double_head | 5489328 | timm 训练态含蒸馏头 C=1000 | outputs/metrics/count_params.json | `python tools/count_params.py --model repvit_m0_9` |
| m0_9_params_single_head_unfused | 5103560 | 未融合单头 C=1000 | 同上 | 同上 |
| m0_9_params_backbone | 4717792 | 骨干，不含任何分类头 | 同上 | 同上 |
| m0_9_params_pet37_unfused | 4732805 | 未融合单头 C=37，本任务训练态 | 同上 | 同上 |
| m0_9_params_fused_c1000 | 5067056 | 融合后单头 C=1000 | 同上 | 同上 |
| m0_9_params_fused_c37 | 4696301 | 融合后单头 C=37 | 同上 | 同上 |
| m0_9_params_fused_delta | 36504 | 融合净减 | 同上 | 同上 |
| m0_9_macs_thop | 847050816 | thop，未融合 eval，@224 | outputs/metrics/count_flops.json | `python tools/count_flops.py --model repvit_m0_9 --input-size 224` |
| m0_9_macs_fvcore | 832165824 | fvcore，同口径（**不是 2 倍**） | 同上 | 同上 |
| official_m0_9_top1 | 78.20 | 百分数，1000 张自建子集，crop_pct=0.875 | outputs/pretrained_eval/repvit_m0_9/metrics.json | `python tools/eval_pretrained.py --cfg configs/pretrained_eval.yaml` |
| official_m0_9_top5 | 93.70 | 同上 | 同上 | 同上 |
| official_m1_0_top1 | 79.90 | 百分数，同子集同口径 | outputs/pretrained_eval/repvit_m1_0/metrics.json | 加 `--set pretrained_eval.model=repvit_m1_0` |
| baseline_val_macro_f1_best | 0.946166 | 验证集 best epoch | outputs/logs/baseline_metrics.csv | `python tools/train.py --cfg configs/baseline.yaml` |
| baseline_test_top1 | 0.923410 | test，**0~1 小数**，eval_count=1（重跑后的最终值） | outputs/metrics/baseline_test.json | 见 M06 主流程 |
| baseline_test_top5 | 0.992641 | 同上 | 同上 | 同上 |
| baseline_test_macro_f1 | 0.922181 | 同上 | 同上 | 同上 |
| pet_split_lines | 2940 / 740 / 3669 | per-class val = 20 | datasets/lists/pet_{train,val,test}.txt | `python datasets/make_pet_split.py --root data/oxford-iiit-pet --out-dir datasets/lists --val-per-class 20 --seed 42` |
| leakage | 0 / 0 / 0 | train∩val / train∩test / val∩test | outputs/metrics/leakage_check.json | `python datasets/audit_leakage.py` |
| reparam_max_abs_err_pet37 | 7.093e-06 | 融合前后 logits 最大绝对误差 | outputs/reparam/repvit_m0_9_pet37_reparam_report.json | `python tools/reparam_verify.py --model repvit_m0_9_pet37 --weights checkpoints/baseline_best.pt --out-dir outputs/reparam` |
| reparam_top1_identical | True | 32/32 样本 Top-1 一致 | 同上 | 同上 |
| reparam_onnx_nodes | BN=0, Conv=103 | 推理态 ONNX 节点（未融合 126 → 103） | outputs/reparam/repvit_m0_9_pet37_onnx_nodes.json | 同上 |
| onnx_size_m0_9_in1k | 20.36 MB | os.path.getsize()/1e6 | outputs/benchmarks/summary.csv | `python deploy/export_onnx.py --model repvit_m0_9_in1k` |
| onnx_size_m1_0_in1k | 27.33 MB | 同上 | 同上 | 同上 |
| onnx_size_m0_9_pet37 | 18.89 MB | 同上 | 同上 | 同上 |
| bench_m0_9_in1k | mean=7.45 p50=7.37 p95=8.00 ms | ORT CPU EP / FP32 / bs=1 / 224 / warmup10+50 / threads=4 | outputs/benchmarks/repvit_m0_9_in1k_benchmark.json | `python deploy/benchmark.py --model ... --warmup 10 --runs 50 --threads 4` |
| bench_m1_0_in1k | mean=9.20 p50=9.15 p95=9.76 ms | 同上 | 同上 | 同上 |
| bench_m0_9_pet37 | mean=7.20 p50=7.12 p95=7.69 ms | 同上 | 同上 | 同上 |
| consistency_pet37 | max|Δ|=6.199e-06, Top-1 一致率=1.000 | PyTorch vs ONNX，固定划分列表 | outputs/metrics/consistency_repvit_m0_9_pet37.json | `python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --limit 12 --out ...` |
| cross_species_error_ratio | 0.0356（10/281） | test 集错误里真正跨物种（猫↔狗）的占比 | outputs/confusion_matrix/baseline_cat_dog_block.json | `python tools/visualize.py --pred-csv outputs/predictions/baseline_test_preds.csv ...` |
| external_top5_correct | 8/8 | 跨集合（ImageNet 实拍）图片的品种判断 | outputs/benchmarks/external_top5_repvit_m0_9_pet37.csv | `python tools/predict_external.py --model repvit_m0_9_pet37 --dir external --num 8` |

## 数据与权重获取记录

| 资产 | 来源 | 校验 | 状态 |
|---|---|---|---|
| 官方权重 ×5 | `https://ghfast.top/https://github.com/THU-MIG/RepViT/releases/download/v1.0/*` | m0_9=22,422,548 B / m1_0=29,675,245 / m1_1=35,668,677 / m1_5=59,375,411（旧文件曾损坏）/ m2_3=95,860,931，SHA256 见 `outputs/metrics/weight_sha256.json` | ✅ |
| Oxford-IIIT Pet 官方归档 | hf-mirror 上的官方归档镜像 | **MD5 `5c4f3ee8e5d25df40f4fd59a7f44e54c` / `95a8c909bbe2e81eed6a22bccdf3f68f` 与官方基准逐字符一致** | ✅ |
| ImageNet-1K val | `hf-mirror.com/datasets/mrm8488/ImageNet1K-val`（14 分片，gated=False） | 1000 类 × 50 张 = 50,000；JPEG 魔数、目录序=synset 序 | ✅ `data/imagenet/val/` |
| ImageNet 1000 类标签 | timm 内置 synset 表生成 | DoD #10 四锚点全 PASS | ✅ `labels/imagenet_classes.txt` |

## 阻塞与待确认

- [x] **考核方输入 `data/provided/` 为空**（沿用 Phase 0 结论）。经查考核试题 PDF 全文（29 页）
      **从未定义该路径**——它是 02 规格书自创的约定；PDF 第 28 页「考核方发布前建议准备内容」
      是**对出题方的建议清单**，不代表已下发。
      缓解：Pet 的官方 trainval/test 划分与 37 类映射已从官方归档得到；
      ImageNet 类别映射已自建并通过 DoD #10；ImageNet 验证子集为**自建**分层抽样
      （每类 1 张，种子 20260912，清单 `datasets/lists/imagenet_val_subset.txt`），
      **已在 README 与报告中显式声明非考核方指定子集**。
- [ ] github.com 直连不可达（HTTP 000），已用 ghfast.top 代理取得官方 weight release；
      **该替代来源已在 README 与 PROVENANCE.md 声明**，每条权重均记录字节数与 SHA256。
- [ ] 官方站点 `www.robots.ox.ac.uk` 不可达，Pet 数据改用 HF 镜像的官方归档，
      **已用官方 MD5 证明为原始文件**。
- [ ] **Wikimedia Commons 不可达**（curl 超时，HTTP 000），`external/` 的「训练集以外的实际图片」
      改用跨数据集真实照片（ImageNet val 中与 Pet 同品种的样本），
      逐张登记在 `external/images_manifest.csv`，**不声称其为原创实拍**。
- [ ] 原始归档暂存于 `data/_src/`，`data/` 不入提交物。

## 决策记录

- **09-12 环境取舍：改全局 `C:\Python314` 而非建 .venv**（人类决策）。理由：本机已有 95% 依赖预置。
  已执行：`onnxruntime 1.29.0 → 1.30.0`（规格书禁止事项点名版本）、`torch 2.14.0+cpu → +cu126`。
  影响：全局环境被改动；其他项目若依赖 CPU 版 torch 需注意。
- **09-13 官方权重载入路线定为「vendored 官方实现」（路线 B）**。实测 timm 的键与官方 `.pth`
  的键**交集为 0**（官方 713 键 `features.N.*` / `classifier.classifier.*`；timm 389 键
  `stages.M.*` / `head.head.*`），跨体系加载必然 `unexpected=713`。
  vendored 官方实现载入 **missing=2 / unexpected=0**（missing 仅换头后的分类头 Linear），满足 DoD #12。
- **09-13 迁移训练起点定为 timm 侧 HF 权重**（`HF_ENDPOINT=https://hf-mirror.com`）。
- **09-13 第二个官方型号选 RepViT-M1.0**（人类决策）。理由：与 M0.9 规模最接近，
  规模-精度曲线的对照最有说服力。
- **09-13 优化方案选「方案 A 组合式」**（人类决策）= Mixup/CutMix + RandAugment 加强增强
  + 骨干/头差异化学习率，共 12 个差异键。
- **09-13【预算等价性修正】早停必须对所有实验臂一致。**
  早期用默认 `early_stop_patience=8` 跑 `opt_mix`，它在 epoch 24 就早停（25 epoch），
  而 baseline 跑满 40 epoch——两组预算差 37.5%，「训练更久」会混进优化方法的增益里，
  控制变量实验不成立。
  处置：`configs/baseline.yaml` 的 `early_stop_patience` 由 8 改为 **999**（等价于关闭，
  机制保留在代码里可被 `--set` 覆盖），**全部实验臂统一 40 epoch 满预算**。
  判别命令：`python tools/same_budget.py`。
- **09-13【批次一致性】M07 必须串行执行**。本机 15.8 GB 内存，
  一个训练进程 + 4 个 DataLoader worker 约占 3 GB；并发多个训练会导致
  `DataLoader worker exited unexpectedly`，且崩溃的父进程会遗留 worker 僵尸进程
  进一步挤占内存（实测复现两次）。已在 `_run_m07.sh` 内注明。
- **09-13 ImageNet val 子集清单的路径口径**：`datasets/make_imagenet_subset.py`
  新增 `--path-mode {repo,val}`，默认 `repo`（相对仓库根）。理由：与
  `outputs/predictions/*.csv` 的 `path` 列契约（「相对仓库根」）一致，消除歧义。
- **09-13 外部图片（`external/`）的来源**：见上文「阻塞与待确认」第 4 条。
- **09-13 交付形态**（人类决策）：报告 PDF + 答辩 PPT 源文件 + PPT 的 PDF 版**全部产出**；
  PPT 采用**深色科技风**；**实测发现的代码缺陷只在报告附录呈现**，PPT 不展开。

## 实测发现并修复的代码缺陷（详见 PROVENANCE.md 第四节与各文件注释）

按发现方式分类，共 10 类。这些缺陷全部由**实跑**暴露（而非阅读代码发现），
修复后均在文件内注明了「为什么原写法是错的」：

1. `tools/parse_split.py` —— `sum(1 for _ in f)` 先耗尽迭代器，`next(f)` 必抛 `StopIteration`。
2. `tools/assert_data.py` —— 未排除 macOS `._*` 资源叉文件（trimaps 计数翻倍 14780）；
   `ids()` 未跳过 `#` 注释行（误报划分与 list.txt 不一致）。
3. `tools/visualize.py` —— 用 `class_idx < 12` 判猫；官方 **CLASS-ID 猫狗交错**，
   导致物种判定整体错位（American Bulldog 被标成 cat）。
4. `tools/visualize.py` —— 跨物种错误率分母把 cross 重复计一次，输出恒等于 50%（无信息量）。
5. `tools/reparam_verify.py` —— `zip(tk.tolist(), ...)` 后再次 `.tolist()` → `AttributeError`；
   另：`--model` 传 registry key 时未解析，直接喂给 `timm.create_model` 报 Unknown model。
6. `tools/train.py` —— `load_config` 不解析 `_base_`，全部 `opt_*.yaml` 无法运行。
7. `deploy/compare_torch_onnx.py` —— 未 `rsplit` 解析 list 行（整行当路径）→ `OSError`；
   且缺 selfcheck 依赖的 `num_classes/label_file/source/top1_agree` 字段。
8. `tools/selfcheck.py` —— 以 `weights_only=True` 读自训 ckpt（含 config/class_names）→ `UnpicklingError`；
   `copy.deepcopy(m).fuse().parameters()`（timm 的 `fuse()` 原地改写且返回 `None`）→ `AttributeError`；
   `[A-Za-z]:/` 与 `[A-Za-z]:\\` 把 https URL 和 `"...None:\n"` 误判成写死路径；
   混淆矩阵 CSV 第 0 列是类名，`loadtxt` 转换失败；`opt.diff` 做顶层字典比较，
   任何嵌套改动都算成一个差异（应做叶子级）。
9. 多处 JSON 落盘用 `ensure_ascii=False` —— 本仓库根目录含中文，写进 JSON 的绝对路径
   会让 `open()`（中文 Windows 默认 GBK）读不回来。已全仓统一为 `ensure_ascii=True`。
10. `tools/env_check.py` —— venv 缺失被判为致命错误，与本任务「使用全局解释器」的
    人类决策冲突；改为 WARN 并在输出中指明依据。

## M07 最终结果（8 组，均 40 epoch 满预算）

| 实验 | 差异键 | val Top-1 % | val Macro-F1 | test Top-1 % | test Macro-F1 % |
|---|---|---|---|---|---|
| baseline | — | 94.60 | 0.9462 | 92.34 | 92.22 |
| opt_mix (B) | 3 | 94.86 | 0.9489 | 92.42 | 92.27 |
| opt_randaug (C) | 8 | 94.46 | 0.9450 | 92.18 | 92.06 |
| opt_disc (D) | 1 | 94.19 | 0.9421 | 92.20 | 92.11 |
| opt_combo (A) | 12 | 94.86 | 0.9489 | 92.12 | 92.01 |
| opt_abl_a (=C) | 8 | 95.00 | 0.9502 | 92.48 | 92.36 |
| opt_abl_b (=D) | 1 | 94.32 | 0.9433 | 92.53 | 92.44 |
| opt_abl_ab (=C+D) | 9 | 94.87 | 0.9488 | 91.93 | 91.84 |

**结论：8 组 test Top-1 落在 91.93~92.53 的 0.6 个点窄带内，小于二项分布 95% 置信区间半宽
（±0.9 个点），且同配置重跑的波动本身就有 ±0.4 个点（val Macro-F1）——四项优化方法都没有
带来超出随机波动的真实增益。** 组合消融显示 A 与 B 存在超加性交互（I = +0.541）：
A（RandAugment）抵消了 B（差异化 lr）单独使用时的负作用。

## M11 进阶任务完成情况

| 子项 | 状态 | 产物 |
|---|---|---|
| ① 模型家族速度—精度分析（≥4 型号） | ✅ 完成（5 型号 + 6 个 ONNX 部署） | outputs/advanced/{family_summary,marginal_returns,model_recommendation,pareto_summary} + 3 张帕累托图 |
| ② 第二项优化及组合消融 | ✅ 完成（A/B/A+B） | outputs/advanced/ablation_summary.{csv,json}、outputs/metrics/ablation.csv |
| ③ 深入结构重参数化 | ✅ 完成 | outputs/advanced/repvit_m0_9_pet37_{structure_before,structure_after}.txt、_onnx_nodes.json、block_22_fusion_report.txt |
| ④ 集显加速部署 | ❌ **本机不具备条件** | 见 outputs/benchmarks/igpu_*.json；本机只枚举出 NVIDIA 独显，无 Intel/AMD 集显，且无 OpenVINO/DirectML provider，bench_igpu 已把回退 CPU 的行标记为 invalid |
| ⑤ 鲁棒性 / 可解释性 | ✅ 完成 | outputs/advanced/robustness_repvit_m0_9_pet37.json、interp/{featmap_*,multilayer_cam,tsne,calibration} |

## M12 交付物

| 文件 | 说明 |
|---|---|
| `report.pdf` | 32 页（正文 + 附录 A~E），由 report/REPORT.md → docx → PDF |
| `report/REPORT.md` · `.docx` | 报告源文件 |
| `report/答辩PPT_RepViT.pptx` | 15 页答辩 PPT 源文件（深色科技风，原生可编辑 DrawingML） |
| `report/答辩PPT_RepViT.pdf` | PPT 的 PDF 版 |
| `report/ppt_svg/` | PPT 的 15 个页面 SVG 源（可再次导入 ppt-master 重新导出） |
| `outputs/report_assets/` | 5 张报告素材表（口径对照 / 图表清单 / 配置总表 / 训练摘要 / 性能元信息） |
| `outputs/architecture/repvit_m0_9_arch.{png,md}` | 自绘整网结构图，9 要素齐全 |
- **09-13【权重完整性】字节数一致不等于内容完好。** 重新下载 `repvit_m1_5_distill_300e.pth` 前，
  它在**字节数与当时记录的期望值完全一致**的情况下 `torch.load` 直接抛
  `PytorchStreamReader failed reading zip archive: failed finding central directory`。
  教训：`tools/check_weights.py` 原来只比字节数，会把它判成 OK；已加固为
  **字节数 + zip 中央目录自检 + torch.load 实测**三道校验（`loadable=False` 计数必须为 0），
  并修正了 m1_5 的期望字节数为 59,375,411。

## M13 全局验收与推送

- `python tools/selfcheck.py` → **34/34 PASS，0 FAIL**
- DoD 37 条逐条自检通过；工具零硬编码绝对路径（`paths` 检查 0 处命中）
- 已推送到公开仓库：**https://github.com/8ga-tech/RepViT-Reproduction**
  （447 个文件、187.8 MB；2 个 checkpoint + 4 个 ONNX 已入库）

### 推送过程中的环境约束（全部实测，已如实处理）

| 通道 | 实测结果 | 处置 |
|---|---|---|
| `git push` over HTTPS (`github.com`) | `Recv failure: Connection was reset`（直连被阻断，HTTP 000） | 不可用 |
| SSH (`ssh.github.com:443` / `:22`) | TCP + SSH 握手成功，但 `Permission denied (publickey)` | 缺 SSH key |
| `gh ssh-key add` | `HTTP 404`——token 缺 `admin:public_key` scope | 无法自助加 key |
| `gh auth refresh -s admin:public_key` | 需要 `github.com/login/device/code`，同样被阻断 | 不可用 |
| **`api.github.com`** | **HTTP 200**，token 有 `repo` scope | **✅ 唯一可用通道** |

最终用 **GitHub Git Data API** 推送（`_build/push_via_api.py`）：逐文件建 blob → 建 tree
→ 建 commit → 更新 ref。两处踩坑并已修正：

1. **空仓库不能直接建 blob**（`409 Git Repository is empty`）——
   先用 Contents API 写一个文件建立首个 commit，再走 Git Data API。
2. **内容必须取自 git 对象，不能读工作区文件**——Windows 上 `core.autocrlf=true`
   会把仓库里的 LF 在工作区转成 CRLF；直接读文件上传导致 **220/447 个 blob SHA
   与本地 commit 不符**。改用 `git cat-file --batch` 取字节后，
   远端 447 个 blob 与本地 commit **逐字节一致**（已逐 blob 比对验证）。
3. API 对单个 blob 有请求体上限：57.6 MB 的完整 checkpoint 会被
   `422 input too large` 拒绝（实测）。故交付**推理态瘦身版**（19.4 MB/份，
   模型张量逐位相同，仅去掉 optimizer/scheduler/scaler/rng_state），
   完整版保留在本机 `checkpoints/_full/`。

> 因此远端提交 SHA（`81693dd4`）与本地（`726a55c`）不同——commit SHA 编码了
> 提交时间与 committer，API 写入无法复现本地时间戳。但**两边的 tree 内容逐字节一致**，
> 这是可以验证的强等价。
