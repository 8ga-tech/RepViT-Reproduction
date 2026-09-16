# report/ppt_svg —— 答辩 PPT 的合并前设计源（设计记录，不是交付物）

本目录下的 15 份 SVG 是 `report/答辩PPT_RepViT.pptx` 的**合并前设计源**（早期逐页设计稿），只作设计记录与历史参照；**权威交付物是 `report/答辩PPT_RepViT.pptx`**，两者不一致时**一律以 PPTX 为准**。

为容纳试题要求的可视化（≥8 张测试集预测 / ≥4 张同图对比 / ≥5 张训练集外实拍），最终版已把 P03（M0.9 结构图）与 P04（为什么不是标准 ViT）**合并为一页**：因此 `03_arch.svg` 与 `04_notvit.svg` 对应的是合并前的两个独立页面，并且**全部文件的页脚编号仍是合并前的旧页序** —— `01_cover.svg`/`02_status.svg` → 最终 P01/P02，`05..08` 与 `10..11` 号文件 → 最终页序 −1（例如 `05_pretrained.svg` 页脚是 05，对应最终版 P04），`12..15` 号文件 → 与最终页序相同；**不要为了对齐而批量改这些页脚编号**，本注记就是用来替代那次改动的。

`09_results.svg`（8 组实验定量结果表）对应的旧 P09 已被最终版的**训练曲线页（P08）**取代，该文件仅为历史设计记录，**不作为答辩口径**；最终版另有新增的 P11「预测结果」页，本目录中没有它的设计源。

另外：本目录**没有任何脚本会自动消费这些 SVG 重建 PPTX**；PPTX 与 `report/PPT_CONTENT.md` 由 `python tools/update_defense.py` 统一维护，PDF 由 `python tools/export_defense_pdf.py` 用 PowerPoint COM 导出。
